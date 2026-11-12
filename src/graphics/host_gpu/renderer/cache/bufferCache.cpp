#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/cache/hostImportProbe.h"
#include "graphics/host_gpu/renderer/cache/readbackQueue.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/speculation.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stallWatch.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size, bool guest_store) {
	// Round 167 (copies:lean): in the command buffer's batch of copies (copyBufferBatched). The
	// guest's own stores are not ordered after the GPU work before them unless it waits for that
	// work, which its sync points then order: with them, a store's copy waits for nothing either.
	const bool batched     = Bisect::LeanCopies() && GuestGpu::IsGpuThread();
	const bool wait_before = !guest_store || !GuestGpu::GuestPm4Barriers();
	auto*      bytes       = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		if (batched) {
			const auto destination = buffer.Offset(address);
			if (destination > buffer.Size() || chunk > buffer.Size() - destination) {
				EXIT("BufferCache: store 0x%016" PRIx64 "+0x%" PRIx64 " out of its buffer\n",
				     address, chunk);
			}
			auto& command = m_scheduler.Current();
			command.EndRendering();
			const vk::BufferCopy copy {offset, destination, chunk};
			command.Recorder().copyBufferBatched(m_staging_buffer.Handle(), buffer.Handle(), 1,
			                                     &copy, wait_before);
		} else {
			buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
			                chunk, vk::AccessFlagBits::eHostWrite);
		}
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

// KYTY_BUFFER_REGIONS=0: a new cached buffer leaves the tracker regions under it to their first
// use, as before round 103 (ChangeRegister).
static bool BufferRegions() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_REGIONS");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		// Round 103: the tracker regions under it exist from now on. The BDA walk
		// (SynchronizeBdaPending) only reaches regions that exist, and a region is created by
		// the tracker's first use of it (a binding's synchronization, a fault's MakeResident). A
		// buffer that a merge or a stream leap extended over memory no such use had reached held
		// zeros there that nothing uploaded, while shaders read it through addresses without a
		// fault (the page table names the buffer). A new region is all CPU-dirty and pending, so
		// the next walk uploads what the buffer covers of it. KYTY_BUFFER_REGIONS=0 leaves them
		// to their first use, as before.
		if (BufferRegions()) {
			if (const auto created =
			        m_memory_tracker.TrackRegions(buffer.CpuAddress(), buffer.Size());
			    created != 0) {
				m_new_regions = true;
				static std::atomic<uint64_t> buffers {0};
				static std::atomic<uint64_t> regions {0};
				const auto n     = buffers.fetch_add(1, std::memory_order_relaxed) + 1;
				const auto total = regions.fetch_add(created, std::memory_order_relaxed) + created;
				if (n <= 8 || std::has_single_bit(n)) {
					LOGF("BufferCache: buffer 0x%016" PRIx64 "+0x%" PRIx64 " covers %zu tracker "
					     "regions no use had created: created for the BDA walk to upload (%" PRIu64
					     " buffers, %" PRIu64 " regions so far; KYTY_BUFFER_REGIONS=0 turns this "
					     "off)\n",
					     buffer.CpuAddress(), buffer.Size(), created, n, total);
				}
			}
		}
		// The BDA walk synchronizes it (the CPU-dirty pages it covers may have none over them
		// until now).
		m_memory_tracker.MarkBdaPending(buffer.CpuAddress(), buffer.Size());
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		buffer.lru_id = m_lru_cache.Insert(id, LruClock());
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, LruClock());
		if (buffer.gc_pending) {
			buffer.gc_reused = true;
		}
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	// Round 152: a fast readback's copy may still be reading it on the readback queue, which the
	// renderer's ticks (the release below) do not wait for.
	if (m_readback_queue != nullptr) {
		m_readback_queue->WaitCopiesFrom(m_slot_buffers[id].Handle());
	}
	Unregister(id);
	// As TextureCache::DeleteImage: earlier command buffers may still use the buffer on the GPU.
	m_scheduler.DeferRelease([this, id] { ReleaseBuffer(id); });
}

void BufferCache::ReleaseBuffer(BufferId id) {
	const auto* buffer = m_slot_buffers.try_get(id);
	if (buffer != nullptr && !m_scheduler.IsFree(buffer->last_use_tick)) {
		// Bound after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: buffer guest=0x%016" PRIx64 "+0x%" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     buffer->CpuAddress(), buffer->Size(), buffer->last_use_tick);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseBuffer(id); });
		return;
	}
	m_slot_buffers.erase(id);
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                       uint64_t* downloaded) {
	// One reservation cannot exceed the download ring, so a larger range goes in ring-sized
	// windows; a window that does not fit drains the ring before it is mapped.
	const auto capacity = m_download_buffer.Size();
	bool       any      = false;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		any |= DownloadBufferWindow<async>(buffer, vaddr + offset, std::min(capacity, size - offset),
		                                   downloaded);
	}
	return any;
}

template <bool async>
bool BufferCache::DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                       uint64_t* downloaded) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges.Ranges(), address, bytes,
		                                           "buffer download");
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
			    m_downloading_ranges.Add(start, end - start);
			    m_downloading_list.emplace_back(start, end - start);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	if (downloaded != nullptr) {
		for (const auto& copy: copies) {
			*downloaded += copy.size;
		}
	}
	const auto capacity = m_download_buffer.Size();
	for (size_t first = 0; first < copies.size();) {
		const auto base       = copies[first].dstOffset;
		auto       last       = first;
		uint64_t   batch_size = 0;
		while (last < copies.size()) {
			const auto end = copies[last].dstOffset - base + Common::AlignUp(copies[last].size, 64);
			if (end > capacity) {
				break;
			}
			batch_size = end;
			last++;
		}
		EXIT_IF(last == first);
		std::vector<vk::BufferCopy> batch(copies.begin() + static_cast<std::ptrdiff_t>(first),
		                                  copies.begin() + static_cast<std::ptrdiff_t>(last));
		for (auto& copy: batch) {
			copy.dstOffset -= base;
		}
		DownloadBufferCopies<async>(buffer, std::move(batch), batch_size);
		first = last;
	}
	return true;
}

template <bool async>
void BufferCache::DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
                                       uint64_t total_size) {
	Profiler::Add(Profiler::Counter::Readbacks);
	Profiler::Add(Profiler::Counter::ReadbackBytes, static_cast<int64_t>(total_size));
	const auto buffer_address = buffer.CpuAddress();
	auto [mapped, offset]     = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	command.MarkTransfer(CommandBufferDebugOp::InternalWork, buffer_address, 0,
	                     static_cast<uint32_t>(InternalWork::BufferDownload),
	                     static_cast<uint32_t>(std::min<uint64_t>(total_size, UINT32_MAX)),
	                     static_cast<uint32_t>(copies.size()));
	// Recorded through the recording thread: a raw Handle() made every readback wait for it to
	// drain first (4.5 % of the GPU thread at the Leap Attack prompt, 2026-10-05).
	const auto              native = command.Recorder();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	// Round 153: an asynchronous download's publication tick and the bounds of its bytes, for the
	// reads that wait for it (DownloadTickFor).
	uint64_t batch_id = 0; // not const: a constant 0 would need no capture
	if constexpr (async) {
		batch_id = ++m_download_batch_ids;
	}
	uint64_t batch_begin = UINT64_MAX;
	uint64_t batch_end   = 0;
	for (const auto& copy: copies) {
		batch_begin = std::min(batch_begin, buffer_address + copy.srcOffset);
		batch_end   = std::max(batch_end, buffer_address + copy.srcOffset + copy.size);
	}
	auto publish = [this, mapped, offset, total_size, buffer_address, copies = std::move(copies),
	                owner = std::move(temporary), batch_id] {
		const Libs::LibKernel::Memory::WriteSourceScope source(
		    Libs::LibKernel::Memory::WriteSource::Download);
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			const auto  address = buffer_address + copy.srcOffset;
			const auto* bytes   = mapped + (copy.dstOffset - offset);
			if (Libs::LibKernel::Memory::TryWriteBacking(address, bytes, copy.size)) {
				continue;
			}
			// The guest unmapped some of these bytes after the GPU wrote them (a download window
			// reaches past the mapping the read faulted in, and a buffer can span mappings):
			// nothing is left to write those to. The pages still mapped get theirs. Route X's
			// loading ended the emulator here (2026-10-14).
			uint64_t dropped = 0;
			for (uint64_t done = 0; done < copy.size;) {
				const auto at    = address + done;
				const auto piece = std::min<uint64_t>(
				    copy.size - done, Common::AlignDown(at, 0x1000u) + 0x1000u - at);
				if (!Libs::LibKernel::Memory::TryWriteBacking(at, bytes + done, piece)) {
					if (Libs::LibKernel::Memory::TryClampRangeSize(at, piece) == piece) {
						EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64
						     " size=0x%016" PRIx64 "\n",
						     at, piece);
					}
					dropped += piece;
				}
				done += piece;
			}
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("BufferCache: 0x%" PRIx64 " GPU-written bytes of 0x%016" PRIx64 "+0x%" PRIx64
				     " dropped: the guest unmapped them before their download landed\n",
				     dropped, address, static_cast<uint64_t>(copy.size));
			}
		}
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			const auto entry = std::find(m_downloading_list.begin(), m_downloading_list.end(),
			                             std::pair {buffer_address + copy.srcOffset, copy.size});
			if (entry != m_downloading_list.end()) {
				*entry = m_downloading_list.back();
				m_downloading_list.pop_back();
			}
			m_downloading_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
		}
		// Ranges another download still has in flight stay marked.
		for (const auto& [start, bytes]: m_downloading_list) {
			m_downloading_ranges.Add(start, bytes);
		}
		if (batch_id != 0) {
			const auto batch =
			    std::find_if(m_download_batches.begin(), m_download_batches.end(),
			                 [&](const DownloadBatch& b) { return b.id == batch_id; });
			if (batch != m_download_batches.end()) {
				*batch = m_download_batches.back();
				m_download_batches.pop_back();
			}
		}
	};
	if constexpr (async) {
		{
			// The tick DeferPriorityOperation gives the publication: nothing submits in between.
			std::unique_lock lock(m_dirty_ranges_mutex);
			m_download_batches.push_back(
			    {batch_id, m_scheduler.CurrentTick(), batch_begin, batch_end});
		}
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
}

// Round 92 (from BryanKAdams' BryKytyPS5, a25a2b97): the stream ring takes the GPU thread's
// per-draw copies (small CPU-written buffers, flattened resource tables, shader data). That fork
// measured them faster in cached host memory than in write-combined device memory; here, in a
// race, the device ring was 0.8 ms a frame faster (19.7 against 19.3 fps, 51 A/B pairs, round 92),
// so it stays the default. KYTY_STREAM_HOST=1 puts the ring in cached host memory; the step
// stream:host switches to one there.
static bool StreamRingInHostMemory() {
	const char* value = std::getenv("KYTY_STREAM_HOST");
	return value != nullptr && value[0] == '1';
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB, StreamRingInHostMemory()),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!StreamRingInHostMemory() && Bisect::StreamHostConfigured()) {
		m_stream_other = std::make_unique<StreamBuffer>(graphics, scheduler, MemoryUsage::Stream,
		                                                64 * MiB, true);
	}
	// Vulkan memory property bits: 1 device-local, 2 host-visible, 4 coherent, 8 cached.
	LOGF("Stream ring: memory 0x%x (%s)%s\n", m_stream_buffer.MemoryProperties(),
	     StreamRingInHostMemory() ? "cached host memory, KYTY_STREAM_HOST=1" : "device memory",
	     m_stream_other != nullptr ? fmt::format(", for stream:host a ring in memory 0x{:x}",
	                                             m_stream_other->MemoryProperties())
	                                     .c_str()
	                               : "");
	// Readahead's downloads go at the end of the command buffer that wrote their bytes.
	m_scheduler.SetBeforeFlush(&BufferCache::IssueReadaheadHook, this);
	// Game threads' reads of bytes the GPU has finished writing go on a queue of their own.
	m_readback_queue =
	    std::make_unique<ReadbackQueue>(m_graphics, m_scheduler, m_scheduler.GetMasterSemaphore());
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
	// KYTY_BUFFER_CACHE_MB: an upper bound whatever the card's size (default 6 GiB; 0 = none), as
	// KYTY_TEXTURE_CACHE_MB for images: on a 32 GB card buffers of scenes the game had left stayed
	// until ~26 GB.
	const char* value = std::getenv("KYTY_BUFFER_CACHE_MB");
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 6144ull;
	if (mib != 0) {
		const auto cap       = mib << 20u;
		m_trigger_gc_memory  = std::min<uint64_t>(m_trigger_gc_memory, cap);
		m_critical_gc_memory = std::min<uint64_t>(m_critical_gc_memory, cap + cap / 4);
		LOGF("BufferCache: thresholds capped at %" PRIu64 " MB (critical %" PRIu64 " MB)\n",
		     m_trigger_gc_memory >> 20u, m_critical_gc_memory >> 20u);
	}
}

BufferCache::~BufferCache() {
	m_scheduler.SetBeforeFlush(nullptr, nullptr);
	m_readback_queue.reset();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::ForgetFills(uint64_t vaddr, uint64_t size) {
	if (m_known_fills.empty()) {
		return;
	}
	std::erase_if(m_known_fills, [&](const KnownFillRecord& fill) {
		return fill.address < vaddr + size && vaddr < fill.address + fill.size;
	});
	m_known_fill_count.store(m_known_fills.size(), std::memory_order_relaxed);
}

void BufferCache::ForgetFillsUnder(uint64_t vaddr, uint64_t size) {
	if (m_known_fill_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	std::unique_lock lock(m_dirty_ranges_mutex);
	ForgetFills(vaddr, size);
}

namespace {
struct DynamicWatch {
	uint64_t    address = 0;
	uint64_t    size    = 0;
	const char* label   = "";
};
std::mutex                                    g_dynamic_watch_mutex;
std::vector<DynamicWatch>                     g_dynamic_watches;
std::atomic<bool>                             g_dynamic_watching {false};
std::set<std::pair<uint64_t, uint64_t>>       g_dynamic_watch_seen; // (range, shader)
} // namespace

void BufferCache::WatchGpuWrites(uint64_t vaddr, uint64_t size, const char* label) {
	std::lock_guard lock(g_dynamic_watch_mutex);
	constexpr size_t MaxWatches = 8;
	for (const auto& watch: g_dynamic_watches) {
		if (watch.address == vaddr) return;
	}
	if (g_dynamic_watches.size() >= MaxWatches) return;
	g_dynamic_watches.push_back({vaddr, size, label});
	g_dynamic_watching.store(true, std::memory_order_release);
	LOGF("GPU write watch: %s 0x%016" PRIx64 "+0x%" PRIx64 "\n", label, vaddr, size);
}

bool BufferCache::KnownFill(uint64_t vaddr, uint64_t size, uint32_t& value) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	for (const auto& fill: m_known_fills) {
		if (fill.address <= vaddr && vaddr + size <= fill.address + fill.size) {
			value = fill.value;
			return true;
		}
	}
	return false;
}

bool BufferCache::FillOfNewImage(uint64_t vaddr, uint64_t size, uint32_t& value, bool take) {
	if (size == 0 || size > UINT64_MAX - vaddr) {
		return false;
	}
	std::unique_lock lock(m_dirty_ranges_mutex);
	const auto       end   = vaddr + size;
	const auto       found = std::ranges::find_if(m_known_fills, [&](const KnownFillRecord& fill) {
        return fill.address <= vaddr && end <= fill.address + fill.size;
    });
	if (found == m_known_fills.end()) {
		return false;
	}
	// The part of the image the fill wrote on the GPU: still GPU-written, or something wrote
	// over it unseen (its buffer was collected: the pages are no longer tracked).
	const auto gpu_end = std::min(end, found->address + found->gpu_bytes);
	if (gpu_end > vaddr && !m_gpu_modified_ranges.Contains(vaddr, gpu_end - vaddr)) {
		m_known_fills.erase(found);
		m_known_fill_count.store(m_known_fills.size(), std::memory_order_relaxed);
		return false;
	}
	value = found->value;
	if (!take) {
		return true;
	}
	// Keep what is left on either side, each with the part of it written on the GPU.
	const auto fill        = *found;
	const auto fill_end    = fill.address + fill.size;
	const auto fill_gpu_to = fill.address + fill.gpu_bytes;
	m_known_fills.erase(found);
	if (vaddr > fill.address) {
		m_known_fills.push_back({fill.address, vaddr - fill.address, fill.value,
		                         std::min(fill.gpu_bytes, vaddr - fill.address)});
	}
	if (end < fill_end) {
		m_known_fills.push_back(
		    {end, fill_end - end, fill.value, fill_gpu_to > end ? fill_gpu_to - end : 0});
	}
	m_known_fill_count.store(m_known_fills.size(), std::memory_order_relaxed);
	return true;
}

void BufferCache::NoteFill(uint64_t vaddr, uint64_t size, uint32_t value, uint64_t gpu_bytes,
                           bool merge) {
	if (size == 0 || size > UINT64_MAX - vaddr) {
		return;
	}
	gpu_bytes = std::min(gpu_bytes, size);
	std::unique_lock lock(m_dirty_ranges_mutex);
	ForgetFills(vaddr, size);
	if (merge && gpu_bytes == size) {
		for (auto& fill: m_known_fills) {
			if (fill.value != value || fill.gpu_bytes != fill.size) {
				continue;
			}
			if (fill.address + fill.size == vaddr) {
				fill.size += size;
				fill.gpu_bytes = fill.size;
				return;
			}
			if (vaddr + size == fill.address) {
				fill.address = vaddr;
				fill.size += size;
				fill.gpu_bytes = fill.size;
				return;
			}
		}
	}
	constexpr size_t   MaxKnownFills = 256;
	constexpr uint64_t SmallFill     = 1u << 20u;
	if (m_known_fills.size() >= MaxKnownFills) {
		// The oldest small fill goes first: a large one is an image's bytes, not created yet.
		const auto evict = std::ranges::find_if(
		    m_known_fills, [](const KnownFillRecord& fill) { return fill.size < SmallFill; });
		m_known_fills.erase(evict != m_known_fills.end() ? evict : m_known_fills.begin());
	}
	m_known_fills.push_back({vaddr, size, value, gpu_bytes});
	m_known_fill_count.store(m_known_fills.size(), std::memory_order_relaxed);
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	// Every CPU write fault comes through here (tens of thousands a second in races): without
	// known fills there is nothing to forget, and the exclusive lock only made the game's threads
	// queue behind each other and the GPU thread.
	if (m_known_fill_count.load(std::memory_order_relaxed) != 0) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		ForgetFills(vaddr, size);
	}
	m_memory_tracker.InvalidateRegion(vaddr, size, [this, vaddr, size] {
		// A CPU store to GPU-written bytes: a fast clear checked on the GPU there no longer holds.
		if (m_metadata_watch_count.load(std::memory_order_relaxed) != 0) {
			const auto       page = vaddr & ~(TRACKER_PAGE_SIZE - 1);
			std::unique_lock lock(m_dirty_ranges_mutex);
			NoteMetadataWrite(page, TRACKER_PAGE_SIZE);
		}
		ReadMemory(vaddr, size, true);
	});
	// Round 90: a write of the game's (the walk for memory-reading shaders after a wait looks for
	// one, RenderContext::PrepareBda). A store to GPU-written bytes is made CPU-dirty on the GPU
	// thread (ReadMemory), where it does not count by itself.
	if (!t_guest_gpu_thread) {
		g_cpu_write_epoch.fetch_add(1, std::memory_order_release);
	}
}

bool BufferCache::ReleaseBlockForCpuWrite(uint64_t block, uint64_t size) {
	// One tracker region, so a refusal leaves the whole block as it was.
	if (!GuestRange {block, size}.Valid() ||
	    block / TRACKER_REGION_SIZE != (block + size - 1) / TRACKER_REGION_SIZE ||
	    !m_memory_tracker.ReleaseForCpuWrite(block, size)) {
		return false;
	}
	if (m_known_fill_count.load(std::memory_order_relaxed) != 0) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		ForgetFills(block, size);
	}
	return true;
}

size_t BufferCache::MarkHotPages(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	const auto pages = m_memory_tracker.MarkHot(vaddr, size);
	if (pages != 0 && m_known_fill_count.load(std::memory_order_relaxed) != 0) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		ForgetFills(vaddr, size);
	}
	return pages;
}

void BufferCache::ForgetUnmapped(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		return;
	}
	uint64_t dropped = 0;
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		// Memory mapped there again holds other bytes: its metadata checks start over.
		NoteMetadataWrite(vaddr, size);
		if (!m_gpu_modified_ranges.Intersects(vaddr, size)) {
			return;
		}
		m_gpu_modified_ranges.ForEachInRange(
		    vaddr, size, [&](uint64_t begin, uint64_t end) { dropped += end - begin; });
		m_gpu_modified_ranges.Subtract(vaddr, size);
	}
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
		LOGF("BufferCache: 0x%" PRIx64 " GPU-written bytes of unmapped memory 0x%016" PRIx64
		     "+0x%" PRIx64 " dropped\n",
		     dropped, vaddr, size);
	}
}

// Round 132: KYTY_STORE_CLEAN=0 (or F8's storeclean:off): a GPU-thread store to CPU-clean watched
// pages makes them CPU-dirty, as before (WriteClean).
static bool StoreCleanEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STORE_CLEAN");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::StoreCleanOff();
}

bool BufferCache::WriteClean(uint64_t vaddr, const void* data, uint64_t size, bool guest_store) {
	if (!GuestGpu::IsGpuThread()) {
		return false;
	}
	// Round 111: the GPU thread's stores (labels, WRITE_DATA) come after its deferred work.
	FinishDeferredMemoryWork();
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	const bool gpu_written = m_memory_tracker.IsRegionGpuModified(vaddr, size);
	if (!gpu_written) {
		// Round 132: CPU-clean bytes; only worth it on pages watched for the CPU's writes (a
		// writable page takes the store as it is).
		if (!StoreCleanEnabled()) {
			return false;
		}
		auto& pages = m_memory_tracker.Pages();
		for (uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1); page < vaddr + size;
		     page += TRACKER_PAGE_SIZE) {
			if (!pages.IsWriteWatched(page)) {
				return false;
			}
		}
	}
	{
		// A download publishes the GPU's older value of these bytes when it lands.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}
	// Round 133: an image write-back on its way puts the image's bytes in guest memory when it
	// lands (on the priority thread), over this store: it goes the usual way, as a fast readback's
	// read does (FastReadback).
	if (const auto writebacks = m_texture_cache.PendingWriteBacks();
	    !writebacks.empty() &&
	    std::ranges::any_of(writebacks, [&](const std::pair<uint64_t, uint64_t>& pending) {
		    return pending.first < vaddr + size && vaddr < pending.first + pending.second;
	    })) {
		return false;
	}

	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	// A caller that names its writes (a metadata clear) keeps its name.
	using Libs::LibKernel::Memory::WriteSource;
	const auto outer = Libs::LibKernel::Memory::CurrentWriteSource();
	const Libs::LibKernel::Memory::WriteSourceScope source(
	    outer == WriteSource::Other ? WriteSource::StoreInPlace : outer);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, data, size, guest_store);
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		ForgetFills(vaddr, size);
		NoteMetadataWrite(vaddr, size);
		if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
			// Overwritten in full: guest memory holds the value the GPU copy will have.
			m_gpu_modified_ranges.Subtract(vaddr, size);
		}
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	if (!gpu_written) {
		// As a CPU write fault would tell them (RenderContext::InvalidateMemory).
		if (m_clean_store_hook != nullptr) {
			m_clean_store_hook(m_clean_store_user, vaddr, size);
		}
		GpuPhases::Count(GpuPhases::Phase::CleanStore, 1);
	}
	return true;
}

void BufferCache::WatchMetadata(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// A bound on the watches kept: past it they start again (every range then checks once more).
	constexpr size_t MaxWatches = 4096;
	std::unique_lock lock(m_dirty_ranges_mutex);
	if (m_metadata_watches.size() >= MaxWatches) {
		m_metadata_watches.clear();
		m_metadata_watch_span = 0;
	}
	// Watches over the range that start elsewhere go: the range is current as checked now.
	for (auto it = m_metadata_watches.lower_bound(
	         vaddr >= m_metadata_watch_span ? vaddr - m_metadata_watch_span : 0);
	     it != m_metadata_watches.end() && it->first < vaddr + size;) {
		if (it->second.end > vaddr && it->first != vaddr) {
			it = m_metadata_watches.erase(it);
		} else {
			++it;
		}
	}
	m_metadata_watches[vaddr] = {vaddr + size, false};
	m_metadata_watch_span     = std::max(m_metadata_watch_span, size);
	m_metadata_watch_count.store(m_metadata_watches.size(), std::memory_order_relaxed);
}

bool BufferCache::MetadataUnwritten(uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	const auto       found = m_metadata_watches.find(vaddr);
	return found != m_metadata_watches.end() && found->second.end == vaddr + size &&
	       !found->second.written;
}

void BufferCache::NoteMetadataWrite(uint64_t vaddr, uint64_t size) {
	if (m_metadata_watches.empty() || size == 0) {
		return;
	}
	const auto end = vaddr + size;
	for (auto it = m_metadata_watches.lower_bound(
	         vaddr >= m_metadata_watch_span ? vaddr - m_metadata_watch_span : 0);
	     it != m_metadata_watches.end() && it->first < end; ++it) {
		if (it->second.end > vaddr) {
			it->second.written = true;
		}
	}
}

// A game thread's fault on GPU-written memory no longer makes the GPU thread wait for the GPU.
// The GPU thread records the download, submits it and goes on; the faulting thread waits for
// that download's publication itself, then asks again (the window's pages then have no
// GPU-written bytes left and their protection is lifted). KYTY_ASYNC_READBACK=0 waits on the GPU
// thread as before.
static const bool g_async_readback = [] {
	const char* value = std::getenv("KYTY_ASYNC_READBACK");
	return value == nullptr || value[0] != '0';
}();

// Game threads' accesses through ReadMemory: done in one step (the bytes were in guest memory),
// or after waiting for a download. Readahead's summary line reports them.
static std::atomic<uint64_t> g_reads_direct {0};
static std::atomic<uint64_t> g_reads_waited {0};

// Round 135, KYTY_CLEAN_PAGE=old (or KYTY_BISECT's cleanpage:old): a fault on pages another
// thread's read made current since downloads the rest of its window and waits for it, as before.
static bool CleanPageOld() {
	static const bool old = [] {
		const char* value = std::getenv("KYTY_CLEAN_PAGE");
		return value != nullptr && std::strcmp(value, "old") == 0;
	}();
	return old || Bisect::CleanPageOld();
}

// Round 135, KYTY_READ_WAIT=writer (or KYTY_BISECT's readwait:writer): a game thread's read of
// bytes whose writer the GPU is still running waits for that writer's command buffer, then copies
// them on the readback queue, instead of downloading them behind the renderer's queue.
static bool ReadWaitWriter() {
	static const bool writer = [] {
		const char* value = std::getenv("KYTY_READ_WAIT");
		return value != nullptr && std::strcmp(value, "writer") == 0;
	}();
	return writer || Bisect::ReadWaitWriter();
}

// Round 153, KYTY_LIFT_WAIT=old (or KYTY_BISECT's liftwait:old): a read of pages whose bytes a
// download is still bringing in waits for the latest download of all, as before, instead of the
// ones that cover its pages.
static bool LiftWaitOld() {
	static const bool old = [] {
		const char* value = std::getenv("KYTY_LIFT_WAIT");
		return value != nullptr && std::strcmp(value, "old") == 0;
	}();
	return old || Bisect::LiftWaitOld();
}

namespace {

// Round 135: game threads' reads of GPU-written memory (ReadMemory) by the instruction whose fault
// asked for them (Common::PerfStats::t_fault_pc; 0: not from a fault), with how each went (its
// first step) and how long the thread was held, and every 10 s the costliest ("Reads of
// GPU-written memory" lines). Only while the Perf lines are on.
constexpr size_t kReadOutcomes = 11;
constexpr size_t kFastRefusals = 9;

constexpr std::array<const char*, kReadOutcomes> kReadOutcomeNames {
    "?",                               // None
    "nothing to do",                   // Direct
    "write bitmap waits",              // Bitmap
    "lifted",                          // Lifted
    "lifted once a download landed",   // LiftWaited
    "current already",                 // Clean
    "copied at once",                  // Fast
    "waited for the writer",           // WaitWriter
    "downloaded",                      // Download
    "nothing left to download",        // Nothing
    "waited for another read's copy"}; // FastWait
constexpr std::array<const char*, kFastRefusals> kFastRefusalNames {
    "?",                       // None
    "no readback queue",       // Unavailable
    "a download on its way",   // Downloading
    "shader address stores",   // Address
    "writer being recorded",   // Recording
    "writer on the GPU",       // InFlight
    "other",                   // Other
    "nothing to copy at once", // Nothing
    "too big"};                // TooBig

struct ReadSite {
	uint64_t                            count      = 0;
	uint64_t                            ns         = 0;
	uint64_t                            longest_ns = 0;
	uint64_t                            last_vaddr = 0;
	uint32_t                            tid        = 0;
	std::array<uint64_t, kReadOutcomes> outcomes {};
	std::array<uint64_t, kReadOutcomes> outcome_ns {};
	// Downloads by why the fast readback did not copy, and their time.
	std::array<uint64_t, kFastRefusals> downloads {};
	std::array<uint64_t, kFastRefusals> download_ns {};
	uint64_t                            download_bytes = 0;
	// Downloads whose faulting pages' writer the GPU was running: how many command buffers it
	// still had to finish before the writer's, and before the download's.
	uint64_t inflight       = 0;
	uint64_t writer_ahead   = 0;
	uint64_t download_ahead = 0;
	// Downloads a fast readback found nothing to copy for, with ranges left elsewhere in the
	// window (still being written).
	uint64_t nothing_left = 0;
	// Writer waits (readwait:writer) whose next step copied at once.
	uint64_t writer_then_fast = 0;
};

struct ReadSiteNote {
	uint64_t pc             = 0;
	uint64_t vaddr          = 0;
	uint64_t ns             = 0;
	size_t   outcome        = 0;
	size_t   refusal        = 0;
	uint64_t bytes          = 0;
	uint64_t writer_ahead   = 0;
	uint64_t download_ahead = 0;
	bool     inflight       = false;
	bool     left           = false;
	bool     then_fast      = false;
};

std::mutex                                                    g_read_sites_mutex;
std::map<std::string, std::unordered_map<uint64_t, ReadSite>> g_read_sites;
uint64_t                                                      g_read_sites_since_ns = 0;

// " 1160 reads, 5300 ms (4.57 ms each, longest 14.0): lifted 60 (0.21 ms each), ..."
std::string DescribeReadOutcomes(const ReadSite& site) {
	const auto ms   = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	auto       text = fmt::format(
        " {} reads, {:.0f} ms ({:.2f} ms each, longest {:.1f}):", site.count, ms(site.ns),
        site.count == 0 ? 0.0 : ms(site.ns) / static_cast<double>(site.count), ms(site.longest_ns));
	bool first = true;
	for (size_t outcome = 1; outcome < kReadOutcomes; outcome++) {
		const auto count = site.outcomes[outcome];
		if (count == 0) {
			continue;
		}
		text +=
		    fmt::format("{} {} {} ({:.2f} ms each)", first ? "" : ",", kReadOutcomeNames[outcome],
		                count, ms(site.outcome_ns[outcome]) / static_cast<double>(count));
		first = false;
		if (outcome == 7 && site.writer_then_fast != 0) {
			text += fmt::format(" then copied at once {}", site.writer_then_fast);
		}
		if (outcome == 8) {
			// The downloads by why the fast readback did not copy.
			std::string why;
			for (size_t refusal = 1; refusal < kFastRefusals; refusal++) {
				if (site.downloads[refusal] == 0) {
					continue;
				}
				why += fmt::format("{}{} {} ({:.2f} ms each)", why.empty() ? "" : ", ",
				                   kFastRefusalNames[refusal], site.downloads[refusal],
				                   ms(site.download_ns[refusal]) /
				                       static_cast<double>(site.downloads[refusal]));
			}
			if (site.inflight != 0) {
				why += fmt::format(
				    "; with the writer on the GPU, it had {:.1f} command buffers "
				    "to finish before the writer's, {:.1f} before the download's",
				    static_cast<double>(site.writer_ahead) / static_cast<double>(site.inflight),
				    static_cast<double>(site.download_ahead) / static_cast<double>(site.inflight));
			}
			if (site.nothing_left != 0) {
				why += fmt::format("; {} of them for ranges elsewhere in the window only",
				                   site.nothing_left);
			}
			text += fmt::format(" [{}{:.1f} KB each]", why.empty() ? std::string() : why + "; ",
			                    static_cast<double>(site.download_bytes) / 1024.0 /
			                        static_cast<double>(count));
		}
	}
	return text;
}

std::string DescribeReadSite(uint64_t pc, const ReadSite& site) {
	return (pc == 0 ? std::string("not from a fault")
	                : fmt::format("0x{:016x} ({}, at 0x{:x})", pc,
	                              Common::PerfStats::ThreadNameOf(site.tid), site.last_vaddr)) +
	       DescribeReadOutcomes(site);
}

void NoteReadSite(const ReadSiteNote& note) {
	// The Perf line's configuration (KYTY_BISECT's step, "<step> base" in the A/B's other half):
	// each has its own counts.
	auto configuration = Common::PerfStats::Configuration();
	std::map<std::string, std::unordered_map<uint64_t, ReadSite>> report;
	uint64_t                                                      seconds_ns = 0;
	uint64_t                                                      now        = 0;
	{
		std::lock_guard lock(g_read_sites_mutex);
		// Inside the lock: a thread that read the clock before another reported would find the
		// report's start after its own time.
		now = StallWatch::NowNs();
		if (g_read_sites_since_ns == 0) {
			g_read_sites_since_ns = now;
		}
		auto& sites = g_read_sites[std::move(configuration)];
		if (sites.size() < 4096 || sites.contains(note.pc)) {
			auto& site = sites[note.pc];
			site.count++;
			site.ns += note.ns;
			site.longest_ns = std::max(site.longest_ns, note.ns);
			site.last_vaddr = note.vaddr;
			site.tid        = Common::PerfStats::CurrentOsThreadId();
			if (note.outcome < kReadOutcomes) {
				site.outcomes[note.outcome]++;
				site.outcome_ns[note.outcome] += note.ns;
			}
			if (note.outcome == 8 && note.refusal < kFastRefusals) {
				site.downloads[note.refusal]++;
				site.download_ns[note.refusal] += note.ns;
				site.download_bytes += note.bytes;
				if (note.inflight) {
					site.inflight++;
					site.writer_ahead += note.writer_ahead;
					site.download_ahead += note.download_ahead;
				}
				site.nothing_left += note.left ? 1u : 0u;
			}
			site.writer_then_fast += note.then_fast ? 1u : 0u;
		}
		if (now - g_read_sites_since_ns < 10'000'000'000ull) {
			return;
		}
		seconds_ns = now - g_read_sites_since_ns;
		report.swap(g_read_sites);
		g_read_sites_since_ns = now;
	}
	for (const auto& [name, sites]: report) {
		ReadSite                                   total;
		std::vector<std::pair<uint64_t, ReadSite>> sorted(sites.begin(), sites.end());
		for (const auto& [pc, site]: sorted) {
			total.count += site.count;
			total.ns += site.ns;
			total.longest_ns = std::max(total.longest_ns, site.longest_ns);
			for (size_t i = 0; i < kReadOutcomes; i++) {
				total.outcomes[i] += site.outcomes[i];
				total.outcome_ns[i] += site.outcome_ns[i];
			}
			for (size_t i = 0; i < kFastRefusals; i++) {
				total.downloads[i] += site.downloads[i];
				total.download_ns[i] += site.download_ns[i];
			}
			total.download_bytes += site.download_bytes;
			total.inflight += site.inflight;
			total.writer_ahead += site.writer_ahead;
			total.download_ahead += site.download_ahead;
			total.nothing_left += site.nothing_left;
			total.writer_then_fast += site.writer_then_fast;
		}
		std::ranges::sort(sorted,
		                  [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		const auto tag = name.empty() ? std::string() : " [" + name + "]";
		LOGF("Reads of GPU-written memory (%.0f s)%s: %zu instructions,%s\n",
		     static_cast<double>(seconds_ns) / 1e9, tag.c_str(), sorted.size(),
		     DescribeReadOutcomes(total).c_str());
		for (size_t i = 0; i < std::min<size_t>(sorted.size(), 6); i++) {
			LOGF("Reads of GPU-written memory #%zu%s: %s\n", i + 1, tag.c_str(),
			     DescribeReadSite(sorted[i].first, sorted[i].second).c_str());
		}
	}
}

} // namespace

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	static_assert(static_cast<size_t>(ReadOutcome::Count) == kReadOutcomes);
	static_assert(static_cast<size_t>(FastRefusal::Count) == kFastRefusals);
	static_assert(static_cast<size_t>(ReadOutcome::WaitWriter) == 7 &&
	              static_cast<size_t>(ReadOutcome::Download) == 8);
	const bool from_gpu_thread = GuestGpu::IsGpuThread();
	const bool async           = g_async_readback && !from_gpu_thread;
	// KYTY_PERF: what a game thread's access waits for (the "Perf GPU:" line).
	const bool measure = !from_gpu_thread && Common::PerfStats::Active();
	const auto now_ns  = [] {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
		                                  .count());
	};
	const auto started      = measure ? now_ns() : 0;
	uint64_t   pickup_ns    = 0;
	uint64_t   record_ns    = 0;
	uint64_t   wait_ns      = 0;
	uint64_t   ticks_behind = 0;
	uint32_t   steps        = 0;
	bool       fast         = false; // a fast readback copied its bytes (FastReadback)
	// Round 135: what the first step did, for the counts by instruction, and whether a writer
	// wait (readwait:writer, once a read) may be asked for.
	bool         writer_wait = async && ReadWaitWriter();
	ReadSiteNote note;
	note.pc    = Common::PerfStats::t_fault_pc;
	note.vaddr = vaddr;
	for (;;) {
		// What a step did, filled on the GPU thread.
		struct Step {
			uint64_t    wait_tick = 0;
			uint64_t    taken_up  = 0;
			uint64_t    recorded  = 0;
			uint64_t    behind    = 0;
			ReadOutcome outcome   = ReadOutcome::None;
			FastRefusal refusal   = FastRefusal::None;
			uint64_t    bytes     = 0;
			uint64_t    writer    = 0; // the refused writer's tick
			uint64_t    current   = 0;
			uint64_t    known     = 0;
			uint32_t    left      = 0;
			// Round 152: the fast readback copy to wait for, then ask again.
			ReadbackQueue::Ticket ticket;
		} step;
		const auto sent = measure ? now_ns() : 0;
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write,
		                                                from_gpu_thread, async, measure,
		                                                writer_wait, &now_ns, &step, &fast] {
			if (measure) {
				step.taken_up = now_ns();
				auto& master  = m_scheduler.GetMasterSemaphore();
				master.Refresh();
				const auto current = m_scheduler.CurrentTick();
				const auto done    = master.KnownGpuTick();
				step.behind        = current > done ? current - done - 1 : 0;
			}
			m_step_bytes = 0;
			step.wait_tick =
			    ReadMemoryStep(vaddr, size, is_write, from_gpu_thread, async, writer_wait);
			fast         = fast || std::exchange(m_fast_step, false);
			step.outcome = m_step_outcome;
			step.refusal = m_fast_refusal;
			step.bytes   = m_step_bytes;
			step.writer  = m_fast_refused_tick;
			step.current = m_fast_current_tick;
			step.known   = m_fast_known_tick;
			step.left    = m_fast_left;
			step.ticket  = std::exchange(m_step_ticket, {});
			if (measure) {
				step.recorded = now_ns();
			}
		});
		if (measure) {
			pickup_ns += step.taken_up - sent;
			record_ns += step.recorded - step.taken_up;
			if (steps == 0) {
				ticks_behind = step.behind;
			}
		}
		if (steps == 0) {
			note.outcome = static_cast<size_t>(step.outcome);
			note.refusal = static_cast<size_t>(step.refusal);
			note.bytes   = step.bytes;
			if (step.refusal == FastRefusal::InFlight && step.writer > step.known) {
				note.inflight       = true;
				note.writer_ahead   = step.writer - step.known;
				note.download_ahead = step.current > step.known ? step.current - step.known : 0;
			}
			note.left = step.refusal == FastRefusal::Nothing && step.left != 0;
		} else if (steps == 1 && note.outcome == static_cast<size_t>(ReadOutcome::WaitWriter)) {
			note.then_fast = step.outcome == ReadOutcome::Fast;
		}
		steps++;
		if (step.ticket.Valid()) {
			// Round 152: the copy of the faulting pages' bytes (this read's, or another's) is on
			// its way on the readback queue. This thread waits for it and puts the bytes in guest
			// memory, not the GPU thread, then asks again (the pages are then lifted).
			const auto wait_from = measure ? now_ns() : 0;
			CompleteFastReadback(step.ticket, FastCompleter::Reader);
			if (measure) {
				wait_ns += now_ns() - wait_from;
			}
			continue;
		}
		if (step.wait_tick == 0) {
			break;
		}
		const auto wait_from  = measure ? now_ns() : 0;
		const auto known      = m_scheduler.GetMasterSemaphore().KnownGpuTick();
		const auto stall_from = StallWatch::NowNs();
		if (step.outcome == ReadOutcome::WaitWriter) {
			// The writer's command buffer, submitted already; then the read asks again, its
			// bytes now final (one such wait a read).
			writer_wait = false;
			m_scheduler.GetMasterSemaphore().Wait(step.wait_tick);
		} else {
			m_scheduler.WaitPriorityOperations(step.wait_tick);
		}
		m_scheduler.OnWaited(step.wait_tick, known, StallWatch::NowNs() - stall_from);
		if (measure) {
			wait_ns += now_ns() - wait_from;
		}
	}
	if (measure) {
		Common::PerfStats::AddReadback(pickup_ns, record_ns, wait_ns, steps, ticks_behind);
		note.ns = now_ns() - started;
		NoteReadSite(note);
	}
	if (!from_gpu_thread) {
		// A fast readback's bytes were not in memory yet either.
		(steps <= 1 && !fast ? g_reads_direct : g_reads_waited)
		    .fetch_add(1, std::memory_order_relaxed);
	}
}

// Runs on the GPU thread. Returns 0 when the access may proceed, or (asynchronous readback only)
// the tick whose download publication the caller must wait for before asking again.
// The readback window (KYTY_READBACK_WINDOW_KB, default 4096; 0 reads only the requested pages,
// as upstream does). The game threads wait in readbacks for much of each frame at the Leap
// Attack prompt; wider windows mean fewer of them. Jungle route, 2026-10-07, prompt / 140-240 s:
// 512 KB 20.4 / 16.4 fps (6 runs), 4 MB 23.3 / 18.2 (3), 8 MB 22.6 / 17.3 (4), 16 MB 22.2 / 16.4,
// 32 MB 22.5 / 15.9 (slower in the fight).
static uint64_t ReadbackWindow() {
	static const uint64_t window = [] {
		const char* value = std::getenv("KYTY_READBACK_WINDOW_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {4096}) * 1024u;
	}();
	// KYTY_BISECT's readback:<n>, an A/B test within one run.
	if (const auto kb = Bisect::ReadbackWindowKB(); kb >= 0) {
		return static_cast<uint64_t>(kb) * 1024u;
	}
	return window;
}

static bool ReadbackRetick() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_RETICK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void BufferCache::LiftCurrentPages(const Buffer& buffer, uint64_t vaddr) {
	const auto window = ReadbackWindow();
	if (window == 0) {
		return;
	}
	const auto            begin = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
	const auto            end   = std::min(begin + window, buffer.CpuAddress() + buffer.Size());
	std::vector<uint64_t> current;
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		for (auto page = Common::AlignDown(begin, TRACKER_PAGE_SIZE); page < end;
		     page += TRACKER_PAGE_SIZE) {
			if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
			    !m_downloading_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
				current.push_back(page);
			}
		}
	}
	// One lift per run of neighbouring pages: each lift is a protection change, and page by page
	// they were most of the GPU thread's VirtualProtect calls.
	for (size_t first = 0; first < current.size();) {
		size_t last = first + 1;
		while (last < current.size() && current[last] == current[last - 1] + TRACKER_PAGE_SIZE) {
			last++;
		}
		const auto run_begin = current[first];
		const auto run_size  = current[last - 1] + TRACKER_PAGE_SIZE - run_begin;
		if (m_memory_tracker.IsRegionGpuModified(run_begin, run_size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(run_begin, run_size);
		}
		first = last;
	}
}

// KYTY_FAST_READBACK=0 (or KYTY_BISECT's fastread:off): every game-thread read of GPU-written
// bytes downloads them behind the renderer's queue, as before round 64.
static bool FastReadbackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAST_READBACK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::FastReadbackOff();
}

// Round 152: KYTY_FAST_READBACK=gpu (or KYTY_BISECT's fastread:gpu): the GPU thread waits for each
// fast readback's copy and puts its bytes in guest memory, as before, instead of the reading
// thread.
static bool FastReadbackWaitsOnGpuThread() {
	static const bool gpu = [] {
		const char* value = std::getenv("KYTY_FAST_READBACK");
		const bool  on    = value != nullptr && std::string_view(value) == "gpu";
		LOGF("BufferCache: a fast readback's copy on the readback queue is waited for by %s\n",
		     on ? "the GPU thread (KYTY_FAST_READBACK=gpu, as before round 152)"
		        : "the reading thread, not the GPU thread (round 152; KYTY_FAST_READBACK=gpu or "
		          "KYTY_BISECT's fastread:gpu: the GPU thread, as before)");
		return on;
	}();
	return gpu || Bisect::FastReadbackOnGpuThread();
}

namespace {
// What the fast readback did, and why reads went the old way, for the "Fast readback:" line (GPU
// thread only).
struct FastReadbackStats {
	uint64_t attempts    = 0;
	uint64_t done        = 0;
	uint64_t bytes       = 0;
	uint64_t gpu_ns      = 0; // the GPU thread's time in the copied ones
	uint64_t ranges      = 0; // the copied ones' ranges
	uint64_t full        = 0; // every copy slot was in use: the oldest was finished first
	uint64_t left        = 0; // ranges elsewhere in the windows, left for a later read
	uint64_t recording   = 0; // the faulting pages' writer is in the command buffer being recorded
	uint64_t in_flight   = 0; // submitted, not finished
	uint64_t downloading = 0; // a download of the faulting pages' bytes is on its way
	uint64_t too_big     = 0;
	uint64_t nothing     = 0; // no GPU-written bytes it could copy in the window
	uint64_t address     = 0; // a command storing through shader addresses is still on the GPU
	uint64_t other       = 0; // a range outside the buffer, or the copy could not be made
};
FastReadbackStats g_fast_readback;

// Round 152, any thread: the copies waited for and put in guest memory, how long their waits were,
// and by whom (BufferCache::FastCompleter: the reading thread, the GPU thread for its own read,
// the GPU thread for another's, the priority thread).
struct FastCompletions {
	std::atomic<uint64_t>                count {0};
	std::atomic<uint64_t>                wait_ns {0};
	std::atomic<uint64_t>                longest_ns {0};
	std::array<std::atomic<uint64_t>, 4> by {};
};
FastCompletions g_fast_completions;

// Every 4096 attempts, when an attempt ends: what those did (then the counts start again), with
// the KYTY_BISECT step they ran in.
struct FastReadbackReport {
	FastReadbackStats& stats;
	~FastReadbackReport() {
		if (stats.attempts < 4096) {
			return;
		}
		auto&          completions = g_fast_completions;
		const uint64_t waits       = completions.count.exchange(0, std::memory_order_relaxed);
		const uint64_t wait_ns     = completions.wait_ns.exchange(0, std::memory_order_relaxed);
		const uint64_t longest_ns  = completions.longest_ns.exchange(0, std::memory_order_relaxed);
		std::array<uint64_t, 4> by {};
		for (size_t i = 0; i < by.size(); i++) {
			by[i] = completions.by[i].exchange(0, std::memory_order_relaxed);
		}
		const auto average = [](uint64_t ns, uint64_t n) {
			return n != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(n) : 0.0;
		};
		const auto step = Common::PerfStats::WindowConfiguration();
		const auto tag  = step.empty() ? std::string() : " [" + step + "]";
		LOGF("Fast readback: %" PRIu64 " of %" PRIu64
		     " reads copied on the readback queue%s (%.1f MB in %.1f ranges each, the GPU thread "
		     "%.0f us on each; "
		     "%" PRIu64 " ranges elsewhere in their windows left for later); their copies waited "
		     "for %.0f us on average, longest %.0f us, by the reading thread %" PRIu64
		     ", the GPU thread %" PRIu64 " for its own reads and %" PRIu64 " for others' (%" PRIu64
		     " with every copy slot in use), the priority thread %" PRIu64
		     "; the others waited for the renderer's queue: %" PRIu64
		     " written in the command buffer being recorded, %" PRIu64
		     " in one not finished, %" PRIu64 " being downloaded, %" PRIu64 " too big, %" PRIu64
		     " without GPU-written bytes to copy, %" PRIu64
		     " behind shader address stores, %" PRIu64 " other\n",
		     stats.done, stats.attempts, tag.c_str(), static_cast<double>(stats.bytes) / 1e6,
		     stats.done != 0 ? static_cast<double>(stats.ranges) / static_cast<double>(stats.done)
		                     : 0.0,
		     average(stats.gpu_ns, stats.done), stats.left, average(wait_ns, waits),
		     static_cast<double>(longest_ns) / 1e3, by[0], by[1], by[2], stats.full, by[3],
		     stats.recording, stats.in_flight, stats.downloading, stats.too_big, stats.nothing,
		     stats.address, stats.other);
		stats = {};
	}
};
} // namespace

void BufferCache::CompleteFastReadback(ReadbackQueue::Ticket ticket, FastCompleter who) {
	if (m_readback_queue == nullptr || !ticket.Valid()) {
		return;
	}
	// A game thread the game runs at a low priority would hold up the threads that wait for this
	// copy (the GPU thread, the priority thread) whenever it is preempted.
	std::optional<Common::ThreadPriorityFloor> floor;
	if (who == FastCompleter::Reader) {
		floor.emplace();
	}
	const ReadbackQueue::Job* job       = nullptr;
	uint64_t                  waited_ns = 0;
	const uint8_t*            bytes     = m_readback_queue->Claim(ticket, job, waited_ns);
	if (bytes == nullptr) {
		// Another thread got to it first (or it is in guest memory already).
		m_readback_queue->WaitDone(ticket);
		return;
	}
	const auto buffer_address = job->buffer_address;
	const auto total          = job->total_size;
	const auto job_begin      = job->begin;
	const auto job_end        = job->end;
	{
		// Into guest memory, as a download's publication puts them (DownloadBufferCopies).
		const Libs::LibKernel::Memory::WriteSourceScope source(
		    Libs::LibKernel::Memory::WriteSource::FastReadback);
		for (const auto& copy: job->copies) {
			const auto  address = buffer_address + copy.srcOffset;
			const auto* from    = bytes + copy.dstOffset;
			if (Libs::LibKernel::Memory::TryWriteBacking(address, from, copy.size)) {
				continue;
			}
			// Bytes the guest has unmapped since the GPU wrote them: the pages still mapped get
			// theirs.
			for (uint64_t done = 0; done < copy.size;) {
				const auto at    = address + done;
				const auto piece = std::min<uint64_t>(
				    copy.size - done, Common::AlignDown(at, 0x1000u) + 0x1000u - at);
				if (!Libs::LibKernel::Memory::TryWriteBacking(at, from + done, piece) &&
				    Libs::LibKernel::Memory::TryClampRangeSize(at, piece) == piece) {
					EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64
					     " size=0x%016" PRIx64 "\n",
					     at, piece);
				}
				done += piece;
			}
		}
	}
	{
		// The bytes are no longer on their way (as a download's publication): only this copy's
		// entries go, so ranges another download (or copy) still has in flight stay marked; only
		// those within the copy's bounds can have lost bytes.
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: job->copies) {
			const std::pair<uint64_t, uint64_t> range {buffer_address + copy.srcOffset, copy.size};
			const auto                          entry =
			    std::find(m_downloading_list.begin(), m_downloading_list.end(), range);
			if (entry != m_downloading_list.end()) {
				*entry = m_downloading_list.back();
				m_downloading_list.pop_back();
			}
			m_downloading_ranges.Subtract(range.first, range.second);
			// Copies on their way never overlap (FastReadback leaves out bytes on their way).
			m_fast_pending_ranges.Subtract(range.first, range.second);
		}
		for (const auto& [start, length]: m_downloading_list) {
			if (start < job_end && job_begin < start + length) {
				m_downloading_ranges.Add(start, length);
			}
		}
	}
	m_readback_queue->Done(ticket);
	auto& completions = g_fast_completions;
	completions.count.fetch_add(1, std::memory_order_relaxed);
	completions.wait_ns.fetch_add(waited_ns, std::memory_order_relaxed);
	for (auto longest = completions.longest_ns.load(std::memory_order_relaxed);
	     waited_ns > longest && !completions.longest_ns.compare_exchange_weak(
	                                longest, waited_ns, std::memory_order_relaxed);) {
	}
	completions.by[static_cast<size_t>(who)].fetch_add(1, std::memory_order_relaxed);
	Common::PerfStats::AddFastReadback(waited_ns, total);
}

bool BufferCache::FastReadback(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t window_begin,
                               uint64_t window_end, bool is_write, bool wait_here) {
	// Round 135: why nothing was copied, for ReadMemoryStep (readwait:writer) and the counts.
	m_fast_refusal      = FastRefusal::Unavailable;
	m_fast_writer_tick  = 0;
	m_fast_left         = 0;
	m_fast_refused_tick = 0;
	m_fast_current_tick = 0;
	m_fast_known_tick   = 0;
	if (m_readback_queue == nullptr || !m_readback_queue->Available() || !FastReadbackEnabled() ||
	    window_end <= window_begin) {
		return false;
	}
	const auto started = StallWatch::NowNs();
	auto&      stats   = g_fast_readback;
	stats.attempts++;
	const FastReadbackReport report {stats};
	// Between commands: every write marked so far has its command in a command buffer, and its
	// range the tick of that command buffer.
	WritesRecorded();

	const auto page_begin     = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end       = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	const auto buffer_address = buffer.CpuAddress();
	const auto buffer_end     = buffer_address + buffer.Size();
	// Image write-backs still to land put their bytes in guest memory after these (on the
	// priority thread): bytes under one stay for a later read.
	const auto writebacks      = m_texture_cache.PendingWriteBacks();
	const auto under_writeback = [&](uint64_t start, uint64_t end) {
		return std::ranges::any_of(writebacks, [&](const std::pair<uint64_t, uint64_t>& pending) {
			return pending.first < end && start < pending.first + pending.second;
		});
	};
	{
		// A download or an image write-back of the faulting pages on its way publishes its bytes
		// when it lands: the read waits for it the old way.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(page_begin, page_end - page_begin) ||
		    (!writebacks.empty() && under_writeback(page_begin, page_end))) {
			stats.downloading++;
			m_fast_refusal = FastRefusal::Downloading;
			return false;
		}
	}
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto known    = master.KnownGpuTick();
	const auto current  = m_scheduler.CurrentTick();
	m_fast_current_tick = current;
	m_fast_known_tick   = known;
	if (m_address_write_tick > known && !Bisect::FastReadbackAnyAddress()) {
		// A command still on the GPU stores through shader addresses, which mark no range: these
		// bytes may be among its stores, which a download behind it takes. (KYTY_BISECT's
		// fastread:any copies anyway, to measure what this costs.)
		stats.address++;
		m_fast_refusal = FastRefusal::Address;
		return false;
	}

	// The GPU-written ranges of the window's GPU-written pages, as a download takes them
	// (DownloadBufferWindow). One whose last writer has not finished stays for a later read, as
	// does one a download or an image write-back still on its way also holds: its publication
	// (on the priority thread) would land over these bytes after them. On the faulting pages
	// either sends the read the old way. So does one the GPU write bitmap tracks: its bits were
	// taken as downloaded with the window (RefineWithWriteBitmap).
	enum class Refusal : uint8_t { None, Recording, InFlight, Downloading, Other };
	auto                        refusal = Refusal::None;
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total = 0;
	uint64_t                    left  = 0;
	// Round 135: the faulting pages' refused ranges: the latest writer tick among them, the
	// latest among those the GPU is running, and whether one was refused for another reason.
	uint64_t faulting_tick     = 0;
	uint64_t faulting_inflight = 0;
	bool     faulting_blocked  = false;
	m_memory_tracker.ForEachDownloadRange<false>(
	    window_begin, window_end - window_begin, [&](uint64_t address, uint64_t bytes) noexcept {
		    std::shared_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    const auto tick   = m_write_ticks.Max(start, end);
			    auto       reason = Refusal::None;
			    if (start < buffer_address || end > buffer_end) {
				    reason = Refusal::Other;
			    } else if (m_downloading_ranges.Intersects(start, end - start) ||
			               (!writebacks.empty() && under_writeback(start, end))) {
				    reason = Refusal::Downloading;
			    } else if (tick >= current) {
				    reason = Refusal::Recording;
			    } else if (tick > known) {
				    reason = Refusal::InFlight;
			    }
			    if (reason == Refusal::None) {
				    copies.emplace_back(start - buffer_address, total, end - start);
				    // Packed ranges on separate cache lines, as the old path does.
				    total += Common::AlignUp(end - start, 64);
				    return;
			    }
			    const bool faulting = start < page_end && end > page_begin;
			    if (faulting) {
				    faulting_tick = std::max(faulting_tick, tick);
				    if (reason == Refusal::InFlight) {
					    faulting_inflight = std::max(faulting_inflight, tick);
				    } else {
					    faulting_blocked = true;
				    }
			    }
			    const bool bitmap =
			        m_write_bitmap != nullptr && m_bitmap_tracked.Intersects(start, end - start);
			    if (refusal == Refusal::None && (faulting || reason == Refusal::Other)) {
				    refusal = reason;
			    } else if (refusal == Refusal::None && bitmap) {
				    refusal = Refusal::Other;
			    }
			    left++;
		    });
	    });
	m_fast_refused_tick = faulting_tick;
	switch (refusal) {
		case Refusal::None: break;
		case Refusal::Recording:
			stats.recording++;
			m_fast_refusal = FastRefusal::Recording;
			return false;
		case Refusal::InFlight:
			stats.in_flight++;
			m_fast_refusal = FastRefusal::InFlight;
			// Only unfinished writers on the faulting pages: once the GPU has run the latest of
			// them, the next try can copy (readwait:writer).
			if (!faulting_blocked && faulting_inflight > known && faulting_inflight < current) {
				m_fast_writer_tick = faulting_inflight;
			}
			return false;
		case Refusal::Downloading:
			stats.downloading++;
			m_fast_refusal = FastRefusal::Downloading;
			return false;
		case Refusal::Other:
			stats.other++;
			m_fast_refusal = FastRefusal::Other;
			return false;
	}
	if (copies.empty()) {
		stats.nothing++;
		m_fast_refusal = FastRefusal::Nothing;
		m_fast_left    = static_cast<uint32_t>(std::min<uint64_t>(left, UINT32_MAX));
		return false;
	}
	if (total > m_readback_queue->Capacity()) {
		stats.too_big++;
		m_fast_refusal = FastRefusal::TooBig;
		return false;
	}
	// Round 152: every copy slot holds a copy not yet in guest memory: the oldest goes there first
	// (by this thread, if its reader has not got to it yet).
	if (m_readback_queue->Full()) {
		stats.full++;
		CompleteFastReadback(m_readback_queue->Oldest(), FastCompleter::GpuThread);
	}
	// After every tick the GPU is known to have finished, not only the copied ranges' writers (all
	// done by then): a command storing through shader addresses may have written them later. The
	// wait is free, the tick being done.
	ReadbackQueue::Job job;
	job.source         = buffer.Handle();
	job.buffer_address = buffer_address;
	job.total_size     = total;
	job.begin          = UINT64_MAX;
	job.end            = 0;
	for (const auto& copy: copies) {
		job.begin = std::min(job.begin, buffer_address + copy.srcOffset);
		job.end   = std::max(job.end, buffer_address + copy.srcOffset + copy.size);
	}
	job.copies        = copies;
	const auto ticket = m_readback_queue->Submit(std::move(job), known);
	if (!ticket.Valid()) {
		stats.other++;
		m_fast_refusal = FastRefusal::Other;
		return false;
	}
	{
		// As a download's (DownloadBufferWindow): the bytes are no longer waiting for a download
		// but on their way, until the copy's publication (CompleteFastReadback) puts them in guest
		// memory. A GPU write recorded since marks them written again.
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			const auto start = buffer_address + copy.srcOffset;
			m_gpu_modified_ranges.Subtract(start, copy.size);
			m_downloading_ranges.Add(start, copy.size);
			m_downloading_list.emplace_back(start, copy.size);
			m_fast_pending_ranges.Add(start, copy.size);
		}
	}
	m_fast_step    = true;
	m_fast_refusal = FastRefusal::None;
	stats.done++;
	stats.bytes += total;
	stats.ranges += copies.size();
	stats.left += left;
	if (!wait_here) {
		// Round 152: the reading thread waits for the copy and puts its bytes in guest memory
		// (ReadMemory), and its next step lifts the pages. A publication recorded after this one
		// (a download's, an image write-back's) waits for it on the priority thread first, so it
		// cannot land under these older bytes, even if the reading thread is held up.
		m_scheduler.DeferPriorityOperation(
		    [this, ticket] { CompleteFastReadback(ticket, FastCompleter::Priority); });
		m_step_ticket = ticket;
		stats.gpu_ns += StallWatch::NowNs() - started;
		return true;
	}
	CompleteFastReadback(ticket, FastCompleter::Inline);
	// A page under an image write-back still to land is lifted too: a guest read of its bytes,
	// not GPU-written ones, took them from guest memory before, as the clean-read emulation does.
	LiftCurrentPages(buffer, vaddr);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	stats.gpu_ns += StallWatch::NowNs() - started;
	return true;
}

uint64_t BufferCache::ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
                                     bool from_gpu_thread, bool async, bool writer_wait) {
	m_step_outcome = ReadOutcome::Direct;
	m_fast_refusal = FastRefusal::None;
	m_step_ticket  = {};
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return 0;
	}
	// Round 152: bytes of the faulting pages a fast readback has copied and not yet put in guest
	// memory (FastReadback): the pages stay protected until they are there. A game thread waits
	// for them (its own copy's or another read's) and asks again; the GPU thread puts them there
	// itself.
	if (m_readback_queue != nullptr && m_readback_queue->AnyPending()) {
		const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		for (auto ticket = m_readback_queue->PendingIn(first, last); ticket.Valid();
		     ticket      = m_readback_queue->PendingIn(first, last)) {
			if (!from_gpu_thread) {
				m_step_ticket  = ticket;
				m_step_outcome = ReadOutcome::FastWait;
				return 0;
			}
			CompleteFastReadback(ticket, FastCompleter::GpuThread);
		}
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Diagnostics: whether the bytes the faulting access touches were written by the GPU
	// (true sharing) or only share the page with bytes that were (false sharing).
	{
		static uint64_t exact = 0;
		static uint64_t shared = 0;
		const bool      dirty  = HasGpuDirtyBytes(vaddr, std::max<uint64_t>(size, 4));
		(dirty ? exact : shared)++;
		static uint32_t logged = 0;
		if (!dirty && logged < 32) {
			logged++;
			// Which bytes of the 4 KiB page the GPU did write, in 16-byte steps.
			std::string dirty_ranges;
			const auto  page  = vaddr & ~uint64_t {0xfff};
			uint64_t    begin = 0;
			bool        open  = false;
			for (uint64_t at = page; at <= page + 0x1000; at += 16) {
				const bool here = at < page + 0x1000 && HasGpuDirtyBytes(at, 16);
				if (here && !open) {
					begin = at;
					open  = true;
				} else if (!here && open) {
					dirty_ranges += fmt::format(" {:x}-{:x}", begin - page, at - page);
					open = false;
				}
			}
			LOGF("ReadMemory false sharing: vaddr=0x%016" PRIx64 " write=%d page dirty:%s\n",
			     vaddr, is_write ? 1 : 0, dirty_ranges.c_str());
		}
		if ((exact + shared) % 1024 == 0) {
			LOGF("ReadMemory faults: gpu_written=%" PRIu64 " false_sharing=%" PRIu64 "\n", exact,
			     shared);
		}
	}

	// The page is protected as GPU-written, but none of its bytes are waiting for a download:
	// the bytes the GPU wrote are elsewhere on the page, or were downloaded with another
	// page. The page is current, so lift its protection. Downloading the window instead drained
	// the GPU (~30 ms a fault) for guest reads of structs next to the command processor's
	// labels.
	// GPU write bitmap: pages of tracked bindings the GPU did not write lose their GPU-written
	// bytes here, so the window below downloads only what was written, or nothing.
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	const bool page_gpu   = m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin);
	// Round 135: the pages are neither GPU-written for the tracker nor hold GPU-written bytes,
	// bytes of a download on its way or bytes an image write-back still to land puts there:
	// another thread's read of the window made them current between this fault and this step
	// (game threads read the same results at the same time: "nothing to copy at once" was a
	// fifth of the fast readbacks in a race). The access needs nothing. Before, the window's
	// other GPU-written ranges were downloaded behind the renderer's queue and waited for, though
	// they were not this access's (KYTY_CLEAN_PAGE=old, cleanpage:old). Before the write
	// bitmap's refinement, which takes the window's written pages as downloaded.
	if (!page_gpu && !CleanPageOld()) {
		bool busy = false;
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			busy = HasGpuDirtyBytes(page_begin, page_end - page_begin) ||
			       m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
		}
		if (!busy) {
			const auto writebacks = m_texture_cache.PendingWriteBacks();
			busy =
			    std::ranges::any_of(writebacks, [&](const std::pair<uint64_t, uint64_t>& pending) {
				    return pending.first < page_end && page_begin < pending.first + pending.second;
			    });
		}
		if (!busy) {
			if (!from_gpu_thread && ReadaheadEnabled()) {
				// As the lifted pages below: the window's blocks stay learned while game threads
				// keep reading them.
				LearnReadahead(page_begin, page_end, true);
			}
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			m_step_outcome = ReadOutcome::Clean;
			static std::atomic<uint64_t> clean {0};
			if (const auto n = clean.fetch_add(1, std::memory_order_relaxed) + 1;
			    std::has_single_bit(n) && n >= 64) {
				LOGF("ReadMemory: %" PRIu64 " faults on pages another read had made current went "
				     "on at once (round 135; cleanpage:old waits as before)\n",
				     n);
			}
			return 0;
		}
	}
	if (uint64_t wait_tick = 0; RefineWithWriteBitmap(buffer, vaddr, size, async, wait_tick)) {
		m_step_outcome = ReadOutcome::Bitmap;
		return wait_tick;
	}
	if (page_gpu && !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		m_step_outcome = ReadOutcome::Lifted;
		// An asynchronous readback of these bytes may not be in guest memory yet: the access
		// must not see the older bytes, so it waits for that publication first.
		bool     downloading   = false;
		uint64_t download_tick = 0;
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			downloading = m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
			// Round 153: the downloads that cover these pages, not the latest of all (that one
			// publishes after them, often several command buffers later). liftwait:old (or
			// KYTY_LIFT_WAIT=old) waits for the latest, as before.
			if (downloading && !LiftWaitOld()) {
				for (const auto& batch: m_download_batches) {
					if (batch.begin < page_end && page_begin < batch.end) {
						download_tick = std::max(download_tick, batch.tick);
					}
				}
			}
		}
		if (downloading && download_tick == 0) {
			download_tick = m_last_async_download_tick;
		}
		if (downloading && download_tick != 0) {
			// Downloads are submitted as they are recorded (readahead's in its before-flush hook,
			// a game thread's below), so this tick should be submitted already. Were it not, it
			// would be submitted here: on the GPU thread without the hook, which must not run in
			// the middle of a command (it downloads ranges the command has yet to write).
			if (download_tick >= m_scheduler.CurrentTick()) {
				if (from_gpu_thread) {
					m_scheduler.Wait(download_tick);
				} else {
					m_scheduler.Flush();
				}
			}
			if (async) {
				m_step_outcome = ReadOutcome::LiftWaited;
				return download_tick;
			}
			m_scheduler.WaitPriorityOperations(download_tick);
		}
		if (async && ReadbackWindow() != 0) {
			// The whole download window, as the synchronous path lifts it: page by page, each
			// page of the window faulted on its own and went through the GPU thread again.
			if (!from_gpu_thread && ReadaheadEnabled()) {
				// Its bytes were in memory (a readahead, or a window downloaded before): the
				// window's blocks stay learned while game threads keep reading them (its other
				// pages are lifted now, so they fault no more).
				const auto window = ReadbackWindow();
				const auto begin  = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
				const auto end    = std::min(begin + window, buffer.CpuAddress() + buffer.Size());
				LearnReadahead(begin, end, true);
			}
			LiftCurrentPages(buffer, vaddr);
		} else if (!from_gpu_thread && ReadaheadEnabled()) {
			LearnReadahead(page_begin, page_end, true);
		}
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		static uint64_t lifted = 0;
		if (++lifted % 1024 == 1) {
			LOGF("ReadMemory: lifted %" PRIu64 " protections of pages without GPU-written "
			     "bytes\n",
			     lifted);
		}
		return 0;
	}

	// Diagnostics: the GPU-written ranges (whole) on the faulting page of the first downloads.
	{
		static uint32_t logged = 0;
		if (logged < 96) {
			logged++;
			std::string ranges;
			uint32_t    count = 0;
			m_gpu_modified_ranges.ForEachOverlapping(
			    page_begin, page_end - page_begin, [&](uint64_t begin, uint64_t end) {
				    if (count++ < 4) {
					    ranges += fmt::format(" {:x}+{:x}", begin, end - begin);
				    }
			    });
			LOGF("ReadMemory download: vaddr=0x%016" PRIx64
			     " write=%d gpu_thread=%d ranges=%u:%s\n",
			     vaddr, is_write ? 1 : 0, from_gpu_thread ? 1 : 0, count, ranges.c_str());
		}
	}

	// Widen nearby CPU reads so they share one GPU drain. Upstream (0ce84e59) reads only the
	// requested pages; Wolverine's game threads read ~2,500 GPU-written pages a frame at the Leap
	// Attack prompt, each its own fault and download that way (7.5 fps against 20.3).
	auto window_begin = vaddr;
	auto window_end   = vaddr + size;
	if (const auto window = ReadbackWindow(); window != 0) {
		const auto buffer_begin = buffer.CpuAddress();
		const auto buffer_end   = buffer_begin + buffer.Size();
		window_begin = std::max(Common::AlignDown(vaddr, window), buffer_begin);
		window_end   = std::min(std::max(window_begin + window, vaddr + size), buffer_end);
	}

	if (!from_gpu_thread && ReadaheadEnabled()) {
		LearnReadahead(window_begin, window_end, false);
	}
	Timeline::Mark("readmem", vaddr, from_gpu_thread ? 1u : 0u);
	// Bytes the GPU has finished writing: copied on the readback queue at once, instead of behind
	// what the renderer's queue still has to run. The GPU thread's own reads too where it allows
	// (GpuThreadFastReads): the copy is done when FastReadback returns. A game thread's is left
	// to it (round 152): ReadMemory waits for it, then asks again.
	const bool gpu_thread_fast =
	    from_gpu_thread && m_gpu_thread_fast_reads && !Bisect::FlipCheckOld();
	if ((async || gpu_thread_fast) &&
	    FastReadback(buffer, vaddr, size, window_begin, window_end, is_write,
	                 from_gpu_thread || FastReadbackWaitsOnGpuThread())) {
		m_step_outcome = ReadOutcome::Fast;
		if (gpu_thread_fast) {
			static std::atomic<uint64_t> fast_reads {0};
			const auto                   n = fast_reads.fetch_add(1, std::memory_order_relaxed) + 1;
			if (std::has_single_bit(n)) {
				LOGF("ReadMemory: %" PRIu64 " reads of the GPU thread copied on the readback "
				     "queue (round 80; flipcheck:old waits for the GPU as before)\n",
				     n);
			}
		}
		return 0;
	}
	// Round 135 (KYTY_READ_WAIT=writer, readwait:writer): the faulting pages' bytes are only
	// waiting for writers the GPU is running: the caller waits for the latest one's command
	// buffer, which the GPU runs before the ones recorded since, and asks again; its bytes are
	// then copied at once. A download here would come after all of those.
	// Not with pages the write bitmap tracks in the window: the refinement above took their
	// written pages as downloaded with it, so they are downloaded now.
	if (async && writer_wait && m_fast_refusal == FastRefusal::InFlight &&
	    m_fast_writer_tick != 0 && [&] {
		    std::shared_lock lock(m_dirty_ranges_mutex);
		    return m_write_bitmap == nullptr ||
		           !m_bitmap_tracked.Intersects(window_begin, window_end - window_begin);
	    }()) {
		m_step_outcome = ReadOutcome::WaitWriter;
		return m_fast_writer_tick;
	}
	// The downloads game threads wait for, by writer (the fast readbacks are not among them),
	// with the bytes downloaded (round 135; the window's size before), and the time and the GPU's
	// progress from before the download (a synchronous one, or a ring that wraps, waits).
	uint64_t report_known = 0;
	uint64_t report_ns    = 0;
	if (!from_gpu_thread) {
		auto& master = m_scheduler.GetMasterSemaphore();
		master.Refresh();
		report_known = master.KnownGpuTick();
		report_ns    = StallWatch::NowNs();
	}
	const auto report_download = [&](uint64_t bytes) {
		if (!from_gpu_thread) {
			ReportGameThreadDownload(vaddr, bytes, report_known, report_ns);
		}
	};
	uint64_t downloaded = 0;
	m_step_outcome      = ReadOutcome::Nothing;
	if (async) {
		// The download publishes when its tick completes; the page stays protected until the
		// caller's next step finds its bytes downloaded and lifts it.
		auto tick = m_scheduler.CurrentTick();
		if (DownloadBufferMemory<true>(buffer, window_begin, window_end - window_begin,
		                               &downloaded)) {
			m_step_outcome = ReadOutcome::Download;
			m_step_bytes   = downloaded;
			report_download(downloaded);
			// The tick of the last copy, not of the first: a download ring that wraps within this
			// recording submits it (StreamBuffer::WaitPendingOperations), and the later copies
			// and their publication then land in the next tick. A caller waiting for the earlier
			// tick went on before its bytes were published, and the GPU thread then lifted the
			// page's protection over the older bytes (Jetsku bf8d6148, same idea for writers).
			// KYTY_READBACK_RETICK=0 keeps the tick from before the download.
			if (ReadbackRetick()) {
				tick = m_scheduler.CurrentTick();
			}
			m_scheduler.Flush();
			// Readahead's downloads, recorded as Flush submitted, can be in a later tick (a
			// download ring that wraps there): the latest one is kept.
			m_last_async_download_tick = std::max(m_last_async_download_tick, tick);
			return tick;
		}
	} else if (DownloadBufferMemory<false>(buffer, window_begin, window_end - window_begin,
	                                       &downloaded)) {
		m_step_outcome = ReadOutcome::Download;
		m_step_bytes   = downloaded;
		report_download(downloaded);
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	return 0;
}

uint32_t BufferCache::MakeResident(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr > UINT64_MAX - size) {
		return 0;
	}
	uint32_t   made  = 0;
	const auto first = vaddr & ~(CACHING_PAGESIZE - 1u);
	const auto last  = (vaddr + size - 1u) & ~(CACHING_PAGESIZE - 1u);
	for (uint64_t page = first; page <= last; page += CACHING_PAGESIZE) {
		if (!GuestRange {page, CACHING_PAGESIZE}.Valid()) {
			break;
		}
		const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
		if (owner != nullptr && *owner) {
			// Read through the page table alone, it would age out of the cache.
			TouchBuffer(m_slot_buffers[*owner]);
			continue;
		}
		if (Libs::LibKernel::Memory::TryClampRangeSize(page, CACHING_PAGESIZE) !=
		    CACHING_PAGESIZE) {
			continue;
		}
		(void)FindBuffer(page, CACHING_PAGESIZE);
		SynchronizeBuffersInRange(page, CACHING_PAGESIZE);
		made++;
	}
	return made;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	{
		// The new buffer holds the old one's GPU-written bytes once this tick has run: a fast
		// readback of them (FastReadback) waits for it.
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_write_ticks.Assign(overlap.CpuAddress(), overlap.CpuAddress() + overlap.Size(),
		                     m_scheduler.CurrentTick());
	}
	NoteBufferEvent(BufferEventKind::Joined, overlap);
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	if (vaddr < CACHING_PAGESIZE) {
		// Guest page 0 is never mapped; a request here is a garbage or null descriptor that
		// slipped past the null checks. The memory tracker cannot hold address 0, so name the
		// caller now rather than in the garbage collector later.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 8) {
			LOGF("BufferCache: buffer requested in guest page 0: vaddr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n%s",
			     vaddr, size, Common::HostBacktrace().c_str());
		}
	}
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	// A mirror far larger than the request means the merge, not a descriptor, chose this
	// size: buffers only ever grow here, so one that swallows its neighbours as a title
	// streams across its heap ends up spanning it. Name the request that did it.
	if (overlap.end - overlap.begin >= 256ull * 1024 * 1024) {
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 12) {
			size_t absorbed = 0;
			for (auto it = overlap.first; it != overlap.last; ++it) {
				absorbed++;
			}
			LOGF("BufferCache: large mirror %" PRIu64 "MB at 0x%016" PRIx64
			     " for request 0x%016" PRIx64 "+%" PRIu64 "KB, absorbing %zu buffers, leap=%d\n",
			     (overlap.end - overlap.begin) >> 20u, overlap.begin, vaddr, size >> 10u, absorbed,
			     overlap.has_stream_leap ? 1 : 0);
		}
	}
	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	// Round 115, KYTY_ZERO_NEW_MEMORY=1: zeros where neither an older buffer's copy nor an upload
	// puts bytes, instead of what the driver's memory held there.
	if (Bisect::ZeroNewMemory()) {
		m_slot_buffers[id].Fill(0, overlap.end - overlap.begin, 0);
	}
	uint32_t absorbed = 0;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
		absorbed++;
	}
	Register(id);
	NoteBufferEvent(BufferEventKind::Created, m_slot_buffers[id], vaddr, size, absorbed);
	// KYTY_HOST_IMPORT_PROBE=1 (round 116), once.
	HostImportProbe::MaybeRun(m_graphics, overlap.begin, overlap.end - overlap.begin);
	return id;
}

namespace {
// What the per-draw synchronization is asked to cover. A giant raw descriptor makes every
// draw walk the tracker for hundreds of regions; the histogram says whether that is the case.
struct SyncStats {
	uint64_t calls      = 0;
	uint64_t uploads    = 0;
	uint64_t bytes      = 0;
	uint64_t buckets[6] = {}; // <=4K, <=64K, <=1M, <=16M, <=256M, larger
	uint64_t written_buckets[6] = {};
	uint64_t written    = 0;
	double   walk_s     = 0.0;
};
SyncStats g_sync_stats;

void RecordSync(uint64_t size, bool is_written, bool uploaded, double walk_s) {
	auto& st = g_sync_stats;
	st.calls++;
	st.uploads += uploaded ? 1 : 0;
	st.bytes += size;
	st.written += is_written ? 1 : 0;
	st.walk_s += walk_s;
	const int bucket = size <= 0x1000 ? 0 : size <= 0x10000 ? 1 : size <= 0x100000 ? 2
	                 : size <= 0x1000000 ? 3 : size <= 0x10000000 ? 4 : 5;
	st.buckets[bucket]++;
	if (is_written) {
		st.written_buckets[bucket]++;
	}
	if (st.calls % 200000 == 0) {
		LOGF("SyncStats: calls=%" PRIu64 " uploads=%" PRIu64 " written=%" PRIu64
		     " avg_bytes=%" PRIu64 " walk_total=%.1fms sizes(<=4K,64K,1M,16M,256M,>)=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 " written_sizes=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
		     st.calls, st.uploads, st.written, st.bytes / st.calls, st.walk_s * 1000.0,
		     st.buckets[0], st.buckets[1], st.buckets[2], st.buckets[3], st.buckets[4],
		     st.buckets[5], st.written_buckets[0], st.written_buckets[1], st.written_buckets[2],
		     st.written_buckets[3], st.written_buckets[4], st.written_buckets[5]);
		st = {};
	}
}
} // namespace

// Round 111 (Bisect::AsyncMemorySync): whether the GPU thread's syncs leave the page protections
// they ask for to the page manager's worker (PageManager::DeferScope). Their uploads then read
// guest memory on the worker too, after those protections (UploadCopies).
static bool DeferProtections(const CommandScheduler& /*scheduler*/) {
	return t_guest_gpu_thread && Bisect::AsyncMemorySync();
}

// Round 172, diagnostics: how many of the GPU thread's buffer syncs repeat one of the same range
// (and kind) made since the last time pages became CPU-dirty anywhere, which a memo could skip.
// A "Buffer syncs" line every 10 s.
static void NoteSyncRepeat(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer) {
	struct Entry {
		uint64_t vaddr = 0;
		uint64_t size  = 0;
		uint64_t epoch = 0;
		uint32_t kind  = 0;
	};
	struct Stats {
		std::array<Entry, 1024>               table {};
		uint64_t                              calls            = 0;
		uint64_t                              written          = 0;
		uint64_t                              texel            = 0;
		uint64_t                              repeats          = 0;
		uint64_t                              repeated_written = 0;
		std::chrono::steady_clock::time_point last {};
	};
	// The GPU thread's alone (its callers check): no table in every thread's storage.
	static Stats stats;
	const auto   kind  = (is_written ? 1u : 0u) | (is_texel_buffer ? 2u : 0u);
	const auto   epoch = g_cpu_dirty_epoch.load(std::memory_order_relaxed);
	const auto slot  = ((vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull) ^ kind) % stats.table.size();
	auto&      entry = stats.table[slot];
	stats.calls++;
	stats.written += is_written ? 1u : 0u;
	stats.texel += is_texel_buffer ? 1u : 0u;
	if (entry.vaddr == vaddr && entry.size == size && entry.kind == kind && entry.epoch == epoch) {
		stats.repeats++;
		stats.repeated_written += is_written ? 1u : 0u;
	}
	entry = {vaddr, size, epoch, kind};
	if ((stats.calls & 4095u) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (stats.last.time_since_epoch().count() == 0) {
		stats.last = now;
		return;
	}
	const double seconds = std::chrono::duration<double>(now - stats.last).count();
	if (seconds < 10.0) {
		return;
	}
	LOGF("Buffer syncs (%.1f s): %.0f a second (%.0f of written ranges, %.0f of texel buffers); "
	     "%.0f a second repeat one of the same range made since pages last became CPU-dirty "
	     "(%.0f of them written)\n",
	     seconds, static_cast<double>(stats.calls) / seconds,
	     static_cast<double>(stats.written) / seconds, static_cast<double>(stats.texel) / seconds,
	     static_cast<double>(stats.repeats) / seconds,
	     static_cast<double>(stats.repeated_written) / seconds);
	stats.calls = stats.written = stats.texel = stats.repeats = stats.repeated_written = 0;
	stats.last                                                                         = now;
}

// gt7, KYTY_BUFCACHE_V2=1 (off by default). Written buffer syncs of ranges that were last synced
// as written and whose GPU-dirty state has not been cleared since: a direct-mapped memo of
// (range, g_gpu_clear_epoch) on the GPU thread, the only thread that changes GPU-dirty bits.
static bool BufCacheV2() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFCACHE_V2");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

namespace {
struct WrittenMemo {
	uint64_t vaddr = 0;
	uint64_t size  = 0;
	uint64_t epoch = 0; // 0 never matches: the epoch starts at 1
};
constexpr size_t kWrittenMemoSlots = 256;
// GPU thread only.
std::array<WrittenMemo, kWrittenMemoSlots> g_written_memo {};

size_t WrittenMemoSlot(uint64_t vaddr, uint64_t size) {
	return static_cast<size_t>(((vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull)) %
	                           kWrittenMemoSlots);
}
} // namespace

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	KYTY_PROFILER_FUNCTION();
	const GpuPhases::Scope phase(GpuPhases::Phase::BufferSync);
	if (t_guest_gpu_thread) {
		NoteSyncRepeat(vaddr, size, is_written, is_texel_buffer);
	}
	const bool written_memo = BufCacheV2() && is_written && !is_texel_buffer && t_guest_gpu_thread;
	if (written_memo) {
		const auto& memo = g_written_memo[WrittenMemoSlot(vaddr, size)];
		if (memo.vaddr == vaddr && memo.size == size &&
		    memo.epoch == g_gpu_clear_epoch.load(std::memory_order_acquire)) {
			// Still all GPU-dirty: a full sync would find no CPU-dirty page to copy and change no
			// bit nor protection.
			RecordSync(size, true, false, 0.0);
			return false;
		}
	}
	// Most calls (~22k a frame in a race) read a range whose tracker regions are all clean: what
	// ForEachUploadRange finds for it is nothing to copy. Answer from the lock-free summary bitmap
	// without the protection defer scope, the copy list, the lambdas and their profiler zones.
	if (!is_written && !is_texel_buffer && g_tracker_skip_clean && g_tracker_bitmap &&
	    m_memory_tracker.IsRangeCpuCleanHint(vaddr, size)) {
		RecordSync(size, false, false, 0.0);
		return false;
	}
	// gt7-caches-1112, KYTY_SYNC_SKIP=1: a written sync of a range whose pages are all GPU-dirty
	// already has nothing to upload (no page can be CPU-dirty or hot) and sets bits and
	// protections that are set: skipped. Every 64th hit runs the full sync to measure the saving.
	uint64_t skip_start = 0;
	if (is_written && !is_texel_buffer && t_guest_gpu_thread &&
	    Libs::Graphics::Speculation::SyncSkipEnabled()) {
		namespace Spec = Libs::Graphics::Speculation;
		skip_start     = GpuPhases::Now();
		if (m_memory_tracker.IsRangeAllGpuDirtyHint(vaddr, size)) {
			if (!Spec::CalibrateSyncNext()) {
				Spec::CacheAdd(Spec::CacheCounter::SyncSkipHit);
				Spec::AddSyncCheckCycles(GpuPhases::Now() - skip_start);
				RecordSync(size, true, false, 0.0);
				return false;
			}
		} else {
			Spec::CacheAdd(Spec::CacheCounter::SyncSkipMiss);
			skip_start = 0;
		}
	}
	// Timing every call cost ~2 % of the GPU thread (two performance-counter reads); SyncStats
	// times the tracker walk only with KYTY_SYNC_STATS=1.
	static const bool timed = std::getenv("KYTY_SYNC_STATS") != nullptr;
	Common::Timer               walk_timer;
	if (timed) {
		walk_timer.Start();
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	{
		KYTY_PROFILER_BLOCK("Sync::Tracker");
		// Round 111: the tracker's protections only (a texel buffer's image sync below keeps its
		// own).
		const PageManager::DeferScope defer(m_memory_tracker.Pages(),
		                                    DeferProtections(m_scheduler));
		m_memory_tracker.ForEachUploadRange(
		    vaddr, size, is_written,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    copies.emplace_back(total_size, buffer.Offset(address), bytes);
			    total_size += bytes;
		    },
		    [&]() noexcept {
			    KYTY_PROFILER_BLOCK("Sync::Upload");
			    source = UploadCopies(buffer, copies, total_size);
		    });
	}
	RecordSync(size, is_written, static_cast<bool>(source), timed ? walk_timer.GetTimeS() : 0.0);
	if (written_memo) {
		// The whole range is GPU-dirty now (ChangeState<Gpu, true> over every region of it).
		auto& memo = g_written_memo[WrittenMemoSlot(vaddr, size)];
		memo       = {vaddr, size, g_gpu_clear_epoch.load(std::memory_order_acquire)};
	}
	if (source) {
		RecordUpload(buffer, copies, total_size, source);
	}
	if (skip_start != 0) {
		// A calibration: the full sync of a range the skip would have passed.
		Libs::Graphics::Speculation::AddSyncFullCycles(GpuPhases::Now() - skip_start);
	}
	if (is_texel_buffer && !is_written) {
		KYTY_PROFILER_BLOCK("Sync::TexelImage");
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::RecordUpload(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                               uint64_t total_size, vk::Buffer source) {
	if (m_metadata_watch_count.load(std::memory_order_relaxed) != 0) {
		// Bytes the CPU wrote, uploaded over what a fast-clear check left there.
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			NoteMetadataWrite(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
	}
	KYTY_PROFILER_BLOCK("Sync::Copy");
	m_scheduler.NoteTransfer(StallWatch::Transfer::BufferUpload, total_size);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Recorder();
	if (Bisect::LeanCopies() && GuestGpu::IsGpuThread()) {
		// Round 167 (copies:lean): in the command buffer's batch of copies, after the GPU work
		// before it (draws may still read the bytes it replaces).
		native.copyBufferBatched(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                         copies.data(), true);
		return;
	}
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()), copies.data());
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

// Reads guest memory for an upload. The guest can unmap memory a cached buffer still covers: AMM
// maps streaming textures in blocks, and a texture upload once read one byte past the end of the
// block it had mapped. Such bytes read as zero, like ObtainBufferForImage's direct path.
static void ReadGuestForUpload(uint8_t* destination, uint64_t address, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadBacking(address, destination, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(address, destination, size)) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: upload of unmapped guest memory 0x%016" PRIx64 " size=0x%" PRIx64
			     " reads as zero\n",
			     address, size);
		}
		std::memset(destination, 0, static_cast<size_t>(size));
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Profiler::Add(Profiler::Counter::UploadBytes, static_cast<int64_t>(total_size));
	GpuPhases::NoteUpload(total_size, copies.size());

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		// Reading guest memory into staging was ~28 MB a frame of memcpy on the GPU thread in
		// the jungle (~8,500 uploads). With a recording thread the reads run there instead, in
		// the recorded command stream right before the copies that use them, which the caller
		// records next; the staging memory must be host-coherent, since nothing flushes it after.
		// A read sees the guest bytes of slightly later than the upload decision: newer CPU
		// writes re-dirty their pages and are uploaded again, as before. Off by default: no
		// measurable gain in the jungle (16.3 fps from 140 to 240 s with and without, 2026-10-07),
		// and writes that bypass tracking land in the upload (two strict cache tests fail).
		// KYTY_UPLOAD_READS_ON_RECORD_THREAD=1 turns it on.
		static const bool deferred = [] {
			const char* value = std::getenv("KYTY_UPLOAD_READS_ON_RECORD_THREAD");
			return value != nullptr && value[0] == '1';
		}();
		// Round 111, Bisect::AsyncMemorySync: the reads run on the page manager's worker, after
		// the protections deferred before them (these pages' among them). The GPU thread waits
		// for them before its own stores to guest memory (FinishDeferredMemoryWork) and the
		// command buffer is submitted after them, so the upload holds the bytes as they were
		// before anything the guest can do once it sees this command done.
		auto&      pages      = m_memory_tracker.Pages();
		const bool async_sync = t_guest_gpu_thread && Bisect::AsyncMemorySync();
		if (async_sync && m_staging_buffer.IsCoherent()) {
			std::vector<UploadRead> reads;
			reads.reserve(copies.size());
			for (auto& copy: copies) {
				reads.push_back(
				    {mapped + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size});
				copy.srcOffset += base_offset;
			}
			pages.PostDeferred([reads = std::move(reads)] {
				for (const auto& read: reads) {
					ReadGuestForUpload(read.destination, read.address, read.size);
				}
			});
			m_staging_buffer.Commit();
			return m_staging_buffer.Handle();
		}
		if (deferred && m_staging_buffer.IsCoherent() &&
		    m_scheduler.Current().Recorder().Threaded()) {
			const auto protections = pages.DeferredIssued();
			m_upload_reads.clear();
			for (auto& copy: copies) {
				m_upload_reads.push_back(
				    {mapped + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size});
				copy.srcOffset += base_offset;
			}
			const auto* reads = m_scheduler.Stash(m_upload_reads.data(), m_upload_reads.size());
			const auto  count = m_upload_reads.size();
			auto*       deferring = &pages;
			m_scheduler.Record([reads, count, protections, deferring](vk::CommandBuffer) {
				if (protections != 0) {
					deferring->WaitDeferred(protections);
				}
				for (size_t i = 0; i < count; i++) {
					ReadGuestForUpload(reads[i].destination, reads[i].address, reads[i].size);
				}
			});
			m_staging_buffer.Commit();
			return m_staging_buffer.Handle();
		}
		// Read here: after the page protections deferred so far (round 111).
		pages.FinishDeferred();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			ReadGuestForUpload(mapped + copy.srcOffset, address, copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	KYTY_PROFILER_BLOCK("BufferCache::UploadCopies(temporary)");
	m_memory_tracker.Pages().FinishDeferred();
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		ReadGuestForUpload(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool needs_device_address) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (auto& stream = ActiveStream(); !is_written && size <= CACHING_PAGESIZE &&
	                                   (!needs_device_address || stream.HasDeviceAddress()) &&
	                                   !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	                                   m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = stream.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			stream.Commit();
			return {&stream, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	buffer.last_use_tick = m_scheduler.CurrentTick();
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		MarkGpuWritten(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferWritten(uint64_t vaddr, uint64_t size,
                                                              uint64_t written_vaddr,
                                                              uint64_t written_size, BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid() ||
	    written_vaddr < vaddr || written_size > size || written_vaddr - vaddr > size - written_size) {
		EXIT("BufferCache: invalid written-range buffer request\n");
	}
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	buffer.last_use_tick = m_scheduler.CurrentTick();
	TouchBuffer(buffer);
	// The whole binding is current on the GPU, as for a written buffer; only the bytes the
	// stores can reach become GPU-written.
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	if (written_size != 0) {
		(void)SynchronizeBuffer(buffer, written_vaddr, written_size, true, false);
		MarkGpuWritten(written_vaddr, written_size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

namespace {
constexpr uint64_t WriteBitmapPageBits = 12;
constexpr uint64_t WriteBitmapPage     = uint64_t {1} << WriteBitmapPageBits;
constexpr uint64_t WriteBitmapLimit    = uint64_t {1} << 40u; // guest addresses it covers
constexpr uint64_t WriteBitmapBytes    = WriteBitmapLimit / WriteBitmapPage / 8u;
constexpr uint64_t WriteBitmapRegionBits = 22; // ticks per 4 MiB: a multiple of a bitmap word's span

// KYTY_WRITE_BITMAP_MIN_KB (default 1024): smaller unbounded written bindings keep downloading
// whole, as before; the bitmap pays off where a binding spans much more than it writes.
uint64_t WriteBitmapMinBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP_MIN_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {1024}) * 1024u;
	}();
	return bytes;
}
} // namespace

vk::DeviceAddress BufferCache::WriteBitmapAddress() {
	if (!ShaderRecompiler::IR::WriteBitmapEnabled() || m_write_bitmap_failed) {
		return 0;
	}
	if (m_write_bitmap == nullptr) {
		m_write_bitmap = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    WriteBitmapBytes);
		// The host reads and clears bits in place, so the memory must be host-coherent.
		if (!m_write_bitmap->IsCoherent() || m_write_bitmap->Mapped().size() < WriteBitmapBytes ||
		    !m_write_bitmap->HasDeviceAddress()) {
			LOGF("BufferCache: GPU write bitmap unavailable (no host-coherent device memory)\n");
			m_write_bitmap.reset();
			m_write_bitmap_failed = true;
			return 0;
		}
		std::memset(m_write_bitmap->Mapped().data(), 0, static_cast<size_t>(WriteBitmapBytes));
		LOGF("BufferCache: GPU write bitmap on (%" PRIu64 " MB, pages of written bindings of at "
		     "least %" PRIu64 " KB)\n",
		     WriteBitmapBytes >> 20u, WriteBitmapMinBytes() >> 10u);
	}
	return m_write_bitmap->BufferDeviceAddress();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferTracked(uint64_t vaddr, uint64_t size,
                                                              BufferId id) {
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	buffer.last_use_tick = m_scheduler.CurrentTick();
	(void)SynchronizeBuffer(buffer, vaddr, size, true, false);
	MarkGpuWritten(vaddr, size, true);
	return {&buffer, buffer.Offset(vaddr)};
}

bool BufferCache::RefineWithWriteBitmap(const Buffer& buffer, uint64_t vaddr, uint64_t size,
                                        bool async, uint64_t& wait_tick) {
	if (m_write_bitmap == nullptr) {
		return false;
	}
	// The readback window ReadMemoryStep downloads (see there).
	auto begin = vaddr;
	auto end   = vaddr + size;
	if (const auto window = ReadbackWindow(); window != 0) {
		begin = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
		end   = std::min(std::max(begin + window, vaddr + size), buffer.CpuAddress() + buffer.Size());
	}
	begin = Common::AlignDown(begin, WriteBitmapPage);
	end   = Common::AlignUp(end, WriteBitmapPage);
	std::unique_lock lock(m_dirty_ranges_mutex);
	if (m_bitmap_tracked.Empty() || !m_bitmap_tracked.Intersects(begin, end - begin)) {
		return false;
	}
	// The bits are final only once no tracked write to these regions is in flight; that also
	// keeps the GPU's atomic ORs off the words the host clears below.
	uint64_t tick = 0;
	for (auto region = begin >> WriteBitmapRegionBits; region <= (end - 1) >> WriteBitmapRegionBits;
	     region++) {
		if (const auto found = m_bitmap_ticks.find(region); found != m_bitmap_ticks.end()) {
			tick = std::max(tick, found->second);
		}
	}
	if (tick != 0 && !m_scheduler.IsFree(tick)) {
		if (!async) {
			return false; // the GPU thread downloads the window as before
		}
		lock.unlock();
		if (tick >= m_scheduler.CurrentTick()) {
			m_scheduler.Flush();
		}
		wait_tick = tick;
		return true;
	}
	// Only GPU-written pieces of the window are looked at, and unwritten pages leave the set in
	// runs: page-by-page set operations cost the GPU thread more than the downloads they saved.
	auto*                                      words = reinterpret_cast<uint32_t*>(m_write_bitmap->Mapped().data());
	std::vector<std::pair<uint64_t, uint64_t>> pieces;
	m_gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](uint64_t from, uint64_t to) {
		pieces.emplace_back(from, to);
	});
	std::vector<std::pair<uint64_t, uint64_t>> unwritten; // runs of whole pages
	uint64_t                                   dropped = 0;
	for (const auto [from, to]: pieces) {
		m_bitmap_tracked.ForEachInRange(from, to - from, [&](uint64_t tb, uint64_t te) {
			for (auto page = Common::AlignUp(tb, WriteBitmapPage);
			     page + WriteBitmapPage <= te && page + WriteBitmapPage <= WriteBitmapLimit;
			     page += WriteBitmapPage) {
				const auto index = page >> WriteBitmapPageBits;
				auto&      word  = words[index >> 5u];
				const auto bit   = uint32_t {1} << (index & 31u);
				if ((word & bit) == 0) {
					if (!unwritten.empty() && unwritten.back().second == page) {
						unwritten.back().second = page + WriteBitmapPage;
					} else {
						unwritten.emplace_back(page, page + WriteBitmapPage);
					}
					dropped++;
				} else {
					word &= ~bit; // downloaded with the window now
					m_bitmap_written_pages++;
				}
			}
		});
	}
	for (const auto [from, to]: unwritten) {
		m_gpu_modified_ranges.Subtract(from, to - from);
	}
	m_bitmap_dropped_pages += dropped;
	static uint64_t calls = 0;
	if (++calls % 4096 == 1) {
		LOGF("WriteBitmap: %" PRIu64 " refinements, %" PRIu64 " pages dropped unwritten, %" PRIu64
		     " pages written\n",
		     calls, m_bitmap_dropped_pages, m_bitmap_written_pages);
	}
	return false;
}

// Diagnostics: which guest shader last marked a range GPU-written (0: a copy, fill or label
// write), so that the downloads game threads wait for can be put down to their writers.
namespace {
struct GpuWriter {
	uint64_t end     = 0;
	uint64_t shader  = 0;
	uint64_t tick    = 0; // the command buffer the write was recorded in
	uint64_t when_ns = 0; // when it was recorded
};
std::mutex                     g_writers_mutex;
std::map<uint64_t, GpuWriter>  g_writers;

struct WriterDownloads {
	uint64_t count = 0;
	uint64_t bytes = 0;
	uint64_t first = 0;
	// Downloads whose writer record was found: how long after the write they came, and how many
	// of them came after the GPU had run the write (readahead's case).
	uint64_t aged    = 0;
	uint64_t age_ns  = 0;
	uint64_t done    = 0;
};
std::unordered_map<uint64_t, WriterDownloads> g_writer_downloads;
uint64_t                                      g_game_downloads = 0;
} // namespace

// Whether GpuWritersOf may be asked (round 80): its records were made for every GPU write, a map
// insertion under a lock and a clock read each, ~1 % of the GPU thread in a race, for logs that
// only the frame captures (F9), KYTY_WATCH_SHADER and KYTY_TRACE_IMAGES write, and the game
// threads' downloads by writer. KYTY_GPU_WRITERS=1 keeps them always (a capture then also names
// the writers of earlier frames).
static bool GpuWriterRecordsWanted() {
	static const bool always = [] {
		const auto set = [](const char* name) {
			const char* value = std::getenv(name);
			return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
		};
		return set("KYTY_GPU_WRITERS") || set("KYTY_WATCH_SHADER") || set("KYTY_TRACE_IMAGES");
	}();
	return always || FrameGraph::Logging();
}

void BufferCache::RecordGpuWriter(uint64_t vaddr, uint64_t size, uint64_t shader, uint64_t tick) {
	// Records stop being made outside the stretches that want them: those of a stretch are
	// dropped when it ends, so nothing names old writers later.
	static std::atomic<bool> recording {false};
	if (!GpuWriterRecordsWanted()) {
		if (recording.exchange(false, std::memory_order_relaxed)) {
			std::lock_guard lock(g_writers_mutex);
			g_writers.clear();
		}
		return;
	}
	recording.store(true, std::memory_order_relaxed);
	const auto      now = StallWatch::NowNs();
	std::lock_guard lock(g_writers_mutex);
	if (g_writers.size() >= 65536) {
		g_writers.clear();
	}
	g_writers[vaddr] = {vaddr + size, shader, tick, now};
}

std::string BufferCache::GpuWritersOf(uint64_t vaddr, uint64_t size) {
	std::lock_guard       lock(g_writers_mutex);
	std::vector<uint64_t> shaders;
	// Ranges are keyed by their start; one that starts below vaddr may still reach into it.
	constexpr uint64_t reach = uint64_t {1} << 28u;
	for (auto it = g_writers.lower_bound(vaddr > reach ? vaddr - reach : 0);
	     it != g_writers.end() && it->first < vaddr + size && shaders.size() < 4; ++it) {
		if (it->second.end <= vaddr) continue;
		if (std::ranges::find(shaders, it->second.shader) == shaders.end()) {
			shaders.push_back(it->second.shader);
		}
	}
	std::string text;
	for (const auto shader: shaders) {
		text += text.empty() ? "" : ",";
		text += shader == 0 ? std::string("copy") : fmt::format("{:016x}", shader);
	}
	return text;
}

void BufferCache::ReportGameThreadDownload(uint64_t vaddr, uint64_t bytes, uint64_t done_tick,
                                           uint64_t now) {
	Common::PerfStats::AddDownload(bytes);
	std::lock_guard  lock(g_writers_mutex);
	uint64_t         shader = UINT64_MAX; // no record
	const GpuWriter* writer = nullptr;
	if (auto it = g_writers.upper_bound(vaddr); it != g_writers.begin()) {
		--it;
		// Ranges overlap: the closest start at or below the address that still covers it.
		for (int steps = 0; steps < 64; steps++) {
			if (vaddr < it->second.end) {
				shader = it->second.shader;
				writer = &it->second;
				break;
			}
			if (it == g_writers.begin()) {
				break;
			}
			--it;
		}
	}
	auto& entry = g_writer_downloads[shader];
	if (entry.count++ == 0) {
		entry.first = vaddr;
	}
	entry.bytes += bytes;
	if (writer != nullptr && writer->when_ns != 0 && now >= writer->when_ns) {
		entry.aged++;
		entry.age_ns += now - writer->when_ns;
		entry.done += writer->tick != 0 && writer->tick <= done_tick ? 1u : 0u;
	}
	if (++g_game_downloads % 1024 != 0) {
		return;
	}
	std::vector<std::pair<uint64_t, WriterDownloads>> top(g_writer_downloads.begin(),
	                                                      g_writer_downloads.end());
	std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	std::string line;
	for (size_t index = 0; index < std::min<size_t>(top.size(), 10); index++) {
		const auto& [hash, writer] = top[index];
		line += fmt::format(" {}={}/{}KB@{:x}",
		                    hash == UINT64_MAX ? std::string("unknown")
		                    : hash == 0        ? std::string("copy")
		                                       : fmt::format("{:016x}", hash),
		                    writer.count, writer.bytes >> 10, writer.first);
		if (writer.aged != 0) {
			// How long after the write the game thread read it, and how often the GPU had run
			// the write by then.
			line += fmt::format(" (written {:.1f} ms before, done on the GPU in {:.0f}%)",
			                    static_cast<double>(writer.age_ns) / 1e6 /
			                        static_cast<double>(writer.aged),
			                    100.0 * static_cast<double>(writer.done) /
			                        static_cast<double>(writer.aged));
		}
	}
	LOGF("ReadMemory: %" PRIu64 " game-thread downloads, by writer:%s%s\n", g_game_downloads,
	     line.c_str(),
	     GpuWriterRecordsWanted() ? ""
	                              : " (writers are recorded only with KYTY_GPU_WRITERS=1 or in "
	                                "F9 captures: \"unknown\" otherwise)");
}

bool BufferCache::ReadaheadEnabled() {
	static const bool whole_run = [] {
		const char* value = std::getenv("KYTY_READAHEAD");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Readahead: on for the whole run (KYTY_READAHEAD=1): what game threads read back "
			     "is downloaded as soon as the GPU writes it again\n");
		}
		return on;
	}();
	return whole_run || Bisect::Readahead();
}

void BufferCache::RotateReadahead() {
	const auto now = StallWatch::NowNs();
	if (m_readahead_generation_ns == 0) {
		// The counts of the first summary start here.
		m_readahead_generation_ns = now;
		g_reads_direct.store(0, std::memory_order_relaxed);
		g_reads_waited.store(0, std::memory_order_relaxed);
		return;
	}
	constexpr uint64_t Generation = 4'000'000'000ull;
	if (now - m_readahead_generation_ns < Generation) {
		return;
	}
	const double seconds      = static_cast<double>(now - m_readahead_generation_ns) / 1e9;
	m_readahead_generation_ns = now;
	uint64_t learned          = 0;
	m_readahead_current.ForEach([&](uint64_t begin, uint64_t end) { learned += end - begin; });
	const auto direct = g_reads_direct.exchange(0, std::memory_order_relaxed);
	const auto waited = g_reads_waited.exchange(0, std::memory_order_relaxed);
	if (ReadaheadEnabled() || m_readahead_downloads != 0) {
		LOGF("Readahead: %.0f downloads/s (%.1f MB/s); game-thread reads of GPU-written memory "
		     "%.0f/s, %.0f%% of them found their bytes in memory; %.1f MB learned\n",
		     static_cast<double>(m_readahead_downloads) / seconds,
		     static_cast<double>(m_readahead_bytes) / 1e6 / seconds,
		     static_cast<double>(direct + waited) / seconds,
		     direct + waited == 0 ? 0.0
		                          : 100.0 * static_cast<double>(direct) /
		                                static_cast<double>(direct + waited),
		     static_cast<double>(learned) / 1048576.0);
	}
	m_readahead_downloads = 0;
	m_readahead_bytes     = 0;
	m_readahead_previous  = std::move(m_readahead_current);
	m_readahead_current.Clear();
	m_readahead_learned_bytes = 0;
}

void BufferCache::LearnReadahead(uint64_t begin, uint64_t end, bool refresh) {
	RotateReadahead();
	const auto first = Common::AlignDown(begin, ReadaheadBlock);
	const auto last  = Common::AlignUp(end, ReadaheadBlock);
	if (last <= first || m_readahead_current.Contains(first, last - first)) {
		return;
	}
	if (refresh) {
		// Only the blocks of the older generation: a read that finds its bytes in memory because
		// another window brought them is not a reason to start downloading them.
		m_readahead_previous.ForEachInRange(first, last - first, [&](uint64_t b, uint64_t e) {
			const auto block_begin = Common::AlignDown(b, ReadaheadBlock);
			m_readahead_current.Add(block_begin, Common::AlignUp(e, ReadaheadBlock) - block_begin);
		});
		return;
	}
	if (m_readahead_learned_bytes >= ReadaheadMaxLearned) {
		return;
	}
	m_readahead_current.Add(first, last - first);
	m_readahead_learned_bytes += last - first;
}

void BufferCache::QueueReadahead(uint64_t vaddr, uint64_t size) {
	if ((m_readahead_current.Empty() && m_readahead_previous.Empty()) || size == 0 ||
	    !GuestGpu::IsGpuThread() || !ReadaheadEnabled()) {
		return;
	}
	const auto add = [this](uint64_t begin, uint64_t end) {
		if (!m_readahead_queue.empty() && m_readahead_queue.back().first <= end &&
		    begin <= m_readahead_queue.back().second) {
			auto& last  = m_readahead_queue.back();
			last.first  = std::min(last.first, begin);
			last.second = std::max(last.second, end);
			return;
		}
		if (m_readahead_queue.size() < ReadaheadMaxQueue) {
			m_readahead_queue.emplace_back(begin, end);
		}
	};
	m_readahead_current.ForEachInRange(vaddr, size, add);
	m_readahead_previous.ForEachInRange(vaddr, size, add);
}

void BufferCache::IssueReadaheadHook(void* cache) {
	static_cast<BufferCache*>(cache)->IssueReadahead();
}

void BufferCache::IssueReadahead() {
	// Inside an upload's callback the tracker cannot be walked: the next flush takes them.
	if (m_readahead_queue.empty() || !GuestGpu::IsGpuThread() ||
	    m_memory_tracker.InUploadCallback()) {
		return;
	}
	auto queue = std::move(m_readahead_queue);
	m_readahead_queue.clear();
	if (!ReadaheadEnabled()) {
		return; // the step went off
	}
	RotateReadahead();
	// At most 32 MB a command buffer: what is left over waits for the game thread's read, as
	// without readahead (the priority thread copies every byte into guest memory).
	constexpr uint64_t MaxBytes = 32ull << 20u;
	uint64_t           bytes    = 0;
	for (const auto& [begin, end]: queue) {
		for (auto at = begin; at < end && bytes < MaxBytes;) {
			const auto  next  = Common::AlignDown(at, CACHING_PAGESIZE) + CACHING_PAGESIZE;
			const auto* owner = m_page_table.Find(at >> PageTable::kPageBits);
			if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
				at = next;
				continue;
			}
			auto&      buffer = m_slot_buffers[*owner];
			const auto stop   = std::min(end, buffer.CpuAddress() + buffer.Size());
			if (stop <= at) {
				at = next;
				continue;
			}
			// Only the bytes still GPU-written are copied: a game thread's download, or another
			// readahead, may have taken them already. A block with a download still in flight is
			// left to the game thread's read: re-downloading a block the GPU keeps rewriting
			// could keep that read waiting for one publication after another.
			std::vector<std::pair<uint64_t, uint64_t>> runs;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				for (auto block = at; block < stop;) {
					const auto block_end =
					    std::min(stop, Common::AlignDown(block, ReadaheadBlock) + ReadaheadBlock);
					if (!m_downloading_ranges.Intersects(block, block_end - block)) {
						if (!runs.empty() && runs.back().second == block) {
							runs.back().second = block_end;
						} else {
							runs.emplace_back(block, block_end);
						}
					}
					block = block_end;
				}
			}
			for (const auto& [run_begin, run_end]: runs) {
				(void)DownloadBufferMemory<true>(buffer, run_begin, run_end - run_begin, &bytes);
			}
			at = stop;
		}
	}
	if (bytes != 0) {
		// The tick of the last copy (a download ring that wraps submits the earlier ones).
		m_last_async_download_tick = std::max(m_last_async_download_tick, m_scheduler.CurrentTick());
		m_readahead_downloads++;
		m_readahead_bytes += bytes;
		Common::PerfStats::AddReadahead(bytes);
	}
}

void BufferCache::WritesRecorded(bool address_writes, uint64_t shader) {
	const auto current = m_scheduler.CurrentTick();
	if (address_writes) {
		m_address_write_tick = std::max(m_address_write_tick, current);
		if (!m_address_writes_seen.exchange(true, std::memory_order_relaxed)) {
			LOGF("Write faults: shader %016" PRIx64
			     " stores through shader addresses, which mark no range: each write fault "
			     "releases its own page from now on\n",
			     shader);
		}
		// Stores through shader addresses mark no range: any watched metadata may be among them.
		if (m_metadata_watch_count.load(std::memory_order_relaxed) != 0) {
			std::unique_lock lock(m_dirty_ranges_mutex);
			for (auto& [address, watch]: m_metadata_watches) {
				(void)address;
				watch.written = true;
			}
		}
		// Which shaders hold fast readbacks up this way (the first 16).
		static std::unordered_set<uint64_t> logged;
		if (logged.size() < 16 && logged.insert(shader).second) {
			LOGF("Fast readback: shader %016" PRIx64 " stores through shader addresses: fast "
			     "readbacks wait for each of its commands to finish\n",
			     shader);
		}
	}
	if (m_unrecorded_writes.empty()) {
		return;
	}
	std::unique_lock lock(m_dirty_ranges_mutex);
	for (const auto& write: m_unrecorded_writes) {
		if (write.tick == current) {
			continue;
		}
		// The scheduler submitted between the binding and its command: the command went into a
		// later tick, this one at the latest. The write bitmap's regions wait for it too
		// (RefineWithWriteBitmap takes their bits as final once their tick is done).
		m_write_ticks.Assign(write.begin, write.end, current);
		if (write.tracked) {
			const auto first = Common::AlignDown(write.begin, WriteBitmapPage);
			const auto last  = Common::AlignUp(write.end, WriteBitmapPage);
			for (auto region = first >> WriteBitmapRegionBits;
			     region <= (last - 1) >> WriteBitmapRegionBits; region++) {
				auto& tick = m_bitmap_ticks[region];
				tick       = std::max(tick, current);
			}
		}
	}
	m_unrecorded_writes.clear();
}

void BufferCache::MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked) {
	KYTY_PROFILER_BLOCK("Obtain::MarkWritten");
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		const auto first = Common::AlignDown(vaddr, WriteBitmapPage);
		const auto last  = Common::AlignUp(vaddr + size, WriteBitmapPage);
		if (tracked && !m_bitmap_tracked.Contains(first, last - first)) {
			// A page stays untracked while it holds bytes an untracked write left GPU-written:
			// the GPU-written pieces outside the tracked set, widened to pages, are left out.
			RangeSet add;
			add.Add(first, last - first);
			std::vector<std::pair<uint64_t, uint64_t>> dirty;
			m_gpu_modified_ranges.ForEachInRange(first, last - first, [&](uint64_t b, uint64_t e) {
				dirty.emplace_back(b, e);
			});
			for (const auto [b, e]: dirty) {
				RangeSet untracked;
				untracked.Add(b, e - b);
				m_bitmap_tracked.ForEachInRange(b, e - b, [&](uint64_t tb, uint64_t te) {
					untracked.Subtract(tb, te - tb);
				});
				untracked.ForEach([&](uint64_t ub, uint64_t ue) {
					const auto pb = Common::AlignDown(ub, WriteBitmapPage);
					add.Subtract(pb, Common::AlignUp(ue, WriteBitmapPage) - pb);
				});
			}
			add.ForEach([&](uint64_t b, uint64_t e) { m_bitmap_tracked.Add(b, e - b); });
		}
		if (tracked) {
			const auto tick = m_scheduler.CurrentTick();
			for (auto region = first >> WriteBitmapRegionBits;
			     region <= (last - 1) >> WriteBitmapRegionBits; region++) {
				m_bitmap_ticks[region] = tick;
			}
		} else if (!m_bitmap_tracked.Empty()) {
			m_bitmap_tracked.Subtract(first, last - first);
		}
		// KYTY_MT_PROG: prepared draws' seals over these bytes break before they are dirty.
		DrawPrep::NoteGpuWrite(vaddr, size);
		m_gpu_modified_ranges.Add(vaddr, size);
		// The tick the writing command goes into, unless the scheduler submits before it is
		// recorded (WritesRecorded then moves it on).
		const auto tick = m_scheduler.CurrentTick();
		m_write_ticks.Assign(vaddr, vaddr + size, tick);
		m_unrecorded_writes.push_back({vaddr, vaddr + size, tick, tracked});
		ForgetFills(vaddr, size);
		NoteMetadataWrite(vaddr, size);
	}
	RecordGpuWriter(vaddr, size, s_diag_shader_hash, m_scheduler.CurrentTick());
	QueueReadahead(vaddr, size);
	if (Libs::LibKernel::Memory::TracesMemory(vaddr, size)) {
		// KYTY_TRACE_MEMORY: each writer's first eight writes, then at powers of two.
		static std::map<uint64_t, uint64_t> writes;
		const auto                          count = ++writes[s_diag_shader_hash];
		if (count <= 8 || std::has_single_bit(count)) {
			Libs::LibKernel::Memory::TraceMemory(
			    "GPU write 0x%016" PRIx64 "+0x%" PRIx64 " by %s %016" PRIx64 " (%" PRIu64
			    " by it so far)",
			    vaddr, size, s_diag_shader_hash != 0 ? "shader" : "a copy or fill, shader",
			    s_diag_shader_hash, count);
		}
	}
	// Diagnostics: KYTY_WATCH_GPU_WRITE=<address>[+<size>][,<address>[+<size>]...] (hex) names
	// the bindings that mark bytes of those ranges GPU-written: the first write of each
	// (range, shader), with a host stack when no shader is being bound (copies, fills).
	static const std::vector<std::pair<uint64_t, uint64_t>> watches = [] {
		std::vector<std::pair<uint64_t, uint64_t>> result;
		const char*                                value = std::getenv("KYTY_WATCH_GPU_WRITE");
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = 1;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			result.emplace_back(begin, bytes);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static std::atomic<uint32_t> watched {0};
	if (g_dynamic_watching.load(std::memory_order_acquire)) {
		std::lock_guard lock(g_dynamic_watch_mutex);
		for (const auto& watch: g_dynamic_watches) {
			if (vaddr >= watch.address + watch.size || watch.address >= vaddr + size) continue;
			if (!g_dynamic_watch_seen.emplace(watch.address, s_diag_shader_hash).second ||
			    watched.fetch_add(1) >= 128) {
				continue;
			}
			LOGF("GPU write covers %s 0x%016" PRIx64 "+0x%" PRIx64 ": range=0x%016" PRIx64
			     " size=0x%" PRIx64 " shader=0x%016" PRIx64 "\n%s",
			     watch.label, watch.address, watch.size, vaddr, size, s_diag_shader_hash,
			     s_diag_shader_hash == 0 ? Common::HostBacktrace().c_str() : "");
		}
	}
	for (size_t index = 0; index < watches.size(); index++) {
		const auto& [watch_begin, watch_size] = watches[index];
		if (watch_size == 0 || vaddr >= watch_begin + watch_size ||
		    watch_begin >= vaddr + size) {
			continue;
		}
		static std::mutex                             seen_mutex;
		static std::set<std::pair<size_t, uint64_t>> seen;
		bool                                          first = false;
		{
			std::lock_guard lock(seen_mutex);
			first = seen.emplace(index, s_diag_shader_hash).second;
		}
		if (first && watched.fetch_add(1) < 128) {
			LOGF("GPU write covers watched #%zu 0x%016" PRIx64 "+0x%" PRIx64
			     ": range=0x%016" PRIx64 " size=0x%" PRIx64 " shader=0x%016" PRIx64 "\n%s",
			     index, watch_begin, watch_size, vaddr, size, s_diag_shader_hash,
			     s_diag_shader_hash == 0 ? Common::HostBacktrace().c_str() : "");
		}
	}
}

// KYTY_IMAGE_UPLOAD_FROM_BUFFERS=1: an image upload over a cached buffer copies from that
// buffer even when the GPU wrote none of its bytes, as before 2026-10-14. By default the upload
// takes the guest bytes then: they are what the CPU wrote, while the buffer is only as fresh as
// its own tracking, which misses writes next to other cached data (GT7's light grids were stale
// that way, 2026-10-13).
static bool UploadImagesFromCachedBuffers() {
	static const bool from_buffers = [] {
		const char* value = std::getenv("KYTY_IMAGE_UPLOAD_FROM_BUFFERS");
		return value != nullptr && value[0] == '1';
	}();
	return from_buffers;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size) && !UploadImagesFromCachedBuffers() &&
		    !IsRegionGpuModified(vaddr, size)) {
			static std::atomic<uint64_t> staged {0};
			const auto                   count = ++staged;
			if (count <= 4 || (count & (count - 1u)) == 0) {
				LOGF("BufferCache: image upload 0x%012" PRIx64 "+0x%" PRIx64
				     " takes its guest bytes, not the cached buffer over it (%" PRIu64
				     " so far; KYTY_IMAGE_UPLOAD_FROM_BUFFERS=1 copies from the buffer)\n",
				     vaddr, size, count);
			}
			return StageGuestBytesForImage(vaddr, size);
		}
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging != nullptr) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
			std::memset(staging, 0, static_cast<size_t>(size));
		}
		m_staging_buffer.Commit();
		return {&m_staging_buffer, stage_offset};
	}
	// Larger than the staging ring: a one-time host allocation, freed once the GPU is done.
	auto temporary = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer, size);
	auto* mapped = temporary->Mapped().data();
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
		std::memset(mapped, 0, static_cast<size_t>(size));
	}
	temporary->Flush(0, size);
	auto* result = temporary.get();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {result, 0};
}

std::pair<Buffer*, uint64_t> BufferCache::StageGuestBytesForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: staging reservation failed for guest image\n");
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		std::memset(staging, 0, static_cast<size_t>(size));
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	const bool known_meta = m_texture_cache.ClearMeta(vaddr, value);
	if (Libs::LibKernel::Memory::TracesMemory(vaddr, size)) {
		Libs::LibKernel::Memory::TraceMemory(
		    "fill 0x%016" PRIx64 "+0x%" PRIx64 " with 0x%08x on the %s%s", vaddr, size, value,
		    IsRegionGpuModified(vaddr, size) ? "GPU" : "CPU",
		    known_meta ? " (a known surface's metadata: a clear)" : "");
	}
	if (!IsRegionGpuModified(vaddr, size)) {
		// Round 111: after the deferred protections (this store's faults need them) and the
		// earlier uploads' reads.
		FinishDeferredMemoryWork();
		ForgetFillsUnder(vaddr, size);
		// Hot pages do not fault (MemoryTracker::DirtyHotIn).
		m_memory_tracker.DirtyHotIn(vaddr, size);
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		Libs::LibKernel::Memory::JournalWrite(vaddr, size,
		                                      Libs::LibKernel::Memory::WriteSource::Fill);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	// After ObtainBuffer marked the range GPU-written (which forgets older fills of it). Not
	// merged: neighbouring surfaces' metadata clears stay apart, so a write to one keeps the
	// other's.
	NoteFill(vaddr, size, value, size, false);
}

void BufferCache::FillMappedSpans(std::span<const std::pair<uint64_t, uint64_t>> spans,
                                  uint32_t                                       value) {
	if (spans.empty()) {
		return;
	}
	for (const auto& [vaddr, size]: spans) {
		if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 ||
		    !GuestRange {vaddr, size}.Valid()) {
			EXIT("BufferCache: invalid mapped fill span 0x%016" PRIx64 "+0x%" PRIx64 "\n", vaddr,
			     size);
		}
	}
	// One cached buffer over all of them (holes included, which read as zero), as an image over
	// the range will want later, rather than one a span; then one barrier pair for all the fills.
	const auto first = spans.front().first;
	const auto last  = spans.back().first + spans.back().second;
	const auto id    = FindBuffer(first, last - first);
	struct Fill {
		vk::Buffer buffer;
		uint64_t   offset;
		uint64_t   size;
	};
	std::vector<Fill> fills;
	fills.reserve(spans.size());
	for (const auto& [vaddr, size]: spans) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true, id);
		fills.push_back({dst->Handle(), dst_offset, size});
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	for (const auto& fill: fills) {
		native.fillBuffer(fill.buffer, fill.offset, fill.size, value);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                       nullptr);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	const bool cpu_copy = src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	                      !IsRegionGpuModified(src_vaddr, size) &&
	                      !m_texture_cache.FindImageFromRange(src_vaddr, size);
	if (dst_memory && Libs::LibKernel::Memory::TracesMemory(dst_vaddr, size)) {
		Libs::LibKernel::Memory::TraceMemory(
		    "copy into 0x%016" PRIx64 "+0x%" PRIx64 " from 0x%016" PRIx64 "%s on the %s", dst_vaddr,
		    size, src_vaddr, src_gds ? " (gds)" : "", cpu_copy ? "CPU" : "GPU");
	}
	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log("dma 0x%012" PRIx64 "%s <- 0x%012" PRIx64 "%s, 0x%" PRIx64 " bytes, %s%s",
		                dst_vaddr, dst_gds ? " (gds)" : "", src_vaddr, src_gds ? " (gds)" : "",
		                size, cpu_copy ? "on the CPU" : "on the GPU",
		                dst_memory && m_texture_cache.FindImageFromRange(dst_vaddr, size)
		                    ? ", into an image's memory"
		                    : "");
	}
	if (cpu_copy) {
		FinishDeferredMemoryWork(); // round 111, as FillBuffer
		ForgetFillsUnder(dst_vaddr, size);
		// Hot pages do not fault (MemoryTracker::DirtyHotIn).
		m_memory_tracker.DirtyHotIn(dst_vaddr, size);
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		Libs::LibKernel::Memory::JournalWrite(dst_vaddr, size,
		                                      Libs::LibKernel::Memory::WriteSource::Copy);
		return;
	}

	auto& command = m_scheduler.Current();
	command.MarkTransfer(CommandBufferDebugOp::BufferCopy, dst_vaddr, src_vaddr,
	                     static_cast<uint32_t>(std::min<uint64_t>(size, UINT32_MAX)),
	                     dst_gds ? 1u : 0u, src_gds ? 1u : 0u);
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::AddressWritesInFlight() {
	return m_address_write_tick != 0 && !m_scheduler.IsFree(m_address_write_tick);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Round 152: bytes a fast readback has copied are not in guest memory until the thread that
	// waits for the copy puts them there: the GPU thread's own reads (the SRT walks, prepared
	// draws, an image's first fill) take them as GPU-written until then.
	return m_gpu_modified_ranges.Intersects(vaddr, size) ||
	       (m_readback_queue != nullptr && m_readback_queue->AnyPending() &&
	        m_readback_queue->PendingIn(vaddr, vaddr + size).Valid());
}

void BufferCache::FinishFastReadbacks(uint64_t vaddr, uint64_t size) {
	if (m_readback_queue == nullptr || size == 0 || !m_readback_queue->AnyPending()) {
		return;
	}
	for (auto ticket = m_readback_queue->PendingIn(vaddr, vaddr + size); ticket.Valid();
	     ticket      = m_readback_queue->PendingIn(vaddr, vaddr + size)) {
		CompleteFastReadback(ticket, FastCompleter::GpuThread);
	}
}

bool BufferCache::HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size) {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return m_gpu_modified_ranges.Intersects(vaddr, size) ||
	       m_fast_pending_ranges.Intersects(vaddr, size);
}

void BufferCache::DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size) {
	const std::pair<uint64_t, uint64_t> range {vaddr, size};
	DownloadRangesForDiagnostics(std::span {&range, 1u});
}

void BufferCache::DownloadRangesForDiagnostics(
    std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	std::vector<std::pair<uint64_t, uint64_t>> downloaded;
	for (const auto& [vaddr, size]: ranges) {
		if (size == 0) {
			continue;
		}
		auto it = m_buffers.upper_bound(vaddr);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < vaddr + size; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto begin  = std::max(vaddr, buffer.CpuAddress());
			const auto end    = std::min(vaddr + size, buffer.CpuAddress() + buffer.Size());
			if (begin < end && DownloadBufferMemory<true>(buffer, begin, end - begin)) {
				downloaded.emplace_back(begin, end - begin);
			}
		}
	}
	if (downloaded.empty()) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	for (const auto& [begin, bytes]: downloaded) {
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
	}
}

void BufferCache::WaitForDiagnostics() {
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
}

void BufferCache::SnapshotForDiagnostics(uint64_t vaddr, uint64_t size, uint32_t op,
                                         std::string label, bool after_dispatch) {
	// At most this many a capture (each a buffer of its own, up to KYTY_FG_DUMP_BYTES).
	constexpr size_t MaxSnapshots = 1024;
	size &= ~uint64_t {3};
	if (size == 0 || m_diagnostic_snapshots.size() >= MaxSnapshots ||
	    !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// What the binding uploads first (FindBuffers runs after the stage's capture lines): the
	// CPU's writes since the last upload, so the copy is what the dispatch reads.
	if (!after_dispatch && !HasGpuDirtyBytes(vaddr, size)) {
		SynchronizeBuffersInRange(vaddr, size);
	}
	PendingSnapshot pending;
	pending.snapshot.address = vaddr;
	pending.snapshot.op      = op;
	pending.snapshot.after   = after_dispatch;
	pending.snapshot.label   = std::move(label);
	pending.size             = size;
	const Buffer* source     = nullptr;
	if (auto it = m_buffers.upper_bound(vaddr); it != m_buffers.begin()) {
		--it;
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.IsInBounds(vaddr, size)) {
			source = &buffer;
		}
	}
	if (source == nullptr) {
		pending.snapshot.words.resize(size / 4u);
		if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, pending.snapshot.words.data(), size)) {
			pending.snapshot.words.clear();
		}
		m_diagnostic_snapshots.push_back(std::move(pending));
		return;
	}
	pending.snapshot.from_gpu = true;
	pending.copy  = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         vk::BufferUsageFlagBits::eTransferDst, size);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Recorder();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = source->Handle();
	before.offset              = 0;
	before.size                = source->Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	const vk::BufferCopy copy {source->Offset(vaddr), 0, size};
	native.copyBuffer(source->Handle(), pending.copy->Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = pending.copy->Handle();
	after.offset        = 0;
	after.size          = size;
	// The source goes back to what the dispatch after it reads or writes.
	auto release          = before;
	release.srcAccessMask = vk::AccessFlagBits::eTransferRead;
	release.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	const std::array<vk::BufferMemoryBarrier, 2> done {after, release};
	native.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer,
	    vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	    static_cast<uint32_t>(done.size()), done.data(), 0, nullptr);
	m_diagnostic_snapshots.push_back(std::move(pending));
}

std::vector<BufferCache::DiagnosticSnapshot> BufferCache::TakeDiagnosticSnapshots() {
	std::vector<DiagnosticSnapshot> result;
	if (m_diagnostic_snapshots.empty()) {
		return result;
	}
	WaitForDiagnostics();
	result.reserve(m_diagnostic_snapshots.size());
	for (auto& pending: m_diagnostic_snapshots) {
		if (pending.copy != nullptr) {
			pending.copy->Invalidate(0, pending.size);
			const auto mapped = pending.copy->Mapped();
			if (mapped.size() >= pending.size) {
				pending.snapshot.words.resize(pending.size / 4u);
				std::memcpy(pending.snapshot.words.data(), mapped.data(), pending.size);
			}
		}
		result.push_back(std::move(pending.snapshot));
	}
	m_diagnostic_snapshots.clear();
	return result;
}

bool BufferCache::IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return !m_gpu_modified_ranges.Intersects(vaddr, size) &&
	       !m_downloading_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

uint64_t BufferCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

void BufferCache::RunGarbageCollector() {
	KYTY_PROFILER_FUNCTION();
	m_gc_tick++;
	const auto clock = LruClock();
	// Memory pressure, as in TextureCache::RunGarbageCollector.
	if (m_trigger_gc_memory != m_applied_trigger_gc_memory ||
	    m_critical_gc_memory != m_applied_critical_gc_memory) {
		m_base_trigger_gc_memory  = m_trigger_gc_memory;
		m_base_critical_gc_memory = m_critical_gc_memory;
	}
	if (const auto shortfall = m_graphics.MemoryShortfall(); shortfall != 0) {
		const auto cap = std::max<uint64_t>(
		    m_total_used_memory > shortfall ? m_total_used_memory - shortfall : 0, 512ull << 20u);
		m_trigger_gc_memory  = std::min(m_base_trigger_gc_memory, cap);
		m_critical_gc_memory = std::min(m_base_critical_gc_memory, cap + cap / 8);
	} else {
		m_trigger_gc_memory  = m_base_trigger_gc_memory;
		m_critical_gc_memory = m_base_critical_gc_memory;
	}
	m_applied_trigger_gc_memory  = m_trigger_gc_memory;
	m_applied_critical_gc_memory = m_critical_gc_memory;
	// Pressure is judged by this cache's own bytes. Device-wide usage also counts the images
	// spilled to host memory, which on a 6 GB card keeps it above the critical mark forever
	// and has the collector destroy and recreate every buffer twice a second (a third of
	// the GPU thread's time in the world). KYTY_GC_DEVICE_BYTES=1 restores that policy.
	static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
	if (device_bytes && m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// KYTY_BISECT's gc:off: neither cache collects while the step is on (an A/B test of what
	// collecting costs).
	if (Bisect::GcOff()) {
		return;
	}
	struct GcSection {
		uint64_t  begin = StallWatch::NowNs();
		uint64_t* deleted;
		~GcSection() {
			StallWatch::NoteSection(StallWatch::Section::BufferGc, begin, StallWatch::NowNs(),
			                        *deleted);
		}
	};
	uint64_t  section_deleted = 0;
	GcSection gc_section {.deleted = &section_deleted};

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	// Ages in frames, as in the texture cache: a buffer used this frame or the last is
	// never a candidate, whatever the submission count.
	const uint64_t age        = std::min<uint64_t>(aggressive ? 2 : 4, clock);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	static uint64_t       gc_runs = 0, gc_visited = 0, gc_retained = 0, gc_deleted = 0;
	static double         gc_seconds = 0.0;
	Common::Timer         gc_timer;
	gc_timer.Start();
	gc_runs++;
	m_lru_cache.ForEachItemBelow(clock - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		gc_visited++;
		if (buffer.CpuAddress() == 0) {
			// See CreateBuffer: the tracker rejects address 0, so this one is never collected.
			return false;
		}
		if (buffer.gc_pending) {
			// Its download is in flight; the deferred step at the end drops it.
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges.Ranges(),
		                                           buffer.CpuAddress(), buffer.Size(),
		                                           "garbage collection");
		// An aged GPU-dirty buffer is downloaded and dropped under either policy. Retaining it
		// (the non-aggressive rule before) leaves its GPU-written bytes owning the pages
		// forever, and the images that share those pages are then re-sourced from the
		// buffer's stale contents: the world renders black. Measured 2026-09-21.
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size())) {
			// Its pages are still marked GPU-written, but no byte waits for a download: they were
			// downloaded with another range, or a game thread's asynchronous readback of them is
			// still in flight (ReadMemoryStep's "page is current" case). This ended the session on
			// cards small enough for the collector to run (a 4 GB laptop, 2026-10-06). In flight:
			// leave the buffer to a later pass, after the publication. Otherwise the bytes are
			// current in guest memory and the buffer goes like a clean one.
			bool downloading = false;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				downloading = m_downloading_ranges.Intersects(buffer.CpuAddress(), buffer.Size());
			}
			if (downloading) {
				return false;
			}
			m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			NoteBufferEvent(BufferEventKind::Collected, buffer);
			DeleteBuffer(id);
		} else if (dirty) {
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			NoteBufferEvent(BufferEventKind::Collected, buffer);
			DeleteBuffer(id);
		}
		gc_deleted++;
		section_deleted++;
		return ++retire_count == limit;
	});
	gc_seconds += gc_timer.GetTimeS();
	if (gc_runs % 500 == 0) {
		LOGF("GcStats: runs=%" PRIu64 " visited=%" PRIu64 " retained=%" PRIu64 " deleted=%" PRIu64
		     " scan_total=%.1fms buffers=%zu used=%" PRIu64 "MB trigger=%" PRIu64 "MB aggressive=%d\n",
		     gc_runs, gc_visited, gc_retained, gc_deleted, gc_seconds * 1000.0, m_buffers.size(),
		     m_total_used_memory >> 20u, m_trigger_gc_memory >> 20u, static_cast<int>(aggressive));
		gc_visited = gc_retained = gc_deleted = 0;
		gc_seconds = 0.0;
		// The largest buffers: the whole-heap descriptors of a title mirror as whole device
		// buffers, and a handful of them can be most of the cache.
		std::vector<std::pair<uint64_t, uint64_t>> largest;
		for (const auto& [address, id]: m_buffers) {
			const auto* buffer = m_slot_buffers.try_get(id);
			if (buffer != nullptr && !buffer->is_deleted) {
				largest.emplace_back(buffer->Size(), address);
			}
		}
		std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(largest.size(), 6),
		                  largest.end(), std::greater<>());
		for (size_t i = 0; i < std::min<size_t>(largest.size(), 6); i++) {
			LOGF("\t largest[%zu] = %" PRIu64 "MB at 0x%016" PRIx64 "\n", i, largest[i].first >> 20u,
			     largest[i].second);
		}
	}
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners. Must be
	// CurrentTick(): DownloadBufferMemory above queued copy-out commands into the currently open
	// recording, so that recording has to actually submit and complete -- see ReadMemory's wait
	// for why waiting on an older per-buffer tick here would skip that entirely.
	const auto completion_tick = m_scheduler.CurrentTick();
	// KYTY_GC_ASYNC_DOWNLOAD=0 waits here as before. By default the GPU thread goes on, and the
	// buffers are dropped by a deferred operation once the GPU has finished this tick and its
	// downloads are published (deferred operations run after their tick's priority callbacks). In
	// a 1440p race on a 6 GB card the wait took 37 ms of each 290 ms frame (2026-10-24). A buffer
	// a draw or dispatch uses again meanwhile stays: its pages may hold newer GPU writes.
	static const bool async_download = [] {
		const char* value = std::getenv("KYTY_GC_ASYNC_DOWNLOAD");
		return value == nullptr || value[0] != '0';
	}();
	if (async_download) {
		struct Pending {
			BufferId id;
			uint64_t address;
		};
		std::vector<Pending> pending;
		pending.reserve(dirty_buffers.size());
		for (const auto id: dirty_buffers) {
			auto& buffer      = m_slot_buffers[id];
			buffer.gc_pending = true;
			buffer.gc_reused  = false;
			pending.push_back({id, buffer.CpuAddress()});
		}
		m_scheduler.DeferOperation([this, completion_tick, pending = std::move(pending)] {
			static uint64_t dropped = 0, kept = 0;
			for (const auto& [id, address]: pending) {
				auto* buffer = m_slot_buffers.try_get(id);
				if (buffer == nullptr || buffer->is_deleted || !buffer->gc_pending ||
				    buffer->CpuAddress() != address) {
					continue;
				}
				buffer->gc_pending = false;
				// GPU writes recorded since the downloads (also through buffer device addresses,
				// which bind nothing and so leave gc_reused alone) are in m_gpu_modified_ranges
				// again: then the buffer holds bytes newer than guest memory and stays.
				if (buffer->gc_reused || buffer->last_use_tick > completion_tick ||
				    m_gpu_modified_ranges.Intersects(buffer->CpuAddress(), buffer->Size())) {
					buffer->gc_reused = false;
					kept++;
					continue;
				}
				m_memory_tracker.UnmarkRegionAsGpuModified(buffer->CpuAddress(), buffer->Size());
				if (m_memory_tracker.IsRegionGpuModified(buffer->CpuAddress(), buffer->Size())) {
					kept++;
					continue;
				}
				m_memory_tracker.UntrackMemory(buffer->CpuAddress(), buffer->Size());
				NoteBufferEvent(BufferEventKind::Collected, *buffer);
				// Not destroyed at once: work recorded after the downloads may still reach it
				// through its device address on the GPU (a device loss on 2026-10-24: a write
				// into a buffer destroyed one tick before the GPU finished). DeleteBuffer clears
				// its address table entries and destroys it once the current tick is done.
				DeleteBuffer(id);
				if (++dropped % 1024 == 1) {
					LOGF("BufferCache: %" PRIu64 " GPU-written buffers dropped after their "
					     "downloads landed, %" PRIu64 " kept (used again meanwhile)\n",
					     dropped, kept);
				}
			}
		});
		return;
	}
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		NoteBufferEvent(BufferEventKind::Collected, buffer);
		// The fast readbacks' copies recorded before the wait above are done (their safety-net
		// operations were among those waited for); none is recorded since (round 152).
		if (m_readback_queue != nullptr) {
			m_readback_queue->WaitCopiesFrom(buffer.Handle());
		}
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.ForEachMaybeCpuDirtyRegion(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) {
		    SynchronizeBuffersInRange(address, bytes);
	    });
}

uint64_t BufferCache::SynchronizeBdaPending(const RangeSet& mapped) {
	// Before round 69 each walk visited every tracker region of the mapped ranges with a CPU-dirty
	// page anywhere, and synchronized every cached buffer of it: thousands of buffers a walk whose
	// pages had not changed (the regions keep CPU-dirty the pages no buffer covers), ~30 us a
	// dispatch that reads memory through addresses, 8 % of the GPU thread in a race. Now only the
	// regions where pages became CPU-dirty or a buffer appeared since the last walk, and in them
	// only the buffers over CPU-dirty pages: what a sync of the others found was nothing to do.
	//
	// Each cached buffer over the runs synchronizes once, over the span from its first CPU-dirty
	// byte to its last in the region (round 70): its sync uploads only the CPU-dirty pages of the
	// span and changes their protection in one call, where one sync a run made one protection
	// call a run (~3.7 us each; 216,000 protection changes in 10 s of a race in round 69, 8 % of
	// the GPU thread). KYTY_BISECT's bdaspan:off synchronizes each run on its own again.
	// Round 111: the walk's protections are deferred too (its uploads read on the page manager's
	// worker, after them).
	const PageManager::DeferScope defer(m_memory_tracker.Pages(), DeferProtections(m_scheduler));
	thread_local std::vector<std::pair<uint64_t, uint64_t>> runs;
	uint64_t                                                regions = 0;
	const bool                                              by_run  = Bisect::BdaSpanOff();
	// Round 80: the spans of a region's buffers write-protect their pages again in one change
	// (MemoryTracker::TakeCpuDirtySpans); KYTY_BISECT's bdaprotect:old syncs each span on its own.
	const bool                         together = !by_run && !Bisect::BdaProtectOld();
	thread_local std::vector<BufferId> span_buffers;
	thread_local std::vector<std::pair<uint64_t, uint64_t>> spans;
	thread_local std::vector<std::vector<vk::BufferCopy>>   span_copies;
	// A buffer's runs in one mapped part of the region: the buffers do not overlap and the runs
	// ascend, so they come one after another.
	const auto sync_runs = [&] {
		if (by_run) {
			for (const auto& [address, bytes]: runs) {
				SynchronizeBuffersInRange(address, bytes);
			}
			return;
		}
		BufferId   current {};
		uint64_t   span_begin = 0;
		uint64_t   span_end   = 0;
		span_buffers.clear();
		spans.clear();
		const auto flush      = [&] {
            if (current) {
				if (together) {
					span_buffers.push_back(current);
					spans.emplace_back(span_begin, span_end - span_begin);
				} else {
					(void)SynchronizeBuffer(m_slot_buffers[current], span_begin,
					                        span_end - span_begin, false, false);
				}
				current = {};
			}
		};
		for (const auto& [address, bytes]: runs) {
			const auto end = address + bytes;
			auto       it  = m_buffers.upper_bound(address);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < end; ++it) {
				const auto& buffer = m_slot_buffers[it->second];
				const auto  start  = std::max(buffer.CpuAddress(), address);
				const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
				if (start >= finish) {
					continue;
				}
				if (current == it->second) {
					span_end = finish;
					continue;
				}
				flush();
				current    = it->second;
				span_begin = start;
				span_end   = finish;
			}
		}
		flush();
		if (!together || spans.empty()) {
			return;
		}
		const GpuPhases::Scope phase(GpuPhases::Phase::BufferSync);
		if (span_copies.size() < spans.size()) {
			span_copies.resize(spans.size());
		}
		std::vector<uint64_t> totals(spans.size(), 0);
		for (size_t i = 0; i < spans.size(); i++) {
			span_copies[i].clear();
		}
		m_memory_tracker.TakeCpuDirtySpans(
		    spans, [&](size_t index, uint64_t address, uint64_t bytes) noexcept {
			    auto& buffer = m_slot_buffers[span_buffers[index]];
			    span_copies[index].emplace_back(totals[index], buffer.Offset(address), bytes);
			    totals[index] += bytes;
		    });
		for (size_t i = 0; i < spans.size(); i++) {
			auto& buffer = m_slot_buffers[span_buffers[i]];
			RecordSync(spans[i].second, false, !span_copies[i].empty(), 0.0);
			if (span_copies[i].empty()) {
				continue;
			}
			const auto source = UploadCopies(buffer, span_copies[i], totals[i]);
			if (source) {
				RecordUpload(buffer, span_copies[i], totals[i], source);
			}
		}
	};
	m_memory_tracker.TakeBdaPending([&](uint64_t region) {
		regions++;
		mapped.ForEachInRange(region, TRACKER_REGION_SIZE, [&](uint64_t begin, uint64_t end) {
			runs.clear();
			m_memory_tracker.CollectCpuDirtyRuns(begin, end - begin, runs);
			sync_runs();
		});
	});
	return regions;
}

void BufferCache::VerifyWithGuest(uint64_t vaddr, uint64_t size, uint32_t op,
                                  const std::string& what) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid() || HasGpuDirtyBytes(vaddr, size)) {
		return;
	}
	// What the binding uploads first; what is left to differ, the tracking did not see.
	SynchronizeBuffersInRange(vaddr, size);
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr) & ~uint64_t {3};
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end) & ~uint64_t {3};
		if (start < buffer.CpuAddress() || start >= finish ||
		    !FrameGraph::ClaimVerify(start, finish - start)) {
			continue;
		}
		const auto bytes = finish - start;
		auto       guest = std::make_shared<std::vector<uint32_t>>(bytes / sizeof(uint32_t));
		if (!Libs::LibKernel::Memory::TryReadBacking(start, guest->data(), bytes)) {
			continue;
		}
		auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         AllFlags, bytes);
		if (download->Mapped().data() == nullptr) {
			continue;
		}
		download->CopyFrom(m_scheduler.Current(), buffer, buffer.Offset(start), 0, bytes,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = download->Handle();
		barrier.offset              = 0;
		barrier.size                = bytes;
		m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
		                                                 nullptr, 1, &barrier, 0, nullptr);
		const auto description =
		    fmt::format("{} 0x{:012x}+0x{:x} (cached buffer 0x{:012x}+0x{:x})", what, start, bytes,
		                buffer.CpuAddress(), buffer.Size());
		m_scheduler.DeferPriorityOperation([download, guest, start, bytes, op, description] {
			download->Invalidate(0, bytes);
			const auto*  held   = reinterpret_cast<const uint32_t*>(download->Mapped().data());
			const size_t words  = guest->size();
			size_t       differ = 0, first = words, last = 0;
			for (size_t i = 0; i < words; i++) {
				if (held[i] != (*guest)[i]) {
					first = differ++ == 0 ? i : first;
					last  = i;
				}
			}
			if (differ != 0) {
				LOGF("FG %u stale %s: %zu of %zu words differ, from +0x%zx to +0x%zx (the first: "
				     "guest %08x, GPU %08x)\n",
				     op, description.c_str(), differ, words, first * 4u, last * 4u, (*guest)[first],
				     held[first]);
			}
			FrameGraph::CountVerify(false, bytes, differ != 0);
			(void)start;
		});
	}
}

void BufferCache::TraceContents(uint64_t vaddr, uint64_t size, const std::string& prefix) {
	size = std::min<uint64_t>(size, 4096);
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr) & ~uint64_t {3};
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end) & ~uint64_t {3};
		if (start < buffer.CpuAddress() || start >= finish) {
			continue;
		}
		const auto bytes = finish - start;
		auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         AllFlags, bytes);
		if (download->Mapped().data() == nullptr) {
			return;
		}
		download->CopyFrom(m_scheduler.Current(), buffer, buffer.Offset(start), 0, bytes,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
		m_scheduler.DeferPriorityOperation([download, bytes, start, prefix] {
			download->Invalidate(0, bytes);
			const auto*  words     = reinterpret_cast<const uint32_t*>(download->Mapped().data());
			const size_t count     = bytes / sizeof(uint32_t);
			size_t       nonfinite = 0;
			for (size_t i = 0; i < count; i++) {
				nonfinite += (words[i] & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
			}
			LOGF("%s 0x%012" PRIx64 "+0x%" PRIx64
			     " as the GPU holds it: %zu of %zu words are not finite as floats\n",
			     prefix.c_str(), start, bytes, nonfinite, count);
			for (size_t row = 0; row < std::min<size_t>(count, 64); row += 16) {
				std::string line;
				for (size_t k = row; k < std::min(count, row + 16); k++) {
					line += fmt::format(" {:08x}", words[k]);
				}
				LOGF("%s +0x%03zx:%s\n", prefix.c_str(), row * 4u, line.c_str());
			}
		});
		return; // the first cached buffer over the range
	}
}

void BufferCache::NoteBufferEvent(BufferEventKind kind, const Buffer& buffer, uint64_t request,
                                  uint64_t request_size, uint32_t absorbed) {
	constexpr size_t  Capacity = 16384;
	const BufferEvent event {.frame   = m_graphics.presented_frames.load(std::memory_order_relaxed),
	                         .address = buffer.CpuAddress(),
	                         .size    = buffer.Size(),
	                         .request = request,
	                         .request_size = request_size,
	                         .absorbed     = absorbed,
	                         .kind         = kind};
	if (m_buffer_events.size() < Capacity) {
		m_buffer_events.push_back(event);
	} else {
		m_buffer_events[m_buffer_events_next] = event;
	}
	m_buffer_events_next = (m_buffer_events_next + 1) % Capacity;
	m_buffer_events_total++;
	if (kind == BufferEventKind::Created) {
		m_buffer_origins[event.address] = event;
	} else {
		m_buffer_origins.erase(event.address);
	}
}

size_t BufferCache::RedirtyMapped(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	if (m_known_fill_count.load(std::memory_order_relaxed) != 0) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		ForgetFills(vaddr, size);
	}
	return m_memory_tracker.RedirtyForMap(vaddr, size);
}

void BufferCache::MarkMappedPending(uint64_t vaddr, uint64_t size) noexcept {
	m_memory_tracker.MarkBdaPending(vaddr, size);
	g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
	if (!t_guest_gpu_thread) {
		g_cpu_write_epoch.fetch_add(1, std::memory_order_release);
	}
}

uint64_t BufferCache::CompareWithGuest(uint64_t vaddr, uint64_t size, const std::string& label,
                                       bool                                           heal,
                                       const std::function<bool(uint64_t, uint64_t)>& mapped,
                                       bool                                           quiet) {
	if (size == 0 || m_scheduler.Current().IsInvalid()) {
		return 0;
	}
	const auto first = vaddr & ~(CACHING_PAGESIZE - 1u);
	const auto last =
	    Common::AlignUp(vaddr + std::min<uint64_t>(size, 64ull << 20u), CACHING_PAGESIZE);
	if (first == 0 || !GuestRange {first, last - first}.Valid()) {
		return 0;
	}
	const auto bytes = last - first;
	const auto pages = static_cast<size_t>(bytes >> CACHING_PAGEBITS);
	struct PageInfo {
		BufferId                  buffer {};
		bool                      covered    = false; // a cached buffer is over the page
		bool                      owner      = false; // the page table names a buffer
		bool                      owner_same = false; // ... the one over the page
		bool                      mapped     = false;
		bool                      pending    = false;
		bool                      readable   = false;
		bool                      changed    = false; // the tracker's state changed meanwhile
		MemoryTracker::PageStates states;
		uint64_t                  expected   = 0; // the BDA page table entry the buffer wrote
		uint64_t                  entry      = 0;
		size_t                    differ     = 0; // words
		size_t                    gpu_zero   = 0;
		size_t                    guest_zero = 0;
	};
	std::vector<PageInfo> info(pages);
	auto download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, bytes);
	auto table    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, pages * sizeof(vk::DeviceAddress));
	if (download->Mapped().data() == nullptr || table->Mapped().data() == nullptr) {
		return 0;
	}
	auto&      command = m_scheduler.Current();
	const auto copy    = [&](Buffer& target, const Buffer& source, uint64_t source_offset,
                          uint64_t target_offset, uint64_t count) {
        target.CopyFrom(command, source, source_offset, target_offset, count,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	};
	auto it = m_buffers.upper_bound(first);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < last; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), first);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), last);
		if (start >= finish) {
			continue;
		}
		copy(*download, buffer, buffer.Offset(start), start - first, finish - start);
		for (auto page = start; page < finish; page += CACHING_PAGESIZE) {
			auto& entry    = info[(page - first) >> CACHING_PAGEBITS];
			entry.buffer   = it->second;
			entry.covered  = true;
			entry.expected = buffer.BufferDeviceAddress() + (page - buffer.CpuAddress());
		}
	}
	copy(*table, m_bda_pagetable_buffer, PageIndex(first) * sizeof(vk::DeviceAddress), 0,
	     pages * sizeof(vk::DeviceAddress));
	for (const auto* target: {download.get(), table.get()}) {
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = target->Handle();
		barrier.offset              = 0;
		barrier.size                = VK_WHOLE_SIZE;
		command.Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                                   vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
		                                   &barrier, 0, nullptr);
	}
	// The CPU side as the copies see it: the uploads recorded so far run before them.
	for (size_t i = 0; i < pages; i++) {
		const auto  page  = first + (uint64_t {i} << CACHING_PAGEBITS);
		auto&       entry = info[i];
		const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
		entry.owner       = owner != nullptr && *owner;
		entry.owner_same  = entry.owner && entry.covered && *owner == entry.buffer;
		entry.states      = m_memory_tracker.DescribePages(page, CACHING_PAGESIZE);
		entry.pending     = m_memory_tracker.IsBdaPending(page);
		entry.mapped      = mapped && mapped(page, CACHING_PAGESIZE);
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	download->Invalidate(0, bytes);
	table->Invalidate(0, pages * sizeof(vk::DeviceAddress));

	std::vector<uint32_t> guest(CACHING_PAGESIZE / sizeof(uint32_t));
	const auto*           held_bytes = download->Mapped().data();
	// stale: differing while the tracker holds them clean (or in no region); waiting: differing
	// while CPU-dirty, an upload is due; healable: either of them.
	size_t covered = 0, differing = 0, zero = 0, stale = 0, waiting = 0, written = 0;
	for (size_t i = 0; i < pages; i++) {
		const auto page  = first + (uint64_t {i} << CACHING_PAGEBITS);
		auto&      entry = info[i];
		std::memcpy(&entry.entry, table->Mapped().data() + i * sizeof(vk::DeviceAddress),
		            sizeof(vk::DeviceAddress));
		if (!entry.covered) {
			continue;
		}
		covered++;
		entry.readable =
		    Libs::LibKernel::Memory::TryReadBacking(page, guest.data(), CACHING_PAGESIZE) ||
		    Libs::LibKernel::Memory::TryReadSparseBacking(page, guest.data(), CACHING_PAGESIZE);
		if (!entry.readable) {
			continue;
		}
		const auto* held = reinterpret_cast<const uint32_t*>(held_bytes + (page - first));
		for (size_t word = 0; word < guest.size(); word++) {
			entry.differ += held[word] != guest[word] ? 1u : 0u;
			entry.gpu_zero += held[word] == 0u ? 1u : 0u;
			entry.guest_zero += guest[word] == 0u ? 1u : 0u;
		}
		const auto after = m_memory_tracker.DescribePages(page, CACHING_PAGESIZE);
		entry.changed    = after.cpu != entry.states.cpu || after.gpu != entry.states.gpu;
		if (entry.differ != 0) {
			differing++;
			zero += entry.gpu_zero == guest.size() ? 1u : 0u;
			if (entry.states.gpu != 0) {
				written++;
			} else if (!entry.changed) {
				stale += entry.states.cpu == 0 && entry.states.hot == 0 ? 1u : 0u;
				waiting += entry.states.cpu != 0 ? 1u : 0u;
			}
		}
	}
	LOGF("%s: 0x%012" PRIx64 "+0x%" PRIx64 ", %zu pages of 16 KB: %zu with a cached buffer, %zu of "
	     "them differ from guest memory (%zu the GPU holds as zeros): %zu STALE (clean in the "
	     "tracker, or in no tracker region), %zu CPU-dirty (an upload is due), %zu GPU-written; "
	     "shader stores through addresses: %s\n",
	     label.c_str(), vaddr, size, pages, covered, differing, zero, stale, waiting, written,
	     AddressWritesSeen() ? "seen" : "none");
	const auto describe = [&](const PageInfo& entry) {
		if (!entry.covered) {
			return fmt::format("no cached buffer | page table: {} | BDA entry: {} | mapped: {}",
			                   entry.owner ? "names one" : "none",
			                   entry.entry == 0 ? "none" : "SET", entry.mapped ? "yes" : "no");
		}
		const auto& buffer = m_slot_buffers[entry.buffer];
		std::string compare;
		if (!entry.readable) {
			compare = "guest memory not readable";
		} else if (entry.differ == 0) {
			compare = "the same as guest memory";
		} else {
			compare = fmt::format("DIFFER from guest memory{}",
			                      entry.gpu_zero == guest.size() ? " (the GPU holds zeros)" : "");
		}
		return fmt::format(
		    "buffer 0x{:012x}+0x{:x} | page table: {} | BDA entry: {} | tracker: {} of 4 pages "
		    "CPU-dirty, {} GPU-written, {} hot{}{} | pending: {} | mapped: {} | {}{}",
		    buffer.CpuAddress(), buffer.Size(),
		    entry.owner_same ? "this buffer"
		    : entry.owner    ? "ANOTHER buffer"
		                     : "NONE",
		    entry.entry == entry.expected ? "right"
		    : entry.entry == 0            ? "EMPTY"
		                                  : "WRONG",
		    entry.states.cpu, entry.states.gpu, entry.states.hot,
		    entry.states.present == 0 ? ", no tracker region" : "",
		    entry.changed ? ", changed during the check" : "", entry.pending ? "yes" : "no",
		    entry.mapped ? "yes" : "no", compare,
		    entry.differ != 0 && entry.differ != guest.size()
		        ? fmt::format(" ({} of 4096 words)", entry.differ)
		        : std::string {});
	};
	size_t lines = 0;
	for (size_t i = 0; i < pages;) {
		const auto text = describe(info[i]);
		size_t     j    = i + 1;
		while (j < pages && describe(info[j]) == text) {
			j++;
		}
		const bool plain = info[i].covered && info[i].readable && info[i].differ == 0;
		if (quiet && plain) {
			i = j;
			continue;
		}
		if (lines++ < 64) {
			LOGF("%s: 0x%012" PRIx64 "..0x%012" PRIx64 " (%zu pages): %s\n", label.c_str(),
			     first + (uint64_t {i} << CACHING_PAGEBITS),
			     first + (uint64_t {j} << CACHING_PAGEBITS), j - i, text.c_str());
		}
		i = j;
	}
	if (lines > 64) {
		LOGF("%s: %zu more runs of pages not logged\n", label.c_str(), lines - 64);
	}
	const auto healable = stale + waiting;
	if (heal && healable != 0) {
		size_t made = 0;
		for (size_t i = 0; i < pages; i++) {
			const auto& entry = info[i];
			if (entry.covered && entry.readable && entry.differ != 0 && entry.states.gpu == 0 &&
			    !entry.changed) {
				const auto page = first + (uint64_t {i} << CACHING_PAGEBITS);
				made += RedirtyMapped(page, CACHING_PAGESIZE);
				SynchronizeBuffersInRange(page, CACHING_PAGESIZE);
			}
		}
		LOGF("%s: heal (KYTY_PROBE_LIST_HEAL): the %zu differing pages the GPU did not write "
		     "uploaded again from guest memory (%zu of their 4 KB pages were clean in the "
		     "tracker)\n",
		     label.c_str(), healable, made);
	}
	if (quiet && differing == 0) {
		return healable;
	}
	// Where the buffers over the range came from.
	BufferId previous {};
	for (const auto& entry: info) {
		if (!entry.covered || entry.buffer == previous) {
			continue;
		}
		previous           = entry.buffer;
		const auto& buffer = m_slot_buffers[entry.buffer];
		const auto  origin = m_buffer_origins.find(buffer.CpuAddress());
		if (origin == m_buffer_origins.end() || origin->second.size != buffer.Size()) {
			LOGF("%s: buffer 0x%012" PRIx64 "+0x%" PRIx64 " was created before the history\n",
			     label.c_str(), buffer.CpuAddress(), buffer.Size());
			continue;
		}
		LOGF("%s: buffer 0x%012" PRIx64 "+0x%" PRIx64 " was created at frame %" PRIu64
		     " for 0x%012" PRIx64 "+0x%" PRIx64 ", taking over %u buffers\n",
		     label.c_str(), buffer.CpuAddress(), buffer.Size(), origin->second.frame,
		     origin->second.request, origin->second.request_size, origin->second.absorbed);
	}
	std::vector<const BufferEvent*> events;
	for (size_t k = 0; k < m_buffer_events.size(); k++) {
		const auto  index = m_buffer_events.size() < 16384 ? k : (m_buffer_events_next + k) % 16384;
		const auto& event = m_buffer_events[index];
		if (event.address < last && event.address + event.size > first) {
			events.push_back(&event);
		}
	}
	const auto shown = std::min<size_t>(events.size(), 16);
	for (size_t k = events.size() - shown; k < events.size(); k++) {
		const auto& event = *events[k];
		LOGF("%s: history: frame %" PRIu64 " %s 0x%012" PRIx64 "+0x%" PRIx64 "\n", label.c_str(),
		     event.frame,
		     event.kind == BufferEventKind::Created  ? "created"
		     : event.kind == BufferEventKind::Joined ? "joined into a newer buffer"
		                                             : "collected",
		     event.address, event.size);
	}
	LOGF("%s: %zu buffer events over the range among the last %zu of %" PRIu64 "\n", label.c_str(),
	     events.size(), m_buffer_events.size(), m_buffer_events_total);
	return healable;
}

uint64_t BufferCache::VerifyCacheWithGuest(bool                                           heal,
                                           const std::function<bool(uint64_t, uint64_t)>& mapped) {
	if (m_scheduler.Current().IsInvalid()) {
		return 0;
	}
	constexpr uint64_t Chunk = 64ull << 20u;
	auto download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, Chunk);
	if (download->Mapped().data() == nullptr) {
		return 0;
	}
	struct Piece {
		uint64_t address        = 0;
		uint64_t size           = 0;
		uint64_t offset         = 0; // in the download buffer
		uint64_t buffer_address = 0;
		uint64_t buffer_size    = 0;
	};
	struct Run {
		uint64_t begin          = 0;
		uint64_t end            = 0;
		uint64_t buffer_address = 0;
		uint64_t buffer_size    = 0;
		bool     zero           = false;
		bool     mapped         = false;
		bool     untracked      = false; // no tracker region
	};
	std::vector<Piece>                     pieces;
	std::vector<MemoryTracker::PageStates> states; // each page of the pieces, in order
	std::vector<Run>                       stale_runs;
	std::vector<Run>                       lost_runs;
	std::vector<uint32_t>                  guest(CACHING_PAGESIZE / sizeof(uint32_t));
	uint64_t scanned = 0, clean = 0, stale = 0, stale_zero = 0, stale_unmapped = 0, lost = 0,
	         changed = 0, unreadable = 0, healed = 0, used = 0, untracked = 0, untracked_stale = 0;
	const auto add_run = [](std::vector<Run>& runs, const Piece& piece, uint64_t page, bool zero,
	                        bool in_mapped, bool no_region) {
		if (!runs.empty() && runs.back().end == page &&
		    runs.back().buffer_address == piece.buffer_address && runs.back().zero == zero &&
		    runs.back().mapped == in_mapped && runs.back().untracked == no_region) {
			runs.back().end += CACHING_PAGESIZE;
			return;
		}
		runs.push_back({page, page + CACHING_PAGESIZE, piece.buffer_address, piece.buffer_size,
		                zero, in_mapped, no_region});
	};
	Common::Timer timer;
	timer.Start();
	const auto flush = [&] {
		if (pieces.empty()) {
			return;
		}
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = download->Handle();
		barrier.offset              = 0;
		barrier.size                = VK_WHOLE_SIZE;
		m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
		                                                 nullptr, 1, &barrier, 0, nullptr);
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		download->Invalidate(0, used);
		size_t index = 0;
		for (const auto& piece: pieces) {
			for (uint64_t at = 0; at < piece.size; at += CACHING_PAGESIZE, index++) {
				const auto  page   = piece.address + at;
				const auto& before = states[index];
				// A page in no tracker region counts as CPU-dirty, but no BDA walk uploads it.
				const bool no_region = before.present == 0;
				if (page == 0 || before.cpu != 0 || before.gpu != 0 || before.hot != 0) {
					continue;
				}
				if (no_region) {
					untracked++;
				} else {
					clean++;
				}
				if (!Libs::LibKernel::Memory::TryReadBacking(page, guest.data(),
				                                             CACHING_PAGESIZE) &&
				    !Libs::LibKernel::Memory::TryReadSparseBacking(page, guest.data(),
				                                                   CACHING_PAGESIZE)) {
					unreadable++;
					continue;
				}
				const auto* held = download->Mapped().data() + piece.offset + at;
				if (std::memcmp(held, guest.data(), CACHING_PAGESIZE) == 0) {
					continue;
				}
				const auto after = m_memory_tracker.DescribePages(page, CACHING_PAGESIZE);
				if (after.present != before.present || after.cpu != 0 || after.gpu != 0 ||
				    after.hot != 0) {
					changed++;
					continue;
				}
				const bool zero      = std::all_of(held, held + CACHING_PAGESIZE,
				                                   [](uint8_t byte) { return byte == 0; });
				const bool in_mapped = mapped && mapped(page, CACHING_PAGESIZE);
				stale++;
				stale_zero += zero ? 1u : 0u;
				stale_unmapped += in_mapped ? 0u : 1u;
				untracked_stale += no_region ? 1u : 0u;
				add_run(stale_runs, piece, page, zero, in_mapped, no_region);
				if (heal) {
					healed += RedirtyMapped(page, CACHING_PAGESIZE) != 0 ? 1u : 0u;
					SynchronizeBuffersInRange(page, CACHING_PAGESIZE);
				}
			}
		}
		pieces.clear();
		states.clear();
		used = 0;
	};
	std::vector<std::pair<uint64_t, BufferId>> buffers(m_buffers.begin(), m_buffers.end());
	for (const auto& [address, id]: buffers) {
		if (IsBufferInvalid(id)) {
			continue;
		}
		const auto& buffer = m_slot_buffers[id];
		for (uint64_t offset = 0; offset < buffer.Size();) {
			const auto bytes = std::min(buffer.Size() - offset, Chunk - used);
			download->CopyFrom(m_scheduler.Current(), buffer, offset, used, bytes,
			                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
			                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
			                   vk::AccessFlagBits::eHostRead);
			const Piece piece {buffer.CpuAddress() + offset, bytes, used, buffer.CpuAddress(),
			                   buffer.Size()};
			pieces.push_back(piece);
			for (uint64_t at = 0; at < bytes; at += CACHING_PAGESIZE) {
				const auto page = piece.address + at;
				// Guest page 0 is never tracked (CreateBuffer).
				states.push_back(page == 0
				                     ? MemoryTracker::PageStates {}
				                     : m_memory_tracker.DescribePages(page, CACHING_PAGESIZE));
				if (states.back().cpu != 0 && !m_memory_tracker.IsBdaPending(page) && mapped &&
				    mapped(page, CACHING_PAGESIZE)) {
					lost++;
					add_run(lost_runs, piece, page, false, true, false);
				}
			}
			used += bytes;
			offset += bytes;
			scanned += bytes;
			if (used == Chunk) {
				flush();
			}
		}
	}
	flush();
	if (heal && lost != 0) {
		// After the scan, so that its later pages are judged as they were.
		for (const auto& run: lost_runs) {
			m_memory_tracker.MarkBdaPending(run.begin, run.end - run.begin);
		}
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
	}
	LOGF("Buffer verify: %zu buffers, %" PRIu64 " MB in %.1f s: %" PRIu64
	     " pages of 16 KB clean in the tracker and %" PRIu64 " in no tracker region, %" PRIu64
	     " of them STALE (the GPU's bytes differ from guest memory; %" PRIu64
	     " in no tracker region, %" PRIu64 " all zeros, %" PRIu64
	     " outside the mapped ranges); %" PRIu64
	     " LOST pages (CPU-dirty in mapped regions the BDA walk is not pending for); %" PRIu64
	     " changed during the check, %" PRIu64
	     " not readable; shader stores through addresses (which no range records): %s\n",
	     buffers.size(), scanned >> 20u, timer.GetTimeS(), clean, untracked, stale, untracked_stale,
	     stale_zero, stale_unmapped, lost, changed, unreadable,
	     AddressWritesSeen() ? "SEEN, a stale page may hold them" : "none");
	for (size_t i = 0; i < std::min<size_t>(stale_runs.size(), 24); i++) {
		const auto& run = stale_runs[i];
		LOGF("Buffer verify: stale 0x%012" PRIx64 "+0x%" PRIx64 " (%" PRIu64
		     " pages) in buffer 0x%012" PRIx64 "+0x%" PRIx64 ", the GPU holds %s, %s, %s\n",
		     run.begin, run.end - run.begin, (run.end - run.begin) >> CACHING_PAGEBITS,
		     run.buffer_address, run.buffer_size, run.zero ? "zeros" : "other bytes",
		     run.mapped ? "mapped" : "NOT mapped",
		     run.untracked ? "in NO tracker region" : "clean in the tracker");
	}
	for (size_t i = 0; i < std::min<size_t>(lost_runs.size(), 8); i++) {
		const auto& run = lost_runs[i];
		LOGF("Buffer verify: lost 0x%012" PRIx64 "+0x%" PRIx64 " in buffer 0x%012" PRIx64
		     "+0x%" PRIx64 "\n",
		     run.begin, run.end - run.begin, run.buffer_address, run.buffer_size);
	}
	if (stale_runs.size() > 24 || lost_runs.size() > 8) {
		LOGF("Buffer verify: %zu stale runs and %zu lost runs in all\n", stale_runs.size(),
		     lost_runs.size());
	}
	if (heal && (stale != 0 || lost != 0)) {
		LOGF("Buffer verify: heal (KYTY_VERIFY_BUFFERS_HEAL): %" PRIu64
		     " stale pages uploaded again (%" PRIu64
		     " had clean 4 KB pages to make CPU-dirty), %" PRIu64
		     " lost pages made pending for the BDA walk\n",
		     stale, healed, lost);
	}
	return stale;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
