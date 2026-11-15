#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

// The upload ring. 512 MB held ~0.5 GB of commit for uploads of ~50 MB a frame at Marvel's Wolverine's Trask outpost; a
// larger transfer takes a one-time buffer either way (UploadMemory). KYTY_STAGING_MB=512 restores the old size.
uint64_t StagingBufferSize() {
	const char* value = std::getenv("KYTY_STAGING_MB");
	const auto  mb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
	return (mb >= 16 && mb <= 4096 ? mb : 256) * MiB;
}

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Recorder();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
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
		EXIT_IF(!inserted && !KYTY_UPSTREAM_PATH("21f3b9de6"));
		{
			std::lock_guard lock(m_registered_mutex);
			m_registered.Add(buffer.CpuAddress(), buffer.Size());
		}
		m_total_used_memory += buffer.Size();
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		m_bda_pagetable_buffer.Back(table_offset, addresses.size() * sizeof(vk::DeviceAddress));
		{
			std::lock_guard lock(m_bda_sync_mutex);
			m_dma_accessed.ForEachInRange(
			    buffer.CpuAddress(), buffer.Size(), [&](uint64_t begin, uint64_t end) {
				    for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
					    addresses[buffer.Offset(address) >> CACHING_PAGEBITS] += BDA_ACCESSED;
				    }
			    });
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF((found == m_buffers.end() || found->second != id) && !KYTY_UPSTREAM_PATH("21f3b9de6"));
		m_buffers.erase(found);
		{
			std::lock_guard lock(m_registered_mutex);
			m_registered.Subtract(buffer.CpuAddress(), buffer.Size());
		}
		EXIT_IF(buffer.Size() > m_total_used_memory && !KYTY_UPSTREAM_PATH("21f3b9de6"));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted && buffer.lru_tick < m_gc_tick) {
		buffer.lru_tick = m_gc_tick;
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	std::erase(m_sparse_buffers, id);
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	std::erase_if(ranges, [&](const std::pair<uint64_t, uint64_t>& range) {
		const auto first = copies.size();
		m_memory_tracker.ForEachDownloadRange<false>(
		    range.first, range.second, [&](uint64_t address, uint64_t bytes) noexcept {
			    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
			                                           "buffer download");
			    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
				    if (m_in_read_back) {
					    for (auto page = start / TRACKER_PAGE_SIZE; page <= (end - 1) / TRACKER_PAGE_SIZE; page++) {
						    m_preflush.OnReadBack(page, m_gc_tick);
					    }
				    }
				    copies.emplace_back(start - buffer_address, total_size, end - start);
				    // Keep packed ranges on separate cache lines, as in shadPS4.
				    total_size += Common::AlignUp(end - start, 64);
			    });
			    m_gpu_modified_ranges.Subtract(address, bytes);
			    m_known_written.Subtract(address, bytes);
			    ForgetColorMetadataChecks(address, bytes);
		    });
		return copies.size() == first;
	});
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
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
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [total_size, buffer_address, copies = std::move(copies),
	                download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(
			    buffer_address + copy.srcOffset,
			    download.mapped + (copy.dstOffset - download.offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE, graphics.sparse_page_table),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, StagingBufferSize()),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_track_device(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                     TRACK_WORDS * sizeof(uint32_t)),
      m_track_host(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
                   TRACK_WORDS * sizeof(uint32_t)),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		if (!buffer->IsSparse()) buffer->Fill(0, buffer->Size(), 0); // a sparse one's unbound pages read 0
	}
	// Zeroed: an empty guest buffer is bound to it, and with robustBufferAccess2 its loads have no
	// guard of their own (its stores and atomics do: its length in the shader data is 0).
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::Upload, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	std::memset(GetBuffer(null_id).Mapped().data(), 0, static_cast<size_t>(GetBuffer(null_id).Size()));
	GetBuffer(null_id).Flush(0, GetBuffer(null_id).Size());
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	m_scheduler.SetEndHook([this] { RecordPreflushCopies(); });
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
	m_budget_gc_memory   = std::max(static_cast<uint64_t>(budget), m_critical_gc_memory);
}

BufferCache::~BufferCache() {
	m_scheduler.SetEndHook({});
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

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	g_caches_touched.fetch_add(1, std::memory_order_relaxed);
	if (!GuestRange {vaddr, size}.Valid() && !KYTY_UPSTREAM_PATH("25fb20e53")) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size, true);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	g_caches_touched.fetch_add(1, std::memory_order_relaxed);
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendReadBackSync([this, vaddr, size, is_write] {
		// A read of what a tracked dispatch may have written: the bitmaps that have come back clear what it did not.
		if (std::ranges::any_of(m_tracked_writes, [&](const TrackedWrite& write) {
			    return vaddr < write.vaddr + write.size && write.vaddr < vaddr + size;
		    })) {
			ResolveTrackedWrites();
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		m_in_read_back = true;
		// A game that reads what its shaders wrote reads the same pages again every frame (Marvel's
		// Wolverine's cabin title: ~770 ray-tracing structure headers a frame, from 7 threads), and
		// each read back waits for all GPU work. The first page of the previous frame's read backs to
		// come back takes along the others that shaders wrote again since, in the same wait.
		std::vector<uint64_t> read_ahead;
		if (!is_write) {
			const auto page = vaddr / TRACKER_PAGE_SIZE;
			if (!m_read_ahead_done && m_previous_read_backs.contains(page)) {
				m_read_ahead_done = true;
				// One download per buffer: a download's barriers drain the GPU, and these are thousands of
				// pages a frame (Marvel's Wolverine's title scenes: ~4,000, ~11 ms of GPU time one at a time).
				std::map<uint64_t, std::pair<BufferId, std::vector<std::pair<uint64_t, uint64_t>>>> buffers;
				for (const auto& [other, address]: m_previous_read_backs) {
					if (other != page && IsRegionRegistered(address, 1) &&
					    m_memory_tracker.IsRegionGpuModified(address, 1)) {
						if (PublishPreflushed(address, 1)) {
							m_read_backs[other] = address;
							continue;
						}
						const auto id    = FindBuffer(address, 1);
						auto&      entry = buffers[m_slot_buffers[id].CpuAddress()];
						entry.first      = id;
						entry.second.emplace_back(address, 1);
					}
				}
				for (auto& [base, entry]: buffers) {
					auto& [id, pages] = entry;
					if (!DownloadBufferMemory<true>(m_slot_buffers[id], pages)) continue;
					for (const auto& [address, size]: pages) {
						read_ahead.push_back(address);
						m_read_backs[address / TRACKER_PAGE_SIZE] = address;
					}
				}
			}
			m_read_backs[page] = vaddr;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		// Read backs come in groups: a game writes its records of a table that a shader also
		// writes, a page at a time, every frame, and each read back is a wait for the GPU. One in
		// the megabyte of the buffer's previous one takes that megabyte along. Not the
		// invalidation of a range (an unmap), which names what it wants: a write fault names a byte.
		const auto megabyte = vaddr / MiB;
		const bool follows  = (!is_write || size == 1u) && megabyte == buffer.read_back &&
		                      (vaddr + size - 1u) / MiB == megabyte;
		buffer.read_back    = megabyte;
		const auto begin    = !follows ? vaddr : std::max(buffer.CpuAddress(), megabyte * MiB);
		const auto end      = !follows ? vaddr + size
		                               : std::min(buffer.CpuAddress() + buffer.Size(), (megabyte + 1u) * MiB);
		if (read_ahead.empty()) {
			if (!PublishPreflushed(begin, end - begin) && DownloadBufferMemory<false>(buffer, begin, end - begin)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
			}
		} else {
			const bool own = !PublishPreflushed(begin, end - begin) && DownloadBufferMemory<true>(buffer, begin, end - begin);
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Wait(tick);
			m_scheduler.WaitPriorityOperations(tick);
			if (own) {
				m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
			}
			for (const auto address: read_ahead) {
				m_memory_tracker.UnmarkRegionAsGpuModified(address, 1);
			}
		}
		m_in_read_back = false;
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size, true);
		}
	});
}

void BufferCache::PreflushWrite(uint64_t vaddr, uint64_t size) {
	if (size != 0) {
		m_preflush.OnWrite(vaddr / TRACKER_PAGE_SIZE, (vaddr + size - 1) / TRACKER_PAGE_SIZE, m_scheduler.CurrentTick());
	}
}

void BufferCache::RecordPreflushCopies() {
	// By source buffer and slot chunk: one copy command each.
	std::map<std::pair<VkBuffer, uint32_t>, std::vector<vk::BufferCopy>> copies;
	uint64_t bytes = 0;
	m_preflush.TakeCopies(m_scheduler.CurrentTick(), [&](uint64_t page, uint32_t slot) {
		const auto  address = page * TRACKER_PAGE_SIZE;
		const auto* owner   = m_page_table.Find(address >> PageTable::kPageBits);
		if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(address, TRACKER_PAGE_SIZE)) {
			return false; // a slot holds a whole page from one buffer, or nothing
		}
		const auto& buffer = m_slot_buffers[*owner];
		const auto  chunk  = slot / PREFLUSH_CHUNK_SLOTS;
		while (m_preflush_chunks.size() <= chunk) {
			m_preflush_chunks.push_back(std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                                     vk::BufferUsageFlagBits::eTransferDst,
			                                                     PREFLUSH_CHUNK_SLOTS * TRACKER_PAGE_SIZE));
		}
		copies[{static_cast<VkBuffer>(buffer.Handle()), chunk}].emplace_back(
		    buffer.Offset(address), (slot % PREFLUSH_CHUNK_SLOTS) * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
		bytes += TRACKER_PAGE_SIZE;
		return true;
	});
	if (copies.empty()) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferPreflushBytes, bytes);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before,
	                       0, nullptr, 0, nullptr);
	for (const auto& [key, regions]: copies) {
		native.copyBuffer(key.first, m_preflush_chunks[key.second]->Handle(), static_cast<uint32_t>(regions.size()),
		                  regions.data());
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &after, 0,
	                       nullptr, 0, nullptr);
}

bool BufferCache::PublishPreflushed(uint64_t vaddr, uint64_t size) {
	// The GPU-written bytes a download would copy, and the newest copy among their pages.
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes,
		                                     [&](uint64_t start, uint64_t end) { ranges.emplace_back(start, end); });
	});
	if (ranges.empty()) {
		return false;
	}
	uint64_t tick = 0;
	for (const auto& [start, end]: ranges) {
		for (auto page = start / TRACKER_PAGE_SIZE; page <= (end - 1) / TRACKER_PAGE_SIZE; page++) {
			const auto found = m_preflush.Lookup(page);
			if (!found) {
				return false;
			}
			tick = std::max(tick, found->second);
		}
	}
	EXIT_IF(tick >= m_scheduler.CurrentTick());
	if (!m_scheduler.IsFree(tick)) {
		m_scheduler.Wait(tick);
	}
	// An asynchronous download published after this would write older bytes (the read-ahead's).
	m_scheduler.WaitPriorityOperations(tick);
	for (const auto& [start, end]: ranges) {
		for (auto first = start; first < end;) {
			const auto page  = first / TRACKER_PAGE_SIZE;
			const auto last  = std::min(end, (page + 1) * TRACKER_PAGE_SIZE);
			const auto slot  = m_preflush.Lookup(page)->first;
			auto&      chunk = *m_preflush_chunks[slot / PREFLUSH_CHUNK_SLOTS];
			const auto offset = (slot % PREFLUSH_CHUNK_SLOTS) * TRACKER_PAGE_SIZE + (first - page * TRACKER_PAGE_SIZE);
			chunk.Invalidate(offset, last - first);
			Libs::LibKernel::Memory::WriteBacking(first, chunk.Mapped().data() + offset, last - first);
			first = last;
		}
	}
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes, "pre-flush publish");
		m_gpu_modified_ranges.Subtract(address, bytes);
		m_known_written.Subtract(address, bytes);
		ForgetColorMetadataChecks(address, bytes);
	});
	for (const auto& [start, end]: ranges) {
		for (auto page = start / TRACKER_PAGE_SIZE; page <= (end - 1) / TRACKER_PAGE_SIZE; page++) {
			m_preflush.OnReadBack(page, m_gc_tick);
		}
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferPreflushHits);
	return true;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
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
	m_joins++;
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	// Its bytes go to backed pages only: a page left on the zero page must stay zero. An old sparse buffer's backed pages,
	// all of an old dense one.
	if (new_buffer.IsSparse()) {
		const auto base = overlap.CpuAddress() - new_buffer.CpuAddress();
		if (!overlap.IsSparse()) {
			new_buffer.Back(base, overlap.Size());
		} else {
			for (uint64_t offset = 0; offset < overlap.Size(); offset += overlap.PageSize()) {
				if (overlap.IsBacked(offset)) new_buffer.Back(base + offset, std::min(overlap.PageSize(), overlap.Size() - offset));
			}
		}
	}
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF((m_scheduler.Current().IsInvalid()) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto bytes = overlap.end - overlap.begin;
	if (m_graphics.max_buffer_size != 0 && bytes > m_graphics.max_buffer_size) {
		LOGF("BufferCache: a buffer of %llu MiB at guest 0x%016" PRIx64 " is larger than the device's maxBufferSize (%llu MiB)\n",
		     static_cast<unsigned long long>(bytes >> 20u), overlap.begin,
		     static_cast<unsigned long long>(m_graphics.max_buffer_size >> 20u));
	}
	// KYTY_SPARSE_GEOMETRY: a buffer of 1 GiB or more has memory only where the guest has (Marvel's Wolverine's 2.7 GB geometry
	// pool with the game's lazy commit: ~0.8 GB mapped at Trask).
	const bool sparse = m_graphics.sparse_buffers && bytes >= (uint64_t {1} << 30u);
	const auto id     = m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	                                          AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, bytes, sparse,
	                                          sparse && m_graphics.sparse_buffers_aliased);
	auto& buffer = m_slot_buffers[id];
	if (buffer.IsSparse()) {
		m_sparse_buffers.push_back(id);
		BackCommitted(buffer, overlap.begin, bytes);
		LOGF("BufferCache: sparse buffer of %llu MiB at guest 0x%016" PRIx64 ", %llu MiB backed\n",
		     static_cast<unsigned long long>(bytes >> 20u), overlap.begin,
		     static_cast<unsigned long long>(buffer.BackedBytes() >> 20u));
	}
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

void BufferCache::NoteMapped(uint64_t vaddr, uint64_t size) {
	if (!m_graphics.sparse_buffers) return;
	std::lock_guard lock(m_pending_maps_mutex);
	m_pending_maps.Add(vaddr, size);
	m_has_pending_maps.store(true, std::memory_order_release);
}

void BufferCache::BackPendingMaps() {
	RangeSet maps;
	{
		std::lock_guard lock(m_pending_maps_mutex);
		maps = std::exchange(m_pending_maps, {});
		m_has_pending_maps.store(false, std::memory_order_relaxed);
	}
	for (const auto id: m_sparse_buffers) {
		auto& buffer = m_slot_buffers[id];
		maps.ForEachInRange(buffer.CpuAddress(), buffer.Size(),
		                    [&](uint64_t begin, uint64_t end) { BackCommitted(buffer, begin, end - begin); });
	}
}

void BufferCache::BackCommitted(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// Guest pages are 16 KiB; a sparse page is backed when any of them holds memory.
	constexpr uint64_t GuestPage = 16 * 1024;
	const auto         page_size = buffer.PageSize();
	const auto         first     = Common::AlignDown(buffer.Offset(vaddr), page_size);
	const auto         end       = std::min(buffer.Size(), buffer.Offset(vaddr) + size);
	for (uint64_t offset = first; offset < end; offset += page_size) {
		if (buffer.IsBacked(offset)) continue;
		const auto bytes = std::min(page_size, buffer.Size() - offset);
		for (uint64_t part = 0; part < bytes; part += GuestPage) {
			if (Libs::LibKernel::Memory::IsCommittedRange(buffer.CpuAddress() + offset + part, 1)) {
				buffer.Back(offset, bytes);
				break;
			}
		}
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (m_has_pending_maps.load(std::memory_order_acquire) && !m_sparse_buffers.empty()) {
		BackPendingMaps();
	}
	UploadBatch batch;
	vk::Buffer  source;
	CollectUploads(buffer, vaddr, size, is_written, batch, &source);
	if (!source) {
		UploadInParts(batch); // copies that outgrew half the ring (none: nothing to do)
	}
	RecordUploads(source, batch);
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::CollectUploads(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                 UploadBatch& batch, vk::Buffer* source) {
	// A draw binds the same buffers again and again, one of them gigabytes: a range that was
	// found clean is walked again only after a page in its part of the address space got dirty.
	const bool known_clean = !is_written && vaddr >= buffer.clean_begin && size <= buffer.clean_end - vaddr &&
	                         buffer.clean_end > vaddr &&
	                         RegionManager::CpuDirtied(buffer.clean_begin, buffer.clean_end - buffer.clean_begin) ==
	                             buffer.clean_count;
	if (known_clean) {
		return;
	}
	const auto count      = RegionManager::CpuDirtied(vaddr, size);
	// Nor does a range none of whose regions has a CPU-dirty page: the walk would take each region's lock to find
	// nothing (Marvel's Wolverine's Trask outpost: 7,600 such walks a frame over the geometry heap's parts, 99.6 % of the
	// read walks, ~4 ms of the GPU thread). The count is read first, so a write after it shows at the next bind.
	if (!is_written && m_memory_tracker.IsRangeCpuClean(vaddr, size)) {
		buffer.clean_begin = vaddr;
		buffer.clean_end   = vaddr + size;
		buffer.clean_count = count;
		return;
	}
	const auto first_copy = batch.copies.size();
	// Known-zero runs are filled on the GPU: reading guest memory nothing wrote would make Windows back
	// every page with a fresh zeroed page of RAM (Marvel's Wolverine: ~2.7 GB in its first 12 s).
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes, bool zero) noexcept {
		    if (zero) {
			    batch.fills.push_back({&buffer, buffer.Offset(address), bytes});
			    return;
		    }
		    if (batch.uploads.empty() || batch.uploads.back().buffer != &buffer) {
			    batch.uploads.push_back({&buffer, 0});
		    }
		    batch.copies.emplace_back(batch.total_size, buffer.Offset(address), bytes);
		    batch.uploads.back().end = batch.copies.size();
		    batch.total_size += bytes;
	    },
	    [&]() noexcept {
		    // More than the upload ring holds goes through it in parts after the walk: a one-time host buffer of the
		    // whole size was charged to commit in full (Marvel's Wolverine, Pro mode at Trask: 2.8 GB for the geometry
		    // pool, session 60).
		    if (source != nullptr && batch.total_size <= m_staging_buffer.Size() / 2) {
			    *source = UploadCopies(batch.uploads, batch.copies, batch.total_size);
		    }
	    });
	// Never data onto the zero page: back the copies' pages first (their fills are recorded before the copy below).
	// Not inside the walk: that may hold the tracker's lock, and a guest map takes the two the other way round.
	if (buffer.IsSparse()) {
		for (size_t i = first_copy; i < batch.copies.size(); ++i) {
			BackCommitted(buffer, buffer.CpuAddress() + batch.copies[i].dstOffset, batch.copies[i].size);
		}
	}
	buffer.clean_begin = vaddr;
	buffer.clean_end   = is_written ? vaddr : vaddr + size;
	buffer.clean_count = count;
}

vk::Buffer BufferCache::UploadCopies(std::span<const BufferUpload> uploads, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    size_t first = 0;
		    for (const auto& upload: uploads) {
			    for (auto& copy: copies.subspan(first, upload.end - first)) {
				    const auto address = upload.buffer->CpuAddress() + copy.dstOffset;
				    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
					                                         copy.size)) {
					    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				    }
				    copy.srcOffset += base_offset;
			    }
			    first = upload.end;
		    }
	    });
	return source.first->Handle();
}

void BufferCache::UploadInParts(Buffer& buffer, std::span<const vk::BufferCopy> copies) {
	// Half the ring per part: one part fills while the one before it is copied.
	const auto                  limit = m_staging_buffer.Size() / 2;
	std::vector<vk::BufferCopy> part;
	uint64_t                    part_size = 0;
	const auto                  send      = [&] {
        Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, part_size);
        // Map may submit the current commands and wait for an earlier part, so the command buffer is taken after it.
        const auto [mapped, offset] = m_staging_buffer.Map(part_size, 4);
        EXIT_IF(mapped == nullptr);
        for (auto& copy: part) {
            if (!LibKernel::Memory::CopySparseMemory(buffer.CpuAddress() + copy.dstOffset, mapped + copy.srcOffset,
                                                     copy.size)) {
                EXIT("BufferCache: failed to read sparse guest buffer memory\n");
            }
            copy.srcOffset += offset;
        }
        m_staging_buffer.Commit();
        auto& command = m_scheduler.Current();
        command.EndRendering();
        const auto              native = command.Recorder();
        vk::BufferMemoryBarrier before {};
        before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
                               vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
        before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.buffer              = buffer.Handle();
        before.offset              = 0;
        before.size                = buffer.Size();
        native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
        native.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), static_cast<uint32_t>(part.size()), part.data());
        auto after          = before;
        after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
        native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
        part.clear();
        part_size = 0;
	};
	for (const auto& copy: copies) {
		for (uint64_t done = 0; done < copy.size;) {
			const auto bytes = std::min(copy.size - done, limit - part_size);
			part.emplace_back(part_size, copy.dstOffset + done, bytes);
			part_size += bytes;
			done += bytes;
			if (part_size == limit) send();
		}
	}
	if (!part.empty()) send();
}

// The batch's copies, each buffer's through the ring in parts (session 60); its fills stay for RecordUploads.
void BufferCache::UploadInParts(const UploadBatch& batch) {
	size_t first = 0;
	for (const auto& upload: batch.uploads) {
		UploadInParts(*upload.buffer, std::span(batch.copies).subspan(first, upload.end - first));
		first = upload.end;
	}
}

void BufferCache::RecordUploads(vk::Buffer source, const UploadBatch& batch) {
	if (!source && batch.fills.empty()) {
		return;
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Recorder();
	const vk::MemoryBarrier before {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                 vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eTransferWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before, 0, nullptr, 0, nullptr);
	if (source) {
		size_t first = 0;
		for (const auto& upload: batch.uploads) {
			native.copyBuffer(source, upload.buffer->Handle(), static_cast<uint32_t>(upload.end - first),
			                  batch.copies.data() + first);
			first = upload.end;
		}
	}
	for (const auto& fill: batch.fills) {
		native.fillBuffer(fill.buffer->Handle(), fill.offset, fill.bytes, 0);
	}
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                               .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
	                                                vk::AccessFlagBits::eMemoryWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, uint64_t written_offset,
                                                       uint64_t written_size,
                                                       std::span<const std::pair<uint64_t, uint64_t>> written_ranges,
                                                       bool tracked) {
	auto& command = m_scheduler.Current();
	if ((command.IsInvalid() && !KYTY_UPSTREAM_PATH("e827c92a2")) || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// The same constants are bound by draw after draw: within one command buffer the stream buffer keeps a copy
		// (a wrap never reaches the current tick's data), and the guest can only change them under a packet that
		// writes or waits on memory (NewStreamEpoch).
		const auto tick = m_scheduler.CurrentTick();
		auto& copy = m_stream_copies[((vaddr >> 4u) * 0x9E3779B97F4A7C15ull >> 54u) ^ (size & 1023u)];
		if (copy.vaddr == vaddr && copy.size == size && copy.tick == tick && copy.epoch == m_stream_epoch) {
			return {&m_stream_buffer, copy.offset};
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			// CopySparseMemory reads private memory at its guest address under the address-space lock, which a
			// fault's protection change needs: no page of the range may be kept read-protected.
			m_memory_tracker.ReleaseKeptRange(vaddr, size);
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			copy = {vaddr, size, tick, m_stream_epoch, offset};
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// A written binding whose stores reach only part of it: all of it is synchronized, only that
	// part becomes GPU-written.
	const auto written = is_written ? std::min(written_size, size - std::min(written_offset, size)) : 0u;
	(void)SynchronizeBuffer(buffer, vaddr, size, written == size, is_texel_buffer);
	if (written != 0u && !written_ranges.empty()) {
		// A scatter's stores (ResourceSnapshot::buffer_write_ranges): only the pages they reach become
		// GPU-written, in runs of whole pages; the binding's other bytes were synchronized above.
		uint64_t run_begin = 0, run_end = 0;
		const auto mark = [&] {
			if (run_end <= run_begin) return;
			(void)SynchronizeBuffer(buffer, run_begin, run_end - run_begin, true, is_texel_buffer);
			m_gpu_modified_ranges.Add(run_begin, run_end - run_begin);
			PreflushWrite(run_begin, run_end - run_begin);
			ForgetColorMetadataChecks(run_begin, run_end - run_begin);
			m_known_written.Add(run_begin, run_end - run_begin);
		};
		for (const auto& [first, last]: written_ranges) {
			if (first >= size) break;
			const auto begin = std::max(vaddr, Common::AlignDown(vaddr + first, TRACKER_PAGE_SIZE));
			const auto end   = std::min(vaddr + size, Common::AlignUp(vaddr + std::min(last, size), TRACKER_PAGE_SIZE));
			if (begin > run_end) {
				mark();
				run_begin = begin;
			}
			run_end = std::max(run_end, end);
		}
		mark();
		return {&buffer, buffer.Offset(vaddr)};
	}
	if (written != 0u && written != size) {
		(void)SynchronizeBuffer(buffer, vaddr + written_offset, written, true, is_texel_buffer);
	}
	if (written != 0u) {
		m_gpu_modified_ranges.Add(vaddr + written_offset, written);
		PreflushWrite(vaddr + written_offset, written);
		ForgetColorMetadataChecks(vaddr + written_offset, written);
		if (!tracked) {
			m_known_written.Add(vaddr + written_offset, written);
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

uint64_t BufferCache::TrackWrites(const Buffer& buffer, uint64_t base, uint64_t vaddr, uint64_t size) {
	// No resolve here: this binding is marked GPU-written but not queued yet, so an earlier bitmap would clear it.
	constexpr uint64_t word_bytes = 32u * TRACKER_PAGE_SIZE;
	const auto         words      = static_cast<uint32_t>(
        std::min<uint64_t>((vaddr + size - base + word_bytes - 1u) / word_bytes, TRACK_WORDS));
	// A bitmap costs a fill and a copy a dispatch: only for what the CPU reads, a buffer read back before.
	uint32_t offset = m_tracked_writes.empty() ? 0u : m_track_head;
	bool     fits   = buffer.read_back != UINT64_MAX && words * word_bytes >= vaddr + size - base;
	if (fits && !m_tracked_writes.empty()) {
		const auto tail = m_tracked_writes.front().offset;
		if (offset < tail) {
			fits = offset + words < tail;
		} else if (offset + words > TRACK_WORDS) {
			offset = 0;
			fits   = words < tail;
		}
	}
	if (!fits) {
		m_known_written.Add(vaddr, size);
		return 0;
	}
	m_track_head  = offset + words;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	// The dispatch's ShaderWriteHazardBarrier orders the fill before its writes.
	command.Recorder().fillBuffer(m_track_device.Handle(), offset * sizeof(uint32_t), words * sizeof(uint32_t), 0);
	m_tracked_writes.push_back({.base = base, .vaddr = vaddr, .size = size, .offset = offset, .words = words});
	return m_track_device.BufferDeviceAddress() + offset * sizeof(uint32_t);
}

void BufferCache::CloseTrackedWrites() {
	std::vector<vk::BufferCopy> copies;
	for (auto it = m_tracked_writes.rbegin(); it != m_tracked_writes.rend() && it->tick == 0; ++it) {
		it->tick = m_scheduler.CurrentTick();
		copies.emplace_back(it->offset * sizeof(uint32_t), it->offset * sizeof(uint32_t), it->words * sizeof(uint32_t));
	}
	if (copies.empty()) {
		return;
	}
	// The dispatch's ShaderAccessBarrier orders its writes before the copy.
	const auto native = m_scheduler.Current().Recorder();
	native.copyBuffer(m_track_device.Handle(), m_track_host.Handle(), static_cast<uint32_t>(copies.size()),
	                  copies.data());
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &barrier,
	                       0, nullptr, 0, nullptr);
}

void BufferCache::ResolveTrackedWrites() {
	while (!m_tracked_writes.empty() && m_tracked_writes.front().tick != 0 &&
	       m_scheduler.IsFree(m_tracked_writes.front().tick)) {
		const auto write = m_tracked_writes.front();
		m_tracked_writes.pop_front();
		// A later tracked dispatch may still write what this one did not. ResolveWriteBitmap looks at keep within
		// this write's range only, so only the later writes that overlap it go in: all of them, a map node each
		// for every write resolved, were a third of the GPU thread in Marvel's Wolverine's title-scene load.
		RangeSet keep;
		for (const auto& later: m_tracked_writes) {
			if (later.vaddr < write.vaddr + write.size && write.vaddr < later.vaddr + later.size) {
				keep.Add(later.vaddr, later.size);
			}
		}
		m_track_host.Invalidate(write.offset * sizeof(uint32_t), write.words * sizeof(uint32_t));
		const std::span bits(reinterpret_cast<const uint32_t*>(m_track_host.Mapped().data()) + write.offset,
		                     write.words);
		// Tracker pages with no GPU-written byte left, in runs.
		uint64_t                                   run_begin = 0, run_end = 0;
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		const auto                                 unmark = [&] {
			if (run_end > run_begin) {
				runs.emplace_back(run_begin, run_end);
			}
		};
		ResolveWriteBitmap(m_gpu_modified_ranges, m_known_written, keep, write.base, write.vaddr, write.size, bits,
		                   [&](uint64_t begin, uint64_t end) {
			                   // Within [begin, end) no GPU-written byte is left: only its first and last
			                   // tracker pages can keep one outside it.
			                   auto first = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
			                   auto last  = Common::AlignUp(end, TRACKER_PAGE_SIZE);
			                   if (m_gpu_modified_ranges.Intersects(first, TRACKER_PAGE_SIZE)) first += TRACKER_PAGE_SIZE;
			                   if (last > first && m_gpu_modified_ranges.Intersects(last - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE)) {
				                   last -= TRACKER_PAGE_SIZE;
			                   }
			                   if (first >= last) return;
			                   if (first != run_end) {
				                   unmark();
				                   run_begin = first;
			                   }
			                   run_end = last;
		                   });
		unmark();
		// Not written by the dispatch: no longer GPU-written, but kept read-protected, so that the next tracked
		// dispatch over the binding marks it with no protection change (the CPU's first touch lifts it). Arming and
		// lifting a whole binding every dispatch cost Marvel's Wolverine's burning-jeep fight ~7 ms of its GPU thread
		// a frame (4,000 calls and 3 M pages a second each way). All runs in one call: a write over its 2.7 GB
		// geometry heap has thousands, after a close-up title scene.
		m_memory_tracker.KeepGpuProtection(runs);
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid() && !KYTY_UPSTREAM_PATH("6d3891362")) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	// Known-zero runs (a new render target's untouched memory) are written as zeros, not read.
	std::vector<std::pair<uint64_t, uint64_t>> zeros;
	m_memory_tracker.ForEachZeroRange(vaddr, size, [&](uint64_t address, uint64_t bytes) {
		const auto begin = std::max(address, vaddr);
		const auto end   = std::min(address + bytes, vaddr + size);
		if (begin < end) {
			zeros.emplace_back(begin, end);
		}
	});
	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		const auto read = [&](uint64_t begin, uint64_t end) {
			if (begin < end &&
			    !Libs::LibKernel::Memory::TryReadSparseBacking(begin, mapped + (begin - vaddr), end - begin)) {
				EXIT("BufferCache: failed to read mapped guest image backing\n");
			}
		};
		auto cursor = vaddr;
		for (const auto& [begin, end]: zeros) {
			read(cursor, begin);
			std::memset(mapped + (begin - vaddr), 0, end - begin);
			cursor = end;
		}
		read(cursor, vaddr + size);
	});
}

bool BufferCache::ReleaseKept(uint64_t vaddr) {
	return m_memory_tracker.ReleaseKept(vaddr, 256u * 1024u);
}

void BufferCache::MarkMemoryZero(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.MarkRegionAsZero(vaddr, size);
}

void BufferCache::UnmarkMemoryZero(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.UnmarkRegionAsZero(vaddr, size);
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
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::ReadGuestBytes(uint64_t vaddr, void* data, uint64_t size) {
	if (HasGpuDirtyBytes(vaddr, size)) {
		ReadMemory(vaddr, size);
	}
	if (!Libs::LibKernel::Memory::TryReadBufferBacking(vaddr, data, size)) {
		std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
	}
}

bool BufferCache::WriteGuestBytes(uint64_t vaddr, const void* data, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid() || !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	Libs::LibKernel::Memory::WriteBacking(vaddr, data, size);
	WriteDataBuffer(m_slot_buffers[FindBuffer(vaddr, size)], vaddr, data, size);
	// A pre-flushed copy from before this store would undo it on the backing.
	PreflushWrite(vaddr, size);
	// The backing now holds these bytes, and the buffer copy above lands after every earlier GPU write to them, so a
	// read back would only copy the same bytes again: they are no longer GPU-dirty. Later GPU writes mark them anew.
	m_gpu_modified_ranges.Subtract(vaddr, size);
	m_known_written.Subtract(vaddr, size);
	ForgetColorMetadataChecks(vaddr, size);
	return true;
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
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
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

bool BufferCache::HoldsRegion(uint64_t vaddr, uint64_t size) const {
	std::lock_guard lock(m_registered_mutex);
	return m_registered.Intersects(vaddr, size);
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

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	ResolveTrackedWrites();
	// A new frame (see ReadMemory).
	m_previous_read_backs = std::exchange(m_read_backs, {});
	m_read_ahead_done     = false;
	m_preflush.Expire(m_gc_tick + 1);
	const auto tick       = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted && !KYTY_UPSTREAM_PATH("21f3b9de6"));
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		// A buffer a shader wrote is read back to be deleted, and uploaded again at its next use. The device's
		// memory is mostly images, which this frees none of: only at the budget itself (Marvel's Wolverine's
		// PRESENT DAY at 12.4 of 13.3 GB: ~120 buffers, ~600 MB read back in two seconds).
		if (dirty && m_total_used_memory < m_budget_gc_memory) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			if (m_record_released) {
				m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
			}
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if ((m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		     m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) &&
		    !KYTY_UPSTREAM_PATH("21f3b9de6")) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		std::erase(m_sparse_buffers, id); // as DeleteBuffer: BackPendingMaps walks these ids
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

const Buffer& BufferCache::GetProgramConstants(const std::shared_ptr<const std::vector<uint32_t>>& words) {
	EXIT_IF(words == nullptr || words->empty());
	auto& [owner, buffer] = m_program_constants[words.get()];
	if (buffer == nullptr) {
		// Mapped, as the GDS buffer: the words are written before any command that reads them is submitted.
		const auto size = words->size() * sizeof(uint32_t);
		owner           = words;
		buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		                                  vk::BufferUsageFlagBits::eStorageBuffer, size);
		std::memcpy(buffer->Mapped().data(), words->data(), size);
		buffer->Flush(0, size);
		SetVulkanObjectNameF(m_graphics.device, buffer->Handle(), "Kyty.ProgramConstants");
	}
	return *buffer;
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	if (!is_write) {
		m_bda_sync.Add(begin, end - begin);
		return;
	}
	m_dma_accessed.ForEachInRange(begin, end - begin, [this](uint64_t first, uint64_t last) {
		m_bda_sync.Add(first, last - first);
	});
}

void BufferCache::MarkDmaAccess(const RangeSet& pages) {
	{
		std::lock_guard lock(m_bda_sync_mutex);
		pages.ForEach([this](uint64_t begin, uint64_t end) {
			m_dma_accessed.Add(begin, end - begin);
			m_bda_sync.Add(begin, end - begin);
		});
	}
	RangeSet                       unbacked;
	std::vector<vk::DeviceAddress> entries;
	pages.ForEach([&](uint64_t begin, uint64_t end) {
		entries.clear();
		for (auto address = begin; address < end; address += CACHING_PAGESIZE) {
			const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
			if (owner == nullptr || !*owner) {
				LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
				unbacked.Add(address, CACHING_PAGESIZE);
				entries.push_back(0);
				continue;
			}
			const auto& buffer = m_slot_buffers[*owner];
			entries.push_back(buffer.BufferDeviceAddress() + buffer.Offset(address) + BDA_ACCESSED);
		}
		if (!unbacked.Contains(begin, end - begin)) {
			const auto table_offset = PageIndex(begin) * sizeof(vk::DeviceAddress);
			m_bda_pagetable_buffer.Back(table_offset, entries.size() * sizeof(vk::DeviceAddress));
			WriteDataBuffer(m_bda_pagetable_buffer, table_offset, entries.data(),
			                entries.size() * sizeof(vk::DeviceAddress));
		}
	});
	unbacked.ForEach(
	    [this](uint64_t start, uint64_t end) { (void)FindBuffer(start, end - start); });
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		TouchBuffer(buffer);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
		    start, finish - start, [&](uint64_t page, uint64_t bytes) {
			    const auto first = std::max(page, start);
			    const auto last  = std::min(page + bytes, finish);
			    m_gpu_modified_ranges.Add(first, last - first);
			    PreflushWrite(first, last - first);
			    ForgetColorMetadataChecks(first, last - first);
			    m_known_written.Add(first, last - first); // these record no bitmap
		    });
	}
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	const auto queued = [this] {
		std::lock_guard lock(m_bda_sync_mutex);
		return std::exchange(m_bda_sync, {});
	}();
	if (m_has_pending_maps.load(std::memory_order_acquire) && !m_sparse_buffers.empty()) {
		BackPendingMaps();
	}
	UploadBatch batch;
	queued.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < finish; ++it) {
				auto&      buffer = m_slot_buffers[it->second];
				const auto first  = std::max(buffer.CpuAddress(), start);
				const auto last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
				if (first < last) {
					// Reads follow every range's protection; a write in between re-queues its page.
					CollectUploads(buffer, first, last - first, false, batch, nullptr);
				}
			}
		});
	});
	vk::Buffer source;
	if (batch.total_size <= m_staging_buffer.Size() / 2) {
		source = UploadCopies(batch.uploads, batch.copies, batch.total_size);
	} else {
		UploadInParts(batch);
	}
	RecordUploads(source, batch);
}

} // namespace Libs::Graphics
