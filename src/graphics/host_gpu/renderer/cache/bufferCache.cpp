#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <fmt/format.h>
#include <unordered_map>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
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
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
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
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	if (const auto& buffer = m_slot_buffers[id]; buffer.IsShared()) {
		// Its GPU-written marks stand for writes into guest memory itself: once those have run,
		// guest memory is current, and no later download from another buffer may overwrite it.
		if (const auto found = m_shared_write_ticks.find(buffer.CpuAddress());
		    found != m_shared_write_ticks.end()) {
			if (!m_scheduler.IsFree(found->second)) {
				m_scheduler.Wait(found->second);
			}
			m_shared_write_ticks.erase(found);
			std::unique_lock lock(m_dirty_ranges_mutex);
			m_gpu_modified_ranges.Subtract(buffer.CpuAddress(), buffer.Size());
		}
		m_shared_ranges.Subtract(buffer.CpuAddress(), buffer.Size());
		for (auto& zone: m_host_zones) {
			if (zone.has_id && zone.id == id) {
				zone.has_id = false;
			}
		}
		std::fprintf(stderr, "[shared-buffer] dropped 0x%llx+%llu KB\n",
		             static_cast<unsigned long long>(buffer.CpuAddress()),
		             static_cast<unsigned long long>(buffer.Size() >> 10u));
	}
	Unregister(id);
	// As TextureCache::DeleteImage: earlier command buffers may still use the buffer on the GPU.
	m_scheduler.DeferRelease([this, id] { ReleaseBuffer(id); });
}

namespace {
// KYTY_SHARED_BUFFERS=1 turns sharing on; KYTY_SHARED_BUFFER_MAX_KB (default 256) bounds the
// buffers it takes and KYTY_SHARED_BUFFER_VOTES (default 16) the game-thread readbacks first.
bool SharedBuffersEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHARED_BUFFERS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
uint64_t SharedBufferMaxBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_SHARED_BUFFER_MAX_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {256}) * 1024u;
	}();
	return bytes;
}
uint32_t SharedBufferVotes() {
	static const uint32_t votes = [] {
		const char* value = std::getenv("KYTY_SHARED_BUFFER_VOTES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 16u;
	}();
	return votes;
}
} // namespace

bool BufferCache::VoteShared(BufferId id) {
	if (!SharedBuffersEnabled() || IsBufferInvalid(id) || id == NULL_BUFFER_ID) {
		return false;
	}
	auto&      old   = m_slot_buffers[id];
	const auto begin = old.CpuAddress();
	const auto size  = old.Size();
	if (old.IsShared() || size > SharedBufferMaxBytes() || begin % 4096u != 0 || size % 4096u != 0) {
		return false;
	}
	if (++m_shared_votes[begin] < SharedBufferVotes() || m_shared_attempts[begin] >= 3) {
		return false;
	}
	m_shared_votes[begin] = 0;
	m_shared_attempts[begin]++;
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(begin, size)) {
			return false; // its bytes are still on their way back; vote again later
		}
	}
	if (FindCopyReadback(begin, size) != 0) {
		return false;
	}
	// Guest memory must hold everything the GPU wrote before the GPU writes there directly. The
	// synchronous download waits for all earlier GPU work, so nothing still writes the old buffer.
	if (HasGpuDirtyBytes(begin, size)) {
		(void)DownloadBufferMemory<false>(old, begin, size);
	}
	if (HasGpuDirtyBytes(begin, size)) {
		return false;
	}
	// A download still in flight would publish older bytes into what is then the GPU's memory.
	WaitDownloadsLanded(begin, size);
	if (m_memory_tracker.IsRegionGpuModified(begin, size)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, size);
	}
	const auto* backing = Libs::LibKernel::Memory::FindGpuCleanBacking(begin, size);
	if (backing == nullptr || reinterpret_cast<uint64_t>(backing) % 4096u != 0) {
		std::fprintf(stderr, "[shared-buffer] 0x%llx+%llu KB: no aligned backing\n",
		             static_cast<unsigned long long>(begin),
		             static_cast<unsigned long long>(size >> 10u));
		return false;
	}
	const auto shared_id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, begin, const_cast<uint8_t*>(backing),
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size, Buffer::ImportHostMemory {});
	if (!m_slot_buffers[shared_id].Handle()) {
		m_slot_buffers.erase(shared_id);
		std::fprintf(stderr, "[shared-buffer] 0x%llx+%llu KB: the driver refused the import\n",
		             static_cast<unsigned long long>(begin),
		             static_cast<unsigned long long>(size >> 10u));
		m_shared_attempts[begin] = 3;
		return false;
	}
	DeleteBuffer(id);
	Register(shared_id);
	m_shared_ranges.Add(begin, size);
	std::fprintf(stderr, "[shared-buffer] shared 0x%llx+%llu KB (attempt %u)\n",
	             static_cast<unsigned long long>(begin), static_cast<unsigned long long>(size >> 10u),
	             m_shared_attempts[begin]);
	return true;
}

namespace {
bool HostZonesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOST_ZONES");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
} // namespace

// KYTY_HOST_ZONE_RANGES=<hex addr>+<hex size>[,...]: the zones (default: Wolverine's per-object
// constants, where ~75 % of the write faults land; Revivedx's list).
BufferCache::HostZone* BufferCache::FindHostZone(uint64_t vaddr, uint64_t size) {
	if (!HostZonesEnabled()) {
		return nullptr;
	}
	if (m_host_zones.empty()) {
		const char* value = std::getenv("KYTY_HOST_ZONE_RANGES");
		if (value == nullptr) {
			value = "10a0000000+800000,11e0000000+400000,11e0400000+400000,11e0800000+400000,"
			        "11e0c00000+400000";
		}
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = CACHING_PAGESIZE;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			if (begin % CACHING_PAGESIZE == 0 && bytes % CACHING_PAGESIZE == 0 && bytes != 0 &&
			    GuestRange {begin, bytes}.Valid()) {
				m_host_zones.push_back({.begin = begin, .size = bytes});
			}
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		std::fprintf(stderr, "[host-zone] %zu zones\n", m_host_zones.size());
		if (m_host_zones.empty()) {
			m_host_zones.push_back({.begin = 0, .size = 0, .failed = true});
		}
	}
	for (auto& zone: m_host_zones) {
		if (!zone.failed && vaddr < zone.begin + zone.size && zone.begin < vaddr + size) {
			return &zone;
		}
	}
	return nullptr;
}

void BufferCache::DropHostZone(HostZone& zone, const char* reason) {
	zone.failed = true;
	if (zone.has_id && !IsBufferInvalid(zone.id)) {
		DeleteBuffer(zone.id);
	}
	zone.has_id = false;
	std::fprintf(stderr, "[host-zone] 0x%llx+0x%llx: normal buffers from now on (%s)\n",
	             static_cast<unsigned long long>(zone.begin),
	             static_cast<unsigned long long>(zone.size), reason);
}

bool BufferCache::CreateHostZoneBuffer(HostZone& zone) {
	if (zone.failed) {
		return false;
	}
	if (zone.has_id && !IsBufferInvalid(zone.id)) {
		return true;
	}
	zone.has_id = false;
	if (HasGpuDirtyBytes(zone.begin, zone.size) ||
	    m_memory_tracker.IsRegionGpuModified(zone.begin, zone.size)) {
		DropHostZone(zone, "GPU-written bytes");
		return false;
	}
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(zone.begin, zone.size)) {
			return false; // try again at the next request
		}
	}
	const auto* backing = Libs::LibKernel::Memory::FindGpuCleanBacking(zone.begin, zone.size);
	if (backing == nullptr || reinterpret_cast<uint64_t>(backing) % 4096u != 0) {
		if (zone.waits++ % 256 == 0) {
			std::fprintf(stderr, "[host-zone] 0x%llx+0x%llx: not in one mapping yet\n",
			             static_cast<unsigned long long>(zone.begin),
			             static_cast<unsigned long long>(zone.size));
		}
		return false;
	}
	// The buffers already in the zone are replaced; one that crosses its edge ends the zone.
	std::vector<BufferId> inside;
	auto                  it = m_buffers.lower_bound(zone.begin);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < zone.begin + zone.size; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.CpuAddress() + buffer.Size() <= zone.begin) {
			continue;
		}
		if (buffer.CpuAddress() < zone.begin ||
		    buffer.CpuAddress() + buffer.Size() > zone.begin + zone.size) {
			DropHostZone(zone, "a cached buffer crosses the zone");
			return false;
		}
		inside.push_back(it->second);
	}
	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, zone.begin, const_cast<uint8_t*>(backing),
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, zone.size, Buffer::ImportHostMemory {});
	if (!m_slot_buffers[id].Handle()) {
		m_slot_buffers.erase(id);
		DropHostZone(zone, "the driver refused the import");
		return false;
	}
	for (const auto old: inside) {
		DeleteBuffer(old);
	}
	SetVulkanObjectNameF(m_graphics.device, m_slot_buffers[id].Handle(),
	                     "Kyty.HostZone[guest=0x{:016x} size=0x{:x}]", zone.begin, zone.size);
	Register(id);
	m_shared_ranges.Add(zone.begin, zone.size);
	zone.id     = id;
	zone.has_id = true;
	// Lift the write protection: the pages stay CPU-dirty and writable, and syncs of a shared
	// buffer upload nothing.
	m_memory_tracker.MarkRegionAsCpuModified(zone.begin, zone.size);
	std::fprintf(stderr, "[host-zone] 0x%llx+0x%llx imported (replaced %zu buffers)\n",
	             static_cast<unsigned long long>(zone.begin),
	             static_cast<unsigned long long>(zone.size), inside.size());
	return true;
}

void BufferCache::UnmapShared(uint64_t vaddr, uint64_t size) {
	if (m_shared_ranges.Empty() || !m_shared_ranges.Intersects(vaddr, size)) {
		return;
	}
	std::vector<BufferId> ids;
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		if (buffer.IsShared() && address < vaddr + size && vaddr < address + buffer.Size()) {
			ids.push_back(id);
		}
	}
	for (const auto id: ids) {
		DeleteBuffer(id);
		// The caller drained the GPU; the import must not outlive the memory it points into.
		m_slot_buffers[id].ReleaseImport();
	}
}

void BufferCache::ReleaseBuffer(BufferId id) {
	// A copy-queue readback may still be reading it (the main queue's ticks do not order it).
	WaitCopyReadbacks();
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
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// One reservation cannot exceed the download ring, so a larger range goes in ring-sized
	// windows; a window that does not fit drains the ring before it is mapped.
	const auto capacity = m_download_buffer.Size();
	bool       any      = false;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		any |= DownloadBufferWindow<async>(buffer, vaddr + offset, std::min(capacity, size - offset));
	}
	return any;
}

template <bool async>
bool BufferCache::DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
			    m_downloading_ranges.Add(start, end - start);
			    m_downloading_list.emplace_back(start, end - start);
		    });
		    NoteDownloading();
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ClearWriteBits(address, bytes);
	    });
	if (copies.empty()) {
		return false;
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
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
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
		NoteDownloading();
	};
	if constexpr (async) {
		// The tick whose completion publishes these bytes (the priority operation runs then).
		const auto publish_tick = m_scheduler.CurrentTick();
		for (auto seen = m_download_publish_tick.load(std::memory_order_relaxed);
		     publish_tick > seen &&
		     !m_download_publish_tick.compare_exchange_weak(seen, publish_tick, std::memory_order_relaxed);) {
		}
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
}

// KYTY_STAGING_BUFFER_MB / KYTY_STREAM_BUFFER_MB: the upload rings' sizes (default 512 / 64). A ring
// that wraps before the GPU has finished with its oldest part waits for it ([ring-wait]).
static uint64_t RingMiB(const char* name, uint64_t fallback) {
	const char* value = std::getenv(name);
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : fallback;
	return (mib >= 16 && mib <= 4096 ? mib : fallback) << 20u;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, RingMiB("KYTY_STAGING_BUFFER_MB", 512)),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, RingMiB("KYTY_STREAM_BUFFER_MB", 64)),
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
	// Unset: the larger of 6 GiB and 60 % of the budget (as KYTY_TEXTURE_CACHE_MB).
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10)
	                                     : std::max<uint64_t>(6144ull, static_cast<uint64_t>(budget) / 10 * 6 >> 20u);
	if (mib != 0) {
		const auto cap       = mib << 20u;
		m_trigger_gc_memory  = std::min<uint64_t>(m_trigger_gc_memory, cap);
		m_critical_gc_memory = std::min<uint64_t>(m_critical_gc_memory, cap + cap / 4);
		LOGF("BufferCache: thresholds capped at %" PRIu64 " MB (critical %" PRIu64 " MB)\n",
		     m_trigger_gc_memory >> 20u, m_critical_gc_memory >> 20u);
	}
}

BufferCache::~BufferCache() {
	WaitCopyReadbacks();
	for (auto& slot: m_copy_slots) {
		slot.staging.reset();
		if (slot.fence != nullptr) {
			m_graphics.device.destroyFence(slot.fence, nullptr);
		}
		if (slot.pool != nullptr) {
			m_graphics.device.destroyCommandPool(slot.pool, nullptr);
		}
		slot.fence = nullptr;
		slot.pool  = nullptr;
	}
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
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size, [this, vaddr, size] {
		KYTY_PROFILER_BLOCK("ReadMemory: CPU write over GPU-written bytes");
		// KYTY_FPS_LOG: on the GPU thread, which pages its own writes flush most (every 10 s).
		if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log && GuestGpu::IsGpuThread()) {
			static std::unordered_map<uint64_t, uint64_t> pages;
			static auto                                  last = std::chrono::steady_clock::now();
			pages[vaddr & ~uint64_t {0xfff}]++;
			if (const auto now = std::chrono::steady_clock::now(); now - last > std::chrono::seconds(10)) {
				last = now;
				std::vector<std::pair<uint64_t, uint64_t>> list(pages.begin(), pages.end());
				std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				std::string text;
				for (size_t i = 0; i < list.size() && i < 8; i++) {
					text += fmt::format(" 0x{:x}x{}", list[i].first, list[i].second);
				}
				std::fprintf(stderr, "[gpu-thread-write-flush] pages=%zu top:%s\n", list.size(), text.c_str());
				pages.clear();
			}
		}
		ReadMemory(vaddr, size, true);
	});
}

bool BufferCache::WriteClean(uint64_t vaddr, const void* data, uint64_t size) {
	if (!GuestGpu::IsGpuThread() || size == 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	{
		// A download publishes the GPU's older value of these bytes when it lands.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}

	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, data, size);
	if (HasGpuDirtyBytes(vaddr, size)) {
		// Overwritten in full: guest memory holds the value the GPU copy will have.
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
		ClearWriteBits(vaddr, size);
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
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

static thread_local int t_rm_reason = 0;
static std::atomic<uint64_t> g_rm_iters[6] {};
void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const bool from_gpu_thread = GuestGpu::IsGpuThread();
	const bool async           = g_async_readback && !from_gpu_thread;
	// Timeline: the whole read, whatever path it takes.
	struct TimelineEnd {
		uint64_t vaddr;
		~TimelineEnd() { Timeline::Mark("readmem-end", vaddr); }
	} timeline_end {vaddr};
	// KYTY_READBACK_STATS=1: time spent in readbacks, game threads and GPU thread, every 10 s.
	struct ReadbackTimer {
		bool                                  enabled;
		int                                   kind; // 0 game read, 1 game write, 2 GPU thread
		uint64_t                              vaddr;
		std::chrono::steady_clock::time_point start;
		~ReadbackTimer() {
			if (!enabled) return;
			static std::mutex mutex;
			static uint64_t   us[3] {}, n[3] {};
			// Pages faulting again within the window: same 4 KB page seen before.
			static std::unordered_map<uint64_t, uint32_t> pages[2];
			static auto       last = std::chrono::steady_clock::now();
			const auto        now  = std::chrono::steady_clock::now();
			std::lock_guard   lock(mutex);
			us[kind] += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
			n[kind]++;
			if (kind < 2) pages[kind][vaddr >> 12u]++;
			if (now - last > std::chrono::seconds(10)) {
				std::fprintf(stderr,
				             "[readback] game reads %llu ms n=%llu pages=%zu, game writes %llu ms "
				             "n=%llu pages=%zu, gpu thread %llu ms n=%llu (%.1f s)\n",
				             static_cast<unsigned long long>(us[0] / 1000), static_cast<unsigned long long>(n[0]),
				             pages[0].size(),
				             static_cast<unsigned long long>(us[1] / 1000), static_cast<unsigned long long>(n[1]),
				             pages[1].size(),
				             static_cast<unsigned long long>(us[2] / 1000), static_cast<unsigned long long>(n[2]),
				             std::chrono::duration<double>(now - last).count());
				for (int k = 0; k < 2; k++) {
					std::vector<std::pair<uint64_t, uint32_t>> list(pages[k].begin(), pages[k].end());
					std::sort(list.begin(), list.end(),
					          [](const auto& a, const auto& b) { return a.second > b.second; });
					std::string text;
					for (size_t i = 0; i < list.size() && i < 12; i++) {
						text += fmt::format(" {:x}:{}", list[i].first << 12u, list[i].second);
					}
					std::fprintf(stderr, "[readback]   top %s pages:%s\n", k == 0 ? "read" : "write",
					             text.c_str());
					pages[k].clear();
				}
				us[0] = us[1] = us[2] = n[0] = n[1] = n[2] = 0;
				last = now;
			}
		}
	};
	static const bool readback_stats = std::getenv("KYTY_READBACK_STATS") != nullptr;
	ReadbackTimer timer {readback_stats, from_gpu_thread ? 2 : (is_write ? 1 : 0), vaddr,
	                     readback_stats ? std::chrono::steady_clock::now()
	                                    : std::chrono::steady_clock::time_point {}};
	uint64_t rm_iters = 0;
	// KYTY_READBACK_STATS: the steps of slow readbacks (where a game thread waits).
	struct SlowTrace {
		bool        enabled;
		uint64_t    vaddr;
		bool        is_write;
		std::chrono::steady_clock::time_point start;
		std::string steps;
		void Add(const char* what, double ms, int reason, uint64_t tick, uint64_t cur) {
			if (!enabled || steps.size() > 600) return;
			steps += fmt::format(" [{} {:.2f}ms r{} t{} c{}]", what, ms, reason, tick, cur);
		}
		~SlowTrace() {
			if (!enabled) return;
			const double total = std::chrono::duration<double, std::milli>(
			                         std::chrono::steady_clock::now() - start).count();
			static std::atomic<uint32_t> logged {0};
			if (total >= 8.0 && logged.fetch_add(1) < 300) {
				std::fprintf(stderr, "[readback-slow] %s 0x%llx %.1fms:%s\n", is_write ? "W" : "R",
				             static_cast<unsigned long long>(vaddr), total, steps.c_str());
			}
		}
	} slow {readback_stats && !from_gpu_thread, vaddr, is_write, std::chrono::steady_clock::now(), {}};
	for (;;) {
		uint64_t wait_tick = 0;
		int      rm_reason = 0;
		rm_iters++;
		const auto step_begin = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point exec_begin;
		m_scheduler.Context().GetGpu().SendCommandSync(
		    [this, vaddr, size, is_write, from_gpu_thread, async, &wait_tick, &rm_reason,
		     &exec_begin] {
			    exec_begin  = std::chrono::steady_clock::now();
			    t_rm_reason = 0;
			    wait_tick   = ReadMemoryStep(vaddr, size, is_write, from_gpu_thread, async);
			    rm_reason   = t_rm_reason;
		    });
		slow.Add("queue",
		         std::chrono::duration<double, std::milli>(exec_begin - step_begin).count(), 0, 0,
		         0);
		g_rm_iters[rm_reason].fetch_add(1, std::memory_order_relaxed);
		slow.Add("step", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_begin).count(),
		         rm_reason, wait_tick & ~CopyReadbackMarker, m_scheduler.CurrentTick());
		if (readback_stats && (rm_iters & (rm_iters - 1)) == 0 && rm_iters >= 1024) {
			std::fprintf(stderr,
			             "[readback-spin] vaddr=0x%llx iters=%llu reason=%d tick=0x%llx free=%d cur=%llu "
			             "last_async=%llu | iters by reason: bitmap=%llu copyfind=%llu lastasync=%llu "
			             "copyq=%llu dl=%llu done=%llu\n",
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(rm_iters),
			             rm_reason, static_cast<unsigned long long>(wait_tick),
			             ((wait_tick & CopyReadbackMarker) == 0 && wait_tick != 0) ? (m_scheduler.IsFree(wait_tick) ? 1 : 0) : -1,
			             static_cast<unsigned long long>(m_scheduler.CurrentTick()),
			             static_cast<unsigned long long>(m_last_async_download_tick),
			             static_cast<unsigned long long>(g_rm_iters[1].load()),
			             static_cast<unsigned long long>(g_rm_iters[2].load()),
			             static_cast<unsigned long long>(g_rm_iters[3].load()),
			             static_cast<unsigned long long>(g_rm_iters[4].load()),
			             static_cast<unsigned long long>(g_rm_iters[5].load()),
			             static_cast<unsigned long long>(g_rm_iters[0].load()));
		}
		if (wait_tick == 0) {
			return;
		}
		const auto wait_begin = std::chrono::steady_clock::now();
		if ((wait_tick & CopyReadbackMarker) != 0) {
			CompleteCopyReadback(wait_tick);
			slow.Add("copywait", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_begin).count(),
			         rm_reason, 0, 0);
			continue;
		}
		// WaitPriorityOperations only waits for publications queued at or before the tick; a
		// tick with none (a pending write-bitmap writer) returned at once and this loop went
		// through the GPU thread again, ~500k times a second on the title screen. Wait for the
		// host GPU to reach the tick first (KYTY_READBACK_SPIN_FIX=0: as before).
		static const bool spin_fix = [] {
			const char* value = std::getenv("KYTY_READBACK_SPIN_FIX");
			return value == nullptr || value[0] != '0';
		}();
		if (spin_fix && wait_tick < m_scheduler.CurrentTick() && !m_scheduler.IsFree(wait_tick)) {
			m_scheduler.GetMasterSemaphore().Wait(wait_tick);
		}
		m_scheduler.WaitPriorityOperations(wait_tick);
		slow.Add("wait", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_begin).count(),
		         rm_reason, wait_tick, m_scheduler.CurrentTick());
	}
}

uint32_t BufferCache::PeekDword(uint64_t vaddr) {
	KYTY_PROFILER_FUNCTION();
	static constexpr uint32_t PeekSlots = 65536;
	if (!GuestGpu::IsGpuThread() || (vaddr & 3u) != 0u || !GuestRange {vaddr, 4}.Valid()) {
		return 0;
	}
	if (m_peek_buffer == nullptr) {
		m_peek_buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst,
		                                         uint64_t {PeekSlots} * sizeof(uint32_t));
	}
	auto found = m_peek_slots.find(vaddr);
	if (found == m_peek_slots.end()) {
		if (m_peek_slots.size() >= PeekSlots) {
			return 0;
		}
		found = m_peek_slots.emplace(vaddr, PeekSlot {.index = static_cast<uint32_t>(m_peek_slots.size())})
		            .first;
	}
	auto& slot = found->second;
	if (slot.tick != 0) {
		if (!m_scheduler.IsFree(slot.tick)) {
			return slot.value; // the last copy is still in flight
		}
		m_peek_buffer->Invalidate(uint64_t {slot.index} * sizeof(uint32_t), sizeof(uint32_t));
		std::memcpy(&slot.value, m_peek_buffer->Mapped().data() + uint64_t {slot.index} * sizeof(uint32_t),
		            sizeof(uint32_t));
		slot.tick = 0;
	}
	if (!HasGpuDirtyBytes(vaddr, sizeof(uint32_t))) {
		uint32_t value = 0;
		if (Libs::LibKernel::Memory::TryReadBacking(vaddr, &value, sizeof(value))) {
			slot.value = value;
		}
		return slot.value;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, sizeof(uint32_t))) {
		return slot.value;
	}
	const auto& source  = m_slot_buffers[*owner];
	if (source.IsShared() || !source.Handle()) {
		// Its bytes live in guest memory: the GPU's writes land there.
		uint32_t value = 0;
		if (Libs::LibKernel::Memory::TryReadBacking(vaddr, &value, sizeof(value))) {
			slot.value = value;
		}
		return slot.value;
	}
	auto&       command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	const vk::BufferCopy copy {source.Offset(vaddr), uint64_t {slot.index} * sizeof(uint32_t),
	                           sizeof(uint32_t)};
	native.copyBuffer(source.Handle(), m_peek_buffer->Handle(), 1, &copy);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
	slot.tick = m_scheduler.CurrentTick();
	return slot.value;
}

// KYTY_SRT_NARROW_READBACK=1: an SRT defer readback downloads only the pages the refresh reads,
// not the 4 MiB window game-thread reads take (so the window's other bytes do not count as
// downloading behind the whole GPU backlog). Off: no measurable change in the jungle (2026-10-18).
static bool SrtNarrowReadback() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_NARROW_READBACK");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

uint64_t BufferCache::StartSrtReadback(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() || size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	const auto flushed = [this](uint64_t tick) {
		if (tick >= m_scheduler.CurrentTick()) {
			(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
		}
		return tick;
	};
	if (!HasGpuDirtyBytes(vaddr, size)) {
		bool downloading = false;
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			downloading = m_downloading_ranges.Intersects(vaddr, size);
		}
		// A download in flight publishes these bytes when its tick completes. A copy-queue
		// readback is left to the ordinary (waiting) path: its caller publishes it.
		if (downloading && FindCopyReadback(vaddr, size) == 0 && m_last_async_download_tick != 0 &&
		    !m_scheduler.IsFree(m_last_async_download_tick)) {
			return flushed(m_last_async_download_tick);
		}
		return 0;
	}
	const auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (buffer.IsShared()) {
		// Its bytes live in guest memory: they are current once the writes recorded so far land.
		if (const auto found = m_shared_write_ticks.find(buffer.CpuAddress());
		    found != m_shared_write_ticks.end() && !m_scheduler.IsFree(found->second)) {
			return flushed(found->second);
		}
		return 0;
	}
	m_srt_readback  = true;
	const auto tick = ReadMemoryStep(vaddr, size, false, true, true);
	m_srt_readback  = false;
	if (tick == 0) {
		return 0;
	}
	if ((tick & CopyReadbackMarker) != 0) {
		// The writers had finished: the copy queue's readback waits for nothing else.
		CompleteCopyReadback(tick);
		return 0;
	}
	return flushed(tick);
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
	return window;
}

static bool ReadbackRetick() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_RETICK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t BufferCache::ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
                                     bool from_gpu_thread, bool async) {
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return 0;
	}
	// MEASUREMENT ONLY (wrong results): KYTY_TEST_NO_READBACK_WAIT=1 lets game-thread reads of
	// GPU-written pages see guest memory as it is, without any readback, to bound what the
	// readback waits cost. The bytes stay GPU-written for the GPU thread.
	if (static const bool no_wait = [] {
		    const char* value = std::getenv("KYTY_TEST_NO_READBACK_WAIT");
		    return value != nullptr && value[0] == '1';
	    }();
	    no_wait && !is_write && !from_gpu_thread) {
		const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
		return 0;
	}
	if (!from_gpu_thread && SharedBuffersEnabled()) {
		if (const auto id = FindBuffer(vaddr, size); VoteShared(id)) {
			return 0;
		}
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (buffer.IsShared()) {
		// Its bytes live in guest memory; nothing to download. The GPU thread's own reads wait
		// for the writes recorded before them (see MarkGpuWritten).
		const auto begin = buffer.CpuAddress();
		const auto bytes = buffer.Size();
		if (const auto found = m_shared_write_ticks.find(begin); found != m_shared_write_ticks.end()) {
			const auto tick = found->second;
			if (from_gpu_thread && !m_scheduler.IsFree(tick)) {
				m_scheduler.Wait(tick);
			}
			if (from_gpu_thread || m_scheduler.IsFree(tick)) {
				m_shared_write_ticks.erase(begin);
				std::unique_lock lock(m_dirty_ranges_mutex);
				m_gpu_modified_ranges.Subtract(begin, bytes);
			}
		}
		if (m_memory_tracker.IsRegionGpuModified(begin, bytes)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
		}
		return 0;
	}

	// Shared guest memory groundwork (KYTY_SHARED_MEMORY_PROBE=1): whether the driver can import
	// this guest memory as a GPU buffer (VK_EXT_external_memory_host), and which cache buffers
	// the game threads read back most, i.e. the candidates for sharing.
	if (static const bool probe = std::getenv("KYTY_SHARED_MEMORY_PROBE") != nullptr; probe) {
		auto& graphics = m_scheduler.Context().GetGraphics();
		static bool probed = false;
		if (!probed) {
			probed = true;
			if (!graphics.external_memory_host_enabled ||
			    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT == nullptr) {
				std::fprintf(stderr, "[shared-memory] VK_EXT_external_memory_host not available\n");
			} else {
				vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host_props {};
				vk::PhysicalDeviceProperties2 props2 {};
				props2.pNext = &host_props;
				graphics.physical_device.getProperties2(&props2);
				const auto align = host_props.minImportedHostPointerAlignment;
				const auto probe_at = [&](const char* what, void* pointer) {
					VkMemoryHostPointerPropertiesEXT out {};
					out.sType      = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
					const auto res = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT(
					    graphics.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
					    pointer, &out);
					std::fprintf(stderr,
					             "[shared-memory] probe %s ptr=%p align=0x%llx result=%d types=0x%x\n",
					             what, pointer, static_cast<unsigned long long>(align),
					             static_cast<int>(res), out.memoryTypeBits);
				};
				probe_at("guest", reinterpret_cast<void*>(Common::AlignDown(vaddr, align)));
				void* plain = std::malloc(static_cast<size_t>(align) * 4u);
				probe_at("malloc", reinterpret_cast<void*>(
				                       Common::AlignUp(reinterpret_cast<uint64_t>(plain), align)));
				std::free(plain);
			}
		}
		if (!from_gpu_thread) {
			struct Hot {
				uint64_t count = 0;
				uint64_t size  = 0;
			};
			static std::unordered_map<uint64_t, Hot> hot;
			static uint64_t                          total = 0;
			auto& h = hot[buffer.CpuAddress()];
			h.count++;
			h.size = buffer.Size();
			// Where in guest memory, by 1 MiB block: what a shared range would have to cover.
			static std::unordered_map<uint64_t, uint64_t> blocks;
			blocks[vaddr >> 20u]++;
			if (total % 8192 == 8191) {
				std::vector<std::pair<uint64_t, uint64_t>> list(blocks.begin(), blocks.end());
				std::sort(list.begin(), list.end(),
				          [](const auto& a, const auto& b) { return a.second > b.second; });
				std::string text;
				for (size_t i = 0; i < list.size() && i < 16; i++) {
					text += fmt::format(" {:x}:{}", list[i].first << 20u, list[i].second);
				}
				std::fprintf(stderr, "[shared-memory] hot 1MiB blocks=%zu:%s\n", list.size(), text.c_str());
			}
			if (++total % 2048 == 0) {
				std::vector<std::pair<uint64_t, Hot>> list(hot.begin(), hot.end());
				std::sort(list.begin(), list.end(),
				          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
				std::fprintf(stderr, "[shared-memory] game-thread readbacks=%llu buffers=%zu, top:\n",
				             static_cast<unsigned long long>(total), list.size());
				for (size_t i = 0; i < list.size() && i < 10; i++) {
					std::fprintf(stderr, "[shared-memory]   buffer 0x%llx size=%llu KB faults=%llu\n",
					             static_cast<unsigned long long>(list[i].first),
					             static_cast<unsigned long long>(list[i].second.size / 1024),
					             static_cast<unsigned long long>(list[i].second.count));
				}
			}
		}
	}

	// Diagnostics: whether the bytes the faulting access touches were written by the GPU
	// (true sharing) or only share the page with bytes that were (false sharing).
	{
		static uint64_t exact = 0;
		static uint64_t shared = 0;
		const bool      dirty  = HasGpuDirtyBytes(vaddr, std::max<uint64_t>(size, 4));
		(dirty ? exact : shared)++;
		static uint32_t logged = 0;
		static uint32_t shared_seen = 0;
		static const bool shared_stats = std::getenv("KYTY_READBACK_STATS") != nullptr;
		if (!dirty && shared_stats && !from_gpu_thread && (shared_seen++ % 256) == 0 &&
		    shared_seen < 256u * 120u) {
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
			std::fprintf(stderr, "[readback] shared %s 0x%llx+%llu: page GPU-written bytes%s\n",
			             is_write ? "write" : "read", static_cast<unsigned long long>(vaddr),
			             static_cast<unsigned long long>(size), dirty_ranges.c_str());
		}
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
			if (static const bool stats = std::getenv("KYTY_READBACK_STATS") != nullptr; stats) {
				std::fprintf(stderr, "[readback] faults: bytes GPU-written %llu, page shared %llu\n",
				             static_cast<unsigned long long>(exact),
				             static_cast<unsigned long long>(shared));
			}
		}
		// KYTY_READBACK_STATS: the bytes of the page that count as GPU-written, for game reads of
		// GPU-written bytes, the first 48.
		if (static const bool stats = std::getenv("KYTY_READBACK_STATS") != nullptr;
		    stats && dirty && !from_gpu_thread) {
			static std::atomic<uint32_t> seen_reads {0};
			static std::atomic<uint32_t> shown {0};
			if (seen_reads.fetch_add(1) % 256 == 0 && shown.fetch_add(1) < 120) {
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
				std::fprintf(stderr, "[readback] game %s 0x%llx+%llu: page GPU-written bytes%s\n",
				             is_write ? "write" : "read", static_cast<unsigned long long>(vaddr),
				             static_cast<unsigned long long>(size), dirty_ranges.c_str());
			}
		}
	}

	// The page is protected as GPU-written, but none of its bytes are waiting for a download:
	// the bytes the GPU wrote are elsewhere on the page, or were downloaded with another
	// page. The page is current, so lift its protection. Downloading the window instead drained
	// the GPU (~30 ms a fault) for guest reads of structs next to the command processor's
	// labels.
	// GPU write bitmap: pages of tracked bindings the GPU did not write lose their GPU-written
	// bytes here, so the window below downloads only what was written, or nothing.
	m_narrow_window = m_srt_readback && SrtNarrowReadback();
	if (uint64_t wait_tick = 0; RefineWithWriteBitmap(buffer, vaddr, size, async, wait_tick)) {
		{ t_rm_reason = 1; return wait_tick; }
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		// An asynchronous readback of these bytes may not be in guest memory yet: the access
		// must not see the older bytes, so it waits for that publication first.
		bool downloading = false;
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			downloading = m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
		}
		if (downloading) {
			// A copy-queue readback of these bytes: its caller publishes it; wait for that.
			if (const auto marker = FindCopyReadback(page_begin, page_end - page_begin);
			    marker != 0) {
				if (async) {
					{ t_rm_reason = 2; return marker; }
				}
				CompleteCopyReadback(marker);
				std::shared_lock lock(m_dirty_ranges_mutex);
				downloading = m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
			}
		}
		if (downloading && m_last_async_download_tick != 0) {
			if (async) {
				{ t_rm_reason = 3; return m_last_async_download_tick; }
			}
			m_scheduler.WaitPriorityOperations(m_last_async_download_tick);
		}
		if (async && ReadbackWindow() != 0 && !m_narrow_window) {
			// The whole download window, as the synchronous path lifts it: page by page, each
			// page of the window faulted on its own and went through the GPU thread again.
			const auto window = ReadbackWindow();
			const auto begin  = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
			const auto end    = std::min(begin + window, buffer.CpuAddress() + buffer.Size());
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
			// One lift per run of neighbouring pages: each lift is a protection change, and
			// page by page they were most of the GPU thread's VirtualProtect calls.
			for (size_t first = 0; first < current.size();) {
				size_t last = first + 1;
				while (last < current.size() &&
				       current[last] == current[last - 1] + TRACKER_PAGE_SIZE) {
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
	if (const auto window = ReadbackWindow(); window != 0 && !m_narrow_window) {
		const auto buffer_begin = buffer.CpuAddress();
		const auto buffer_end   = buffer_begin + buffer.Size();
		window_begin = std::max(Common::AlignDown(vaddr, window), buffer_begin);
		window_end   = std::min(std::max(window_begin + window, vaddr + size), buffer_end);
	} else if (m_narrow_window) {
		// Writers still running elsewhere in the window: only the faulting pages, whose writers
		// have finished, download (the rest of the window keeps its protection).
		window_begin = std::max(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), buffer.CpuAddress());
		window_end   = std::min(Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE),
		                        buffer.CpuAddress() + buffer.Size());
	}

	Timeline::Mark("readmem", vaddr, from_gpu_thread ? 1u : 0u);
	// KYTY_READ_SNAPSHOT=1: a game-thread read of bytes whose GPU writers are still running does
	// not wait for them. On the console the CPU reads memory as it is: bytes the running work has
	// not written yet read as before, and a game polling for a GPU result keeps polling. Here the
	// GPU's current bytes of the window are copied on the second queue (behind finished work
	// only) into guest memory, and CPU reads of the window go through until the work recorded so
	// far has finished; then the pages fault again, and a read downloads the final bytes (cheap:
	// the writers are done). Writes still fault and wait, and the bytes stay GPU-written for the
	// GPU thread's own reads, so nothing the GPU writes is lost or overwritten.
	// KYTY_READ_SNAPSHOT=2: every game-thread read snapshots (copy behind finished work), also
	// when no writer is known to run: the writer ticks are kept per 1 MiB block and per heap-wide
	// binding, so a writer always seemed to run, and a read that did not snapshot downloaded on the
	// main queue behind all the work recorded so far (~7 ms a read for the BVH headers the game
	// reads each frame).
	static const int snapshot_mode = [] {
		const char* value = std::getenv("KYTY_READ_SNAPSHOT");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	if (const bool snapshots = snapshot_mode != 0; snapshots && async && !is_write && !from_gpu_thread) {
		constexpr uint64_t SnapshotWindow = 64u * 1024u;
		const auto snap_begin = std::max(Common::AlignDown(vaddr, SnapshotWindow), buffer.CpuAddress());
		const auto snap_end   = std::min(std::max(snap_begin + SnapshotWindow,
		                                          Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE)),
		                                 buffer.CpuAddress() + buffer.Size());
		bool ready = false;
		{
			std::lock_guard lock(m_snapshot_mutex);
			for (auto it = m_snapshots_ready.begin(); it != m_snapshots_ready.end(); ++it) {
				if (it->first <= page_begin && page_end <= it->second) {
					ready = true;
					const auto [b, e] = *it;
					m_snapshots_ready.erase(it);
					m_memory_tracker.LiftReads(b, e - b);
					m_scheduler.DeferOperation([this, b, e] { m_memory_tracker.RestoreReads(b, e - b); });
					break;
				}
			}
		}
		if (ready) {
			{ t_rm_reason = 6; return 0; }
		}
		const auto latest = LatestGpuWriteTick(snap_begin, snap_end);
		if (snapshot_mode == 2 || (latest != 0 && !m_scheduler.IsFree(latest))) {
			if (const auto marker = TrySnapshotReadback(buffer, snap_begin, snap_end); marker != 0) {
				{ t_rm_reason = 7; return marker; }
			}
		}
	}
	if (async) {
		NoteHotReadback(vaddr);
		// The writers of the window have finished: copy it on the second queue, behind nothing,
		// and let the caller wait for that copy alone.
		if (const auto marker = TryCopyQueueReadback(buffer, window_begin, window_end);
		    marker != 0) {
			{ t_rm_reason = 4; return marker; }
		}
		// The download publishes when its tick completes; the page stays protected until the
		// caller's next step finds its bytes downloaded and lifts it.
		auto tick = m_scheduler.CurrentTick();
		if (DownloadBufferMemory<true>(buffer, window_begin, window_end - window_begin)) {
			// The tick of the last copy, not of the first: a download ring that wraps within this
			// recording submits it (StreamBuffer::WaitPendingOperations), and the later copies
			// and their publication then land in the next tick. A caller waiting for the earlier
			// tick went on before its bytes were published, and the GPU thread then lifted the
			// page's protection over the older bytes (Jetsku bf8d6148, same idea for writers).
			// KYTY_READBACK_RETICK=0 keeps the tick from before the download.
			if (ReadbackRetick()) {
				tick = m_scheduler.CurrentTick();
			}
			(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
			m_last_async_download_tick = tick;
			{ t_rm_reason = 5; return tick; }
		}
	} else if (DownloadBufferMemory<false>(buffer, window_begin, window_end - window_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	return 0;
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
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	m_gc_stats.created++;
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
	if (auto* zone = FindHostZone(vaddr, size); zone != nullptr) {
		if (vaddr >= zone->begin && end <= zone->begin + zone->size) {
			if (CreateHostZoneBuffer(*zone)) {
				return zone->id;
			}
		} else {
			DropHostZone(*zone, "a buffer crosses the zone");
		}
	}
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
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
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
	double   walk_written_s = 0.0;
};
SyncStats g_sync_stats;

void RecordSync(uint64_t size, bool is_written, bool uploaded, double walk_s) {
	auto& st = g_sync_stats;
	st.calls++;
	st.uploads += uploaded ? 1 : 0;
	st.bytes += size;
	st.written += is_written ? 1 : 0;
	st.walk_s += walk_s;
	if (is_written) st.walk_written_s += walk_s;
	const int bucket = size <= 0x1000 ? 0 : size <= 0x10000 ? 1 : size <= 0x100000 ? 2
	                 : size <= 0x1000000 ? 3 : size <= 0x10000000 ? 4 : 5;
	st.buckets[bucket]++;
	if (is_written) {
		st.written_buckets[bucket]++;
	}
	if (st.calls % 200000 == 0) {
		std::fprintf(stderr, "[sync-stats] calls=%" PRIu64 " uploads=%" PRIu64 " written=%" PRIu64
		     " avg_bytes=%" PRIu64 " walk_total=%.1fms (written %.1fms) sizes(<=4K,64K,1M,16M,256M,>)=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 " written_sizes=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
		     st.calls, st.uploads, st.written, st.bytes / st.calls, st.walk_s * 1000.0,
		     st.walk_written_s * 1000.0,
		     st.buckets[0], st.buckets[1], st.buckets[2], st.buckets[3], st.buckets[4],
		     st.buckets[5], st.written_buckets[0], st.written_buckets[1], st.written_buckets[2],
		     st.written_buckets[3], st.written_buckets[4], st.written_buckets[5]);
		st = {};
	}
}
} // namespace

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	KYTY_PROFILER_FUNCTION();
	if (buffer.IsShared()) {
		// The buffer is guest memory itself: no upload, and GPU writes need no tracking.
		return false;
	}
	// KYTY_SYNC_SKIP=1: the same sync (buffer, range, written) as one done while no page anywhere
	// became CPU-dirty or stopped being GPU-dirty finds nothing to upload and marks nothing new:
	// skip the tracker walk. The title syncs ~500,000 buffer ranges a second, most of them
	// repeats within a submission (the per-record dispatch chains bind the same buffers).
	static const bool skip_repeats = [] {
		const char* value = std::getenv("KYTY_SYNC_SKIP");
		return value != nullptr && value[0] == '1';
	}();
	struct SyncMemo {
		const Buffer* buffer     = nullptr;
		uint64_t      vaddr      = 0;
		uint64_t      size       = 0;
		uint64_t      cpu_epoch  = 0;
		uint64_t      gpu_epoch  = 0;
		bool          written    = false;
	};
	static std::array<SyncMemo, 4096> memo {};
	static uint64_t                   memo_hits = 0, memo_calls = 0;
	SyncMemo*                         slot      = nullptr;
	if (skip_repeats && !is_texel_buffer) {
		const auto index = static_cast<size_t>(
		    ((vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull) ^ (is_written ? 0x5bd1e995u : 0u)) %
		    memo.size());
		slot = &memo[index];
		memo_calls++;
		if (slot->buffer == &buffer && slot->vaddr == vaddr && slot->size == size &&
		    slot->written == is_written &&
		    slot->cpu_epoch == g_cpu_dirty_epoch.load(std::memory_order_acquire) &&
		    slot->gpu_epoch == g_gpu_clean_epoch.load(std::memory_order_acquire)) {
			memo_hits++;
			return false;
		}
		if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
		    log && (memo_calls & ((1u << 20u) - 1u)) == 0) {
			std::fprintf(stderr, "[sync-skip] %llu of %llu syncs skipped\n",
			             static_cast<unsigned long long>(memo_hits),
			             static_cast<unsigned long long>(memo_calls));
		}
	}
	// The epochs as they were before the walk: a page that changes state during it (another
	// thread's write fault) moves them, and the next sync walks again.
	struct MemoStore {
		SyncMemo*     slot;
		const Buffer* buffer;
		uint64_t      vaddr, size;
		bool          written;
		uint64_t      cpu_epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
		uint64_t      gpu_epoch = g_gpu_clean_epoch.load(std::memory_order_acquire);
		~MemoStore() {
			if (slot != nullptr) {
				*slot = {buffer, vaddr, size, cpu_epoch, gpu_epoch, written};
			}
		}
	} memo_store {slot, &buffer, vaddr, size, is_written};
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
	if (source) {
		KYTY_PROFILER_BLOCK("Sync::Copy");
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Recorder();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		KYTY_PROFILER_BLOCK("Sync::TexelImage");
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
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

// KYTY_UPLOAD_WORKER=1 (after Revivedx c3bf5829): the guest-memory reads of staging uploads run
// on a worker thread in order. The GPU thread only queues them; a submission waits for every read
// queued before it, on the thread that calls vkQueueSubmit (the recording thread, so the GPU
// thread does not wait). As on the console, the GPU sees the bytes as they are a little later
// than the upload decision; a CPU write after the decision dirties its page again and is
// uploaded again, as before.
namespace {
class UploadWorker {
public:
	static bool Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_UPLOAD_WORKER");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}
	static UploadWorker& Get() {
		static UploadWorker worker;
		return worker;
	}
	// GPU thread only. False when the queue is full (the caller reads itself).
	bool Push(uint8_t* destination, uint64_t address, uint64_t size) {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (head - m_tail.load(std::memory_order_acquire) >= Capacity) {
			return false;
		}
		m_jobs[head % Capacity] = {destination, address, size};
		m_head.store(head + 1, std::memory_order_release);
		if (m_sleeping.load(std::memory_order_acquire)) {
			m_head.notify_one();
		}
		return true;
	}
	[[nodiscard]] uint64_t Pushed() const { return m_head.load(std::memory_order_acquire); }
	// Any thread: the jobs before sequence number `upto` are done.
	void WaitFor(uint64_t upto) const {
		while (m_tail.load(std::memory_order_acquire) < upto) {
			_mm_pause();
		}
	}

private:
	struct Job {
		uint8_t* destination = nullptr;
		uint64_t address     = 0;
		uint64_t size        = 0;
	};
	static constexpr uint64_t Capacity = 8192;

	UploadWorker() {
		std::thread([this] {
			Profiler::SetThreadName("UploadWorker");
			uint64_t tail = 0;
			for (;;) {
				uint64_t head = m_head.load(std::memory_order_acquire);
				for (uint32_t spin = 0; head == tail && spin < 20000u; spin++) {
					_mm_pause();
					head = m_head.load(std::memory_order_acquire);
				}
				if (head == tail) {
					m_sleeping.store(true, std::memory_order_seq_cst);
					head = m_head.load(std::memory_order_seq_cst);
					if (head == tail) {
						m_head.wait(tail, std::memory_order_acquire);
					}
					m_sleeping.store(false, std::memory_order_relaxed);
					continue;
				}
				while (tail != head) {
					const auto job = m_jobs[tail % Capacity];
					ReadGuestForUpload(job.destination, job.address, job.size);
					tail++;
					m_tail.store(tail, std::memory_order_release);
				}
			}
		}).detach();
	}

	std::array<Job, Capacity> m_jobs {};
	alignas(64) std::atomic<uint64_t> m_head {0};
	alignas(64) std::atomic<uint64_t> m_tail {0};
	alignas(64) std::atomic<bool> m_sleeping {false};
};
} // namespace

uint64_t UploadWorkerPushed() {
	return UploadWorker::Enabled() ? UploadWorker::Get().Pushed() : 0;
}

void WaitUploadWorker(uint64_t upto) {
	if (upto != 0) {
		UploadWorker::Get().WaitFor(upto);
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Profiler::Add(Profiler::Counter::UploadBytes, static_cast<int64_t>(total_size));

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
		if (deferred && m_staging_buffer.IsCoherent() && m_scheduler.Current().Recorder().Threaded()) {
			m_upload_reads.clear();
			for (auto& copy: copies) {
				m_upload_reads.push_back(
				    {mapped + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size});
				copy.srcOffset += base_offset;
			}
			const auto* reads = m_scheduler.Stash(m_upload_reads.data(), m_upload_reads.size());
			const auto  count = m_upload_reads.size();
			m_scheduler.Record([reads, count](vk::CommandBuffer) {
				for (size_t i = 0; i < count; i++) {
					ReadGuestForUpload(reads[i].destination, reads[i].address, reads[i].size);
				}
			});
			m_staging_buffer.Commit();
			return m_staging_buffer.Handle();
		}
		const bool to_worker = UploadWorker::Enabled() && m_staging_buffer.IsCoherent() &&
		                       GuestGpu::IsGpuThread();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!to_worker || !UploadWorker::Get().Push(mapped + copy.srcOffset, address, copy.size)) {
				ReadGuestForUpload(mapped + copy.srcOffset, address, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	KYTY_PROFILER_BLOCK("BufferCache::UploadCopies(temporary)");
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

	if (!is_written && size <= CACHING_PAGESIZE && !IsShared(vaddr, size) &&
	    (!needs_device_address || m_stream_buffer.HasDeviceAddress()) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
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

bool BitmapRangeTicks() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BITMAP_RANGE_TICKS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

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

vk::DeviceAddress BufferCache::SrtVerifyReportAddress() {
	if (m_srt_verify_report_failed) {
		return 0;
	}
	if (m_srt_verify_report == nullptr) {
		m_srt_verify_report = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    256);
		if (!m_srt_verify_report->IsCoherent() || m_srt_verify_report->Mapped().size() < 32 ||
		    !m_srt_verify_report->HasDeviceAddress()) {
			LOGF("BufferCache: SRT verification reports unavailable (no host-coherent device memory)\n");
			m_srt_verify_report.reset();
			m_srt_verify_report_failed = true;
			return 0;
		}
		std::memset(m_srt_verify_report->Mapped().data(), 0, 32);
	}
	return m_srt_verify_report->BufferDeviceAddress();
}

uint64_t BufferCache::TakeSrtVerifyReport(std::array<uint32_t, 3>* detail) {
	if (m_srt_verify_report == nullptr) {
		return 0;
	}
	const auto* words = reinterpret_cast<const volatile uint32_t*>(m_srt_verify_report->Mapped().data());
	const uint32_t count = words[0];
	if (count == m_srt_verify_reports_seen) {
		return 0;
	}
	m_srt_verify_reports_seen = count;
	if (detail != nullptr) {
		*detail = {words[3], words[4], words[5]};
	}
	return (uint64_t {words[2]} << 32u) | words[1];
}

vk::DeviceAddress BufferCache::WriteBitmapAddress() {
	if (!ShaderRecompiler::IR::WriteBitmapEnabled() || m_write_bitmap_failed) {
		return 0;
	}
	if (m_write_bitmap == nullptr) {
		m_write_bitmap = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    WriteBitmapBytes * WriteBitmapEpochs);
		// The host reads and clears bits in place, so the memory must be host-coherent.
		if (!m_write_bitmap->IsCoherent() ||
		    m_write_bitmap->Mapped().size() < WriteBitmapBytes * WriteBitmapEpochs ||
		    !m_write_bitmap->HasDeviceAddress()) {
			LOGF("BufferCache: GPU write bitmap unavailable (no host-coherent device memory)\n");
			m_write_bitmap.reset();
			m_write_bitmap_failed = true;
			return 0;
		}
		std::memset(m_write_bitmap->Mapped().data(), 0,
		            static_cast<size_t>(WriteBitmapBytes * WriteBitmapEpochs));
		LOGF("BufferCache: GPU write bitmap on (%u x %" PRIu64 " MB, pages of written bindings of "
		     "at least %" PRIu64 " KB)\n",
		     WriteBitmapEpochs, WriteBitmapBytes >> 20u, WriteBitmapMinBytes() >> 10u);
	}
	// This frame's epoch (AdvanceWriteBitmapEra). GPU thread, as the marks of the same draws.
	return m_write_bitmap->BufferDeviceAddress() +
	       (m_bitmap_era % WriteBitmapEpochs) * WriteBitmapBytes;
}

// SSE4.1 (streaming loads) is enabled for this function only, so the build does not need
// -march=native; every x86-64 CPU able to run the emulator has SSE4.1.
#if defined(__clang__) || defined(__GNUC__)
__attribute__((target("sse4.1")))
#endif
void BufferCache::LoadWrittenWords(uint64_t word_begin, uint64_t word_end,
                                   std::vector<uint32_t>& out, int only_epoch) const {
	// The bitmap lives in write-combined device memory (ReBAR): every scalar CPU read there is
	// its own PCIe round trip (~1 us), which made a settle cost ~55 ms. Streaming loads read 64
	// bytes a round trip.
	const auto count = word_end - word_begin;
	out.assign(count, 0);
	const auto* base = reinterpret_cast<const uint32_t*>(m_write_bitmap->Mapped().data());
	for (uint32_t epoch = 0; epoch < WriteBitmapEpochs; epoch++) {
		if (only_epoch >= 0 && epoch != static_cast<uint32_t>(only_epoch)) {
			continue;
		}
		const auto* words = base + epoch * (WriteBitmapBytes / 4u) + word_begin;
		uint64_t    i     = 0;
		for (; i < count && (reinterpret_cast<uintptr_t>(words + i) & 15u) != 0; i++) {
			out[i] |= words[i];
		}
		for (; i + 4 <= count; i += 4) {
			alignas(16) uint32_t loaded[4];
			_mm_store_si128(reinterpret_cast<__m128i*>(loaded),
			                _mm_stream_load_si128(const_cast<__m128i*>(
			                    reinterpret_cast<const __m128i*>(words + i))));
			out[i] |= loaded[0];
			out[i + 1] |= loaded[1];
			out[i + 2] |= loaded[2];
			out[i + 3] |= loaded[3];
		}
		for (; i < count; i++) {
			out[i] |= words[i];
		}
	}
}

bool BufferCache::PredictWrites() const {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_PREDICT");
		return value == nullptr || value[0] != '0';
	}();
	// The first eras mark whole bindings, as without prediction, while the set is learnt.
	return enabled && m_write_bitmap != nullptr && m_bitmap_era >= 32;
}

void BufferCache::AdvanceWriteBitmapEra() {
	KYTY_PROFILER_BLOCK("BufferCache::AdvanceWriteBitmapEra");
	const auto            era = m_bitmap_era;
	std::vector<uint64_t> regions;
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& [region, last]: m_region_era) {
			if (last == era) {
				regions.push_back(region);
			}
		}
		m_bitmap_era = era + 1; // marks recorded from here on belong to the next era
	}
	std::sort(regions.begin(), regions.end());
	BitmapCopy copy;
	copy.era              = era;
	copy.tick             = m_scheduler.CurrentTick();
	uint64_t total_words  = 0;
	for (size_t first = 0; first < regions.size();) {
		auto last = first + 1;
		while (last < regions.size() && regions[last] == regions[last - 1] + 1) {
			last++;
		}
		const auto begin = regions[first] << WriteBitmapRegionBits;
		const auto end =
		    std::min((regions[last - 1] + 1) << WriteBitmapRegionBits, WriteBitmapLimit);
		first = last;
		if (begin >= end) {
			continue;
		}
		const auto word_begin = (begin >> WriteBitmapPageBits) >> 5u;
		const auto word_end   = (end >> WriteBitmapPageBits) >> 5u;
		copy.spans.emplace_back(word_begin, word_end);
		copy.offsets.push_back(total_words);
		total_words += word_end - word_begin;
	}
	if (total_words != 0) {
		const auto bytes = total_words * 4u;
		for (auto it = m_bitmap_staging_pool.begin(); it != m_bitmap_staging_pool.end(); ++it) {
			if ((*it)->Size() >= bytes) {
				copy.staging = std::move(*it);
				m_bitmap_staging_pool.erase(it);
				break;
			}
		}
		if (copy.staging == nullptr) {
			copy.staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                        vk::BufferUsageFlagBits::eTransferDst,
			                                        Common::AlignUp(bytes, uint64_t {64 * 1024}));
		}
		std::vector<vk::BufferCopy> copies;
		copies.reserve(copy.spans.size());
		for (size_t i = 0; i < copy.spans.size(); i++) {
			copies.emplace_back(copy.spans[i].first * 4u, copy.offsets[i] * 4u,
			                    (copy.spans[i].second - copy.spans[i].first) * 4u);
		}
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Recorder();
		const auto bitmap = m_write_bitmap->Handle();
		// The era's shader writes (atomic ORs through the bitmap's address), then copy, then
		// zero, then the next era's shaders.
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask =
		    vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                       nullptr);
		native.copyBuffer(bitmap, copy.staging->Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		vk::MemoryBarrier copied {};
		copied.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		copied.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &copied, 0, nullptr, 0,
		                       nullptr);
		for (const auto& [word_begin, word_end]: copy.spans) {
			native.fillBuffer(bitmap, word_begin * 4u, (word_end - word_begin) * 4u, 0);
		}
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
		                      vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                      vk::AccessFlagBits::eHostRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands |
		                           vk::PipelineStageFlagBits::eHost,
		                       {}, 1, &after, 0, nullptr, 0, nullptr);
	}
	m_bitmap_copies.push_back(std::move(copy));
	m_scheduler.DeferOperation([this] { ProcessBitmapCopies(); });
}

void BufferCache::ProcessBitmapCopies() {
	while (!m_bitmap_copies.empty() && m_scheduler.IsFree(m_bitmap_copies.front().tick)) {
		auto& copy = m_bitmap_copies.front();
		ProcessBitmapCopy(copy);
		if (copy.staging != nullptr) {
			m_bitmap_staging_pool.push_back(std::move(copy.staging));
		}
		m_bitmap_copies.pop_front();
	}
}

void BufferCache::ProcessBitmapCopy(BitmapCopy& copy) {
	KYTY_PROFILER_BLOCK("BufferCache::ProcessBitmapCopy");
	constexpr uint64_t PredictKeepEras = 256;
	const auto         start           = std::chrono::steady_clock::now();
	const auto*        words =
	    copy.staging != nullptr ? reinterpret_cast<const uint32_t*>(copy.staging->Mapped().data())
	                            : nullptr;
	if (words != nullptr && !copy.spans.empty()) {
		const auto words_total =
		    copy.offsets.back() + (copy.spans.back().second - copy.spans.back().first);
		copy.staging->Invalidate(0, words_total * 4u);
	}
	std::vector<uint64_t>                      protect;
	std::vector<std::pair<uint64_t, uint64_t>> release;
	uint64_t                                   written = 0, missed = 0, fresh = 0, released = 0;
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		// The pages the era wrote: written and not synced since (a sync recorded after them
		// would have found them GPU-written and taken them out of m_bitmap_written).
		for (size_t i = 0; words != nullptr && i < copy.spans.size(); i++) {
			const auto [word_begin, word_end] = copy.spans[i];
			const auto* span_words            = words + copy.offsets[i];
			for (uint64_t k = 0; k < word_end - word_begin; k++) {
				// Runs of written pages: the title writes ~33,000 pages an era, mostly whole words
				// (32 pages), and page by page the set operations below were ~5 ms of the command
				// thread an era. A whole word is one predicted block and one range operation.
				uint32_t bits = span_words[k];
				while (bits != 0) {
					const uint32_t first = static_cast<uint32_t>(std::countr_zero(bits));
					const uint32_t rest  = first + 1u < 32u ? (~bits >> (first + 1u)) : 0u;
					uint32_t       count = 1u;
					if (first + 1u < 32u) {
						count += rest == 0u ? 31u - first
						                    : static_cast<uint32_t>(std::countr_zero(rest));
					}
					count = std::min(count, 32u - first);
					bits &= count == 32u ? 0u : ~(((1u << count) - 1u) << first);
					const auto run_begin = (((word_begin + k) << 5u) + first) << WriteBitmapPageBits;
					const auto run_bytes = uint64_t {count} << WriteBitmapPageBits;
					written += count;
					if (count == 32u) {
						if (const auto [it, inserted] = m_predict_seen_blocks.try_emplace(run_begin, copy.era);
						    inserted) {
							m_predicted.Add(run_begin, run_bytes);
							m_predicted_generation++;
							fresh += 32u;
						} else {
							it->second = copy.era;
						}
					} else {
						for (uint32_t i = 0; i < count; i++) {
							const auto page = run_begin + (uint64_t {i} << WriteBitmapPageBits);
							if (const auto [it, inserted] = m_predict_seen.try_emplace(page, copy.era);
							    inserted) {
								m_predicted.Add(page, WriteBitmapPage);
								m_predicted_generation++;
								fresh++;
							} else {
								it->second = copy.era;
							}
						}
					}
					if (m_gpu_modified_ranges.Contains(run_begin, run_bytes)) {
						m_bitmap_written.Add(run_begin, run_bytes);
					} else {
						for (uint32_t i = 0; i < count; i++) {
							const auto page = run_begin + (uint64_t {i} << WriteBitmapPageBits);
							if (m_gpu_modified_ranges.Intersects(page, WriteBitmapPage)) {
								m_bitmap_written.Add(page, WriteBitmapPage);
							} else {
								protect.push_back(page); // marked below, outside the lock
							}
						}
					}
				}
			}
		}
		// Regions whose marks have all run and been accounted for: their tracked pages the GPU
		// did not write stop counting as GPU-written.
		std::vector<uint64_t> done;
		for (auto it = m_region_era.begin(); it != m_region_era.end();) {
			if (it->second <= copy.era) {
				done.push_back(it->first);
				m_bitmap_ticks.erase(it->first);
				it = m_region_era.erase(it);
			} else {
				++it;
			}
		}
		std::sort(done.begin(), done.end());
		for (size_t first = 0; first < done.size();) {
			auto last = first + 1;
			while (last < done.size() && done[last] == done[last - 1] + 1) {
				last++;
			}
			const auto begin = done[first] << WriteBitmapRegionBits;
			const auto end   = (done[last - 1] + 1) << WriteBitmapRegionBits;
			first            = last;
			RangeSet   candidates;
			m_gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](uint64_t from, uint64_t to) {
				m_bitmap_tracked.ForEachInRange(from, to - from, [&](uint64_t tb, uint64_t te) {
					const auto pb = Common::AlignUp(tb, WriteBitmapPage);
					const auto pe = Common::AlignDown(te, WriteBitmapPage);
					if (pb < pe) {
						candidates.Add(pb, pe - pb);
					}
				});
			});
			if (candidates.Empty()) {
				continue;
			}
			std::vector<std::pair<uint64_t, uint64_t>> keep;
			m_bitmap_written.ForEachInRange(begin, end - begin,
			                                [&](uint64_t from, uint64_t to) { keep.emplace_back(from, to); });
			// A download of these bytes in flight publishes them itself.
			m_downloading_ranges.ForEachInRange(begin, end - begin, [&](uint64_t from, uint64_t to) {
				keep.emplace_back(Common::AlignDown(from, WriteBitmapPage),
				                  Common::AlignUp(to, WriteBitmapPage));
			});
			for (const auto [from, to]: keep) {
				candidates.Subtract(from, to - from);
			}
			candidates.ForEach([&](uint64_t from, uint64_t to) { release.emplace_back(from, to); });
		}
		for (const auto [from, to]: release) {
			m_gpu_modified_ranges.Subtract(from, to - from);
			released += (to - from) >> WriteBitmapPageBits;
		}
		if (copy.era % 64 == 0) {
			for (auto it = m_predict_seen_blocks.begin(); it != m_predict_seen_blocks.end();) {
				if (it->second + PredictKeepEras < copy.era) {
					m_predicted.Subtract(it->first, WriteBitmapPage * 32u);
					m_predicted_generation++;
					it = m_predict_seen_blocks.erase(it);
				} else {
					++it;
				}
			}
			for (auto it = m_predict_seen.begin(); it != m_predict_seen.end();) {
				if (it->second + PredictKeepEras < copy.era) {
					m_predicted.Subtract(it->first, WriteBitmapPage);
					m_predicted_generation++;
					it = m_predict_seen.erase(it);
				} else {
					++it;
				}
			}
		}
	}
	const auto computed = std::chrono::steady_clock::now();
	// Writes the prediction did not protect: GPU-written from now on, as if marked (a game access
	// between the write and here saw the old bytes). Pages the CPU wrote since stay CPU-dirty.
	std::vector<uint64_t> marked;
	for (const auto page: protect) {
		if (m_memory_tracker.TryMarkRegionAsGpuModified(page, WriteBitmapPage)) {
			marked.push_back(page);
		}
	}
	if (!marked.empty()) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto page: marked) {
			m_gpu_modified_ranges.Add(page, WriteBitmapPage);
			m_bitmap_written.Add(page, WriteBitmapPage);
			m_bitmap_tracked.Add(page, WriteBitmapPage);
		}
		missed += marked.size();
	}
	for (const auto [from, to]: release) {
		if (m_memory_tracker.IsRegionGpuModified(from, to - from)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(from, to - from);
		}
	}
	static uint64_t s_written = 0, s_missed = 0, s_fresh = 0, s_released = 0;
	static double   s_compute_ms = 0, s_protect_ms = 0;
	const auto      done_at = std::chrono::steady_clock::now();
	s_written += written;
	s_missed += missed;
	s_fresh += fresh;
	s_released += released;
	s_compute_ms += std::chrono::duration<double, std::milli>(computed - start).count();
	s_protect_ms += std::chrono::duration<double, std::milli>(done_at - computed).count();
	if (static const bool log = std::getenv("KYTY_READBACK_STATS") != nullptr;
	    log && copy.era % 64 == 63) {
		std::fprintf(stderr,
		             "[write-bitmap] eras %llu-%llu: %llu pages written (%llu missed by the "
		             "prediction, %llu new), %llu pages released; predicted %zu pages in %zu "
		             "ranges, written set %zu ranges; compute %.1f ms, protect %.1f ms; predict %s\n",
		             static_cast<unsigned long long>(copy.era - 63),
		             static_cast<unsigned long long>(copy.era),
		             static_cast<unsigned long long>(s_written),
		             static_cast<unsigned long long>(s_missed),
		             static_cast<unsigned long long>(s_fresh),
		             static_cast<unsigned long long>(s_released), m_predict_seen.size(),
		             m_predicted.Count(), m_bitmap_written.Count(), s_compute_ms, s_protect_ms,
		             PredictWrites() ? "on" : "off");
		s_written = s_missed = s_fresh = s_released = 0;
		s_compute_ms = s_protect_ms = 0;
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferTracked(uint64_t vaddr, uint64_t size,
                                                              BufferId id) {
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	buffer.last_use_tick = m_scheduler.CurrentTick();
	m_tracked_serial++;
	if (PredictWrites() && !buffer.IsShared()) {
		// Only the pages the GPU wrote in recent eras become GPU-written (protected); the
		// era's bitmap copy accounts for the rest (ProcessBitmapCopy). Once per binding and era,
		// unless the prediction changed.
		constexpr uint32_t PredictLearnEras = 8;
		bool               learn            = false;
		bool               skip             = false;
		{
			std::unique_lock lock(m_dirty_ranges_mutex);
			auto&            marked = m_predict_marked[vaddr * 31u + size];
			if (marked.era != m_bitmap_era + 1) {
				marked.eras_seen++;
			}
			learn = marked.eras_seen <= PredictLearnEras;
			skip  = !learn && marked.era == m_bitmap_era + 1 &&
			       marked.generation == m_predicted_generation;
			marked.era        = m_bitmap_era + 1;
			marked.generation = m_predicted_generation;
		}
		if (learn) {
			(void)SynchronizeBuffer(buffer, vaddr, size, true, false);
			MarkGpuWritten(vaddr, size, true);
			return {&buffer, buffer.Offset(vaddr)};
		}
		(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
		if (skip) {
			return {&buffer, buffer.Offset(vaddr)};
		}
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		{
			std::unique_lock lock(m_dirty_ranges_mutex);
			const auto       tick = m_scheduler.CurrentTick();
			const auto last = std::min(vaddr + size, WriteBitmapLimit);
			for (auto region = vaddr >> WriteBitmapRegionBits;
			     vaddr < last && region <= (last - 1) >> WriteBitmapRegionBits; region++) {
				m_bitmap_ticks[region] = tick;
				m_region_era[region]   = m_bitmap_era;
			}
			if (BitmapRangeTicks() && vaddr < last) {
				m_pending_bitmap_writes.push_back({vaddr, last, tick});
			}
			m_predicted.ForEachInRange(vaddr, size, [&](uint64_t from, uint64_t to) {
				ranges.emplace_back(from, to);
			});
		}
		for (const auto [from, to]: ranges) {
			// Uploads bytes the CPU wrote meanwhile and marks the pages, atomically per region.
			(void)SynchronizeBuffer(buffer, from, to - from, true, false);
			MarkGpuWritten(from, to - from, true);
		}
		return {&buffer, buffer.Offset(vaddr)};
	}
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
	static std::atomic<uint64_t> s_range_narrowed {0}, s_range_kept {0};
	if (tick != 0 && BitmapRangeTicks() && !m_scheduler.IsFree(tick)) {
		// The region has a writer still to run; wait only if one of the pending writers covers
		// this window. Pages of a region whose bits are not processed yet count as written below
		// and download (their writers have finished), instead of waiting for the region's tick.
		while (!m_pending_bitmap_writes.empty() &&
		       m_scheduler.IsFree(m_pending_bitmap_writes.front().tick)) {
			m_pending_bitmap_writes.pop_front();
		}
		uint64_t window_tick = 0;
		for (const auto& w: m_pending_bitmap_writes) {
			if (w.begin < end && begin < w.end) {
				window_tick = std::max(window_tick, w.tick);
			}
		}
		static std::atomic<uint64_t> s_page_narrowed {0};
		if (window_tick != 0) {
			const auto pb = Common::AlignDown(vaddr, WriteBitmapPage);
			const auto pe = Common::AlignUp(vaddr + size, WriteBitmapPage);
			uint64_t   page_tick = 0;
			const PendingBitmapWrite* cover = nullptr;
			for (const auto& w: m_pending_bitmap_writes) {
				if (w.begin < pe && pb < w.end && w.tick >= page_tick) {
					page_tick = w.tick;
					cover     = &w;
				}
			}
			if (page_tick == 0) {
				begin           = pb;
				end             = pe;
				m_narrow_window = true;
				window_tick     = 0;
				s_page_narrowed++;
			} else if (static const bool log = std::getenv("KYTY_READBACK_STATS") != nullptr; log) {
				static std::atomic<uint32_t> logged {0};
				if (logged++ < 32) {
					std::fprintf(stderr,
					             "[bitmap-range] page 0x%llx covered by pending write 0x%llx-0x%llx tick %llu\n",
					             static_cast<unsigned long long>(pb),
					             static_cast<unsigned long long>(cover->begin),
					             static_cast<unsigned long long>(cover->end),
					             static_cast<unsigned long long>(cover->tick));
				}
			}
		}
		(window_tick == 0 ? s_range_narrowed : s_range_kept)++;
		if (static const bool log = std::getenv("KYTY_READBACK_STATS") != nullptr; log) {
			static std::atomic<uint32_t> logged {0};
			if ((s_range_narrowed + s_range_kept) % 64 == 1 && logged++ < 64) {
				std::fprintf(stderr,
				             "[bitmap-range] window 0x%llx+0x%llx region tick %llu: %s (narrowed %llu, "
				             "kept %llu, page-narrowed %llu, pending writes %zu)\n",
				             static_cast<unsigned long long>(begin),
				             static_cast<unsigned long long>(end - begin),
				             static_cast<unsigned long long>(tick),
				             window_tick == 0 ? "no pending writer covers it" : "a pending writer covers it",
				             static_cast<unsigned long long>(s_range_narrowed.load()),
				             static_cast<unsigned long long>(s_range_kept.load()),
				             static_cast<unsigned long long>(s_page_narrowed.load()),
				             m_pending_bitmap_writes.size());
			}
		}
		tick = window_tick;
	}
	static std::atomic<uint64_t> s_refines {0}, s_refine_waits {0}, s_refine_gpu {0},
	    s_refine_written {0}, s_refine_dropped {0};
	if (static const bool log = std::getenv("KYTY_READBACK_STATS") != nullptr;
	    log && ++s_refines % 1024 == 0) {
		std::fprintf(stderr,
		             "[write-bitmap] refine: 1024 calls: %llu wait for a pending write, %llu on "
		             "the GPU thread with one pending, %llu pages written (download), %llu "
		             "pages unwritten (no download)\n",
		             static_cast<unsigned long long>(s_refine_waits.exchange(0)),
		             static_cast<unsigned long long>(s_refine_gpu.exchange(0)),
		             static_cast<unsigned long long>(s_refine_written.exchange(0)),
		             static_cast<unsigned long long>(s_refine_dropped.exchange(0)));
	}
	if (tick != 0 && !m_scheduler.IsFree(tick)) {
		if (!async) {
			s_refine_gpu++;
			return false; // the GPU thread downloads the window as before
		}
		s_refine_waits++;
		lock.unlock();
		if (tick >= m_scheduler.CurrentTick()) {
			(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
		}
		wait_tick = tick;
		return true;
	}
	// Only GPU-written pieces of the window are looked at, and unwritten pages leave the set in
	// runs: page-by-page set operations cost the GPU thread more than the downloads they saved.
	// A page counts as written when an epoch still holds its bit or an earlier, cleared epoch
	// carried it into m_bitmap_written; bits are never cleared here (the epochs are cleared at
	// their next turn, and downloads take pages out of m_bitmap_written).
	if (begin >= WriteBitmapLimit) {
		return false;
	}
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
				// A region marked in an era whose bitmap copy is not processed yet may hold
				// writes not known here: its pages count as written.
				if (!m_region_era.contains(page >> WriteBitmapRegionBits) &&
				    !m_bitmap_written.Intersects(page, WriteBitmapPage)) {
					if (!unwritten.empty() && unwritten.back().second == page) {
						unwritten.back().second = page + WriteBitmapPage;
					} else {
						unwritten.emplace_back(page, page + WriteBitmapPage);
					}
					dropped++;
				} else {
					m_bitmap_written_pages++; // downloaded with the window
					s_refine_written++;
				}
			}
		});
	}
	for (const auto [from, to]: unwritten) {
		m_gpu_modified_ranges.Subtract(from, to - from);
	}
	s_refine_dropped += dropped;
	m_bitmap_dropped_pages += dropped;
	static uint64_t calls = 0;
	if (++calls % 4096 == 1) {
		LOGF("WriteBitmap: %" PRIu64 " refinements, %" PRIu64 " pages dropped unwritten, %" PRIu64
		     " pages written\n",
		     calls, m_bitmap_dropped_pages, m_bitmap_written_pages);
	}
	return false;
}

void BufferCache::ClearWriteBits(uint64_t vaddr, uint64_t size) {
	if (m_write_bitmap == nullptr || size == 0 || vaddr >= WriteBitmapLimit) {
		return;
	}
	if (m_bitmap_written.Empty() ||
	    !m_bitmap_written.Intersects(Common::AlignDown(vaddr, WriteBitmapPage),
	                                 Common::AlignUp(vaddr + size, WriteBitmapPage) -
	                                     Common::AlignDown(vaddr, WriteBitmapPage))) {
		return;
	}
	// Only the CPU-side set changes (the GPU never touches it): a page leaves it once none of
	// its bytes is GPU-written any more.
	const auto first = Common::AlignDown(vaddr, WriteBitmapPage);
	const auto last  = Common::AlignUp(vaddr + size, WriteBitmapPage);
	std::vector<uint64_t> clean;
	m_bitmap_written.ForEachInRange(first, last - first, [&](uint64_t from, uint64_t to) {
		for (auto page = from; page < to; page += WriteBitmapPage) {
			if (!m_gpu_modified_ranges.Intersects(page, WriteBitmapPage)) {
				clean.push_back(page);
			}
		}
	});
	for (const auto page: clean) {
		m_bitmap_written.Subtract(page, WriteBitmapPage);
	}
}

void BufferCache::SettleWriteBitmap(uint64_t completed_tick) {
	if (m_write_bitmap == nullptr || m_bitmap_ticks.empty()) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::SettleWriteBitmap");
	static uint64_t s_settle_pieces = 0, s_settle_tracked = 0;
	// Which tracked pages the GPU really writes, and how often a page is written for the first time
	static const bool                   s_log_written = std::getenv("KYTY_READBACK_STATS") != nullptr;
	static uint64_t                     s_written = 0, s_new_written = 0;
	static std::unordered_set<uint64_t> s_ever_written;
	static std::vector<uint64_t>        s_new_samples;
	static double                       s_load_ms = 0, s_lock_ms = 0;
	static uint64_t                     s_load_words = 0;
	const auto  settle_start = std::chrono::steady_clock::now();
	std::vector<std::pair<uint64_t, uint64_t>> unwritten; // runs of whole pages
	uint64_t                                   dropped = 0;
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		s_lock_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
		                                                       settle_start)
		                 .count();
		// The completed regions, joined into spans: one range-set walk per span, not per region
		// (per region, the walks cost ~50 ms a settle over the 2.7 GB heap).
		std::vector<uint64_t> regions;
		for (auto it = m_bitmap_ticks.begin(); it != m_bitmap_ticks.end();) {
			if (it->second > completed_tick) {
				++it; // a tracked write to this region is still to run
				continue;
			}
			regions.push_back(it->first);
			it = m_bitmap_ticks.erase(it);
		}
		std::sort(regions.begin(), regions.end());
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		for (const auto region: regions) {
			if (!spans.empty() && spans.back().second == region) {
				spans.back().second = region + 1;
			} else {
				spans.emplace_back(region, region + 1);
			}
		}
		// Each span's bits, ORed over the epochs, copied out of device memory once
		// (LoadWrittenWords); pages a cleared epoch carried into m_bitmap_written stay written.
		thread_local std::vector<uint32_t> local_words;
		const bool                         any_written = !m_bitmap_written.Empty();
		for (const auto [first_region, end_region]: spans) {
			const auto region_begin = first_region << WriteBitmapRegionBits;
			const auto region_size  = (end_region - first_region) << WriteBitmapRegionBits;
			if (region_begin >= WriteBitmapLimit) {
				continue;
			}
			const auto span_end   = std::min(region_begin + region_size, WriteBitmapLimit);
			const auto word_begin = (region_begin >> WriteBitmapPageBits) >> 5u; // 32 pages a word
			const auto word_end   = (span_end >> WriteBitmapPageBits) >> 5u;
			const auto load_start = std::chrono::steady_clock::now();
			LoadWrittenWords(word_begin, word_end, local_words);
			s_load_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
			                                                       load_start)
			                 .count();
			s_load_words += word_end - word_begin;
			const auto word_at = [&](uint64_t word_index) {
				// Outside the span cannot happen (pieces are clipped to it); count it as written.
				return word_index >= word_begin && word_index < word_end
				           ? local_words[word_index - word_begin]
				           : ~uint32_t {0};
			};
			const auto unsynced = [&](uint64_t page, uint64_t bytes) {
				return any_written && m_bitmap_written.Intersects(page, bytes);
			};
			std::vector<std::pair<uint64_t, uint64_t>> pieces;
			m_gpu_modified_ranges.ForEachInRange(region_begin, region_size,
			                                     [&](uint64_t from, uint64_t to) {
				                                     pieces.emplace_back(from, to);
			                                     });
			s_settle_pieces += pieces.size();
			for (const auto [from, to]: pieces) {
				m_bitmap_tracked.ForEachInRange(from, to - from, [&](uint64_t tb, uint64_t te) {
					s_settle_tracked++;
					for (auto page = Common::AlignUp(tb, WriteBitmapPage);
					     page + WriteBitmapPage <= te && page + WriteBitmapPage <= WriteBitmapLimit;) {
						const auto index = page >> WriteBitmapPageBits;
						const auto word  = word_at(index >> 5u);
						if (word == 0 && (index & 31u) == 0 && page + 32 * WriteBitmapPage <= te &&
						    !unsynced(page, 32 * WriteBitmapPage)) {
							// 32 unwritten pages at once
							if (!unwritten.empty() && unwritten.back().second == page) {
								unwritten.back().second = page + 32 * WriteBitmapPage;
							} else {
								unwritten.emplace_back(page, page + 32 * WriteBitmapPage);
							}
							dropped += 32;
							page += 32 * WriteBitmapPage;
							continue;
						}
						if ((word & (uint32_t {1} << (index & 31u))) == 0 &&
						    !unsynced(page, WriteBitmapPage)) {
							if (!unwritten.empty() && unwritten.back().second == page) {
								unwritten.back().second = page + WriteBitmapPage;
							} else {
								unwritten.emplace_back(page, page + WriteBitmapPage);
							}
							dropped++;
						} else if (s_log_written) {
							s_written++;
							if (s_ever_written.insert(page).second) {
								s_new_written++;
								if (s_new_samples.size() < 8) {
									s_new_samples.push_back(page);
								}
							}
						}
						page += WriteBitmapPage;
					}
				});
			}
		}
		// A download of these bytes in flight publishes them itself; leave those pages alone.
		std::erase_if(unwritten, [&](const auto& run) {
			return m_downloading_ranges.Intersects(run.first, run.second - run.first);
		});
		for (const auto [from, to]: unwritten) {
			m_gpu_modified_ranges.Subtract(from, to - from);
		}
		m_bitmap_dropped_pages += dropped;
	}
	const auto computed = std::chrono::steady_clock::now();
	for (const auto [from, to]: unwritten) {
		if (m_memory_tracker.IsRegionGpuModified(from, to - from)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(from, to - from);
		}
	}
	static uint64_t settles = 0;
	static double   compute_ms = 0, unmark_ms = 0;
	const auto      done = std::chrono::steady_clock::now();
	compute_ms += std::chrono::duration<double, std::milli>(computed - settle_start).count();
	unmark_ms += std::chrono::duration<double, std::milli>(done - computed).count();
	if (static const bool log = std::getenv("KYTY_READBACK_STATS") != nullptr;
	    log && ++settles % 64 == 1) {
		std::fprintf(stderr,
		             "[write-bitmap] settle #%llu: %llu pages released (%zu runs), total %llu; "
		             "compute %.1f ms, unprotect %.1f ms over the last 64; dirty pieces %llu, tracked "
		             "pieces %llu, dirty ranges %zu, tracked ranges %zu\n",
		             static_cast<unsigned long long>(settles),
		             static_cast<unsigned long long>(dropped), unwritten.size(),
		             static_cast<unsigned long long>(m_bitmap_dropped_pages), compute_ms, unmark_ms,
		             static_cast<unsigned long long>(s_settle_pieces),
		             static_cast<unsigned long long>(s_settle_tracked), m_gpu_modified_ranges.Count(),
		             m_bitmap_tracked.Count());
		std::string samples;
		for (const auto page: s_new_samples) {
			samples += fmt::format(" {:x}", page);
		}
		std::fprintf(stderr,
		             "[write-bitmap]   written pages %llu over the last 64 settles, %llu never written "
		             "before (ever written %zu):%s\n",
		             static_cast<unsigned long long>(s_written),
		             static_cast<unsigned long long>(s_new_written), s_ever_written.size(),
		             samples.c_str());
		std::fprintf(stderr,
		             "[write-bitmap]   of the compute: lock wait %.1f ms, bitmap loads %.1f ms "
		             "(%llu words)\n",
		             s_lock_ms, s_load_ms, static_cast<unsigned long long>(s_load_words));
		s_lock_ms = s_load_ms = 0;
		s_load_words          = 0;
		compute_ms = unmark_ms = 0;
		s_settle_pieces = s_settle_tracked = 0;
		s_written = s_new_written = 0;
		s_new_samples.clear();
	}
}

static void WatchGpuWriteDiag(uint64_t vaddr, uint64_t size, bool shared) {
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
			first = seen.emplace(index, BufferCache::s_diag_shader_hash).second;
		}
		if (first && watched.fetch_add(1) < 128) {
			std::fprintf(stderr, "GPU write%s covers watched #%zu 0x%016" PRIx64 "+0x%" PRIx64
			     ": range=0x%016" PRIx64 " size=0x%" PRIx64 " shader=0x%016" PRIx64 "\n%s",
			     shared ? " (shared)" : "", index, watch_begin, watch_size, vaddr, size, BufferCache::s_diag_shader_hash,
			     BufferCache::s_diag_shader_hash == 0 ? Common::HostBacktrace().c_str() : "");
		}
	}
}

void BufferCache::MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked) {
	KYTY_PROFILER_BLOCK("Obtain::MarkWritten");
	if (IsShared(vaddr, size) && m_shared_ranges.Contains(vaddr, size)) {
		// GPU writes to a shared buffer land in guest memory: no download, no page protection.
		// Until they have executed, the bytes count as GPU-written for the GPU thread's own reads
		// (descriptors, indirect arguments), which must see them in command order; game threads
		// read guest memory as it is, as on the console.
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner != nullptr && *owner && m_slot_buffers[*owner].IsShared()) {
			const auto begin = m_slot_buffers[*owner].CpuAddress();
			const auto bytes = m_slot_buffers[*owner].Size();
			const auto tick  = m_scheduler.CurrentTick();
			m_shared_write_ticks[begin] = tick;
			{
				std::unique_lock lock(m_dirty_ranges_mutex);
				m_gpu_modified_ranges.Add(vaddr, size);
			}
			WatchGpuWriteDiag(vaddr, size, true);
			m_scheduler.DeferOperation([this, begin, bytes, tick] {
				const auto found = m_shared_write_ticks.find(begin);
				if (found == m_shared_write_ticks.end() || found->second != tick) {
					return;
				}
				m_shared_write_ticks.erase(found);
				std::unique_lock lock(m_dirty_ranges_mutex);
				m_gpu_modified_ranges.Subtract(begin, bytes);
			});
		}
		return;
	}
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
				m_region_era[region]   = m_bitmap_era; // its bits go to this era's epoch
			}
			if (BitmapRangeTicks()) {
				m_pending_bitmap_writes.push_back({first, last, tick});
			}
		} else if (!m_bitmap_tracked.Empty()) {
			m_bitmap_tracked.Subtract(first, last - first);
		}
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	NoteGpuWriteTick(vaddr, size);
	NoteEagerWrite(vaddr, size);
	// KYTY_LOG_HUGE_WRITES=1: the shaders whose bindings mark 64 MB or more GPU-written at once.
	if (static const bool huge_log = std::getenv("KYTY_LOG_HUGE_WRITES") != nullptr;
	    huge_log && size >= (64ull << 20u)) {
		static std::mutex                             huge_mutex;
		static std::unordered_map<uint64_t, uint64_t> huge;
		std::lock_guard                               lock(huge_mutex);
		if (huge[s_diag_shader_hash]++ % 500 == 0) {
			std::fprintf(stderr, "[huge-write] shader=%016llx range=0x%llx size=%llu MB count=%llu\n",
			             static_cast<unsigned long long>(s_diag_shader_hash),
			             static_cast<unsigned long long>(vaddr),
			             static_cast<unsigned long long>(size >> 20u),
			             static_cast<unsigned long long>(huge[s_diag_shader_hash]));
		}
	}
	WatchGpuWriteDiag(vaddr, size, false);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
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

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size) {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasPendingDownload(uint64_t vaddr, uint64_t size) {
	if (!m_downloading_any.load(std::memory_order_acquire)) {
		return false;
	}
	std::shared_lock lock(m_dirty_ranges_mutex);
	return m_downloading_ranges.Intersects(vaddr, size);
}

// A read of guest memory that bypasses page protection (the SRT walker, the clean-page caches)
// finds bytes downloading as not GPU-dirty, but guest memory holds their older value until the
// download is published. An asynchronous readback (a game thread's fault, a readback window, the
// collector, an SRT defer) publishes when its tick completes, while the GPU thread goes on: an
// SRT refresh in that window read the previous layout of a BVH blob the game had just instanced
// again, and the refit it bound ran with another instance's buffers (counters double-decremented,
// parent links broken, the refit looping until the device was lost).
void BufferCache::WaitDownloadsLanded(uint64_t vaddr, uint64_t size) {
	for (int round = 0; round < 8 && HasPendingDownload(vaddr, size); round++) {
		if (const auto marker = FindCopyReadback(vaddr, size); marker != 0) {
			CompleteCopyReadback(marker);
			continue;
		}
		const auto tick = std::max(m_download_publish_tick.load(std::memory_order_relaxed),
		                           m_last_async_download_tick);
		if (tick == 0) {
			break;
		}
		if (!m_scheduler.IsFree(tick)) {
			m_scheduler.Wait(tick);
		}
		m_scheduler.WaitPriorityOperations(tick);
	}
	if (HasPendingDownload(vaddr, size)) {
		// Not published by the latest download's tick: drain everything recorded so far.
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
}

std::string BufferCache::DescribeWriteState(uint64_t vaddr) {
	const auto       page = Common::AlignDown(vaddr, WriteBitmapPage);
	std::unique_lock lock(m_dirty_ranges_mutex);
	const auto       found  = m_host_read_serial.find(page);
	const auto       era_it = m_region_era.find(page >> WriteBitmapRegionBits);
	const auto       seen   = m_predict_seen.find(page);
	const auto       id     = FindBuffer(page, WriteBitmapPage);
	return fmt::format(
	    "dirty={} tracked={} written_set={} predicted={} (seen era {}) region_era={} era={} "
	    "serial={}/{} buffer=0x{:x}+0x{:x} shared={} predict={} tracker_gpu={} tracker_cpu={} "
	    "downloading={}",
	    m_gpu_modified_ranges.Intersects(page, WriteBitmapPage),
	    m_bitmap_tracked.Intersects(page, WriteBitmapPage),
	    m_bitmap_written.Intersects(page, WriteBitmapPage),
	    m_predicted.Intersects(page, WriteBitmapPage),
	    seen != m_predict_seen.end() ? static_cast<int64_t>(seen->second) : -1,
	    era_it != m_region_era.end() ? static_cast<int64_t>(era_it->second) : -1, m_bitmap_era,
	    found != m_host_read_serial.end() ? found->second : 0, m_tracked_serial,
	    m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size(), m_slot_buffers[id].IsShared(),
	    PredictWrites(), m_memory_tracker.IsRegionGpuModified(page, WriteBitmapPage),
	    m_memory_tracker.IsRegionCpuModified(page, WriteBitmapPage),
	    m_downloading_ranges.Intersects(page, WriteBitmapPage));
}

void BufferCache::WaitSharedWrites(uint64_t vaddr, uint64_t size) {
	// Write prediction: an unprotected page of a region whose tracked writes are not accounted
	// for yet may hold GPU writes this read must see. It becomes GPU-written (and predicted), so
	// the read faults into a download, in command order.
	if (PredictWrites() && vaddr < WriteBitmapLimit) {
		const auto first = Common::AlignDown(vaddr, WriteBitmapPage);
		const auto last  = Common::AlignUp(vaddr + size, WriteBitmapPage);
		bool       mark  = false;
		{
			std::unique_lock lock(m_dirty_ranges_mutex);
			// Only when a tracked binding was obtained since this page was last protected here:
			// otherwise nothing can have written it since its download.
			if (m_region_era.contains(first >> WriteBitmapRegionBits) &&
			    !m_gpu_modified_ranges.Intersects(first, last - first)) {
				auto& serial = m_host_read_serial[first];
				mark         = serial != m_tracked_serial;
				serial       = m_tracked_serial;
			}
		}
		if (mark && m_memory_tracker.TryMarkRegionAsGpuModified(first, last - first)) {
			std::unique_lock lock(m_dirty_ranges_mutex);
			m_gpu_modified_ranges.Add(first, last - first);
			m_bitmap_tracked.Add(first, last - first);
			for (auto page = first; page < last; page += WriteBitmapPage) {
				if (m_predict_seen.try_emplace(page, m_bitmap_era).second) {
					m_predicted.Add(page, WriteBitmapPage);
					m_predicted_generation++;
				}
			}
		}
		// Page protection set during this submission may not be in force yet, so the read
		// would not fault: download GPU-written bytes here, as the fault would.
		bool dirty = false;
		{
			std::unique_lock lock(m_dirty_ranges_mutex);
			dirty = m_gpu_modified_ranges.Intersects(first, last - first);
		}
		KYTY_PROFILER_BLOCK("WaitSharedWrites::HostReadDownload");
		for (int attempt = 0; dirty && attempt < 64; attempt++) {
			const auto tick = ReadMemoryStep(first, last - first, false, true, false);
			if (tick == 0) {
				break;
			}
			if ((tick & CopyReadbackMarker) != 0) {
				CompleteCopyReadback(tick);
				continue;
			}
			if (tick >= m_scheduler.CurrentTick()) {
				(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
			}
			m_scheduler.Wait(tick);
		}
	}
	if (m_shared_ranges.Empty() || !m_shared_ranges.Intersects(vaddr, size)) {
		return;
	}
	uint64_t tick = 0;
	auto     it   = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < vaddr + size; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (!buffer.IsShared() || buffer.CpuAddress() + buffer.Size() <= vaddr) {
			continue;
		}
		if (const auto found = m_shared_write_ticks.find(buffer.CpuAddress());
		    found != m_shared_write_ticks.end()) {
			tick = std::max(tick, found->second);
		}
	}
	if (tick == 0 || m_scheduler.IsFree(tick)) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::WaitSharedWrites");
	if (tick >= m_scheduler.CurrentTick()) {
		(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
	}
	m_scheduler.Wait(tick);
}

bool BufferCache::PeekGpuCopy(uint64_t vaddr, uint64_t size, std::vector<uint32_t>& out) {
	auto it = m_buffers.upper_bound(vaddr);
	if (it == m_buffers.begin()) {
		return false;
	}
	--it;
	auto& buffer = m_slot_buffers[it->second];
	if (vaddr < buffer.CpuAddress() || vaddr + size > buffer.CpuAddress() + buffer.Size() || size == 0) {
		return false;
	}
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                          vk::BufferUsageFlagBits::eTransferDst, size);
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
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 0, nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {vaddr - buffer.CpuAddress(), 0, size};
	native.copyBuffer(buffer.Handle(), temporary->Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = temporary->Handle();
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 0,
	                       nullptr, 1, &after, 0, nullptr);
	const auto tick = m_scheduler.CurrentTick();
	(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
	m_scheduler.Wait(tick);
	temporary->Invalidate(0, size);
	out.resize(static_cast<size_t>(size / 4u));
	std::memcpy(out.data(), temporary->Mapped().data(), out.size() * 4u);
	return true;
}

void BufferCache::SnapshotGpu(const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                              vk::Buffer raw, uint64_t raw_offset, uint64_t raw_size,
                              SnapshotDone done) {
	struct Piece {
		vk::Buffer source;
		uint64_t   source_offset;
		uint64_t   size;
		uint64_t   at;
	};
	std::vector<Piece>    pieces;
	std::vector<uint64_t> placed(ranges.size(), UINT64_MAX);
	uint64_t              total = 0;
	for (size_t i = 0; i < ranges.size(); i++) {
		const auto [vaddr, size] = ranges[i];
		auto it = m_buffers.upper_bound(vaddr);
		if (size == 0 || (size & 3u) != 0 || it == m_buffers.begin()) {
			continue;
		}
		--it;
		const auto& buffer = m_slot_buffers[it->second];
		if (vaddr < buffer.CpuAddress() || vaddr + size > buffer.CpuAddress() + buffer.Size()) {
			continue;
		}
		pieces.push_back({buffer.Handle(), vaddr - buffer.CpuAddress(), size, total});
		placed[i] = total;
		total += Common::AlignUp(size, 16);
	}
	const uint64_t raw_at = total;
	if (raw && raw_size != 0) {
		pieces.push_back({raw, raw_offset, raw_size, raw_at});
		total += Common::AlignUp(raw_size, 16);
	}
	if (total == 0) {
		return;
	}
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                          vk::BufferUsageFlagBits::eTransferDst, total);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto        native = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 1, &before, 0, nullptr, 0, nullptr);
	for (const auto& piece: pieces) {
		const vk::BufferCopy copy {piece.source_offset, piece.at, piece.size};
		native.copyBuffer(piece.source, temporary->Handle(), 1, &copy);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1,
	                       &after, 0, nullptr, 0, nullptr);
	std::vector<uint64_t> sizes;
	sizes.reserve(ranges.size());
	for (const auto& range: ranges) {
		sizes.push_back(range.second);
	}
	m_scheduler.DeferOperation([owner = std::move(temporary), placed = std::move(placed),
	                            sizes = std::move(sizes), raw_at, raw_size, total,
	                            done = std::move(done)]() mutable {
		owner->Invalidate(0, total);
		const auto*                        bytes = owner->Mapped().data();
		std::vector<std::vector<uint32_t>> out(placed.size());
		for (size_t i = 0; i < placed.size(); i++) {
			if (placed[i] == UINT64_MAX) continue;
			out[i].resize(static_cast<size_t>(sizes[i] / 4u));
			std::memcpy(out[i].data(), bytes + placed[i], out[i].size() * 4u);
		}
		std::vector<uint32_t> raw_words(static_cast<size_t>(raw_size / 4u));
		if (!raw_words.empty()) {
			std::memcpy(raw_words.data(), bytes + raw_at, raw_words.size() * 4u);
		}
		done(out, raw_words);
	});
}

void BufferCache::DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size) {
	std::vector<std::pair<uint64_t, uint64_t>> downloaded;
	auto                                       it = m_buffers.upper_bound(vaddr);
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
	// KYTY_FPS_LOG: [buffer-gc] per 5 s, the collections, what they deleted and waited for, the
	// cache's bytes against its thresholds and the buffers created.
	if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_gc_stats.last > std::chrono::seconds(5)) {
			std::fprintf(stderr,
			             "[buffer-gc] 5 s: runs %llu (aggressive %llu) deleted %llu waits %llu (%.0f ms) "
			             "created %llu used %llu MB trigger %llu MB critical %llu MB buffers %zu\n",
			             static_cast<unsigned long long>(m_gc_stats.runs),
			             static_cast<unsigned long long>(m_gc_stats.aggressive),
			             static_cast<unsigned long long>(m_gc_stats.deleted),
			             static_cast<unsigned long long>(m_gc_stats.waits), m_gc_stats.wait_ms,
			             static_cast<unsigned long long>(m_gc_stats.created),
			             static_cast<unsigned long long>(m_total_used_memory >> 20u),
			             static_cast<unsigned long long>(m_trigger_gc_memory >> 20u),
			             static_cast<unsigned long long>(m_critical_gc_memory >> 20u), m_buffers.size());
			m_gc_stats      = {};
			m_gc_stats.last = now;
		}
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	m_gc_stats.runs++;
	m_gc_stats.aggressive += aggressive ? 1 : 0;
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
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
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
			DeleteBuffer(id);
		} else if (dirty) {
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		gc_deleted++;
		m_gc_stats.deleted++;
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
	const auto wait_start      = std::chrono::steady_clock::now();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	WaitCopyReadbacks();
	m_gc_stats.waits++;
	m_gc_stats.wait_ms +=
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count();
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

// KYTY_BDA_TOUCHED=0: the BDA walk visits every CPU-dirty region of every mapped range (as
// before), not only the regions where pages became CPU-dirty since the walk last ran.
bool BufferCache::BdaTouchedEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BDA_TOUCHED");
		return (value == nullptr || value[0] != '0') && !HotPagesEnabled();
	}();
	return enabled;
}

void BufferCache::SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.ForEachMaybeCpuDirtyRegion(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) {
		    SynchronizeBuffersInRange(address, bytes);
	    });
}

void BufferCache::SynchronizeTouchedMappedBuffers(const RangeSet& mapped) {
	// The BDA walk (once per guest submission, 126 a second at env3) went through each of ~13,000
	// mapped ranges and synced every buffer of every CPU-dirty region again; a region stays
	// CPU-dirty while the game writes memory no buffer covers: ~0.5 ms a walk (2026-10-20). Pages
	// a sync uploads are protected again, so only regions where pages became CPU-dirty since the
	// last walk (the tracker's touched bits) need it, clipped to the mapped ranges.
	// KYTY_FPS_LOG: [bda-walk] per 5 s, walks, touched regions and buffers synced.
	if (mapped.Empty()) {
		return;
	}
	static const bool log   = std::getenv("KYTY_FPS_LOG") != nullptr;
	static uint64_t   walks = 0, regions = 0, synced = 0;
	static auto       last  = std::chrono::steady_clock::now();
	walks++;
	const auto sync = [&](uint64_t address, uint64_t bytes) {
		regions++;
		mapped.ForEachInRange(address, bytes, [&](uint64_t begin, uint64_t end) {
			synced += SynchronizeBuffersInRange(begin, end - begin);
		});
	};
	// The tracker covers the lower address space and the extended memory window (GuestRange).
	const auto walk = [&](uint64_t window_begin, uint64_t window_end) {
		const auto begin = std::max({mapped.Lowest(), window_begin, TRACKER_PAGE_SIZE});
		const auto end   = std::min(mapped.Highest(), window_end);
		if (begin < end) {
			m_memory_tracker.ForEachCpuTouchedRegion(begin, end - begin, sync);
		}
	};
	walk(0, LOWER_ADDRESS_SIZE);
	walk(LibKernel::Memory::kExtendedMemoryBase,
	     LibKernel::Memory::kExtendedMemoryBase + LibKernel::Memory::kExtendedMemorySize);
	if (log && (walks & 63u) == 0) {
		if (const auto now = std::chrono::steady_clock::now(); now - last > std::chrono::seconds(5)) {
			std::fprintf(stderr, "[bda-walk] 5 s: walks %llu, touched regions %llu, buffers synced %llu\n",
			             static_cast<unsigned long long>(walks), static_cast<unsigned long long>(regions),
			             static_cast<unsigned long long>(synced));
			walks = regions = synced = 0;
			last  = now;
		}
	}
}

uint64_t BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end    = vaddr + size;
	auto       it     = m_buffers.upper_bound(vaddr);
	uint64_t   synced = 0;
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
			synced++;
		}
	}
	return synced;
}

// Copy-queue readbacks (KYTY_COPY_QUEUE_READBACK, default on; =0 restores the main-queue
// download). A game thread's fault on GPU-written memory recorded the window's download into
// the current command buffer, submitted it and waited for that tick: for everything recorded
// and queued before it to execute (median 1.9 ms, mean 3.7 ms a readback; the game's worker
// threads spent half their time there, Wolverine's jungle, 2026-10-09). The GPU writes to the
// faulting bytes have usually long finished. When every GPU write to the window has executed,
// its GPU-written bytes are copied on the second queue of the graphics family instead, behind
// nothing, and the faulting thread waits for that copy alone, then publishes the bytes itself.
// IDXTRI's wolverine-perf has the idea (bbad1b39), with the GPU thread waiting for the copy.
static bool CopyQueueReadbackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_QUEUE_READBACK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void BufferCache::NoteGpuWriteTick(uint64_t vaddr, uint64_t size) {
	if (!CopyQueueReadbackEnabled() || size == 0) {
		return;
	}
	constexpr uint64_t BlockBits = 20;
	const auto         tick      = m_scheduler.CurrentTick();
	const auto         first     = vaddr >> BlockBits;
	const auto         last      = (vaddr + size - 1) >> BlockBits;
	if (last - first < 64) {
		for (auto block = first; block <= last; block++) {
			m_write_ticks[block] = tick;
		}
		return;
	}
	for (auto& write: m_large_writes) {
		if (write[0] == vaddr && write[1] == vaddr + size) {
			write[2] = tick;
			return;
		}
	}
	if (m_large_writes.size() >= 16) {
		std::erase_if(m_large_writes,
		              [this](const auto& write) { return m_scheduler.IsFree(write[2]); });
	}
	m_large_writes.push_back({vaddr, vaddr + size, tick});
}

uint64_t BufferCache::LatestGpuWriteTick(uint64_t begin, uint64_t end) const {
	constexpr uint64_t BlockBits = 20;
	uint64_t           latest    = 0;
	for (auto block = begin >> BlockBits; block <= (end - 1) >> BlockBits; block++) {
		if (const auto found = m_write_ticks.find(block); found != m_write_ticks.end()) {
			latest = std::max(latest, found->second);
		}
	}
	for (const auto& write: m_large_writes) {
		if (write[0] < end && begin < write[1]) {
			latest = std::max(latest, write[2]);
		}
	}
	// 0: no write noted (none since the cache started tracking them).
	return latest;
}

// KYTY_COPY_QUEUE_INFLIGHT=1 (opt-in): a window whose last writer has not finished is copied on
// the second queue too, behind that writer's tick only (submitted first when it is the tick being
// recorded), rather than behind everything recorded since. Off by default: the GPU thread can
// then wait for a copy whose writer's tick it has not submitted yet (the title hung at its first
// BDA dispatch, 2026-10-09), and it was no faster at the title menu.
static bool CopyQueueInflight() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_QUEUE_INFLIGHT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

uint64_t BufferCache::FindCopyReadback(uint64_t vaddr, uint64_t size) const {
	if (m_copy_in_flight.load(std::memory_order_acquire) == 0) {
		return 0;
	}
	for (uint32_t i = 0; i < CopyReadbackSlots; i++) {
		const auto& slot = m_copy_slots[i];
		// The GPU thread fills a slot before it marks it busy, and only it does.
		if (slot.busy.load(std::memory_order_acquire) != 0 && slot.window_begin < vaddr + size &&
		    vaddr < slot.window_end) {
			return CopyReadbackMarker | (uint64_t {slot.generation} << 8u) | i;
		}
	}
	return 0;
}

uint64_t BufferCache::TryCopyQueueReadback(Buffer& buffer, uint64_t window_begin,
                                           uint64_t window_end) {
	constexpr uint64_t StagingSize = 8ull * 1024 * 1024;
	if (!CopyQueueReadbackEnabled() || m_graphics.readback_queue == nullptr ||
	    window_begin >= window_end) {
		return 0;
	}
	const auto latest_write = LatestGpuWriteTick(window_begin, window_end);
	const bool finished     = latest_write == 0 || m_scheduler.IsFree(latest_write);
	if (!finished) {
		m_copy_running++;
	}
	if (!finished && !CopyQueueInflight()) {
		if (m_copy_running % 2048 == 1) {
			std::fprintf(stderr,
			             "[copy-readback] writers_running=%" PRIu64 " taken=%" PRIu64 "\n",
			             m_copy_running, m_copy_taken);
		}
		return 0;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		// A main-queue download of these bytes publishes them again when it lands.
		if (m_downloading_ranges.Intersects(window_begin, window_end - window_begin)) {
			return 0;
		}
		m_gpu_modified_ranges.ForEachInRange(
		    window_begin, window_end - window_begin, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    total_size += Common::AlignUp(end - start, 64);
		    });
	}
	if (copies.empty() || total_size > StagingSize) {
		return 0;
	}
	uint32_t index = CopyReadbackSlots;
	for (uint32_t i = 0; i < CopyReadbackSlots; i++) {
		if (m_copy_slots[i].busy.load(std::memory_order_acquire) == 0) {
			index = i;
			break;
		}
	}
	if (index == CopyReadbackSlots) {
		m_copy_busy++;
		return 0;
	}
	KYTY_PROFILER_BLOCK("BufferCache::CopyQueueReadback");
	auto& slot   = m_copy_slots[index];
	auto  device = m_graphics.device;
	if (slot.pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.queue_family;
		RequireVulkanSuccess(device.createCommandPool(&pool_info, nullptr, &slot.pool),
		                     "create readback command pool");
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = slot.pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		RequireVulkanSuccess(device.allocateCommandBuffers(&allocate, &slot.command),
		                     "allocate readback command buffer");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(device.createFence(&fence_info, nullptr, &slot.fence),
		                     "create readback fence");
		slot.staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst,
		                                        StagingSize);
	}

	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(slot.command.begin(&begin), "begin readback command buffer");
	slot.command.copyBuffer(buffer.Handle(), slot.staging->Handle(),
	                        static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier to_host {};
	to_host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer              = slot.staging->Handle();
	to_host.offset              = 0;
	to_host.size                = total_size;
	slot.command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                             vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &to_host, 0,
	                             nullptr);
	RequireVulkanSuccess(slot.command.end(), "end readback command buffer");

	// The memory dependency on the writers: a wait for the timeline value of the last one (one
	// the GPU has reached when they have finished). A writer in the tick being recorded is
	// submitted first; a timeline wait may precede its signal's submission.
	if (!finished && latest_write >= m_scheduler.CurrentTick()) {
		(KYTY_SUBMIT_REASON(), m_scheduler.Flush());
	}
	const uint64_t reached =
	    finished ? m_scheduler.GetMasterSemaphore().KnownGpuTick() : latest_write;
	const vk::Semaphore             master     = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &reached;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &master;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &slot.command;
	RequireVulkanSuccess(m_graphics.readback_queue.submit(1, &submit, slot.fence),
	                     "submit readback");

	// The bytes are on their way back: they stop counting as GPU-written and count as
	// downloading until the caller publishes them, as with a main-queue download.
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			m_downloading_ranges.Add(buffer_address + copy.srcOffset, copy.size);
			m_gpu_modified_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
			ClearWriteBits(buffer_address + copy.srcOffset, copy.size);
		}
		NoteDownloading();
	}
	slot.buffer_address = buffer_address;
	slot.window_begin   = window_begin;
	slot.window_end     = window_end;
	slot.total_size     = total_size;
	slot.copies         = std::move(copies);
	slot.snapshot       = false;
	slot.generation     = (slot.generation + 1u) & 0xffffffu;
	m_copy_in_flight.fetch_add(1, std::memory_order_acq_rel);
	slot.busy.store(1, std::memory_order_release);
	Profiler::Add(Profiler::Counter::Readbacks);
	Profiler::Add(Profiler::Counter::ReadbackBytes, static_cast<int64_t>(total_size));
	if (++m_copy_taken % 2048 == 1) {
		std::fprintf(stderr,
		             "[copy-readback] taken=%" PRIu64 " writers_running=%" PRIu64
		             " slots_busy=%" PRIu64 "\n",
		             m_copy_taken, m_copy_running, m_copy_busy);
	}
	return CopyReadbackMarker | (uint64_t {slot.generation} << 8u) | index;
}

uint64_t BufferCache::TrySnapshotReadback(Buffer& buffer, uint64_t window_begin,
                                          uint64_t window_end) {
	constexpr uint64_t StagingSize = 8ull * 1024 * 1024;
	if (m_graphics.readback_queue == nullptr || window_begin >= window_end) {
		return 0;
	}
	// The window's GPU-written bytes, leaving out bytes a download is publishing (that
	// publication writes the newer bytes, and must not be overwritten).
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		RangeSet wanted;
		m_gpu_modified_ranges.ForEachInRange(window_begin, window_end - window_begin,
		                                     [&](uint64_t start, uint64_t end) {
			                                     wanted.Add(start, end - start);
		                                     });
		m_downloading_ranges.ForEachInRange(window_begin, window_end - window_begin,
		                                    [&](uint64_t start, uint64_t end) {
			                                    wanted.Subtract(start, end - start);
		                                    });
		wanted.ForEach([&](uint64_t start, uint64_t end) {
			copies.emplace_back(start - buffer_address, total_size, end - start);
			total_size += Common::AlignUp(end - start, 64);
		});
	}
	if (copies.empty() || total_size > StagingSize) {
		return 0;
	}
	uint32_t index = CopyReadbackSlots;
	for (uint32_t i = 0; i < CopyReadbackSlots; i++) {
		if (m_copy_slots[i].busy.load(std::memory_order_acquire) == 0) {
			index = i;
			break;
		}
	}
	if (index == CopyReadbackSlots) {
		m_copy_busy++;
		return 0;
	}
	KYTY_PROFILER_BLOCK("BufferCache::SnapshotReadback");
	auto& slot   = m_copy_slots[index];
	auto  device = m_graphics.device;
	if (slot.pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.queue_family;
		RequireVulkanSuccess(device.createCommandPool(&pool_info, nullptr, &slot.pool),
		                     "create readback command pool");
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = slot.pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		RequireVulkanSuccess(device.allocateCommandBuffers(&allocate, &slot.command),
		                     "allocate readback command buffer");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(device.createFence(&fence_info, nullptr, &slot.fence),
		                     "create readback fence");
		slot.staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst,
		                                        StagingSize);
	}
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(slot.command.begin(&begin), "begin readback command buffer");
	slot.command.copyBuffer(buffer.Handle(), slot.staging->Handle(),
	                        static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier to_host {};
	to_host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer              = slot.staging->Handle();
	to_host.offset              = 0;
	to_host.size                = total_size;
	slot.command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                             vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &to_host, 0,
	                             nullptr);
	RequireVulkanSuccess(slot.command.end(), "end readback command buffer");
	// Behind the work the GPU has finished (its writes are visible to the copy), not behind the
	// work still running: that work's writes to these bytes land when they land, as on the console.
	const uint64_t                  reached    = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	const vk::Semaphore             master     = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &reached;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &master;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &slot.command;
	RequireVulkanSuccess(m_graphics.readback_queue.submit(1, &submit, slot.fence),
	                     "submit snapshot readback");
	slot.buffer_address = buffer_address;
	slot.window_begin   = window_begin;
	slot.window_end     = window_end;
	slot.total_size     = total_size;
	slot.copies         = std::move(copies);
	slot.snapshot       = true;
	slot.generation     = (slot.generation + 1u) & 0xffffffu;
	m_copy_in_flight.fetch_add(1, std::memory_order_acq_rel);
	slot.busy.store(1, std::memory_order_release);
	static uint64_t snapshots = 0;
	if (++snapshots % 1024 == 1) {
		std::fprintf(stderr, "[read-snapshot] %llu snapshot reads\n",
		             static_cast<unsigned long long>(snapshots));
	}
	return CopyReadbackMarker | (uint64_t {slot.generation} << 8u) | index;
}

void BufferCache::CompleteCopyReadback(uint64_t marker) {
	const auto index      = static_cast<uint32_t>(marker & 0xffu);
	const auto generation = static_cast<uint32_t>((marker >> 8u) & 0xffffffu);
	EXIT_IF((marker & CopyReadbackMarker) == 0 || index >= CopyReadbackSlots);
	auto&           slot = m_copy_slots[index];
	std::lock_guard guard(slot.mutex);
	if (slot.busy.load(std::memory_order_acquire) == 0 || slot.generation != generation) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::CompleteCopyReadback");
	auto device = m_graphics.device;
	RequireVulkanSuccess(device.waitForFences(1, &slot.fence, VK_TRUE, UINT64_MAX),
	                     "wait for readback");
	slot.staging->Invalidate(0, slot.total_size);
	const auto* mapped = slot.staging->Mapped().data();
	if (slot.snapshot) {
		// Only bytes still GPU-written: a download recorded since may have published newer
		// bytes of the rest, which the older snapshot must not overwrite.
		std::shared_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: slot.copies) {
			const auto begin = slot.buffer_address + copy.srcOffset;
			m_gpu_modified_ranges.ForEachInRange(begin, copy.size, [&](uint64_t from, uint64_t to) {
				if (m_downloading_ranges.Intersects(from, to - from)) {
					return;
				}
				Libs::LibKernel::Memory::WriteBacking(from, mapped + copy.dstOffset + (from - begin),
				                                      to - from);
			});
		}
	} else {
		for (const auto& copy: slot.copies) {
			Libs::LibKernel::Memory::WriteBacking(slot.buffer_address + copy.srcOffset,
			                                      mapped + copy.dstOffset, copy.size);
		}
	}
	if (slot.snapshot) {
		// The bytes stay GPU-written; the GPU thread lets reads of the window through.
		std::lock_guard lock(m_snapshot_mutex);
		m_snapshots_ready.emplace_back(Common::AlignDown(slot.window_begin, TRACKER_PAGE_SIZE),
		                               Common::AlignUp(slot.window_end, TRACKER_PAGE_SIZE));
		if (m_snapshots_ready.size() > 64) {
			m_snapshots_ready.erase(m_snapshots_ready.begin());
		}
	} else {
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: slot.copies) {
			m_downloading_ranges.Subtract(slot.buffer_address + copy.srcOffset, copy.size);
		}
		NoteDownloading();
	}
	RequireVulkanSuccess(device.resetFences(1, &slot.fence), "reset readback fence");
	m_copy_in_flight.fetch_sub(1, std::memory_order_acq_rel);
	slot.busy.store(0, std::memory_order_release);
}

// Eager readbacks (KYTY_EAGER_READBACK, default on; =0 off). The game threads fault on ~135
// GPU-written windows a frame in the jungle, mostly the same windows every frame, and each fault
// waits for a download it starts itself (median 2 ms, ~80 s of game-thread time per 20 s,
// 2026-10-09). A window the game threads read back in the last EagerFrames frames is "hot": when a
// submission writes to it, its download is recorded at the end of that submission and published
// when the tick completes. The fault that follows finds the bytes downloaded (or on their way) and
// only lifts the page's protection.
// Opt-in (KYTY_EAGER_READBACK=1): in the save's opening fight the faults did not get cheaper
// (8.8 fps with it, 9.3 without, 2026-10-09); the windows are rewritten after the download.
static bool EagerReadbackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EAGER_READBACK");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

static constexpr uint64_t EagerWindowBits = 22; // 4 MiB, the readback window
static constexpr uint64_t EagerFrames     = 30;

void BufferCache::NoteHotReadback(uint64_t vaddr) {
	if (!EagerReadbackEnabled()) {
		return;
	}
	m_hot_windows[vaddr >> EagerWindowBits] =
	    m_graphics.presented_frames.load(std::memory_order_relaxed);
}

void BufferCache::NoteEagerWrite(uint64_t vaddr, uint64_t size) {
	if (!EagerReadbackEnabled() || m_hot_windows.empty() || size == 0) {
		return;
	}
	const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const auto first = vaddr >> EagerWindowBits;
	const auto last  = (vaddr + size - 1) >> EagerWindowBits;
	if (last - first > 64) {
		return;
	}
	for (auto window = first; window <= last; window++) {
		const auto hot = m_hot_windows.find(window);
		if (hot != m_hot_windows.end() && frame - hot->second <= EagerFrames) {
			m_eager_pending.insert(window);
		}
	}
}

void BufferCache::EagerDownloads() {
	// KYTY_WRITE_BITMAP_SETTLE (default on with the bitmap): when this submission completes, the
	// tracked pages its shaders did not write stop counting as GPU-written and lose their
	// protection, so the game threads read them without a fault (RefineWithWriteBitmap does the
	// same lazily, at the fault, after waiting for the GPU).
	static const bool settle = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP_SETTLE");
		return value == nullptr || value[0] != '0';
	}();
	// Once per presented frame: a settle walks every tracked region the frame wrote (the game's
	// 2.7 GB heap is ~700 of them), which per submission cost the GPU thread 12 ms each time.
	// Each frame also starts a new bitmap epoch (AdvanceWriteBitmapEra).
	if (m_write_bitmap != nullptr) {
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		if (frame != m_bitmap_settle_frame) {
			m_bitmap_settle_frame = frame;
			// The era's bitmap copy settles its pages (ProcessBitmapCopy), exactly.
			(void)settle;
			AdvanceWriteBitmapEra();
		}
	}
	if (m_eager_pending.empty()) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::EagerDownloads");
	bool any = false;
	for (const auto window: m_eager_pending) {
		const auto begin = window << EagerWindowBits;
		const auto end   = begin + (uint64_t {1} << EagerWindowBits);
		auto       it    = m_buffers.upper_bound(begin);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto from   = std::max(begin, buffer.CpuAddress());
			const auto to     = std::min(end, buffer.CpuAddress() + buffer.Size());
			if (from < to) {
				any |= DownloadBufferMemory<true>(buffer, from, to - from);
			}
		}
	}
	m_eager_pending.clear();
	if (any) {
		m_eager_count++;
		m_last_async_download_tick = m_scheduler.CurrentTick();
	}
	if (m_hot_windows.size() > 4096) {
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		std::erase_if(m_hot_windows,
		              [frame](const auto& entry) { return frame - entry.second > EagerFrames; });
	}
}

void BufferCache::WaitCopyReadbacks() {
	if (m_copy_in_flight.load(std::memory_order_acquire) == 0) {
		return;
	}
	for (auto& slot: m_copy_slots) {
		if (slot.busy.load(std::memory_order_acquire) == 0) {
			continue;
		}
		std::lock_guard guard(slot.mutex);
		if (slot.busy.load(std::memory_order_acquire) != 0) {
			RequireVulkanSuccess(
			    m_graphics.device.waitForFences(1, &slot.fence, VK_TRUE, UINT64_MAX),
			    "wait for readback");
		}
	}
}

} // namespace Libs::Graphics
