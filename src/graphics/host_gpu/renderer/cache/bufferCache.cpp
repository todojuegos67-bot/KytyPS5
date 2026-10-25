#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "lane-sync.h"
#include "slow-log.h"

#include "native-buffer-residency.h"
#include "live-census.h"
#include "live-counters.h"
#include "local-platform.h"
#include "live-trace.h"
#include "native-resource-state.h"
#include "async-upload.h"
#include "parallel-pages.h"
#include "readback-queue.h"
#include "sync-stats.h"
#include "fresh-memory.h"
#include "texture-offload.h"
#include "ram-guard.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "gpu_tiler_shaders/lod_stats_pack_spv.h"

extern "C" {
volatile std::atomic<uint32_t> kyty_local_buffer_residency_mode {0};
volatile std::atomic<uint32_t> kyty_local_copy_feedback_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_detach_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_write_readback_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_slots_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_upload_mode {0};
// 1: RangeSet fast paths (rangeSet.h) for the GPU-modified range set.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_range_set_fast_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_reprotect_mode {0};
// Live switch (same-process A/B): ObtainBuffer's memo of repeated shader writes.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_write_memo_mode {1};
// 1: a guest readback whose window meets an image falls back to the request's own pages.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_narrow_mode {0};
// KYTY_READBACK_QUEUE (readback-queue.h): 1: readbacks whose bytes no in-flight GPU work writes
// copy on the transfer queue; 2: also those whose last writer is submitted but still running;
// 3: as 1, and each such copy is repeated on the graphics queue and compared (verification).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_queue_mode {0};
// 1: pack the LOD report on the GPU instead of a CPU wait and copy.
volatile std::atomic<uint32_t> kyty_local_async_lod_stats_mode {0};
// 1: shader constants stream through a host-visible upload ring.
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_stream_upload_mode {0};
extern volatile std::atomic_uint32_t kyty_local_frame_pipeline_mode;
}

#include "common/assert.h"
#include "common/guardedCopy.h"
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

#include <tuple>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "speculation-state.h"

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

// AsyncUpload call (context: the guest address): an image span with unmapped parts (a streamed
// texture pool) into staging, the mapped parts as they are and the rest as zeros.
// KYTY_SPARSE_UPLOAD_PIECES=0: a partly unmapped image span is read by one worker call (TryReadSparseBacking:
// one thread, under the address space's locks) as before. Default: its mapped parts and holes are pushed as
// plain copies, which the worker runs as shared runs on the copy pool (a new area's ~1 GB stalled 100+ ms here).
bool SparseUploadPieces() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_UPLOAD_PIECES");
		return value == nullptr || *value != '0';
	}();
	return enabled;
}

void ReadMappedOrZeroCall(void* context, uint64_t destination, uint64_t size) {
	auto* target = reinterpret_cast<uint8_t*>(destination);
	if (!Libs::LibKernel::Memory::TryReadMappedOrZero(reinterpret_cast<uint64_t>(context), target, size)) {
		std::memset(target, 0, size);
	}
}

// DownloadBufferMemory can process priority operations while waiting. Its
// exact intervals are retired after the entire copy batch, so a reentrant
// query must use the page tracker until that transaction has finished.
thread_local uint32_t native_buffer_download_depth = 0;
struct NativeBufferDownloadScope {
	NativeBufferDownloadScope() { ++native_buffer_download_depth; }
	~NativeBufferDownloadScope() { --native_buffer_download_depth; }
};

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

struct BufferCache::DownloadCopy {
	Buffer*  buffer        = nullptr;
	uint64_t source_offset = 0;
	uint64_t address       = 0;
	uint64_t size          = 0;
};

struct BufferCache::GuestReadback {
	struct Part { uint64_t address, size, offset; };
	std::vector<Part> parts;
	std::vector<GuestRange> pages;
	uint64_t begin = 0, size = 0, tick = 0, packed_size = 0;
	// KYTY_READBACK_QUEUE: the copy-engine value that completes the copy (0: graphics queue
	// at `tick`), and the graphics tick of the last GPU write it waited for. Mode 3 also copies
	// on the graphics queue at `tick` into `verify`.
	uint64_t queue_value = 0, producer_tick = 0;
	Buffer* verify = nullptr;
	size_t slot = 0;
	Buffer* download = nullptr;
	std::atomic<bool> copying {false};
	std::atomic<bool> copied {false};
	// Detach protocol: the reader claims the copy (Pending -> Copying) after the GPU
	// finished; the GPU thread may detach a request whose copy has not started
	// (Pending -> Detached). A detached copy never writes the backing: its pages stay
	// GPU-owned, so the reader faults again and reads the newer GPU state.
	enum State : uint32_t { Pending, Copying, Detached };
	std::atomic<uint32_t> state {Pending};
};

// A value kept between calls on this thread for its storage (a vector's capacity): a call takes it out of `home` and
// puts it back, so a nested call gets an empty one of its own. (A guest readback's vectors and range set made ~10 heap
// allocations: 1% of the GPU thread on the 1-1 walk.)
template <typename T>
class ScratchValue {
public:
	explicit ScratchValue(T& home): m_home(home), m_value(std::move(home)) {}
	~ScratchValue() { m_home = std::move(m_value); }
	ScratchValue(const ScratchValue&)            = delete;
	ScratchValue& operator=(const ScratchValue&) = delete;
	T& operator*() { return m_value; }

private:
	T& m_home;
	T  m_value;
};

// With the frame pipeline, guest reads of GPU-written memory are copied out
// asynchronously; read-only GPU bindings may overlap the copy.
static bool GuestReadbacksEnabled() {
	return kyty_local_frame_pipeline_mode.load(std::memory_order_relaxed) != 0;
}

std::shared_ptr<BufferCache::GuestReadback> BufferCache::BeginGuestReadback(
    uint64_t address, uint64_t size, bool* completed) {
	if (completed) *completed = false;
	constexpr uint64_t WindowSize = 512 * 1024;
	constexpr uint64_t Capacity = 2 * WindowSize;
	if (!m_resources || !GuestRange {address, size}.Valid() || size > WindowSize) {
		LiveCounters::Add(LiveCounters::RbRejectSize);
		return {};
	}
	if (!GuestReadbacksEnabled()) return {};
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto& pending = m_guest_readbacks[slot];
		if (!pending) continue;
		if (pending->copied.load(std::memory_order_acquire)) {
			FinishGuestReadback(slot);
			continue;
		}
		if (std::ranges::any_of(pending->pages, [&](const GuestRange& page) {
			return address >= page.address && address < page.End() && size <= page.End() - address;
		})) {
			return pending;
		}
	}
	// A request spanning several pending page spans cannot share just one ticket.
	DrainGuestReadback(address, size);
	auto& buffer = m_slot_buffers[FindBuffer(address, size)];
	auto begin = std::max(address & ~(WindowSize - 1), buffer.CpuAddress());
	auto end = std::min(std::max(begin + WindowSize, address + size),
	                          buffer.CpuAddress() + buffer.Size());
	if (!m_resources->IsMapped(begin, end - begin) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(begin, end - begin)) {
		// The widening is speculative: a buffer registered over memory the game has since unmapped in part (or maps
		// again elsewhere) fails the window, while the pages around the request may still be one mapping. The window
		// keeps those (page by page outwards from the request's): the guest reads such memory page after page, and a
		// window of the request's pages alone faulted again at each next page (3x the write faults, 5x the readbacks).
		const auto good = [&](uint64_t page_begin, uint64_t page_end) {
			return m_resources->IsMapped(page_begin, page_end - page_begin) &&
			       LibKernel::Memory::IsUniqueGuestBackingRange(page_begin, page_end - page_begin);
		};
		auto first = std::max(address & ~(TRACKER_PAGE_SIZE - 1), buffer.CpuAddress());
		auto last  = std::min((address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1),
		                      buffer.CpuAddress() + buffer.Size());
		if (!good(first, last)) {
			LiveCounters::Add(LiveCounters::RbRejectBacking);
			return {};
		}
		while (first > begin && good(first - std::min(TRACKER_PAGE_SIZE, first - begin), first))
			first -= std::min(TRACKER_PAGE_SIZE, first - begin);
		while (last < end && good(last, last + std::min(TRACKER_PAGE_SIZE, end - last)))
			last += std::min(TRACKER_PAGE_SIZE, end - last);
		begin = first;
		end   = last;
		LiveCounters::Add(LiveCounters::RbWindowNarrowed);
	}
	// Only an image the GPU wrote holds bytes the buffer lacks. Other images over the pages
	// (textures the game streams in, images the buffer is newer than) take no part in the
	// copy, as in the synchronous download, and FindImage drains the copy before any use of
	// them. Refusing them sent 1-byte guest reads of texture pages (~5 a frame) to a
	// render-thread download of ~70 us each.
	if (m_texture_cache.HasGpuWrittenImageOverlap(begin, end - begin)) {
		// KYTY_READBACK_NARROW: the widening is speculative; an image elsewhere in the window
		// leaves the request's own pages, if no image covers them, to an asynchronous copy.
		const bool narrow = kyty_local_readback_narrow_mode.load(std::memory_order_relaxed) != 0;
		if (narrow) {
			begin = std::max(address & ~(TRACKER_PAGE_SIZE - 1), buffer.CpuAddress());
			end   = std::min((address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1),
			               buffer.CpuAddress() + buffer.Size());
		}
		if (!narrow || m_texture_cache.HasGpuWrittenImageOverlap(begin, end - begin)) {
			LiveCounters::Add(LiveCounters::RbRejectImage);
			return {};
		}
	}
	thread_local RangeSet   available_home;
	ScratchValue            available_scratch(available_home);
	auto&                   available_pages = *available_scratch;
	available_pages.Clear();
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin,
	    [&](uint64_t a, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, a, bytes, "guest readback");
	    },
	    [&](uint64_t a, uint64_t bytes) noexcept {
		    available_pages.Add(a, bytes);
	    });
	// Widening is speculative: pages another ticket already covers must not cause
	// this unrelated request to wait or download the same bytes a second time.
	for (const auto& pending: m_guest_readbacks) {
		if (pending) for (const auto& page: pending->pages)
			available_pages.Subtract(page.address, page.size);
	}
	// KYTY_READBACK_QUEUE: the bytes written back are those of these pages (the window's others are at most copied as
	// envelope fill, never written back): they take the transfer queue while no unfinished GPU write overlaps them,
	// whatever other writes in the window do (at Latria the culling's next stage writing the window sent ~4 readbacks
	// a frame of final bytes behind the whole graphics queue, ~1.2 ms each). When one does but the requested pages'
	// bytes are final, the pieces it overlaps stay GPU-owned for a later request.
	thread_local std::vector<std::pair<uint64_t, uint64_t>> runs_home;
	ScratchValue                                            runs_scratch(runs_home);
	auto&                                                   runs = *runs_scratch;
	runs.clear();
	available_pages.ForEach([&](uint64_t a, uint64_t e) { runs.emplace_back(a, e); });
	if (const auto queue_mode = kyty_local_readback_queue_mode.load(std::memory_order_relaxed);
	    (queue_mode == 1 || queue_mode == 3) && m_graphics.readback_queue != nullptr && m_gpu_writes_on &&
	    m_gpu_writes_from != UINT64_MAX && !runs.empty()) {
		auto& master = m_scheduler.GetMasterSemaphore();
		master.Refresh();
		const uint64_t completed  = master.KnownGpuTick();
		const uint64_t page_begin = address & ~(TRACKER_PAGE_SIZE - 1);
		const uint64_t page_end   = (address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
		if (completed >= m_gpu_writes_from && InflightWriteTick(page_begin, page_end, completed, true) == 0) {
			constexpr uint64_t Piece = uint64_t {1} << GpuWriteGranuleBits;
			const auto         drop  = [&](uint64_t from, uint64_t to) {
				if (from < to && InflightWriteTick(from, to, completed, true) != 0) available_pages.Subtract(from, to - from);
			};
			bool narrowed = false;
			for (const auto& [run_begin, run_end]: runs) {
				if (InflightWriteTick(run_begin, run_end, completed, true) == 0) continue;
				narrowed = true;
				// (By write granule, the requested pages apart.)
				for (uint64_t at = run_begin; at < run_end;) {
					const uint64_t next = std::min(run_end, (at / Piece + 1) * Piece);
					if (next <= page_begin || at >= page_end) {
						drop(at, next);
					} else {
						drop(at, std::max(at, page_begin));
						drop(std::min(next, page_end), next);
					}
					at = next;
				}
			}
			if (narrowed) {
				LiveCounters::Add(LiveCounters::RbNarrowed);
				runs.clear();
				available_pages.ForEach([&](uint64_t a, uint64_t e) { runs.emplace_back(a, e); });
			}
		}
	}
	thread_local std::vector<DownloadCopy> copies_home;
	ScratchValue                           copies_scratch(copies_home);
	auto&                                  copies = *copies_scratch;
	copies.clear();
	std::vector<GuestRange> pages;
	pages.reserve(available_pages.Count());
	available_pages.ForEach([&](uint64_t a, uint64_t end) { pages.push_back({a, end - a}); });
	m_gpu_modified_ranges.ForEachIntersection(available_pages, [&](RangeSet::Range range) {
		copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
	});
	if (copies.empty()) {
		if (completed) *completed = true;
		return {};
	}
	// The copies' envelopes in packed regions, one region for envelopes less than 64 KiB apart (the
	// bytes between are copied too and never written back): GPU writes split a window into up to
	// thousands of small ranges, a copy command each otherwise (seconds of driver calls in one frame).
	// The parts place each copy's bytes in the download.
	struct Envelope {
		uint64_t source, size, cursor;
	};
	constexpr uint64_t                 Gap = 64 * 1024;
	thread_local std::vector<Envelope> envelopes_home;
	ScratchValue                       envelopes_scratch(envelopes_home);
	auto&                              envelopes = *envelopes_scratch;
	envelopes.clear();
	std::vector<GuestReadback::Part> parts;
	parts.reserve(copies.size());
	uint64_t packed_size = 0;
	for (const auto& copy: copies) {
		const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
		if (!envelopes.empty() && source_begin <= envelopes.back().source + envelopes.back().size + Gap) {
			auto&      last = envelopes.back();
			const auto end  = std::max(last.source + last.size, source_begin + envelope_size);
			packed_size += AlignDownload(end - last.source) - AlignDownload(last.size);
			last.size = end - last.source;
		} else {
			envelopes.push_back({source_begin, envelope_size, packed_size});
			packed_size += AlignDownload(envelope_size);
		}
		parts.push_back({copy.address, copy.size, envelopes.back().cursor + copy.source_offset - envelopes.back().source});
		if (packed_size > Capacity) {
			LiveCounters::Add(LiveCounters::RbRejectCapacity);
			return {};
		}
	}
	LiveCounters::Add(LiveCounters::ReadbackRegions, envelopes.size());
	// KYTY_READBACK_SLOTS: when every slot is pending, the render thread waits for the
	// oldest copy (a GPU tick). Bursts from many guest threads fill the original eight.
	const size_t slots = kyty_local_readback_slots_mode.load(std::memory_order_relaxed) != 0
	                         ? GuestReadbackSlots
	                         : 8;
	size_t slot = 0;
	while (slot < slots && m_guest_readbacks[slot]) ++slot;
	if (slot == slots) {
		slot = 0;
		for (size_t i = 1; i < slots; ++i)
			if (m_guest_readbacks[i]->tick < m_guest_readbacks[slot]->tick) slot = i;
		LiveCounters::Add(LiveCounters::ReadbackEvictions);
		FinishGuestReadback(slot);
	}
	auto& download = m_guest_downloads[slot];
	if (!download)
		download = std::make_unique<Buffer>(m_graphics, m_scheduler,
		    MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, Capacity);
	// A detached copy-engine request may still write this slot.
	if (m_download_queue_values[slot] != 0) {
		m_readback_queue->Wait(m_download_queue_values[slot]);
		m_download_queue_values[slot] = 0;
	}
	auto request = std::make_shared<GuestReadback>();
	request->begin = begin;
	request->size = end - begin;
	request->pages = std::move(pages);
	request->slot = slot;
	request->download = download.get();
	request->packed_size = packed_size;
	// KYTY_READBACK_QUEUE: bytes whose writers the GPU finished (or, mode 2, has been handed)
	// need not wait behind the rest of the graphics queue.
	if (ReadbackQueueReady(runs, true, request->producer_tick)) {
		// Also after the slot's last graphics-queue copy (it is submitted: that path flushes).
		request->producer_tick = std::max(request->producer_tick, m_download_ticks[slot]);
		thread_local std::vector<ReadbackQueue::Queue::Region> regions_home;
		ScratchValue                                           regions_scratch(regions_home);
		auto&                                                  regions = *regions_scratch;
		regions.clear();
		for (const auto& envelope: envelopes)
			regions.push_back({buffer.Handle(), {envelope.source, envelope.cursor, envelope.size}});
		request->parts = std::move(parts);
		request->queue_value = m_readback_queue->Copy(regions, download->Handle(),
		                                              m_scheduler.GetMasterSemaphore().HandleOf(request->producer_tick),
		                                              request->producer_tick);
		LiveTrace::Event(LiveTrace::RbFast, begin, (end - begin) | request->producer_tick << 32u);
		m_download_queue_values[slot] = request->queue_value;
		request->tick                 = m_scheduler.CurrentTick();
		if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 3) {
			auto& verify = m_verify_downloads[slot];
			if (!verify)
				verify = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
				                                  vk::BufferUsageFlagBits::eTransferDst, Capacity);
			for (const auto& region: regions)
				verify->CopyFrom(m_scheduler.Current(), buffer, region.copy.srcOffset, region.copy.dstOffset,
				                 region.copy.size, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
				                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
				                 vk::AccessFlagBits::eHostRead);
			request->verify = verify.get();
			m_scheduler.Flush();
		}
		if (LiveTrace::WriteTicks())
			LiveTrace::Event(LiveTrace::ReadbackTicks, address, size | request->tick << 32u);
		m_guest_readbacks[slot] = request;
		m_active_guest_readbacks |= 1u << slot;
		MoveWrittenObtainEpoch();
		LiveCounters::Add(LiveCounters::AsyncReadbacks);
		return request;
	}
	// KYTY_ASYNC_COMPUTE: the copy goes on the lane that wrote the bytes last (a culling result: the compute lane,
	// not behind the graphics work queued on the other one).
	const int lane_before = m_scheduler.Lane();
	// (Measured 12-10: on the compute lane the copy took longer, ~6 ms; KYTY_ACQ_READBACK_LANE=1 turns it on.)
	static const bool by_writer = [] {
		const char* value = std::getenv("KYTY_ACQ_READBACK_LANE");
		return value != nullptr && *value == '1';
	}();
	const int lane = by_writer && CommandScheduler::ThreadRecorder() == nullptr ? LaneSync::WriterLane(begin, end - begin) : lane_before;
	if (lane != lane_before) m_scheduler.SetLane(lane);
	LaneSync::NoteRange(buffer.lane_marks, begin, end - begin, false);
	for (const auto& envelope: envelopes)
		download->CopyFrom(m_scheduler.Current(), buffer, envelope.source, envelope.cursor, envelope.size,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	request->parts = std::move(parts);
	request->tick = m_scheduler.CurrentTick();
	m_download_ticks[slot] = request->tick;
	if (LiveTrace::WriteTicks()) LiveTrace::Event(LiveTrace::ReadbackTicks, address, size | request->tick << 32u);
	m_scheduler.Flush();
	if (lane != lane_before) m_scheduler.SetLane(lane_before);
	// Keep GPU ownership and page protection until the caller has copied every byte.
	// Subsequent accesses to this window must retire this request before proceeding.
	m_guest_readbacks[slot] = request;
	m_active_guest_readbacks |= 1u << slot;
	MoveWrittenObtainEpoch();
	LiveCounters::Add(LiveCounters::AsyncReadbacks);
	return request;
}

// Local diagnostic: every GPU write into guest buffers (tracew), noted or not.
static void TraceGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick) {
	if (LiveTrace::WriteTicks()) LiveTrace::Event(LiveTrace::GpuWrite, vaddr, size | tick << 32u);
}

void BufferCache::ResetGpuWrites() {
	m_gpu_writes_base += m_gpu_writes.size(); // (the counters' last numbers stay below it)
	m_gpu_writes.clear();
	m_gpu_writes_head = m_gpu_writes_stamped = 0;
	m_gpu_big_writes.clear();
	m_gpu_write_stamps.clear();
}

// The tick of the write at `index` (UINT64_MAX: not stamped yet).
uint64_t BufferCache::GpuWriteTick(size_t index) const {
	if (index >= m_gpu_writes_stamped) return UINT64_MAX;
	const auto number = m_gpu_writes_base + index;
	const auto stamp  = std::upper_bound(m_gpu_write_stamps.begin(), m_gpu_write_stamps.end(), number,
	                                     [](uint64_t value, const auto& entry) { return value < entry.first; });
	return stamp != m_gpu_write_stamps.end() ? stamp->second : UINT64_MAX;
}

#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(unsigned long, unsigned long, void**,
                                                                                    unsigned long*);
#endif

// KYTY_GPU_WRITE_WATCH=<hex begin>-<hex end> (diagnostics): the GPU writes (written buffer bindings) that cover a part
// of the range, each new (address, size, caller chain) once: WRITE-WATCH lines (the first 64).
static void LogGpuWriteWatch(uint64_t vaddr, uint64_t size) {
#if defined(_WIN32)
	static const auto watch = [] {
		std::pair<uint64_t, uint64_t> range {0, 0};
		if (const char* text = std::getenv("KYTY_GPU_WRITE_WATCH"); text != nullptr) {
			char* end    = nullptr;
			range.first  = std::strtoull(text, &end, 16);
			range.second = end != nullptr && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : 0;
		}
		return range;
	}();
	if (watch.second <= watch.first || vaddr >= watch.second || vaddr + size <= watch.first) return;
	std::array<void*, 12> frames {};
	unsigned long         hash  = 0;
	const auto            count = RtlCaptureStackBackTrace(1, static_cast<unsigned long>(frames.size()), frames.data(), &hash);
	static std::mutex                   mutex;
	static std::unordered_map<uint64_t, uint32_t> seen;
	const uint64_t key = (vaddr * 31 + size) * 1000003 + hash;
	{
		std::lock_guard lock(mutex);
		if (seen.size() >= 64 && !seen.contains(key)) return;
		if (seen[key]++ != 0) return;
	}
	std::string text = fmt::format("WRITE-WATCH addr=0x{:x} size=0x{:x} srt_shader=0x{:x}:", vaddr, size,
	                               Libs::LibKernel::Memory::g_srt_shader_hash);
	for (unsigned i = 0; i < count; i++) {
		const auto address = reinterpret_cast<uint64_t>(frames[i]);
		text += address >= 0x140000000ull && address < 0x150000000ull ? fmt::format(" exe+0x{:x}", address - 0x140000000ull)
		                                                               : fmt::format(" 0x{:x}", address);
	}
	std::printf("%s\n", text.c_str());
	std::fflush(stdout);
#else
	(void)vaddr, (void)size;
#endif
}

// KYTY_READBACK_QUEUE. GPU thread, inside a command: the write is recorded at or after the
// current tick, so its tick is only set at the next point between commands. Every GPU write
// into a guest buffer is noted: shader writes, uploads, image copies and buffer joins.
void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t caller) {
	if (caller == 0 && SyncStats::Enabled()) caller = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	LogGpuWriteWatch(vaddr, size);
	TraceGpuWrite(vaddr, size, m_scheduler.CurrentTick());
	if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 0 || m_graphics.readback_queue == nullptr) {
		if (m_gpu_writes_on) {
			m_gpu_writes_on = false;
			ResetGpuWrites();
		}
		return;
	}
	// Starting, or no point between commands for very long: forget what is not stamped and
	// refuse the fast path until everything recorded so far has completed.
	constexpr size_t Limit = size_t {1} << 20;
	if (!m_gpu_writes_on || m_gpu_writes.size() - m_gpu_writes_head >= Limit) {
		m_gpu_writes_on = true;
		ResetGpuWrites();
		m_gpu_writes_from = UINT64_MAX;
	}
	// Writes not stamped yet all get the same tick at the next point between commands: a range
	// noted again before then (a dispatch run writing the same buffers) adds nothing.
	for (size_t i = m_gpu_writes.size(), last = std::max(m_gpu_writes_stamped, i > 4 ? i - 4 : 0); i > last; --i)
		if (m_gpu_writes[i - 1].begin == vaddr && m_gpu_writes[i - 1].end == vaddr + size) return;
	const bool big = ((vaddr + size - 1) >> GpuWriteGranuleBits) - (vaddr >> GpuWriteGranuleBits) >= GpuWriteSpan;
	m_gpu_writes.push_back({vaddr, vaddr + size, big, SyncStats::Enabled() ? SyncStats::g_shader : 0, caller});
	if (big) {
		m_gpu_big_writes.push_back(m_gpu_writes_base + m_gpu_writes.size() - 1);
	} else {
		const auto number = m_gpu_writes_base + m_gpu_writes.size(); // (1 + the write's)
		for (auto g = vaddr >> GpuWriteGranuleBits; g <= (vaddr + size - 1) >> GpuWriteGranuleBits; ++g)
			m_gpu_write_last[GpuWriteCounter(g)] = number;
	}
}


// KYTY_READBACK_QUEUE. GPU thread: the latest tick of a possibly unfinished GPU write
// overlapping [begin, end); 0 when there is none. Between commands (guest readbacks run there)
// every noted write is recorded by now and gets its tick; inside a command an undated write
// may still be recorded later, so it yields UINT64_MAX.
uint64_t BufferCache::InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed,
                                        bool between_commands) {
	if (between_commands) {
		const uint64_t current = m_scheduler.CurrentTick();
		if (m_gpu_writes_stamped < m_gpu_writes.size()) {
			const auto end = m_gpu_writes_base + m_gpu_writes.size();
			if (!m_gpu_write_stamps.empty() && m_gpu_write_stamps.back().second == current)
				m_gpu_write_stamps.back().first = end;
			else
				m_gpu_write_stamps.emplace_back(end, current);
			m_gpu_writes_stamped = m_gpu_writes.size();
		}
		if (m_gpu_writes_from == UINT64_MAX) m_gpu_writes_from = current;
	}
	// Stamps never decrease: the completed writes are a prefix of the dated ones.
	while (!m_gpu_write_stamps.empty() && m_gpu_write_stamps.front().second <= completed) {
		m_gpu_writes_head = static_cast<size_t>(m_gpu_write_stamps.front().first - m_gpu_writes_base);
		m_gpu_write_stamps.pop_front();
	}
	if (m_gpu_writes_head >= 4096 && m_gpu_writes_head * 2 >= m_gpu_writes.size()) {
		m_gpu_writes.erase(m_gpu_writes.begin(), m_gpu_writes.begin() + static_cast<std::ptrdiff_t>(m_gpu_writes_head));
		m_gpu_writes_stamped -= m_gpu_writes_head;
		m_gpu_writes_base += m_gpu_writes_head;
		m_gpu_writes_head = 0;
	}
	// Ticks grow with the index (undated writes are last): the latest overlapping write has the largest.
	const auto tick_of = [&](size_t index) { return GpuWriteTick(index); };
	if (end <= begin) {
		for (size_t i = m_gpu_writes.size(); i > m_gpu_writes_head; --i) {
			const auto& write = m_gpu_writes[i - 1];
			if (write.begin < end && begin < write.end) return tick_of(i - 1);
		}
		return 0;
	}
	// The big writes (not in m_gpu_write_last), from their own list: the latest overlapping one.
	while (!m_gpu_big_writes.empty() && m_gpu_big_writes.front() < m_gpu_writes_base + m_gpu_writes_head)
		m_gpu_big_writes.pop_front();
	size_t found = 0; // 1 + the index of the latest overlapping write found
	for (auto it = m_gpu_big_writes.rbegin(); it != m_gpu_big_writes.rend(); ++it) {
		const auto  index = static_cast<size_t>(*it - m_gpu_writes_base);
		const auto& write = m_gpu_writes[index];
		if (write.begin < end && begin < write.end) {
			found = index + 1;
			break;
		}
	}
	// The last small write in the range's granules completed (the in-flight writes are the log's tail): none in flight
	// overlaps it. Else none after that one does (a later one would be the last there), and one before the overlapping
	// big write is older than it.
	uint64_t last = 0;
	for (auto g = begin >> GpuWriteGranuleBits; g <= (end - 1) >> GpuWriteGranuleBits; ++g)
		last = std::max(last, m_gpu_write_last[GpuWriteCounter(g)]);
	if (last > m_gpu_writes_base + m_gpu_writes_head) {
		const auto start = static_cast<size_t>(std::min<uint64_t>(last - m_gpu_writes_base, m_gpu_writes.size()));
		for (size_t i = start; i > std::max(m_gpu_writes_head, found); --i) {
			const auto& write = m_gpu_writes[i - 1];
			if (!write.big && write.begin < end && begin < write.end) {
				found = i;
				break;
			}
		}
	}
	return found == 0 ? 0 : tick_of(found - 1);
}

bool BufferCache::ReadbackQueueReady(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                     bool between_commands, uint64_t& wait_tick) {
	const auto mode = kyty_local_readback_queue_mode.load(std::memory_order_relaxed);
	if (mode == 0 || m_graphics.readback_queue == nullptr || !m_gpu_writes_on) return false;
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const uint64_t completed = master.KnownGpuTick();
	uint64_t       producer  = 0;
	for (const auto& [begin, end]: ranges)
		producer = std::max(producer, InflightWriteTick(begin, end, completed, between_commands));
	// Writes before the log started are dated only once everything recorded then completed.
	if (m_gpu_writes_from == UINT64_MAX || completed < m_gpu_writes_from) return false;
	// Mode 2 waits for an unfinished last writer on the transfer queue; mode 3 does the same and
	// compares every copy with a graphics-queue copy (the write log must know every writer).
	if (producer != 0 && (mode < 2 || producer >= m_scheduler.CurrentTick())) return false;
	if (!m_readback_queue) m_readback_queue = std::make_unique<ReadbackQueue::Queue>(m_graphics);
	LiveCounters::Add(producer == 0 ? LiveCounters::RbQueueDone : LiveCounters::RbQueueInflight);
	wait_tick = std::max(producer, completed);
	return true;
}

// KYTY_READBACK_QUEUE=3. Any thread.
void BufferCache::VerifyReadback(uint64_t address, const uint8_t* fast, const uint8_t* reference, uint64_t size,
                                 const char* path, uint64_t waited) {
	LiveCounters::Add(LiveCounters::RbQueueVerified);
	if (std::memcmp(fast, reference, size) == 0) return;
	LiveCounters::Add(LiveCounters::RbQueueMismatch);
	LiveTrace::Event(LiveTrace::RbMismatch, address, size | waited << 32u);
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 40) {
		uint64_t first = 0, differing = 0;
		while (fast[first] == reference[first]) ++first;
		for (uint64_t i = 0; i < size; ++i) differing += fast[i] != reference[i];
		// Every logged write overlapping the range, oldest first (the log is GPU-thread state;
		// a guest thread's report may race, it is diagnostic only).
		std::string writes;
		for (size_t i = m_gpu_writes_head; i < m_gpu_writes.size() && writes.size() < 300; ++i) {
			const auto& write = m_gpu_writes[i];
			if (write.begin < address + size && address < write.end)
				writes += fmt::format(" [{:#x}+{:#x} t{}{}]", write.begin, write.end - write.begin, GpuWriteTick(i),
				                      i < m_gpu_writes_stamped ? "" : "?");
		}
		std::string bytes;
		for (uint64_t i = first & ~uint64_t {3}; i < std::min(size, (first & ~uint64_t {3}) + 16); ++i)
			bytes += fmt::format("{:02x}/{:02x} ", fast[i], reference[i]);
		std::printf("READBACK_QUEUE_MISMATCH %s address=0x%" PRIx64 " size=0x%" PRIx64 " first=0x%" PRIx64
		            " differing=%" PRIu64 " waited=%" PRIu64 " current=%" PRIu64 " known=%" PRIu64
		            " bytes(fast/ref)=%s writes:%s\n",
		            path, address, size, first, differing, waited, m_scheduler.CurrentTick(),
		            m_scheduler.GetMasterSemaphore().KnownGpuTick(), bytes.c_str(), writes.c_str());
		std::fflush(stdout);
	}
}

void BufferCache::CopyGuestReadback(const std::shared_ptr<GuestReadback>& request) {
	if (request->copying.exchange(true, std::memory_order_acq_rel)) {
		while (!request->copied.load(std::memory_order_acquire)) request->copied.wait(false);
		return;
	}
	if (request->queue_value != 0) {
		m_readback_queue->Wait(request->queue_value);
		m_scheduler.WaitPriorityOperations(request->producer_tick);
	} else {
		m_scheduler.GetMasterSemaphore().Wait(request->tick);
		m_scheduler.WaitPriorityOperations(request->tick);
	}
	if (request->verify != nullptr) {
		m_scheduler.GetMasterSemaphore().Wait(request->tick);
		m_scheduler.WaitPriorityOperations(request->tick);
	}
	uint32_t pending = GuestReadback::Pending;
	if (request->state.compare_exchange_strong(pending, GuestReadback::Copying, std::memory_order_acq_rel)) {
		request->download->Invalidate(0, request->packed_size);
		if (request->verify != nullptr) {
			request->verify->Invalidate(0, request->packed_size);
			for (const auto& part: request->parts)
				VerifyReadback(part.address, request->download->Mapped().data() + part.offset,
				               request->verify->Mapped().data() + part.offset, part.size, "guest",
				               request->producer_tick);
		}
		const auto* source = (request->verify != nullptr ? request->verify : request->download)->Mapped().data();
		for (const auto& part: request->parts) WriteBackGpuOwned(part.address, source + part.offset, part.size, "readback");
	}
	request->copied.store(true, std::memory_order_release);
	request->copied.notify_all();
}

void BufferCache::DrainGuestReadback(uint64_t address, uint64_t size, bool gpu_read_only, bool gpu_write) {
	// Read-only bindings use device bytes. Pending copies retain GPU ownership
	// and CPU-clean pages, so synchronization cannot upload over those bytes. (A speculative translation's ranges
	// are obtained again when it is committed.)
	if (!m_active_guest_readbacks || gpu_read_only || Spec::Current() != nullptr) return;
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto& request = m_guest_readbacks[slot];
		if (!request) continue;
		if (address != 0 && (address >= request->begin + request->size ||
		                    (address < request->begin && size <= request->begin - address))) continue;
		if (address != 0 && std::ranges::none_of(request->pages, [&](const GuestRange& page) {
			return address < page.End() && (address >= page.address || size > page.address - address);
		})) continue;
		FinishGuestReadback(slot, gpu_write && kyty_local_readback_detach_mode.load(std::memory_order_relaxed) != 0);
	}
}

void BufferCache::FinishGuestReadback(size_t slot, bool detach) {
	auto request = m_guest_readbacks[slot];
	if (!request) return;
	MoveWrittenObtainEpoch();
	uint32_t pending = GuestReadback::Pending;
	if (detach && !request->copied.load(std::memory_order_acquire) &&
	    request->state.compare_exchange_strong(pending, GuestReadback::Detached, std::memory_order_acq_rel)) {
		// The pages stay GPU-owned (ranges and page marks untouched): the next GPU
		// write keeps them so, and a reader faults again for the newer bytes.
		LiveCounters::Add(LiveCounters::ReadbackDetaches);
		m_guest_readbacks[slot].reset();
		m_active_guest_readbacks &= ~(1u << slot);
		return;
	}
	if (!request->copied.load(std::memory_order_acquire)) {
		LiveCensus::Scope census(LiveCensus::ReadbackWait, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
		                         reinterpret_cast<uint64_t>(__builtin_return_address(1)));
		LiveCensus::WaitScope waiting(LiveCensus::WaitReadback);
		while (!request->copied.load(std::memory_order_acquire)) request->copied.wait(false);
	}
	LiveCounters::Add(LiveCounters::ReadbackParts, request->parts.size());
	LiveCounters::Add(LiveCounters::GpuRangeEntries, m_gpu_modified_ranges.Count());
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.SubtractSorted(request->parts);
		MoveWrittenObtainEpoch();
		++m_gpu_coverage_epoch;
	}
	// The renderer may have dirtied other pages in the widened window since handoff.
	for (const auto& page: request->pages)
		m_memory_tracker.UnmarkRegionAsGpuModified(page.address, page.size);
	m_guest_readbacks[slot].reset();
	m_active_guest_readbacks &= ~(1u << slot);
}

// All metadata and mapped reads belong to the GPU thread. A slot is reused only
// after its copy tick completes; speculative copies never publish CPU ownership.
struct BufferCache::CopyFeedback {
	static constexpr uint64_t SlotSize = 64 * 1024;
	static constexpr size_t SlotCount = 512;
	struct Slot {
		uint64_t address = 0, size = 0, tick = 0, mapping_epoch = 0;
		vk::Buffer owner = nullptr;
	};
	Buffer download;
	std::array<Slot, SlotCount> slots {};
	// Slot of each snapshot by its guest address, sorted by address (at most SlotCount entries:
	// a search of one array, where a map walked cold tree nodes on every GPU write).
	std::vector<std::pair<uint64_t, size_t>> index;
	size_t cursor = 0;
	[[nodiscard]] auto UpperBound(uint64_t address) {
		return std::upper_bound(index.begin(), index.end(), address,
		                        [](uint64_t value, const auto& entry) { return value < entry.first; });
	}
	[[nodiscard]] auto LowerBound(uint64_t address) {
		return std::lower_bound(index.begin(), index.end(), address,
		                        [](const auto& entry, uint64_t value) { return entry.first < value; });
	}
	bool Erase(uint64_t address) {
		const auto it = LowerBound(address);
		if (it == index.end() || it->first != address) return false;
		index.erase(it);
		return true;
	}
	bool Insert(uint64_t address, size_t slot) {
		const auto it = LowerBound(address);
		if (it != index.end() && it->first == address) return false;
		index.insert(it, {address, slot});
		return true;
	}
	// Index entries per 16 MiB granule, hashed into 4096 counters: a range whose granules count
	// none overlaps no entry, and the walk of the index (a cold tree) is skipped. Counters two
	// granules share only cost that walk.
	std::array<uint16_t, 4096> granules {};
	void Count(const Slot& slot, int delta) {
		for (uint64_t g = slot.address >> 24u; g <= (slot.address + slot.size - 1) >> 24u; ++g)
			granules[g & 4095u] = static_cast<uint16_t>(granules[g & 4095u] + delta);
	}
	[[nodiscard]] bool MayOverlap(uint64_t vaddr, uint64_t size) const {
		const uint64_t first = vaddr >> 24u, last = (vaddr + size - 1) >> 24u;
		if (size == 0 || last - first >= granules.size()) return true;
		for (uint64_t g = first; g <= last; ++g)
			if (granules[g & 4095u] != 0) return true;
		return false;
	}
	CopyFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
	    : download(graphics, scheduler, MemoryUsage::Download, 0,
	               vk::BufferUsageFlagBits::eTransferDst, SlotCount * SlotSize) {
		// Each slot has a separate non-coherent atom, even when adjacent slots are in flight.
		const auto atom = graphics.physical_device_properties.limits.nonCoherentAtomSize;
		EXIT_IF(atom == 0 || SlotSize % atom != 0);
	}
};

void BufferCache::InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size)) return;
	auto it = feedback.LowerBound(vaddr);
	if (it != feedback.index.begin()) {
		const auto prior = std::prev(it);
		const auto& slot = feedback.slots[prior->second];
		if (slot.address + slot.size > vaddr) it = prior;
	}
	auto last = it;
	for (; last != feedback.index.end() && last->first < vaddr + size; ++last) {
		feedback.Count(feedback.slots[last->second], -1);
		feedback.slots[last->second].address = 0;
	}
	feedback.index.erase(it, last);
}

void BufferCache::ScheduleCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (kyty_local_copy_feedback_mode.load(std::memory_order_relaxed) == 0) return;
	if (auto* spec = Spec::Current()) return void(spec->feedbacks.push_back({vaddr, vaddr + size})); // (at its commit)
	const auto mapping_epoch = m_resources ? m_resources->MappingEpoch() : 0;
	if (!m_resources || !GuestRange {vaddr, size}.Valid() ||
	    ((vaddr | size) & 3u) != 0 || size > CopyFeedback::SlotSize ||
	    !m_gpu_modified_ranges.Contains(vaddr, size) ||
	    m_texture_cache.HasTrackedDataOverlap(vaddr, size) ||
	    !m_resources->IsMapped(vaddr, size) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(vaddr, size)) {
		return;
	}
	const auto* owner_id = m_page_table.Find(vaddr >> PageTable::kPageBits);
	auto* owner = owner_id && *owner_id ? m_slot_buffers.try_get(*owner_id) : nullptr;
	if (!owner || owner->is_deleted || !owner->IsInBounds(vaddr, size)) {
		return;
	}
	if (!m_copy_feedback) m_copy_feedback = std::make_unique<CopyFeedback>(m_graphics, m_scheduler);
	auto& feedback = *m_copy_feedback;
	auto& master = m_scheduler.GetMasterSemaphore();
	size_t selected = CopyFeedback::SlotCount;
	for (unsigned refresh = 0; refresh < 2 && selected == CopyFeedback::SlotCount; ++refresh) {
		if (refresh) master.Refresh();
		const auto completed = master.KnownGpuTick();
		// Prefer an invalidated slot, then evict an old completed snapshot if necessary.
		for (unsigned evict = 0; evict < 2 && selected == CopyFeedback::SlotCount; ++evict) {
			for (size_t n = 0; n < CopyFeedback::SlotCount; ++n) {
				const auto i = (feedback.cursor + n) % CopyFeedback::SlotCount;
				const auto& slot = feedback.slots[i];
				if (slot.tick <= completed && (evict || slot.address == 0)) {
					selected = i;
					break;
				}
			}
		}
	}
	if (selected == CopyFeedback::SlotCount) {
		return; // Never submit or wait merely to obtain speculative storage.
	}
	if (m_resources->MappingEpoch() != mapping_epoch) return;
	auto& slot = feedback.slots[selected];
	if (slot.address && feedback.Erase(slot.address)) {
		feedback.Count(slot, -1);
	}
	InvalidateCopyFeedback(vaddr, size);
	auto& command = m_scheduler.Current();
	if (command.InCopyRun()) {
		// The copy just made is in the open run: the snapshot follows the run's copies (CommandBuffer::CopyAfterRun).
		feedback.download.written_serial = m_scheduler.CommandSerial();
		command.CopyAfterRun(owner->Handle(), owner->Offset(vaddr), feedback.download.Handle(),
		                     selected * CopyFeedback::SlotSize, size);
	} else {
		feedback.download.CopyFrom(command, *owner, owner->Offset(vaddr), selected * CopyFeedback::SlotSize, size,
		                           vk::AccessFlagBits::eMemoryWrite,
		                           vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostRead,
		                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                           vk::AccessFlagBits::eHostRead);
	}
	slot = {vaddr, size, m_scheduler.CurrentTick(), mapping_epoch, owner->Handle()};
	if (feedback.Insert(vaddr, selected)) feedback.Count(slot, 1);
	MoveWrittenObtainEpoch();
	feedback.cursor = (selected + 1) % CopyFeedback::SlotCount;
}

bool BufferCache::TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (kyty_local_copy_feedback_mode.load(std::memory_order_relaxed) == 0 || !m_copy_feedback || !m_resources)
		return false;
	const auto begin = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto end = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	if (!buffer.IsInBounds(begin, end - begin) || end - begin > CopyFeedback::SlotSize ||
	    m_texture_cache.HasTrackedDataOverlap(begin, end - begin)) {
		return false;
	}
	std::vector<DownloadCopy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "copy feedback");
	    },
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachIntersection(address, bytes, [&](RangeSet::Range range) {
			    copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
		    });
	    });
	if (copies.empty()) return false;
	struct Part { uint64_t address, size, offset; };
	std::vector<Part> parts;
	auto& feedback = *m_copy_feedback;
	const auto mapping_epoch = m_resources->MappingEpoch();
	uint64_t latest_tick = 0;
	for (const auto& copy: copies) {
		auto cursor = copy.address;
		while (cursor < copy.address + copy.size) {
			auto it = feedback.UpperBound(cursor);
			if (it == feedback.index.begin()) return false;
			--it;
			const auto& slot = feedback.slots[it->second];
			if (cursor >= slot.address + slot.size) return false;
			// A snapshot of memory unmapped since (every remap unmaps first) is another mapping's. Mappings elsewhere
			// change nothing in it: the global mapping epoch, which the game's streaming moves every frame or two,
			// dropped every snapshot then (half of the 1-1 walk's frames read their ~60 snapshotted result pages back
			// on the copy engine, ~80 us each, on the thread that submits the main batch).
			if (slot.owner != buffer.Handle() || m_resources->UnmappedSince(slot.mapping_epoch, slot.address, slot.size) ||
			    !m_resources->IsMapped(slot.address, slot.size) ||
			    !LibKernel::Memory::IsUniqueGuestBackingRange(slot.address, slot.size)) {
				return false;
			}
			const auto bytes = std::min(copy.address + copy.size, slot.address + slot.size) - cursor;
			parts.push_back({cursor, bytes, it->second * CopyFeedback::SlotSize + cursor - slot.address});
			latest_tick = std::max(latest_tick, slot.tick);
			cursor += bytes;
		}
	}
	if (!m_scheduler.IsFree(latest_tick)) return false;
	m_scheduler.WaitPriorityOperations(latest_tick);
	if (m_resources->MappingEpoch() != mapping_epoch) return false;
	for (const auto& part: parts) feedback.download.Invalidate(part.offset, part.size);
	for (const auto& part: parts) {
		WriteBackGpuOwned(part.address, feedback.download.Mapped().data() + part.offset, part.size, "feedback");
	}
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.SubtractSorted(copies);
		MoveWrittenObtainEpoch();
		++m_gpu_coverage_epoch;
	}
	// Only complete dirty-page coverage allows dropping protection. CPU-clean bytes
	// are never published from the snapshot. Any later upload/GPU write invalidates it.
	m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
	return true;
}

// GPU thread: a guest access to bytes a copy-feedback snapshot holds (copied after their last GPU
// write, the copy complete) is served from it at once, instead of a transfer and a second command
// for the reader; the whole snapshot then, whose other pages would fault one by one. Not while a
// pending readback covers the pages: its copy would land later.
bool BufferCache::TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return false;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size) || !IsRegionRegistered(vaddr, size)) return false;
	auto first = vaddr, last = vaddr + size;
	if (auto it = feedback.UpperBound(vaddr); it != feedback.index.begin()) {
		const auto& slot = feedback.slots[std::prev(it)->second];
		if (slot.address <= vaddr && vaddr < slot.address + slot.size) {
			first = std::min(first, slot.address);
			last  = std::max(last, slot.address + slot.size);
		}
	}
	const auto begin = first & ~(TRACKER_PAGE_SIZE - 1);
	const auto end   = (last + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (const auto& pending: m_guest_readbacks)
		if (pending && pending->begin < end && begin < pending->begin + pending->size) return false;
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (last - first == size || !buffer.IsInBounds(first, last - first) || !TryReadCopyFeedback(buffer, first, last - first)) {
		if (!TryReadCopyFeedback(buffer, vaddr, size)) return false;
	}
	if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	return true;
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
	// (Its allocation names it: a defragmentation pass's moves are allocations.)
	if (m_graphics.game_buffer_pool != nullptr && m_slot_buffers[id].Allocation() != nullptr)
		vmaSetAllocationUserData(m_graphics.allocator, m_slot_buffers[id].Allocation(),
		                         reinterpret_cast<void*>((uint64_t {id.index} << 32u) | id.generation));
	// (As before below 12 GiB of budget, where GPU buffers ignore it: one in system memory had no video memory at all;
	// and without the driver's figures, which give no room to move one into.)
	if (!m_graphics.small_video_memory && m_graphics.CanReportMemoryUsage() && m_slot_buffers[id].IsInSystemMemory())
		m_spilled.push_back(id);
}

void BufferCache::Unregister(BufferId id) {
	++m_unregister_epoch;
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	DrainGuestReadback(m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size());
	m_sync_buffers_valid = false;
	if (!m_sync_changes_lost) {
		if (m_sync_changes.size() < 256)
			m_sync_changes.push_back({id, m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size(), insert});
		else
			m_sync_changes_lost = true;
	}
	const auto epoch = m_registration_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (m_registration_spans.size() < 64) {
		m_registration_spans.push_back({m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size()});
	} else {
		m_registration_spans_lost = true;
	}
	// Region requests prove their state against the registration epoch of their own region: only the
	// requests of these regions need to be synchronized again.
	for (auto region = m_slot_buffers[id].CpuAddress() / TRACKER_REGION_SIZE,
	          last   = (m_slot_buffers[id].CpuAddress() + m_slot_buffers[id].Size() - 1) / TRACKER_REGION_SIZE;
	     region <= last && region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE; ++region) {
		m_region_registrations[region].store(epoch, std::memory_order_release);
		BdaDirtyRegions::Mark(region);
	}
	LiveCounters::Add(LiveCounters::BufferRegistrations);
	auto& buffer = m_slot_buffers[id];
	if (static const bool log_registrations = std::getenv("KYTY_BUFREG_LOG") != nullptr; log_registrations)
		std::printf("[tsc %llu] BUFREG %c addr=0x%llx size=0x%llx\n", static_cast<unsigned long long>(__rdtsc()),
		            insert ? '+' : '-', static_cast<unsigned long long>(buffer.CpuAddress()),
		            static_cast<unsigned long long>(buffer.Size()));
	if constexpr (!insert) InvalidateCopyFeedback(buffer.CpuAddress(), buffer.Size());
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		m_cache_bytes += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		if (pages.last_exclusive * sizeof(vk::DeviceAddress) > GetBdaPageTableBuffer()->Size()) {
			GrowBdaTable(pages.last_exclusive);
		}
		WriteBdaEntries(pages.first, size_pages, addresses.data());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_cache_bytes -= std::min(m_cache_bytes, buffer.Size());
		m_lru_cache.Free(buffer.lru_id);
		WriteBdaEntries(pages.first, size_pages, nullptr);
		buffer.is_deleted = true;
	}
}

bool BufferCache::IsGpuMapped(uint64_t vaddr, uint64_t size) const noexcept {
	return m_resources == nullptr || m_resources->IsMapped(vaddr, size);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted) {
		return;
	}
	std::erase(m_spilled, id);
	Unregister(id);
	// KYTY_READBACK_QUEUE: a copy on the transfer queue may still read it after the graphics timeline
	// passes (it waits on that timeline only for the bytes' last writer). Freed then, its memory was
	// in use again on GPUs short of VRAM (frequent GC): VK_ERROR_DEVICE_LOST.
	const uint64_t readback = m_readback_queue ? m_readback_queue->Submitted() : 0;
	// (A speculation's packet may still read it: Spec::PacketNow.)
	const auto     packet   = Spec::PacketNow();
	const auto     erase    = [this, id, readback, packet] {
		if (readback != 0 && m_readback_queue) m_readback_queue->Wait(readback);
		m_retired.emplace_back(id, packet);
		EraseRetired();
	};
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation(erase);
	} else {
		erase();
	}
}

std::pair<uint64_t, uint64_t> BufferCache::DownloadEnvelope(const DownloadCopy& copy) {
	if (copy.buffer == nullptr || copy.size == 0 || copy.source_offset > copy.buffer->Size() ||
	    copy.size > copy.buffer->Size() - copy.source_offset) {
		EXIT("BufferCache: invalid download copy\n");
	}
	const auto begin = copy.source_offset & ~uint64_t {3};
	if (copy.source_offset > UINT64_MAX - copy.size ||
	    copy.source_offset + copy.size > UINT64_MAX - 3) {
		EXIT("BufferCache: download copy alignment overflow\n");
	}
	const auto end = (copy.source_offset + copy.size + 3) & ~uint64_t {3};
	if (end > copy.buffer->Size()) {
		EXIT("BufferCache: aligned download copy exceeds its owner\n");
	}
	return {begin, end - begin};
}

// GPU data into guest memory, only on pages the GPU still owns. Every write-back path gives pages back to
// the CPU only after writing them, so a CPU-owned page here holds what the CPU wrote since the copy was
// taken (or nothing the GPU ever wrote): the stale GPU bytes would overwrite newer game data, which shows
// up much later as a broken game structure (a null pointer in a job worker while streaming). Logged.
void BufferCache::WriteBackGpuOwned(uint64_t address, const uint8_t* data, uint64_t size, const char* source) {
	if (m_memory_tracker.IsRegionFullyGpuModified(address, size)) {
		LibKernel::Memory::WriteBacking(address, data, size, source);
		return;
	}
	uint64_t skipped = 0;
	for (uint64_t at = address, end = address + size; at < end;) {
		const auto next = std::min((at & ~(TRACKER_PAGE_SIZE - 1)) + TRACKER_PAGE_SIZE, end);
		if (m_memory_tracker.IsRegionFullyGpuModified(at, next - at)) {
			LibKernel::Memory::WriteBacking(at, data + (at - address), next - at, source);
		} else {
			skipped += next - at;
		}
		at = next;
	}
	LiveCounters::Add(LiveCounters::WriteBackSkips, skipped);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("BufferCache: %s write-back skipped 0x%" PRIx64 " of 0x%" PRIx64 " bytes at 0x%016" PRIx64
		            ": CPU-owned pages\n",
		            source, skipped, size, address);
		std::fflush(stdout);
	}
}

void BufferCache::DownloadBufferMemory(std::span<const DownloadCopy> copies) {
	NativeBufferDownloadScope native_download_scope;
	std::vector<DownloadCopy> batch;
	batch.reserve(copies.size());
	uint64_t                  packed_size = 0;
	auto&                     download    = m_download_buffer;
	const auto flush = [&] {
		const auto [mapped, base_offset] = download.Map(packed_size, DOWNLOAD_ALIGNMENT);
		EXIT_IF(mapped == nullptr);
		uint64_t cursor = 0;
		// KYTY_READBACK_QUEUE: bytes no unfinished GPU write touches copy on the transfer queue
		// instead of draining the graphics queue. This may run inside a command.
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		ranges.reserve(batch.size());
		for (const auto& copy: batch) ranges.emplace_back(copy.address, copy.address + copy.size);
		if (uint64_t wait_tick = 0; ReadbackQueueReady(ranges, false, wait_tick)) {
			std::vector<ReadbackQueue::Queue::Region> regions;
			regions.reserve(batch.size());
			for (const auto& copy: batch) {
				const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
				regions.push_back({copy.buffer->Handle(), {source_begin, base_offset + cursor, envelope_size}});
				cursor += AlignDownload(envelope_size);
			}
			download.Commit();
			if (LiveTrace::WriteTicks())
				for (const auto& copy: batch)
					LiveTrace::Event(LiveTrace::SyncReadback, copy.address, copy.size | m_scheduler.CurrentTick() << 32u);
			m_readback_queue->Wait(m_readback_queue->Copy(regions, download.Handle(),
			                                              m_scheduler.GetMasterSemaphore().HandleOf(wait_tick), wait_tick));
			m_scheduler.WaitPriorityOperations(wait_tick);
			if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 3) {
				// The reference copy lands in a second reservation; the backing gets its bytes.
				const auto [reference, reference_offset] = download.Map(packed_size, DOWNLOAD_ALIGNMENT);
				EXIT_IF(reference == nullptr);
				for (size_t i = 0; i < batch.size(); ++i)
					download.CopyFrom(m_scheduler.Current(), *batch[i].buffer, regions[i].copy.srcOffset,
					                  reference_offset + regions[i].copy.dstOffset - base_offset, regions[i].copy.size,
					                  vk::AccessFlagBits::eMemoryWrite, vk::AccessFlags {},
					                  vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
					                  vk::AccessFlagBits::eHostRead);
				download.Commit();
				const auto completion_tick = m_scheduler.CurrentTick();
				m_scheduler.Finish();
				m_scheduler.WaitPriorityOperations(completion_tick);
				uint64_t at = 0;
				for (const auto& copy: batch) {
					const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
					const auto offset = at + copy.source_offset - source_begin;
					download.Invalidate(base_offset + offset, copy.size);
					download.Invalidate(reference_offset + offset, copy.size);
					VerifyReadback(copy.address, mapped + offset, reference + offset, copy.size, "sync", wait_tick);
					std::memcpy(mapped + offset, reference + offset, copy.size);
					at += AlignDownload(envelope_size);
				}
			}
		} else {
			for (const auto& copy: batch) {
				const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
				download.CopyFrom(m_scheduler.Current(), *copy.buffer, source_begin, base_offset + cursor,
				                  envelope_size, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlags {},
				                  vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
				                  vk::AccessFlagBits::eHostRead);
				cursor += AlignDownload(envelope_size);
			}
			download.Commit();
			const auto completion_tick = m_scheduler.CurrentTick();
			if (LiveTrace::WriteTicks())
				for (const auto& copy: batch)
					LiveTrace::Event(LiveTrace::SyncReadback, copy.address, copy.size | completion_tick << 32u);
			m_scheduler.Finish();
			m_scheduler.WaitPriorityOperations(completion_tick);
		}
		cursor = 0;
		for (const auto& copy: batch) {
			const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
			const auto offset = cursor + copy.source_offset - source_begin;
			download.Invalidate(base_offset + offset, copy.size);
			WriteBackGpuOwned(copy.address, mapped + offset, copy.size, "download");
			cursor += AlignDownload(envelope_size);
		}
		batch.clear();
		packed_size = 0;
	};
	for (auto copy: copies) {
		while (copy.size != 0) {
			const auto available = download.Size() - packed_size;
			const auto prefix    = copy.source_offset & 3u;
			const auto bytes     = std::min(copy.size, available - prefix);
			DownloadCopy part {copy.buffer, copy.source_offset, copy.address, bytes};
			const auto [source_begin, envelope_size] = DownloadEnvelope(part);
			(void)source_begin;
			packed_size += AlignDownload(envelope_size);
			batch.push_back(part);
			copy.source_offset += bytes;
			copy.address += bytes;
			copy.size -= bytes;
			if (packed_size == download.Size()) {
				flush();
			}
		}
	}
	if (!batch.empty()) {
		flush();
	}
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.SubtractSorted(copies);
		MoveWrittenObtainEpoch();
		++m_gpu_coverage_epoch;
	}
}

bool BufferCache::TryReportLodStatsOnGpu(uint64_t address, bool reset) {
    constexpr uint64_t ReportSize = 0x840;
    if (address % 4 || !m_resources || !m_resources->IsMapped(address, ReportSize) ||
        !LibKernel::Memory::IsUniqueGuestBackingRange(address, ReportSize) ||
        m_texture_cache.HasTrackedDataOverlap(address, ReportSize)) return false;
    if (!m_lod_pack_pipeline) {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
        for (uint32_t i = 0; i < bindings.size(); ++i)
            bindings[i] = {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
        vk::DescriptorSetLayoutCreateInfo descriptors {};
        descriptors.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
        descriptors.bindingCount = bindings.size(); descriptors.pBindings = bindings.data();
        RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptors, nullptr, &m_lod_pack_descriptors),
                             "create LOD report descriptors");
        vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 8};
        vk::PipelineLayoutCreateInfo layout {};
        layout.setLayoutCount = 1; layout.pSetLayouts = &m_lod_pack_descriptors;
        layout.pushConstantRangeCount = 1; layout.pPushConstantRanges = &push;
        RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&layout, nullptr, &m_lod_pack_layout),
                             "create LOD report pipeline layout");
        vk::ShaderModuleCreateInfo shader {};
        shader.codeSize = sizeof(LOD_STATS_PACK_SPV); shader.pCode = LOD_STATS_PACK_SPV;
        vk::ShaderModule module;
        RequireVulkanSuccess(m_graphics.device.createShaderModule(&shader, nullptr, &module), "create LOD report shader");
        vk::ComputePipelineCreateInfo pipeline {};
        pipeline.stage.stage = vk::ShaderStageFlagBits::eCompute;
        pipeline.stage.module = module; pipeline.stage.pName = "main";
        // (The subgroup size the game's compute pipelines require here: GraphicContext::InternalComputeSubgroupSize.)
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};
        subgroup.requiredSubgroupSize = m_graphics.InternalComputeSubgroupSize();
        if (subgroup.requiredSubgroupSize != 0) pipeline.stage.pNext = &subgroup;
        pipeline.layout = m_lod_pack_layout;
        RequireVulkanSuccess(m_graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &m_lod_pack_pipeline),
                             "create LOD report pipeline");
        m_graphics.device.destroyShaderModule(module, nullptr);
    }
    // The output remains GPU-owned until the normal checked readback path
    // publishes all bytes. A CPU polling the ready word faults and waits for
    // this dispatch; never publish readiness on the CPU before GPU completion.
    const auto [output, offset] = ObtainBuffer(address, ReportSize, true);
    const auto alignment = m_graphics.GetPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    const auto base = offset & ~(alignment - 1);
    const uint32_t constants[] {uint32_t((offset - base) / 4), reset ? 1u : 0u};
    const vk::DescriptorBufferInfo buffers[] {{m_lod_stats_buffer.Handle(), 0, 256 * 16},
                                             {output->Handle(), base, offset - base + ReportSize}};
    std::array<vk::WriteDescriptorSet, 2> writes {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = vk::DescriptorType::eStorageBuffer; writes[i].pBufferInfo = &buffers[i];
    }
    m_scheduler.EndRendering();
    const auto command = m_scheduler.Current().Handle();
    vk::MemoryBarrier barrier {};
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
                            {}, 1, &barrier, 0, nullptr, 0, nullptr);
    command.bindPipeline(vk::PipelineBindPoint::eCompute, m_lod_pack_pipeline);
    command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_lod_pack_layout, 0, writes);
    command.pushConstants(m_lod_pack_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), constants);
    command.dispatch(4, 1, 1);
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
                            {}, 1, &barrier, 0, nullptr, 0, nullptr);
    return true;
}

void BufferCache::ReportLodStats(void* dst, uint32_t size, bool reset) {
	// Pack the 64-byte completion header and 256 eight-byte LOD counters.
	// The command processor keeps other packet layouts on the existing path.
	EXIT_IF(dst == nullptr || size != 0x840);
	if (kyty_local_async_lod_stats_mode.load(std::memory_order_relaxed) &&
	    TryReportLodStatsOnGpu(reinterpret_cast<uint64_t>(dst), reset))
		return;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = m_lod_stats_buffer.Handle();
	barrier.size = 256 * 16;
	command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	    vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.Finish();
	m_lod_stats_buffer.Invalidate(0, 256 * 16);
	auto* words = reinterpret_cast<uint32_t*>(m_lod_stats_buffer.Mapped().data());
	std::memset(dst, 0, size);
	const uint32_t ready = 1;
	std::memcpy(dst, &ready, sizeof(ready));
	for (uint32_t i = 0; i < 256; ++i) {
		const uint64_t entry = (uint64_t(words[i * 4] & 15u) << 56u) |
		    (uint64_t(std::min(words[i * 4 + 1], 0xffffffu)) << 32u) | words[i * 4 + 1];
		std::memcpy(static_cast<uint8_t*>(dst) + 64 + i * 8, &entry, sizeof(entry));
		if (reset) {
			words[i * 4] = 15;
			words[i * 4 + 1] = words[i * 4 + 2] = words[i * 4 + 3] = 0;
		}
	}
	if (reset) m_lod_stats_buffer.Flush(0, 256 * 16);
}

static BufferCache* g_sync_cache = nullptr;
// Uploads the ring could not take (a temporary buffer each): the cost side of KYTY_STAGING_RING_MB, in the RAM census.
static std::atomic<uint64_t> g_ring_overflow_count {0}, g_ring_overflow_bytes {0};

// The image upload ring: 1 GiB on a PC with 32 GB of RAM or more (512 MiB before). A frame where the game streams
// in an area uploads 0.8-1.4 GB of textures; past the ring's size the render thread waited for the GPU to finish
// the first copies before it could reuse the ring (60-80 ms of gpu_wait in those frames, Boletaria).
static uint64_t StagingRingSize() {
	const uint64_t     ram = LocalPlatform::PhysicalMemory();
	uint64_t           size = ram >= (uint64_t {32} << 30u) ? 1024 * MiB : 512 * MiB;
	if (LocalPlatform::LowMemVram()) size = 256 * MiB; // KYTY_LOWMEM_VRAM=1
	// KYTY_STAGING_RING_MB=<n> (KYTY_MEMCUT): the ring's size, 64-1024. Smaller only waits for the GPU sooner in a
	// streaming frame (or uploads through a temporary buffer past it): the bytes uploaded are the same.
	if (const char* text = std::getenv("KYTY_STAGING_RING_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0)
		size = std::clamp<uint64_t>(std::strtoull(text, nullptr, 10), 64, 1024) * MiB;
	std::printf("[memcut] image upload ring %llu MiB (host-visible)\n", static_cast<unsigned long long>(size / MiB));
	static std::once_flag once;
	std::call_once(once, [] {
		RamGuard::RegisterFigure("upload ring overflows so far (the MiB figure is the number of uploads)",
		                         [] { return g_ring_overflow_count.load(std::memory_order_relaxed) << 20u; });
		RamGuard::RegisterFigure("upload ring overflow bytes so far",
		                         [] { return g_ring_overflow_bytes.load(std::memory_order_relaxed); });
	});
	return size;
}

// KYTY_BDA_SPARSE=1 (where the device has strict sparse residency buffers; off by default while a GPU hang in 2-2's
// culling chain on an 8 GB card is open, the default it had: on with less than 12 GiB of video memory): the BDA page
// table (an entry per 16 KiB page of the 1 TiB guest address space, 512 MiB) gets video memory only for the 64 KiB
// pages of entries a buffer registration wrote (Buffer::MakeResident); the others read zero, as the whole table did.
// The game's memory spans ~100 such pages (128 MiB of guest address space each).
static MemoryUsage PageTableUsage(const GraphicContext& graphics) {
	static const char* const value  = std::getenv("KYTY_BDA_SPARSE");
	const bool               wanted = value != nullptr && value[0] != '\0' && value[0] != '0';
	return wanted && graphics.sparse_residency_buffers ? MemoryUsage::SparseDeviceLocal : MemoryUsage::DeviceLocal;
}

// The guest addresses the BDA page table covers at first: 128 GiB, 64 MiB of video memory instead of the 512 MiB of
// the whole 1 TiB. The PS5's kernel places mappings in its system-managed range by default (to 32 GiB); Demon's Souls
// registers buffers there, by its executable (36 GiB) and at the start of the user range among its PRT pages (66 GiB).
// 1-1 tours with 8 GB (KYTY_SIMULATE_VRAM_MB=8192) back to back: walk 63.6 -> 68.9 fps, spin 62.1 -> 67.7, points under
// 60 fps 44 -> 24 of 84 (the route's middle third 54.3 -> 63.8, 27 of 29 points faster; its last third even: the room
// filled up again). A registration past it grows it (GrowBdaTable; 1-1 starting at 12 GiB grew three times while
// loading, no error). KYTY_BDA_TABLE_GIB=<n>: n GiB at first (tests of the growth). A sparse table is whole (its pages
// get memory as entries are written).
static uint64_t InitialBdaTableBytes(MemoryUsage usage) {
	constexpr uint64_t Whole = BufferCache::BDA_PAGETABLE_SIZE;
	if (usage == MemoryUsage::SparseDeviceLocal) return Whole;
	const char* value   = std::getenv("KYTY_BDA_TABLE_GIB");
	uint64_t    covered = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) << 30u
	                                                           : uint64_t {128} << 30u;
	covered = std::clamp<uint64_t>(covered, uint64_t {1} << 30u, uint64_t {1} << 40u);
	return std::min(Whole, (covered >> BufferCache::CACHING_PAGEBITS) * sizeof(vk::DeviceAddress));
}

void BufferCache::WriteBdaEntries(uint64_t first, uint64_t count, const vk::DeviceAddress* entries) {
	// (The first table's memory may have held other data: zero before its first entry. A sparse one is zeroed as its
	// pages get memory.)
	if (!m_bda_zeroed) {
		m_bda_zeroed = true;
		auto& table  = *m_bda_tables.front();
		if (table.Usage() != MemoryUsage::SparseDeviceLocal) table.Fill(0, table.Size(), 0);
	}
	if (count == 0) return;
	for (const auto& table: m_bda_tables) {
		const uint64_t held = table->Size() / sizeof(vk::DeviceAddress);
		if (first >= held) continue;
		const uint64_t part = std::min(count, held - first);
		table->MakeResident(first * sizeof(vk::DeviceAddress), part * sizeof(vk::DeviceAddress));
		if (entries != nullptr) {
			WriteDataBuffer(*table, first * sizeof(vk::DeviceAddress), entries, part * sizeof(vk::DeviceAddress));
		} else {
			table->Fill(first * sizeof(vk::DeviceAddress), part * sizeof(vk::DeviceAddress), 0);
		}
	}
}

// A registration past the newest table: a table twice its size at least (to the whole 1 TiB), the old entries copied,
// the rest zero, takes its place. The old one stays and is written alike (WriteBdaEntries): a set made with it - a
// native record's, a table set, one written into a command buffer already recorded - reads right entries for what it
// covers and zero past it, as before, never a freed buffer's; native records and table sets made before are made again
// before their next use (their generation: NativeXprValidate, NativeXprDraw, TableValidateSet).
void BufferCache::GrowBdaTable(uint64_t entries) {
	if (!m_bda_zeroed) WriteBdaEntries(0, 0, nullptr);
	auto*    old   = GetBdaPageTableBuffer();
	uint64_t bytes = old->Size();
	while (bytes < entries * sizeof(vk::DeviceAddress) && bytes < BDA_PAGETABLE_SIZE) bytes *= 2;
	bytes      = std::min(bytes, BDA_PAGETABLE_SIZE);
	auto table = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, bytes);
	SetVulkanObjectNameF(m_graphics.device, table->Handle(), "BDA Page Table Buffer");
	table->CopyFrom(m_scheduler.Current(), *old, 0, 0, old->Size());
	table->Fill(old->Size(), bytes - old->Size(), 0);
	m_bda_table.store(table.get(), std::memory_order_release);
	m_bda_tables.push_back(std::move(table));
	s_bda_table_generation.fetch_add(1, std::memory_order_acq_rel);
	std::printf("BDA page table: %llu MiB for guest addresses up to %llu GiB (a buffer at 0x%llx)\n",
	            static_cast<unsigned long long>(bytes >> 20u),
	            static_cast<unsigned long long>((bytes / sizeof(vk::DeviceAddress)) >> (30u - CACHING_PAGEBITS)),
	            static_cast<unsigned long long>((entries - 1) << CACHING_PAGEBITS));
	std::fflush(stdout);
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache,
                         GpuResourceManager* resources)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_lod_stats_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, 256 * 16),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, (RequestRebarPlacement("image upload ring"), StagingRingSize()),
                       AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_host_shader_upload(graphics, scheduler, MemoryUsage::Upload, (RequestRebarPlacement("shader upload ring"), 64 * MiB)),
      m_table_upload(graphics, scheduler, MemoryUsage::Upload, (RequestRebarPlacement("table upload ring"), 32 * MiB),
                     AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_staging_device(graphics, scheduler, MemoryUsage::Stream, 64 * MiB,
                       AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 32 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache), m_resources(resources) {
	RequestRebarPlacement(nullptr);
	if (const auto report = TakeRebarReport(); !report.empty()) {
		std::printf("[rebar] KYTY_REBAR_STAGING: %s\n", report.c_str());
		std::fflush(stdout);
	} else if (const char* value = std::getenv("KYTY_REBAR_STAGING"); value != nullptr && value[0] == '1') {
		std::printf("[rebar] KYTY_REBAR_STAGING: no ring requested\n");
	}
	{
		const auto usage = PageTableUsage(graphics);
		m_bda_tables.push_back(
		    std::make_unique<Buffer>(graphics, scheduler, usage, 0, AllFlags, InitialBdaTableBytes(usage)));
		m_bda_table.store(m_bda_tables.back().get(), std::memory_order_release);
	}
	m_scheduler.SetPrologueHook([this](vk::CommandBuffer command) { FlushPrologueCopies(command); });
	m_gpu_modified_ranges.AllowFastPath();
	g_sync_cache = this;
	LiveCounters::g_sync_report = [] {
		if (g_sync_cache != nullptr) g_sync_cache->PrintSyncState();
	};
	LiveCounters::g_buffer_report = [](const char* path) {
		if (g_sync_cache != nullptr) g_sync_cache->WriteBufferReport(path);
	};
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	std::memset(m_lod_stats_buffer.Mapped().data(), 0, 256 * 16);
	for (uint32_t i = 0; i < 256; ++i) {
		reinterpret_cast<uint32_t*>(m_lod_stats_buffer.Mapped().data())[i * 4] = 15;
	}
	m_lod_stats_buffer.Flush(0, 256 * 16);
	SetVulkanObjectNameF(m_graphics.device, GetBdaPageTableBuffer()->Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	UpdateGcThresholds();
}

// As TextureCache::UpdateGcThresholds: from the current budget, re-read during play.
void BufferCache::UpdateGcThresholds() {
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	LiveCounters::g_buffer_report = nullptr;
	LiveCounters::g_sync_report = nullptr;
	g_sync_cache                = nullptr;
	DrainGuestReadback();
	m_readback_queue.reset();
    m_graphics.device.destroyPipeline(m_lod_pack_pipeline, nullptr);
    m_graphics.device.destroyPipelineLayout(m_lod_pack_layout, nullptr);
    m_graphics.device.destroyDescriptorSetLayout(m_lod_pack_descriptors, nullptr);
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

void BufferCache::WriteBufferReport(const char* path) {
	m_scheduler.Context().GetGpu().SendCommandSync([this, path] {
		FILE* file = std::fopen(path, "wb");
		if (file == nullptr) return;
		std::fputs("id\taddress\tsize\tlru_tick\tgc_tick\tmapped\tgpu_modified\tcpu_modified\tbda_read\tmapped_bytes\tmapped_parts\n",
		           file);
		std::vector<std::pair<uint64_t, uint64_t>> parts;
		m_slot_buffers.ForEach([&](BufferId id, const Buffer& buffer) {
			if (buffer.is_deleted || buffer.CpuAddress() == 0) return;
			const auto address = buffer.CpuAddress(), size = buffer.Size();
			// (How much of it guest mappings cover, in how many pieces: video memory over unmapped guest memory.)
			LibKernel::Memory::MappedParts(address, size, &parts);
			uint64_t mapped_bytes = 0;
			for (const auto& part: parts) mapped_bytes += part.second;
			std::fprintf(file, "%u\t0x%" PRIx64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%d\t%d\t%d\t%d\t%" PRIu64 "\t%zu\n",
			             id.index, address, size, m_lru_cache.TickOf(buffer.lru_id), m_gc_tick,
			             LibKernel::Memory::IsFullyMapped(address, size) ? 1 : 0,
			             m_memory_tracker.IsRegionGpuModified(address, size) ? 1 : 0,
			             m_memory_tracker.IsRegionCpuModified(address, size) ? 1 : 0, buffer.bda_read ? 1 : 0, mapped_bytes,
			             parts.size());
		});
		std::fclose(file);
	});
}

void BufferCache::PrintSyncState() {
	auto&    master = m_scheduler.GetMasterSemaphore();
	uint64_t gpu    = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(master.Handle(), &gpu);
	uint64_t queued = 0, done = 0;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	queued = LocalVulkanRecording::DeferredSubmitsQueued();
	done   = LocalVulkanRecording::DeferredSubmitsDone();
#endif
	std::printf("SYNC master gpu=%" PRIu64 " known=%" PRIu64 " current=%" PRIu64 " deferred queued=%" PRIu64 " done=%" PRIu64
	            " upload pushed=%" PRIu64 "\n",
	            gpu, master.KnownGpuTick(), m_scheduler.CurrentTick(), queued, done, AsyncUpload::SubmitSequence());
	if (m_readback_queue)
		std::printf("SYNC readback queue submitted=%" PRIu64 " counter=%" PRIu64 "\n", m_readback_queue->Submitted(),
		            m_readback_queue->Counter());
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto request = m_guest_readbacks[slot];
		if (!request) continue;
		std::printf("SYNC readback slot=%zu begin=0x%" PRIx64 " size=0x%" PRIx64 " tick=%" PRIu64 " producer=%" PRIu64
		            " queue_value=%" PRIu64 " copying=%d copied=%d state=%u\n",
		            slot, request->begin, request->size, request->tick, request->producer_tick, request->queue_value,
		            request->copying.load() ? 1 : 0, request->copied.load() ? 1 : 0, request->state.load());
	}
	std::fflush(stdout);
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

// The game's large memmoves (DemonsSoulsCopy, >= 256 KiB) prepare their destination this way: at 1-1 a few a frame of
// 4 MiB into memory GPU work had written, each read back first on the GPU thread (~200 us of its time: the copy and the
// wait for it) for bytes the memmove then overwrote.
void BufferCache::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	bool requested = false; // (the range, once for all of its regions)
	m_memory_tracker.InvalidateRegion(vaddr, size, [&] {
		if (std::exchange(requested, true)) return;
		if (GuestGpu::IsGpuThread()) {
			DiscardGpuWrites(vaddr, size);
			return;
		}
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size] { DiscardGpuWrites(vaddr, size); });
	});
}

// As ReadMemoryOnGpu for a write, but only the GPU-written bytes of the first and last page outside [vaddr, vaddr +
// size) are read back (the rest of those pages); the others are dropped: the write replaces them. A GPU write still
// under way to the range races the guest's write on the console too.
void BufferCache::DiscardGpuWrites(uint64_t vaddr, uint64_t size) {
	LiveCensus::Scope census(LiveCensus::SyncDownload, vaddr >> 20u << 20u, 2);
	const auto end   = vaddr + size;
	const auto first = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto last  = (end + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	// (A guest readback of these pages under way writes back onto pages still GPU-owned: finished first.)
	DrainGuestReadback(first, last - first);
	if (!IsRegionRegistered(vaddr, size)) {
		return;
	}
	std::vector<DownloadCopy> copies;
	for (const auto& [begin, stop]: {std::pair {first, vaddr}, std::pair {end, last}}) {
		if (begin >= stop || !m_gpu_modified_ranges.Intersects(begin, stop - begin)) continue;
		// (The buffer of that page alone: one over the whole range would merge every buffer in it.)
		auto& buffer = m_slot_buffers[FindBuffer(begin & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE)];
		m_gpu_modified_ranges.ForEachIntersection(begin, stop - begin, [&](RangeSet::Range range) {
			copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
		});
	}
	if (!copies.empty()) {
		LiveCounters::Add(LiveCounters::SyncDownloads);
		DownloadBufferMemory(copies);
	}
	uint64_t dropped = 0;
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.ForEachIntersection(first, last - first, [&](RangeSet::Range range) { dropped += range.size; });
		m_gpu_modified_ranges.Subtract(first, last - first);
		MoveWrittenObtainEpoch();
		++m_gpu_coverage_epoch;
	}
	LiveCounters::Add(LiveCounters::DiscardedGpuBytes, dropped);
	InvalidateCopyFeedback(first, last - first);
	m_memory_tracker.UnmarkRegionAsGpuModified(first, last - first);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

// GpuResourceManager::UnmapMemory, after InvalidateMemory: the buffers the unmapped range holds whole go. Nothing the
// guest maps reaches them any more (the BDA page table drops them with the registration), and kept they stayed in
// video memory for good (the collector waits for a frame without BDA shaders, which this game never has): at 1-1, 151
// buffers of 2.5 GiB after 12 minutes. Their GPU-written bytes were read back by InvalidateMemory; one still GPU-owned
// stays. A buffer the range holds in part stays (its mapped rest is still the game's).
void BufferCache::DeleteUnmappedBuffers(uint64_t vaddr, uint64_t size) {
	if (size == 0 || UINT64_MAX - vaddr < size) return;
	const auto end = vaddr + size;
	std::vector<BufferId> ids;
	for (auto it = m_buffers.lower_bound(vaddr); it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.is_deleted || buffer.CpuAddress() + buffer.Size() > end ||
		    m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()))
			continue;
		ids.push_back(it->second);
	}
	for (const auto id: ids) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		DeleteBuffer(id);
	}
	LiveCounters::Add(LiveCounters::UnmapBufferDeletes, ids.size());
}

// A buffer the unmapped range holds in part stays for its mapped rest, and joins of buffers next to it grew it on (stream
// leaps over memory the game maps and unmaps as it streams): at 1-1, partly unmapped buffers of 3.6 GiB of the 4.9 GiB
// registered after 9 rounds of the user's route (36 minutes), one of 1.2 GiB, all kept by the collector (GPU-owned
// bytes, read through BDA); with this, 1.1 of 2.3 GiB after 5 rounds (3.1 of 4.3 before), the route's last points 45 ->
// 56 fps in the fifth. Once mapped in no more than half of it, it is cut to its mapped runs: new buffers with its
// bytes copied on the GPU (as a join copies them; the BDA page table points at them), so the bytes copied are at most
// the bytes freed. Its unmapped bytes are nobody's: their GPU-owned ones were read back when unmapped (InvalidateMemory),
// and a buffer with some there left is kept as it is.
void BufferCache::SplitUnmappedBuffers(uint64_t vaddr, uint64_t size) {
	if (m_resources == nullptr || size == 0 || UINT64_MAX - vaddr < size) return;
	const auto            end = vaddr + size;
	std::vector<BufferId> candidates;
	auto                  it = m_buffers.lower_bound(vaddr);
	if (it != m_buffers.begin()) {
		const auto& buffer = m_slot_buffers[std::prev(it)->second];
		if (buffer.CpuAddress() + buffer.Size() > vaddr) candidates.push_back(std::prev(it)->second);
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.CpuAddress() + buffer.Size() > end) candidates.push_back(it->second);
	}
	// Mapped runs closer than the game's 64 KiB pages stay one buffer. (A stream leap no longer joins them again over the
	// gap: ResolveOverlaps. With 2 MiB here a pool mapping 256 KiB of each MiB stayed whole: 1-1 after two world trips,
	// 349 MiB of video memory held for unmapped bytes.)
	constexpr uint64_t Gap = CACHING_PAGESIZE * 4;
	for (const auto id: candidates) {
		const auto& old = m_slot_buffers[id];
		if (old.is_deleted) continue;
		const auto begin = old.CpuAddress(), stop = begin + old.Size();
		RangeSet   mapped;
		m_resources->MappedRanges(begin, stop - begin, mapped);
		mapped.Subtract(vaddr, size); // (not subtracted yet: UnmapMemory does it after this)
		std::vector<std::pair<uint64_t, uint64_t>> pieces;
		mapped.ForEach([&](uint64_t from, uint64_t to) {
			from = std::max(begin, from & ~(CACHING_PAGESIZE - 1));
			to   = std::min(stop, (to + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1));
			if (!pieces.empty() && from <= pieces.back().second + Gap) {
				pieces.back().second = std::max(pieces.back().second, to);
			} else if (from < to) {
				pieces.emplace_back(from, to);
			}
		});
		uint64_t kept = 0;
		for (const auto& [from, to]: pieces) kept += to - from;
		if (kept * 2 > stop - begin) continue;
		// (GPU-owned bytes outside the pieces would be lost with it.)
		bool owned_outside = false;
		uint64_t at = begin;
		for (const auto& [from, to]: pieces) {
			owned_outside = owned_outside || (from > at && m_gpu_modified_ranges.Intersects(at, from - at));
			at            = to;
		}
		if (owned_outside || (stop > at && m_gpu_modified_ranges.Intersects(at, stop - at))) continue;
		// New buffers first (inserting can move the slots: the old one is looked up again after), then the copies, the
		// old one out, the new ones in (CreateBuffer's order).
		std::vector<BufferId> ids;
		for (const auto& [from, to]: pieces) {
			ids.push_back(m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, from,
			                                    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, to - from));
		}
		const auto& source = m_slot_buffers[id];
		for (size_t i = 0; i < ids.size(); ++i) {
			auto&      piece        = m_slot_buffers[ids[i]];
			const auto [from, to]   = pieces[i];
			piece.bda_read          = source.bda_read;
			SetVulkanObjectNameF(m_graphics.device, piece.Handle(), "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", from,
			                     to - from);
			piece.CopyFrom(m_scheduler.Current(), source, from - begin, 0, to - from);
			NoteGpuWrite(from, to - from); // (as JoinOverlap)
		}
		LiveCounters::Add(LiveCounters::UnmapBufferSplits);
		LiveCounters::Add(LiveCounters::UnmapSplitFreedBytes, stop - begin - kept);
		DeleteBuffer(id);
		for (const auto piece: ids) Register(piece);
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// A write takes the same asynchronous copy (KYTY_ASYNC_WRITE_READBACK): the
	// GPU thread records the download instead of draining the GPU; the writer
	// waits and copies, then the written range becomes CPU-owned.
	const bool async_write = is_write && kyty_local_async_write_readback_mode.load(std::memory_order_relaxed) != 0;
	const bool guest       = !GuestGpu::IsGpuThread();
	if (guest) LiveTrace::Event(LiveTrace::GuestReadback, 1, vaddr);
	struct TraceEnd {
		bool     on;
		uint64_t vaddr;
		~TraceEnd() {
			if (on) LiveTrace::Event(LiveTrace::GuestReadback, 0, vaddr);
		}
	} trace_end {guest, vaddr};
	// The thread's whole wait (KYTY_SLOW_LOG_MS): the GPU thread's turn, the GPU work before the copy, the copy.
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW ReadMemory %.1f ms addr=0x%llx size=0x%llx write=%d %s\n", ms, static_cast<unsigned long long>(vaddr),
		            static_cast<unsigned long long>(size), is_write ? 1 : 0, guest ? "guest" : "render");
	});
	if ((!is_write || async_write) && !GuestGpu::IsGpuThread() && GuestReadbacksEnabled()) {
		std::shared_ptr<GuestReadback> request;
		auto& gpu = m_scheduler.Context().GetGpu();
		gpu.SendCommandSync([&] {
			if (TryGuestReadFromFeedback(vaddr, size, is_write)) return;
			bool completed = false;
			request = BeginGuestReadback(vaddr, size, &completed);
			if (!request && (is_write || !completed)) {
				LiveCounters::Add(LiveCounters::SyncReadsGuest);
				ReadMemoryOnGpu(vaddr, size, is_write);
			}
		});
		if (request) {
			CopyGuestReadback(request);
			gpu.SendCommandSync([this, request, vaddr, size, is_write] {
				// An overlapping access may already have retired this request.
				if (m_guest_readbacks[request->slot] == request) FinishGuestReadback(request->slot);
				if (is_write) FinishWriteReadback(vaddr, size);
			});
		}
		return;
	}
	LiveCounters::Add(GuestGpu::IsGpuThread() ? LiveCounters::SyncReadsRender : LiveCounters::SyncReadsGuest);
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		ReadMemoryOnGpu(vaddr, size, is_write);
	});
}

// GPU thread, after an asynchronous readback for a guest write: the tail of
// ReadMemoryOnGpu(is_write). A range the readback left GPU-owned (detached, or
// outside its window) is downloaded as before.
void BufferCache::FinishWriteReadback(uint64_t vaddr, uint64_t size) {
	if (!IsRegionRegistered(vaddr, size)) return;
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		ReadMemoryOnGpu(vaddr, size, true);
		return;
	}
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

void BufferCache::NoteWriterOf(uint64_t begin, uint64_t end) {
	SyncStats::g_writer_shader = SyncStats::g_writer_caller = 0;
	size_t scanned = 0;
	for (size_t i = m_gpu_writes.size(); i > 0 && scanned < 200000; --i, ++scanned) {
		const auto& write = m_gpu_writes[i - 1];
		if (write.begin < end && begin < write.end) {
			SyncStats::g_writer_shader = write.shader;
			SyncStats::g_writer_caller = write.caller != 0 ? write.caller : 1;
			return;
		}
	}
}

// KYTY_SYNC_STACK (diagnostics): the host frames of each caller chain of the render thread's synchronous downloads
// (SYNC-STACK lines, exe offsets for the linker map; 64 chains), printed at its 1st, 2nd, 4th, ... call with the count.
static void LogSyncStack(uint64_t vaddr, uint64_t size, bool is_write) {
#if defined(_WIN32)
	std::array<void*, 14> frames {};
	unsigned long         hash  = 0;
	const auto            count = RtlCaptureStackBackTrace(2, static_cast<unsigned long>(frames.size()), frames.data(), &hash);
	static std::mutex                        mutex;
	static std::unordered_map<unsigned long, uint32_t> seen;
	uint32_t calls = 0;
	{
		std::lock_guard lock(mutex);
		if (seen.size() >= 64 && !seen.contains(hash)) return;
		calls = ++seen[hash];
		if ((calls & (calls - 1)) != 0) return;
	}
	std::string text = fmt::format("SYNC-STACK addr=0x{:x} size=0x{:x} write={} calls={}:", vaddr, size, is_write ? 1 : 0, calls);
	for (unsigned i = 0; i < count; i++) {
		const auto address = reinterpret_cast<uint64_t>(frames[i]);
		text += address >= 0x140000000ull && address < 0x150000000ull ? fmt::format(" exe+0x{:x}", address - 0x140000000ull)
		                                                               : fmt::format(" 0x{:x}", address);
	}
	std::printf("%s\n", text.c_str());
	std::fflush(stdout);
#else
	(void)vaddr, (void)size, (void)is_write;
#endif
}

void BufferCache::ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write) {
	static const bool sync_stack = std::getenv("KYTY_SYNC_STACK") != nullptr;
	if (sync_stack && GuestGpu::IsGpuThread()) LogSyncStack(vaddr, size, is_write);
	// (Cleared after the waits below were noted: other waits name no writer.)
	struct WriterReset {
		~WriterReset() { SyncStats::g_writer_shader = SyncStats::g_writer_caller = 0; }
	} writer_reset;
	if (SyncStats::Enabled() && GuestGpu::IsGpuThread()) NoteWriterOf(vaddr, vaddr + size);
	LiveCensus::Scope census(LiveCensus::SyncDownload, vaddr >> 20u << 20u, is_write);
	LiveCensus::WaitScope waiting(LiveCensus::WaitDownload);
	// Only the guest readbacks of the bytes this download covers must finish first (their copy would write
	// older GPU bytes over the window's, and their marks race with its). The others are the guest threads'
	// business: at Boletaria's dragon path the render thread waited ~4 ms a frame for all of them here,
	// 10 light-table reads a frame each waiting for a readback of some other range to come back from the GPU.
	if (is_write) {
		DrainGuestReadback(vaddr, size);
		if (!IsRegionRegistered(vaddr, size)) return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	LaneSync::NoteRange(buffer.lane_marks, vaddr, size, false);
	if (TryReadCopyFeedback(buffer, vaddr, size)) {
		if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		return;
	}

	// Widen nearby CPU reads so they share one GPU drain.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto         window_begin = std::max(vaddr & ~(WindowSize - 1), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	DrainGuestReadback(window_begin, window_end - window_begin);

	std::vector<DownloadCopy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(
	    window_begin, window_end - window_begin,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "memory invalidation");
	    },
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachIntersection(address, bytes, [&](RangeSet::Range range) {
			    copies.push_back(
			        {&buffer, buffer.Offset(range.address), range.address, range.size});
		    });
	    });
	if (!copies.empty()) {
		LiveCounters::Add(LiveCounters::SyncDownloads);
		DownloadBufferMemory(copies);
		// The enumeration covered whole dirty pages and every exact interval on them.
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	auto* const spec = Spec::Current();
	// Registration changes still drain separately before retiring an owner. (Not a speculation's: its work reads the
	// buffer on the GPU, after its commit.)
	if (spec == nullptr) DrainGuestReadback(vaddr, size, true);
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
			// (A speculation's work holds it: its commit checks nothing registered over the range since.)
			if (spec != nullptr) spec->NoteBufferRange(vaddr, size);
			return *owner;
		}
	}
	// (A speculative translation registers nothing.)
	if (spec != nullptr) return Spec::Refuse("buffer registration"), NULL_BUFFER_ID;
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
	// A leap covers memory the game maps only (its run from the edge): over an unmapped gap it joined the buffers of
	// pieces mapped apart into one that held the gaps in video memory too (1-1 after two world trips: a pool mapping 256
	// KiB of each MiB in one buffer of 369 MiB, 236 MiB of it unmapped; 349 MiB in 29 buffers).
	const auto leap_right = [&](uint64_t from, uint64_t to) {
		if (m_resources == nullptr) return to;
		RangeSet mapped;
		m_resources->MappedRanges(from, to - from, mapped);
		uint64_t reach = from;
		mapped.ForEach([&](uint64_t a, uint64_t b) {
			if (a <= reach) reach = std::max(reach, std::min(b, to));
		});
		return reach & ~(CACHING_PAGESIZE - 1);
	};
	const auto leap_left = [&](uint64_t from, uint64_t to) {
		if (m_resources == nullptr) return from;
		RangeSet mapped;
		m_resources->MappedRanges(from, to - from, mapped);
		uint64_t reach = to;
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		mapped.ForEach([&](uint64_t a, uint64_t b) { runs.emplace_back(a, b); });
		for (auto it = runs.rbegin(); it != runs.rend() && it->second >= reach; ++it) reach = std::min(reach, std::max(it->first, from));
		return (reach + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1);
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
			// Fix the shadPS4 bug that reserves space opposite to the incoming stream's growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end = std::max(end, leap_right(end, end + std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end)));
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin = std::min(begin, leap_left(begin - std::min(StreamLeapSize, begin - minimum), begin));
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
	new_buffer.bda_read |= overlap.bda_read;
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	// The new owner holds these bytes only once the copy ran.
	NoteGpuWrite(overlap.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = (vaddr + size + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1);
	vaddr &= ~(CACHING_PAGESIZE - 1);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

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

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (!is_written && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// CPU cleanliness does not prove that an aliased image is current.
		return is_texel_buffer && SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	LiveCounters::ScopedUs timed(LiveCounters::BufferSyncUs);
	bool fully_gpu_modified = false;
	if (is_written && kyty_local_buffer_residency_mode.load(std::memory_order_relaxed) != 0 &&
	    native_buffer_download_depth == 0 && m_gpu_modified_ranges.Contains(vaddr, size)) {
		// Exact ranges are added only after SynchronizeBuffer marks their
		// pages GPU-owned. Readback subtracts them before clearing page
		// ownership; CPU writes and unmapping take that readback path.
		// A complete byte cover is sufficient, while a hole or partial
		// cover still takes the original page query.
		fully_gpu_modified = true;
	}
	if (is_written && (fully_gpu_modified || m_memory_tracker.IsRegionFullyGpuModified(vaddr, size))) {
		// ObtainBuffer still records the exact write range and invalidates its epoch.
		return false;
	}
	// Only mapped guest memory is read. Pages the guest unmapped stay CPU-dirty (UnmapMemory
	// invalidates them) until memory is mapped there again, and an image's range can reach into
	// them: a texture of a streamed pool whose layers the game released while respawning. The
	// guest mappings tell (a PRT aperture is mapped for the GPU as a whole, its pages one by one);
	// not GpuResourceManager's ranges, whose lock PrepareBdaReadRanges holds around this.
	if (LibKernel::Memory::IsFullyMapped(vaddr, size)) {
		UploadDirtyRanges(buffer, vaddr, size, is_written);
	} else {
		std::vector<std::pair<uint64_t, uint64_t>> parts;
		LibKernel::Memory::MappedParts(vaddr, size, &parts);
		for (const auto& [address, bytes]: parts) UploadDirtyRanges(buffer, address, bytes, is_written);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// The CPU-dirty pages of a mapped range copied into the buffer (the written ones then GPU-owned).
void BufferCache::UploadDirtyRanges(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written) {
	// Reused per thread: the memory tracker refuses a nested upload (CheckNotInUploadCallback).
	thread_local std::vector<vk::BufferCopy> copies;
	copies.clear();
	uint64_t                    total_size = 0;
	uint64_t                    last_dirty = 0;
	vk::Buffer                  source;
	const Buffer*               ring = nullptr; // (the staging ring the copies read)
	// The backing mapping the last run piece was in, from its address to its end: runs mostly lie in one mapping, which
	// nothing unmaps meanwhile (a guest unmap waits for this thread). Found through the backing store's translations, which
	// TryGetBackingPointer then finds it in. (The guest ranges' MappingEnd for every run was 1.6% of the render thread at
	// Latria, more as the game's direct memory gets cut finer at every level load.)
	uint64_t mapping_from = 0, mapping_end = 0;
	const auto mapping_end_of = [&](uint64_t address) {
		if (address < mapping_from || address >= mapping_end) {
			mapping_from = address;
			mapping_end  = LibKernel::Memory::BackingMappingEnd(address);
		}
		return mapping_end;
	};
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    // A run over consecutive mappings (the game maps its heaps a few hundred KiB at a time) in a copy per
		    // mapping: each has a backing pointer of its own, so the upload worker reads them and takes their write
		    // protections (UploadCopies); one copy over a boundary had none, and all of the upload's bytes were read and
		    // its pages protected on this thread (Latria standing: ReapplyProtection 6% of the render thread).
		    for (uint64_t at = 0; at < bytes;) {
			    const uint64_t end   = mapping_end_of(address + at);
			    const uint64_t piece = end > address + at ? std::min(bytes - at, end - (address + at)) : bytes - at;
			    copies.emplace_back(total_size, buffer.Offset(address + at), piece);
			    total_size += piece;
			    at += piece;
		    }
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size, &ring); }, &last_dirty);
	if (total_size != 0) LaneSync::NoteRange(buffer.lane_marks, vaddr, size, true);
	if (source) {
		// The pieces of one run (split at mapping boundaries for the upload worker's reads, which UploadCopies pushed)
		// are adjacent in the staging ring and in the buffer: one copy region each run from here on. (The game's
		// mappings get finer at every level load: 434 -> 586 copies a frame at Stonefang over four rounds.)
		size_t merged = 0;
		for (const auto& copy: copies) {
			if (merged != 0 && copies[merged - 1].srcOffset + copies[merged - 1].size == copy.srcOffset &&
			    copies[merged - 1].dstOffset + copies[merged - 1].size == copy.dstOffset) {
				copies[merged - 1].size += copy.size;
			} else {
				copies[merged++] = copy;
			}
		}
		copies.resize(merged);
		for (const auto& copy: copies) {
			InvalidateCopyFeedback(buffer.CpuAddress() + copy.dstOffset, copy.size);
			NoteGpuWrite(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
		// Pages dirty since before the open command buffer began: copied ahead of all its commands.
		if (const auto prologue = m_scheduler.UploadPrologue(last_dirty, buffer.written_serial)) {
			// By device address, all of the prologue's as one command (FlushPrologueCopies): a copy command per buffer
			// was ~1 us of GPU time each, ~200 in one prologue of the async culling chain at 1-1 (~175 us there).
			if (m_graphics.copy_memory_indirect_enabled && ring != nullptr && ring->HasDeviceAddress() &&
			    buffer.HasDeviceAddress()) {
				const auto from = ring->BufferDeviceAddress(), to = buffer.BufferDeviceAddress();
				uint64_t   bits = from | to;
				for (const auto& copy: copies) bits |= copy.srcOffset | copy.dstOffset | copy.size;
				// (Addresses and sizes of an indirect copy are dword aligned: else copy commands.)
				if ((bits & 3u) == 0) {
					for (const auto& copy: copies)
						m_prologue_copies.push_back({from + copy.srcOffset, to + copy.dstOffset, copy.size});
					return;
				}
			}
			prologue.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()), copies.data());
			return;
		}
		LiveCounters::Add(LiveCounters::InOrderUploads);
		LiveCounters::Add(LiveCounters::InOrderUploadBytes, total_size);
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
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
}

namespace {
struct IndirectCopies {
	VkCommandBuffer command;
	VkDeviceAddress list;
	uint32_t        count;
};
void ReplayIndirectCopies(std::span<const LocalVulkanRecording::Segment> segments,
                          const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& copies = *static_cast<const IndirectCopies*>(segments[0].data);
	dispatch.vkCmdCopyMemoryIndirectNV(copies.command, copies.list, copies.count, sizeof(VkCopyMemoryIndirectCommandNV));
}
} // namespace

// The closing upload prologue's copies (UploadDirtyRanges), from a list in the table ring, as one command recorded in
// order with the prologue's other commands (the recording worker replays it).
void BufferCache::FlushPrologueCopies(vk::CommandBuffer command) {
	if (m_prologue_copies.empty()) return;
	const auto bytes          = m_prologue_copies.size() * sizeof(VkCopyMemoryIndirectCommandNV);
	auto [mapped, offset]     = m_table_upload.Map(bytes, 16);
	EXIT_IF(mapped == nullptr);
	std::memcpy(mapped, m_prologue_copies.data(), bytes);
	m_table_upload.Commit();
	const IndirectCopies copies {command, m_table_upload.BufferDeviceAddress() + offset,
	                             static_cast<uint32_t>(m_prologue_copies.size())};
	m_prologue_copies.clear();
	const LocalVulkanRecording::Segment segments[] {{&copies, sizeof(copies)}};
	if (!LocalVulkanRecording::EnqueueDeferred(ReplayIndirectCopies, segments, false)) {
		LocalVulkanRecording::Drain();
		ReplayIndirectCopies(segments, VULKAN_HPP_DEFAULT_DISPATCHER);
	}
}

// An upload's source bytes (a CPU-dirty range) into host memory, from inside the memory tracker's upload callback: through
// the backing view, whatever the guest range's protection, else through the guest view without faulting. A read of the
// guest view that faulted re-entered the tracker from the callback (another tracker's page protection: the fault
// handler read GPU data back) and ended the emulator (10-12, Tower of Latria, third visit). Bytes not readable at all (no
// mapping) are uploaded as zeros, reported.
// An async upload of `size` backing-view bytes from `source` into staging `target` (guest `vaddr`): the fresh memory
// in it (src/local/fresh-memory.h, bytes nobody wrote) filled with zeros here instead of read by the worker, whose read
// made those pages resident.
static void PushUploadCopy(uint8_t* target, const uint8_t* source, uint64_t size, uint64_t vaddr) {
	thread_local std::vector<std::tuple<uint64_t, uint64_t, bool>> runs;
	runs.clear();
	bool any_fresh = false;
	if (FreshMemory::ForEachBackingRun(source, size,
	                                   [&](uint64_t offset, uint64_t bytes, bool fresh) {
		                                   runs.emplace_back(offset, bytes, fresh);
		                                   any_fresh = any_fresh || fresh;
	                                   }) &&
	    any_fresh) {
		for (const auto& [offset, bytes, fresh]: runs) {
			if (fresh) {
				std::memset(target + offset, 0, bytes);
				FreshMemory::Get()->zero_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
			} else {
				AsyncUpload::Get().Push(target + offset, source + offset, bytes, vaddr + offset);
			}
		}
		return;
	}
	AsyncUpload::Get().Push(target, source, size, vaddr);
}

static void ReadUploadSource(uint8_t* destination, uint64_t address, uint64_t size) {
	if (Libs::LibKernel::Memory::TryReadBackingToHost(address, destination, size)) return;
#if defined(_WIN32)
	if (const auto missed = Common::GuardedCopy(destination, reinterpret_cast<const void*>(address), size); missed != 0) {
		std::memset(destination + (size - missed), 0, missed);
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("BufferCache: an upload's source 0x%llx+0x%llx is unreadable from +0x%llx: zeros uploaded\n",
			            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(size - missed));
	}
#else
	std::memcpy(destination, reinterpret_cast<const void*>(address), size);
#endif
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size, const Buffer** ring_used) {
	if (copies.empty()) {
		return nullptr;
	}
	// Bytes of a texture whose image went are being given back (texture-offload.h, an asynchronous restore): read once
	// they are there, not the discarded pages' zeros (submitted first: the offload worker writes them when the GPU is
	// done).
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		if (!TextureOffload::MustWaitBeforeRead(address, copy.size)) continue;
		if (CommandScheduler::ThreadRecorder() != nullptr) throw CommandScheduler::RecorderRefusal {"texture offload restore"};
		TextureOffload::NoteReadWait(copy.size);
		m_scheduler.Flush();
		TextureOffload::WaitRestores(address, copy.size);
	}
	LiveCounters::Add(LiveCounters::UploadCopies, copies.size());
	LiveCounters::Add(LiveCounters::UploadBytes, total_size);
	if (LiveCounters::g_granules_on.load(std::memory_order_relaxed)) {
		for (const auto& copy: copies) {
			LiveCounters::AddGranule(buffer.CpuAddress() + copy.dstOffset, LiveCounters::GUploadBytes, copy.size);
			LiveCounters::AddGranule(buffer.CpuAddress() + copy.dstOffset, LiveCounters::GUploadCalls, 1);
		}
	}

	// In video memory while that ring has room without waiting (its copies were ~1 ms of GPU work a frame at 1-1,
	// reading system memory across the bus, and the frame's chain waits on them), else in the host ring.
	auto* ring                 = &m_staging_device;
	auto [mapped, base_offset] = m_staging_device.Map(total_size, 4, false);
	if (mapped == nullptr) {
		ring                            = &m_staging_buffer;
		std::tie(mapped, base_offset) = m_staging_buffer.Map(total_size, 4);
	}
	auto& staging_ring = *ring;
	// KYTY_ASYNC_REPROTECT: the pages being copied were marked clean with their write
	// protection deferred. It precedes every copy: queued ahead of them when all go to the
	// worker, applied here otherwise.
	if (auto deferred = MemoryTracker::TakeDeferredProtects(); !deferred.empty()) {
		bool queued = mapped != nullptr && AsyncUpload::Enabled() && staging_ring.IsCoherent();
		for (const auto& copy: copies)
			queued = queued && LibKernel::Memory::TryGetBackingPointer(buffer.CpuAddress() + copy.dstOffset, copy.size);
		for (const auto& item: deferred) {
			if (queued) {
				AsyncUpload::Get().PushProtection(
				    [](void* manager, uint64_t address, uint64_t size) {
					    static_cast<RegionManager*>(manager)->ApplyDeferredProtection(address, size);
				    },
				    item.manager, item.address, item.size, item.address, item.size);
			} else {
				item.manager->ApplyDeferredProtection(item.address, item.size);
			}
		}
	}
	if (mapped != nullptr) {
		// KYTY_ASYNC_UPLOAD: the worker fills coherent staging memory from the backing view
		// before the submission that carries these copies; ranges without one copy here.
		const bool async = AsyncUpload::Enabled() && staging_ring.IsCoherent();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			const auto* source = async ? LibKernel::Memory::TryGetBackingPointer(address, copy.size) : nullptr;
			if (source != nullptr) {
				PushUploadCopy(mapped + copy.srcOffset, source, copy.size, address);
			} else {
				ReadUploadSource(mapped + copy.srcOffset, address, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		if (async) AsyncUpload::Get().Kick();
		staging_ring.Commit();
		if (ring_used != nullptr) *ring_used = &staging_ring;
		return staging_ring.Handle();
	}

	g_ring_overflow_count.fetch_add(1, std::memory_order_relaxed);
	g_ring_overflow_bytes.fetch_add(total_size, std::memory_order_relaxed);
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		ReadUploadSource(temporary->Mapped().data() + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

// A read through a pointer into memory no buffer holds misses the GPU's page table: it reads nothing (zeros) and leaves
// a fault, which gets those pages a buffer only once the submission's faults are processed (FaultManager). A Bink video's
// first decode read its blocks' flags that way, and the frames predicted from it kept the error until the video looped
// (issue #52: the archstone preview's colours wrong for 10 s whenever the decoder's memory was new to the GPU). So the
// start of each range a dispatch or a draw reads through pointers - 64 KiB from its base, its offsets known or not - is
// given a buffer with the guest's bytes before the work is recorded (GpuResourceManager::PrepareBdaReadRanges).
void BufferCache::EnsureBdaReadable(std::span<const GuestRange> ranges) {
	constexpr uint64_t HeadBytes = uint64_t {64} << 10u;
	const auto         held      = [this](uint64_t page) {
        const auto* owner = m_page_table.Find(page);
        return owner != nullptr && *owner;
	};
	for (const auto& range: ranges) {
		const auto last = (range.address + std::min(range.size, HeadBytes) - 1) >> PageTable::kPageBits;
		for (auto page = range.address >> PageTable::kPageBits; page <= last;) {
			if (held(page)) {
				++page;
				continue;
			}
			// (One buffer for a run of such pages.)
			auto end = page + 1;
			while (end <= last && !held(end)) ++end;
			const auto begin = page << PageTable::kPageBits, bytes = (end - page) << PageTable::kPageBits;
			if (IsGpuMapped(begin, bytes) && LibKernel::Memory::IsFullyMapped(begin, bytes)) {
				auto& buffer    = m_slot_buffers[FindBuffer(begin, bytes)];
				buffer.bda_read = true;
				TouchBuffer(buffer);
				(void)SynchronizeBuffer(buffer, begin, bytes, false, false);
			}
			page = end;
		}
	}
}

void BufferCache::EnsureBufferContents(uint64_t vaddr, uint64_t size) {
	if (auto* spec = Spec::Current()) {
		// Made current when the speculative translation is committed (an earlier segment's guest memory writes
		// before it, the open segment's after its work).
		if (spec->open_journal_set.Intersects(vaddr, size)) return Spec::Refuse("memory the processor writes");
		if (m_memory_tracker.IsRegionCpuModified(vaddr, size) || spec->JournalIntersects(vaddr, size))
			spec->NoteCurrent(vaddr, size);
		return;
	}
	const auto id     = FindBuffer(vaddr, size);
	auto&      buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
}

StreamBuffer& BufferCache::GetShaderUploadBuffer() noexcept {
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->shader_upload != nullptr) return *spec->shader_upload;
	return kyty_local_stream_upload_mode.load(std::memory_order_relaxed) != 0 ? m_host_shader_upload
	                                                                        : m_stream_buffer;
}

StreamBuffer& BufferCache::GetTableUploadBuffer() noexcept {
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->table_upload != nullptr) return *spec->table_upload;
	return m_table_upload;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	if (auto* spec = Spec::Current()) {
		// As below: a small range the guest wrote, read from a copy in the (speculation's) ring, as no GPU work wrote
		// it (speculation-state.h) and with no guest memory write of the speculation pending over it.
		if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(vaddr, size) &&
		    m_memory_tracker.IsRegionCpuModified(vaddr, size) && !spec->JournalIntersects(vaddr, size)) {
			auto&      upload    = GetShaderUploadBuffer();
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 4);
			auto [mapped, offset] = upload.Map(size, alignment, false);
			if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(vaddr, mapped, size)) {
				upload.Commit();
				spec->NoteCleanRead(vaddr, size);
				return {&upload, offset};
			}
		}
		// A speculative translation (speculation-state.h): the registered buffer over the range, which its commit
		// obtains again before the work (a range it reads made current, one it writes made GPU-owned).
		const auto* owner  = m_page_table.Find(vaddr >> PageTable::kPageBits);
		auto*       buffer = owner != nullptr && *owner ? m_slot_buffers.try_get(*owner) : nullptr;
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size) ||
		    (is_texel_buffer && m_texture_cache.HasGpuWrittenImageOverlap(vaddr, size)))
			return Spec::Refuse("buffer lookup"), std::pair<Buffer*, uint64_t> {nullptr, 0};
		// (An earlier segment's guest memory writes are made before its work, the open segment's after it, and its
		// copies read their sources then.)
		if (spec->open_journal_set.Intersects(vaddr, size) || (is_written && spec->open_copy_sources.Intersects(vaddr, size)))
			return Spec::Refuse("memory the processor writes"), std::pair<Buffer*, uint64_t> {nullptr, 0};
		if (is_written) spec->NoteWritten(vaddr, size);
		else if (m_memory_tracker.IsRegionCpuModified(vaddr, size) || spec->JournalIntersects(vaddr, size))
			spec->NoteCurrent(vaddr, size);
		if (spec->touched_buffers.empty() || spec->touched_buffers.back() != *owner) spec->touched_buffers.push_back(*owner);
		spec->NoteBufferRange(vaddr, size);
		return {buffer, buffer->Offset(vaddr)};
	}
	DrainGuestReadback(vaddr, size, !is_written && !is_texel_buffer, is_written);
	if (LiveTrace::WriteTicks())
		LiveTrace::Event(LiveTrace::BufferUse, vaddr, size | (is_written ? uint64_t {1} << 63u : 0));
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		auto& upload = GetShaderUploadBuffer();
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 4);
		auto [mapped, offset] = upload.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(vaddr, mapped, size)) {
			upload.Commit();
			return {&upload, offset};
		}
	}

	// A range written again with nothing changed since (m_write_memo): no lookup, synchronization or range Add. (The
	// light-loop chains write the same few shared blocks in every dispatch: ~2100 of ~3200 shader-written ranges a frame
	// at Latria were repeats, each a page-table lookup, a coverage query and a range Add; ~57% of them hit, Stonefang
	// standing 9.11 -> 8.91 ms render in the same process.)
	auto& memo = m_write_memo[((vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull >> 40u)) % m_write_memo.size()];
	if (is_written && memo.vaddr == vaddr && memo.size == size && memo.texel == is_texel_buffer &&
	    memo.registration == RegistrationEpoch() && memo.coverage == m_gpu_coverage_epoch &&
	    kyty_local_write_memo_mode.load(std::memory_order_relaxed) != 0) {
		if (auto* buffer = m_slot_buffers.try_get(memo.id); buffer != nullptr && !buffer->is_deleted && buffer->IsInBounds(vaddr, size)) {
			TouchBuffer(*buffer);
			InvalidateCopyFeedback(vaddr, size);
			NoteGpuWrite(vaddr, size);
			Spec::NoteGpuWrite(vaddr, size);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
			return {buffer, buffer->Offset(vaddr)};
		}
	}

	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size)) {
		id     = FindBuffer(vaddr, size);
		buffer = &m_slot_buffers[id];
	}
	TouchBuffer(*buffer);
	LaneSync::NoteRange(buffer->lane_marks, vaddr, size, is_written);
	(void)SynchronizeBuffer(*buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		// (Not Buffer::written_serial: the write's pages were synchronized above, none of them is a CPU-dirty page an
		// upload prologue copies later.)
		InvalidateCopyFeedback(vaddr, size);
		{
			const std::unique_lock lock(m_gpu_modified_mutex);
			m_gpu_modified_ranges.Add(vaddr, size);
		}
		NoteGpuWrite(vaddr, size);
		Spec::NoteGpuWrite(vaddr, size);
		LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
		LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
		memo = {vaddr, size, RegistrationEpoch(), m_gpu_coverage_epoch, id, is_texel_buffer};
	}
	return {buffer, buffer->Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size, bool* refillable) {
	using Clock = std::chrono::steady_clock;
	Clock::time_point marks[4] {Clock::now()};
	const char*       path = "staging";
	SlowLog::Scope    slow([&](double ms) {
        const auto at = [&](int i) {
            return marks[i] == Clock::time_point {}
		               ? 0.0
		               : std::chrono::duration<double, std::milli>(marks[i] - marks[0]).count();
        };
        std::printf("SLOW ObtainBufferForImage %.1f ms addr=0x%llx size=0x%llx path=%s drain=%.1f gpu_check=%.1f "
		               "map=%.1f\n",
		               ms, static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size), path,
		               at(1), at(2), at(3));
    }, SlowLog::HitchThreshold());
	DrainGuestReadback(vaddr, size);
	marks[1] = Clock::now();
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			path = "buffer";
			ApplyImageProtects();
			TouchBuffer(buffer);
			LaneSync::NoteRange(buffer.lane_marks, vaddr, size, false);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		path = "gpu";
		ApplyImageProtects();
		return ObtainBuffer(vaddr, size, false, false);
	}
	// More than the whole staging ring holds (a texture over a streamed pool expanded to 1.35 GB while
	// playing in Boletaria, and the upload exited): a buffer over the range, which SynchronizeBuffer
	// fills a staging-sized piece at a time (a temporary buffer past that) and only where memory is mapped.
	if (size > m_staging_buffer.Size()) {
		path = "large";
		ApplyImageProtects();
		return ObtainBuffer(vaddr, size, false, false);
	}
	marks[2] = Clock::now();

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	marks[3] = Clock::now();
	// KYTY_ASYNC_UPLOAD=2: the upload worker fills the staging copy too (the image upload only
	// records GPU work on it, and every submission waits for the copies pushed before it).
	if (staging != nullptr && kyty_local_async_upload_mode.load(std::memory_order_relaxed) >= 2 &&
	    m_staging_buffer.IsCoherent()) {
		if (!PushImageStagingCopies(staging, vaddr, size)) path = "sparse";
		AsyncUpload::Get().Kick();
		LiveCounters::Add(LiveCounters::AsyncImageBytes, size);
		m_staging_buffer.Commit();
		if (refillable != nullptr) *refillable = true;
		return {&m_staging_buffer, stage_offset};
	}
	const char* prt_failure = "not-attempted";
	ApplyImageProtects();
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size, &prt_failure))) {
		EXIT("BufferCache: failed to read mapped guest image backing: "
		     "address=0x%016" PRIx64 " size=0x%016" PRIx64
		     " staging=%p staging_capacity=0x%016" PRIx64 " mapped=%u prt_failure=%s\n",
		     vaddr, size, static_cast<void*>(staging), m_staging_buffer.Size(),
		     m_resources != nullptr && m_resources->IsMapped(vaddr, size), prt_failure);
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::ApplyImageProtects() {
	for (const auto& range: m_image_protects) m_image_protect_manager->SyncProtection(range.address, range.size);
	m_image_protects.clear();
}

void BufferCache::QueueImageProtects(std::span<const StagingPiece> copied, bool whole_image) {
	if (m_image_protects.empty()) return;
	const auto page = whole_image ? m_image_protect_manager->GetPageSize() : 1;
	for (const auto& range: m_image_protects) {
		const bool read_whole = std::ranges::any_of(copied, [&](const StagingPiece& piece) {
			return range.address >= piece.vaddr / page * page &&
			       range.address + range.size <= (piece.vaddr + piece.size + page - 1) / page * page;
		});
		if (!read_whole) {
			m_image_protect_manager->SyncProtection(range.address, range.size);
			continue;
		}
		// (The page manager's protection of the pages as their watchers are then: the count of this one may be gone.)
		AsyncUpload::Get().PushProtection(
		    [](void* manager, uint64_t address, uint64_t bytes) {
			    static_cast<PageManager*>(manager)->SyncProtection(address, bytes);
		    },
		    m_image_protect_manager, range.address, range.size, range.address, range.size);
	}
	m_image_protects.clear();
}

bool BufferCache::PushImageStagingCopies(uint8_t* staging, uint64_t vaddr, uint64_t size) {
	const StagingPiece whole {vaddr, size, 0};
	QueueImageProtects({&whole, 1}, true);
	if (const auto* source = Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, size)) {
		PushUploadCopy(staging, source, size, vaddr);
		return true;
	}
	if (Libs::LibKernel::Memory::TryGetBackingPieces(vaddr, size, m_backing_pieces)) {
		// Large images usually span several guest mappings.
		uint64_t offset = 0;
		for (const auto& [piece, bytes]: m_backing_pieces) {
			PushUploadCopy(staging + offset, piece, bytes, vaddr + offset);
			offset += bytes;
		}
		return true;
	}
	// Partly unmapped (a texture pool with released layers): the worker reads the mapped
	// parts; the render thread copied hundreds of MiB here synchronously before.
	if (SparseUploadPieces() && Libs::LibKernel::Memory::TryGetSparseBackingPieces(vaddr, size, m_backing_pieces)) {
		uint64_t offset = 0;
		for (const auto& [piece, bytes]: m_backing_pieces) {
			AsyncUpload::Get().Push(staging + offset, piece, bytes, vaddr + offset);
			offset += bytes;
		}
		LiveCounters::Add(LiveCounters::SparseUploadPieces, m_backing_pieces.size());
		return false;
	}
	AsyncUpload::Get().PushCall(ReadMappedOrZeroCall, reinterpret_cast<void*>(vaddr),
	                            reinterpret_cast<uint64_t>(staging), size, vaddr, size);
	return false;
}

bool BufferCache::RefillImageStaging(const Buffer& ring, uint64_t offset, uint64_t vaddr, uint64_t size) {
	// ObtainBufferForImage's staging path, in its order: not over a cached buffer holding the range or GPU-written pages.
	if (&ring != &m_staging_buffer || !GuestRange {vaddr, size}.Valid() || offset > ring.Size() ||
	    size > ring.Size() - offset || kyty_local_async_upload_mode.load(std::memory_order_relaxed) < 2 ||
	    !ring.IsCoherent())
		return false;
	DrainGuestReadback(vaddr, size);
	if (const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	    owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size))
		return false;
	if (IsRegionGpuModified(vaddr, size)) return false;
	(void)PushImageStagingCopies(ring.Mapped().data() + offset, vaddr, size);
	AsyncUpload::Get().Kick();
	LiveCounters::Add(LiveCounters::AsyncImageBytes, size);
	return true;
}

std::pair<Buffer*, uint64_t> BufferCache::StageImagePieces(const std::vector<StagingPiece>& pieces,
                                                           uint64_t total, const char** why) {
	static std::atomic<uint32_t> reports {0};
	const auto refuse = [&](const char* reason, uint64_t vaddr, uint64_t size, const Buffer* owner) {
		if (why != nullptr) *why = reason;
		if (reports.fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("StageImagePieces refused (%s): piece 0x%llx size 0x%llx, owner 0x%llx size 0x%llx\n", reason,
			            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(owner != nullptr ? owner->CpuAddress() : 0),
			            static_cast<unsigned long long>(owner != nullptr ? owner->Size() : 0));
		return std::pair<Buffer*, uint64_t> {nullptr, 0};
	};
	for (const auto& piece: pieces) {
		if (!GuestRange {piece.vaddr, piece.size}.Valid() || piece.offset > total ||
		    piece.size > total - piece.offset) {
			EXIT("BufferCache: invalid image staging piece\n");
		}
		DrainGuestReadback(piece.vaddr, piece.size);
		// (A cached buffer over a piece holds no newer bytes than guest memory where none of them is GPU-written: the
		// buffer only takes CPU writes from it. Pieces in one refused before, and the whole image was uploaded: the
		// shadow-map pool at 4-2, inside a buffer over a texture pool, 320 MiB at every layer the game streamed.)
		if (IsRegionGpuModified(piece.vaddr, piece.size)) {
			const auto* owner = m_page_table.Find(piece.vaddr >> PageTable::kPageBits);
			return refuse("gpu", piece.vaddr, piece.size, owner != nullptr && *owner ? &m_slot_buffers[*owner] : nullptr);
		}
	}
	auto [staging, stage_offset] = m_staging_buffer.Map(total, 16);
	if (staging == nullptr) {
		return refuse("map", 0, total, nullptr);
	}
	const bool async = kyty_local_async_upload_mode.load(std::memory_order_relaxed) >= 2 &&
	                   m_staging_buffer.IsCoherent();
	// (Parts of the image: a deferred protection goes ahead of the worker's copies only of pages one of them reads whole.)
	if (async) {
		QueueImageProtects(pieces, false);
	} else {
		ApplyImageProtects();
	}
	bool queued = false;
	for (const auto& piece: pieces) {
		auto* target = staging + piece.offset;
		if (async) {
			if (const auto* source = Libs::LibKernel::Memory::TryGetBackingPointer(piece.vaddr, piece.size)) {
				PushUploadCopy(target, source, piece.size, piece.vaddr);
				queued = true;
				continue;
			}
			if (Libs::LibKernel::Memory::TryGetBackingPieces(piece.vaddr, piece.size, m_backing_pieces)) {
				uint64_t offset = 0;
				for (const auto& [source, bytes]: m_backing_pieces) {
					PushUploadCopy(target + offset, source, bytes, piece.vaddr + offset);
					offset += bytes;
				}
				queued = true;
				continue;
			}
		}
		// Part of the span is unmapped (a streamed texture pool layer the game released): those
		// pages read as zeros, the rest as they are. The unmapped pages are marked dirty again
		// when memory is mapped there (TextureCache::MapMemory).
		if (async) {
			if (SparseUploadPieces() &&
			    Libs::LibKernel::Memory::TryGetSparseBackingPieces(piece.vaddr, piece.size, m_backing_pieces)) {
				uint64_t offset = 0;
				for (const auto& [source, bytes]: m_backing_pieces) {
					AsyncUpload::Get().Push(target + offset, source, bytes, piece.vaddr + offset);
					offset += bytes;
				}
				LiveCounters::Add(LiveCounters::SparseUploadPieces, m_backing_pieces.size());
			} else {
				AsyncUpload::Get().PushCall(ReadMappedOrZeroCall, reinterpret_cast<void*>(piece.vaddr),
				                            reinterpret_cast<uint64_t>(target), piece.size, piece.vaddr, piece.size);
			}
			queued = true;
			continue;
		}
		ReadMappedOrZeroCall(reinterpret_cast<void*>(piece.vaddr), reinterpret_cast<uint64_t>(target),
		                     piece.size);
	}
	if (queued) {
		AsyncUpload::Get().Kick();
		LiveCounters::Add(LiveCounters::AsyncImageBytes, total);
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
	(void)m_texture_cache.ClearMeta(vaddr, value);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		Spec::NoteHostWrite(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, "fill");
	const auto id          = FindBuffer(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true, id);
	EXIT_IF(dst == nullptr);
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
	auto* const spec = Spec::Current();
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		if (spec != nullptr) {
			// A speculative translation's copy is made at its commit, in order with its other guest memory writes,
			// between ranges no GPU work wrote (speculation-state.h: work that writes them stops the commit first).
			spec->NoteCpuCopy(src_vaddr, size);
			spec->NoteCpuCopy(dst_vaddr, size);
			spec->Journal(dst_vaddr, src_vaddr, static_cast<uint32_t>(size), true);
			return;
		}
		CopyGuestMemory(dst_vaddr, src_vaddr, size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, "copy");
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	if (spec != nullptr && spec->refused != nullptr) return; // (a buffer it would register)
	EXIT_IF(src == nullptr || dst == nullptr);
	if (src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size) {
		EXIT("BufferCache: resolved Vulkan copy ranges overlap\n");
	}
	dst->CopyInRun(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::PrepareAliasesForCpuWrite(uint64_t vaddr, uint64_t size) {
	// (Not in a speculative translation: the aliases' state changes here and now.)
	if (Spec::Current() != nullptr) return false;
	thread_local std::vector<std::pair<uint64_t, uint64_t>> aliases;
	if (!LibKernel::Memory::GuestBackingAliases(vaddr, size, &aliases)) return false;
	for (const auto& [alias, bytes]: aliases)
		if (IsRegionGpuModified(alias, bytes) || m_texture_cache.IsRegionGpuModified(alias, bytes)) return false;
	for (const auto& [alias, bytes]: aliases) {
		DrainGuestReadback(alias, bytes);
		m_texture_cache.InvalidateMemory(alias, bytes);
		InvalidateMemory(alias, bytes);
	}
	return true;
}

// Copies from this size on compare their pages on helper threads too (ParallelPages): below it, waking them costs about
// what they would take off this thread.
static constexpr uint64_t ParallelCompareBytes = uint64_t {2} << 20u;
static constexpr uint64_t ParallelComparePages = 64; // (a chunk: 256 KiB)

void BufferCache::CopyGuestMemory(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size) {
	// The game's linear copies mostly rewrite what the destination already holds (95% of the bytes in the fixed
	// scene): only the destination pages whose bytes differ are written, so the others stay clean (no upload of their
	// bytes to the GPU again). A run of them with a write-protected page (uploaded since it was last written) is marked
	// CPU-dirty with its protection kept and written through the backing view: no write fault a page (~190 a frame at
	// 1-1) and no protection change, here or when it is uploaded again (the game rewrites these pages every frame).
	uint64_t   run = 0, run_end = 0;
	const auto write = [&] {
		if (run_end == run) return;
		const auto  address = dst_vaddr + run, bytes = run_end - run;
		const auto* from    = reinterpret_cast<const void*>(src_vaddr + run);
		if (!m_memory_tracker.IsRegionFullyCpuModified(address, bytes)) {
			// (As the write faults would: the images over the pages first.)
			m_texture_cache.InvalidateMemory(address, bytes);
			// (The pages are not GPU-modified, CopyBuffer checked: no GPU result is written back there.)
			if (m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(address, bytes) &&
			    (Libs::LibKernel::Memory::TryWriteCpuBacking(address, from, bytes) ||
			     Libs::LibKernel::Memory::TryWriteBacking(address, from, bytes))) {
				Spec::NoteHostWrite(address, bytes);
				return;
			}
			// (Memory the backing view does not hold: written through the guest's view, unprotected first.)
			InvalidateMemory(address, bytes);
		}
		// (Pages CPU-dirty already change no state: their epoch moves, a copy of their bytes taken meanwhile is
		// stale. A linear copy is a dispatch, no point the observation epoch moves at.)
		m_memory_tracker.NoteHostWrite(address, bytes);
		std::memcpy(reinterpret_cast<void*>(address), from, bytes);
		Spec::NoteHostWrite(address, bytes);
	};
	// Which destination pages differ: compared on helper threads too when the copy is large (the 4 MiB copy the culling
	// chain makes every frame, on the frame's critical path: 0.39 ms on this thread alone, 0.24 ms shared at Latria).
	const auto page_of = [&](uint64_t at) { return (dst_vaddr + at) / TRACKER_PAGE_SIZE - dst_vaddr / TRACKER_PAGE_SIZE; };
	thread_local std::vector<uint8_t> differs;
	// (Helpers only over pages no read faults on: a helper's fault waits for this thread's
	// readback, which waits for the helpers. CopyBuffer left out GPU-modified memory, so these are
	// the pages the texture and file offloads hold; this thread compares such a copy alone, its own
	// faults are served here. Only this thread adds holds.)
	const bool parallel = size >= ParallelCompareBytes &&
	                      !Libs::LibKernel::Memory::ReadFaults(dst_vaddr, size) &&
	                      !Libs::LibKernel::Memory::ReadFaults(src_vaddr, size);
	if (parallel) {
		struct Compare {
			uint64_t dst, src, size;
			uint8_t* differs;
		} compare {dst_vaddr, src_vaddr, size, nullptr};
		differs.assign(page_of(size - 1) + 1, 0);
		compare.differs = differs.data();
		const ParallelPages::Job pages = [](void* context, uint64_t first, uint64_t last) {
			const auto& job = *static_cast<const Compare*>(context);
			for (auto page = first; page < last; ++page) {
				const auto begin = page == 0 ? 0 : (job.dst / TRACKER_PAGE_SIZE + page) * TRACKER_PAGE_SIZE - job.dst;
				const auto end   = std::min((job.dst / TRACKER_PAGE_SIZE + page + 1) * TRACKER_PAGE_SIZE - job.dst, job.size);
				job.differs[page] = std::memcmp(reinterpret_cast<const void*>(job.dst + begin),
				                                reinterpret_cast<const void*>(job.src + begin), end - begin) != 0;
			}
		};
		ParallelPages::For(differs.size(), ParallelComparePages, pages, &compare);
	}
	for (uint64_t at = 0; at < size;) {
		const auto bytes = std::min(TRACKER_PAGE_SIZE - (dst_vaddr + at) % TRACKER_PAGE_SIZE, size - at);
		if (parallel ? differs[page_of(at)] != 0
		             : std::memcmp(reinterpret_cast<const void*>(dst_vaddr + at),
		                           reinterpret_cast<const void*>(src_vaddr + at), bytes) != 0) {
			if (at != run_end) {
				write();
				run = at;
			}
			run_end = at + bytes;
		}
		at += bytes;
	}
	write();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// A short range asks the page table: buffers cover whole caching pages (CreateBuffer aligns them, a split cuts at
	// mapping bounds) and registration fills their pages. (The map's search was 0.5% of the GPU thread at the 1-1
	// ramparts: every guest readback asks.)
	if (const auto first = vaddr >> PageTable::kPageBits, last = (vaddr + size - 1) >> PageTable::kPageBits;
	    last - first < 4) {
		for (auto page = first; page <= last; ++page)
			if (const auto* owner = m_page_table.Find(page); owner != nullptr && *owner) return true;
		return false;
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
	// (A speculative translation's own writes count.)
	const auto* spec = Spec::Current();
	return m_memory_tracker.IsRegionGpuModified(vaddr, size) || (spec != nullptr && spec->WrittenIntersects(vaddr, size));
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// (Another thread's question, a speculation's: under the lock the GPU thread changes them under.)
	if (!GuestGpu::IsGpuThread()) {
		const std::shared_lock lock(m_gpu_modified_mutex);
		return m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	// (Not a range a speculative translation's commit makes current.)
	const auto* spec = Spec::Current();
	return m_memory_tracker.IsRegionCpuModified(vaddr, size) && (spec == nullptr || !spec->current_set.Contains(vaddr, size));
}

// Per flip, whatever the memory use: the CPU-written buffers no binding used for StaleTime go (at most MaxRetirements a
// flip); with the video memory in use at the critical level or over it, after PressureTime and up to PressureRetirements
// (only then: ten seconds also retire what a turned camera left behind, re-registered when seen again at Latria).
// Idleness is counted in time, not flips: the 1800 flips it was are 2 minutes at an 8 GB laptop's 14 fps, where
// a 5-minute 1-1 tour left 0.5 GB of such buffers idle while the video memory in use went past the budget (8.0 GB of
// 7.25). Same exe, the 1800-flip rule against this, twice each: 14.8/17.3 -> 23.5/23.9 fps, render 45 -> 38.6 ms, frames
// over 100 ms 137/137 -> 25/21, region syncs 3650 -> 1900 a frame, buffers 2.9 -> 2.1 GB, bda_fault_pages 3.3/4.6 ->
// 3.0/2.0. At high frame rates the StaleTicks flips can come first. Never one used in the last MinTicks flips (a long
// frame is no idle time). The video memory in use is read every 30 flips.
// The collector below waits for a frame without BDA shaders, which this game never has, so every buffer it ever
// registered stayed; those inside the BDA shaders' unbounded read windows had their CPU-written pages uploaded and
// write-protected again at every BDA preparation (the game reuses that memory every frame). At 1-1, three rounds of the
// user's route (12 minutes): 3.8K -> 11.5K buffers, region syncs 1.4K -> 2.5K a frame, uploads 22 -> 32 MB a frame,
// 60.5 -> 48 fps; with this, ~5K buffers and 63 -> 60 fps. Passed over: GPU-owned bytes (their download is the
// collector's) and buffers BDA shaders read with no buffer registered (Buffer::bda_read: those reads touch no LRU
// entry, and one after the buffer went would read zeros once) - none faulted back over that route (bda_fault_pages ~0).
void BufferCache::RetireStaleBuffers(uint64_t tick) {
	constexpr auto     StaleTime           = std::chrono::seconds(30);
	constexpr auto     PressureTime        = std::chrono::seconds(10);
	constexpr size_t   MaxRetirements      = 64;
	constexpr size_t   PressureRetirements = 256;
	constexpr uint64_t MinTicks            = 120;
	constexpr uint64_t StaleTicks          = 1800;
	const auto         now                 = std::chrono::steady_clock::now();
	m_tick_times[tick % m_tick_times.size()] = now.time_since_epoch().count();
	if (tick % 30 == 0) {
		uint64_t video = 0, budget = 0;
		m_retire_pressure = m_graphics.QueryLocalVideoMemory(&video, &budget) && video >= m_critical_gc_memory;
	}
	if (tick <= MinTicks) return;
	const bool pressure = m_retire_pressure;
	const auto horizon  = (now - (pressure ? PressureTime : StaleTime)).time_since_epoch().count();
	// The newest tick that began by the horizon among the ticks the ring holds (their times only grow); when even the
	// oldest held began after it, the ticks older than the ring.
	uint64_t first = tick - std::min<uint64_t>(tick, m_tick_times.size() - 1), last = tick;
	if (m_tick_times[first % m_tick_times.size()] > horizon) {
		last = first;
	} else {
		while (first < last) {
			const uint64_t middle = first + (last - first + 1) / 2;
			if (m_tick_times[middle % m_tick_times.size()] <= horizon) first = middle;
			else last = middle - 1;
		}
	}
	// (Never later than the StaleTicks rule it replaced: at 120 fps, 1800 flips are 15 s.)
	const uint64_t below = std::min(std::max(last, tick > StaleTicks ? tick - StaleTicks : 0), tick - MinTicks);
	const size_t   limit = pressure ? PressureRetirements : MaxRetirements;
	thread_local std::vector<BufferId> stale, kept;
	stale.clear();
	kept.clear();
	m_lru_cache.ForEachItemBelow(below, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		if (buffer.is_deleted) return false;
		if (buffer.bda_read || m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()))
			kept.push_back(id);
		else
			stale.push_back(id);
		return stale.size() == limit;
	});
	// (The ones passed over go to the back of the list: the next flips' walks do not cross them again.)
	for (const auto id: kept) TouchBuffer(m_slot_buffers[id]);
	if (stale.empty()) return;
	for (const auto id: stale) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		DeleteBuffer(id);
	}
	LiveCounters::Add(LiveCounters::GcBufferDeletes, stale.size());
}

void BufferCache::EraseRetired() {
	std::erase_if(m_retired, [&](const auto& retired) {
		if (!Spec::PacketsPassed(retired.second)) return false;
		// (A defragmentation move's new buffer destroyed before its pass ended: the pass frees the moved allocation, or,
		// ending on the worker thread, Defragment once it ended. Defragment.)
		for (const auto& move: m_defrag_moves) {
			if (move.destination != retired.first) continue;
			auto& buffer = m_slot_buffers[retired.first];
			if (m_defrag_ending.valid() && buffer.Allocation() != nullptr) m_defrag_free_later.push_back(buffer.Allocation());
			buffer.ReleaseAllocation();
		}
		m_slot_buffers.erase(retired.first);
		return true;
	});
}

bool BufferCache::MoveSpilledBufferBack() {
	// What a buffer moved back leaves of the budget: VMA's largest new block and the next frames' allocations. (A game
	// buffer in system memory is read over the bus every frame: 28 fps where there were 60 on a 16 GB GPU, until a
	// restart.)
	constexpr uint64_t Headroom = uint64_t {256} << 20u;
	// In use: read through the page table (bda_read: those reads leave no trace in the LRU), or bound in the last 600
	// flips (the texture collector's staleness).
	constexpr uint64_t InUseTicks = 600;
	if (m_spilled.empty()) {
		m_spilled_room = 0;
		return false;
	}
	const uint64_t room   = m_graphics.GetVideoMemoryRoom();
	uint64_t       needed = 0;
	BufferId       id {};
	bool           fits   = false;
	for (const auto spilled: m_spilled) {
		const auto& buffer = m_slot_buffers[spilled];
		if (!buffer.bda_read && m_lru_cache.TickOf(buffer.lru_id) + InUseTicks < m_gc_tick) continue;
		needed += buffer.Size();
		if (!fits && buffer.Size() + Headroom <= room) {
			id   = spilled;
			fits = true;
		}
	}
	// The room the ones in use need, which the texture collector makes (SpilledRoom).
	m_spilled_room = needed != 0 ? needed + Headroom : 0;
	// (On the command buffer being recorded, outside a speculation.)
	if (!fits || !m_scheduler.Active() || m_scheduler.Current().IsInvalid() || Spec::Current() != nullptr) {
		return false;
	}
	const auto begin    = m_slot_buffers[id].CpuAddress();
	const auto size     = m_slot_buffers[id].Size();
	const auto moved_id = m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, begin,
	                                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
	if (m_slot_buffers[moved_id].IsInSystemMemory()) {
		m_slot_buffers.erase(moved_id);
		return false;
	}
	// SplitUnmappedBuffers' steps with one piece (the stream score kept: the same range): the copy, the old buffer out,
	// the new one in.
	auto&       moved  = m_slot_buffers[moved_id];
	const auto& source = m_slot_buffers[id];
	moved.bda_read     = source.bda_read;
	moved.IncreaseStreamScore(source.StreamScore());
	SetVulkanObjectNameF(m_graphics.device, moved.Handle(), "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", begin,
	                     size);
	moved.CopyFrom(m_scheduler.Current(), source, 0, 0, size);
	NoteGpuWrite(begin, size);
	DeleteBuffer(id);
	Register(moved_id);
	m_spilled_room = needed > size ? needed - size + Headroom : 0;
	std::printf("Vulkan: a GPU buffer is back in video memory (%llu bytes)\n", static_cast<unsigned long long>(size));
	return true;
}

// After the game frees memory its buffers go, and the blocks of the game buffers' pool stay as full as the buffers left
// in them: video memory nobody uses between them (1-1 after two world trips on an 8 GB GPU: 764 of 1,616 MiB free, 13 of
// 27 blocks 50-90% free). VMA's passes move buffers out of the emptiest blocks: a new buffer at a move's destination
// (holding the moved allocation, which points there once the pass ends), the bytes copied on the GPU, the old one out and
// the new one in (MoveSpilledBufferBack's steps). Once every old buffer of a pass is destroyed (the GPU done with it), a
// worker thread ends the pass: VMA frees their memory then and the blocks left empty (a driver call each, up to 18 ms on
// the GPU thread).
void BufferCache::Defragment(uint64_t tick) {
	constexpr uint64_t MinFree = uint64_t {256} << 20u, PassBytes = uint64_t {32} << 20u;
	const auto         pool    = m_graphics.game_buffer_pool;
	auto*              allocator = m_graphics.allocator;
	if (pool == nullptr || (m_defrag == nullptr && tick < m_defrag_next_check)) return;
	if (m_defrag_ending.valid()) {
		if (m_defrag_ending.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
		const auto result = m_defrag_ending.get();
		// (Moved allocations of new buffers destroyed while the pass ended: EraseRetired.)
		for (auto* allocation: m_defrag_free_later) vmaFreeMemory(allocator, allocation);
		m_defrag_free_later.clear();
		m_defrag_moves.clear();
		ReleaseBufferFrees();
		if (result != VK_INCOMPLETE) EndDefragmentation();
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	struct Timed {
		BufferCache*                          cache;
		std::chrono::steady_clock::time_point start;
		bool                                  on    = false;
		uint32_t                              moves = 0, ignored = 0;
		double                                begin_ms = 0;
		~Timed() {
			if (!on) return;
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			cache->m_defrag_ms += ms;
			cache->m_defrag_max_ms = std::max(cache->m_defrag_max_ms, ms);
			if (ms >= 4.0)
				std::printf("Game buffers compacting: a flip's pass took %.2f ms (%u moves, %u ignored; VMA's pass begin %.2f "
				            "ms)\n",
				            ms, moves, ignored, begin_ms);
		}
	} timed {this, start};
	if (m_defrag_pass_open) {
		for (const auto& move: m_defrag_moves)
			if (move.source && m_slot_buffers.is_allocated(move.source)) return; // (one the GPU may still read)
		// (A new buffer destroyed since: its memory and the moved allocation go with the pass. EraseRetired.)
		for (uint32_t i = 0; i < m_defrag_pass.moveCount; ++i)
			if (m_defrag_moves[i].destination && !m_slot_buffers.is_allocated(m_defrag_moves[i].destination))
				m_defrag_pass.pMoves[i].operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY;
		m_defrag_pass_open = false;
		m_defrag_ending    = std::async(std::launch::async, [allocator, context = m_defrag, pass = &m_defrag_pass] {
            return vmaEndDefragmentationPass(allocator, context, pass);
        });
		return;
	}
	if (m_defrag == nullptr) {
		m_defrag_next_check = tick + 120;
		VmaStatistics statistics {};
		vmaGetPoolStatistics(allocator, pool, &statistics);
		const uint64_t free = statistics.blockBytes - statistics.allocationBytes;
		// (More than after the last compaction: the rest it could not move.)
		if (free < MinFree + m_defrag_free_after) return;
		VmaDefragmentationInfo info {};
		info.flags                 = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
		info.pool                  = pool;
		info.maxBytesPerPass       = PassBytes;
		info.maxAllocationsPerPass = 64;
		if (vmaBeginDefragmentation(allocator, &info, &m_defrag) != VK_SUCCESS) {
			m_defrag = nullptr;
			return;
		}
	}
	// (On the command buffer being recorded, outside a speculation: MoveSpilledBufferBack.)
	if (!m_scheduler.Active() || m_scheduler.Current().IsInvalid() || Spec::Current() != nullptr) return;
	// (The allocations of buffers erased before it stay till its end: their free on the reclaimer's thread waits.)
	if (!HoldBufferFrees()) return;
	timed.on         = true;
	const auto begun = vmaBeginDefragmentationPass(allocator, m_defrag, &m_defrag_pass);
	timed.begin_ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	if (begun != VK_INCOMPLETE) {
		ReleaseBufferFrees();
		EndDefragmentation();
		return;
	}
	m_defrag_passes++;
	m_defrag_moves.assign(m_defrag_pass.moveCount, {});
	for (uint32_t i = 0; i < m_defrag_pass.moveCount; ++i) {
		auto&             move = m_defrag_pass.pMoves[i];
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, move.srcAllocation, &info);
		const auto key = reinterpret_cast<uint64_t>(info.pUserData);
		const auto id  = BufferId {static_cast<uint32_t>(key >> 32u), static_cast<uint32_t>(key)};
		auto*      old = key != 0 ? m_slot_buffers.try_get(id) : nullptr;
		if (old == nullptr || old->Allocation() != move.srcAllocation) {
			move.operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
			timed.ignored++;
			m_defrag_ignored++;
			continue;
		}
		if (old->is_deleted) {
			// (Deleted, waiting for the GPU: the pass frees its memory, after it is destroyed. Destroyed first, it freed an
			// allocation the pass holds.)
			move.operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY;
			old->ReleaseAllocation();
			m_defrag_moves[i] = {id, {}};
			continue;
		}
		const auto begin    = old->CpuAddress();
		const auto size     = old->Size();
		const auto moved_id = m_slot_buffers.insert(m_graphics, m_scheduler, begin,
		                                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size,
		                                            move.dstTmpAllocation);
		auto&       moved  = m_slot_buffers[moved_id];
		auto&       source = m_slot_buffers[id];
		moved.AdoptAllocation(move.srcAllocation);
		moved.bda_read = source.bda_read;
		moved.IncreaseStreamScore(source.StreamScore());
		SetVulkanObjectNameF(m_graphics.device, moved.Handle(), "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", begin,
		                     size);
		moved.CopyFrom(m_scheduler.Current(), source, 0, 0, size);
		NoteGpuWrite(begin, size);
		// (Its memory is the pass's now: destroyed, it leaves it.)
		source.ReleaseAllocation();
		DeleteBuffer(id);
		Register(moved_id);
		m_defrag_moves[i] = {id, moved_id};
		timed.moves++;
	}
	m_defrag_pass_open = true;
}

void BufferCache::EndDefragmentation() {
	if (m_defrag == nullptr) return;
	VmaDefragmentationStats stats {};
	vmaEndDefragmentation(m_graphics.allocator, m_defrag, &stats);
	m_defrag = nullptr;
	VmaStatistics statistics {};
	vmaGetPoolStatistics(m_graphics.allocator, m_graphics.game_buffer_pool, &statistics);
	m_defrag_free_after = statistics.blockBytes - statistics.allocationBytes;
	std::printf("Game buffers compacted: %u moved (%llu MiB, %u ignored), %u blocks freed (%llu MiB); %llu MiB of %llu MiB "
	            "free; %u passes, %.1f ms (at most %.2f in a flip)\n",
	            stats.allocationsMoved, static_cast<unsigned long long>(stats.bytesMoved >> 20u), m_defrag_ignored,
	            stats.deviceMemoryBlocksFreed, static_cast<unsigned long long>(stats.bytesFreed >> 20u),
	            static_cast<unsigned long long>(m_defrag_free_after >> 20u),
	            static_cast<unsigned long long>(statistics.blockBytes >> 20u), m_defrag_passes, m_defrag_ms, m_defrag_max_ms);
	m_defrag_passes = m_defrag_ignored = 0;
	m_defrag_ms = m_defrag_max_ms = 0;
}

void BufferCache::RunGarbageCollector(bool collect) {
	EraseRetired();
	const auto tick = m_gc_tick++;
	RetireStaleBuffers(tick);
	// The game buffers in system memory are looked at every 60 flips, collection or not (this game's frames all read
	// guest memory through device addresses). At the end of a submission its command buffer still records
	// (EndSubmission, before the flush).
	if (tick % 60 == 0) (void)MoveSpilledBufferBack();
	Defragment(tick);
	if (!collect) {
		return;
	}
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		if ((tick & 31u) == 0) UpdateGcThresholds();
	}
	if (KytyGcLiveBudget() && m_graphics.CanReportMemoryUsage()) {
		static const uint64_t base_trigger       = m_trigger_gc_memory;
		static const uint64_t base_critical      = m_critical_gc_memory;
		static auto           kyty_budget_window = std::chrono::steady_clock::time_point {};
		const auto            now                = std::chrono::steady_clock::now();
		const uint64_t        budget             = KytyGcLiveBudgetValue(m_graphics);
		if (now - kyty_budget_window >= std::chrono::seconds(2)) {
			kyty_budget_window     = now;
			constexpr uint64_t MiB = 1024ull * 1024;
			const uint64_t     live_trigger  = budget > 2560 * MiB ? budget - 1536 * MiB : 1024 * MiB;
			const uint64_t     live_critical = budget > 1536 * MiB ? budget - 512 * MiB : 1024 * MiB;
			const uint64_t     trigger       = std::min(base_trigger, live_trigger);
			const uint64_t     critical      = std::max(std::min(base_critical, live_critical), trigger + 256 * MiB);
			const auto moved = [](uint64_t a, uint64_t b) { return (a > b ? a - b : b - a) >= (uint64_t {64} << 20u); };
			if (moved(trigger, m_trigger_gc_memory) || moved(critical, m_critical_gc_memory)) {
				std::printf("KYTYGCBUDGET buffer budget=%lluMB trigger=%lluMB critical=%lluMB used=%lluMB\n",
				            static_cast<unsigned long long>(budget >> 20), static_cast<unsigned long long>(trigger >> 20),
				            static_cast<unsigned long long>(critical >> 20),
				            static_cast<unsigned long long>(m_total_used_memory >> 20));
				std::fflush(stdout);
			}
			m_trigger_gc_memory  = trigger;
			m_critical_gc_memory = critical;
		}
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// Most collection checks do no work. Only a real collection can enumerate
	// pending dirty bytes or retire their owner, so only that path needs a drain.
	DrainGuestReadback();

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	std::vector<DownloadCopy> copies;
	size_t                    retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			m_memory_tracker.ForEachDownloadRange<false>(
			    buffer.CpuAddress(), buffer.Size(),
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, dirty_address,
				                                           dirty_size, "garbage collection");
			    },
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_gpu_modified_ranges.ForEachIntersection(
				        dirty_address, dirty_size, [&](RangeSet::Range range) {
					    copies.push_back({&buffer, range.address - buffer.CpuAddress(),
					                      range.address, range.size});
				        });
				});
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	LiveCounters::Add(LiveCounters::GcBufferDeletes, retire_count);
	if (dirty_buffers.empty()) {
		return;
	}

	EXIT_IF(copies.empty());
	DownloadBufferMemory(copies);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		// Freed once the GPU is done with it: a download on the readback queue (KYTY_READBACK_QUEUE)
		// waits only for the last write, not for the reads in flight or the commands recorded so far.
		DeleteBuffer(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	// This records into FaultManager's own buffer. Its later registration callbacks
	// pass through ChangeRegister, which drains before changing a guest owner.
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
                                                std::vector<RangeSet::Range>& ranges) const {
	ranges.clear();
	auto it = m_buffers.upper_bound(begin);
	if (it != m_buffers.begin()) --it; // (the last buffer starting before `begin` may reach into it)
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto first = std::max(it->first, begin), last = std::min(it->first + m_slot_buffers[it->second].Size(), end);
		if (first >= last) continue;
		mapped.ForEachIntersection(first, last - first, [&](RangeSet::Range range) {
			if (!ranges.empty() && ranges.back().address + ranges.back().size == range.address)
				ranges.back().size += range.size;
			else ranges.push_back(range);
		});
	}
}

bool BufferCache::TakeRegistrationSpans(std::vector<GuestRange>& spans) {
	spans.insert(spans.end(), m_registration_spans.begin(), m_registration_spans.end());
	m_registration_spans.clear();
	return !std::exchange(m_registration_spans_lost, false);
}

void BufferCache::SynchronizeRegionRequest(SyncRegionRequest& request) {
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW SynchronizeRegionRequest %.1f ms addr=0x%llx size=0x%llx\n", ms,
		            static_cast<unsigned long long>(request.address), static_cast<unsigned long long>(request.size));
	});
	DrainGuestReadback(request.address, request.size, true);
	const auto epoch = m_memory_tracker.CpuModificationEpoch(request.address, request.size);
	// A request lies in one tracker region: registrations elsewhere do not touch its buffers.
	const auto registered = RegionRegistrationEpoch(request.address);
	if (epoch != 0 && request.cpu_epoch == epoch && request.registration_epoch == registered) {
		LiveCounters::Add(LiveCounters::RegionSkips);
		return;
	}
	LiveCounters::Add(LiveCounters::RegionSyncs);
	LiveCounters::ScopedUs timed(LiveCounters::RegionSyncUs);
	SynchronizeBuffersInRange(request.address, request.size);
	request.cpu_epoch = 0;
	if (epoch != 0 && RegionRegistrationEpoch(request.address) == registered &&
	    m_memory_tracker.CpuModificationEpoch(request.address, request.size) == epoch) {
		request.cpu_epoch = epoch;
		request.registration_epoch = registered;
	}
}

// The index's pieces of a buffer: its range split at tracker region boundaries.
template <typename Fn>
static void ForEachSyncPiece(uint64_t address, uint64_t size, Fn&& fn) {
	const auto end = address + size;
	for (auto start = address; start < end;) {
		const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
		fn(start, finish);
		start = finish;
	}
}

// The registration changes into the index, in order; false when one does not fit it (pieces not where they would be:
// the caller rebuilds). Pieces are sorted by address and disjoint, as registered buffers are.
bool BufferCache::ApplySyncChanges() {
	for (const auto& change: m_sync_changes) {
		const auto end = change.address + change.size;
		const auto first = std::lower_bound(m_sync_buffers.begin(), m_sync_buffers.end(), change.address,
		                                    [](const SyncBuffer& piece, uint64_t address) { return piece.start < address; });
		const auto at    = static_cast<size_t>(first - m_sync_buffers.begin());
		if (!change.insert) {
			auto last = at;
			while (last < m_sync_buffers.size() && m_sync_buffers[last].id == change.id && m_sync_buffers[last].start < end)
				++last;
			if (last == at || m_sync_buffers[at].start != change.address || m_sync_buffers[last - 1].end != end) return false;
			m_sync_buffers.erase(m_sync_buffers.begin() + static_cast<ptrdiff_t>(at),
			                     m_sync_buffers.begin() + static_cast<ptrdiff_t>(last));
			m_sync_stamps.erase(m_sync_stamps.begin() + static_cast<ptrdiff_t>(at),
			                    m_sync_stamps.begin() + static_cast<ptrdiff_t>(last));
			continue;
		}
		auto* buffer = m_slot_buffers.try_get(change.id);
		if (buffer == nullptr) continue; // (erased since: its removal follows, and finds nothing either)
		if ((at > 0 && m_sync_buffers[at - 1].end > change.address) ||
		    (at < m_sync_buffers.size() && m_sync_buffers[at].start < end))
			return false;
		size_t count = 0;
		ForEachSyncPiece(change.address, change.size, [&](uint64_t, uint64_t) { ++count; });
		m_sync_buffers.insert(m_sync_buffers.begin() + static_cast<ptrdiff_t>(at), count, SyncBuffer {});
		m_sync_stamps.insert(m_sync_stamps.begin() + static_cast<ptrdiff_t>(at), count, SyncStamp {});
		auto piece = at;
		ForEachSyncPiece(change.address, change.size,
		                 [&](uint64_t start, uint64_t finish) { m_sync_buffers[piece++] = {start, finish, buffer, change.id}; });
	}
	return true;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	DrainGuestReadback(vaddr, size, true);
	if (!m_sync_buffers_valid && !m_sync_changes_lost && ApplySyncChanges()) {
		m_sync_changes.clear();
		m_sync_buffers_valid = true;
	}
	if (!m_sync_buffers_valid) {
		// The pieces of buffers that stayed registered keep their stamps: a stamp is about its own buffer's
		// contents against the CPU epoch of its region, which another buffer's registration does not move.
		std::swap(m_sync_buffers, m_old_sync_buffers);
		std::swap(m_sync_stamps, m_old_sync_stamps);
		m_sync_buffers.clear();
		m_sync_stamps.clear();
		m_sync_buffers.reserve(m_buffers.size());
		size_t old = 0;
		for (const auto& [address, id]: m_buffers) {
			auto& buffer = m_slot_buffers[id];
			const auto end = address + buffer.Size();
			for (auto start = address; start < end;) {
				const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
				while (old < m_old_sync_buffers.size() && m_old_sync_buffers[old].start < start) ++old;
				const bool kept = old < m_old_sync_buffers.size() && m_old_sync_buffers[old].start == start &&
				                  m_old_sync_buffers[old].end == finish && m_old_sync_buffers[old].id == id;
				m_sync_buffers.push_back({start, finish, &buffer, id});
				m_sync_stamps.push_back(kept ? m_old_sync_stamps[old] : SyncStamp {});
				start = finish;
			}
		}
		m_sync_buffers_valid = true;
		m_sync_changes.clear();
		m_sync_changes_lost = false;
	}
	const auto end = vaddr + size;

	auto it = std::lower_bound(m_sync_buffers.begin(), m_sync_buffers.end(), vaddr,
	                           [](const SyncBuffer& buffer, uint64_t address) {
		                           return buffer.end <= address;
	                           });
	for (; it != m_sync_buffers.end() && it->start < end; ++it) {
		const auto start  = std::max(it->start, vaddr);
		const auto finish = std::min(it->end, end);
		if (start < finish) {
			SyncStamp* stamp = nullptr;
			uint64_t epoch = 0;
			{
				stamp = &m_sync_stamps[static_cast<size_t>(it - m_sync_buffers.begin())];
				epoch = m_memory_tracker.CpuModificationEpoch(start, finish - start);
				if (epoch != 0 && stamp->epoch == epoch && start >= stamp->begin && finish <= stamp->end) continue;
			}
			(void)SynchronizeBuffer(*it->buffer, start, finish - start, false, false);
			if (stamp != nullptr && epoch != 0 && m_sync_buffers_valid &&
			    m_memory_tracker.CpuModificationEpoch(start, finish - start) == epoch) {
				*stamp = {start, finish, epoch};
			}
		}
	}

}

} // namespace Libs::Graphics
