#include "local/faltas-reciclado.h" // (sync3)
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "local/envio1.h"
#include "gpu-residency.h"
#include "lane-sync.h"
#include "slow-log.h"

#include "native-buffer-residency.h"
#include "live-census.h"
#include "live-counters.h"
#include "local-platform.h"
#include "live-switch.h"
#include "live-trace.h"
#include "native-resource-state.h"
#include "async-upload.h"
#include "local/tiron7.h" // (tiron7: KYTY_TIRON7)
#include "raiz.h" // (raiz2: KYTY_RAIZ)
#include "parpadeo7.h" // (parpadeo7: KYTY_PARPADEO7)
#include "local/juego3.h" // (juego3, KYTY_JUEGO3)
#include "f0-measure.h" // (f0: KYTY_F0)
#include "local/serie1.h" // (serie1: KYTY_SERIE1)
#include "local/trad2.h" // (trad2: KYTY_TRAD2)
#include "local/live-control.h" // (trad2: LiveControl::g_flips, the [trad2] line's frames)
#include "raiz-wb-guard.h" // (crash-ba2b1f: KYTY_RAIZ_WB_GUARD)
#include "parallel-pages.h"
#include "readback-queue.h"
#include "sync-stats.h"
#include "fresh-memory.h"
#include "texture-offload.h"
#include "malla-nativa.h" // (MALLA-NATIVA: KYTY_MALLA_NATIVA)
#include "local/frame-nativo.h"  // (FRAME-NATIVO2: the DefaultTagg protection counter of the [frame-nativo2] line)
#include "local/frame-nativo2.h" // (FRAME-NATIVO2: KYTY_FRAME_NATIVO2)
#include "local/nativo-cubierto.h" // (VIGILANCIA-DOBLE: KYTY_SIN_VIGILANCIA_DOBLE)
#include "local/arena-dueno.h" // (ARENA-DUENO: KYTY_ARENA_DUENO=1, medidor)
#include "loader/demonsSoulsMemMap.h" // (FRAME-NATIVO2 v3: the arenas' addresses at run time)
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
// KYTY_UPLOAD_LANES (local/async-upload.h): 1 = one upload lane (default), 2 = two, 3 = two with the order checks.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_upload_lanes {1};
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

// (BUF-VRAM) KYTY_BUF_VRAM (por defecto 1; 0 = como antes). Con la VRAM al nivel critico: los bufferes escritos por la
// GPU que llevan >= 10 s sin usarse ya no se "tocan" en RetireStaleBuffers (antes volvian al final de la lista LRU y el
// recolector no los veia nunca: [bufgc] 0 borrados en toda la partida); el recolector los descarga a la RAM del juego y
// los borra (se vuelven a subir si se usan). La compactacion del pool empieza con 64 MiB libres (antes 256). Linea
// [buf-vram] cada 30 s.
namespace {
bool BufVramOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BUF_VRAM");
		return v == nullptr || *v != '0';
	}();
	return on;
}
uint64_t g_bufvram_evicted = 0, g_bufvram_evicted_bytes = 0, g_bufvram_dirty = 0, g_bufvram_dirty_bytes = 0; // (BUF-VRAM)
} // namespace
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <tuple>
#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "speculation-state.h"
#include "local/vram-drv.h" // (VRAM-DRV)
#include "vram-tope.h"      // (BUF-RAM: nivel 2 del tope)
#include "buf-mem.h"         // (BUF-MEM: KYTY_BUF_MEM)

#define VRAM_EMU_OWNER ::VramEmu::kBufCache
#include "local/vram-emu.h" // (VRAM-EMU) sus asignaciones de VMA y CreateImage apuntadas con su dueno

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// (BUF-RAM) Bufferes del juego que BufRamPass paso a memoria del sistema: id, marca LRU al migrar y tamano.
struct BufRamEntry {
	Common::SlotId id;
	uint64_t       tick_mig = 0, size = 0;
};
std::vector<BufRamEntry>               g_bufram;
std::unordered_map<uint64_t, int64_t>  g_bufram_cooldown; // direccion -> ms hasta el que no se vuelve a migrar
// (M7) Id del gestor central (GestorMem) de cada bufer registrado, por su lru_id (denso, reutilizado: acotado por los
// bufferes vivos). Tocar en TouchBuffer sin locks. g_mover_gid: el id que pasa al bufer nuevo al migrar (CambiarLugar).
constexpr size_t                       kBufGid = size_t {1} << 18u;
std::atomic<uint64_t>                  g_buf_gid[kBufGid] {};
uint64_t                               g_mover_gid = 0;

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
	// (serie1) KYTY_SERIE1=1 too, unless KYTY_ACQ_READBACK_LANE=0: the culling alone is on lane 1 there, and its bits'
	// copy behind the graphics lane would wait for what the culling no longer waits for (local/serie1.h).
	static const bool by_writer = [] {
		const char* value = std::getenv("KYTY_ACQ_READBACK_LANE");
		return (value != nullptr && *value == '1') || ((value == nullptr || *value != '0') && Serie1::WantsLanes());
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
	Juego3::NoteWrite(vaddr, size); // (juego3: what a dispatch writes, by its shader; off: one load)
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
	const bool     medir = FaltasReciclado::t_medir; // (sync3, KYTY_TECHO: the fault's wait vs copy)
	const uint64_t w0    = medir ? __rdtsc() : 0;
	struct CopyClock {
		bool     on;
		uint64_t w0, c0 = 0;
		~CopyClock() {
			if (!on) return;
			const uint64_t end = __rdtsc();
			FaltasReciclado::g_rb_wait.fetch_add((c0 != 0 ? c0 : end) - w0, std::memory_order_relaxed);
			if (c0 != 0) FaltasReciclado::g_rb_copy.fetch_add(end - c0, std::memory_order_relaxed);
			FaltasReciclado::g_rb_n.fetch_add(1, std::memory_order_relaxed);
		}
	} copy_clock {medir, w0};
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
	if (medir) copy_clock.c0 = __rdtsc();
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
		// (raiz2, KYTY_RAIZ: parts that follow each other in guest memory and in the download, one write-back - one
		// window over their no-access pages instead of one a part -.)
		const auto& parts = request->parts;
		for (size_t i = 0; i < parts.size();) {
			uint64_t size = parts[i].size;
			size_t   next = i + 1;
			if (Raiz::On())
				for (; next < parts.size() && parts[next].address == parts[i].address + size &&
				       parts[next].offset == parts[i].offset + size;
				     ++next) {
					size += parts[next].size;
					Raiz::g_counters.merged_writebacks.fetch_add(1, std::memory_order_relaxed);
				}
			WriteBackGpuOwned(parts[i].address, source + parts[i].offset, size, "readback");
			i = next;
		}
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
		Trad1StampCoverageAll(request->parts); // (trad1b)
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

// KYTY_NATIVE_MEMORY=1 (off by default): no read watcher protects a page (PageManager applies none), so a game read of
// GPU-written memory faults nowhere. GPU writes are written back to guest memory as their work completes instead:
// copy-feedback snapshots always (as KYTY_EAGER_FEEDBACK), shader-written buffer ranges by NoteNativeWrite /
// WriteBackNativeWrites, bounded below.
// (nativemem1) Not with KYTY_WRITE_WATCH=1 (LibKernel::Memory::NativeReadFaultsEnabled): GPU-written pages are no-access
// again there and a read of them faults, so none of these eager write-backs is needed.
static bool NativeMemoryOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_MEMORY");
		return v != nullptr && v[0] == '1' && !Libs::LibKernel::Memory::NativeReadFaultsEnabled();
	}();
	return on;
}

// (nativemem1) KYTY_NATIVE_MEMORY=1 with or without KYTY_WRITE_WATCH: CPU writes are seen by the write watch only, so a
// write of the emulator's own straight to guest memory invalidates what it must itself (CopyGuestMemory).
static bool NativeWriteWatchOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_MEMORY");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// (raiz2) The upload worker (KYTY_ASYNC_UPLOAD) with native memory: not with KYTY_WRITE_WATCH=1 (ww2: the worker read
// guest memory itself, faulted on a page GPU work had made no-access meanwhile and asked the GPU thread - maybe waiting
// for it - for the readback: a hang), except with KYTY_RAIZ=1, where its copies go through Common::GuardedCopy and a
// window over the no-access pages (AsyncUpload's CopyFrom, Memory::NativeGuardedRead): no guest fault, the bytes the
// backing view gave. 11-16 (raiz1, Boletaria): all uploads on the GPU thread, buffer_sync 8.6-10.5 ms a frame against
// 0.2-0.5 with the worker.
static bool UploadWorkerAllowed() noexcept {
	return !PageManager::NativeReadFaults() || Raiz::On();
}
// (raiz2) The worker's source pointer: TryGetUploadSourcePointer with KYTY_RAIZ (over protected pages too: its copies
// are guarded), TryGetBackingPointer otherwise.
static const uint8_t* UploadSourcePointer(uint64_t vaddr, uint64_t size) {
	return Raiz::On() ? Libs::LibKernel::Memory::TryGetUploadSourcePointer(vaddr, size)
	                  : Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, size);
}

static uint64_t NativeEnvKiB(const char* name, uint64_t fallback_kib) noexcept {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == 0) return fallback_kib * 1024;
	return std::strtoull(v, nullptr, 10) * 1024;
}

// GPU thread, beside each shader write of a guest buffer (GetBuffer, is_written). Writes of one recording share a tick:
// the same range noted again only moves its tick.
void BufferCache::NoteNativeWrite(uint64_t vaddr, uint64_t size) {
	if (!NativeMemoryOn() || Spec::Current() != nullptr || size == 0) return;
	const auto tick = m_scheduler.CurrentTick();
	for (size_t i = m_native_writes.size(), last = i > 8 ? i - 8 : 0; i > last; --i) {
		auto& w = m_native_writes[i - 1];
		if (w.address == vaddr && w.size == size) {
			if (i == m_native_writes.size()) {
				w.tick = std::max(w.tick, tick);
				return;
			}
			break; // (an older entry: a new one keeps the queue ordered by tick)
		}
	}
	constexpr size_t Limit = size_t {1} << 16;
	if (m_native_writes.size() >= Limit) {
		// No point between packets for very long: the oldest is written back by the next CPU-side sync, if ever read
		// by the emulator; a game read of it may see stale bytes.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("BufferCache: native write-back queue full, dropping 0x%016" PRIx64 "\n", m_native_writes.front().address);
			std::fflush(stdout);
		}
		m_native_writes.pop_front();
	}
	m_native_writes.push_back({vaddr, size, tick});
}

// GPU thread, between guest packets (WriteBackCompletedFeedback). Completed writes still GPU-owned are downloaded
// (ReadMemoryOnGpu: a snapshot if one holds them, else one download; the pages then CPU-owned). Bounded:
//   KYTY_NATIVE_EAGER_SMALL_KB (64): larger ranges only over a 64 KiB granule some CPU-side read pulled back before;
//   KYTY_NATIVE_EAGER_BUDGET_KB (4096): bytes per call; the rest waits for the next call.
void BufferCache::WriteBackNativeWrites() {
	if (m_native_writes.empty()) return;
	auto& master = m_scheduler.GetMasterSemaphore();
	if (m_native_writes.front().tick > master.KnownGpuTick()) return;
	static const uint64_t small_limit  = NativeEnvKiB("KYTY_NATIVE_EAGER_SMALL_KB", 64);
	static const uint64_t budget = NativeEnvKiB("KYTY_NATIVE_EAGER_BUDGET_KB", 4096);
	const auto completed = master.KnownGpuTick();
	uint64_t   spent     = 0;
	while (!m_native_writes.empty() && m_native_writes.front().tick <= completed && spent < budget) {
		const auto w = m_native_writes.front();
		m_native_writes.pop_front();
		if (!IsRegionRegistered(w.address, w.size) || !m_memory_tracker.IsRegionGpuModified(w.address, w.size)) {
			LiveCounters::Add(LiveCounters::EagerClean);
			continue;
		}
		if (w.size > small_limit) {
			bool read_before = false;
			{
				const std::scoped_lock lock(m_native_read_mutex);
				for (auto g = w.address >> 16u; g <= (w.address + w.size - 1) >> 16u && !read_before; ++g)
					read_before = m_native_read_granules.count(g) != 0;
			}
			if (!read_before) {
				LiveCounters::Add(LiveCounters::EagerRefused);
				continue;
			}
		}
		ReadMemoryOnGpu(w.address, w.size, false);
		LiveCounters::Add(LiveCounters::EagerWritten);
		spent += w.size;
	}
}

// KYTY_NATIVE_MEMORY=1, GPU thread, right before the game can see a label, an end-of-pipe event or a flip
// (GpuResourceManager::NativeFlushGpuWrites): every GPU-written range of a registered buffer goes to guest memory now,
// with no size bound or budget (WriteBackNativeWrites' bounds left games reading stale bytes). ReadMemoryOnGpu downloads
// what its own buffer holds of the range (one buffer each: a range is cut at the buffers' ends) and waits for the GPU
// work before the copy; the pages are GPU-clean after it, so a later call downloads only what the GPU wrote since.
uint64_t BufferCache::NativeWriteBackAll(uint64_t* bytes) {
	if (!NativeMemoryOn() || m_gpu_modified_ranges.Empty()) return 0;
	// (A copy of the ranges: ReadMemoryOnGpu takes the downloaded ones out of the set.)
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(m_gpu_modified_ranges.Count());
	m_gpu_modified_ranges.ForEach([&](uint64_t begin, uint64_t end) { ranges.emplace_back(begin, end); });
	uint64_t pieces = 0;
	for (const auto& [begin, end]: ranges) {
		for (uint64_t at = begin; at < end;) {
			const auto* owner_id = m_page_table.Find(at >> PageTable::kPageBits);
			auto*       owner    = owner_id != nullptr && *owner_id ? m_slot_buffers.try_get(*owner_id) : nullptr;
			if (owner == nullptr || owner->is_deleted || !owner->IsInBounds(at, 1)) {
				// (No buffer holds this caching page: nothing of it to download.)
				at = ((at >> PageTable::kPageBits) + 1) << PageTable::kPageBits;
				continue;
			}
			const uint64_t stop = std::min<uint64_t>(end, owner->CpuAddress() + owner->Size());
			if (stop <= at) break; // (cannot happen: IsInBounds)
			if (IsRegionRegistered(at, stop - at) && m_memory_tracker.IsRegionGpuModified(at, stop - at)) {
				ReadMemoryOnGpu(at, stop - at, false);
				++pieces;
				if (bytes != nullptr) *bytes += stop - at;
			}
			at = stop;
		}
	}
	return pieces;
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
	} else if (static auto& batch = LiveSwitch::Find("feedback_batch", 0); batch.load(std::memory_order_relaxed) != 0) {
		// KYTY_FEEDBACK_BATCH (live feedback_batch, on by default): the snapshot is copied
		// as the open buffer closes, with the others there behind one barrier each side, not behind two of its own
		// here. Later work in the buffer that writes the range invalidates the slot (InvalidateCopyFeedback), so what
		// the copy reads then is what it would read here whenever the slot is still used. (The source outlives the
		// buffer: DeleteBuffer defers.)
		feedback.download.written_serial = m_scheduler.CommandSerial();
		m_feedback_batch.push_back({owner->Handle(), owner->Offset(vaddr), selected * CopyFeedback::SlotSize, size});
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
	// (FB-PEREZOSO, 11-25, KYTY_FB_PEREZOSO=1 por defecto, =0 como antes) Log 20261125-111241-732: con las piezas
	// nativas la escritura anticipada de estas instantaneas abria 4081-4971 ventanas de escritura/s, TODAS en el hilo de
	// render (0.61-0.65 ms/frame de ventanas + write-release/write-track del hilo GPU 0.43 -> 1.1-1.3 ms/frame), y
	// [wb-guard] conto 30767 paginas que el juego escribia el mismo mientras: bytes escritos para nada. Base (log
	// 101543-027): 0 ventanas. Ahora la instantanea solo se entrega cuando el juego LEE el rango (ReadMemory ->
	// TryGuestReadFromFeedback, ya existente): ni copia doble ni ventana por frame en el hilo de render.
	static const bool fb_perezoso = [] {
		const char* v = std::getenv("KYTY_FB_PEREZOSO");
		const bool  on = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[fb-perezoso] %s (KYTY_FB_PEREZOSO=%s): instantaneas de copy feedback entregadas %s\n", on ? "on" : "off",
		            v != nullptr ? v : "(no puesta)", on ? "solo al leerlas el juego" : "en cuanto la GPU acaba (como antes)");
		std::fflush(stdout);
		return on;
	}();
	if (static auto& eager = LiveSwitch::Find("eager_feedback", 0);
	    eager.load(std::memory_order_relaxed) != 0 || (NativeMemoryOn() && !fb_perezoso)) {
		if (m_eager_feedback.size() >= CopyFeedback::SlotCount) m_eager_feedback.pop_front();
		m_eager_feedback.push_back({selected, vaddr, size, slot.tick});
	}
}

// KYTY_EAGER_FEEDBACK. A snapshot is written back only once its tick completed and while its slot still holds it: a
// later GPU write over the range invalidates the slot (InvalidateCopyFeedback), so no later write is in flight.
// TryGuestReadFromFeedback publishes only GPU-modified, CPU-clean bytes and unprotects as a fault's read would.
void BufferCache::WriteBackCompletedFeedback(bool may_wait) {
	if (NativeMemoryOn()) WriteBackNativeWrites();
	// (Called between guest packets: nothing to do until the oldest snapshot's tick completes.)
	if (m_eager_feedback.empty() || m_eager_feedback.front().tick > m_scheduler.GetMasterSemaphore().KnownGpuTick()) return;
	static auto& eager  = LiveSwitch::Find("eager_feedback", 0);
	static auto& verify = LiveSwitch::Find("eager_feedback_verify", 0);
	if ((eager.load(std::memory_order_relaxed) == 0 && !NativeMemoryOn()) || !m_copy_feedback) {
		m_eager_feedback.clear();
		return;
	}
	const auto completed = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	while (!m_eager_feedback.empty() && m_eager_feedback.front().tick <= completed) {
		const auto entry = m_eager_feedback.front();
		m_eager_feedback.pop_front();
		const auto& slot = m_copy_feedback->slots[entry.slot];
		if (slot.address != entry.address || slot.tick != entry.tick) {
			LiveCounters::Add(LiveCounters::EagerStale);
		} else if (!IsRegionRegistered(entry.address, entry.size) ||
		           !m_memory_tracker.IsRegionGpuModified(entry.address, entry.size)) {
			LiveCounters::Add(LiveCounters::EagerClean);
		} else if (const auto every = verify.load(std::memory_order_relaxed);
		           may_wait && every > 0 && ++m_eager_verify_count % static_cast<uint64_t>(every) == 0 &&
		           VerifyEagerFeedback(entry)) {
			LiveCounters::Add(LiveCounters::EagerWritten);
		} else if (TryGuestReadFromFeedback(entry.address, entry.size, false)) {
			LiveCounters::Add(LiveCounters::EagerWritten);
		} else {
			LiveCounters::Add(LiveCounters::EagerRefused);
		}
	}
}

// KYTY_EAGER_FEEDBACK_VERIFY (live eager_feedback_verify N, default 0): every Nth write-back, the snapshot's bytes
// against what a fault's read would fetch with no snapshot at that moment: the source buffer's GPU-written bytes read
// back now (DownloadBufferMemory, synchronous: the verify arm's time is not the lever's). Those bytes, read back, are
// what the guest gets; a byte the snapshot holds otherwise is counted (eager_mismatch, a line for the first 8). False
// when it did not run (the write-back then as usual).
bool BufferCache::VerifyEagerFeedback(const EagerFeedback& entry) {
	auto&      feedback = *m_copy_feedback;
	const auto begin    = entry.address & ~(TRACKER_PAGE_SIZE - 1);
	const auto end      = (entry.address + entry.size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	if (m_scheduler.Current().InCopyRun()) return false;
	for (const auto& pending: m_guest_readbacks)
		if (pending && pending->begin < end && begin < pending->begin + pending->size) return false;
	auto& buffer = m_slot_buffers[FindBuffer(entry.address, entry.size)];
	if (!buffer.IsInBounds(begin, end - begin)) return false;
	// Every GPU-written byte of its pages read back (the protection goes for the pages whole, as a fault's read would
	// drop it); the snapshot compared over its own range, the GPU-written bytes in it.
	std::vector<DownloadCopy> copies;
	m_gpu_modified_ranges.ForEachIntersection(begin, end - begin, [&](RangeSet::Range range) {
		copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
	});
	if (copies.empty()) return false;
	struct Compared {
		uint64_t address, size, at;
	};
	std::vector<Compared> compared;
	std::vector<uint8_t>  snapshot;
	for (const auto& copy: copies) {
		const auto first = std::max(copy.address, entry.address);
		const auto last  = std::min(copy.address + copy.size, entry.address + entry.size);
		if (first >= last) continue;
		const auto offset = entry.slot * CopyFeedback::SlotSize + (first - entry.address);
		feedback.download.Invalidate(offset, last - first);
		const auto* from = feedback.download.Mapped().data() + offset;
		compared.push_back({first, last - first, snapshot.size()});
		snapshot.insert(snapshot.end(), from, from + (last - first));
	}
	LiveCounters::Add(LiveCounters::SyncDownloads);
	DownloadBufferMemory(copies);
	uint64_t mismatched = 0;
	for (const auto& part: compared) {
		const auto* now = LibKernel::Memory::TryGetBackingPointer(part.address, part.size);
		if (now == nullptr) continue; // (unmapped meanwhile: nothing to compare)
		for (uint64_t i = 0; i < part.size; ++i) mismatched += now[i] != snapshot[part.at + i] ? 1u : 0u;
	}
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.SubtractSorted(copies);
		++m_gpu_coverage_epoch;
		Trad1StampCoverageAll(copies); // (trad1b)
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
	LiveCounters::Add(LiveCounters::EagerVerified);
	if (mismatched != 0) {
		LiveCounters::Add(LiveCounters::EagerMismatch);
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("EAGER_FEEDBACK_MISMATCH address=0x%llx size=0x%llx bytes=%llu tick=%llu\n",
			            static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.size),
			            static_cast<unsigned long long>(mismatched), static_cast<unsigned long long>(entry.tick));
	}
	return true;
}

void BufferCache::FlushFeedbackBatch(CommandBuffer& command) {
	if (m_feedback_batch.empty()) return;
	const auto native = command.Handle();
	// As Buffer::CopyFrom's barriers, once for the batch: what wrote the sources (and the host's reads of the slots)
	// before the copies, the copies before what reads the slots after (the host).
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostRead;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion, 1, &before, 0, nullptr,
	                       0, nullptr);
	const vk::Buffer destination = m_copy_feedback->download.Handle();
	for (const auto& copy: m_feedback_batch) {
		const vk::BufferCopy region {copy.source_offset, copy.destination_offset, copy.size};
		native.copyBuffer(copy.source, destination, 1, &region);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask =
	    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
	LiveCounters::Add(LiveCounters::FeedbackBatches);
	m_feedback_batch.clear();
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
		Trad1StampCoverageAll(copies); // (trad1b)
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
	// (The upload lanes split protections by this region: its region managers' and the page manager's locks.)
	static_assert(AsyncUpload::Worker::LaneRegion == TRACKER_REGION_SIZE);
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
		{ // (M7) al gestor central: bufer de la GPU, en VRAM o en memoria del sistema (BUF-RAM)
			const int lugar = buffer.IsInSystemMemory() ? GestorMem::Ram : GestorMem::Vram;
			uint64_t  gid   = std::exchange(g_mover_gid, uint64_t {0});
			if (gid != 0) GestorMem::CambiarLugar(gid, lugar); // migrado: el mismo recurso, otro sitio
			else gid = GestorMem::Registrar(GestorMem::TipoBuffer, lugar, buffer.Size(), 0, nullptr);
			if (buffer.lru_id < kBufGid) g_buf_gid[buffer.lru_id].store(gid, std::memory_order_relaxed);
			else GestorMem::Liberar(gid);
		}
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
		if (buffer.lru_id < kBufGid) GestorMem::Liberar(g_buf_gid[buffer.lru_id].exchange(0, std::memory_order_relaxed)); // (M7)
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
		// (BDA-RETIRO) Todo uso real pasa por aqui; el recolector toca la LRU sin esto (RetireStaleBuffers).
		if (buffer.bda_read)
			buffer.bda_use_ms =
			    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
		if (buffer.lru_id < kBufGid) GestorMem::Tocar(g_buf_gid[buffer.lru_id].load(std::memory_order_relaxed)); // (M7)
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	const Envio1::Scope envio1_miss(Envio1::BufMiss); // (envio1: cleanup)
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted) {
		return;
	}
	std::erase(m_spilled, id);
	if (!g_bufram.empty()) { // (BUF-RAM)
		const auto it = std::find_if(g_bufram.begin(), g_bufram.end(), [&](const BufRamEntry& e) { return e.id == id; });
		if (it != g_bufram.end()) {
			VramTope::g_bufram_bytes.fetch_sub(it->size, std::memory_order_relaxed);
			VramTope::g_bufram_n.fetch_sub(1, std::memory_order_relaxed);
			*it = g_bufram.back();
			g_bufram.pop_back();
		}
	}
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
// (crash-ba2b1f, KYTY_RAIZ=1 + KYTY_RAIZ_WB_GUARD, local/raiz-wb-guard.h) And not on a GPU-owned page the game wrote since
// the GPU took it (its write watch, reset then; or a game write found during a put-in): that write reached no fault and
// no upload, so the GPU's copy of the page lacks it and writing the copy back would undo it (20261116-043751: a job
// worker's shared_ptr control blocks replaced by the page's old bytes, eboot+0xba2b1f). =1: the page is left alone
// (the game's bytes win; it is queued as CPU-written, so the GPU's copy is uploaded again); =2: counted only.
void BufferCache::WriteBackGpuOwned(uint64_t address, const uint8_t* data, uint64_t size, const char* source) {
	std::vector<uint64_t> game_pages; // (sorted; empty - no allocation - unless the guard found some)
	const int             guard = LibKernel::Memory::RaizWbGuardMode();
	if (guard != 0) {
		LibKernel::Memory::RaizWbGameWritten(address, size, &game_pages);
		if (guard != 1) {
			for (const auto page: game_pages)
				if (m_memory_tracker.IsRegionFullyGpuModified(page, TRACKER_PAGE_SIZE))
					LibKernel::Memory::RaizWbNoteSkipped(page, address, size, source, false);
			game_pages.clear();
		}
	}
	const auto game_wrote = [&game_pages](uint64_t at) {
		return !game_pages.empty() && std::binary_search(game_pages.begin(), game_pages.end(), at & ~(TRACKER_PAGE_SIZE - 1));
	};
	if (game_pages.empty() && m_memory_tracker.IsRegionFullyGpuModified(address, size)) {
		LibKernel::Memory::WriteBacking(address, data, size, source);
		return;
	}
	uint64_t skipped = 0;
	// (raiz2, KYTY_RAIZ: the GPU-owned pages in a row written back together - one window -, not a page at a time.)
	const bool join      = Raiz::On();
	uint64_t   run_begin = 0, run_end = 0;
	const auto flush     = [&] {
		if (run_end > run_begin) LibKernel::Memory::WriteBacking(run_begin, data + (run_begin - address), run_end - run_begin, source);
		run_begin = run_end = 0;
	};
	for (uint64_t at = address, end = address + size; at < end;) {
		const auto next = std::min((at & ~(TRACKER_PAGE_SIZE - 1)) + TRACKER_PAGE_SIZE, end);
		const bool owned = m_memory_tracker.IsRegionFullyGpuModified(at, next - at);
		if (owned && game_wrote(at)) {
			// (crash-ba2b1f) Left alone: the game's bytes win (counted, logged and queued as CPU-written there).
			flush();
			LibKernel::Memory::RaizWbNoteSkipped(at & ~(TRACKER_PAGE_SIZE - 1), address, size, source, true);
		} else if (owned) {
			if (join && run_end == at && run_end > run_begin) {
				run_end = next;
				Raiz::g_counters.merged_writebacks.fetch_add(1, std::memory_order_relaxed);
			} else {
				flush();
				run_begin = at;
				run_end   = next;
				if (!join) flush();
			}
		} else {
			flush();
			skipped += next - at;
		}
		at = next;
	}
	flush();
	LiveCounters::Add(LiveCounters::WriteBackSkips, skipped);
	static std::atomic<uint32_t> logged {0};
	if (skipped != 0 && logged.fetch_add(1, std::memory_order_relaxed) < 64) { // (crash-ba2b1f: 0 when only the guard skipped)
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
		Trad1StampCoverageAll(copies); // (trad1b)
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

// The image upload ring: 1 GiB on a PC with 32 GB of RAM or more (512 MiB before). A frame where the game streams
// in an area uploads 0.8-1.4 GB of textures; past the ring's size the render thread waited for the GPU to finish
// the first copies before it could reuse the ring (60-80 ms of gpu_wait in those frames, Boletaria).
// RAM diet (port default): the ring is host-visible memory the driver keeps in the process's private commit (the
// 1024 MiB private region of the RAM breakdown; GPU system memory host-visible 1216 MiB = this ring + 64 + 64 + 32 +
// 32 MiB of the other rings): 256 MiB, as KYTY_LOWMEM_VRAM=1 already had. KYTY_STAGING_RING_MB=<n> (64-2048) sets it;
// KYTY_RAM_DIET_STAGING=0 gives the size before (1 GiB with 32 GB of RAM).
uint64_t BufferCacheStagingRingSize() {
	static const uint64_t size = [] {
		if (const char* mb = std::getenv("KYTY_STAGING_RING_MB"); mb != nullptr && mb[0] != '\0') {
			const uint64_t value = std::strtoull(mb, nullptr, 10);
			if (value >= 64 && value <= 2048) return value * MiB;
		}
		// (tiron9, KYTY_TIRON9=1) 768 MiB: 20261120-011558, the 622.7 ms frame uploaded 362 MB of images through the 256 MiB
		// ring - the render thread waited in StreamBuffer::Map <- CommandScheduler::Wait (gpu_wait 124 ms, 22% of its samples)
		// for the GPU to free ring space, and the GPU's submissions waited for the upload copies (upload_wait 253 ms).
		if (const char* t9 = std::getenv("KYTY_TIRON9"); t9 == nullptr || t9[0] != '0') {
			std::printf("[tiron9] KYTY_TIRON9=1: anillo de subida de imagenes 768 MiB (antes 256)\n");
			std::fflush(stdout);
			return uint64_t {768} * MiB;
		}
		const char* diet = std::getenv("KYTY_RAM_DIET_STAGING");
		const bool  off  = diet != nullptr && diet[0] == '0';
		if (LocalPlatform::LowMemVram() || !off) return 256 * MiB; // KYTY_LOWMEM_VRAM=1 or the RAM diet
		const uint64_t ram = LocalPlatform::PhysicalMemory();
		return ram >= (uint64_t {32} << 30u) ? 1024 * MiB : 512 * MiB;
	}();
	return size;
}
static uint64_t StagingRingSize() {
	const uint64_t size = BufferCacheStagingRingSize();
	std::printf("RAM diet: image upload ring %llu MiB of host-visible memory (KYTY_STAGING_RING_MB, "
	            "KYTY_RAM_DIET_STAGING=0: 1 GiB as before)\n",
	            static_cast<unsigned long long>(size / MiB));
	std::fflush(stdout);
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

// (FRAME-NATIVO2, KYTY_FRAME_NATIVO2=1: local/frame-nativo2.h) The 4 DefaultTagg arenas imported whole as VkBuffers
// (VK_EXT_external_memory_host), never registered in the cache (no watch, no protection, no upload). Render thread
// only for the imports; the lookup is lock-free (an arena's buffer is set before its flag, and lives to the end).
namespace {
struct FrameNativo2Arena {
	Buffer*           buffer     = nullptr; // (kept to the end of the process: the arenas are never unmapped)
	uint64_t          guest_base = 0;       // guest address of the buffer's byte 0
	vk::DeviceAddress device     = 0;
};
struct FrameNativo2State {
	FrameNativo2Arena                     arena[FrameNativo2::kArenas];
	std::chrono::steady_clock::time_point next_try[FrameNativo2::kArenas] {};
	std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
	uint64_t                              last_hits = 0, last_bytes = 0, last_prot = 0;
	bool                                  prot_base = false;
	int                                   refusals_printed = 0;
	bool                                  need_dirty[FrameNativo2::kArenas] {}; // (v2: arena unprotected, trackers told)
};
// (v9, =3) Hilo de la GPU: los rangos EXACTOS que el juego libero (gancho FreeTag 0x7fe540, local/frame-nativo2.h) en un
// arena sin proteger: lo que la GPU escribio alli se descarta y pasan a la CPU (el juego los reescribira sin fallo).
template <typename Tracker, typename Discard>
void FrameNativo2DrainFreed(Tracker& tracker, Discard&& discard) {
	std::vector<std::pair<uint64_t, uint64_t>> freed;
	FrameNativo2::TakeFreed(freed);
	if (freed.empty()) return;
	// (v10) Copias de la epoca (v7) y memo BDA invalidadas: lo liberado puede reescribirse ya.
	FrameNativo2::g_epoch.fetch_add(1, std::memory_order_seq_cst);
	for (const auto& [address, size]: freed) {
		const int a = FrameNativo2::Touching(address, size);
		if (a < 0 || !FrameNativo2::g_excluded[a].load(std::memory_order_acquire)) continue;
		// (v10) Nada se marca CPU-modificado aqui (v9: 1976 MB/s re-subidos): solo se suelta lo que la GPU tenia, sin
		// descargarlo; la siguiente lectura copia (v7) o marca solo lo que lee.
		if (tracker.IsRegionGpuModified(address, size)) {
			discard(address, size);
			FrameNativo2::g_drained_gpu.fetch_add(1, std::memory_order_relaxed);
		}
		FrameNativo2::g_drained_ranges.fetch_add(1, std::memory_order_relaxed);
		FrameNativo2::g_drained_bytes.fetch_add(size, std::memory_order_relaxed);
	}
}
// (v13, =6) Solo el hilo de la GPU: bloques cuyas entradas BDA pasan al arena importado (entran sin vigilar) o vuelven a
// los buferes de la cache (vuelven a la vigilancia).
inline std::vector<std::pair<uint64_t, uint64_t>> g_fn2_bda_enter, g_fn2_bda_restore;
// (v11, =4) Hilo de la GPU: los bloques que vuelven a la vigilancia (AllocTag los dio a una instancia vigilada, o la red
// de seguridad vio una escritura de la GPU): la CPU los escribio sin que el tracker lo viera, asi que pasan enteros a
// CPU-modificados (lo que la GPU tuviera alli se suelta antes: nunca los dos estados a la vez).
template <typename Tracker, typename Discard>
void FrameNativo2DrainRewatch(Tracker& tracker, Discard&& discard) {
	if (!FrameNativo2::SoloLectura()) return;
	std::vector<std::pair<uint64_t, uint64_t>> back;
	FrameNativo2::TakeRewatch(back);
	for (const auto& [address, size]: back) {
		if (FrameNativo2::RoAny(address, size)) continue; // (AllocTag ya los volvio a dar a una de solo lectura)
		if (tracker.IsRegionGpuModified(address, size)) discard(address, size);
		tracker.MarkRegionAsCpuModified(address, size);
		if (FrameNativo2::Nativo() && g_fn2_bda_restore.size() < 65536) g_fn2_bda_restore.emplace_back(address, size); // (v13)
	}
}
// (v11b, =4) Hilo de la GPU: los bloques que AllocTag acaba de dar a una instancia de solo lectura. Lo que la GPU tuviera
// alli (de un dueno anterior) se suelta SIN descargar (el juego lo reescribe; descargarlo pisaba sus descriptores) y las
// texturas del rango se invalidan; en el primer fotograma nuevo (epoca de v7, gancho NextFrame) otra vez las texturas,
// por lo que el juego escribio despues de pedir el bloque. Las tablas/variantes (table-xpr, native-xpr) van por
// contenido leido de la memoria real y los samplers por valor: no guardan nada de estos bloques que invalidar.
// Texturas por invalidar pedidas donde la cache de texturas puede tener su cerrojo (ObtainBuffer desde
// ObtainBufferForImage): se invalidan despues, en EnsureBdaReadable (hilo de la GPU, sin ese cerrojo).
inline std::vector<std::pair<uint64_t, uint64_t>> g_fn2_tex_pending; // (solo el hilo de la GPU)
template <typename InvalidateTextures>
void FrameNativo2FlushTextures(InvalidateTextures&& textures) {
	if (g_fn2_tex_pending.empty()) return;
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	pending.swap(g_fn2_tex_pending);
	for (const auto& [address, size]: pending)
		if (FrameNativo2::RoAny(address, size)) textures(address, size);
}
// (v12, =5) Hilo de la GPU, sin el cerrojo de la cache de texturas (EnsureBdaReadable): de los bloques que entraron, solo
// los que tienen imagenes encima; de esos se invalidan solo si ninguna imagen del rango la pinto la GPU (sus bytes son
// del juego: el CPU los reescribe al reasignar el bloque). Las pintadas por la GPU no se tocan (nunca se resuben).
template <typename HasImages, typename HasGpuImages, typename InvalidateTextures>
void FrameNativo2FlushTexturesV12(HasImages&& has, HasGpuImages&& gpu, InvalidateTextures&& textures) {
	if (g_fn2_tex_pending.empty()) return;
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	pending.swap(g_fn2_tex_pending);
	for (const auto& [address, size]: pending) {
		if (!FrameNativo2::RoAny(address, size) || !has(address, size)) continue;
		FrameNativo2::g_v12_img_ranges.fetch_add(1, std::memory_order_relaxed);
		if (gpu(address, size)) {
			FrameNativo2::g_v12_img_gpu.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		textures(address, size);
		FrameNativo2::g_v12_img_inval.fetch_add(1, std::memory_order_relaxed);
	}
}
template <typename Tracker, typename Discard, typename InvalidateTextures>
void FrameNativo2DrainEnter(Tracker& tracker, Discard&& discard, InvalidateTextures&& textures) {
	if (!FrameNativo2::SoloLectura()) return;
	if (FrameNativo2::V12() || FrameNativo2::Nativo()) {
		// (v12, =5) Sin segunda invalidacion por fotograma: lo entregado no se reescribe hasta FreeTag (Ghidra, Alloc
		// 0x811f60). Lo de la GPU de un dueno anterior se suelta igual; las texturas solo se miran (FlushTexturesV12).
		std::vector<std::pair<uint64_t, uint64_t>> in;
		FrameNativo2::TakeEnter(in);
		for (const auto& [address, size]: in) {
			if (!FrameNativo2::RoAny(address, size)) continue;
			FrameNativo2::g_v12_enter_ranges.fetch_add(1, std::memory_order_relaxed);
			if (tracker.IsRegionGpuModified(address, size)) discard(address, size);
			if (g_fn2_tex_pending.size() < 65536) g_fn2_tex_pending.emplace_back(address, size);
			if (FrameNativo2::Nativo() && g_fn2_bda_enter.size() < 65536) g_fn2_bda_enter.emplace_back(address, size); // (v13)
		}
		(void)textures;
		return;
	}
	static std::vector<std::pair<uint64_t, uint64_t>> recent; // (solo el hilo de la GPU)
	static uint64_t                                    recent_frame = 0;
	const uint64_t                                     frame = FrameNativo2::g_frames.load(std::memory_order_acquire);
	if (frame != recent_frame) {
		for (const auto& [address, size]: recent) {
			if (!FrameNativo2::RoAny(address, size)) continue;
			textures(address, size);
			FrameNativo2::g_ro_epoch_invals.fetch_add(1, std::memory_order_relaxed);
		}
		recent.clear();
		recent_frame = frame;
	}
	std::vector<std::pair<uint64_t, uint64_t>> in;
	FrameNativo2::TakeEnter(in);
	for (const auto& [address, size]: in) {
		if (!FrameNativo2::RoAny(address, size)) continue; // (ya devuelto a la vigilancia)
		if (tracker.IsRegionGpuModified(address, size)) discard(address, size);
		textures(address, size);
		FrameNativo2::g_ro_epoch_invals.fetch_add(1, std::memory_order_relaxed);
		if (recent.size() < 65536) recent.emplace_back(address, size);
	}
}
FrameNativo2State& FrameNativo2S() {
	static FrameNativo2State state;
	return state;
}

// Imports arena i when it is fully mapped and every page of it read-write (NVIDIA refuses READONLY pages at the
// import: log 20261124-091857-518; a refused import is never used). True: imported now.
bool FrameNativo2Import(GraphicContext& graphics, CommandScheduler& scheduler, int i) {
	auto&          st    = FrameNativo2S();
	const uint64_t guest = FrameNativo2::Base(i);
	const uint64_t bytes = FrameNativo2::kArenaSize;
	FrameNativo2::g_tries.fetch_add(1, std::memory_order_relaxed);
	// (v3) Every early exit counted by reason (log 20261124-100322-504: 92 tries, 0 fails, 0 imports, no reason).
	const auto why = [](FrameNativo2::Reason r) {
		FrameNativo2::g_reason[r].fetch_add(1, std::memory_order_relaxed);
		return false;
	};
	if (guest == 0) return why(FrameNativo2::RNoBase);
	if (!Tiron7::g_extension.load(std::memory_order_relaxed)) return why(FrameNativo2::RNoExtension);
	if (!LibKernel::Memory::IsFullyMapped(guest, bytes)) return why(FrameNativo2::RNotMapped);
	// (FRAME-NATIVO2 v6, COPIA) Ghidra (VA del juego; eboot.raw = VA + 0xbdb0): NextFrame 0x811860 (archivo 0x81d610)
	// hace idx=([+0x90]+1)%[+0x94] y devuelve al padre (vtable+0x100) los trozos de ese indice sin esperar nada; lo llama
	// el bucle de fotograma (juego 0xa988e6, tras incl [+0x156fc]). Con 1 buffer (8 de 15 instancias) un trozo se
	// reescribe en el fotograma siguiente. Sin proteccion (v5 excluia el arena) esas escrituras no llegan al tracker:
	// un rango que la GPU escribio sigue "de la GPU", se lee de la cache con bytes viejos y su write-back pisa los del
	// juego (log 20261124-105755-202: [wb-guard] en 0x374780000, arena 2; S# basura 'unknown ratio' -> cuelgue). En
	// COPIA no hace falta importar ni desproteger: la copia al traducir lee el respaldo. El arena queda vigilado.
	if (!FrameNativo2::Live() && !FrameNativo2::Nativo()) { // (v13, =6: el import de verdad, como VIVO)
#if defined(_WIN32)
		// (v8, KYTY_FRAME_NATIVO2=3) Sin proteccion: excluido (ProtectTransient descarta sus protecciones) y RW ya.
		if (FrameNativo2::NoProt()) {
			FrameNativo2::g_excluded[i].store(true, std::memory_order_release);
			if (!LibKernel::Memory::RestoreGuestWritable(guest, bytes)) {
				FrameNativo2::g_excluded[i].store(false, std::memory_order_release);
				FrameNativo2::g_fails.fetch_add(1, std::memory_order_relaxed);
				return why(FrameNativo2::RRestore);
			}
			st.need_dirty[i] = true;
		}
#endif
		FrameNativo2::g_imported[i].store(true, std::memory_order_release);
		const int count = FrameNativo2::g_imported_count.fetch_add(1, std::memory_order_acq_rel) + 1;
		std::printf("[frame-nativo2] arena %d (0x%llx) en modo %s; %d de %d\n", i, static_cast<unsigned long long>(guest),
		            FrameNativo2::NoProt() ? "COPIA SIN PROTEGER (v8, =3): excluido, sin vigilar"
		                                   : "COPIA: sin importar ni desproteger (sigue vigilado)",
		            count, FrameNativo2::kArenas);
		std::fflush(stdout);
		return true;
	}
#if defined(_WIN32)
	// (FRAME-NATIVO2 v2) RAIZ and the trackers protect the arena long before the first try (log 20261124-093927-391:
	// 52 tries, 0 imported). So: excluded first (ProtectTransient drops every later protection of it, under the
	// address space lock that this unprotect also takes), then made read-write through the guest's own protection
	// path (RAIZ puts its kept bytes in first), then imported while read-write. Never protected again.
	FrameNativo2::g_excluded[i].store(true, std::memory_order_release);
	const auto unexclude = [&] {
		// (Import refused: the arena back to the trackers; Obtain marks it CPU-modified, so nothing read stale.)
		FrameNativo2::g_excluded[i].store(false, std::memory_order_release);
		st.need_dirty[i] = true;
		return false;
	};
	if (!LibKernel::Memory::RestoreGuestWritable(guest, bytes)) {
		FrameNativo2::g_reason[FrameNativo2::RRestore].fetch_add(1, std::memory_order_relaxed);
		FrameNativo2::g_fails.fetch_add(1, std::memory_order_relaxed);
		if (st.refusals_printed++ < 8) {
			std::printf("[frame-nativo2] arena %d (0x%llx): no se pudo quitar la proteccion (RestoreGuestWritable)\n", i,
			            static_cast<unsigned long long>(guest));
			std::fflush(stdout);
		}
		return unexclude();
	}
	st.need_dirty[i] = true; // (the trackers believed these pages protected: re-read from memory on next use)
	const auto* source = LibKernel::Memory::TryGetBackingPointer(guest, bytes); // (after the unprotect: RAIZ refuses protected pages)
	if (source == nullptr) {
		FrameNativo2::g_reason[FrameNativo2::RNoBacking].fetch_add(1, std::memory_order_relaxed);
		return unexclude();
	}
	const auto host = reinterpret_cast<uint64_t>(source);
	for (uint64_t at = host; at < host + bytes;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0) {
			FrameNativo2::g_reason[FrameNativo2::RQuery].fetch_add(1, std::memory_order_relaxed);
			return unexclude();
		}
		if (info.State != MEM_COMMIT || (info.Protect & 0xffu) != PAGE_READWRITE) {
			FrameNativo2::g_reason[FrameNativo2::RNotRw].fetch_add(1, std::memory_order_relaxed);
			FrameNativo2::g_fails.fetch_add(1, std::memory_order_relaxed);
			if (st.refusals_printed++ < 8) {
				std::printf("[frame-nativo2] arena %d (0x%llx): paginas no RW tras quitar la proteccion (0x%lx)\n", i,
				            static_cast<unsigned long long>(guest), static_cast<unsigned long>(info.Protect));
				std::fflush(stdout);
			}
			return unexclude();
		}
		at = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
	}
#else
	const uint8_t* source = nullptr;
	const uint64_t host   = 0;
	return false;
#endif
	auto&      vk     = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto device = static_cast<VkDevice>(graphics.device);
	VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_limits {};
	host_limits.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
	VkPhysicalDeviceProperties2 props {};
	props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props.pNext = &host_limits;
	vk.vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(graphics.physical_device), &props);
	const uint64_t alignment = host_limits.minImportedHostPointerAlignment;
	VkBuffer       buffer    = VK_NULL_HANDLE;
	VkDeviceMemory memory    = VK_NULL_HANDLE;
	const auto     start     = std::chrono::steady_clock::now();
	const auto     fail      = [&](const char* stage, VkResult result) {
        if (buffer != VK_NULL_HANDLE) vk.vkDestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) vk.vkFreeMemory(device, memory, nullptr);
        FrameNativo2::g_excluded[i].store(false, std::memory_order_release); // (FRAME-NATIVO2 v2: back to the trackers)
        FrameNativo2::g_fails.fetch_add(1, std::memory_order_relaxed);
        FrameNativo2::g_reason[FrameNativo2::RVulkan].fetch_add(1, std::memory_order_relaxed);
        if (st.refusals_printed++ < 8) {
            std::printf("[frame-nativo2] arena %d (0x%llx) NO importado en %s (VkResult %d): sigue el camino de antes\n", i,
		                static_cast<unsigned long long>(guest), stage, static_cast<int>(result));
            std::fflush(stdout);
        }
        return false;
	};
	if (alignment == 0 || (alignment & (alignment - 1)) != 0 || (host & (alignment - 1)) != 0 ||
	    (bytes & (alignment - 1)) != 0 || vk.vkGetMemoryHostPointerPropertiesEXT == nullptr)
		return fail("alineacion", VK_SUCCESS);
	VkMemoryHostPointerPropertiesEXT host_props {};
	host_props.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	if (const auto result = vk.vkGetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	                                                               source, &host_props);
	    result != VK_SUCCESS || host_props.memoryTypeBits == 0)
		return fail("vkGetMemoryHostPointerPropertiesEXT", result);
	VkExternalMemoryBufferCreateInfo external {};
	external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.pNext       = &external;
	buffer_info.size        = bytes;
	buffer_info.usage       = static_cast<VkBufferUsageFlags>(AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	const uint32_t families[] {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) { // (as Buffer::Buffer)
		buffer_info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families;
	}
	if (const auto result = vk.vkCreateBuffer(device, &buffer_info, nullptr, &buffer); result != VK_SUCCESS)
		return fail("vkCreateBuffer", result);
	VkMemoryRequirements requirements {};
	vk.vkGetBufferMemoryRequirements(device, buffer, &requirements);
	const uint32_t types = requirements.memoryTypeBits & host_props.memoryTypeBits;
	VkPhysicalDeviceMemoryProperties memory_props {};
	vk.vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(graphics.physical_device), &memory_props);
	uint32_t type = UINT32_MAX;
	for (uint32_t k = 0; k < memory_props.memoryTypeCount && type == UINT32_MAX; ++k)
		if ((types & (1u << k)) != 0 && (memory_props.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
			type = k;
	if (type == UINT32_MAX || requirements.size > bytes) return fail("tipo de memoria", VK_SUCCESS);
	VkMemoryAllocateFlagsInfo flags_info {};
	flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkImportMemoryHostPointerInfoEXT import_info {};
	import_info.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
	import_info.pNext        = &flags_info;
	import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import_info.pHostPointer = const_cast<uint8_t*>(source);
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.pNext           = &import_info;
	allocate.allocationSize  = bytes;
	allocate.memoryTypeIndex = type;
	VramDrv::Scope vram_drv(VramDrv::CatHostImport); // (VRAM-DRV)
	if (const auto result = vk.vkAllocateMemory(device, &allocate, nullptr, &memory); result != VK_SUCCESS)
		return fail("vkAllocateMemory", result);
	if (const auto result = vk.vkBindBufferMemory(device, buffer, memory, 0); result != VK_SUCCESS)
		return fail("vkBindBufferMemory", result);
	VkBufferDeviceAddressInfo address_info {};
	address_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = buffer;
	const VkDeviceAddress address = vk.vkGetBufferDeviceAddress(device, &address_info);
	if (address == 0) return fail("vkGetBufferDeviceAddress", VK_SUCCESS);
	auto& arena      = st.arena[i];
	arena.buffer     = new Buffer(graphics, scheduler, guest, vk::Buffer(buffer), vk::DeviceMemory(memory), bytes);
	arena.guest_base = guest;
	arena.device     = address;
	FrameNativo2::g_imported[i].store(true, std::memory_order_release);
	const int count = FrameNativo2::g_imported_count.fetch_add(1, std::memory_order_acq_rel) + 1;
	std::printf("[frame-nativo2] arena %d importado: guest 0x%llx, %llu MiB (host %p) en %.1f ms; %d de %d arenas DefaultTagg "
	            "nativos; %s\n",
	            i, static_cast<unsigned long long>(guest), static_cast<unsigned long long>(bytes >> 20u),
	            static_cast<const void*>(source),
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), count,
	            FrameNativo2::kArenas,
	            FrameNativo2::Nativo() ? "modo NATIVO (=6): bloques de solo lectura leidos en vivo, cuarentena de FreeTag, etiquetas al traducir"
	            : FrameNativo2::Live() ? "modo VIVO: etiquetas de fin de GPU al completar (C5)"
	                                   : "modo COPIA: lecturas copiadas al traducir, etiquetas al traducir (sin C5, sin espera)");
	std::fflush(stdout);
	return true;
}

// (FRAME-NATIVO2 v3) The arenas' addresses from the game's allocator list ([memmap] snapshot, Ghidra-read allocator
// objects: DemonsSoulsMemMap), not fixed: log 20261124-100322-504 had arenas 3 and 4 at 0x36f800000 / 0x3b1900000.
// Every "DefaultTagg" range of exactly kArenaSize not held yet takes a free slot (once; the arenas are never
// unmapped). Render thread; the snapshot read is lock-free.
void FrameNativo2Discover() {
	for (int i = 0; i < FrameNativo2::kArenas; ++i)
		if (FrameNativo2::Base(i) == 0) goto look;
	return;
look:
	static const char* const labels[] = {"DefaultTagg"};
	uint64_t ranges[2 * 32];
	const size_t n = Loader::DemonsSoulsMemMap::RangesOf(labels, 1, ranges, 32);
	if (n == SIZE_MAX) return;
	for (size_t r = 0; r < n; ++r) {
		const uint64_t b = ranges[2 * r], e = ranges[2 * r + 1];
		if (e - b != FrameNativo2::kArenaSize) continue;
		bool known = false;
		int  slot  = -1;
		for (int i = 0; i < FrameNativo2::kArenas; ++i) {
			const uint64_t have = FrameNativo2::Base(i);
			if (have == b) known = true;
			if (have == 0 && slot < 0) slot = i;
		}
		if (known || slot < 0) continue;
		FrameNativo2::g_base[slot].store(b, std::memory_order_release);
		std::printf("[frame-nativo2] arena %d encontrado en el asignador del juego: guest 0x%llx, %llu MiB\n", slot,
		            static_cast<unsigned long long>(b), static_cast<unsigned long long>((e - b) >> 20u));
		std::fflush(stdout);
	}
}

// The [frame-nativo2] line, every 5 s (render thread).
void FrameNativo2Report() {
	auto&      st  = FrameNativo2S();
	const auto now = std::chrono::steady_clock::now();
	const auto dt  = std::chrono::duration<double>(now - st.last_report).count();
	if (dt < 5.0) return;
	st.last_report     = now;
	const auto hits    = FrameNativo2::g_hits.load(std::memory_order_relaxed);
	const auto hbytes  = FrameNativo2::g_hit_bytes.load(std::memory_order_relaxed);
	const bool map     = FrameNativo::g_map.load(std::memory_order_acquire) != nullptr;
	const auto prot    = FrameNativo::g_calls[FrameNativo::CDefaultTagg].load(std::memory_order_relaxed);
	char       prot_text[64];
	if (map && st.prot_base)
		std::snprintf(prot_text, sizeof(prot_text), "%.0f/s", static_cast<double>(prot - st.last_prot) / dt);
	else
		std::snprintf(prot_text, sizeof(prot_text), "sin mapa");
	int      n   = 0;
	uint64_t mib = 0;
	for (int i = 0; i < FrameNativo2::kArenas; ++i)
		if (FrameNativo2::g_imported[i].load(std::memory_order_acquire)) {
			n++;
			mib += FrameNativo2::kArenaSize >> 20u;
		}
	std::printf("[frame-nativo2] importados %d/%d arenas DefaultTagg (%llu MiB); lecturas servidas sin vigilar ni resubir "
	            "%.0f/s (%.1f MB/s que antes se copiaban y protegian); protecciones DefaultTagg ahora %s; intentos %llu, "
	            "fallos %llu; protecciones descartadas (arena excluido) %llu; motivos sin importar: sin direccion %llu, sin "
	            "VK_EXT_external_memory_host %llu, no mapeado %llu, RestoreGuestWritable %llu, sin puntero (TryGetBackingPointer) "
	            "%llu, VirtualQuery %llu, no RW %llu, Vulkan %llu; arenas 0x%llx 0x%llx 0x%llx 0x%llx\n",
	            n, FrameNativo2::kArenas, static_cast<unsigned long long>(mib), static_cast<double>(hits - st.last_hits) / dt,
	            static_cast<double>(hbytes - st.last_bytes) / dt / 1e6, prot_text,
	            static_cast<unsigned long long>(FrameNativo2::g_tries.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_fails.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_skipped.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RNoBase].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RNoExtension].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RNotMapped].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RRestore].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RNoBacking].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RQuery].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RNotRw].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_reason[FrameNativo2::RVulkan].load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::Base(0)), static_cast<unsigned long long>(FrameNativo2::Base(1)),
	            static_cast<unsigned long long>(FrameNativo2::Base(2)), static_cast<unsigned long long>(FrameNativo2::Base(3)));
	std::printf("[frame-nativo2] modo %s: copiadas al traducir %llu (%.1f MB en total), al camino de la cache por anilla "
	            "llena %llu, rangos BDA del arena subidos al traducir %llu; etiquetas %s\n",
	            FrameNativo2::Live() ? "VIVO (KYTY_FRAME_NATIVO2=2)" : "COPIA (KYTY_FRAME_NATIVO2=1)",
	            static_cast<unsigned long long>(FrameNativo2::g_copies.load(std::memory_order_relaxed)),
	            static_cast<double>(FrameNativo2::g_copy_bytes.load(std::memory_order_relaxed)) / 1e6,
	            static_cast<unsigned long long>(FrameNativo2::g_copy_fallbacks.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_bda_skips.load(std::memory_order_relaxed)),
	            FrameNativo2::Active() ? "al completar la GPU (C5)" : "al traducir (sin espera)");
	{
		// (v7) Cost of the COPIA path per game frame (frame-loop NextFrame calls; without the hook: per second).
		static uint64_t last_ns = 0, last_frames = 0, last_copies = 0, last_cbytes = 0, last_reuses = 0, last_rbytes = 0;
		const uint64_t  ns     = FrameNativo2::g_copy_ns.load(std::memory_order_relaxed);
		const uint64_t  frames = FrameNativo2::g_frames.load(std::memory_order_relaxed);
		const uint64_t  cps    = FrameNativo2::g_copies.load(std::memory_order_relaxed);
		const uint64_t  cby    = FrameNativo2::g_copy_bytes.load(std::memory_order_relaxed);
		const uint64_t  rus    = FrameNativo2::g_reuses.load(std::memory_order_relaxed);
		const uint64_t  rby    = FrameNativo2::g_reuse_bytes.load(std::memory_order_relaxed);
		const double    fr     = frames > last_frames ? static_cast<double>(frames - last_frames) : dt;
		std::printf("[frame-nativo2] coste COPIA %.2f ms/frame (%s %.0f); por frame: copias %.0f (%.2f MB), reutilizadas %.0f "
		            "(%.2f MB ahorrados); gancho NextFrame 0x811860 %s, epocas %llu\n",
		            static_cast<double>(ns - last_ns) / 1e6 / fr, frames > last_frames ? "frames" : "SIN GANCHO: por segundo,",
		            fr, static_cast<double>(cps - last_copies) / fr, static_cast<double>(cby - last_cbytes) / fr / 1e6,
		            static_cast<double>(rus - last_reuses) / fr, static_cast<double>(rby - last_rbytes) / fr / 1e6,
		            FrameNativo2::g_hooked.load(std::memory_order_relaxed) ? "puesto" : "NO puesto",
		            static_cast<unsigned long long>(FrameNativo2::g_epoch.load(std::memory_order_relaxed)));
		last_ns = ns, last_frames = frames, last_copies = cps, last_cbytes = cby, last_reuses = rus, last_rbytes = rby;
	}
	if (FrameNativo2::NoProt()) {
		// (v8) Protecciones/s del arena (deberian ser ~0) e invalidaciones por NextFrame.
		static uint64_t lf = 0, lr = 0, lb = 0, lw = 0, ls = 0;
		const uint64_t  f = FrameNativo2::g_frames.load(std::memory_order_relaxed); // (v10: NextFrame del bucle, gancho 0x811860)
		const uint64_t  r = FrameNativo2::g_inval_ranges.load(std::memory_order_relaxed);
		const uint64_t  b = FrameNativo2::g_inval_bytes.load(std::memory_order_relaxed);
		const uint64_t  w = FrameNativo2::g_gpu_writes.load(std::memory_order_relaxed);
		const uint64_t  s = FrameNativo2::g_skipped.load(std::memory_order_relaxed);
		const double    nf = f > lf ? static_cast<double>(f - lf) : 1.0;
		std::printf("[frame-nativo2 v8] SIN PROTEGER: protecciones DefaultTagg %s (descartadas %.0f/s); NextFrame vistos %.0f/s; "
		            "invalidaciones por NextFrame: %.1f rangos (%.2f MB) de escrituras GPU; escrituras GPU apuntadas %.0f/s\n",
		            prot_text, static_cast<double>(s - ls) / dt, static_cast<double>(f - lf) / dt,
		            static_cast<double>(r - lr) / nf, static_cast<double>(b - lb) / nf / 1e6, static_cast<double>(w - lw) / dt);
		lf = f, lr = r, lb = b, lw = w, ls = s;
		// (v9) FreeTag (juego 0x7fe540): llamadas, con bloques, rangos exactos liberados y descartados en el hilo GPU.
		static uint64_t lc = 0, lfo = 0, lfr = 0, lfb = 0, ldr = 0, ldb = 0, ldg = 0;
		const uint64_t  c  = FrameNativo2::g_free_calls.load(std::memory_order_relaxed);
		const uint64_t  fo = FrameNativo2::g_free_found.load(std::memory_order_relaxed);
		const uint64_t  fr = FrameNativo2::g_freed_ranges.load(std::memory_order_relaxed);
		const uint64_t  fb = FrameNativo2::g_freed_bytes.load(std::memory_order_relaxed);
		const uint64_t  dr = FrameNativo2::g_drained_ranges.load(std::memory_order_relaxed);
		const uint64_t  db = FrameNativo2::g_drained_bytes.load(std::memory_order_relaxed);
		const uint64_t  dg = FrameNativo2::g_drained_gpu.load(std::memory_order_relaxed);
		const int       vt = FrameNativo2::g_vtable_ok.load(std::memory_order_relaxed);
		std::printf("[frame-nativo2 v9] FreeTag 0x7fe540 (vtable+0x100 %s): %.0f llamadas/s (%.0f con bloques); liberados "
		            "%.0f rangos/s (%.1f MB/s); en el arena, descartados/CPU %.0f rangos/s (%.1f MB/s), con escrituras GPU %.0f/s\n",
		            vt == 1 ? "comprobada" : vt == 0 ? "NO COINCIDE" : "sin comprobar", static_cast<double>(c - lc) / dt,
		            static_cast<double>(fo - lfo) / dt, static_cast<double>(fr - lfr) / dt,
		            static_cast<double>(fb - lfb) / dt / 1e6, static_cast<double>(dr - ldr) / dt,
		            static_cast<double>(db - ldb) / dt / 1e6, static_cast<double>(dg - ldg) / dt);
		lc = c, lfo = fo, lfr = fr, lfb = fb, ldr = dr, ldb = db, ldg = dg;
		// (v10) Lo marcado CPU-modificado (solo rangos usados) y lo re-subido por BDA, en MB/s.
		static uint64_t lm = 0, lu = 0, lg = 0, lmm = 0;
		const uint64_t  m  = FrameNativo2::g_marked_bytes.load(std::memory_order_relaxed);
		const uint64_t  u  = FrameNativo2::g_reupload_bytes.load(std::memory_order_relaxed);
		const uint64_t  g  = FrameNativo2::g_dropped_gpu_bytes.load(std::memory_order_relaxed);
		const uint64_t  mm = FrameNativo2::g_bda_memo.load(std::memory_order_relaxed);
		std::printf("[frame-nativo2 v10] marcados CPU %.1f MB/s (solo lo usado); re-subidos BDA %.1f MB/s (ahorrados por "
		            "epoca %.0f/s); escrituras GPU soltadas sin descargar %.1f MB/s\n",
		            static_cast<double>(m - lm) / dt / 1e6, static_cast<double>(u - lu) / dt / 1e6,
		            static_cast<double>(mm - lmm) / dt, static_cast<double>(g - lg) / dt / 1e6);
		lm = m, lu = u, lg = g, lmm = mm;
	}
	if (FrameNativo2::SoloLectura()) {
		// (v11, =4) Protecciones DefaultTagg/s, bloques sin vigilar, escrituras GPU inesperadas (debe ser 0), coste copia.
		static uint64_t l_ns = 0, l_fr = 0, l_sk = 0, l_un = 0, l_al = 0, l_ab = 0, l_rw = 0, l_rb = 0, l_cp = 0, l_cb = 0;
		const uint64_t  ns = FrameNativo2::g_copy_ns.load(std::memory_order_relaxed);
		const uint64_t  fr = FrameNativo2::g_frames.load(std::memory_order_relaxed);
		const uint64_t  sk = FrameNativo2::g_skipped.load(std::memory_order_relaxed);
		const uint64_t  un = FrameNativo2::g_ro_unexpected.load(std::memory_order_relaxed);
		const uint64_t  al = FrameNativo2::g_ro_allocs.load(std::memory_order_relaxed);
		const uint64_t  ab = FrameNativo2::g_ro_alloc_bytes.load(std::memory_order_relaxed);
		const uint64_t  rw = FrameNativo2::g_ro_rewatch.load(std::memory_order_relaxed);
		const uint64_t  rb = FrameNativo2::g_ro_rewatch_bytes.load(std::memory_order_relaxed);
		const uint64_t  cp = FrameNativo2::g_copies.load(std::memory_order_relaxed);
		const uint64_t  cb = FrameNativo2::g_copy_bytes.load(std::memory_order_relaxed);
		const double    nf = fr > l_fr ? static_cast<double>(fr - l_fr) : dt;
		const int64_t   gr = FrameNativo2::g_ro_granules.load(std::memory_order_relaxed);
		static uint64_t l_dr = 0, l_ds = 0, l_ei = 0; // (v11b)
		const uint64_t  dr = FrameNativo2::g_ro_desc_reads.load(std::memory_order_relaxed);
		const uint64_t  ds = FrameNativo2::g_ro_sync_skips.load(std::memory_order_relaxed);
		const uint64_t  ei = FrameNativo2::g_ro_epoch_invals.load(std::memory_order_relaxed);
		std::printf("[frame-nativo2 v11] SOLO LECTURA (=4): protecciones DefaultTagg %s (descartadas en bloques sin vigilar "
		            "%.0f/s); bloques sin vigilar %.1f MiB; escrituras GPU inesperadas %llu (total %llu, debe ser 0); copia "
		            "%.2f ms/frame (%.0f copias, %.2f MB por frame); AllocTag solo lectura %.0f/s (%.1f MB/s), vuelven a "
		            "vigilancia %.0f/s (%.1f MB/s); RestoreGuestWritable fallidos %llu; lista KYTY_FRAME_NATIVO2_RO: %d "
		            "llamadores, %d tags; (v11b) descriptores leidos de bloques sin vigilar %.0f/s (memoria real; descargas "
		            "GPU evitadas %.0f/s); invalidaciones de cache por epoca %.0f/s\n",
		            prot_text, static_cast<double>(sk - l_sk) / dt, static_cast<double>(gr > 0 ? gr : 0) * 0.5,
		            static_cast<unsigned long long>(un - l_un), static_cast<unsigned long long>(un),
		            static_cast<double>(ns - l_ns) / 1e6 / nf, static_cast<double>(cp - l_cp) / nf,
		            static_cast<double>(cb - l_cb) / nf / 1e6, static_cast<double>(al - l_al) / dt,
		            static_cast<double>(ab - l_ab) / dt / 1e6, static_cast<double>(rw - l_rw) / dt,
		            static_cast<double>(rb - l_rb) / dt / 1e6,
		            static_cast<unsigned long long>(FrameNativo2::g_ro_restore_fail.load(std::memory_order_relaxed)),
		            FrameNativo2::Ro().nc, FrameNativo2::Ro().nt, static_cast<double>(dr - l_dr) / dt,
		            static_cast<double>(ds - l_ds) / dt, static_cast<double>(ei - l_ei) / dt);
		l_dr = dr, l_ds = ds, l_ei = ei;
		if (FrameNativo2::V12()) {
			// (v12, =5) Una linea: protecciones/s, bloques sin vigilar, aciertos/fallos por epoca de bloque, MB/frame,
			// ventanas/s ([raiz]: deben ser ~0 nuevas), imagenes en los bloques, escrituras GPU inesperadas, ms/frame.
			static uint64_t v_ch = 0, v_cm = 0, v_bh = 0, v_bm = 0, v_bb = 0, v_bs = 0, v_en = 0, v_ir = 0, v_ii = 0, v_ig = 0;
			static uint64_t v_win = 0, v_wp = 0;
			const uint64_t  ch  = FrameNativo2::g_v12_copy_hits.load(std::memory_order_relaxed);
			const uint64_t  cm  = FrameNativo2::g_v12_copy_miss.load(std::memory_order_relaxed);
			const uint64_t  bh  = FrameNativo2::g_v12_bda_hits.load(std::memory_order_relaxed);
			const uint64_t  bm  = FrameNativo2::g_v12_bda_miss.load(std::memory_order_relaxed);
			const uint64_t  bb  = FrameNativo2::g_v12_bda_bytes.load(std::memory_order_relaxed);
			const uint64_t  bs  = FrameNativo2::g_v12_bda_split.load(std::memory_order_relaxed);
			const uint64_t  en  = FrameNativo2::g_v12_enter_ranges.load(std::memory_order_relaxed);
			const uint64_t  ir  = FrameNativo2::g_v12_img_ranges.load(std::memory_order_relaxed);
			const uint64_t  ii  = FrameNativo2::g_v12_img_inval.load(std::memory_order_relaxed);
			const uint64_t  ig  = FrameNativo2::g_v12_img_gpu.load(std::memory_order_relaxed);
			const uint64_t  win = ::Raiz::g_counters.read_windows.load(std::memory_order_relaxed) +
			                     ::Raiz::g_counters.write_windows.load(std::memory_order_relaxed);
			const uint64_t wp  = ::Raiz::g_counters.window_protects.load(std::memory_order_relaxed);
			const double   fps = fr > l_fr ? static_cast<double>(fr - l_fr) : 0.0;
			std::printf("[frame-nativo2 v12] POR BLOQUES (=5): protecciones DefaultTagg %s; bloques sin vigilar %.1f MiB; "
			            "copias V# por epoca de bloque: aciertos %.0f/s, fallos %.0f/s (%.2f MB/frame, %.2f ms/frame); BDA: "
			            "aciertos %.0f/s, fallos %.0f/s, %.2f MB/frame subidos, rangos partidos %.0f/s; ventanas %.0f/s "
			            "(VirtualProtect de ventanas %.0f/s); bloques que entran %.0f/s, con imagenes %.0f/s (invalidadas %.0f/s, "
			            "pintadas por la GPU sin tocar %.0f/s; total con imagenes %llu); escrituras GPU inesperadas %llu (total "
			            "%llu, debe ser 0); fotograma %.2f ms (%.0f frames del bucle)\n",
			            prot_text, static_cast<double>(gr > 0 ? gr : 0) * 0.5, static_cast<double>(ch - v_ch) / dt,
			            static_cast<double>(cm - v_cm) / dt, static_cast<double>(cb - l_cb) / (fps > 0 ? fps : dt) / 1e6,
			            static_cast<double>(ns - l_ns) / 1e6 / (fps > 0 ? fps : dt), static_cast<double>(bh - v_bh) / dt,
			            static_cast<double>(bm - v_bm) / dt, static_cast<double>(bb - v_bb) / (fps > 0 ? fps : dt) / 1e6,
			            static_cast<double>(bs - v_bs) / dt, static_cast<double>(win - v_win) / dt,
			            static_cast<double>(wp - v_wp) / dt, static_cast<double>(en - v_en) / dt,
			            static_cast<double>(ir - v_ir) / dt, static_cast<double>(ii - v_ii) / dt,
			            static_cast<double>(ig - v_ig) / dt, static_cast<unsigned long long>(ir),
			            static_cast<unsigned long long>(un - l_un), static_cast<unsigned long long>(un),
			            fps > 0 ? dt * 1000.0 / fps : 0.0, fps);
			v_ch = ch, v_cm = cm, v_bh = bh, v_bm = bm, v_bb = bb, v_bs = bs, v_en = en, v_ir = ir, v_ii = ii, v_ig = ig;
			v_win = win, v_wp = wp;
		}
		l_ns = ns, l_fr = fr, l_sk = sk, l_un = un, l_al = al, l_ab = ab, l_rw = rw, l_rb = rb, l_cp = cp, l_cb = cb;
	}
	std::fflush(stdout);
	st.last_hits  = hits;
	st.last_bytes = hbytes;
	st.last_prot  = prot;
	st.prot_base  = map;
}
} // namespace

// (FRAME-NATIVO2) Render thread: the arenas not imported yet tried (each at most every 2 s), their BDA entries written,
// the report. Then: the imported buffer holding [vaddr, vaddr + size) for a read, or null.
std::pair<Buffer*, uint64_t> BufferCache::FrameNativo2Obtain(uint64_t vaddr, uint64_t size) {
	auto& st = FrameNativo2S();
	if (!AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation() && CommandScheduler::ThreadRecorder() == nullptr &&
	    GuestGpu::IsGpuThread()) {
		if ((!FrameNativo2::SoloLectura() || FrameNativo2::Nativo()) && // (v11, =4: nada que importar; v13 =6: si)
		    FrameNativo2::g_imported_count.load(std::memory_order_relaxed) < FrameNativo2::kArenas) {
			const auto now = std::chrono::steady_clock::now();
			FrameNativo2Discover();
			for (int i = 0; i < FrameNativo2::kArenas; ++i) {
				if (FrameNativo2::g_imported[i].load(std::memory_order_relaxed) || now < st.next_try[i]) continue;
				st.next_try[i] = now + std::chrono::seconds(2);
				if (FrameNativo2Import(m_graphics, m_scheduler, i)) {
					if (FrameNativo2::Live()) FrameNativo2WriteBda(i);
					// (v13, =6) El arena vuelve a la vigilancia normal entero (need_dirty abajo lo marca CPU-modificado): solo
					// los bloques de solo lectura que AllocTag de desde ahora salen de ella (OnAllocTagRo pide Find >= 0).
					if (FrameNativo2::Nativo()) FrameNativo2::g_excluded[i].store(false, std::memory_order_release);
				}
				// (v2) The unprotect hid CPU writes from the trackers: the whole arena CPU-modified once.
				if (st.need_dirty[i]) {
					st.need_dirty[i] = false;
					// (v4, log 20261124-101723-073: EXIT "CPU dirty state conflicts with GPU dirty state" here) Pages of the
					// arena the GPU wrote are downloaded into guest memory first and their GPU state dropped, so the
					// whole arena is CPU-modified with no page in both states (then the imported buffer reads them).
					const uint64_t base = FrameNativo2::Base(i);
					const uint64_t size = FrameNativo2::kArenaSize;
					if (m_memory_tracker.IsRegionGpuModified(base, size)) {
						DrainGuestReadback(base, size);
						std::vector<DownloadCopy> copies;
						if (m_gpu_modified_ranges.Intersects(base, size)) {
							m_gpu_modified_ranges.ForEachIntersection(base, size, [&](RangeSet::Range range) {
								// (per tracker page: the buffer of that page alone, as DiscardGpuWrites)
								for (uint64_t at = range.address; at < range.address + range.size;) {
									const uint64_t page = at & ~(TRACKER_PAGE_SIZE - 1);
									const uint64_t stop = std::min(range.address + range.size, page + TRACKER_PAGE_SIZE);
									copies.push_back({nullptr, 0, at, stop - at});
									at = stop;
								}
							});
						}
						for (auto& copy: copies) {
							auto& buffer = m_slot_buffers[FindBuffer(copy.address & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE)];
							copy          = {&buffer, buffer.Offset(copy.address), copy.address, copy.size};
						}
						if (!copies.empty()) {
							LiveCounters::Add(LiveCounters::SyncDownloads);
							DownloadBufferMemory(copies);
						}
						{
							const std::unique_lock lock(m_gpu_modified_mutex);
							m_gpu_modified_ranges.Subtract(base, size);
							MoveWrittenObtainEpoch();
							++m_gpu_coverage_epoch;
							Trad1StampCoverage(base, size);
						}
						InvalidateCopyFeedback(base, size);
						m_memory_tracker.UnmarkRegionAsGpuModified(base, size);
					}
					m_memory_tracker.MarkRegionAsCpuModified(base, size);
				}
			}
		}
		// (v8) Nuevo NextFrame del bucle: lo que la GPU escribio en el arena en fotogramas anteriores se descarta (el juego
		// ya puede reescribirlo sin que nadie lo vea): pasa a la CPU y la siguiente lectura sube los bytes del juego.
		// (v9) Ya no "por fotograma" (v8 descartaba tambien lo de fotogramas aun vivos, y sin gancho no descartaba nada:
		// log 20261124-120036-175): solo los bloques que FreeTag acaba de liberar.
		if (FrameNativo2::NoProt() || FrameNativo2::SoloLectura())
			FrameNativo2DrainFreed(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); });
		FrameNativo2DrainRewatch(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); });
		FrameNativo2DrainEnter(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); },
		                       [](uint64_t a, uint64_t s) {
			                       if (g_fn2_tex_pending.size() < 65536) g_fn2_tex_pending.emplace_back(a, s);
		                       });
		if (FrameNativo2::Nativo()) FrameNativo2Native6();
		FrameNativo2Report();
	}
	// (v11, =4) Solo rangos enteros dentro de bloques sin vigilar (instancias de solo lectura): copia por epoca (v7).
	// Lo demas del arena (la instancia tag 41, que escribe la GPU) va por la cache como sin FRAME-NATIVO2.
	// (v13, =6) Rangos enteros sin vigilar dentro de un arena importado: el VkBuffer del arena, en vivo.
	const int i = FrameNativo2::Nativo()      ? (FrameNativo2::RoAll(vaddr, size) ? FrameNativo2::Find(vaddr, size) : -1)
	              : FrameNativo2::SoloLectura() ? (FrameNativo2::RoAll(vaddr, size) ? 0 : -1)
	                                            : FrameNativo2::Find(vaddr, size);
	if (i < 0) return {nullptr, 0};
	// (Bytes the GPU wrote there live in a cache buffer: that path.)
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) return {nullptr, 0};
	// (v5, COPIA) The bytes as they are now, at translation, into the upload ring (never waits on it: a full ring
	// takes the cache's way, the range marked CPU-modified by GetBuffer).
	if (!FrameNativo2::Live() && !FrameNativo2::Nativo()) {
		const auto t0     = std::chrono::steady_clock::now();
		auto&      upload = GetShaderUploadBuffer();
		// (v7) The same range already copied in this NextFrame epoch and this tick of this ring: that copy (frame-nativo2.h).
		struct Memo {
			uint64_t      vaddr = 0, size = 0, epoch = 0, tick = 0, offset = 0;
			StreamBuffer* ring  = nullptr;
		};
		static thread_local Memo memo[4096];
		const bool     hooked = FrameNativo2::g_hooked.load(std::memory_order_acquire);
		// (v12, =5) La epoca de los bloques del rango (no la global): cambia solo si uno de ellos se libera o reasigna.
		uint64_t       gen12  = 0;
		const bool     v12    = FrameNativo2::V12() && FrameNativo2::RoGen(vaddr, size, &gen12);
		if (FrameNativo2::V12() && !v12) return {nullptr, 0}; // (un bloque cambio de estado: el camino de la cache)
		const uint64_t epoch  = v12 ? gen12 : FrameNativo2::g_epoch.load(std::memory_order_seq_cst);
		const uint64_t tick   = m_scheduler.CurrentTick();
		auto&          slot   = memo[((vaddr >> 4u) ^ (vaddr >> 16u) ^ (size * 0x9e3779b1ull)) & 4095u];
		if (hooked && slot.ring == &upload && slot.vaddr == vaddr && slot.size == size && slot.epoch == epoch &&
		    slot.tick == tick) {
			FrameNativo2::g_hits.fetch_add(1, std::memory_order_relaxed);
			FrameNativo2::g_reuses.fetch_add(1, std::memory_order_relaxed);
			if (v12) FrameNativo2::g_v12_copy_hits.fetch_add(1, std::memory_order_relaxed);
			FrameNativo2::g_reuse_bytes.fetch_add(size, std::memory_order_relaxed);
			FrameNativo2::g_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                      std::chrono::steady_clock::now() - t0).count()),
			                                  std::memory_order_relaxed);
			return {&upload, slot.offset};
		}
		const auto alignment = std::max<uint64_t>(
		    std::max<uint64_t>(m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment,
		                       m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment),
		    16);
		auto [mapped, offset] = upload.Map(size, alignment, false);
		if (mapped == nullptr || !Libs::LibKernel::Memory::TryReadBackingToHost(vaddr, mapped, size)) {
			FrameNativo2::g_copy_fallbacks.fetch_add(1, std::memory_order_relaxed);
			return {nullptr, 0};
		}
		upload.Commit();
		// (v7) Remembered only if no NextFrame began during the copy (else the bytes may be of the next epoch).
		uint64_t after = 0;
		if (v12) FrameNativo2::g_v12_copy_miss.fetch_add(1, std::memory_order_relaxed);
		if (hooked && (v12 ? (FrameNativo2::RoGen(vaddr, size, &after) && after == epoch)
		                   : FrameNativo2::g_epoch.load(std::memory_order_seq_cst) == epoch))
			slot = {vaddr, size, epoch, tick, offset, &upload};
		FrameNativo2::g_hits.fetch_add(1, std::memory_order_relaxed);
		FrameNativo2::g_hit_bytes.fetch_add(size, std::memory_order_relaxed);
		FrameNativo2::g_copies.fetch_add(1, std::memory_order_relaxed);
		FrameNativo2::g_copy_bytes.fetch_add(size, std::memory_order_relaxed);
		FrameNativo2::g_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                      std::chrono::steady_clock::now() - t0).count()),
		                                  std::memory_order_relaxed);
		return {&upload, offset};
	}
	FrameNativo2::g_hits.fetch_add(1, std::memory_order_relaxed);
	FrameNativo2::g_hit_bytes.fetch_add(size, std::memory_order_relaxed);
	if (FrameNativo2::Nativo()) FrameNativo2::g_n6_native.fetch_add(1, std::memory_order_relaxed);
	const auto& arena = st.arena[i];
	return {arena.buffer, vaddr - arena.guest_base};
}

// (FRAME-NATIVO2) The BDA entries of imported arena i (or, i < 0, of every imported arena inside [first, first +
// count) pages) pointed at its buffer.
void BufferCache::FrameNativo2WriteBda(int only, uint64_t first, uint64_t count) {
	auto& st = FrameNativo2S();
	for (int i = 0; i < FrameNativo2::kArenas; ++i) {
		if ((only >= 0 && i != only) || !FrameNativo2::g_imported[i].load(std::memory_order_acquire)) continue;
		const auto&          arena = st.arena[i];
		PageTable::PageRange pages {};
		if (!PageTable::TryGetPageRange(arena.guest_base, FrameNativo2::kArenaSize, pages)) continue;
		uint64_t lo = pages.first, hi = pages.last_exclusive;
		if (only < 0) {
			lo = std::max<uint64_t>(lo, first);
			hi = std::min<uint64_t>(hi, first + count);
			if (lo >= hi) continue;
		}
		if (hi * sizeof(vk::DeviceAddress) > GetBdaPageTableBuffer()->Size()) GrowBdaTable(hi);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(hi - lo);
		for (uint64_t page = lo; page < hi; ++page)
			addresses.push_back(arena.device + ((page << CACHING_PAGEBITS) - arena.guest_base));
		for (const auto& table: m_bda_tables) {
			const uint64_t held = table->Size() / sizeof(vk::DeviceAddress);
			if (lo >= held) continue;
			const uint64_t part = std::min(hi - lo, held - lo);
			table->MakeResident(lo * sizeof(vk::DeviceAddress), part * sizeof(vk::DeviceAddress));
			WriteDataBuffer(*table, lo * sizeof(vk::DeviceAddress), addresses.data(), part * sizeof(vk::DeviceAddress));
		}
	}
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
	// (FRAME-NATIVO2) Pages of an imported arena keep pointing at its buffer (a cache buffer over them registered or
	// gone wrote its own entries or zeros).
	if (FrameNativo2::Active()) FrameNativo2WriteBda(-1, first, count);
	// (v13, =6) Los tramos sin vigilar de un arena importado siguen apuntando al arena.
	if (FrameNativo2::Nativo() && FrameNativo2::g_ro_granules.load(std::memory_order_relaxed) > 0) FrameNativo2BdaToArena(first, count);
}

// (v13, =6) Entradas BDA de las paginas [first, first + count) que caen en tramos sin vigilar de un arena importado:
// al arena (FrameNativo2WriteBda escribe solo la parte de cada arena importado).
void BufferCache::FrameNativo2BdaToArena(uint64_t first, uint64_t count) {
	const uint64_t lo = first << CACHING_PAGEBITS, hi = (first + count) << CACHING_PAGEBITS;
	uint64_t       b = 0, e = 0;
	for (uint64_t at = lo; at < hi && FrameNativo2::RoNextRun(at, hi, &b, &e); at = e) {
		const uint64_t p0 = b >> CACHING_PAGEBITS;
		const uint64_t p1 = (e + (uint64_t {1} << CACHING_PAGEBITS) - 1) >> CACHING_PAGEBITS;
		if (p1 > p0 && FrameNativo2::Touching(b, e - b) >= 0) FrameNativo2WriteBda(-1, p0, p1 - p0);
	}
}

// (v13, =6) La linea [frame-nativo2 v13] cada 10 s (hilo de la GPU): usos de bufer servidos en nativo frente al camino
// normal, coste de ObtainBuffer por uso y por fotograma traducido, rangos BDA, cuarentena.
static void FrameNativo2Report6() {
	struct Last {
		std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
		uint64_t tsc = __rdtsc(), calls = 0, native = 0, cycles = 0, bda_n = 0, bda_o = 0, frames = 0, released = 0, waits = 0;
		uint64_t held_total = 0, enter = 0, restore = 0;
	};
	static Last    last;
	const auto     now = std::chrono::steady_clock::now();
	const double   dt  = std::chrono::duration<double>(now - last.at).count();
	if (dt < 10.0) return;
	const uint64_t tsc      = __rdtsc();
	const double   ns_tick  = (static_cast<double>(tsc - last.tsc) > 0) ? dt * 1e9 / static_cast<double>(tsc - last.tsc) : 0.0;
	const uint64_t calls    = FrameNativo2::g_n6_calls.load(std::memory_order_relaxed);
	const uint64_t native   = FrameNativo2::g_n6_native.load(std::memory_order_relaxed);
	const uint64_t cycles   = FrameNativo2::g_n6_tsc.load(std::memory_order_relaxed);
	const uint64_t bda_n    = FrameNativo2::g_n6_bda_native.load(std::memory_order_relaxed);
	const uint64_t bda_o    = FrameNativo2::g_n6_bda_normal.load(std::memory_order_relaxed);
	const uint64_t frames   = FrameNativo2::g_guest_consumed.load(std::memory_order_relaxed);
	const uint64_t released = FrameNativo2::g_n6_released.load(std::memory_order_relaxed);
	const uint64_t waits    = FrameNativo2::g_n6_wait_frames.load(std::memory_order_relaxed);
	const uint64_t heldt    = FrameNativo2::g_n6_held_total.load(std::memory_order_relaxed);
	const uint64_t enter    = FrameNativo2::g_n6_enter_bda.load(std::memory_order_relaxed);
	const uint64_t restore  = FrameNativo2::g_n6_restore_bda.load(std::memory_order_relaxed);
	const double   f        = static_cast<double>(std::max<uint64_t>(frames - last.frames, 1));
	const uint64_t dc = calls - last.calls, dn = native - last.native, drel = released - last.released;
	const double   ns_use = dc != 0 ? static_cast<double>(cycles - last.cycles) * ns_tick / static_cast<double>(dc) : 0.0;
	const double   ms_frm = static_cast<double>(cycles - last.cycles) * ns_tick / 1e6 / f;
	int            imported = 0;
	for (int i = 0; i < FrameNativo2::kArenas; ++i)
		if (FrameNativo2::g_imported[i].load(std::memory_order_acquire)) ++imported;
	std::printf("[frame-nativo2 v13] NATIVO (=6), %.0f fotogramas traducidos en %.1f s: usos de bufer servidos en nativo "
	            "%.0f/s (%.0f por fotograma), por el camino normal %.0f/s (%.0f por fotograma); ObtainBuffer %.0f ns por uso, "
	            "%.2f ms por fotograma (comparar con el log sin FN2: [p7] frame/submissions y [fps] cpu); rangos BDA directos al "
	            "arena %.0f/s, normales %.0f/s; entradas BDA al arena %.0f/s, devueltas a la cache %.0f/s; cuarentena FreeTag: "
	            "%llu bloques de 1 MiB retenidos ahora (max %llu), retenidos %.0f/s, soltados %.0f/s tras %.1f fotogramas de "
	            "media, perdidos (padre reiniciado) %llu; arenas importados %d/%d; bloques sin vigilar %.1f MiB; escrituras GPU "
	            "inesperadas %llu (debe ser 0)\n",
	            f, dt, static_cast<double>(dn) / dt, static_cast<double>(dn) / f, static_cast<double>(dc - dn) / dt,
	            static_cast<double>(dc - dn) / f, ns_use, ms_frm, static_cast<double>(bda_n - last.bda_n) / dt,
	            static_cast<double>(bda_o - last.bda_o) / dt, static_cast<double>(enter - last.enter) / dt,
	            static_cast<double>(restore - last.restore) / dt,
	            static_cast<unsigned long long>(FrameNativo2::g_n6_held_blocks.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(FrameNativo2::g_n6_max_held.load(std::memory_order_relaxed)),
	            static_cast<double>(heldt - last.held_total) / dt, static_cast<double>(drel) / dt,
	            drel != 0 ? static_cast<double>(waits - last.waits) / static_cast<double>(drel) : 0.0,
	            static_cast<unsigned long long>(FrameNativo2::g_n6_lost.load(std::memory_order_relaxed)), imported,
	            FrameNativo2::kArenas,
	            static_cast<double>(FrameNativo2::g_ro_granules.load(std::memory_order_relaxed)) *
	                static_cast<double>(uint64_t {1} << FrameNativo2::kRoShift) / 1048576.0,
	            static_cast<unsigned long long>(FrameNativo2::g_ro_unexpected.load(std::memory_order_relaxed)));
	std::fflush(stdout);
	last = {now, tsc, calls, native, cycles, bda_n, bda_o, frames, released, waits, heldt, enter, restore};
}

// (v13, =6) Hilo de la GPU: entradas BDA de los bloques que entraron (al arena) y de los que vuelven a la vigilancia (a
// los buferes de la cache que tengan esas paginas, o 0: un fallo de pagina les dara uno), y las retenciones de FreeTag
// cuyo trabajo de GPU ya termino, soltadas.
void BufferCache::FrameNativo2Native6() {
	if (!g_fn2_bda_enter.empty()) {
		std::vector<std::pair<uint64_t, uint64_t>> in;
		in.swap(g_fn2_bda_enter);
		for (const auto& [address, size]: in) {
			if (!FrameNativo2::RoAny(address, size)) continue;
			PageTable::PageRange pages {};
			if (!PageTable::TryGetPageRange(address, size, pages)) continue;
			FrameNativo2BdaToArena(pages.first, pages.last_exclusive - pages.first);
			FrameNativo2::g_n6_enter_bda.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (!g_fn2_bda_restore.empty()) {
		std::vector<std::pair<uint64_t, uint64_t>> back;
		back.swap(g_fn2_bda_restore);
		for (const auto& [address, size]: back) {
			if (FrameNativo2::RoAny(address, size)) continue; // (ya otra vez sin vigilar)
			PageTable::PageRange pages {};
			if (!PageTable::TryGetPageRange(address, size, pages)) continue;
			std::vector<vk::DeviceAddress> entries;
			entries.reserve(pages.last_exclusive - pages.first);
			for (uint64_t page = pages.first; page < pages.last_exclusive; ++page) {
				const auto* owner = m_page_table.Find(page);
				if (owner != nullptr && *owner) {
					const auto& buffer = m_slot_buffers[*owner];
					entries.push_back(buffer.BufferDeviceAddress() + ((page << CACHING_PAGEBITS) - buffer.CpuAddress()));
				} else {
					entries.push_back(0);
				}
			}
			if (pages.last_exclusive * sizeof(vk::DeviceAddress) > GetBdaPageTableBuffer()->Size()) GrowBdaTable(pages.last_exclusive);
			WriteBdaEntries(pages.first, entries.size(), entries.data());
			FrameNativo2::g_n6_restore_bda.fetch_add(1, std::memory_order_relaxed);
		}
	}
	std::vector<FrameNativo2::Held> ready;
	auto&                           master = m_scheduler.GetMasterSemaphore();
	bool                            refreshed = false;
	FrameNativo2::TakeReleasable(
	    ready, [this] { return m_scheduler.CurrentTick(); },
	    [&](uint64_t tick) {
		    if (!refreshed) {
			    master.Refresh();
			    refreshed = true;
		    }
		    return master.KnownGpuTick() >= tick;
	    });
	for (const auto& h: ready) FrameNativo2::Release(h);
	FrameNativo2Report6();
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

// (unificada, KYTY_UNIFICADA=1; measurement only) Can this driver import host memory of the kind the guest's is - private,
// committed, MEM_WRITE_WATCH - and does it still once the pages are made read-only after the import, or only while
// writable? TIRON7 v2 (20261120-000053): every import of guest pages refused at vkAllocateMemory with
// VK_ERROR_OUT_OF_DEVICE_MEMORY, pages MEM_COMMIT PAGE_READONLY MEM_PRIVATE. Four cases, 64 MiB each, one "[unificada]"
// line per case; nothing of the emulator uses them.
static void UnificadaProbe(GraphicContext& graphics) {
#if defined(_WIN32)
	const char* on = std::getenv("KYTY_UNIFICADA");
	if (on == nullptr || on[0] != '1') return;
	if (!Tiron7::g_extension.load(std::memory_order_relaxed)) {
		std::printf("[unificada] VK_EXT_external_memory_host no activada: sin prueba\n");
		std::fflush(stdout);
		return;
	}
	auto&          vk     = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto     device = static_cast<VkDevice>(graphics.device);
	constexpr SIZE_T Bytes = SIZE_T {64} << 20u;
	struct Case {
		const char* name;
		DWORD       type;
		DWORD       protect_before; // at the import
		DWORD       protect_after;  // set after the import (0: none)
	};
	const Case cases[] {
	    {"privada RW", MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, 0},
	    {"privada RW + MEM_WRITE_WATCH (como la del juego)", MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE, 0},
	    {"WRITE_WATCH, importada RW y luego READONLY (como RAIZ)", MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE,
	     PAGE_READONLY},
	    {"WRITE_WATCH en READONLY al importar", MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READONLY, 0},
	};
	for (const auto& c: cases) {
		void* host = VirtualAlloc(nullptr, Bytes, c.type, PAGE_READWRITE);
		if (host == nullptr) {
			std::printf("[unificada] %s: VirtualAlloc fallo (%lu)\n", c.name, GetLastError());
			continue;
		}
		std::memset(host, 0x5a, Bytes); // (committed and touched, as the game's)
		DWORD old = 0;
		if (c.protect_before != PAGE_READWRITE) (void)VirtualProtect(host, Bytes, c.protect_before, &old);
		VkMemoryHostPointerPropertiesEXT props {};
		props.sType           = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
		const VkResult query  = vk.vkGetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                                        host, &props);
		VkPhysicalDeviceMemoryProperties memory_props {};
		vk.vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(graphics.physical_device), &memory_props);
		uint32_t type = UINT32_MAX;
		for (uint32_t i = 0; i < memory_props.memoryTypeCount && type == UINT32_MAX; ++i)
			if ((props.memoryTypeBits & (1u << i)) != 0 &&
			    (memory_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
				type = i;
		VkResult       alloc  = VK_ERROR_UNKNOWN;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		const auto     start  = std::chrono::steady_clock::now();
		if (query == VK_SUCCESS && type != UINT32_MAX) {
			VkImportMemoryHostPointerInfoEXT import_info {};
			import_info.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
			import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
			import_info.pHostPointer = host;
			VkMemoryAllocateInfo allocate {};
			allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocate.pNext           = &import_info;
			allocate.allocationSize  = Bytes;
			allocate.memoryTypeIndex = type;
			{
				VramDrv::Scope vram_drv(VramDrv::CatHostImport); // (VRAM-DRV)
				alloc = vk.vkAllocateMemory(device, &allocate, nullptr, &memory);
			}
		}
		const double ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		BOOL         after   = TRUE;
		DWORD        after_e = 0;
		if (alloc == VK_SUCCESS && c.protect_after != 0) {
			after = VirtualProtect(host, Bytes, c.protect_after, &old);
			if (!after) after_e = GetLastError();
		}
		std::printf("[unificada] %s: query %d (memoryTypeBits 0x%x, tipo %d), import %d en %.1f ms%s%s\n", c.name,
		            static_cast<int>(query), props.memoryTypeBits, type == UINT32_MAX ? -1 : static_cast<int>(type),
		            static_cast<int>(alloc), ms, alloc == VK_SUCCESS ? " -> ACEPTADA" : " -> RECHAZADA",
		            c.protect_after == 0 || alloc != VK_SUCCESS ? ""
		            : after                                   ? "; protegerla READONLY despues: OK"
		                                                      : "; protegerla READONLY despues: FALLO");
		if (!after) std::printf("[unificada]   (VirtualProtect error %lu)\n", after_e);
		std::fflush(stdout);
		if (memory != VK_NULL_HANDLE) vk.vkFreeMemory(device, memory, nullptr);
		(void)VirtualProtect(host, Bytes, PAGE_READWRITE, &old);
		VirtualFree(host, 0, MEM_RELEASE);
	}
#else
	(void)graphics;
#endif
}

// ---- (VRAM-BUF) KYTY_VRAM_BUF (1 por defecto; =0 apagado) -------------------------------------------------------------
// Log 20261125-050455 (Boletaria, 8 GB simulados): VRAM 5863 de 5449 MiB de presupuesto, 4141 pipelines expulsados por
// falta de VRAM. De lo de la cache de bufferes, m_device_buffer era un anillo de 128 MiB de VRAM (DeviceLocal) reservado al
// arrancar y usado SOLO por TextureCache::CopyImage cuando copia entre dos imagenes de formato distinto que no son D16
// (CopyImageWithBuffer): el unico llamador de GetUtilityBuffer(MemoryUsage::DeviceLocal) en todo src. Ghidra (eboot
// 01.007.000): el juego copia texturas con sus propios compute shaders (Agc::Toolkit::cs_copytexture1d/2d/3d_*,
// cs_copyrawtexture*, cs_copybuffer), que el emulador ejecuta como dispatches: no pasan por ese anillo. Ahora se crea
// al primer uso: si la sesion no tiene esa copia, 128 MiB de VRAM menos; si la tiene, igual que antes (mismo tamano).
static bool VramBufOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_BUF");
		const bool  r = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[vram-buf] KYTY_VRAM_BUF=%d: anillo de copias entre formatos (128 MiB de VRAM) %s; linea [vram-buf] cada 5 s\n",
		            r ? 1 : 0, r ? "creado al primer uso" : "creado al arrancar (como antes)");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

StreamBuffer& BufferCache::DeviceUtilityBuffer() {
	if (m_device_buffer_made.load(std::memory_order_acquire)) return *m_device_buffer;
	std::lock_guard lock(m_device_buffer_mutex);
	if (!m_device_buffer) {
		m_device_buffer = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 128 * MiB);
		m_device_buffer_made.store(true, std::memory_order_release);
		if (VramBufOn()) {
			std::printf("[vram-buf] anillo de copias entre formatos creado ahora (primera copia de ese tipo): 128 MiB\n");
			std::fflush(stdout);
		}
	}
	return *m_device_buffer;
}

// Cada 5 s: VRAM liberada por VRAM-BUF y de que son los bufferes de la cache (para atribuir los ~1,2 GB del log 050455).
void BufferCache::VramBufReport() {
	if (!VramBufOn()) return;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last = now;
	uint64_t n = 0, total = 0, bda = 0, big = 0, nbig = 0, largest = 0;
	m_slot_buffers.ForEach([&](auto id, const Buffer& b) {
		if (id.index == NULL_BUFFER_ID.index || b.is_deleted.load(std::memory_order_relaxed) || b.Size() == 0) return;
		n++;
		total += b.Size();
		if (b.bda_read) bda += b.Size();
		if (b.Size() >= (uint64_t {16} << 20u)) big += b.Size(), nbig++;
		largest = std::max<uint64_t>(largest, b.Size());
	});
	const bool made  = m_device_buffer_made.load(std::memory_order_acquire);
	const auto mib   = [](uint64_t v) { return static_cast<unsigned long long>(v >> 20u); };
	std::printf("[vram-buf] 5 s: liberados %llu MiB (anillo de copias entre formatos %s); bufferes de la cache %llu MiB en %llu "
	            "(leidos por puntero %llu MiB; de 16 MiB o mas %llu en %llu MiB, el mayor %llu MiB)\n",
	            made ? 0ull : 128ull, made ? "creado: usado por el juego" : "sin crear", mib(total),
	            static_cast<unsigned long long>(n), mib(bda), static_cast<unsigned long long>(nbig), mib(big), mib(largest));
	std::fflush(stdout);
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache,
                         GpuResourceManager* resources)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_lod_stats_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, 256 * 16),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, StagingRingSize(),
                       AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_host_shader_upload(graphics, scheduler, MemoryUsage::Upload, 64 * MiB),
      m_table_upload(graphics, scheduler, MemoryUsage::Upload, 32 * MiB,
                     AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_staging_device(graphics, scheduler, MemoryUsage::Stream, 64 * MiB,
                       AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 32 * MiB),
      m_texture_cache(texture_cache), m_resources(resources) {
	{
		const auto usage = PageTableUsage(graphics);
		m_bda_tables.push_back(
		    std::make_unique<Buffer>(graphics, scheduler, usage, 0, AllFlags, InitialBdaTableBytes(usage)));
		m_bda_table.store(m_bda_tables.back().get(), std::memory_order_release);
	}
	if (!VramBufOn()) (void)DeviceUtilityBuffer(); // (VRAM-BUF=0: el anillo de 128 MiB al arrancar, como antes)
	m_scheduler.SetPrologueHook([this](vk::CommandBuffer command) { FlushPrologueCopies(command); });
	m_scheduler.SetEpilogueHook([this](CommandBuffer& command) { FlushFeedbackBatch(command); });
	m_gpu_modified_ranges.AllowFastPath();
	g_sync_cache = this;
	// (crash-ba2b1f) Crash reports: the trackers' state of a page (MemoryTracker::PeekPageUnlocked).
	LibKernel::Memory::RaizWbSetPagePeek(
	    [](void* user, uint64_t page, uint32_t* bits) {
		    return static_cast<const BufferCache*>(user)->m_memory_tracker.PeekPageUnlocked(page, bits);
	    },
	    this);
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
	UnificadaProbe(graphics); // (unificada)
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
	StopPrefetchHelper(); // (parpadeo7: its copies made before the pending readbacks are drained below)
	LibKernel::Memory::RaizWbSetPagePeek(nullptr, nullptr); // (crash-ba2b1f)
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

void BufferCache::GuestUseBreakdown(uint64_t& total, uint64_t& gpu_any, uint64_t& gpu_full, uint64_t& cpu_any,
                                    uint64_t& count) {
	total = gpu_any = gpu_full = cpu_any = count = 0;
	m_slot_buffers.ForEach([&](BufferId /*id*/, const Buffer& buffer) {
		if (buffer.is_deleted || buffer.CpuAddress() == 0) return;
		const uint64_t address = buffer.CpuAddress(), size = buffer.Size();
		if (size == 0) return;
		count++;
		total += size;
		if (m_memory_tracker.IsRegionGpuModified(address, size)) {
			gpu_any += size;
			if (m_memory_tracker.IsRegionFullyGpuModified(address, size)) gpu_full += size;
		}
		if (m_memory_tracker.IsRegionCpuModified(address, size)) cpu_any += size;
	});
}

void BufferCache::CollectResidencyRanges(std::vector<GpuResidency::Range>& out, uint64_t& count) {
	count = 0;
	for (const auto& entry: m_buffers) {
		const auto* buffer = m_slot_buffers.try_get(entry.second);
		if (buffer == nullptr || buffer->is_deleted.load(std::memory_order_relaxed)) continue;
		const uint64_t address = buffer->CpuAddress(), size = buffer->Size();
		if (address == 0 || size == 0) continue;
		const auto kind = m_memory_tracker.IsRegionGpuModified(address, size) ? GpuResidency::GpuWrittenBuffer
		                                                                      : GpuResidency::GpuBuffer;
		out.push_back({address, size, kind});
		count++;
	}
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
		Trad1StampCoverage(first, last - first); // (trad1b)
	}
	LiveCounters::Add(LiveCounters::DiscardedGpuBytes, dropped);
	InvalidateCopyFeedback(first, last - first);
	m_memory_tracker.UnmarkRegionAsGpuModified(first, last - first);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

// (FRAME-NATIVO2 v10, =3) Bloques que FreeTag (juego 0x7fe540) acaba de liberar: lo que la GPU escribio alli se
// suelta SIN descargarlo (el juego los va a sobrescribir) y SIN marcar nada CPU-modificado (v9 lo hacia con todo el
// MiB: 1976 MB/s re-subidos, log 20261124-121448-571). Solo paginas del tracker enteras dentro del rango. Las lecturas
// siguientes: copia por epoca (v7) o la marca exacta de cada uso (GetBuffer / EnsureBdaReadable).
void BufferCache::FrameNativo2DropGpu(uint64_t vaddr, uint64_t size) {
	const auto first = (vaddr + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	const auto last  = (vaddr + size) & ~(TRACKER_PAGE_SIZE - 1);
	if (first >= last) return;
	// (Una lectura de vuelta en curso escribiria bytes viejos encima de los nuevos del juego: se cierra antes.)
	DrainGuestReadback(first, last - first, false, true);
	uint64_t dropped = 0;
	{
		const std::unique_lock lock(m_gpu_modified_mutex);
		m_gpu_modified_ranges.ForEachIntersection(first, last - first, [&](RangeSet::Range range) { dropped += range.size; });
		m_gpu_modified_ranges.Subtract(first, last - first);
		MoveWrittenObtainEpoch();
		++m_gpu_coverage_epoch;
		Trad1StampCoverage(first, last - first);
	}
	LiveCounters::Add(LiveCounters::DiscardedGpuBytes, dropped);
	FrameNativo2::g_dropped_gpu_bytes.fetch_add(dropped, std::memory_order_relaxed);
	InvalidateCopyFeedback(first, last - first);
	m_memory_tracker.UnmarkRegionAsGpuModified(first, last - first);
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
	if (NativeMemoryOn() && size != 0) {
		// (WriteBackNativeWrites' bound: ranges a CPU-side read pulled back once are written back eagerly from then on.)
		const std::scoped_lock lock(m_native_read_mutex);
		for (auto g = vaddr >> 16u; g <= (vaddr + size - 1) >> 16u; ++g) m_native_read_granules.insert(g);
	}
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
	Juego3::Scope juego3_readback(guest ? Juego3::TagReadback : 0u); // (juego3: the game thread's whole readback)
	if ((!is_write || async_write) && !GuestGpu::IsGpuThread() && GuestReadbacksEnabled()) {
		std::shared_ptr<GuestReadback> request;
		auto& gpu = m_scheduler.Context().GetGpu();
		gpu.SendCommandSync([&] {
			Juego3::LearnReadback(vaddr, size); // (juego3: the range's writer, for the readback at the label)
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
			// (raiz8, KYTY_RAIZ_ASYNC_RELEASE: the read releases the GPU thread left of this readback's pages applied
			// here, by the thread that waited for them - the GPU thread goes on meanwhile -.)
			const uint64_t r0 = FaltasReciclado::t_medir ? __rdtsc() : 0; // (sync3)
			// (sync6, KYTY_TECHO: measurement only) Of the window's pages put in here, how many RECICLADO knows dead
			// (every GPU-written byte re-reserved since): what a "defer only dead pages" variant could leave.
			if (FaltasReciclado::t_medir)
				for (const auto& page: request->pages)
					for (uint64_t at = page.address & ~uint64_t {0xfff}; at < page.address + page.size; at += 0x1000)
						(FaltasReciclado::PageDead(at) ? FaltasReciclado::g_win_dead : FaltasReciclado::g_win_live)
						    .fetch_add(1, std::memory_order_relaxed);
			if (FaltasReciclado::PmActive() && !FaltasReciclado::g_off.load(std::memory_order_relaxed)) {
				// (sync7, KYTY_PUTIN_MUERTAS) Dead pages (not the fault's own: applied below) left; live ones in runs, as before.
				const uint64_t own = vaddr & ~uint64_t {0xfff};
				for (const auto& page: request->pages) {
					uint64_t run = 0, run_end = 0;
					const uint64_t end = page.address + page.size;
					for (uint64_t at = page.address & ~uint64_t {0xfff}; at < end; at += 0x1000) {
						bool defer = at != own && FaltasReciclado::PageDead(at);
						if (defer && (FaltasReciclado::g_pm_sample.fetch_add(1, std::memory_order_relaxed) & 15u) == 0) {
							defer = false;
							FaltasReciclado::g_pm_verified.fetch_add(1, std::memory_order_relaxed);
						}
						if (defer) {
							if (run_end > run) PageManager::ApplyDeferredReleases(run, run_end - run);
							run = run_end = 0;
							FaltasReciclado::PmNote(at);
							FaltasReciclado::g_pm_deferred.fetch_add(1, std::memory_order_relaxed);
							continue;
						}
						const uint64_t from = std::max(at, page.address), to = std::min(at + 0x1000, end);
						if (run_end == from) run_end = to;
						else {
							if (run_end > run) PageManager::ApplyDeferredReleases(run, run_end - run);
							run = from, run_end = to;
						}
					}
					if (run_end > run) PageManager::ApplyDeferredReleases(run, run_end - run);
				}
			} else
				for (const auto& page: request->pages) PageManager::ApplyDeferredReleases(page.address, page.size);
			if (r0 != 0) FaltasReciclado::g_rb_release.fetch_add(__rdtsc() - r0, std::memory_order_relaxed);
		}
		const uint64_t r1 = FaltasReciclado::t_medir ? __rdtsc() : 0; // (sync3)
		PageManager::ApplyDeferredReleases(vaddr, size); // (raiz8: the bytes this read is for, whatever path released them)
		if (r1 != 0) FaltasReciclado::g_rb_release.fetch_add(__rdtsc() - r1, std::memory_order_relaxed);
		return;
	}
	LiveCounters::Add(GuestGpu::IsGpuThread() ? LiveCounters::SyncReadsRender : LiveCounters::SyncReadsGuest);
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		ReadMemoryOnGpu(vaddr, size, is_write);
	});
	if (!GuestGpu::IsGpuThread()) PageManager::ApplyDeferredReleases(vaddr, size); // (raiz8: as above)
}

// ---- (parpadeo7) KYTY_PARPADEO7: readbacks begun at the XPR label (local/parpadeo7.h) ---------------------------------
struct BufferCache::PrefetchHelper {
	struct Item {
		std::shared_ptr<GuestReadback> request;
		int64_t                        label_ns = 0;
		uint64_t                       recent   = 0; // Parpadeo7::NotePrefetched's index
		bool                           apply    = false; // retired: its pages' deferred releases are left
	};
	std::mutex              mutex;
	std::condition_variable available;
	std::deque<Item>        items;
	bool                    stop = false;
	std::thread             thread;
};

uint32_t BufferCache::PrefetchGuestRead(uint64_t vaddr, uint64_t size, int64_t label_ns) {
	if (!GuestGpu::IsGpuThread() || Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr ||
	    !GuestReadbacksEnabled() || size == 0 || UINT64_MAX - vaddr < size)
		return Parpadeo7::PrefetchOff;
	const uint64_t end = vaddr + size;
	// A pending readback over a part of it: in flight already (BeginGuestReadback would share it, or wait for its copy
	// to retire it first). A free slot (else it would retire the oldest, waiting for that copy).
	const size_t slots = kyty_local_readback_slots_mode.load(std::memory_order_relaxed) != 0 ? GuestReadbackSlots : 8;
	size_t       free  = 0;
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto& pending = m_guest_readbacks[slot];
		if (!pending || pending->copied.load(std::memory_order_acquire)) {
			if (slot < slots) ++free;
			continue;
		}
		if (std::ranges::any_of(pending->pages, [&](const GuestRange& page) { return page.address < end && vaddr < page.End(); }))
			return Parpadeo7::PrefetchInflight;
	}
	// (juego3: its prefetches leave 8 slots to the game's own reads - a read finding every slot pending retires the oldest,
	// the GPU thread waiting for that copy -.)
	if (free == 0 || (Juego3::FixNow() && free < 8)) return Parpadeo7::PrefetchNoSlot;
	{
		std::lock_guard lock(m_prefetch_helper_mutex);
		if (!m_prefetch_helper) {
			m_prefetch_helper         = std::make_unique<PrefetchHelper>();
			m_prefetch_helper->thread = std::thread([this] { PrefetchHelperRun(); });
		} else if (m_prefetch_helper->stop) {
			return Parpadeo7::PrefetchOff;
		}
	}
	if (NativeMemoryOn()) { // (as ReadMemory: ranges a CPU-side read pulled back are written back eagerly from then on)
		const std::scoped_lock lock(m_native_read_mutex);
		for (auto g = vaddr >> 16u; g <= (end - 1) >> 16u; ++g) m_native_read_granules.insert(g);
	}
	bool completed = false;
	auto request   = BeginGuestReadback(vaddr, size, &completed);
	if (!request) return completed ? Parpadeo7::PrefetchNothing : Parpadeo7::PrefetchRefused;
	if (request->copying.load(std::memory_order_acquire)) return Parpadeo7::PrefetchInflight;
	auto& helper = *m_prefetch_helper;
	{
		std::lock_guard lock(helper.mutex);
		if (!helper.stop) {
			helper.items.push_back({request, label_ns,
			                        Parpadeo7::NotePrefetched(request->begin, request->begin + request->size, label_ns), false});
			helper.available.notify_one();
			return Parpadeo7::PrefetchBegun;
		}
	}
	// (Stopping: nobody will copy it. Detached - its pages stay GPU-owned, a reader faults and reads them back -.)
	if (m_guest_readbacks[request->slot] == request) FinishGuestReadback(request->slot, true);
	return Parpadeo7::PrefetchOff;
}

// The GPU thread, the request retired.
void BufferCache::QueuePrefetchApply(const std::shared_ptr<GuestReadback>& request) {
	if (!m_prefetch_helper) return;
	auto&           helper = *m_prefetch_helper;
	std::lock_guard lock(helper.mutex);
	if (helper.stop) return;
	helper.items.push_back({request, 0, 0, true});
	helper.available.notify_one();
}

// The helper. It waits for the GPU (and the copy engine) as a reader's CopyGuestReadback does, never for the GPU
// thread: the retirement is a command posted to it. The GPU thread may wait for a copy claimed here (it retires the
// request's slot, FinishGuestReadback), as for a reader's; this thread claims each copy as soon as it is queued.
void BufferCache::PrefetchHelperRun() {
	LocalPlatform::SetThreadName("Kyty.P7Readback");
	auto& helper = *m_prefetch_helper;
	for (;;) {
		PrefetchHelper::Item item;
		bool                 stopping = false;
		{
			std::unique_lock lock(helper.mutex);
			helper.available.wait(lock, [&] { return helper.stop || !helper.items.empty(); });
			if (helper.items.empty()) return;
			item = std::move(helper.items.front());
			helper.items.pop_front();
			stopping = helper.stop;
		}
		if (item.apply) {
			// (raiz8: the put-in of kept bytes here, not by the game thread that reads them)
			for (const auto& page: item.request->pages) PageManager::ApplyDeferredReleases(page.address, page.size);
			Parpadeo7::g_c.applied.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		CopyGuestReadback(item.request);
		Parpadeo7::NoteCopied(item.recent, item.label_ns,
		                      item.request->state.load(std::memory_order_acquire) == GuestReadback::Copying);
		if (stopping) continue; // (the GPU thread is stopping: the destructor's drain retires it)
		auto request = std::move(item.request);
		(void)m_scheduler.Context().GetGpu().PostCommand([this, request] {
			if (m_guest_readbacks[request->slot] == request) FinishGuestReadback(request->slot);
			QueuePrefetchApply(request);
		});
	}
}

void BufferCache::StopPrefetchHelper() {
	std::thread thread;
	{
		std::lock_guard lock(m_prefetch_helper_mutex);
		if (!m_prefetch_helper) return;
		{
			std::lock_guard helper_lock(m_prefetch_helper->mutex);
			m_prefetch_helper->stop = true;
			m_prefetch_helper->available.notify_all();
		}
		thread = std::move(m_prefetch_helper->thread);
	}
	if (thread.joinable()) thread.join();
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
			// Fix: space was reserved opposite to the incoming stream's growth.
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
	const Envio1::Scope envio1_miss(Envio1::BufMiss); // (envio1: registration and merge)
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
	const Envio1::Scope envio1_dirty(Envio1::Dirty); // (envio1)
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
	// (ww2, KYTY_WRITE_WATCH_MODE=poll; nothing otherwise) GPU work is about to write this range: its pages become
	// GPU-written (no-access) below. What the game wrote there since the write watch was last read is read now, and
	// those pages are made CPU-dirty at once (the tracker's part of InvalidateMemory; a page the GPU wrote already is
	// left as it is): the upload below copies them before the GPU's write, the order a write fault gave. Before, the
	// next sweep found the page no-access and dropped the write: the buffer kept the bytes from before it (the
	// "written pages dropped" of the log, 180 000 in 40 s). The images over those pages are invalidated at the next
	// sweep (PageManager keeps the pages for it).
	if (is_written && PageManager::WatchPolled()) {
		thread_local std::vector<uint64_t> written_pages;
		PageManager::TakeWritesBeforeGpuWrite(vaddr, size, &written_pages);
		for (const auto page: written_pages) m_memory_tracker.InvalidateRegion(page, TRACKER_PAGE_SIZE, [] {});
	}
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
	if (total_size != 0) {
		LaneSync::NoteRange(buffer.lane_marks, vaddr, size, true);
		++Trad2::g_upload_runs; // (trad2: its verification)
	}
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

// (anuncio2) KYTY_RAIZ_ANNOUNCE's audits around each upload copy (AsyncUpload::g_copy_hook; the copies made here too).
static void UploadAuditHook(uint64_t guest, const uint8_t* destination, uint64_t size, bool done) {
	if (done) ::Loader::DemonsSoulsWriteAudit::NoteUploadCopied(guest, destination, size);
	else ::Loader::DemonsSoulsWriteAudit::NoteUploadCopyBegin(guest, size);
}

static void ReadUploadSourceBytes(uint8_t* destination, uint64_t address, uint64_t size);
static void ReadUploadSource(uint8_t* destination, uint64_t address, uint64_t size) {
	if (::Loader::DemonsSoulsWriteAudit::UploadCopyHookOn()) { // (anuncio2)
		UploadAuditHook(address, destination, size, false);
		ReadUploadSourceBytes(destination, address, size);
		UploadAuditHook(address, destination, size, true);
		return;
	}
	ReadUploadSourceBytes(destination, address, size);
}
static void ReadUploadSourceBytes(uint8_t* destination, uint64_t address, uint64_t size) {
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
	if (F0Measure::On()) for (const auto& copy: copies) F0Measure::NoteUpload(buffer.CpuAddress() + copy.dstOffset, copy.size); // (f0)
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
		// (ww2, KYTY_NATIVE_MEMORY=1 with KYTY_WRITE_WATCH=1: nothing is handed to the upload worker, see below.)
		bool queued = mapped != nullptr && AsyncUpload::Enabled() && staging_ring.IsCoherent() && UploadWorkerAllowed();
		for (const auto& copy: copies)
			queued = queued && UploadSourcePointer(buffer.CpuAddress() + copy.dstOffset, copy.size);
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
	// (anuncio2) The audits' hook on the upload worker's copies, while one is on (KYTY_RAIZ_ANNOUNCE).
	if (::Loader::DemonsSoulsWriteAudit::UploadCopyHookOn()) {
		if (AsyncUpload::g_copy_hook.load(std::memory_order_relaxed) == nullptr) AsyncUpload::g_copy_hook.store(UploadAuditHook, std::memory_order_release);
	} else if (AsyncUpload::g_copy_hook.load(std::memory_order_relaxed) != nullptr) {
		AsyncUpload::g_copy_hook.store(nullptr, std::memory_order_release);
	}
	if (mapped != nullptr) {
		// KYTY_ASYNC_UPLOAD: the worker fills coherent staging memory from the backing view
		// before the submission that carries these copies; ranges without one copy here.
		// (ww2, KYTY_NATIVE_MEMORY=1 with KYTY_WRITE_WATCH=1: copied here, never by the upload worker. Its "backing"
		// pointer is guest memory itself, read later: a page GPU work made no-access meanwhile faulted on the worker,
		// which then asked this thread - maybe waiting for the worker - for the read-back: a hang. ram4's notes, risk A.)
		const bool async = AsyncUpload::Enabled() && staging_ring.IsCoherent() && UploadWorkerAllowed(); // (raiz2)
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			const auto* source = async ? UploadSourcePointer(address, copy.size) : nullptr;
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

// (BDA-RETIRO, 11-25) KYTY_BDA_RETIRO (encendido por defecto; =0 como antes). Log 20261125-201501-496 (8 GB simulados):
// los bufferes leidos por punteros (bda_read) del pool del juego 349 -> 378 -> 422 -> 547 -> 550 MiB ([buf-mem] bda), sin
// bajar nunca, ni al dejar m01/m02; los estaticos y los escritos por la GPU suben y bajan (el recolector los quita). Causa:
// RetireStaleBuffers pone los bda_read en "kept" y les hace TouchBuffer cada vez que salen viejos, asi que su edad en la
// LRU vuelve a 0 y nada los quita salvo un unmap del juego ([buf-vram] "sin usar >30 s: bda 0" siempre). Arreglo: un sello
// de uso REAL (la preparacion BDA, EnsureBdaReadable, que lee su cabeza o una pagina suya) aparte de la LRU; con la VRAM
// en nivel critico, uno sin uso real en IdleMs, sin bytes escritos por la GPU, se borra como un estatico (la GPU termina
// antes: DeleteBuffer -> EraseRetired). Uno borrado que vuelve a registrarse (misma direccion) queda fijo para siempre:
// como mucho una vuelta por bufer en la sesion, y la linea lo cuenta.
namespace BdaRetiro {
inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BDA_RETIRO");
		const bool  r = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[bda-retiro] KYTY_BDA_RETIRO=%d: %s\n", r ? 1 : 0,
		            r ? "bufferes bda_read sin uso real en 120 s fuera con la VRAM critica" : "apagado (los bda_read no se quitan nunca, como antes)");
		std::fflush(stdout);
		return r;
	}();
	return on;
}
inline int64_t AhoraMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void Usado(const Buffer& b) { b.bda_use_ms = AhoraMs(); }
constexpr int64_t IdleMs = 120000;
std::unordered_set<uint64_t> g_retirados; // direcciones ya retiradas una vez (hilo de la GPU)
uint64_t g_n = 0, g_bytes = 0, g_vueltos = 0, g_fijos_bytes = 0;
} // namespace BdaRetiro

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
        // (FRAME-NATIVO2: a page of an imported arena is read through its buffer; no watched buffer made for it.)
        return (owner != nullptr && *owner) ||
               (FrameNativo2::Active() && FrameNativo2::Find(page << PageTable::kPageBits, 1) >= 0) ||
               (FrameNativo2::Nativo() && FrameNativo2::RoAll(page << PageTable::kPageBits, 1) &&
                FrameNativo2::Find(page << PageTable::kPageBits, 1) >= 0); // (v13: leida por la entrada BDA del arena)
	};
	for (const auto& range: ranges) {
		// (FRAME-NATIVO2 v5, COPIA) A range in an excluded arena: its pages are never protected again, so no write
		// fault tells the tracker and a buffer already over them would keep old bytes. The range (up to 1 MiB) is taken
		// as CPU-modified at each use and its buffer synchronized now, at translation (the bytes the GPU reads are
		// those of this moment, as before the import), whether a buffer held its pages or not.
		// (v13, =6) Lo que entro o salio de la vigilancia, primero; un rango entero sin vigilar se lee por las entradas BDA
		// del arena (ni bufer ni subida); si no, el camino normal de abajo (las paginas sin vigilar cuentan como "con bufer").
		if (FrameNativo2::Nativo()) {
			if (!AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation() && CommandScheduler::ThreadRecorder() == nullptr &&
			    GuestGpu::IsGpuThread()) {
				FrameNativo2DrainRewatch(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); });
				FrameNativo2DrainEnter(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); },
				                       [](uint64_t, uint64_t) {});
				FrameNativo2FlushTexturesV12([this](uint64_t a, uint64_t s) { return m_texture_cache.HasTrackedDataOverlap(a, s); },
				                             [this](uint64_t a, uint64_t s) { return m_texture_cache.HasGpuWrittenImageOverlap(a, s); },
				                             [this](uint64_t a, uint64_t s) { m_texture_cache.InvalidateMemory(a, s); });
				FrameNativo2Native6();
			}
			const uint64_t head = std::max<uint64_t>(std::min(range.size, HeadBytes), 1);
			if (FrameNativo2::RoAll(range.address, head) && FrameNativo2::Find(range.address, head) >= 0) {
				FrameNativo2::g_n6_bda_native.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			FrameNativo2::g_n6_bda_normal.fetch_add(1, std::memory_order_relaxed);
		} else if (FrameNativo2::Wanted() && !FrameNativo2::Live()) {
			constexpr uint64_t ArenaBytes = uint64_t {1} << 20u;
			if (FrameNativo2::SoloLectura() && !AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation() &&
			    CommandScheduler::ThreadRecorder() == nullptr && GuestGpu::IsGpuThread()) {
				FrameNativo2DrainRewatch(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); });
				if (FrameNativo2::V12()) {
					FrameNativo2DrainEnter(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); },
					                       [](uint64_t, uint64_t) {});
					FrameNativo2FlushTexturesV12([this](uint64_t a, uint64_t s) { return m_texture_cache.HasTrackedDataOverlap(a, s); },
					                             [this](uint64_t a, uint64_t s) { return m_texture_cache.HasGpuWrittenImageOverlap(a, s); },
					                             [this](uint64_t a, uint64_t s) { m_texture_cache.InvalidateMemory(a, s); });
				} else {
					FrameNativo2FlushTextures([this](uint64_t a, uint64_t s) { m_texture_cache.InvalidateMemory(a, s); });
					FrameNativo2DrainEnter(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); },
					                       [this](uint64_t a, uint64_t s) { m_texture_cache.InvalidateMemory(a, s); });
				}
			}
			// (v12, =5) Solo los tramos sin vigilar del rango (recortados a sus bloques: nunca se sale a un bloque
			// vigilado); cada tramo recordado por (pagina, bytes, epoca de sus bloques, tick). Si el rango tiene tambien
			// partes vigiladas, siguen por el camino normal de abajo.
			if (FrameNativo2::V12() && FrameNativo2::RoAny(range.address, std::max<uint64_t>(range.size, 1))) {
				const uint64_t mask  = (uint64_t {1} << PageTable::kPageBits) - 1;
				const uint64_t lo    = range.address;
				const uint64_t hi    = range.address + std::max<uint64_t>(std::min(range.size, ArenaBytes), 1);
				struct BdaMemo12 {
					uint64_t begin = 0, bytes = 0, gen = 0, tick = ~uint64_t {0};
				};
				static BdaMemo12 memo12[2048];
				const uint64_t   tick  = m_scheduler.CurrentTick();
				int              parts = 0;
				uint64_t         rb = 0, re = 0;
				for (uint64_t at = lo; at < hi && FrameNativo2::RoNextRun(at, hi, &rb, &re); at = re) {
					// (Paginas enteras, sin salir del tramo: los tramos van por granulos de 512 KiB, alineados a pagina.)
					const uint64_t begin = rb & ~mask;
					const uint64_t end   = (re + mask) & ~mask;
					const uint64_t bytes = end - begin;
					++parts;
					uint64_t gen = 0;
					if (bytes == 0 || !FrameNativo2::RoGen(begin, bytes, &gen)) continue;
					auto& memo = memo12[((begin >> 12u) ^ (begin >> 22u) ^ bytes) & 2047u];
					if (FrameNativo2::g_hooked.load(std::memory_order_acquire) && memo.begin == begin && memo.bytes == bytes &&
					    memo.gen == gen && memo.tick == tick && !m_memory_tracker.IsRegionGpuModified(begin, bytes)) {
						FrameNativo2::g_v12_bda_hits.fetch_add(1, std::memory_order_relaxed);
						continue;
					}
					if (IsGpuMapped(begin, bytes) && LibKernel::Memory::IsFullyMapped(begin, bytes)) {
						(void)m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(begin, bytes);
						FrameNativo2::g_v12_bda_miss.fetch_add(1, std::memory_order_relaxed);
						FrameNativo2::g_v12_bda_bytes.fetch_add(bytes, std::memory_order_relaxed);
						auto& buffer    = m_slot_buffers[FindBuffer(begin, bytes)];
						buffer.bda_read = true;
						BdaRetiro::Usado(buffer); // (BDA-RETIRO)
						PolicyBdaRead(buffer.CpuAddress(), buffer.Size()); // (policy4)
						TouchBuffer(buffer);
						(void)SynchronizeBuffer(buffer, begin, bytes, false, false);
						// (Recordado solo si ningun bloque cambio durante la subida.)
						uint64_t after = 0;
						if (FrameNativo2::RoGen(begin, bytes, &after) && after == gen) memo = {begin, bytes, gen, tick};
					}
				}
				if (FrameNativo2::RoAll(lo, hi - lo)) continue;
				if (parts > 0) FrameNativo2::g_v12_bda_split.fetch_add(1, std::memory_order_relaxed);
			}
			const int a = FrameNativo2::Touching(range.address, std::max<uint64_t>(range.size, 1));
			// (v11, =4) Tambien un rango con bloques sin vigilar de instancias de solo lectura (la marca de cada uso).
			if ((a >= 0 && FrameNativo2::g_excluded[a].load(std::memory_order_acquire)) ||
			    (FrameNativo2::SoloLectura() && !FrameNativo2::V12() &&
			     FrameNativo2::RoAny(range.address, std::max<uint64_t>(range.size, 1)))) {
				const uint64_t mask  = (uint64_t {1} << PageTable::kPageBits) - 1;
				const uint64_t begin = range.address & ~mask;
				const uint64_t end   = (range.address + std::max<uint64_t>(std::min(range.size, ArenaBytes), 1) + mask) & ~mask;
				const uint64_t bytes = end - begin;
				// (v10, =3) El mismo rango ya subido en esta epoca (sin NextFrame ni FreeTag desde entonces): sus bytes no
				// han podido cambiar (regla v7), no se vuelve a marcar ni subir.
				struct BdaMemo {
					uint64_t begin = 0, bytes = 0, epoch = ~uint64_t {0};
				};
				static BdaMemo bda_memo[1024];
				const uint64_t epoch = FrameNativo2::g_epoch.load(std::memory_order_seq_cst);
				auto&          memo  = bda_memo[((begin >> 12u) ^ (begin >> 22u) ^ bytes) & 1023u];
				if ((FrameNativo2::NoProt() || FrameNativo2::SoloLectura()) && FrameNativo2::g_hooked.load(std::memory_order_acquire) && memo.begin == begin &&
				    memo.bytes == bytes && memo.epoch == epoch && !m_memory_tracker.IsRegionGpuModified(begin, bytes)) {
					FrameNativo2::g_bda_memo.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
				if (IsGpuMapped(begin, bytes) && LibKernel::Memory::IsFullyMapped(begin, bytes)) {
					if (m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(begin, bytes))
						FrameNativo2::g_marked_bytes.fetch_add(bytes, std::memory_order_relaxed);
					FrameNativo2::g_reupload_bytes.fetch_add(bytes, std::memory_order_relaxed);
					memo = {begin, bytes, epoch};
					auto& buffer    = m_slot_buffers[FindBuffer(begin, bytes)];
					buffer.bda_read = true;
					BdaRetiro::Usado(buffer); // (BDA-RETIRO)
					PolicyBdaRead(buffer.CpuAddress(), buffer.Size()); // (policy4)
					TouchBuffer(buffer);
					(void)SynchronizeBuffer(buffer, begin, bytes, false, false);
					FrameNativo2::g_bda_skips.fetch_add(1, std::memory_order_relaxed);
				}
				continue;
			}
		}
		const auto last = (range.address + std::min(range.size, HeadBytes) - 1) >> PageTable::kPageBits;
		for (auto page = range.address >> PageTable::kPageBits; page <= last;) {
			if (held(page)) {
				// (BDA-RETIRO) Una pagina con bufer ya registrado que la preparacion BDA lee: uso real de ese bufer.
				if (const auto* owner_id = m_page_table.Find(page); owner_id != nullptr && *owner_id) {
					if (auto* owner = m_slot_buffers.try_get(*owner_id); owner != nullptr && !owner->is_deleted && owner->bda_read)
						BdaRetiro::Usado(*owner);
				}
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
				BdaRetiro::Usado(buffer); // (BDA-RETIRO)
				PolicyBdaRead(buffer.CpuAddress(), buffer.Size()); // (policy4)
				TouchBuffer(buffer);
				(void)SynchronizeBuffer(buffer, begin, bytes, false, false);
			}
			page = end;
		}
	}
}

// ---- (nativo2) KYTY_DS_NATIVE_HEAP ----------------------------------------------------------------------------------
// The game's general heaps (cpuheap, cpubkt) hold data GPU work reads that no hook of the engine announces. Until now a
// read of such a no-watch page watched it again for good ("VETADA antes": ~5,000 write faults/s in 1-1, 20258 log
// 085737, each a protection change both ways). Native: the page stays writable; the OS write watch of native memory
// (MEM_WRITE_WATCH, set by the kernel for every write, no fault) says which were written, read and reset before every
// submission. The written ones become CPU-dirty, so the submission's uses upload them exactly as a fault would have.
// What it cannot see: a write landing after the sweep and before a use in the same submission (the frame policy has the
// same window); the next submission's sweep takes it.
namespace {
struct NativeHeapStats {
	std::atomic<uint64_t> kept_pages {0}, kept_uses {0}, kept_bda {0};
	// GPU thread only:
	uint64_t sweeps = 0, sweep_pages = 0, runs = 0, written = 0, dirtied = 0, whole = 0, failed = 0, max_us = 0;
	double   sweep_us   = 0;
	double   collect_us = 0; // (nativo4: of sweep_us, CollectNativeRuns and the merge)
	uint64_t last_print = 0;
};
NativeHeapStats g_native_heap;
uint64_t        NativeHeapNowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
void NativeHeapNoteKept(uint64_t pages, bool bda) {
	g_native_heap.kept_uses.fetch_add(1, std::memory_order_relaxed);
	if (pages != 0) g_native_heap.kept_pages.fetch_add(pages, std::memory_order_relaxed);
	if (bda) g_native_heap.kept_bda.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

// ---- (HEAP-NATIVO) KYTY_HEAP_NATIVO -----------------------------------------------------------------------------------
// Log 20261124-105347-821 (Boletaria): protecciones/s otros 17,116, cpuheap 30,473, cpubkt 8,597. [anuncio]: 77,377
// write faults per 5 s refused as "VETADA antes" (~15,500/s, each an unprotect + a reprotect = ~31,000 protection changes/s,
// the cpuheap + cpubkt share). Ghidra (eboot 01.007.000; file address = game + 0xbdb0): the top writers are per-frame CPU
// bookkeeping of scene objects - FUN_00ab46f0 (file 0xac04a0) storing the distance fade/LOD at object+0x174..+0x17c
// (0xab4939, file 0xac06e9, 113,277 refusals), FUN_00a8ecf0 (file 0xa9aaa0) clearing +0x168 of 0x170-byte records and
// computing camera distances (17,895), 0xd21840 (file 0xd2d5f0), 0xabe187/0xabe3c1, 0xa8f0a6/0xa8f573 - plus the
// HeapAllocator's own headers (0x802a6c..0x8035cf, file 0x80e81c..0x80f37f) written in-band at every Alloc/Free. Those
// pages share 4 KiB with bytes GPU work reads (cpuheap: 0.15 GB with a GPU buffer copy, [ram19]), so a no-hook read
// vetoed them (watched again for good) and every frame's write then faulted.
// Native (copy at translation, no protection): a read of a heap/unlabelled page outside the hooks' rules leaves it
// writable for good and makes it CPU-dirty at THIS use, so the use uploads the guest's current bytes - what the write
// fault did, without the fault, the veto or a write-watch sweep (nativo2's sweep: 1.1 ms x ~870 submissions/s, log
// 20261123-100237-483). Same window as the frame policy: a CPU write after this submission's translation is taken at the
// next use. Labels (DemonsSoulsMemMap via FrameNativo::Classify): cpuheap, cpubkt, otros; DefaultTagg/ring/ctx/gpupage keep
// their path. A use whose heap part is over KYTY_HEAP_NATIVO_MAX_KB (default 4096) keeps the old path (upload bounded).
namespace {
struct HeapNativoStats {
	std::atomic<uint64_t> uses {0}, pages {0}, big {0}, bda {0};
	std::atomic<uint64_t> same_epoch {0}, same_hash {0}, hashed {0}, hash_ns {0}; // (v2)
	uint64_t              last_ms = 0, last_flips = 0;
	uint64_t              calls[FrameNativo::CCount] {};
};
HeapNativoStats g_heap_nativo;
// (HEAP-NATIVO v2) Log 20261124-121041-478 (=1): 27,800 uses/s, 970 MB/s re-uploaded (~30 MB/frame), 31 ms/frame against
// 13 ms without it (log 120220). Every use made its pages CPU-dirty again. Ghidra: the writers are per-frame passes
// (FUN_00ab46f0, file 0xac04a0: fade/LOD at +0x174/+0x178/+0x17c of every object; FUN_00a8ecf0, file 0xa9aaa0: +0x168
// cleared in every 0x170-byte record, then camera distances) and the HeapAllocator headers only at Alloc/Free: the bytes
// change at most once per game frame, most pages not at all. So a page is looked at once per epoch (a flip or a NextFrame
// of the frame hook) and uploaded only if its 64-bit signature changed since the last upload.
// (HEAP-NATIVO v3) Log 20261124-122101-164 (v2): 20.1 ms/frame (cpu 19.5, idle 0) against 12.9 without it (log 120220);
// uploads only 0.33 MB/frame and signatures 2,981/frame = 0.75 ms/frame, so neither is the cost. The rest: ~1,170 uses/frame
// (58,314/s) each walking ~21 pages (101 MB/frame "ahorrado" = ~26,000 page visits/frame), every visit 2-3 label lookups
// (FrameNativo::Classify) + a mutex + an unordered_map lookup (23,000/frame were "ya visto en el frame"). Ghidra confirms
// the writers touch EVERY object each frame (FUN_00ab46f0, file 0xac04a0: loop over the list at +0xa0/+0xc8 storing
// +0x174 = 1.0f / +0x17c = 0xff00; FUN_00a8ecf0, file 0xa9aaa0: +0x168 = 0 of every 0x170-byte record) but mostly with the
// same values (same_hash 2,900 of 2,981), so per-page protection (via A) would fault on every page every frame, and a hook
// (via B) would mark every object dirty: both worse. v3 keeps v2's rule (one look per page per epoch + signature) and
// removes the overhead around it: (1) a use already taken this epoch (same address and size) returns at once; (2) a page's
// epoch is claimed with one atomic compare-exchange in a fixed lock-free table that also caches its label (no mutex, no
// map, no Classify after the first time).
struct HeapNativoSlot {
	std::atomic<uint64_t> tag {0};   // page | 1 (0 = empty)
	std::atomic<uint64_t> epoch {0}; // last epoch it was looked at
	std::atomic<uint64_t> hash {0};  // signature of the last upload
	std::atomic<uint32_t> label {0}; // 1 = heap (ours), 2 = not ours, 0 = unknown
};
constexpr size_t HEAP_NATIVO_SLOTS = size_t {1} << 18u; // 8 MiB, 1 GiB of distinct pages before two share a slot
HeapNativoSlot*  g_heap_nativo_slots = new HeapNativoSlot[HEAP_NATIVO_SLOTS];
struct HeapNativoUse {
	std::atomic<uint64_t> key {0}, epoch {0}, heap {0};
};
constexpr size_t HEAP_NATIVO_USES = size_t {1} << 16u;
HeapNativoUse*   g_heap_nativo_uses = new HeapNativoUse[HEAP_NATIVO_USES];
struct HeapNativoStats3 {
	std::atomic<uint64_t> use_hits {0}, take_ns {0}, slot_misses {0}, label_calls {0}, mark_ns {0};
};
HeapNativoStats3 g_heap_nativo3;
[[nodiscard]] HeapNativoSlot& HeapNativoSlotFor(uint64_t page) noexcept {
	auto&          s   = g_heap_nativo_slots[((page >> 12u) * 0x9E3779B97F4A7C15ull) >> (64u - 18u)];
	const uint64_t tag = page | 1u;
	if (s.tag.load(std::memory_order_acquire) != tag) {
		// (a new page, or another one shared the slot: start it over - it is looked at and uploaded, as the first time)
		g_heap_nativo3.slot_misses.fetch_add(1, std::memory_order_relaxed);
		s.label.store(0, std::memory_order_relaxed);
		s.hash.store(0, std::memory_order_relaxed);
		s.epoch.store(0, std::memory_order_relaxed);
		s.tag.store(tag, std::memory_order_release);
	}
	return s;
}
[[nodiscard]] bool HeapNativoLabel(uint64_t vaddr) noexcept {
	if (FrameNativo::g_map.load(std::memory_order_acquire) == nullptr) return false; // (memory map not built yet)
	const auto c = FrameNativo::Classify(vaddr);
	return c == FrameNativo::CCpuheap || c == FrameNativo::CCpubkt || c == FrameNativo::COtros;
}
[[nodiscard]] uint64_t HeapNativoMaxBytes() noexcept {
	static const uint64_t max = [] {
		const char* v = std::getenv("KYTY_HEAP_NATIVO_MAX_KB");
		if (v != nullptr && v[0] >= '0' && v[0] <= '9') return std::strtoull(v, nullptr, 10) * 1024ull;
		return uint64_t {4096} * 1024ull;
	}();
	return max;
}
[[nodiscard]] uint64_t HeapNativoEpoch() noexcept {
	return LiveControl::g_flips.load(std::memory_order_acquire) +
	       (FrameNativo2::g_epoch.load(std::memory_order_acquire) << 32u);
}
// 0 = the page could not be read (then it is uploaded as before).
[[nodiscard]] uint64_t HeapNativoHash(uint64_t page) noexcept {
	alignas(16) uint64_t w[TRACKER_PAGE_SIZE / 8];
	if (!Libs::LibKernel::Memory::TryReadBackingToHost(page, w, TRACKER_PAGE_SIZE)) return 0;
	uint64_t h0 = 0x9e3779b97f4a7c15ull, h1 = 0xc2b2ae3d27d4eb4full, h2 = 0x165667b19e3779f9ull, h3 = 0x27d4eb2f165667c5ull;
	for (size_t i = 0; i < TRACKER_PAGE_SIZE / 8; i += 4) {
		h0 = (h0 ^ w[i + 0]) * 0x100000001b3ull;
		h1 = (h1 ^ w[i + 1]) * 0x100000001b3ull;
		h2 = (h2 ^ w[i + 2]) * 0x100000001b3ull;
		h3 = (h3 ^ w[i + 3]) * 0x100000001b3ull;
	}
	const uint64_t h = (h0 ^ (h1 >> 7u) ^ (h1 << 57u)) + (h2 ^ (h3 >> 13u) ^ (h3 << 51u));
	return h == 0 ? 1 : h;
}
// (HEAP-NATIVO v4) Log 20261124-190741-497 (v3): signatures 2,657-3,017 pages/frame = 0.76-0.85 ms/frame (~280 ns a page):
// one TryReadBackingToHost per page (read transaction + mapping lookup + guarded 4 KiB copy) and a hash bound by the
// multiply chain. v4 (KYTY_HEAP_NATIVO_V4=1, off by default until validated): the pages of a use claimed in this epoch are
// read in runs of up to 16 contiguous pages with ONE read (the lookup and the transaction paid once per run), and signed
// with a two-sum signature per 64-bit lane (additions only: the compiler vectorizes it; the second sum weights by position,
// so a moved or changed word changes it). Same rule as v3 (every claimed page signed each epoch): nothing skipped, so the
// same bytes reach the GPU. A run that cannot be read whole goes page by page as in v3 (a different signature only makes
// that page upload once).
[[nodiscard]] bool HeapNativoV4On() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_HEAP_NATIVO_V4");
		const bool  yes = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[heap-nativo] KYTY_HEAP_NATIVO_V4=%d: %s\n", yes ? 1 : 0,
		            yes ? "firmas por tramos (una lectura cada 16 paginas seguidas) y firma de sumas (vectorizable)"
		                : "apagado (firmas de v3: una lectura y una firma por pagina)");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}
[[nodiscard]] uint64_t HeapNativoHash4(const uint64_t* w) noexcept {
	uint64_t a[8] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull,
	                 0x85ebca77c2b2ae63ull, 0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull, 0x94d049bb133111ebull};
	uint64_t b[8] = {};
	for (size_t i = 0; i < TRACKER_PAGE_SIZE / 8; i += 8) {
		for (size_t k = 0; k < 8; ++k) {
			a[k] += w[i + k];
			b[k] += a[k];
		}
	}
	uint64_t h = 0x100000001b3ull;
	for (size_t k = 0; k < 8; ++k) {
		h = (h ^ a[k]) * 0x9E3779B97F4A7C15ull;
		h = (h ^ b[k] ^ (h >> 29u)) * 0xC2B2AE3D27D4EB4Full;
	}
	return h == 0 ? 1 : h;
}
struct HeapNativoStats4 {
	std::atomic<uint64_t> reads {0}, pages {0}, fallback {0}, inplace {0}; // (v5: inplace = signed where they are)
};
HeapNativoStats4 g_heap_nativo4;
constexpr size_t HEAP_NATIVO_RUN = 16;
} // namespace

bool BufferCache::HeapNativoOn() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_HEAP_NATIVO");
		const bool  yes = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto; KYTY_HEAP_NATIVO=1 lo enciende
		std::printf("[heap-nativo] KYTY_HEAP_NATIVO=%d: %s\n", yes ? 1 : 0,
		            yes ? "ENCENDIDO por defecto (v3): cpuheap, cpubkt y otros sin proteccion: lo que la GPU lee de ahi se copia "
		                  "al traducir (sin veto, sin fallos, sin write watch)"
		                : "apagado con KYTY_HEAP_NATIVO=0 (como antes)");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}

// (HEAP-NATIVO) Any thread that asks the cache. Returns the bytes of [vaddr, vaddr + size) it took (runs of heap pages,
// made CPU-dirty now, left writable); the caller handles the rest as before. 0 = none taken.
uint64_t BufferCache::HeapNativoTake(uint64_t vaddr, uint64_t size, bool bda) const {
	if (size == 0 || FrameNativo::g_map.load(std::memory_order_acquire) == nullptr) return 0;
	const auto     t0    = std::chrono::steady_clock::now();
	const uint64_t epoch = HeapNativoEpoch();
	// (v3 1) The same use again in this epoch: its pages were already looked at; same answer, nothing to do.
	const uint64_t key = (vaddr ^ (size << 40u) ^ (size >> 24u)) | 1u;
	auto&          use = g_heap_nativo_uses[((vaddr ^ (size * 0x9E3779B97F4A7C15ull)) * 0xC2B2AE3D27D4EB4Full) >> (64u - 16u)];
	if (use.key.load(std::memory_order_acquire) == key && use.epoch.load(std::memory_order_acquire) == epoch) {
		g_heap_nativo3.use_hits.fetch_add(1, std::memory_order_relaxed);
		g_heap_nativo.uses.fetch_add(1, std::memory_order_relaxed);
		const uint64_t heap = use.heap.load(std::memory_order_relaxed);
		g_heap_nativo3.take_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                     std::chrono::steady_clock::now() - t0).count()),
		                                 std::memory_order_relaxed);
		return heap;
	}
	auto&          tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	const uint64_t first   = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const uint64_t end     = vaddr + size;
	// Labels from the table (Classify once per page, not per visit).
	auto ours = [](uint64_t page) {
		auto&    s = HeapNativoSlotFor(page);
		uint32_t l = s.label.load(std::memory_order_relaxed);
		if (l == 0) {
			g_heap_nativo3.label_calls.fetch_add(1, std::memory_order_relaxed);
			l = HeapNativoLabel(page) ? 1u : 2u;
			s.label.store(l, std::memory_order_relaxed);
		}
		return l == 1u;
	};
	uint64_t heap = 0;
	for (uint64_t p = first; p < end; p += TRACKER_PAGE_SIZE) {
		if (ours(p)) heap += (std::min)(p + TRACKER_PAGE_SIZE, end) - (std::max)(p, vaddr);
	}
	// (only a use that is all heap, or all not-heap, is remembered: a mixed one re-runs ClearNoWatch on its other pages)
	const bool pure = heap == 0 || heap == size;
	auto finish = [&](uint64_t h) {
		if (pure) {
			use.epoch.store(0, std::memory_order_relaxed);
			use.heap.store(h, std::memory_order_relaxed);
			use.key.store(key, std::memory_order_relaxed);
			use.epoch.store(epoch, std::memory_order_release);
		}
		g_heap_nativo3.take_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                     std::chrono::steady_clock::now() - t0).count()),
		                                 std::memory_order_relaxed);
		return h;
	};
	if (heap == 0) return finish(0);
	if (heap > HeapNativoMaxBytes()) {
		g_heap_nativo.big.fetch_add(1, std::memory_order_relaxed);
		g_heap_nativo3.take_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                     std::chrono::steady_clock::now() - t0).count()),
		                                 std::memory_order_relaxed);
		return 0; // (old path; not cached: the caller does the work each time)
	}
	uint64_t pages = 0, mark_ns = 0;
	uint64_t run = 0, run_end = 0;
	auto     flush = [&] {
        if (run_end != run) {
            const auto m0 = std::chrono::steady_clock::now();
            pages += tracker.MarkNoWatchDirty(run, run_end - run);
            mark_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m0).count());
        }
        run = run_end = 0;
	};
	for (uint64_t at = vaddr; at < end;) {
		const bool o    = ours(at & ~(TRACKER_PAGE_SIZE - 1));
		uint64_t   next = (std::min)((at | (TRACKER_PAGE_SIZE - 1)) + 1, end);
		while (next < end && ours(next) == o) next = (std::min)(next + TRACKER_PAGE_SIZE, end);
		if (o && HeapNativoV4On()) {
			// (v4) Claim as v3, sign the claimed pages by runs read at once, then the same upload rule.
			uint64_t pend[HEAP_NATIVO_RUN];
			size_t   np        = 0;
			auto     sign_pend = [&] {
                if (np == 0) return;
                thread_local uint64_t buf[HEAP_NATIVO_RUN * (TRACKER_PAGE_SIZE / 8)];
                uint64_t              hs[HEAP_NATIVO_RUN];
                const auto            h0 = std::chrono::steady_clock::now();
                if (Libs::LibKernel::Memory::NativeMemoryEnabled()) {
                    // (v5) In place: native memory, the guest address IS the PC address. NativeWindowRun reads the
                    // pages where they are (no copy when they are not protected; a window when a tracker protects
                    // them; read again if a put-in of kept GPU bytes overlapped) and the signature runs on them.
                    struct Ctx {
                        const uint64_t* w;
                        size_t          n;
                        uint64_t*       hs;
                    } ctx {reinterpret_cast<const uint64_t*>(pend[0]), np, hs};
                    Libs::LibKernel::Memory::NativeWindowRun(
                        ctx.w, np * TRACKER_PAGE_SIZE,
                        [](void* c) {
                            auto* x = static_cast<Ctx*>(c);
                            for (size_t i = 0; i < x->n; ++i) x->hs[i] = HeapNativoHash4(x->w + i * (TRACKER_PAGE_SIZE / 8));
                        },
                        &ctx);
                    g_heap_nativo4.reads.fetch_add(1, std::memory_order_relaxed);
                    g_heap_nativo4.inplace.fetch_add(np, std::memory_order_relaxed);
                } else if (Libs::LibKernel::Memory::TryReadBackingToHost(pend[0], buf, np * TRACKER_PAGE_SIZE)) {
                    for (size_t i = 0; i < np; ++i) hs[i] = HeapNativoHash4(buf + i * (TRACKER_PAGE_SIZE / 8));
                    g_heap_nativo4.reads.fetch_add(1, std::memory_order_relaxed);
                } else {
                    for (size_t i = 0; i < np; ++i) hs[i] = HeapNativoHash(pend[i]);
                    g_heap_nativo4.fallback.fetch_add(np, std::memory_order_relaxed);
                }
                g_heap_nativo4.pages.fetch_add(np, std::memory_order_relaxed);
                g_heap_nativo.hashed.fetch_add(np, std::memory_order_relaxed);
                g_heap_nativo.hash_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::steady_clock::now() - h0).count()),
                                                std::memory_order_relaxed);
                for (size_t i = 0; i < np; ++i) {
                    const uint64_t p = pend[i], h = hs[i];
                    auto&          s = HeapNativoSlotFor(p);
                    if (h != 0 && s.hash.load(std::memory_order_relaxed) == h) {
                        g_heap_nativo.same_hash.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    s.hash.store(h, std::memory_order_relaxed);
                    if (run_end != p) {
                        flush();
                        run = p;
                    }
                    run_end = p + TRACKER_PAGE_SIZE;
                }
                np = 0;
			};
			for (uint64_t p = at & ~(TRACKER_PAGE_SIZE - 1); p < next; p += TRACKER_PAGE_SIZE) {
				auto&    s    = HeapNativoSlotFor(p);
				uint64_t seen = s.epoch.load(std::memory_order_acquire);
				if (seen == epoch || !s.epoch.compare_exchange_strong(seen, epoch, std::memory_order_acq_rel)) {
					g_heap_nativo.same_epoch.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
				if (np != 0 && (np == HEAP_NATIVO_RUN || pend[np - 1] + TRACKER_PAGE_SIZE != p)) sign_pend();
				pend[np++] = p;
			}
			sign_pend();
			flush();
		} else if (o) {
			for (uint64_t p = at & ~(TRACKER_PAGE_SIZE - 1); p < next; p += TRACKER_PAGE_SIZE) {
				auto&    s    = HeapNativoSlotFor(p);
				uint64_t seen = s.epoch.load(std::memory_order_acquire);
				if (seen == epoch || !s.epoch.compare_exchange_strong(seen, epoch, std::memory_order_acq_rel)) {
					g_heap_nativo.same_epoch.fetch_add(1, std::memory_order_relaxed);
					continue; // (this epoch's look is done, or another thread is doing it)
				}
				const auto     h0 = std::chrono::steady_clock::now();
				const uint64_t h  = HeapNativoHash(p);
				g_heap_nativo.hashed.fetch_add(1, std::memory_order_relaxed);
				g_heap_nativo.hash_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                    std::chrono::steady_clock::now() - h0).count()),
				                                std::memory_order_relaxed);
				if (h != 0 && s.hash.load(std::memory_order_relaxed) == h) {
					g_heap_nativo.same_hash.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
				s.hash.store(h, std::memory_order_relaxed);
				if (run_end != p) {
					flush();
					run = p;
				}
				run_end = p + TRACKER_PAGE_SIZE;
			}
			flush();
		} else {
			const uint64_t n = tracker.ClearNoWatch(at, next - at, true);
			::Loader::DemonsSoulsWriteAudit::NoteReverted(at, n, bda ? 2 : 0);
		}
		at = next;
	}
	g_heap_nativo.uses.fetch_add(1, std::memory_order_relaxed);
	g_heap_nativo.pages.fetch_add(pages, std::memory_order_relaxed);
	g_heap_nativo3.mark_ns.fetch_add(mark_ns, std::memory_order_relaxed);
	if (bda) g_heap_nativo.bda.fetch_add(1, std::memory_order_relaxed);
	return finish(heap);
}

// (HEAP-NATIVO) The [heap-nativo] line, every 5 s (GPU thread, before each submission): protection changes/s of the three
// labels now, against the reference of log 20261124-105347-821 (KYTY_HEAP_NATIVO=0).
void BufferCache::HeapNativoReport() {
	auto&          st  = g_heap_nativo;
	const uint64_t now = NativeHeapNowMs();
	if (st.last_ms == 0) {
		st.last_ms = now;
		for (size_t i = 0; i < FrameNativo::CCount; ++i) st.calls[i] = FrameNativo::g_calls[i].load(std::memory_order_relaxed);
		return;
	}
	if (now - st.last_ms < 5000) return;
	const double secs = static_cast<double>(now - st.last_ms) / 1000.0;
	double       ps[FrameNativo::CCount];
	for (size_t i = 0; i < FrameNativo::CCount; ++i) {
		const uint64_t c = FrameNativo::g_calls[i].load(std::memory_order_relaxed);
		ps[i]            = static_cast<double>(c - st.calls[i]) / secs;
		st.calls[i]      = c;
	}
	const uint64_t uses = st.uses.exchange(0, std::memory_order_relaxed), pages = st.pages.exchange(0, std::memory_order_relaxed);
	const uint64_t big = st.big.exchange(0, std::memory_order_relaxed), bda = st.bda.exchange(0, std::memory_order_relaxed);
	const uint64_t same_epoch = st.same_epoch.exchange(0, std::memory_order_relaxed);
	const uint64_t same_hash = st.same_hash.exchange(0, std::memory_order_relaxed), hashed = st.hashed.exchange(0, std::memory_order_relaxed);
	const uint64_t hash_ns    = st.hash_ns.exchange(0, std::memory_order_relaxed);
	const uint64_t flips      = LiveControl::g_flips.load(std::memory_order_relaxed);
	const double   frames     = st.last_flips != 0 && flips > st.last_flips ? static_cast<double>(flips - st.last_flips) : 1.0;
	st.last_flips             = flips;
	std::printf("[heap-nativo] %.1f s (KYTY_HEAP_NATIVO=%d): protecciones/s antes (log 105347) -> ahora: cpuheap 30473 -> %.0f; "
	            "cpubkt 8597 -> %.0f; otros 17116 -> %.0f; (DefaultTagg %.0f, gpupage %.0f: no son de este cambio); copiado al "
	            "traducir %.0f usos/s (%llu de BDA), %.2f MB/s; usos demasiado grandes (camino de antes) %llu\n",
	            secs, HeapNativoOn() ? 1 : 0, ps[FrameNativo::CCpuheap], ps[FrameNativo::CCpubkt], ps[FrameNativo::COtros],
	            ps[FrameNativo::CDefaultTagg], ps[FrameNativo::CGpupage], static_cast<double>(uses) / secs,
	            static_cast<unsigned long long>(bda), static_cast<double>(pages) * 4096.0 / 1e6 / secs,
	            static_cast<unsigned long long>(big));
	std::printf("[heap-nativo] (v2: una vez por epoca + firma) %.0f frames: subido %.2f MB/frame; ahorrado %.2f MB/frame (ya "
	            "visto en el frame %.2f + sin cambios %.2f); firmas %.0f/frame %.3f ms/frame (antes de v2: ~30 MB/frame, log 121041)\n",
	            frames, static_cast<double>(pages) * 4096.0 / 1e6 / frames,
	            static_cast<double>(same_epoch + same_hash) * 4096.0 / 1e6 / frames,
	            static_cast<double>(same_epoch) * 4096.0 / 1e6 / frames, static_cast<double>(same_hash) * 4096.0 / 1e6 / frames,
	            static_cast<double>(hashed) / frames, static_cast<double>(hash_ns) / 1e6 / frames);
	{
		const uint64_t use_hits = g_heap_nativo3.use_hits.exchange(0, std::memory_order_relaxed);
		const uint64_t take_ns  = g_heap_nativo3.take_ns.exchange(0, std::memory_order_relaxed);
		const uint64_t misses   = g_heap_nativo3.slot_misses.exchange(0, std::memory_order_relaxed);
		const uint64_t labels   = g_heap_nativo3.label_calls.exchange(0, std::memory_order_relaxed);
		const uint64_t mark_ns  = g_heap_nativo3.mark_ns.exchange(0, std::memory_order_relaxed);
		std::printf("[heap-nativo] (v3: sin mutex/mapa, etiqueta en tabla, uso repetido al instante) coste total %.3f ms/frame = "
		            "firmas %.3f + marcar subida %.3f + resto (busquedas) %.3f; usos %.0f/frame (repetidos %.0f); paginas nuevas "
		            "en tabla %.0f/frame, Classify %.0f/frame (v2, log 122101: 20.1 ms/frame, 0.75 ms firmas)\n",
		            static_cast<double>(take_ns) / 1e6 / frames, static_cast<double>(hash_ns) / 1e6 / frames,
		            static_cast<double>(mark_ns) / 1e6 / frames,
		            static_cast<double>(take_ns > hash_ns + mark_ns ? take_ns - hash_ns - mark_ns : 0) / 1e6 / frames,
		            static_cast<double>(uses) / frames, static_cast<double>(use_hits) / frames, static_cast<double>(misses) / frames,
		            static_cast<double>(labels) / frames);
	}
	{
		const uint64_t r4 = g_heap_nativo4.reads.exchange(0, std::memory_order_relaxed);
		const uint64_t p4 = g_heap_nativo4.pages.exchange(0, std::memory_order_relaxed);
		const uint64_t f4 = g_heap_nativo4.fallback.exchange(0, std::memory_order_relaxed);
		const uint64_t i4 = g_heap_nativo4.inplace.exchange(0, std::memory_order_relaxed);
		std::printf("[heap-nativo] (v4 KYTY_HEAP_NATIVO_V4=%d: firmas por tramos) paginas firmadas %.0f/frame en %.0f lecturas/frame "
		            "(%.1f paginas por lectura), pagina a pagina por no poder leer el tramo %.0f/frame; firmas %.3f ms/frame (v3, log "
		            "190741: 0.76-0.85 ms/frame, una lectura por pagina). Protecciones que quedan en cpuheap/otros: de la GPU "
		            "(write-release, deferred-release, reapply en la linea [raiz] PROTECT BY ORIGIN), no de lecturas de la CPU\n",
		            HeapNativoV4On() ? 1 : 0, static_cast<double>(p4) / frames, static_cast<double>(r4) / frames,
		            r4 != 0 ? static_cast<double>(p4 - f4) / static_cast<double>(r4) : 0.0, static_cast<double>(f4) / frames,
		            static_cast<double>(hash_ns) / 1e6 / frames);
		std::printf("[heap-nativo] (v5: firma en su sitio, sin copia) paginas firmadas en su sitio %.0f/frame de %.0f; firmas %.3f "
	            "ms/frame (v4 con copia, log 20261125-075201-988: 1.05 ms/frame, 1679 paginas)\n",
	            static_cast<double>(i4) / frames, static_cast<double>(p4) / frames, static_cast<double>(hash_ns) / 1e6 / frames);
	}
	std::fflush(stdout);
	st.last_ms = now;
}

bool BufferCache::NativeAllOn() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_DS_NATIVE_ALL");
		const bool  yes = v != nullptr && v[0] == '1';
		std::printf("[nativo3] KYTY_DS_NATIVE_ALL=%d: %s\n", yes ? 1 : 0,
		            yes ? "TODA pagina sin vigilancia (frame y montones) por el write watch del sistema antes de cada envio "
		                  "(exacto por construccion; las marcas de frame no se aplican; un envio encolado antes de un barrido no "
		                  "barre otra vez)"
		                : "apagado");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}

bool BufferCache::NativeHeapOn() noexcept {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_DS_NATIVE_HEAP");
		const bool  yes = (v != nullptr && v[0] == '1') || NativeAllOn();
		std::printf("[nativo2] KYTY_DS_NATIVE_HEAP=%d: %s\n", yes ? 1 : 0,
		            yes ? "lecturas de la GPU fuera de los ganchos: paginas nativas (write watch del sistema en cada envio), sin "
		                  "volver a vigilarlas"
		                : "apagado (las paginas que la GPU lee sin gancho se vuelven a vigilar, como siempre)");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}

void BufferCache::PolicyNativeSweep() {
	HeapNativoReport(); // (HEAP-NATIVO: the [heap-nativo] line, on or off)
	if (!NativeHeapOn()) return;
	const auto start = std::chrono::steady_clock::now();
	struct Run {
		uint64_t address, size;
	};
	thread_local std::vector<Run>      runs;
	thread_local std::vector<uint64_t> found;
	runs.clear();
	const bool     all   = NativeAllOn();
	const uint64_t pages = m_memory_tracker.CollectNativeRuns(&runs, all);
	const auto     collected = std::chrono::steady_clock::now(); // (nativo4)
	// (nativo3) Fewer system calls: runs less than 1 MiB apart read in one call (the pages between are watched ones:
	// MarkNoWatchDirty leaves them alone, and nothing reads their write watch under KYTY_RAIZ).
	if (all && runs.size() > 1) {
		std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.address < b.address; });
		size_t out = 0;
		for (size_t i = 1; i < runs.size(); ++i) {
			const uint64_t end = runs[out].address + runs[out].size;
			if (runs[i].address >= end && runs[i].address - end <= (uint64_t {1} << 20u)) {
				runs[out].size = (std::max)(end, runs[i].address + runs[i].size) - runs[out].address;
			} else if (runs[i].address < end) {
				runs[out].size = (std::max)(end, runs[i].address + runs[i].size) - runs[out].address;
			} else {
				runs[++out] = runs[i];
			}
		}
		runs.resize(out + 1);
	}
	const uint64_t Max = all ? 65536 : 4096;
	found.resize(Max);
	uint64_t written = 0, dirtied = 0, whole = 0, failed_runs = 0;
	for (const auto& run: runs) {
		bool           failed = false;
		const uint64_t count  = Libs::LibKernel::Memory::NativeTakeWritten(run.address, run.size, found.data(), Max, &failed);
		written += count;
		if (failed || count > Max) {
			// (Its writes cannot all be told: the whole run is taken as written - never stale, an upload more.)
			dirtied += m_memory_tracker.MarkNoWatchDirty(run.address, run.size);
			++whole;
			if (failed) ++failed_runs;
			continue;
		}
		// Adjacent written pages in one call (one region lock each).
		for (uint64_t i = 0; i < count;) {
			uint64_t j = i + 1;
			while (j < count && found[j] == found[j - 1] + TRACKER_PAGE_SIZE) ++j;
			dirtied += m_memory_tracker.MarkNoWatchDirty(found[i], (j - i) * TRACKER_PAGE_SIZE);
			i = j;
		}
	}
	auto&          st = g_native_heap;
	const double   us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
	st.sweeps++;
	st.sweep_pages += pages;
	st.runs += runs.size();
	st.written += written;
	st.dirtied += dirtied;
	st.whole += whole;
	st.failed += failed_runs;
	st.sweep_us += us;
	st.collect_us += std::chrono::duration<double, std::micro>(collected - start).count();
	st.max_us = (std::max)(st.max_us, static_cast<uint64_t>(us));
	const uint64_t now = NativeHeapNowMs();
	if (st.last_print == 0) st.last_print = now;
	if (now - st.last_print >= 5000) {
		const double secs = static_cast<double>(now - st.last_print) / 1000.0;
		std::printf("[nativo2] %.1f s: paginas nativas nuevas %llu (en %llu usos, %llu de BDA) que ya no se vuelven a vigilar; "
		            "barridos %llu (%.0f paginas y %.0f tramos de media), write watch: escritas %llu -> marcadas sucias %llu "
		            "(tramos tomados enteros %llu, sin write watch %llu); coste %.2f ms en total (%.3f ms por barrido, el mayor "
		            "%.3f ms)\n",
		            secs, static_cast<unsigned long long>(st.kept_pages.exchange(0, std::memory_order_relaxed)),
		            static_cast<unsigned long long>(st.kept_uses.exchange(0, std::memory_order_relaxed)),
		            static_cast<unsigned long long>(st.kept_bda.exchange(0, std::memory_order_relaxed)),
		            static_cast<unsigned long long>(st.sweeps),
		            st.sweeps != 0 ? static_cast<double>(st.sweep_pages) / static_cast<double>(st.sweeps) : 0.0,
		            st.sweeps != 0 ? static_cast<double>(st.runs) / static_cast<double>(st.sweeps) : 0.0,
		            static_cast<unsigned long long>(st.written), static_cast<unsigned long long>(st.dirtied),
		            static_cast<unsigned long long>(st.whole), static_cast<unsigned long long>(st.failed), st.sweep_us / 1000.0,
		            st.sweeps != 0 ? st.sweep_us / 1000.0 / static_cast<double>(st.sweeps) : 0.0, static_cast<double>(st.max_us) / 1000.0);
		std::printf("[nativo2] de ese coste, recorrer las paginas sin vigilancia: %.2f ms (el resto: GetWriteWatch y marcar sucio)\n", st.collect_us / 1000.0); // (nativo4)
		// (nativo3) The cheap exactness check (KYTY_DS_WRITE_FRAME=2: 1 page in 64 hashed when uploaded, compared when GPU
		// work uses it): uses whose bytes changed with no mark - what the GPU would read stale. Total since start.
		if (::Loader::DemonsSoulsWriteAudit::CheckStaleOn())
			std::printf("[nativo-check] viejos: %llu (desde el arranque; comprobacion muestreada 1 de cada 64 paginas)\n",
			            static_cast<unsigned long long>(::Loader::DemonsSoulsWriteAudit::StaleUsesTotal()));
		std::fflush(stdout);
		st.sweeps = st.sweep_pages = st.runs = st.written = st.dirtied = st.whole = st.failed = st.max_us = 0;
		st.sweep_us   = 0;
		st.collect_us = 0;
		st.last_print = now;
	}
}

// (policy1) See PolicyBeforeUse. Any thread that asks the cache (the GPU thread, a speculative translation's).
// (policy6, KYTY_DS_WRITE_FRAME=2) GPU work is about to use [vaddr, vaddr + size): each sampled page with a hash taken
// when it last became clean is looked at again; still no-watch and clean, with other bytes now, it is one the GPU would
// use stale (DemonsSoulsWriteAudit::CheckNoWatchUse counts and lists it). At most 16384 pages a call.
void BufferCache::PolicyCheckStale(uint64_t vaddr, uint64_t size, bool wide) const {
	auto&          tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	const uint64_t end     = vaddr + size;
	uint64_t       pages   = 0;
	for (uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1); page < end && pages < 16384; page += TRACKER_PAGE_SIZE, ++pages) {
		if (!::Loader::DemonsSoulsWriteAudit::SampledPage(page) || !::Loader::DemonsSoulsWriteAudit::HasCleanHash(page)) continue;
		// (PROT-MENOS7) Hot pages too (never write-protected: only the write watch read makes them dirty).
		const bool clean = (tracker.AnyNoWatch(page, TRACKER_PAGE_SIZE) || tracker.AnyHot(page, TRACKER_PAGE_SIZE)) &&
		                   !tracker.IsRegionCpuModified(page, TRACKER_PAGE_SIZE);
		::Loader::DemonsSoulsWriteAudit::CheckNoWatchUse(page, clean, vaddr, end, wide || size > (uint64_t {1} << 20u));
	}
}

void BufferCache::PolicyBeforeUseSlow(uint64_t vaddr, uint64_t size, bool is_written) const {
	auto& tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	if (!GuestRange {vaddr, size}.Valid()) return;
	// (VIGILANCIA-DOBLE, SEGURIDAD) The emulator's cache asked for pages a native piece said it serves: the piece is not
	// the only path after all. They go back to the normal watch at once (Descubrir; no-watch pages watched again; the
	// whole range CPU-modified, since unwatched pages may have been written unseen: this use uploads them); counted.
	if (::NativoCubierto::SeguridadUsoEmulador(vaddr, size)) {
		const uint64_t back = tracker.ClearNoWatch(vaddr, size, true);
		::Loader::DemonsSoulsWriteAudit::NoteReverted(vaddr, back, is_written ? 1 : 0);
		if (!is_written) tracker.MarkRegionAsCpuModified(vaddr, size);
		return;
	}
	if (!tracker.AnyNoWatch(vaddr, size)) return;
	if (::Loader::DemonsSoulsWriteAudit::CheckStaleOn()) PolicyCheckStale(vaddr, size, false); // (policy6)
	if (!is_written && ::Loader::DemonsSoulsWriteAudit::ReadCovered(vaddr, size)) {
		// (anuncio1) The no-watch pages are trusted here: KYTY_RAIZ_ANNOUNCE + KYTY_RAIZ_VERIFY audits them.
		if (::Loader::DemonsSoulsWriteAudit::AnnounceVerifyOn()) PolicyCheckAnnounced(vaddr, size, size > (uint64_t {1} << 20u));
		return;
	}
	// (anuncio2) KYTY_RAIZ_ANNOUNCE: a GPU write over frame memory leaves its pages no-watch (PolicyGpuWrite).
	if (is_written && ::Loader::DemonsSoulsWriteAudit::AnnounceOn()) {
		PolicyGpuWrite(vaddr, size);
		return;
	}
	// (HEAP-NATIVO) KYTY_HEAP_NATIVO=1: heap pages copied at this use, left writable (no veto); the rest as before.
	if (!is_written && HeapNativoOn() && HeapNativoTake(vaddr, size, false) != 0) return;
	// (nativo2) KYTY_DS_NATIVE_HEAP: a read no hook covers keeps the pages no-watch, native (written ones found by the OS
	// write watch before each submission); the pages new to it are CPU-dirty now, so this use uploads them.
	if (!is_written && NativeHeapOn()) {
		NativeHeapNoteKept(tracker.KeepNative(vaddr, size), false);
		return;
	}
	const uint64_t pages = tracker.ClearNoWatch(vaddr, size, true);
	::Loader::DemonsSoulsWriteAudit::NoteReverted(vaddr, pages, is_written ? 1 : 0);
	if (::Loader::DemonsSoulsWriteAudit::AnnounceOn()) ::Loader::DemonsSoulsWriteAudit::NoteVetoOther(vaddr, size, pages, 0); // (anuncio2)
}

// (anuncio2) KYTY_RAIZ_ANNOUNCE=1. GPU work is about to write [vaddr, vaddr + size). Measured (035513): ~16,000 pages/s
// of frame memory went back to their write watcher here, refused for good ("la GPU escribe ahi", 78,891 pages in 5 s),
// and each cost one write fault more (~20,000/s, up from ~1,000/s) when a frame allocator handed it out again
// (anuncio1's lift), plus the protection changes both ways. Frame-policy pages (DemonsSoulsWriteAudit::FramePolicyPage:
// handed out by a frame allocator lately, not a ring) stay no-watch now; they are only made CPU-dirty, so the upload
// that precedes the write (ForEachUploadRange) copies the guest's bytes into the buffer first - what ClearNoWatch's
// CPU-dirty did. Why that is enough:
//  - while GPU-dirty the page has its read watcher (no access: RegionManager::UpdateGpuProtection, independent of
//    no-watch); a CPU access faults and takes the normal path (SetNoWatch refuses a GPU-dirty page: why 1), as today;
//  - given back (written back, the read watcher gone) it is writable with no write watcher - no fault when the next
//    owner writes it -, and the CPU's writes there are announced by R1-R3 like those of any no-watch frame page
//    (ReadCovered trusts exactly these pages already); MarkNoWatchDirty and SetNoWatch never touch GPU-dirty pages;
//  - the other pages of the range: as before (watched again and refused).
void BufferCache::PolicyGpuWrite(uint64_t vaddr, uint64_t size) const {
	namespace Audit = ::Loader::DemonsSoulsWriteAudit;
	auto&          tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	const uint64_t end     = vaddr + size;
	uint64_t       kept = 0, reverted = 0;
	// Runs of pages of one kind: frame-policy pages (kept) or not.
	for (uint64_t at = vaddr; at < end;) {
		const bool ours = Audit::FramePolicyPage(at);
		uint64_t   next = (std::min)((at | (TRACKER_PAGE_SIZE - 1)) + 1, end);
		while (next < end && Audit::FramePolicyPage(next) == ours) next = (std::min)(next + TRACKER_PAGE_SIZE, end);
		if (ours) {
			(void)tracker.MarkNoWatchDirty(at, next - at);
			kept += ((next - 1) / TRACKER_PAGE_SIZE) - (at / TRACKER_PAGE_SIZE) + 1;
		} else {
			const uint64_t pages = tracker.ClearNoWatch(at, next - at, true);
			Audit::NoteReverted(at, pages, 1);
			Audit::NoteVetoOther(at, next - at, pages, 1);
			reverted += pages;
		}
		at = next;
	}
	Audit::NoteGpuWriteUse(kept, reverted);
}

// (anuncio1) KYTY_RAIZ_ANNOUNCE=1 + KYTY_RAIZ_VERIFY=1. GPU work is about to use [vaddr, vaddr + size), whose no-watch
// pages the policy trusts (ReadCovered). Each no-watch page the write watch says was written since it last became clean
// (RegionManager: reset then, before the upload's copy) and that no announcement made CPU-dirty since is one the GPU
// would use stale: counted and listed (DemonsSoulsWriteAudit::NoteUnannounced), and made CPU-dirty at once, so the use
// that follows uploads it - the audit repairs what it finds. A page written in bytes this use does not read counts the
// same (the write watch is per page): an upper bound. At most 512 written pages a call are looked at (more: counted).
void BufferCache::PolicyCheckAnnounced(uint64_t vaddr, uint64_t size, bool wide) const {
	auto&              tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	constexpr uint64_t Max     = 512;
	uint64_t           written[Max];
	const uint64_t     found = Libs::LibKernel::Memory::NativeAnnounceWritten(vaddr, size, written, Max);
	uint64_t           dirty = 0, stale = 0;
	for (uint64_t i = 0; i < found && i < Max; ++i) {
		const uint64_t page = written[i];
		if (!tracker.AnyNoWatch(page, TRACKER_PAGE_SIZE)) continue; // (watched: its writes faulted)
		if (tracker.IsRegionCpuModified(page, TRACKER_PAGE_SIZE)) {
			++dirty; // (announced, or never clean since it became no-watch)
			continue;
		}
		// (VIGILANCIA-DOBLE) Written with no announcement: besides the repair, the page goes back to the normal watch
		// (ClearNoWatch: CPU-dirty and watched), so the next unannounced write faults as before.
		const bool repaired = ::NativoCubierto::On() ? tracker.ClearNoWatch(page, TRACKER_PAGE_SIZE, true) != 0
		                                             : tracker.MarkNoWatchDirty(page, TRACKER_PAGE_SIZE) != 0;
		::Loader::DemonsSoulsWriteAudit::NoteUnannounced(page, vaddr, size, wide, repaired);
		++stale;
	}
	::Loader::DemonsSoulsWriteAudit::NoteAnnounceCheck(found, dirty, stale, found > Max);
}

// (policy4) See PolicyBdaRead.
void BufferCache::PolicyBdaReadSlow(uint64_t vaddr, uint64_t size) const {
	auto& tracker = const_cast<MemoryTracker&>(m_memory_tracker);
	if (!GuestRange {vaddr, size}.Valid() || !tracker.AnyNoWatch(vaddr, size)) return;
	if (::Loader::DemonsSoulsWriteAudit::CheckStaleOn()) PolicyCheckStale(vaddr, size, true); // (policy6: a BDA read)
	if (::Loader::DemonsSoulsWriteAudit::ReadCovered(vaddr, size)) {
		if (::Loader::DemonsSoulsWriteAudit::AnnounceVerifyOn()) PolicyCheckAnnounced(vaddr, size, true); // (anuncio1)
		return;
	}
	if (HeapNativoOn() && HeapNativoTake(vaddr, size, true) != 0) return; // (HEAP-NATIVO)
	if (NativeHeapOn()) { // (nativo2: a buffer read by address, as a read above)
		NativeHeapNoteKept(tracker.KeepNative(vaddr, size), true);
		return;
	}
	const uint64_t pages = tracker.ClearNoWatch(vaddr, size, true);
	::Loader::DemonsSoulsWriteAudit::NoteReverted(vaddr, pages, 2);
	if (::Loader::DemonsSoulsWriteAudit::AnnounceOn()) ::Loader::DemonsSoulsWriteAudit::NoteVetoOther(vaddr, size, pages, 2); // (anuncio2)
}

void BufferCache::EnsureBufferContents(uint64_t vaddr, uint64_t size) {
	PolicyBeforeUse(vaddr, size, false); // (policy1)
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

// (MALLA-NATIVA fase 1b) Memoria de GPU de un bloque de malla preparada por el hilo Preparer (malla-nativa.h): o VRAM
// visible por la CPU ya copiada (mapped), o VRAM normal + un staging ya copiado que el hilo de render copia en el
// prologo de subidas. Sin Buffer aun: el hilo de render lo envuelve (Buffer con asignacion, vkCreateBuffer + bind).
struct MallaPrep {
	VmaAllocation alloc         = nullptr;
	VkBuffer      staging       = VK_NULL_HANDLE;
	VmaAllocation staging_alloc = nullptr;
	uint8_t*      mapped        = nullptr;
};

static void* MallaPrepare(GraphicContext* g, uint64_t bytes, const uint8_t* data, int* reason) {
	// (fase 1c) Presupuesto: el monton de VRAM mas grande debe quedar con un 10 % libre tras el bloque; si no, camino normal.
	{
		const auto& mem     = g->physical_device_memory_properties;
		uint32_t    heap    = UINT32_MAX;
		uint64_t    largest = 0;
		for (uint32_t h = 0; h < mem.memoryHeapCount; h++) {
			if ((static_cast<VkMemoryHeapFlags>(mem.memoryHeaps[h].flags) & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
			    mem.memoryHeaps[h].size > largest)
				largest = mem.memoryHeaps[h].size, heap = h;
		}
		if (heap != UINT32_MAX) {
			VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(g->allocator, budgets);
			// (fase 1h) Dentro de la reserva de mallas basta con caber bajo el presupuesto del driver: la geometria es
			// pequena (~44 MB) y se usa cada fotograma; las texturas ceden (VRAM-DESG expulsa por budget_want).
			auto&      mr      = MallaNativa::Get();
			const bool reserve = mr.vram_bytes.load(std::memory_order_relaxed) + bytes <= MallaNativa::ReserveBytes();
			const uint64_t limit = reserve ? budgets[heap].budget : budgets[heap].budget - budgets[heap].budget / 10;
			if (budgets[heap].usage + bytes > limit) {
				*reason = MallaNativa::FailBudget;
				return nullptr;
			}
			if (reserve && budgets[heap].usage + bytes > budgets[heap].budget - budgets[heap].budget / 10)
				mr.in_reserve.fetch_add(1, std::memory_order_relaxed);
		}
	}
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.size  = bytes;
	info.usage = static_cast<VkBufferUsageFlags>(AllFlags);
	const uint32_t families[] {g->queue_family, g->readback_family};
	if (g->readback_family != static_cast<uint32_t>(-1)) { // (como Buffer::Buffer: los mismos requisitos)
		info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		info.queueFamilyIndexCount = 2;
		info.pQueueFamilyIndices   = families;
	}
	const auto dev = static_cast<VkDevice>(g->device);
	// (fase 1d) Antes: vmaAllocateMemoryForBuffer con VMA_MEMORY_USAGE_AUTO_*: VMA no conoce el uso del bufer por esa
	// via y rechaza toda reserva AUTO (VK_ERROR_FEATURE_NOT_PRESENT): "reserva VMA" en cada bloque. Ahora el mismo
	// camino que Buffer::Buffer (streamBuffer.cpp): vmaCreateBuffer con el VkBufferCreateInfo real (VMA ve el uso, BDA
	// incluido), sin el pool compactado de la cache (lo moveria la desfragmentacion); el VkBuffer de VMA se destruye y
	// la asignacion queda para el Buffer que la envuelve en el hilo de render. VMA es seguro entre hilos (asignador sin
	// EXTERNALLY_SYNCHRONIZED, el mismo g->allocator).
	VkResult   first_result = VK_SUCCESS;
	const char* first_what  = nullptr;
	const auto note = [&](VkResult r, const char* what) {
		if (r != VK_SUCCESS && first_result == VK_SUCCESS) first_result = r, first_what = what;
	};
	const auto create = [&](const VmaAllocationCreateInfo& ci, VmaAllocation* out, VmaAllocationInfo* res, const char* what) {
		VkBuffer   tmp = VK_NULL_HANDLE;
		const auto r   = vmaCreateBuffer(g->allocator, &info, &ci, &tmp, out, res);
		note(r, what);
		if (r != VK_SUCCESS) return false;
		VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(dev, tmp, nullptr); // (la asignacion sigue)
		return true;
	};
	auto p = std::make_unique<MallaPrep>();
	VmaAllocationInfo result {};
	// 1) VRAM visible por la CPU (ReBAR/BAR), dentro del presupuesto: copia directa. (Sin ReBAR: 256 MB, se agota pronto.)
	VmaAllocationCreateInfo visible {};
	visible.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
	                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	visible.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	visible.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	if (create(visible, &p->alloc, &result, "VRAM visible") && result.pMappedData != nullptr) {
		p->mapped = static_cast<uint8_t*>(result.pMappedData);
		std::memcpy(p->mapped, data, bytes);
		vmaFlushAllocation(g->allocator, p->alloc, 0, bytes);
		return p.release();
	}
	if (p->alloc != nullptr) vmaFreeMemory(g->allocator, p->alloc);
	p->alloc = nullptr;
	const VkResult visible_result = first_result;
	first_result = VK_SUCCESS, first_what = nullptr; // (sin BAR libre es lo normal: no es el fallo que se busca)
	// 2) VRAM normal (como Buffer::Buffer con MemoryUsage::DeviceLocal: con poca VRAM, sin el bit de presupuesto; el
	// presupuesto ya lo vigila la comprobacion de arriba) + staging copiado aqui (la copia de GPU, en el prologo).
	VmaAllocationCreateInfo local {};
	local.flags         = g->small_video_memory ? VmaAllocationCreateFlags {} : VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	local.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	local.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkBufferCreateInfo stage_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	stage_info.size  = bytes;
	stage_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo stage {};
	stage.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	stage.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	VmaAllocationInfo stage_result {};
	bool ok = create(local, &p->alloc, &result, "VRAM");
	if (ok) {
		const auto r = vmaCreateBuffer(g->allocator, &stage_info, &stage, &p->staging, &p->staging_alloc, &stage_result);
		note(r, "staging");
		ok = r == VK_SUCCESS && stage_result.pMappedData != nullptr;
	}
	if (!ok) {
		*reason = p->alloc == nullptr ? MallaNativa::FailAlloc : MallaNativa::FailStaging;
		static std::atomic<bool> logged {false};
		if (!logged.exchange(true)) {
			const auto& mem  = g->physical_device_memory_properties;
			const auto  type = result.memoryType;
			std::printf("[malla-nativa] primer fallo de reserva: %s, VkResult %d (VRAM visible antes: %d), %llu bytes, tipo de "
			            "memoria %u (flags 0x%x, monton %u), poca VRAM %d\n",
			            first_what != nullptr ? first_what : "staging sin mapear", static_cast<int>(first_result),
			            static_cast<int>(visible_result), static_cast<unsigned long long>(bytes), type,
			            type < mem.memoryTypeCount ? static_cast<unsigned>(mem.memoryTypes[type].propertyFlags) : 0u,
			            type < mem.memoryTypeCount ? mem.memoryTypes[type].heapIndex : 0u, g->small_video_memory ? 1 : 0);
		}
		if (p->staging != VK_NULL_HANDLE) vmaDestroyBuffer(g->allocator, p->staging, p->staging_alloc);
		if (p->alloc != nullptr) vmaFreeMemory(g->allocator, p->alloc);
		return nullptr; // (fallido para siempre: camino normal)
	}
	std::memcpy(stage_result.pMappedData, data, bytes);
	vmaFlushAllocation(g->allocator, p->staging_alloc, 0, bytes);
	return p.release();
}

static void MallaDiscard(GraphicContext* g, void* opaque) {
	auto* p = static_cast<MallaPrep*>(opaque);
	if (p->staging != VK_NULL_HANDLE) vmaDestroyBuffer(g->allocator, p->staging, p->staging_alloc);
	if (p->alloc != nullptr) vmaFreeMemory(g->allocator, p->alloc);
	delete p;
}

// (F5 pase 2, sombras nativas) Solo MALLA-NATIVA: el VkBuffer del bloque de malla que contiene [vaddr, vaddr + size)
// y su desplazamiento, por el mismo Lookup que ObtainBuffer; {nullptr, 0} si no lo hay ahora (sin camino normal).
std::pair<Buffer*, uint64_t> BufferCache::ObtainMallaNativa(uint64_t vaddr, uint64_t size) {
	if (!MallaNativa::On()) return {nullptr, 0};
	if (Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr) return {nullptr, 0};
	if (!MallaNativa::Get().backend.load(std::memory_order_acquire)) return {nullptr, 0}; // (aun sin preparar: ObtainBuffer lo hace)
	uint64_t    page_lo = 0;
	void* const native  = MallaNativa::Lookup(
        vaddr, size, &page_lo, [this](void* opaque, uint64_t lo, uint64_t bytes, uint8_t** mapped, bool* vram) -> void* {
            auto*             p = static_cast<MallaPrep*>(opaque);
            vk::CommandBuffer prologue {};
            if (p->staging != VK_NULL_HANDLE) {
                prologue = m_scheduler.UploadPrologue(0, UINT64_MAX);
                if (!prologue) return nullptr;
            }
            auto* buffer = new Buffer(m_graphics, m_scheduler, lo, AllFlags, bytes, p->alloc);
            if (p->staging != VK_NULL_HANDLE) {
                const vk::BufferCopy region {0, 0, bytes};
                prologue.copyBuffer(vk::Buffer(p->staging), buffer->Handle(), 1, &region);
                m_scheduler.DeferOperation([allocator = m_graphics.allocator, sb = p->staging, sa = p->staging_alloc] {
                    vmaDestroyBuffer(allocator, sb, sa);
                });
                MallaNativa::Get().staged.fetch_add(1, std::memory_order_relaxed);
            }
            *mapped = p->mapped;
            *vram   = true;
            delete p;
            return buffer;
        });
	if (native == nullptr) return {nullptr, 0};
	return {static_cast<Buffer*>(native), vaddr - page_lo};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	const FrameNativo2::ObtainTimer fn2_obtain_timer; // (FRAME-NATIVO2 v13, =6: coste por uso; nada con otros modos)
	const Envio1::Scope envio1_obtain(Envio1::BufHit); // (envio1: the read-only rest; its serial parts nested below)
	if (ArenaDueno::On()) ArenaDueno::OnGpuUse(vaddr, size, is_written); // (ARENA-DUENO: uso de la GPU por instancia duena)
	// (MALLA-NATIVA, KYTY_MALLA_NATIVA=1) Un rango dentro de un bloque de malla (registrado en 0xaa06a0): su VkBuffer
	// propio, creado una vez con las paginas del bloque; sin vigilancia, sin sincronizar, nada registrado en la cache.
	// (fase 1j) Nunca dentro de una traduccion especulativa ni de un grabador de otro hilo: alli el VkBuffer no queda
	// atado al tick en que se envia de verdad (la fosa lo podia destruir antes) y DeferOperation lanza RecorderRefusal
	// (el unique_ptr lo destruia al instante, con la GPU usandolo). Ese uso va por el camino normal.
	if (MallaNativa::On() && !is_written && !is_texel_buffer &&
	    (Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr)) {
		MallaNativa::Get().spec_skipped.fetch_add(1, std::memory_order_relaxed);
	} else if (MallaNativa::On() && !is_written && !is_texel_buffer) {
		for (void* dead: MallaNativa::TakeGraveyard()) { // (mallas destruidas en 0xaa1d10: tras la GPU)
			if (!m_scheduler.Active()) { // (fase 1j: antes se destruia aqui mismo; se guarda hasta poder diferirlo)
				MallaNativa::KeepGraveyard(dead);
				continue;
			}
			std::unique_ptr<Buffer> gone(static_cast<Buffer*>(dead));
			m_scheduler.DeferOperation([owner = std::move(gone)]() mutable { owner.reset(); });
		}
		if (!MallaNativa::Get().backend.load(std::memory_order_acquire)) {
			GraphicContext* g = &m_graphics;
			MallaNativa::SetBackend([g](uint64_t, uint64_t bytes, const uint8_t* data, int* reason) {
				return MallaPrepare(g, bytes, data, reason);
			},
			                        [g](void* p) { MallaDiscard(g, p); });
		}
		uint64_t    page_lo = 0;
		void* const native  = MallaNativa::Lookup(
            vaddr, size, &page_lo, [this](void* opaque, uint64_t lo, uint64_t bytes, uint8_t** mapped, bool* vram) -> void* {
                auto*             p = static_cast<MallaPrep*>(opaque);
                vk::CommandBuffer prologue {};
                if (p->staging != VK_NULL_HANDLE) {
                    prologue = m_scheduler.UploadPrologue(0, UINT64_MAX); // (antes de todo el bufer abierto)
                    if (!prologue) return nullptr;                        // (ahora no: el siguiente uso)
                }
                auto* buffer = new Buffer(m_graphics, m_scheduler, lo, AllFlags, bytes, p->alloc);
                if (p->staging != VK_NULL_HANDLE) {
                    const vk::BufferCopy region {0, 0, bytes};
                    prologue.copyBuffer(vk::Buffer(p->staging), buffer->Handle(), 1, &region);
                    m_scheduler.DeferOperation([allocator = m_graphics.allocator, sb = p->staging, sa = p->staging_alloc] {
                        vmaDestroyBuffer(allocator, sb, sa);
                    });
                    MallaNativa::Get().staged.fetch_add(1, std::memory_order_relaxed);
                }
                *mapped = p->mapped;
                *vram   = true;
                delete p;
                return buffer;
            });
		if (native != nullptr) return {static_cast<Buffer*>(native), vaddr - page_lo};
	}
	// (FRAME-NATIVO2, KYTY_FRAME_NATIVO2=1) A read inside an imported DefaultTagg arena: its imported buffer, the guest's
	// bytes where they are (no watch, no protection, no upload, nothing registered).
	// (v8, =3) Escritura de la GPU en un arena sin proteger: apuntada con su fotograma (se descarta tras NextFrame).
	if (is_written && FrameNativo2::NoProt()) {
		if (const int a = FrameNativo2::Touching(vaddr, size);
		    a >= 0 && FrameNativo2::g_excluded[a].load(std::memory_order_acquire)) {
			// (v9) Antes de usarlo, lo liberado ya descartado (solo en el hilo de la GPU; si no, la siguiente lectura).
			if (!AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation() &&
			    CommandScheduler::ThreadRecorder() == nullptr && GuestGpu::IsGpuThread())
				FrameNativo2DrainFreed(m_memory_tracker, [this](uint64_t a, uint64_t s) { FrameNativo2DropGpu(a, s); });
			FrameNativo2::g_gpu_writes.fetch_add(1, std::memory_order_relaxed);
			// (v10) Lo soltado por FreeTag ya no se marca entero: el rango exacto de este uso, si no es de la GPU.
			if (m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(vaddr, size))
				FrameNativo2::g_marked_bytes.fetch_add(size, std::memory_order_relaxed);
		}
	}
	// (v11, =4) Bloques sin vigilar de instancias de solo lectura.
	if (FrameNativo2::SoloLectura() && FrameNativo2::RoAny(vaddr, size)) {
		if (is_written) {
			// Red de seguridad: la GPU escribe donde se creia de solo lectura. Se apunta y esa instancia vuelve a la
			// vigilancia: lo que la CPU escribio sin verse pasa a CPU-modificado (ya, y entero en el hilo de la GPU).
			std::vector<std::pair<uint64_t, uint64_t>> back;
			FrameNativo2::OnUnexpectedGpuWrite(vaddr, size, back);
			for (const auto& [a, s]: back) {
				(void)m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(a, s);
				FrameNativo2::PushRewatch(a, s);
			}
		} else if (is_texel_buffer && m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(vaddr, size)) {
			FrameNativo2::g_marked_bytes.fetch_add(size, std::memory_order_relaxed);
		}
	}
	// (v10, =3) Lectura texel del arena sin proteger: el rango exacto de este uso CPU-modificado (si no es de la GPU).
	if (!is_written && is_texel_buffer && FrameNativo2::NoProt()) {
		if (const int a = FrameNativo2::Touching(vaddr, size);
		    a >= 0 && FrameNativo2::g_excluded[a].load(std::memory_order_acquire) &&
		    m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(vaddr, size))
			FrameNativo2::g_marked_bytes.fetch_add(size, std::memory_order_relaxed);
	}
	if (FrameNativo2::Wanted() && !is_written && !is_texel_buffer) {
		if (const auto native = FrameNativo2Obtain(vaddr, size); native.first != nullptr) return native;
		// (v2) Other uses (texel, written) of an excluded arena go the old way, but its pages are never protected again:
		// no write fault tells the tracker, so the range is taken as CPU-modified at each use (unless the GPU wrote it).
		if (const int a = FrameNativo2::Touching(vaddr, size);
		    (a >= 0 && FrameNativo2::g_excluded[a].load(std::memory_order_acquire)) ||
		    (FrameNativo2::SoloLectura() && FrameNativo2::RoAny(vaddr, size))) { // (v11: tambien bloques sin vigilar)
			// (v4) Only when no page of the range is GPU-dirty (checked under the regions' locks: never both states;
			// a GPU-written range stays GPU-owned and goes the cache's way).
			if (m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(vaddr, size))
				FrameNativo2::g_marked_bytes.fetch_add(size, std::memory_order_relaxed);
		}
	}
	PolicyBeforeUse(vaddr, size, is_written); // (policy1)
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
		const Envio1::Scope envio1_dirty(Envio1::Dirty); // (envio1)
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
			const Envio1::Scope envio1_mod(Envio1::BufMod); // (envio1)
			InvalidateCopyFeedback(vaddr, size);
			NoteGpuWrite(vaddr, size);
			Spec::NoteGpuWrite(vaddr, size);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
			NoteNativeWrite(vaddr, size);
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
		const Envio1::Scope envio1_mod(Envio1::BufMod); // (envio1)
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
		NoteNativeWrite(vaddr, size);
		memo = {vaddr, size, RegistrationEpoch(), m_gpu_coverage_epoch, id, is_texel_buffer};
	}
	return {buffer, buffer->Offset(vaddr)};
}

// (trad1, KYTY_TRAD1; local/trad1.h) ObtainBuffer's write memo path for a range the caller proved: the same calls, in the
// same order, without the memo's lookup, the page table, SynchronizeBuffer and the range Add (the coverage epoch
// unchanged since the obtain that added the range: none of its bytes left m_gpu_modified_ranges, its pages are still
// GPU-owned: SynchronizeBuffer would return at once, the Add would add nothing).
bool BufferCache::Trad1RenoteWritten(BufferId id, const Buffer* expect, uint64_t vaddr, uint64_t size,
                                     uint64_t registration, uint64_t coverage) {
	if (Spec::Current() != nullptr || size == 0) return false;
	PolicyBeforeUse(vaddr, size, true); // (policy1)
	DrainGuestReadback(vaddr, size, false, true);
	// (After the drain: a readback it finished gave bytes back and moved the coverage epoch. trad1b: only a loss in the
	// range's own tracker regions counts; the rest of m_gpu_modified_ranges is not what SynchronizeBuffer looks at.)
	if (Trad1CoverageMoved(vaddr, size, coverage) || RegistrationEpoch() != registration) return false;
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer != expect || buffer->is_deleted || !buffer->IsInBounds(vaddr, size)) return false;
	if (LiveTrace::WriteTicks()) LiveTrace::Event(LiveTrace::BufferUse, vaddr, size | uint64_t {1} << 63u);
	TouchBuffer(*buffer);
	LaneSync::NoteRange(buffer->lane_marks, vaddr, size, true);
	const Envio1::Scope envio1_mod(Envio1::BufMod); // (envio1)
	InvalidateCopyFeedback(vaddr, size);
	NoteGpuWrite(vaddr, size);
	Spec::NoteGpuWrite(vaddr, size);
	LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
	LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
	NoteNativeWrite(vaddr, size);
	return true;
}

// (trad1, KYTY_TRAD1=2) SynchronizeBuffer's test for a written range, without its work.
bool BufferCache::Trad1SyncWouldUpload(uint64_t vaddr, uint64_t size) {
	if (kyty_local_buffer_residency_mode.load(std::memory_order_relaxed) != 0 && native_buffer_download_depth == 0 &&
	    m_gpu_modified_ranges.Contains(vaddr, size))
		return false;
	return !m_memory_tracker.IsRegionFullyGpuModified(vaddr, size);
}

// (tiron7, KYTY_TIRON7=1: local/tiron7.h) Imported guest memory a recorded image upload reads on the GPU. Render thread only
// (imports are not made on a recorder's thread, in a deferred operation or on the upload worker).
namespace {
struct Tiron7Import {
	std::unique_ptr<Buffer> buffer;
	uint64_t                guest = 0, guest_size = 0; // the guest range it covers
	uint64_t                tick  = 0;                 // the work that reads it
};
struct Tiron7State {
	GraphicContext*          graphics  = nullptr;
	CommandScheduler*        scheduler = nullptr;
	uint64_t                 alignment = 0;
	bool                     usable    = false;
	std::vector<Tiron7Import> live;
	// (v2) Guest ranges refused before, and when: not tried again for 10 s (a refusal costs a driver call each time).
	std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::chrono::steady_clock::time_point>> refused;
	std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};
Tiron7State& Tiron7S() {
	static Tiron7State state;
	return state;
}

void Tiron7Report() {
	auto&      st  = Tiron7S();
	const auto now = std::chrono::steady_clock::now();
	if (now - st.last_report < std::chrono::seconds(5)) return;
	st.last_report = now;
	std::printf("[tiron7] importadas %llu (%.0f MB, %.1f ms en total de import), fallidas %llu (fueron por el camino de antes), "
	            "liberadas %llu, vivas %llu (%.0f MB); esperas a la GPU por un drenado %llu (%.1f ms); fallos %.1f ms, saltados "
	            "por fallo previo %llu\n",
	            static_cast<unsigned long long>(Tiron7::g_imports.load()), Tiron7::g_import_bytes.load() / 1e6,
	            Tiron7::g_import_us.load() / 1000.0, static_cast<unsigned long long>(Tiron7::g_fails.load()),
	            static_cast<unsigned long long>(Tiron7::g_frees.load()), static_cast<unsigned long long>(st.live.size()),
	            Tiron7::g_live_bytes.load() / 1e6, static_cast<unsigned long long>(Tiron7::g_drain_waits.load()),
	            Tiron7::g_drain_wait_us.load() / 1000.0, Tiron7::g_fail_us.load() / 1000.0,
	            static_cast<unsigned long long>(Tiron7::g_skipped.load()));
	std::fflush(stdout);
}

// Frees the imports whose work is complete.
void Tiron7Reclaim() {
	auto& st = Tiron7S();
	std::erase_if(st.live, [&](Tiron7Import& import) {
		if (import.tick >= CommandScheduler::PendingTick || !st.scheduler->IsFree(import.tick)) return false;
		Tiron7::g_live_bytes.fetch_sub(import.buffer->Size(), std::memory_order_relaxed);
		Tiron7::g_frees.fetch_add(1, std::memory_order_relaxed);
		return true;
	});
}

// AsyncUpload::DrainRange: guest [vaddr, vaddr + size) is about to change (unmap, offload, a command processor write):
// the GPU work reading it through an import completes first, then the import is freed.
void Tiron7Drain(uint64_t vaddr, uint64_t size) {
	auto& st = Tiron7S();
	if (st.live.empty() || AsyncUpload::t_worker || CommandScheduler::ThreadRecorder() != nullptr) return;
	uint64_t wait = 0;
	for (const auto& import: st.live)
		if (import.guest < vaddr + size && vaddr < import.guest + import.guest_size) wait = std::max(wait, import.tick);
	if (wait == 0) return;
	if (!st.scheduler->IsFree(wait)) {
		const auto start = std::chrono::steady_clock::now();
		st.scheduler->Wait(wait);
		Tiron7::g_drain_waits.fetch_add(1, std::memory_order_relaxed);
		Tiron7::g_drain_wait_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                                            std::chrono::steady_clock::now() - start)
		                                                            .count()),
		                                  std::memory_order_relaxed);
	}
	Tiron7Reclaim();
}

// The guest's bytes of [vaddr, vaddr + size) as a buffer the GPU reads them from, at the returned offset; null: not
// importable (the caller takes the old path).
std::pair<Buffer*, uint64_t> Tiron7Import(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t vaddr,
                                          uint64_t size, const uint8_t* source) {
	auto& st = Tiron7S();
	if (st.graphics == nullptr) {
		st.graphics  = &graphics;
		st.scheduler = &scheduler;
		if (Tiron7::g_extension.load(std::memory_order_relaxed)) {
			VkPhysicalDeviceExternalMemoryHostPropertiesEXT host {};
			host.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
			VkPhysicalDeviceProperties2 props {};
			props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			props.pNext = &host;
			VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(graphics.physical_device),
			                                                             &props);
			st.alignment = host.minImportedHostPointerAlignment;
			st.usable    = st.alignment != 0 && (st.alignment & (st.alignment - 1)) == 0 &&
			            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT != nullptr;
			AsyncUpload::g_import_drain = Tiron7Drain;
		}
		std::printf("[tiron7] import de memoria del juego %s (alineacion %llu)\n", st.usable ? "listo" : "NO usable: como antes",
		            static_cast<unsigned long long>(st.alignment));
		std::fflush(stdout);
	}
	Tiron7Reclaim();
	Tiron7Report();
	// (A copy's buffer offset is a multiple of the texel block: the staging ring's 16, kept.)
	if (!st.usable || (reinterpret_cast<uint64_t>(source) & 15u) != 0) return {nullptr, 0};
	const auto start   = std::chrono::steady_clock::now();
	std::erase_if(st.refused, [&](const auto& entry) { return start - entry.second > std::chrono::seconds(10); });
	for (const auto& entry: st.refused)
		if (entry.first.first < vaddr + size && vaddr < entry.first.second) {
			Tiron7::g_skipped.fetch_add(1, std::memory_order_relaxed);
			return {nullptr, 0};
		}
	const auto address = reinterpret_cast<uint64_t>(source);
	const auto base    = address & ~(st.alignment - 1);
	const auto end     = (address + size + st.alignment - 1) & ~(st.alignment - 1);
	const auto bytes   = end - base;
	const auto device  = static_cast<VkDevice>(graphics.device);
	auto&      vk      = VULKAN_HPP_DEFAULT_DISPATCHER;
	uint32_t   host_bits = 0, buffer_bits = 0;
	const auto fail    = [&](const char* stage, VkResult result, VkBuffer buffer, VkDeviceMemory memory) {
		if (buffer != VK_NULL_HANDLE) vk.vkDestroyBuffer(device, buffer, nullptr);
		if (memory != VK_NULL_HANDLE) vk.vkFreeMemory(device, memory, nullptr);
		st.refused.push_back({{vaddr, vaddr + size}, std::chrono::steady_clock::now()});
		Tiron7::g_fail_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                                      std::chrono::steady_clock::now() - start)
		                                                      .count()),
		                            std::memory_order_relaxed);
		if (Tiron7::g_fails.fetch_add(1, std::memory_order_relaxed) < 16) {
			unsigned long state = 0, protect = 0, type = 0, allocation_protect = 0;
			uint64_t      region = 0, allocation = 0;
#if defined(_WIN32)
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(base), &info, sizeof(info)) != 0) {
				state = info.State; protect = info.Protect; type = info.Type; allocation_protect = info.AllocationProtect;
				region = info.RegionSize; allocation = reinterpret_cast<uint64_t>(info.AllocationBase);
			}
#endif
			std::printf("[tiron7] import refused at %s: VkResult %d, guest 0x%llx size 0x%llx host %p base 0x%llx bytes 0x%llx "
			            "(alignment %llu), memoryTypeBits host 0x%x buffer 0x%x; host memory: state 0x%lx protect 0x%lx type 0x%lx "
			            "allocation 0x%llx (protect 0x%lx) region 0x%llx: the old path\n",
			            stage, static_cast<int>(result), static_cast<unsigned long long>(vaddr),
			            static_cast<unsigned long long>(size), static_cast<const void*>(source),
			            static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes),
			            static_cast<unsigned long long>(st.alignment), host_bits, buffer_bits, state, protect, type,
			            static_cast<unsigned long long>(allocation), allocation_protect, static_cast<unsigned long long>(region));
			std::fflush(stdout);
		}
		return std::pair<Buffer*, uint64_t> {nullptr, 0};
	};

	VkMemoryHostPointerPropertiesEXT host_props {};
	host_props.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	if (const auto result = vk.vkGetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	                                                               reinterpret_cast<const void*>(base), &host_props);
	    result != VK_SUCCESS || host_props.memoryTypeBits == 0)
		return fail("vkGetMemoryHostPointerPropertiesEXT", result, VK_NULL_HANDLE, VK_NULL_HANDLE);
	host_bits = host_props.memoryTypeBits;
	VkExternalMemoryBufferCreateInfo external {};
	external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.pNext       = &external;
	buffer_info.size        = bytes;
	buffer_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	const uint32_t families[] {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) { // (as Buffer::Buffer)
		buffer_info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families;
	}
	VkBuffer buffer = VK_NULL_HANDLE;
	if (const auto result = vk.vkCreateBuffer(device, &buffer_info, nullptr, &buffer); result != VK_SUCCESS)
		return fail("vkCreateBuffer", result, VK_NULL_HANDLE, VK_NULL_HANDLE);
	VkMemoryRequirements requirements {};
	vk.vkGetBufferMemoryRequirements(device, buffer, &requirements);
	buffer_bits          = requirements.memoryTypeBits;
	const uint32_t types = requirements.memoryTypeBits & host_props.memoryTypeBits;
	VkPhysicalDeviceMemoryProperties memory_props {};
	vk.vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(graphics.physical_device), &memory_props);
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory_props.memoryTypeCount && type == UINT32_MAX; ++i)
		if ((types & (1u << i)) != 0 && (memory_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
			type = i;
	if (type == UINT32_MAX || requirements.size > bytes) return fail("memory type", VK_SUCCESS, buffer, VK_NULL_HANDLE);
	VkImportMemoryHostPointerInfoEXT import_info {};
	import_info.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
	import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import_info.pHostPointer = reinterpret_cast<void*>(base);
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.pNext           = &import_info;
	allocate.allocationSize  = bytes;
	allocate.memoryTypeIndex = type;
	VkDeviceMemory memory    = VK_NULL_HANDLE;
	VramDrv::Scope vram_drv(VramDrv::CatHostImport); // (VRAM-DRV)
	if (const auto result = vk.vkAllocateMemory(device, &allocate, nullptr, &memory); result != VK_SUCCESS)
		return fail("vkAllocateMemory", result, buffer, VK_NULL_HANDLE);
	if (const auto result = vk.vkBindBufferMemory(device, buffer, memory, 0); result != VK_SUCCESS)
		return fail("vkBindBufferMemory", result, buffer, memory);
	// cpu_address: the guest address the buffer's byte 0 stands for (Offset(vaddr) = the image's bytes).
	const uint64_t guest_base = vaddr - (address - base);
	auto&          import     = st.live.emplace_back();
	import.buffer     = std::make_unique<Buffer>(graphics, scheduler, guest_base, vk::Buffer(buffer), vk::DeviceMemory(memory),
                                             bytes);
	import.guest      = vaddr;
	import.guest_size = size;
	import.tick       = scheduler.CurrentTick();
	Tiron7::g_imports.fetch_add(1, std::memory_order_relaxed);
	Tiron7::g_import_bytes.fetch_add(size, std::memory_order_relaxed);
	Tiron7::g_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
	Tiron7::g_import_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                                        std::chrono::steady_clock::now() - start)
	                                                        .count()),
	                              std::memory_order_relaxed);
	return {import.buffer.get(), address - base};
}
} // namespace

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size, bool* refillable) {
	PolicyBeforeUse(vaddr, size, false); // (policy1)
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
	// (tiron7, KYTY_TIRON7=1) A large upload: the GPU reads the guest's bytes where they are (local/tiron7.h); neither
	// copied by the upload worker nor waited for at the submission. Not importable: the paths below.
	if (Tiron7::ImportOn() && size >= Tiron7::MinBytes() && !AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation() &&
	    CommandScheduler::ThreadRecorder() == nullptr) {
		if (const auto* source = Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, size); source != nullptr) {
			ApplyImageProtects();
			if (const auto imported = Tiron7Import(m_graphics, m_scheduler, vaddr, size, source); imported.first != nullptr) {
				path = "tiron7";
				if (refillable != nullptr) *refillable = false;
				return imported;
			}
		}
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
	// (ww2: not with KYTY_NATIVE_MEMORY=1 and KYTY_WRITE_WATCH=1, as UploadCopies: the worker would read guest memory
	// itself later, and fault on a page GPU work made no-access meanwhile.)
	if (staging != nullptr && kyty_local_async_upload_mode.load(std::memory_order_relaxed) >= 2 &&
	    UploadWorkerAllowed() && m_staging_buffer.IsCoherent()) { // (raiz2)
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
	if (const auto* source = UploadSourcePointer(vaddr, size)) { // (raiz2)
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
	    !UploadWorkerAllowed() || !ring.IsCoherent()) // (ww2: see ObtainBufferForImage; raiz2: UploadWorkerAllowed)
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
	                   UploadWorkerAllowed() && m_staging_buffer.IsCoherent(); // (ww2: see ObtainBufferForImage; raiz2)
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
			if (const auto* source = UploadSourcePointer(piece.vaddr, piece.size)) { // (raiz2)
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
	{
		const Spec::ClearOriginScope origin(3); // (spec5)
		(void)m_texture_cache.ClearMeta(vaddr, value);
	}
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

// An in-order copy's bytes that change: the upload jobs pending over them (work recorded before the copy, whose bytes the
// upload worker has not taken yet) are done first, so that work reads the bytes from before the copy, as it does on the
// PS5; as a label's write waits for them (WriteGuestMemory). Only the runs written: a page the copy leaves as it was is
// the same for those jobs either way. (Found by CarpetHead reading mode 4's commit against the in-order path, PR #349's
// "known, not fixed": at 1-1 such jobs were pending, ~70 us a frame of waits when drained over the whole destination.)
// Live switch copy_drain_range (KYTY_COPY_DRAIN_RANGE=0: not waited for). (A function of its own: CopyGuestMemory keeps
// its shape, and with it its PGO profile.)
static void DrainCopyDestination(uint64_t vaddr, uint64_t size) {
	static auto& drain = LiveSwitch::Find("copy_drain_range", 1);
	if (drain.load(std::memory_order_relaxed) != 0) AsyncUpload::DrainRange(vaddr, size);
}

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
		DrainCopyDestination(address, bytes); // (up1118, upstream 7f7ec8a: in-order copies)
		if (!m_memory_tracker.IsRegionFullyCpuModified(address, bytes)) {
			// (As the write faults would: the images over the pages first.)
			m_texture_cache.InvalidateMemory(address, bytes);
			// (The pages are not GPU-modified, CopyBuffer checked: no GPU result is written back there.)
			// (raiz3, KYTY_RAIZ: not kept read-only - no alias view: writing through it took a window, 2 VirtualProtect
			// and the gate alone, 6,600-9,300 a second at Boletaria -; unprotected below, as a write fault would.)
			if (!Raiz::On() && m_memory_tracker.MarkRegionAsCpuDirtyKeepProtection(address, bytes) &&
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
		if (NativeWriteWatchOn()) {
			// (KYTY_NATIVE_MEMORY: the write fault of this memcpy on a page an image watches invalidated the image before
			// the bytes landed; here no fault comes, and NativeForgetWrites below hides the write from the sweeps: the
			// images over it go now. Pages CPU-dirty for the buffers skipped that above. And the game's own writes there
			// since the last sweep are kept for it first: the forgetting would lose them.)
			m_texture_cache.InvalidateMemory(address, bytes);
			PageManager::CaptureCpuWrites(address, bytes);
		}
		std::memcpy(reinterpret_cast<void*>(address), from, bytes); Libs::LibKernel::Memory::NativeForgetWrites(address, bytes);
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
	PolicyBeforeUse(vaddr, size, false); // (policy1)
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
	// (BUF-VRAM) Under pressure the GPU-written ones stay where they are: the collector below (RunGarbageCollector,
	// aggressive) downloads their bytes to guest memory and deletes them. bda_read ones are still touched (a read of one
	// gone through a pointer past its first 64 KiB would read zeros once).
	for (const auto id: kept) {
		auto& buffer = m_slot_buffers[id];
		if (BufVramOn() && pressure && m_total_used_memory >= m_critical_gc_memory && !buffer.bda_read) continue;
		// (BDA-RETIRO) Un bda_read sin uso real en IdleMs, con la VRAM critica y sin bytes de la GPU: fuera (ver arriba).
		if (buffer.bda_read && BdaRetiro::On()) {
			const int64_t ahora = BdaRetiro::AhoraMs();
			if (buffer.bda_use_ms == 0) buffer.bda_use_ms = ahora; // (primera vez visto: cuenta desde ahora)
			if (!buffer.bda_contado) {
				buffer.bda_contado = true;
				if (BdaRetiro::g_retirados.contains(buffer.CpuAddress())) { // volvio tras retirarlo: se queda fijo
					BdaRetiro::g_vueltos++;
					BdaRetiro::g_fijos_bytes += buffer.Size();
				}
			}
			if (pressure && stale.size() < limit && ahora - buffer.bda_use_ms >= BdaRetiro::IdleMs &&
			    !BdaRetiro::g_retirados.contains(buffer.CpuAddress()) &&
			    !m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
				BdaRetiro::g_retirados.insert(buffer.CpuAddress());
				BdaRetiro::g_n++;
				BdaRetiro::g_bytes += buffer.Size();
				stale.push_back(id);
				continue;
			}
			// (Tocado solo en la LRU: el sello de uso real no cambia.)
			m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
			if (buffer.lru_id < kBufGid) GestorMem::Tocar(g_buf_gid[buffer.lru_id].load(std::memory_order_relaxed)); // (M7)
			continue;
		}
		TouchBuffer(buffer);
	}
	if (stale.empty()) return;
	for (const auto id: stale) {
		auto& buffer = m_slot_buffers[id];
		g_bufvram_evicted++;                       // (BUF-VRAM)
		g_bufvram_evicted_bytes += buffer.Size(); // (BUF-VRAM)
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		DeleteBuffer(id);
	}
	LiveCounters::Add(LiveCounters::GcBufferDeletes, stale.size());
}

void BufferForceSystemMemory(bool on); // (BUF-RAM) streamBuffer.cpp

// (BUF-RAM, 11-25) Nivel 2 de local/vram-tope.h. Log 20261125-074401 (1440p + FSR 4, 8 GB simulados): bufferes de la
// cache 1000-1300 MiB, todos en VRAM ("en RAM 0 MiB"); en [buf-vram] "sin usar >30 s" 0-17 MiB (casi todo escrito por la
// GPU), y el recolector borra 120-370 MiB/30 s de bufferes estaticos sin usar >10 s que luego se vuelven a subir.
// Aqui, con el uso >= 95 % del tope: un bufer sin uso de la GPU en >= ColdMs (no bda_read: esas lecturas no dejan marca
// en la LRU, no se puede saber si esta frio) pasa a un bufer nuevo en memoria del sistema (VMA, tipos sin DEVICE_LOCAL;
// la GPU lo lee por PCIe), con los mismos pasos que MoveSpilledBufferBack / Defragment: copia GPU->GPU en el command
// buffer que se graba (CopyFrom pone las barreras), el viejo fuera (DeleteBuffer: su VRAM se libera cuando la GPU acaba
// con el, EraseRetired), el nuevo dentro (Register: pagina BDA nueva). Una sola copia en la GPU en todo momento.
// Vuelta: si la LRU lo marca usado tras migrar (o pasa a bda_read) y uso + tamano < 90 % del tope; uno devuelto no se
// vuelve a migrar en 60 s. Limites: 64 MiB/s a RAM, 64 MiB/s de vuelta, 1536 MiB en RAM como mucho. Nada a disco.
// (BUF-MEM) Censo de cada ~5 s: donde vive cada bufer de la cache (VRAM / memoria del sistema visible por la GPU) por
// tipo, las mallas nativas, el pool y la VRAM que el pool devolvio al driver (sus bloques liberados por VMA).
void BufferCache::BufMemCensus() {
	BufMem::Censo c;
	m_slot_buffers.ForEach([&](auto /*id*/, const Buffer& b) {
		if (b.is_deleted.load(std::memory_order_relaxed) || b.Size() == 0 || b.Allocation() == nullptr) return;
		const bool ram = b.IsInSystemMemory();
		const bool gpu = !b.bda_read && GuestRange {b.CpuAddress(), b.Size()}.Valid() &&
		                  m_memory_tracker.IsRegionGpuModified(b.CpuAddress(), b.Size());
		(b.bda_read ? (ram ? c.ram_bda : c.vram_bda) : gpu ? (ram ? c.ram_gpu : c.vram_gpu) : (ram ? c.ram_static : c.vram_static)) +=
		    b.Size();
	});
	if (MallaNativa::On()) {
		c.mesh_vram = MallaNativa::Get().vram_bytes.load(std::memory_order_relaxed);
		c.mesh_ram  = MallaNativa::Get().sys_bytes.load(std::memory_order_relaxed);
	}
	if (m_graphics.game_buffer_pool != nullptr) {
		VmaStatistics st {};
		vmaGetPoolStatistics(m_graphics.allocator, m_graphics.game_buffer_pool, &st);
		c.pool_blocks = st.blockBytes, c.pool_used = st.allocationBytes;
		static uint64_t last_blocks = 0;
		if (last_blocks > st.blockBytes) BufMem::g_returned.fetch_add(last_blocks - st.blockBytes, std::memory_order_relaxed);
		last_blocks = st.blockBytes;
	}
	BufMem::Log(c);
}

void BufferCache::BufRamPass(uint64_t tick) {
	// (VG-AB, 11-25) 5 s -> 30 s: log 102943 migro 616 MiB en 30 s y volvieron ~540 (en RAM 73 MiB a los 60 s; el
	// recolector borro 582 MiB vs 154 sin el grupo) y [buf-vram] daba "sin usar >30 s" 0 MiB: eran bufferes en uso.
	constexpr int64_t  ColdMs      = 30000;
	constexpr uint64_t RateBytes   = uint64_t {64} << 20u; // por segundo, cada sentido
	constexpr uint64_t MaxRamBytes = uint64_t {1536} << 20u;
	constexpr uint64_t MinSize     = uint64_t {64} << 10u;
	constexpr int64_t  CooldownMs  = 60000;
	const bool bufmem = BufMem::On(); // (BUF-MEM) siempre activo: frio = sin uso de la GPU en ColdFrames fotogramas
	if (!VramTope::BufRamOn() && !bufmem) return;
	BufMem::Enlazar(); // (M7) BufMem::Presion como callback de presion de VRAM del gestor (una vez)
	// (M7) Migrar a RAM lo decide la presion de VRAM del gestor central (>= 95 % de su presupuesto), no el nivel 2 de
	// vram-tope.h (VramTope::g_bufram_permitido). Sin gestor, como antes.
	const bool bufram_permitido = GestorMem::On() ? GestorMem::Presion(GestorMem::Vram) >= 0.95
	                                              : VramTope::g_bufram_permitido.load(std::memory_order_relaxed);
	if (bufmem && tick % 300 == 0) BufMemCensus(); // (linea [buf-mem], ~5 s)
	// (W17) KYTY_VRAM_MITAD=1: sin tope de MiB en RAM, 512 MiB/s, frio = sin uso en los ultimos 60 fotogramas (la LRU
	// da los menos usados primero), y vuelta a VRAM solo bajo el 90 % de la mitad del tope.
	const bool     mitad     = VramTope::Mitad();
	const uint64_t rate      = mitad ? (uint64_t {512} << 20u) : bufmem ? BufMem::RateBytes : RateBytes;
	const uint64_t max_ram   = mitad ? (uint64_t {1} << 40u) : bufmem ? BufMem::MaxRamBytes : MaxRamBytes;
	constexpr uint64_t FriosFotogramas = 60;
	static int64_t  window_ms = 0;
	static uint64_t out_win = 0, back_win = 0;
	const int64_t   now_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()).count();
	if (now_ms - window_ms >= 1000) {
		window_ms = now_ms;
		out_win = back_win = 0;
	}
	// (En limite de fotograma, sobre el command buffer que se graba, fuera de especulacion y de una compactacion.)
	if (!m_scheduler.Active() || m_scheduler.Current().IsInvalid() || Spec::Current() != nullptr) return;
	if (m_defrag != nullptr || m_defrag_pass_open || m_defrag_ending.valid()) return;
	uint64_t tope = VramTope::g_tope_bytes.load(std::memory_order_relaxed);
	uint64_t uso  = VramTope::g_uso_bytes.load(std::memory_order_relaxed);
	if (GestorMem::On() && GestorMem::Presupuesto(GestorMem::Vram) != 0) { // (M7) el presupuesto y el uso del gestor
		tope = GestorMem::Presupuesto(GestorMem::Vram);
		uso  = GestorMem::Uso(GestorMem::Vram);
	}
	if (tope == 0 && bufmem) { // (BUF-MEM sin vram-tope: el presupuesto del driver del monton de VRAM mas grande)
		const auto& mem  = m_graphics.physical_device_memory_properties;
		uint32_t    heap = UINT32_MAX;
		uint64_t    big  = 0;
		for (uint32_t h = 0; h < mem.memoryHeapCount; h++)
			if ((static_cast<VkMemoryHeapFlags>(mem.memoryHeaps[h].flags) & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
			    mem.memoryHeaps[h].size > big)
				big = mem.memoryHeaps[h].size, heap = h;
		if (heap != UINT32_MAX) {
			VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(m_graphics.allocator, budgets);
			tope = budgets[heap].budget, uso = budgets[heap].usage;
		}
	}
	if (tope == 0) return;
	const uint64_t tope_ef = mitad ? tope / 2 : tope; // (W17)
	const auto copy_to = [&](BufferId id, bool to_ram) -> bool {
		const auto begin = m_slot_buffers[id].CpuAddress();
		const auto size  = m_slot_buffers[id].Size();
		if (to_ram) BufferForceSystemMemory(true);
		const auto moved_id = m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, begin,
		                                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
		BufferForceSystemMemory(false);
		if (m_slot_buffers[moved_id].IsInSystemMemory() != to_ram) { // (sin sitio / sin tipo de memoria)
			m_slot_buffers.erase(moved_id);
			return false;
		}
		auto&       moved  = m_slot_buffers[moved_id];
		const auto& source = m_slot_buffers[id];
		moved.bda_read     = source.bda_read;
		moved.IncreaseStreamScore(source.StreamScore());
		if (to_ram) {
			SetVulkanObjectNameF(m_graphics.device, moved.Handle(), "Kyty.GameBufferRam[guest=0x{:016x} size=0x{:x}]", begin, size);
		} else {
			SetVulkanObjectNameF(m_graphics.device, moved.Handle(), "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", begin, size);
		}
		moved.CopyFrom(m_scheduler.Current(), source, 0, 0, size);
		NoteGpuWrite(begin, size);
		// (M7) El id del gestor pasa al bufer nuevo: Register(moved_id) hace CambiarLugar (VRAM <-> RAM), no alta/baja.
		const uint64_t mover = source.lru_id < kBufGid ? g_buf_gid[source.lru_id].exchange(0, std::memory_order_relaxed) : 0;
		DeleteBuffer(id); // (si estaba en g_bufram, sale de ahi)
		g_mover_gid = mover;
		Register(moved_id);
		g_mover_gid = 0;
		(to_ram ? BufMem::g_out : BufMem::g_back).fetch_add(size, std::memory_order_relaxed); // (BUF-MEM)
		if (to_ram) BufMem::g_out_since_defrag.fetch_add(size, std::memory_order_relaxed);
		if (to_ram) {
			std::erase(m_spilled, moved_id); // (lo gestiona BufRamPass, no MoveSpilledBufferBack)
			g_bufram.push_back({moved_id, m_lru_cache.TickOf(moved.lru_id), size});
			VramTope::g_bufram_bytes.fetch_add(size, std::memory_order_relaxed);
			VramTope::g_bufram_n.fetch_add(1, std::memory_order_relaxed);
			VramTope::g_bufram_out.fetch_add(size, std::memory_order_relaxed);
		} else {
			VramTope::g_bufram_back.fetch_add(size, std::memory_order_relaxed);
			g_bufram_cooldown[begin] = now_ms + CooldownMs;
		}
		return true;
	};
	// 1) Vuelta de los usados (cada 30 flips: el recorrido es de los que estan en RAM).
	if (tick % 30 == 0 && !g_bufram.empty()) {
		uint64_t           hot_bytes = 0;
		uint64_t           room_uso  = uso;
		std::vector<std::pair<BufferId, uint64_t>> back;
		for (const auto& e: g_bufram) {
			const auto* b = m_slot_buffers.try_get(e.id);
			if (b == nullptr || b->is_deleted) continue;
			if (!b->bda_read && m_lru_cache.TickOf(b->lru_id) <= e.tick_mig) continue;
			hot_bytes += e.size;
			if (back_win + e.size <= rate && (room_uso + e.size) * 100 < tope_ef * VramTope::BufRamBackPct) {
				back.emplace_back(e.id, e.size);
				back_win += e.size;
				room_uso += e.size;
			}
		}
		for (const auto& [id, size]: back) {
			if (copy_to(id, false)) hot_bytes -= std::min(hot_bytes, size);
		}
		VramTope::g_bufram_calientes.store(hot_bytes, std::memory_order_relaxed);
	}
	// 2) Migracion de frios con el nivel 2 activo.
	// (BUF-MEM) Lo sacado deja huecos en los bloques del pool: compactacion ya (Defragment), los bloques vacios se liberan.
	if (bufmem && BufMem::g_out_since_defrag.load(std::memory_order_relaxed) >= BufMem::DefragAfterBytes) {
		BufMem::g_out_since_defrag.store(0, std::memory_order_relaxed);
		m_defrag_next_check = 0;
		m_defrag_free_after = 0;
	}
	// (BUF-MEM) Presion del gestor central: ese mismo flip saca frios sin esperar a la ventana.
	const uint64_t presion = bufmem ? BufMem::TomarPresion() : 0;
	if (!(bufmem || bufram_permitido) || (tick % 10 != 0 && presion == 0)) return;
	if (out_win >= rate || VramTope::g_bufram_bytes.load(std::memory_order_relaxed) >= max_ram) return;
	const auto horizon = (std::chrono::steady_clock::now() - std::chrono::milliseconds(ColdMs)).time_since_epoch().count();
	uint64_t   first = tick - std::min<uint64_t>(tick, m_tick_times.size() - 1), last = tick;
	if (m_tick_times[first % m_tick_times.size()] > horizon) {
		last = first;
	} else {
		while (first < last) {
			const uint64_t middle = first + (last - first + 1) / 2;
			if (m_tick_times[middle % m_tick_times.size()] <= horizon) first = middle;
			else last = middle - 1;
		}
	}
	if (mitad) last = tick > FriosFotogramas ? tick - FriosFotogramas : 0; // (W17) por fotogramas, no por 30 s
	else if (bufmem && !bufram_permitido)
		last = tick > BufMem::ColdFrames ? tick - BufMem::ColdFrames : 0; // (BUF-MEM) sin uso en ColdFrames fotogramas
	if (last == 0) return;
	std::erase_if(g_bufram_cooldown, [&](const auto& c) { return c.second <= now_ms; });
	std::vector<BufferId> cold;
	uint64_t              want = 0;
	m_lru_cache.ForEachItemBelow(last, [&](BufferId id) {
		const auto& b = m_slot_buffers[id];
		if (b.is_deleted || b.bda_read || b.IsInSystemMemory() || b.Usage() != MemoryUsage::DeviceLocal ||
		    b.Allocation() == nullptr || b.Size() < MinSize || b.CpuAddress() == 0)
			return false;
		if (g_bufram_cooldown.contains(b.CpuAddress())) return false;
		cold.push_back(id);
		want += b.Size();
		return out_win + want >= std::max(rate / 10, presion) || cold.size() >= (mitad || presion != 0 ? size_t {64} : size_t {16}); // (~1/10 del limite por pasada: 10 pasadas/s)
	});
	for (const auto id: cold) {
		const auto size = m_slot_buffers[id].Size();
		if (!copy_to(id, true)) break; // (sin memoria del sistema para el: se para hasta la proxima pasada)
		out_win += size;
	}
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
	constexpr uint64_t PassBytes = uint64_t {32} << 20u;
	// (BUF-VRAM) 64 MiB free start a compaction (256 before: the ~210 MiB free between compactions stayed).
	// (VRAM-EMU, KYTY_EMU_POOL_MINFREE_MB, por defecto 32; =64 como antes) Log 20261125-202747: tras cada compactacion
	// quedaban 24-37 MiB libres y entre compactaciones el hueco subia a 50-116 MiB (lineas [buf-mem] "hueco") antes de la
	// siguiente; con el umbral en un bloque (32 MiB) se compacta antes. Mismas pasadas (32 MiB, 64 movimientos por flip).
	static const uint64_t emu_min_free = [] {
		const char*    v  = std::getenv("KYTY_EMU_POOL_MINFREE_MB");
		const uint64_t mb = v != nullptr && *v != '\0' ? std::strtoull(v, nullptr, 10) : 64; // (11-26 limpieza) 64 como antes
		std::printf("[vram-emu-pool] KYTY_EMU_POOL_MINFREE_MB=%llu: el pool de bufferes del juego se compacta con %llu MiB "
		            "libres (antes 64); la linea \"Game buffers compacted\" da lo libre tras cada una\n",
		            static_cast<unsigned long long>(mb), static_cast<unsigned long long>(mb));
		std::fflush(stdout);
		return (mb == 0 ? 64 : mb) << 20u;
	}();
	const uint64_t MinFree = BufVramOn() ? emu_min_free : uint64_t {256} << 20u;
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
	VramBufReport(); // (VRAM-BUF)
	const auto tick = m_gc_tick++;
	RetireStaleBuffers(tick);
	// (BUF-VRAM) Every 30 s: the buffers' video memory now and 30 s ago, what went, and where the rest is (not used for
	// 30 s: GPU-written / read through pointers (bda_read) / other; the pool's free bytes; the device-local rings).
	if (BufVramOn()) {
		static auto     last_report = std::chrono::steady_clock::now();
		static uint64_t last_bytes  = 0;
		const auto      now         = std::chrono::steady_clock::now();
		if (now - last_report >= std::chrono::seconds(30)) {
			last_report = now;
			const auto horizon = (now - std::chrono::seconds(30)).time_since_epoch().count();
			uint64_t   n = 0, total = 0, sys = 0, idle_gpu = 0, idle_bda = 0, idle_other = 0, bda = 0, gpu = 0, bda_frio = 0;
			const auto horizon5 = (now - std::chrono::seconds(5)).time_since_epoch().count();
			uint64_t   c5_gpu = 0, c5_bda = 0, c5_est = 0; // (BUF-RAM)
			m_slot_buffers.ForEach([&](auto id, const Buffer& b) {
				if (id.index == NULL_BUFFER_ID.index || b.is_deleted.load(std::memory_order_relaxed) || b.Size() == 0) return;
				n++;
				total += b.Size();
				if (b.IsInSystemMemory()) sys += b.Size();
				const bool written = m_memory_tracker.IsRegionGpuModified(b.CpuAddress(), b.Size());
				if (b.bda_read) bda += b.Size();
				if (b.bda_read && b.bda_use_ms != 0 && BdaRetiro::AhoraMs() - b.bda_use_ms >= BdaRetiro::IdleMs) bda_frio += b.Size();
				if (written) gpu += b.Size();
				const uint64_t t    = m_lru_cache.TickOf(b.lru_id);
				const bool     idle = t + m_tick_times.size() <= tick || m_tick_times[t % m_tick_times.size()] <= horizon;
				// (BUF-RAM) frios >5 s (lo que migra el nivel 2) por tipo; en RAM no cuenta.
				if (!b.IsInSystemMemory() &&
				    (t + m_tick_times.size() <= tick || m_tick_times[t % m_tick_times.size()] <= horizon5)) {
					if (b.bda_read) c5_bda += b.Size();
					else if (written) c5_gpu += b.Size();
					else c5_est += b.Size();
				}
				if (!idle) return;
				if (b.bda_read) idle_bda += b.Size();
				else if (written) idle_gpu += b.Size();
				else idle_other += b.Size();
			});
			uint64_t pool_blocks = 0, pool_used = 0;
			if (m_graphics.game_buffer_pool != nullptr) {
				VmaStatistics statistics {};
				vmaGetPoolStatistics(m_graphics.allocator, m_graphics.game_buffer_pool, &statistics);
				pool_blocks = statistics.blockBytes;
				pool_used   = statistics.allocationBytes;
			}
			const auto mib = [](uint64_t v) { return static_cast<unsigned long long>(v >> 20u); };
			std::printf("[buf-vram] 30 s: bufferes %llu MiB antes -> %llu MiB ahora (%llu vivos, en RAM %llu MiB); expulsados %llu "
			            "(%llu MiB) + escritos por la GPU descargados y expulsados %llu (%llu MiB); sin usar >30 s: escritos "
			            "GPU %llu, bda %llu, otros %llu MiB; bda total %llu, GPU total %llu MiB; pool %llu MiB (libre %llu); "
			            "presion %s\n",
			            mib(last_bytes), mib(total), static_cast<unsigned long long>(n), mib(sys),
			            static_cast<unsigned long long>(g_bufvram_evicted), mib(g_bufvram_evicted_bytes),
			            static_cast<unsigned long long>(g_bufvram_dirty), mib(g_bufvram_dirty_bytes), mib(idle_gpu),
			            mib(idle_bda), mib(idle_other), mib(bda), mib(gpu), mib(pool_blocks),
			            mib(pool_blocks - std::min(pool_blocks, pool_used)), m_retire_pressure ? "si" : "no");
			std::printf("[buf-ram] 30 s: en VRAM sin uso de la GPU >5 s: estaticos %llu MiB, escritos por la GPU %llu MiB "
			            "(migrables al nivel 2), bda %llu MiB (no migrables: su uso no se ve); en RAM %llu MiB (%llu)\n",
			            mib(c5_est), mib(c5_gpu), mib(c5_bda), mib(VramTope::g_bufram_bytes.load()),
			            static_cast<unsigned long long>(VramTope::g_bufram_n.load()));
			// (BDA-RETIRO) La cifra a vigilar: bda total al principio y al final de la partida (log 201501: 349 -> 550 MiB).
			std::printf("[bda-retiro] 30 s (KYTY_BDA_RETIRO=%d): bufferes bda_read %llu MiB en VRAM, sin uso real >120 s %llu MiB; "
			            "retirados en total %llu (%llu MiB); vueltos a registrar (fijos desde entonces) %llu (%llu MiB)\n",
			            BdaRetiro::On() ? 1 : 0, mib(bda), mib(bda_frio), static_cast<unsigned long long>(BdaRetiro::g_n),
			            mib(BdaRetiro::g_bytes), static_cast<unsigned long long>(BdaRetiro::g_vueltos), mib(BdaRetiro::g_fijos_bytes));
			std::fflush(stdout);
			last_bytes = total;
			g_bufvram_evicted = g_bufvram_evicted_bytes = g_bufvram_dirty = g_bufvram_dirty_bytes = 0;
		}
	}
	// (VRAM-DESG) KYTY_VRAM_DESG=0 lo apaga. Cada 1800 pasadas (~30 s, un recorrido de ~4000 bufferes): cuantos hay, su
	// memoria, la que esta en RAM del sistema y la que se solapa (el mismo rango del juego en dos bufferes: copia doble en
	// la memoria de video). Para el desglose de la VRAM; no cambia nada.
	static const bool vram_desg = [] { const char* v = std::getenv("KYTY_VRAM_DESG"); return v == nullptr || *v != '0'; }();
	if (vram_desg && tick % 1800 == 0) {
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		spans.reserve(4096);
		uint64_t n = 0, total = 0, sys = 0;
		m_slot_buffers.ForEach([&](auto /*id*/, const Buffer& b) {
			if (b.is_deleted.load(std::memory_order_relaxed) || b.Size() == 0) return;
			n++;
			total += b.Size();
			if (b.IsInSystemMemory()) sys += b.Size();
			spans.emplace_back(b.CpuAddress(), b.CpuAddress() + b.Size());
		});
		std::sort(spans.begin(), spans.end());
		uint64_t uni = 0, cur_b = 0, cur_e = 0;
		for (const auto& [sb, se]: spans) {
			if (sb >= cur_e) { uni += cur_e - cur_b; cur_b = sb; cur_e = se; }
			else if (se > cur_e) cur_e = se;
		}
		uni += cur_e - cur_b;
		std::printf("[vram-desg] bufferes: %llu vivos, %llu MiB (en RAM del sistema %llu MiB); rango del juego unico %llu MiB, "
		            "solapado %llu MiB (bufferes sobre la misma memoria del juego)\n",
		            static_cast<unsigned long long>(n), static_cast<unsigned long long>(total >> 20u),
		            static_cast<unsigned long long>(sys >> 20u), static_cast<unsigned long long>(uni >> 20u),
		            static_cast<unsigned long long>((total - std::min(total, uni)) >> 20u));
		std::fflush(stdout);
	}
	// The game buffers in system memory are looked at every 60 flips, collection or not (this game's frames all read
	// guest memory through device addresses). At the end of a submission its command buffer still records
	// (EndSubmission, before the flush).
	if (tick % 60 == 0) (void)MoveSpilledBufferBack();
	BufRamPass(tick); // (BUF-RAM)
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
	// (11-23) KYTY_BUFFER_GC_MIN_IDLE_S (default 10, 0 = as before): never a buffer used in the last n seconds. With the
	// video memory at the critical level (an 8 GB card: images fill their budget, buffers ~1 GB, driver ~1.5 GB) the 80
	// flips (under a second at 100 fps) deleted buffers the game uses again a second later: in a 16-minute log, 16
	// frames of 100-220 ms in 75 s each re-uploaded ~200 MB (buffer_sync ~110 ms, backing reads ~2 GB/s), the same
	// buffers every 1-2 s. Keeping fewer deleted is always correct (the collector only frees); the memory stays used
	// for those seconds (RetireStaleBuffers' PressureTime is the same 10 s).
	static const int64_t min_idle_s = [] {
		const char* v = std::getenv("KYTY_BUFFER_GC_MIN_IDLE_S");
		return v != nullptr && *v != 0 ? std::max<int64_t>(0, std::atoll(v)) : int64_t {10};
	}();
	uint64_t below = tick - age;
	if (min_idle_s > 0) {
		const auto horizon = (std::chrono::steady_clock::now() - std::chrono::seconds(min_idle_s)).time_since_epoch().count();
		uint64_t   first   = tick - std::min<uint64_t>(tick, m_tick_times.size() - 1), last = tick;
		if (m_tick_times[first % m_tick_times.size()] > horizon) {
			last = first;
		} else {
			while (first < last) {
				const uint64_t middle = first + (last - first + 1) / 2;
				if (m_tick_times[middle % m_tick_times.size()] <= horizon) first = middle;
				else last = middle - 1;
			}
		}
		below = std::min(below, last);
	}
	static uint64_t gc_deleted = 0, gc_deleted_bytes = 0;
	static auto     gc_report  = std::chrono::steady_clock::now();

	std::vector<BufferId> dirty_buffers;
	std::vector<DownloadCopy> copies;
	size_t                    retire_count = 0;
	m_lru_cache.ForEachItemBelow(below, [&](BufferId id) {
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
			g_bufvram_dirty++;                       // (BUF-VRAM)
			g_bufvram_dirty_bytes += buffer.Size(); // (BUF-VRAM)
		} else {
			gc_deleted++;
			gc_deleted_bytes += buffer.Size();
			g_bufvram_evicted++;                       // (BUF-VRAM)
			g_bufvram_evicted_bytes += buffer.Size(); // (BUF-VRAM)
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	LiveCounters::Add(LiveCounters::GcBufferDeletes, retire_count);
	if (const auto now = std::chrono::steady_clock::now(); now - gc_report >= std::chrono::seconds(5)) {
		// (11-23) The collector's buffer deletions (clean ones; the GPU-written ones are downloaded first, below).
		std::printf("[bufgc] 5 s: buffers borrados por memoria de video %llu (%llu MiB); edad minima %lld s "
		            "(KYTY_BUFFER_GC_MIN_IDLE_S), %s\n",
		            static_cast<unsigned long long>(gc_deleted), static_cast<unsigned long long>(gc_deleted_bytes >> 20u),
		            static_cast<long long>(min_idle_s), aggressive ? "nivel critico" : "nivel de aviso");
		gc_deleted = gc_deleted_bytes = 0;
		gc_report  = now;
	}
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
	// (trad2, KYTY_TRAD2) A request with no CPU-dirty page (and no no-watch page the policy could give back) is not
	// walked: SynchronizeBuffersInRange would find every piece clean (SynchronizeBuffer's first check, over a part of the
	// request) and upload nothing. local/trad2.h.
	if (const uint32_t trad2 = Trad2::Mode(); trad2 != 0) {
		Trad2::g.why[request.cpu_epoch == 0 ? Trad2::Unproven
		             : epoch == 0           ? Trad2::EpochZero
		             : request.registration_epoch != registered ? Trad2::Registration
		                                                        : Trad2::EpochMoved] += 1;
		const bool     sample_fast = Trad2::SampleFast();
		const uint64_t fast_start  = sample_fast ? __rdtsc() : 0;
		const bool     dirty       = m_memory_tracker.IsRegionCpuModified(request.address, request.size);
		const bool     nowatch     = !dirty && ::Loader::DemonsSoulsWriteAudit::SplitOn() &&
		                     m_memory_tracker.AnyNoWatch(request.address, request.size);
		if (dirty || nowatch) {
			Trad2::g.result[dirty ? Trad2::Dirty : Trad2::NoWatch] += 1;
			const bool     sample = dirty && Trad2::SampleDirty();
			const uint64_t start  = sample ? __rdtsc() : 0;
			{
				LiveCounters::Add(LiveCounters::RegionSyncs);
				LiveCounters::ScopedUs timed(LiveCounters::RegionSyncUs);
				SynchronizeBuffersInRange(request.address, request.size);
			}
			if (sample) {
				Trad2::g.dirty_cycles += __rdtsc() - start;
				++Trad2::g.dirty_samples;
			}
		} else {
			Trad2::g.result[Trad2::Shortcut] += 1;
			// (The proof as the old path's below: the epochs read before the check, unchanged after it.)
			const bool proven = epoch != 0 && RegionRegistrationEpoch(request.address) == registered &&
			                    m_memory_tracker.CpuModificationEpoch(request.address, request.size) == epoch;
			if (sample_fast) {
				Trad2::g.fast_cycles += __rdtsc() - fast_start;
				++Trad2::g.fast_samples;
			}
			if (!Trad2::ShortcutRunsOld()) {
				request.cpu_epoch = 0;
				if (proven) {
					++Trad2::g.proven;
					request.cpu_epoch          = epoch;
					request.registration_epoch = registered;
				}
				Trad2::MaybeReport(LiveControl::g_flips.load(std::memory_order_relaxed));
				return;
			}
			// The old path over it (mode 2: always; mode 1: 1 in 32), timed and checked: it must upload nothing.
			++Trad2::g.checked;
			const bool     sample = Trad2::SampleCleanTiming();
			const uint64_t runs   = Trad2::g_upload_runs;
			const uint64_t start  = sample ? __rdtsc() : 0;
			{
				LiveCounters::Add(LiveCounters::RegionSyncs);
				LiveCounters::ScopedUs timed(LiveCounters::RegionSyncUs);
				SynchronizeBuffersInRange(request.address, request.size);
			}
			if (sample) {
				Trad2::g.clean_cycles += __rdtsc() - start;
				++Trad2::g.clean_samples;
			}
			if (Trad2::g_upload_runs != runs) Trad2::NoteDifference(request.address, request.size, Trad2::g_upload_runs - runs);
		}
		request.cpu_epoch = 0;
		if (epoch != 0 && RegionRegistrationEpoch(request.address) == registered &&
		    m_memory_tracker.CpuModificationEpoch(request.address, request.size) == epoch) {
			request.cpu_epoch = epoch;
			request.registration_epoch = registered;
		}
		Trad2::MaybeReport(LiveControl::g_flips.load(std::memory_order_relaxed));
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
			// (policy4) A buffer shaders read by address: no page of it stays without its watcher unless hooks
			// announce all of its bytes (before the stamp: the pages given back are CPU-dirty and move the epoch).
			if (it->buffer->bda_read) PolicyBdaRead(start, finish - start);
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
