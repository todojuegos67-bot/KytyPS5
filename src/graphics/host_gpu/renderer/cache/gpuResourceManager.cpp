#include "local/envio3.h"
#include "local/sync-stats.h"
#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "live-census.h"
#include "loader/demonsSoulsTextureProbe.h"
#include "loader/demonsSoulsMemMap.h"
#include "loader/demonsSoulsWriteAudit.h"
#include "live-counters.h"
#include "raiz.h" // (raiz1: KYTY_RAIZ)
#include "parpadeo7.h" // (parpadeo7: KYTY_PARPADEO7)
#include "lane-sync.h"
#include "local/serie1.h" // (serie1: KYTY_SERIE1)
#include "async-upload.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <optional>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <utility>
#include <x86intrin.h>

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"
#include "speculation-state.h"
#include "fresh-memory.h"
#include "texture-offload.h"
#include "gpu-residency.h"

extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_bda_dirty_regions_mode {0};
// 1: watchers removed while memory is released skip the host protection change there
// (PageManager::SetUnmappingRange); a write fault on a page nothing watches unprotects it.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_unmap_protect_skip_mode {0};
// Live switch (same-process A/B): a write fault in the 16 KiB window right after the one the same thread's last window
// fault ended in, within 2 ms (a sequential stream of writes), takes the next 64 KiB at once. (1-1 route, 120 cap, A-B-A
// over 29 points: window faults -13.6%, fps +0.7%, 1% low +3.8%, p99 -4.1%; uploads +9%, on the upload worker.)
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_window_stream_mode {1};
// (performance-switches.h, KYTY_SPECULATE; spec5: the sweeps log for its commits.)
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
}

namespace Libs::Graphics {

namespace {
// (render3) Defined below with what they switch.
bool BigWindowGrow();
bool BdaMemoOn();
} // namespace

GpuResourceManager::GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler), m_buffer_cache(graphics, scheduler, m_page_manager, m_texture_cache, this),
      m_texture_cache(graphics, scheduler, m_page_manager, m_buffer_cache), m_graphics(graphics) {
	GpuResidency::Start();
	// (render3) Read here (and their start-up lines printed), not at a guest thread's first write fault.
	(void)BigWindowGrow();
	(void)BdaMemoOn();
}

GpuResourceManager::~GpuResourceManager() = default;

void GpuResourceManager::BeginAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_add((uint64_t{1} << 32u) + 1u,
		    std::memory_order_acq_rel);
		EXIT_IF(uint32_t(old >> 32u) == UINT32_MAX || uint32_t(old) == UINT32_MAX);
	}
}

void GpuResourceManager::EndAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_sub(1u, std::memory_order_release);
		EXIT_IF(uint32_t(old) == 0);
	}
}

uint64_t GpuResourceManager::PreparationAliasEpoch() const noexcept {
	const auto value = m_preparation_alias_epoch.load(std::memory_order_acquire);
	return uint32_t(value) == 0 ? value : 0;
}

namespace {
// (render3) KYTY_BIG_WINDOW_GROW=1 (off by default; needs KYTY_BIG_WINDOW=1): in the big-window granules a write fault
// no longer makes its whole aligned 1 MiB dirty. Every page a window opens is CPU-dirty until uploaded, written or
// not, and the next BDA preparation uploads all of them and protects them again: with flat 1 MiB windows the buffer
// uploads went from ~9 MB to 50-66 MB a frame for the same ~250 window faults a frame (logs of 14-11 21:06 and 15-11
// 06:56). With this switch a fault there opens the normal window (16 KiB, 64 KiB for a stream), and a stream that
// goes on (the fault right after the window this thread's last fault opened, within ~15 ms) doubles its window each
// time: 128, 256, 512 KiB, 1 MiB, forward only and within the 4 MiB tracker region. What is uploaded is then at most
// about twice what a sequential writer wrote; memory behind the fault (already uploaded) is not made dirty again.
// (rend1) KYTY_REND1=1 turns this on too (unless KYTY_BIG_WINDOW_GROW=0 says otherwise): logs 070419-003 / 081749-853
// (KYTY_RAIZ=1) have 37-47 MB of buffer uploads a frame at 1.6-1.8 k table draws and 59-77 MB at 5-11 k, 18-37 k pages a
// frame opened by ~130-260 window faults (1 MiB each: big_window flat 70-160 a frame), and every opened page uploaded
// and write-protected again (reprotect_pages 10-18 k a frame); render3's table: 9 MB a frame with the normal window for
// the same ~250 faults. Same primitive, smaller geometry: nothing else changes.
bool BigWindowGrow() {
	static const bool on = [] {
		const char* v    = std::getenv("KYTY_BIG_WINDOW_GROW");
		const char* rend = std::getenv("KYTY_REND1");
		const bool  is   = (v != nullptr && v[0] == '1') ||
		                ((v == nullptr || v[0] != '0') && rend != nullptr && rend[0] == '1' && rend[1] == '\0');
		if (is) {
			std::printf("Big window: grows with the stream (KYTY_BIG_WINDOW_GROW=1%s): 16/64 KiB, then 128 KiB to 1 MiB "
			            "forward, instead of an aligned 1 MiB at every fault\n",
			            (v != nullptr && v[0] == '1') ? "" : ", by KYTY_REND1=1");
			std::fflush(stdout);
		}
		return is;
	}();
	return on;
}
} // namespace

// (esperas4) KYTY_BIG_WINDOW_HOT (on by default; =0 turns it off): log 20261123-080842-324 has ~24,000 window faults a
// second, ~85% of them in the big-window granules (cpuheap/DefaultTagg/cpubkt/ctx), in 268 regions of 2 MiB: the
// game's per-frame allocators (eboot+0x812046 writes each new block's header, eboot+0x9f411f the first word of a block
// just handed out) write forward through memory every upload made clean again. The grow rule above only sees a stream
// when the SAME thread's next fault is the window right after its last one; the game's workers interleave, so most of
// these faults got the 16 KiB window. Two rules, same primitive (TryInvalidateCpuWriteWindow: no GPU-dirty page, no
// image, no pending download; nothing else changes):
//   - region stream: the window right after the last window ANY thread opened in that 4 MiB tracker region, within
//     ~20 ms, doubles that window (64 KiB to 1 MiB, forward, inside the region): what is opened is at most about twice
//     what a forward writer wrote, as the per-thread rule;
//   - hot block: a 64 KiB block that took a window fault in the last ~250 ms (rewritten every frame: the upload made
//     it clean and protected, and the game wrote it again) gets its whole aligned 64 KiB at once, not 16 KiB.
// Counters on the "[esperas] ventanas calientes" line (with KYTY_ESPERAS=1).
namespace {
bool BigWindowHot() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BIG_WINDOW_HOT");
		const bool  is = !(v != nullptr && v[0] == '0');
		std::printf("Big window hot (esperas4): %s (KYTY_BIG_WINDOW_HOT=0 turns it off): region streams of any thread and "
		            "64 KiB blocks rewritten every frame open bigger windows\n", is ? "on" : "off");
		std::fflush(stdout);
		return is;
	}();
	return on;
}
struct HotRegionSlot {
	std::atomic<uint64_t> key {0}, end {0}, size {0}, tsc {0};
};
constexpr size_t HotRegionSlots = 2048;
HotRegionSlot    g_hot_regions[HotRegionSlots];
constexpr size_t HotBlockSlots = 16384;
std::atomic<uint64_t> g_hot_blocks[HotBlockSlots] {}; // (block index << 24 | tsc >> 26: a stamp of ~20 ms at 3.2 GHz)
std::atomic<uint64_t> g_hot_stats[5] {};             // stream windows, stream pages, hot windows, hot pages, small
HotRegionSlot& HotRegion(uint64_t region) {
	return g_hot_regions[static_cast<size_t>((region * 0x9E3779B97F4A7C15ull) >> 53u) % HotRegionSlots];
}
void HotNote(uint64_t begin, uint64_t size, uint64_t now) {
	const uint64_t region = begin / TRACKER_REGION_SIZE;
	auto&          slot   = HotRegion(region);
	slot.key.store(region + 1, std::memory_order_relaxed);
	slot.end.store(begin + size, std::memory_order_relaxed);
	slot.size.store(size, std::memory_order_relaxed);
	slot.tsc.store(now, std::memory_order_relaxed);
	const uint64_t block = begin >> 16u;
	g_hot_blocks[static_cast<size_t>((block * 0x9E3779B97F4A7C15ull) >> 50u) % HotBlockSlots].store(
	    (block << 24u) | ((now >> 26u) & 0xffffffu), std::memory_order_relaxed);
}
bool HotBlock(uint64_t fault, uint64_t now) {
	const uint64_t block = fault >> 16u;
	const uint64_t v     = g_hot_blocks[static_cast<size_t>((block * 0x9E3779B97F4A7C15ull) >> 50u) % HotBlockSlots].load(
	    std::memory_order_relaxed);
	if ((v >> 24u) != (block & ((uint64_t {1} << 40u) - 1))) return false;
	const uint64_t then  = v & 0xffffffu;
	const uint64_t stamp = (now >> 26u) & 0xffffffu;
	return ((stamp - then) & 0xffffffu) <= 12; // (~250 ms at 3.2 GHz)
}
void HotReport() {
	static std::atomic<uint64_t> last {0};
	const uint64_t               now = __rdtsc();
	uint64_t                     l   = last.load(std::memory_order_relaxed);
	if (l == 0) {
		last.compare_exchange_strong(l, now);
		return;
	}
	if (now - l < 16'000'000'000ull || !last.compare_exchange_strong(l, now)) return; // (~5 s)
	const char* e = std::getenv("KYTY_ESPERAS");
	uint64_t    v[5];
	for (int i = 0; i < 5; i++) v[i] = g_hot_stats[i].exchange(0, std::memory_order_relaxed);
	if (e == nullptr || e[0] != '1') return;
	std::printf("[esperas] ventanas calientes 5 s: %llu de flujo de region (%llu paginas), %llu de bloque caliente de 64 KiB "
	            "(%llu paginas), %llu normales en la ventana grande\n",
	            static_cast<unsigned long long>(v[0]), static_cast<unsigned long long>(v[1]), static_cast<unsigned long long>(v[2]),
	            static_cast<unsigned long long>(v[3]), static_cast<unsigned long long>(v[4]));
	std::fflush(stdout);
}
} // namespace

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault) {
	// A small window amortizes faults from sequential CPU writes. Expanding a
	// fault is allowed only for mapped, unaliased pages with no GPU/image owner.
	constexpr uint64_t window_size  = 4 * TRACKER_PAGE_SIZE;
	constexpr uint64_t stream_size  = 16 * TRACKER_PAGE_SIZE;
	constexpr uint64_t big          = uint64_t {1} << 20u;
	const auto         window_begin = fault & ~(window_size - 1);
	// KYTY_BIG_WINDOW=1: per-frame memory the game rewrites whole (DefaultTagg, render-context DCBs, found by
	// DemonsSoulsMemMap) unprotects 1 MiB at once: still tracked (dirty bits exact for what is unprotected), with
	// ~64x fewer faults. Falls back to the normal window when the 1 MiB has GPU/image owners.
	const bool in_big = ::Loader::DemonsSoulsMemMap::InBigWindowGranule(fault);
	const bool grow   = in_big && BigWindowGrow();
	if (in_big && !grow) {
		if (TryInvalidateCpuWriteWindow(fault, fault & ~(big - 1), big)) {
			LiveCensus::g_big_window[0].fetch_add(1, std::memory_order_relaxed);
			return true;
		}
	}
	// (esperas4, KYTY_BIG_WINDOW_HOT) See BigWindowHot: region streams and hot 64 KiB blocks in the big-window granules.
	if (grow && BigWindowHot()) {
		HotReport();
		const uint64_t now        = __rdtsc();
		const uint64_t region     = window_begin / TRACKER_REGION_SIZE;
		const uint64_t region_end = (window_begin | (TRACKER_REGION_SIZE - 1)) + 1;
		auto&          slot       = HotRegion(region);
		if (slot.key.load(std::memory_order_relaxed) == region + 1 && slot.end.load(std::memory_order_relaxed) == window_begin &&
		    now - slot.tsc.load(std::memory_order_relaxed) < 64'000'000) {
			const uint64_t last_size = std::max<uint64_t>(slot.size.load(std::memory_order_relaxed), stream_size / 2);
			const uint64_t wanted    = std::min<uint64_t>(std::min<uint64_t>(last_size * 2, big), region_end - window_begin);
			if (wanted > window_size && TryInvalidateCpuWriteWindow(fault, window_begin, wanted)) {
				HotNote(window_begin, wanted, now);
				g_hot_stats[0].fetch_add(1, std::memory_order_relaxed);
				g_hot_stats[1].fetch_add(wanted / TRACKER_PAGE_SIZE, std::memory_order_relaxed);
				return true;
			}
		}
		const uint64_t block_begin = fault & ~(stream_size - 1);
		if (HotBlock(fault, now) && TryInvalidateCpuWriteWindow(fault, block_begin, stream_size)) {
			HotNote(block_begin, stream_size, now);
			g_hot_stats[2].fetch_add(1, std::memory_order_relaxed);
			g_hot_stats[3].fetch_add(stream_size / TRACKER_PAGE_SIZE, std::memory_order_relaxed);
			return true;
		}
		if (TryInvalidateCpuWriteWindow(fault, window_begin, window_size)) {
			HotNote(window_begin, window_size, now);
			g_hot_stats[4].fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		return false;
	}
	if (kyty_local_window_stream_mode.load(std::memory_order_relaxed) != 0) {
		// The window after the one this thread's last window fault ended in, within 2 ms (~6.4M TSC ticks at 3.2 GHz):
		// a stream, whose next 64 KiB goes at once (the next fault in it is the window after that).
		thread_local uint64_t last_end = 0, last_tsc = 0, last_size = 0;
		const uint64_t        now      = __rdtsc();
		// (render3, KYTY_BIG_WINDOW_GROW) A stream that already had its 64 KiB (or more) and goes on: twice the last
		// window, up to 1 MiB, to the end of its tracker region at most (a window lies in one region).
		if (grow && last_size >= stream_size && window_begin == last_end && now - last_tsc < 64'000'000) {
			const uint64_t region_end = (window_begin | (TRACKER_REGION_SIZE - 1)) + 1;
			const uint64_t wanted     = std::min<uint64_t>(std::min<uint64_t>(last_size * 2, big), region_end - window_begin);
			if (wanted > stream_size && TryInvalidateCpuWriteWindow(fault, window_begin, wanted)) {
				last_end  = window_begin + wanted;
				last_size = wanted;
				last_tsc  = now;
				LiveCensus::g_big_window[1].fetch_add(1, std::memory_order_relaxed);
				LiveCensus::g_big_window[2].fetch_add(wanted / TRACKER_PAGE_SIZE, std::memory_order_relaxed);
				return true;
			}
		}
		const bool            stream   = window_begin == last_end && now - last_tsc < 6'400'000;
		if (stream && TryInvalidateCpuWriteWindow(fault, window_begin, stream_size)) {
			last_end  = window_begin + stream_size;
			last_size = stream_size;
			last_tsc  = now;
			return true;
		}
		if (!TryInvalidateCpuWriteWindow(fault, window_begin, window_size)) return false;
		last_end  = window_begin + window_size;
		last_size = window_size;
		last_tsc  = now;
		return true;
	}
	return TryInvalidateCpuWriteWindow(fault, window_begin, window_size);
}

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t window_begin, uint64_t window_size) {
	GuestRange selected {};
	uint64_t   mapping_epoch;
	{
		std::shared_lock mapped_lock(m_mapped_ranges_mutex);
		mapping_epoch = m_mapping_epoch;
		m_mapped_ranges.ForEachIntersection(window_begin, window_size, [&](RangeSet::Range range) {
			if (fault >= range.address && fault - range.address < range.size) {
				const auto begin =
				    (range.address + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
				const auto end = (range.address + range.size) & ~(TRACKER_PAGE_SIZE - 1);
				if (begin < end) selected = {begin, end - begin};
			}
		});
	}
	if (selected.size <= TRACKER_PAGE_SIZE || fault < selected.address ||
	    fault - selected.address >= selected.size ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(selected.address, selected.size))
		return false;
	// Kernel backing queries must not run while holding the resource-map lock.
	std::shared_lock mapped_lock(m_mapped_ranges_mutex);
	if (mapping_epoch != m_mapping_epoch ||
	    !m_mapped_ranges.Contains(selected.address, selected.size))
		return false;
	return m_buffer_cache.TryInvalidateCpuWriteWindow(fault, selected.address, selected.size);
}

// KYTY_FAULT_STATS=1 (diagnostic): every 5 s, the 2 MiB guest regions that took the most CPU-tracking faults
// (write-window, write, read), so the memory the game keeps rewriting can be told apart. A fixed open-addressed table.
namespace {
struct FaultSlot {
	std::atomic<uint64_t> region {0};
	std::atomic<uint32_t> window {0}, write {0}, read {0};
};
constexpr size_t           FaultSlots = 4096;
FaultSlot                  g_fault_slots[FaultSlots];
std::atomic<int64_t>       g_fault_report_ms {0};
bool FaultStatsOn() {
	static const bool on = [] { const char* v = std::getenv("KYTY_FAULT_STATS"); return v != nullptr && v[0] == '1'; }();
	return on;
}
void NoteFault(uint64_t vaddr, int kind) {
	const uint64_t region = (vaddr >> 21u) + 1u;
	size_t         i      = static_cast<size_t>((region * 0x9E3779B97F4A7C15ull) >> 52u) % FaultSlots;
	for (size_t probe = 0; probe < 64; ++probe, i = (i + 1) % FaultSlots) {
		uint64_t cur = g_fault_slots[i].region.load(std::memory_order_relaxed);
		if (cur == 0 && g_fault_slots[i].region.compare_exchange_strong(cur, region)) cur = region;
		if (cur != region) continue;
		(kind == 0 ? g_fault_slots[i].window : kind == 1 ? g_fault_slots[i].write : g_fault_slots[i].read)
		    .fetch_add(1, std::memory_order_relaxed);
		break;
	}
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch()).count();
	auto last = g_fault_report_ms.load(std::memory_order_relaxed);
	if (now - last < 5000 || !g_fault_report_ms.compare_exchange_strong(last, now)) return;
	struct Row { uint64_t region; uint32_t window, write, read; };
	std::vector<Row> rows;
	for (auto& slot: g_fault_slots) {
		const uint64_t r = slot.region.load(std::memory_order_relaxed);
		if (r == 0) continue;
		Row row {r - 1u, slot.window.exchange(0), slot.write.exchange(0), slot.read.exchange(0)};
		if (row.window + row.write + row.read != 0) rows.push_back(row);
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
		return a.window + a.write + a.read > b.window + b.write + b.read; });
	uint64_t total = 0;
	for (const auto& r: rows) total += r.window + r.write + r.read;
	std::printf("[faults] 5 s: %llu faults in %zu regions of 2 MiB; top:\n", static_cast<unsigned long long>(total), rows.size());
	for (size_t k = 0; k < rows.size() && k < 12; ++k)
	{
		// (Which of it the game streams textures into: DemonsSoulsTextureProbe's live ranges; ? = probe busy or off.)
		const uint64_t base = rows[k].region << 21u;
		const uint64_t tex  = ::Loader::DemonsSoulsTextureProbe::LiveOverlap(base, base + (uint64_t {1} << 21u));
		char           what[48];
		if (tex == UINT64_MAX) std::snprintf(what, sizeof(what), "textura ?");
		else std::snprintf(what, sizeof(what), "textura %llu%%", static_cast<unsigned long long>(tex * 100 / (uint64_t {1} << 21u)));
		// (And which of it is CBuffer memory created with initial data: the buffer probe; ? = busy or off.)
		const uint64_t buf = ::Loader::DemonsSoulsTextureProbe::BufferOverlap(base, base + (uint64_t {1} << 21u));
		const size_t   used = std::strlen(what);
		if (buf == UINT64_MAX) std::snprintf(what + used, sizeof(what) - used, "  buffer ?");
		else std::snprintf(what + used, sizeof(what) - used, "  buffer %llu%%", static_cast<unsigned long long>(buf * 100 / (uint64_t {1} << 21u)));
		// (And what else the game keeps there: submitted DCBs, render contexts, engine heaps, pool, rings; KYTY_MEMMAP.)
		char           map[192];
		::Loader::DemonsSoulsMemMap::Tags(base, base + (uint64_t {1} << 21u), map, sizeof(map));
		std::printf("[faults]   0x%010llx  window %u  write %u  read %u  %s%s%s\n", static_cast<unsigned long long>(base),
		            rows[k].window, rows[k].write, rows[k].read, what, map[0] != 0 ? "  " : "", map);
	}
	std::fflush(stdout);
}
} // namespace

// (raiz8) PageManager's view of the GPU thread (pageManager.h).
bool PageManagerOnGpuThread() noexcept {
	return GuestGpu::IsGpuThread();
}

bool GpuResourceManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	const Techo::CohScope techo_scope(Techo::CFault, LiveCensus::g_render); // (techo, KYTY_TECHO)
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// (raiz8, KYTY_RAIZ_ASYNC_RELEASE) A page whose read release the GPU thread left to others: applied now (its state's
	// protection, the kept bytes in) and the access retried. A page watched again by then faults again, on the paths
	// below. First: the paths below read the page's state, which already says released.
	if (m_page_manager.ApplyDeferredReleaseAt(fault_vaddr)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		// A never-written granule's first write (fresh memory): unwatched, done unless another watcher holds the page.
		if (FreshFirstWrite(fault_vaddr)) {
			uint64_t first = 0, end = 0;
			if (!m_page_manager.WatchedSpan(fault_vaddr & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE, &first, &end)) {
				// (audit1, KYTY_DS_WRITE_AUDIT: measurement only, here and below.)
				if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteWriteFault(fault_vaddr, 2);
				return true;
			}
		}
		// (policy1) KYTY_DS_NEIGHBOR_SPLIT (off by default: one relaxed load): a write to bytes no hook of the game
		// announces, in a page whose only GPU ranges (if any) are ones the hooks do announce: the page stops being
		// write-protected for the buffers (MemoryTracker::SetNoWatchPage; it is CPU-dirty once more and then left
		// alone). When another watcher still holds the page (an image, fresh memory) the write faults again and takes
		// the normal path below, as it does when the policy does not want the page.
		if (::Loader::DemonsSoulsWriteAudit::SplitOn() && ::Loader::DemonsSoulsWriteAudit::WantsNoWatch(fault_vaddr)) {
			int  why  = 0;
			bool done = m_buffer_cache.PolicySetNoWatch(fault_vaddr, &why);
			// (anuncio1) KYTY_RAIZ_ANNOUNCE: a frame page refused for good (a GPU write, a consumer) that a frame allocator
			// has handed out again since: its refusal was the old owner's. Lifted, and no-watch now.
			if (!done && why == 3 && ::Loader::DemonsSoulsWriteAudit::UnblockWanted(fault_vaddr)) {
				done = m_buffer_cache.PolicyUnblockSetNoWatch(fault_vaddr);
				::Loader::DemonsSoulsWriteAudit::NoteUnblocked(fault_vaddr, done);
			}
			if (!done) ::Loader::DemonsSoulsWriteAudit::NoteNoWatchRefused(fault_vaddr, why); // (anuncio1: why, by reason)
			::Loader::DemonsSoulsWriteAudit::NoteNoWatch(fault_vaddr, done);
			if (done) {
				if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteWriteFault(fault_vaddr, 1);
				LiveCounters::Add(LiveCounters::WriteFault);
				if (FaultStatsOn()) NoteFault(fault_vaddr, 1);
				return true;
			}
		}
		// (A write to a watched page of a view a partial unmap is mapping again: unseen while such a view came back
		// writable.)
		const bool in_remap = LibKernel::Memory::InRemapWindow(fault_vaddr);
		if (TryInvalidateCpuWriteWindow(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::WindowFault);
			if (FaultStatsOn()) NoteFault(fault_vaddr, 0);
			if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteWriteFault(fault_vaddr, 0);
			if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
			return true;
		}
		LiveCounters::Add(LiveCounters::WriteFault);
		if (FaultStatsOn()) NoteFault(fault_vaddr, 1);
		if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteWriteFault(fault_vaddr, 1);
		// No watcher holds the page, so no cache expects this fault: the protection outlived its
		// watchers and would fault forever. It gets the guest's own protection back.
		if (m_page_manager.RestoreIfUnwatched(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::StaleProtectRepairs);
			return true;
		}
		if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		LiveCounters::Add(LiveCounters::ReadFault);
		Raiz::NoteGameReadFault(fault_vaddr); // (raiz2: the windows' exposure estimate)
		if (FaultStatsOn()) NoteFault(fault_vaddr, 2);
		// (policy1, KYTY_DS_WRITE_AUDIT: the "[read-audit]" lines. Measurement only.)
		if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteReadFault(fault_vaddr);
		if (LiveCensus::g_render) LiveCounters::Add(LiveCounters::RenderReadFaults);
		// A speculative translation's thread read GPU-written memory: the readback is the normal path's (the
		// scheduler's own command buffers and ticks), as the translation in order would make it there.
		struct NormalPath {
			Spec::State*                state    = std::exchange(Spec::t_state, nullptr);
			CommandScheduler::Recorder* recorder = CommandScheduler::ThreadRecorder();
			NormalPath() { CommandScheduler::SetThreadRecorder(nullptr); }
			~NormalPath() {
				Spec::t_state = state;
				CommandScheduler::SetThreadRecorder(recorder);
			}
		} normal_path;
		SyncStats::g_fault_address = fault_vaddr;
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

// (policy2) KYTY_DS_READBACK_HOOK: the game is about to read a whole readback buffer (its accessor was entered). The
// same readback its read faults make (HandleFault above, with the same normal-path guard), asked once for the range:
// the pages become readable together, and the reads that follow do not fault.
bool GpuResourceManager::ReadbackRange(uint64_t vaddr, uint64_t size) noexcept {
	if (!IsMapped(vaddr, size)) return false;
	struct NormalPath {
		Spec::State*                state    = std::exchange(Spec::t_state, nullptr);
		CommandScheduler::Recorder* recorder = CommandScheduler::ThreadRecorder();
		NormalPath() { CommandScheduler::SetThreadRecorder(nullptr); }
		~NormalPath() {
			Spec::t_state = state;
			CommandScheduler::SetThreadRecorder(recorder);
		}
	} normal_path;
	// (policy4) Only when a page of the range itself is GPU-written: what a read fault there means. (The caller's
	// hint, PageManager::HasReadWatchers, is per 64 KiB: without this, a buffer with no GPU-written page of its own
	// had the readback's whole 512 KiB window read back early because of a neighbour's pages. A fault never does that.)
	if (!m_buffer_cache.IsRegionGpuModified(vaddr, size)) return false;
	SyncStats::g_fault_address = vaddr;
	m_buffer_cache.ReadMemory(vaddr, size);
	return true;
}

// (parpadeo7) The chunk bits an XPR label hands over, read back from the label on (local/parpadeo7.h). The same guard as
// ReadbackRange (the normal path's command buffers and ticks), and the same test a read fault's readback makes: a range
// without a GPU-written page has nothing to read back.
void GpuResourceManager::PrefetchGuestReads(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, int64_t label_ns) noexcept {
	const auto start = std::chrono::steady_clock::now();
	try {
		struct NormalPath {
			Spec::State*                state    = std::exchange(Spec::t_state, nullptr);
			CommandScheduler::Recorder* recorder = CommandScheduler::ThreadRecorder();
			NormalPath() { CommandScheduler::SetThreadRecorder(nullptr); }
			~NormalPath() {
				Spec::t_state = state;
				CommandScheduler::SetThreadRecorder(recorder);
			}
		} normal_path;
		auto& c = Parpadeo7::g_c;
		for (const auto& [vaddr, size]: ranges) {
			if (size == 0 || !IsMapped(vaddr, size) || !m_page_manager.HasReadWatchers(vaddr, size) ||
			    !m_buffer_cache.IsRegionGpuModified(vaddr, size)) {
				c.not_gpu.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			switch (m_buffer_cache.PrefetchGuestRead(vaddr, size, label_ns)) {
				case Parpadeo7::PrefetchBegun: c.begun.fetch_add(1, std::memory_order_relaxed); break;
				case Parpadeo7::PrefetchInflight: c.inflight.fetch_add(1, std::memory_order_relaxed); break;
				case Parpadeo7::PrefetchNoSlot: c.no_slot.fetch_add(1, std::memory_order_relaxed); break;
				case Parpadeo7::PrefetchNothing: c.not_gpu.fetch_add(1, std::memory_order_relaxed); break;
				default: c.refused.fetch_add(1, std::memory_order_relaxed); break;
			}
		}
	} catch (...) {
	}
	Parpadeo7::g_c.gpu_us.fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()),
	    std::memory_order_relaxed);
}

// KYTY_NATIVE_MEMORY's cost, every 5 s (GPU thread only: SweepCpuWrites and NativeFlushGpuWrites): the sweeps of the
// write watch, their time and the written watched pages they invalidated; the GPU-to-guest flushes, the time they
// waited (the GPU, the downloads, their write-backs) and what they downloaded.
namespace {
struct NativeStats {
	uint64_t sweeps = 0, sweep_us = 0, sweep_max_us = 0, pages = 0;
	// (ww2) Hand-overs that needed no sweep (KYTY_WRITE_WATCH_MODE=poll): submissions a sweep had covered, WAIT_REG_MEMs
	// satisfied by a value the command processors wrote; the images' pages of TakeWritesBeforeGpuWrite invalidated; flips.
	uint64_t covered_submissions = 0, processor_waits = 0, image_pages = 0, flips = 0, timed_sweeps = 0;
	uint64_t flushes = 0, flush_us = 0, flush_max_us = 0;
	uint64_t buffer_ranges = 0, buffer_bytes = 0, images = 0, image_bytes = 0, refused = 0;
	std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
};
NativeStats g_native_stats;

void ReportNativeStats() {
	auto&      st  = g_native_stats;
	const auto now = std::chrono::steady_clock::now();
	if (now - st.report < std::chrono::seconds(5)) return;
	constexpr double MiB = 1024.0 * 1024.0;
	if (PageManager::NativeReadFaults()) {
		// (nativemem1, KYTY_WRITE_WATCH) Protection changes per second (all of guest memory's: the trackers' no-access
		// ones and their releases, the guest's mprotects; the emulator's opened copies apart) against the write watch's
		// work: its sweeps and the written pages they found that a write watcher held.
		// (ww2) And what the tracking of the game's writes cost a frame (a flip): the sweeps' time, the GetWriteWatch
		// calls of everything (sweeps, the reads before GPU work and after a no-access protection, the emulator's own
		// writes), the hand-overs that needed no sweep, and the written pages left out.
		static uint64_t last_protects = 0, last_copy_protects = 0, last_copies = 0, last_calls = 0, last_retries = 0;
		uint64_t        protects = 0, copy_protects = 0, copies = 0, full = 0, partial = 0, regions = 0, queried = 0;
		uint64_t        calls = 0, retries = 0;
		LibKernel::Memory::NativeProtectCounters(&protects, &copy_protects, &copies);
		LibKernel::Memory::NativeWriteWatchCounters(&calls, &retries);
		PageManager::TakeSweepStats(&full, &partial, &regions, &queried);
		PageManager::WatchStats watch;
		PageManager::TakeWatchStats(&watch);
		const double secs  = std::chrono::duration<double>(now - st.report).count();
		const double per_s = secs > 0.0 ? 1.0 / secs : 0.0;
		const auto   sweeps = std::max<uint64_t>(1, st.sweeps);
		const double frames = static_cast<double>(std::max<uint64_t>(1, st.flips));
		std::printf("Write watch (%.1f s): protect calls %.0f/s (+ %.0f/s opening %.0f copies/s over protected pages); "
		            "write-watch pages %.0f/s (%.1f MiB/s) seen written and invalidated; %llu sweeps (%llu full, %llu "
		            "partial) %.1f ms (%.3f ms each, max %.3f), %.1f regions / %.0f MiB read each\n",
		            secs, static_cast<double>(protects - last_protects) * per_s,
		            static_cast<double>(copy_protects - last_copy_protects) * per_s,
		            static_cast<double>(copies - last_copies) * per_s, static_cast<double>(st.pages) * per_s,
		            static_cast<double>(st.pages) * 4096.0 / MiB * per_s, static_cast<unsigned long long>(st.sweeps),
		            static_cast<unsigned long long>(full), static_cast<unsigned long long>(partial),
		            static_cast<double>(st.sweep_us) / 1000.0, static_cast<double>(st.sweep_us) / 1000.0 / static_cast<double>(sweeps),
		            static_cast<double>(st.sweep_max_us) / 1000.0, static_cast<double>(regions) / static_cast<double>(sweeps),
		            static_cast<double>(queried) / MiB / static_cast<double>(sweeps));
		std::printf("Write watch+ (%.1f s, mode %s, %llu frames): per frame %.3f ms reading the write watch (%.3f in %.1f "
		            "sweeps with their invalidations + %.3f around GPU writes), %.0f GetWriteWatch calls (%llu retried by "
		            "allocation); no sweep needed: %llu submissions already covered, %llu waits on labels "
		            "the GPU side wrote; read before GPU writes: %llu pages in %llu ranges (+ %llu pages of images), after a "
		            "no-access protection: %llu pages in %llu; written pages lost: %llu (no-access when swept); region list "
		            "built %llu times; %llu sweeps by KYTY_WRITE_WATCH_MAX_MS\n",
		            secs, PageManager::WatchProtects() ? "protect" : PageManager::WatchPolled() ? "poll" : "sweep",
		            static_cast<unsigned long long>(st.flips),
		            static_cast<double>(st.sweep_us + watch.capture_us) / 1000.0 / frames,
		            static_cast<double>(st.sweep_us) / 1000.0 / frames, static_cast<double>(st.sweeps) / frames,
		            static_cast<double>(watch.capture_us) / 1000.0 / frames, static_cast<double>(calls - last_calls) / frames,
		            static_cast<unsigned long long>(retries - last_retries), static_cast<unsigned long long>(st.covered_submissions),
		            static_cast<unsigned long long>(st.processor_waits), static_cast<unsigned long long>(watch.before_gpu_pages),
		            static_cast<unsigned long long>(watch.before_gpu_calls), static_cast<unsigned long long>(st.image_pages),
		            static_cast<unsigned long long>(watch.after_protect_pages),
		            static_cast<unsigned long long>(watch.after_protect_calls),
		            static_cast<unsigned long long>(watch.noaccess_pages), static_cast<unsigned long long>(watch.list_rebuilds),
		            static_cast<unsigned long long>(st.timed_sweeps));
		std::fflush(stdout);
		last_protects      = protects;
		last_copy_protects = copy_protects;
		last_copies        = copies;
		last_calls         = calls;
		last_retries       = retries;
		st                 = NativeStats {};
		st.report          = now;
		return;
	}
	std::printf("Native memory (%.1f s): %llu sweeps, %.1f ms (max %.3f ms), %llu written watched pages invalidated; "
	            "%llu GPU->guest flushes, %.1f ms waiting (max %.3f ms), forced downloads: %llu buffer ranges %.1f MiB, "
	            "%llu images %.1f MiB, %llu images refused\n",
	            std::chrono::duration<double>(now - st.report).count(), static_cast<unsigned long long>(st.sweeps),
	            static_cast<double>(st.sweep_us) / 1000.0, static_cast<double>(st.sweep_max_us) / 1000.0,
	            static_cast<unsigned long long>(st.pages), static_cast<unsigned long long>(st.flushes),
	            static_cast<double>(st.flush_us) / 1000.0, static_cast<double>(st.flush_max_us) / 1000.0,
	            static_cast<unsigned long long>(st.buffer_ranges), static_cast<double>(st.buffer_bytes) / MiB,
	            static_cast<unsigned long long>(st.images), static_cast<double>(st.image_bytes) / MiB,
	            static_cast<unsigned long long>(st.refused));
	std::fflush(stdout);
	st        = NativeStats {};
	st.report = now;
}

// (ww2, KYTY_WRITE_WATCH_MODE=poll) KYTY_WRITE_WATCH_MAX_MS=N (default 0: off): a hand-over that needs no sweep sweeps
// all the same when the last sweep is N ms old. A net for a game write that reaches GPU work with neither a submission
// nor a label of the game's after it (a race on the console; the sweeps at every WAIT_REG_MEM used to hide it).
// Read at first use (the launcher sets the variables after the exe started).
std::chrono::microseconds WatchMaxGap() {
	static const std::chrono::microseconds gap = [] {
		const char* v = std::getenv("KYTY_WRITE_WATCH_MAX_MS");
		return std::chrono::microseconds(v != nullptr && v[0] >= '0' && v[0] <= '9' ? std::strtoull(v, nullptr, 10) * 1000u : 0u);
	}();
	return gap;
}
std::chrono::steady_clock::time_point g_last_sweep = std::chrono::steady_clock::now();

// KYTY_NATIVE_IMAGE_DOWNLOAD=0: NativeFlushGpuWrites leaves the images out (buffers only). Default on.
bool NativeImageDownloadOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_IMAGE_DOWNLOAD");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}
} // namespace

// KYTY_NATIVE_MEMORY=1: no page is ever protected, so no write faults; the write watch, read here at the points where
// the guest's writes become the GPU's to use, stands for them.
//
// Why these points are enough (the argument of BdaDirtyRegions::Publish): the guest hands memory to GPU work only
// through a later submission, or through a label a WAIT_REG_MEM of that work waits for; a write after both, with no
// label, races the GPU on the console too. Each new submission starts with a sweep (GuestGpu::Process, its first slice:
// the guest wrote its command buffers, descriptors and data before submitting, and the submission queue's lock orders
// those writes before this thread takes it), every satisfied WAIT_REG_MEM is followed by one
// (CommandProcessor::WaitRegMem) before the packets after it, and each GPU flip has one. A slice resumed at a wait
// sweeps only when the wait is satisfied (the guest handed nothing new over meanwhile otherwise). A write made later is
// seen by the next sweep; the bytes the GPU thread uploads meanwhile may include it (an upload is a copy of what memory
// holds then), and the next sweep invalidates the page again (an upload more, never a stale one).
void GpuResourceManager::SyncProtectPause() {
	if (!GuestGpu::IsGpuThread() || Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr) return;
	thread_local std::vector<PageManager::DeferredRange> reprotect;
	thread_local std::vector<PageManager::DeferredRange> resync;
	thread_local std::vector<GuestRange>                 pieces;
	thread_local std::vector<PageManager::WatchedRun>    runs;
	reprotect.clear();
	resync.clear();
	bool ending = false;
	if (!PageManager::TakeProtectPause(&reprotect, &resync, &ending)) return;
	const auto mapped = [this](const std::vector<PageManager::DeferredRange>& in, std::vector<GuestRange>& out) {
		out.clear();
		std::shared_lock lock(m_mapped_ranges_mutex);
		for (const auto& run: in)
			m_mapped_ranges.ForEachIntersection(run.address, run.size, [&](RangeSet::Range range) {
				out.push_back({range.address, range.size});
			});
	};
	// Ending: the skipped write protections applied first (a write after this faults as usual), then every page that
	// went unprotected is invalidated as written (a write made meanwhile is not lost).
	if (ending) {
		mapped(reprotect, pieces);
		for (const auto& piece: pieces) m_page_manager.ReapplyProtection(piece.address, piece.size);
	}
	mapped(resync, pieces);
	uint64_t pages = 0;
	for (const auto& piece: pieces) {
		pages += (piece.size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
		InvalidateCpuWrittenPages(piece.address, piece.size);
	}
	// Still paused: the pages some watcher still write-watches stay unprotected, so they stay in the set.
	if (!ending) {
		for (const auto& piece: pieces) {
			runs.clear();
			m_page_manager.WatchedRuns(piece.address, piece.size, &runs);
			for (const auto& run: runs)
				if (run.mode == Common::VirtualMemory::Mode::Read) PageManager::KeepProtectPauseResync(run.address, run.size);
		}
	}
	PageManager::EndProtectPause(ending, pages);
}

void GpuResourceManager::NoteFlipForWatchStats() {
	if (!PageManager::NativeReadFaults() || !GuestGpu::IsGpuThread()) return;
	g_native_stats.flips++;
	// (KYTY_WRITE_WATCH_MODE=protect: no sweep comes to print the line.)
	if (PageManager::WatchProtects()) ReportNativeStats();
	// (raiz1, KYTY_RAIZ: the "[raiz]" line, every 5 s.)
	if (Raiz::On()) {
		static uint64_t raiz_flips = 0; // (raiz2: the line's per-frame figures; this thread only)
		++raiz_flips;
		uint64_t protects = 0;
		LibKernel::Memory::NativeProtectCounters(&protects, nullptr, nullptr);
		Raiz::Report(LiveCounters::Value(LiveCounters::WriteFault), LiveCounters::Value(LiveCounters::WindowFault),
		             LiveCounters::Value(LiveCounters::ReadFault), protects, raiz_flips,
		             LiveCounters::Value(LiveCounters::AddressSpaceBusyWaits));
	}
}

namespace {
std::atomic<uint64_t> g_native_sweep_stamp {0}; // (nativo3)
std::atomic<uint64_t> g_native_marks_skipped {0};
std::atomic<uint64_t> g_native_sweeps_covered {0};
std::atomic<uint64_t> g_native_sweeps_label {0}; // (nativo4)
std::atomic<uint64_t> g_native_sweeps_done {0};
std::atomic<uint64_t> g_native4_last_ms {0};
} // namespace

uint64_t GpuResourceManager::NativeSweepStamp() noexcept {
	return g_native_sweep_stamp.load();
}
bool GpuResourceManager::NativeAllOn() noexcept {
	return BufferCache::NativeAllOn();
}
bool GpuResourceManager::NativeSweptSince(uint64_t stamp) noexcept {
	return BufferCache::NativeAllOn() && g_native_sweep_stamp.load() > stamp;
}

void GpuResourceManager::SweepCpuWrites(SweepReason reason) {
	// KYTY_VOLATILE=1: the volatile regions' pages dirty again before this submission (or the work after a satisfied
	// WAIT_REG_MEM) reads guest memory (MemoryTracker::VolatileRedirty; one relaxed load when off). Every submission
	// path (AgcDriverSubmitDcb/MultiDcbs/CommandBuffer/MultiCommandBuffers/Acb/MultiAcbs) comes here at its first slice.
	if (GuestGpu::IsGpuThread() && Spec::Current() == nullptr && CommandScheduler::ThreadRecorder() == nullptr) {
		MemoryTracker::VolatileRedirty();
		MemoryTracker::HotRedirty(); // (PROT-MENOS) KYTY_PROT_MENOS: hot pages dirty again before this submission
		// (policy7) This is the GPU thread: its write faults are the emulated GPU's own (the frame audit). A thread_local store.
		if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteGpuThread();
		// (policy1) KYTY_DS_NEIGHBOR_SPLIT: what the game's hooks announced since the last call, applied to the no-watch
		// pages before this submission's work (or the work after a satisfied WAIT_REG_MEM) reads guest memory: the game
		// creates, wraps or copies its data before it emits the draw that uses it, and the submission comes after.
		if (::Loader::DemonsSoulsWriteAudit::SplitOn()) {
			::Loader::DemonsSoulsWriteAudit::Request requests[128];
			bool     everything = false;
			uint64_t dirty = 0, reverted = 0;
			for (size_t count = 0; (count = ::Loader::DemonsSoulsWriteAudit::TakeRequests(requests, 128, &everything)) != 0;) {
				for (size_t i = 0; i < count; ++i) {
					// (nativo2) KYTY_DS_NATIVE_HEAP: a range no hook covers appeared there - its writes are the OS write
					// watch's to tell now: native pages (CPU-dirty if new), not watched again for good.
					if (requests[i].revert)
						reverted += BufferCache::NativeHeapOn() ? m_buffer_cache.PolicyKeepNative(requests[i].address, requests[i].size)
						                                        : m_buffer_cache.PolicyRevert(requests[i].address, requests[i].size);
					else dirty += m_buffer_cache.PolicyMarkDirty(requests[i].address, requests[i].size);
				}
				if (count < 128) break;
			}
			// (policy6) KYTY_DS_WRITE_FRAME's marks (the lot submits, the command buffer submits, the emulator's copies made
			// since the last call): their no-watch pages are taken as written. Its overflow: all of them are (never stale).
			if (::Loader::DemonsSoulsWriteAudit::FrameOn()) {
				bool     all_frame = false;
				uint64_t marked    = 0;
				// (nativo3, KYTY_DS_NATIVE_ALL: drained, not applied - the write watch sweep below sees every write to those
				// no-watch pages; the marks were ~500,000 ranges a second on this thread, 093444.)
				const bool skip_marks = BufferCache::NativeAllOn();
				for (size_t count = 0; (count = ::Loader::DemonsSoulsWriteAudit::TakeFrameMarks(requests, 128, &all_frame)) != 0;) {
					if (skip_marks) g_native_marks_skipped.fetch_add(count, std::memory_order_relaxed);
					else
						for (size_t i = 0; i < count; ++i) marked += m_buffer_cache.PolicyMarkDirty(requests[i].address, requests[i].size);
					if (count < 128) break;
				}
				if (all_frame) marked += m_buffer_cache.PolicyMarkAllDirty();
				if (marked != 0) ::Loader::DemonsSoulsWriteAudit::NoteFrameDrained(marked);
			}
			// (The hooks' queue overflowed: what was lost may be an announcement or a request to watch a range again, so
			// every no-watch page is watched again, CPU-dirty. policy4; policy1 only marked them written.)
			if (everything) reverted += m_buffer_cache.PolicyRevertAll();
			// (nativo2) KYTY_DS_NATIVE_HEAP: the native pages the game wrote since the last sweep (the OS write watch,
			// read and reset) CPU-dirty before this submission's work reads them. One cached load when off.
			// (nativo3, KYTY_DS_NATIVE_ALL) A submission queued before a native sweep began (NativeSweptSince): that sweep
			// read every write the game made before queueing it - no other here (exact; the GuestGpu queue is a few deep).
			// (nativo4) A WAIT_REG_MEM satisfied by a label the command processor wrote itself (ProcessorWait: the game
			// handed nothing over through it, as ww2's poll mode) sweeps no more: 068773 sweeps in 5 s (100237) were
			// mostly those, ~500 ms a frame on this thread. A covered submission neither (nativo3).
			if (BufferCache::NativeAllOn() && reason == SweepReason::CoveredSubmission) {
				g_native_sweeps_covered.fetch_add(1, std::memory_order_relaxed);
			} else if (BufferCache::NativeAllOn() && reason == SweepReason::ProcessorWait) {
				g_native_sweeps_label.fetch_add(1, std::memory_order_relaxed);
			} else {
				if (BufferCache::NativeAllOn()) {
					g_native_sweep_stamp.fetch_add(1);
					g_native_sweeps_done.fetch_add(1, std::memory_order_relaxed);
				}
				m_buffer_cache.PolicyNativeSweep();
			}
			if (BufferCache::NativeAllOn()) {
				const uint64_t now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
				                            std::chrono::steady_clock::now().time_since_epoch()).count());
				uint64_t last = g_native4_last_ms.load(std::memory_order_relaxed);
				if (last == 0) g_native4_last_ms.store(now_ms, std::memory_order_relaxed);
				else if (now_ms - last >= 5000) {
					g_native4_last_ms.store(now_ms, std::memory_order_relaxed);
					std::printf("[nativo4] 5 s: barridos hechos %llu; saltados: etiqueta del procesador %llu, envio cubierto %llu\n",
					            static_cast<unsigned long long>(g_native_sweeps_done.exchange(0, std::memory_order_relaxed)),
					            static_cast<unsigned long long>(g_native_sweeps_label.exchange(0, std::memory_order_relaxed)),
					            static_cast<unsigned long long>(g_native_sweeps_covered.exchange(0, std::memory_order_relaxed)));
					std::fflush(stdout);
				}
			}
			if (dirty != 0 || reverted != 0 || everything) ::Loader::DemonsSoulsWriteAudit::NoteDrained(dirty, reverted, everything);
			::Loader::DemonsSoulsWriteAudit::NoteKeptPages(g_nowatch_kept_pages.load(std::memory_order_relaxed));
		}
	}
	SyncProtectPause();
	if (!PageManager::NativeWriteWatch()) return;
	// (The GPU thread's own translation only: a speculation's thread or a recorder owns no cache it could invalidate.
	// KYTY_SPECULATE stays off with native memory: GuestGpu.)
	if (!GuestGpu::IsGpuThread() || Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr) return;
	// (raiz1, KYTY_RAIZ) The pages the emulator's write windows opened from read-only (kernel NativeGuardedAccess:
	// a game write that landed there during the window took no fault): invalidated as CPU-written before this
	// submission's commands read memory, as their write faults would have. Only what is mapped now (as below).
	if (Raiz::On()) {
		thread_local std::vector<std::pair<uint64_t, uint64_t>> raiz_resync;
		thread_local std::vector<GuestRange>                    raiz_pieces;
		Raiz::TakeResync(&raiz_resync);
		if (!raiz_resync.empty()) {
			raiz_pieces.clear();
			{
				std::shared_lock lock(m_mapped_ranges_mutex);
				for (const auto& [begin, end]: raiz_resync)
					m_mapped_ranges.ForEachIntersection(begin, end - begin, [&](RangeSet::Range range) {
						raiz_pieces.push_back({range.address, range.size});
					});
			}
			uint64_t raiz_pages = 0;
			for (const auto& piece: raiz_pieces) {
				raiz_pages += (piece.size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
				InvalidateCpuWrittenPages(piece.address, piece.size);
			}
			Raiz::g_counters.resync_taken_pages.fetch_add(raiz_pages, std::memory_order_relaxed);
		}
	}
	// (ww2, KYTY_WRITE_WATCH_MODE=protect: the game's writes fault; nothing to read.)
	if (PageManager::WatchProtects()) return;
	// (ww2, KYTY_WRITE_WATCH_MODE=poll) A hand-over that carries no write the last sweep did not read: no sweep.
	//  - A submission the game queued before the last sweep began (the caller's PageManager::SweptSince): the game made
	//    its writes before it queued it, so before that sweep read the write watch.
	//  - A WAIT_REG_MEM satisfied by the value a command processor wrote there itself (a label one queue's commands
	//    write and another's wait for): the game handed nothing over through it. A write the game makes after it sees
	//    such a label races the commands after the wait on the console too.
	if (reason != SweepReason::Handover && PageManager::WatchPolled()) {
		auto& skipped = g_native_stats;
		(reason == SweepReason::CoveredSubmission ? skipped.covered_submissions : skipped.processor_waits)++;
		// (KYTY_WRITE_WATCH_MAX_MS: unless the last sweep is that old.)
		const auto gap = WatchMaxGap();
		if (gap.count() == 0 || std::chrono::steady_clock::now() - g_last_sweep < gap) {
			ReportNativeStats();
			return;
		}
		skipped.timed_sweeps++;
	}
	const auto start = std::chrono::steady_clock::now();
	g_last_sweep     = start;
	thread_local std::vector<PageManager::DeferredRange> runs;
	thread_local std::vector<PageManager::DeferredRange> image_runs;
	thread_local std::vector<GuestRange>                 pieces;
	runs.clear();
	image_runs.clear();
	pieces.clear();
	m_page_manager.SweepCpuWrites(&runs, &image_runs);
	uint64_t pages = 0, image_pages = 0;
	if (!runs.empty()) {
		{
			// Only what is mapped now (HandleFault's IsMapped): a page unmapped since it was written is nobody's. (The
			// lock is let go before the caches run: a buffer lookup there takes it again.)
			std::shared_lock lock(m_mapped_ranges_mutex);
			for (const auto& run: runs)
				m_mapped_ranges.ForEachIntersection(run.address, run.size, [&](RangeSet::Range range) {
					pieces.push_back({range.address, range.size});
				});
		}
		for (const auto& piece: pieces) {
			pages += (piece.size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
			// (audit1, KYTY_DS_WRITE_AUDIT: measurement only.)
			if (::Loader::DemonsSoulsWriteAudit::On()) ::Loader::DemonsSoulsWriteAudit::NoteSweptRun(piece.address, piece.size);
			InvalidateCpuWrittenPages(piece.address, piece.size);
		}
	}
	// (ww2, poll) The pages the buffers took right before GPU work wrote them (BufferCache::UploadDirtyRanges): the
	// images over them, as their write faults invalidated them too. The buffers are done (the pages were uploaded and
	// are GPU-written now: invalidated here, each would be read back).
	if (!image_runs.empty()) {
		pieces.clear();
		{
			std::shared_lock lock(m_mapped_ranges_mutex);
			for (const auto& run: image_runs)
				m_mapped_ranges.ForEachIntersection(run.address, run.size, [&](RangeSet::Range range) {
					pieces.push_back({range.address, range.size});
				});
		}
		for (const auto& piece: pieces) {
			image_pages += (piece.size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
			m_texture_cache.InvalidateCpuWrittenPages(piece.address, piece.size);
		}
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	auto& st = g_native_stats;
	st.sweeps++;
	st.sweep_us += us;
	st.sweep_max_us = std::max(st.sweep_max_us, us);
	st.pages += pages;
	st.image_pages += image_pages;
	ReportNativeStats();
}

// KYTY_NATIVE_MEMORY=1: the game reads GPU-written memory with no read fault to serve it (no page is protected). Before
// the game can learn that GPU work is done, everything that work wrote is put in guest memory: the label it waits for
// (or the event, or the flip) comes after this. The buffers first (an image over GPU-written buffer bytes is not
// downloaded: TextureCache::SafeToDownload), each download waiting for the GPU work before it; then the images, whose
// downloads are recorded behind the work that wrote them, the GPU finished and their write-backs (priority thread)
// waited for. Nothing to do (no GPU-written range, no image written since the last flush): no wait at all.
void GpuResourceManager::NativeFlushGpuWrites() {
	if (!PageManager::NativeWriteWatch()) return;
	// (nativemem1, KYTY_WRITE_WATCH: the GPU-written pages stay no-access: a read of them faults and reads them back, as
	// without native memory. Nothing to flush: no drain of the GPU at labels, end-of-pipe events and flips.)
	if (PageManager::NativeReadFaults()) return;
	if (!GuestGpu::IsGpuThread() || Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr) return;
	const bool images = NativeImageDownloadOn() && m_texture_cache.NativeHasGpuWrites();
	if (!m_buffer_cache.NativeHasGpuWrites() && !images) return;
	const auto start = std::chrono::steady_clock::now();
	auto&      st    = g_native_stats;
	st.buffer_ranges += m_buffer_cache.NativeWriteBackAll(&st.buffer_bytes);
	if (images) {
		const auto recorded = m_texture_cache.NativeDownloadWritten(&st.image_bytes, &st.refused);
		st.images += recorded;
		if (recorded != 0) {
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Finish();
			m_scheduler.WaitPriorityOperations(tick);
		}
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	st.flushes++;
	st.flush_us += us;
	st.flush_max_us = std::max(st.flush_max_us, us);
	ReportNativeStats();
}

void GpuResourceManager::InvalidateCpuWrittenPages(uint64_t vaddr, uint64_t size) {
	// (envio3, KYTY_ENVIO3) A compute submission in flight on Kyty.Acb2 (Spec::t_effects set on the GPU thread) checks
	// what was found CPU-written since it was handed: its commit stops at a clean read or image over it and re-uploads
	// the buffers it reads there (Speculation::CommitSegments / Before). Off: nothing (t_effects is null).
	// (spec5, KYTY_SPECULATE under KYTY_NATIVE_MEMORY: the same for the graphics speculation's commits; t_effects is its
	// log at every sweep while it is on, GraphicsRun.)
	if (Envio3::Mode() != 0 || kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) Spec::NoteHostWrite(vaddr, size);
	// (Fresh memory stands down with native memory - no backing view -: a first write is noted all the same. Unlike
	// the fault, the page is invalidated in any case: another watcher may hold it, and an extra invalidation is safe.)
	if (FreshMemory::Get() != nullptr)
		for (uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1); page < vaddr + size; page += TRACKER_PAGE_SIZE)
			(void)FreshFirstWrite(page);
	// The buffers, as the write faults of these pages did (no window: it only saved faults). A run with no GPU-written
	// page at once (its pages CPU-dirty). One with some, a page at a time as the faults came: InvalidateRegion reads a
	// GPU-written page back first (ReadMemory, is_write: the GPU's exact bytes over guest memory, the guest's own bytes
	// elsewhere, then CPU-dirty) - a page each, so that no buffer spanning the run is made for the readback.
	if (!m_buffer_cache.IsRegionGpuModified(vaddr, size)) {
		m_buffer_cache.InvalidateMemory(vaddr, size);
	} else {
		const auto end = vaddr + size;
		for (uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1); page < end; page += TRACKER_PAGE_SIZE) {
			const auto begin = std::max(page, vaddr);
			const auto stop  = std::min(page + TRACKER_PAGE_SIZE, end);
			m_buffer_cache.InvalidateMemory(begin, stop - begin);
		}
	}
	// Then the images over them (the whole pages: the write watch knows no bytes; a GPU-written image met only on a page
	// it shares at its first or last page is left alone, as its fault - a byte outside it - left it:
	// TextureCache::InvalidateCpuWrittenPages).
	m_texture_cache.InvalidateCpuWrittenPages(vaddr, size);
}

bool GpuResourceManager::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	TextureOffload::OnHostAccess(vaddr, size, false, false);
	FreshOverwrite(vaddr, size); // (a host write follows: its pages are writable, its granules written)
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::InvalidateImages(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) return false;
	TextureOffload::OnHostAccess(vaddr, size, false, true); // (its caller writes all of it: CompareMove)
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	TextureOffload::OnHostAccess(vaddr, size, false, true); // (it writes all of it)
	FreshOverwrite(vaddr, size);
	m_buffer_cache.InvalidateMemoryForWrite(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

void GpuResourceManager::MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEachIntersection(vaddr, size, [&](RangeSet::Range range) { out.Add(range.address, range.size); });
}

bool GpuResourceManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void GpuResourceManager::MapMemory(uint64_t vaddr, uint64_t size) {
	const auto map = [this, vaddr, size] {
		m_texture_cache.MapMemory(vaddr, size);
		{
			std::lock_guard lock(m_mapped_ranges_mutex);
			m_mapped_ranges.Add(vaddr, size);
			++m_mapping_epoch;
			if (size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size)) NoteBdaSpan(vaddr, size);
		}
		FreshWatch(vaddr, size);
	};
	if (m_gpu) {
		// A pending readback writes its backing when it finishes, which a new mapping aliasing
		// mapped memory would show; memory nothing else maps has no readback pending.
		const bool aliased = !LibKernel::Memory::IsUniqueGuestBackingRange(vaddr, size);
		m_gpu->SendCommandSync([this, map, aliased] {
			if (aliased) m_buffer_cache.DrainGuestReadback();
			map();
		});
	} else {
		map();
	}
}

void GpuResourceManager::UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	TextureOffload::OnHostAccess(vaddr, size, true, false);
	const auto unmap = [this, vaddr, size, releasing] {
		// Released memory loses its host protection with the mapping: the watchers removed
		// below need not restore it page run by page run first.
		struct UnmappingRange {
			explicit UnmappingRange(uint64_t begin, uint64_t end) { PageManager::SetUnmappingRange(begin, end); }
			~UnmappingRange() { PageManager::SetUnmappingRange(0, 0); }
		};
		std::optional<UnmappingRange> unmapping;
		if (releasing && kyty_local_unmap_protect_skip_mode.load(std::memory_order_relaxed) != 0) {
			unmapping.emplace(vaddr, vaddr + size);
		}
		// Readbacks write guest memory when they finish: the ones into the range finish first
		// (the others do not touch it).
		m_buffer_cache.DrainGuestReadback(vaddr, size);
		// Pending upload jobs on the range read its backing (or change its protection): they finish before
		// it changes owner. Jobs elsewhere keep running.
		AsyncUpload::DrainRange(vaddr, size);
		// The GPU only works on the caches' own buffers and images, which are freed after the
		// work in flight completes (deferred operations), so the unmap need not wait for the GPU
		// (it used to finish all of it: up to a frame of GPU work, for every guest unmap). What
		// writes guest memory after the GPU are the readbacks drained above and GPU-written
		// images' downloads, which complete on the priority thread: wait for those over the range.
		// (Deferred fault-buffer processing skips ranges no longer mapped.)
		const auto download_tick = m_scheduler.Active() ? m_texture_cache.PendingDownloadTick(vaddr, size) : 0;
		if (download_tick != 0) {
			const auto start = std::chrono::steady_clock::now();
			if (!m_scheduler.IsFree(download_tick)) m_scheduler.Wait(download_tick);
			m_scheduler.WaitPriorityOperations(download_tick);
			LiveCounters::Add(LiveCounters::UnmapFinishes);
			LiveCounters::Add(LiveCounters::UnmapFinishUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
		FreshUnwatch(vaddr, size, !releasing);
		const bool registered = size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		if (registered) {
			m_buffer_cache.DeleteUnmappedBuffers(vaddr, size);
			m_buffer_cache.SplitUnmappedBuffers(vaddr, size);
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		const auto epoch = ++m_mapping_epoch;
		if (registered) NoteBdaSpan(vaddr, size);
		if (size == 0) return;
		m_unmap_latest = epoch;
		constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
		for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last; ++g) {
			auto& leaf = m_unmap_epochs[g >> UnmapLeafBits];
			if (!leaf) leaf = std::make_unique<uint64_t[]>(leaf_mask + 1);
			leaf[g & leaf_mask] = epoch;
		}
	};
	if (m_gpu == nullptr) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
	// (raiz8, KYTY_RAIZ_ASYNC_RELEASE: the releases the GPU thread left over the range applied now, while it is this
	// mapping's - none applied later over a mapping made there next -.)
	PageManager::ApplyDeferredReleases(vaddr, size);
}

bool GpuResourceManager::UnmappedSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept {
	// (Nothing unmapped since: mostly so. Else each leaf the range spans looked up once.)
	if (size == 0 || m_unmap_latest <= epoch) return false;
	constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
	for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last;) {
		const auto stop = std::min(last, g | leaf_mask);
		if (const auto leaf = m_unmap_epochs.find(g >> UnmapLeafBits); leaf != m_unmap_epochs.end()) {
			const auto* epochs = leaf->second.get();
			for (auto i = g & leaf_mask; i <= (stop & leaf_mask); ++i)
				if (epochs[i] > epoch) return true;
		}
		g = stop + 1;
	}
	return false;
}

void GpuResourceManager::NoteBdaSpan(uint64_t vaddr, uint64_t size) {
	if (m_bda_spans.size() < 256) {
		m_bda_spans.push_back({vaddr, size});
	} else {
		m_bda_rebuild = true;
	}
}

void GpuResourceManager::RefreshBdaRanges() {
	if (!m_buffer_cache.TakeRegistrationSpans(m_bda_spans) || m_bda_spans.size() > 256) m_bda_rebuild = true;
	if (!m_bda_rebuild && m_bda_spans.empty()) return;
	LiveCounters::Add(LiveCounters::BdaRebuilds);
	// The tracker regions to collect again, as merged [begin, end) windows (requests lie in one region each).
	std::vector<std::pair<uint64_t, uint64_t>> windows;
	if (m_bda_rebuild) {
		windows.emplace_back(0, UINT64_MAX);
	} else {
		std::ranges::sort(m_bda_spans, {}, &GuestRange::address);
		for (const auto& span : m_bda_spans) {
			const auto begin = span.address / TRACKER_REGION_SIZE * TRACKER_REGION_SIZE;
			const auto end   = (span.End() + TRACKER_REGION_SIZE - 1) / TRACKER_REGION_SIZE * TRACKER_REGION_SIZE;
			if (!windows.empty() && begin <= windows.back().second) {
				windows.back().second = std::max(windows.back().second, end);
			} else {
				windows.emplace_back(begin, end);
			}
		}
	}
	m_bda_spans.clear();
	m_bda_rebuild = false;
	std::vector<RangeSet::Range> ranges;
	for (const auto& [window_begin, window_end] : windows) {
		m_buffer_cache.CollectMappedRegisteredRanges(m_mapped_ranges, window_begin, window_end, ranges);
		auto& fresh = m_fresh_bda_region_requests;
		fresh.clear();
		for (const auto& range : ranges) {
			const auto end = range.address + range.size;
			for (auto start = range.address; start < end;) {
				const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
				fresh.push_back({start, finish - start});
				start = finish;
			}
		}
		// An unchanged request keeps its proof: SynchronizeRegionRequest checks it against its region's CPU and
		// registration epochs (an unmap marks the range CPU-dirty, which moves the CPU epoch).
		const auto before = [](const BufferCache::SyncRegionRequest& request, uint64_t address) {
			return request.address < address;
		};
		const auto first = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), window_begin, before);
		const auto last  = std::lower_bound(first, m_bda_region_requests.end(), window_end, before);
		auto       old   = first;
		for (auto& request : fresh) {
			while (old != last && old->address < request.address) ++old;
			if (old != last && old->address == request.address && old->size == request.size) {
				request = *old;
			} else if (request.address / TRACKER_REGION_SIZE < BdaDirtyRegions::Regions) {
				// A new request starts unproven.
				BdaDirtyRegions::Mark(request.address / TRACKER_REGION_SIZE);
			}
		}
		m_bda_region_requests.insert(m_bda_region_requests.erase(first, last), fresh.begin(), fresh.end());
	}
}

namespace {
// (render3) KYTY_BDA_MEMO=1 (off by default; with KYTY_BDA_DIRTY_REGIONS): see GpuResourceManager::BdaMemo.
bool BdaMemoOn() {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_BDA_MEMO");
		const bool  is = v != nullptr && v[0] == '1';
		if (is) {
			std::printf("BDA memo: on (KYTY_BDA_MEMO=1): a bounded BDA preparation over regions already prepared, with no "
			            "dirty bit set, registration or mapping change since, is skipped\n");
			std::fflush(stdout);
		}
		return is;
	}();
	return on;
}
} // namespace

// (serie1) KYTY_SERIE1=1: the XPR culling recorded on lane 1 (local/serie1.h). Reads through the page table (flat / global
// memory) ask no buffer and leave no lane mark (lane-sync.h): its submission would not wait for the graphics work that
// wrote them. Bounded ranges: noted as reads, per 64 KiB, as a buffer's are (the waits for their last graphics writes).
// Unbounded (PrepareBda, any address): it waits for every graphics tick before it - no overtaking, nothing stale.
static void Serie1NoteBda(CommandScheduler& scheduler, const GuestRange* ranges, size_t count) {
	if (!Serie1::g_open_cull || !LaneSync::g_on || LaneSync::g_lane != 1) return;
	uint64_t pages = 0;
	for (size_t i = 0; ranges != nullptr && i < count; ++i)
		pages += ((ranges[i].address + ranges[i].size - 1) >> LaneSync::PageBits) - (ranges[i].address >> LaneSync::PageBits) + 1;
	if (ranges == nullptr || pages > 4096) {
		const uint64_t newest = scheduler.GetMasterSemaphore().LastTick(0);
		if (newest > LaneSync::g_required[0]) LaneSync::g_required[0] = newest;
		++Serie1::g_c.bda_full;
		return;
	}
	for (size_t i = 0; i < count; ++i) {
		const uint64_t first = ranges[i].address >> LaneSync::PageBits;
		const uint64_t last  = (ranges[i].address + ranges[i].size - 1) >> LaneSync::PageBits;
		for (uint64_t page = first; page <= last; ++page)
			LaneSync::Note(LaneSync::g_pages[page & (LaneSync::PageSlots - 1)], false);
	}
	++Serie1::g_c.bda_ranges;
}

bool GpuResourceManager::PrepareBdaReadRanges(std::span<const GuestRange> ranges) {
	if (ranges.empty() || ranges.size() > 128 ||
	    std::ranges::any_of(ranges, [](const auto& range) { return !range.Valid(); })) return false;
	if (auto* spec = Spec::Current()) {
		// Prepared when it is committed (and its guest memory writes made after its work).
		if (std::ranges::any_of(ranges, [&](const auto& range) { return spec->open_journal_set.Intersects(range.address, range.size); }))
			Spec::Refuse("memory the processor writes");
		spec->bda.insert(spec->bda.end(), ranges.begin(), ranges.end());
		return true;
	}
	Serie1NoteBda(m_scheduler, ranges.data(), ranges.size()); // (serie1)
	for (const auto& range: ranges) MemoryTracker::HotLazyListed(range.address, range.size); // (PROT-MENOS7)
	// (Before the lock: making buffers asks for it.)
	m_buffer_cache.EnsureBdaReadable(ranges);
	// (render3) KYTY_BDA_MEMO: the state the memo stands on, read after the buffers above were made (a buffer made
	// there moved the registration epoch and marked its regions: the memo misses) and before any work below (a bit set
	// meanwhile moves the generation: the next call misses).
	const bool memo = kyty_local_bda_dirty_regions_mode.load(std::memory_order_relaxed) != 0 && BdaMemoOn();
	uint64_t   memo_generation = 0, memo_registration = 0, memo_mapping = 0;
	// (The tracker regions SynchronizeDirtyBdaRegions visits for a range: a run of them, or none.)
	const auto memo_regions = [](const GuestRange& range, uint64_t& first, uint64_t& last) {
		first = range.address / TRACKER_REGION_SIZE;
		if (first >= BdaDirtyRegions::Regions) return false;
		last = std::min<uint64_t>((range.End() - 1) / TRACKER_REGION_SIZE, BdaDirtyRegions::Regions - 1);
		return true;
	};
	if (memo) {
		memo_generation   = BdaDirtyRegions::Generation();
		memo_registration = m_buffer_cache.RegistrationEpoch();
		memo_mapping      = m_mapping_epoch.load(std::memory_order_acquire);
		if (m_bda_memo.count != 0 && m_bda_memo.generation == memo_generation &&
		    m_bda_memo.registration == memo_registration && m_bda_memo.mapping == memo_mapping) {
			bool covered = true;
			for (const auto& range : ranges) {
				uint64_t first = 0, last = 0;
				if (!memo_regions(range, first, last)) continue; // (no region of it is visited)
				bool held = false;
				for (size_t i = 0; i < m_bda_memo.count && !held; ++i)
					held = m_bda_memo.regions[i].first <= first && last <= m_bda_memo.regions[i].second;
				if (!held) {
					covered = false;
					break;
				}
			}
			if (covered) {
				// Every region of these ranges had its bit consumed since the state above was read, and none was set
				// again: the walk below would find no bit (and RefreshBdaRanges no change).
				LiveCensus::g_bda_memo[0] += 1;
				m_fault_process_pending = true;
				m_bda_used              = true;
				return true;
			}
		}
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	LiveCounters::Add(LiveCounters::BdaRangeCalls);
	LiveCounters::Add(LiveCounters::BdaRanges, ranges.size());
	for (const auto range : ranges) LiveCounters::Add(LiveCounters::BdaRangeMiB, range.size >> 20u);
	if (kyty_local_bda_dirty_regions_mode.load(std::memory_order_relaxed) != 0) {
		// KYTY_BDA_DIRTY_REGIONS: only regions whose bit is set can hold a stale request.
		for (const auto range : ranges) SynchronizeDirtyBdaRegions(range);
		if (memo) {
			LiveCensus::g_bda_memo[1] += 1;
			// The regions just walked join the memo's while its state is the one read above; else they begin a new one.
			// (A bit set during the walk - a request left unproven, a registration - moved the generation: the memo
			// made here misses at the next call, which walks again.)
			if (m_bda_memo.generation != memo_generation || m_bda_memo.registration != memo_registration ||
			    m_bda_memo.mapping != memo_mapping) {
				m_bda_memo.generation   = memo_generation;
				m_bda_memo.registration = memo_registration;
				m_bda_memo.mapping      = memo_mapping;
				m_bda_memo.count        = 0;
			}
			for (const auto& range : ranges) {
				uint64_t first = 0, last = 0;
				if (!memo_regions(range, first, last)) continue;
				// Runs that touch or overlap the new one are merged into it (until none does).
				for (bool merged = true; merged;) {
					merged = false;
					for (size_t i = 0; i < m_bda_memo.count; ++i) {
						const auto run = m_bda_memo.regions[i];
						if (run.first > last + 1 || first > run.second + 1) continue;
						first                 = std::min<uint64_t>(first, run.first);
						last                  = std::max<uint64_t>(last, run.second);
						m_bda_memo.regions[i] = m_bda_memo.regions[m_bda_memo.count - 1];
						--m_bda_memo.count;
						merged = true;
						break;
					}
				}
				if (m_bda_memo.count == BdaMemo::Capacity) m_bda_memo.count = 0; // (full: the older runs are forgotten)
				m_bda_memo.regions[m_bda_memo.count++] = {first, last};
			}
		}
		m_fault_process_pending = true;
		m_bda_used              = true;
		return true;
	}
	for (const auto range : ranges) {
		auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), range.address,
		    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
		for (; it != m_bda_region_requests.end() && it->address < range.End(); ++it)
			m_buffer_cache.SynchronizeRegionRequest(*it);
	}
	m_fault_process_pending = true;
	m_bda_used              = true;
	return true;
}

void GpuResourceManager::SynchronizeDirtyBdaRegions(GuestRange range) {
	const auto first = range.address / TRACKER_REGION_SIZE;
	if (first >= BdaDirtyRegions::Regions) return;
	const auto last = std::min((range.End() - 1) / TRACKER_REGION_SIZE, BdaDirtyRegions::Regions - 1);
	for (auto word = first / 64; word <= last / 64; ++word) {
		auto mask = ~uint64_t {0};
		if (word == first / 64) mask &= ~uint64_t {0} << (first % 64);
		if (word == last / 64) mask &= ~uint64_t {0} >> (63 - last % 64);
		auto& bits = BdaDirtyRegions::g_bits[word];
		for (auto pending = bits.load(std::memory_order_acquire) & mask; pending != 0; pending &= pending - 1) {
			const auto bit = uint64_t {1} << std::countr_zero(pending);
			// Clear before synchronizing: a CPU write after the epoch read sets it again.
			if ((bits.fetch_and(~bit, std::memory_order_acq_rel) & bit) == 0) continue;
			const auto region_begin = (word * 64 + static_cast<uint64_t>(std::countr_zero(pending))) * TRACKER_REGION_SIZE;
			auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), region_begin,
			    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
			bool unproven = false;
			for (; it != m_bda_region_requests.end() && it->address < region_begin + TRACKER_REGION_SIZE; ++it) {
				m_buffer_cache.SynchronizeRegionRequest(*it);
				unproven |= it->cpu_epoch == 0;
			}
			// A request left unproven by its own uploads' queued write protection needs no other visit:
			// a write before the protection lands is in the copies taken after it, and a write after it
			// faults (and marks the region).
			if (unproven && !m_buffer_cache.DeferredProtectionPending(region_begin))
				BdaDirtyRegions::Mark(region_begin / TRACKER_REGION_SIZE);
		}
	}
}

void GpuResourceManager::PrepareBda() {
	if (auto* spec = Spec::Current()) {
		// Prepared when it is committed (and its guest memory writes made after its work).
		if (spec->writes.size() > spec->Sealed().writes) Spec::Refuse("memory the processor writes");
		spec->bda_all = true;
		return;
	}
	Serie1NoteBda(m_scheduler, nullptr, 0); // (serie1)
	MemoryTracker::HotLazyListed(0, 0); // (PROT-MENOS7: every region with hot pages)
	// Unknown shader addresses cannot prove disjointness from an in-flight readback.
	LiveCounters::Add(LiveCounters::BdaFullSyncs);
	m_buffer_cache.DrainGuestReadback();
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	for (auto& request : m_bda_region_requests) m_buffer_cache.SynchronizeRegionRequest(request);
	m_fault_process_pending = true;
	m_bda_used              = true;
}

void GpuResourceManager::EndSubmission() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	// (ram19) KYTY_RAM19=1: the GPU residency request answered here too (see AdvanceFrame): with the game's own flips no
	// FlipPreparation comes and AdvanceFrame never ran. Same copy, same thread (the buffer cache is this thread's).
	if (GpuResidency::Ram19On() && GpuResidency::Requested()) {
		const auto                       start = std::chrono::steady_clock::now();
		std::vector<GpuResidency::Range> ranges;
		if (const uint64_t request = GpuResidency::Take(&ranges); request != 0) {
			uint64_t images = 0, buffers = 0;
			m_texture_cache.CollectResidencyRanges(ranges, images);
			m_buffer_cache.CollectResidencyRanges(ranges, buffers);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			GpuResidency::Deliver(request, std::move(ranges), images, buffers, ms);
		}
	}
	if (std::exchange(m_gpu_flip_pending, false)) RunCollectors();
}

void GpuResourceManager::AdvanceFrame() {
	// KYTY_GPU_RESIDENCY (gpu-residency.h): the registered ranges copied for the measurement's thread, once every 30 s
	// (otherwise an atomic load). Here no cache lock is held, and the buffer cache is this thread's.
	if (GpuResidency::Requested()) {
		const auto                       start = std::chrono::steady_clock::now();
		std::vector<GpuResidency::Range> ranges;
		// (0: the measurement's thread holds the exchange's mutex; never waited for, tried again at the next flip.)
		if (const uint64_t request = GpuResidency::Take(&ranges); request != 0) {
			uint64_t images = 0, buffers = 0;
			m_texture_cache.CollectResidencyRanges(ranges, images);
			m_buffer_cache.CollectResidencyRanges(ranges, buffers);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			GpuResidency::Deliver(request, std::move(ranges), images, buffers, ms);
		}
	}
	m_texture_cache.AdvanceFrame();
	RunCollectors();
	if (auto* fresh = FreshMemory::Get(); fresh != nullptr && ++m_fresh_report_flip % 3600 == 0) {
		size_t parts = 0;
		{
			std::lock_guard lock(m_fresh_mutex);
			parts = m_fresh_parts.size();
		}
		std::printf("Fresh memory: %llu MiB of uploads were never-written bytes (zeros, not read); %zu parts watched; "
		            "%llu MiB written at a first write, %llu MiB by other writes\n",
		            static_cast<unsigned long long>(fresh->zero_upload_bytes.load() >> 20u), parts,
		            static_cast<unsigned long long>(fresh->faults.load() >> 20u),
		            static_cast<unsigned long long>(fresh->marked.load() >> 20u));
		std::fflush(stdout);
	}
}

void GpuResourceManager::FreshWatch(uint64_t vaddr, uint64_t size) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr || size == 0) return;
	bool gpu = false, cpu_write = false;
	if (!LibKernel::Memory::GuestRangeAccess(vaddr, size, &gpu, &cpu_write)) {
		FreshMemory::MarkGuestRange(vaddr, size);
		return;
	}
	if (!cpu_write) return; // (no guest write here goes unseen)
	if (!gpu) {
		FreshMemory::MarkGuestRange(vaddr, size); // (written unwatched)
		return;
	}
	std::lock_guard lock(m_fresh_mutex);
	uint64_t        run_begin = 0, run_end = 0;
	const auto      flush     = [&] {
		if (run_end > run_begin) m_page_manager.UpdatePageWatchers<true>(run_begin, run_end - run_begin);
		run_begin = run_end = 0;
	};
	for (uint64_t done = 0; done < size;) {
		const uint64_t at      = vaddr + done;
		const auto     end     = LibKernel::Memory::BackingMappingEnd(at);
		const auto     piece   = end > at ? std::min(end - at, size - done) : size - done;
		const auto*    backing = end > at ? LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
		if (backing == nullptr) {
			done += piece;
			continue;
		}
		for (uint64_t offset = 0; offset < piece;) {
			const auto address = reinterpret_cast<uint64_t>(backing) + offset;
			const auto granule = FreshMemory::GranuleOf(*fresh, address);
			const auto bytes   = granule == UINT64_MAX
			                         ? piece - offset
			                         : std::min(piece - offset,
			                                    FreshMemory::Granule - ((address - fresh->base) & (FreshMemory::Granule - 1)));
			const auto begin = at + offset;
			offset += bytes;
			if (granule == UINT64_MAX || FreshMemory::IsWritten(*fresh, granule)) continue;
			if ((begin & (TRACKER_PAGE_SIZE - 1)) != 0 || (bytes & (TRACKER_PAGE_SIZE - 1)) != 0 ||
			    m_fresh_parts.contains(begin)) {
				// (Not whole pages, or watched already: counted as written rather than half watched.)
				if (FreshMemory::MarkGranule(*fresh, granule)) fresh->marked.fetch_add(FreshMemory::Granule);
				FreshUnwatchGranule(granule);
				continue;
			}
			m_fresh_parts.emplace(begin, FreshPart {begin + bytes, granule});
			m_fresh_granules[granule].push_back(begin);
			FreshMemory::SetArmed(*fresh, granule, true);
			fresh->watched_bytes.fetch_add(bytes, std::memory_order_relaxed);
			if (begin != run_end) flush();
			if (run_end == 0) run_begin = begin;
			run_end = begin + bytes;
		}
		done += piece;
	}
	flush();
}

void GpuResourceManager::FreshUnwatchGranule(uint64_t granule) {
	const auto found = m_fresh_granules.find(granule);
	if (found == m_fresh_granules.end()) return;
	for (const auto begin: found->second) {
		const auto part = m_fresh_parts.find(begin);
		if (part == m_fresh_parts.end()) continue;
		m_page_manager.UpdatePageWatchers<false>(begin, part->second.end - begin);
		m_fresh_parts.erase(part);
	}
	m_fresh_granules.erase(found);
	if (auto* fresh = FreshMemory::Get()) FreshMemory::SetArmed(*fresh, granule, false);
}

void GpuResourceManager::FreshUnwatch(uint64_t vaddr, uint64_t size, bool mark) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr || size == 0) return;
	std::lock_guard       lock(m_fresh_mutex);
	std::vector<uint64_t> granules;
	auto                  it = m_fresh_parts.upper_bound(vaddr);
	if (it != m_fresh_parts.begin()) --it;
	for (; it != m_fresh_parts.end() && it->first < vaddr + size; ++it) {
		if (it->second.end <= vaddr) continue;
		// (A part left partly mapped loses its watch with the others: written.)
		const bool whole = it->first >= vaddr && it->second.end <= vaddr + size;
		if ((mark || !whole) && FreshMemory::MarkGranule(*fresh, it->second.granule))
			fresh->marked.fetch_add(FreshMemory::Granule);
		granules.push_back(it->second.granule);
	}
	for (const auto granule: granules) FreshUnwatchGranule(granule);
}

bool GpuResourceManager::FreshFirstWrite(uint64_t vaddr) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr) return false;
	std::lock_guard lock(m_fresh_mutex);
	auto            it = m_fresh_parts.upper_bound(vaddr);
	if (it == m_fresh_parts.begin()) return false;
	--it;
	if (vaddr >= it->second.end) return false;
	const auto granule = it->second.granule;
	if (FreshMemory::MarkGranule(*fresh, granule)) fresh->faults.fetch_add(FreshMemory::Granule);
	FreshUnwatchGranule(granule);
	return true;
}

void GpuResourceManager::FreshOverwrite(uint64_t vaddr, uint64_t size) {
	if (FreshMemory::Get() == nullptr || size == 0) return;
	FreshMemory::MarkGuestRange(vaddr, size);
	FreshUnwatch(vaddr, size, true);
}

void GpuResourceManager::RunCollectors() {
	// Collection runs per frame: per submission (dozens a frame here) its ages of 16 to 160 ticks
	// were a few frames, and it deleted what the next frames used again.
	m_texture_cache.RunGarbageCollector();
	// A shader that reads guest memory through the BDA page table touches no buffer in the LRU: a
	// buffer collected under it faults back at once, and each registration invalidates every BDA
	// region proof. Buffers are collected only after a frame without such shaders.
	// Demon's Souls runs such shaders every frame, so its buffers were never collected: the buffer cache grew from 0.5 to
	// 4.2 GB in 17 minutes of 01.005.000 and took the video memory to 12.8 GB of a 10 GB budget. Past the budget they
	// are collected once a second anyway (the faults and BDA proofs that costs, once a second, instead of a full card).
	const bool bda_free    = !std::exchange(m_bda_used, false);
	const bool over_budget = m_graphics.CanReportMemoryUsage() &&
	                         m_graphics.GetDeviceMemoryUsage() >= m_graphics.GetTotalMemoryBudget();
	m_buffer_cache.RunGarbageCollector(bda_free || (over_budget && (++m_collections % 60) == 0));
}

} // namespace Libs::Graphics
