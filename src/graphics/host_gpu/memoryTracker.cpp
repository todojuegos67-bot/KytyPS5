#include "graphics/host_gpu/memoryTracker.h"
#include "live-counters.h"

#include "common/assert.h"
#include "common/guardedCopy.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsMemMap.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

// KYTY_VOLATILE (see MemoryTracker::VolatileRedirty).
namespace {
// (v4) How the allocator/ctx regions (StepTagged) are re-dirtied. KYTY_VOLATILE_MODE:
//   frame  (default) the regions some upload made clean, once per game frame (RenderContextFrameCPU +0xc4 moved;
//          without it, every 16 ms), plus at each call the ctx block of every DCB submitted since the last call
//   ww     KYTY_NATIVE_MEMORY=1 only (else frame): the pages the write watch saw written since the last call, at
//          every call (GetWriteWatch + reset over the registered regions only)
//   submit v3: the regions some upload made clean, at every call
// KYTY_VOLATILE_EVERY=N (frame mode, default 0 = off): also every N calls inside a game frame.
enum class VolatileMode { Frame, WriteWatch, Submit };
struct VolatileConfig {
	bool                    on = false; // (the game's rings and the manual ranges)
	std::vector<GuestRange> ranges;
	VolatileMode            mode  = VolatileMode::Frame;
	uint64_t                every = 0;
};

VolatileConfig ParseVolatileConfig() {
	VolatileConfig config;
	const char*    mode = std::getenv("KYTY_VOLATILE");
	if (mode == nullptr || std::strcmp(mode, "1") != 0) {
		std::printf("Volatile: off\n");
		std::fflush(stdout);
		return config;
	}
	config.on = true;
	if (const char* m = std::getenv("KYTY_VOLATILE_MODE"); m != nullptr) {
		if (std::strcmp(m, "submit") == 0) config.mode = VolatileMode::Submit;
		if (std::strcmp(m, "ww") == 0) config.mode = VolatileMode::WriteWatch;
	}
	if (config.mode == VolatileMode::WriteWatch && !Libs::LibKernel::Memory::NativeMemoryEnabled()) {
		std::printf("Volatile: mode ww needs KYTY_NATIVE_MEMORY=1 (guest memory without MEM_WRITE_WATCH): frame\n");
		config.mode = VolatileMode::Frame;
	}
	if (const char* e = std::getenv("KYTY_VOLATILE_EVERY"); e != nullptr) config.every = std::strtoull(e, nullptr, 10);
	std::printf("Volatile: alloc/ctx mode %s, every %llu\n",
	            config.mode == VolatileMode::Frame ? "frame" : config.mode == VolatileMode::WriteWatch ? "ww" : "submit",
	            static_cast<unsigned long long>(config.every));
	const char* list = std::getenv("KYTY_VOLATILE_RANGES");
	std::string text = list != nullptr ? list : "";
	uint64_t    total = 0;
	std::string shown;
	size_t      pos = 0;
	while (pos < text.size()) {
		size_t next = text.find(',', pos);
		if (next == std::string::npos) next = text.size();
		const std::string item = text.substr(pos, next - pos);
		pos                    = next + 1;
		const size_t colon     = item.find(':');
		if (item.empty()) continue;
		char*          end_addr = nullptr;
		char*          end_size = nullptr;
		const uint64_t addr     = std::strtoull(item.c_str(), &end_addr, 0);
		const uint64_t size =
		    colon == std::string::npos ? 0 : std::strtoull(item.c_str() + colon + 1, &end_size, 0);
		const GuestRange range {addr, size};
		if (colon == std::string::npos || end_addr != item.c_str() + colon || end_size == nullptr ||
		    *end_size != '\0' || !range.Valid()) {
			std::printf("Volatile: ignored range \"%s\" (want 0xADDR:0xSIZE)\n", item.c_str());
			continue;
		}
		config.ranges.push_back(range);
		total += size;
		char buf[64];
		std::snprintf(buf, sizeof(buf), "%s0x%llx:0x%llx", shown.empty() ? "" : ",",
		              static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size));
		shown += buf;
	}
	if (config.ranges.empty()) {
		std::printf("Volatile: on (game rings)\n");
	} else {
		std::printf("Volatile: on (game rings), %zu manual ranges (%.1f MiB): %s\n", config.ranges.size(),
		            static_cast<double>(total) / (1024.0 * 1024.0), shown.c_str());
	}
	std::fflush(stdout);
	return config;
}

const VolatileConfig& GetVolatileConfig() {
	static const VolatileConfig config = ParseVolatileConfig();
	return config;
}

// The volatile regions, append-only (a region manager lives as long as its tracker). Written under the tracker's
// region mutex; VolatileRedirty reads it without it.
constexpr size_t                   VOLATILE_MAX = 4096; // (16 GiB of regions)
std::atomic<RegionManager*>        g_volatile_regions[VOLATILE_MAX];
std::atomic<size_t>                g_volatile_count {0};

// The tracker the GPU thread re-dirties (one per process).
std::atomic<MemoryTracker*> g_tracker {nullptr};

// The [volatile] line every 5 s (GPU thread only).
struct VolatileStats {
	uint64_t calls = 0, frames = 0, ring_pages = 0, manual_pages = 0, zone_bytes = 0, full = 0;
	uint64_t tagged_pages = 0, tagged_regions = 0;
	uint64_t frame_redirties = 0, ctx_submits = 0, ctx_pages = 0, ww_pages = 0, ww_fail = 0;
	std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
};
VolatileStats g_volatile_stats;

// Demon's Souls 01.007.000 (eboot at 0x900000000, as demonsSoulsWarp.cpp): the render backend singleton
// (CRenderBackend, 0xe00 bytes, built by 0x902630) is stored at eboot+0x2f65f58 by its base constructor 0xaa3890. The
// backend holds two frame allocators (vtable eboot+0x25afc58, init 0x838480: frames +0xc0 = 2, base +0x90, end +0x98,
// cursor +0xa0, the frame starts' array +0x88, the current frame +0xc4, size +0xc8): RenderContextFrameCPU at
// backend+0x6d8 over the 130 MiB block at [backend+0x8d8] (Alloc returns at 0x902f20) and RenderContextFrameGPU at
// backend+0x7d8 over the 20 MiB block at [backend+0x8e0] (0x902f3f). BeginFrame (0x838500) moves +0xc4 and records
// the cursor as the new frame's start.
constexpr uint64_t DsImage      = 0x900000000ull;
constexpr uint64_t DsBackendPtr = DsImage + 0x2f65f58;
constexpr uint64_t DsAllocVtbl  = DsImage + 0x25afc58;
constexpr uint64_t DsBackendLen = 0xe00;

struct GameRing {
	const char* name;
	uint64_t    object_offset, block_offset, size;
	uint64_t    object = 0, base = 0;
	uint64_t    lower  = 0;     // (re-dirty from here: the start of the game frame seen at the last call)
	uint32_t    frame  = ~0u;
};
struct GameRings {
	bool     checked = false, supported = false, found = false;
	uint32_t tries = 0;
	GameRing rings[2] = {{"CPU", 0x6d8, 0x8d8, 0x8200000}, {"GPU", 0x7d8, 0x8e0, 0x1400000}};
};
GameRings g_game; // (GPU thread only)

// The allocator (DefaultTagg...) and ctx regions made volatile from the memory map's granules (GPU thread only; a
// region already volatile, a ring or a manual one, is not taken: it keeps its own re-dirty). Never removed.
struct Tagged {
	RegionManager* regions[VOLATILE_MAX] {};
	uint64_t       bases[VOLATILE_MAX] {}; // (v4: the region's guest address)
	uint64_t       submit_cursor = 0;      // (v4: DemonsSoulsMemMap::SubmitsSince)
	uint64_t       calls_in_frame = 0;
	std::chrono::steady_clock::time_point last_full {};
	size_t         count = 0, taken = 0; // (regions; granules read from the map)
	uint64_t       kind_regions[2] {};
	size_t         printed = 0;
	std::chrono::steady_clock::time_point print_at {};
};
Tagged g_tagged;

template <typename T>
bool Peek(uint64_t address, T* out) {
	return address >= 0x10000 && Common::GuardedCopy(out, reinterpret_cast<const void*>(address), sizeof(T)) == 0;
}

// The code that stores the singleton and the two blocks, as 01.007.000 has it.
bool GameCodeMatches() {
	// 0xaa3890: push rbp; mov rbp, rsp; push rbx; push rax; vxorps; vmovups [rdi]; mov rbx, [rip -> 0x2f65f58]
	static constexpr uint8_t Singleton[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0xc5, 0xf8, 0x57, 0xc0, 0xc5,
	                                        0xf8, 0x11, 0x07, 0x48, 0x8b, 0x1d, 0xb3, 0x26, 0x4c, 0x02};
	// 0x902f33: mov [rbx + 0x8d8], rax; 0x902f3f: mov [rbx + 0x8e0], rax (the call between them not checked: a patch
	// of the allocator's call sites may own it)
	static constexpr uint8_t Cpu[] = {0x48, 0x89, 0x83, 0xd8, 0x08, 0x00, 0x00};
	static constexpr uint8_t Gpu[] = {0x48, 0x89, 0x83, 0xe0, 0x08, 0x00, 0x00};
	uint8_t singleton[sizeof(Singleton)] {};
	uint8_t cpu[sizeof(Cpu)] {}, gpu[sizeof(Gpu)] {};
	return Peek(DsImage + 0xaa3890, &singleton) && std::memcmp(singleton, Singleton, sizeof(Singleton)) == 0 &&
	       Peek(DsImage + 0x902f33, &cpu) && std::memcmp(cpu, Cpu, sizeof(Cpu)) == 0 &&
	       Peek(DsImage + 0x902f3f, &gpu) && std::memcmp(gpu, Gpu, sizeof(Gpu)) == 0;
}
} // namespace

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
	for (size_t i = 0; i < REGION_COUNT; i++) {
		m_regions[i].store(nullptr, std::memory_order_relaxed);
	}
	if (GetVolatileConfig().on) g_tracker.store(this, std::memory_order_release); // (and the startup line)
}

MemoryTracker::~MemoryTracker() {
	// (Its managers go with it: none is re-dirtied after this.)
	MemoryTracker* self = this;
	g_tracker.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
	if (self == this) g_tagged.count = 0; // (its regions go with it; shutdown only)
	std::lock_guard lock(m_region_mutex);
	for (const auto& manager: m_region_storage) {
		if (!manager->IsVolatile()) continue;
		const size_t count = g_volatile_count.load(std::memory_order_acquire);
		for (size_t i = 0; i < count; i++) {
			RegionManager* expected = manager.get();
			g_volatile_regions[i].compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
		}
	}
}

size_t MemoryTracker::MakeVolatile(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	size_t changed = 0;
	for (uint64_t index = vaddr / TRACKER_REGION_SIZE; index <= (vaddr + size - 1) / TRACKER_REGION_SIZE; index++) {
		auto*           manager = GetOrCreateRegion(index);
		std::scoped_lock lock(manager->lock);
		if (manager->IsVolatile()) continue;
		(void)manager->SetVolatile();
		changed++;
	}
	return changed;
}

uint64_t MemoryTracker::RedirtyVolatile(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) return 0;
	uint64_t pages = 0;
	uint64_t end   = vaddr + size;
	for (uint64_t at = vaddr; at < end;) {
		const uint64_t index  = at / TRACKER_REGION_SIZE;
		const uint64_t region = index * TRACKER_REGION_SIZE;
		const uint64_t stop   = std::min(end, region + TRACKER_REGION_SIZE);
		if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
			std::scoped_lock lock(manager->lock);
			if (manager->IsVolatile()) {
				pages += manager->MarkVolatileDirty(static_cast<size_t>((at - region) / TRACKER_PAGE_SIZE),
				                                    static_cast<size_t>((stop - region + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE));
			}
		}
		at = stop;
	}
	return pages;
}

// KYTY_VOLATILE=1: finds the game's rings (once), then re-dirties what the game may have written in them since the
// last call. GPU thread only.
void MemoryTracker::StepGameRings(uint64_t& ring_pages, bool& new_frame, bool& frame_known) {
	auto& game = g_game;
	if (!game.checked) {
		game.checked   = true;
		game.supported = DemonsSouls::IsSupportedGame() && GameCodeMatches();
		if (!game.supported) {
			std::printf("[volatile] game rings: not Demon's Souls 01.007.000 (code differs), only manual ranges\n");
			std::fflush(stdout);
		}
	}
	if (!game.supported) return;
	if (!game.found) {
		// Until the game built its backend (a few loads per call; nothing else happens meanwhile).
		uint64_t backend = 0;
		if (!Peek(DsBackendPtr, &backend) || backend == 0) return;
		for (auto& ring: game.rings) {
			uint64_t vtbl = 0, base = 0, end = 0, size = 0, block = 0;
			uint32_t frames = 0;
			const uint64_t object = backend + ring.object_offset;
			if (!Peek(backend + ring.block_offset, &block) || !Peek(object, &vtbl) || !Peek(object + 0x90, &base) ||
			    !Peek(object + 0x98, &end) || !Peek(object + 0xc8, &size) || !Peek(object + 0xc0, &frames) ||
			    vtbl != DsAllocVtbl || block == 0 || base != block || size != ring.size || end != base + size ||
			    frames != 2 || !GuestRange {base, size}.Valid()) {
				if (++game.tries == 600) { // (seen 600 times: say why once)
					std::printf("[volatile] game rings: backend 0x%llx, %s ring not initialized yet (vtbl 0x%llx base "
					            "0x%llx size 0x%llx frames %u)\n",
					            static_cast<unsigned long long>(backend), ring.name, static_cast<unsigned long long>(vtbl),
					            static_cast<unsigned long long>(base), static_cast<unsigned long long>(size), frames);
					std::fflush(stdout);
				}
				return;
			}
			ring.object = object;
			ring.base   = base;
		}
		game.found      = true;
		size_t   regions = 0;
		// (v5) The rings stay tracked as before unless KYTY_VOLATILE_RINGS=1: the GPU ring holds GPU-written sync
		// values a WAIT_REG_MEM waits on; made volatile, a load screen stalled ("tick not reached"). They are only
		// found here: the frame counter (+0xc4) and the exclusion below use them.
		static const bool rings_volatile = [] {
			const char* v = std::getenv("KYTY_VOLATILE_RINGS");
			return v != nullptr && v[0] == '1';
		}();
		for (auto& ring: game.rings) {
			if (rings_volatile) regions += MakeVolatile(ring.base, ring.size);
			ring.lower = ring.base; // (the first call re-dirties each ring whole)
		}
		uint64_t pool = 0;
		(void)Peek(backend + 0xde8, &pool);
		std::printf("[volatile] game rings: CPU 0x%010llx+130 MiB (0x%010llx), GPU 0x%010llx+20 MiB (0x%010llx), "
		            "%zu regions of 4 MiB made volatile [%010llx..%010llx) [%010llx..%010llx); backend 0x%llx; "
		            "not volatile: 112 MiB pool 0x%010llx\n",
		            static_cast<unsigned long long>(game.rings[0].base),
		            static_cast<unsigned long long>(game.rings[0].base + game.rings[0].size),
		            static_cast<unsigned long long>(game.rings[1].base),
		            static_cast<unsigned long long>(game.rings[1].base + game.rings[1].size), regions,
		            static_cast<unsigned long long>(game.rings[0].base & ~(TRACKER_REGION_SIZE - 1)),
		            static_cast<unsigned long long>((game.rings[0].base + game.rings[0].size + TRACKER_REGION_SIZE - 1) &
		                                            ~(TRACKER_REGION_SIZE - 1)),
		            static_cast<unsigned long long>(game.rings[1].base & ~(TRACKER_REGION_SIZE - 1)),
		            static_cast<unsigned long long>((game.rings[1].base + game.rings[1].size + TRACKER_REGION_SIZE - 1) &
		                                            ~(TRACKER_REGION_SIZE - 1)),
		            static_cast<unsigned long long>(backend), static_cast<unsigned long long>(pool));
		std::fflush(stdout);
	}
	auto& st = g_volatile_stats;
	for (auto& ring: game.rings) {
		const uint64_t base = ring.base, end = ring.base + ring.size;
		uint64_t       cursor = 0, starts = 0, start = 0;
		uint32_t       frame  = 0;
		// The game may move these while they are read: each value is checked, a bad read re-dirties the ring whole.
		const bool ok = Peek(ring.object + 0xa0, &cursor) && Peek(ring.object + 0xc4, &frame) && frame < 2 &&
		                Peek(ring.object + 0x88, &starts) && Peek(starts + frame * 8ull, &start) && cursor >= base &&
		                cursor <= end && start >= base && start <= end;
		if (!ok) {
			ring_pages += RedirtyVolatile(base, ring.size);
			st.zone_bytes += ring.size;
			st.full++;
			ring.lower = base;
			continue;
		}
		if (&ring == &game.rings[0]) frame_known = true;
		if (frame != ring.frame) {
			if (&ring == &game.rings[0]) {
				st.frames++;
				new_frame = true;
			}
			ring.frame = frame;
		}
		// [lower, cursor) in ring order: every allocation made since the frame seen at the last call began (the game
		// writes a frame's data while that frame is current, before submitting it).
		const uint64_t lower = ring.lower & ~(TRACKER_PAGE_SIZE - 1);
		if (cursor >= lower) {
			ring_pages += RedirtyVolatile(lower, cursor - lower);
			st.zone_bytes += cursor - lower;
		} else { // (wrapped)
			ring_pages += RedirtyVolatile(lower, end - lower) + RedirtyVolatile(base, cursor - base);
			st.zone_bytes += (end - lower) + (cursor - base);
		}
		ring.lower = start;
	}
}

namespace {
void CollectWrittenPage(uint64_t page_vaddr, void* user) {
	static_cast<std::vector<uint64_t>*>(user)->push_back(page_vaddr);
}
} // namespace

void MemoryTracker::StepTagged(bool new_frame, bool frame_known, uint64_t& tagged_pages, uint64_t& tagged_regions) {
	auto& t = g_tagged;
	// (v5) Not before the rings are known: a granule over them must be left out.
	if (!g_game.found) return;
	// New granules: each region made volatile once (GetOrCreateRegion, then its lock: SetVolatile unprotects what is
	// still protected, once). The same path v2 takes for the rings.
	uint64_t granules[256];
	for (;;) {
		const size_t n = Loader::DemonsSoulsMemMap::VolatileGranules(t.taken, granules, 256);
		if (n == 0) break;
		t.taken += n;
		for (size_t i = 0; i < n; i++) {
			const uint64_t kind  = granules[i] & 1u;
			const uint64_t vaddr = (granules[i] >> 1u) << 22u;
			if (!GuestRange {vaddr, uint64_t {1} << 22u}.Valid()) continue;
			{
				// (v5) Never a 4 MiB granule over a ring (see StepGameRings): ctx blocks may live in the GPU ring.
				bool over_ring = false;
				for (const auto& ring: g_game.rings)
					if (ring.base != 0 && vaddr < ring.base + ring.size + TRACKER_REGION_SIZE &&
					    ring.base < vaddr + (uint64_t {1} << 22u) + TRACKER_REGION_SIZE)
						over_ring = true;
				if (over_ring) continue;
			}
			for (uint64_t index = vaddr / TRACKER_REGION_SIZE; index <= (vaddr + (uint64_t {1} << 22u) - 1) / TRACKER_REGION_SIZE;
			     index++) {
				if (t.count == VOLATILE_MAX) break;
				auto*            manager = GetOrCreateRegion(index);
				std::scoped_lock lock(manager->lock);
				if (manager->IsVolatile()) continue;
				(void)manager->SetVolatile();
				(void)manager->TakeVolatileClean();
				t.bases[t.count]     = index * TRACKER_REGION_SIZE;
				t.regions[t.count++] = manager;
				t.kind_regions[kind]++;
			}
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (t.count != t.printed && now - t.print_at >= std::chrono::seconds(10)) {
		t.printed  = t.count;
		t.print_at = now;
		const double mib = static_cast<double>(TRACKER_REGION_SIZE) / (1024.0 * 1024.0);
		std::printf("[volatile] registered %zu regions (%.0f MiB): DefaultTagg %.0f MiB, ctx %.0f MiB\n", t.count,
		            static_cast<double>(t.count) * mib, static_cast<double>(t.kind_regions[0]) * mib,
		            static_cast<double>(t.kind_regions[1]) * mib);
		std::fflush(stdout);
	}
	const auto& config = GetVolatileConfig();
	auto&       st     = g_volatile_stats;
	if (config.mode == VolatileMode::WriteWatch) {
		// (v4, native memory) Exactly the pages written since the last call (by the game or the emulator), read and
		// reset per region: only those become CPU-dirty again (not the GPU-dirty ones, MarkVolatileDirty).
		thread_local std::vector<uint64_t> written;
		for (size_t i = 0; i < t.count; i++) {
			written.clear();
			if (!Libs::LibKernel::Memory::NativeWriteWatch(t.bases[i], TRACKER_REGION_SIZE, &CollectWrittenPage, &written,
			                                               true)) {
				// (Not readable: the whole region when an upload made it clean, as v3.)
				st.ww_fail++;
				RegionManager* manager = t.regions[i];
				if (!manager->VolatileCleanHint()) continue;
				std::scoped_lock lock(manager->lock);
				if (!manager->TakeVolatileClean()) continue;
				tagged_pages += manager->MarkVolatileDirty();
				tagged_regions++;
				continue;
			}
			if (written.empty()) continue;
			RegionManager*   manager = t.regions[i];
			std::scoped_lock lock(manager->lock);
			for (const uint64_t page_vaddr: written) {
				if (page_vaddr < t.bases[i] || page_vaddr >= t.bases[i] + TRACKER_REGION_SIZE) continue;
				const auto page = static_cast<size_t>((page_vaddr - t.bases[i]) / TRACKER_PAGE_SIZE);
				st.ww_pages += manager->MarkVolatileDirty(page, page + 1);
			}
			tagged_regions++;
		}
		return;
	}
	bool full = true;
	if (config.mode == VolatileMode::Frame) {
		// (v4) Every region an upload made clean: once per game frame (or every 16 ms when the frame is not known),
		// and every KYTY_VOLATILE_EVERY calls inside a frame.
		t.calls_in_frame++;
		full = new_frame || (config.every != 0 && t.calls_in_frame >= config.every) ||
		       (!frame_known && now - t.last_full >= std::chrono::milliseconds(16));
		// The ctx blocks of the DCBs submitted since the last call (a context's commands and embedded data are written
		// before its submit): those blocks whole, at once.
		uint64_t pairs[2 * 64];
		for (;;) {
			const size_t n = Loader::DemonsSoulsMemMap::SubmitsSince(&t.submit_cursor, pairs, 64);
			for (size_t k = 0; k < n; k++) {
				uint64_t begin = pairs[2 * k], end = pairs[2 * k + 1];
				uint64_t block_begin = 0, block_end = 0;
				if (Loader::DemonsSoulsMemMap::CtxBlockOf(begin, &block_begin, &block_end)) {
					begin = std::min(begin, block_begin);
					end   = std::max(end, block_end);
				}
				if (end <= begin || end - begin > (uint64_t {64} << 20u)) continue;
				st.ctx_submits++;
				st.ctx_pages += RedirtyVolatile(begin & ~(TRACKER_PAGE_SIZE - 1), end - (begin & ~(TRACKER_PAGE_SIZE - 1)));
			}
			if (n < 64) break;
		}
		if (!full) return;
		t.calls_in_frame = 0;
		t.last_full      = now;
		st.frame_redirties++;
	}
	// Re-dirty only the regions an upload (or the GPU) made some page clean in since the last call: every page not
	// GPU-dirty is CPU-dirty again (bits only; volatile pages are never protected, so no syscall).
	for (size_t i = 0; i < t.count; i++) {
		RegionManager* manager = t.regions[i];
		if (!manager->VolatileCleanHint()) continue;
		std::scoped_lock lock(manager->lock);
		if (!manager->TakeVolatileClean()) continue;
		tagged_pages += manager->MarkVolatileDirty();
		tagged_regions++;
	}
}

void MemoryTracker::VolatileRedirty() noexcept {
	auto* tracker = g_tracker.load(std::memory_order_acquire);
	if (tracker == nullptr) return; // (off)
	uint64_t ring_pages = 0, manual_pages = 0;
	bool new_frame = false, frame_known = false;
	tracker->StepGameRings(ring_pages, new_frame, frame_known);
	uint64_t tagged_pages = 0, tagged_regions = 0;
	tracker->StepTagged(new_frame, frame_known, tagged_pages, tagged_regions);
	const size_t count = g_volatile_count.load(std::memory_order_acquire);
	for (size_t i = 0; i < count; i++) {
		auto* manager = g_volatile_regions[i].load(std::memory_order_acquire);
		if (manager == nullptr) continue;
		std::scoped_lock lock(manager->lock);
		manual_pages += manager->MarkVolatileDirty();
	}
	auto& st = g_volatile_stats;
	st.calls++;
	st.ring_pages += ring_pages;
	st.manual_pages += manual_pages;
	st.tagged_pages += tagged_pages;
	st.tagged_regions += tagged_regions;
	const auto now = std::chrono::steady_clock::now();
	if (now - st.report >= std::chrono::seconds(5)) {
		const auto mib = [](uint64_t pages) { return static_cast<double>(pages * TRACKER_PAGE_SIZE) / (1024.0 * 1024.0); };
		std::printf("[volatile] %.1f s: %llu calls, %llu game frames, rings %llu pages (%.1f MiB) dirty again, "
		            "zone %.1f MiB/call, %llu whole-ring; manual %llu pages (%.1f MiB) in %zu regions\n",
		            std::chrono::duration<double>(now - st.report).count(), static_cast<unsigned long long>(st.calls),
		            static_cast<unsigned long long>(st.frames), static_cast<unsigned long long>(st.ring_pages),
		            mib(st.ring_pages),
		            st.calls ? static_cast<double>(st.zone_bytes) / static_cast<double>(st.calls) / (1024.0 * 1024.0) : 0.0,
		            static_cast<unsigned long long>(st.full), static_cast<unsigned long long>(st.manual_pages),
		            mib(st.manual_pages), count);
		std::printf("[volatile] %.1f s: alloc/ctx %llu pages (%.1f MiB) dirty again, %llu region re-dirties, %zu "
		            "regions\n",
		            std::chrono::duration<double>(now - st.report).count(), static_cast<unsigned long long>(st.tagged_pages),
		            mib(st.tagged_pages), static_cast<unsigned long long>(st.tagged_regions), g_tagged.count);
		// (v4) What the chosen mode re-dirtied: full passes (frame mode), the ctx blocks of submitted DCBs, the pages
		// the write watch reported (ww mode) and the regions it could not read.
		std::printf("[volatile] %.1f s: v4 %llu full passes, ctx %llu submits %.1f MiB dirty again, write watch %.1f "
		            "MiB (%llu unreadable)\n",
		            std::chrono::duration<double>(now - st.report).count(),
		            static_cast<unsigned long long>(st.frame_redirties), static_cast<unsigned long long>(st.ctx_submits),
		            mib(st.ctx_pages), mib(st.ww_pages), static_cast<unsigned long long>(st.ww_fail));
		std::fflush(stdout);
		const auto report = now;
		st                = VolatileStats {};
		st.report         = report;
	}
}

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto end   = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	// KYTY_VOLATILE: a region over a KYTY_VOLATILE_RANGES range is volatile from its creation (not published yet).
	if (const auto& config = GetVolatileConfig(); config.on) {
		const uint64_t base = index * TRACKER_REGION_SIZE;
		for (const auto& range: config.ranges) {
			if (range.address >= base + TRACKER_REGION_SIZE || range.End() <= base) continue;
			const size_t slot = g_volatile_count.load(std::memory_order_relaxed);
			if (slot >= VOLATILE_MAX) break; // (more than 16 GiB: the rest stays as it was)
			{
				std::scoped_lock region_lock(ptr->lock);
				ptr->SetVolatile();
			}
			g_volatile_regions[slot].store(ptr, std::memory_order_release);
			g_volatile_count.store(slot + 1, std::memory_order_release);
			// (No print here: this can run inside a fault handler. The [volatile] line counts the regions.)
			break;
		}
	}
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

// The bit queries read under the region lock's sequence (TrackingSpinLock::ReadShared): every
// change of the bits holds the lock.
template <typename Query>
static bool QueryRegion(RegionManager* manager, Query&& query) {
	return manager->lock.ReadShared(query);
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return QueryRegion(manager, [&] { return manager->IsModified<DirtySource::Cpu>(offset, bytes); });
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return QueryRegion(manager, [&] { return manager->IsModified<DirtySource::Gpu>(offset, bytes); });
	});
}

bool MemoryTracker::IsRegionFullyCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	// (Missing regions were never tracked: all CPU-dirty.)
	return !Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return !QueryRegion(manager, [&] { return manager->IsModified<DirtySource::Cpu, true>(offset, bytes); });
	});
}

bool MemoryTracker::IsRegionFullyGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	// Missing regions start CPU-dirty, so they must also participate in this test.
	return !Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return !QueryRegion(manager, [&] { return manager->IsModified<DirtySource::Gpu, true>(offset, bytes); });
	});
}

bool MemoryTracker::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size,
                                                std::unique_lock<TrackingSpinLock>* handoff) noexcept {
	CheckNotInUploadCallback();
	if (!GuestRange {begin, size}.Valid() || size <= TRACKER_PAGE_SIZE ||
	    begin % TRACKER_PAGE_SIZE || size % TRACKER_PAGE_SIZE || fault < begin ||
	    fault - begin >= size ||
	    begin / TRACKER_REGION_SIZE != (begin + size - 1) / TRACKER_REGION_SIZE)
		return false;
	auto* manager = m_regions[begin / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
	if (!manager) return false;
	std::scoped_lock lock(manager->lock);
	if (handoff != nullptr) handoff->unlock();
	if (manager->IsModified<DirtySource::Gpu>(begin - manager->GetCpuAddr(), size)) return false;
	// The GPU ownership check and permission change share one region lock.
	manager->ChangeState<DirtySource::Cpu, true>(begin, size);
	LiveCounters::Add(LiveCounters::WindowPages, size / TRACKER_PAGE_SIZE);
	LiveCounters::AddGranule(begin, LiveCounters::GWindowPages, size / TRACKER_PAGE_SIZE);
	return true;
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

bool MemoryTracker::MarkRegionAsCpuDirtyKeepProtection(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	if (IsRegionGpuModified(vaddr, size)) return false;
	// (Untracked regions are all CPU-dirty; a page the GPU marks meanwhile is the GPU thread's, which calls this.)
	return !Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return !manager->MarkCpuDirtyKeepProtection(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::NoteHostWrite(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t, uint64_t) { manager->NoteHostWrite(); });
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UntrackMemory(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	std::vector<RegionManager*> managers;
	managers.reserve((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) /
	                 TRACKER_REGION_SIZE);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
		managers.push_back(manager);
	});

	std::vector<std::unique_lock<TrackingSpinLock>> locks;
	locks.reserve(managers.size());
	for (auto* manager: managers) {
		locks.emplace_back(manager->lock);
	}
	if (Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	    })) {
		EXIT("cannot untrack GPU-dirty memory\n");
	}
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
		manager->ForgetNoWatch(offset, bytes); // (policy1; nothing without KYTY_DS_NEIGHBOR_SPLIT)
	});
}

} // namespace Libs::Graphics
