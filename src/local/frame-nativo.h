#pragma once
// (FRAME-NATIVO) KYTY_FRAME_NATIVO (default 1; =0 exactly as before).
//
// Ghidra (eboot 01.007.000): TaggedFrameAllocator::Alloc 0x811f60 / reserve 0x811d50 ("TaggedFrameAllocator.cpp")
// keep 20 per-thread cursors (this+0xb8 + thread*0x30: cursor, bytes left, free list) plus one shared slot (+0x478,
// spinlock at [+0x80]); chunks come from the parent (+0x88, vtable +0xf8 allocate, +0xd0 granularity) and the leftovers
// go to a free list stored INSIDE the arena. There is no single "advanced offset" per frame to upload, and the arena is
// ordinary guest memory (the emulator's backing), not a block Vulkan can place at the game's address. So the native
// path is the frame policy (KYTY_DS_WRITE_FRAME, policy6): the pages a frame allocator hands out (hook on 0x811f60)
// stop being write-protected, and the lot submits / command buffer submits mark them written before the GPU reads them
// (no fault, no reprotect). KYTY_FRAME_NATIVO=1 turns that policy on (level 1, no stale check) when KYTY_DS_WRITE_FRAME
// is not set; an explicit KYTY_DS_WRITE_FRAME (=0 too) wins.
//
// Also a cheap counter of EVERY protection change (ProtectMappedUnlocked, any thread) by the memory map's label of its
// first page (64 KiB granules below 128 GiB, rebuilt every 5 s from DemonsSoulsMemMap::RangesOf), and a "[frame-nativo]"
// line every 30 s. The map is rebuilt in place: a call during the rebuild may count under the wrong label (us window).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace FrameNativo {

[[nodiscard]] inline bool Wanted() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_FRAME_NATIVO");
		return v != nullptr && v[0] == '1'; // apagado por defecto: se prueba PROT-MENOS solo
	}();
	return on;
}

enum Cls : uint8_t { COtros = 0, CDefaultTagg, CRing, CCtx, CCpuheap, CCpubkt, CGpupage, CCount };
inline const char* const kNames[CCount] = {"otros", "DefaultTagg", "ring", "ctx", "cpuheap", "cpubkt", "gpupage"};

constexpr uint64_t kGranShift = 16;                     // 64 KiB
constexpr uint64_t kLimit     = uint64_t {1} << 37u;    // 128 GiB of guest VA (the game's ranges are below)
constexpr size_t   kGranules  = static_cast<size_t>(kLimit >> kGranShift); // 2 Mi entries = 2 MiB

inline std::atomic<std::atomic<uint8_t>*> g_map {nullptr};
inline std::atomic<uint64_t>              g_calls[CCount] {};
inline std::atomic<uint64_t>              g_pages[CCount] {};
// (PROT-MENOS) Protection changes the hot pages spared, by label (local/prot-menos.h): each kept reprotect call also
// spares the write fault's unprotect that would follow it.
inline std::atomic<uint64_t>              g_avoided[CCount] {};

// (PROT-MENOS) The label of `vaddr` (COtros when unknown).
[[nodiscard]] inline Cls Classify(uint64_t vaddr) noexcept {
	std::atomic<uint8_t>* map = g_map.load(std::memory_order_acquire);
	if (map == nullptr || vaddr >= kLimit) return COtros;
	const uint8_t c = map[vaddr >> kGranShift].load(std::memory_order_relaxed);
	return c < CCount ? static_cast<Cls>(c) : COtros;
}
inline void NoteAvoided(uint64_t vaddr, uint64_t calls) noexcept {
	g_avoided[Classify(vaddr)].fetch_add(calls, std::memory_order_relaxed);
}

// Any thread, under the address space lock: two relaxed loads and two adds.
inline void NoteProtect(uint64_t vaddr, uint64_t size) noexcept {
	std::atomic<uint8_t>* map = g_map.load(std::memory_order_acquire);
	uint8_t               c   = COtros;
	if (map != nullptr && vaddr < kLimit) {
		c = map[vaddr >> kGranShift].load(std::memory_order_relaxed);
		if (c >= CCount) c = COtros;
	}
	g_calls[c].fetch_add(1, std::memory_order_relaxed);
	g_pages[c].fetch_add((size + 0xfffu) >> 12u, std::memory_order_relaxed);
}

// One thread (the write audit's 5 s loop). `ranges_of(labels, n, out, max)` = DemonsSoulsMemMap::RangesOf.
template <typename F>
void Refresh(F ranges_of) noexcept {
	std::atomic<uint8_t>* map = g_map.load(std::memory_order_relaxed);
	if (map == nullptr) {
		map = new (std::nothrow) std::atomic<uint8_t>[kGranules];
		if (map == nullptr) return;
		for (size_t i = 0; i < kGranules; ++i) map[i].store(0, std::memory_order_relaxed);
	}
	static uint64_t pairs[2 * 16384];
	// Bigger labels first, the small blocks last (a 64 KiB granule shared by two labels takes the later one).
	static constexpr Cls order[] = {CDefaultTagg, CGpupage, CCpuheap, CRing, CCtx, CCpubkt};
	bool any = false;
	for (size_t i = 0; i < kGranules; ++i) map[i].store(0, std::memory_order_relaxed);
	for (const Cls c: order) {
		const char* label = kNames[c];
		const size_t n = ranges_of(&label, 1, pairs, 16384);
		if (n == SIZE_MAX) continue;
		for (size_t r = 0; r < n; ++r) {
			const uint64_t b = pairs[2 * r], e = pairs[2 * r + 1];
			if (e <= b || b >= kLimit) continue;
			const uint64_t last = ((e < kLimit ? e : kLimit) - 1) >> kGranShift;
			for (uint64_t g = b >> kGranShift; g <= last; ++g) map[g].store(c, std::memory_order_relaxed);
			any = true;
		}
	}
	if (any) g_map.store(map, std::memory_order_release);
	else if (g_map.load(std::memory_order_relaxed) == nullptr) delete[] map;
}

} // namespace FrameNativo
