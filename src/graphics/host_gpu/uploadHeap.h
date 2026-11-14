#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_

// KYTY_NATIVE_MEM=3: guest memory the engine fills with the CPU every frame for the GPU (its per-frame upload arenas
// and the CPU heap, found at run time by loader/demonsSoulsNativeMem.cpp) is marked here. In marked memory a HOT page -
// one the guest rewrites every frame (it was dirty at uploads of two frames at most HotFrames apart) - is kept: after
// its copy it stays CPU-dirty and writable instead of write-protected, so the guest's next write costs no fault and no
// protection change. A kept page is copied at its uses, at most once per publish epoch (BdaDirtyRegions::Publish: a
// submission taken, a WAIT_REG_MEM satisfied, a speculation committed - guest writes reach GPU work only after one).
// It is tracked again (clean, write-watched) when it is copied again sooner than its write period (in a second publish
// epoch of a frame, or a later frame before the guest rewrites it: tracking would copy it once; it is not kept again for
// CoolFrames frames) or after KeepFrames frames (still hot, it faults once and is kept
// again). Never more copies than tracking makes, beyond a bounded learning cost; correctness never depends on what is
// marked: a kept page is only "CPU-dirty, copied at every use of a new publish epoch".
//
// Marks are only ever added; nothing here frees, decommits or moves memory.

#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::UploadHeap {

inline constexpr uint64_t GranuleBits = 16; // 64 KiB: 64 granules in a 4 MiB tracker region, one word each
inline constexpr uint64_t Regions     = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
static_assert(TRACKER_REGION_SIZE >> GranuleBits == 64);

inline std::atomic<uint32_t> g_on {0};
// Publish epochs (BdaDirtyRegions::Publish: a submission taken, a WAIT_REG_MEM satisfied, a speculation committed - the
// points after which guest writes reach GPU work) and frames (FrameBoundary submissions), counted by the GPU thread.
inline std::atomic<uint64_t> g_publish {1};
inline std::atomic<uint64_t> g_frame {1};
// A marked page is kept once it was dirty at uploads of two different frames at most HotFrames apart (the engine's
// per-frame memory is rewritten every frame of its ring); one copied again sooner than that period (read in two publish
// epochs of a frame, or in two frames of one write period) is tracked again, and not kept for CoolFrames frames.
// A kept page is tracked again after KeepFrames frames (at its next copy): one still written every frame faults once and
// is kept again at once; one the guest stopped writing is copied at most KeepFrames more times.
inline constexpr uint32_t HotFrames = 4, CoolFrames = 64, KeepFrames = 8;
struct PageInfo {
	uint32_t dirty_frame     = 0; // frame + 1 of its last copy as a tracked dirty page (0: none)
	uint32_t kept_copy_frame = 0; // frame of its last copy while kept
	uint32_t no_keep_before  = 0; // frame from which it may be kept again
	uint32_t kept_since      = 0; // frame it was kept
	uint32_t period          = 1; // frames between its last two tracked copies when it was kept (its write period)
	// KYTY_ARENA_NATIVE: the frame of its kept copies' count, that count, and the frames it was copied in while kept.
	uint32_t copies_frame = 0;
	uint16_t frame_copies = 0, use_frames = 0;
};
inline std::atomic<uint64_t> g_granules[Regions] = {};
// Statistics (relaxed): pages kept CPU-dirty after a copy, kept pages whose copy the epoch rule skipped, kept pages
// handed to a GPU write, and pages marked.
inline std::atomic<uint64_t> g_kept_pages {0}, g_deduped_pages {0}, g_gpu_taken_pages {0}, g_marked_bytes {0};
inline std::atomic<uint64_t> g_kept_copies {0}, g_promoted_pages {0}, g_demoted_pages {0};

// A publish epoch (BdaDirtyRegions::Publish calls it; the tests too).
inline void Publish() noexcept { g_publish.fetch_add(1, std::memory_order_acq_rel); }

[[nodiscard]] inline uint64_t RegionGranules(uint64_t region_base) noexcept {
	if (g_on.load(std::memory_order_relaxed) == 0) return 0;
	const uint64_t region = region_base / TRACKER_REGION_SIZE;
	return region < Regions ? g_granules[region].load(std::memory_order_acquire) : 0;
}

// Marks the 64 KiB granules wholly inside [vaddr, vaddr + size).
inline void Mark(uint64_t vaddr, uint64_t size) noexcept {
	constexpr uint64_t Granule = uint64_t {1} << GranuleBits;
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) return;
	uint64_t begin = (vaddr + Granule - 1) & ~(Granule - 1);
	const uint64_t end = (vaddr + size) & ~(Granule - 1);
	for (; begin < end; begin += Granule) {
		const uint64_t region = begin / TRACKER_REGION_SIZE;
		const uint64_t bit    = uint64_t {1} << ((begin % TRACKER_REGION_SIZE) >> GranuleBits);
		if ((g_granules[region].fetch_or(bit, std::memory_order_acq_rel) & bit) == 0)
			g_marked_bytes.fetch_add(Granule, std::memory_order_relaxed);
	}
}

// KYTY_ARENA_NATIVE=1 (off by default; needs KYTY_NATIVE_MEM=3): the engine's GPU frame arena (the GPU "Default
// TaggedArena", parent of RenderSingleFrameGPU / GameToGPU) treated as a native port's upload heap, and the marked
// heaps' multi-epoch rule relaxed to a copy budget. Static facts (01.007.000): CBuffer::Create 0x909460 routes flag 0x2
// (0x90948a: shl 0xb -> 0x1000) to RenderSingleFrameGPU 0x2f643f0 (0x909526) only when initial data is given
// (0x9094f3), and copies that data with the CPU at once (memcpy 0x90954b): the arena's buffers are CPU-written, GPU-read.
// CPU-side / readback buffers (flag 0x1) go to the CPU context 0x2ea0d00 (0x9094e4 / 0x9094fd). Field logs (all 161
// census rows of the arena): 0 CPU read faults in it.
// What changes (correctness never depends on it: a kept page is CPU-dirty and copied at every use of a new publish
// epoch, exactly as above):
// 1. In the frame arena a CPU-dirty page is kept at its first upload (the arena's 1 MiB chunks rotate through hundreds
//    of MiB, so its pages are rewritten many frames apart and never met the HotFrames rule), and KeepFrames counts the
//    frames it was copied in, not frames passed (a page rewritten at each reuse of its chunk is kept across reuses).
// 2. In every marked heap (the frame arena and CPUMemory's HeapAllocator) a kept page read in several publish epochs of
//    a frame stays kept up to ArenaCopiesPerFrame copies in that frame (a 4 KiB copy against a fault and two protection
//    calls), instead of being tracked again at its second.
// 3. A CPU access to GPU-written frame-arena memory reads back at most ArenaWindow around it (not 512 KiB):
//    gpu-only-writes.h.
// Nothing here frees, moves or decommits memory.
inline constexpr uint32_t ArenaCopiesPerFrame = 4;
inline constexpr uint64_t ArenaWindow         = uint64_t {64} << 10u;
inline bool ArenaNativeRequested() noexcept {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_ARENA_NATIVE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}
// Set by the census once the level-3 checks passed and KYTY_ARENA_NATIVE=1.
inline std::atomic<uint32_t> g_arena_on {0};
inline std::atomic<uint64_t> g_arena_granules[Regions] = {};
inline std::atomic<uint64_t> g_arena_bytes {0}, g_arena_promoted {0}, g_budget_kept {0}, g_budget_demoted {0};

[[nodiscard]] inline bool ArenaOn() noexcept { return g_arena_on.load(std::memory_order_relaxed) != 0; }

[[nodiscard]] inline uint64_t RegionArenaGranules(uint64_t region_base) noexcept {
	if (!ArenaOn()) return 0;
	const uint64_t region = region_base / TRACKER_REGION_SIZE;
	return region < Regions ? g_arena_granules[region].load(std::memory_order_acquire) : 0;
}

// Whether all of [vaddr, vaddr + size) is frame-arena memory (whole 64 KiB granules marked).
[[nodiscard]] inline bool InArena(uint64_t vaddr, uint64_t size) noexcept {
	if (!ArenaOn() || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) return false;
	for (uint64_t g = vaddr >> GranuleBits, last = (vaddr + size - 1) >> GranuleBits; g <= last; ++g) {
		const uint64_t at = g << GranuleBits;
		const uint64_t bit = uint64_t {1} << ((at % TRACKER_REGION_SIZE) >> GranuleBits);
		if ((g_arena_granules[at / TRACKER_REGION_SIZE].load(std::memory_order_acquire) & bit) == 0) return false;
	}
	return true;
}

// Marks the frame arena's 64 KiB granules wholly inside [vaddr, vaddr + size) (also as an upload heap: Mark). Only added.
inline void MarkArena(uint64_t vaddr, uint64_t size) noexcept {
	constexpr uint64_t Granule = uint64_t {1} << GranuleBits;
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) return;
	Mark(vaddr, size);
	uint64_t       begin = (vaddr + Granule - 1) & ~(Granule - 1);
	const uint64_t end   = (vaddr + size) & ~(Granule - 1);
	for (; begin < end; begin += Granule) {
		const uint64_t bit = uint64_t {1} << ((begin % TRACKER_REGION_SIZE) >> GranuleBits);
		if ((g_arena_granules[begin / TRACKER_REGION_SIZE].fetch_or(bit, std::memory_order_acq_rel) & bit) == 0)
			g_arena_bytes.fetch_add(Granule, std::memory_order_relaxed);
	}
}

} // namespace Libs::Graphics::UploadHeap

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_
