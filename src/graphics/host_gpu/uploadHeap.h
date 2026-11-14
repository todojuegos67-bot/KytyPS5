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

} // namespace Libs::Graphics::UploadHeap

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_
