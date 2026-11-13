#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_UPLOADHEAP_H_

// KYTY_NATIVE_MEM=3: guest memory the engine fills with the CPU every frame for the GPU (its per-frame upload arenas,
// found at run time by loader/demonsSoulsNativeMem.cpp) is tracked as a native port's upload heap: its pages are
// never write-protected for the buffer cache and never made clean; every GPU use copies what it reads, as a native
// renderer copies from its upload heap. A CPU write there no longer faults, is never missed (the pages stay CPU-dirty,
// so every use synchronizes them), and costs no protection change.
//
// Correctness does not depend on which memory is marked: a marked page is only ever "always CPU-dirty". What is marked
// decides the cost (a page marked that is rarely written is copied at each use instead of being proven clean).
// Marks are only ever added; nothing here frees, decommits or moves memory.
//
// Within one observation epoch of the GPU thread (PageManager::ObservationEpoch: until the next submission, label or
// guest command the guest could learn GPU progress from) a kept page already copied is not copied again: the guest
// cannot write memory that work reads in between without racing its own GPU work. A CPU write the trackers see
// (a fault through another watcher, a host write) drops that.

#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::UploadHeap {

inline constexpr uint64_t GranuleBits = 16; // 64 KiB: 64 granules in a 4 MiB tracker region, one word each
inline constexpr uint64_t Regions     = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
static_assert(TRACKER_REGION_SIZE >> GranuleBits == 64);

inline std::atomic<uint32_t> g_on {0};
inline std::atomic<uint64_t> g_granules[Regions] = {};
// Statistics (relaxed): pages kept CPU-dirty after a copy, kept pages whose copy the epoch rule skipped, kept pages
// handed to a GPU write, and pages marked.
inline std::atomic<uint64_t> g_kept_pages {0}, g_deduped_pages {0}, g_gpu_taken_pages {0}, g_marked_bytes {0};

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
