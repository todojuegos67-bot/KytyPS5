#pragma once
// (trust1) What the write audit (loader/demonsSoulsWriteAudit*: KYTY_DS_WRITE_AUDIT=1 or one of its policies) tells the
// render thread's trusted-depth registry (local/trusted-depth.h, KYTY_DS_TRUSTED_DEPTH). Only std types: the audit
// (any game thread) and the render thread both include it.
//
// THE AUDIT DOES NOT FILL IT YET: the diff that makes it (native/trust1/NOTAS.md, section "Diff para el audit") goes in
// the memory files, which this delivery does not touch. Until it is applied g_classify stays null: the registry acts
// on nothing and says so ("sin feed"), and the measurement falls back to the memory map's labels.
//
//   g_classify(address, size)   what the hooks know of a range (Class below). Set by the audit's Install, null while
//                               the registry is off. Takes the registry's shared lock: not for every draw.
//   g_generation[granule]       moves (+1) whenever a hook announces a write over a registered range, or a range is
//                               added, replaced, removed or swept (any kind): the trusted ranges' proof (Generation).
//   g_everything                moves when a range was too big to bump granule by granule: every proof fails.
//
// Ordering: a hook bumps after the game wrote (Create after its memcpy, the copy hooks after the copy) and before the
// game submits the command buffer that uses it; the render thread reads after taking that submission (the queue's lock).

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace DsTrustFeed {

enum Class : int {
	Other    = 0, // in no range the hooks announce every write of (unknown memory, GPU-written, stale per-use ranges)
	Static   = 1, // every byte in ranges the hooks announce every write of (created with data, textures, layers, the
	              // XPR section blocks once the diff registers them): what the registry trusts
	Frame    = 2, // per-frame memory (rings, TaggedArenas, small DCBs): new every frame
	Excluded = 3, // a byte in the 0x20-byte table 0x9f6800-0x9f6b20 rewrite (CBufRewritten) or a Bink block: never trusted
	PerUse   = 4, // wrapped (0x800) or linear copy source memory, fresh only while announced again at each use
	Classes  = 5
};
using ClassifyFn = int (*)(uint64_t address, uint64_t size) noexcept;
inline std::atomic<ClassifyFn> g_classify {nullptr};
[[nodiscard]] inline bool Live() noexcept {
	return g_classify.load(std::memory_order_acquire) != nullptr;
}

constexpr unsigned GranuleShift = 16; // 64 KiB
constexpr size_t   Granules     = size_t {1} << 16u;
constexpr uint64_t MaxBumpGranules = 4096; // (256 MiB: past it, everything)
inline std::atomic<uint32_t> g_generation[Granules] {};
inline std::atomic<uint32_t> g_everything {0};
[[nodiscard]] inline size_t GranuleIndex(uint64_t granule) noexcept {
	return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ull) >> 48u) & (Granules - 1);
}
// The audit (any thread): the range's proofs fail from now on.
inline void Bump(uint64_t address, uint64_t size) noexcept {
	if (size == 0) return;
	const uint64_t first = address >> GranuleShift, last = (address + size - 1) >> GranuleShift;
	if (last < first || last - first >= MaxBumpGranules) {
		g_everything.fetch_add(1, std::memory_order_release);
		return;
	}
	for (uint64_t granule = first; granule <= last; ++granule)
		g_generation[GranuleIndex(granule)].fetch_add(1, std::memory_order_release);
}
// The render thread: a value that moves whenever Bump touched the range (or g_everything moved).
[[nodiscard]] inline uint32_t Generation(uint64_t address, uint64_t size) noexcept {
	uint32_t sum = g_everything.load(std::memory_order_acquire) * 0x9E3779B1u;
	if (size == 0) return sum;
	const uint64_t first = address >> GranuleShift, last = (address + size - 1) >> GranuleShift;
	for (uint64_t granule = first; granule <= last && granule - first < MaxBumpGranules; ++granule)
		sum += g_generation[GranuleIndex(granule)].load(std::memory_order_acquire);
	return sum;
}

} // namespace DsTrustFeed
