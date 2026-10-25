#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SHADOWPOOLTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SHADOWPOOLTRACE_H_

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShadowPoolTrace {

// KYTY_SHADOW_POOL_LOG: what the game does to its layered shadow pools (depth arrays of 6+ layers: the 1024x1024x80 spot
// pool, the 384x384x360 cube pool), a flip at a time, so a run log tells which pass leaves a shadow layer wrong.
// 1 (default): a pool's first 8 active flips, then one every 600 flips; 2: every flip it is used; 0: off.
// Per flip: depth-target binds (by load clear, HTile clear, depth write; distinct layers rendered), sampled and storage
// binds, clears, depth/colour switches, layers refreshed from guest memory (CPU-dirty), whole uploads.
inline int Level() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_SHADOW_POOL_LOG");
		return value == nullptr ? 1 : std::atoi(value);
	}();
	return level;
}

struct Pool {
	std::atomic<uint64_t> address {0};
	uint32_t              layers = 0;
	uint32_t              width  = 0;
	uint32_t              height = 0;
	std::atomic<uint32_t> depth_binds {0};
	std::atomic<uint32_t> load_clears {0};
	std::atomic<uint32_t> meta_clears {0};
	std::atomic<uint32_t> depth_writes {0};
	std::atomic<uint32_t> no_write_no_clear {0};
	std::atomic<uint32_t> samples {0};
	std::atomic<uint32_t> storage {0};
	std::atomic<uint32_t> clears {0};
	std::atomic<uint32_t> switches {0};
	std::atomic<uint32_t> refreshed_layers {0};
	std::atomic<uint32_t> whole_uploads {0};
	std::atomic<uint32_t> refresh_kept {0};    // KYTY_SHADOW_REFRESH_FIX: rendered layers kept (guest bytes unchanged)
	std::atomic<uint32_t> refresh_changed {0}; // ... rendered layers uploaded (guest bytes changed)
	std::atomic<uint64_t> rendered[8] {};  // bit per layer (first 512)
	uint32_t              first_layer = UINT32_MAX;
	uint32_t              last_layer  = 0;
	uint64_t              active_flips = 0;
	uint64_t              last_active  = 0;
};

inline Pool g_pools[8];

inline bool Candidate(uint32_t layers, uint32_t width, uint32_t height) {
	return layers >= 6 && width >= 128 && height >= 128 && Level() > 0;
}

// The pool of this image (made at its first event); nullptr when not one or the table is full.
inline Pool* Find(uint64_t address, uint32_t layers, uint32_t width, uint32_t height) {
	if (!Candidate(layers, width, height) || address == 0) return nullptr;
	for (auto& pool: g_pools) {
		if (pool.address.load(std::memory_order_relaxed) == address) return &pool;
	}
	for (auto& pool: g_pools) {
		uint64_t expected = 0;
		if (pool.address.compare_exchange_strong(expected, address)) {
			pool.layers = layers;
			pool.width  = width;
			pool.height = height;
			return &pool;
		}
	}
	// (Full: the pool of a level left long ago gives its entry; its counts start again.)
	Pool* oldest = &g_pools[0];
	for (auto& pool: g_pools) {
		if (pool.last_active < oldest->last_active) oldest = &pool;
	}
	oldest->address.store(address, std::memory_order_relaxed);
	oldest->layers       = layers;
	oldest->width        = width;
	oldest->height       = height;
	oldest->active_flips = 0;
	oldest->last_active  = 0;
	return oldest;
}

inline void DepthBind(Pool* pool, uint32_t base_layer, uint32_t layer_count, bool load_clear, bool meta_clear,
                      bool depth_write) {
	if (pool == nullptr) return;
	pool->depth_binds.fetch_add(1, std::memory_order_relaxed);
	if (load_clear) pool->load_clears.fetch_add(1, std::memory_order_relaxed);
	if (meta_clear) pool->meta_clears.fetch_add(1, std::memory_order_relaxed);
	if (depth_write) pool->depth_writes.fetch_add(1, std::memory_order_relaxed);
	if (!depth_write && !load_clear) pool->no_write_no_clear.fetch_add(1, std::memory_order_relaxed);
	for (uint32_t layer = base_layer; layer < base_layer + layer_count && layer < 512; ++layer) {
		pool->rendered[layer / 64].fetch_or(uint64_t {1} << (layer % 64), std::memory_order_relaxed);
	}
	if (base_layer < pool->first_layer) pool->first_layer = base_layer;
	if (base_layer + layer_count > pool->last_layer) pool->last_layer = base_layer + layer_count;
}

inline void Count(Pool* pool, std::atomic<uint32_t> Pool::* counter, uint32_t n = 1) {
	if (pool != nullptr) (pool->*counter).fetch_add(n, std::memory_order_relaxed);
}

// At each flip (the texture cache's collector): one line per pool used since the last flip, rate-limited.
inline void Flip(uint64_t flip) {
	const int level = Level();
	if (level <= 0) return;
	for (auto& pool: g_pools) {
		const auto address = pool.address.load(std::memory_order_relaxed);
		if (address == 0) continue;
		const uint32_t binds    = pool.depth_binds.exchange(0, std::memory_order_relaxed);
		const uint32_t clears_l = pool.load_clears.exchange(0, std::memory_order_relaxed);
		const uint32_t meta     = pool.meta_clears.exchange(0, std::memory_order_relaxed);
		const uint32_t writes   = pool.depth_writes.exchange(0, std::memory_order_relaxed);
		const uint32_t idle     = pool.no_write_no_clear.exchange(0, std::memory_order_relaxed);
		const uint32_t samples  = pool.samples.exchange(0, std::memory_order_relaxed);
		const uint32_t storage  = pool.storage.exchange(0, std::memory_order_relaxed);
		const uint32_t clears   = pool.clears.exchange(0, std::memory_order_relaxed);
		const uint32_t switches = pool.switches.exchange(0, std::memory_order_relaxed);
		const uint32_t refresh  = pool.refreshed_layers.exchange(0, std::memory_order_relaxed);
		const uint32_t whole    = pool.whole_uploads.exchange(0, std::memory_order_relaxed);
		const uint32_t kept     = pool.refresh_kept.exchange(0, std::memory_order_relaxed);
		const uint32_t changed  = pool.refresh_changed.exchange(0, std::memory_order_relaxed);
		uint32_t       distinct = 0;
		for (auto& bits: pool.rendered) distinct += static_cast<uint32_t>(std::popcount(bits.exchange(0, std::memory_order_relaxed)));
		const uint32_t first = pool.first_layer;
		const uint32_t last  = pool.last_layer;
		pool.first_layer     = UINT32_MAX;
		pool.last_layer      = 0;
		const bool write_activity = binds != 0 || storage != 0 || clears != 0 || switches != 0 || refresh != 0 || whole != 0 ||
		                            kept != 0;
		if (!write_activity) continue;
		const uint64_t n  = ++pool.active_flips;
		pool.last_active  = flip;
		// (Anything that may lose rendered layers is always reported, at most 64 times.)
		static std::atomic<uint32_t> risky_reports {0};
		const bool risky = (refresh != 0 || whole != 0 || kept != 0) && risky_reports.fetch_add(1, std::memory_order_relaxed) < 64;
		if (!(level >= 2 || n <= 8 || n % 600 == 0 || risky)) continue;
		std::printf("Shadow pool 0x%llx (%u layers %ux%u) flip %llu (active %llu): depth binds %u (load clear %u, htile "
		            "clear %u, depth write %u, neither %u; %u layers in %u..%u), sampled %u, storage %u, clears %u, "
		            "switches %u, layers refreshed from guest memory %u, whole uploads %u; rendered layers kept %u, "
		            "uploaded as changed %u (KYTY_SHADOW_POOL_LOG=0: off, 2: every flip)\n",
		            static_cast<unsigned long long>(address), pool.layers, pool.width, pool.height,
		            static_cast<unsigned long long>(flip), static_cast<unsigned long long>(n), binds, clears_l, meta,
		            writes, idle, distinct, first == UINT32_MAX ? 0 : first, last, samples, storage, clears, switches,
		            refresh, whole, kept, changed);
		std::fflush(stdout);
	}
}

} // namespace Libs::Graphics::ShadowPoolTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SHADOWPOOLTRACE_H_
