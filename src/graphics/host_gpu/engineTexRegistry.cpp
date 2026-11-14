// KYTY_ENGINE_TEX_REGISTRY: see engineTexRegistry.h.
#include "graphics/host_gpu/engineTexRegistry.h"

#include "graphics/host_gpu/hostMemory.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace Libs::Graphics::EngineTexRegistry {

namespace {

// TextureArrayManager (B-streaming.md section 3; 0xac4160 creates pools 0..10 through 0xac3f20, entries of 0x60 bytes
// at [mgr], holder at +0, the pool CTexture at [holder+0x20] (0xac45bd / 0xac45d9)).
constexpr uint64_t ArrayManagerGlobal = 0x2f684b0, PoolEntrySize = 0x60, HolderTexture = 0x20;
constexpr int      PoolCount          = 11;

struct State {
	std::atomic<bool>                      enabled {false};
	std::atomic<uint64_t>                  eboot {0};
	std::shared_mutex                      mutex;
	std::map<uint64_t, Entry>              by_base;  // disjoint [base, base + max(size, 1))
	std::unordered_map<uint64_t, uint64_t> by_owner; // CTexture* -> base
	std::atomic<uint64_t>                  inserted[static_cast<uint32_t>(Source::Count)] {};
	std::atomic<uint64_t>                  released {0}, replaced {0};
	std::atomic<uint64_t>                  counts[static_cast<uint32_t>(Count::Count)] {};
	std::atomic<uint64_t>                  largest_bytes {0};
	std::atomic<uint64_t>                  largest_shape {0}; // w:16 h:16 layers:16 levels:16
	std::atomic<int64_t>                   next_log {0};
	std::atomic<uint32_t>                  reports {0};
};

State& G() {
	static State* s = new State; // never destroyed: game threads may call in during shutdown
	return *s;
}

uint64_t Extent(const Entry& e) { return e.size != 0 ? e.size : 1u; }

uint64_t Load64(uint64_t address) {
	uint64_t v = 0;
	std::memcpy(&v, reinterpret_cast<const void*>(address), sizeof(v));
	return v;
}

} // namespace

bool Enabled() noexcept { return G().enabled.load(std::memory_order_relaxed); }
void SetEnabled(bool on) noexcept { G().enabled.store(on, std::memory_order_release); }
void SetEbootBase(uint64_t base) noexcept { G().eboot.store(base, std::memory_order_release); }

void Insert(const Entry& entry) noexcept {
	if (entry.base == 0) return;
	auto&          s     = G();
	const uint64_t begin = entry.base, end = entry.base + Extent(entry);
	{
		std::unique_lock lock(s.mutex);
		// An older entry over the same memory is stale (its release was not seen): the engine's newest texture wins.
		auto it = s.by_base.lower_bound(begin);
		if (it != s.by_base.begin()) {
			auto prev = std::prev(it);
			if (prev->first + Extent(prev->second) > begin) it = prev;
		}
		while (it != s.by_base.end() && it->first < end) {
			s.by_owner.erase(it->second.owner);
			it = s.by_base.erase(it);
			s.replaced.fetch_add(1, std::memory_order_relaxed);
		}
		if (auto o = s.by_owner.find(entry.owner); o != s.by_owner.end()) {
			s.by_base.erase(o->second);
			s.by_owner.erase(o);
		}
		s.by_base.emplace(begin, entry);
		s.by_owner[entry.owner] = begin;
	}
	s.inserted[static_cast<uint32_t>(entry.source)].fetch_add(1, std::memory_order_relaxed);
}

void Release(uint64_t owner, uint64_t base) noexcept {
	auto&            s = G();
	std::unique_lock lock(s.mutex);
	if (auto o = s.by_owner.find(owner); o != s.by_owner.end()) {
		s.by_base.erase(o->second);
		s.by_owner.erase(o);
		s.released.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (auto b = s.by_base.find(base); base != 0 && b != s.by_base.end()) {
		s.by_owner.erase(b->second.owner);
		s.by_base.erase(b);
		s.released.fetch_add(1, std::memory_order_relaxed);
	}
}

bool Find(uint64_t address, Entry* out) noexcept {
	auto&            s = G();
	std::shared_lock lock(s.mutex);
	auto             it = s.by_base.upper_bound(address);
	if (it == s.by_base.begin()) return false;
	--it;
	if (address - it->first >= Extent(it->second)) return false;
	if (out != nullptr) *out = it->second;
	return true;
}

int16_t PoolOf(uint64_t owner) noexcept {
	const uint64_t eboot = G().eboot.load(std::memory_order_acquire);
	if (eboot == 0 || owner == 0) return -1;
	const uint64_t global = eboot + ArrayManagerGlobal;
	if (!HostMemoryRangeIsReadable(global, 8)) return -1;
	const uint64_t mgr = Load64(global);
	if (mgr == 0 || !HostMemoryRangeIsReadable(mgr, 8)) return -1;
	const uint64_t pools = Load64(mgr);
	if (pools == 0 || !HostMemoryRangeIsReadable(pools, PoolEntrySize * PoolCount)) return -1;
	for (int id = 0; id < PoolCount; id++) {
		const uint64_t holder = Load64(pools + PoolEntrySize * static_cast<uint64_t>(id));
		if (holder == 0 || !HostMemoryRangeIsReadable(holder + HolderTexture, 8)) continue;
		if (Load64(holder + HolderTexture) == owner) return static_cast<int16_t>(id);
	}
	return -1;
}

bool Resolve(uint32_t fields[8], Entry* entry) noexcept {
	const uint64_t base = ((fields[0] | (static_cast<uint64_t>(fields[1]) << 32u)) & 0xFFFFFFFFFFull) << 8u;
	if (base == 0 || !Find(base, entry)) {
		Note(Count::Miss);
		return false;
	}
	Note(Count::Hit);
	if (entry->base == base) {
		// dword1: MIN_LOD (bits 8..19); dword3: BASE_LEVEL / LAST_LEVEL (bits 12..19): the engine streams through them.
		const bool same = fields[0] == entry->tsharp[0] &&
		                  (fields[1] & ~0x000FFF00u) == (entry->tsharp[1] & ~0x000FFF00u) &&
		                  fields[2] == entry->tsharp[2] &&
		                  (fields[3] & ~0x000FF000u) == (entry->tsharp[3] & ~0x000FF000u);
		const bool upper_differs = std::memcmp(fields + 4, entry->tsharp + 4, 16) != 0;
		// A layer view the engine made of this texture (0x8ff4e0 copies the 32-byte T# and rewrites only DEPTH /
		// BASE_ARRAY in word 4, same base): a layer range inside the engine's own with words 5..7 unchanged is the
		// game's view and is kept (adopting the engine's words would reset its BASE_ARRAY to layer 0).
		const uint32_t own_last = entry->tsharp[4] & 0x1fffu, last = fields[4] & 0x1fffu,
		               first = (fields[4] >> 16u) & 0x1fffu;
		const bool engine_view = first <= last && last <= own_last && std::memcmp(fields + 5, entry->tsharp + 5, 12) == 0;
		if (same && upper_differs && !engine_view) {
			Report("engine T# words 4..7 used", base, fields, *entry, 0, 0);
			std::memcpy(fields + 4, entry->tsharp + 4, 16);
			Note(Count::Adopted);
		}
	}
	if (entry->pool_id < 0 && entry->layers > 1) {
		entry->pool_id = PoolOf(entry->owner);
		if (entry->pool_id >= 0) {
			auto&            s = G();
			std::unique_lock lock(s.mutex);
			if (auto it = s.by_base.find(entry->base); it != s.by_base.end() && it->second.owner == entry->owner)
				it->second.pool_id = entry->pool_id;
		}
	}
	return true;
}

void Note(Count c) noexcept { G().counts[static_cast<uint32_t>(c)].fetch_add(1, std::memory_order_relaxed); }

void NoteImage(uint64_t bytes, uint32_t width, uint32_t height, uint32_t layers, uint32_t levels) noexcept {
	auto&    s   = G();
	uint64_t cur = s.largest_bytes.load(std::memory_order_relaxed);
	while (bytes > cur) {
		if (s.largest_bytes.compare_exchange_weak(cur, bytes, std::memory_order_relaxed)) {
			s.largest_shape.store((uint64_t {width & 0xffffu} << 48u) | (uint64_t {height & 0xffffu} << 32u) |
			                          (uint64_t {layers & 0xffffu} << 16u) | uint64_t {levels & 0xffffu},
			                      std::memory_order_relaxed);
			break;
		}
	}
}

void Report(const char* what, uint64_t address, const uint32_t d[8], const Entry& e, uint32_t view_layers,
            uint32_t base_layer) noexcept {
	if (G().reports.fetch_add(1, std::memory_order_relaxed) >= 16) return;
	std::printf("EngineTex: %s at 0x%llx (view %u layers from %u): T# %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x; engine "
	            "texture 0x%llx size 0x%llx %ux%u depth %u layers %u mips %u fmt %u flags 0x%x pool %d src %u owner 0x%llx "
	            "T# %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	            what, static_cast<unsigned long long>(address), view_layers, base_layer, d[0], d[1], d[2], d[3], d[4], d[5],
	            d[6], d[7], static_cast<unsigned long long>(e.base), static_cast<unsigned long long>(e.size), e.width,
	            e.height, e.depth, e.layers, e.mips, e.engine_format, e.flags, e.pool_id, static_cast<uint32_t>(e.source),
	            static_cast<unsigned long long>(e.owner), e.tsharp[0], e.tsharp[1], e.tsharp[2], e.tsharp[3], e.tsharp[4],
	            e.tsharp[5], e.tsharp[6], e.tsharp[7]);
	std::fflush(stdout);
}

void MaybeLog() noexcept {
	if (!Enabled()) return;
	auto&         s   = G();
	const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t due = s.next_log.load(std::memory_order_relaxed);
	if (now < due || !s.next_log.compare_exchange_strong(due, now + 30)) return;
	size_t entries = 0;
	{
		std::shared_lock lock(s.mutex);
		entries = s.by_base.size();
	}
	const auto     count = [&](Count c) { return static_cast<unsigned long long>(s.counts[static_cast<uint32_t>(c)].load()); };
	const uint64_t shape = s.largest_shape.load();
	std::printf("EngineTex: %zu engine textures (made: factory %llu, streamed %llu, create %llu; released %llu, replaced "
	            "%llu); resolves: hits %llu, misses %llu, engine T# words used %llu, clamps %llu, conflicts %llu; largest "
	            "image %llu MiB (%ux%u, %u layers, %u levels)\n",
	            entries, static_cast<unsigned long long>(s.inserted[0].load()),
	            static_cast<unsigned long long>(s.inserted[1].load()), static_cast<unsigned long long>(s.inserted[2].load()),
	            static_cast<unsigned long long>(s.released.load()), static_cast<unsigned long long>(s.replaced.load()),
	            count(Count::Hit), count(Count::Miss), count(Count::Adopted), count(Count::Clamp), count(Count::Conflict),
	            static_cast<unsigned long long>(s.largest_bytes.load() >> 20u), static_cast<uint32_t>(shape >> 48u),
	            static_cast<uint32_t>((shape >> 32u) & 0xffffu), static_cast<uint32_t>((shape >> 16u) & 0xffffu),
	            static_cast<uint32_t>(shape & 0xffffu));
	std::fflush(stdout);
}

} // namespace Libs::Graphics::EngineTexRegistry
