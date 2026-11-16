#pragma once
// (audit1) KYTY_DS_WRITE_AUDIT: the registry of the ranges the game's hooks announce (loader/demonsSoulsWriteAudit.cpp).
// Plain C++ (no emulator headers) so that it can be run in a test.
//
// Disjoint [begin, end) ranges in a map under a shared mutex: the hooks (game threads) take it exclusively for one
// map operation; the fault path and the reports take it shared for one lookup. Nothing done under it touches guest
// memory, so a thread holding it never faults into the tracking (which would ask for it again).
// A counting filter of the ranges' first pages lets the free hooks (every free of the game) leave without the lock
// when the pointer cannot be a registered range's start.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace Loader::DemonsSoulsWriteAuditRegistry {

enum Kind : uint8_t {
	CBufData = 0, // CBuffer::Create with initial data, persistent memory (its memcpy seen)
	CBufGpu,      // CBuffer with flags 4 or 1 and no data seen: the GPU writes it
	CBufWrap,     // CBuffer with flag 0x800: wraps memory it does not own (meshes, pools)
	CBufPlain,    // any other persistent CBuffer (no data seen, no flag 4/1)
	TexStream,    // a streamed mip chunk (SetCurrentStreamingChunk's memcpy)
	TexLoad,      // a texture read from its file (FTextureFactory)
	// (policy1)
	TexLayer,      // a texture array's layer filled by the CPU at load (0x8fedc0: read at 0x8fef15 / converted at 0x8feed1)
	Bink,          // a Bink video's GPU block (the CPU writes its input buffers before 0x9808a0)
	CBufRewritten, // the 0x20-byte stream table made at 0x9f67d7, rewritten in place by 0x9f6800..0x9f6b20: no hook
	LinearSrc,     // the source of a linear copy outside the frame arenas and the registry (announced at each copy)
	KindCount
};
// The kinds a hook announces every CPU write of (bit = 1 << kind). `level` 1: buffers created with their data and the
// textures; 2: also the memory a CBuffer only wraps (0x800) and the linear copies' sources, whose announcement is the
// wrap / the copy itself (the owner writes them through its own pointer between two of them).
[[nodiscard]] constexpr uint32_t TrustedKinds(int level) noexcept {
	uint32_t mask = 0;
	if (level >= 1) mask |= (1u << CBufData) | (1u << TexStream) | (1u << TexLoad) | (1u << TexLayer);
	if (level >= 2) mask |= (1u << CBufWrap) | (1u << LinearSrc);
	return mask;
}
// (bit = 1 << kind) Announced at each use, not written through hooks: fresh only for a while after their last use.
constexpr uint32_t PerUseKinds = (1u << CBufWrap) | (1u << LinearSrc);
inline const char* KindName(uint8_t kind) {
	static constexpr const char* Names[KindCount] = {"cbuf-datos", "cbuf-gpu", "cbuf-envuelto", "cbuf-sin-datos", "tex-stream", "tex-carga",
	                                                    "tex-capa",   "bink",     "cbuf-reescrito", "linearcopy-origen"};
	return kind < KindCount ? Names[kind] : "?";
}

struct Entry {
	uint64_t end     = 0;
	uint32_t flags   = 0; // (CBuffer +0x18)
	uint32_t format  = 0; // (CBuffer +0x1c)
	uint8_t  kind    = 0;
	bool     pending = false; // CBufData registered by Create's memcpy, not yet confirmed by the end of Create
	// (policy1) The block whose free ends this range when it is not the range's own start (a streamed texture's
	// virtual block, CTexture +0x98: its chunks are registered where they are copied, inside it). 0: none.
	uint64_t owner = 0;
	// (policy1) Changed under the shared lock (atomics): how many times a hook announced the range (a Create over
	// the same memory, a linear copy from it, a write a hook saw), the wrappers alive over it, and when it was last
	// announced (Registry::Now's clock).
	std::atomic<uint32_t> generation {0};
	std::atomic<int32_t>  live {0};
	std::atomic<uint32_t> stamp {0};
};

enum class Where : uint8_t { None, Inside, Neighbor };
struct Hit {
	Where    where = Where::None;
	uint64_t begin = 0, end = 0; // the range (Inside) or the range sharing the page (Neighbor)
	uint32_t flags = 0, format = 0;
	uint8_t  kind = 0;
	// (policy1)
	uint32_t generation = 0, stamp = 0;
	int32_t  live = 0;
	bool     pending = false;
};

class Registry {
public:
	static constexpr uint64_t PageSize   = 4096;
	static constexpr size_t   FilterBits = 20;

	Registry(): m_filter(new std::atomic<uint16_t>[size_t {1} << FilterBits]), m_granules(new std::atomic<uint64_t>[GranuleWords]) {
		for (size_t i = 0; i < (size_t {1} << FilterBits); ++i) m_filter[i].store(0, std::memory_order_relaxed);
		for (size_t i = 0; i < GranuleWords; ++i) m_granules[i].store(0, std::memory_order_relaxed);
	}

	// (policy1) Lock-free: false when no range was ever registered in the 64 KiB granules of [address, address + size)
	// (bits are never cleared: true may be a range long gone). For every memcpy of the game.
	static constexpr uint64_t GranuleShift = 16;
	static constexpr size_t   GranuleWords = (uint64_t {1} << (40u - GranuleShift)) / 64;
	[[nodiscard]] bool MayContain(uint64_t address, uint64_t size) const noexcept {
		if (size == 0 || address + size < address) return false;
		const uint64_t first = address >> GranuleShift, last = (address + size - 1) >> GranuleShift;
		if (last >= GranuleWords * 64) return false;
		if (last - first >= 64) return true;
		for (uint64_t granule = first; granule <= last; ++granule)
			if ((m_granules[granule / 64].load(std::memory_order_relaxed) >> (granule % 64) & 1u) != 0) return true;
		return false;
	}

	// Lock-free: false when no registered range starts in `address`'s page (true may be a hash collision).
	[[nodiscard]] bool MayBeginAt(uint64_t address) const noexcept {
		return m_filter[FilterIndex(address)].load(std::memory_order_relaxed) != 0;
	}

	// Registers [begin, begin + size). Ranges it overlaps are taken out first (a new occupant of the memory); returns
	// how many. `inside_keeps`: when the new range lies whole inside one registered range, that one stays and the new
	// one is not added (a wrapper over part of registered memory); *nested is set then.
	// (policy1) `owner`: see Entry::owner. `live`: a wrapper is alive over it (Entry::live starts at 1).
	size_t Add(uint64_t begin, uint64_t size, uint8_t kind, uint32_t flags, uint32_t format, bool pending, bool inside_keeps,
	           bool* nested = nullptr, uint64_t owner = 0, bool live = false) {
		if (nested != nullptr) *nested = false;
		if (size == 0 || begin + size < begin) return 0;
		const uint64_t   end = begin + size;
		std::unique_lock lock(m_mutex);
		auto             it = FirstOverlap(begin);
		if (inside_keeps && it != m_ranges.end() && it->first <= begin && it->second.end >= end &&
		    !(it->first == begin && it->second.end == end)) {
			if (nested != nullptr) *nested = true;
			return 0;
		}
		size_t replaced = 0;
		while (it != m_ranges.end() && it->first < end) {
			it = Erase(it);
			++replaced;
		}
		Entry& entry  = m_ranges.try_emplace(begin).first->second;
		entry.end     = end;
		entry.flags   = flags;
		entry.format  = format;
		entry.kind    = kind;
		entry.pending = pending;
		entry.owner   = owner != begin ? owner : 0;
		entry.generation.store(NextGeneration(), std::memory_order_relaxed);
		entry.live.store(live ? 1 : 0, std::memory_order_relaxed);
		entry.stamp.store(Now(), std::memory_order_relaxed);
		m_filter[FilterIndex(begin)].fetch_add(1, std::memory_order_relaxed);
		for (uint64_t granule = begin >> GranuleShift; granule <= (end - 1) >> GranuleShift && granule < GranuleWords * 64; ++granule) {
			auto& word = m_granules[granule / 64];
			const uint64_t bit = uint64_t {1} << (granule % 64);
			if ((word.load(std::memory_order_relaxed) & bit) == 0) word.fetch_or(bit, std::memory_order_relaxed);
		}
		if (entry.owner != 0) m_filter[FilterIndex(entry.owner)].fetch_add(1, std::memory_order_relaxed);
		m_count[kind].fetch_add(1, std::memory_order_relaxed);
		m_bytes[kind].fetch_add(size, std::memory_order_relaxed);
		return replaced;
	}

	// The pending range exactly [begin, end): confirmed with the CBuffer's flags and format. False: there is none.
	bool Confirm(uint64_t begin, uint64_t end, uint32_t flags, uint32_t format) {
		if (!MayBeginAt(begin)) return false;
		std::unique_lock lock(m_mutex);
		const auto       it = m_ranges.find(begin);
		if (it == m_ranges.end() || !it->second.pending || it->second.end != end) return false;
		it->second.pending = false;
		it->second.flags   = flags;
		it->second.format  = format;
		it->second.generation.store(NextGeneration(), std::memory_order_relaxed);
		return true;
	}

	// Takes out the range that starts at `begin` (only a pending one with `only_pending`). False: there is none.
	bool RemoveAt(uint64_t begin, bool only_pending = false) {
		if (!MayBeginAt(begin)) return false;
		std::unique_lock lock(m_mutex);
		const auto       it = m_ranges.find(begin);
		if (it == m_ranges.end() || (only_pending && !it->second.pending)) return false;
		Erase(it);
		return true;
	}

	// Takes out every range that overlaps [begin, begin + size); returns how many.
	size_t RemoveIn(uint64_t begin, uint64_t size) {
		if (size == 0 || begin + size < begin) return 0;
		const uint64_t   end = begin + size;
		std::unique_lock lock(m_mutex);
		auto             it      = FirstOverlap(begin);
		size_t           removed = 0;
		while (it != m_ranges.end() && it->first < end) {
			it = Erase(it);
			++removed;
		}
		return removed;
	}

	// Where `address` lies: inside a range, in a page that holds part of one (the nearest one is given), or neither.
	[[nodiscard]] Hit Find(uint64_t address) const {
		Hit              hit;
		const uint64_t   page = address & ~(PageSize - 1);
		std::shared_lock lock(m_mutex);
		auto             it = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (address < previous->second.end) {
				Fill(&hit, Where::Inside, previous->first, previous->second);
				return hit;
			}
			if (previous->second.end > page) Fill(&hit, Where::Neighbor, previous->first, previous->second);
		}
		if (hit.where == Where::None && it != m_ranges.end() && it->first - page < PageSize)
			Fill(&hit, Where::Neighbor, it->first, it->second);
		return hit;
	}

	// The bytes of [begin, begin + size) inside registered ranges.
	[[nodiscard]] uint64_t Overlap(uint64_t begin, uint64_t size) const {
		if (size == 0 || begin + size < begin) return 0;
		const uint64_t   end   = begin + size;
		uint64_t         bytes = 0;
		std::shared_lock lock(m_mutex);
		auto             it = m_ranges.upper_bound(begin);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second.end > begin) it = previous;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const uint64_t from = it->first > begin ? it->first : begin;
			const uint64_t to   = it->second.end < end ? it->second.end : end;
			if (to > from) bytes += to - from;
		}
		return bytes;
	}

	// ---- (policy1) ------------------------------------------------------------------------------------------------
	// The registry's clock: a count of 1/8 s set by the audit's thread (0 in a test that never sets it). Per-use
	// ranges (PerUseKinds) are fresh while announced within FreshTicks of it.
	static constexpr uint32_t FreshTicks = 16; // (2 s)
	inline static std::atomic<uint32_t> s_now {0};
	// Generations never repeat (a range registered again does not get an old one's): two equal generations of a
	// range mean that no hook announced it in between.
	inline static std::atomic<uint32_t> s_generation {0};
	[[nodiscard]] static uint32_t NextGeneration() noexcept { return s_generation.fetch_add(1, std::memory_order_relaxed) + 1; }
	[[nodiscard]] static uint32_t Now() noexcept { return s_now.load(std::memory_order_relaxed); }
	[[nodiscard]] static bool     Fresh(uint8_t kind, int32_t live, uint32_t stamp) noexcept {
        return ((1u << kind) & PerUseKinds) == 0 || live > 0 || Now() - stamp <= FreshTicks;
	}

	// The range exactly [begin, begin + size) of `kind` is there already: announced again (generation, stamp;
	// `live_delta` added to its live wrappers) under the shared lock, and true. False: there is none.
	bool Touch(uint64_t begin, uint64_t size, uint8_t kind, int32_t live_delta) const {
		if (!MayBeginAt(begin)) return false;
		std::shared_lock lock(m_mutex);
		const auto       it = m_ranges.find(begin);
		if (it == m_ranges.end() || it->second.end != begin + size || it->second.kind != kind || it->second.pending) return false;
		auto& entry = const_cast<Entry&>(it->second); // (atomics: shared writers)
		entry.generation.store(NextGeneration(), std::memory_order_relaxed);
		entry.stamp.store(Now(), std::memory_order_relaxed);
		if (live_delta != 0) entry.live.fetch_add(live_delta, std::memory_order_relaxed);
		return true;
	}
	// A wrapper over the range that starts at `begin` was destroyed: one live wrapper less (the range stays). False:
	// no range of `kind` starts there.
	bool Release(uint64_t begin, uint8_t kind) const {
		if (!MayBeginAt(begin)) return false;
		std::shared_lock lock(m_mutex);
		const auto       it = m_ranges.find(begin);
		if (it == m_ranges.end() || it->second.kind != kind) return false;
		auto&   entry = const_cast<Entry&>(it->second);
		int32_t live  = entry.live.load(std::memory_order_relaxed);
		while (live > 0 && !entry.live.compare_exchange_weak(live, live - 1, std::memory_order_relaxed)) {
		}
		return true;
	}
	// A hook saw a write inside the range that holds `address`: its generation moves. False: no range holds it.
	bool Announce(uint64_t address) const {
		std::shared_lock lock(m_mutex);
		auto             it = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) return false;
		--it;
		if (address >= it->second.end) return false;
		auto& entry = const_cast<Entry&>(it->second);
		entry.generation.store(NextGeneration(), std::memory_order_relaxed);
		entry.stamp.store(Now(), std::memory_order_relaxed);
		return true;
	}
	// Takes out the ranges whose owner block is `owner` (Entry::owner): they lie from `owner` on, with no other
	// range between them (a block is one texture's). Returns how many.
	size_t RemoveOwned(uint64_t owner) {
		if (owner == 0 || !MayBeginAt(owner)) return 0;
		std::unique_lock lock(m_mutex);
		auto             it      = m_ranges.lower_bound(owner);
		size_t           removed = 0;
		while (it != m_ranges.end() && (it->second.owner == owner || it->first == owner)) {
			it = Erase(it);
			++removed;
		}
		return removed;
	}
	// (bit = 1 << kind) The kinds of the fresh ranges with a byte in the 4 KiB page of `address` (stale per-use
	// ranges left out).
	[[nodiscard]] uint32_t PageKinds(uint64_t address) const {
		const uint64_t   page = address & ~(PageSize - 1);
		uint32_t         mask = 0;
		std::shared_lock lock(m_mutex);
		auto             it = m_ranges.upper_bound(page);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second.end > page) it = previous;
		}
		for (; it != m_ranges.end() && it->first < page + PageSize; ++it) {
			const Entry& entry = it->second;
			if (Fresh(entry.kind, entry.live.load(std::memory_order_relaxed), entry.stamp.load(std::memory_order_relaxed)))
				mask |= 1u << entry.kind;
		}
		return mask;
	}
	// Every byte of [begin, begin + size) lies in a fresh range of one of `kinds` (bit = 1 << kind).
	[[nodiscard]] bool Covered(uint64_t begin, uint64_t size, uint32_t kinds) const {
		if (size == 0 || begin + size < begin) return false;
		const uint64_t   end = begin + size;
		uint64_t         at  = begin;
		std::shared_lock lock(m_mutex);
		auto             it = m_ranges.upper_bound(begin);
		if (it == m_ranges.begin()) return false;
		--it;
		for (; it != m_ranges.end() && at < end; ++it) {
			const Entry& entry = it->second;
			if (it->first > at) return false;
			if (entry.end <= at) {
				if (it->first < at) continue; // (only the first one can end before `at`)
				return false;
			}
			if (((1u << entry.kind) & kinds) == 0 ||
			    !Fresh(entry.kind, entry.live.load(std::memory_order_relaxed), entry.stamp.load(std::memory_order_relaxed)))
				return false;
			at = entry.end;
		}
		return at >= end;
	}
	// Takes out the per-use ranges with no live wrapper not announced for `ticks` of the clock; returns how many.
	size_t SweepStale(uint32_t ticks) {
		std::unique_lock lock(m_mutex);
		size_t           removed = 0;
		const uint32_t   now     = Now();
		for (auto it = m_ranges.begin(); it != m_ranges.end();) {
			const Entry& entry = it->second;
			if (((1u << entry.kind) & PerUseKinds) != 0 && entry.live.load(std::memory_order_relaxed) <= 0 &&
			    now - entry.stamp.load(std::memory_order_relaxed) > ticks) {
				it = Erase(it);
				++removed;
			} else {
				++it;
			}
		}
		return removed;
	}
	// The `index`-th range from `from` on in address order (wrapping to the first), for sampling. False: none.
	[[nodiscard]] bool Nth(uint64_t from, size_t index, Hit* out) const {
		std::shared_lock lock(m_mutex);
		if (m_ranges.empty()) return false;
		auto it = m_ranges.lower_bound(from);
		for (size_t i = 0;; ++i) {
			if (it == m_ranges.end()) it = m_ranges.begin();
			if (i == index) break;
			++it;
		}
		Fill(out, Where::Inside, it->first, it->second);
		return true;
	}
	// The range that starts at `begin`, when it still ends at `end`. False: gone or another one.
	[[nodiscard]] bool At(uint64_t begin, uint64_t end, Hit* out) const {
		if (!MayBeginAt(begin)) return false;
		std::shared_lock lock(m_mutex);
		const auto       it = m_ranges.find(begin);
		if (it == m_ranges.end() || it->second.end != end) return false;
		Fill(out, Where::Inside, it->first, it->second);
		return true;
	}

	[[nodiscard]] uint64_t Count(uint8_t kind) const noexcept { return m_count[kind].load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Bytes(uint8_t kind) const noexcept { return m_bytes[kind].load(std::memory_order_relaxed); }

private:
	using Map = std::map<uint64_t, Entry>;

	static size_t FilterIndex(uint64_t address) noexcept {
		return static_cast<size_t>(((address / PageSize) * 0x9E3779B97F4A7C15ull) >> (64u - FilterBits));
	}
	static void Fill(Hit* hit, Where where, uint64_t begin, const Entry& entry) {
		hit->where  = where;
		hit->begin  = begin;
		hit->end    = entry.end;
		hit->flags  = entry.flags;
		hit->format = entry.format;
		hit->kind   = entry.kind;
		hit->generation = entry.generation.load(std::memory_order_relaxed);
		hit->stamp      = entry.stamp.load(std::memory_order_relaxed);
		hit->live       = entry.live.load(std::memory_order_relaxed);
		hit->pending    = entry.pending;
	}
	// The first range that ends after `begin` (under the lock).
	Map::iterator FirstOverlap(uint64_t begin) {
		auto it = m_ranges.upper_bound(begin);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second.end > begin) it = previous;
		}
		return it;
	}
	Map::iterator Erase(Map::iterator it) {
		m_filter[FilterIndex(it->first)].fetch_sub(1, std::memory_order_relaxed);
		if (it->second.owner != 0) m_filter[FilterIndex(it->second.owner)].fetch_sub(1, std::memory_order_relaxed);
		m_count[it->second.kind].fetch_sub(1, std::memory_order_relaxed);
		m_bytes[it->second.kind].fetch_sub(it->second.end - it->first, std::memory_order_relaxed);
		return m_ranges.erase(it);
	}

	mutable std::shared_mutex                m_mutex;
	Map                                      m_ranges;
	std::unique_ptr<std::atomic<uint16_t>[]> m_filter;
	std::unique_ptr<std::atomic<uint64_t>[]> m_granules; // (policy1: MayContain)
	std::atomic<uint64_t>                    m_count[KindCount] {};
	std::atomic<uint64_t>                    m_bytes[KindCount] {};
};

} // namespace Loader::DemonsSoulsWriteAuditRegistry
