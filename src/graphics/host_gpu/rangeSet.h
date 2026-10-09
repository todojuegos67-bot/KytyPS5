#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

extern "C" {
// 1: Add returns early for a range already covered, and lookups first try the
// interval the previous lookup found (valid until the set changes).
extern volatile std::atomic<uint32_t> kyty_local_range_set_fast_mode;
}

namespace Libs::Graphics {

class RangeSet final {
public:
	// The owner reads this set from one thread only (lookups then update the hint).
	void AllowFastPath() { m_fast_eligible = true; }

	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
	};

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		if (Fast() && Covered(address, end)) return;
		++m_version;
		// Intervals that overlap or touch [address, end) merge with it.
		auto first = LowerBound(address);
		if (first != m_ranges.begin() && std::prev(first)->end >= address) {
			--first;
		}
		uint64_t begin = address;
		uint64_t last  = end;
		auto     it    = first;
		for (; it != m_ranges.end() && it->begin <= last; ++it) {
			begin = std::min(begin, it->begin);
			last  = std::max(last, it->end);
		}
		if (first == it) {
			m_ranges.insert(first, {begin, last});
			return;
		}
		*first = {begin, last};
		m_ranges.erase(first + 1, it);
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		++m_version;
		auto first = LowerBound(address);
		if (first != m_ranges.begin() && std::prev(first)->end > address) {
			--first;
		}
		auto last = first;
		while (last != m_ranges.end() && last->begin < end) {
			++last;
		}
		if (first == last) {
			return;
		}
		// What remains of the first and the last interval replaces them all.
		Interval parts[2];
		size_t   count = 0;
		if (first->begin < address) {
			parts[count++] = {first->begin, address};
		}
		if (std::prev(last)->end > end) {
			parts[count++] = {end, std::prev(last)->end};
		}
		const auto index   = static_cast<size_t>(first - m_ranges.begin());
		const auto removed = static_cast<size_t>(last - first);
		if (count > removed) {
			m_ranges[index] = parts[0];
			m_ranges.insert(m_ranges.begin() + static_cast<ptrdiff_t>(index + 1), parts[1]);
			return;
		}
		std::copy(parts, parts + count, m_ranges.begin() + static_cast<ptrdiff_t>(index));
		m_ranges.erase(m_ranges.begin() + static_cast<ptrdiff_t>(index + count),
		               m_ranges.begin() + static_cast<ptrdiff_t>(index + removed));
	}

	void Clear() {
		++m_version;
		m_ranges.clear();
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& range: m_ranges) {
			func(range.begin, range.end);
		}
	}

	[[nodiscard]] std::vector<Range> Intersections(uint64_t address, uint64_t size) const {
		std::vector<Range> result;
		ForEachIntersection(address, size, [&result](Range range) { result.push_back(range); });
		return result;
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = LowerBound(address);
		if (it != m_ranges.begin() && std::prev(it)->end > address) {
			return true;
		}
		return it != m_ranges.end() && it->begin < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		if (Fast()) return Covered(address, end);
		auto       it  = UpperBound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->begin <= address && it->end >= end;
	}

	template <typename Func>
	void ForEachIntersection(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = UpperBound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->begin < end; ++it) {
			const auto begin = std::max(address, it->begin);
			const auto last  = std::min(end, it->end);
			if (begin < last) {
				func(Range {begin, last - begin});
			}
		}
	}

	// The intersections with each interval of `queries` in turn, as ForEachIntersection gives them one query at a time:
	// one search for the first, then a walk (the queries are sorted and disjoint, as a set's own). A search per query
	// against ~50K GPU-written ranges after an area streamed in took 80 ms of a walk's slow frames (guest readbacks).
	template <typename Func>
	void ForEachIntersection(const RangeSet& queries, Func&& func) const {
		if (queries.m_ranges.empty() || m_ranges.empty()) return;
		auto it = UpperBound(queries.m_ranges.front().begin);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (const auto& query: queries.m_ranges) {
			while (it != m_ranges.end() && it->end <= query.begin) {
				++it;
			}
			for (auto at = it; at != m_ranges.end() && at->begin < query.end; ++at) {
				const auto begin = std::max(query.begin, at->begin);
				const auto last  = std::min(query.end, at->end);
				if (begin < last) {
					func(Range {begin, last - begin});
				}
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	[[nodiscard]] size_t Count() const { return m_ranges.size(); }

private:
	[[nodiscard]] bool Fast() const {
		return m_fast_eligible && kyty_local_range_set_fast_mode.load(std::memory_order_relaxed) != 0;
	}

	// [address, end) lies inside one interval; remembers that interval. An interval found at
	// version V covers the same bytes while the set stays at V: a few are kept, since lookups
	// alternate between ranges (a dispatch's written buffers).
	bool Covered(uint64_t address, uint64_t end) const {
		for (const auto& hint: m_hints)
			if (hint.version == m_version && hint.begin <= address && end <= hint.end) return true;
		auto it = UpperBound(address);
		if (it == m_ranges.begin()) return false;
		--it;
		if (it->begin > address || it->end < end) return false;
		m_hints[m_next_hint++ % m_hints.size()] = {m_version, it->begin, it->end};
		return true;
	}

	// Disjoint, non-adjacent intervals sorted by address: lookups (thousands per frame, most of
	// them for SRT words and bindings) search one contiguous array instead of walking tree nodes.
	struct Interval {
		uint64_t begin = 0, end = 0;
	};
	using Intervals = std::vector<Interval>;

	[[nodiscard]] Intervals::const_iterator LowerBound(uint64_t address) const {
		return std::lower_bound(m_ranges.begin(), m_ranges.end(), address,
		                        [](const Interval& range, uint64_t value) { return range.begin < value; });
	}
	[[nodiscard]] Intervals::iterator LowerBound(uint64_t address) {
		return std::lower_bound(m_ranges.begin(), m_ranges.end(), address,
		                        [](const Interval& range, uint64_t value) { return range.begin < value; });
	}
	[[nodiscard]] Intervals::const_iterator UpperBound(uint64_t address) const {
		return std::upper_bound(m_ranges.begin(), m_ranges.end(), address,
		                        [](uint64_t value, const Interval& range) { return value < range.begin; });
	}

	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	Intervals m_ranges;
	bool      m_fast_eligible = false;
	uint64_t  m_version       = 1;
	struct Hint {
		uint64_t version = 0, begin = 0, end = 0;
	};
	mutable std::array<Hint, 4> m_hints {};
	mutable uint32_t            m_next_hint = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
