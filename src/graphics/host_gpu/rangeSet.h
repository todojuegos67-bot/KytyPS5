#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iterator>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class RangeSet final {
public:
	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		if (it == m_ranges.end() || it->first > address) {
			it = m_ranges.emplace_hint(it, address, end);
		} else {
			if (it->second >= end) {
				return;
			}
			it->second = end;
		}
		auto next = std::next(it);
		while (next != m_ranges.end() && next->first <= it->second) {
			it->second = std::max(it->second, next->second);
			next       = m_ranges.erase(next);
		}
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		// The parts that stay keep their nodes: a hole punched in a range allocates one node, not two after a free.
		while (it != m_ranges.end() && it->first < end) {
			if (it->first < address) {
				const auto last = it->second;
				it->second      = address;
				if (last > end) {
					m_ranges.emplace_hint(std::next(it), end, last);
					break;
				}
				++it;
				continue;
			}
			if (it->second > end) {
				auto node  = m_ranges.extract(it++);
				node.key() = end;
				m_ranges.insert(it, std::move(node));
				break;
			}
			it = m_ranges.erase(it);
		}
	}

	void Clear() { m_ranges.clear(); }

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
};

// GPU write tracking: `bits` has a bit for each 4 KiB from `base`, set where a dispatch wrote. Of
// [vaddr, vaddr + size), the GPU-written (`dirty`) bytes on a set bit become `known` written; those on
// a clear bit leave `dirty` unless `known` or `keep` holds them, and go to `cleared(begin, end)`.
template <typename Func>
void ResolveWriteBitmap(RangeSet& dirty, RangeSet& known, const RangeSet& keep, uint64_t base, uint64_t vaddr,
                        uint64_t size, std::span<const uint32_t> bits, Func&& cleared) {
	constexpr uint64_t Page = 4096;
	const auto         bit  = [&](uint64_t page) { return ((bits[page / 32u] >> (page % 32u)) & 1u) != 0u; };
	const uint64_t     pages = bits.size() * 32u;
	// known and keep are read in place, not merged into a copy: known's ranges in a run the bitmap cleared are the
	// same before and after the runs it set (those add only within themselves). A copy was a node for each of the
	// thousands of known ranges of Marvel's Wolverine's 2.7 GB geometry heap, at every tracked write over it.
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	const auto outside_keep = [&](uint64_t from, uint64_t to) {
		auto at = from;
		keep.ForEachInRange(from, to - from, [&](uint64_t first, uint64_t past) {
			if (at < first) {
				ranges.emplace_back(at, first);
			}
			at = past;
		});
		if (at < to) {
			ranges.emplace_back(at, to);
		}
	};
	for (uint64_t page = 0; page < pages;) {
		const bool set  = bit(page);
		auto       last = page + 1;
		// The run ends at the next bit that differs, found a word at a time.
		while (last < pages) {
			const auto differ = (bits[last / 32u] ^ (set ? ~0u : 0u)) >> (last % 32u);
			if (differ != 0u) {
				last += static_cast<uint64_t>(std::countr_zero(differ));
				break;
			}
			last = (last / 32u + 1u) * 32u;
		}
		last = std::min(last, pages);
		const auto begin = std::max(vaddr, base + page * Page);
		const auto end   = std::min(vaddr + size, base + last * Page);
		page             = last;
		if (begin >= end) {
			continue;
		}
		ranges.clear();
		dirty.ForEachInRange(begin, end - begin, [&](uint64_t first, uint64_t past) {
			if (set) {
				ranges.emplace_back(first, past);
				return;
			}
			auto at = first;
			known.ForEachInRange(first, past - first, [&](uint64_t from, uint64_t to) {
				if (at < from) {
					outside_keep(at, from);
				}
				at = to;
			});
			if (at < past) {
				outside_keep(at, past);
			}
		});
		for (const auto& [first, past]: ranges) {
			if (set) {
				known.Add(first, past - first);
				continue;
			}
			dirty.Subtract(first, past - first);
			cleared(first, past);
		}
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
