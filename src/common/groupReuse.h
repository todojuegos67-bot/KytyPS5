#ifndef EMULATOR_SRC_COMMON_GROUPREUSE_H_
#define EMULATOR_SRC_COMMON_GROUPREUSE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Common {

// What one candidate's resolution depends on.
enum class GroupOutcome : uint8_t {
	Fixed,   // a null from the value alone: holds for good
	Absent,  // a null for want of something over its bytes (GroupResult's range): holds while nothing over those
	         // bytes changes (the caller's change list)
	Held,    // a result the caller can repeat (its repeat callback) instead of resolving again
	Recheck, // resolved again every time
};

// What a resolve reports beside its result.
struct GroupResult {
	GroupOutcome outcome = GroupOutcome::Recheck;
	uint64_t     address = 0; // Absent: the bytes [address, end) whose changes end it
	uint64_t     end     = 0;
};

// A group's members as last resolved: the values and key they came from, what each result depends on, and the
// position in the caller's change list up to which the Absent results were checked.
template <typename Value, typename Key>
struct GroupReuse {
	std::vector<Value>                         values;
	Key                                        key {};
	std::vector<GroupOutcome>                  outcomes;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;  // by rank; an Absent result's bytes
	std::vector<uint32_t>                      walk;    // the Held and Recheck ranks, ascending
	std::vector<uint32_t>                      stale;   // Absent ranks whose bytes changed, ascending
	std::vector<std::pair<uint64_t, uint32_t>> absent;  // the Absent ranks by address (as indexed)
	std::vector<uint32_t>                      fresh;   // ranks that became Absent in this call (not yet in `absent`)
	std::vector<uint32_t>                      next;    // stale ranks for the next call
	std::vector<uint32_t>                      touched; // the ranks the last call resolved or repeated, ascending
	uint64_t                                   longest = 0; // the longest Absent range
	uint64_t                                   clock   = 0;
	bool                                       whole   = false; // the last call resolved every rank
	bool                                       valid   = false;
};

namespace GroupReuseDetail {

template <typename Value, typename Key>
void Index(GroupReuse<Value, Key>& reuse) {
	reuse.walk.clear();
	reuse.absent.clear();
	reuse.fresh.clear();
	reuse.longest = 0;
	for (uint32_t rank = 0; rank < reuse.outcomes.size(); ++rank) {
		switch (reuse.outcomes[rank]) {
			case GroupOutcome::Fixed: break;
			case GroupOutcome::Absent:
				reuse.absent.emplace_back(reuse.ranges[rank].first, rank);
				reuse.longest = std::max(reuse.longest, reuse.ranges[rank].second - reuse.ranges[rank].first);
				break;
			default: reuse.walk.push_back(rank); break;
		}
	}
	std::ranges::sort(reuse.absent);
}

// Calls stale(rank) for every Absent rank whose bytes overlap [address, end).
template <typename Value, typename Key, typename Stale>
void Overlapping(GroupReuse<Value, Key>& reuse, uint64_t address, uint64_t end, Stale&& stale) {
	const auto from  = address > reuse.longest ? address - reuse.longest : 0u;
	auto first = std::ranges::lower_bound(reuse.absent, std::pair {from, uint32_t {0}});
	for (; first != reuse.absent.end() && first->first < end; ++first) {
		const auto rank = first->second;
		if (reuse.outcomes[rank] == GroupOutcome::Absent && reuse.ranges[rank].first < end &&
		    reuse.ranges[rank].second > address)
			stale(rank);
	}
	for (const auto rank: reuse.fresh) {
		if (reuse.outcomes[rank] == GroupOutcome::Absent && reuse.ranges[rank].first < end &&
		    reuse.ranges[rank].second > address)
			stale(rank);
	}
}

template <typename Value, typename Key>
void Store(GroupReuse<Value, Key>& reuse, uint32_t rank, const GroupResult& result, bool& reindex) {
	const auto old = reuse.outcomes[rank];
	reindex |= (old == GroupOutcome::Fixed || old == GroupOutcome::Absent) !=
	           (result.outcome == GroupOutcome::Fixed || result.outcome == GroupOutcome::Absent);
	if (result.outcome == GroupOutcome::Absent) {
		const std::pair range {result.address, result.end};
		if (old != GroupOutcome::Absent || reuse.ranges[rank] != range) {
			reuse.ranges[rank] = range;
			reuse.fresh.push_back(rank);
			reindex = true;
		}
	} else if (old == GroupOutcome::Absent) {
		reindex = true;
	}
	reuse.outcomes[rank] = result.outcome;
}

} // namespace GroupReuseDetail

// Fills `group` with resolve(rank, result) for every value, in rank order, as a plain loop would, and returns true
// when earlier results were reused. When the key and the number of values are those of the previous call on the
// same group and reuse, only these ranks are visited again, in rank order: the ones whose value changed (they
// resolve), the Held ones (repeat(rank, result) keeps the result when it returns true; otherwise the rank resolves),
// the Recheck ones, and the Absent ones whose bytes the caller's change list (changes.Clock(), changes.Since(clock,
// visit(address, end)): false when it no longer holds the changes after `clock`) shows changed since they were
// checked, including changes made by this call's own resolves before their rank comes. The Fixed and other Absent
// results stay in `group`. A result set is stored for the next call only when it is `reusable`. reuse.touched lists
// the ranks visited (reuse.whole: all). `group` keeps its capacity.
template <typename Binding, typename Value, typename Key, typename Changes, typename Resolve, typename Repeat>
bool ResolveGroup(std::span<const Value> values, const Key& key, bool reusable, std::vector<Binding>& group,
                  GroupReuse<Value, Key>& reuse, Changes&& changes, Resolve&& resolve, Repeat&& repeat) {
	using namespace GroupReuseDetail;
	reuse.touched.clear();
	if (reuse.valid && group.size() == values.size() && reuse.values.size() == values.size() && reuse.key == key) {
		reuse.whole = false;
		reuse.next.clear();
		bool reindex = false;
		// A changed value resolves: a member in the walk loses its Held repeat, any other joins the stale ones. Nearly
		// every call changes none, and its thousands of values compare faster as one block.
		bool same_values = false;
		if constexpr (std::has_unique_object_representations_v<Value>) {
			same_values = std::memcmp(reuse.values.data(), values.data(), values.size_bytes()) == 0;
		}
		for (uint32_t member = 0; !same_values && member < values.size(); ++member) {
			if (reuse.values[member] == values[member]) continue;
			reuse.values[member] = values[member];
			const auto outcome   = reuse.outcomes[member];
			if (outcome == GroupOutcome::Held || outcome == GroupOutcome::Recheck) {
				reuse.outcomes[member] = GroupOutcome::Recheck;
			} else {
				reuse.next.push_back(member);
			}
		}
		if (!reuse.next.empty()) {
			const auto middle = reuse.stale.insert(reuse.stale.end(), reuse.next.begin(), reuse.next.end());
			std::inplace_merge(reuse.stale.begin(), middle, reuse.stale.end());
			const auto [first, last] = std::ranges::unique(reuse.stale);
			reuse.stale.erase(first, last);
			reuse.next.clear();
		}
		bool   lost    = false; // the change list lost track: every later rank resolves again
		size_t walk    = 0;
		size_t stale   = 0;
		// Absent ranks whose bytes changed since they were checked: those after `rank` join this call's walk, the
		// others (already visited, as the plain loop would have before the change) the next call's.
		const auto catch_up = [&](int64_t rank) {
			if (lost) return;
			const auto clock = changes.Clock();
			if (clock == reuse.clock) return;
			const bool listed = changes.Since(reuse.clock, [&](uint64_t address, uint64_t end) {
				Overlapping(reuse, address, end, [&](uint32_t member) {
					if (static_cast<int64_t>(member) <= rank) {
						reuse.next.push_back(member);
						return;
					}
					const auto at = std::lower_bound(reuse.stale.begin() + static_cast<ptrdiff_t>(stale),
					                                 reuse.stale.end(), member);
					if (at == reuse.stale.end() || *at != member) reuse.stale.insert(at, member);
				});
			});
			reuse.clock = clock;
			if (!listed) lost = true;
		};
		catch_up(-1);
		int64_t rank = -1;
		while (!lost && (walk < reuse.walk.size() || stale < reuse.stale.size())) {
			const bool from_walk = stale >= reuse.stale.size() ||
			                       (walk < reuse.walk.size() && reuse.walk[walk] < reuse.stale[stale]);
			const auto member    = from_walk ? reuse.walk[walk++] : reuse.stale[stale++];
			rank                 = member;
			reuse.touched.push_back(member);
			if (reuse.outcomes[member] == GroupOutcome::Held && repeat(size_t {member}, group[member])) continue;
			GroupResult result;
			group[member] = resolve(size_t {member}, result);
			Store(reuse, member, result, reindex);
			catch_up(rank);
		}
		if (lost) {
			for (auto member = static_cast<size_t>(rank + 1); member < values.size(); ++member) {
				reuse.touched.push_back(static_cast<uint32_t>(member));
				GroupResult result;
				group[member] = resolve(member, result);
			}
			reuse.valid = false;
			return true;
		}
		std::ranges::sort(reuse.next);
		const auto [first, last] = std::ranges::unique(reuse.next);
		reuse.next.erase(first, last);
		reuse.stale.swap(reuse.next);
		if (reindex) Index(reuse);
		reuse.fresh.clear();
		return true;
	}
	reuse.valid = false;
	reuse.whole = true;
	reuse.outcomes.assign(values.size(), GroupOutcome::Recheck);
	reuse.ranges.resize(values.size());
	reuse.stale.clear();
	reuse.fresh.clear();
	const auto start = changes.Clock();
	group.clear();
	group.reserve(values.size());
	bool reindex = false;
	for (size_t rank = 0; rank < values.size(); ++rank) {
		GroupResult result;
		group.push_back(resolve(rank, result));
		Store(reuse, static_cast<uint32_t>(rank), result, reindex);
	}
	if (!reusable) return false;
	Index(reuse);
	// What changed while the group resolved: an Absent result checked before the change is checked again next time.
	reuse.clock = start;
	if (changes.Clock() != start &&
	    !changes.Since(start, [&](uint64_t address, uint64_t end) {
		    Overlapping(reuse, address, end, [&](uint32_t member) { reuse.stale.push_back(member); });
	    })) {
		return false;
	}
	reuse.clock = changes.Clock();
	std::ranges::sort(reuse.stale);
	const auto [first, last] = std::ranges::unique(reuse.stale);
	reuse.stale.erase(first, last);
	reuse.values.assign(values.begin(), values.end());
	reuse.key   = key;
	reuse.valid = true;
	return false;
}

} // namespace Common

#endif /* EMULATOR_SRC_COMMON_GROUPREUSE_H_ */
