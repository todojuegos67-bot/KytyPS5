#include "common/groupReuse.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GroupReuseTests: failed: %s\n", message);
		std::abort();
	}
}

// A texture cache in miniature. A value is a candidate's number; its bytes are [value * 16, value * 16 + 16), and a
// page is 64 bytes. A value 0 mod 4 is a null descriptor, one over a target binds as null, one without an image binds
// as null, and any other binds as its number. Every image made or freed gives the images over its page a new stamp
// and appends its page to the change list, which keeps the last `kept` changes. Resolving `creator` makes the image
// `created` if it is missing, as a texture lookup can; such a lookup is never repeated.
struct Cache {
	std::set<int>                              images;
	std::set<int>                              targets;
	std::map<int, uint64_t>                    stamps;
	std::vector<std::pair<uint64_t, uint64_t>> log; // page ranges; the clock is its size
	size_t                                     kept    = 1000;
	int                                        creator = -1;
	int                                        created = -1;
	size_t                                     calls   = 0;
	size_t                                     repeats = 0;
	std::map<int, int>                         looked; // value -> lookups of it as bound (a lookup's side effect)

	static uint64_t Page(int value) { return uint64_t(value) * 16u / 64u; }
	void            Change(int value) {
        const auto page = Page(value);
        for (auto& [image, stamp]: stamps)
            if (Page(image) == page) stamp = log.size() + 1000u;
        log.emplace_back(page * 64u, page * 64u + 64u);
	}
	void Add(int value) {
		images.insert(value);
		stamps[value] = 0;
		Change(value);
	}
	void Remove(int value) {
		images.erase(value);
		stamps.erase(value);
		Change(value);
	}

	int Resolve(int value, Common::GroupResult* result, uint64_t* stamp) {
		++calls;
		if (result != nullptr) *result = {};
		if (value % 4 == 0) {
			if (result != nullptr) result->outcome = Common::GroupOutcome::Fixed;
			return 0;
		}
		if (targets.contains(value)) return 0;
		if (!images.contains(value)) {
			if (result != nullptr)
				*result = {Common::GroupOutcome::Absent, uint64_t(value) * 16u, uint64_t(value) * 16u + 16u};
			return 0;
		}
		if (value == creator && !images.contains(created)) Add(created);
		++looked[value];
		// A lookup that can make an image is not one to repeat (FindImage's `plain`).
		if (result != nullptr && value != creator) result->outcome = Common::GroupOutcome::Held;
		if (stamp != nullptr) *stamp = stamps[value];
		return value;
	}
	bool Repeat(int value, uint64_t stamp) {
		if (targets.contains(value) || !images.contains(value) || stamps[value] != stamp) return false;
		++repeats;
		++looked[value];
		return true;
	}

	[[nodiscard]] uint64_t Clock() const { return log.size(); }
	template <typename Visit>
	bool Since(uint64_t clock, Visit&& visit) const {
		if (log.size() - clock > kept) return false;
		for (auto at = clock; at < log.size(); ++at) visit(log[at].first, log[at].second);
		return true;
	}
};

using Reuse = Common::GroupReuse<int, int>;

struct Group {
	std::vector<int>      bindings;
	std::vector<uint64_t> stamps;
	Reuse                 reuse;
};

bool Resolve(Cache& cache, const std::vector<int>& values, int key, bool reusable, Group& group) {
	group.stamps.resize(values.size());
	return Common::ResolveGroup(
	    std::span(values), key, reusable, group.bindings, group.reuse, cache,
	    [&](size_t rank, Common::GroupResult& result) {
		    return cache.Resolve(values[rank], &result, &group.stamps[rank]);
	    },
	    [&](size_t rank, int& binding) { return cache.Repeat(binding, group.stamps[rank]); });
}

// The plain loop, on a copy of the cache.
std::vector<int> Plain(Cache cache, const std::vector<int>& values) {
	std::vector<int> group;
	for (const auto value: values) group.push_back(cache.Resolve(value, nullptr, nullptr));
	return group;
}

std::vector<int> Values() {
	std::vector<int> values;
	for (int value = 1; value <= 64; ++value) values.push_back(value);
	return values;
}

Cache Images() {
	Cache cache;
	for (int value = 1; value <= 64; value += 3) cache.Add(value);
	return cache;
}

void TestHitAndMiss() {
	auto       cache  = Images();
	const auto values = Values();
	Group      group;
	Check(!Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && group.reuse.valid &&
	          group.reuse.whole,
	      "the first resolution is not the plain loop's or was not stored");
	const auto all = cache.calls;
	cache.calls    = 0;
	cache.repeats  = 0;
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values),
	      "an unchanged group was not reused or differs from the plain loop");
	Check(cache.calls == 0 && cache.repeats == group.reuse.walk.size() && cache.repeats < all / 2 &&
	          group.reuse.touched == group.reuse.walk,
	      "a reused group did more than repeat its bound candidates");
	// A changed value resolves that member alone.
	auto changed = values;
	changed[5]   = 7;
	cache.calls  = 0;
	Check(Resolve(cache, changed, 1, true, group) && group.bindings == Plain(cache, changed) && cache.calls == 1 &&
	          std::ranges::binary_search(group.reuse.touched, 5u),
	      "a changed value was not resolved alone");
	changed[6] = 8; // a bound member becomes a null descriptor
	cache.calls = 0;
	Check(Resolve(cache, changed, 1, true, group) && group.bindings == Plain(cache, changed) && cache.calls == 1,
	      "a bound member whose value changed repeated its old lookup");
	cache.calls = 0;
	Check(Resolve(cache, changed, 1, true, group) && cache.calls == 0, "the changed values were not kept");
	Check(!Resolve(cache, changed, 2, true, group) && group.bindings == Plain(cache, changed), "another key was reused");
	Group unstored;
	Check(!Resolve(cache, changed, 2, false, unstored) && !unstored.reuse.valid &&
	          !Resolve(cache, changed, 2, false, unstored) && unstored.bindings == Plain(cache, changed),
	      "a group that may not be reused was stored");
	Check(!Resolve(cache, std::vector<int>(values.begin(), values.begin() + 10), 1, true, group) &&
	          group.bindings.size() == 10u,
	      "a shorter group kept old bindings");
}

void TestChanges() {
	auto       cache  = Images();
	const auto values = Values();
	Group      group;
	Resolve(cache, values, 1, true, group);
	// An image made elsewhere: the nulls over its page resolve again (2 and 3; 1 is bound), the others stay.
	cache.Add(2);
	cache.calls = 0;
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && group.bindings[1] == 2,
	      "a null over a new image was kept");
	Check(cache.calls == 3, "not only the bound candidate (stamp moved) and the nulls over the page resolved again");
	cache.calls = 0;
	Check(Resolve(cache, values, 1, true, group) && cache.calls == 0, "a group was not reused after the change");
	// An image freed: its binding does not repeat (its stamp moved), and it becomes a null that holds.
	cache.Remove(7);
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && group.bindings[6] == 0,
	      "a freed image stayed bound");
	cache.calls = 0;
	Check(Resolve(cache, values, 1, true, group) && cache.calls == 0, "a freed image's null was not kept");
	// A bound candidate's lookup makes an image for a later null one, which binds in this same call. 13 is made again
	// so that it resolves (and makes the image) instead of repeating.
	cache.creator = 13;
	cache.created = 30;
	cache.Remove(13);
	cache.Add(13);
	auto expected = Plain(cache, values);
	Check(expected[29] == 30, "the test's plain loop did not make the image");
	Check(Resolve(cache, values, 1, true, group) && group.bindings == expected && group.reuse.valid,
	      "a resolve that made an image left a later null stale");
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values),
	      "the group differs after a resolve made an image");
	// ... and one that makes an image for an earlier null: that null stays for this call (the plain loop resolved it
	// before) and binds in the next.
	cache.creator = 41;
	cache.created = 6;
	cache.Remove(41);
	cache.Add(41);
	expected = Plain(cache, values);
	Check(expected[5] == 0, "the test's plain loop bound the earlier null");
	Check(Resolve(cache, values, 1, true, group) && group.bindings == expected, "an earlier null changed too early");
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && group.bindings[5] == 6,
	      "an earlier null over a new image was kept");
}

void TestLostChanges() {
	auto       cache  = Images();
	const auto values = Values();
	Group      group;
	cache.kept = 2;
	Resolve(cache, values, 1, true, group);
	for (const int value: {2, 3, 5}) cache.Add(value);
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && !group.reuse.valid &&
	          group.reuse.touched.size() == values.size(),
	      "changes the list no longer held did not resolve the group whole");
	Check(!Resolve(cache, values, 1, true, group) && group.reuse.valid, "the group was not stored again");
}

void TestTargets() {
	auto       cache  = Images();
	const auto values = Values();
	Group      group;
	cache.targets.insert(10);
	Resolve(cache, values, 1, true, group);
	// A bound candidate becomes a target of the draw, and the target binds again.
	cache.targets = {13};
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) &&
	          group.bindings[12] == 0 && group.bindings[9] == 10,
	      "a reused group missed a change of the draw's targets");
	cache.targets.clear();
	Check(Resolve(cache, values, 1, true, group) && group.bindings == Plain(cache, values) && group.bindings[12] == 13,
	      "a candidate that stopped being a target did not bind again");
}

// Random image churn, targets and creators against the plain loop, with the lookups' side effects counted.
void TestRandom() {
	std::mt19937 random(6);
	for (int round = 0; round < 300; ++round) {
		auto  cache  = Images();
		auto  plain  = cache;
		auto  values = Values();
		Group group;
		cache.kept    = 1 + random() % 8;
		cache.creator = plain.creator = 1 + static_cast<int>(random() % 64);
		cache.created = plain.created = 1 + static_cast<int>(random() % 64);
		for (int call = 0; call < 30; ++call) {
			for (auto change = random() % 4; change > 0; --change) {
				const int value = 1 + static_cast<int>(random() % 64);
				if (cache.images.contains(value)) {
					cache.Remove(value);
					plain.Remove(value);
				} else {
					cache.Add(value);
					plain.Add(value);
				}
			}
			for (auto change = random() % 3; change > 0; --change) values[random() % values.size()] = 1 + static_cast<int>(random() % 64);
			cache.targets.clear();
			if (random() % 2) cache.targets.insert(1 + static_cast<int>(random() % 64));
			plain.targets = cache.targets;
			std::vector<int> expected;
			for (const auto value: values) expected.push_back(plain.Resolve(value, nullptr, nullptr));
			Resolve(cache, values, 1, true, group);
			Check(group.bindings == expected, "a reused group differs from the plain loop");
			Check(cache.images == plain.images, "a reused group made other images than the plain loop");
			Check(cache.looked == plain.looked, "a reused group's bound candidates were looked up other times");
			Check(std::ranges::is_sorted(group.reuse.touched), "the visited ranks are not in rank order");
		}
	}
}

void TestKeepsCapacity() {
	auto       cache  = Images();
	const auto values = Values();
	Group      group;
	Resolve(cache, values, 1, false, group);
	const auto* data = group.bindings.data();
	Resolve(cache, values, 2, false, group);
	Check(group.bindings.data() == data, "resolving a group again reallocated its storage");
}

} // namespace

int main() {
	TestHitAndMiss();
	TestChanges();
	TestLostChanges();
	TestTargets();
	TestRandom();
	TestKeepsCapacity();
	std::printf("GroupReuseTests: all passed\n");
	return 0;
}
