#include "common/hashMemo.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "HashMemoTests: failed: %s\n", message);
		std::abort();
	}
}

using Memo = Common::HashMemo<int, int, 4, 2>;

int* Find(Memo& memo, uint64_t hash, int key) {
	auto* entry = memo.Find(hash, [&](const Memo::Entry& stored) { return stored.key == key; });
	return entry != nullptr ? &entry->value : nullptr;
}

void Put(Memo& memo, uint64_t hash, int key, int value) {
	auto& entry = memo.Store(hash);
	entry.key   = key;
	entry.value = value;
}

void TestFind() {
	Memo memo;
	Check(Find(memo, 1, 10) == nullptr, "an empty memo finds nothing");
	Put(memo, 1, 10, 100);
	Check(Find(memo, 1, 10) != nullptr && *Find(memo, 1, 10) == 100, "a stored key is found");
	Check(Find(memo, 1, 11) == nullptr, "another key with the same hash is not");
	Check(Find(memo, 5, 10) == nullptr, "the same key under another hash (same set) is not");
	Check(memo.Find(1, [](const Memo::Entry&) { return false; }) == nullptr, "an entry the caller refuses is not");
	memo.Clear();
	Check(Find(memo, 1, 10) == nullptr, "Clear drops every entry");
}

void TestReplacement() {
	Memo memo;
	// Hashes 0, 4, 8, 12 share set 0 of two ways: stores take the free ways, then replace round robin.
	Put(memo, 0, 1, 1);
	Put(memo, 4, 2, 2);
	Check(Find(memo, 0, 1) != nullptr && Find(memo, 4, 2) != nullptr, "two ways hold two keys");
	Put(memo, 8, 3, 3);
	Check(Find(memo, 0, 1) == nullptr && Find(memo, 4, 2) != nullptr && Find(memo, 8, 3) != nullptr,
	      "the third replaces the first way");
	Put(memo, 12, 4, 4);
	Check(Find(memo, 4, 2) == nullptr && Find(memo, 8, 3) != nullptr && Find(memo, 12, 4) != nullptr,
	      "the fourth replaces the second way");
	Check(Find(memo, 1, 1) == nullptr, "other sets are untouched");
}

// Against a map that remembers everything: whatever the memo finds is what was last stored for that key, and an entry
// the caller's validity refuses (here: stored before the key's last invalidation) is never given.
void TestAgainstModel() {
	std::mt19937                                  random(5);
	Common::HashMemo<int, std::pair<int, int>, 8, 4> memo; // value, generation
	std::map<int, int>                            model;   // key -> value last stored
	std::map<int, int>                            generation;
	size_t                                        hits = 0;
	for (int step = 0; step < 200000; ++step) {
		const int key = static_cast<int>(random() % 64);
		if (random() % 50 == 0) ++generation[key];
		const auto hash  = static_cast<uint64_t>(key) * 0x9e3779b97f4a7c15ull;
		auto*      found = memo.Find(hash, [&](const auto& entry) {
            return entry.key == key && entry.value.second == generation[key];
        });
		if (found != nullptr) {
			++hits;
			Check(model.contains(key) && model[key] == found->value.first,
			      "a hit gives the value last stored for its key");
			continue;
		}
		const int value = static_cast<int>(random());
		auto&     entry = memo.Store(hash);
		entry.key       = key;
		entry.value     = {value, generation[key]};
		model[key]      = value;
	}
	Check(hits > 50000, "lookups hit (64 keys, 32 entries)");
}

} // namespace

int main() {
	TestFind();
	TestReplacement();
	TestAgainstModel();
	std::printf("HashMemoTests: all passed\n");
	return 0;
}
