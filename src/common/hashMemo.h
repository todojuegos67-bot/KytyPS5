#ifndef EMULATOR_SRC_COMMON_HASHMEMO_H_
#define EMULATOR_SRC_COMMON_HASHMEMO_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace Common {

// A set-associative memo: `Sets` sets of `Ways` entries, the set picked by the caller's hash. Find gives an entry of
// that hash whose key (and validity) the caller's `match` accepts. A store takes the set's first free entry, else the
// one after the set's last replacement (round robin).
template <typename Key, typename Value, size_t Sets, size_t Ways>
class HashMemo {
	static_assert(Sets != 0 && (Sets & (Sets - 1)) == 0, "Sets must be a power of two");
	static_assert(Ways != 0);

public:
	struct Entry {
		Key      key {};
		Value    value {};
		uint64_t hash  = 0;
		bool     valid = false;
	};

	HashMemo(): m_sets(std::make_unique<std::array<Set, Sets>>()) {}

	template <typename Match>
	[[nodiscard]] Entry* Find(uint64_t hash, Match&& match) {
		auto& set = (*m_sets)[hash & (Sets - 1)];
		for (auto& entry: set.ways) {
			if (entry.valid && entry.hash == hash && match(entry)) return &entry;
		}
		return nullptr;
	}

	// The entry to fill for `hash`; the caller sets its key and value.
	Entry& Store(uint64_t hash) {
		auto&  set   = (*m_sets)[hash & (Sets - 1)];
		Entry* taken = nullptr;
		for (auto& entry: set.ways) {
			if (!entry.valid) {
				taken = &entry;
				break;
			}
		}
		if (taken == nullptr) {
			taken    = &set.ways[set.next];
			set.next = static_cast<uint32_t>((set.next + 1u) % Ways);
		}
		taken->hash  = hash;
		taken->valid = true;
		return *taken;
	}

	void Clear() {
		for (auto& set: *m_sets) {
			for (auto& entry: set.ways) entry.valid = false;
			set.next = 0;
		}
	}

private:
	struct Set {
		std::array<Entry, Ways> ways {};
		uint32_t                next = 0;
	};
	std::unique_ptr<std::array<Set, Sets>> m_sets;
};

} // namespace Common

#endif /* EMULATOR_SRC_COMMON_HASHMEMO_H_ */
