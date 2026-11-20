#ifndef EMULATOR_SRC_LOCAL_LABEL_PAGES_H_
#define EMULATOR_SRC_LOCAL_LABEL_PAGES_H_

// The command processor's label words (4-byte guest words it wrote; Spec::g_labels, speculation-state.h) by 4 KiB guest
// page: a two-level radix (16 MiB chunks in a small map, each 4,096 page slots) of pages, each a 1,024-bit bitset, one
// bit a 4-byte word, made at the page's first label. No step ever moves more than one entry: a page's first label
// allocates its page (128 bytes), a chunk's first page its slots (32 KiB); the chunk map holds one entry for each 16 MiB
// of guest memory with labels, so its rehashes are of a few hundred entries. (A std::unordered_set of the word
// addresses rehashed all of its entries when it outgrew its buckets, under the labels' mutex: 11 ms at 262,144 words,
// 27 ms at 524,288, 64 ms at 1,048,576.)
// Membership is the set's for every address: a word's bit is (address & 0xfff) >> 2 in the page address >> 12, so the
// address's low two bits are dropped as the set's key (address & ~3) drops them, and a 4-byte word never spans pages.
// Not thread-safe: its owner's lock guards it, exclusively (Find updates the last chunk's cache).

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace LabelPages {

// The radix: pages of type Page by the guest address's page.
template <typename Page>
class Radix {
public:
	[[nodiscard]] Page* Find(uint64_t address) const {
		const Chunk* chunk = FindChunk(address >> ChunkShift);
		return chunk != nullptr ? (*chunk)[SlotOf(address)].get() : nullptr;
	}
	Page& Make(uint64_t address) {
		const uint64_t key   = address >> ChunkShift;
		Chunk*         chunk = FindChunk(key);
		if (chunk == nullptr) {
			auto made = std::make_unique<Chunk>();
			chunk     = made.get();
			m_chunks.emplace(key, std::move(made));
			m_last_key = key, m_last = chunk;
		}
		auto& slot = (*chunk)[SlotOf(address)];
		if (slot == nullptr) slot = std::make_unique<Page>();
		return *slot;
	}

private:
	static constexpr uint32_t ChunkShift = 24, PageShift = 12;
	using Chunk                          = std::array<std::unique_ptr<Page>, size_t {1} << (ChunkShift - PageShift)>;
	[[nodiscard]] static size_t SlotOf(uint64_t address) {
		return static_cast<size_t>((address >> PageShift) & ((uint64_t {1} << (ChunkShift - PageShift)) - 1));
	}
	Chunk* FindChunk(uint64_t key) const {
		if (m_last != nullptr && m_last_key == key) return m_last;
		const auto found = m_chunks.find(key);
		if (found == m_chunks.end()) return nullptr;
		m_last_key = key, m_last = found->second.get();
		return m_last;
	}
	std::unordered_map<uint64_t, std::unique_ptr<Chunk>> m_chunks;
	mutable uint64_t                                     m_last_key = 0;
	mutable Chunk*                                       m_last     = nullptr;
};

// A page's words: one bit each.
struct Bits {
	std::array<uint64_t, 16> words {};
	[[nodiscard]] static uint32_t IndexOf(uint64_t address) { return static_cast<uint32_t>((address & 0xfffu) >> 2u); }
	[[nodiscard]] bool            Test(uint64_t address) const {
		           const auto i = IndexOf(address);
		           return ((words[i >> 6u] >> (i & 63u)) & 1u) != 0;
	}
	void Set(uint64_t address) {
		const auto i = IndexOf(address);
		words[i >> 6u] |= uint64_t {1} << (i & 63u);
	}
};

// The set of label words (an address's word: address & ~3).
class Set {
public:
	void Insert(uint64_t address) { m_pages.Make(address).Set(address); }
	[[nodiscard]] bool Contains(uint64_t address) const {
		const auto* page = m_pages.Find(address);
		return page != nullptr && page->Test(address);
	}

private:
	Radix<Bits> m_pages;
};

} // namespace LabelPages

#endif /* EMULATOR_SRC_LOCAL_LABEL_PAGES_H_ */
