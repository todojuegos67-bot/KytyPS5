#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"

#include <atomic>
#include <compare>
#include <cstdint>
#include <memory>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// Stable-address slot storage for cache resources that are intentionally non-movable. A slot may be read from another
// thread while the owner inserts and erases (a speculative translation's thread, src/graphics/guest_gpu/speculation.h):
// the slots are in chunks that never move (published atomically), a slot's generation and liveness are atomic. An
// erased object is destroyed by erase: a caller whose object another thread may still be reading defers it.
template <typename T>
class SlotVector {
public:
	SlotVector() = default;
	~SlotVector() {
		for (uint32_t chunk = 0; chunk < MaxChunks; ++chunk) delete[] m_chunks[chunk].load(std::memory_order_relaxed);
	}
	KYTY_CLASS_NO_COPY(SlotVector);

	[[nodiscard]] T& operator[](SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		return *At(id.index).value;
	}

	[[nodiscard]] const T& operator[](SlotId id) const noexcept {
		EXIT_IF(!is_allocated(id));
		return *At(id.index).value;
	}

	[[nodiscard]] T* try_get(SlotId id) noexcept { return is_allocated(id) ? &*At(id.index).value : nullptr; }

	[[nodiscard]] const T* try_get(SlotId id) const noexcept { return is_allocated(id) ? &*At(id.index).value : nullptr; }

	// The slot's first line (its liveness and the object's first members) loading, for a lookup soon.
	void prefetch(SlotId id) const noexcept {
		if (id && id.index < m_capacity.load(std::memory_order_acquire)) __builtin_prefetch(&At(id.index));
	}
	[[nodiscard]] bool is_allocated(SlotId id) const noexcept {
		if (!id || id.index >= m_capacity.load(std::memory_order_acquire)) return false;
		const auto& slot = At(id.index);
		return slot.alive.load(std::memory_order_acquire) && slot.generation.load(std::memory_order_acquire) == id.generation;
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index = m_capacity.load(std::memory_order_relaxed);
			EXIT_IF(index >= MaxChunks * ChunkSlots);
			auto& chunk = m_chunks[index >> ChunkBits];
			if (chunk.load(std::memory_order_relaxed) == nullptr) chunk.store(new Slot[ChunkSlots], std::memory_order_release);
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
		}
		auto& slot = At(index);
		EXIT_IF(slot.value.has_value());
		slot.value.emplace(std::forward<Args>(args)...);
		slot.alive.store(true, std::memory_order_release);
		if (index == m_capacity.load(std::memory_order_relaxed)) m_capacity.store(index + 1, std::memory_order_release);
		++m_size;
		return SlotId {index, slot.generation.load(std::memory_order_relaxed)};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		auto& slot = At(id.index);
		slot.alive.store(false, std::memory_order_release);
		const auto generation = slot.generation.load(std::memory_order_relaxed) + 1;
		slot.generation.store(generation == 0 ? 1 : generation, std::memory_order_release);
		slot.value.reset();
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_capacity.load(std::memory_order_acquire); }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0, count = m_capacity.load(std::memory_order_acquire); index < count; ++index) {
			auto& slot = At(index);
			if (slot.value) fn(SlotId {index, slot.generation.load(std::memory_order_relaxed)}, *slot.value);
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0, count = m_capacity.load(std::memory_order_acquire); index < count; ++index) {
			const auto& slot = At(index);
			if (slot.value) fn(SlotId {index, slot.generation.load(std::memory_order_relaxed)}, *slot.value);
		}
	}

private:
	// (The liveness check reads generation and alive, then the caller reads the object: in front of it they share its
	// first cache line. After it, an object of hundreds of bytes put them on a line of their own: a second cold miss for
	// every lookup, ~3% of the GPU thread at 1-1.)
	struct Slot {
		std::atomic<uint32_t> generation {1};
		std::atomic<bool>     alive {false};
		std::optional<T>      value;
	};
	static constexpr uint32_t ChunkBits = 10, ChunkSlots = 1u << ChunkBits, MaxChunks = 1u << 12;

	[[nodiscard]] Slot& At(uint32_t index) const noexcept {
		return m_chunks[index >> ChunkBits].load(std::memory_order_acquire)[index & (ChunkSlots - 1)];
	}

	std::unique_ptr<std::atomic<Slot*>[]> m_chunks = std::make_unique<std::atomic<Slot*>[]>(MaxChunks);
	std::atomic<uint32_t>                 m_capacity {0};
	std::vector<uint32_t>                 m_free_list;
	size_t                                m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
