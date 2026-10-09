#ifndef EMULATOR_SRC_COMMON_LRUCACHE_H_
#define EMULATOR_SRC_COMMON_LRUCACHE_H_

#include <cstddef>
#include <deque>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace Common {

template <typename Object, typename Tick>
class LeastRecentlyUsedCache {
	struct Item {
		Object object {};
		Tick   tick {};
		Item*  next = nullptr;
		Item*  prev = nullptr;
	};

public:
	[[nodiscard]] size_t Insert(Object object, Tick tick) {
		const auto id   = Build();
		auto&      item = At(id);
		item.object     = std::move(object);
		item.tick       = tick;
		Attach(item);
		return id;
	}

	void Touch(size_t id, Tick tick) {
		auto& item = At(id);
		if (item.tick >= tick) {
			return;
		}
		item.tick = tick;
		if (&item != m_last) {
			Detach(item);
			Attach(item);
		}
	}

	[[nodiscard]] Tick TickOf(size_t id) { return At(id).tick; }

	void Free(size_t id) {
		auto& item = At(id);
		Detach(item);
		item.next = nullptr;
		item.prev = nullptr;
		m_free.push_back(id);
	}

	template <typename Function>
	void ForEachItemBelow(Tick tick, Function&& function) {
		constexpr bool ReturnsBool = std::is_same_v<std::invoke_result_t<Function, Object>, bool>;
		for (auto* item = m_first; item != nullptr;) {
			if (item->tick > tick) {
				return;
			}
			auto* next = item->next;
			if constexpr (ReturnsBool) {
				if (function(item->object)) {
					return;
				}
			} else {
				function(item->object);
			}
			item = next;
		}
	}

private:
	// Items in fixed chunks, which never move (the list links point into them). A std::deque of these 32-byte items
	// keeps one per block on MSVC: every index was a map lookup and another heap line (~1.4% of the render thread).
	static constexpr size_t ChunkItems = 1024;

	[[nodiscard]] Item& At(size_t id) { return m_chunks[id / ChunkItems][id % ChunkItems]; }

	[[nodiscard]] size_t Build() {
		if (m_free.empty()) {
			const auto id = m_size++;
			if (id % ChunkItems == 0) m_chunks.push_back(std::make_unique<Item[]>(ChunkItems));
			return id;
		}
		const auto id = m_free.front();
		m_free.pop_front();
		return id;
	}

	void Attach(Item& item) {
		if (m_first == nullptr) {
			m_first = &item;
		}
		if (m_last == nullptr) {
			m_last = &item;
			return;
		}
		item.prev    = m_last;
		m_last->next = &item;
		item.next    = nullptr;
		m_last       = &item;
	}

	void Detach(Item& item) {
		if (item.prev != nullptr) {
			item.prev->next = item.next;
		}
		if (item.next != nullptr) {
			item.next->prev = item.prev;
		}
		if (m_first == &item) {
			m_first = item.next;
		}
		if (m_last == &item) {
			m_last = item.prev;
		}
	}

	std::vector<std::unique_ptr<Item[]>> m_chunks;
	size_t                               m_size = 0;
	std::deque<size_t>                   m_free;
	Item*              m_first = nullptr;
	Item*              m_last  = nullptr;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LRUCACHE_H_
