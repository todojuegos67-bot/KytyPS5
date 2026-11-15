#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PREFLUSHTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PREFLUSHTABLE_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The bookkeeping of the buffer cache's pre-flush (Ryujinx's BufferPreFlush). A game that reads what its shaders
// wrote reads the same pages again a frame later. Each such page gets a host slot, and every command buffer that
// writes the page copies it there at its end. A read back of the page then waits only for that command buffer, and
// copies from the slot, instead of recording a download behind all the work being recorded and waiting for it.
//
// Exact by ticks (a tick is a command buffer): a slot serves its page only when it was copied at the end of a
// command buffer no older than the page's last GPU write. A write after the copy, or in the command buffer still
// being recorded, makes the page miss until a later command buffer copies it again. Pages are in tracker-page units;
// slots are indices the caller maps to memory. GPU thread only.
class PreflushTable {
public:
	// A page is copied at most `max_copies` times between two read backs: one written again and again but rarely
	// read costs copies and serves nothing (Marvel's Wolverine's Japan title: ~13 copies a hit), so after that it
	// misses until it is read back again. Most pages that are read have one writer between two reads.
	explicit PreflushTable(uint32_t max_slots, uint32_t max_copies = 2): m_max_slots(max_slots), m_max_copies(max_copies) {}

	// The page was read back in `frame` (downloaded, or served from its slot): it gets a slot if one is free, and
	// counts as unwritten from now on.
	void OnReadBack(uint64_t page, uint64_t frame) {
		auto it = m_pages.find(page);
		if (it == m_pages.end()) {
			uint32_t slot = 0;
			if (!m_free.empty()) {
				slot = m_free.back();
				m_free.pop_back();
			} else if (m_next_slot < m_max_slots) {
				slot = m_next_slot++;
			} else {
				return;
			}
			it = m_pages.emplace(page, Page {.slot = slot}).first;
		}
		it->second.read_frame = frame;
		it->second.write_tick = 0;
		it->second.copy_tick  = 0;
		it->second.copies     = 0;
	}

	// A GPU write of the pages [first, last] recorded in the command buffer `tick`. Only the range is kept: a draw
	// may write a whole buffer that holds thousands of history pages, and draws come by the thousand a frame. The
	// pages are stamped once, at the command buffer's end.
	void OnWrite(uint64_t first, uint64_t last, uint64_t tick) {
		if (tick != m_written_tick) {
			Apply(); // a command buffer whose end was not seen: a later end copies its pages
		}
		m_written_tick = tick;
		m_written.Add(first, last - first + 1);
	}

	// At the end of the command buffer `tick`, after all its writes: copy(page, slot) records the copy of each page
	// written since the last call and returns whether it did; a copied page is stamped with `tick`. A page past its
	// copy limit leaves the table (it misses, and is walked no more) until it is read back again.
	template <typename Copy>
	void TakeCopies(uint64_t tick, Copy&& copy) {
		Apply();
		for (const auto address: m_pending) {
			const auto it = m_pages.find(address);
			if (it == m_pages.end() || !it->second.pending) {
				continue; // expired since, or read back and taken again
			}
			auto& page   = it->second;
			page.pending = false;
			if (page.write_tick == 0) {
				continue;
			}
			if (page.copies >= m_max_copies) {
				m_free.push_back(page.slot);
				m_pages.erase(it);
			} else if (copy(address, page.slot)) {
				page.copy_tick = tick;
				page.copies++;
			}
		}
		m_pending.clear();
	}

	// The page's slot and the command buffer whose end copied it, when that copy holds the page's last GPU write.
	[[nodiscard]] std::optional<std::pair<uint32_t, uint64_t>> Lookup(uint64_t page) const {
		if (m_written.Intersects(page, 1)) {
			return std::nullopt; // written in the command buffer being recorded
		}
		const auto it = m_pages.find(page);
		if (it == m_pages.end() || it->second.write_tick == 0 || it->second.copy_tick < it->second.write_tick) {
			return std::nullopt;
		}
		return std::pair {it->second.slot, it->second.copy_tick};
	}

	// A new frame: pages not read back in it or the one before give their slots back.
	void Expire(uint64_t frame) {
		std::erase_if(m_pages, [&](const auto& entry) {
			if (entry.second.read_frame + 1 >= frame) {
				return false;
			}
			m_free.push_back(entry.second.slot);
			return true;
		});
	}

	[[nodiscard]] size_t Pages() const noexcept { return m_pages.size(); }
	[[nodiscard]] uint32_t SlotsUsed() const noexcept { return m_next_slot; }

private:
	// Stamps the history pages the writes of m_written_tick reached.
	void Apply() {
		m_written.ForEach([&](uint64_t first, uint64_t end) {
			for (auto it = m_pages.lower_bound(first); it != m_pages.end() && it->first < end; ++it) {
				it->second.write_tick = m_written_tick;
				if (!it->second.pending) {
					it->second.pending = true;
					m_pending.push_back(it->first);
				}
			}
		});
		m_written.Clear();
	}

	struct Page {
		uint64_t read_frame = 0;
		uint64_t write_tick = 0; // 0: not GPU-written since its last read back
		uint64_t copy_tick  = 0;
		uint32_t slot       = 0;
		uint32_t copies     = 0;     // since its last read back
		bool     pending    = false; // written since the last TakeCopies
	};
	std::map<uint64_t, Page> m_pages;
	RangeSet                 m_written; // pages, written in the command buffer m_written_tick
	uint64_t                 m_written_tick = 0;
	std::vector<uint64_t>    m_pending;
	std::vector<uint32_t>    m_free;
	uint32_t                 m_next_slot = 0;
	uint32_t                 m_max_slots;
	uint32_t                 m_max_copies;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PREFLUSHTABLE_H_
