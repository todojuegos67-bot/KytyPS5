#pragma once
// (parpadeo7) KYTY_PARPADEO7=1: the XPR chunk bits read back as soon as the label that hands them to the game is
// translated, not when the game reads them; KYTY_PARPADEO6's settle (0xa1f290's entry) stays as the net. Implies
// KYTY_PARPADEO6=1 (local/parpadeo6.h).
//
// Why (native/parpadeo7/INFORME.md):
//   - parpadeo6's settle made every 0xa1f290 call wait until the bits' pages had the GPU's bytes (no flicker since). Its
//     line: ~3,200-3,400 waits every 5 s, ~1.7-1.9 ms each, almost all "read back": each settle that met a GPU-owned
//     range asked a readback of its own (BufferCache::ReadMemory), whose copy is recorded behind everything translated
//     by then and whose bytes come in only once the GPU ran all of it. Several per frame, one after the other on the
//     pass threads (each pass reads the bits of its own slots, in other pages).
//   - The game reads them right after the XPR label: the RELEASE_MEM that sets *(view+0x120)+0x120 (shadows) or +0x1c0
//     (main) to 1 after the culling dispatches of 0xa1f930 / 0xa20750, which write each section's bits; the addresses are
//     in the frame's table *(view+0x58) + section*0x2d0 + slot*8 (written by the same job right after it records the
//     dispatch: parpadeo3/ghidra, 0xa1f930). The emulator writes the label when it translates the packet.
// Here:
//   - Learn (the game thread, 0xa1f290's hook): per view and label (its offset from *(view+0x120)), the (section, slot,
//     words) the game reads after it: the bits slot and, on the 0xad92f0 path, mask2's.
//   - At the label (the translating thread, CommandProcessor::WriteAtEndOfPipe, a 1 written as data): the view's ranges
//     from this frame's table, and one command queued for the GPU thread (run between packets, never inside the packet).
//   - That command (GpuResourceManager::PrefetchGuestReads): the ranges, sorted and joined, each begun as an asynchronous
//     guest readback (BufferCache::PrefetchGuestRead: the copy right behind the culling work, the GPU thread never waits:
//     nothing when a pending readback overlaps, no slot is free or no page of it is GPU-written).
//   - A helper thread copies each one into guest memory as soon as the GPU is done (CopyGuestReadback, as a reader
//     would), the GPU thread retires it (a command, never waited for), and the helper applies its pages' deferred
//     releases (the put-in of kept bytes, done there instead of by the game's reader).
//   - The game's settle then finds the bytes in (no readback, no wait), or the readback in flight (it waits for that
//     one, begun earlier, instead of one of its own).
// Locks: the GPU side only try-locks the registry (a busy one: that label is skipped, counted); the helper never waits
// for the GPU thread (it posts to it); the GPU thread may wait for a copy the helper claimed, as for a reader's.
//
// Line "[parpadeo7]" every 5 s (beside "[parpadeo6]", whose ESPERAS are what this removes).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Parpadeo7 {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		// (esperas1, 11-25) On by default: KYTY_PARPADEO7=0 turns it off. Ghidra: 0xab6080 waits for the label with
		// 0x8351b0 (a spin on the label == 1, sceKernelUsleep every 1000 spins) and only then loops 0xa1f290 over the
		// sections: on PS5 the game waits for THIS frame's culling (the label and the bits' table are in *(view+0x58), the
		// frame's table), so last frame's bits cannot be served (old bits = the flicker). What is left to remove is the
		// emulator's own cost: one readback per section, each behind all the work translated by then (log
		// 20261125-092605-970: ~2,900 readbacks / 5 s, ~2 ms each, 1.1-1.25 s of waits per second). Here the bits of the
		// whole label are read back in one asynchronous copy right behind the culling; the settle stays as the net.
		const char* v = std::getenv("KYTY_PARPADEO7");
		// (esperas2, 11-25) Off by default again: KYTY_PARPADEO7=1 turns it on. Log 20261125-104537-645 (all on): the game's
		// settle still waits for 2,000-2,500 readbacks every 5 s (0.88-1.5 s/s summed over the pass threads), as without it;
		// "precargado y aun en vuelo" ~= those waits and label->bytes (1.9 ms) ~= each wait (1.8 ms): the game reads right
		// after the label and the prefetch's copy is behind the same GPU work as its own readback, so it removes nothing and
		// costs the GPU thread, a helper thread and ~200-290 detached copies / 5 s. parpadeo6's settle stays on by itself.
		return v != nullptr && v[0] == '1' && v[1] == 0;
	}();
	return on;
}

inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// BufferCache::PrefetchGuestRead's result.
enum PrefetchResult : uint32_t {
	PrefetchBegun    = 0, // a new readback, handed to the helper
	PrefetchInflight = 1, // a pending readback overlaps the range already
	PrefetchNoSlot   = 2, // every readback slot pending (retiring one would wait)
	PrefetchNothing  = 3, // no GPU-written page left in it
	PrefetchRefused  = 4, // the asynchronous readback refused it (size, backing, image, capacity)
	PrefetchOff      = 5  // not the GPU thread, or guest readbacks off
};

// ---- What the game reads after each label (learned in 0xa1f290's hook) ----------------------------------------------
struct Slot {
	uint32_t words = 0;
	uint64_t seen  = 0; // the entry's label count when the game last read it
};
struct Entry {
	uint64_t                           view       = 0;
	int64_t                            off        = INT64_MIN; // label - *(view+0x120) (INT64_MIN: not near it)
	uint64_t                           last_label = 0;          // the label address last seen
	uint64_t                           labels     = 0;          // labels matched
	std::unordered_map<uint64_t, Slot> slots;                   // (section << 32 | slot) -> words
	// (lab1 p7b2) The view's last frames: (*(view+0x120), *(view+0x58)) as the game had them when it read. The label
	// is translated a frame or two after the game set them up, by then view+0x120/0x58 name the next frame's block and
	// table (0xa0d2d0 makes both anew each frame), so the label is matched against these and the slots read from the
	// table of ITS frame.
	uint64_t hist_base[8] {}, hist_table[8] {};
	uint32_t hist_next = 0;
	uint64_t used      = 0; // (p7c) g_use_seq at the last Learn: the oldest entry is replaced when the registry is full
};
inline uint64_t g_use_seq = 0; // (p7c) under g_mutex
// (lab1 p7b2) Learn calls with the label within 0x1000 of *(view+0x120) (offset known) or not.
inline std::atomic<uint64_t> g_learn_off {0}, g_learn_nooff {0}, g_hist_hits {0};
constexpr size_t   MaxEntries = 16;
constexpr size_t   MaxSlots   = 4096;
constexpr uint64_t StaleAfter = 8; // labels of the entry without the game reading the slot: dropped

inline std::mutex           g_mutex;
inline std::vector<Entry>   g_entries;
inline std::atomic<size_t>  g_entry_count {0};

// The game thread (the hook), after it found the call's ranges. `base`: *(view+0x120) (0: unread).
inline void Learn(uint64_t view, uint64_t label, uint64_t base, uint32_t section, uint32_t slot, uint32_t words,
                  uint64_t table = 0) noexcept {
	if (view == 0 || label == 0 || words == 0) return;
	try {
		const int64_t   off = base != 0 && label >= base && label - base < 0x1000 ? static_cast<int64_t>(label - base) : INT64_MIN;
		std::lock_guard lock(g_mutex);
		Entry*          entry = nullptr;
		(off != INT64_MIN ? g_learn_off : g_learn_nooff).fetch_add(1, std::memory_order_relaxed);
		// (lab1 p7b2) One entry per view and offset; an unknown offset joins the view's entry (it was one entry per
		// frame before: the label is new each frame, and the 16 entries filled up with dead ones).
		for (auto& e: g_entries)
			if (e.view == view && (e.last_label == label || e.off == off || off == INT64_MIN)) {
				entry = &e;
				break;
			}
		if (entry == nullptr) {
			// (p7c) Full: the least recently learned entry is replaced (log 20261125-094604-438: the 16 entries stayed full
			// with views the game no longer reads, the new ones were never learned: "aprendido 16 ... rangos 0").
			if (g_entries.size() >= MaxEntries) {
				entry = &*std::min_element(g_entries.begin(), g_entries.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
				*entry = Entry {};
			} else {
				g_entries.emplace_back();
				entry = &g_entries.back();
				g_entry_count.store(g_entries.size(), std::memory_order_release);
			}
			entry->view = view;
		}
		entry->used = ++g_use_seq;
		if (off != INT64_MIN || entry->off == INT64_MIN) entry->off = off;
		entry->last_label = label;
		if (base != 0 && table >= 0x10000) {
			const uint32_t last = (entry->hist_next + 7u) & 7u;
			if (entry->hist_base[last] != base || entry->hist_table[last] != table) {
				entry->hist_base[entry->hist_next]  = base;
				entry->hist_table[entry->hist_next] = table;
				entry->hist_next                    = (entry->hist_next + 1u) & 7u;
			}
		}
		const uint64_t key = uint64_t {section} << 32u | slot;
		auto           it  = entry->slots.find(key);
		if (it == entry->slots.end()) {
			if (entry->slots.size() >= MaxSlots) return;
			it = entry->slots.emplace(key, Slot {}).first;
		}
		it->second.words = std::max(it->second.words, words);
		it->second.seen  = entry->labels;
	} catch (...) {
	}
}

struct Counters {
	std::atomic<uint64_t> labels {0};          // labels written with 1 (data) while something was learned
	std::atomic<uint64_t> matched {0};         // ... an XPR label of a learned view
	std::atomic<uint64_t> busy {0};            // ... skipped: the registry was locked by a game thread
	std::atomic<uint64_t> other_frame {0};     // ... skipped: *(view+0x58) is another frame's table already
	std::atomic<uint64_t> ranges {0};          // ranges named at matched labels
	std::atomic<uint64_t> spans {0};           // ... joined
	std::atomic<uint64_t> begun {0};           // readbacks begun
	std::atomic<uint64_t> inflight {0};        // spans a pending readback overlapped already
	std::atomic<uint64_t> no_slot {0};         // spans with every slot pending
	std::atomic<uint64_t> not_gpu {0};         // spans with no GPU-written page (unmapped, CPU-owned, read already)
	std::atomic<uint64_t> refused {0};         // spans the asynchronous readback refused
	std::atomic<uint64_t> gpu_us {0};          // the GPU thread's time on the commands (us)
	std::atomic<uint64_t> copied {0};          // the helper's copies done
	std::atomic<uint64_t> detached {0};        // ... that wrote nothing (detached: a GPU write came first)
	std::atomic<uint64_t> copy_us {0};         // label -> bytes in, summed (us)
	std::atomic<uint64_t> copy_max_us {0};
	std::atomic<uint64_t> applied {0};         // deferred releases applied by the helper (readbacks)
	std::atomic<uint64_t> settle_in {0};       // game settles of a prefetched range: bytes in before it came
	std::atomic<uint64_t> settle_late {0};     // ... it came first (waited for the prefetch's copy)
	std::atomic<uint64_t> settle_miss {0};     // game settles that read back a range nothing prefetched
	std::atomic<uint64_t> label_to_game_us {0}; // label -> the game's settle, prefetched ranges (us)
	std::atomic<uint64_t> label_to_game_n {0};
};
inline Counters g_c;

inline void Max(std::atomic<uint64_t>& m, uint64_t v) noexcept {
	uint64_t cur = m.load(std::memory_order_relaxed);
	while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
	}
}

// The translating thread, a label written with value 1 as data. `read` (uint64_t address, uint64_t* out) -> bool: a
// guarded guest read. Fills `out` with the ranges ({address, bytes}) the game will read after it; true when it is an XPR
// label. Never waits: a registry a game thread holds is skipped.
template <typename Read>
bool MatchLabel(uint64_t label, Read&& read, std::vector<std::pair<uint64_t, uint64_t>>& out) noexcept {
	out.clear();
	if (g_entry_count.load(std::memory_order_acquire) == 0) return false;
	g_c.labels.fetch_add(1, std::memory_order_relaxed);
	std::unique_lock lock(g_mutex, std::try_to_lock);
	if (!lock.owns_lock()) {
		g_c.busy.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	try {
		bool matched_any = false;
		for (auto& e: g_entries) {
			// (lab1 p7b2) The label against the view's last frames: b + its offset, or the two XPR labels 0x912960 writes
			// (0xa0b020: *(view+0x120)+0x1c0; 0xa0b120 shadows: +0x120; parpadeo3 Ghidra). The slots come from the
			// table of that frame, checked as before: *(table+0x2cd30) == b + 0x120 (0xa0d2d0).
			// (p7c) Log 20261125-094604-438: "tabla de otro frame" ~= "halladas en el historial" (4028 / 5 s). The history is
			// learned in 0xa1f290, i.e. AFTER each frame's label, so at the label of frame N it holds frames <= N-1 only;
			// the label block *(view+0x120) is reused across frames while the table is new each frame (0xa0d2d0), so the
			// first history slot with that base named an old table, failed the check and the live *(view+0x58) (frame N's)
			// was never tried. Now: every candidate is checked, newest first, then the live table; the first table whose
			// *(table+0x2cd30) == b + 0x120 (0xa0d2d0 writes it so) is the label's frame.
			uint64_t table = 0, b = 0;
			bool     from_hist = false;
			const auto valid = [&](uint64_t t, uint64_t base) {
				uint64_t first = 0;
				return t >= 0x10000 && base != 0 && read(t + 0x2cd30, &first) && first == base + 0x120;
			};
			for (uint32_t k = 0; k < 8u && table == 0; ++k) {
				const uint32_t h  = (e.hist_next + 7u - k) & 7u; // newest first
				const uint64_t hb = e.hist_base[h];
				if (hb == 0) continue;
				if ((e.off != INT64_MIN && hb + static_cast<uint64_t>(e.off) == label) || hb + 0x120u == label || hb + 0x1c0u == label) {
					if (b == 0) b = hb;
					if (valid(e.hist_table[h], hb)) {
						table     = e.hist_table[h];
						from_hist = true;
					}
				}
			}
			if (table == 0) {
				// The live view (the frame the game is building now).
				uint64_t lb = 0, lt = 0;
				const bool have_lb = read(e.view + 0x120, &lb) && lb != 0;
				if (b == 0) {
					bool hit = e.last_label == label;
					if (have_lb && (lb + 0x120u == label || lb + 0x1c0u == label || (e.off != INT64_MIN && lb + static_cast<uint64_t>(e.off) == label))) hit = true;
					if (!hit) continue;
					b = have_lb && (lb + 0x120u == label || lb + 0x1c0u == label || (e.off != INT64_MIN && lb + static_cast<uint64_t>(e.off) == label))
					        ? lb
					        : (e.off != INT64_MIN ? label - static_cast<uint64_t>(e.off) : 0);
				}
				if (!read(e.view + 0x58, &lt) || !valid(lt, b)) {
					g_c.other_frame.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
				table = lt;
			}
			if (from_hist) g_hist_hits.fetch_add(1, std::memory_order_relaxed);
			++e.labels;
			for (auto it = e.slots.begin(); it != e.slots.end();) {
				if (e.labels - it->second.seen > StaleAfter) {
					it = e.slots.erase(it);
					continue;
				}
				const uint64_t section = it->first >> 32u, slot = it->first & 0xffffffffu;
				uint64_t       at      = 0;
				if (read(table + section * 0x2d0u + slot * 8u, &at) && at >= 0x10000) out.emplace_back(at, uint64_t {it->second.words} * 8u);
				++it;
			}
			matched_any = true;
			// (p7c) No return here: the passes 40..45 wait on the SAME label (one per view entry); all their ranges go in
			// one readback (it used to stop at the first entry, often one with no live slots: "del XPR 35 ... rangos 0").
		}
		if (matched_any) {
			g_c.matched.fetch_add(1, std::memory_order_relaxed);
			g_c.ranges.fetch_add(out.size(), std::memory_order_relaxed);
		}
		return !out.empty();
	} catch (...) {
		out.clear();
	}
	return false;
}

// Sorted and joined: ranges less than 4 KiB apart in one span, spans at most 256 KiB (a guest readback's request is at
// most 512 KiB; it widens it to its 512 KiB window itself).
inline void JoinRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	std::sort(ranges.begin(), ranges.end());
	size_t n = 0;
	for (const auto& [at, bytes]: ranges) {
		if (bytes == 0) continue;
		if (n != 0) {
			auto&          last = ranges[n - 1];
			const uint64_t end  = std::max(last.first + last.second, at + bytes);
			if (at <= last.first + last.second + 4096u && end - last.first <= 256u * 1024u) {
				last.second = end - last.first;
				continue;
			}
		}
		ranges[n++] = {at, bytes};
	}
	ranges.resize(n);
}

// ---- The prefetched spans (diagnostics: what the game's settle met) --------------------------------------------------
struct Recent {
	std::atomic<uint64_t> begin {0}, end {0};
	std::atomic<int64_t>  label_ns {0}, copied_ns {0};
};
constexpr size_t                  RecentCount = 1024;
inline std::array<Recent, RecentCount> g_recent;
inline std::atomic<uint64_t>      g_recent_next {0};

// The GPU thread: a span begun (or met in flight). Returns its index + 1 (for NoteCopied).
inline uint64_t NotePrefetched(uint64_t begin, uint64_t end, int64_t label_ns) noexcept {
	const uint64_t i = g_recent_next.fetch_add(1, std::memory_order_relaxed);
	auto&          r = g_recent[i % RecentCount];
	r.begin.store(0, std::memory_order_relaxed);
	r.copied_ns.store(0, std::memory_order_relaxed);
	r.label_ns.store(label_ns, std::memory_order_relaxed);
	r.end.store(end, std::memory_order_relaxed);
	r.begin.store(begin, std::memory_order_release);
	return i + 1;
}
// The helper: the span's bytes are in (`label_ns`: its label's time, for the latency).
inline void NoteCopied(uint64_t index, int64_t label_ns, bool wrote) noexcept {
	const int64_t now = NowNs();
	if (index != 0) g_recent[(index - 1) % RecentCount].copied_ns.store(wrote ? now : -1, std::memory_order_release);
	g_c.copied.fetch_add(1, std::memory_order_relaxed);
	if (!wrote) g_c.detached.fetch_add(1, std::memory_order_relaxed);
	const uint64_t us = now > label_ns ? static_cast<uint64_t>(now - label_ns) / 1000u : 0;
	g_c.copy_us.fetch_add(us, std::memory_order_relaxed);
	Max(g_c.copy_max_us, us);
}
// The game thread, after its settle of [at, at + bytes): `read_back` whether the settle asked a readback.
inline void NoteSettle(uint64_t at, uint64_t bytes, bool read_back) noexcept {
	const int64_t now      = NowNs();
	const uint64_t last    = g_recent_next.load(std::memory_order_acquire);
	const uint64_t first   = last > RecentCount ? last - RecentCount : 0;
	int64_t        label   = 0, copied = 0;
	for (uint64_t i = last; i > first; --i) {
		const auto& r = g_recent[(i - 1) % RecentCount];
		const auto  b = r.begin.load(std::memory_order_acquire);
		if (b == 0 || b >= at + bytes || at >= r.end.load(std::memory_order_relaxed)) continue;
		label  = r.label_ns.load(std::memory_order_relaxed);
		copied = r.copied_ns.load(std::memory_order_relaxed);
		break;
	}
	if (label == 0) {
		if (read_back) g_c.settle_miss.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (read_back || copied <= 0 || copied > now) g_c.settle_late.fetch_add(1, std::memory_order_relaxed);
	else g_c.settle_in.fetch_add(1, std::memory_order_relaxed);
	if (now > label && now - label < 1'000'000'000) {
		g_c.label_to_game_us.fetch_add(static_cast<uint64_t>(now - label) / 1000u, std::memory_order_relaxed);
		g_c.label_to_game_n.fetch_add(1, std::memory_order_relaxed);
	}
}

// The 5 s line: printed by the first game thread past the 5 s (one CAS).
inline std::atomic<int64_t> g_last_ns {0};
inline void MaybeReport() noexcept {
	const int64_t now  = NowNs();
	int64_t       last = g_last_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 5'000'000'000ll || !g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	auto&      c    = g_c;
	const auto take = [](std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.exchange(0, std::memory_order_relaxed)); };
	const auto labels = take(c.labels), matched = take(c.matched), busy = take(c.busy), other = take(c.other_frame), ranges = take(c.ranges),
	           spans = take(c.spans), begun = take(c.begun), inflight = take(c.inflight), no_slot = take(c.no_slot),
	           not_gpu = take(c.not_gpu), refused = take(c.refused), gpu_us = take(c.gpu_us), copied = take(c.copied),
	           detached = take(c.detached), copy_us = take(c.copy_us), copy_max = take(c.copy_max_us),
	           applied = take(c.applied), s_in = take(c.settle_in), s_late = take(c.settle_late),
	           s_miss = take(c.settle_miss), l2g_us = take(c.label_to_game_us), l2g_n = take(c.label_to_game_n);
	size_t learned = 0, views = 0;
	{
		std::unique_lock lock(g_mutex, std::try_to_lock);
		if (lock.owns_lock()) {
			views = g_entries.size();
			for (const auto& e: g_entries) learned += e.slots.size();
		}
	}
	const double seconds = static_cast<double>(now - last) / 1e9;
	std::printf("[parpadeo7] %.1f s (KYTY_PARPADEO7=1, bits del XPR leidos de vuelta al traducir su etiqueta): aprendido %zu "
	            "rangos en %zu etiquetas de vista; etiquetas a 1: %llu, del XPR %llu (saltadas: registro ocupado %llu, tabla de otro frame %llu); rangos "
	            "%llu -> %llu tramos: precargas %llu, ya en vuelo %llu, sin hueco %llu, nada de GPU %llu, rechazadas %llu; "
	            "hilo GPU %.2f ms (%.3f ms/s); copias del ayudante %llu (sin escribir %llu), etiqueta->bytes dentro media "
	            "%.0f us max %llu us; liberaciones aplicadas %llu; EL JUEGO: precargado y ya dentro %llu, precargado y aun "
	            "en vuelo %llu, SIN PRECARGA (readback propio) %llu; etiqueta->juego media %.0f us\n",
	            seconds, learned, views, labels, matched, busy, other, ranges, spans, begun, inflight, no_slot, not_gpu, refused,
	            static_cast<double>(gpu_us) / 1000.0, seconds > 0.0 ? static_cast<double>(gpu_us) / 1000.0 / seconds : 0.0,
	            copied, detached, copied != 0 ? static_cast<double>(copy_us) / static_cast<double>(copied) : 0.0, copy_max,
	            applied, s_in, s_late, s_miss, l2g_n != 0 ? static_cast<double>(l2g_us) / static_cast<double>(l2g_n) : 0.0);
	std::fflush(stdout);
	// (lab1 p7b2) How the matching went: the learn calls with a known offset, and the labels found in the history.
	std::printf("[p7b2] aprendizaje: con offset %llu, sin offset %llu; etiquetas halladas en el historial de frames %llu\n",
	            static_cast<unsigned long long>(g_learn_off.exchange(0, std::memory_order_relaxed)),
	            static_cast<unsigned long long>(g_learn_nooff.exchange(0, std::memory_order_relaxed)),
	            static_cast<unsigned long long>(g_hist_hits.exchange(0, std::memory_order_relaxed)));
	std::fflush(stdout);
}

} // namespace Parpadeo7
