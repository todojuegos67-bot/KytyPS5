#ifndef EMULATOR_SRC_LOCAL_COLD_STORE_H_
#define EMULATOR_SRC_LOCAL_COLD_STORE_H_

// (ram11) KYTY_COLD_STORE=1 (off by default; only with KYTY_RAIZ=1 - the guest memory private -; read once, in its
// thread, 10 s after the start).
//
// With KYTY_RAIZ the game's memory is private: what Windows trims of it goes to its memory compression store, in RAM
// (logs 11-16 00:50 and 01:26, 16 GB: the emulator 3.7-4.8 GB in RAM, compression 5.4-6.2 GB, available 0.4-0.6 GB),
// never to disk. Here its cold part goes to a file of the emulator's own and its RAM is given back (decommitted: no
// compression, no commit charge), and comes back at its first touch.
//
// (ram12) The probe no longer makes pages NO-ACCESS: raiz reads a no-access page as "GPU-written, a tracker's" - a
// write-back over it is KEPT (lazy) instead of written, a read window opens it and fills a shadow, a window restores it
// to no-access after the copy - so a probed granule could take a GPU result the game never saw, or be left no-access
// with no owner (crash 11-16 01:48: eboot+0x8120b8, TaggedFrameAllocator, a null read from its free list, with only the
// probe acting). Now a probed page keeps its protection plus PAGE_GUARD: raiz sees its own protection (the low byte),
// keeps nothing for it, opens no window for it; any touch raises a guard violation that clears the guard of that page by
// itself (Windows), and nothing of ours ever puts a saved protection back over one somebody else set:
//
//  1. Probe. While available physical memory is under KYTY_COLD_STORE_BELOW_MB (2048): granules of 64 KiB of the game's
//     mappings - all 16 pages committed, private, of one allocation, of a plain protection (read-write, read-only,
//     execute+read/write; no guard), no GPU tracker's watch on them (the read and write hints), no lazy write-back
//     kept, not in the game's per-frame memory (KYTY_COLD_STORE_EXCLUDE: the labels DefaultTagg, ctx, ring and the
//     render-target arenas, where the map knows them), not touched in the last KYTY_COLD_STORE_HOT_S (120) s - get
//     PAGE_GUARD added to their protection. With the protection gate taken alone (no window open, no protection change).
//  2. A touch (any thread, guest or emulator code) raises a guard violation; the vectored handler here (first in the
//     chain) clears the guard of the granule's other pages - each one only if its protection is still ours, and a
//     protection set by somebody else in between is put back -, forgets the granule (hot) and the access runs again.
//     A protection change of guest memory (ProtectMappedUnlocked, gate held) and a kernel-mode access
//     (GuestLazyCommit::Prefault) clear them first the same way (Ensure).
//  3. Evict. Still probed after KYTY_COLD_STORE_IDLE_S (20) s, gate alone: each page's protection must still be exactly
//     its own + guard (nobody touched it or changed it); then read-only for the copy, written to the file, decommitted
//     (VirtualFree MEM_DECOMMIT; never DiscardVirtualMemory). At most KYTY_COLD_STORE_MB_S (64) MiB a second.
//  4. Back. A touch of an evicted granule faults on reserved memory; the handler commits it read-write, reads its
//     64 KiB from the file, gives each page its protection back (nobody can have changed a reserved page's: a protection
//     change brings it back first) and the access runs again. Ensure does the same.
//  5. Unmap (UnmapBacking): probed granules lose their guard, evicted ones are committed again read-write (zeros).
//
// The file: <exe folder or KYTY_COLD_STORE_DIR>\kyty-cold-<pid>.tmp, sparse, deleted on close, buffered.
// The table of granules is fixed in size (no allocation in the fault handler): 131,072 granules (8 GiB).
// Log, every 30 s (its thread): "Cold store (30 s): ...".

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "lock-watch.h" // (ram13b)

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ColdStore {

constexpr uint64_t Granule      = 0x10000;
constexpr uint64_t PageSize     = 0x1000;
constexpr uint32_t Pages        = static_cast<uint32_t>(Granule / PageSize);
constexpr uint32_t Capacity     = 1u << 17u; // granules in the table
constexpr uint32_t HotSlots     = 1u << 16u;
constexpr uint8_t  StateFree    = 0;
constexpr uint8_t  StateProbe   = 1;
constexpr uint8_t  StateCold    = 2;
constexpr uint8_t  StatePending = 3; // (ram13) decommitted, its bytes in a buffer of the worker's until the file has them
constexpr uint32_t Buffers      = 32; // granules evicted per batch

// The operating system's side (Windows by default; a test sets its own).
struct Os {
	// The 16 pages of the granule: false unless all committed, private and of a plain protection (each one's in prot).
	bool (*query)(uint64_t granule, uint32_t* prot)                     = nullptr;
	bool (*protect)(uint64_t at, uint64_t size, uint32_t prot, uint32_t* old) = nullptr; // set; the old one of the first page
	uint32_t (*page_prot)(uint64_t page)                                 = nullptr; // a committed page's protection now, 0 otherwise
	bool (*decommit)(uint64_t granule)                                   = nullptr;
	bool (*commit_rw)(uint64_t granule)                                  = nullptr; // committed read-write (zeros)
	bool (*write_file)(uint64_t offset, const void* data, uint64_t size) = nullptr;
	bool (*read_file)(uint64_t offset, void* data, uint64_t size)        = nullptr;
	uint64_t (*available)()                                              = nullptr; // physical memory available, bytes
	bool (*allows)(uint64_t address, bool write)                         = nullptr; // the page lets that access through now
	uint32_t prot_no_access = 0x01, prot_read_only = 0x02, prot_guard = 0x100;
};

// The emulator's side (memoryAddressSpace.inc sets it).
struct Hooks {
	bool (*tracked)(uint64_t vaddr, uint64_t size) = nullptr; // a GPU tracker's watch may be on it (a hint, superset)
	bool (*pending)(uint64_t vaddr, uint64_t size) = nullptr; // lazy write-back kept for it
	void (*gate_lock)()                            = nullptr; // the protection gate, alone
	void (*gate_unlock)()                          = nullptr;
	void (*gate_lock_shared)()                     = nullptr; // (ram13) the gate as a protection change takes it
	void (*gate_unlock_shared)()                   = nullptr;
	bool (*gpu_thread)()                           = nullptr;
	void (*ranges)(void* user, std::vector<std::pair<uint64_t, uint64_t>>* out) = nullptr; // the game's mappings
	void* ranges_user                              = nullptr;
	// (ram12) The game's per-frame memory (never cold: rewritten every frame, GPU results read back into it).
	bool (*excluded)(uint64_t vaddr, uint64_t size) = nullptr;
	// (raiz5) The emulator's shadow copies of the range's pages forgotten: its bytes are about to change (probed,
	// evicted, back). Called with this module's lock held (the shadows' lock after it, never the other way).
	void (*forget)(uint64_t vaddr, uint64_t size) = nullptr;
};

struct Entry {
	uint64_t key   = 0; // granule address, 0 = empty
	uint32_t slot  = 0; // file slot (cold)
	uint32_t since = 0; // seconds tick of the probe
	uint8_t  state = StateFree;
	uint8_t  buf   = 0; // (StatePending: the worker's buffer)
	uint8_t  prot[Pages] {};
};

struct Stats {
	std::atomic<uint64_t> probed {0}, evicted {0}, back_reads {0}, back_us {0}, back_us_max {0}, back_gpu {0},
	    probe_hits {0}, ensures {0}, dropped {0}, write_fail {0}, read_fail {0}, full {0};
	// (ram12) Pages whose protection somebody else changed while they were probed (theirs kept; at an evict: not
	// evicted), and protections of others put back after a guard clear crossed them.
	std::atomic<uint64_t> changed {0}, put_back {0}, excluded {0};
	// (ram13) What it costs the others: guard faults handled and the time in the handler (all faults it handled), the
	// gate held alone (evictions: every protection change of guest memory waits) and shared (probes: only windows
	// wait), the time to get it, and the work per granule.
	std::atomic<uint64_t> handler_us {0}, handler_us_max {0}, handled {0};
	std::atomic<uint64_t> gate_alone_us {0}, gate_alone_max_us {0}, gate_shared_us {0}, gate_wait_us {0};
	std::atomic<uint64_t> probe_work_us {0}, evict_work_us {0}, file_us {0};
};

struct State {
	std::mutex            mutex; // the table, the slots, the file's use
	Entry*                table = nullptr;
	uint32_t*             free_slots = nullptr;
	uint32_t              free_count = 0, next_slot = 0;
	std::atomic<uint32_t> count {0};  // entries in use (the handler's fast path)
	std::atomic<uint64_t> lo {~uint64_t {0}}, hi {0};
	std::atomic<uint32_t> hot_until[HotSlots] {};
	// Entries per bucket of granules and the tick of the bucket's last erase: the fault handler's look without the lock
	// (every fault of guest memory - the trackers' ~10,000 a second - passes here first).
	std::atomic<uint32_t> bucket[HotSlots] {};
	std::atomic<uint32_t> bucket_erased[HotSlots] {};
	// (ram12) A bit per 64 KiB granule of guest addresses (under 1 TiB) that ever had a guard of ours: a guard violation
	// there with no entry is a guard of ours left by a copy's window (it restores the protection it found) - running
	// the access again is right -, never a guest stack's guard (a granule with a guard page is never probed).
	std::atomic<uint64_t>* ever = nullptr;
	std::atomic<uint32_t> cold_now {0}, probe_now {0};
	Os                    os;
	uint8_t*              buffers = nullptr; // (ram13) Buffers x 64 KiB, the worker's
	std::atomic<uint64_t> per_s_now {0};     // (ram13) the adaptive rate, bytes a second
	Hooks                 hooks;
	Stats                 stats;
	std::atomic<bool>     on {false};
	std::atomic<uint32_t> tick {0}; // seconds since start (the worker)
	uint32_t              hot_s = 120; // KYTY_COLD_STORE_HOT_S
	uint32_t              recent_ticks = 2; // (a bucket a granule left counts as ours that long: 1-2 s)
};

inline State& Get() {
	static State* state = new State; // (never destroyed: the handler may run during exit)
	return *state;
}

// (ram13b) The table's lock, watched (LockWatch: who waits for it, who holds it).
class ColdLock {
public:
	explicit ColdLock(State& s): m_s(s) {
		LockWatch::Wait wait(LockWatch::ColdMutex);
		m_s.mutex.lock();
	}
	~ColdLock() {
		m_s.mutex.unlock();
		LockWatch::Release(LockWatch::ColdMutex);
	}
	ColdLock(const ColdLock&)            = delete;
	ColdLock& operator=(const ColdLock&) = delete;

private:
	State& m_s;
};

inline uint32_t Hash(uint64_t key) {
	return static_cast<uint32_t>(((key >> 16u) * 0x9E3779B97F4A7C15ull) >> 40u) & (Capacity - 1u);
}

// (mutex held) The entry of `key`, or null.
inline Entry* FindLocked(State& s, uint64_t key) {
	for (uint32_t i = Hash(key), n = 0; n < Capacity; i = (i + 1u) & (Capacity - 1u), ++n) {
		if (s.table[i].key == key) return &s.table[i];
		if (s.table[i].key == 0) return nullptr;
	}
	return nullptr;
}
// (mutex held) A new entry for `key` (null when the table is too full: kept under 3/4).
inline Entry* InsertLocked(State& s, uint64_t key) {
	if (s.count.load(std::memory_order_relaxed) >= Capacity / 4u * 3u) return nullptr;
	for (uint32_t i = Hash(key);; i = (i + 1u) & (Capacity - 1u)) {
		if (s.table[i].key == 0) {
			s.table[i]     = Entry {};
			s.table[i].key = key;
			s.count.fetch_add(1, std::memory_order_release);
			s.bucket[Hash(key) & (HotSlots - 1u)].fetch_add(1, std::memory_order_release);
			if (key < s.lo.load(std::memory_order_relaxed)) s.lo.store(key, std::memory_order_relaxed);
			if (key + Granule > s.hi.load(std::memory_order_relaxed)) s.hi.store(key + Granule, std::memory_order_relaxed);
			return &s.table[i];
		}
	}
}
// (mutex held) The entry out of the table (linear probing: the ones after it moved back).
inline void EraseLocked(State& s, Entry* e) {
	uint32_t i = static_cast<uint32_t>(e - s.table);
	const uint32_t b = Hash(e->key) & (HotSlots - 1u);
	s.bucket_erased[b].store(s.tick.load(std::memory_order_relaxed) + s.recent_ticks, std::memory_order_relaxed);
	s.bucket[b].fetch_sub(1, std::memory_order_release);
	s.table[i] = Entry {};
	s.count.fetch_sub(1, std::memory_order_release);
	for (uint32_t j = (i + 1u) & (Capacity - 1u); s.table[j].key != 0; j = (j + 1u) & (Capacity - 1u)) {
		const uint32_t home = Hash(s.table[j].key);
		// (Moves back when its home is not in (i, j] cyclically.)
		const bool between = i <= j ? (home > i && home <= j) : (home > i || home <= j);
		if (!between) {
			s.table[i] = s.table[j];
			s.table[j] = Entry {};
			i          = j;
		}
	}
}

inline void MarkHot(State& s, uint64_t key, uint32_t seconds) {
	s.hot_until[Hash(key) & (HotSlots - 1u)].store(s.tick.load(std::memory_order_relaxed) + seconds, std::memory_order_relaxed);
}
inline bool IsHot(State& s, uint64_t key) {
	return s.hot_until[Hash(key) & (HotSlots - 1u)].load(std::memory_order_relaxed) > s.tick.load(std::memory_order_relaxed);
}

// (mutex held) Each page's protection back (runs of equal ones).
inline void RestoreProtLocked(State& s, const Entry& e) {
	for (uint32_t p = 0; p < Pages;) {
		uint32_t q = p + 1;
		while (q < Pages && e.prot[q] == e.prot[p]) ++q;
		uint32_t old = 0;
		(void)s.os.protect(e.key + p * PageSize, (q - p) * PageSize, e.prot[p], &old);
		p = q;
	}
}

inline uint64_t NowUs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline void Max(std::atomic<uint64_t>& m, uint64_t v) {
	uint64_t now = m.load(std::memory_order_relaxed);
	while (v > now && !m.compare_exchange_weak(now, v, std::memory_order_relaxed)) {
	}
}

// (mutex held; ram12) The guard off each page of probed entry `e` that still has it, its own protection kept: a page
// whose protection is no longer ours + guard (somebody set another; Windows cleared the guard at a touch) is left as
// it is, and one somebody changed between the look and the change gets theirs back (the old protection the change
// returns tells). Never a saved protection over a page: nothing of others is undone.
inline void ClearGuardsLocked(State& s, const Entry& e) {
	for (uint32_t p = 0; p < Pages; ++p) {
		const uint64_t page = e.key + p * PageSize;
		const uint32_t now  = s.os.page_prot(page);
		if (now != (e.prot[p] | s.os.prot_guard)) {
			if (now != 0 && now != e.prot[p]) s.stats.changed.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		uint32_t old = 0;
		if (!s.os.protect(page, PageSize, e.prot[p], &old)) continue;
		if (old != now) { // (changed in between: theirs back)
			uint32_t ignored = 0;
			(void)s.os.protect(page, PageSize, old, &ignored);
			s.stats.put_back.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

// (mutex held) The granule of entry `e` usable again as it was; the entry erased. False: its bytes could not be read
// back (the granule left committed read-write with what the read gave: the caller's access runs, counted).
inline bool BringBackLocked(State& s, Entry* e, bool from_fault) {
	bool ok = true;
	// (ram13b) No shadow forgotten here: this runs in the fault handler, on any thread - one holding raiz's shadow lock
	// or its window included -; a shadow is only ever of a no-access page, and a granule of ours never is one (probe:
	// its own protection + guard; evicted: reserved). Forgotten at the probe and at the eviction (the worker, no lock of
	// raiz's held).
	if (e->state == StateProbe) {
		ClearGuardsLocked(s, *e);
		if (from_fault) s.stats.probe_hits.fetch_add(1, std::memory_order_relaxed);
		s.probe_now.fetch_sub(1, std::memory_order_relaxed);
	} else if (e->state == StateCold || e->state == StatePending) {
		const uint64_t t0 = NowUs();
		if (e->state == StatePending) { // (ram13: not in the file yet - the worker writes it -: from its buffer)
			ok = s.os.commit_rw(e->key);
			if (ok) std::memcpy(reinterpret_cast<void*>(e->key), s.buffers + static_cast<size_t>(e->buf) * Granule, Granule);
		} else {
			ok = s.os.commit_rw(e->key) &&
			     s.os.read_file(static_cast<uint64_t>(e->slot) * Granule, reinterpret_cast<void*>(e->key), Granule);
		}
		if (!ok) s.stats.read_fail.fetch_add(1, std::memory_order_relaxed);
		RestoreProtLocked(s, *e);
		if (e->state == StateCold) s.free_slots[s.free_count++] = e->slot; // (a pending one's: the worker's to free)
		const uint64_t us = NowUs() - t0;
		s.stats.back_reads.fetch_add(1, std::memory_order_relaxed);
		s.stats.back_us.fetch_add(us, std::memory_order_relaxed);
		uint64_t m = s.stats.back_us_max.load(std::memory_order_relaxed);
		while (us > m && !s.stats.back_us_max.compare_exchange_weak(m, us, std::memory_order_relaxed)) {
		}
		if (s.hooks.gpu_thread != nullptr && s.hooks.gpu_thread()) s.stats.back_gpu.fetch_add(1, std::memory_order_relaxed);
		s.cold_now.fetch_sub(1, std::memory_order_relaxed);
	}
	MarkHot(s, e->key, s.hot_s);
	EraseLocked(s, e);
	return ok;
}

// A fault at `address` (the vectored handler, any thread): true when it was a probed or evicted granule's and it is
// usable again (the access runs again).
// A fault raised while the granule was still out, served after another thread brought it back (two threads touching
// it at once: the second finds no entry), runs again when the page lets the access through now - at most 4 times in a
// row at one address on one thread, then it is left to the other handlers -.
// (ram12) `guard`: a guard violation (Windows has cleared that page's guard already); else an access violation (an
// evicted granule's reserved pages).
inline bool OnFault(uint64_t address, bool write, bool guard) {
	State& s = Get();
	if (address < s.lo.load(std::memory_order_relaxed) || address >= s.hi.load(std::memory_order_relaxed)) return false;
	const uint64_t key = address & ~(Granule - 1u);
	const bool     ours_once = s.ever != nullptr && (address >> 40u) == 0 &&
	                       (s.ever[address >> 22u].load(std::memory_order_relaxed) >> ((address >> 16u) & 63u) & 1u) != 0;
	const uint32_t b   = Hash(key) & (HotSlots - 1u);
	if (s.bucket[b].load(std::memory_order_acquire) == 0 &&
	    s.bucket_erased[b].load(std::memory_order_relaxed) < s.tick.load(std::memory_order_relaxed))
		return guard && ours_once; // (none of ours there now: a leftover guard of ours runs again; anything else is theirs)
	{
		ColdLock lock(s);
		Entry*          e = FindLocked(s, key);
		if (e == nullptr) {
			thread_local uint64_t last = 0;
			thread_local uint32_t again = 0;
			again = last == address ? again + 1 : 0;
			last  = address;
			// (A guard violation already took the guard off: running again is right whoever's page it was here.)
			// (ram13: or the page has a guard of ours again - put back since: running again raises it, and it is served -.)
			const bool guarded_now = ours_once && s.os.page_prot != nullptr &&
			                         (s.os.page_prot(address & ~(PageSize - 1u)) & s.os.prot_guard) != 0;
			return (guard && ours_once) || (again < 4 && (guarded_now || (s.os.allows != nullptr && s.os.allows(address, write))));
		}
		if (guard && e->state != StateProbe) return true; // (evicted since the guard went off: the access runs again, faults on it)
		const uint64_t t0 = NowUs();
		if (!guard && e->state == StateProbe) {
			// (An access violation of a probed page: its own protection's - not ours - unless the page lets it through
			// now: raised before a change that came in between, it runs again.)
			thread_local uint64_t last = 0;
			thread_local uint32_t again = 0;
			again = last == address ? again + 1 : 0;
			last  = address;
			const bool guarded_now = s.os.page_prot != nullptr && (s.os.page_prot(address & ~(PageSize - 1u)) & s.os.prot_guard) != 0;
			return again < 4 && (guarded_now || (s.os.allows != nullptr && s.os.allows(address, write)));
		}
		(void)BringBackLocked(s, e, true);
		const uint64_t us = NowUs() - t0;
		s.stats.handled.fetch_add(1, std::memory_order_relaxed);
		s.stats.handler_us.fetch_add(us, std::memory_order_relaxed);
		Max(s.stats.handler_us_max, us);
	}
	return true;
}

// Before a protection change or a kernel-mode access of guest [vaddr, vaddr + size): its granules usable again.
inline void Ensure(uint64_t vaddr, uint64_t size) {
	State& s = Get();
	if (size == 0 || s.count.load(std::memory_order_acquire) == 0 || vaddr + size <= s.lo.load(std::memory_order_relaxed) ||
	    vaddr >= s.hi.load(std::memory_order_relaxed))
		return;
	for (uint64_t key = vaddr & ~(Granule - 1u); key < vaddr + size; key += Granule) {
		{
			ColdLock lock(s);
			Entry*          e = FindLocked(s, key);
			if (e == nullptr) continue;
			s.stats.ensures.fetch_add(1, std::memory_order_relaxed);
			(void)BringBackLocked(s, e, false);
		}
	}
}

// Guest [vaddr, vaddr + size) unmapped: its entries dropped (evicted granules committed again read-write, zeros).
inline void Drop(uint64_t vaddr, uint64_t size) {
	State& s = Get();
	if (size == 0 || s.count.load(std::memory_order_acquire) == 0 || vaddr + size <= s.lo.load(std::memory_order_relaxed) ||
	    vaddr >= s.hi.load(std::memory_order_relaxed))
		return;
	ColdLock lock(s);
	for (uint64_t key = vaddr & ~(Granule - 1u); key < vaddr + size; key += Granule) {
		Entry* e = FindLocked(s, key);
		if (e == nullptr) continue;
		if (e->state == StateProbe) {
			ClearGuardsLocked(s, *e);
			s.probe_now.fetch_sub(1, std::memory_order_relaxed);
		} else if (e->state == StateCold || e->state == StatePending) {
			(void)s.os.commit_rw(e->key);
			if (e->state == StateCold) s.free_slots[s.free_count++] = e->slot;
			s.cold_now.fetch_sub(1, std::memory_order_relaxed);
		}
		s.stats.dropped.fetch_add(1, std::memory_order_relaxed);
		EraseLocked(s, e);
	}
}

// One step of the worker (once a second; public for the test): probes and evicts within the budgets.
struct Config {
	uint64_t below = uint64_t {2048} << 20u;
	uint64_t per_s = uint64_t {128} << 20u;   // the rate it starts at (KYTY_COLD_STORE_MB_S)
	uint64_t per_s_max = uint64_t {512} << 20u; // (KYTY_COLD_STORE_MB_S_MAX)
	uint64_t gate_us_per_s = 10000;           // the gate held alone, at most, a second (KYTY_COLD_STORE_GATE_MS)
	uint32_t idle_s = 20;
};
struct Cursor {
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	size_t                                     range = 0;
	uint64_t                                   at    = 0;
	std::vector<uint64_t>                      probed; // granules probed, in order
	uint32_t                                   listed = 0;
	// (ram13) The adaptive rate's last look (every 5 s).
	uint32_t adapted = 0;
	uint64_t seen_back = 0, seen_back_us = 0, seen_alone = 0;
};

// (ram13) The gate, timed: alone (an eviction: every protection change waits) or shared (a probe: only windows wait).
struct GateTimer {
	State& s;
	bool   alone;
	uint64_t held_from = 0;
	GateTimer(State& state, bool exclusive): s(state), alone(exclusive) {
		const uint64_t t0 = NowUs();
		{
			LockWatch::Wait wait(alone || s.hooks.gate_lock_shared == nullptr ? LockWatch::GateAlone : LockWatch::GateShared);
			if (alone) s.hooks.gate_lock();
			else if (s.hooks.gate_lock_shared != nullptr) s.hooks.gate_lock_shared();
			else s.hooks.gate_lock();
		}
		held_from = NowUs();
		s.stats.gate_wait_us.fetch_add(held_from - t0, std::memory_order_relaxed);
	}
	~GateTimer() {
		if (alone || s.hooks.gate_unlock_shared == nullptr) {
			s.hooks.gate_unlock();
			LockWatch::Release(LockWatch::GateAlone);
		} else {
			s.hooks.gate_unlock_shared();
			LockWatch::Release(LockWatch::GateShared);
		}
		const uint64_t us = NowUs() - held_from;
		if (alone) {
			s.stats.gate_alone_us.fetch_add(us, std::memory_order_relaxed);
			Max(s.stats.gate_alone_max_us, us);
		} else {
			s.stats.gate_shared_us.fetch_add(us, std::memory_order_relaxed);
		}
	}
	GateTimer(const GateTimer&)            = delete;
	GateTimer& operator=(const GateTimer&) = delete;
};

inline void Step(const Config& c, Cursor& cur) {
	State&         s      = Get();
	const uint32_t now    = s.tick.load(std::memory_order_relaxed);
	const uint64_t avail  = s.os.available != nullptr ? s.os.available() : 0;
	const bool     short_ = avail != 0 && avail < c.below;
	// (ram13) The rate, every 5 s: up by half while the granules coming back stay few and cheap and the gate's time
	// alone stays in its budget; halved when they do not.
	if (s.per_s_now.load(std::memory_order_relaxed) == 0) s.per_s_now.store(c.per_s, std::memory_order_relaxed);
	if (now - cur.adapted >= 5) {
		const uint64_t back = s.stats.back_reads.load() - cur.seen_back, back_us = s.stats.back_us.load() - cur.seen_back_us;
		const uint64_t alone = s.stats.gate_alone_us.load() - cur.seen_alone;
		const double   secs  = static_cast<double>(now - cur.adapted);
		const double   per_s_back = cur.adapted == 0 ? 0.0 : static_cast<double>(back) / secs;
		const double   mean_us    = back != 0 ? static_cast<double>(back_us) / static_cast<double>(back) : 0.0;
		const double   alone_us_s = cur.adapted == 0 ? 0.0 : static_cast<double>(alone) / secs;
		uint64_t       rate       = s.per_s_now.load(std::memory_order_relaxed);
		if (per_s_back > 60.0 || mean_us > 1500.0 || alone_us_s > static_cast<double>(c.gate_us_per_s)) {
			rate = std::max<uint64_t>(rate / 2u, uint64_t {16} << 20u);
		} else if (short_ && per_s_back < 20.0 && mean_us < 400.0 && alone_us_s < 0.7 * static_cast<double>(c.gate_us_per_s)) {
			rate = std::min<uint64_t>(rate + rate / 2u, c.per_s_max);
		}
		s.per_s_now.store(rate, std::memory_order_relaxed);
		cur.adapted      = now;
		cur.seen_back    = s.stats.back_reads.load();
		cur.seen_back_us = s.stats.back_us.load();
		cur.seen_alone   = s.stats.gate_alone_us.load();
	}
	const uint64_t per_s = s.per_s_now.load(std::memory_order_relaxed);
	// Evict what stayed probed long enough (a touch took it out of the table): a granule per gate hold (alone, ~20 us:
	// look, read-only, copy to a buffer, decommit), the file written after with no gate and no lock held.
	if (short_ && !cur.probed.empty()) {
		std::vector<uint64_t> keep;
		uint64_t              budget = per_s, alone_used = 0;
		size_t                i = 0;
		for (; i < cur.probed.size();) {
			uint32_t pending[Buffers];
			uint64_t pending_key[Buffers];
			uint32_t n = 0;
			for (; i < cur.probed.size() && n < Buffers; ++i) {
				const uint64_t key = cur.probed[i];
				if (budget < Granule || alone_used >= c.gate_us_per_s) {
					keep.push_back(key);
					continue;
				}
				{
					ColdLock lock(s);
					Entry*          e = FindLocked(s, key);
					if (e == nullptr || e->state != StateProbe) continue; // (touched, ensured or dropped)
					if (now - e->since < c.idle_s) {
						keep.push_back(key);
						continue;
					}
				}
				const uint64_t t0 = NowUs();
				{
					GateTimer gate(s, true);
					// (ram13b) Raiz's lazy store asked before the table's lock: its lock is never taken under ours.
					const bool kept = s.hooks.pending != nullptr && s.hooks.pending(key, Granule);
					ColdLock   lock(s);
					Entry*     e = FindLocked(s, key);
					if (e == nullptr || e->state != StateProbe) continue;
					// (ram12) Each page exactly its own protection + guard still: nobody touched it (a touch clears the
					// guard) and nobody changed it. Otherwise it is not cold, or not ours to take: forgotten, theirs left.
					bool intact = true;
					for (uint32_t p = 0; p < Pages && intact; ++p)
						intact = s.os.page_prot(key + p * PageSize) == (e->prot[p] | s.os.prot_guard);
					if (!intact || kept) {
						ClearGuardsLocked(s, *e);
						s.stats.changed.fetch_add(intact ? 0 : 1, std::memory_order_relaxed);
						s.probe_now.fetch_sub(1, std::memory_order_relaxed);
						MarkHot(s, key, s.hot_s);
						EraseLocked(s, e);
						continue;
					}
					if (s.hooks.forget != nullptr) s.hooks.forget(key, Granule);
					uint32_t old_ro = 0;
					(void)s.os.protect(key, Granule, s.os.prot_read_only, &old_ro);
					std::memcpy(s.buffers + static_cast<size_t>(n) * Granule, reinterpret_cast<const void*>(key), Granule);
					if (!s.os.decommit(key)) {
						s.stats.write_fail.fetch_add(1, std::memory_order_relaxed);
						RestoreProtLocked(s, *e); // (the read-only we set a moment ago: gate held alone, nobody else)
						s.probe_now.fetch_sub(1, std::memory_order_relaxed);
						EraseLocked(s, e);
						continue;
					}
					e->state = StatePending;
					e->buf   = static_cast<uint8_t>(n);
					s.probe_now.fetch_sub(1, std::memory_order_relaxed);
					s.cold_now.fetch_add(1, std::memory_order_relaxed);
				}
				const uint64_t us = NowUs() - t0;
				alone_used += us;
				s.stats.evict_work_us.fetch_add(us, std::memory_order_relaxed);
				pending[n]       = n;
				pending_key[n++] = key;
				budget -= Granule;
			}
			// The batch to the file: no gate, no lock while writing (a touch meanwhile is served from the buffer).
			for (uint32_t k = 0; k < n; ++k) {
				uint32_t slot = 0;
				{
					ColdLock lock(s);
					Entry*          e = FindLocked(s, pending_key[k]);
					if (e == nullptr || e->state != StatePending || e->buf != pending[k]) continue; // (back already)
					if (s.free_count != 0) slot = s.free_slots[--s.free_count];
					else slot = s.next_slot++;
				}
				const uint64_t t0 = NowUs();
				const bool     ok = s.os.write_file(static_cast<uint64_t>(slot) * Granule,
				                                    s.buffers + static_cast<size_t>(pending[k]) * Granule, Granule);
				s.stats.file_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
				ColdLock lock(s);
				Entry*          e = FindLocked(s, pending_key[k]);
				if (e != nullptr && e->state == StatePending && e->buf == pending[k]) {
					if (ok) {
						e->state = StateCold;
						e->slot  = slot;
						s.stats.evicted.fetch_add(1, std::memory_order_relaxed);
						continue;
					}
					s.stats.write_fail.fetch_add(1, std::memory_order_relaxed);
					(void)BringBackLocked(s, e, false); // (from its buffer)
				}
				s.free_slots[s.free_count++] = slot;
			}
		}
		cur.probed.swap(keep);
	}
	if (!short_) return;
	if (cur.ranges.empty() || now - cur.listed >= 10) {
		cur.ranges.clear();
		if (s.hooks.ranges != nullptr) s.hooks.ranges(s.hooks.ranges_user, &cur.ranges);
		cur.listed = now;
		if (cur.range >= cur.ranges.size()) {
			cur.range = 0;
			cur.at    = 0;
		}
	}
	// Probe more (ram13): as much as the rate lets out in an idle time stays probed (the pipeline full), two seconds'
	// worth a step.
	const uint64_t depth   = per_s * c.idle_s;
	const uint64_t probing = static_cast<uint64_t>(s.probe_now.load(std::memory_order_relaxed)) * Granule;
	uint64_t       want    = probing >= depth ? 0 : std::min(depth - probing, 2 * per_s);
	size_t         looked  = 0;
	uint32_t       prot[Pages];
	// (The trackers', the lazy store's and the labels' checks with no gate: they take locks of their own. A watch that
	// comes after goes through a protection change - Ensure, gate held - first.)
	std::vector<uint64_t> candidates;
	while (want >= Granule * (candidates.size() + 1) && !cur.ranges.empty() && looked < 262144) {
		if (cur.range >= cur.ranges.size()) cur.range = 0;
		const auto&    r     = cur.ranges[cur.range];
		const uint64_t first = (r.first + Granule - 1u) & ~(Granule - 1u);
		if (cur.at < first) cur.at = first;
		if (cur.at + Granule > r.second) {
			cur.range++;
			cur.at = 0;
			looked++;
			continue;
		}
		const uint64_t key = cur.at;
		cur.at += Granule;
		looked++;
		if (IsHot(s, key)) continue;
		if ((s.hooks.tracked != nullptr && s.hooks.tracked(key, Granule)) ||
		    (s.hooks.pending != nullptr && s.hooks.pending(key, Granule)))
			continue;
		if (s.hooks.excluded != nullptr && s.hooks.excluded(key, Granule)) {
			s.stats.excluded.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		candidates.push_back(key);
	}
	// (ram13) In batches of 16 under the gate SHARED - as a protection change takes it: only windows wait (raiz's
	// ~100 a second), never a protection change -. A protection change running beside it: each page is guarded by
	// itself and the protection it had is checked against the one looked at; one changed in between gets its own back
	// and the granule is left.
	for (size_t at = 0; at < candidates.size();) {
		const uint64_t t0 = NowUs();
		GateTimer      gate(s, false);
		for (size_t end = std::min(candidates.size(), at + 16); at < end; ++at) {
			const uint64_t  key = candidates[at];
			ColdLock lock(s);
			if (FindLocked(s, key) != nullptr) continue;
			if (!s.os.query(key, prot)) continue;
			Entry* e = InsertLocked(s, key);
			if (e == nullptr) {
				s.stats.full.fetch_add(1, std::memory_order_relaxed);
				at = candidates.size();
				break;
			}
			for (uint32_t p = 0; p < Pages; ++p) e->prot[p] = static_cast<uint8_t>(prot[p]);
			e->state = StateProbe;
			e->since = now;
			if ((key >> 40u) == 0 && s.ever != nullptr)
				s.ever[key >> 22u].fetch_or(uint64_t {1} << ((key >> 16u) & 63u), std::memory_order_relaxed);
			bool guarded = true;
			for (uint32_t p = 0; p < Pages; ++p) {
				uint32_t old = 0;
				if (!s.os.protect(key + p * PageSize, PageSize, e->prot[p] | s.os.prot_guard, &old)) {
					guarded = false;
					break;
				}
				if (old != e->prot[p]) { // (changed since the look: theirs back, the granule left)
					uint32_t ignored = 0;
					(void)s.os.protect(key + p * PageSize, PageSize, old, &ignored);
					s.stats.put_back.fetch_add(1, std::memory_order_relaxed);
					guarded = false;
					break;
				}
			}
			if (!guarded) {
				ClearGuardsLocked(s, *e);
				EraseLocked(s, e);
				continue;
			}
			if (s.hooks.forget != nullptr) s.hooks.forget(key, Granule);
			s.probe_now.fetch_add(1, std::memory_order_relaxed);
			s.stats.probed.fetch_add(1, std::memory_order_relaxed);
			cur.probed.push_back(key);
		}
		s.stats.probe_work_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
	}
}

inline uint64_t EnvNumber(const char* name, uint64_t fallback) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] >= '0' && v[0] <= '9' ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : fallback;
}

// The "Cold store (30 s)" line.
inline void Report(double seconds) {
	State&          s = Get();
	static uint64_t prev[9] {};
	const uint64_t  cur[9] = {s.stats.evicted.load(), s.stats.back_reads.load(), s.stats.back_us.load(), s.stats.back_gpu.load(),
	                          s.stats.probe_hits.load(), s.stats.ensures.load(), s.stats.probed.load(), s.stats.dropped.load(),
	                          s.stats.write_fail.load() + s.stats.read_fail.load()};
	uint64_t        d[9];
	for (int i = 0; i < 9; ++i) d[i] = cur[i] - prev[i];
	const double ps = seconds > 0 ? 1.0 / seconds : 0.0;
	std::printf("Cold store (30 s): out now %.0f MiB, probed now %.0f MiB; written %.1f MiB/s; back %.1f/s (%.0f us mean, %llu "
	            "us longest), %.2f/s on the GPU thread; probe hits %.1f/s; ensures %.1f/s; probed %.1f MiB/s; dropped at unmap "
	            "%.1f/s; failures %llu; protections of others met %llu, put back %llu; per-frame memory left out %llu "
	            "granules in all; available %llu MiB\n",
	            s.cold_now.load() * 64.0 / 1024.0, s.probe_now.load() * 64.0 / 1024.0, d[0] * ps / 16.0, d[1] * ps,
	            d[1] != 0 ? static_cast<double>(d[2]) / static_cast<double>(d[1]) : 0.0,
	            static_cast<unsigned long long>(s.stats.back_us_max.exchange(0)), d[3] * ps, d[4] * ps, d[5] * ps,
	            d[6] * ps / 16.0, d[7] * ps, static_cast<unsigned long long>(d[8]),
	            static_cast<unsigned long long>(s.stats.changed.load()), static_cast<unsigned long long>(s.stats.put_back.load()),
	            static_cast<unsigned long long>(s.stats.excluded.load()),
	            static_cast<unsigned long long>((s.os.available != nullptr ? s.os.available() : 0) >> 20u));
	// (ram13) What it costs the others, over the same 30 s.
	static uint64_t pc[10] {};
	const uint64_t  cc[10] = {s.stats.handled.load(), s.stats.handler_us.load(), s.stats.gate_alone_us.load(),
	                          s.stats.gate_shared_us.load(), s.stats.gate_wait_us.load(), s.stats.probe_work_us.load(),
	                          s.stats.evict_work_us.load(), s.stats.file_us.load(), s.stats.probed.load(),
	                          s.stats.evicted.load()};
	uint64_t        dc[10];
	for (int i = 0; i < 10; ++i) dc[i] = cc[i] - pc[i];
	std::printf("Cold store cost (30 s): faults it served %.1f/s, %.0f us mean, %llu us longest; gate held ALONE %.2f ms/s "
	            "(longest %llu us; every guest protection change waits), shared %.2f ms/s (only windows wait), waited for it "
	            "%.2f ms/s; per granule: probe %.1f us, evict %.1f us; file %.2f ms/s; rate now %llu MiB/s\n",
	            dc[0] * ps, dc[0] != 0 ? static_cast<double>(dc[1]) / static_cast<double>(dc[0]) : 0.0,
	            static_cast<unsigned long long>(s.stats.handler_us_max.exchange(0)), dc[2] * ps / 1000.0,
	            static_cast<unsigned long long>(s.stats.gate_alone_max_us.exchange(0)), dc[3] * ps / 1000.0,
	            dc[4] * ps / 1000.0, dc[8] != 0 ? static_cast<double>(dc[5]) / static_cast<double>(dc[8]) : 0.0,
	            dc[9] != 0 ? static_cast<double>(dc[6]) / static_cast<double>(dc[9]) : 0.0, dc[7] * ps / 1000.0,
	            static_cast<unsigned long long>(s.per_s_now.load() >> 20u));
	std::fflush(stdout);
	for (int i = 0; i < 9; ++i) prev[i] = cur[i];
	for (int i = 0; i < 10; ++i) pc[i] = cc[i];
}

// Table and slots (once).
inline bool Prepare() {
	State& s = Get();
	if (s.table != nullptr) return true;
	s.table      = new Entry[Capacity];
	s.free_slots = new uint32_t[Capacity];
	s.ever       = new std::atomic<uint64_t>[uint64_t {1} << 18u] {}; // (1 TiB / 64 KiB bits: 2 MiB)
	s.buffers    = new uint8_t[static_cast<size_t>(Buffers) * Granule];     // (ram13: 2 MiB)
	return true;
}

#if defined(_WIN32)
namespace Win {
inline HANDLE g_file = INVALID_HANDLE_VALUE;
inline bool   Plain(DWORD p) {
	return p == PAGE_READWRITE || p == PAGE_READONLY || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE;
}
inline bool Query(uint64_t granule, uint32_t* prot) {
	void* base = nullptr; // (one allocation: a protection change or a decommit cannot cross two)
	for (uint64_t at = granule; at < granule + Granule;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
		if (info.State != MEM_COMMIT || info.Type != MEM_PRIVATE || !Plain(info.Protect)) return false;
		if (base == nullptr) base = info.AllocationBase;
		if (info.AllocationBase != base) return false;
		const uint64_t stop = std::min(granule + Granule, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
		for (uint64_t p = at; p < stop; p += PageSize) prot[(p - granule) / PageSize] = info.Protect;
		at = stop;
	}
	return true;
}
inline bool Protect(uint64_t at, uint64_t size, uint32_t prot, uint32_t* old_prot) {
	DWORD old = 0;
	const bool ok = VirtualProtect(reinterpret_cast<void*>(at), size, prot, &old) != 0;
	*old_prot     = old;
	return ok;
}
inline uint32_t PageProt(uint64_t page) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) return 0;
	return info.Protect;
}
inline bool Decommit(uint64_t granule) {
	return VirtualFree(reinterpret_cast<void*>(granule), Granule, MEM_DECOMMIT) != 0;
}
inline bool CommitRw(uint64_t granule) {
	return VirtualAlloc(reinterpret_cast<void*>(granule), Granule, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}
inline bool Io(bool write, uint64_t offset, void* data, uint64_t size) {
	OVERLAPPED o {};
	o.Offset     = static_cast<DWORD>(offset);
	o.OffsetHigh = static_cast<DWORD>(offset >> 32u);
	DWORD done   = 0;
	const BOOL ok = write ? WriteFile(g_file, data, static_cast<DWORD>(size), &done, &o)
	                      : ReadFile(g_file, data, static_cast<DWORD>(size), &done, &o);
	return ok != 0 && done == size;
}
inline bool WriteF(uint64_t offset, const void* data, uint64_t size) {
	return Io(true, offset, const_cast<void*>(data), size);
}
inline bool ReadF(uint64_t offset, void* data, uint64_t size) {
	return Io(false, offset, data, size);
}
inline bool Allows(uint64_t address, bool write) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT)
		return false;
	const DWORD p = info.Protect & 0xffu;
	return write ? (p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE) : (p != PAGE_NOACCESS && p != 0);
}
inline uint64_t Available() {
	MEMORYSTATUSEX m {};
	m.dwLength = sizeof(m);
	return GlobalMemoryStatusEx(&m) != 0 ? m.ullAvailPhys : 0;
}
inline LONG CALLBACK Handler(PEXCEPTION_POINTERS info) {
	const auto* r = info->ExceptionRecord;
	const bool guard = r->ExceptionCode == STATUS_GUARD_PAGE_VIOLATION;
	if ((r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION && !guard) || r->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
	return OnFault(static_cast<uint64_t>(r->ExceptionInformation[1]), r->ExceptionInformation[0] == 1, guard)
	           ? EXCEPTION_CONTINUE_EXECUTION
	           : EXCEPTION_CONTINUE_SEARCH;
}
inline bool OpenFile() {
	wchar_t folder[MAX_PATH] {};
	if (const char* dir = std::getenv("KYTY_COLD_STORE_DIR"); dir != nullptr && dir[0] != '\0') {
		(void)MultiByteToWideChar(CP_UTF8, 0, dir, -1, folder, MAX_PATH);
	} else {
		const DWORD n = GetModuleFileNameW(nullptr, folder, MAX_PATH);
		for (DWORD i = n; i > 0; --i)
			if (folder[i - 1] == L'\\' || folder[i - 1] == L'/') {
				folder[i - 1] = 0;
				break;
			}
	}
	wchar_t path[MAX_PATH + 64] {};
	(void)swprintf(path, MAX_PATH + 64, L"%s\\kyty-cold-%lu.tmp", folder, static_cast<unsigned long>(GetCurrentProcessId()));
	g_file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
	                     FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
	if (g_file == INVALID_HANDLE_VALUE) return false;
	DWORD returned = 0;
	(void)DeviceIoControl(g_file, 0x000900c4 /* FSCTL_SET_SPARSE */, nullptr, 0, nullptr, 0, &returned, nullptr);
	return true;
}
} // namespace Win
#endif

// Started once the emulator's side is known (memoryAddressSpace.inc, native memory): its thread reads the switches 10 s
// later and does nothing more without KYTY_COLD_STORE=1.
inline void Start(const Hooks& hooks) {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [hooks] {
		std::thread([hooks] {
			std::this_thread::sleep_for(std::chrono::seconds(10));
			const char* v = std::getenv("KYTY_COLD_STORE");
			if (v == nullptr || v[0] != '1') return;
			State& s = Get();
			Prepare();
			s.hooks                = hooks;
			s.os.query             = &Win::Query;
			s.os.protect           = &Win::Protect;
			s.os.page_prot         = &Win::PageProt;
			s.os.prot_guard        = PAGE_GUARD;
			s.os.decommit          = &Win::Decommit;
			s.os.commit_rw         = &Win::CommitRw;
			s.os.write_file        = &Win::WriteF;
			s.os.read_file         = &Win::ReadF;
			s.os.available         = &Win::Available;
			s.os.allows            = &Win::Allows;
			s.os.prot_no_access    = PAGE_NOACCESS;
			s.os.prot_read_only    = PAGE_READONLY;
			if (!Win::OpenFile() || AddVectoredExceptionHandler(1, &Win::Handler) == nullptr) {
				std::printf("Cold store: could not start (file or handler; error %lu)\n", static_cast<unsigned long>(GetLastError()));
				return;
			}
			Config c;
			c.below  = EnvNumber("KYTY_COLD_STORE_BELOW_MB", 2048) << 20u;
			c.per_s  = std::max<uint64_t>(EnvNumber("KYTY_COLD_STORE_MB_S", 128), 1) << 20u;
			c.per_s_max     = std::max<uint64_t>(EnvNumber("KYTY_COLD_STORE_MB_S_MAX", 512), 1) << 20u;
			c.gate_us_per_s = std::max<uint64_t>(EnvNumber("KYTY_COLD_STORE_GATE_MS", 10), 1) * 1000u;
			c.idle_s = static_cast<uint32_t>(std::max<uint64_t>(EnvNumber("KYTY_COLD_STORE_IDLE_S", 20), 2));
			s.hot_s  = static_cast<uint32_t>(std::max<uint64_t>(EnvNumber("KYTY_COLD_STORE_HOT_S", 120), 1));
			s.on.store(true, std::memory_order_release);
			LockWatch::Start(); // (ram13b)
			std::printf("Cold store: on (KYTY_COLD_STORE=1): below %llu MiB available, %llu MiB/s to start (adaptive, at most "
			            "%llu), the gate alone at most %llu ms a second, idle %u s\n",
			            static_cast<unsigned long long>(c.below >> 20u), static_cast<unsigned long long>(c.per_s >> 20u),
			            static_cast<unsigned long long>(c.per_s_max >> 20u),
			            static_cast<unsigned long long>(c.gate_us_per_s / 1000u), c.idle_s);
			std::fflush(stdout);
			Cursor cur;
			for (uint32_t t = 1;; ++t) {
				std::this_thread::sleep_for(std::chrono::seconds(1));
				s.tick.store(t, std::memory_order_relaxed);
				Step(c, cur);
				if (t % 30 == 0) Report(30.0);
			}
		}).detach();
	});
#else
	(void)hooks;
#endif
}

} // namespace ColdStore

#endif // EMULATOR_SRC_LOCAL_COLD_STORE_H_
