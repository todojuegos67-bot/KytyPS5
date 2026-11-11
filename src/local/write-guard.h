#ifndef KYTY_LOCAL_WRITE_GUARD_H_
#define KYTY_LOCAL_WRITE_GUARD_H_

// KYTY_WRITE_GUARD (1 on, 0 off; unset: on under KYTY_SPECULATE=4 only): a guest memory write of the command processor
// (WriteGuestMemory: WRITE_DATA, an end-of-pipe label, the reference clock) made while older upload jobs are pending
// lands at once and is hidden from the game until those jobs are done, instead of waiting for them (AsyncUpload::Drain,
// issue #196).
//
// The write is a release: the game takes a label as the GPU being done and may then rewrite the buffers the pending
// copies still read. So the game must not see it before every job queued ahead of it ran; the render thread need not
// wait for that. Each 4 KiB page the write touches gets no access, and a job pushed behind the pending ones lifts it.
//
// * Every access to such a page from any thread faults and waits for the lift (OnFault): the game's, and any reader
//   of the emulator's that this form does not list. A wait behind a moving queue lasts as long as it takes; a queue
//   that stood still for a second is a cycle, reported, and the guard lifted then. The upload worker, which cannot
//   wait for its own queue, lifts the guard itself, reported. Neither should ever happen.
// * The GPU thread's own reads of the bytes it wrote (GuardedBacking) take the guest backing view, the same bytes
//   through the second mapping, without waiting: WAIT_REG_MEM's (a cross-queue label), the command buffers and the
//   indirect register tables, which GNM allocates next to its labels. The render thread passing a label is no release
//   to the game: its later writes are guarded behind their own jobs, and the end-of-pipe interrupt fires only when
//   the submission that carries it completes, which waits for the upload jobs pushed before it.
// * A protection change of any other kind (the trackers', the guest's mprotect) leaves a guarded page no-access and
//   becomes its protection when the guard lifts (Honour, from GuestAddressSpace::ProtectMappedUnlocked).
// * An unmap drops the guards of its range (Drop), and so the GPU thread's own table's entries of them (g_gone).

#include "common/virtualMemory.h"
#include "live-switch.h"
#include "mode-toggle.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_write_guard_mode;
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (performance-switches.h)
}

namespace Libs::LibKernel::Memory {
const uint8_t* TryGetBackingPointer(uint64_t vaddr, uint64_t size); // (kernel/memory.h)
} // namespace Libs::LibKernel::Memory

namespace WriteGuard {

inline constexpr uint64_t Page = 4096;

struct Entry {
	uint64_t                    token = 0;                                     // the lift job's
	Common::VirtualMemory::Mode saved = Common::VirtualMemory::Mode::NoAccess; // the protection once lifted
};

inline std::shared_mutex                   g_lock;
inline std::unordered_map<uint64_t, Entry> g_pages;     // under g_lock
inline uint64_t                            g_token = 0; // under g_lock
inline std::atomic<uint32_t>               g_count {0}; // g_pages.size(), for the readers' fast path
// Pages newly guarded so far (KYTY_WRITE_GUARD_SPEC: a speculation's thread, translating while the GPU thread writes,
// rebases its command buffers when it moved: ProcessPm4).
inline std::atomic<uint64_t>               g_arms {0};
// The upload worker: it cannot wait for a lift its own queue holds (async-upload.h sets it).
inline thread_local bool t_no_wait = false;
// The upload worker's completed sequence (set by the GPU thread's first guarded write): a wait whose queue has not moved
// for a second is a cycle (OnFault).
inline uint64_t (*g_upload_done)() = nullptr;

// KYTY_WRITE_GUARD: 1 on, 0 off; not set (2): on under KYTY_SPECULATE=4 only (graphicsRun.cpp).
[[nodiscard]] inline bool Enabled() {
	const auto mode = kyty_local_write_guard_mode.load(std::memory_order_relaxed);
	return mode == 1 || (mode == 2 && ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4);
}
// KYTY_WRITE_GUARD_SPEC (live write_guard_spec, on by default under KYTY_SPECULATE=4): the guard in mode 4 too: the
// normal path's guest writes and a commit's (Speculation::After), instead of draining the upload queue before them.
// The speculation's threads read a guarded page through the backing view (Spec::State::CopyGuest, GuardTable), the
// bytes the GPU thread wrote, as its own reads do: a speculation reads the processor's writes, it does not release
// the game.
[[nodiscard]] inline bool InSpeculation() {
	// (The lambda's type named: atoll's long long and ModeToggle::Default's int64_t, long on Linux, are two types there.)
	static auto& on = LiveSwitch::Find("write_guard_spec", []() -> int64_t {
		const char* value = std::getenv("KYTY_WRITE_GUARD_SPEC");
		if (value != nullptr) return std::atoll(value);
		return ModeToggle::Default("write_guard_spec");
	}());
	return on.load(std::memory_order_relaxed) != 0;
}

// KYTY_WRITE_GUARD_LOCKFREE (live write_guard_lockfree, default 1): the readers' check without g_lock first
// (GuardedBacking: mode 4's threads spent 7-8.5 % of their time acquiring it shared, 4 Nov), and a speculation's thread
// rebases its command buffers only for a page newly guarded inside them (ProcessPm4). 0: the lock at every check and a
// rebase at every new guard, as before. The tables below are kept whatever the switch (it can be flipped live).
[[nodiscard]] inline bool LockFree() {
	static auto& on = LiveSwitch::Find("write_guard_lockfree", 1);
	return on.load(std::memory_order_relaxed) != 0;
}

// KYTY_WRITE_GUARD_FAST (live write_guard_fast, default 0): the GPU thread's guarded writes and its reads of the pages
// they guarded without g_lock where they can (4 Nov: GuardedWrite was 7.2-7.7 % of the render thread, half of it
// waiting for g_lock; its reads of the pages it had just guarded took the filter's maybe, the lock and a mapping
// lookup each). A write probes the filter and takes the lock shared once, only on a maybe, for both of its lookups
// (GuardedWrite); its reads of its own guarded pages are answered from its own table (OwnBacking). 0: as before. The
// tables below are kept whatever the switch (it can be flipped live).
[[nodiscard]] inline bool Fast() {
	static auto& on = LiveSwitch::Find("write_guard_fast", 0); // (default 0: mode 0 lost ~2.4 fps with it on, wg5m0 10:30)
	return on.load(std::memory_order_relaxed) != 0;
}

// The guarded pages as a counting filter: a count of them a hash bucket, read without the lock (MayBeGuarded). Two
// pages may share a bucket (a check of either takes the lock: a false "maybe", never a false "no"); the pages of one
// 64 MiB block never do. Changed under the exclusive g_lock only: a bucket counts up before its page has no access
// (Arm) and down once the page is out of g_pages (LiftLocked, Drop), so it is never zero while a page of it is guarded.
inline constexpr uint32_t                  FilterBits = 14; // 16384 buckets, 64 KiB
inline std::atomic<uint32_t>               g_filter[1u << FilterBits] {};
// A range of more pages than this takes the lock without probing them.
inline constexpr uint64_t                  ProbeMax = 256;
// The page of each arm, by its number (g_arms before it) mod ArmRing: a speculation's thread reads the pages armed since
// it last looked (ArmedSinceIn).
inline constexpr uint64_t                  ArmRing = 256;
inline std::atomic<uint64_t>               g_armed[ArmRing] {};

[[nodiscard]] inline uint32_t FilterIndex(uint64_t page) {
	const uint64_t index = page / Page;
	return static_cast<uint32_t>((index ^ (index >> FilterBits) ^ (index >> (2u * FilterBits))) & ((1u << FilterBits) - 1u));
}
// (Exclusive g_lock.) Published with release stores: a reader that sees g_count's or g_arms's change after it sees it.
inline void FilterAdd(uint64_t page) {
	g_filter[FilterIndex(page)].fetch_add(1, std::memory_order_release);
}
inline void FilterRemove(uint64_t page) {
	g_filter[FilterIndex(page)].fetch_sub(1, std::memory_order_release);
}

// The pages that left g_pages (lifted, LiftLocked; dropped by an unmap, Drop) counted a filter bucket: an entry of the
// arming thread's own table (Own) holds while its bucket's count is the one read when it was made under the exclusive
// lock with the page guarded, so its page has not left g_pages since (another page of the bucket leaving drops it too:
// a miss, never a stale hit). Changed under the exclusive g_lock, after the page is out of g_pages and its protection
// restored, with a release store: a reader whose acquire load sees the change sees those too.
inline std::atomic<uint32_t> g_gone[1u << FilterBits] {};
inline void                  NoteGone(uint64_t page) {
    g_gone[FilterIndex(page)].fetch_add(1, std::memory_order_release);
}

// (KYTY_WRITE_GUARD_FAST.) The arming thread's own guarded pages (only GuardedWrite arms, on the GPU thread): a page,
// its backing view when one mapping covers it, and its bucket's g_gone when it was armed. Direct-mapped by the filter's
// index (a collision overwrites: a miss). Made on the thread's first fast arm; null on every other thread, which then
// asks g_pages as before.
struct Own {
	uint64_t       page    = 0;
	const uint8_t* backing = nullptr;
	uint32_t       gone    = 0;
};
inline constexpr uint32_t OwnSlots    = 256;
inline constexpr uint64_t OwnMaxPages = 4; // a read spanning more pages asks g_pages
inline thread_local Own*  t_own       = nullptr;

[[nodiscard]] inline Own& OwnSlot(Own* own, uint64_t page) {
	return own[FilterIndex(page) & (OwnSlots - 1u)];
}
// The backing of `page` from this thread's table when its entry holds (the page guarded since this thread armed it),
// else null (an entry that no longer holds is dropped).
[[nodiscard]] inline const uint8_t* OwnPage(uint64_t page) {
	Own* own = t_own;
	if (own == nullptr) return nullptr;
	Own& slot = OwnSlot(own, page);
	if (slot.page != page) return nullptr;
	if (g_gone[FilterIndex(page)].load(std::memory_order_acquire) != slot.gone) {
		slot.page = 0;
		return nullptr;
	}
	return slot.backing;
}
// (Exclusive g_lock, `page` just armed by this thread.) Its entry: `backing` is the page's backing view (null: none
// noted, its reads ask g_pages).
inline void OwnNote(uint64_t page, const uint8_t* backing) {
	if (backing == nullptr) return;
	if (t_own == nullptr) t_own = new Own[OwnSlots] {}; // (the GPU thread's, for the process's life)
	OwnSlot(t_own, page) = Own {page, backing, g_gone[FilterIndex(page)].load(std::memory_order_relaxed)};
}
// (KYTY_WRITE_GUARD_FAST.) The backing view of [vaddr, vaddr + size) when each of its pages is one this thread guarded
// and still is (their backing contiguous), from its own table: no lock, no mapping lookup. Null: not known here (the
// caller asks g_pages, as before). A lift racing this check is the race GuardedBacking's lock had too (the answer of
// a check just before the lift): the backing view of a page still mapped holds the guest view's bytes. An unmap drops
// the page's guard (Drop, under the exclusive lock, from GpuResourceManager::UnmapMemory, which runs on the GPU thread
// itself through SendCommandSync before the range is unmapped), so this thread's later checks see its own g_gone change.
[[nodiscard]] inline const void* OwnBacking(uint64_t vaddr, uint64_t size) {
	if (t_own == nullptr || size == 0) return nullptr;
	const uint64_t first = vaddr & ~(Page - 1), end = vaddr + size;
	if (end < vaddr || end - first > OwnMaxPages * Page) return nullptr;
	const uint8_t* base = nullptr;
	for (uint64_t page = first; page < end; page += Page) {
		const uint8_t* backing = OwnPage(page);
		if (backing == nullptr) return nullptr;
		if (page == first)
			base = backing;
		else if (backing != base + (page - first))
			return nullptr;
	}
	return base + (vaddr - first);
}

// No lock: false only when no page of [vaddr, vaddr + size) is guarded (as of the probe); true: maybe (the caller
// checks under g_lock).
[[nodiscard]] inline bool MayBeGuarded(uint64_t vaddr, uint64_t size) {
	if (size == 0) return false;
	const uint64_t first = vaddr & ~(Page - 1), end = vaddr + size;
	if (end < vaddr || (end - first + Page - 1) / Page > ProbeMax) return true;
	for (uint64_t page = first; page < end; page += Page)
		if (g_filter[FilterIndex(page)].load(std::memory_order_acquire) != 0) return true;
	return false;
}

// No lock: whether a page armed since arm number `seen` (up to `arms`, g_arms as the caller read it with acquire) may
// satisfy `in_range` (the caller's buffers). True when the ring no longer holds them all, or was overwritten while read:
// an arm that overwrote a slot read here had g_arms at seen + ArmRing or more before it, which the acquire loads make
// the reload below see.
template <typename InRange>
[[nodiscard]] inline bool ArmedSinceIn(uint64_t seen, uint64_t arms, InRange&& in_range) {
	if (arms - seen >= ArmRing) return true;
	for (uint64_t arm = seen; arm != arms; ++arm)
		if (in_range(g_armed[arm % ArmRing].load(std::memory_order_acquire))) return true;
	return g_arms.load(std::memory_order_acquire) - seen >= ArmRing;
}

// (Under g_lock.) Whether a page of [vaddr, vaddr + size) is guarded.
[[nodiscard]] inline bool AnyIn(uint64_t vaddr, uint64_t size) {
	if (g_pages.empty() || size == 0) return false;
	const uint64_t first = vaddr & ~(Page - 1), end = vaddr + size;
	if ((end - first) / Page < g_pages.size()) {
		for (uint64_t page = first; page < end; page += Page)
			if (g_pages.count(page) != 0) return true;
		return false;
	}
	return std::any_of(g_pages.begin(), g_pages.end(),
	                   [&](const auto& item) { return item.first < end && vaddr < item.first + Page; });
}

// The guest backing view of [vaddr, vaddr + size) when a page of it is guarded, else null (the GPU thread's own reads:
// the bytes it wrote are there).
[[nodiscard]] inline const void* GuardedBacking(uint64_t vaddr, uint64_t size) {
	if (g_count.load(std::memory_order_acquire) == 0) return nullptr;
	// (KYTY_WRITE_GUARD_FAST: the GPU thread's reads of the pages it guarded from its own table.)
	if (Fast())
		if (const void* own = OwnBacking(vaddr, size)) return own;
	// (KYTY_WRITE_GUARD_LOCKFREE: the lock only on a maybe, and the answer then from g_pages, as before: a lift since
	// the probe leaves the page out of it, its protection restored.)
	if (LockFree() && !MayBeGuarded(vaddr, size)) return nullptr;
	{
		std::shared_lock lock(g_lock);
		if (!AnyIn(vaddr, size)) return nullptr;
	}
	return Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, size);
}

// (Exclusive g_lock.) A new token for a write's lift job.
[[nodiscard]] inline uint64_t NextToken() {
	return ++g_token;
}

// (Exclusive g_lock.) The page guarded until the lift job with `token` (a guarded page's guard moves to the new job).
[[nodiscard]] inline bool Arm(uint64_t page, uint64_t token) {
	if (auto it = g_pages.find(page); it != g_pages.end()) {
		it->second.token = token;
		return true;
	}
	Entry entry {token};
	FilterAdd(page); // (before the page has no access)
	if (!Common::VirtualMemory::Protect(page, Page, Common::VirtualMemory::Mode::NoAccess, &entry.saved)) {
		FilterRemove(page);
		return false;
	}
	if (entry.saved == Common::VirtualMemory::Mode::NoAccess) {
		// Mostly a view page whose protection was never recorded (issue #289, KYTY_WRITE_GUARD_UFFD): the lift leaves it
		// no-access with no guard and no tracker holding it. (Also right after a lift to a tracker's no access.)
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("Write guard: page 0x%016llx armed with no access as its saved protection "
			            "(issue #289 if it repeats)\n",
			            static_cast<unsigned long long>(page));
	}
	g_pages.emplace(page, entry);
	g_count.fetch_add(1, std::memory_order_release);
	const uint64_t arm = g_arms.load(std::memory_order_relaxed);
	g_armed[arm % ArmRing].store(page, std::memory_order_release); // (before g_arms counts it)
	g_arms.fetch_add(1, std::memory_order_release);
	return true;
}

// (Exclusive g_lock.) The page's guard lifted: its protection as it would be without it.
inline void LiftLocked(std::unordered_map<uint64_t, Entry>::iterator it) {
	Common::VirtualMemory::Mode now = Common::VirtualMemory::Mode::NoAccess;
	(void)Common::VirtualMemory::Protect(it->first, Page, it->second.saved, &now);
	if (now != Common::VirtualMemory::Mode::NoAccess) {
		// Something changed the page's protection behind Honour: the guard was gone before its job.
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("Write guard: the guard of page 0x%016llx was lifted before its job (mode %u)\n",
			            static_cast<unsigned long long>(it->first), static_cast<unsigned>(now));
		(void)Common::VirtualMemory::Protect(it->first, Page, now);
	}
	const uint64_t page = it->first;
	g_pages.erase(it);
	g_count.fetch_sub(1, std::memory_order_release);
	FilterRemove(page); // (after: its protection is restored and it is out of g_pages)
	NoteGone(page);     // (the same)
}

// The lift job (the upload worker, behind the jobs pending at the write).
inline void Lift(uint64_t page, uint64_t token) {
	std::unique_lock lock(g_lock);
	if (auto it = g_pages.find(page); it != g_pages.end() && it->second.token == token) LiftLocked(it);
}

// A protection change of guest memory (GuestAddressSpace::ProtectMappedUnlocked): its guarded pages keep no access
// and take `mode` for when they are lifted; `protect` changes the others.
template <typename Protect>
[[nodiscard]] inline bool Honour(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode, Protect&& protect) {
	if (!Enabled()) return protect(vaddr, size);
	{
		// (A guard armed meanwhile waits for this lock: the change cannot undo it.)
		std::shared_lock lock(g_lock);
		if (!AnyIn(vaddr, size)) return protect(vaddr, size);
	}
	std::unique_lock      lock(g_lock);
	std::vector<uint64_t> pages;
	for (auto& [page, entry]: g_pages)
		if (page < vaddr + size && vaddr < page + Page) {
			entry.saved = mode;
			pages.push_back(page);
		}
	std::sort(pages.begin(), pages.end());
	uint64_t at = vaddr;
	bool     ok = true;
	for (const auto page: pages) {
		if (at < page) ok = protect(at, page - at) && ok;
		at = std::max(at, page + Page);
	}
	if (at < vaddr + size) ok = protect(at, vaddr + size - at) && ok;
	return ok;
}

// An unmap of [vaddr, vaddr + size): its guards dropped (the pages go; a lift after it would protect what is mapped
// there next).
inline void Drop(uint64_t vaddr, uint64_t size) {
	if (g_count.load(std::memory_order_acquire) == 0) return;
	std::unique_lock lock(g_lock);
	for (auto it = g_pages.begin(); it != g_pages.end();) {
		if (it->first < vaddr + size && vaddr < it->first + Page) {
			const uint64_t page = it->first;
			it                  = g_pages.erase(it);
			g_count.fetch_sub(1, std::memory_order_release);
			FilterRemove(page); // (after it is out of g_pages)
			NoteGone(page);     // (the same)
		} else {
			++it;
		}
	}
}


// The exception handler, first (runtimeLinker.cpp): a fault on a guarded page waits for its lift and is retried. A wait
// behind a moving queue waits as long as it takes (a level load's megabytes of uploads). A queue that has not moved for a
// second while the page waits is a cycle, a fault of this form: the faulting thread holds what the jobs ahead of the
// lift need. It is reported and the guard lifted then (the game may see the write early, the hazard) rather than the
// process hanging.
[[nodiscard]] inline bool OnFault(uint64_t vaddr, uint64_t rip) {
	if (g_count.load(std::memory_order_acquire) == 0) return false;
	const uint64_t page = vaddr & ~(Page - 1);
	{
		std::unique_lock lock(g_lock);
		const auto       it = g_pages.find(page);
		if (it == g_pages.end()) return false;
		if (t_no_wait) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
				std::printf("Write guard: the upload worker touched guarded page 0x%016llx at 0x%016llx: lifted\n",
				            static_cast<unsigned long long>(page), static_cast<unsigned long long>(rip));
			LiftLocked(it);
			return true;
		}
	}
	const auto start      = std::chrono::steady_clock::now();
	auto       moved      = start;
	uint64_t   done       = g_upload_done != nullptr ? g_upload_done() : 0;
	for (;;) {
		{
			std::shared_lock lock(g_lock);
			if (g_pages.count(page) == 0) break;
		}
		const auto now = std::chrono::steady_clock::now();
		if (const uint64_t at = g_upload_done != nullptr ? g_upload_done() : 0; at != done) done = at, moved = now;
		if (now - moved > std::chrono::seconds(1)) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
				std::printf("Write guard: the upload queue stood still a second while a thread waited for guarded page 0x%016llx "
				            "(at 0x%016llx): a cycle, lifted\n",
				            static_cast<unsigned long long>(page), static_cast<unsigned long long>(rip));
			std::unique_lock lock(g_lock);
			if (const auto it = g_pages.find(page); it != g_pages.end()) LiftLocked(it);
			break;
		}
		std::this_thread::yield();
	}
	return true;
}

} // namespace WriteGuard

#endif // KYTY_LOCAL_WRITE_GUARD_H_
