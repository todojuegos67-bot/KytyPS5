#pragma once
// Guest lazy commit (off by default, KYTY_GUEST_LAZY_COMMIT=1 turns it on; Windows, page-file store only).
//
// The guest backing store (one 13824 MiB page-file section) is created SEC_RESERVE instead of SEC_COMMIT, and its
// pages are committed when first touched, in granules of KYTY_GUEST_LAZY_COMMIT_KB (default 64 KiB, the fresh-memory
// granule), instead of when the game maps them (KYTY_BACKING_ON_DEMAND) or all at launch. As on a native PC: the game
// maps its whole budget, only what it writes (or reads) is charged to the commit limit. A section page is committed
// once for every view of it (VirtualAlloc MEM_COMMIT through any view of a SEC_RESERVE section commits the section's
// page; each view keeps its own page protection - the view's mapping protection unless that view's pages were given
// another one).
//
// Nothing here ever decommits, frees, unmaps or moves memory: a page goes from reserved to committed once, at its first
// touch, and stays committed for the rest of the run (a view of a section cannot be decommitted anyway).
//
// The rule that keeps it equal to a store committed at map: whatever commits a page gives every view of it exactly the
// protection it would have had, committed from the start. So:
//  - CPU touch (guest or host code, through a guest view or the backing alias): a reserved page faults; the vectored
//    handler here (first in the chain) commits the granule around it and the instruction runs again. It does so for a
//    fault on ANY reserved page of ANY mapped view in the process (MEM_RESERVE + MEM_MAPPED), known here or not: the
//    granule (clipped to that reserved run) is committed through the faulting view - for a known guest view or the
//    alias, the recorded protections first (below) and the alias's pages checked read-write; for a view not known here,
//    simply with its mapping protection. A fault on a page that is already committed (another thread committed its
//    granule between the fault and the handler's look) is resumed when the page is in a known guest view or the alias
//    and its protection now allows the access (read, write or execute); otherwise, and after 4 such resumptions in a
//    row at one address on one thread, it is left to the other handlers. Only this module's lock is taken there (no
//    emulator lock), and nothing is allocated.
//  - Protection changes of a guest view (the GPU trackers' write/read watches, the fresh-memory watches, guest
//    mprotect, a no-access view): committed pages are protected at once. For still-reserved pages of a guest view that
//    is the only guest view of those store pages, the protection is recorded instead (per store offset, with the view
//    that asked); a page is never accessible before it is committed, so nothing is missed meanwhile. When the page is
//    committed - through that view, another view, or the alias - it is committed first THROUGH THAT VIEW WITH THE
//    RECORDED PROTECTION (one VirtualAlloc: the view never has another protection), then the rest of the granule.
//    Pages that another guest view also maps (aliasing), the alias itself (texture offload), protections the view
//    cannot take, and views not known here are committed first and then protected (the commit-on-protect fallback),
//    so at most one view has recorded protections for a store page.
//  - Kernel-mode access (ReadFile/WriteFile, socket send/recv, getsockopt/setsockopt, a CRT write to stdout of a guest
//    buffer) does not fault in user mode: the buffer is committed first (Prefault, with its recorded protections), as
//    are guest stacks the game gives (a fault on a reserved stack page cannot be delivered: the system writes the
//    exception frame there) and ranges the emulator checks for readability (HostMemoryQueryRange).
// Every commit holds one lock and commits only pages still reserved: a commit of a committed page would reset its
// protection (a watch lost).
//
// The guest views are known here (Register before each view is mapped, Unregister after it is unmapped), with their
// store offsets: a commit through any view finds the recorded protections of its pages by offset. A count of guest
// views per 16 KiB of store says whether a page is aliased.

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <intrin.h>

#include "mem-guard.h"
#include "pin-watched.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace GuestLazyCommit {

constexpr uint64_t CountBits = 14; // views per 16 KiB of store (the guest page)

struct View {
	uint64_t size   = 0;
	uint64_t offset = 0; // store offset of its first byte
};

// A protection asked for reserved store pages [key, end) of offset, by the guest view `owner` (its base), whose page at
// the run's first offset is `guest`.
struct Wanted {
	uint64_t end     = 0;
	uint64_t guest   = 0;
	uint64_t owner   = 0;
	DWORD    protect = 0;
};

struct State {
	bool                  enabled = false;
	uint64_t              granule = 64u << 10u;
	uint64_t              alias_base = 0, alias_size = 0; // the backing alias
	uint64_t              store_size = 0;
	SRWLOCK               lock = SRWLOCK_INIT;
	std::atomic<uint64_t> committed {0};     // bytes committed by this module
	std::atomic<uint64_t> calls {0};         // commits (granules or ranges)
	std::atomic<uint64_t> fault_commits {0}; // of them, from a fault
	std::atomic<uint64_t> ticks {0};         // time in the commits (QueryPerformanceCounter ticks)
	std::atomic<uint64_t> max_ticks {0};     // the longest single commit
	std::atomic<uint64_t> failures {0};
	std::atomic<uint64_t> mapped {0};          // bytes of the store a guest mapping has covered (64 KiB granules)
	std::atomic<uint64_t> deferred_bytes {0};  // protection changes recorded for reserved pages (cumulative)
	std::atomic<uint64_t> fallback_bytes {0};  // protection changes that committed first (aliased, alias...)
	std::atomic<uint64_t> mismatch_bytes {0};  // pages a commit left with another protection than expected (fixed)
	std::atomic<uint32_t> last_error {0};      // the last failed commit's GetLastError
	// (Lock held for all of these. The fault handler only reads them: it allocates nothing.)
	std::map<uint64_t, View>   views;  // guest views of the store by base
	std::map<uint64_t, Wanted> wanted; // by store offset; runs do not overlap
	std::vector<uint8_t>       counts; // guest views per 16 KiB of store (255: many, kept)
};

inline State& Get() {
	static State state;
	return state;
}

// Read once: the store's constructor decides SEC_RESERVE with it. Off unless KYTY_GUEST_LAZY_COMMIT starts with '1'.
inline bool Requested() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GUEST_LAZY_COMMIT");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

inline bool Enabled() {
	return Get().enabled;
}

inline uint64_t Now() {
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	return static_cast<uint64_t>(t.QuadPart);
}

inline double TicksToMs(uint64_t ticks) {
	static const double per_ms = [] {
		LARGE_INTEGER f;
		QueryPerformanceFrequency(&f);
		return static_cast<double>(f.QuadPart) / 1000.0;
	}();
	return static_cast<double>(ticks) / per_ms;
}

// A reserved page of a mapped view: the alias, a guest view of the store, or any other view of a SEC_RESERVE section
// in the process (not told apart here: the handler and the commits treat them all alike, a view not known here
// committed with its mapping protection - see CommitPieceLocked).
inline bool IsMappedReserved(const MEMORY_BASIC_INFORMATION& info) {
	return info.State == MEM_RESERVE && info.Type == MEM_MAPPED;
}

// The protection a view's page has when committed with no other asked: its mapping protection.
inline DWORD CommitProtect(DWORD allocation_protect) {
	switch (allocation_protect & 0xffu) {
		case PAGE_READONLY:
		case PAGE_READWRITE:
		case PAGE_EXECUTE:
		case PAGE_EXECUTE_READ:
		case PAGE_EXECUTE_READWRITE: return allocation_protect & 0xffu;
		case PAGE_WRITECOPY: return PAGE_READONLY;
		case PAGE_EXECUTE_WRITECOPY: return PAGE_EXECUTE_READ;
		default: return PAGE_READWRITE;
	}
}

inline bool CanWrite(DWORD p) {
	p &= 0xffu;
	return p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE || p == PAGE_WRITECOPY || p == PAGE_EXECUTE_WRITECOPY;
}
inline bool CanExecute(DWORD p) {
	p &= 0xffu;
	return p == PAGE_EXECUTE || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE || p == PAGE_EXECUTE_WRITECOPY;
}
inline bool CanRead(DWORD p) {
	p &= 0xffu;
	return p != PAGE_NOACCESS && p != PAGE_EXECUTE && p != 0;
}

// Whether a view mapped with `own` can be given `protect` (else the commit-on-protect fallback, which fails as the
// committed store did).
inline bool Compatible(DWORD protect, DWORD own) {
	if ((protect & ~DWORD {0xffu}) != 0) return false; // (guard, no-cache...: not recorded)
	if ((protect & 0xffu) == PAGE_NOACCESS) return true;
	return !(CanWrite(protect) && !CanWrite(own)) && !(CanExecute(protect) && !CanExecute(own)) &&
	       !(CanRead(protect) && !CanRead(own));
}

// The store offset of `address` and the guest view holding it (owner 0: the alias); false when it is in neither.
inline bool ViewOffsetLocked(const State& state, uint64_t address, uint64_t* offset, uint64_t* owner, uint64_t* view_end) {
	if (address >= state.alias_base && address - state.alias_base < state.alias_size) {
		*offset   = address - state.alias_base;
		*owner    = 0;
		*view_end = state.alias_base + state.alias_size;
		return true;
	}
	auto it = state.views.upper_bound(address);
	if (it == state.views.begin()) return false;
	--it;
	if (address - it->first >= it->second.size) return false;
	*offset   = it->second.offset + (address - it->first);
	*owner    = it->first;
	*view_end = it->first + it->second.size;
	return true;
}

// Whether store offsets [begin, end) are mapped by one guest view only (the alias not counted).
inline bool UniqueLocked(const State& state, uint64_t begin, uint64_t end) {
	if (begin >= end) return false;
	const uint64_t first = begin >> CountBits, last = (end - 1) >> CountBits;
	if (last >= state.counts.size()) return false;
	for (uint64_t i = first; i <= last; ++i) {
		if (state.counts[i] != 1) return false;
	}
	return true;
}

// Removes the records of offsets [begin, end) (lock held; splits the runs crossing its edges).
inline void EraseWantedLocked(State& state, uint64_t begin, uint64_t end) {
	if (begin >= end || state.wanted.empty()) return;
	auto it = state.wanted.lower_bound(begin);
	if (it != state.wanted.begin()) {
		auto prev = std::prev(it);
		if (prev->second.end > begin) {
			const Wanted old       = prev->second;
			const uint64_t old_key = prev->first;
			prev->second.end       = begin;
			if (old.end > end) {
				Wanted tail = old;
				tail.guest  = old.guest + (end - old_key);
				state.wanted.emplace(end, tail);
				return;
			}
		}
	}
	while (it != state.wanted.end() && it->first < end) {
		if (it->second.end > end) {
			Wanted tail     = it->second;
			tail.guest      = it->second.guest + (end - it->first);
			state.wanted.erase(it);
			state.wanted.emplace(end, tail);
			return;
		}
		it = state.wanted.erase(it);
	}
}

// Records `protect` for reserved store offsets [begin, end) of guest view `owner`, whose page at `begin` is `guest`
// (lock held). Joined with neighbours of the same view and protection: as many records as protection runs.
inline void RecordWantedLocked(State& state, uint64_t begin, uint64_t end, uint64_t guest, uint64_t owner, DWORD protect) {
	if (begin >= end) return;
	EraseWantedLocked(state, begin, end);
	auto it = state.wanted.emplace(begin, Wanted {end, guest, owner, protect}).first;
	if (it != state.wanted.begin()) {
		auto prev = std::prev(it);
		if (prev->second.end == begin && prev->second.owner == owner && prev->second.protect == protect &&
		    prev->second.guest + (begin - prev->first) == guest) {
			prev->second.end = end;
			state.wanted.erase(it);
			it = prev;
		}
	}
	if (auto next = std::next(it); next != state.wanted.end() && next->first == it->second.end &&
	                               next->second.owner == owner && next->second.protect == protect &&
	                               it->second.guest + (next->first - it->first) == next->second.guest) {
		it->second.end = next->second.end;
		state.wanted.erase(next);
	}
	state.deferred_bytes.fetch_add(end - begin, std::memory_order_relaxed);
}

// The protection recorded for store offset `offset`, else `fallback` (lock held).
inline DWORD WantedAtLocked(const State& state, uint64_t offset, DWORD fallback) {
	auto it = state.wanted.upper_bound(offset);
	if (it == state.wanted.begin()) return fallback;
	--it;
	return offset < it->second.end ? it->second.protect : fallback;
}

// One VirtualAlloc MEM_COMMIT of [b, e) through its view with `protect`; a protection the view refuses: committed with
// no access first (nobody gets in meanwhile), then given `protect`, else `own`. No allocation (fault handler).
inline bool CommitWith(uint64_t b, uint64_t e, DWORD protect, DWORD own) {
	if (b >= e) return true;
	if (VirtualAlloc(reinterpret_cast<void*>(b), e - b, MEM_COMMIT, protect) != nullptr) return true;
	if (protect == own) return false;
	if (VirtualAlloc(reinterpret_cast<void*>(b), e - b, MEM_COMMIT, PAGE_NOACCESS) == nullptr) return false;
	DWORD old = 0;
	return VirtualProtect(reinterpret_cast<void*>(b), e - b, protect, &old) != 0 ||
	       VirtualProtect(reinterpret_cast<void*>(b), e - b, own, &old) != 0;
}

// A self-check of the per-view protections Windows gives (lock held): the committed pages of [b, e) of one view, just
// committed, should have `protect`; a page with another is given it, and counted (expected never: a commit through a
// view gives that view the protection asked and every other view its own mapping protection, as protections of views
// of one section are per view). No allocation (fault handler).
inline void EnsureProtectLocked(State& state, uint64_t b, uint64_t e, DWORD protect) {
	for (uint64_t at = b; at < e;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) return;
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const uint64_t piece_end  = region_end < e ? region_end : e;
		if (piece_end <= at) return;
		if (info.State == MEM_COMMIT && (info.Protect & 0xffu) != (protect & 0xffu)) {
			DWORD old = 0;
			if (VirtualProtect(reinterpret_cast<void*>(at), piece_end - at, protect, &old) != 0)
				state.mismatch_bytes.fetch_add(piece_end - at, std::memory_order_relaxed);
		}
		at = piece_end;
	}
}

inline void NoteCommitError(State& state) {
	state.failures.fetch_add(1, std::memory_order_relaxed);
	state.last_error.store(static_cast<uint32_t>(GetLastError()), std::memory_order_relaxed);
}

// Commits the reserved run [b, e) of one view region (lock held; `own`: that view's mapping protection). The pages
// with recorded protections first, each through the view that asked, with that protection (the store page is then
// committed for every view, the asking one never without its protection); then what is still reserved of [b, e)
// through this view with its own. A record's commit refused for lack of memory: nothing more (false; the caller
// releases and retries), so no page of it is committed without its protection. Its view gone (unmapped, not yet
// unregistered): skipped, the page committed through this one. No allocation (fault handler).
inline bool CommitPieceLocked(State& state, uint64_t b, uint64_t e, DWORD own) {
	uint64_t   offset = 0, owner = 0, view_end = 0;
	const bool known  = ViewOffsetLocked(state, b, &offset, &owner, &view_end);
	if (known && !state.wanted.empty()) {
		const uint64_t ob = offset, oe = offset + (e - b);
		auto           it = state.wanted.upper_bound(ob);
		if (it != state.wanted.begin() && std::prev(it)->second.end > ob) it = std::prev(it);
		for (; it != state.wanted.end() && it->first < oe; ++it) {
			const uint64_t rb = it->first > ob ? it->first : ob;
			const uint64_t re = it->second.end < oe ? it->second.end : oe;
			if (rb >= re) continue;
			const uint64_t guest = it->second.guest + (rb - it->first);
			// (Its view's page is the same store page as this one's: reserved still, unless its view is gone.)
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<void*>(guest), &info, sizeof(info)) != sizeof(info) || !IsMappedReserved(info))
				continue;
			if (CommitWith(guest, guest + (re - rb), it->second.protect, CommitProtect(info.AllocationProtect))) {
				state.committed.fetch_add(re - rb, std::memory_order_relaxed);
				EnsureProtectLocked(state, guest, guest + (re - rb), it->second.protect);
				continue;
			}
			NoteCommitError(state);
			if (MemGuard::IsOutOfMemoryError(state.last_error.load(std::memory_order_relaxed))) return false;
		}
	}
	// What is still reserved: the pages with no record (or whose view is gone).
	bool ok = true;
	for (uint64_t at = b; at < e;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const uint64_t piece_end  = region_end < e ? region_end : e;
		if (piece_end <= at) return false;
		if (IsMappedReserved(info)) {
			if (CommitWith(at, piece_end, own, own)) state.committed.fetch_add(piece_end - at, std::memory_order_relaxed);
			else {
				NoteCommitError(state);
				ok = false;
			}
		}
		at = piece_end;
	}
	// The alias's pages of it, just committed through some view: read-write, as the alias always is (checked: a check of
	// the per-view protections Windows gives, see EnsureProtectLocked).
	if (known) EnsureProtectLocked(state, state.alias_base + offset, state.alias_base + offset + (e - b), PAGE_READWRITE);
	return ok;
}

// Commits the reserved pages of mapped views in [begin, end) (lock held). Pages already committed are skipped: their
// protection stays.
inline bool CommitReservedLocked(State& state, uint64_t begin, uint64_t end, bool from_fault) {
	bool ok = true;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const uint64_t piece_end  = region_end < end ? region_end : end;
		if (piece_end <= at) return false;
		if (IsMappedReserved(info)) {
			const auto start = Now();
			ok               = CommitPieceLocked(state, at, piece_end, CommitProtect(info.AllocationProtect)) && ok;
			const auto spent = Now() - start;
			state.calls.fetch_add(1, std::memory_order_relaxed);
			if (from_fault) state.fault_commits.fetch_add(1, std::memory_order_relaxed);
			state.ticks.fetch_add(spent, std::memory_order_relaxed);
			for (auto max = state.max_ticks.load(std::memory_order_relaxed);
			     spent > max && !state.max_ticks.compare_exchange_weak(max, spent, std::memory_order_relaxed);) {
			}
		}
		at = piece_end;
	}
	return ok;
}

// Page bounds of [address, address + size).
inline void PageBounds(const void* address, uint64_t size, uint64_t* begin, uint64_t* end) {
	*begin = reinterpret_cast<uint64_t>(address) & ~uint64_t {0xfff};
	*end   = (reinterpret_cast<uint64_t>(address) + size + 0xfff) & ~uint64_t {0xfff};
}

// Whether [begin, end) has a reserved page of a mapped view (no lock: a committed page stays committed).
inline bool AnyMappedReserved(uint64_t begin, uint64_t end, bool* other_uncommitted) {
	bool any = false;
	if (other_uncommitted != nullptr) *other_uncommitted = false;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) {
			if (other_uncommitted != nullptr) *other_uncommitted = true;
			return any;
		}
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		if (region_end <= at) {
			if (other_uncommitted != nullptr) *other_uncommitted = true;
			return any;
		}
		if (IsMappedReserved(info)) any = true;
		else if (info.State != MEM_COMMIT && other_uncommitted != nullptr) *other_uncommitted = true;
		at = region_end;
	}
	return any;
}

// Runs `attempt` under the lock; *error: the error of a commit it saw refused (0: none), read under the lock.
template <typename F>
inline bool LockedAttempt(State& state, uint32_t* error, F&& attempt) {
	AcquireSRWLockExclusive(&state.lock);
	state.last_error.store(0, std::memory_order_relaxed);
	const bool ok = attempt();
	*error        = state.last_error.load(std::memory_order_relaxed);
	ReleaseSRWLockExclusive(&state.lock);
	return ok;
}

// Commits the reserved pages of [address, address + size), each with the protection it would have had committed from
// the start (before a kernel-mode access, a stack, a readability check).
inline bool Prefault(const void* address, uint64_t size) {
	auto& state = Get();
	if (!state.enabled || address == nullptr || size == 0) return true;
	uint64_t begin = 0, end = 0;
	PageBounds(address, size, &begin, &end);
	if (!AnyMappedReserved(begin, end, nullptr)) return true;
	uint32_t   error = 0;
	const auto run   = [&] { return CommitReservedLocked(state, begin, end, false); };
	bool       ok    = LockedAttempt(state, &error, run);
	for (int attempt = 0; !ok && MemGuard::IsOutOfMemoryError(error) &&
	                      MemGuard::ShouldRetry("guest memory commit (prefault)", end - begin, attempt);
	     ++attempt)
		ok = LockedAttempt(state, &error, run);
	return ok;
}

// The locked pass of Protect: committed pages protected, reserved pages of a sole guest view recorded, the others
// committed (with what is recorded for them) and protected.
inline bool ProtectLocked(State& state, uint64_t begin, uint64_t end, DWORD protect, DWORD* old_protect) {
	bool  first = true;
	DWORD dummy = 0;
	for (uint64_t at = begin; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
		const uint64_t region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const uint64_t piece_end  = region_end < end ? region_end : end;
		if (piece_end <= at) return false;
		DWORD* old = first && old_protect != nullptr ? old_protect : &dummy;
		first      = false;
		if (IsMappedReserved(info)) {
			const DWORD own    = CommitProtect(info.AllocationProtect);
			uint64_t    offset = 0, owner = 0, view_end = 0;
			const bool  known  = ViewOffsetLocked(state, at, &offset, &owner, &view_end);
			if (known && owner != 0 && piece_end <= view_end && Compatible(protect, own) &&
			    UniqueLocked(state, offset, offset + (piece_end - at))) {
				*old = WantedAtLocked(state, offset, own);
				RecordWantedLocked(state, offset, offset + (piece_end - at), at, owner, protect);
				at = piece_end;
				continue;
			}
			// (Aliased, the alias, a protection the view cannot take, or a view not known here: committed first.)
			state.fallback_bytes.fetch_add(piece_end - at, std::memory_order_relaxed);
			if (!CommitReservedLocked(state, at, piece_end, false)) return false;
		}
		if (VirtualProtect(reinterpret_cast<void*>(at), piece_end - at, protect, old) == 0) return false;
		at = piece_end;
	}
	return true;
}

// A VirtualProtect for guest memory (SysVirtualProtect, the store's views, the texture offload's alias pages): see the
// top. *old_protect: the first page's (for a reserved page, what it would have had).
inline bool ProtectUnpinned(void* address, uint64_t size, DWORD protect, DWORD* old_protect);
// KYTY_PIN_WATCHED (pin-watched.h): the section pages of the range locked in RAM before their protection changes, and
// unlocked after a protection back to read-write (bug check 0x1A/0x41202: a page paged while its protection changes).
inline bool Protect(void* address, uint64_t size, DWORD protect, DWORD* old_protect) {
	PinWatched::Scope pin(reinterpret_cast<uint64_t>(address), size, protect);
	pin.applied = ProtectUnpinned(address, size, protect, old_protect);
	return pin.applied;
}
inline bool ProtectUnpinned(void* address, uint64_t size, DWORD protect, DWORD* old_protect) {
	auto& state = Get();
	if (!state.enabled || size == 0) return VirtualProtect(address, size, protect, old_protect) != 0;
	uint64_t begin = 0, end = 0;
	PageBounds(address, size, &begin, &end);
	// (No reserved page of a view - most calls once the pages are touched -: no lock. A range with pages neither
	// committed nor ours (free, private reserved): the plain call, which refuses it as before.)
	bool other = false;
	if (!AnyMappedReserved(begin, end, &other) || other) return VirtualProtect(address, size, protect, old_protect) != 0;
	uint32_t   error = 0;
	const auto run   = [&] { return ProtectLocked(state, begin, end, protect, old_protect); };
	bool       ok    = LockedAttempt(state, &error, run);
	for (int attempt = 0; !ok && MemGuard::IsOutOfMemoryError(error) &&
	                      MemGuard::ShouldRetry("guest memory commit (protect)", end - begin, attempt);
	     ++attempt)
		ok = LockedAttempt(state, &error, run);
	return ok;
}

inline void CountLocked(State& state, uint64_t offset, uint64_t size, bool add) {
	if (size == 0) return;
	const uint64_t first = offset >> CountBits, last = (offset + size - 1) >> CountBits;
	if (last >= state.counts.size()) return;
	for (uint64_t i = first; i <= last; ++i) {
		auto& count = state.counts[i];
		if (count == 255) continue; // (many: kept, never unique again)
		if (add) ++count;
		else if (count != 0) --count;
	}
}

// Forgets a guest view: its records (only it asked for them) and its count (lock held).
inline void RemoveViewLocked(State& state, std::map<uint64_t, View>::iterator it) {
	const uint64_t base = it->first;
	const View     view = it->second;
	for (auto w = state.wanted.lower_bound(view.offset); w != state.wanted.end() && w->first < view.offset + view.size;) {
		if (w->second.owner == base) w = state.wanted.erase(w);
		else ++w;
	}
	CountLocked(state, view.offset, view.size, false);
	state.views.erase(it);
}

// A guest view of store offsets [offset, offset + size) about to be mapped at `base` (before the map: a fault on it
// then finds its offset). Unregister(base) undoes it when the map fails. (Views still known in [base, base + size) -
// none, every unmap unregisters - are forgotten first.)
inline void Register(uint64_t base, uint64_t size, uint64_t offset) {
	auto& state = Get();
	if (!state.enabled || size == 0) return;
	AcquireSRWLockExclusive(&state.lock);
	auto it = state.views.upper_bound(base);
	if (it != state.views.begin() && std::prev(it)->first + std::prev(it)->second.size > base) it = std::prev(it);
	while (it != state.views.end() && it->first < base + size) {
		auto next = std::next(it);
		RemoveViewLocked(state, it);
		it = next;
	}
	state.views.emplace(base, View {size, offset});
	CountLocked(state, offset, size, true);
	ReleaseSRWLockExclusive(&state.lock);
}

// The guest view holding `address` unmapped (after the unmap: its records were its own, and are forgotten).
inline void Unregister(uint64_t address) {
	auto& state = Get();
	if (!state.enabled) return;
	AcquireSRWLockExclusive(&state.lock);
	auto it = state.views.upper_bound(address);
	if (it != state.views.begin() && address - std::prev(it)->first < std::prev(it)->second.size)
		RemoveViewLocked(state, std::prev(it));
	ReleaseSRWLockExclusive(&state.lock);
}

// Whether a committed page's protection allows the access of an access violation (ExceptionInformation[0]: 0 read,
// 1 write, 8 execute; anything else: no).
inline bool AllowsAccess(DWORD protect, ULONG_PTR access) {
	if ((protect & PAGE_GUARD) != 0) return false;
	if (access == 0) return CanRead(protect);
	if (access == 1) return CanWrite(protect);
	if (access == 8) return CanExecute(protect);
	return false;
}

// Per thread (no allocation: static TLS, constant-initialized): the address of the last fault resumed because its page
// was found committed already, and how many times in a row it was. Capped, so an access that keeps faulting on a page
// that looks accessible goes to the other handlers instead of looping.
constexpr uint32_t MaxCommittedRetries = 4;
inline thread_local uint64_t g_retry_address = 0;
inline thread_local uint32_t g_retry_count   = 0;

// A fault on a page found committed: another thread committed its granule between this fault and the handler's look
// (two threads first-touching one granule). Resumed when the page is in a known guest view or the alias and its
// protection allows the access, at most MaxCommittedRetries times in a row at one address; else left to the others
// (a watch's protection: the emulator's handler, as on a committed store).
inline LONG CommittedMeanwhile(State& state, const MEMORY_BASIC_INFORMATION& info, uint64_t address, ULONG_PTR access) {
	if (info.State != MEM_COMMIT || info.Type != MEM_MAPPED || !AllowsAccess(info.Protect, access))
		return EXCEPTION_CONTINUE_SEARCH;
	uint64_t offset = 0, owner = 0, view_end = 0;
	AcquireSRWLockShared(&state.lock);
	const bool known = ViewOffsetLocked(state, address, &offset, &owner, &view_end);
	ReleaseSRWLockShared(&state.lock);
	if (!known) return EXCEPTION_CONTINUE_SEARCH;
	if (g_retry_address != address) {
		g_retry_address = address;
		g_retry_count   = 0;
	}
	if (++g_retry_count > MaxCommittedRetries) {
		g_retry_count = 0;
		return EXCEPTION_CONTINUE_SEARCH;
	}
	return EXCEPTION_CONTINUE_EXECUTION;
}

inline LONG CALLBACK Handler(EXCEPTION_POINTERS* pointers) {
	auto& state = Get();
	if (!state.enabled || pointers == nullptr || pointers->ExceptionRecord == nullptr ||
	    pointers->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
	    pointers->ExceptionRecord->NumberParameters < 2)
		return EXCEPTION_CONTINUE_SEARCH;
	const auto access  = pointers->ExceptionRecord->ExceptionInformation[0];
	const auto address = static_cast<uint64_t>(pointers->ExceptionRecord->ExceptionInformation[1]);
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) != sizeof(info))
		return EXCEPTION_CONTINUE_SEARCH;
	if (!IsMappedReserved(info)) return CommittedMeanwhile(state, info, address, access);
	g_retry_count = 0; // (a commit: the run of resumptions without one ends)
	// The granule around the fault, inside the reserved run and its view.
	const uint64_t run_begin = reinterpret_cast<uint64_t>(info.BaseAddress);
	const uint64_t run_end   = run_begin + info.RegionSize;
	uint64_t       begin     = address & ~(state.granule - 1);
	uint64_t       end       = begin + state.granule;
	if (begin < run_begin) begin = run_begin;
	if (end > run_end) end = run_end;
	uint32_t   error     = 0;
	const auto run       = [&] { return CommitReservedLocked(state, begin, end, true); };
	bool       committed = LockedAttempt(state, &error, run);
	// Refused for lack of commit: a release pass (MemGuard, outside the lock) and another try; still refused, the clear
	// out-of-memory message instead of an access violation in guest code.
	for (int attempt = 0; !committed && MemGuard::IsOutOfMemoryError(error); ++attempt) {
		if (!MemGuard::ShouldRetry("guest memory commit (lazy)", end - begin, attempt))
			MemGuard::FatalOutOfMemory("guest memory commit (lazy)", end - begin);
		committed = LockedAttempt(state, &error, run);
	}
	// Committed now: run it again (a protection recorded for it faults next, in the emulator's handler, as on a
	// committed store). Still reserved (refused): left to the others.
	MEMORY_BASIC_INFORMATION after {};
	if (VirtualQuery(reinterpret_cast<void*>(address), &after, sizeof(after)) == sizeof(after) && after.State == MEM_COMMIT)
		return EXCEPTION_CONTINUE_EXECUTION;
	return EXCEPTION_CONTINUE_SEARCH;
}

// The store's mapped bytes (what KYTY_BACKING_ON_DEMAND would have committed at map): granules first covered.
inline void NoteMapped(uint64_t bytes) {
	Get().mapped.fetch_add(bytes, std::memory_order_relaxed);
}

// Whether the page at `address` is a reserved (never committed) page of a view: its bytes are zeros once committed, so
// a zero fill of it is not needed (and would only commit it).
inline bool IsReservedPage(const void* address) {
	if (!Get().enabled) return false;
	MEMORY_BASIC_INFORMATION info {};
	return VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && IsMappedReserved(info);
}

// The end of the run of pages starting at `address` that share its state (VirtualQuery), at most `end`.
inline uint64_t RunEnd(uint64_t address, uint64_t end) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) != sizeof(info)) return end;
	const uint64_t run_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
	return run_end > address && run_end < end ? run_end : end;
}

// Called by the store once its reserved section and alias exist.
inline void Start(uint64_t alias_base, uint64_t store_size) {
	auto& state = Get();
	if (const char* kb = std::getenv("KYTY_GUEST_LAZY_COMMIT_KB"); kb != nullptr) {
		const auto value = std::strtoull(kb, nullptr, 10);
		// A power of two from 4 KiB to 4 MiB.
		if (value >= 4 && value <= 4096 && (value & (value - 1)) == 0) state.granule = value << 10u;
	}
	state.alias_base = alias_base;
	state.alias_size = store_size;
	state.store_size = store_size;
	state.counts.assign(static_cast<size_t>((store_size + (uint64_t {1} << CountBits) - 1) >> CountBits), 0);
	if (AddVectoredExceptionHandler(1, Handler) == nullptr) {
		std::printf("Guest lazy commit: no exception handler; the store is committed instead\n");
		VirtualAlloc(reinterpret_cast<void*>(alias_base), store_size, MEM_COMMIT, PAGE_READWRITE);
		return;
	}
	state.enabled = true;
	std::printf("Guest lazy commit: on (KYTY_GUEST_LAZY_COMMIT=1), %llu MiB store reserved, committed at "
	            "first touch in %llu KiB granules; nothing is ever decommitted or moved\n",
	            static_cast<unsigned long long>(store_size >> 20u), static_cast<unsigned long long>(state.granule >> 10u));
	std::fflush(stdout);
}

// The periodic line (30 s; printf: LOGF is silent in the release build). written_bytes/watched_bytes: fresh memory's.
inline void Report(uint64_t written_bytes, uint64_t watched_bytes) {
	auto& state = Get();
	if (!state.enabled) return;
	const auto committed = state.committed.load(std::memory_order_relaxed);
	const auto mapped    = state.mapped.load(std::memory_order_relaxed);
	const auto calls     = state.calls.load(std::memory_order_relaxed);
	const auto ticks     = state.ticks.load(std::memory_order_relaxed);
	AcquireSRWLockShared(&state.lock);
	const auto records = static_cast<unsigned long long>(state.wanted.size());
	const auto views   = static_cast<unsigned long long>(state.views.size());
	ReleaseSRWLockShared(&state.lock);
	std::printf("Guest lazy commit (30 s): %llu MiB committed of %llu MiB mapped by the game (%llu MiB mapped, never "
	            "committed: the saving over commit-at-map; store %llu MiB)\n",
	            static_cast<unsigned long long>(committed >> 20u), static_cast<unsigned long long>(mapped >> 20u),
	            static_cast<unsigned long long>((mapped > committed ? mapped - committed : 0) >> 20u),
	            static_cast<unsigned long long>(state.store_size >> 20u));
	std::printf("Guest lazy commit: %llu commits (%llu at a fault, %llu failed), %.1f ms committing (%.3f ms longest, "
	            "%.1f us average); protections of reserved pages: %llu MiB recorded (%llu runs kept, %llu views), %llu "
	            "MiB committed first (aliased/alias); %llu KiB of protections fixed after a commit (expected 0); fresh "
	            "memory: %llu MiB written, %llu MiB watched\n",
	            static_cast<unsigned long long>(calls),
	            static_cast<unsigned long long>(state.fault_commits.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(state.failures.load(std::memory_order_relaxed)), TicksToMs(ticks),
	            TicksToMs(state.max_ticks.load(std::memory_order_relaxed)),
	            calls != 0 ? TicksToMs(ticks) * 1000.0 / static_cast<double>(calls) : 0.0,
	            static_cast<unsigned long long>(state.deferred_bytes.load(std::memory_order_relaxed) >> 20u), records, views,
	            static_cast<unsigned long long>(state.fallback_bytes.load(std::memory_order_relaxed) >> 20u),
	            static_cast<unsigned long long>(state.mismatch_bytes.load(std::memory_order_relaxed) >> 10u),
	            static_cast<unsigned long long>(written_bytes >> 20u), static_cast<unsigned long long>(watched_bytes >> 20u));
}

} // namespace GuestLazyCommit

#else

#include <cstdint>

namespace GuestLazyCommit {
inline bool Requested() { return false; }
inline bool Enabled() { return false; }
inline bool Prefault(const void*, uint64_t) { return true; }
inline void NoteMapped(uint64_t) {}
inline bool IsReservedPage(const void*) { return false; }
inline uint64_t RunEnd(uint64_t, uint64_t end) { return end; }
inline void Register(uint64_t, uint64_t, uint64_t) {}
inline void Unregister(uint64_t) {}
inline void Report(uint64_t, uint64_t) {}
} // namespace GuestLazyCommit

#endif
