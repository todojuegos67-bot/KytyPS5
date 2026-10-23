#pragma once
// Guest lazy commit (KYTY_GUEST_LAZY_COMMIT=1, default off; Windows, page-file store only).
//
// The guest backing store (one 13824 MiB page-file section) is created SEC_RESERVE instead of SEC_COMMIT, and its
// pages are committed when first touched, in granules of KYTY_GUEST_LAZY_COMMIT_KB (default 64 KiB, the fresh-memory
// granule), instead of all 13.5 GiB charged to the commit limit at launch. Commit of a section page is made through
// whichever view touches it (VirtualAlloc MEM_COMMIT on a view of a SEC_RESERVE section commits the section's page
// for every view of it).
//
//  - CPU touch (guest code or host code, through a guest view or the backing alias): a reserved page faults; the
//    vectored handler here (first in the chain) commits the granule around it with the view's own protection and the
//    instruction runs again. Nothing else is in that path (no emulator lock): a page still reserved has no protection
//    of anyone's to keep, every protection change commits its pages first (below).
//  - Protection changes (the GPU trackers' write/read watches, guest mprotect, a no-access view): the pages are
//    committed first (Protect), as Windows cannot protect reserved pages. A watched range is committed at its watch.
//  - Kernel-mode access (ReadFile into a guest buffer) does not fault in user mode: the buffer is committed first
//    (Prefault, called by the file reads).
// Both commit paths hold one lock and commit only pages still reserved: a commit of a committed page would reset
// its protection (a watch lost).
//
// Decommit of freed guest memory is not done: a view of a section cannot be decommitted (VirtualFree MEM_DECOMMIT
// refuses mapped views); only unmapping every view and recreating the section would free it.

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace GuestLazyCommit {

struct State {
	bool                  enabled = false;
	uint64_t              granule = 64u << 10u;
	uint64_t              alias_base = 0, alias_size = 0; // the backing alias
	uint64_t              store_size = 0;
	SRWLOCK               lock = SRWLOCK_INIT;
	std::atomic<uint64_t> committed {0};     // bytes committed by this module
	std::atomic<uint64_t> calls {0};         // VirtualAlloc commit calls
	std::atomic<uint64_t> fault_commits {0}; // of them, from a fault
	std::atomic<uint64_t> ticks {0};         // time in the commits (QueryPerformanceCounter ticks)
	std::atomic<uint64_t> max_ticks {0};     // the longest single commit
	std::atomic<uint64_t> failures {0};
};

inline State& Get() {
	static State state;
	return state;
}

// Read once: the store's constructor decides SEC_RESERVE with it.
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

// A view (MEM_MAPPED) of the store: the alias, or a guest view of the same section (any other mapped view is not
// ours: the section of a SEC_RESERVE mapping elsewhere in the process - none is made - would be treated alike, which
// is harmless: a reserved page of a view faults either way).
inline bool IsOurReserved(const MEMORY_BASIC_INFORMATION& info) {
	return info.State == MEM_RESERVE && info.Type == MEM_MAPPED;
}

inline DWORD CommitProtect(DWORD allocation_protect) {
	switch (allocation_protect & 0xffu) {
		case PAGE_READONLY:
		case PAGE_READWRITE:
		case PAGE_EXECUTE_READ:
		case PAGE_EXECUTE_READWRITE: return allocation_protect & 0xffu;
		case PAGE_WRITECOPY: return PAGE_READONLY;
		case PAGE_EXECUTE_WRITECOPY: return PAGE_EXECUTE_READ;
		default: return PAGE_READWRITE;
	}
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
		if (IsOurReserved(info)) {
			const auto start = Now();
			const bool done  = VirtualAlloc(reinterpret_cast<void*>(at), piece_end - at, MEM_COMMIT,
			                                CommitProtect(info.AllocationProtect)) != nullptr;
			const auto spent = Now() - start;
			state.calls.fetch_add(1, std::memory_order_relaxed);
			if (from_fault) state.fault_commits.fetch_add(1, std::memory_order_relaxed);
			state.ticks.fetch_add(spent, std::memory_order_relaxed);
			for (auto max = state.max_ticks.load(std::memory_order_relaxed);
			     spent > max && !state.max_ticks.compare_exchange_weak(max, spent, std::memory_order_relaxed);) {
			}
			if (done) state.committed.fetch_add(piece_end - at, std::memory_order_relaxed);
			else {
				state.failures.fetch_add(1, std::memory_order_relaxed);
				ok = false;
			}
		}
		at = piece_end;
	}
	return ok;
}

// Commits the reserved pages of [address, address + size) (a protection change or a kernel-mode access follows).
inline bool Prefault(const void* address, uint64_t size) {
	auto& state = Get();
	if (!state.enabled || address == nullptr || size == 0) return true;
	const auto begin = reinterpret_cast<uint64_t>(address) & ~uint64_t {0xfff};
	const auto end   = (reinterpret_cast<uint64_t>(address) + size + 0xfff) & ~uint64_t {0xfff};
	// (A quick look first: a range already committed - most of them - takes no lock.)
	{
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(begin), &info, sizeof(info)) == sizeof(info) &&
		    !IsOurReserved(info) && reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize >= end)
			return true;
	}
	AcquireSRWLockExclusive(&state.lock);
	const bool ok = CommitReservedLocked(state, begin, end, false);
	ReleaseSRWLockExclusive(&state.lock);
	return ok;
}

// A VirtualProtect that commits the reserved pages first.
inline bool Protect(void* address, uint64_t size, DWORD protect, DWORD* old_protect) {
	if (Get().enabled) Prefault(address, size);
	if (VirtualProtect(address, size, protect, old_protect) != 0) return true;
	// (Refused: a page committed between the walk's look and the call's, or a region the quick look missed.)
	return Get().enabled && Prefault(address, size) && VirtualProtect(address, size, protect, old_protect) != 0;
}

inline LONG CALLBACK Handler(EXCEPTION_POINTERS* pointers) {
	auto& state = Get();
	if (!state.enabled || pointers == nullptr || pointers->ExceptionRecord == nullptr ||
	    pointers->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
	    pointers->ExceptionRecord->NumberParameters < 2)
		return EXCEPTION_CONTINUE_SEARCH;
	const auto address = static_cast<uint64_t>(pointers->ExceptionRecord->ExceptionInformation[1]);
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) != sizeof(info) || !IsOurReserved(info))
		return EXCEPTION_CONTINUE_SEARCH;
	// The granule around the fault, inside the reserved run and its view.
	const uint64_t run_begin = reinterpret_cast<uint64_t>(info.BaseAddress);
	const uint64_t run_end   = run_begin + info.RegionSize;
	uint64_t       begin     = address & ~(state.granule - 1);
	uint64_t       end       = begin + state.granule;
	if (begin < run_begin) begin = run_begin;
	if (end > run_end) end = run_end;
	AcquireSRWLockExclusive(&state.lock);
	CommitReservedLocked(state, begin, end, true);
	ReleaseSRWLockExclusive(&state.lock);
	// Retried: committed now (a protection fault next goes to the emulator's handler), or, when the commit failed
	// (commit limit), the same fault is not ours the next time round only if the page changed; leave it to the others.
	MEMORY_BASIC_INFORMATION after {};
	if (VirtualQuery(reinterpret_cast<void*>(address), &after, sizeof(after)) == sizeof(after) && after.State == MEM_COMMIT)
		return EXCEPTION_CONTINUE_EXECUTION;
	return EXCEPTION_CONTINUE_SEARCH;
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
	if (AddVectoredExceptionHandler(1, Handler) == nullptr) {
		std::printf("Guest lazy commit: no exception handler; the store is committed instead\n");
		VirtualAlloc(reinterpret_cast<void*>(alias_base), store_size, MEM_COMMIT, PAGE_READWRITE);
		return;
	}
	state.enabled = true;
	std::printf("Guest lazy commit: on (KYTY_GUEST_LAZY_COMMIT=1), %llu MiB store reserved, committed at first touch in "
	            "%llu KiB granules\n",
	            static_cast<unsigned long long>(store_size >> 20u), static_cast<unsigned long long>(state.granule >> 10u));
	std::fflush(stdout);
}

// The periodic line (with the fresh-memory figures: written bytes are the floor of what must be committed).
inline void Report(uint64_t written_bytes, uint64_t watched_bytes) {
	auto& state = Get();
	if (!state.enabled) return;
	const auto committed = state.committed.load(std::memory_order_relaxed);
	const auto calls     = state.calls.load(std::memory_order_relaxed);
	const auto ticks     = state.ticks.load(std::memory_order_relaxed);
	std::printf("Guest lazy commit: %llu of %llu MiB store committed (%llu MiB never committed: the commit saved); "
	            "%llu commit calls (%llu at a fault, %llu failed), %.1f ms committing (%.3f ms longest, %.1f us average); "
	            "fresh memory: %llu MiB written, %llu MiB watched (at most %llu MiB can stay uncommitted: the store less "
	            "the written)\n",
	            static_cast<unsigned long long>(committed >> 20u), static_cast<unsigned long long>(state.store_size >> 20u),
	            static_cast<unsigned long long>((state.store_size - (committed < state.store_size ? committed : state.store_size)) >> 20u),
	            static_cast<unsigned long long>(calls),
	            static_cast<unsigned long long>(state.fault_commits.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(state.failures.load(std::memory_order_relaxed)), TicksToMs(ticks),
	            TicksToMs(state.max_ticks.load(std::memory_order_relaxed)),
	            calls != 0 ? TicksToMs(ticks) * 1000.0 / static_cast<double>(calls) : 0.0,
	            static_cast<unsigned long long>(written_bytes >> 20u), static_cast<unsigned long long>(watched_bytes >> 20u),
	            static_cast<unsigned long long>((written_bytes < state.store_size ? state.store_size - written_bytes : 0) >> 20u));
}

} // namespace GuestLazyCommit

#else

#include <cstdint>

namespace GuestLazyCommit {
inline bool Requested() { return false; }
inline bool Enabled() { return false; }
inline bool Prefault(const void*, uint64_t) { return true; }
inline void Report(uint64_t, uint64_t) {}
} // namespace GuestLazyCommit

#endif
