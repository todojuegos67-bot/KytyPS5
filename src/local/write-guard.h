#ifndef KYTY_LOCAL_WRITE_GUARD_H_
#define KYTY_LOCAL_WRITE_GUARD_H_

// KYTY_WRITE_GUARD (default 0; 1 turns it on): a guest memory write of the command processor (WriteGuestMemory:
// WRITE_DATA, an end-of-pipe label, the reference clock) made while older upload jobs are pending lands at once and
// is hidden from the game until those jobs are done, instead of waiting for them (AsyncUpload::Drain, issue #196).
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
// * An unmap drops the guards of its range (Drop).

#include "common/virtualMemory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_write_guard_mode;
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
// The upload worker: it cannot wait for a lift its own queue holds (async-upload.h sets it).
inline thread_local bool t_no_wait = false;
// The upload worker's completed sequence (set by the GPU thread's first guarded write): a wait whose queue has not moved
// for a second is a cycle (OnFault).
inline uint64_t (*g_upload_done)() = nullptr;

[[nodiscard]] inline bool Enabled() {
	return kyty_local_write_guard_mode.load(std::memory_order_relaxed) != 0;
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
	if (!Common::VirtualMemory::Protect(page, Page, Common::VirtualMemory::Mode::NoAccess, &entry.saved)) return false;
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
	g_pages.erase(it);
	g_count.fetch_sub(1, std::memory_order_release);
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
			it = g_pages.erase(it);
			g_count.fetch_sub(1, std::memory_order_release);
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
