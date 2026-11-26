#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "live-counters.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "writeback-verify.h"
#include "raiz-wb-guard.h" // (crash-ba2b1f: KYTY_RAIZ_WB_GUARD)
#include "loader/demonsSoulsWriteAudit.h"
#include "local/prot-menos.h" // (PROT-MENOS) KYTY_PROT_MENOS: hot pages
#include "local/nativo-cubierto.h" // (CPUHEAP-SIN-VIGILAR) KYTY_CPUHEAP_SIN_VIGILAR

#include <algorithm>
#include <atomic>
#include <immintrin.h>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <pthread.h>
#include <sched.h>
#elif defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		if (m_owner.load(std::memory_order_relaxed) == thread) {
			EXIT("recursive region tracking lock\n");
		}
		uint32_t spins = 0;
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			if (m_owner.load(std::memory_order_relaxed) == thread) {
				EXIT("recursive region tracking lock while contended\n");
			}
			// Wait with plain loads: repeated locked writes would keep the line away from the
			// owner and slow the very critical section being waited for. Past a short spin the
			// waiter yields its CPU: guest threads streaming textures fault into the texture
			// cache at once, and pure spinners starved a descheduled owner (the render thread,
			// 100+ ms frames with its samples parked inside the critical section).
			do {
				if (++spins < 512) {
					_mm_pause();
				} else {
					YieldCpu();
				}
			} while (m_lock.test(std::memory_order_relaxed));
		}
		m_owner.store(thread, std::memory_order_relaxed);
		// Odd while held: ReadShared retries a read that overlapped a holder.
		m_seq.store(m_seq.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
	}
	void unlock() noexcept {
		if (m_owner.load(std::memory_order_relaxed) != CurrentThread()) {
			EXIT("region tracking lock released by non-owner\n");
		}
		m_seq.store(m_seq.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}
	[[nodiscard]] bool OwnedByCurrentThread() const noexcept { return m_owner.load(std::memory_order_relaxed) == CurrentThread(); }
	// A read of state only holders of this lock write, without taking it: valid when no holder
	// was inside while it ran (the sequence even and unchanged), else done again under the lock.
	// The query then costs no locked instruction and leaves the lock's line to the writers.
	template <typename Read>
	auto ReadShared(Read&& read) {
		const auto before = m_seq.load(std::memory_order_acquire);
		if ((before & 1u) == 0) {
			auto result = read();
			std::atomic_thread_fence(std::memory_order_acquire);
			if (m_seq.load(std::memory_order_relaxed) == before) return result;
		}
		lock();
		auto result = read();
		unlock();
		return result;
	}

	static void YieldCpu() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SwitchToThread();
#else
		sched_yield();
#endif
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// TEB ClientId.UniqueThread, what GetCurrentThreadId returns: one load instead of a
		// call into kernel32 on every lock and unlock.
		return static_cast<uint32_t>(__readgsqword(0x48));
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
	std::atomic_uint32_t m_seq {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Advances whenever pages become CPU-dirty (ChangeState<Cpu, true>, a new region's pages). A region's LastDirtyClock is
// the value of its last such change: the CPU-dirty pages of a region whose clock is not above a value read earlier were
// dirty all the time since (nothing made them clean: see CommandScheduler::UploadPrologue).
inline std::atomic<uint64_t> g_cpu_dirty_clock {1};
// (policy1, KYTY_DS_NEIGHBOR_SPLIT) Clean pages an upload left without their write watcher because they are no-watch
// (RegionManager::SetNoWatch): the protection changes the policy saved, for its line.
inline std::atomic<uint64_t> g_nowatch_kept_pages {0};
[[nodiscard]] inline uint64_t NextCpuDirtyClock() noexcept {
	return g_cpu_dirty_clock.fetch_add(1, std::memory_order_acq_rel) + 1;
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	// (crash-ba2b1f) Crash reports: the bits of the page of `vaddr` (inside this region), read WITHOUT the lock (a hint:
	// the crashing thread may hold it). Libs::LibKernel::Memory::RaizPeek* bits.
	[[nodiscard]] uint32_t PeekPageUnlocked(uint64_t vaddr) const noexcept {
		if (vaddr < m_cpu_addr || vaddr - m_cpu_addr >= TRACKER_REGION_SIZE) return 0;
		const auto page = static_cast<size_t>((vaddr - m_cpu_addr) / TRACKER_PAGE_SIZE);
		namespace M     = ::Libs::LibKernel::Memory;
		return M::RaizPeekRegion | (m_cpu_dirty.Get(page) ? M::RaizPeekCpuDirty : 0u) | (m_gpu_dirty.Get(page) ? M::RaizPeekGpuDirty : 0u) |
		       (m_writable.Get(page) ? M::RaizPeekWritable : 0u) | (m_readable.Get(page) ? M::RaizPeekReadable : 0u) |
		       (m_nowatch.Get(page) ? M::RaizPeekNoWatch : 0u) | (m_nosplit.Get(page) ? M::RaizPeekNoSplit : 0u);
	}
	// See g_cpu_dirty_clock.
	[[nodiscard]] uint64_t LastDirtyClock() const noexcept { return m_last_dirty.load(std::memory_order_acquire); }
	[[nodiscard]] uint64_t CpuModificationEpoch() const {
		// Pages marked clean whose write protection is still queued can change unseen:
		// nothing is provable until the worker applied it (and moved the epoch).
		if (m_deferred_protects.load(std::memory_order_acquire) != 0) return 0;
		return m_cpu_epoch.load(std::memory_order_acquire);
	}
	// KYTY_ASYNC_REPROTECT: a write protection of pages in this region is queued.
	void BeginDeferredProtection() noexcept { m_deferred_protects.fetch_add(1, std::memory_order_acq_rel); }
	[[nodiscard]] bool DeferredProtectionPending() const noexcept {
		return m_deferred_protects.load(std::memory_order_acquire) != 0;
	}
	// The upload worker (or the render thread as a fallback) applies it.
	void ApplyDeferredProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ReapplyProtection(vaddr, size);
		m_cpu_epoch.fetch_add(1, std::memory_order_acq_rel);
		m_deferred_protects.fetch_sub(1, std::memory_order_acq_rel);
	}

	template <DirtySource source, bool all = false>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const auto& bits        = GetBits<source>();
		if constexpr (all) return bits.AllInRange(start, end);
		return bits.AnyInRange(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.AnyInRange(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (m_cpu_dirty.AnyInRange(start, end)) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		// (PROT-MENOS) A CPU write (fault) over write-protected pages: the second one inside a window makes them hot.
		if constexpr (source == DirtySource::Cpu && enable) {
			if (ProtMenos::On()) NoteCpuWrite(start, end);
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			if constexpr (enable) {
				// Invalidate cached clean proofs before another CPU thread can write
				// through the relaxed host protection. A cache miss takes this lock.
				m_cpu_epoch.fetch_add(1, std::memory_order_release);
				m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
				BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
			}
			UpdateCpuProtection<!enable>(start, end);
		} else {
			// (GPU-written pages given back to the CPU hold the GPU's bytes now: the epoch moves, NoteHostWrite.)
			if constexpr (!enable) {
				m_cpu_epoch.fetch_add(1, std::memory_order_release);
				NoteVolatileClean(); // (the pages given back are clean now)
			}
			UpdateGpuProtection<enable>();
		}
	}

	// The emulator wrote CPU-dirty pages of the region (their state stays): what a proof by the epoch holds of their
	// bytes holds no longer (TableXpr::RingCopy's copies of guest-written ranges). The epoch moves as well when pages
	// become CPU-dirty and when GPU-written pages are given back (their bytes the GPU's): with the guest's writes to
	// dirty pages it covers every change of the region's bytes the CPU can see but those.
	void NoteHostWrite() noexcept { m_cpu_epoch.fetch_add(1, std::memory_order_release); }

	// CPU-dirty with the pages' write protection kept: the emulator wrote them through the backing view (no fault
	// saw it). A page still write-watched stays so (UpdateCpuProtection moves only pages of a change's range, in its
	// direction): its next upload finds it protected already, and a guest write before that faults once and drops
	// the watcher as for any dirty page (ChangeState). Returns false (nothing changed) when the GPU wrote a page.
	[[nodiscard]] bool MarkCpuDirtyKeepProtection(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (m_gpu_dirty.AnyInRange(start, end)) return false;
		m_cpu_dirty.SetRange(start, end);
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		return true;
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (!GetBits<source>().AnyInRange(start, end)) return;
		RegionBits mask(GetBits<source>(), start, end);
		if constexpr (clear) {
			GetBits<source>().UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu && clear) {
			UpdateCpuProtection<true>(start, end);
			ForEachRange(mask, std::forward<Func>(func));
			return;
		}
		if constexpr (source == DirtySource::Gpu && clear) {
			m_cpu_epoch.fetch_add(1, std::memory_order_release); // (as ChangeState's)
			NoteVolatileClean();
			UpdateGpuProtection<false>();
		}
		ForEachRange(mask, std::forward<Func>(func));
	}

	// KYTY_VOLATILE=1 (MemoryTracker): memory the game rewrites every frame (its frame rings, KYTY_VOLATILE_RANGES). Its
	// pages are never write-protected for the CPU: an upload leaves them clean but writable (no syscall, no later
	// fault), and MemoryTracker::VolatileRedirty makes the pages the game may have written since CPU-dirty again, so
	// what the next draws read is uploaded again. GPU-dirty pages keep the usual rules.
	// Set under the region lock (at creation, or later for the game's rings); read under it.
	[[nodiscard]] bool IsVolatile() const noexcept { return m_volatile.load(std::memory_order_relaxed); }
	// (Region lock held.) Returns the pages that became dirty.
	size_t SetVolatile() {
		m_volatile.store(true, std::memory_order_relaxed);
		const size_t pages = MarkVolatileDirty();
		UpdateCpuProtection<false>(0, TRACKER_REGION_PAGES); // (a page still write-protected loses it, once)
		return pages;
	}
	// (Region lock held.) The pages of [start, end) neither GPU-dirty nor CPU-dirty become CPU-dirty (bits only).
	// Nothing changed (all dirty already): nothing is done, no epoch, no clock, no protection call. A page that
	// changed and is still write-protected (one the GPU gave back) is unprotected: the only syscall it can make.
	// Returns the pages that became dirty.
	size_t MarkVolatileDirty(size_t start = 0, size_t end = TRACKER_REGION_PAGES) {
		RegionBits clean = ~m_gpu_dirty;
		clean &= ~m_cpu_dirty;
		const RegionBits add(clean, start, end);
		if (add.None()) return 0;
		const size_t pages = add.Count();
		m_cpu_dirty ^= add; // (disjoint: an OR)
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		UpdateCpuProtection<false>(start, end);
		return pages;
	}
	// KYTY_VOLATILE: some page of this volatile region became clean (an upload, the GPU giving pages back) since the
	// last TakeVolatileClean. MemoryTracker re-dirties the allocator/ctx regions only when it is set, so an idle region
	// costs one load per call. Hint: readable without the lock; Take under it.
	[[nodiscard]] bool VolatileCleanHint() const noexcept { return m_volatile_clean.load(std::memory_order_acquire); }
	bool               TakeVolatileClean() noexcept { return m_volatile_clean.exchange(false, std::memory_order_acq_rel); }

	// (policy1) KYTY_DS_NEIGHBOR_SPLIT (loader/demonsSoulsWriteAudit.h; nothing below is ever set without it): a
	// "no-watch" page is never write-protected for the CPU. An upload leaves it clean and writable, so the game's
	// writes to it are not seen; it becomes CPU-dirty again only when a hook of the game announces a write
	// (MarkNoWatchDirty) or when it stops being no-watch (ClearNoWatch: a consumer asked for bytes of it no hook
	// covers). GPU-dirty pages keep the usual rules. All of it under the region lock; HasNoWatch is a hint.
	[[nodiscard]] bool HasNoWatch() const noexcept { return m_has_nowatch.load(std::memory_order_relaxed); }
	// (Region lock held, or TrackingSpinLock::ReadShared.) Some page of the range is no-watch.
	[[nodiscard]] bool AnyNoWatch(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return m_nowatch.AnyInRange(start, end);
	}
	// (Region lock held.) The page of `vaddr` becomes no-watch: CPU-dirty now (the write that faulted lands after
	// this returns and is in the page's next upload) and its write watcher released. False, nothing changed: the GPU
	// wrote the page, it is no-watch already, or a consumer refused it before (ClearNoWatch with `block`).
	// (anuncio1) `why` (may be null), when false: 1 GPU-dirty, 2 no-watch already (another watcher keeps the page
	// protected: an image, fresh memory), 3 refused before (ClearNoWatch with `block`). The no-watch policy's counters.
	[[nodiscard]] bool SetNoWatch(uint64_t vaddr, int* why = nullptr) {
		const auto [start, end] = GetPageRange(vaddr, 1);
		if (m_gpu_dirty.Get(start) || m_nowatch.Get(start) || m_nosplit.Get(start)) {
			if (why != nullptr) *why = m_gpu_dirty.Get(start) ? 1 : m_nowatch.Get(start) ? 2 : 3;
			return false;
		}
		m_nowatch.Set(start);
		m_has_nowatch.store(true, std::memory_order_relaxed);
		if (!m_cpu_dirty.Get(start)) {
			m_cpu_dirty.Set(start);
			m_cpu_epoch.fetch_add(1, std::memory_order_release);
			m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
			BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		}
		UpdateCpuProtection<false>(start, end);
		return true;
	}
	// (Region lock held.) The no-watch pages of the range neither GPU-dirty nor CPU-dirty become CPU-dirty (bits
	// only: they are writable already). Returns how many.
	size_t MarkNoWatchDirty(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		RegionBits clean        = m_nowatch & ~m_gpu_dirty;
		clean &= ~m_cpu_dirty;
		const RegionBits add(clean, start, end);
		if (add.None()) return 0;
		m_cpu_dirty ^= add; // (disjoint: an OR)
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		return add.Count();
	}
	// (Region lock held.) The no-watch pages of the range are watched again: CPU-dirty now (their bytes may have
	// changed unseen; the next upload makes them clean and write-protected as any page), a GPU-dirty one given the
	// write watcher it never got. `block`: they are not made no-watch again. Returns how many.
	size_t ClearNoWatch(uint64_t offset, uint64_t size, bool block) {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const RegionBits mask(m_nowatch, start, end);
		if (mask.None()) return 0;
		m_nowatch ^= mask;
		// (nativo2) Watched again: no longer native either.
		if (m_has_native.load(std::memory_order_relaxed)) {
			const RegionBits native = m_native & mask;
			m_native ^= native;
		}
		if (block) {
			for (const auto [first, last]: mask) {
				m_nosplit.SetRange(first, last);
				// (anuncio1, KYTY_RAIZ_ANNOUNCE) When it was refused: a frame page handed out again after it may be
				// made no-watch again (DemonsSoulsWriteAudit::UnblockWanted). One relaxed load when off.
				if (::Loader::DemonsSoulsWriteAudit::AnnounceOn())
					::Loader::DemonsSoulsWriteAudit::NoteBlockedRun(m_cpu_addr + first * TRACKER_PAGE_SIZE, m_cpu_addr + last * TRACKER_PAGE_SIZE);
			}
		}
		RegionBits add = mask & ~m_gpu_dirty;
		add &= ~m_cpu_dirty;
		if (!add.None()) {
			m_cpu_dirty ^= add;
			m_cpu_epoch.fetch_add(1, std::memory_order_release);
			m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
			BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		}
		UpdateCpuProtection<true>(start, end);
		return mask.Count();
	}
	// (anuncio1, KYTY_RAIZ_ANNOUNCE; region lock held) The page of `vaddr` is no longer refused (its refusal belonged to a
	// reservation of a frame allocator that has been handed out again since): SetNoWatch may take it. True: it was.
	bool UnblockNoWatch(uint64_t vaddr) {
		const auto [start, end] = GetPageRange(vaddr, 1);
		if (!m_nosplit.Get(start)) return false;
		m_nosplit.Unset(start);
		return true;
	}
	// (Region lock held.) The range is no tracked memory any more (MemoryTracker::UntrackMemory made it CPU-dirty and
	// unwatched): its pages forget the policy's state.
	void ForgetNoWatch(uint64_t offset, uint64_t size) {
		if (!HasNoWatch()) return;
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		m_nowatch.UnsetRange(start, end);
		m_nosplit.UnsetRange(start, end);
		m_native.UnsetRange(start, end); // (nativo2)
	}

	// ---- (nativo2) KYTY_DS_NATIVE_HEAP: native no-watch pages. Never set without it. --------------------------------
	// A no-watch page outside the hooks' rules (the game's general heaps: cpuheap, cpubkt) that GPU work reads is no
	// longer watched again for good (ClearNoWatch with `block`, then a write fault each time the game writes it). It
	// stays no-watch and becomes "native": the OS write watch of its page (native memory is MEM_WRITE_WATCH) is read and
	// reset before every submission (BufferCache::PolicyNativeSweep), and a written page is made CPU-dirty then - the
	// engine's own writes seen without a fault, as a port would know them.
	[[nodiscard]] bool HasNative() const noexcept { return m_has_native.load(std::memory_order_relaxed); }
	// (Region lock held, or ReadShared.) Some no-watch page of the range is not native yet.
	[[nodiscard]] bool AnyNoWatchNotNative(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const RegionBits candidates(m_nowatch & ~m_native, start, end);
		return !candidates.None();
	}
	// (Region lock held.) The no-watch pages of the range not native yet become native, and CPU-dirty now (their bytes
	// may have changed unseen since they became clean: the use that follows uploads them). Returns how many.
	size_t KeepNative(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const RegionBits add(m_nowatch & ~m_native, start, end);
		if (add.None()) return 0;
		m_native ^= add; // (disjoint: an OR)
		m_has_native.store(true, std::memory_order_relaxed);
		RegionBits dirty = add & ~m_gpu_dirty;
		dirty &= ~m_cpu_dirty;
		if (!dirty.None()) {
			m_cpu_dirty ^= dirty;
			m_cpu_epoch.fetch_add(1, std::memory_order_release);
			m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
			BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		}
		return add.Count();
	}
	// (Region lock held.) The runs of native pages, as page indices [first, last), in `out` (appended).
	template <typename F>
	void ForEachNativeRun(F&& f) const {
		for (const auto [first, last]: m_native) f(first, last);
	}

	// Own cache line: guest fault handling updates the epoch and bitmaps below.
	alignas(64) TrackingSpinLock lock;

	// ---- (PROT-MENOS) KYTY_PROT_MENOS: hot pages (local/prot-menos.h). Never set without it. ---------------------------
	// Hint without the lock: this region has hot pages and something to do (a hot page became clean, or a demotion).
	[[nodiscard]] bool HasHot() const noexcept { return m_has_hot.load(std::memory_order_relaxed); } // (PROT-MENOS4)
	[[nodiscard]] bool HotHint(uint64_t now_ms) const noexcept {
		if (!m_has_hot.load(std::memory_order_acquire)) return false;
		return m_hot_clean.load(std::memory_order_acquire) || now_ms - m_hot_since.load(std::memory_order_relaxed) >= ProtMenos::kDemoteMs;
	}
	// (Region lock held; before a submission reads guest memory.) The hot pages neither GPU-dirty nor CPU-dirty become
	// CPU-dirty (bits only: they are writable), so the submission's uses upload what the game wrote since. Every
	// kDemoteMs the hot pages not GPU-dirty go back to the normal watch (dirty and writable now: their next upload
	// write-protects them as usual). Returns the pages made dirty.
	size_t HotStep(uint64_t now_ms) {
		if (!m_has_hot.load(std::memory_order_relaxed)) return 0;
		const bool demote = now_ms - m_hot_since.load(std::memory_order_relaxed) >= ProtMenos::kDemoteMs;
		if (!m_hot_clean.exchange(false, std::memory_order_acq_rel) && !demote) return 0;
		RegionBits clean = m_hot & ~m_gpu_dirty;
		clean &= ~m_cpu_dirty;
		const bool watch = ProtMenos::WatchOn(); // (PROT-MENOS2)
		RegionBits add;
		if (!clean.None()) {
			if (watch) {
				add = clean & m_nowatch; // (PROT-MENOS2: a no-watch page's write watch is nativo2's: dirty as before)
				RegionBits runs = m_hot & ~m_gpu_dirty;
				runs &= ~m_nowatch;
				runs &= ~m_native; // (nativo2 reads those pages' write watch)
				// (PROT-MENOS4) Before: one GetWriteWatch+reset per run of hot pages, at EVERY submission, for every region
				// (HotHint stays true while a hot page is unwritten): thousands of system calls per frame (log 103232:
				// queue_run 23-25 ms with draw_us <70 and uploads <4 MB). Now one read WITHOUT reset over the span of the
				// clean hot pages (others' write watch untouched: wb-guard, nativo2), and the reset+read only over the
				// runs of hot pages that read says written (the ab46f0 objects: a few per frame).
				RegionBits want = clean & runs;
				if (!want.None()) {
					size_t lo = TRACKER_REGION_PAGES, hi = 0;
					for (const auto [first, last]: want) {
						if (first < lo) lo = first;
						if (last > hi) hi = last;
					}
					thread_local uint64_t peek[TRACKER_REGION_PAGES];
					bool                  failed = false;
					const uint64_t        count  = ::Libs::LibKernel::Memory::NativePeekWritten(
                        m_cpu_addr + lo * TRACKER_PAGE_SIZE, (hi - lo) * TRACKER_PAGE_SIZE, peek, TRACKER_REGION_PAGES, &failed);
					ProtMenos::g_peek_calls.fetch_add(1, std::memory_order_relaxed);
					if (failed || count > TRACKER_REGION_PAGES) {
						for (const auto [first, last]: runs) { // (as before: per run)
							if (clean.AnyInRange(first, last)) TakeHotWritten(first, last, clean, add);
						}
					} else if (count != 0) {
						RegionBits hit;
						for (uint64_t i = 0; i < count; ++i) {
							if (peek[i] < m_cpu_addr) continue;
							const auto page = static_cast<size_t>((peek[i] - m_cpu_addr) / TRACKER_PAGE_SIZE);
							if (page >= lo && page < hi && want.Get(page) && !hit.Get(page)) hit.Set(page);
						}
						for (const auto [first, last]: hit) TakeHotWritten(first, last, clean, add);
					}
				}
				{
					RegionBits w = add & m_hot;
					w &= ~m_hot_written;
					m_hot_written ^= w; // (PROT-MENOS4: written in this demotion window; an OR)
				}
				RegionBits left = clean;
				left &= ~add;
				if (!left.None()) {
					ProtMenos::g_ww_unchanged.fetch_add(left.Count(), std::memory_order_relaxed);
					m_hot_clean.store(true, std::memory_order_release); // (looked at again next submission)
				}
			} else {
				add = clean;
			}
		}
		size_t pages = 0;
		if (!add.None()) {
			pages = add.Count();
			m_cpu_dirty ^= add; // (disjoint: an OR)
			m_cpu_epoch.fetch_add(1, std::memory_order_release);
			m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
			BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		}
		if (demote) {
			RegionBits drop = m_hot & ~m_gpu_dirty; // (GPU-dirty ones stay hot: given back, they are writable)
			// (PROT-MENOS4) With the write watch, the hot pages written in the last window stay hot (the per-frame
			// ab46f0 fields): only the ones not written go back to the normal watch. Hot pages are only pages written.
			if (watch) {
				const RegionBits keep = drop & m_hot_written;
				if (!keep.None()) {
					ProtMenos::g_kept.fetch_add(keep.Count(), std::memory_order_relaxed);
					drop &= ~m_hot_written;
				}
			}
			m_hot_written.Clear();
			const size_t n = drop.Count();
			if (n != 0) {
				m_hot ^= drop;
				ProtMenos::Release(static_cast<int64_t>(n));
				ProtMenos::g_demoted.fetch_add(n, std::memory_order_relaxed);
				if (watch) pages += DemoteClean(drop); // (PROT-MENOS2)
			}
			m_hot_since.store(now_ms, std::memory_order_relaxed);
			if (m_hot.None()) m_has_hot.store(false, std::memory_order_release);
		}
		return pages;
	}
	// (PROT-MENOS6, lock NOT held; GPU thread) Before a query of this submission reads the region's state: its clean hot
	// pages' write watch read (HotStep) once per submission `gen`. Nothing to do (no hot page, none clean): two loads.
	void HotLazy(uint64_t gen) {
		if (!m_has_hot.load(std::memory_order_acquire)) return;
		if (!m_hot_clean.load(std::memory_order_acquire)) return; // (every hot page dirty already: uploaded as dirty)
		if (m_hot_gen.load(std::memory_order_relaxed) == gen) return;
		const auto       s0 = std::chrono::steady_clock::now();
		std::scoped_lock guard(lock);
		if (m_hot_gen.load(std::memory_order_relaxed) == gen) return;
		m_hot_gen.store(gen, std::memory_order_relaxed);
		const size_t pages = HotStep(ProtMenos::NowMs());
		if (pages != 0) ProtMenos::g_redirty_pages.fetch_add(pages, std::memory_order_relaxed);
		ProtMenos::g_lazy_calls.fetch_add(1, std::memory_order_relaxed);
		ProtMenos::g_lazy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                         std::chrono::steady_clock::now() - s0)
		                                                         .count()),
		                               std::memory_order_relaxed);
	}
	// (Region lock held.) The range is no tracked memory any more (MemoryTracker::UntrackMemory): its pages are not hot.
	void ForgetHot(uint64_t offset, uint64_t size) {
		if (!m_has_hot.load(std::memory_order_relaxed)) return;
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const RegionBits drop(m_hot, start, end);
		const size_t     n = drop.Count();
		if (n == 0) return;
		m_hot ^= drop;
		ProtMenos::Release(static_cast<int64_t>(n));
		if (m_hot.None()) m_has_hot.store(false, std::memory_order_release);
	}

private:
	// (PROT-MENOS, region lock held) [start, end) is becoming CPU-dirty. Its pages still write-protected (a fault, or an
	// invalidation) that were already seen so in this window become hot; the others are remembered for the window.
	void NoteCpuWrite(size_t start, size_t end) {
		ProtMenos::g_cpu_writes.fetch_add(1, std::memory_order_relaxed); // (PROT-MENOS4)
		RegionBits prot(~m_writable, start, end);
		prot &= ~m_nowatch;
		prot &= ~m_hot;
		prot &= ~m_gpu_dirty;
		if (prot.None()) return;
		const uint64_t now    = ProtMenos::NowMs();
		const uint64_t window = now / ProtMenos::kWindowMs;
		if (m_seen_window != window) {
			m_seen.Clear();
			m_seen_window = window;
		}
		const RegionBits promote = prot & m_seen;
		m_seen ^= prot & ~m_seen; // (an OR)
		if (promote.None()) return;
		const auto want = static_cast<int64_t>(promote.Count());
		if (!m_hot_listed) {
			m_hot_listed = ProtMenos::ListRegion(this);
			if (!m_hot_listed) return;
		}
		const int64_t got = ProtMenos::Reserve(want);
		if (got != want) { // (the cap: all or nothing for this change)
			if (got != 0) ProtMenos::Release(got);
			ProtMenos::g_refused.fetch_add(static_cast<uint64_t>(want), std::memory_order_relaxed);
			return;
		}
		if (!m_has_hot.load(std::memory_order_relaxed)) m_hot_since.store(now, std::memory_order_relaxed);
		m_hot_written ^= promote & ~m_hot_written; // (PROT-MENOS4: just written; an OR)
		m_hot ^= promote; // (disjoint: an OR)
		m_has_hot.store(true, std::memory_order_release);
		ProtMenos::g_promoted.fetch_add(static_cast<uint64_t>(want), std::memory_order_relaxed);
	}

	// (PROT-MENOS2, region lock held) The write watch of pages [first, last) read and reset: the ones of `clean` written
	// are added to `add`. A refused read: the whole run taken as written (as without the write watch).
	void TakeHotWritten(size_t first, size_t last, const RegionBits& clean, RegionBits& add) {
		thread_local uint64_t found[TRACKER_REGION_PAGES];
		bool                  failed = false;
		const uint64_t        count  = ::Libs::LibKernel::Memory::NativeTakeWritten(
            m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE, found, TRACKER_REGION_PAGES, &failed);
		ProtMenos::g_ww_calls.fetch_add(1, std::memory_order_relaxed);
		if (failed || count > TRACKER_REGION_PAGES) {
			ProtMenos::g_ww_failed.fetch_add(1, std::memory_order_relaxed);
			RegionBits part(clean, first, last);
			part &= ~add;
			add ^= part; // (an OR)
			return;
		}
		for (uint64_t i = 0; i < count; ++i) {
			if (found[i] < m_cpu_addr) continue;
			const auto page = static_cast<size_t>((found[i] - m_cpu_addr) / TRACKER_PAGE_SIZE);
			if (page >= first && page < last && clean.Get(page) && !add.Get(page)) add.Set(page);
		}
	}
	// (PROT-MENOS2, region lock held) The demoted pages `drop` (no longer hot) still clean: write-protected again (the
	// normal watch), then their write watch read: those written since the last read become dirty and writable again
	// (as a fault would make them). Returns the pages made dirty.
	size_t DemoteClean(const RegionBits& drop) {
		RegionBits still = drop & ~m_cpu_dirty;
		still &= ~m_nowatch;
		if (still.None()) return 0;
		for (const auto [first, last]: still) UpdateCpuProtection<true>(first, last);
		RegionBits late;
		for (const auto [first, last]: still) TakeHotWritten(first, last, still, late);
		if (late.None()) return 0;
		m_cpu_dirty ^= late; // (disjoint: an OR)
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		for (const auto [first, last]: late) UpdateCpuProtection<false>(first, last);
		const size_t n = late.Count();
		ProtMenos::g_ww_late.fetch_add(n, std::memory_order_relaxed);
		return n;
	}

	// The pages of [start, end) whose write watcher no longer matches their dirty state, in the change's direction:
	// clean pages without one get it (track), dirty pages with one lose it. (Dirty pages that keep their protection,
	// MarkCpuDirtyKeepProtection, are left alone by a change that only made other pages clean or dirty.)
	template <bool track>
	void UpdateCpuProtection(size_t start, size_t end) {
		// (Volatile: clean pages stay writable, see SetVolatile.)
		if constexpr (track) {
			if (m_volatile.load(std::memory_order_relaxed)) {
				m_volatile_clean.store(true, std::memory_order_release); // (pages were made clean: an upload)
				return;
			}
		}
		auto changed = track ? m_writable & ~m_cpu_dirty : m_cpu_dirty & ~m_writable;
		// (policy1: a no-watch page stays writable when it becomes clean. One load unless the region has some.)
		if (track && m_has_nowatch.load(std::memory_order_relaxed)) {
			const RegionBits kept(changed & m_nowatch, start, end);
			if (!kept.None()) {
				g_nowatch_kept_pages.fetch_add(kept.Count(), std::memory_order_relaxed);
				// (anuncio1, KYTY_RAIZ_ANNOUNCE + KYTY_RAIZ_VERIFY) Clean now, before the upload copies them
				// (MemoryTracker::ForEachUploadRange): their write watch starts over. A write found there when GPU work
				// is about to use them, with no announcement in between, is one the GPU would miss.
				if (::Loader::DemonsSoulsWriteAudit::AnnounceVerifyOn()) {
					for (const auto [first, last]: kept)
						::Loader::DemonsSoulsWriteAudit::NoteNoWatchCleanRun(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
				}
				// (policy6, KYTY_DS_WRITE_FRAME=2) The sampled ones are hashed: their upload copies them now.
				if (::Loader::DemonsSoulsWriteAudit::CheckStaleOn()) {
					for (const auto [first, last]: kept) {
						for (size_t page = first; page < last; ++page) {
							const uint64_t address = m_cpu_addr + page * TRACKER_PAGE_SIZE;
							if (::Loader::DemonsSoulsWriteAudit::SampledPage(address)) ::Loader::DemonsSoulsWriteAudit::NoteNoWatchClean(address);
						}
					}
				}
			}
			changed &= ~m_nowatch;
		}
		// (PROT-MENOS) A hot page stays writable when it becomes clean; HotStep makes it dirty again before the next
		// submission. The reprotect calls spared are counted by label (each also spares the next write fault's unprotect).
		if (track && m_has_hot.load(std::memory_order_relaxed)) {
			const RegionBits kept(changed & m_hot, start, end);
			if (!kept.None()) {
				uint64_t runs = 0;
				for (const auto run: kept) {
					(void)run;
					runs++;
				}
				FrameNativo::NoteAvoided(m_cpu_addr + start * TRACKER_PAGE_SIZE, runs);
				m_hot_clean.store(true, std::memory_order_release);
				changed &= ~m_hot;
			}
		}
		RegionBits mask(changed, start, end);
		if (mask.None()) {
			return;
		}
		m_writable ^= mask;
		LiveCounters::Add(track ? LiveCounters::Reprotect : LiveCounters::Unprotect);
		LiveCounters::Add(track ? LiveCounters::ReprotectPages : LiveCounters::UnprotectPages, mask.Count());
		if (track && LiveCounters::g_granules_on.load(std::memory_order_relaxed)) {
			for (const auto [start, end]: mask) {
				LiveCounters::AddGranule(m_cpu_addr + start * TRACKER_PAGE_SIZE, LiveCounters::GReprotectPages, end - start);
			}
		}
		m_page_manager.UpdatePageWatchersForRegion<track>(m_cpu_addr, mask);
	}

	template <bool track>
	void UpdateGpuProtection() {
		auto readable = ~m_gpu_dirty;
		auto mask     = readable ^ m_readable;
		m_readable    = readable;
		if (mask.None()) {
			return;
		}
		// (Diagnostic, KYTY_WRITEBACK_VERIFY: the pages becoming GPU-owned hashed, the ones given back forgotten.)
		if (WritebackVerify::Enabled()) {
			for (const auto [start, end]: mask) {
				if constexpr (track) {
					WritebackVerify::Owned(m_cpu_addr + start * TRACKER_PAGE_SIZE, end - start);
				} else {
					WritebackVerify::Released(m_cpu_addr + start * TRACKER_PAGE_SIZE, end - start);
				}
			}
		}
		// (CPUHEAP-SIN-VIGILAR, KYTY_CPUHEAP_SIN_VIGILAR=1 por defecto; local/nativo-cubierto.h) Las paginas de cpuheap que
		// la GPU toma no reciben vigilante de lectura (sin acceso): se quedan con su proteccion de escritura. Lo decidido al
		// tomarlas se guarda (m_noread), asi la devolucion no abre lo que no se cerro aunque la etiqueta cambie entre medias.
		if (::CpuheapSinVigilar::On() || ::DefaultTaggSinVigilar::On()) { // (+ DEFAULTTAGG-SIN-VIGILAR)
			if constexpr (track) {
				RegionBits skip;
				for (const auto [first, last]: mask) {
					for (size_t p = first; p < last;) {
						const uint64_t at   = m_cpu_addr + p * TRACKER_PAGE_SIZE;
						const size_t   gran = static_cast<size_t>(((at | 0xffffull) + 1 - at) / TRACKER_PAGE_SIZE); // (to the 64 KiB granule's end)
						const size_t   stop = (std::min)(static_cast<size_t>(last), p + gran);
						const auto     cls  = FrameNativo::Classify(at);
						if (cls == FrameNativo::CCpuheap && ::CpuheapSinVigilar::On()) skip.SetRange(p, stop);
						if (cls == FrameNativo::CDefaultTagg && ::DefaultTaggSinVigilar::On()) {
							skip.SetRange(p, stop);
							::DefaultTaggSinVigilar::g_taken.fetch_add(stop - p, std::memory_order_relaxed);
						}
						p = stop;
					}
				}
				// (A page taken now that is not cpuheap loses an old mark: its watcher is added and must be removed.)
				m_noread &= ~(mask & ~skip);
				if (!skip.None()) {
					const size_t n = skip.Count();
					m_noread ^= skip & ~m_noread; // (an OR)
					mask &= ~skip;
					::CpuheapSinVigilar::g_taken.fetch_add(n, std::memory_order_relaxed);
					::CpuheapSinVigilar::g_now.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
				}
			} else {
				const RegionBits skip = mask & m_noread;
				if (!skip.None()) {
					const size_t n = skip.Count();
					m_noread ^= skip;
					mask &= ~skip;
					::CpuheapSinVigilar::g_given.fetch_add(n, std::memory_order_relaxed);
					::CpuheapSinVigilar::g_now.fetch_sub(static_cast<int64_t>(n), std::memory_order_relaxed);
				}
			}
			if (mask.None()) return;
		}
		if constexpr (track) {
			// (crash-ba2b1f, KYTY_RAIZ_WB_GUARD) The pages the GPU takes: their write watch starts over before they lose
			// access, so a game write that reaches no fault from here on is seen at their write-back (BufferCache::
			// WriteBackGpuOwned). One relaxed load when off.
			if (::Libs::LibKernel::Memory::RaizWbGuardOn()) {
				for (const auto [start, end]: mask)
					::Libs::LibKernel::Memory::RaizWbGpuOwned(m_cpu_addr + start * TRACKER_PAGE_SIZE, (end - start) * TRACKER_PAGE_SIZE);
			}
			m_page_manager.UpdatePageWatchersForRegion<true, true>(m_cpu_addr, mask);
		} else {
			m_page_manager.UpdatePageWatchersForRegion<false, true>(m_cpu_addr, mask);
		}
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	template <typename Func>
	void ForEachRange(const RegionBits& bits, Func&& func) const {
		for (const auto [start, end]: bits) {
			func(m_cpu_addr + start * TRACKER_PAGE_SIZE, (end - start) * TRACKER_PAGE_SIZE);
		}
	}

	alignas(64) PageManager& m_page_manager;
	uint64_t              m_cpu_addr = 0;
	std::atomic<uint64_t> m_cpu_epoch {1};
	std::atomic<uint64_t> m_last_dirty {NextCpuDirtyClock()}; // (all pages start CPU-dirty)
	std::atomic<uint32_t> m_deferred_protects {0};
	RegionBits            m_cpu_dirty;
	RegionBits            m_gpu_dirty;
	RegionBits            m_writable;
	RegionBits            m_readable;
	RegionBits            m_noread; // (CPUHEAP-SIN-VIGILAR: GPU-taken cpuheap pages left without a read watcher)
	std::atomic<bool>     m_volatile {false}; // (SetVolatile)
	std::atomic<bool>     m_has_nowatch {false}; // (policy1: some bit of m_nowatch was set at some time)
	RegionBits            m_nowatch;             // (policy1: SetNoWatch; all clear without the policy)
	RegionBits            m_nosplit;             // (policy1: ClearNoWatch with `block`)
	std::atomic<bool>     m_has_native {false};  // (nativo2: some bit of m_native was set at some time)
	RegionBits            m_native;              // (nativo2: KeepNative; a subset of m_nowatch)
	std::atomic<bool>     m_volatile_clean {false}; // (VolatileCleanHint)
	// (PROT-MENOS) Hot pages: all clear without KYTY_PROT_MENOS.
	std::atomic<bool>     m_has_hot {false};   // (some bit of m_hot is set)
	std::atomic<bool>     m_hot_clean {false}; // (a hot page became clean since the last HotStep)
	std::atomic<uint64_t> m_hot_since {0};     // (ms: first promotion, then the last demotion pass)
	bool                  m_hot_listed  = false; // (in ProtMenos::g_regions)
	std::atomic<uint64_t> m_hot_gen {0}; // (PROT-MENOS6: the submission HotLazy last read this region for)
	uint64_t              m_seen_window = 0;
	RegionBits            m_hot;  // (never write-protected for the CPU; dirty again before each submission)
	RegionBits            m_seen; // (write-protected pages written once in the window m_seen_window)
	RegionBits            m_hot_written; // (PROT-MENOS4: hot pages found written since the last demotion pass)

	void NoteVolatileClean() noexcept {
		if (m_volatile.load(std::memory_order_relaxed)) m_volatile_clean.store(true, std::memory_order_release);
		// (PROT-MENOS) A hot page given back by the GPU is clean and writable: dirty again before the next submission.
		if (m_has_hot.load(std::memory_order_relaxed)) m_hot_clean.store(true, std::memory_order_release);
	}
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
