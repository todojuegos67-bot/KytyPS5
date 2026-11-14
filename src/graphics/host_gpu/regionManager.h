#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "live-counters.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/uploadHeap.h"
#include "writeback-verify.h"

#include <atomic>
#include <bit>
#include <immintrin.h>
#include <memory>
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
	// See g_cpu_dirty_clock.
	[[nodiscard]] uint64_t LastDirtyClock() const noexcept { return m_last_dirty.load(std::memory_order_acquire); }
	[[nodiscard]] uint64_t CpuModificationEpoch() const {
		// Pages marked clean whose write protection is still queued can change unseen:
		// nothing is provable until the worker applied it (and moved the epoch).
		if (m_deferred_protects.load(std::memory_order_acquire) != 0) return 0;
		// Kept pages (UploadHeap) are written without a fault: nothing about them is provable.
		if (m_has_kept.load(std::memory_order_acquire)) return 0;
		return m_cpu_epoch.load(std::memory_order_acquire);
	}
	// KYTY_ASYNC_REPROTECT: a write protection of pages in this region is queued.
	void BeginDeferredProtection() noexcept { m_deferred_protects.fetch_add(1, std::memory_order_acq_rel); }
	[[nodiscard]] bool DeferredProtectionPending() const noexcept {
		return m_deferred_protects.load(std::memory_order_acquire) != 0;
	}
	// UploadHeap: pages kept CPU-dirty and writable after a copy (written without a fault).
	[[nodiscard]] bool HasKeptPages() const noexcept { return m_has_kept.load(std::memory_order_acquire); }
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
				// Kept pages (UploadHeap) are CPU-dirty only because they are never made clean: their bytes went to
				// the GPU at their last use, as a clean page's had. The GPU's write takes them, as it takes clean
				// pages: clean and write-watched again. Any other CPU-dirty page is the conflict it always was.
				RegionBits conflict(m_cpu_dirty, start, end);
				if ((conflict & ~m_kept).Any()) {
					EXIT("GPU dirty state conflicts with CPU dirty state\n");
				}
				m_cpu_dirty ^= conflict;
				m_kept ^= conflict;
				m_kept_done &= ~conflict;
				m_has_kept.store(m_kept.Any(), std::memory_order_release);
				UploadHeap::g_gpu_taken_pages.fetch_add(conflict.Count(), std::memory_order_relaxed);
				UpdateCpuProtection<true>(start, end);
			}
		}
		if constexpr ((source == DirtySource::Cpu && enable) || source == DirtySource::Gpu) {
			// (A CPU write the trackers saw, or a GPU change: a kept page's copy of this epoch is stale. m_kept_done is
			// a subset of m_kept, empty whenever m_has_kept is false (both change under the lock): off, one branch.)
			if (m_has_kept.load(std::memory_order_relaxed)) m_kept_done.UnsetRange(start, end);
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
			if constexpr (!enable) m_cpu_epoch.fetch_add(1, std::memory_order_release);
			UpdateGpuProtection<enable>();
		}
	}

	// The emulator wrote CPU-dirty pages of the region (their state stays): what a proof by the epoch holds of their
	// bytes holds no longer (TableXpr::RingCopy's copies of guest-written ranges). The epoch moves as well when pages
	// become CPU-dirty and when GPU-written pages are given back (their bytes the GPU's): with the guest's writes to
	// dirty pages it covers every change of the region's bytes the CPU can see but those.
	void NoteHostWrite() noexcept {
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		// (Not under the lock in every caller: a stamp change makes the next kept upload forget its copies.)
		m_kept_stamp_invalid.store(true, std::memory_order_release);
	}

	// CPU-dirty with the pages' write protection kept: the emulator wrote them through the backing view (no fault
	// saw it). A page still write-watched stays so (UpdateCpuProtection moves only pages of a change's range, in its
	// direction): its next upload finds it protected already, and a guest write before that faults once and drops
	// the watcher as for any dirty page (ChangeState). Returns false (nothing changed) when the GPU wrote a page.
	[[nodiscard]] bool MarkCpuDirtyKeepProtection(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (m_gpu_dirty.AnyInRange(start, end)) return false;
		m_cpu_dirty.SetRange(start, end);
		if (m_has_kept.load(std::memory_order_relaxed)) m_kept_done.UnsetRange(start, end);
		m_cpu_epoch.fetch_add(1, std::memory_order_release);
		m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		return true;
	}

	// The upload of the CPU-dirty pages of [vaddr, vaddr + size) (MemoryTracker::ForEachUploadRange): func gets the
	// ranges to copy (contiguous pages as one range), and the pages become clean and write-watched, as
	// ForEachModifiedRange<Cpu, true> makes them. With `keep` (a read-only use on the GPU thread), a page UploadHeap
	// marks that is hot - written in consecutive frames (it faulted and was copied in an earlier frame, at most
	// UploadHeap::HotFrames before) - stays CPU-dirty and writable instead (kept): no protection change, no fault at the
	// guest's next write; it is copied at its uses, at most once per publish epoch (uploadHeap.h). A kept page used
	// again sooner than its write period (a later publish epoch of the same frame - read by several submissions - or a
	// later frame of the period: tracking would copy it once) is made clean and write-watched after that copy and not kept again for UploadHeap::CoolFrames frames. So a kept
	// page costs one copy a write period where tracking costs a fault, two protection changes and the same copy.
	template <typename Func>
	void ForEachCpuUploadRange(uint64_t vaddr, uint64_t size, bool keep, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (!m_cpu_dirty.AnyInRange(start, end)) return;
		RegionBits     mask(m_cpu_dirty, start, end);
		// (Only on the GPU thread translating guest work: the publish epoch and frame are its own.)
		const bool     gpu_thread = PageManager::ReadProtectionDeferred();
		const uint64_t granules   = keep ? UploadHeap::RegionGranules(m_cpu_addr) : 0;
		if (granules == 0 && !m_has_kept.load(std::memory_order_relaxed)) {
			m_cpu_dirty.UnsetRange(start, end);
			UpdateCpuProtection<true>(start, end);
			ForEachRange(mask, std::forward<Func>(func));
			return;
		}
		if (granules != m_pin_granules) {
			m_pin_granules = granules;
			m_pin_mask.Clear();
			constexpr size_t PagesPerGranule = (size_t {1} << UploadHeap::GranuleBits) / TRACKER_PAGE_SIZE;
			for (uint64_t bits = granules; bits != 0; bits &= bits - 1) {
				const auto g = static_cast<size_t>(std::countr_zero(bits));
				m_pin_mask.SetRange(g * PagesPerGranule, (g + 1) * PagesPerGranule);
			}
		}
		const uint64_t publish = UploadHeap::g_publish.load(std::memory_order_acquire);
		const uint32_t frame   = static_cast<uint32_t>(UploadHeap::g_frame.load(std::memory_order_acquire));
		const bool     host_wrote = m_kept_stamp_invalid.exchange(false, std::memory_order_acq_rel);
		if (!gpu_thread || publish != m_kept_stamp || host_wrote) {
			m_kept_done.Clear();
			m_kept_stamp = gpu_thread ? publish : 0;
		}
		if (granules != 0 && m_page_info == nullptr) m_page_info = std::make_unique<UploadHeap::PageInfo[]>(TRACKER_REGION_PAGES);
		// clean: dirty pages made clean now (write-watched); copy: pages copied now.
		RegionBits clean = mask & ~m_pin_mask;
		RegionBits copy  = clean;
		// A kept page that is not marked any more for this use (a written use, another thread): as before, clean.
		if (granules == 0) {
			clean = mask;
			copy  = mask & ~(m_kept & m_kept_done);
		}
		uint64_t kept_copies = 0, deduped = 0, promoted = 0, demoted = 0;
		if (granules != 0) {
			const RegionBits candidates = mask & m_pin_mask;
			for (const auto [first, last]: candidates) {
				for (size_t page = first; page < last; page++) {
					auto& info = m_page_info[page];
					if (m_kept.Get(page)) {
						if (!gpu_thread) { // (another thread's use: copied, kept as it is)
							copy.Set(page);
							kept_copies++;
							continue;
						}
						if (m_kept_done.Get(page)) { // (copied in this publish epoch)
							deduped++;
							continue;
						}
						copy.Set(page);
						m_kept_done.Set(page);
						kept_copies++;
						if (frame - info.kept_copy_frame < info.period) {
							// Read again sooner than the guest rewrites it (in a later publish epoch of the frame, or in
							// a later frame of its write period: tracking would not copy it again): tracked from here on.
							clean.Set(page);
							m_kept.Unset(page);
							m_kept_done.Unset(page);
							info.no_keep_before = frame + UploadHeap::CoolFrames;
							demoted++;
						} else if (frame - info.kept_since >= UploadHeap::KeepFrames) {
							// Kept long enough: tracked again (written next frame, it faults once and is kept again).
							clean.Set(page);
							m_kept.Unset(page);
							m_kept_done.Unset(page);
							info.dirty_frame = frame + 1;
							demoted++;
						} else {
							info.kept_copy_frame = frame;
						}
						continue;
					}
					copy.Set(page);
					if (!gpu_thread) {
						// (Another thread's use: kept as the upload heap was before the hot rule, so its copy, which may
						// be another buffer's, leaves the page CPU-dirty; the GPU thread's next use decides.)
						m_kept.Set(page);
						kept_copies++;
						continue;
					}
					const bool hot = info.dirty_frame != 0 && info.dirty_frame != frame + 1 &&
					                 frame + 1 - info.dirty_frame <= UploadHeap::HotFrames &&
					                 static_cast<int32_t>(frame - info.no_keep_before) >= 0;
					if (hot) {
						m_kept.Set(page);
						m_kept_done.Set(page);
						info.kept_copy_frame = frame;
						info.kept_since      = frame;
						info.period          = frame + 1 - info.dirty_frame; // (1..HotFrames: its write period)
						promoted++;
					} else {
						clean.Set(page);
						if (gpu_thread) info.dirty_frame = frame + 1; // (0 = never)
					}
				}
			}
		}
		m_cpu_dirty &= ~clean;
		m_kept &= ~clean;
		m_kept_done &= ~clean;
		const bool has_kept = m_kept.Any();
		m_has_kept.store(has_kept, std::memory_order_release);
		if (kept_copies + promoted != 0) {
			// Kept pages copied now and dirty since: an upload of them later in this command buffer goes in order
			// (UploadPrologue); what proved the region before proves nothing now.
			m_cpu_epoch.fetch_add(1, std::memory_order_release);
			m_last_dirty.store(NextCpuDirtyClock(), std::memory_order_release);
		}
		if (granules != 0) {
			UploadHeap::g_kept_copies.fetch_add(kept_copies, std::memory_order_relaxed);
			UploadHeap::g_deduped_pages.fetch_add(deduped, std::memory_order_relaxed);
			UploadHeap::g_promoted_pages.fetch_add(promoted, std::memory_order_relaxed);
			UploadHeap::g_demoted_pages.fetch_add(demoted, std::memory_order_relaxed);
			UploadHeap::g_kept_pages.fetch_add(kept_copies + promoted, std::memory_order_relaxed);
		}
		if (clean.Any()) UpdateCpuProtection<true>(start, end);
		if (copy.Any()) ForEachRange(copy, std::forward<Func>(func));
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
			UpdateGpuProtection<false>();
		}
		ForEachRange(mask, std::forward<Func>(func));
	}

	// Own cache line: guest fault handling updates the epoch and bitmaps below.
	alignas(64) TrackingSpinLock lock;

private:
	// The pages of [start, end) whose write watcher no longer matches their dirty state, in the change's direction:
	// clean pages without one get it (track), dirty pages with one lose it. (Dirty pages that keep their protection,
	// MarkCpuDirtyKeepProtection, are left alone by a change that only made other pages clean or dirty.)
	template <bool track>
	void UpdateCpuProtection(size_t start, size_t end) {
		const auto changed = track ? m_writable & ~m_cpu_dirty : m_cpu_dirty & ~m_writable;
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
		if constexpr (track) {
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
	// UploadHeap: kept pages (CPU-dirty, writable, copied at their last use), the kept pages copied in the epoch
	// m_kept_stamp, the region's marked granules as pages (cached for m_pin_granules).
	std::atomic<bool>     m_has_kept {false};
	std::atomic<bool>     m_kept_stamp_invalid {false};
	uint64_t              m_kept_stamp    = 0;
	uint64_t              m_pin_granules  = 0;
	RegionBits            m_kept;
	RegionBits            m_kept_done;
	RegionBits            m_pin_mask;
	std::unique_ptr<UploadHeap::PageInfo[]> m_page_info; // (marked regions only)
	RegionBits            m_cpu_dirty;
	RegionBits            m_gpu_dirty;
	RegionBits            m_writable;
	RegionBits            m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
