#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/perfStats.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

// The lock of the texture cache and of each memory-tracker region. Every CPU write fault takes the
// texture cache's (TextureCache::InvalidateMemory), and the GPU thread holds it for each image it
// looks up: in GT7 the game's six job threads spent near half their time in fault handlers, all
// about as long as one another (round 64), as if queued behind the GPU thread. A waiter used to
// spin on test-and-set, writing the lock's cache line in a loop under the holder and taking the
// processor from the holder when they share a core. It now reads the lock until it is free,
// pausing, and yields the processor after a while (KYTY_LOCKS_SPIN=1 or KYTY_BISECT's locks:spin
// for the old way); the waits are counted on the "Perf locks:" line.
class TrackingSpinLock final {
public:
	explicit TrackingSpinLock(
	    Common::PerfStats::Lock kind = Common::PerfStats::Lock::MemoryTracker) noexcept
	    : m_kind(kind) {}
	KYTY_CLASS_NO_COPY(TrackingSpinLock);

	void lock() noexcept {
		const auto thread = CurrentThread();
		if (m_lock.exchange(true, std::memory_order_acquire)) {
			Wait(thread);
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.store(false, std::memory_order_release);
	}

private:
	// Held by another thread (by this one: a recursive lock ends the emulator).
	void Wait(uint32_t thread) noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
		const bool timed = Common::PerfStats::Active();
		const auto start =
		    timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		if (Common::PerfStats::SpinLocks()) {
			while (m_lock.exchange(true, std::memory_order_acquire)) {
				EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
				std::atomic_signal_fence(std::memory_order_seq_cst);
			}
		} else {
			uint32_t spins = 0;
			// GT7: the GPU thread is the bottleneck and has a core of its own; a yield there can
			// hand its core to a compile thread for a whole time slice (waits of 16 ms and more
			// on the texture cache), so it keeps spinning (KYTY_GPU_LOCK_SPIN=0: yields as before).
			const uint32_t spin_limit =
			    Common::PerfStats::IsGpuThread() && Common::PerfStats::GpuLockSpin() ? ~0u
			                                                                         : 256u;
			do {
				while (m_lock.load(std::memory_order_relaxed)) {
					if (spins < spin_limit) {
						spins++;
						Pause();
					} else {
						std::this_thread::yield();
					}
				}
			} while (m_lock.exchange(true, std::memory_order_acquire));
		}
		if (timed) {
			Common::PerfStats::AddLockWait(
			    m_kind, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                      std::chrono::steady_clock::now() - start)
			                                      .count()));
		}
	}

	static void Pause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
	}

	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
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

	std::atomic<bool>       m_lock {false};
	std::atomic_uint32_t    m_owner {0};
	Common::PerfStats::Lock m_kind;
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Bumped whenever pages become CPU-dirty anywhere (a write fault, an untrack, an
// invalidation). A consumer that must see every CPU write, such as the pre-dispatch sync of
// all buffers for global-memory shaders, can skip its walk while this has not moved.
inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// Bumped by the GPU thread when it starts a guest submission (its first slice).
inline std::atomic<uint64_t> g_guest_submission_seq {1};

// Bumped by the GPU thread when a WAIT_REG_MEM or a rewind wait passes (round 90). What the CPU
// wrote before it let the GPU go on is visible to the commands after the wait, as on the console,
// so the walk for memory-reading shaders runs again after one (RenderContext::PrepareBda).
inline std::atomic<uint64_t> g_guest_wait_seq {0};

// Round 90: set on the GPU thread (GuestGpu::ThreadRun). The pages it makes CPU-dirty (its own
// stores to guest memory: labels, WRITE_DATA, invalidations) are not the game's writes.
inline thread_local bool t_guest_gpu_thread = false;

// Round 90: bumped with g_cpu_dirty_epoch when pages become CPU-dirty on any other thread: the
// game wrote them (a write fault, or a block released from write tracking). A wait the GPU
// released itself (one queue waiting for another's label) then calls for no new walk.
inline std::atomic<uint64_t> g_cpu_write_epoch {1};

// gt7 (KYTY_BUFCACHE_V2): bumped whenever GPU-dirty pages become clean (a download, an unmark),
// and when a MemoryTracker is created or destroyed. A range found all GPU-dirty stays so until the
// next bump, so a written buffer sync that sees the same range and the same value skips its walk
// (BufferCache::SynchronizeBuffer). Only the GPU thread changes GPU-dirty bits.
inline std::atomic<uint64_t> g_gpu_clear_epoch {1};

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
		PublishDirtyBits();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	// Lock-free "any page in this region may be dirty" summaries. A binding that spans a
	// whole heap covers hundreds of regions; without these every draw locks and walks each
	// one. False is exact (set only after a walk saw no bits); true is conservative.
	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}
	// IsModified without the lock (mainline #1284), for the GPU thread, which owns the GPU-dirty
	// bits and clears CPU-dirty bits. Other threads only set CPU-dirty bits, under the lock and
	// before the page becomes writable, so a clean answer cannot miss a write already made.
	// gt7-caches-1112 (KYTY_SYNC_SKIP): whether every page of the range is GPU-dirty, from the
	// published bits without the lock. Exact on the GPU thread, which alone changes GPU-dirty bits;
	// a GPU-dirty page can be neither CPU-dirty (ChangeState refuses it) nor hot (MarkHot).
	[[nodiscard]] bool IsAllGpuDirtyUnlocked(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		if (end <= start) {
			return false;
		}
		const auto first = start / 64;
		const auto last  = (end - 1) / 64;
		for (auto word = first; word <= last; word++) {
			auto mask = ~uint64_t {0};
			if (word == first) {
				mask &= ~uint64_t {0} << (start % 64);
			}
			if (word == last && end % 64 != 0) {
				mask &= (uint64_t {1} << (end % 64)) - 1;
			}
			if ((m_gpu_words[word].load(std::memory_order_acquire) & mask) != mask) {
				return false;
			}
		}
		return true;
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModifiedUnlocked(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const auto& words       = source == DirtySource::Cpu ? m_cpu_words : m_gpu_words;
		const auto  first       = start / 64;
		const auto  last        = (end - 1) / 64;
		for (auto word = first; word <= last; word++) {
			auto bits = words[word].load(std::memory_order_acquire);
			if (word == first) {
				bits &= ~uint64_t {0} << (start % 64);
			}
			if (word == last && end % 64 != 0) {
				bits &= (uint64_t {1} << (end % 64)) - 1;
			}
			if (bits != 0) {
				return true;
			}
		}
		return false;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			PublishDirtyBits();
			StoreSummary<source>(true);
			if constexpr (source == DirtySource::Cpu) {
				NoteBdaPending();
				g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
				if (!t_guest_gpu_thread) {
					g_cpu_write_epoch.fetch_add(1, std::memory_order_release);
				}
			}
		} else {
			bits.UnsetRange(start, end);
			PublishDirtyBits();
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (source == DirtySource::Gpu) {
				g_gpu_clear_epoch.fetch_add(1, std::memory_order_release);
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateCpuProtection();
		} else {
			if constexpr (enable) {
				// Written by the GPU: no longer kept writable for the CPU (hot pages, below).
				if (m_hot_count != 0 && RegionBits(m_hot, start, end).Any()) {
					m_hot.UnsetRange(start, end);
					m_hot_count = m_hot.Count();
					UpdateCpuProtection();
				}
			}
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			PublishDirtyBits();
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (source == DirtySource::Gpu) {
				g_gpu_clear_epoch.fetch_add(1, std::memory_order_release);
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateCpuProtection();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// ForEachModifiedRange<DirtySource::Cpu, true> without its protection change (round 80): the
	// CPU-dirty pages of the range become clean, and func gets their runs, but they stay writable
	// until CommitCpuProtection, which the caller makes before it releases the lock (and before
	// it reads the pages' bytes). Several ranges then change the protection in one call.
	template <typename Func>
	void TakeCpuModifiedRanges(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto& bits              = GetBits<DirtySource::Cpu>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		bits.UnsetRange(start, end);
		PublishDirtyBits();
		if (bits.None()) {
			StoreSummary<DirtySource::Cpu>(false);
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}
	void CommitCpuProtection() { UpdateCpuProtection(); }

	// The tracker mirrors the CPU summary in one bit per region (MemoryTracker::IsRangeCpuCleanHint).
	// Set before the region is published.
	void SetCpuSummaryBit(std::atomic<uint64_t>* word, uint64_t bit) noexcept {
		m_cpu_summary_word = word;
		m_cpu_summary_bit  = bit;
	}

	// The tracker's bits of the regions the next BDA walk visits (MemoryTracker::TakeBdaPending):
	// one in a word of regions and one in the word above it. Set before the region is published.
	void SetBdaPendingBits(std::atomic<uint64_t>* word, uint64_t bit,
	                       std::atomic<uint64_t>* top_word, uint64_t top_bit) noexcept {
		m_bda_word     = word;
		m_bda_bit      = bit;
		m_bda_top_word = top_word;
		m_bda_top_bit  = top_bit;
	}

	// Pages of this region became CPU-dirty: the next BDA walk visits it. The region's bit before
	// the word's, so a walk that sees the word's bit finds the region's.
	void NoteBdaPending() noexcept {
		if (m_bda_word != nullptr) {
			m_bda_word->fetch_or(m_bda_bit, std::memory_order_release);
			m_bda_top_word->fetch_or(m_bda_top_bit, std::memory_order_release);
		}
	}

	// Hot pages (KYTY_BISECT's "hotpages"): pages the CPU writes over and over, kept writable
	// instead of write-protected again after each upload, so that the CPU's next write does not
	// fault. Without a fault the tracker cannot see that write, so they are marked CPU-dirty again
	// wholesale (RedirtyHot) at the points where the GPU thread may next read what the CPU wrote
	// since: a new slice of a submission, a wait on memory. A page the GPU writes stops being hot.
	// All with the lock held.
	//
	// The pages of [vaddr, vaddr + size) that hold no GPU-written bytes become hot; how many did.
	size_t MarkHot(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits pages;
		pages.SetRange(start, end);
		pages &= ~m_hot;
		pages &= ~m_gpu_dirty;
		if (pages.None()) {
			return 0;
		}
		m_hot |= pages;
		m_hot_count = m_hot.Count();
		UpdateCpuProtection();
		return pages.Count();
	}
	// The hot pages become CPU-dirty again (they are writable already); how many.
	size_t RedirtyHot() {
		if (m_hot_count == 0) {
			return 0;
		}
		return DirtyHot(m_hot);
	}
	// The hot pages of [vaddr, vaddr + size) become CPU-dirty: something wrote them without a
	// fault (the GPU thread's own stores to guest memory); how many.
	size_t DirtyHotIn(uint64_t vaddr, uint64_t size) {
		if (m_hot_count == 0) {
			return 0;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		return DirtyHot(RegionBits(m_hot, start, end));
	}
	// The hot pages of [vaddr, vaddr + size) stop being hot, as CPU-dirty pages: the CPU may have
	// written them since their last upload, unseen, so their next upload takes those writes (and
	// write-protects them, so the writes after it fault again). Before a sync that hands the
	// range to the GPU's writes (MemoryTracker::ForEachUploadRange).
	void DropHotIn(uint64_t vaddr, uint64_t size) {
		if (m_hot_count == 0) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits pages(m_hot, start, end);
		if (pages.None()) {
			return;
		}
		(void)DirtyHot(pages);
		m_hot.UnsetRange(start, end);
		m_hot_count = m_hot.Count();
		UpdateCpuProtection();
	}
	// No page is hot any more. They become CPU-dirty first (unseen writes since their last
	// upload are uploaded with them), then write-protected at that upload as any page.
	void ClearHot() {
		if (m_hot_count == 0) {
			return;
		}
		(void)DirtyHot(m_hot);
		m_hot.Clear();
		m_hot_count = 0;
		UpdateCpuProtection();
	}
	[[nodiscard]] size_t HotCount() const noexcept { return m_hot_count; }

	// Memory mapped again (RenderContext::MapMemory, KYTY_MAP_REDIRTY): the pages of
	// [vaddr, vaddr + size) that are neither CPU-dirty nor GPU-written become CPU-dirty, since the
	// mapping brought bytes no write fault announced; and the region is pending for the next BDA
	// walk when any page of the range is CPU-dirty, since a walk while the range was unmapped can
	// take the region's pending bit without synchronizing it. How many pages became CPU-dirty.
	size_t DirtyCleanIn(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits pages;
		pages.SetRange(start, end);
		auto dirty = pages & ~m_gpu_dirty;
		dirty &= ~m_cpu_dirty;
		const auto count = dirty.Count();
		if (count != 0) {
			m_cpu_dirty |= dirty;
			PublishDirtyBits();
			StoreSummary<DirtySource::Cpu>(true);
			UpdateCpuProtection();
		}
		if (count != 0 || RegionBits(m_cpu_dirty, start, end).Any()) {
			NoteBdaPending();
			g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
			if (!t_guest_gpu_thread) {
				g_cpu_write_epoch.fetch_add(1, std::memory_order_release);
			}
		}
		return count;
	}

	// Diagnostics: the CPU-dirty, GPU-written and hot pages of [vaddr, vaddr + size).
	struct PageCounts {
		size_t cpu = 0;
		size_t gpu = 0;
		size_t hot = 0;
	};
	[[nodiscard]] PageCounts CountPages(uint64_t vaddr, uint64_t size) const {
		const auto [start, end] = GetPageRange(vaddr, size);
		return {RegionBits(m_cpu_dirty, start, end).Count(),
		        RegionBits(m_gpu_dirty, start, end).Count(), RegionBits(m_hot, start, end).Count()};
	}

	TrackingSpinLock lock;
	// Whether the tracker lists this region as holding hot pages (MemoryTracker's hot mutex).
	bool in_hot_list = false;

private:
	// Every summary change goes through here; the CPU one is mirrored in the tracker's bitmap at
	// the same point, before any page protection changes.
	template <DirtySource source>
	void StoreSummary(bool value) {
		GetSummary<source>().store(value, std::memory_order_release);
		if constexpr (source == DirtySource::Cpu) {
			if (m_cpu_summary_word != nullptr) {
				if (value) {
					m_cpu_summary_word->fetch_or(m_cpu_summary_bit, std::memory_order_acq_rel);
				} else {
					m_cpu_summary_word->fetch_and(~m_cpu_summary_bit, std::memory_order_acq_rel);
				}
			}
		}
	}

	// After every change of m_cpu_dirty or m_gpu_dirty, with the lock held.
	void PublishDirtyBits() {
		for (size_t word = 0; word < RegionBits::kWordCount; word++) {
			m_cpu_words[word].store(m_cpu_dirty.Word(word), std::memory_order_release);
			m_gpu_words[word].store(m_gpu_dirty.Word(word), std::memory_order_release);
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	// `pages` (hot ones) that are neither GPU-written nor CPU-dirty become CPU-dirty; how many.
	size_t DirtyHot(const RegionBits& pages) {
		auto dirty = pages & ~m_gpu_dirty;
		dirty &= ~m_cpu_dirty;
		if (dirty.None()) {
			return 0;
		}
		m_cpu_dirty |= dirty;
		PublishDirtyBits();
		StoreSummary<DirtySource::Cpu>(true);
		NoteBdaPending();
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		return dirty.Count();
	}

	// The CPU write protection: a page is writable while CPU-dirty or hot. Pages can go both
	// ways in one call (hot pages cleared while others become dirty).
	void UpdateCpuProtection() {
		auto writable = m_cpu_dirty;
		if (m_hot_count != 0) {
			writable |= m_hot;
		}
		auto now_writable  = writable & ~m_writable;
		auto now_protected = m_writable & ~writable;
		if (now_writable.None() && now_protected.None()) {
			return;
		}
		m_writable = writable;
		if (now_protected.Any()) {
			m_page_manager.UpdatePageWatchersForRegion<true, false>(m_cpu_addr, now_protected);
		}
		if (now_writable.Any()) {
			m_page_manager.UpdatePageWatchersForRegion<false, false>(m_cpu_addr, now_writable);
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

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
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

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	// Copies of the dirty bits for IsModifiedUnlocked, stored under the lock.
	std::array<std::atomic<uint64_t>, RegionBits::kWordCount> m_cpu_words {};
	std::array<std::atomic<uint64_t>, RegionBits::kWordCount> m_gpu_words {};
	// The constructor fills m_cpu_dirty, so the CPU summary starts true.
	std::atomic<bool> m_cpu_maybe_dirty {true};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	std::atomic<uint64_t>* m_cpu_summary_word = nullptr;
	uint64_t               m_cpu_summary_bit  = 0;
	std::atomic<uint64_t>* m_bda_word         = nullptr;
	uint64_t               m_bda_bit          = 0;
	std::atomic<uint64_t>* m_bda_top_word     = nullptr;
	uint64_t               m_bda_top_bit      = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
	RegionBits             m_hot;
	size_t                 m_hot_count = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
