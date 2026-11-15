#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <array>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			// Wait on reads, so the holder's cache line is not written until the lock is free.
			while (m_lock.test(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(_M_X64)
				_mm_pause();
#elif defined(_M_ARM64)
				__yield();
#elif defined(__aarch64__)
				asm volatile("yield");
#endif
			}
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#if defined(_M_X64)
		// GetCurrentThreadId's own read of the thread block (TEB ClientId.UniqueThread), without the call into kernel32:
		// every lock and unlock of every tracker walk asks, about a millisecond a frame at the Trask outpost.
		return __readgsdword(0x48);
#else
		return GetCurrentThreadId();
#endif
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0 && !KYTY_UPSTREAM_PATH("25fb20e53")) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_cpu_written.Fill();
		m_writable.Fill();
		m_readable.Fill();
		NoteCpuDirty(m_cpu_addr, TRACKER_REGION_SIZE);
	}

	// How often a page became CPU-dirty, counted per 256 MiB of address space (aliased past
	// 1 TiB). A range whose counts add up to what they did when it was last found clean has no
	// CPU-dirty page: a caller that binds the same large range over and over need not walk it.
	// The count goes up after the page is marked, so a walk that misses the mark sees the count.
	[[nodiscard]] static uint64_t CpuDirtied(uint64_t vaddr, uint64_t size) {
		uint64_t sum = 0;
		for (auto part = vaddr >> 28u; size != 0u && part <= (vaddr + size - 1u) >> 28u; ++part) {
			sum += s_cpu_dirtied[part % s_cpu_dirtied.size()].load(std::memory_order_acquire);
		}
		return sum;
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	// No page of the region is CPU-dirty, so an upload walk of any part of it finds nothing. Read without the lock: a
	// CPU write marks its page, then this, before the write goes on, so a reader that sees it clean came before the
	// write, as a walk that took the lock first would have.
	[[nodiscard]] bool CpuClean() const { return m_cpu_clean.load(std::memory_order_acquire); }
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().Any(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (m_cpu_dirty.Any(start, end) && !KYTY_UPSTREAM_PATH("25fb20e53")) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			// Written (or possibly written) by the CPU or the GPU: no longer known to be zero. A GPU-dirty
			// page is never CPU-dirty, so this changes no write protection.
			m_zero.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) NoteCpuDirty(vaddr, size);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) NoteCpuClean();
		// A page a shader marks, or the CPU dirties, is no longer kept (KeepGpuProtection).
		const bool kept = enable && m_kept.FirstRangeFrom(start).first < end;
		if (kept) {
			m_kept.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			if constexpr (enable) {
				m_cpu_written.SetRange(start, end);
			}
			UpdateProtection<!enable, false>();
			// After the write side, so that a kept page goes from no access to read-write in one change.
			if (kept) {
				UpdateProtection<false, true>();
			}
		} else {
			UpdateProtection<enable, true>();
		}
	}

	// The pages stop being GPU-written but stay read-protected: the tracked dispatch that did not write them
	// binds them again, and its mark is then no protection change. The CPU's first access lifts them
	// (ReleaseKept, or a write's ChangeState). Guest memory holds their data: no GPU byte was written.
	// Pages the CPU has touched before are kept too, so the CPU's touches fault again after each resolve; at
	// Marvel's Wolverine's Trask stand that's about 150 faults a frame on its game threads, for less protection
	// work on the GPU thread, which is the one that limits the frame there.
	void KeepGpuProtection(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		m_kept |= RegionBits(m_gpu_dirty, start, end);
		m_gpu_dirty.UnsetRange(start, end);
	}

	// The same for every page of `mask` (AddPages) at once.
	void KeepGpuProtection(const RegionBits& mask) {
		m_kept |= m_gpu_dirty & mask;
		m_gpu_dirty &= ~mask;
	}

	void AddPages(RegionBits& mask, uint64_t vaddr, uint64_t size) const {
		const auto [start, end] = GetPageRange(vaddr, size);
		mask.SetRange(start, end);
	}

	// Lifts the kept protection of [vaddr, vaddr + size) when `page` is kept (any page of it with
	// `page` == 0); false when there was nothing to lift.
	[[nodiscard]] bool ReleaseKept(uint64_t page, uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (page != 0 ? !m_kept.Get(GetPageRange(page, 1).first) : RegionBits(m_kept, start, end).None()) {
			return false;
		}
		m_kept.UnsetRange(start, end);
		UpdateProtection<false, true>();
		return true;
	}

	// A func taking (address, bytes, zero) gets the CPU-dirty runs split where the known-zero state changes.
	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				NoteCpuClean();
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		if constexpr (std::is_invocable_v<Func&, uint64_t, uint64_t, bool>) {
			const auto zero = mask & m_zero;
			for (const auto [first, last]: mask ^ zero) {
				func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE, false);
			}
			for (const auto [first, last]: zero) {
				func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE, true);
			}
		} else {
			for (const auto [first, last]: mask) {
				func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
			}
		}
	}

	// Known zero: guest memory that nothing has written since it was first mapped (direct memory never
	// mapped before), so it reads as zeros. Such pages stay write-protected while CPU-dirty: the first
	// write by anyone faults and ends the state (ChangeState). GPU-dirty pages are never marked.
	void MarkZero(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		for (const auto [first, last]: RegionBits(~m_gpu_dirty, start, end)) {
			m_zero.SetRange(first, last);
		}
		UpdateProtection<true, false>();
	}

	void ClearZero(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (RegionBits(m_zero, start, end).None()) {
			return;
		}
		m_zero.UnsetRange(start, end);
		UpdateProtection<false, false>();
	}

	template <typename Func>
	void ForEachZeroRange(uint64_t vaddr, uint64_t size, Func&& func) const {
		const auto [start, end] = GetPageRange(vaddr, size);
		for (const auto [first, last]: RegionBits(m_zero, start, end)) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Calls func(address, size) for the runs of pages in the range that became CPU dirty since
	// the previous call over them, or are still CPU dirty (further writes to a dirty page do not
	// fault), and forgets the former. Callers hold the lock.
	template <typename Func>
	void ConsumeCpuWritten(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits mask(m_cpu_written, start, end);
		mask |= RegionBits(m_cpu_dirty, start, end);
		m_cpu_written.UnsetRange(start, end);
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Marks the pages of the range GPU dirty unless they are CPU dirty, and calls func(address,
	// size) for the runs it marked. Callers hold the lock.
	template <typename Func>
	void MarkGpuDirtyUnlessCpuDirty(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits mask(~m_cpu_dirty, start, end);
		if (mask.None()) {
			return;
		}
		for (const auto [first, last]: mask) {
			m_gpu_dirty.SetRange(first, last);
		}
		UpdateProtection<true, true>();
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

private:
	static void NoteCpuDirty(uint64_t vaddr, uint64_t size) {
		for (auto part = vaddr >> 28u; size != 0u && part <= (vaddr + size - 1u) >> 28u; ++part) {
			s_cpu_dirtied[part % s_cpu_dirtied.size()].fetch_add(1, std::memory_order_release);
		}
	}
	inline static std::array<std::atomic<uint64_t>, 4096> s_cpu_dirtied {};

	// After every change of m_cpu_dirty, under the lock.
	void NoteCpuClean() { m_cpu_clean.store(m_cpu_dirty.None(), std::memory_order_release); }

	template <bool track, bool is_read>
	void UpdateProtection() {
		// Writable: CPU-dirty and not known zero.
		const auto protection = is_read ? ~(m_gpu_dirty | m_kept) : m_cpu_dirty & ~m_zero;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
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
		if ((size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		     size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) &&
		    !KYTY_UPSTREAM_PATH("25fb20e53")) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	std::atomic<bool> m_cpu_clean {false}; // m_cpu_dirty.None() (CpuClean)
	// Pages that became CPU dirty since ConsumeCpuWritten last visited them.
	RegionBits   m_cpu_written;
	RegionBits   m_gpu_dirty;
	RegionBits   m_zero;
	RegionBits   m_writable;
	RegionBits   m_readable;
	RegionBits   m_kept; // read-protected without GPU-written bytes (KeepGpuProtection)
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
