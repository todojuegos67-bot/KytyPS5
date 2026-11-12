#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// A/B switch for the clean-region skip (KYTY_TRACKER_SKIP_CLEAN=0 disables it).
inline const bool g_tracker_skip_clean =
    std::getenv("KYTY_TRACKER_SKIP_CLEAN") == nullptr || std::getenv("KYTY_TRACKER_SKIP_CLEAN")[0] == '1';

// A/B switch for the per-region bitmap fast path (KYTY_TRACKER_BITMAP=0 disables it).
inline const bool g_tracker_bitmap =
    std::getenv("KYTY_TRACKER_BITMAP") == nullptr || std::getenv("KYTY_TRACKER_BITMAP")[0] == '1';

// A/B switch for the dirty-region walk of read-only uploads (KYTY_TRACKER_DIRTY_WALK=0 disables
// it): a read-only sync whose regions all exist visits only the regions the bitmap marks
// CPU-dirty, instead of loading every region of the range. The game binds heap-wide V#s (up to
// 2.7 GB, ~700 regions) many times a draw; one dirty region anywhere sent each of those through
// every region (~0.3 us a sync, 27 syncs a draw at the Leap Attack prompt).
inline const bool g_tracker_dirty_walk = std::getenv("KYTY_TRACKER_DIRTY_WALK") == nullptr ||
                                         std::getenv("KYTY_TRACKER_DIRTY_WALK")[0] == '1';

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	// Lock-free and without touching the regions: every tracker region of the range exists and
	// none may hold CPU-dirty pages. The regions' own summaries, mirrored one bit per region, so a
	// buffer spanning hundreds of clean regions is answered from a few words. False is
	// conservative.
	[[nodiscard]] bool IsRangeCpuCleanHint(uint64_t vaddr, uint64_t size) const noexcept;
	// Answered from the regions' published copies of their bits, without their locks (mainline
	// #1284): exact on the GPU thread, which alone changes GPU-dirty bits and clears CPU-dirty
	// ones; other threads only add CPU-dirty pages, under the lock and before the page becomes
	// writable. Other threads' queries (the fault handler's IsGpuWrittenPage, crash reports) may
	// see a moment-old answer, which their callers check again under the lock (WriteClean,
	// InvalidateRegion).
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// gt7-caches-1112 (KYTY_SYNC_SKIP), GPU thread: whether every page of the range is GPU-dirty
	// (its regions all exist). A written sync of such a range uploads nothing and changes no bit
	// nor protection, so BufferCache::SynchronizeBuffer may skip it.
	[[nodiscard]] bool IsRangeAllGpuDirtyHint(uint64_t vaddr, uint64_t size) const noexcept;
	// A CPU write fault's release of [vaddr, vaddr + size) at once: each tracker region of it is
	// marked CPU-dirty (and writable) in the same lock hold that finds no GPU-written page there.
	// False when one holds GPU-written pages: that region is left as it was.
	[[nodiscard]] bool ReleaseForCpuWrite(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Calls func(address, bytes) for each part of the range, one per tracker region, that may
	// hold CPU-dirty pages (lock-free summaries: false is exact, true is conservative).
	template <typename Func>
	void ForEachMaybeCpuDirtyRegion(uint64_t vaddr, uint64_t size, Func&& func) {
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (manager->MaybeModified<DirtySource::Cpu>()) {
				func(manager->GetCpuAddr() + offset, bytes);
			}
		});
	}
	// The BDA walk (RenderContext::PrepareBda). A region is pending from its creation, and again
	// whenever pages of it become CPU-dirty or a buffer is registered over it (MarkBdaPending): a
	// walk that synchronized every buffer of a region leaves CPU-dirty only the pages no buffer
	// covers, so until one of those happens the next walks have nothing to do there.
	void MarkBdaPending(uint64_t vaddr, uint64_t size) noexcept;
	// Creates the tracker regions of the range that do not exist yet: all CPU-dirty and pending
	// for the next BDA walk, as every new region. How many it created.
	size_t TrackRegions(uint64_t vaddr, uint64_t size);
	// Diagnostics: whether the region of `vaddr` is pending for the next BDA walk.
	[[nodiscard]] bool IsBdaPending(uint64_t vaddr) const noexcept;
	// Memory mapped again (RegionManager::DirtyCleanIn): its clean pages become CPU-dirty and its
	// regions with CPU-dirty pages pending for the next BDA walk; how many pages became CPU-dirty.
	// Only regions that exist (a new one starts CPU-dirty and pending).
	size_t RedirtyForMap(uint64_t vaddr, uint64_t size);
	// Diagnostics: the pages of the range, by state (a region that does not exist counts none).
	struct PageStates {
		size_t present = 0; // pages in regions that exist
		size_t cpu     = 0;
		size_t gpu     = 0;
		size_t hot     = 0;
	};
	[[nodiscard]] PageStates DescribePages(uint64_t vaddr, uint64_t size);
	// Calls func(region address) for each pending region, which is pending no more.
	template <typename Func>
	void TakeBdaPending(Func&& func) {
		for (uint64_t top = 0; top < BDA_TOP_WORDS; top++) {
			if (m_bda_pending_top[top].load(std::memory_order_relaxed) == 0) {
				continue;
			}
			auto words = m_bda_pending_top[top].exchange(0, std::memory_order_acq_rel);
			while (words != 0) {
				const uint64_t word = top * 64 + static_cast<uint64_t>(std::countr_zero(words));
				words &= words - 1;
				auto regions = m_bda_pending[word].exchange(0, std::memory_order_acq_rel);
				while (regions != 0) {
					const uint64_t index =
					    word * 64 + static_cast<uint64_t>(std::countr_zero(regions));
					regions &= regions - 1;
					func(index * TRACKER_REGION_SIZE);
				}
			}
		}
	}
	// The BDA walk's uploads in one tracker region (round 80): for each span of `spans` (inside
	// one region, ascending, apart), its CPU-dirty pages become clean and func(span index,
	// address, bytes) gets their runs; then one protection change write-protects them all,
	// before the region's lock is released. Each span's own sync (ForEachUploadRange) made one
	// protection change a span: two to three a region, 7 us each, 4 % of the GPU thread in a race.
	template <typename Func>
	void TakeCpuDirtySpans(std::span<const std::pair<uint64_t, uint64_t>> spans, Func&& func) {
		if (spans.empty()) {
			return;
		}
		CheckNotInUploadCallback();
		const auto index = spans.front().first / TRACKER_REGION_SIZE;
		ValidateRange(spans.front().first, spans.front().second);
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return;
		}
		std::scoped_lock lock(manager->lock);
		for (size_t i = 0; i < spans.size(); i++) {
			const auto [address, bytes] = spans[i];
			if (bytes == 0 || address / TRACKER_REGION_SIZE != index ||
			    (address + bytes - 1) / TRACKER_REGION_SIZE != index) {
				EXIT("MemoryTracker: a span of the BDA walk leaves its region\n");
			}
			manager->TakeCpuModifiedRanges(
			    address, bytes,
			    [&](uint64_t run, uint64_t run_bytes) noexcept { func(i, run, run_bytes); });
		}
		manager->CommitCpuProtection();
	}
	// The CPU-dirty pages of the range, as runs of whole pages (appended to `runs`). Lock-free
	// skip of a region whose summary is clean.
	void CollectCpuDirtyRuns(uint64_t vaddr, uint64_t size,
	                         std::vector<std::pair<uint64_t, uint64_t>>& runs);

	// Hot pages (RegionManager::MarkHot): the pages of the range that hold no GPU-written bytes
	// stay writable from now on; how many became hot. Only regions that exist.
	size_t MarkHot(uint64_t vaddr, uint64_t size);
	// Every hot page becomes CPU-dirty again (RegionManager::RedirtyHot); how many did.
	size_t RedirtyHot();
	// The hot pages of the range become CPU-dirty (RegionManager::DirtyHotIn): a store to them
	// that did not fault (the GPU thread's own). Lock-free return while no page is hot.
	void DirtyHotIn(uint64_t vaddr, uint64_t size);
	// No page is hot any more.
	void ClearHot();
	// The pages that are hot now.
	[[nodiscard]] size_t HotPages() const noexcept {
		return m_hot_pages.load(std::memory_order_relaxed);
	}

	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!manager->IsModifiedUnlocked<DirtySource::Gpu>(offset, bytes)) {
				return;
			}
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		if (!is_written && g_tracker_skip_clean && g_tracker_bitmap &&
		    IsRangeCpuCleanHint(vaddr, size)) {
			// What the walk below finds for a read-only range whose regions all exist and are
			// clean: nothing to copy.
			const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
			upload_func();
			s_upload_owner = previous_upload_owner;
			return;
		}
		if (!is_written && g_tracker_skip_clean && g_tracker_bitmap && g_tracker_dirty_walk &&
		    AllRegionsPresent(vaddr, size)) {
			// The loop below for a read-only range, with the clean regions skipped by their bitmap
			// bits rather than by loading each region (the bits mirror the regions' summaries).
			const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
			ForEachCpuDirtyRegion(vaddr, size,
			                      [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				                      manager->lock.lock();
				                      manager->ForEachModifiedRange<DirtySource::Cpu, true>(
				                          manager->GetCpuAddr() + offset, bytes, range_func);
				                      manager->lock.unlock();
			                      });
			upload_func();
			s_upload_owner = previous_upload_owner;
			return;
		}
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool summary_clean = !manager->MaybeModified<DirtySource::Cpu>();
			if (!is_written && g_tracker_skip_clean && summary_clean) {
				return;
			}
			// A written range takes every lock: they stay held until it is marked GPU-dirty.
			if (!is_written && !manager->IsModifiedUnlocked<DirtySource::Cpu>(offset, bytes)) {
				return;
			}
			manager->lock.lock();
			if (summary_clean && manager->IsModified<DirtySource::Cpu>(offset, bytes)) {
				static std::atomic<uint32_t> misses = 0;
				if (misses.fetch_add(1) < 32) {
					LOGF("Tracker: CPU summary miss region=0x%016" PRIx64 " offset=0x%" PRIx64
					     " bytes=0x%" PRIx64 " written=%d\n",
					     manager->GetCpuAddr(), offset, bytes, static_cast<int>(is_written));
				}
			}
			if (is_written) {
				// Hot pages the CPU may have written since their last upload, unseen: uploaded
				// now, before the GPU's writes take the range (RegionManager::DropHotIn).
				manager->DropHotIn(manager->GetCpuAddr() + offset, bytes);
			}
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

	// Whether the calling thread is inside one of this tracker's upload callbacks, where the
	// tracker cannot be used.
	[[nodiscard]] bool InUploadCallback() const noexcept { return s_upload_owner == this; }
	// The page manager its regions protect through (round 111: deferred protection).
	[[nodiscard]] PageManager& Pages() noexcept { return m_page_manager; }

private:
	static constexpr size_t REGION_COUNT  = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static constexpr size_t BDA_TOP_WORDS = REGION_COUNT / 64 / 64;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	// Bits [low, high] of a bitmap word.
	static uint64_t WordMask(uint64_t low, uint64_t high) noexcept {
		return high - low == 63 ? ~uint64_t {0} : ((uint64_t {1} << (high - low + 1)) - 1) << low;
	}

	// Every tracker region of the range exists (regions are never removed once created).
	[[nodiscard]] bool AllRegionsPresent(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
			return false;
		}
		const uint64_t first = vaddr / TRACKER_REGION_SIZE;
		const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			const auto mask = WordMask(word == first / 64 ? first % 64 : 0,
			                           word == last / 64 ? last % 64 : 63);
			if ((m_present_regions[word].load(std::memory_order_acquire) & mask) != mask) {
				return false;
			}
		}
		return true;
	}

	// func(manager, offset, bytes) for the part of the range in each region whose CPU summary
	// bit is set; the range's regions must all exist (AllRegionsPresent).
	template <typename Func>
	void ForEachCpuDirtyRegion(uint64_t vaddr, uint64_t size, Func&& func) {
		const uint64_t first = vaddr / TRACKER_REGION_SIZE;
		const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		const uint64_t end   = vaddr + size;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			auto dirty = m_cpu_dirty_regions[word].load(std::memory_order_acquire) &
			             WordMask(word == first / 64 ? first % 64 : 0,
			                      word == last / 64 ? last % 64 : 63);
			while (dirty != 0) {
				const uint64_t index = word * 64 + static_cast<uint64_t>(std::countr_zero(dirty));
				dirty &= dirty - 1;
				const uint64_t region_begin = index * TRACKER_REGION_SIZE;
				const uint64_t begin        = std::max(vaddr, region_begin);
				const uint64_t finish       = std::min(end, region_begin + TRACKER_REGION_SIZE);
				func(m_regions[index].load(std::memory_order_acquire), begin - region_begin,
				     finish - begin);
			}
		}
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	// One bit per region: the region exists / its CPU summary is set (RegionManager::StoreSummary).
	std::unique_ptr<std::atomic<uint64_t>[]> m_present_regions;
	std::unique_ptr<std::atomic<uint64_t>[]> m_cpu_dirty_regions;
	// The regions the next BDA walk visits (one bit each), and the words of those with a bit set.
	std::unique_ptr<std::atomic<uint64_t>[]>       m_bda_pending;
	std::unique_ptr<std::atomic<uint64_t>[]>       m_bda_pending_top;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	// The regions holding hot pages (RegionManager::in_hot_list), and how many pages are hot. The
	// mutex is taken before a region's lock.
	std::mutex                                     m_hot_mutex;
	std::vector<RegionManager*>                    m_hot_regions;
	std::atomic<size_t>                            m_hot_pages {0};
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
