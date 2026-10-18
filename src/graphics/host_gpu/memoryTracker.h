#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Calls func(address, bytes) for each part of the range, one per tracker region, that may
	// hold CPU-dirty pages (lock-free summaries: false is exact, true is conservative).
	template <typename Func>
	void ForEachMaybeCpuDirtyRegion(uint64_t vaddr, uint64_t size, Func&& func) {
		if (UseRegionBitmap()) {
			IterateCpuSummary(vaddr, size,
			                  [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				                  func(manager->GetCpuAddr() + offset, bytes);
			                  });
			return;
		}
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (manager->MaybeModified<DirtySource::Cpu>()) {
				func(manager->GetCpuAddr() + offset, bytes);
			}
		});
	}
	// Queue after changing ownership; a concurrent drain must not consume an event
	// before the dirty pages it describes become visible.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
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
				{
					KYTY_PROFILER_BLOCK("MemoryTracker::FaultLockWait");
					manager->lock.lock();
				}
				std::scoped_lock lock(std::adopt_lock, manager->lock);
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
		QueueBdaSync(vaddr, size);
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
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		const auto  upload_region = [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		};
		if (!is_written && UseRegionBitmap()) {
			// The bitmap only lists existing regions: create the missing ones (they start
			// CPU-dirty) first, then visit only the regions whose CPU summary is set.
			Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
			IterateCpuSummary(vaddr, size, upload_region);
		} else {
			// One traversal creates missing regions and uploads them.
			Iterate<true>(vaddr, size,
			              [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				              if (!is_written && !manager->MaybeModified<DirtySource::Cpu>()) {
					              return;
				              }
				              upload_region(manager, offset, bytes);
			              });
		}
		if (is_written) {
			// The region locks stay held until the GPU-modified marks below.
			KYTY_PROFILER_BLOCK("MemoryTracker::UploadHoldingRegionLocks");
			upload_func();
		} else {
			KYTY_PROFILER_BLOCK("MemoryTracker::Upload");
			upload_func();
		}
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

private:
	static constexpr size_t REGION_COUNT  = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static constexpr size_t SUMMARY_WORDS = (REGION_COUNT + 63) / 64;

	// Research: KYTY_REGION_BITMAP=1 (live). Read synchronizations of large ranges (Wolverine
	// binds a 2.7 GiB buffer; ~400k syncs/s averaging 377 MiB) visited every region manager just
	// to read its CPU summary, a cache miss per 4 MiB. Each manager mirrors that summary into
	// m_cpu_summary_bits, so the walk reads a bitmap word per 256 MiB instead.
	static bool UseRegionBitmap();

	// Calls func(manager, offset, bytes) for each existing region of the range whose CPU summary
	// bit is set (conservative like the summary itself).
	template <typename Func>
	void IterateCpuSummary(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		if (size == 0) {
			return;
		}
		const uint64_t end   = vaddr + size;
		const uint64_t first = vaddr / TRACKER_REGION_SIZE;
		const uint64_t last  = (end - 1) / TRACKER_REGION_SIZE;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			uint64_t bits = m_cpu_summary_bits[word].load(std::memory_order_acquire);
			if (word == first / 64) {
				bits &= ~uint64_t {0} << (first % 64);
			}
			if (word == last / 64 && last % 64 != 63) {
				bits &= (uint64_t {1} << (last % 64 + 1)) - 1;
			}
			while (bits != 0) {
				const uint64_t index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
				bits &= bits - 1;
				auto* manager = m_regions[index].load(std::memory_order_acquire);
				if (manager == nullptr) {
					continue;
				}
				const uint64_t region_begin = index * TRACKER_REGION_SIZE;
				const uint64_t begin        = std::max(vaddr, region_begin);
				const uint64_t finish       = std::min(end, region_begin + TRACKER_REGION_SIZE);
				func(manager, begin - region_begin, finish - begin);
			}
		}
	}
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

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::unique_ptr<std::atomic<uint64_t>[]>       m_cpu_summary_bits;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	std::mutex                                     m_bda_sync_mutex;
	RangeSet                                       m_bda_sync;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
