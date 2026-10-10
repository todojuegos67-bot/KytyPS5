#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {
// 1: write protection of pages an upload marks clean runs on the upload worker, queued ahead
// of their copies (needs KYTY_ASYNC_UPLOAD; otherwise it applies before the copies as before).
extern volatile std::atomic<uint32_t> kyty_local_async_reprotect_mode;
}

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	// Zero means untracked, disabled, invalid, or spanning several regions: do not cache.
	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const {
		if (!GuestRange{vaddr, size}.Valid() ||
		    vaddr / TRACKER_REGION_SIZE != (vaddr + size - 1) / TRACKER_REGION_SIZE) return 0;
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager == nullptr ? 0 : manager->CpuModificationEpoch();
	}
	// A write protection of the region's pages is queued (KYTY_ASYNC_REPROTECT): their copies are taken
	// after it lands, so a write before it is uploaded and one after it faults.
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const {
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager != nullptr && manager->DeferredProtectionPending();
	}
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	// Every page CPU-dirty (untracked memory is): none is write-protected for the tracker.
	[[nodiscard]] bool IsRegionFullyCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionFullyGpuModified(uint64_t vaddr, uint64_t size);
	// `handoff` (a held caller lock) is released once the region lock is held, so the
	// protection change does not extend the caller's critical section.
	[[nodiscard]] bool TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size,
	                                               std::unique_lock<TrackingSpinLock>* handoff = nullptr) noexcept;
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	// CPU-dirty, the write protection kept (RegionManager::MarkCpuDirtyKeepProtection): for a write through the
	// backing view. False, with nothing changed, when a page is GPU-dirty.
	[[nodiscard]] bool MarkRegionAsCpuDirtyKeepProtection(uint64_t vaddr, uint64_t size);
	// The emulator wrote CPU-dirty memory directly (RegionManager::NoteHostWrite): the regions' epochs move.
	void               NoteHostWrite(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
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

	template <bool clear, typename Preflight, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Preflight&& preflight, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Preflight&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		std::vector<RegionManager*> managers;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
			managers.push_back(manager);
		});
		std::vector<std::unique_lock<TrackingSpinLock>> locks;
		locks.reserve(managers.size());
		for (auto* manager: managers) {
			locks.emplace_back(manager->lock);
		}
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const auto address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes,
			                                                                preflight);
		});
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(
			    manager->GetCpuAddr() + offset, bytes, func);
		});
		if constexpr (clear) {
			Iterate<false>(vaddr, size,
			               [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               const auto address = manager->GetCpuAddr() + offset;
				               manager->template ForEachModifiedRange<DirtySource::Gpu, true>(
				                   address, bytes, [](uint64_t, uint64_t) noexcept {});
			               });
		}
	}

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		ForEachDownloadRange<clear>(
		    vaddr, size, [](uint64_t, uint64_t) noexcept {}, std::forward<Func>(func));
	}

	// `last_dirty`: the latest LastDirtyClock of the range's regions (g_cpu_dirty_clock), read with the pages it uploads.
	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func, uint64_t* last_dirty = nullptr) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		// KYTY_ASYNC_REPROTECT: the pages this upload marks clean get their write protection
		// from the upload worker, queued ahead of their copies (see TakeDeferredProtects).
		const bool defer = !is_written && kyty_local_async_reprotect_mode.load(std::memory_order_relaxed) != 0;
		// Reused per thread (no nested upload, see CheckNotInUploadCallback above).
		thread_local std::vector<PageManager::DeferredRange> ranges;
		ranges.clear();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			if (last_dirty != nullptr) *last_dirty = std::max(*last_dirty, manager->LastDirtyClock());
			if (defer) PageManager::SetDeferredWriteProtectSink(&ranges);
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (defer) {
				PageManager::SetDeferredWriteProtectSink(nullptr);
				for (const auto& range: ranges) {
					manager->BeginDeferredProtection();
					s_deferred.push_back({manager, range.address, range.size});
				}
				ranges.clear();
			}
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		// Protections the upload did not hand to the worker apply here, still before
		// anything else can observe the pages as clean.
		for (const auto& deferred: s_deferred) deferred.manager->ApplyDeferredProtection(deferred.address, deferred.size);
		s_deferred.clear();
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

	struct DeferredProtect {
		RegionManager* manager = nullptr;
		uint64_t       address = 0, size = 0;
	};
	// The upload callback takes the protections to queue ahead of its copies.
	[[nodiscard]] static std::vector<DeferredProtect> TakeDeferredProtects() { return std::exchange(s_deferred, {}); }

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;
	inline static thread_local std::vector<DeferredProtect> s_deferred;

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
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
