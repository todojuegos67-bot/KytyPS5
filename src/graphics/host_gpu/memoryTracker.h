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
		auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		if (manager != nullptr && s_upload_owner == nullptr) HotLazy(manager, vaddr % TRACKER_REGION_SIZE, size); // (PROT-MENOS6/7)
		return manager == nullptr ? 0 : manager->CpuModificationEpoch();
	}
	// (crash-ba2b1f) Crash reports: the trackers' bits of the page of `vaddr` (RegionManager::PeekPageUnlocked), no lock
	// taken; false where no region tracks it.
	[[nodiscard]] bool PeekPageUnlocked(uint64_t vaddr, uint32_t* bits) const noexcept {
		if (vaddr >= TRACKER_ADDRESS_SIZE || m_regions == nullptr) return false;
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		if (manager == nullptr) return false;
		*bits = manager->PeekPageUnlocked(vaddr);
		return true;
	}
	// A write protection of the region's pages is queued (KYTY_ASYNC_REPROTECT): their copies are taken
	// after it lands, so a write before it is uploaded and one after it faults.
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const {
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager != nullptr && manager->DeferredProtectionPending();
	}
	// (KYTY_NATIVE_MEMORY=1: the CPU writes the write watch saw reach the tracker through GpuResourceManager::
	// SweepCpuWrites - BufferCache::InvalidateMemory, as their write faults did -, at the start of each submission and
	// after each satisfied WAIT_REG_MEM: no query here reads the write watch.)
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
		Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) { HotLazy(manager, offset, bytes); }); // (PROT-MENOS6/7)
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

	// ---- (policy1) KYTY_DS_NEIGHBOR_SPLIT: RegionManager's no-watch pages. Never called without the policy. -------
	// A write fault's page becomes no-watch (RegionManager::SetNoWatch). False: nothing changed (untracked memory,
	// a GPU-written page, no-watch already, refused before, or this thread is inside an upload).
	// (anuncio1) `why`: RegionManager::SetNoWatch's reason (0 when refused here: untracked, inside an upload).
	[[nodiscard]] bool SetNoWatchPage(uint64_t vaddr, int* why = nullptr) {
		if (why != nullptr) *why = 0;
		if (!GuestRange {vaddr, 1}.Valid() || s_upload_owner == this) return false;
		auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		if (manager == nullptr) return false;
		std::scoped_lock lock(manager->lock);
		return manager->SetNoWatch(vaddr, why);
	}
	// (anuncio1, KYTY_RAIZ_ANNOUNCE) The page's refusal lifted, then SetNoWatchPage, under one hold of the region lock.
	[[nodiscard]] bool UnblockSetNoWatchPage(uint64_t vaddr) {
		if (!GuestRange {vaddr, 1}.Valid() || s_upload_owner == this) return false;
		auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		if (manager == nullptr) return false;
		std::scoped_lock lock(manager->lock);
		if (!manager->UnblockNoWatch(vaddr)) return false;
		return manager->SetNoWatch(vaddr);
	}
	// Some page of the range is no-watch (a region without any costs one load).
	[[nodiscard]] bool AnyNoWatch(uint64_t vaddr, uint64_t size) {
		if (!GuestRange {vaddr, size}.Valid() || s_upload_owner == this) return false;
		return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			return manager->HasNoWatch() && manager->lock.ReadShared([&] { return manager->AnyNoWatch(offset, bytes); });
		});
	}
	// (PROT-MENOS7, KYTY_PROT_MENOS_V7=2 stale check) Some page of the range is hot.
	[[nodiscard]] bool AnyHot(uint64_t vaddr, uint64_t size) {
		if (!GuestRange {vaddr, size}.Valid() || s_upload_owner == this) return false;
		return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			return manager->HasHot() && manager->lock.ReadShared([&] { return manager->AnyHot(offset, bytes); });
		});
	}
	// The no-watch pages of the range CPU-dirty again (a hook announced a write); returns how many changed.
	uint64_t MarkNoWatchDirty(uint64_t vaddr, uint64_t size) {
		uint64_t pages = 0;
		if (!GuestRange {vaddr, size}.Valid() || s_upload_owner == this) return pages;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!manager->HasNoWatch()) return;
			std::scoped_lock lock(manager->lock);
			pages += manager->MarkNoWatchDirty(offset, bytes);
		});
		return pages;
	}
	// (nativo2) KYTY_DS_NATIVE_HEAP: the no-watch pages of the range not native yet become native (RegionManager::
	// KeepNative: CPU-dirty now). A range whose no-watch pages are all native already costs shared looks only.
	uint64_t KeepNative(uint64_t vaddr, uint64_t size) {
		uint64_t pages = 0;
		if (!GuestRange {vaddr, size}.Valid() || s_upload_owner == this) return pages;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!manager->HasNoWatch()) return;
			if (!manager->lock.ReadShared([&] { return manager->AnyNoWatchNotNative(offset, bytes); })) return;
			std::scoped_lock lock(manager->lock);
			pages += manager->KeepNative(offset, bytes);
		});
		return pages;
	}
	// (nativo2) The runs of native pages of every region that has some, as guest [address, address + size), appended
	// to `out` (each region's lock held only while its runs are copied). Returns the pages.
	// (compat) bufferCache.cpp (nativo3) la llama con `all`; sin KYTY_RAIZ es la misma recogida.
	template <typename Vector>
	uint64_t CollectNativeRuns(Vector* out, bool /*all*/) { return CollectNativeRuns(out); }
	template <typename Vector>
	uint64_t CollectNativeRuns(Vector* out) {
		std::vector<RegionManager*> managers;
		{
			std::lock_guard guard(m_region_mutex);
			for (const auto& region: m_region_storage)
				if (region->HasNative()) managers.push_back(region.get());
		}
		uint64_t pages = 0;
		for (auto* manager: managers) {
			std::scoped_lock lock(manager->lock);
			const uint64_t base = manager->GetCpuAddr();
			manager->ForEachNativeRun([&](size_t first, size_t last) {
				out->push_back({base + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE});
				pages += last - first;
			});
		}
		return pages;
	}
	// Every no-watch page CPU-dirty again (the hooks' queue overflowed); returns how many changed.
	uint64_t MarkAllNoWatchDirty() {
		std::vector<RegionManager*> managers;
		{
			std::lock_guard guard(m_region_mutex);
			for (const auto& region: m_region_storage)
				if (region->HasNoWatch()) managers.push_back(region.get());
		}
		uint64_t pages = 0;
		for (auto* manager: managers) {
			std::scoped_lock lock(manager->lock);
			pages += manager->MarkNoWatchDirty(0, TRACKER_REGION_SIZE);
		}
		return pages;
	}
	// (policy4) Every no-watch page watched again, CPU-dirty, and free to be made no-watch later (the hooks' queue
	// overflowed: an announcement or a request to watch a range again was lost); returns how many there were.
	uint64_t ClearAllNoWatch() {
		std::vector<RegionManager*> managers;
		{
			std::lock_guard guard(m_region_mutex);
			for (const auto& region: m_region_storage)
				if (region->HasNoWatch()) managers.push_back(region.get());
		}
		uint64_t pages = 0;
		for (auto* manager: managers) {
			std::scoped_lock lock(manager->lock);
			pages += manager->ClearNoWatch(0, TRACKER_REGION_SIZE, false);
		}
		return pages;
	}
	// The no-watch pages of the range watched again (RegionManager::ClearNoWatch); returns how many there were.
	uint64_t ClearNoWatch(uint64_t vaddr, uint64_t size, bool block) {
		uint64_t pages = 0;
		if (!GuestRange {vaddr, size}.Valid() || s_upload_owner == this) return pages;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!manager->HasNoWatch()) return;
			std::scoped_lock lock(manager->lock);
			pages += manager->ClearNoWatch(offset, bytes, block);
		});
		return pages;
	}

	// KYTY_VOLATILE=1 (anything else: off, nothing changes). The game's per-frame rings (Demon's Souls 01.007.000:
	// RenderContextFrameCPU/GPU, found in guest memory once the game made them) and the optional KYTY_VOLATILE_RANGES
	// ("0xADDR:0xSIZE,...") become volatile 4 MiB tracker regions (RegionManager::SetVolatile). Called on the GPU thread
	// at the first slice of each submission and after each satisfied WAIT_REG_MEM (GpuResourceManager::SweepCpuWrites),
	// with no region lock held: of the rings it re-dirties only the part the game may have written since the last call
	// (from the start of the game frame seen then to the ring's cursor now); the manual ranges whole. One relaxed load
	// when off.
	static void VolatileRedirty() noexcept;
	// (PROT-MENOS) KYTY_PROT_MENOS: the hot pages (RegionManager::HotStep) CPU-dirty again before a submission. GPU thread.
	static void HotRedirty() noexcept;
	// (PROT-MENOS6) KYTY_PROT_MENOS lazy check: the region's clean hot pages read for this submission (RegionManager::
	// HotLazy) before a GPU-thread query uses its state. No tracker lock may be held by the caller.
	// (PROT-MENOS7) With V7: only when [offset, offset + bytes) of the region holds a clean hot page, and on every thread
	// (never inside an upload callback: its region locks are held).
	static void HotLazy(RegionManager* manager, uint64_t offset = 0, uint64_t bytes = TRACKER_REGION_SIZE) noexcept {
		if (!ProtMenos::LazyOn()) return;
		if (ProtMenos::V7()) {
			if (s_upload_owner != nullptr) return;
		} else if (!ProtMenos::t_gpu_thread) {
			return;
		}
		manager->HotLazy(ProtMenos::g_submit_gen.load(std::memory_order_acquire), offset, bytes);
	}
	// (PROT-MENOS7) Before a BDA read of [vaddr, vaddr + size) (size 0: all guest memory): the lazy check over the
	// regions with hot pages it touches (ProtMenos::g_regions), so their written hot pages are dirty (and their regions'
	// BdaDirtyRegions bits set) before the BDA walk looks. Nothing without V7.
	static void HotLazyListed(uint64_t vaddr, uint64_t size) noexcept;

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

	// KYTY_VOLATILE: the regions over [vaddr, vaddr + size) volatile (created if missing); returns how many changed.
	size_t MakeVolatile(uint64_t vaddr, uint64_t size);
	// KYTY_VOLATILE: the volatile pages of [vaddr, vaddr + size) not GPU-dirty CPU-dirty again; returns the pages.
	uint64_t RedirtyVolatile(uint64_t vaddr, uint64_t size);
	// (v4) new_frame: RenderContextFrameCPU's frame moved since the last call; frame_known: it was read.
	void     StepGameRings(uint64_t& ring_pages, bool& new_frame, bool& frame_known);
	// KYTY_VOLATILE: registers the allocator/ctx granules the memory map found since the last call, then re-dirties
	// them by KYTY_VOLATILE_MODE (v4): frame (default) the regions in which some page became clean (VolatileCleanHint)
	// once per game frame plus the ctx blocks of the DCBs submitted since the last call; ww (native memory) the pages
	// the write watch saw written; submit (v3) the clean-hinted regions at every call. GPU thread only.
	void StepTagged(bool new_frame, bool frame_known, uint64_t& tagged_pages, uint64_t& tagged_regions);

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
