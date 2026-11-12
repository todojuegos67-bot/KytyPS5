#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions           = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
	m_present_regions   = std::make_unique<std::atomic<uint64_t>[]>(REGION_COUNT / 64);
	m_cpu_dirty_regions = std::make_unique<std::atomic<uint64_t>[]>(REGION_COUNT / 64);
	m_bda_pending       = std::make_unique<std::atomic<uint64_t>[]>(REGION_COUNT / 64);
	m_bda_pending_top   = std::make_unique<std::atomic<uint64_t>[]>(BDA_TOP_WORDS);
}

void MemoryTracker::MarkBdaPending(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
		return;
	}
	const uint64_t first = vaddr / TRACKER_REGION_SIZE;
	const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
	for (uint64_t index = first; index <= last; index++) {
		m_bda_pending[index / 64].fetch_or(uint64_t {1} << (index % 64), std::memory_order_release);
		m_bda_pending_top[index / 4096].fetch_or(uint64_t {1} << ((index / 64) % 64),
		                                         std::memory_order_release);
	}
}

size_t MemoryTracker::TrackRegions(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	size_t created = 0;
	for (uint64_t index = vaddr / TRACKER_REGION_SIZE;
	     index <= (vaddr + size - 1) / TRACKER_REGION_SIZE; index++) {
		if (m_regions[index].load(std::memory_order_acquire) == nullptr) {
			(void)GetOrCreateRegion(index);
			created++;
		}
	}
	return created;
}

bool MemoryTracker::IsBdaPending(uint64_t vaddr) const noexcept {
	if (vaddr >= TRACKER_ADDRESS_SIZE) {
		return false;
	}
	const uint64_t index = vaddr / TRACKER_REGION_SIZE;
	return ((m_bda_pending[index / 64].load(std::memory_order_acquire) >> (index % 64)) & 1u) != 0;
}

size_t MemoryTracker::RedirtyForMap(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	size_t dirtied = 0;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		dirtied += manager->DirtyCleanIn(manager->GetCpuAddr() + offset, bytes);
	});
	return dirtied;
}

MemoryTracker::PageStates MemoryTracker::DescribePages(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	PageStates states;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		const auto       counts = manager->CountPages(manager->GetCpuAddr() + offset, bytes);
		states.present +=
		    (offset % TRACKER_PAGE_SIZE + bytes + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
		states.cpu += counts.cpu;
		states.gpu += counts.gpu;
		states.hot += counts.hot;
	});
	return states;
}

void MemoryTracker::CollectCpuDirtyRuns(uint64_t vaddr, uint64_t size,
                                        std::vector<std::pair<uint64_t, uint64_t>>& runs) {
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		if (!manager->MaybeModified<DirtySource::Cpu>()) {
			return;
		}
		std::scoped_lock lock(manager->lock);
		manager->ForEachModifiedRange<DirtySource::Cpu, false>(
		    manager->GetCpuAddr() + offset, bytes,
		    [&](uint64_t address, uint64_t run_bytes) { runs.emplace_back(address, run_bytes); });
	});
}

bool MemoryTracker::IsRangeCpuCleanHint(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
		return false;
	}
	const uint64_t first = vaddr / TRACKER_REGION_SIZE;
	const uint64_t last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
	for (uint64_t word = first / 64; word <= last / 64; word++) {
		const uint64_t low  = word == first / 64 ? first % 64 : 0;
		const uint64_t high = word == last / 64 ? last % 64 : 63;
		const uint64_t mask =
		    high - low == 63 ? ~uint64_t {0} : ((uint64_t {1} << (high - low + 1)) - 1) << low;
		if ((m_present_regions[word].load(std::memory_order_acquire) & mask) != mask ||
		    (m_cpu_dirty_regions[word].load(std::memory_order_acquire) & mask) != 0) {
			return false;
		}
	}
	return true;
}

bool MemoryTracker::IsRangeAllGpuDirtyHint(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
		return false;
	}
	const uint64_t end = vaddr + size;
	for (uint64_t address = vaddr; address < end;) {
		const uint64_t index        = address / TRACKER_REGION_SIZE;
		const uint64_t region_begin = index * TRACKER_REGION_SIZE;
		const uint64_t region_end   = region_begin + TRACKER_REGION_SIZE;
		const uint64_t piece_end    = end < region_end ? end : region_end;
		const auto*    manager      = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr || !manager->MaybeModified<DirtySource::Gpu>() ||
		    !manager->IsAllGpuDirtyUnlocked(address - region_begin, piece_end - address)) {
			return false;
		}
		address = piece_end;
	}
	return true;
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range: vaddr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n", vaddr,
		     size);
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	// A new region is all CPU-dirty (its summary starts true): mirror that before it exists.
	const uint64_t bit = uint64_t {1} << (index % 64);
	m_cpu_dirty_regions[index / 64].fetch_or(bit, std::memory_order_acq_rel);
	ptr->SetCpuSummaryBit(&m_cpu_dirty_regions[index / 64], bit);
	// And pending for the BDA walk, being all CPU-dirty.
	ptr->SetBdaPendingBits(&m_bda_pending[index / 64], bit, &m_bda_pending_top[index / 4096],
	                       uint64_t {1} << ((index / 64) % 64));
	ptr->NoteBdaPending();
	m_present_regions[index / 64].fetch_or(bit, std::memory_order_acq_rel);
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	if (g_tracker_skip_clean && g_tracker_bitmap && IsRangeCpuCleanHint(vaddr, size)) {
		return false;
	}
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		if (g_tracker_skip_clean && !manager->MaybeModified<DirtySource::Cpu>()) {
			return false;
		}
		return manager->IsModifiedUnlocked<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		if (g_tracker_skip_clean && !manager->MaybeModified<DirtySource::Gpu>()) {
			return false;
		}
		return manager->IsModifiedUnlocked<DirtySource::Gpu>(offset, bytes);
	});
}

bool MemoryTracker::ReleaseForCpuWrite(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return !Iterate<false>(
	    vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    std::scoped_lock lock(manager->lock);
		    if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
			    return true;
		    }
		    manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
		    return false;
	    });
}

size_t MemoryTracker::MarkHot(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	size_t           marked = 0;
	std::scoped_lock hot_lock(m_hot_mutex);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		const auto       pages = manager->MarkHot(manager->GetCpuAddr() + offset, bytes);
		if (pages != 0 && !manager->in_hot_list) {
			manager->in_hot_list = true;
			m_hot_regions.push_back(manager);
		}
		marked += pages;
	});
	m_hot_pages.fetch_add(marked, std::memory_order_relaxed);
	return marked;
}

size_t MemoryTracker::RedirtyHot() {
	CheckNotInUploadCallback();
	size_t           pages = 0;
	size_t           hot   = 0;
	std::scoped_lock hot_lock(m_hot_mutex);
	if (m_hot_regions.empty()) {
		return 0;
	}
	for (auto* manager: m_hot_regions) {
		std::scoped_lock lock(manager->lock);
		pages += manager->RedirtyHot();
		hot += manager->HotCount();
	}
	// Pages the GPU wrote since left the hot set on their own (RegionManager::ChangeState).
	m_hot_pages.store(hot, std::memory_order_relaxed);
	return pages;
}

void MemoryTracker::DirtyHotIn(uint64_t vaddr, uint64_t size) {
	if (m_hot_pages.load(std::memory_order_relaxed) == 0) {
		return;
	}
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		(void)manager->DirtyHotIn(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::ClearHot() {
	CheckNotInUploadCallback();
	std::scoped_lock hot_lock(m_hot_mutex);
	for (auto* manager: m_hot_regions) {
		std::scoped_lock lock(manager->lock);
		manager->ClearHot();
		manager->in_hot_list = false;
	}
	m_hot_regions.clear();
	m_hot_pages.store(0, std::memory_order_relaxed);
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		if (!manager->IsModifiedUnlocked<DirtySource::Gpu>(offset, bytes)) {
			return;
		}
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UntrackMemory(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	std::vector<RegionManager*> managers;
	managers.reserve((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) /
	                 TRACKER_REGION_SIZE);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
		managers.push_back(manager);
	});

	std::vector<std::unique_lock<TrackingSpinLock>> locks;
	locks.reserve(managers.size());
	for (auto* manager: managers) {
		locks.emplace_back(manager->lock);
	}
	if (Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	    })) {
		EXIT("cannot untrack GPU-dirty memory\n");
	}
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

} // namespace Libs::Graphics
