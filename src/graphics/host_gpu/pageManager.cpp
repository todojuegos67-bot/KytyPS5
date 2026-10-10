#include "graphics/host_gpu/pageManager.h"
#include "slow-log.h"

#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <sched.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {
thread_local std::vector<PageManager::DeferredRange>* g_deferred_write_protect = nullptr;
thread_local uint64_t g_unmapping_begin = 0;
thread_local uint64_t g_unmapping_end   = 0;
// PageManager::DeferReadProtection: the GPU thread's pending no-access ranges, disjoint and not adjacent (a range
// next to one extends it).
thread_local bool g_defer_read_protect = false;
struct PendingRange {
	uint64_t begin, end;
};
thread_local std::vector<PendingRange> g_read_protects;

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// The tracker reuses Win32 memory-protection tags as internal page-state values.
// Mirror their canonical numeric values so the shared state-machine logic is identical.
constexpr uint32_t PAGE_NOACCESS  = 0x01;
constexpr uint32_t PAGE_READONLY  = 0x02;
constexpr uint32_t PAGE_READWRITE = 0x04;
#endif
constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;
// HasReadWatchers' filter: a bit per 64 KiB granule.
constexpr uint64_t READ_GRANULE_SIZE  = 64 * 1024;
constexpr uint64_t READ_GRANULE_PAGES = READ_GRANULE_SIZE / PAGE_SIZE;

constexpr uint32_t NO_ACCESS_PROTECTION  = PAGE_NOACCESS;
constexpr uint32_t READ_ONLY_PROTECTION  = PAGE_READONLY;
constexpr uint32_t READ_WRITE_PROTECTION = PAGE_READWRITE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

Common::VirtualMemory::Mode ToMemoryMode(uint32_t protection) {
	switch (protection) {
		case NO_ACCESS_PROTECTION: return Common::VirtualMemory::Mode::NoAccess;
		case READ_ONLY_PROTECTION: return Common::VirtualMemory::Mode::Read;
		case READ_WRITE_PROTECTION: return Common::VirtualMemory::Mode::ReadWrite;
		default: Fatal("unmappable protection 0x%08" PRIx32, protection);
	}
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		// Owners call VirtualProtect inside: past a short spin a waiter yields its CPU instead
		// of competing with a descheduled owner (see TrackingSpinLock).
		uint32_t spins = 0;
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			do {
				if (++spins < 512) {
					_mm_pause();
				} else {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					SwitchToThread();
#else
					sched_yield();
#endif
				}
			} while (m_lock.test(std::memory_order_relaxed));
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

uint64_t PageStart(uint64_t vaddr) {
	return vaddr & ~(PAGE_SIZE - 1);
}

uint64_t PageEnd(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	return PageStart(vaddr + size - 1) + PAGE_SIZE;
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] uint32_t Perms() const noexcept {
			if (access_watchers != 0) {
				return NO_ACCESS_PROTECTION;
			}
			if (write_watchers != 0) {
				return READ_ONLY_PROTECTION;
			}
			return READ_WRITE_PROTECTION;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// Nonzero while a page has read watchers (HasReadWatchers).
		std::array<std::atomic<uint8_t>, REGION_PAGES> read_hint {};
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
		for (uint64_t i = 0; i < REGION_COUNT; i++) {
			regions[i].store(nullptr, std::memory_order_relaxed);
		}
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	static void ProtectHost(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size,
		                                                     ToMemoryMode(protection))) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", new=0x%08" PRIx32, vaddr,
			      protection);
		}
	}

	static void FlushReadProtect() noexcept {
		for (const auto& range: g_read_protects) ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
		g_read_protects.clear();
	}

	// The pending no-access ranges that overlap [begin, end), applied first (a protection change of those pages
	// comes after them; pages of the others keep theirs pending).
	static void FlushReadProtectOverlapping(uint64_t begin, uint64_t end) noexcept {
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			if (ranges[i].begin < end && begin < ranges[i].end) {
				ProtectHost(ranges[i].begin, ranges[i].end - ranges[i].begin, NO_ACCESS_PROTECTION);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
	}

	// A no-access range joins the pending ones: merged with those it overlaps or touches (one call for all).
	static void AddReadProtect(uint64_t begin, uint64_t end) noexcept {
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			if (ranges[i].begin <= end && begin <= ranges[i].end) {
				begin     = std::min(begin, ranges[i].begin);
				end       = std::max(end, ranges[i].end);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
		ranges.push_back({begin, end});
	}

	void Protect(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
		if (g_defer_read_protect) {
			if (protection == NO_ACCESS_PROTECTION) {
				AddReadProtect(vaddr, vaddr + size);
				return;
			}
			// (Another protection of the pages of a pending range comes after it; the other ranges stay pending:
			// protections of disjoint pages do not depend on each other's order.)
			FlushReadProtectOverlapping(vaddr, vaddr + size);
		}
		const auto end = vaddr + size;
		if (protection == READ_WRITE_PROTECTION && g_unmapping_end > g_unmapping_begin &&
		    vaddr < g_unmapping_end && end > g_unmapping_begin) {
			// Releases inside the range being unmapped are left to the unmap (SetUnmappingRange).
			if (vaddr < g_unmapping_begin) ProtectHost(vaddr, g_unmapping_begin - vaddr, protection);
			if (end > g_unmapping_end) ProtectHost(g_unmapping_end, end - g_unmapping_end, protection);
			return;
		}
		ProtectHost(vaddr, size, protection);
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				if (track && !is_read && g_deferred_write_protect != nullptr) {
					g_deferred_write_protect->push_back({base_addr + range_begin * PAGE_SIZE, range_bytes});
				} else {
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();
			if constexpr (is_read) {
				if (new_count != 0) MarkReadGranule(address); // (before the page's hint: the bits stay a superset)
				region.read_hint[page_index].store(new_count != 0, std::memory_order_release);
			}

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		if constexpr (is_read) SyncReadGranules(region, base_addr, first, last);
	}

	// A bit per 64 KiB granule: some page there may have read watchers. Set before a page's read_hint, cleared after
	// the granule's last one (both under the region's lock). HasReadWatchers asks it first: a hint byte per page sat on
	// its own cold line, the region pointer on another, for every page a table read touched (1% of the GPU thread at
	// 1-1); a line of these bits covers 32 MiB.
	std::unique_ptr<std::atomic<uint64_t>[]> read_granules =
	    std::make_unique<std::atomic<uint64_t>[]>(ADDRESS_SIZE / READ_GRANULE_SIZE / 64);

	void MarkReadGranule(uint64_t address) noexcept {
		const auto granule = address / READ_GRANULE_SIZE;
		read_granules[granule / 64].fetch_or(uint64_t {1} << (granule % 64), std::memory_order_release);
	}

	void SyncReadGranules(const Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		if (first >= last) return;
		for (size_t g = first / READ_GRANULE_PAGES; g <= (last - 1) / READ_GRANULE_PAGES; ++g) {
			bool any = false;
			for (size_t p = g * READ_GRANULE_PAGES; p < (g + 1) * READ_GRANULE_PAGES; ++p)
				any |= region.read_hint[p].load(std::memory_order_relaxed) != 0;
			const auto granule = base_addr / READ_GRANULE_SIZE + g;
			const auto bit     = uint64_t {1} << (granule % 64);
			if (any)
				read_granules[granule / 64].fetch_or(bit, std::memory_order_release);
			else if ((read_granules[granule / 64].load(std::memory_order_relaxed) & bit) != 0)
				read_granules[granule / 64].fetch_and(~bit, std::memory_order_release);
		}
	}

	[[nodiscard]] bool MayHaveReadWatchers(uint64_t vaddr, uint64_t last_byte) const noexcept {
		if (last_byte >= ADDRESS_SIZE) return true;
		for (auto g = vaddr / READ_GRANULE_SIZE; g <= last_byte / READ_GRANULE_SIZE; ++g)
			if (((read_granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) != 0) return true;
		return false;
	}

	void SyncProtection(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard lock(region->lock);
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto perms = region->pages[first].Perms();
					auto       next  = first + 1;
					while (next < last && region->pages[next].Perms() == perms) next++;
					Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms);
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	void ReapplyProtection(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard lock(region->lock);
				const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				     first < last;) {
					const auto perms = region->pages[first].Perms();
					auto       next  = first + 1;
					while (next < last && region->pages[next].Perms() == perms) next++;
					if (perms != READ_WRITE_PROTECTION) {
						Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms);
					}
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

const std::atomic<uint64_t>* PageManager::ReadGranules() const noexcept {
	static_assert(READ_GRANULE_SIZE == uint64_t {1} << READ_GRANULE_BITS);
	return m_impl->read_granules.get();
}

bool PageManager::HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) return false;
	if (!m_impl->MayHaveReadWatchers(vaddr, vaddr + size - 1)) return false;
	const auto end = PageStart(vaddr + size - 1);
	for (auto page = PageStart(vaddr);; page += PAGE_SIZE) {
		if (const auto* region = m_impl->FindRegion(page);
		    region != nullptr &&
		    region->read_hint[(page % REGION_SIZE) / PAGE_SIZE].load(std::memory_order_acquire))
			return true;
		if (page == end) return false;
	}
}

void PageManager::SetDeferredWriteProtectSink(std::vector<DeferredRange>* sink) noexcept {
	g_deferred_write_protect = sink;
}

void PageManager::SyncProtection(uint64_t vaddr, uint64_t size) {
	m_impl->SyncProtection(vaddr, size);
}

void PageManager::SetUnmappingRange(uint64_t begin, uint64_t end) noexcept {
	g_unmapping_begin = end == 0 ? 0 : begin;
	g_unmapping_end   = end;
}

bool PageManager::RestoreIfUnwatched(uint64_t vaddr) noexcept {
	// A region that does not exist yet never held a watcher: the protection is not ours.
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	SpinGuard lock(region->lock);
	if (region->pages[(vaddr % REGION_SIZE) / PAGE_SIZE].Perms() != READ_WRITE_PROTECTION) {
		return false;
	}
	return Libs::LibKernel::Memory::RestoreGuestWritable(PageStart(vaddr), PAGE_SIZE);
}

void PageManager::ReapplyProtection(uint64_t vaddr, uint64_t size) {
	m_impl->ReapplyProtection(vaddr, size);
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW UpdatePageWatchers %.1f ms track=%d addr=0x%llx size=0x%llx\n", ms, track ? 1 : 0,
		            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
	});
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

void PageManager::DeferReadProtection(bool on) noexcept {
	if (!on) Impl::FlushReadProtect();
	g_defer_read_protect = on;
}

void PageManager::FlushDeferredProtection() noexcept {
	Impl::FlushReadProtect();
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
