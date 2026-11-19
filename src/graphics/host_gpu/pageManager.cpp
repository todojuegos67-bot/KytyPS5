#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"
#include "kernel/scanStamp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

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

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			std::atomic_signal_fence(std::memory_order_seq_cst);
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

// Round 111: the manager whose watcher additions on this thread leave their host protection to
// its worker (PageManager::DeferScope), null for none.
thread_local const void* t_defer_protection = nullptr;

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
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
		// KYTY_SCAN_STAMP (kernel/scanStamp.h): 1 while the host protection the tracker last gave
		// the page takes write access away (made, not only asked for: a deferred protection sets
		// it when the worker makes it). Cleared before a page becomes writable.
		std::array<std::atomic<uint8_t>, REGION_PAGES> host_ro {};
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
	}

	~Impl() {
		StopWorker();
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

	// KYTY_SCAN_STAMP: the pages' host_ro bits (lock-free; the caller holds the region's lock).
	void SetHostRo(uint64_t vaddr, uint64_t size, bool value) noexcept {
		for (auto page = Common::AlignDown(vaddr, PAGE_SIZE); page < vaddr + size;
		     page += PAGE_SIZE) {
			if (auto* region = FindRegion(page); region != nullptr) {
				region->host_ro[(page % REGION_SIZE) / PAGE_SIZE].store(value ? 1u : 0u,
				                                                       std::memory_order_seq_cst);
			}
		}
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		const GpuPhases::Scope phase(GpuPhases::Phase::Protect);
		GpuPhases::NoteProtect(vaddr, size,
		                       mode == Common::VirtualMemory::Mode::NoAccess ? 0u
		                       : mode == Common::VirtualMemory::Mode::Read   ? 1u
		                                                                     : 2u);
		// KYTY_SCAN_STAMP: a page about to be writable is stamped before (a write after it then
		// has a larger stamp); one write-protected, after (the writes before it, a smaller one).
		const bool stamps   = ScanStamp::On();
		const bool writable = mode == Common::VirtualMemory::Mode::ReadWrite;
		if (stamps && writable) {
			SetHostRo(vaddr, size, false);
			ScanStamp::Note(vaddr, size);
		}
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
		if (stamps && !writable) {
			ScanStamp::Note(vaddr, size);
			SetHostRo(vaddr, size, true);
		}
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
				// Added write watchers only make pages read-only: that can wait for the worker. A
				// GPU write's no-access protection is made at once, as the GPU thread's own reads
				// of those pages (indirect arguments, scans) must fault for the download.
				if (track && !is_read && t_defer_protection == this) {
					Defer(base_addr + range_begin * PAGE_SIZE, range_bytes);
				} else {
					if (track && t_defer_protection == this) {
						stat_kept.fetch_add(1, std::memory_order_relaxed);
					}
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
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
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

	// Round 111, deferred protection (PageManager::DeferScope). Called with the region's lock
	// held: the worker takes no region lock while it holds defer_mutex.
	void Defer(uint64_t vaddr, uint64_t size) {
		bool wake = false;
		{
			std::lock_guard lock(defer_mutex);
			wake = deferred.empty() && jobs.empty();
			deferred.push_back({vaddr, size});
			issued.store(issued.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		}
		// The worker sleeps only with both queues empty.
		if (wake) {
			defer_available.notify_one();
		}
	}

	void Post(Common::UniqueFunction<void> job) {
		bool wake = false;
		{
			std::lock_guard lock(defer_mutex);
			wake = deferred.empty() && jobs.empty();
			jobs.push_back(std::move(job));
			issued.store(issued.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		}
		if (wake) {
			defer_available.notify_one();
		}
	}

	void StartWorker() {
		std::call_once(worker_started, [this] {
			LOGF("Deferred protection: page protections the GPU thread's uploads and GPU writes "
			     "ask for are made on a worker thread (KYTY_ASYNC_MEMORY_SYNC, memsync:async)\n");
			worker = std::thread([this] { WorkerLoop(); });
		});
	}

	void StopWorker() {
		{
			std::lock_guard lock(defer_mutex);
			stop_worker = true;
		}
		defer_available.notify_all();
		if (worker.joinable()) {
			worker.join();
		}
	}

	// Returns once the first `count` deferred protections and jobs are done. With `help` (round
	// 156), a batch nobody is working on is done by the caller rather than waited for: the worker
	// slept through most of them (busy 10 % of a race) and the GPU thread waited for it to wake
	// up, 2-4 % of its time. One batch at a time (`processing`), so they still end in order.
	void WaitDeferred(uint64_t count, bool help) {
		if (completed.load(std::memory_order_acquire) >= count) {
			return;
		}
		const auto       begin = NowNs();
		std::unique_lock lock(defer_mutex);
		while (completed.load(std::memory_order_acquire) < count) {
			if (help && !processing && (!deferred.empty() || !jobs.empty())) {
				std::vector<DeferredRange>                batch;
				std::vector<Common::UniqueFunction<void>> batch_jobs;
				batch.swap(deferred);
				batch_jobs.swap(jobs);
				const auto batch_end = issued.load(std::memory_order_relaxed);
				processing           = true;
				lock.unlock();
				const auto helped_begin = NowNs();
				ProcessBatch(batch, batch_jobs);
				stat_helped.fetch_add(1, std::memory_order_relaxed);
				stat_helped_ns.fetch_add(NowNs() - helped_begin, std::memory_order_relaxed);
				lock.lock();
				completed.store(batch_end, std::memory_order_release);
				processing      = false;
				const bool more = !deferred.empty() || !jobs.empty() || stop_worker;
				lock.unlock();
				defer_done.notify_all();
				if (more) {
					defer_available.notify_one();
				}
				lock.lock();
				continue;
			}
			defer_done.wait(lock);
		}
		stat_waits.fetch_add(1, std::memory_order_relaxed);
		stat_wait_ns.fetch_add(NowNs() - begin, std::memory_order_relaxed);
	}

	// Each page of [vaddr, vaddr + size) gets the protection its counts ask for now, a host call
	// a run of pages that ask for the same one (writable pages are left: taking a watcher away
	// made them so at once). The region's lock is held across its calls, as the counts' other
	// changes protect in it.
	uint64_t ApplyCurrent(uint64_t vaddr, uint64_t size) {
		uint64_t   calls = 0;
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			auto*      region      = FindRegion(chunk_begin);
			chunk_begin            = chunk_end;
			if (region == nullptr) {
				continue;
			}
			SpinGuard lock(region->lock);
			for (size_t page = first; page < last;) {
				const auto mode = region->pages[page].Perms();
				size_t     run  = page + 1;
				while (run < last && region->pages[run].Perms() == mode) {
					run++;
				}
				if (mode != Common::VirtualMemory::Mode::ReadWrite) {
					calls++;
					// As Protect: watched pages left writable would lose the CPU's writes.
					if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(
					        region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE, mode)) {
						Fatal("deferred address-space protection failed at 0x%016" PRIx64
						      ", size=0x%016" PRIx64 ", mode=0x%08" PRIx32,
						      region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE,
						      static_cast<uint32_t>(mode));
					}
					if (ScanStamp::On()) {
						// KYTY_SCAN_STAMP: the pages write-protected only now, stamped after it
						// (those protected already keep their stamp).
						for (size_t p = page; p < run; p++) {
							auto& ro = region->host_ro[p];
							if (ro.load(std::memory_order_relaxed) == 0) {
								ScanStamp::Note(region_base + p * PAGE_SIZE, PAGE_SIZE);
								ro.store(1, std::memory_order_seq_cst);
							}
						}
					}
				}
				page = run;
			}
		}
		return calls;
	}

	struct DeferredRange {
		uint64_t vaddr = 0;
		uint64_t size  = 0;
	};

	// One batch: its protections, merged where adjacent, then its jobs (whoever took it: the
	// worker, or a waiter that helps). Counts for the "Deferred protection" line.
	void ProcessBatch(std::vector<DeferredRange>&                batch,
	                  std::vector<Common::UniqueFunction<void>>& batch_jobs) {
		const auto begin = NowNs();
		stat_ranges.fetch_add(batch.size(), std::memory_order_relaxed);
		std::sort(batch.begin(), batch.end(),
		          [](const DeferredRange& a, const DeferredRange& b) { return a.vaddr < b.vaddr; });
		// Overlapping and adjacent ranges as one.
		size_t out = 0;
		for (size_t i = 1; i < batch.size(); i++) {
			auto& last = batch[out];
			if (batch[i].vaddr <= last.vaddr + last.size) {
				last.size =
				    std::max(last.vaddr + last.size, batch[i].vaddr + batch[i].size) - last.vaddr;
			} else {
				batch[++out] = batch[i];
			}
		}
		batch.resize(batch.empty() ? 0 : out + 1);
		stat_merged.fetch_add(batch.size(), std::memory_order_relaxed);
		uint64_t calls = 0;
		for (const auto& range: batch) {
			calls += ApplyCurrent(range.vaddr, range.size);
		}
		stat_calls.fetch_add(calls, std::memory_order_relaxed);
		batch.clear();
		// The jobs after every protection of the batch, the ones queued before each among them,
		// in the order they were posted.
		for (const auto& job: batch_jobs) {
			job();
		}
		stat_jobs.fetch_add(batch_jobs.size(), std::memory_order_relaxed);
		batch_jobs.clear();
		stat_batches.fetch_add(1, std::memory_order_relaxed);
		stat_busy_ns.fetch_add(NowNs() - begin, std::memory_order_relaxed);
	}

	void WorkerLoop() {
		Common::PerfStats::NameThread("GpuProtect");
		std::vector<DeferredRange>                batch;
		std::vector<Common::UniqueFunction<void>> batch_jobs;
		auto                                      report_time = std::chrono::steady_clock::now();
		for (;;) {
			uint64_t batch_end = 0;
			{
				std::unique_lock lock(defer_mutex);
				// Not while a waiter does a batch (it wakes the worker when it is done and more
				// is queued, or the worker is stopping).
				defer_available.wait(lock, [&] {
					return !processing && (stop_worker || !deferred.empty() || !jobs.empty());
				});
				if (deferred.empty() && jobs.empty()) {
					return;
				}
				batch.swap(deferred);
				batch_jobs.swap(jobs);
				batch_end  = issued.load(std::memory_order_relaxed);
				processing = true;
			}
			ProcessBatch(batch, batch_jobs);
			{
				std::lock_guard lock(defer_mutex);
				completed.store(batch_end, std::memory_order_release);
				processing = false;
			}
			defer_done.notify_all();
			const auto now = std::chrono::steady_clock::now();
			if (now - report_time >= std::chrono::seconds(10)) {
				const auto seconds = std::chrono::duration<double>(now - report_time).count();
				const auto take    = [](std::atomic<uint64_t>& value) {
                    return value.exchange(0, std::memory_order_relaxed);
				};
				const auto ranges    = take(stat_ranges);
				const auto batches   = take(stat_batches);
				const auto merged    = take(stat_merged);
				const auto calls     = take(stat_calls);
				const auto jobs_run  = take(stat_jobs);
				const auto busy_ns   = take(stat_busy_ns);
				const auto waits     = take(stat_waits);
				const auto wait_ns   = take(stat_wait_ns);
				const auto kept      = take(stat_kept);
				const auto helped    = take(stat_helped);
				const auto helped_ns = take(stat_helped_ns);
				LOGF("Deferred protection (%.0f s): %" PRIu64 " ranges in %" PRIu64
				     " batches, %" PRIu64 " after merging, %" PRIu64 " host calls, %" PRIu64
				     " jobs (upload reads), %.1f ms busy; %" PRIu64
				     " waits for it, %.1f ms; %" PRIu64
				     " no-access protections made at once; %" PRIu64
				     " batches done by a waiter instead, %.1f ms (round 156)\n",
				     seconds, ranges, batches, merged, calls, jobs_run,
				     static_cast<double>(busy_ns) / 1e6, waits, static_cast<double>(wait_ns) / 1e6,
				     kept, helped, static_cast<double>(helped_ns) / 1e6);
				report_time = now;
			}
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;

	std::mutex                                defer_mutex;
	std::condition_variable                   defer_available;
	std::condition_variable                   defer_done;
	std::vector<DeferredRange>                deferred;
	std::vector<Common::UniqueFunction<void>> jobs;
	std::atomic<uint64_t>                     issued {0};
	std::atomic<uint64_t>                     completed {0};
	bool                                      stop_worker = false;
	std::once_flag                            worker_started;
	std::thread                               worker;
	bool                                      processing = false; // a batch is being done
	std::atomic<uint64_t>                     stat_waits {0};
	std::atomic<uint64_t>                     stat_wait_ns {0};
	std::atomic<uint64_t>                     stat_kept {0};
	// The worker's counts, kept here since a waiter may do a batch too (round 156).
	std::atomic<uint64_t> stat_ranges {0};
	std::atomic<uint64_t> stat_batches {0};
	std::atomic<uint64_t> stat_merged {0};
	std::atomic<uint64_t> stat_calls {0};
	std::atomic<uint64_t> stat_jobs {0};
	std::atomic<uint64_t> stat_busy_ns {0};
	std::atomic<uint64_t> stat_helped {0};
	std::atomic<uint64_t> stat_helped_ns {0};
};

static_assert(std::atomic<void*>::is_always_lock_free);

namespace {
// KYTY_SCAN_STAMP: the manager HostWriteProtected asks (the renderer's; the last one made).
std::atomic<const PageManager::Impl*> g_stamp_manager {nullptr};
} // namespace

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {
	(void)ScanStamp::Mode(); // read (and logged) here, not first in a fault handler
	g_stamp_manager.store(m_impl.get(), std::memory_order_release);
}

PageManager::~PageManager() {
	const Impl* self = m_impl.get();
	g_stamp_manager.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
}

bool PageManager::HostWriteProtected(uint64_t vaddr, uint64_t size) noexcept {
	const auto* impl = g_stamp_manager.load(std::memory_order_acquire);
	if (impl == nullptr || size == 0 || vaddr >= ADDRESS_SIZE || size > ADDRESS_SIZE - vaddr) {
		return false;
	}
	for (auto page = Common::AlignDown(vaddr, PAGE_SIZE); page < vaddr + size; page += PAGE_SIZE) {
		const auto* region = impl->FindRegion(page);
		if (region == nullptr ||
		    region->host_ro[(page % REGION_SIZE) / PAGE_SIZE].load(std::memory_order_seq_cst) ==
		        0) {
			return false;
		}
	}
	return true;
}

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

bool PageManager::IsReadWatched(uint64_t vaddr) const noexcept {
	const auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto index = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	return region->pages[index].access_watchers != 0;
}

bool PageManager::IsWriteWatched(uint64_t vaddr) const noexcept {
	const auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto  index = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	const auto& page  = region->pages[index];
	return page.write_watchers != 0 || page.access_watchers != 0;
}

PageManager::DeferScope::DeferScope(PageManager& manager, bool on): m_previous(t_defer_protection) {
	// Off keeps an outer scope's.
	if (on) {
		manager.m_impl->StartWorker();
		t_defer_protection = manager.m_impl.get();
	}
}

PageManager::DeferScope::~DeferScope() {
	t_defer_protection = m_previous;
}

void PageManager::PostDeferred(Common::UniqueFunction<void> job) {
	m_impl->StartWorker();
	m_impl->Post(std::move(job));
}

uint64_t PageManager::DeferredIssued() const noexcept {
	return m_impl->issued.load(std::memory_order_acquire);
}

bool PageManager::DeferredPending() const noexcept {
	return m_impl->completed.load(std::memory_order_acquire) <
	       m_impl->issued.load(std::memory_order_acquire);
}

void PageManager::WaitDeferred(uint64_t issued, bool help) {
	m_impl->WaitDeferred(issued, help);
}

void PageManager::ResyncHostProtection(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid() || vaddr >= ADDRESS_SIZE) {
		return;
	}
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
	for (auto chunk_begin = begin; chunk_begin < end;) {
		const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
		const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
		const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
		const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
		auto*      region      = m_impl->FindRegion(chunk_begin);
		chunk_begin            = chunk_end;
		if (region == nullptr) {
			continue; // never watched: the host protection is not the tracker's
		}
		SpinGuard lock(region->lock);
		// Runs of watched pages with the same protection, one host call each.
		for (size_t page = first; page < last;) {
			const auto perms = region->pages[page].Perms();
			size_t     run   = page + 1;
			while (run < last && region->pages[run].Perms() == perms) {
				run++;
			}
			if (perms != Common::VirtualMemory::Mode::ReadWrite) {
				m_impl->Protect(region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE, perms);
			}
			page = run;
		}
	}
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
