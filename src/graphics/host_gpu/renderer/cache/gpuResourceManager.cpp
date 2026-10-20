#include "local/sync-stats.h"
#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "live-census.h"
#include "live-counters.h"
#include "async-upload.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <optional>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <utility>
#include <x86intrin.h>

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"
#include "speculation-state.h"
#include "fresh-memory.h"
#include "texture-offload.h"

extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_bda_dirty_regions_mode {0};
// 1: watchers removed while memory is released skip the host protection change there
// (PageManager::SetUnmappingRange); a write fault on a page nothing watches unprotects it.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_unmap_protect_skip_mode {0};
// Live switch (same-process A/B): a write fault in the 16 KiB window right after the one the same thread's last window
// fault ended in, within 2 ms (a sequential stream of writes), takes the next 64 KiB at once. (1-1 route, 120 cap, A-B-A
// over 29 points: window faults -13.6%, fps +0.7%, 1% low +3.8%, p99 -4.1%; uploads +9%, on the upload worker.)
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_window_stream_mode {1};
}

namespace Libs::Graphics {

GpuResourceManager::GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler), m_buffer_cache(graphics, scheduler, m_page_manager, m_texture_cache, this),
      m_texture_cache(graphics, scheduler, m_page_manager, m_buffer_cache), m_graphics(graphics) {}

GpuResourceManager::~GpuResourceManager() = default;

void GpuResourceManager::BeginAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_add((uint64_t{1} << 32u) + 1u,
		    std::memory_order_acq_rel);
		EXIT_IF(uint32_t(old >> 32u) == UINT32_MAX || uint32_t(old) == UINT32_MAX);
	}
}

void GpuResourceManager::EndAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_sub(1u, std::memory_order_release);
		EXIT_IF(uint32_t(old) == 0);
	}
}

uint64_t GpuResourceManager::PreparationAliasEpoch() const noexcept {
	const auto value = m_preparation_alias_epoch.load(std::memory_order_acquire);
	return uint32_t(value) == 0 ? value : 0;
}

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault) {
	// A small window amortizes faults from sequential CPU writes. Expanding a
	// fault is allowed only for mapped, unaliased pages with no GPU/image owner.
	constexpr uint64_t window_size  = 4 * TRACKER_PAGE_SIZE;
	constexpr uint64_t stream_size  = 16 * TRACKER_PAGE_SIZE;
	const auto         window_begin = fault & ~(window_size - 1);
	if (kyty_local_window_stream_mode.load(std::memory_order_relaxed) != 0) {
		// The window after the one this thread's last window fault ended in, within 2 ms (~6.4M TSC ticks at 3.2 GHz):
		// a stream, whose next 64 KiB goes at once (the next fault in it is the window after that).
		thread_local uint64_t last_end = 0, last_tsc = 0;
		const uint64_t        now      = __rdtsc();
		const bool            stream   = window_begin == last_end && now - last_tsc < 6'400'000;
		if (stream && TryInvalidateCpuWriteWindow(fault, window_begin, stream_size)) {
			last_end = window_begin + stream_size;
			last_tsc = now;
			return true;
		}
		if (!TryInvalidateCpuWriteWindow(fault, window_begin, window_size)) return false;
		last_end = window_begin + window_size;
		last_tsc = now;
		return true;
	}
	return TryInvalidateCpuWriteWindow(fault, window_begin, window_size);
}

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t window_begin, uint64_t window_size) {
	GuestRange selected {};
	uint64_t   mapping_epoch;
	{
		std::shared_lock mapped_lock(m_mapped_ranges_mutex);
		mapping_epoch = m_mapping_epoch;
		m_mapped_ranges.ForEachIntersection(window_begin, window_size, [&](RangeSet::Range range) {
			if (fault >= range.address && fault - range.address < range.size) {
				const auto begin =
				    (range.address + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
				const auto end = (range.address + range.size) & ~(TRACKER_PAGE_SIZE - 1);
				if (begin < end) selected = {begin, end - begin};
			}
		});
	}
	if (selected.size <= TRACKER_PAGE_SIZE || fault < selected.address ||
	    fault - selected.address >= selected.size ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(selected.address, selected.size))
		return false;
	// Kernel backing queries must not run while holding the resource-map lock.
	std::shared_lock mapped_lock(m_mapped_ranges_mutex);
	if (mapping_epoch != m_mapping_epoch ||
	    !m_mapped_ranges.Contains(selected.address, selected.size))
		return false;
	return m_buffer_cache.TryInvalidateCpuWriteWindow(fault, selected.address, selected.size);
}

// KYTY_FAULT_STATS=1 (diagnostic): every 5 s, the 2 MiB guest regions that took the most CPU-tracking faults
// (write-window, write, read), so the memory the game keeps rewriting can be told apart. A fixed open-addressed table.
namespace {
struct FaultSlot {
	std::atomic<uint64_t> region {0};
	std::atomic<uint32_t> window {0}, write {0}, read {0};
};
constexpr size_t           FaultSlots = 4096;
FaultSlot                  g_fault_slots[FaultSlots];
std::atomic<int64_t>       g_fault_report_ms {0};
bool FaultStatsOn() {
	static const bool on = [] { const char* v = std::getenv("KYTY_FAULT_STATS"); return v != nullptr && v[0] == '1'; }();
	return on;
}
void NoteFault(uint64_t vaddr, int kind) {
	const uint64_t region = (vaddr >> 21u) + 1u;
	size_t         i      = static_cast<size_t>((region * 0x9E3779B97F4A7C15ull) >> 52u) % FaultSlots;
	for (size_t probe = 0; probe < 64; ++probe, i = (i + 1) % FaultSlots) {
		uint64_t cur = g_fault_slots[i].region.load(std::memory_order_relaxed);
		if (cur == 0 && g_fault_slots[i].region.compare_exchange_strong(cur, region)) cur = region;
		if (cur != region) continue;
		(kind == 0 ? g_fault_slots[i].window : kind == 1 ? g_fault_slots[i].write : g_fault_slots[i].read)
		    .fetch_add(1, std::memory_order_relaxed);
		break;
	}
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch()).count();
	auto last = g_fault_report_ms.load(std::memory_order_relaxed);
	if (now - last < 5000 || !g_fault_report_ms.compare_exchange_strong(last, now)) return;
	struct Row { uint64_t region; uint32_t window, write, read; };
	std::vector<Row> rows;
	for (auto& slot: g_fault_slots) {
		const uint64_t r = slot.region.load(std::memory_order_relaxed);
		if (r == 0) continue;
		Row row {r - 1u, slot.window.exchange(0), slot.write.exchange(0), slot.read.exchange(0)};
		if (row.window + row.write + row.read != 0) rows.push_back(row);
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
		return a.window + a.write + a.read > b.window + b.write + b.read; });
	uint64_t total = 0;
	for (const auto& r: rows) total += r.window + r.write + r.read;
	std::printf("[faults] 5 s: %llu faults in %zu regions of 2 MiB; top:\n", static_cast<unsigned long long>(total), rows.size());
	for (size_t k = 0; k < rows.size() && k < 12; ++k)
		std::printf("[faults]   0x%010llx  window %u  write %u  read %u\n", static_cast<unsigned long long>(rows[k].region << 21u),
		            rows[k].window, rows[k].write, rows[k].read);
	std::fflush(stdout);
}
} // namespace

bool GpuResourceManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		// A never-written granule's first write (fresh memory): unwatched, done unless another watcher holds the page.
		if (FreshFirstWrite(fault_vaddr)) {
			uint64_t first = 0, end = 0;
			if (!m_page_manager.WatchedSpan(fault_vaddr & ~(TRACKER_PAGE_SIZE - 1), TRACKER_PAGE_SIZE, &first, &end))
				return true;
		}
		// (A write to a watched page of a view a partial unmap is mapping again: unseen while such a view came back
		// writable.)
		const bool in_remap = LibKernel::Memory::InRemapWindow(fault_vaddr);
		if (TryInvalidateCpuWriteWindow(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::WindowFault);
			if (FaultStatsOn()) NoteFault(fault_vaddr, 0);
			if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
			return true;
		}
		LiveCounters::Add(LiveCounters::WriteFault);
		if (FaultStatsOn()) NoteFault(fault_vaddr, 1);
		// No watcher holds the page, so no cache expects this fault: the protection outlived its
		// watchers and would fault forever. It gets the guest's own protection back.
		if (m_page_manager.RestoreIfUnwatched(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::StaleProtectRepairs);
			return true;
		}
		if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		LiveCounters::Add(LiveCounters::ReadFault);
		if (FaultStatsOn()) NoteFault(fault_vaddr, 2);
		if (LiveCensus::g_render) LiveCounters::Add(LiveCounters::RenderReadFaults);
		// A speculative translation's thread read GPU-written memory: the readback is the normal path's (the
		// scheduler's own command buffers and ticks), as the translation in order would make it there.
		struct NormalPath {
			Spec::State*                state    = std::exchange(Spec::t_state, nullptr);
			CommandScheduler::Recorder* recorder = CommandScheduler::ThreadRecorder();
			NormalPath() { CommandScheduler::SetThreadRecorder(nullptr); }
			~NormalPath() {
				Spec::t_state = state;
				CommandScheduler::SetThreadRecorder(recorder);
			}
		} normal_path;
		SyncStats::g_fault_address = fault_vaddr;
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool GpuResourceManager::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	TextureOffload::OnHostAccess(vaddr, size, false, false);
	FreshOverwrite(vaddr, size); // (a host write follows: its pages are writable, its granules written)
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::InvalidateImages(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) return false;
	TextureOffload::OnHostAccess(vaddr, size, false, true); // (its caller writes all of it: CompareMove)
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	TextureOffload::OnHostAccess(vaddr, size, false, true); // (it writes all of it)
	FreshOverwrite(vaddr, size);
	m_buffer_cache.InvalidateMemoryForWrite(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

void GpuResourceManager::MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEachIntersection(vaddr, size, [&](RangeSet::Range range) { out.Add(range.address, range.size); });
}

bool GpuResourceManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void GpuResourceManager::MapMemory(uint64_t vaddr, uint64_t size) {
	const auto map = [this, vaddr, size] {
		m_texture_cache.MapMemory(vaddr, size);
		{
			std::lock_guard lock(m_mapped_ranges_mutex);
			m_mapped_ranges.Add(vaddr, size);
			++m_mapping_epoch;
			if (size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size)) NoteBdaSpan(vaddr, size);
		}
		FreshWatch(vaddr, size);
	};
	if (m_gpu) {
		// A pending readback writes its backing when it finishes, which a new mapping aliasing
		// mapped memory would show; memory nothing else maps has no readback pending.
		const bool aliased = !LibKernel::Memory::IsUniqueGuestBackingRange(vaddr, size);
		m_gpu->SendCommandSync([this, map, aliased] {
			if (aliased) m_buffer_cache.DrainGuestReadback();
			map();
		});
	} else {
		map();
	}
}

void GpuResourceManager::UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	TextureOffload::OnHostAccess(vaddr, size, true, false);
	const auto unmap = [this, vaddr, size, releasing] {
		// Released memory loses its host protection with the mapping: the watchers removed
		// below need not restore it page run by page run first.
		struct UnmappingRange {
			explicit UnmappingRange(uint64_t begin, uint64_t end) { PageManager::SetUnmappingRange(begin, end); }
			~UnmappingRange() { PageManager::SetUnmappingRange(0, 0); }
		};
		std::optional<UnmappingRange> unmapping;
		if (releasing && kyty_local_unmap_protect_skip_mode.load(std::memory_order_relaxed) != 0) {
			unmapping.emplace(vaddr, vaddr + size);
		}
		// Readbacks write guest memory when they finish: the ones into the range finish first
		// (the others do not touch it).
		m_buffer_cache.DrainGuestReadback(vaddr, size);
		// Pending upload jobs on the range read its backing (or change its protection): they finish before
		// it changes owner. Jobs elsewhere keep running.
		AsyncUpload::DrainRange(vaddr, size);
		// The GPU only works on the caches' own buffers and images, which are freed after the
		// work in flight completes (deferred operations), so the unmap need not wait for the GPU
		// (it used to finish all of it: up to a frame of GPU work, for every guest unmap). What
		// writes guest memory after the GPU are the readbacks drained above and GPU-written
		// images' downloads, which complete on the priority thread: wait for those over the range.
		// (Deferred fault-buffer processing skips ranges no longer mapped.)
		const auto download_tick = m_scheduler.Active() ? m_texture_cache.PendingDownloadTick(vaddr, size) : 0;
		if (download_tick != 0) {
			const auto start = std::chrono::steady_clock::now();
			if (!m_scheduler.IsFree(download_tick)) m_scheduler.Wait(download_tick);
			m_scheduler.WaitPriorityOperations(download_tick);
			LiveCounters::Add(LiveCounters::UnmapFinishes);
			LiveCounters::Add(LiveCounters::UnmapFinishUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
		FreshUnwatch(vaddr, size, !releasing);
		const bool registered = size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		if (registered) {
			m_buffer_cache.DeleteUnmappedBuffers(vaddr, size);
			m_buffer_cache.SplitUnmappedBuffers(vaddr, size);
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		const auto epoch = ++m_mapping_epoch;
		if (registered) NoteBdaSpan(vaddr, size);
		if (size == 0) return;
		m_unmap_latest = epoch;
		constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
		for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last; ++g) {
			auto& leaf = m_unmap_epochs[g >> UnmapLeafBits];
			if (!leaf) leaf = std::make_unique<uint64_t[]>(leaf_mask + 1);
			leaf[g & leaf_mask] = epoch;
		}
	};
	if (m_gpu == nullptr) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

bool GpuResourceManager::UnmappedSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept {
	// (Nothing unmapped since: mostly so. Else each leaf the range spans looked up once.)
	if (size == 0 || m_unmap_latest <= epoch) return false;
	constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
	for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last;) {
		const auto stop = std::min(last, g | leaf_mask);
		if (const auto leaf = m_unmap_epochs.find(g >> UnmapLeafBits); leaf != m_unmap_epochs.end()) {
			const auto* epochs = leaf->second.get();
			for (auto i = g & leaf_mask; i <= (stop & leaf_mask); ++i)
				if (epochs[i] > epoch) return true;
		}
		g = stop + 1;
	}
	return false;
}

void GpuResourceManager::NoteBdaSpan(uint64_t vaddr, uint64_t size) {
	if (m_bda_spans.size() < 256) {
		m_bda_spans.push_back({vaddr, size});
	} else {
		m_bda_rebuild = true;
	}
}

void GpuResourceManager::RefreshBdaRanges() {
	if (!m_buffer_cache.TakeRegistrationSpans(m_bda_spans) || m_bda_spans.size() > 256) m_bda_rebuild = true;
	if (!m_bda_rebuild && m_bda_spans.empty()) return;
	LiveCounters::Add(LiveCounters::BdaRebuilds);
	// The tracker regions to collect again, as merged [begin, end) windows (requests lie in one region each).
	std::vector<std::pair<uint64_t, uint64_t>> windows;
	if (m_bda_rebuild) {
		windows.emplace_back(0, UINT64_MAX);
	} else {
		std::ranges::sort(m_bda_spans, {}, &GuestRange::address);
		for (const auto& span : m_bda_spans) {
			const auto begin = span.address / TRACKER_REGION_SIZE * TRACKER_REGION_SIZE;
			const auto end   = (span.End() + TRACKER_REGION_SIZE - 1) / TRACKER_REGION_SIZE * TRACKER_REGION_SIZE;
			if (!windows.empty() && begin <= windows.back().second) {
				windows.back().second = std::max(windows.back().second, end);
			} else {
				windows.emplace_back(begin, end);
			}
		}
	}
	m_bda_spans.clear();
	m_bda_rebuild = false;
	std::vector<RangeSet::Range> ranges;
	for (const auto& [window_begin, window_end] : windows) {
		m_buffer_cache.CollectMappedRegisteredRanges(m_mapped_ranges, window_begin, window_end, ranges);
		auto& fresh = m_fresh_bda_region_requests;
		fresh.clear();
		for (const auto& range : ranges) {
			const auto end = range.address + range.size;
			for (auto start = range.address; start < end;) {
				const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
				fresh.push_back({start, finish - start});
				start = finish;
			}
		}
		// An unchanged request keeps its proof: SynchronizeRegionRequest checks it against its region's CPU and
		// registration epochs (an unmap marks the range CPU-dirty, which moves the CPU epoch).
		const auto before = [](const BufferCache::SyncRegionRequest& request, uint64_t address) {
			return request.address < address;
		};
		const auto first = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), window_begin, before);
		const auto last  = std::lower_bound(first, m_bda_region_requests.end(), window_end, before);
		auto       old   = first;
		for (auto& request : fresh) {
			while (old != last && old->address < request.address) ++old;
			if (old != last && old->address == request.address && old->size == request.size) {
				request = *old;
			} else if (request.address / TRACKER_REGION_SIZE < BdaDirtyRegions::Regions) {
				// A new request starts unproven.
				BdaDirtyRegions::Mark(request.address / TRACKER_REGION_SIZE);
			}
		}
		m_bda_region_requests.insert(m_bda_region_requests.erase(first, last), fresh.begin(), fresh.end());
	}
}

bool GpuResourceManager::PrepareBdaReadRanges(std::span<const GuestRange> ranges) {
	if (ranges.empty() || ranges.size() > 128 ||
	    std::ranges::any_of(ranges, [](const auto& range) { return !range.Valid(); })) return false;
	if (auto* spec = Spec::Current()) {
		// Prepared when it is committed (and its guest memory writes made after its work).
		if (std::ranges::any_of(ranges, [&](const auto& range) { return spec->open_journal_set.Intersects(range.address, range.size); }))
			Spec::Refuse("memory the processor writes");
		spec->bda.insert(spec->bda.end(), ranges.begin(), ranges.end());
		return true;
	}
	// (Before the lock: making buffers asks for it.)
	m_buffer_cache.EnsureBdaReadable(ranges);
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	LiveCounters::Add(LiveCounters::BdaRangeCalls);
	LiveCounters::Add(LiveCounters::BdaRanges, ranges.size());
	for (const auto range : ranges) LiveCounters::Add(LiveCounters::BdaRangeMiB, range.size >> 20u);
	if (kyty_local_bda_dirty_regions_mode.load(std::memory_order_relaxed) != 0) {
		// KYTY_BDA_DIRTY_REGIONS: only regions whose bit is set can hold a stale request.
		for (const auto range : ranges) SynchronizeDirtyBdaRegions(range);
		m_fault_process_pending = true;
		m_bda_used              = true;
		return true;
	}
	for (const auto range : ranges) {
		auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), range.address,
		    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
		for (; it != m_bda_region_requests.end() && it->address < range.End(); ++it)
			m_buffer_cache.SynchronizeRegionRequest(*it);
	}
	m_fault_process_pending = true;
	m_bda_used              = true;
	return true;
}

void GpuResourceManager::SynchronizeDirtyBdaRegions(GuestRange range) {
	const auto first = range.address / TRACKER_REGION_SIZE;
	if (first >= BdaDirtyRegions::Regions) return;
	const auto last = std::min((range.End() - 1) / TRACKER_REGION_SIZE, BdaDirtyRegions::Regions - 1);
	for (auto word = first / 64; word <= last / 64; ++word) {
		auto mask = ~uint64_t {0};
		if (word == first / 64) mask &= ~uint64_t {0} << (first % 64);
		if (word == last / 64) mask &= ~uint64_t {0} >> (63 - last % 64);
		auto& bits = BdaDirtyRegions::g_bits[word];
		for (auto pending = bits.load(std::memory_order_acquire) & mask; pending != 0; pending &= pending - 1) {
			const auto bit = uint64_t {1} << std::countr_zero(pending);
			// Clear before synchronizing: a CPU write after the epoch read sets it again.
			if ((bits.fetch_and(~bit, std::memory_order_acq_rel) & bit) == 0) continue;
			const auto region_begin = (word * 64 + static_cast<uint64_t>(std::countr_zero(pending))) * TRACKER_REGION_SIZE;
			auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), region_begin,
			    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
			bool unproven = false;
			for (; it != m_bda_region_requests.end() && it->address < region_begin + TRACKER_REGION_SIZE; ++it) {
				m_buffer_cache.SynchronizeRegionRequest(*it);
				unproven |= it->cpu_epoch == 0;
			}
			// A request left unproven by its own uploads' queued write protection needs no other visit:
			// a write before the protection lands is in the copies taken after it, and a write after it
			// faults (and marks the region).
			if (unproven && !m_buffer_cache.DeferredProtectionPending(region_begin))
				BdaDirtyRegions::Mark(region_begin / TRACKER_REGION_SIZE);
		}
	}
}

void GpuResourceManager::PrepareBda() {
	if (auto* spec = Spec::Current()) {
		// Prepared when it is committed (and its guest memory writes made after its work).
		if (spec->writes.size() > spec->Sealed().writes) Spec::Refuse("memory the processor writes");
		spec->bda_all = true;
		return;
	}
	// Unknown shader addresses cannot prove disjointness from an in-flight readback.
	LiveCounters::Add(LiveCounters::BdaFullSyncs);
	m_buffer_cache.DrainGuestReadback();
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	for (auto& request : m_bda_region_requests) m_buffer_cache.SynchronizeRegionRequest(request);
	m_fault_process_pending = true;
	m_bda_used              = true;
}

void GpuResourceManager::EndSubmission() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	if (std::exchange(m_gpu_flip_pending, false)) RunCollectors();
}

void GpuResourceManager::AdvanceFrame() {
	m_texture_cache.AdvanceFrame();
	RunCollectors();
	if (auto* fresh = FreshMemory::Get(); fresh != nullptr && ++m_fresh_report_flip % 3600 == 0) {
		size_t parts = 0;
		{
			std::lock_guard lock(m_fresh_mutex);
			parts = m_fresh_parts.size();
		}
		std::printf("Fresh memory: %llu MiB of uploads were never-written bytes (zeros, not read); %zu parts watched; "
		            "%llu MiB written at a first write, %llu MiB by other writes\n",
		            static_cast<unsigned long long>(fresh->zero_upload_bytes.load() >> 20u), parts,
		            static_cast<unsigned long long>(fresh->faults.load() >> 20u),
		            static_cast<unsigned long long>(fresh->marked.load() >> 20u));
		std::fflush(stdout);
	}
}

void GpuResourceManager::FreshWatch(uint64_t vaddr, uint64_t size) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr || size == 0) return;
	bool gpu = false, cpu_write = false;
	if (!LibKernel::Memory::GuestRangeAccess(vaddr, size, &gpu, &cpu_write)) {
		FreshMemory::MarkGuestRange(vaddr, size);
		return;
	}
	if (!cpu_write) return; // (no guest write here goes unseen)
	if (!gpu) {
		FreshMemory::MarkGuestRange(vaddr, size); // (written unwatched)
		return;
	}
	std::lock_guard lock(m_fresh_mutex);
	uint64_t        run_begin = 0, run_end = 0;
	const auto      flush     = [&] {
		if (run_end > run_begin) m_page_manager.UpdatePageWatchers<true>(run_begin, run_end - run_begin);
		run_begin = run_end = 0;
	};
	for (uint64_t done = 0; done < size;) {
		const uint64_t at      = vaddr + done;
		const auto     end     = LibKernel::Memory::BackingMappingEnd(at);
		const auto     piece   = end > at ? std::min(end - at, size - done) : size - done;
		const auto*    backing = end > at ? LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
		if (backing == nullptr) {
			done += piece;
			continue;
		}
		for (uint64_t offset = 0; offset < piece;) {
			const auto address = reinterpret_cast<uint64_t>(backing) + offset;
			const auto granule = FreshMemory::GranuleOf(*fresh, address);
			const auto bytes   = granule == UINT64_MAX
			                         ? piece - offset
			                         : std::min(piece - offset,
			                                    FreshMemory::Granule - ((address - fresh->base) & (FreshMemory::Granule - 1)));
			const auto begin = at + offset;
			offset += bytes;
			if (granule == UINT64_MAX || FreshMemory::IsWritten(*fresh, granule)) continue;
			if ((begin & (TRACKER_PAGE_SIZE - 1)) != 0 || (bytes & (TRACKER_PAGE_SIZE - 1)) != 0 ||
			    m_fresh_parts.contains(begin)) {
				// (Not whole pages, or watched already: counted as written rather than half watched.)
				if (FreshMemory::MarkGranule(*fresh, granule)) fresh->marked.fetch_add(FreshMemory::Granule);
				FreshUnwatchGranule(granule);
				continue;
			}
			m_fresh_parts.emplace(begin, FreshPart {begin + bytes, granule});
			m_fresh_granules[granule].push_back(begin);
			FreshMemory::SetArmed(*fresh, granule, true);
			fresh->watched_bytes.fetch_add(bytes, std::memory_order_relaxed);
			if (begin != run_end) flush();
			if (run_end == 0) run_begin = begin;
			run_end = begin + bytes;
		}
		done += piece;
	}
	flush();
}

void GpuResourceManager::FreshUnwatchGranule(uint64_t granule) {
	const auto found = m_fresh_granules.find(granule);
	if (found == m_fresh_granules.end()) return;
	for (const auto begin: found->second) {
		const auto part = m_fresh_parts.find(begin);
		if (part == m_fresh_parts.end()) continue;
		m_page_manager.UpdatePageWatchers<false>(begin, part->second.end - begin);
		m_fresh_parts.erase(part);
	}
	m_fresh_granules.erase(found);
	if (auto* fresh = FreshMemory::Get()) FreshMemory::SetArmed(*fresh, granule, false);
}

void GpuResourceManager::FreshUnwatch(uint64_t vaddr, uint64_t size, bool mark) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr || size == 0) return;
	std::lock_guard       lock(m_fresh_mutex);
	std::vector<uint64_t> granules;
	auto                  it = m_fresh_parts.upper_bound(vaddr);
	if (it != m_fresh_parts.begin()) --it;
	for (; it != m_fresh_parts.end() && it->first < vaddr + size; ++it) {
		if (it->second.end <= vaddr) continue;
		// (A part left partly mapped loses its watch with the others: written.)
		const bool whole = it->first >= vaddr && it->second.end <= vaddr + size;
		if ((mark || !whole) && FreshMemory::MarkGranule(*fresh, it->second.granule))
			fresh->marked.fetch_add(FreshMemory::Granule);
		granules.push_back(it->second.granule);
	}
	for (const auto granule: granules) FreshUnwatchGranule(granule);
}

bool GpuResourceManager::FreshFirstWrite(uint64_t vaddr) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr) return false;
	std::lock_guard lock(m_fresh_mutex);
	auto            it = m_fresh_parts.upper_bound(vaddr);
	if (it == m_fresh_parts.begin()) return false;
	--it;
	if (vaddr >= it->second.end) return false;
	const auto granule = it->second.granule;
	if (FreshMemory::MarkGranule(*fresh, granule)) fresh->faults.fetch_add(FreshMemory::Granule);
	FreshUnwatchGranule(granule);
	return true;
}

void GpuResourceManager::FreshOverwrite(uint64_t vaddr, uint64_t size) {
	if (FreshMemory::Get() == nullptr || size == 0) return;
	FreshMemory::MarkGuestRange(vaddr, size);
	FreshUnwatch(vaddr, size, true);
}

void GpuResourceManager::RunCollectors() {
	// Collection runs per frame: per submission (dozens a frame here) its ages of 16 to 160 ticks
	// were a few frames, and it deleted what the next frames used again.
	m_texture_cache.RunGarbageCollector();
	// A shader that reads guest memory through the BDA page table touches no buffer in the LRU: a
	// buffer collected under it faults back at once, and each registration invalidates every BDA
	// region proof. Buffers are collected only after a frame without such shaders.
	// Demon's Souls runs such shaders every frame, so its buffers were never collected: the buffer cache grew from 0.5 to
	// 4.2 GB in 17 minutes of 01.005.000 and took the video memory to 12.8 GB of a 10 GB budget. Past the budget they
	// are collected once a second anyway (the faults and BDA proofs that costs, once a second, instead of a full card).
	const bool bda_free    = !std::exchange(m_bda_used, false);
	const bool over_budget = m_graphics.CanReportMemoryUsage() &&
	                         m_graphics.GetDeviceMemoryUsage() >= m_graphics.GetTotalMemoryBudget();
	m_buffer_cache.RunGarbageCollector(bda_free || (over_budget && (++m_collections % 60) == 0));
}

} // namespace Libs::Graphics
