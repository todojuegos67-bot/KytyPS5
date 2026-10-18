#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

// KYTY_FAULT_STATS=1 (live, default 0): every 5 s, the tracked-page faults by kind, the distinct
// pages and the pages that faulted most, to tell a few hot pages from many cold ones.
class FaultStats {
public:
	static bool Enabled() {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_FAULT_STATS", 0);
		return enabled.load(std::memory_order_relaxed) != 0;
	}

	void Add(bool write, uint64_t vaddr) {
		std::lock_guard lock(m_mutex);
		auto&           page = m_pages[vaddr >> 12u];
		(write ? page.writes : page.reads)++;
		(write ? m_writes : m_reads)++;
		const auto now = std::chrono::steady_clock::now();
		if (now - m_report < std::chrono::seconds(5)) {
			return;
		}
		std::vector<std::pair<uint64_t, Counts>> top(m_pages.begin(), m_pages.end());
		const auto                               count = std::min<size_t>(top.size(), 8);
		std::partial_sort(top.begin(), top.begin() + static_cast<ptrdiff_t>(count), top.end(),
		                  [](const auto& a, const auto& b) {
			                  return a.second.reads + a.second.writes >
			                         b.second.reads + b.second.writes;
		                  });
		::printf("Faults (5 s): %" PRIu64 " writes, %" PRIu64 " reads, %zu pages; top:", m_writes,
		         m_reads, m_pages.size());
		for (size_t i = 0; i < count; i++) {
			::printf(" 0x%" PRIx64 " w%" PRIu64 "/r%" PRIu64, top[i].first << 12u,
			         top[i].second.writes, top[i].second.reads);
		}
		::printf("\n");
		m_pages.clear();
		m_writes = 0;
		m_reads  = 0;
		m_report = now;
	}

private:
	struct Counts {
		uint64_t writes = 0;
		uint64_t reads  = 0;
	};
	std::mutex                            m_mutex;
	std::unordered_map<uint64_t, Counts>  m_pages;
	uint64_t                              m_writes = 0;
	uint64_t                              m_reads  = 0;
	std::chrono::steady_clock::time_point m_report = std::chrono::steady_clock::now();
};

FaultStats g_fault_stats;

} // namespace

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
	// Idle until KYTY_RECORD_THREAD turns on for a command buffer (commandScheduler.cpp).
	m_command_scheduler.EnableRecordingThread();
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	KYTY_PROFILER_FUNCTION();
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (FaultStats::Enabled()) {
		g_fault_stats.Add(access == PageFaultAccess::Write, fault_vaddr);
	}
	if (access == PageFaultAccess::Write) {
		// KYTY_WRITE_WINDOW_KB=N (live, 0 = off): a CPU write fault invalidates the whole aligned
		// N KiB window around it, as Demon's Souls' write window does (18e290f7), so the next
		// writes of a sequential fill do not fault page by page. Only a window that is mapped,
		// holds no GPU-written data and no image takes it; otherwise the faulting byte alone.
		// The invalidation is the same call as for one byte: pages are only marked CPU-written
		// earlier, so they upload again before the GPU reads them.
		static auto&   window_kib = Common::LiveSwitches::Get("KYTY_WRITE_WINDOW_KB", 0);
		const uint64_t window     = static_cast<uint64_t>(window_kib.load(std::memory_order_relaxed)) * 1024u;
		if (window >= 8192u && (window & (window - 1)) == 0) {
			const uint64_t begin = fault_vaddr & ~(window - 1);
			if (IsMapped(begin, window) && !m_buffer_cache.IsRegionGpuModified(begin, window) &&
			    !m_texture_cache.IsRegionGpuModified(begin, window) &&
			    !m_texture_cache.HasImagesInRegion(begin, window)) {
				m_buffer_cache.InvalidateMemory(begin, window);
				return true;
			}
		}
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		static uint64_t unmaps_total   = 0;
		static uint64_t unmaps_skipped = 0;
		static auto     stats_begin    = std::chrono::steady_clock::now();
		unmaps_total++;
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still access a
		// range with no cached data, so they must finish before it is unmapped. A cached buffer,
		// a pending asynchronous readback or an in-flight download sharing bytes with the range
		// (MayReferenceRange) also needs the drain.
		const bool drain = m_buffer_cache.MayReferenceRange(vaddr, size) ||
		                   m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		                   m_texture_cache.IsRegionRegistered(vaddr, size) ||
		                   m_command_scheduler.HasPendingPriorityOperations();
		if (!drain) {
			unmaps_skipped++;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - stats_begin >= std::chrono::seconds(5)) {
			char line[128];
			std::snprintf(line, sizeof(line),
			              "Unmaps (5 s): %" PRIu64 " total, %" PRIu64 " skipped the GPU drain\n",
			              unmaps_total, unmaps_skipped);
			Log::WriteToConsoleAndLog(line);
			unmaps_total   = 0;
			unmaps_skipped = 0;
			stats_begin    = now;
		}
		if (drain && m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Buffers are synchronized so a global-memory shader sees the CPU's writes; only the ranges
	// queued since the last synchronization (CPU writes, new buffers, new mappings) are walked.
	// The walk runs at most once per guest submission once something became CPU-dirty: what the
	// game wrote before submitting is visible to all its commands, as on the console, and the GPU
	// thread runs behind the game anyway. Per draw, the walk and the re-protection it triggers
	// took ~38 % of the GPU thread on the title menu (21 -> 32 fps). Ranges skipped here stay
	// queued for the next synchronization.
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		const auto submission = g_guest_submission_seq.load(std::memory_order_relaxed);
		if (submission == m_bda_synced_submission) {
			m_fault_process_pending = true;
			return;
		}
		m_bda_synced_submission = submission;
		m_bda_synced_epoch      = epoch;
	}
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	}
	m_fault_process_pending = true;
}

Common::LockGuard RenderContext::LockMutexProfiled() {
	KYTY_PROFILER_BLOCK("RenderContext::WaitMutex");
	return Common::LockGuard(m_mutex);
}

void RenderContext::RunGarbageCollector() {
	KYTY_PROFILER_FUNCTION();
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	// KYTY_GC_INTERVAL_US (live, default 4000; 0 = every call): collect at most once per interval.
	// This runs after every completed guest submission (~1300/s at the Wolverine spot) while both
	// collectors age in frames and budget per frame. Run 31 A/B 0/4000: collectors 16.1 -> 4.0
	// ms/s, mean frame 73.3 -> 72.6 ms; still ~250 collections per second.
	static auto& gc_interval = Common::LiveSwitches::Get("KYTY_GC_INTERVAL_US", 4000);
	const auto   interval    = gc_interval.load(std::memory_order_relaxed);
	const auto   now_us      = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	if (interval <= 0 || now_us - m_last_gc_us >= interval) {
		m_last_gc_us = now_us;
		m_texture_cache.RunGarbageCollector();
		m_buffer_cache.RunGarbageCollector();
	}

	// KYTY_MEMORY_STATS=1: every 5 s, what each cache holds against the device budget.
	static auto& stats       = Common::LiveSwitches::Get("KYTY_MEMORY_STATS", 0);
	static auto  report_time = std::chrono::steady_clock::now();
	if (stats.load(std::memory_order_relaxed) != 0) {
		const auto now = std::chrono::steady_clock::now();
		if (now - report_time >= std::chrono::seconds(5)) {
			report_time          = now;
			constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
			uint64_t allocation_bytes = 0;
			uint64_t block_bytes      = 0;
			m_graphics.GetDeviceAllocationStats(allocation_bytes, block_bytes);
			::printf("Memory: buffers %.2f GiB, images %.2f GiB, device usage %.2f GiB, "
			         "budget %.2f GiB (driver %.2f GiB), VMA allocations %.2f GiB in blocks %.2f "
			         "GiB\n",
			         static_cast<double>(m_buffer_cache.UsedMemory()) / GiB,
			         static_cast<double>(m_texture_cache.UsedMemory()) / GiB,
			         static_cast<double>(m_graphics.GetDeviceMemoryUsage()) / GiB,
			         static_cast<double>(m_graphics.GetTotalMemoryBudget()) / GiB,
			         static_cast<double>(m_graphics.GetDriverMemoryBudget()) / GiB,
			         static_cast<double>(allocation_bytes) / GiB,
			         static_cast<double>(block_bytes) / GiB);
			std::fflush(stdout);
			m_graphics.LogMemoryBudget();
		}
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
