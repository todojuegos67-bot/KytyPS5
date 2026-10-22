#include "graphics/frameCosts.h"
#include "graphics/phaseStats.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <cinttypes>

#include "common/assert.h"
#include "common/liveSwitches.h"
#include "graphics/host_gpu/graphicContext.h"

#include "common/common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

namespace {

int64_t NowUs() {
	return std::chrono::duration_cast<std::chrono::microseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// KYTY_TICK_STATS=1 (live): every 5 s, how often the GPU's timeline value was queried from the
// driver (vkGetSemaphoreCounterValue, a kernel call on Windows) and how many polls
// KYTY_TICK_POLL_US skipped.
struct TickStats {
	std::atomic<uint64_t> refreshes {0};
	std::atomic<uint64_t> skipped {0};
	std::atomic<int64_t>  report_us {0};
};
TickStats g_tick_stats;

void NoteTickStats(bool refreshed) {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_TICK_STATS", 0);
	if (enabled.load(std::memory_order_relaxed) == 0) {
		return;
	}
	(refreshed ? g_tick_stats.refreshes : g_tick_stats.skipped)
	    .fetch_add(1, std::memory_order_relaxed);
	const auto now  = NowUs();
	auto       last = g_tick_stats.report_us.load(std::memory_order_relaxed);
	if (last == 0) {
		g_tick_stats.report_us.compare_exchange_strong(last, now);
		return;
	}
	if (now - last >= 5'000'000 && g_tick_stats.report_us.compare_exchange_strong(last, now)) {
		::printf("Tick polls (5 s): driver queries %llu, skipped %llu\n",
		         static_cast<unsigned long long>(g_tick_stats.refreshes.exchange(0)),
		         static_cast<unsigned long long>(g_tick_stats.skipped.exchange(0)));
		std::fflush(stdout);
	}
}

// Every blocking wait for the GPU (the tick had not completed), timed and keyed by the source line
// that asked for it, with the waiting thread: "GPU waits (5 s)". Costs nothing on the non-blocking
// path; a blocking wait already costs a kernel wait.
void NoteWaitSite(const std::source_location& where, int64_t us) {
	struct Entry {
		uint64_t count = 0;
		int64_t  us    = 0;
	};
	static std::mutex                             mutex;
	static std::unordered_map<std::string, Entry> sites;
	static int64_t                                report_us = 0;
	char                                          thread_name[32] = "?";
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::snprintf(thread_name, sizeof(thread_name), "tid %lu", static_cast<unsigned long>(GetCurrentThreadId()));
#endif
	const char* file = where.file_name();
	if (const char* slash = std::strrchr(file, '/')) file = slash + 1;
	if (const char* slash = std::strrchr(file, '\\')) file = slash + 1;
	std::lock_guard lock(mutex);
	auto& e = sites[std::string(file) + ":" + std::to_string(where.line()) + " [" + thread_name + "]"];
	e.count++;
	e.us += us;
	const auto now = NowUs();
	if (report_us == 0) {
		report_us = now;
	}
	if (now - report_us >= 5'000'000) {
		std::vector<std::pair<std::string, Entry>> sorted(sites.begin(), sites.end());
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
		::printf("GPU waits (5 s), blocking only, by site:\n");
		for (size_t i = 0; i < sorted.size() && i < 12; i++) {
			::printf("  %8.1f ms %6llu waits  %s\n", static_cast<double>(sorted[i].second.us) / 1000.0,
			         static_cast<unsigned long long>(sorted[i].second.count), sorted[i].first.c_str());
		}
		std::fflush(stdout);
		sites.clear();
		report_us = now;
	}
}

} // namespace

void MasterSemaphore::Poll() {
	static auto& poll_us  = Common::LiveSwitches::Get("KYTY_TICK_POLL_US", 0);
	const auto   interval = poll_us.load(std::memory_order_relaxed);
	if (interval > 0) {
		const auto now  = NowUs();
		const auto last = m_last_poll_us.load(std::memory_order_relaxed);
		if (now - last < interval) {
			NoteTickStats(false);
			return;
		}
		m_last_poll_us.store(now, std::memory_order_relaxed);
	}
	Refresh();
}

void MasterSemaphore::Refresh() {
	NoteTickStats(true);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick, std::source_location where) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	vk::Result result;
	{
		FrameCosts::Timer cost_timer {FrameCosts::g_gpu_wait_us, FrameCosts::g_gpu_waits};
		PhaseStats::Scope phase {PhaseStats::IsGpuThread() ? PhaseStats::GpuWait
		                                                      : PhaseStats::GuestPhase(PhaseStats::RenderGpuWait, PhaseStats::OtherGpuWait)};
		const auto        begin = NowUs();
		result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
		NoteWaitSite(where, NowUs() - begin);
	}
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

} // namespace Libs::Graphics
