#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "live-census.h"
#include "live-counters.h"
#include "device-fault.h"

#include "local-platform.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

<<<<<<< HEAD
	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore[0]);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore[0] == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	for (auto& semaphore: m_semaphore)
		if (semaphore != nullptr) m_graphics.device.destroySemaphore(semaphore, nullptr);
	delete[] m_lane_of;
}

void MasterSemaphore::EnableLanes() {
	if (m_lanes_on) return;
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;
	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;
	EXIT_NOT_IMPLEMENTED(m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore[1]) != vk::Result::eSuccess);
	m_lane_of = new std::atomic<uint8_t>[LaneRing] {};
	// (Ticks allocated before this belong to lane 0: the ring is zero.)
	m_last_tick[0].store(m_current_tick.load() - 1, std::memory_order_release);
	m_lanes_on = true;
=======
	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);

	const char* setting = std::getenv("KYTY_TIMELINE_WATCHER");
	if (setting != nullptr && setting[0] != '\0') {
		m_watched = setting[0] == '1';
	} else {
		// (32 reads of the new timeline, the second half timed: what one costs here.)
		uint64_t   counter = 0;
		const auto read    = [&] { (void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter); };
		for (int i = 0; i < 16; i++) read();
		const auto begin = std::chrono::steady_clock::now();
		for (int i = 0; i < 16; i++) read();
		const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 16;
		m_watched       = us >= 1.0;
		static std::atomic<bool> told {false};
		if (!told.exchange(true)) {
			std::printf("Timeline reads: %.2f us each: %s\n", us,
			            m_watched ? "a thread watches the GPU's progress (KYTY_TIMELINE_WATCHER=0: read where asked)"
			                      : "read where asked");
			std::fflush(stdout);
		}
	}
	if (m_watched) m_watcher = std::jthread([this](const std::stop_token& stop) { Watch(stop); });
}

MasterSemaphore::~MasterSemaphore() {
	if (m_watcher.joinable()) {
		m_watcher.request_stop();
		m_watcher.join();
	}
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
>>>>>>> upstream/experiment/perf-40fps-20260926
}

void MasterSemaphore::Publish(uint64_t counter) noexcept {
	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release, std::memory_order_relaxed)) {
	}
}

// The watcher: each tick past the known one waited for (one not submitted yet too: a timeline wait may come first),
// then the timeline read once (the GPU may be further).
void MasterSemaphore::Watch(const std::stop_token& stop) {
	LocalPlatform::SetThreadName("Kyty.Timeline");
	while (!stop.stop_requested()) {
		const uint64_t        next = KnownGpuTick() + 1;
		vk::SemaphoreWaitInfo wait_info {};
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores    = &m_semaphore;
		wait_info.pValues        = &next;
		const auto result        = m_graphics.device.waitSemaphores(&wait_info, 20'000'000ull); // (20 ms: a stop is seen)
		if (result == vk::Result::eTimeout) continue;
		if (result == vk::Result::eErrorDeviceLost) DeviceFault::Report();
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		uint64_t counter = next;
		if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter) != vk::Result::eSuccess || counter == UINT64_MAX)
			DeviceFault::Report();
		Publish(std::max(counter, next));
	}
}

void MasterSemaphore::Refresh() {
<<<<<<< HEAD
	for (int lane = 0; lane < (m_lanes_on ? 2 : 1); ++lane) {
		uint64_t   counter = 0;
		const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore[lane], &counter);
		// (A lost device's timeline reads UINT64_MAX: the first sign of the loss, often.)
		if (result == vk::Result::eErrorDeviceLost || counter == UINT64_MAX) DeviceFault::Report();
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

		auto known = m_gpu_tick[lane].load(std::memory_order_acquire);
		while (known < counter && !m_gpu_tick[lane].compare_exchange_weak(known, counter, std::memory_order_release,
		                                                                   std::memory_order_relaxed)) {
		}
	}
=======
	if (m_watched) return;
	LiveCounters::Add(LiveCounters::TimelineQueries);
	if (LiveCensus::g_render) LiveCounters::Add(LiveCounters::TimelineQueriesRender);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	// (A lost device's timeline reads UINT64_MAX: the first sign of the loss, often.)
	if (result == vk::Result::eErrorDeviceLost || counter == UINT64_MAX) DeviceFault::Report();
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	Publish(counter);
>>>>>>> upstream/experiment/perf-40fps-20260926
}

void MasterSemaphore::WaitAll() {
	if (!m_lanes_on) {
		Wait(m_current_tick.load(std::memory_order_acquire) - 1);
		return;
	}
	for (int lane = 0; lane < 2; ++lane)
		if (const auto last = m_last_tick[lane].load(std::memory_order_acquire); last != 0) Wait(last);
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}
	const auto semaphore = HandleOf(tick);

	LiveCensus::Scope census(LiveCensus::GpuWait, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
	                         reinterpret_cast<uint64_t>(__builtin_return_address(1)));
	LiveCensus::WaitScope waiting(LiveCensus::WaitGpu);
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &semaphore;
	wait_info.pValues        = &tick;

	// (A tick that does not come says every 5 s where the timeline is: a stalled GPU, 8 GB sim 10-12. KYTY_STALL_SECONDS:
	// another period - under Windows' 2 s GPU timeout, a hung GPU is reported before the reset.)
	static const uint64_t StallSeconds = [] {
		const char* value = std::getenv("KYTY_STALL_SECONDS");
		const auto  seconds = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
		return seconds != 0 ? seconds : uint64_t {5};
	}();
	for (uint32_t round = 1;; ++round) {
		const auto result = m_graphics.device.waitSemaphores(&wait_info, StallSeconds * 1000000000ull);
		if (result == vk::Result::eSuccess) break;
		if (result == vk::Result::eErrorDeviceLost) {
			// (A diagnostic's state too: what the GPU was in when it was reset, if it still reads.)
			if (const auto dump = DeviceFault::g_stall_dump.load(std::memory_order_acquire); dump != nullptr) {
				uint64_t now = 0;
				(void)m_graphics.device.getSemaphoreCounterValue(semaphore, &now);
				dump(now);
			}
			DeviceFault::Report();
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eTimeout);
		uint64_t now = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(semaphore, &now);
		std::printf("STALL %s timeline: tick %llu not reached after %llu s, at %llu\n", LaneOf(tick) == 0 ? "graphics" : "compute",
		            static_cast<unsigned long long>(tick), static_cast<unsigned long long>(round * StallSeconds),
		            static_cast<unsigned long long>(now));
		std::fflush(stdout);
		if (round == 1) {
			DeviceFault::ReportStall((uint64_t)(static_cast<VkSemaphore>(semaphore)), now, [&](uint64_t semaphore) { // NOLINT
				uint64_t value = 0;
				(void)m_graphics.device.getSemaphoreCounterValue(vk::Semaphore(reinterpret_cast<VkSemaphore>(semaphore)), &value);
				return value;
			});
		}
	}
	// (Watched: the tick waited for is reached, the watcher may not have said so yet.)
	if (m_watched) {
		Publish(tick);
	} else {
		Refresh();
	}
}

} // namespace Libs::Graphics
