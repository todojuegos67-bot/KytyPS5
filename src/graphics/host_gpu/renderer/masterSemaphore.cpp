#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "live-census.h"
#include "device-fault.h"

#include <cstdlib>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

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
}

void MasterSemaphore::Refresh() {
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
		if (result != vk::Result::eTimeout) {
			// (Not a timeout - a timeout only logs STALL below and keeps waiting: the device was lost, a GPU hang the
			// driver reset (Windows TDR, 2 s), or another wait error. Say which, so the log is not read as a timeout.)
			EXIT("GPU wait failed: %s (%s timeline, tick %llu) - device lost means the GPU hung and was reset\n",
			     vk::to_string(result).c_str(), LaneOf(tick) == 0 ? "graphics" : "compute",
			     static_cast<unsigned long long>(tick));
		}
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
	Refresh();
}

} // namespace Libs::Graphics
