#include "graphics/phaseStats.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>

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

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		EXIT("MasterSemaphore: vkGetSemaphoreCounterValue returned %s (%d), gpu_tick=%" PRIu64 "\n",
		     vk::to_string(result).c_str(), static_cast<int>(result),
		     m_gpu_tick.load(std::memory_order_relaxed));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
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

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	PhaseStats::Scope phase {PhaseStats::IsGpuThread() ? PhaseStats::GpuWait
	                                                      : PhaseStats::GuestPhase(PhaseStats::RenderGpuWait, PhaseStats::OtherGpuWait)};
	// Wait in 2 s slices: a timeout is not fatal (a slow shader, a long first-use compile in the
	// driver), it is logged and the wait continues. Only a real error (VK_ERROR_DEVICE_LOST) ends
	// the emulator, and the message now names it instead of "Not implemented".
	constexpr uint64_t slice_ns = 2'000'000'000ull;
	const auto         begin    = std::chrono::steady_clock::now();
	for (uint32_t slices = 0;; ++slices) {
		const auto result = m_graphics.device.waitSemaphores(&wait_info, slice_ns);
		if (result == vk::Result::eSuccess) break;
		uint64_t counter = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
		const auto waited_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		if (result == vk::Result::eTimeout) {
			std::printf("GpuWaitTimeout: tick=%" PRIu64 " gpu_counter=%" PRIu64 " waited=%.0f ms (still waiting)\n",
			            tick, counter, waited_ms);
			std::fflush(stdout);
			continue;
		}
		std::printf("GpuWaitError: vkWaitSemaphores returned %s (%d) tick=%" PRIu64 " gpu_counter=%" PRIu64
		            " waited=%.0f ms timeouts=%u\n",
		            vk::to_string(result).c_str(), static_cast<int>(result), tick, counter, waited_ms, slices);
		std::fflush(stdout);
		EXIT("MasterSemaphore: vkWaitSemaphores returned %s (%d) waiting for tick %" PRIu64
		     " (GPU at %" PRIu64 ")\n",
		     vk::to_string(result).c_str(), static_cast<int>(result), tick, counter);
	}
	Refresh();
}

} // namespace Libs::Graphics
