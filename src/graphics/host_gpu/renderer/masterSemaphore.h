#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <thread>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	// Every tick up to this one is complete, on every lane (KYTY_ASYNC_COMPUTE, lane-sync.h): the smallest lane
	// timeline value among the lanes with submissions pending; all lanes idle: the last tick allocated.
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		if (!m_lanes_on) return m_gpu_tick[0].load(std::memory_order_acquire);
		uint64_t known = UINT64_MAX;
		for (int lane = 0; lane < 2; ++lane) {
			const auto value = m_gpu_tick[lane].load(std::memory_order_acquire);
			if (m_last_tick[lane].load(std::memory_order_acquire) > value && value < known) known = value;
		}
		return known == UINT64_MAX ? m_current_tick.load(std::memory_order_acquire) - 1 : known;
	}
	[[nodiscard]] bool IsFree(uint64_t tick) const noexcept {
		if (KnownGpuTick() >= tick) return true;
		return m_lanes_on && m_gpu_tick[LaneOf(tick)].load(std::memory_order_acquire) >= tick;
	}
	[[nodiscard]] uint64_t NextTick(int lane = 0) noexcept {
		const auto tick = m_current_tick.fetch_add(1, std::memory_order_release);
		if (m_lanes_on) {
			m_lane_of[tick % LaneRing].store(static_cast<uint8_t>(lane), std::memory_order_relaxed);
			m_last_tick[lane].store(tick, std::memory_order_release);
		}
		return tick;
	}
	[[nodiscard]] int LaneOf(uint64_t tick) const noexcept {
		return m_lanes_on ? m_lane_of[tick % LaneRing].load(std::memory_order_relaxed) : 0;
	}
	[[nodiscard]] uint64_t LastTick(int lane) const noexcept { return m_last_tick[lane].load(std::memory_order_acquire); }
	[[nodiscard]] vk::Semaphore Handle(int lane = 0) const noexcept { return m_semaphore[lane]; }
	[[nodiscard]] vk::Semaphore HandleOf(uint64_t tick) const noexcept { return m_semaphore[LaneOf(tick)]; }
	[[nodiscard]] bool LanesOn() const noexcept { return m_lanes_on; }
	// The second lane's timeline (KYTY_ASYNC_COMPUTE with a compute queue).
	void EnableLanes();

	void Refresh();
	void Wait(uint64_t tick);
	// Every tick allocated so far, on every lane.
	void WaitAll();

private:
<<<<<<< HEAD
	static constexpr size_t LaneRing = 1u << 20;
=======
	void Publish(uint64_t counter) noexcept;
	void Watch(const std::stop_token& stop);

>>>>>>> upstream/experiment/perf-40fps-20260926
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore[2] {nullptr, nullptr};
	std::atomic<uint64_t> m_gpu_tick[2] {0, 0};
	std::atomic<uint64_t> m_last_tick[2] {0, 0};
	std::atomic<uint64_t> m_current_tick {1};
<<<<<<< HEAD
	bool                  m_lanes_on = false;
	std::atomic<uint8_t>* m_lane_of = nullptr;
=======
	// Where a timeline read is a system call (Linux' RADV, ~9 us; issue #80: ~1,500 reads a frame at 1-1, most of them
	// the render thread's polls of whether the GPU is still busy), a thread waits for each tick and publishes it, and
	// Refresh reads nothing. KYTY_TIMELINE_WATCHER=0|1 (default: when a read takes 1 us or more).
	bool         m_watched = false;
	std::jthread m_watcher;
>>>>>>> upstream/experiment/perf-40fps-20260926
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
