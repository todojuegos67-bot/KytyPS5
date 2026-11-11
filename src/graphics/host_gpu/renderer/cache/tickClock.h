#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TICKCLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TICKCLOCK_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

// When each recent collector tick (one a flip) began, on the steady clock: the collectors count idleness in time, not
// flips. Flips come at the game's frame rate, and that moves: 1800 flips were 2 minutes at an 8 GB laptop's 14 fps, and
// ages scaled by the vblank rate were 4 times as long at a 240 Hz vblank with the game at 45-60 fps (issue #215).
class TickClock {
public:
	// Tick `tick` (they count up from 0) begins now.
	void Begin(uint64_t tick) noexcept {
		m_times[tick % m_times.size()] = std::chrono::steady_clock::now().time_since_epoch().count();
	}

	// The newest tick up to `tick` that began `age` ago or earlier: the ticks before it are at least that old. When
	// even the oldest tick held began later (the last 8192: 34 s at 240 flips a second), that one.
	[[nodiscard]] uint64_t Before(uint64_t tick, std::chrono::steady_clock::duration age) const noexcept {
		const auto horizon = (std::chrono::steady_clock::now() - age).time_since_epoch().count();
		uint64_t   first   = tick - std::min<uint64_t>(tick, m_times.size() - 1);
		uint64_t   last    = tick;
		if (m_times[first % m_times.size()] > horizon) return first;
		// (Their times only grow.)
		while (first < last) {
			const uint64_t middle = first + (last - first + 1) / 2;
			if (m_times[middle % m_times.size()] <= horizon) first = middle;
			else last = middle - 1;
		}
		return first;
	}

private:
	std::array<int64_t, 8192> m_times {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TICKCLOCK_H_
