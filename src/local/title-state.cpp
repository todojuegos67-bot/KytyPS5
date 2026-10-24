#include "local/title-state.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::VideoOut {

namespace {
int64_t NowMs() noexcept {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
std::atomic<int64_t> g_movie_ms {0};
std::atomic<int64_t> g_first_button_ms {0};
} // namespace

void NoteMovieFrame() noexcept {
	g_movie_ms.store(NowMs(), std::memory_order_relaxed);
}

bool MoviePlaying() noexcept {
	const auto at = g_movie_ms.load(std::memory_order_relaxed);
	return at != 0 && NowMs() - at < 500;
}

void NotePlayerInput() noexcept {
	int64_t expected = 0;
	if (g_first_button_ms.compare_exchange_strong(expected, NowMs(), std::memory_order_relaxed)) {
		std::printf("Player input: first button (frame generation may turn on)\n");
		std::fflush(stdout);
	}
}

bool PlayerButtonSeen() noexcept {
	return g_first_button_ms.load(std::memory_order_relaxed) != 0;
}

bool PlayerInputSeen() noexcept {
	static const int64_t gate_ms = [] {
		const char* value = std::getenv("KYTY_WORLD_GATE_S");
		return value != nullptr ? static_cast<int64_t>(std::atof(value) * 1000.0) : int64_t {0};
	}();
	const auto at = g_first_button_ms.load(std::memory_order_relaxed);
	return at != 0 && NowMs() - at >= gate_ms;
}

bool CinematicPlaying() noexcept {
	return false;
}

bool CutsceneHeld() noexcept {
	return false;
}

int SetDynamicFlipFps(int /*fps*/) noexcept {
	return 0;
}

int ReachFps(double /*target*/, int /*multiplier*/) noexcept {
	return 0;
}

} // namespace Libs::VideoOut
