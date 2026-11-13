#pragma once
// (ds-native-lean) What the run log carries by default: what a native PC game's log would, not an emulator's
// per-second and per-object traces. The run log is written through to the disk (and read back by nothing now: the
// metrics panel and the F8 HUD take their numbers from memory, PublishFrames / PublishFg below).
//   KYTY_LOG_VERBOSE=1   every diagnostic line as before: the per-second rate lines ([fsr3], [dlssg], [fsr3-mfg]),
//                        one line per table program and per shader translation ("Table: ...", "Shaders: ...")
//   KYTY_LOG_PERIOD=<s>  the period of the rate lines that stay (default 1 s, every second as before: the emulator's
//                        launcher and its metricas.ps1 window read them each second; always 1 with KYTY_LOG_VERBOSE=1).
//                        The port sets 10 (ds-settings.cpp), 1 with its own metricas.ps1 window (ds-main.cpp).
// The numbers behind the lines are still worked out each second (frame generation's dynamic multiplier decides on
// them): only the printing is rarer.

#include "local/port-live.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace LeanLog {

inline bool Verbose() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LOG_VERBOSE");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

// Seconds between two of a periodic line (>= 1).
inline int Period() {
	static const int seconds = [] {
		const char* value = std::getenv("KYTY_LOG_PERIOD");
		const int   n     = !Verbose() && value != nullptr && *value != '\0' ? std::atoi(value) : 1;
		return n >= 1 ? n : 1;
	}();
	return seconds;
}

// One per periodic line (its own state): true when the line is due (the first call, then once a Period()).
class Every {
public:
	bool Due() {
		const auto now = std::chrono::steady_clock::now();
		if (m_next != std::chrono::steady_clock::time_point {} && now < m_next) return false;
		m_next = now + std::chrono::seconds(Period());
		return true;
	}

private:
	std::chrono::steady_clock::time_point m_next {};
};

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The last second's frames (the [fps] window): mean and longest frame time.
inline void PublishFrames(double mean_ms, double max_ms) {
	PortLive::frames_mean_ms.store(static_cast<float>(mean_ms), std::memory_order_relaxed);
	PortLive::frames_max_ms.store(static_cast<float>(max_ms), std::memory_order_relaxed);
	PortLive::frames_seq.fetch_add(1, std::memory_order_release);
}

// Frame generation's last second: real and shown frame rates; multiplier: the one it runs at (0: not known here).
inline void PublishFg(double real, double shown, int multiplier = 0) {
	PortLive::fg_real_fps.store(static_cast<float>(real), std::memory_order_relaxed);
	PortLive::fg_shown_fps.store(static_cast<float>(shown), std::memory_order_relaxed);
	if (multiplier > 0) PortLive::fg_multiplier.store(multiplier, std::memory_order_relaxed);
	PortLive::fg_at_ns.store(NowNs(), std::memory_order_relaxed);
	PortLive::fg_seq.fetch_add(1, std::memory_order_release);
}

} // namespace LeanLog
