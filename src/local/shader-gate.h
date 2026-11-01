#pragma once
// (ds-pc-port44) The shader gate (KYTY_SHADER_GATE=1, the PC port's default; unset or 0: off, the emulator as before).
// The start-up's shader work (the recorded warmup: its shaders and every recorded pipeline, and the shader prefetch:
// every program of the game) runs beside the PS5 logo, the intro video and the title screen. Should it still run when
// the game loads its first world (the title's "Continue" / "New game": the first .arealoadlist), that load waits here
// until it is done, the progress line on screen ("Cargando shaders X/Y"), so that nothing is translated or compiled in
// play. KYTY_SHADER_GATE_MAX_S (default 900): the longest wait, a guard against a stuck worker.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <thread>

namespace ShaderGate {

inline std::atomic<bool> warmup_running {false};   // PipelineCache::RunWarmup beside the game
inline std::atomic<bool> prefetch_running {false}; // the shader prefetch: loading or translating
inline std::atomic<bool> waiting {false};          // a world load held at the gate (the HUD keeps its progress line)
inline std::atomic<bool> passed {false};           // the gate was met once with nothing left (or waited out)

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SHADER_GATE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return on;
}

inline bool Busy() {
	return warmup_running.load(std::memory_order_acquire) || prefetch_running.load(std::memory_order_acquire);
}

// The game's world load (`what`: the map's name): held while the start-up's shader work runs.
inline void WaitBeforeWorld(std::string_view what) {
	if (!Enabled() || !Busy()) {
		passed.store(true, std::memory_order_relaxed);
		return;
	}
	static const int max_s = [] {
		const char* value = std::getenv("KYTY_SHADER_GATE_MAX_S");
		return value != nullptr && *value != '\0' ? std::max(1, std::atoi(value)) : 900;
	}();
	const auto begin = std::chrono::steady_clock::now();
	std::printf("Shader gate: %.*s waits for the start-up's shaders and pipelines (warmup %s, prefetch %s)\n",
	            static_cast<int>(what.size()), what.data(), warmup_running.load() ? "running" : "done",
	            prefetch_running.load() ? "running" : "done");
	std::fflush(stdout);
	waiting.store(true, std::memory_order_release);
	while (Busy() && std::chrono::steady_clock::now() - begin < std::chrono::seconds(max_s))
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	waiting.store(false, std::memory_order_release);
	passed.store(true, std::memory_order_relaxed);
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
	std::printf("Shader gate: %.*s went on after %lld ms%s\n", static_cast<int>(what.size()), what.data(),
	            static_cast<long long>(ms), Busy() ? " (KYTY_SHADER_GATE_MAX_S reached: the work goes on beside the game)" : "");
	std::fflush(stdout);
}

} // namespace ShaderGate
