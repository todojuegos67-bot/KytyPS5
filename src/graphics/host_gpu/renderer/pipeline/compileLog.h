#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPILELOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPILELOG_H_

// "Compile (5 s)": every vkCreate*Pipelines the render path makes, summed per 5 s window with the
// ten most expensive ones listed (start time since boot, pipeline key, SPIR-V words, create flags,
// milliseconds, driver-cache hit from VK_EXT_pipeline_creation_feedback). It answers "which calls
// held Thread_Gpu for 20 s" without cross-referencing several other lines. Printed when the first
// compile after the window ended finishes, so a window with one 20 s compile is printed right after it.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace Libs::Graphics::CompileLog {

struct Event {
	double   start_ms = 0.0; // since the first call
	double   ms       = 0.0;
	uint64_t id       = 0;
	uint64_t words    = 0;
	uint32_t flags    = 0;
	int      hit      = -1; // 1 driver-cache hit, 0 miss, -1 unknown
	bool     graphics = false;
	bool     async    = false; // background thread: does not block a frame
};

inline double NowMs() {
	static const auto t0 = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// `ms` is the duration of the call that just ended.
inline void Note(Event e) {
	static std::mutex         mutex;
	static std::vector<Event> window;
	static double             window_start = NowMs();
	const double              now          = NowMs();
	e.start_ms                             = now - e.ms;
	std::lock_guard lock(mutex);
	window.push_back(e);
	if (now - window_start < 5000.0) return;
	const double wall = now - window_start;
	double       sync_ms = 0.0, async_ms = 0.0;
	size_t       hits = 0, misses = 0, unknown = 0, gfx = 0;
	for (const auto& w: window) {
		(w.async ? async_ms : sync_ms) += w.ms;
		if (w.graphics) ++gfx;
		(w.hit == 1 ? hits : w.hit == 0 ? misses : unknown)++;
	}
	std::printf("Compile (5 s): %zu pipeline creations (%zu graphics), %.0f ms blocking a frame = %.0f%% of %.0f ms wall"
	            " (+%.0f ms on background threads) | driver cache: %zu hit, %zu miss, %zu unknown\n",
	            window.size(), gfx, sync_ms, 100.0 * sync_ms / wall, wall, async_ms, hits, misses, unknown);
	std::sort(window.begin(), window.end(), [](const Event& a, const Event& b) { return a.ms > b.ms; });
	for (size_t i = 0; i < window.size() && i < 10 && window[i].ms >= 20.0; ++i) {
		const auto& w = window[i];
		std::printf("  Compile top %zu: t=%.1f-%.1f s %s key=%016" PRIx64 " words=%" PRIu64 " flags=0x%x %.0f ms cache=%s%s\n",
		            i + 1, w.start_ms / 1000.0, (w.start_ms + w.ms) / 1000.0, w.graphics ? "gfx" : "cs ", w.id, w.words,
		            w.flags, w.ms, w.hit == 1 ? "hit" : w.hit == 0 ? "MISS" : "?", w.async ? " (background)" : "");
	}
	std::fflush(stdout);
	window.clear();
	window_start = now;
}

} // namespace Libs::Graphics::CompileLog

#endif
