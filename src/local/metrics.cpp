#include "local/metrics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Kyty::Metrics {

namespace {

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

double RefreshMs() {
#if defined(_WIN32)
	DEVMODEA mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 1) {
		return 1000.0 / static_cast<double>(mode.dmDisplayFrequency);
	}
#endif
	return 0.0;
}

std::atomic<int64_t> g_pending_input {0};

struct Stats {
	double sum = 0, sq = 0;
	uint32_t n = 0;
	void   Add(double v) {
		  sum += v;
		  sq += v * v;
		  ++n;
	}
	[[nodiscard]] double Mean() const { return n != 0 ? sum / n : 0.0; }
	[[nodiscard]] double Sd() const {
		if (n == 0) return 0.0;
		const double m = Mean();
		return std::sqrt(std::max(sq / n - m * m, 0.0));
	}
};

struct FlipWindow {
	int64_t  start = 0, last = 0;
	uint32_t frames = 0, over17 = 0, over25 = 0;
	double   sum_ms = 0, max_ms = 0;
} g_flips;

struct PresentWindow {
	std::mutex          mutex;
	Stats               flip_to_present, interval;
	int64_t             last_present = 0;
	std::vector<double> latency;
	int64_t             lat_start = 0, lat_first = 0, lat_last = 0;
	uint32_t            lat_presents = 0;
} g_presents;

} // namespace

bool FpsLogOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_FPS_LOG");
		return v != nullptr && *v != '\0' && std::string_view(v) != "0";
	}();
	return on;
}

bool LatencyOn() {
	static const bool on = [] {
		const char* v       = std::getenv("KYTY_LATENCY");
		const bool  enabled = v != nullptr && v[0] == '1';
		if (enabled) {
			std::printf("[lat] on: input (controller read) -> present of that real frame, once a second\n");
			std::fflush(stdout);
		}
		return enabled;
	}();
	return on;
}

void OnPadRead() {
	if (!LatencyOn()) return;
	int64_t none = 0;
	(void)g_pending_input.compare_exchange_strong(none, NowNs(), std::memory_order_relaxed);
}

FrameStamp OnGuestFlip() {
	FrameStamp stamp;
	if (!FpsLogOn() && !LatencyOn()) return stamp;
	const int64_t now = NowNs();
	stamp.flip_ns     = now;
	if (LatencyOn()) stamp.input_ns = g_pending_input.exchange(0, std::memory_order_relaxed);
	if (!FpsLogOn()) return stamp;
	auto& w = g_flips;
	if (w.last != 0) {
		const double ms = static_cast<double>(now - w.last) / 1e6;
		w.frames++;
		w.sum_ms += ms;
		w.max_ms = std::max(w.max_ms, ms);
		if (ms > 17.0) w.over17++;
		if (ms > 25.0) w.over25++;
	}
	w.last = now;
	if (w.start == 0) w.start = now;
	if (now - w.start >= 1000000000 && w.frames > 0) {
		Stats fp, pi;
		{
			std::lock_guard lock(g_presents.mutex);
			fp = g_presents.flip_to_present;
			pi = g_presents.interval;
			g_presents.flip_to_present = {};
			g_presents.interval        = {};
		}
		std::printf("[fps] %u frames: mean %.1f ms, max %.1f; over 17 ms: %u, over 25 ms: %u; flip->present mean %.2f sd "
		            "%.2f ms, present interval mean %.2f sd %.2f ms\n",
		            w.frames, w.sum_ms / w.frames, w.max_ms, w.over17, w.over25, fp.Mean(), fp.Sd(), pi.Mean(), pi.Sd());
		std::fflush(stdout);
		w = {};
		w.start = w.last = now;
	}
	return stamp;
}

void OnPresented(const FrameStamp& stamp, int generated) {
	if (stamp.flip_ns == 0) return;
	const int64_t   now = NowNs();
	auto&           p   = g_presents;
	std::lock_guard lock(p.mutex);
	if (FpsLogOn()) {
		p.flip_to_present.Add(static_cast<double>(now - stamp.flip_ns) / 1e6);
		if (p.last_present != 0) {
			const double ms = static_cast<double>(now - p.last_present) / 1e6;
			if (ms < 1000.0) p.interval.Add(ms);
		}
		p.last_present = now;
	}
	if (!LatencyOn() || stamp.input_ns == 0 || now <= stamp.input_ns) return;
	const double ms = static_cast<double>(now - stamp.input_ns) / 1e6;
	if (p.lat_start == 0) {
		p.lat_start = now;
		p.latency.reserve(512);
	}
	if (ms < 1000.0) p.latency.push_back(ms); // (a pause or a loading screen: not a frame's latency)
	if (p.lat_presents == 0) p.lat_first = now;
	p.lat_last = now;
	p.lat_presents++;
	if (now - p.lat_start < 1000000000) return;
	auto&        v = p.latency;
	const size_t n = v.size();
	if (n > 0) {
		double sum = 0.0, max = 0.0;
		for (const double x: v) {
			sum += x;
			max = std::max(max, x);
		}
		const double mean = sum / static_cast<double>(n);
		const size_t k    = std::min(n - 1, static_cast<size_t>(static_cast<double>(n) * 0.95));
		std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
		const double p95      = v[k];
		const double scanout  = RefreshMs() * 0.5;
		const int    g        = std::max(generated, 0);
		const double interval = p.lat_presents > 1 ? static_cast<double>(p.lat_last - p.lat_first) / 1e6 /
		                                                 static_cast<double>(p.lat_presents - 1)
		                                           : 0.0;
		const double fg = g > 0 ? interval * g / (g + 1.0) : 0.0;
		std::printf("[lat] input->present mean %.1f ms, p95 %.1f, max %.1f; +scanout %.1f; fg +%.1f (n %u); total %.1f, p95 "
		            "%.1f; mode off, fg x%d\n",
		            mean, p95, max, scanout, fg, static_cast<unsigned>(n), mean + scanout + fg, p95 + scanout + fg, g + 1);
		std::fflush(stdout);
	}
	v.clear();
	p.lat_start    = now;
	p.lat_presents = 0;
}

} // namespace Kyty::Metrics
