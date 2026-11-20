#pragma once
// (p7) The "[p7]" line (local/p7-compat.h), at the flip (live-control.h): what the fixes did per frame and, with
// KYTY_P7_ALTERNA=1, the ON / OFF windows of the live fixes (ON OFF OFF ON ...) with their frame and submission times
// (submissions need KYTY_RENDER_STATS=1). Every 5 s (or every window, if longer) while a fix is on or alternating, else
// every 60 s (the hazard and undefined-read counts are diagnostics, kept on NVIDIA too).
// (p7b) KYTY_P7_VENTANA_MS: the window length (default 5000; 500 to 60000). KYTY_P7_DESDE_S: windows that start before
// this second (from the first flip) are not counted (default 0: the loading). The cost is given two ways: ON - OFF over
// every counted window ("COSTE MEDIDO", as p7), and per ABBA block of 4 windows (ON OFF OFF ON: (ON1 + ON4 - OFF2 - OFF3) / 2
// cancels a drift that is linear over the block) as a mean with its standard error ("COSTE POR BLOQUES x +- e ms, n
// bloques"; 95% interval ~ x +- 2e). Also per state: mean fps, frame time p50 / p99 and the 1% low (the mean fps of the
// slowest 1% of frames), from a histogram of every counted frame (0.1 ms bins).
#include "live-census.h"
#include "p7-compat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace P7 {

struct FrameHistogram {
	static constexpr uint32_t Bins = 2000; // 0.1 ms each, the last one everything from 199.9 ms
	std::array<uint32_t, Bins> count {};
	uint64_t                   frames = 0;
	double                     total_ms = 0;
	void Add(double ms) noexcept {
		const auto bin = ms <= 0 ? 0u : ms >= (Bins - 1) * 0.1 ? Bins - 1 : static_cast<uint32_t>(ms * 10.0);
		++count[bin];
		++frames;
		total_ms += ms;
	}
	[[nodiscard]] double Percentile(double p) const noexcept {
		if (frames == 0) return 0;
		const auto want = static_cast<uint64_t>(p * static_cast<double>(frames));
		uint64_t   seen = 0;
		for (uint32_t b = 0; b < Bins; ++b) {
			seen += count[b];
			if (seen > want) return (b + 0.5) * 0.1;
		}
		return (Bins - 0.5) * 0.1;
	}
	// The mean fps of the slowest 1% of frames.
	[[nodiscard]] double OnePercentLow() const noexcept {
		if (frames == 0) return 0;
		const auto want = std::max<uint64_t>(1, frames / 100);
		uint64_t   taken = 0;
		double     sum   = 0;
		for (uint32_t b = Bins; b-- > 0 && taken < want;) {
			const auto n = std::min<uint64_t>(count[b], want - taken);
			taken += n;
			sum += static_cast<double>(n) * (b + 0.5) * 0.1;
		}
		return sum > 0 ? 1000.0 * static_cast<double>(taken) / sum : 0;
	}
};

inline int64_t EnvMs(const char* name, int64_t fallback, int64_t low, int64_t high) noexcept {
	const char* text = std::getenv(name);
	if (text == nullptr || text[0] == '\0') return fallback;
	const long long value = std::strtoll(text, nullptr, 10);
	return value < low ? low : value > high ? high : value;
}

inline void Report(int64_t now_ns) {
	const auto& s = S();
	if (!s.decided || s.mode == 0) return;
	// (The GPU thread applies the live fixes at its start; should it have started before the device was chosen, here.)
	if (!s.live_captured) ApplyAtGpuStart();
	static const int64_t window_ns = EnvMs("KYTY_P7_VENTANA_MS", 5000, 500, 60000) * 1'000'000;
	static const int64_t from_ns   = EnvMs("KYTY_P7_DESDE_S", 0, 0, 3600) * 1'000'000'000;
	// The line's interval, and the current window's start.
	static int64_t  first_ns = 0, last_flip_ns = 0, print_ns = 0, window_start_ns = 0;
	static uint64_t print_tsc = 0, print_frames = 0, print_sub = 0, tracked_frames = 0;
	static uint64_t window_tsc = 0, window_frames = 0, window_sub = 0;
	static uint64_t window = 0; // its index: ON when % 4 is 0 or 3
	static double   acc[2][3] {}; // per state (OFF, ON): counted windows, frame ms, submission ms
	static FrameHistogram histogram[2];
	static double   block[4] {};
	static bool     block_counted[4] {};
	static double   block_sum = 0, block_squares = 0;
	static uint64_t blocks = 0;
	const uint64_t  tsc = __rdtsc();
	const uint64_t  sub = LiveCensus::g_kind_cycles[LiveCensus::Submission];
	if (first_ns == 0) {
		first_ns = last_flip_ns = print_ns = window_start_ns = now_ns;
		print_tsc = window_tsc = tsc;
		print_sub = window_sub = sub;
		return;
	}
	const bool on      = g_window_on.load(std::memory_order_relaxed);
	const bool counted = s.alternate && window_start_ns - first_ns >= from_ns;
	if (counted) histogram[on ? 1 : 0].Add(static_cast<double>(now_ns - last_flip_ns) / 1e6);
	last_flip_ns = now_ns;
	++print_frames;
	++window_frames;
	if (Tracking()) ++tracked_frames; // (the C1/C2 counts are per frame with tracking on)

	// The window: closed, counted, the next one's state applied.
	if (s.alternate && now_ns - window_start_ns >= window_ns && tsc > window_tsc) {
		const double wall_ms  = static_cast<double>(now_ns - window_start_ns) / 1e6;
		const double ns_cycle = wall_ms * 1e6 / static_cast<double>(tsc - window_tsc);
		const double n        = static_cast<double>(window_frames);
		const double frame_ms = wall_ms / n;
		const double sub_ms   = static_cast<double>(sub - window_sub) * ns_cycle / 1e6 / n;
		const auto   position = window % 4;
		if (counted) {
			auto& a = acc[on ? 1 : 0];
			a[0] += 1.0;
			a[1] += frame_ms;
			a[2] += sub_ms;
		}
		block[position]         = frame_ms;
		block_counted[position] = counted;
		if (position == 3 && block_counted[0] && block_counted[1] && block_counted[2] && block_counted[3]) {
			const double d = (block[0] + block[3] - block[1] - block[2]) / 2.0;
			block_sum += d;
			block_squares += d * d;
			++blocks;
		}
		++window;
		window_start_ns = now_ns;
		window_tsc      = tsc;
		window_sub      = sub;
		window_frames   = 0;
		ApplyLive(window % 4 == 0 || window % 4 == 3);
	}

	const int64_t period = std::max<int64_t>(s.alternate || AnyLiveFix() ? 5'000'000'000 : 60'000'000'000,
	                                         s.alternate ? window_ns : 0);
	if (now_ns - print_ns < period || print_frames == 0 || tsc <= print_tsc) return;
	const double n        = static_cast<double>(print_frames);
	const double wall_ms  = static_cast<double>(now_ns - print_ns) / 1e6;
	const double ns_cycle = wall_ms * 1e6 / static_cast<double>(tsc - print_tsc);
	const double frame_ms = wall_ms / n;
	const double sub_ms   = static_cast<double>(sub - print_sub) * ns_cycle / 1e6 / n;
	const auto   take     = [&](int c) { return static_cast<double>(g_counter[c].exchange(0, std::memory_order_relaxed)) / n; };
	const double tn       = static_cast<double>(std::max<uint64_t>(tracked_frames, 1));
	const auto   tracked  = [&](int c) { return static_cast<double>(g_counter[c].exchange(0, std::memory_order_relaxed)) / tn; };
	const double checked = tracked(ChainChecked), hazards = tracked(ChainHazards), chain_barriers = tracked(ChainBarriers);
	const double h_args = tracked(HazArgs), h_raw = tracked(HazRaw), h_waw = tracked(HazWaw), h_war = tracked(HazWar);
	const double h_unknown = tracked(HazUnknown), h_atomic = tracked(HazAtomic), h_conservative = tracked(HazConservative);
	const double drops = tracked(DedupeDrops), unproven = tracked(DedupeUnproven), uncovered = tracked(StageUncovered);
	const double added = tracked(PreciseAdded), global_drops = tracked(GlobalDedupeDrops);
	const double args = take(ArgsBarriers), deferred = take(LabelsDeferred), ordered = take(LabelsOrdered) * n;
	const double immediate = take(LabelsImmediate) * n, cleared = take(ImagesCleared) * n;
	const double undefined = take(UndefinedReads) * n, skipped = take(UndefinedSkipped) * n;
	std::printf("[p7] %.1f s, %.0f frames (%.0f con conteo), %s | C1 cadenas: dispatches encadenados %.1f/frame, HAZARDS %.2f/frame (args %.2f, "
	            "RAW %.2f, WAW %.2f, WAR %.2f, lecturas globales %.2f, solo atomicos %.2f, sin solape probado %.2f), barreras "
	            "puestas %.2f/frame, coste estimado ~%.2f ms/frame (8 us por barrera, medido p7) | C2: repeticiones de escritura "
	            "quitadas %.1f/frame, NO PROBADAS %.2f, ETAPAS NO CUBIERTAS %.2f, barreras anadidas %.2f; barreras globales "
	            "quitadas %.1f/frame | C1 nivel 3 barreras de args %.1f/frame | C5 etiquetas al completar %.1f/frame (ordenadas %.0f, "
	            "inmediatas %.0f) | C7 imagenes limpiadas %.0f | imagenes nuevas leidas sin escribir %.0f, sin poder limpiar %.0f "
	            "| frame %.2f ms, submissions %.2f ms",
	            wall_ms / 1e3, n, static_cast<double>(tracked_frames), s.alternate ? (on ? "ON" : "OFF") : "fijo", checked, hazards, h_args, h_raw, h_waw, h_war,
	            h_unknown, h_atomic, h_conservative, chain_barriers, hazards * 0.008, drops, unproven, uncovered, added,
	            global_drops, args, deferred, ordered, immediate, cleared, undefined, skipped, frame_ms, sub_ms);
	if (s.alternate && acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int w, int i) { return acc[w][i] / acc[w][0]; };
		std::printf(" | ALTERNA ventanas de %.1f s (%.0f OFF / %.0f ON contadas) frame %.2f->%.2f ms, submissions %.2f->%.2f ms: "
		            "COSTE MEDIDO %.3f ms/frame (frame), %.3f ms/frame (submissions)",
		            static_cast<double>(window_ns) / 1e9, acc[0][0], acc[1][0], m(0, 1), m(1, 1), m(0, 2), m(1, 2),
		            m(1, 1) - m(0, 1), m(1, 2) - m(0, 2));
		if (blocks >= 2) {
			const double b    = static_cast<double>(blocks);
			const double mean = block_sum / b;
			const double var  = std::max(0.0, (block_squares - b * mean * mean) / (b - 1.0));
			std::printf(" | COSTE POR BLOQUES %.3f +- %.3f ms/frame (%llu bloques ABBA; IC95 %.3f..%.3f)", mean, std::sqrt(var / b),
			            static_cast<unsigned long long>(blocks), mean - 1.96 * std::sqrt(var / b),
			            mean + 1.96 * std::sqrt(var / b));
		}
		const auto& off_h = histogram[0];
		const auto& on_h  = histogram[1];
		if (off_h.frames != 0 && on_h.frames != 0)
			std::printf(" | OFF: %.1f fps, 1%% low %.1f fps, frametime p50 %.2f p99 %.2f ms; ON: %.1f fps, 1%% low %.1f fps, "
			            "frametime p50 %.2f p99 %.2f ms",
			            1000.0 * static_cast<double>(off_h.frames) / off_h.total_ms, off_h.OnePercentLow(), off_h.Percentile(0.5),
			            off_h.Percentile(0.99), 1000.0 * static_cast<double>(on_h.frames) / on_h.total_ms, on_h.OnePercentLow(),
			            on_h.Percentile(0.5), on_h.Percentile(0.99));
	}
	std::printf("\n");
	std::fflush(stdout);
	print_ns     = now_ns;
	print_tsc    = tsc;
	print_sub    = sub;
	print_frames   = 0;
	tracked_frames = 0;
}

} // namespace P7
