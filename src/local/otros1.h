#ifndef EMULATOR_SRC_LOCAL_OTROS1_H_
#define EMULATOR_SRC_LOCAL_OTROS1_H_
// (otros1) KYTY_OTROS1=1|2 (off by default; needs KYTY_RENDER_STATS=1, whose PM4 clock it refines): where the [render]
// line's "other" goes, by PM4 packet KIND rather than by class. Each packet's time (from its start to the next packet's
// start, as the [render] "pm4" part measures it) is added to its key: the opcode, or for a NOP its R type (the emulator's
// announcements and hooks are NOPs). Also: the walk's work done before each packet (guest commands + feedback
// write-backs, ProcessPm4's preamble), the register pairs the *_REG_INDIRECT blocks carry, and the clock's own cost (two
// TSC reads a packet, calibrated once). A line [otros1] every 5 s: the 16 dearest keys (ms a frame, packets a frame, ns a
// packet), the preamble, the pairs, the clock's cost and the saving (0: no cut is made yet, see INFORME.md; 1 and 2 both
// only measure). Render thread only: plain counters, read by the same thread at the flip.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace Otros1 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_OTROS1");
		return text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
	}();
	return mode;
}

// Keys: 0-255 opcode, 256-319 NOP by R type (KYTY_PM4_R), 320 one-dword filler NOPs (0x80000000), 321 time no packet
// named (a walk's end, a buffer pop, a deferred advance).
inline constexpr uint32_t NopBase = 256, Filler = 320, Unkeyed = 321, Keys = 322;
inline uint64_t           g_cycles[Keys] {};
inline uint64_t           g_packets[Keys] {};
// The walk's preamble (ProcessCommands + WriteBackCompletedFeedback before each walked packet): cycles and calls.
inline uint64_t g_pre_cycles = 0, g_pre_calls = 0;
// Register pairs in *_REG_INDIRECT blocks (0 CX, 1 SH, 2 UC) and their packets.
inline uint64_t g_pairs[3] {}, g_indirect[3] {};

[[nodiscard]] inline uint32_t KeyOf(uint32_t header) {
	if (header == 0x80000000u) return Filler;
	const uint32_t opcode = (header >> 8u) & 0xffu;
	return opcode == 0x10u ? NopBase + ((header >> 2u) & 0x3fu) /* (KYTY_PM4_R: bits 2-7) */ : opcode;
}

// The cycles of one TSC read pair as the clock takes them (measured once, the minimum of 64 tries of 256).
[[nodiscard]] inline double ClockCycles() {
	static const double cycles = [] {
		uint64_t best = ~uint64_t {0};
		for (int t = 0; t < 64; ++t) {
			const uint64_t a = __rdtsc();
			uint64_t       x = 0;
			for (int i = 0; i < 256; ++i) x += __rdtsc();
			const uint64_t b = __rdtsc();
			if (x == 1) std::printf("%llu", static_cast<unsigned long long>(x)); // (keeps the loop)
			best = std::min(best, b - a);
		}
		return static_cast<double>(best) / 256.0;
	}();
	return cycles;
}

// At each flip (render thread): a line every 5 s. `per_cycle_ms`: ms per TSC cycle (the [render] line's).
inline void Report(int64_t now_ns, double per_cycle_ms_hint) {
	if (Mode() == 0) return;
	static int64_t  last_ns = 0;
	static uint64_t frames  = 0, last_tsc = 0;
	static uint64_t last_cycles[Keys] {}, last_packets[Keys] {}, last_pre[2] {}, last_pairs[3] {}, last_indirect[3] {};
	++frames;
	const uint64_t tsc = __rdtsc();
	// (lab1 p7b2, tiron2) KYTY_OTROS1=3: also each frame whose PM4 walk took over KYTY_TIRON2_MS (default 30) ms of
	// the render thread: a [tiron2] line with that frame's walk, preamble and its 6 dearest packet kinds. For the
	// hitches with 90-125 ms of render-thread CPU and nothing in draws, uploads or images (log 20261119-054809-785).
	if (Mode() == 3) {
		static uint64_t f_cycles[Keys] {}, f_pre = 0, f_tsc = 0;
		static const double limit_ms = [] {
			const char* t = std::getenv("KYTY_TIRON2_MS");
			const double v = t != nullptr ? std::atof(t) : 0.0;
			return v > 0.0 ? v : 30.0;
		}();
		static double per_cycle_f = 0.0;
		if (f_tsc != 0 && now_ns > 0) {
			uint64_t walk = 0;
			for (uint32_t k = 0; k < Keys; ++k) walk += g_cycles[k] - f_cycles[k];
			if (per_cycle_f <= 0.0 && last_ns != 0 && tsc > last_tsc && now_ns > last_ns)
				per_cycle_f = static_cast<double>(now_ns - last_ns) / 1e6 / static_cast<double>(tsc - last_tsc);
			const double pc = per_cycle_ms_hint > 0.0 ? per_cycle_ms_hint : per_cycle_f;
			if (pc > 0.0 && static_cast<double>(walk) * pc >= limit_ms) {
				std::array<uint32_t, Keys> o {};
				for (uint32_t k = 0; k < Keys; ++k) o[k] = k;
				std::partial_sort(o.begin(), o.begin() + 6, o.end(),
				                  [&](uint32_t a, uint32_t b) { return g_cycles[a] - f_cycles[a] > g_cycles[b] - f_cycles[b]; });
				std::printf("[tiron2] frame %.1f ms (flip a flip), recorrido PM4 %.1f ms, preambulo (comandos+write-back) %.1f ms | top",
				            static_cast<double>(tsc - f_tsc) * pc, static_cast<double>(walk) * pc,
				            static_cast<double>(g_pre_cycles - f_pre) * pc);
				for (uint32_t i = 0; i < 6; ++i) {
					const uint32_t k = o[i];
					const uint64_t c = g_cycles[k] - f_cycles[k];
					if (c == 0) break;
					if (k < NopBase) std::printf(" op%02x", k);
					else if (k < Filler) std::printf(" nopR%02x", k - NopBase);
					else if (k == Filler) std::printf(" fill");
					else std::printf(" walk_sin_paquete");
					std::printf("=%.1fms", static_cast<double>(c) * pc);
				}
				std::printf("\n");
				std::fflush(stdout);
			}
		}
		std::copy(std::begin(g_cycles), std::end(g_cycles), std::begin(f_cycles));
		f_pre = g_pre_cycles;
		f_tsc = tsc;
	}
	if (last_ns == 0) {
		last_ns  = now_ns;
		last_tsc = tsc;
		frames   = 0;
		std::copy(std::begin(g_cycles), std::end(g_cycles), std::begin(last_cycles));
		std::copy(std::begin(g_packets), std::end(g_packets), std::begin(last_packets));
		return;
	}
	if (now_ns - last_ns < 5'000'000'000 || frames == 0) return;
	const double n          = static_cast<double>(frames);
	const double per_cycle  = per_cycle_ms_hint > 0.0 ? per_cycle_ms_hint
	                                                  : static_cast<double>(now_ns - last_ns) / 1e6 / static_cast<double>(tsc - last_tsc);
	std::array<uint32_t, Keys> order {};
	uint64_t                   all_cycles = 0, all_packets = 0;
	for (uint32_t k = 0; k < Keys; ++k) {
		order[k] = k;
		all_cycles += g_cycles[k] - last_cycles[k];
		all_packets += g_packets[k] - last_packets[k];
	}
	std::partial_sort(order.begin(), order.begin() + 16, order.end(), [&](uint32_t a, uint32_t b) {
		return g_cycles[a] - last_cycles[a] > g_cycles[b] - last_cycles[b];
	});
	std::printf("[otros1] %.1f s, %.0f frames: walk %.2f ms (%.0f packets) a frame | top", static_cast<double>(now_ns - last_ns) / 1e9,
	            n, static_cast<double>(all_cycles) * per_cycle / n, static_cast<double>(all_packets) / n);
	for (uint32_t i = 0; i < 16; ++i) {
		const uint32_t k = order[i];
		const uint64_t c = g_cycles[k] - last_cycles[k], p = g_packets[k] - last_packets[k];
		if (c == 0) break;
		if (k < NopBase) std::printf(" op%02x", k);
		else if (k < Filler) std::printf(" nopR%02x", k - NopBase);
		else if (k == Filler) std::printf(" fill");
		else std::printf(" walk_sin_paquete");
		std::printf("=%.3f/%.0f/%.0fns", static_cast<double>(c) * per_cycle / n, static_cast<double>(p) / n,
		            p != 0 ? static_cast<double>(c) * per_cycle * 1e6 / static_cast<double>(p) : 0.0);
	}
	const double clock_ms = ClockCycles() * static_cast<double>(all_packets) * per_cycle / n;
	std::printf(" | preamble %.3f ms (%.0f) | indirect pairs cx=%.0f/%.0f sh=%.0f/%.0f uc=%.0f/%.0f | clock_cost %.3f ms (%.1f "
	            "cycles a read pair) | ahorro 0.00 ms (mide: ningun recorte activo)\n",
	            static_cast<double>(g_pre_cycles - last_pre[0]) * per_cycle / n, static_cast<double>(g_pre_calls - last_pre[1]) / n,
	            static_cast<double>(g_pairs[0] - last_pairs[0]) / n, static_cast<double>(g_indirect[0] - last_indirect[0]) / n,
	            static_cast<double>(g_pairs[1] - last_pairs[1]) / n, static_cast<double>(g_indirect[1] - last_indirect[1]) / n,
	            static_cast<double>(g_pairs[2] - last_pairs[2]) / n, static_cast<double>(g_indirect[2] - last_indirect[2]) / n,
	            clock_ms, ClockCycles());
	std::fflush(stdout);
	std::copy(std::begin(g_cycles), std::end(g_cycles), std::begin(last_cycles));
	std::copy(std::begin(g_packets), std::end(g_packets), std::begin(last_packets));
	last_pre[0] = g_pre_cycles;
	last_pre[1] = g_pre_calls;
	std::copy(std::begin(g_pairs), std::end(g_pairs), std::begin(last_pairs));
	std::copy(std::begin(g_indirect), std::end(g_indirect), std::begin(last_indirect));
	last_ns  = now_ns;
	last_tsc = tsc;
	frames   = 0;
}

} // namespace Otros1

#endif // EMULATOR_SRC_LOCAL_OTROS1_H_
