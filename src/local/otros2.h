#ifndef EMULATOR_SRC_LOCAL_OTROS2_H_
#define EMULATOR_SRC_LOCAL_OTROS2_H_
// (otros2) KYTY_OTROS2=1|2 (off by default): two cuts of the PM4 walk's "other" time, chosen from the [otros1] line
// (20261117-040033: preamble 0.84-1.12 ms with 14-17 thousand calls a frame; op9f SET_CONTEXT_REG_INDIRECT 0.45-0.66 ms,
// 1,380-2,000 blocks of ~55 pairs a frame). What is drawn is the same:
//   A) The walk's preamble (ProcessCommands + WriteBackCompletedFeedback) is not run before a walk step that cannot need
//      it: a one-dword filler, a buffer's end, a deferred advance, or a non-predicated register-lane packet (render5's
//      RegLaneBits: register / index / instance / indirect-base fields only, no memory access, no wait). It runs before
//      the next other packet (draw, dispatch, indirect table, NOP, event, wait...), at most 64 steps later, and at the
//      walk's end or suspension: the guest commands and RAIZ's write-backs keep their order and still precede every
//      packet that reads memory, waits on a label or signals one (render5's lane already postpones across the same
//      packets).
//   B) SET_CONTEXT_REG_INDIRECT blocks: the handler's per-pair decisions (fake register, sentinel, unknown, regfile raw /
//      ign / call) are kept per block address with a copy of the block; a block whose bytes are the same again (after
//      SyncCommandMemory, as before) replays the decisions: the same stores and calls with the same values, in order.
// 2 = verification: A runs the preamble anyway and counts the steps where it would have postponed work; B, on a hit,
// replays into the registers, restores them, runs the normal loop and compares the CX/SH/UC register files
// (DISCREPANCIA). With KYTY_RENDER_STATS=1 a line [otros2] every 5 s; with KYTY_OTROS1=1 too, the cuts alternate on and
// off by 5 s windows (only to measure: the saving is walk/preamble/op9f time with them off minus on).
// EVENT_WRITE (op46): only measured here (event types and ns): the barrier dedupe (KYTY_GLOBAL_BARRIER_DEDUPE) already
// drops a barrier with nothing recorded since the previous one.
#include "otros1.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace Otros2 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_OTROS2");
		return text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
	}();
	return mode;
}

// The cuts are on (the mode, and when alternating, an "on" window).
inline std::atomic<bool> g_window_on {true};
[[nodiscard]] inline bool Active() { return Mode() != 0 && g_window_on.load(std::memory_order_relaxed); }
// Timing of B (only with RENDER_STATS + OTROS1, set by Report).
inline std::atomic<bool> g_timed {false};

// A) Steps that may skip the preamble at most this many in a row (render5's RegLaneMaxPackets).
inline constexpr uint32_t MaxSkips = 64;
inline uint64_t           g_pre_skipped = 0, g_pre_flush = 0, g_pre_would_postpone = 0;
// B) hits, misses (compiled), cycles of each (timed), discrepancies (2).
inline uint64_t g_memo_hit = 0, g_memo_miss = 0, g_hit_cycles = 0, g_miss_cycles = 0, g_discrepancias = 0;
// op46 EVENT_WRITE by event type: count and cycles of TriggerEvent (timed).
inline uint64_t g_event_count[64] {}, g_event_cycles[64] {};

// A packet header (or a step) before which the preamble can wait: a filler, or a whole non-predicated lane packet.
[[nodiscard]] inline bool PreambleCanWait(uint32_t header, bool lane_opcode) {
	if (header == 0x80000000u) return true;
	return (header >> 30u) == 3u && (header & 1u) == 0 && lane_opcode;
}

inline void Report(int64_t now_ns) {
	if (Mode() == 0) return;
	static int64_t  last_ns = 0;
	static uint64_t frames  = 0;
	static uint64_t l_skip = 0, l_flush = 0, l_would = 0, l_hit = 0, l_miss = 0, l_hc = 0, l_mc = 0;
	static uint64_t l_walk = 0, l_pre = 0, l_9f = 0, l_ev[64] {}, l_evc[64] {};
	// (alternation) accumulated per window state: frames, walk, preamble, op9f cycles.
	static double acc[2][4] {};
	static uint64_t last_tsc = 0;
	const bool      measure  = Otros1::Mode() != 0;
	g_timed.store(measure, std::memory_order_relaxed);
	++frames;
	const uint64_t tsc  = __rdtsc();
	uint64_t       walk = 0;
	for (uint32_t k = 0; k < Otros1::Keys; ++k) walk += Otros1::g_cycles[k];
	if (last_ns == 0) {
		last_ns = now_ns;
		last_tsc = tsc;
		frames   = 0;
		l_walk   = walk;
		l_pre    = Otros1::g_pre_cycles;
		l_9f     = Otros1::g_cycles[0x9f];
		return;
	}
	if (now_ns - last_ns < 5'000'000'000 || frames == 0) return;
	const double n        = static_cast<double>(frames);
	const double ms_cycle = static_cast<double>(now_ns - last_ns) / 1e6 / static_cast<double>(tsc - last_tsc);
	const bool   was_on   = g_window_on.load(std::memory_order_relaxed);
	const double walk_ms = static_cast<double>(walk - l_walk) * ms_cycle / n;
	const double pre_ms  = static_cast<double>(Otros1::g_pre_cycles - l_pre) * ms_cycle / n;
	const double cx_ms   = static_cast<double>(Otros1::g_cycles[0x9f] - l_9f) * ms_cycle / n;
	if (measure) {
		auto& a = acc[was_on ? 1 : 0];
		a[0] += 1.0;
		a[1] += walk_ms;
		a[2] += pre_ms;
		a[3] += cx_ms;
	}
	const uint64_t hits = g_memo_hit - l_hit, misses = g_memo_miss - l_miss;
	const double   hit_ns  = hits != 0 ? static_cast<double>(g_hit_cycles - l_hc) * ms_cycle * 1e6 / static_cast<double>(hits) : 0.0;
	const double   miss_ns = misses != 0 ? static_cast<double>(g_miss_cycles - l_mc) * ms_cycle * 1e6 / static_cast<double>(misses) : 0.0;
	std::printf("[otros2] %.1f s, %.0f frames, recortes %s (modo %u): walk %.2f ms | A preambulo %.3f ms, saltados %.0f/frame "
	            "(al final %.0f), pospuestos con trabajo (=2) %.0f | B op9f %.3f ms, memo %.0f aciertos / %.0f fallos por frame, "
	            "%.0f ns acierto / %.0f ns fallo, ahorro_estimado %.3f ms, DISCREPANCIA %llu | op46",
	            static_cast<double>(now_ns - last_ns) / 1e9, n, was_on ? "ON" : "OFF", Mode(), walk_ms, pre_ms,
	            static_cast<double>(g_pre_skipped - l_skip) / n, static_cast<double>(g_pre_flush - l_flush) / n,
	            static_cast<double>(g_pre_would_postpone - l_would) / n, cx_ms, static_cast<double>(hits) / n,
	            static_cast<double>(misses) / n, hit_ns, miss_ns,
	            hits != 0 && misses != 0 && miss_ns > hit_ns ? (miss_ns - hit_ns) * static_cast<double>(hits) / n / 1e6 : 0.0,
	            static_cast<unsigned long long>(g_discrepancias));
	for (uint32_t t = 0; t < 64; ++t) {
		const uint64_t c = g_event_count[t] - l_ev[t];
		if (c == 0) continue;
		std::printf(" t%02x=%.0f/%.0fns", t, static_cast<double>(c) / n,
		            static_cast<double>(g_event_cycles[t] - l_evc[t]) * ms_cycle * 1e6 / static_cast<double>(c));
		l_ev[t]  = g_event_count[t];
		l_evc[t] = g_event_cycles[t];
	}
	if (measure && acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int s, int i) { return acc[s][i] / acc[s][0]; };
		std::printf(" | ALTERNA (%.0f ventanas OFF / %.0f ON) walk %.2f->%.2f ms, preambulo %.3f->%.3f, op9f %.3f->%.3f: "
		            "ahorro medido %.3f ms/frame",
		            acc[0][0], acc[1][0], m(0, 1), m(1, 1), m(0, 2), m(1, 2), m(0, 3), m(1, 3), m(0, 1) - m(1, 1));
	}
	std::printf("\n");
	std::fflush(stdout);
	l_skip = g_pre_skipped;
	l_flush = g_pre_flush;
	l_would = g_pre_would_postpone;
	l_hit = g_memo_hit;
	l_miss = g_memo_miss;
	l_hc = g_hit_cycles;
	l_mc = g_miss_cycles;
	l_walk = walk;
	l_pre  = Otros1::g_pre_cycles;
	l_9f   = Otros1::g_cycles[0x9f];
	last_ns  = now_ns;
	last_tsc = tsc;
	frames   = 0;
	// (Measuring: the next window with the cuts the other way.)
	if (measure) g_window_on.store(!was_on, std::memory_order_relaxed);
}

} // namespace Otros2

#endif // EMULATOR_SRC_LOCAL_OTROS2_H_
