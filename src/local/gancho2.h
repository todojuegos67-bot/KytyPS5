#ifndef EMULATOR_SRC_LOCAL_GANCHO2_H_
#define EMULATOR_SRC_LOCAL_GANCHO2_H_
// (gancho2) KYTY_GANCHO2=1|2 (off by default): runs of XPR draws across the game's user-data re-emission.
// The game's XPR renderer (ad8bc0 / ad92f0, Ghidra 01.007.000) calls, per visible submesh, 1d10b70 + DrawIndexIndirect
// (1d0b110) + 1d10840; 1d10840 (-> 1d0edb0) only clears the binder's "emitted" flags (+0xd8, +0x148, +0x1b8), so the
// next draw writes the whole user data again (~5.6 SET_SH_REG a draw) with the same values while the stream (9819d0)
// and the material (a84de0) stay. Those writes cut every run TryNativeXprDraws could take: each draw is a call
// (~1.35-1.57 us a call measured, 1,639 run calls a frame, 860 of them clean continuations; gancho2/lab).
// Here, in the emulator and without touching the game: after a DRAW_INDEX_INDIRECT, a gap of only non-predicated
// SET_SH_REG packets (and one-dword fillers) whose every value is the one the register already holds (the draw state
// observer's shadow, of the current generation) followed by the same DRAW_INDEX_INDIRECT form (only the argument offset
// differs) is a no-op: the run goes on over it (up to 64 draws, as before), the gap's packets are passed over.
// 2 = verification: each gap is also executed by its handlers and the CX/SH/UC register files compared with before
// (DISCREPANCIA if a gap the shadow called a no-op changed a register; the run stops there and the files are restored).
// With KYTY_RENDER_STATS=1 a line [gancho2] every 5 s; with mode 1 the extension alternates 5 s on / 5 s off (only to
// measure: the saving is the submission time with it off minus on). Without RENDER_STATS mode 1 is always on.
#include "live-census.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace Gancho2 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_GANCHO2");
		const auto  m = text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		if (m != 0)
			std::printf("gancho2: KYTY_GANCHO2=%u: XPR draw runs go on over SET_SH_REG gaps that rewrite the same values%s\n", m,
			            m == 2 ? " (verification: gaps executed and register files compared)" : "");
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_window_on {true};
[[nodiscard]] inline bool Active() { return Mode() != 0 && g_window_on.load(std::memory_order_relaxed); }

// Counters (the GPU thread's; read at the flip).
inline uint64_t g_draws = 0;        // draws taken over a gap (each one call less)
inline uint64_t g_packets = 0;      // gap packets passed over (SET_SH_REG and fillers)
inline uint64_t g_refused = 0;      // gaps looked at and not taken (a value changed, another packet, other draw form)
inline uint64_t g_discrepancias = 0;
// (gancho2) Why a run stops / does not go on over a gap: per TryNativeXprDraws call (see Reasons).
enum Reason : uint32_t {
	RCalls, RExitEarly, ROffSpec, ROffState, ROffChunk, ROffStop, ROffCapture, ROffMarker, ROffWindow, REndMax, REndBuffer,
	REndPacket, RDiffUdValue, RDiffUdType, RDiffUdCount, RDiffOther, RNoDraw, RBadSh, RMarkerEnd, RDiscrepancy, RRefusedRun,
	RGapsTaken, Reasons
};
inline constexpr const char* ReasonNames[Reasons] = {
	"llamadas", "salida_temprana", "off_spec", "off_estado", "off_chunk", "off_stop", "off_captura", "off_marcador",
	"off_ventana", "fin_64", "fin_buffer", "fin_otro_paquete", "dif_ud_valor", "dif_ud_tipo", "dif_ud_count", "dif_otro_sh",
	"hueco_sin_draw_igual", "sh_raro", "marcador_final", "discrepancia", "racha_rechazada", "huecos_tomados"};
inline uint64_t g_reason[Reasons] {};
// The first packet that ended a gap (opcode; 256 + NOP R type; 512 a marker of another id) and the SH register whose
// value / type differed.
inline uint64_t g_end_packet[513] {};
inline uint64_t g_diff_offset[0x300] {};
inline void Note(Reason r) { ++g_reason[r]; }

// The draws the caller completes for the last run (consumed dwords are no longer 5 per draw).
inline thread_local uint32_t t_last_draws = 0;

inline void Report(int64_t now_ns) {
	if (Mode() == 0) return;
	static int64_t  last_ns = 0;
	static uint64_t last_tsc = 0, frames = 0, l_draws = 0, l_packets = 0, l_refused = 0, l_sub = 0, l_xpr = 0;
	static double   acc[2][4] {}; // per window state: windows, frame ms, submission ms, xpr0 ms
	const bool      alternate = Mode() == 1;
	++frames;
	const uint64_t tsc = __rdtsc();
	const uint64_t sub = LiveCensus::g_kind_cycles[LiveCensus::Submission], xpr = LiveCensus::g_xpr_cycles[0];
	if (last_ns == 0) {
		last_ns = now_ns;
		last_tsc = tsc;
		frames = 0;
		l_sub = sub;
		l_xpr = xpr;
		return;
	}
	if (now_ns - last_ns < 5'000'000'000 || frames == 0 || tsc <= last_tsc) return;
	const double n        = static_cast<double>(frames);
	const double wall_ms  = static_cast<double>(now_ns - last_ns) / 1e6;
	const double ms_cycle = wall_ms / static_cast<double>(tsc - last_tsc);
	const bool   was_on   = g_window_on.load(std::memory_order_relaxed);
	const double frame_ms = wall_ms / n;
	const double sub_ms   = static_cast<double>(sub - l_sub) * ms_cycle / n;
	const double xpr_ms   = static_cast<double>(xpr - l_xpr) * ms_cycle / n;
	if (alternate) {
		auto& a = acc[was_on ? 1 : 0];
		a[0] += 1.0;
		a[1] += frame_ms;
		a[2] += sub_ms;
		a[3] += xpr_ms;
	}
	std::printf("[gancho2] %.1f s, %.0f frames, %s (modo %u): draws cubiertos %.0f/frame (llamadas evitadas), paquetes evitados "
	            "%.0f/frame, huecos rechazados %.0f/frame | frame %.2f ms, submissions %.2f ms, xpr 0 %.2f ms | DISCREPANCIA %llu",
	            wall_ms / 1e3, n, was_on ? "ON" : "OFF", Mode(), static_cast<double>(g_draws - l_draws) / n,
	            static_cast<double>(g_packets - l_packets) / n, static_cast<double>(g_refused - l_refused) / n, frame_ms,
	            sub_ms, xpr_ms, static_cast<unsigned long long>(g_discrepancias));
	{
		static uint64_t l_reason[Reasons] {}, l_end[513] {}, l_diff[0x300] {};
		std::printf(" | motivos/frame:");
		for (uint32_t r = 0; r < Reasons; ++r)
			if (g_reason[r] != l_reason[r]) std::printf(" %s=%.1f", ReasonNames[r], static_cast<double>(g_reason[r] - l_reason[r]) / n);
		const auto top = [&](const char* name, const uint64_t* now, uint64_t* last, uint32_t size) {
			std::printf(" | %s:", name);
			for (int k = 0; k < 6; ++k) {
				uint32_t best = size;
				uint64_t most = 0;
				for (uint32_t i = 0; i < size; ++i)
					if (now[i] - last[i] > most) {
						most = now[i] - last[i];
						best = i;
					}
				if (best == size) break;
				std::printf(" 0x%x=%.1f", best, static_cast<double>(most) / n);
				last[best] = now[best]; // (shown: out of the next picks)
			}
			for (uint32_t i = 0; i < size; ++i) last[i] = now[i];
		};
		top("fin_otro_paquete (op; 0x1xx NOP R; 0x200 marcador)", g_end_packet, l_end, 513);
		top("dif por registro SH", g_diff_offset, l_diff, 0x300);
		for (uint32_t r = 0; r < Reasons; ++r) l_reason[r] = g_reason[r];
	}
	if (alternate && acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int s, int i) { return acc[s][i] / acc[s][0]; };
		std::printf(" | ALTERNA (%.0f ventanas OFF / %.0f ON) frame %.2f->%.2f ms, submissions %.2f->%.2f ms, xpr 0 %.2f->%.2f ms: "
		            "ahorro medido %.3f ms/frame",
		            acc[0][0], acc[1][0], m(0, 1), m(1, 1), m(0, 2), m(1, 2), m(0, 3), m(1, 3), m(0, 2) - m(1, 2));
	}
	std::printf("\n");
	std::fflush(stdout);
	l_draws = g_draws;
	l_packets = g_packets;
	l_refused = g_refused;
	l_sub = sub;
	l_xpr = xpr;
	last_ns = now_ns;
	last_tsc = tsc;
	frames = 0;
	if (alternate) g_window_on.store(!was_on, std::memory_order_relaxed);
}

} // namespace Gancho2

#endif // EMULATOR_SRC_LOCAL_GANCHO2_H_
