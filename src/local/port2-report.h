#ifndef EMULATOR_SRC_LOCAL_PORT2_REPORT_H_
#define EMULATOR_SRC_LOCAL_PORT2_REPORT_H_
// (port2) The [port2] line, every 5 s at the flip (local/port2.h). Mode 1 flips the ON / OFF window here: ON writes
// markers and applies flows, OFF writes none (the stream is today's). Submission and frame times per window come from
// the live census (KYTY_RENDER_STATS=1); "AHORRO MEDIDO" = submissions OFF - ON, averaged over the windows so far.
#include "live-census.h"
#include "port2.h"

#include <cstdint>
#include <cstdio>
#include <x86intrin.h>

namespace Port2 {

inline void Report(int64_t now_ns) {
	if (Mode() == 0) return;
	static int64_t  last_ns = 0;
	static uint64_t last_tsc = 0, frames = 0, l_sub = 0;
	static double   acc[2][3] {}; // per window state: windows, frame ms, submission ms
	++frames;
	const uint64_t tsc = __rdtsc();
	const uint64_t sub = LiveCensus::g_kind_cycles[LiveCensus::Submission];
	if (last_ns == 0) {
		last_ns  = now_ns;
		last_tsc = tsc;
		frames   = 0;
		l_sub    = sub;
		return;
	}
	if (now_ns - last_ns < 5'000'000'000 || frames == 0 || tsc <= last_tsc) return;
	t_rec.Flush(); // (this thread's; the game threads flush every 128 events)
	const double n        = static_cast<double>(frames);
	const double wall_ms  = static_cast<double>(now_ns - last_ns) / 1e6;
	const double ns_cycle = wall_ms * 1e6 / static_cast<double>(tsc - last_tsc);
	const bool   was_on   = Mode() == 2 || g_window_on.load(std::memory_order_relaxed);
	const double frame_ms = wall_ms / n;
	const double sub_ms   = static_cast<double>(sub - l_sub) * ns_cycle / 1e6 / n;
	if (Mode() == 1) {
		auto& a = acc[was_on ? 1 : 0];
		a[0] += 1.0;
		a[1] += frame_ms;
		a[2] += sub_ms;
	}
	const auto take = [](std::atomic<uint64_t>& v) { return static_cast<double>(v.exchange(0, std::memory_order_relaxed)); };
	const double markers = take(g_rec_markers), sealed = take(g_rec_sealed), empty = take(g_rec_empty);
	const double rec_packets = take(g_rec_packets), rec_cycles = take(g_rec_cycles), rec_timed = take(g_rec_timed);
	const double applied = take(g_gpu_applied), packets = take(g_gpu_packets), dw = take(g_gpu_dw), gpu_empty = take(g_gpu_empty);
	const double gpu_cycles = take(g_gpu_cycles), gpu_timed = take(g_gpu_timed);
	std::printf("[port2] %.1f s, %.0f frames, %s (modo %u%s) | flujo (hilos del juego)/frame: marcadores %.0f, sellados %.0f "
	            "(%.0f paquetes), huecos vacios %.0f, coste %.0f ns/draw | GPU/frame: flujos aplicados %.0f, paquetes PM4 %s %.0f "
	            "(%.0f dw), marcadores sin sellar %.0f, aplicar %.0f ns/flujo | operaciones nativas/frame:",
	            wall_ms / 1e3, n, was_on ? "ON" : "OFF", Mode(),
	            g_eboot.load(std::memory_order_relaxed) == 1 ? "" : ", eboot NO verificado: apagado", markers / n, sealed / n,
	            rec_packets / n, empty / n, rec_timed > 0 ? rec_cycles / rec_timed * ns_cycle : 0.0, applied / n,
	            Mode() == 2 ? "cubiertos (verificados, recorridos)" : "evitados", packets / n, dw / n, gpu_empty / n,
	            gpu_timed > 0 ? gpu_cycles / gpu_timed * ns_cycle : 0.0);
	for (uint32_t t = 0; t < Types; ++t) std::printf(" %s=%.0f", TypeNames[t], take(g_gpu_type[t]) / n);
	std::printf(" | caidas/frame:");
	bool any = false;
	for (uint32_t f = 0; f < Falls; ++f) {
		const double v = take(g_rec_fall[f]);
		if (v != 0) {
			std::printf(" %s=%.1f", FallNames[f], v / n);
			any = true;
		}
	}
	for (uint32_t f = 0; f < GFalls; ++f) {
		const double v = take(g_gpu_fall[f]);
		if (v != 0) {
			std::printf(" %s=%.1f", GFallNames[f], v / n);
			any = true;
		}
	}
	if (!any) std::printf(" ninguna");
	{
		// The packets that kept spans to the walk (otro_paquete): the most frequent opcodes (256: not type 3).
		uint64_t now[257];
		for (uint32_t i = 0; i < 257; ++i) now[i] = g_rec_op[i].exchange(0, std::memory_order_relaxed);
		std::printf(" | otro_paquete por opcode:");
		for (int k = 0; k < 5; ++k) {
			uint32_t best = 257;
			uint64_t most = 0;
			for (uint32_t i = 0; i < 257; ++i)
				if (now[i] > most) {
					most = now[i];
					best = i;
				}
			if (best == 257) break;
			std::printf(" 0x%x=%.1f", best, static_cast<double>(most) / n);
			now[best] = 0;
		}
	}
	std::printf(" | frame %.2f ms, submissions %.2f ms", frame_ms, sub_ms);
	if (Mode() == 1 && acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int s, int i) { return acc[s][i] / acc[s][0]; };
		std::printf(" | ALTERNA (%.0f ventanas OFF / %.0f ON) frame %.2f->%.2f ms, submissions %.2f->%.2f ms: AHORRO MEDIDO %.3f ms/frame "
		            "(submissions), %.3f ms/frame (frame)",
		            acc[0][0], acc[1][0], m(0, 1), m(1, 1), m(0, 2), m(1, 2), m(0, 2) - m(1, 2), m(0, 1) - m(1, 1));
	}
	if (Mode() == 2) {
		std::printf(" | VERIFICA comparados %.0f/frame, DISCREPANCIA %llu (por campo:", take(g_verified) / n,
		            static_cast<unsigned long long>(g_discrepancias.load(std::memory_order_relaxed)));
		for (uint32_t f = 0; f < Fields; ++f)
			std::printf(" %s=%llu", FieldNames[f], static_cast<unsigned long long>(g_field[f].load(std::memory_order_relaxed)));
		std::printf(")");
	}
	std::printf("\n");
	std::fflush(stdout);
	l_sub    = sub;
	last_ns  = now_ns;
	last_tsc = tsc;
	frames   = 0;
	if (Mode() == 1) g_window_on.store(!was_on, std::memory_order_relaxed);
}

} // namespace Port2

#endif // EMULATOR_SRC_LOCAL_PORT2_REPORT_H_
