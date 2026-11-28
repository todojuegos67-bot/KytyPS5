#pragma once
// (emision-draws) KYTY_EMISION_RAPIDA (apagado por defecto): en un draw de tabla (table-xpr.inc TableDraw, hilo de
// render, Role 0, sin especulacion) no se repite vkCmdPushConstants cuando el draw anterior de la tabla dejo en el
// mismo command buffer (misma generacion grafica, mismo pipeline layout) EXACTAMENTE los mismos bytes y entre los dos
// no paso ningun paquete PM4 que no sea de registros/indices (contador t_serial del observador: un dispatch, NOP,
// copia o cualquier otro paquete lo mueve; otro draw tambien). Tampoco con hwrt (cursor) ni render1.
// Ghidra FUN_00ad8bc0: los draws seguidos del mismo subgrupo solo cambian el offset de args (drawIdx*0x14): user data y
// bloque iguales -> push iguales.
//   1 encendido. 2 A/B cada 5 s bajo KYTY_TECHO=1 (B = encendido), linea [emision-draws].
// Con KYTY_TECHO=1 (cualquier modo) mide ns por llamada de las fases 3 (bufs), 9 y 4 (emision), continuadas
// (TableContinue) frente a nuevas, y cuantas llamadas traen user data igual a la anterior.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "local/techo.h"

namespace EmisionDraws {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_EMISION_RAPIDA");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[emision-draws] KYTY_EMISION_RAPIDA=2 sin KYTY_TECHO=1: queda en 1\n");
			m = 1;
		}
		if (m != 0) {
			std::printf("[emision-draws] KYTY_EMISION_RAPIDA=%d (%s): push constants iguales no se repiten\n", m,
			            m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Paquetes PM4 no de registros vistos por el observador (hilo de render; draw-state-observer.h lo mueve).
inline thread_local uint64_t t_serial = 0;
// La llamada en curso viene de TableContinue.
inline thread_local bool t_cont = false;

// Lo ultimo empujado por un draw de tabla.
struct Last {
	uint64_t command = 0, generation = 0, layout = 0, serial = UINT64_MAX;
	uint32_t dwords  = 0;
	uint32_t data[32] {};
};
inline thread_local Last t_last;
inline thread_local uint32_t t_last_ud[2][64] {};
inline thread_local uint32_t t_last_ud_n[2] {};

// Contadores (hilo de render): [0 nueva, 1 continuada]
enum { F3 = 0, F9, F4, Fases };
inline uint64_t g_cyc[2][Fases] {}, g_calls[2] {}, g_ud_same[2] {}, g_push_same[2] {}, g_push_skipped[2] {},
    g_push_diff[2] {}, g_push_foreign[2] {};

// Si el push de este draw puede saltarse (y anota el motivo).
[[nodiscard]] inline bool CanSkip(uint64_t command, uint64_t generation, uint64_t layout, const uint32_t* data, uint32_t dwords,
                                  bool eligible) {
	const int c = t_cont ? 1 : 0;
	if (!eligible) return false;
	const auto& l = t_last;
	if (l.command != command || l.generation != generation || l.layout != layout || l.serial + 1 != t_serial) {
		++g_push_foreign[c];
		return false;
	}
	if (l.dwords != dwords || std::memcmp(l.data, data, dwords * 4u) != 0) {
		++g_push_diff[c];
		return false;
	}
	++g_push_same[c];
	if (!Active()) return false;
	++g_push_skipped[c];
	return true;
}

inline void NotePush(uint64_t command, uint64_t generation, uint64_t layout, const uint32_t* data, uint32_t dwords, bool eligible) {
	auto& l = t_last;
	if (!eligible || dwords > 32) {
		l.serial = UINT64_MAX - 1;
		return;
	}
	l.command    = command;
	l.generation = generation;
	l.layout     = layout;
	l.serial     = t_serial;
	l.dwords     = dwords;
	std::memcpy(l.data, data, dwords * 4u);
}

// User data de la llamada igual a la de la anterior (solo con KYTY_TECHO).
template <class Span>
inline void NoteUserData(const Span* ud, uint32_t stages) {
	const int c    = t_cont ? 1 : 0;
	bool      same = true;
	for (uint32_t st = 0; st < 2; ++st) {
		const uint32_t n = st < stages ? static_cast<uint32_t>(ud[st].size() < 64 ? ud[st].size() : 64) : 0;
		if (n != t_last_ud_n[st] || std::memcmp(t_last_ud[st], n != 0 ? ud[st].data() : t_last_ud[st], n * 4u) != 0) same = false;
		if (n != 0) std::memcpy(t_last_ud[st], ud[st].data(), n * 4u);
		t_last_ud_n[st] = n;
	}
	if (same) ++g_ud_same[c];
}

// Ventana de 5 s de [techo].
inline void Window(double busy_ms, double frames, bool keep, double ms_cyc) {
	if (!Techo::On() || frames <= 0) return;
	const int  m   = Mode();
	const bool was = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
		struct W {
			bool   b;
			double busy;
		};
		static W      w[4096];
		static size_t k = 0;
		if (keep && k < 4096) w[k++] = {was, busy_ms};
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (w[i - 1].b != w[i].b && w[i + 1].b != w[i].b) {
				const double d = w[i].busy - 0.5 * (w[i - 1].busy + w[i + 1].busy);
				sum[w[i].b ? 1 : 0] += d, cnt[w[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	}
	const auto ns = [&](int c, int f) {
		return g_calls[c] != 0 ? static_cast<double>(g_cyc[c][f]) * ms_cyc * 1e6 / static_cast<double>(g_calls[c]) : 0.0;
	};
	const auto pf = [&](uint64_t v) { return static_cast<double>(v) / frames; };
	std::printf("[emision-draws] ventana %s: render ocupado %.2f ms | nuevas %.0f/fot ns f3 %.0f f9 %.0f f4 %.0f | continuadas "
	            "%.0f/fot ns f3 %.0f f9 %.0f f4 %.0f | ud igual nueva %.0f cont %.0f | push igual %.0f/%.0f saltado %.0f/%.0f "
	            "distinto %.0f/%.0f ajeno %.0f/%.0f (nueva/cont)",
	            m == 2 ? (was ? "B (encendido)" : "A (normal)") : m == 1 ? "encendido" : "solo mide", busy_ms, pf(g_calls[0]),
	            ns(0, F3), ns(0, F9), ns(0, F4), pf(g_calls[1]), ns(1, F3), ns(1, F9), ns(1, F4), pf(g_ud_same[0]),
	            pf(g_ud_same[1]), pf(g_push_same[0]), pf(g_push_same[1]), pf(g_push_skipped[0]), pf(g_push_skipped[1]),
	            pf(g_push_diff[0]), pf(g_push_diff[1]), pf(g_push_foreign[0]), pf(g_push_foreign[1]));
	if (m == 2) std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	std::memset(g_cyc, 0, sizeof(g_cyc));
	for (int c = 0; c < 2; ++c)
		g_calls[c] = g_ud_same[c] = g_push_same[c] = g_push_skipped[c] = g_push_diff[c] = g_push_foreign[c] = 0;
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace EmisionDraws
