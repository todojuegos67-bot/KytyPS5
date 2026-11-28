#pragma once
// (regs-rapido) KYTY_REGS_RAPIDO (apagado por defecto): los bloques indirectos de registros de Agc
// (SET_CONTEXT_REG_INDIRECT / SET_SH_REG_INDIRECT / SET_UCONFIG_REG_INDIRECT, 5 dwords) entran en el carril de
// registros (KYTY_REG_LANE=1, graphicsRun.cpp) en vez de cortar la racha y pasar por el recorrido general.
// Ghidra (eboot): FUN_01d108a0 / FUN_01d18200 montan 0x22 registros por draw en un bloque de pares, FUN_01d182f0 /
// FUN_01d185d0 encolan el paquete INDIRECT que apunta al bloque, FUN_01d10840 lo limpia tras cada draw: cada draw lleva
// sus INDIRECT entre SET_SH_REG / SET_CONTEXT_REG, y cada uno partia la racha del carril en dos (dos pasos completos del
// recorrido: preambulo, ganchos, comprobaciones de especulacion/trozo/parada, censo).
// Dentro del carril el INDIRECT hace lo mismo que en el recorrido: el preambulo (ordenes del juego + feedback de copias)
// corre ANTES de el igual que alli (el recorrido nunca lo aplaza para un INDIRECT), el observador de estado lo ve, y su
// handler (SyncCommandMemory + pares) es el mismo. Solo se quitan los ganchos que no actuan sobre un INDIRECT con los
// modos por defecto; con cualquier modo que los use (par1, render1=2, port1, pathtracing, sombras=2) no entra.
//   1 encendido.  2 A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = encendido; delta interpolado.
// Apagado: una carga de un static por paquete del carril.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace RegsRapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_REGS_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 1; // (t67) encendido por defecto: -1 ms/fot medido, DIFERENCIAS 0
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[regs-rapido] KYTY_REGS_RAPIDO=2 sin KYTY_TECHO=1: se usa 1 (siempre encendido)\n");
			m = 1;
		}
		if (m != 0) std::printf("[regs-rapido] KYTY_REGS_RAPIDO=%d (%s)\n", m, m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false}; // (=2) esta ventana usa el carril (B)

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Contadores (hilo de render, sumas simples leidas por el informe: relaxed atomics).
enum Why : uint32_t { WOff, WModes, WLen, WCount };
inline constexpr const char* WhyNames[WCount] = {"ventana A", "modos", "longitud"};
inline std::atomic<uint64_t> g_taken {0};
inline std::atomic<uint64_t> g_why[WCount] {};

inline void Note(std::atomic<uint64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); }

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const double t = static_cast<double>(g_taken.exchange(0, std::memory_order_relaxed));
	double       why[WCount];
	for (uint32_t i = 0; i < WCount; ++i) why[i] = static_cast<double>(g_why[i].exchange(0, std::memory_order_relaxed));
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
		struct W { bool b; double busy; };
		static W      win[4096];
		static size_t k = 0;
		if (keep && k < 4096) win[k++] = {was, busy_ms};
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (win[i - 1].b != win[i].b && win[i + 1].b != win[i].b) {
				const double d = win[i].busy - 0.5 * (win[i - 1].busy + win[i + 1].busy);
				sum[win[i].b ? 1 : 0] += d, cnt[win[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	}
	std::printf("[regs-rapido] ventana %s: render ocupado %.2f ms | INDIRECT en el carril %.0f/fot, fuera (",
	            m == 2 ? (was ? "B (carril)" : "A (normal)") : "carril", busy_ms, t / frames);
	for (uint32_t i = 0; i < WCount; ++i) std::printf("%s%s %.0f", i ? ", " : "", WhyNames[i], why[i] / frames);
	std::printf(")");
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace RegsRapido
