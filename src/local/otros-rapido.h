#pragma once
// (otros-rapido) KYTY_OTROS_RAPIDO (apagado por defecto): el paso del recorrido PM4 (graphicsRun.cpp, bucle de
// ProcessPm4) mira primero si el preambulo tiene algo que hacer (Juego3 apagado, sin ordenes del juego pendientes, sin
// feedback de copias pendiente: la misma condicion con la que preamble() vuelve sin hacer nada). Si no lo tiene, se salta
// la busqueda de OTROS2 (mirar el paquete siguiente, PreambleCanWait, RegLaneOpcode, turno) y no aplaza nada: correr ahi
// el preambulo seria no hacer nada, asi que el resultado es el mismo que el camino normal (pre_skips vuelve a 0 como tras
// preamble()). Con trabajo pendiente, el camino de siempre.
//   1 encendido.  2 A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = encendido; delta interpolado.
// Con KYTY_TECHO=1 la linea [otros-rapido] da ademas el coste estimado de la propia medicion del recorrido (censo-pm4 +
// reloj pm4 de RENDER_STATS: ~4 rdtsc y ~4 sumas atomicas por paso), calibrado en esta maquina, para separar lo que es
// medicion de lo que existe al jugar normal.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

#include "local/reent.h"
#include "local/techo.h"

namespace OtrosRapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_OTROS_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 1; // (t69) encendido por defecto: medido sin fallos, DIFERENCIAS 0
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[otros-rapido] KYTY_OTROS_RAPIDO=2 sin KYTY_TECHO=1: se usa 1 (siempre encendido)\n");
			m = 1;
		}
		if (m != 0) std::printf("[otros-rapido] KYTY_OTROS_RAPIDO=%d (%s)\n", m, m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Contadores (hilo de render; solo con KYTY_TECHO=1).
// (reentrante) celdas por hilo (Reent), sumadas por el informe: 0 pasos del recorrido con preambulo posible, 1 de ellos
// sin nada pendiente (los que el atajo toma).
struct Tag;
inline void Note(bool idle) {
	if (!Techo::On()) return;
	auto& c = Reent::Local<Tag, 2>();
	c.Add(0);
	if (idle) c.Add(1);
}

// Coste de un par rdtsc + suma atomica en esta maquina (ns), medido una vez.
[[nodiscard]] inline double PairNs() {
	static const double ns = [] {
		std::atomic<uint64_t> cell {0};
		uint64_t              last = __rdtsc();
		const auto            t0   = std::chrono::steady_clock::now();
		constexpr int         N    = 200000;
		for (int i = 0; i < N; ++i) {
			const uint64_t now = __rdtsc();
			cell.fetch_add(now - last, std::memory_order_relaxed);
			last = now;
		}
		const auto t1 = std::chrono::steady_clock::now();
		return std::chrono::duration<double, std::nano>(t1 - t0).count() / N + 0.0 * static_cast<double>(cell.load());
	}();
	return ns;
}

inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (!Techo::On() || frames <= 0) return;
	uint64_t     cells[2];
	Reent::Take<Tag, 2>(cells);
	const double steps = static_cast<double>(cells[0]) / frames;
	const double idle  = static_cast<double>(cells[1]) / frames;
	const double pair  = PairNs();
	const bool   was   = g_ab_on.load(std::memory_order_relaxed);
	double       delta = 0, cnt[2] {};
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
	std::printf("[otros-rapido] ventana %s: render ocupado %.2f ms | pasos del recorrido %.0f/fot, sin nada pendiente %.0f/fot "
	            "| medicion: par rdtsc+atomica %.1f ns, coste estimado censo-pm4+reloj pm4 (~4 pares/paso) %.2f ms/fot",
	            m == 2 ? (was ? "B (atajo)" : "A (normal)") : (m == 1 ? "atajo" : "apagado"), busy_ms, steps, idle, pair,
	            steps * 4.0 * pair / 1e6);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace OtrosRapido
