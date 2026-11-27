#pragma once
// (ti1) KYTY_TABLA_INDIRECTO: los DRAW_INDIRECT no indexados (cabecera 0xc0032400, argumentos que escribe la GPU) por el
// camino de tabla, como DRAW_INDEX_AUTO (TryNativeAutoDraw): el mismo par/variante/estado de la tabla y, en vez de
// draw(cuenta) con registros, drawIndirect sobre el bufer de los argumentos (lo que hace el camino normal con
// TryGpuIndirectDraw -> DrawAuto -> drawIndirect). Sin lectura en la CPU, sin esperar a la GPU.
// Medido antes (log 20261127-220131-226, [techo-normal]): 15.1 draws/fot "sin intento" (todos DRAW_INDIRECT) a ~23 us
// cada uno en el camino normal frente a ~1.7-2.2 us en la tabla.
//   0 (por defecto) apagado: como antes.
//   1 encendido.
//   2 A/B en el mismo proceso: alterna cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1) y da el delta de render
//     ocupado interpolado con las ventanas vecinas (deriva de escena quitada).
//   3 validacion: encendido salvo 1 de cada 4 frames por el camino normal (cualquier diferencia de imagen parpadea a
//     15 Hz en un sitio quieto) y la linea da el coste por draw de cada camino.
// Coste apagado: una carga de un static.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace TablaIndirecto {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TABLA_INDIRECTO");
		const int   m = v != nullptr ? std::atoi(v) : 0;
		std::printf("[tabla-indirecto] KYTY_TABLA_INDIRECTO=%d (%s)\n", m,
		            m == 1   ? "DRAW_INDIRECT no indexado por la tabla"
		            : m == 2 ? "A/B cada 5 s con KYTY_TECHO=1 (B = por la tabla)"
		            : m == 3 ? "validacion: por la tabla salvo 1 de cada 4 frames"
		                     : "apagado");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool>     g_ab_on {false};   // (=2) esta ventana por la tabla (B)
inline std::atomic<uint32_t> g_frame {0};       // (=3)
inline std::atomic<uint64_t> g_table {0}, g_table_ns {0}, g_normal {0}, g_normal_ns {0}, g_refused {0};

inline void OnFrame(uint32_t frame) {
	if (Mode() == 3) g_frame.store(frame, std::memory_order_relaxed);
}

// Este draw intenta la tabla.
[[nodiscard]] inline bool Try() {
	const int m = Mode();
	if (m == 0) return false;
	if (m == 1) return true;
	if (m == 2) return g_ab_on.load(std::memory_order_relaxed);
	return (g_frame.load(std::memory_order_relaxed) & 3u) != 0;
}

[[nodiscard]] inline uint64_t Now() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const double t = static_cast<double>(g_table.exchange(0, std::memory_order_relaxed));
	const double tn = static_cast<double>(g_table_ns.exchange(0, std::memory_order_relaxed));
	const double n = static_cast<double>(g_normal.exchange(0, std::memory_order_relaxed));
	const double nn = static_cast<double>(g_normal_ns.exchange(0, std::memory_order_relaxed));
	const double r = static_cast<double>(g_refused.exchange(0, std::memory_order_relaxed));
	const bool   was = g_ab_on.load(std::memory_order_relaxed);
	double       delta = 0, cnt[2] {};
	if (m == 2) {
		struct W { bool b; double busy; };
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
	std::printf("[tabla-indirecto] ventana %s: render ocupado %.2f ms, %.1f fps | DRAW_INDIRECT/fot: tabla %.1f (%.2f us), "
	            "normal %.1f (%.2f us), tabla rechazo %.1f",
	            m == 2 ? (was ? "B (tabla)" : "A (normal)") : m == 3 ? "validacion" : "tabla", busy_ms, fps, t / frames,
	            t > 0 ? tn / t / 1e3 : 0.0, n / frames, n > 0 ? nn / n / 1e3 : 0.0, r / frames);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = la tabla ahorra)", cnt[0],
		            cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace TablaIndirecto
