#ifndef EMULATOR_SRC_LOCAL_BLOQUES_RAPIDO_H_
#define EMULATOR_SRC_LOCAL_BLOQUES_RAPIDO_H_
// (bloques-rapido) KYTY_BLOQUES_RAPIDO (apagado por defecto; 1 encendido; 2 A/B cada 5 s bajo KYTY_TECHO=1).
// TableBlocks (table-xpr.inc), atajo TRAD1 de los V# escritos: el fallo "registro" (~230/f) pedia que NINGUN buffer se
// hubiera registrado/desregistrado en todo el espacio desde la ultima obtencion completa (RegistrationEpoch global).
// Con esto basta que no cambiase ninguno en las regiones de 4 MiB del propio rango (RegionRegistrationEpoch, que
// RegisterBuffer/UnregisterBuffer sellan en cada region que toca el buffer): la misma prueba que ya usan los V# de solo
// lectura de TableBlocks y la memo de regiones de ObtainBuffer. Un buffer que pudiera cambiar quien posee el rango lo
// solapa, luego sella su region. El buffer propio sigue comprobado vivo y el mismo (GetBufferIfLive), la cobertura
// GPU por region (Trad1CoverageMoved) igual, y Trad1RenoteWritten vuelve a mirar el registro global despues de su
// drenado (se le pasa el actual: un registro durante el drenado lo rechaza). Rango fuera de las regiones: prueba global.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace BloquesRapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_BLOQUES_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 1; // (t67) encendido por defecto: -1 ms/fot medido, DIFERENCIAS 0
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[bloques-rapido] KYTY_BLOQUES_RAPIDO=2 sin KYTY_TECHO=1: queda en 1\n");
			m = 1;
		}
		if (m != 0) {
			std::printf("[bloques-rapido] KYTY_BLOQUES_RAPIDO=%d (%s): registro por region de 4 MiB en el atajo trad1\n", m,
			            m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};
// Hilo de render: tomados por region (donde el global habria fallado), global ok, y fallos por motivo.
inline uint64_t g_taken = 0, g_global_ok = 0, g_region_moved = 0, g_far = 0;

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Si el registro del rango sigue en pie: el global igual, o (activo) ninguna region del rango sellada despues.
template <class Cache>
[[nodiscard]] inline bool RegistrationHolds(const Cache& cache, uint64_t address, uint64_t size, uint64_t registration,
                                            uint64_t region_size, uint64_t regions) {
	if (cache.RegistrationEpoch() == registration) {
		++g_global_ok;
		return true;
	}
	if (!Active() || size == 0) return false;
	const auto first = address / region_size, last = (address + size - 1) / region_size;
	if (last >= regions || last - first > 64) {
		++g_far;
		return false;
	}
	if (cache.RegisteredSince(address, size, registration)) {
		++g_region_moved;
		return false;
	}
	++g_taken;
	return true;
}

// Ventana de 5 s de [techo] (hilo del informe; los contadores son del hilo de render: lectura aproximada).
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
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
	std::printf("[bloques-rapido] ventana %s: render ocupado %.2f ms, %.1f fps | tomados por region %.0f/fot, global ok %.0f, "
	            "region movida %.0f, fuera %.0f",
	            m == 2 ? (was ? "B (encendido)" : "A (normal)") : "encendido", busy_ms, fps,
	            static_cast<double>(g_taken) / frames, static_cast<double>(g_global_ok) / frames,
	            static_cast<double>(g_region_moved) / frames, static_cast<double>(g_far) / frames);
	if (m == 2) std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	g_taken = g_global_ok = g_region_moved = g_far = 0;
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace BloquesRapido

#endif
