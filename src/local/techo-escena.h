#pragma once
// (techo-escena) KYTY_TECHO_ESCENA (apagado por defecto; SOLO PRUEBA, rompe la imagen): TECHO del pase de escena.
// Los draws de las DCB del pase de profundidad + GBuffer + decal NO se traducen ni se graban (se saltan enteros; el
// estado PM4 de alrededor si se procesa). Lo que baja el hilo de render es lo maximo que puede ahorrar un renderer
// nativo de ese pase (que aun tendria que grabar sus draws en Vulkan: el ahorro real es menor).
// DCB (sitios de KickoffSubmitJob 0xa08ae0, [censo-dcb]), mascara KYTY_TECHO_ESCENA_DCB (por defecto 7 = las tres):
//   bit 0: a0937a DCB#0 (RenderDepth: DepthPrePass, ~190 draws)
//   bit 1: a09a7b DCB#0 (~550 draws: decals)
//   bit 2: a09a7b DCB#1 (~810 draws: GBuffer + resto)
//   1 siempre saltados.  2 A/B cada ventana de 5 s de [techo] con delta de render ocupado interpolado.
// Necesita KYTY_TECHO=1 (enciende el censo de DCB que da el sitio). Coste apagado: una carga de un static.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace TechoEscena {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TECHO_ESCENA");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m != 0 && !Techo::On()) {
			std::printf("[techo-escena] KYTY_TECHO_ESCENA=%d sin KYTY_TECHO=1: apagado\n", m);
			m = 0;
		}
		if (m != 0) std::printf("[techo-escena] KYTY_TECHO_ESCENA=%d (%s), SOLO PRUEBA: la escena no se dibuja\n", m,
		                        m == 1 ? "draws de escena saltados" : "A/B cada 5 s, B = saltados");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

[[nodiscard]] inline uint32_t Mask() {
	static const uint32_t mask = [] {
		const char* v = std::getenv("KYTY_TECHO_ESCENA_DCB");
		const int   m = v != nullptr ? std::atoi(v) : 7;
		return static_cast<uint32_t>(m > 0 ? m & 7 : 7);
	}();
	return mask;
}

inline std::atomic<bool>     g_ab_on {false}; // (=2) esta ventana salta (B)
inline std::atomic<uint64_t> g_skipped {0}, g_seen {0};
inline thread_local bool     t_skip = false;

// Clave de sitio del censo (ret | etiqueta << 32): 1 si es una DCB del pase de escena elegida.
[[nodiscard]] inline bool IsScene(uint64_t key) {
	const uint32_t ret = static_cast<uint32_t>(key), tag = static_cast<uint32_t>(key >> 32u);
	if ((tag & 0xF0000000u) != 0xD0000000u) return false; // (solo DCB de KickoffSubmitJob)
	const uint32_t i = tag & 0xffffu, m = Mask();
	return (ret == 0xa0937au && i == 0 && (m & 1u)) || (ret == 0xa09a7bu && i == 0 && (m & 2u)) ||
	       (ret == 0xa09a7bu && i == 1 && (m & 4u));
}

struct Scope {
	explicit Scope(uint64_t key) {
		const int m = Mode();
		t_skip = m != 0 && key != 0 && IsScene(key) && (m == 1 || g_ab_on.load(std::memory_order_relaxed));
	}
	~Scope() { t_skip = false; }
	Scope(const Scope&) = delete;
	Scope& operator=(const Scope&) = delete;
};

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const double s   = static_cast<double>(g_skipped.exchange(0, std::memory_order_relaxed));
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
	std::printf("[techo-escena] ventana %s: render ocupado %.2f ms, %.1f fps | draws de escena saltados %.0f/fot (mascara %u)",
	            m == 2 ? (was ? "B (saltados)" : "A (normal)") : "saltados", busy_ms, fps, s / frames, Mask());
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | TECHO delta interpolado B-A %+.3f ms/fot (negativo = lo que ahorraria)",
		            cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace TechoEscena
