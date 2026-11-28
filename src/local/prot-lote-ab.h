#ifndef EMULATOR_SRC_LOCAL_PROT_LOTE_AB_H_
#define EMULATOR_SRC_LOCAL_PROT_LOTE_AB_H_
// (prot-lote-ab) KYTY_PROT_LOTE (apagado por defecto; 1 encendido; 2 A/B cada 5 s bajo KYTY_TECHO=1, B = encendido).
// PROT-LOTE (pageManager.cpp, b7b7b5da/cee437ba): el sin-acceso de lo que escribe la GPU (origen read-flush, log
// 20261128-104552: 148 VirtualProtect/fot 0.76 ms/fot en el hilo de render, con TLB shootdown) lo aplica un hilo aparte
// con cola FIFO; el render solo espera en puntos de sincronia (etiqueta, escritura de memoria, espera, comando del juego)
// o si protege esas mismas paginas. Nunca se midio en el PC del usuario: aqui su A/B. El cambio de modo de un hilo solo
// ocurre en un punto de sincronia completo (nada en vuelo, nada pendiente): A y B nunca se mezclan.
// Linea [prot-lote-ab] por ventana: lotes entregados, rangos, esperas (solape / sincronia) en ms/fot, flushes que
// siguen en el render y delta interpolado B-A de render ocupado (patron 0f9920d3).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace ProtLoteAb {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_PROT_LOTE");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) m = 1;
		std::printf("[prot-lote] KYTY_PROT_LOTE=%d: %s\n", m,
		            m == 0   ? "apagado"
		            : m == 1 ? "ENCENDIDO (el sin-acceso de lo que escribe la GPU lo aplica un hilo aparte; el render solo espera "
		                       "en etiquetas/esperas/comandos o si toca esas paginas)"
		                     : "A/B cada 5 s (B = encendido), linea [prot-lote-ab]");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};

[[nodiscard]] inline bool Wanted() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

struct alignas(64) Side {
	std::atomic<uint64_t> batches {0}, ranges {0}, sync_waits {0}, sync_ns {0}, overlap_waits {0}, overlap_ns {0}, flushes {0},
	    flush_ranges {0};
};
inline Side g_side[2];
[[nodiscard]] inline Side& Now() { return g_side[g_ab_on.load(std::memory_order_relaxed) ? 1 : 0]; }
inline void NoteHandOff(uint64_t ranges) {
	if (Mode() == 0) return;
	auto& s = Now();
	s.batches.fetch_add(1, std::memory_order_relaxed);
	s.ranges.fetch_add(ranges, std::memory_order_relaxed);
}
inline void NoteWait(bool overlap, uint64_t ns) {
	if (Mode() == 0) return;
	auto& s = Now();
	(overlap ? s.overlap_waits : s.sync_waits).fetch_add(1, std::memory_order_relaxed);
	(overlap ? s.overlap_ns : s.sync_ns).fetch_add(ns, std::memory_order_relaxed);
}
// A flush the render thread still applies itself (any mode): ranges applied there.
inline void NoteFlush(uint64_t ranges) {
	if (Mode() == 0 || ranges == 0) return;
	auto& s = Now();
	s.flushes.fetch_add(1, std::memory_order_relaxed);
	s.flush_ranges.fetch_add(ranges, std::memory_order_relaxed);
}

// (techo-report, una vez por ventana de 5 s.)
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || !Techo::On() || frames <= 0) return;
	const bool was = g_ab_on.load(std::memory_order_relaxed);
	auto&      s   = g_side[was ? 1 : 0];
	const auto x   = [frames](std::atomic<uint64_t>& a) { return static_cast<double>(a.exchange(0, std::memory_order_relaxed)) / frames; };
	const double b = x(s.batches), r = x(s.ranges), sw = x(s.sync_waits), sns = x(s.sync_ns), ow = x(s.overlap_waits),
	             ons = x(s.overlap_ns), fl = x(s.flushes), fr = x(s.flush_ranges);
	double delta = 0, cnt[2] {};
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
	std::printf("[prot-lote-ab] ventana %s (KYTY_PROT_LOTE=%d): render ocupado %.2f ms, %.1f fps | lotes al hilo aparte %.1f/fot "
	            "(%.1f rangos) | esperas del render: sincronia %.1f/fot %.3f ms, solape %.1f/fot %.3f ms | flushes en el render "
	            "%.1f/fot (%.1f rangos)",
	            m == 2 ? (was ? "B (encendido)" : "A (apagado)") : "encendido", m, busy_ms, fps, b, r, sw, sns / 1e6, ow,
	            ons / 1e6, fl, fr);
	if (m == 2) std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace ProtLoteAb

#endif // EMULATOR_SRC_LOCAL_PROT_LOTE_AB_H_
