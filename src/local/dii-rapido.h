#pragma once
// (dii-rapido) KYTY_DII_RAPIDO (apagado por defecto): via rapida de DRAW_INDEX_INDIRECT.
// Ghidra (eboot 01.007.000): el bucle de escena FUN_00ad8bc0 (y FUN_00ad92f0) emite por submalla visible
// FUN_01d10b70 (solo lee el modifier del shader actual, no escribe nada) + DrawIndexIndirect (0x1d0b110, offset de
// args = drawIdx*0x14) + FUN_01d10840 (-> FUN_01d0edb0: solo limpia las banderas "reservado" del binder +0xd8/+0x148/
// +0x1b8). El material (FUN_00a84de0) y el user data (FUN_009819d0, puntero de 2 dw por etapa, en un SET_SH_REG nuevo
// tras el draw) solo se emiten al cambiar de grupo / subgrupo.
// Los DRAW_INDEX_INDIRECT seguidos sin nada en medio ya van juntos (TryNativeXprDraws: una racha = un TableDraw con
// hasta 64 comandos, un vkCmdDrawIndexedIndirect por tramo contiguo). Lo que corta las rachas es el hueco
// SET_SH_REG (+ marcas 'hu') que el binder vuelve a escribir: cuando TODOS sus valores son los que ya tienen los
// registros (user data: valor, tipo y cuenta del procesador; resto del SH: sombra del observador de estado) el hueco
// no cambia nada y la racha sigue sobre el (mismo pipeline, conjunto, buffer de indices, base de args: el draw
// siguiente no pasa por evaluacion, recogida, conjunto, validacion ni emision propias; solo su offset de args).
// Si un solo valor difiere, el draw va por la via normal (fallback).
//   1  siempre.  2  A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = via rapida, linea [dii-rapido]
//      con draws tomados / huecos rechazados por fotograma y delta interpolado B-A del render ocupado.
// Coste apagado: una carga de un static por racha.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"
#include "local/gancho2.h"

namespace DiiRapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DII_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[dii-rapido] KYTY_DII_RAPIDO=2 sin KYTY_TECHO=1: queda en 1 (siempre)\n");
			m = 1;
		}
		if (m != 0)
			std::printf("[dii-rapido] KYTY_DII_RAPIDO=%d (%s)\n", m,
			            m == 1 ? "rachas DRAW_INDEX_INDIRECT sobre huecos sin cambios" : "A/B cada 5 s, B = via rapida");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {true}; // (=2) esta ventana usa la via rapida (B)

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Hilo GPU; leidos en la ventana (aproximados entre hilos, solo informe).
inline std::atomic<uint64_t> g_taken {0};    // draws unidos a la racha sobre un hueco (cada uno, una llamada menos)
inline std::atomic<uint64_t> g_fallback {0}; // huecos mirados y no tomados (un valor cambia, otro paquete...)
inline std::atomic<uint64_t> g_runs {0};     // rachas (llamadas TryNativeXprDraws que llegan a la racha)

inline void NoteRun(uint32_t taken, bool fell_back) {
	g_runs.fetch_add(1, std::memory_order_relaxed);
	if (taken != 0) g_taken.fetch_add(taken, std::memory_order_relaxed);
	if (fell_back) g_fallback.fetch_add(1, std::memory_order_relaxed);
}

// Ventana de 5 s de [techo] (hilo del informe), como TechoEscena::Window.
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const double taken = static_cast<double>(g_taken.exchange(0, std::memory_order_relaxed));
	const double fb    = static_cast<double>(g_fallback.exchange(0, std::memory_order_relaxed));
	const double runs  = static_cast<double>(g_runs.exchange(0, std::memory_order_relaxed));
	const bool   was   = g_ab_on.load(std::memory_order_relaxed);
	double       delta = 0, cnt[2] {};
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
	std::printf("[dii-rapido] ventana %s: render ocupado %.2f ms, %.1f fps | rachas %.0f/fot, tomados %.1f/fot, fallback %.1f/fot",
	            m == 2 ? (was ? "B (rapido)" : "A (normal)") : "rapido", busy_ms, fps, runs / frames, taken / frames,
	            fb / frames);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	// Por que acaba cada racha (Gancho2::g_reason, lo anota TryNativeXprDraws con la via encendida), por fotograma;
	// el paquete que corta el hueco (opcode; 256+R de un NOP; 512 marca 'hu' de otro id) y el registro SH distinto.
	{
		static uint64_t last_reason[Gancho2::Reasons] {}, last_end[513] {}, last_diff[0x300] {};
		std::printf(" | motivos");
		for (uint32_t r = Gancho2::REndMax; r < Gancho2::Reasons; ++r) {
			const uint64_t d = Gancho2::g_reason[r] - last_reason[r];
			last_reason[r]   = Gancho2::g_reason[r];
			if (d != 0) std::printf(" %s=%.1f", Gancho2::ReasonNames[r], static_cast<double>(d) / frames);
		}
		const auto top = [&](const uint64_t* now, uint64_t* last, uint32_t n, const char* name, bool sh) {
			uint32_t best[4] = {0, 0, 0, 0};
			uint64_t val[4]  = {0, 0, 0, 0};
			for (uint32_t i = 0; i < n; ++i) {
				const uint64_t d = now[i] - last[i];
				last[i]          = now[i];
				for (uint32_t j = 0; j < 4; ++j)
					if (d > val[j]) {
						for (uint32_t q = 3; q > j; --q) val[q] = val[q - 1], best[q] = best[q - 1];
						val[j] = d, best[j] = i;
						break;
					}
			}
			std::printf(" | %s", name);
			for (uint32_t j = 0; j < 4 && val[j] != 0; ++j)
				std::printf(sh ? " 0x%x=%.1f" : " %u=%.1f", best[j], static_cast<double>(val[j]) / frames);
		};
		top(Gancho2::g_end_packet, last_end, 513, "corta_paquete(op|256+R|512)", false);
		top(Gancho2::g_diff_offset, last_diff, 0x300, "sh_distinto", true);
	}
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace DiiRapido
