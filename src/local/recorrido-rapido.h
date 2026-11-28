#pragma once
// (recorrido-rapido) KYTY_RECORRIDO_RAPIDO (apagado por defecto): el bucle del recorrido PM4 (CommandProcessor::ProcessPm4,
// graphicsRun.cpp) con menos trabajo por paquete en el juego normal, mismo orden de efectos.
//   A) Preambulo (ordenes del juego + write-back del feedback de copias entre paquetes): la puerta en linea pasaba con
//      solo haber una instantanea o escritura nativa EN COLA (FeedbackPending), aunque su tick aun no hubiera terminado;
//      entonces se llamaba a ProcessCommands + WriteBackCompletedFeedback, que no hacian nada (el primer elemento espera
//      su tick: BufferCache::WriteBackCompletedFeedback / WriteBackNativeWrites vuelven sin efecto). Encendido: la puerta
//      mira tambien el tick del primero (FeedbackReady); sin ordenes y sin nada terminado, nada que hacer, como antes.
//   B) Los modos raros que el recorrido consulta en cada paquete (render1, port1, pathtracing, native-walk, f5p1-techo,
//      sombras nativas), estaticos desde el arranque, en un solo flag local por lista: con todos apagados, una prueba de
//      registro en vez de ~9 cargas de statics con guarda y una llamada no inline (PathTrace::Requested) por paquete.
//   1 encendido.  2 A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = encendido; delta interpolado.
// Contadores (solo KYTY_TECHO=1): preambulos que pasaron la puerta por motivo (ordenes, feedback terminado, feedback en
// cola sin terminar = las llamadas que A quita) y listas recorridas con B.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace RecorridoRapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_RECORRIDO_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 1; // (t69) encendido por defecto: medido sin fallos, DIFERENCIAS 0
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[recorrido-rapido] KYTY_RECORRIDO_RAPIDO=2 sin KYTY_TECHO=1: se usa 1 (siempre encendido)\n");
			m = 1;
		}
		if (m != 0)
			std::printf("[recorrido-rapido] KYTY_RECORRIDO_RAPIDO=%d (%s)\n", m, m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
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

// Contadores (solo con KYTY_TECHO=1; hilo de render y el de la cola ACB: relaxed atomics).
// Preambulos que pasaron la puerta antigua, por motivo; WSaved: feedback en cola sin terminar y sin ordenes (A, en la
// ventana B, no llamo; en la A llamo para nada). g_taken: listas PM4 recorridas con B (modos raros fuera).
enum Why : uint32_t { WCommands, WReady, WSaved, WCount };
inline constexpr const char* WhyNames[WCount] = {"ordenes/juego3", "feedback terminado", "feedback sin terminar (A quita)"};
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
	std::printf("[recorrido-rapido] ventana %s: render ocupado %.2f ms | listas con B %.0f/fot, preambulos (",
	            m == 2 ? (was ? "B (rapido)" : "A (normal)") : "rapido", busy_ms, t / frames);
	for (uint32_t i = 0; i < WCount; ++i) std::printf("%s%s %.0f", i ? ", " : "", WhyNames[i], why[i] / frames);
	std::printf(")");
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace RecorridoRapido
