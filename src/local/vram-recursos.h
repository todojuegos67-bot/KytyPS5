#pragma once
// (RECURSOS-JUSTOS) KYTY_RECURSOS_JUSTOS (encendido por defecto; =0 lo apaga): que el emulador solo tenga vivo lo que el
// juego usa. Log 20261125-090852-271 (1080p, 8 GB): la precarga de compute (tiron6, semillas de sesiones anteriores)
// dejo 3746 pipelines vivos para toda la sesion SIN forma de rehacerlos (sin `remake`), asi que ni la expulsion por
// inactividad (VRAM por encima del presupuesto todo el rato: 5453/5362) ni el barrido de mundos podian quitarlos; el
// driver de NVIDIA guarda ~220 KB de RAM y el codigo en VRAM por pipeline ("interno del driver" 761 MiB). Con esto los
// de la precarga llevan como rehacerlos (del cache del driver, como los del juego) y los que nadie enlaza se van.
// Ademas una linea "[recursos]" cada 10 s con el reparto de RAM y VRAM del emulador.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace VramRecursos {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_RECURSOS_JUSTOS");
		const bool  r  = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[recursos] KYTY_RECURSOS_JUSTOS=%d: pipelines de la precarga expulsables si no se usan, linea cada 10 s "
		            "(=0: como antes)\n", r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	return on;
}

// Escritos por pipelineCache.cpp (hilo de render / precarga), leidos por el informe de vma.cpp.
inline std::atomic<uint64_t> g_warm_expulsables {0}; // pipelines de la precarga con remake (expulsables)
inline std::atomic<uint64_t> g_graphics {0};         // en el mapa (vivos + expulsados)
inline std::atomic<uint64_t> g_compute {0};
inline std::atomic<uint64_t> g_expulsados {0};       // total expulsados
inline std::atomic<uint64_t> g_rehechos {0};         // total rehechos al volver a usarse

} // namespace VramRecursos
