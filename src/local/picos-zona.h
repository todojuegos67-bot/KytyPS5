#pragma once
// (picos-zona, KYTY_PICOS_ZONA=1, apagado por defecto) Logs 20261128-014906 / 023114 (16 GB): 40 de los 83 fotogramas
// de mas de 25 ms caen en los 6 s tras entrar al mundo. Son pipelines graficos creados al primer uso (con menos de 30 GB no
// hay precalentado: "Pipeline warmup: skipped (RAM 15936 MiB)") y los de compute que tiron6 salta ("recetas sin semilla
// saltadas 267"). Con esto encendido, cada zona que carga el juego (SceneTags) recibe en el hilo de fondo del warmup (por
// debajo de lo normal) solo los pipelines de esa zona. Se para con menos de KYTY_PICOS_ZONA_FLOOR_MB de RAM disponible
// (1536) o tras KYTY_PICOS_ZONA_MB de RAM privada nueva por zona (512). El render no espera nunca: si llega antes, crea el
// suyo (el resultado es el mismo pipeline). La linea [picos-zona] da la medida (a): MB por pipeline, y cuenta los SLOW Frame
// de mas de 25 ms en los 10 s tras cada carga, encendido o apagado, para comparar A y B en dos partidas cortas.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/local-platform.h"

namespace PicosZona {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PICOS_ZONA");
		const bool  r = v != nullptr && v[0] == '1';
		std::printf("[picos-zona] KYTY_PICOS_ZONA=%s: %s\n", r ? "1" : "0",
		            r ? "ON (pipelines de cada zona creados de fondo al cargarla)" : "off (solo se cuentan los picos tras cada carga)");
		std::fflush(stdout);
		return r;
	}();
	return on;
}
inline uint64_t EnvMb(const char* name, uint64_t def) {
	const char* v = std::getenv(name);
	return (v != nullptr && *v != 0 ? std::strtoull(v, nullptr, 10) : def) << 20u;
}
inline uint64_t FloorBytes() { static const uint64_t b = EnvMb("KYTY_PICOS_ZONA_FLOOR_MB", 1536); return b; }
inline uint64_t CapBytes() { static const uint64_t b = EnvMb("KYTY_PICOS_ZONA_MB", 512); return b; }

// La puerta que mira WarmPipelines (sus hilos) mientras se precalienta una zona.
inline std::atomic<bool>     g_gate {false};
inline std::atomic<bool>     g_stopped {false};
inline std::atomic<int>      g_stop_why {0}; // 1 RAM disponible, 2 tope de MB por zona
inline std::atomic<uint64_t> g_base_private {0};
inline std::atomic<uint32_t> g_checks {0};

// true: el trabajo debe parar (comprobado cada 4 pipelines).
inline bool ShouldStop() {
	if (!g_gate.load(std::memory_order_relaxed)) return false;
	if (g_stopped.load(std::memory_order_relaxed)) return true;
	if ((g_checks.fetch_add(1, std::memory_order_relaxed) & 3u) != 0) return false;
	if (LocalPlatform::AvailablePhysicalMemory() < FloorBytes()) {
		g_stop_why.store(1);
		g_stopped.store(true);
		return true;
	}
	const auto priv = LocalPlatform::ProcessMemory().private_bytes;
	const auto base = g_base_private.load(std::memory_order_relaxed);
	if (priv > base && priv - base >= CapBytes()) {
		g_stop_why.store(2);
		g_stopped.store(true);
		return true;
	}
	return false;
}

// Picos tras cada carga: el hilo de render apunta cada fotograma lento.
inline std::atomic<int64_t>  g_load_ms {0}; // steady ms de la ultima carga (0: ninguna pendiente)
inline std::atomic<uint32_t> g_slow25 {0}, g_slow33 {0};
inline int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void NoteSlow(double ms) {
	const auto at = g_load_ms.load(std::memory_order_relaxed);
	if (at == 0 || NowMs() - at > 10000) return;
	if (ms > 25.0) g_slow25.fetch_add(1, std::memory_order_relaxed);
	if (ms > 33.0) g_slow33.fetch_add(1, std::memory_order_relaxed);
}
inline void StartWindow() {
	g_slow25.store(0);
	g_slow33.store(0);
	g_load_ms.store(NowMs(), std::memory_order_release);
}

} // namespace PicosZona
