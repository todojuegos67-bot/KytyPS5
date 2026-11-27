#pragma once
// (r0, 11-27) Arreglos del recorrido PM4 fuera de draws ([techo] pm4 de 976e96a) y su A/B en un solo proceso.
//   KYTY_R0_MARCAS=1: los NOP R_ZERO que no escriben memoria ni esperan nada (relleno y marcadores de user data del juego,
//     'hu' ids 0 / 4 / 0xd: CpOpMarker solo guarda el tipo del SGPR) dejan pendientes las protecciones diferidas, como ya
//     hacen el estado, los draws y EVENT_WRITE (KeepsDeferredProtection). Antes cada uno llamaba a
//     PageManager::FlushDeferredProtectionAsync (y PROT-LOTE: LoteHandOff). Los flips 0x777/0x778/0x781 no entran.
//   KYTY_R0_AB=1: alterna cada KYTY_R0_AB_S s (5 por defecto) los arreglos encendidos / apagados; lineas "[r0-ab]" con
//     fps y render ocupado (fotograma - espera del hilo de render) por tramo, y RESUMEN interpolado (cada tramo frente a
//     la media de sus dos vecinos del otro modo). Sin KYTY_R0_AB los arreglos encendidos estan siempre encendidos.
// Apagado: una lectura de un static por paquete NOP R_ZERO.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace R0Ab {

[[nodiscard]] inline bool Flag(const char* name) {
	const char* t = std::getenv(name);
	return t != nullptr && t[0] == '1' && t[1] == '\0';
}
[[nodiscard]] inline uint32_t AbSeconds() {
	static const uint32_t s = [] {
		if (!Flag("KYTY_R0_AB")) return 0u;
		const char* t = std::getenv("KYTY_R0_AB_S");
		const long  n = t != nullptr ? std::strtol(t, nullptr, 10) : 5;
		return n >= 1 && n <= 600 ? static_cast<uint32_t>(n) : 5u;
	}();
	return s;
}
[[nodiscard]] inline bool MarcasSwitch() {
	static const bool on = [] {
		const bool o = Flag("KYTY_R0_MARCAS");
		std::printf("[r0] KYTY_R0_MARCAS=%d (NOP R_ZERO de relleno / marcador sin vaciar protecciones diferidas), KYTY_R0_AB=%u s\n",
		            o ? 1 : 0, AbSeconds());
		std::fflush(stdout);
		return o;
	}();
	return on;
}
// Tramo actual del A/B (true = arreglos encendidos). Escrito solo en el flip.
inline std::atomic_bool g_window_on {true};
[[nodiscard]] inline bool Marcas() { return MarcasSwitch() && g_window_on.load(std::memory_order_relaxed); }

// Contador de lo que hizo el arreglo (paquetes que no vaciaron protecciones) en el tramo.
inline std::atomic<uint64_t> g_kept {0};

struct Tramo { bool on; double fps, busy; };
inline std::vector<Tramo> g_serie;

inline void Resumen(const char* cuando) {
	double   sf = 0, sb = 0, sbms = 0;
	uint32_t k  = 0;
	for (size_t i = 1; i + 1 < g_serie.size(); ++i) {
		const Tramo &a = g_serie[i - 1], &c = g_serie[i], &b = g_serie[i + 1];
		if (a.on == c.on || b.on == c.on) continue;
		const double vf = (a.fps + b.fps) * 0.5, vb = (a.busy + b.busy) * 0.5;
		if (vf <= 0 || vb <= 0) continue;
		const double s = c.on ? 1.0 : -1.0;
		sf += s * (c.fps - vf) * 100.0 / vf;
		sb += s * (c.busy - vb) * 100.0 / vb;
		sbms += s * (c.busy - vb);
		++k;
	}
	if (k == 0) return;
	std::printf("[r0-ab] RESUMEN interpolado (%s, %u tramos): encendido - apagado: render ocupado %+.3f ms/fot (%+.1f%%), fps %+.1f%%\n",
	            cuando, k, sbms / k, sb / k, sf / k);
	std::fflush(stdout);
}

// Hilo de render, en cada flip. idle_ns: LiveControl::g_render_idle_ns (acumulado).
inline void OnFlip(int64_t flip_ns, int64_t idle_ns) {
	const uint32_t secs = AbSeconds();
	if (secs == 0 || !MarcasSwitch()) return;
	static int64_t  start_ns = 0, last_ns = 0, last_idle = 0;
	static uint64_t frames = 0;
	static double   wall = 0, busy = 0;
	static uint32_t tramo = 0;
	if (last_ns == 0) {
		static const bool reg = std::atexit([] { Resumen("al cerrar"); }) == 0;
		(void)reg;
		start_ns = last_ns = flip_ns, last_idle = idle_ns;
		return;
	}
	const double ms = static_cast<double>(flip_ns - last_ns) / 1e6, idle = static_cast<double>(idle_ns - last_idle) / 1e6;
	++frames, wall += ms, busy += ms > idle ? ms - idle : 0.0;
	last_ns = flip_ns, last_idle = idle_ns;
	if (flip_ns - start_ns < static_cast<int64_t>(secs) * 1000000000LL) return;
	const bool   on = g_window_on.load(std::memory_order_relaxed);
	const double n  = static_cast<double>(frames);
	std::printf("[r0-ab] tramo %s #%u%s: %llu fot, %.1f fps, render ocupado %.3f ms/fot, marcas sin vaciar %.0f/fot\n",
	            on ? "ENCENDIDO" : "apagado", tramo, tramo == 0 ? " (calentamiento)" : "", static_cast<unsigned long long>(frames),
	            n * 1e3 / wall, busy / n, static_cast<double>(g_kept.exchange(0, std::memory_order_relaxed)) / n);
	if (tramo != 0) {
		g_serie.push_back({on, n * 1e3 / wall, busy / n});
		Resumen("parcial");
	}
	std::fflush(stdout);
	++tramo;
	g_window_on.store(!on, std::memory_order_relaxed);
	start_ns = flip_ns, frames = 0, wall = 0, busy = 0;
}

} // namespace R0Ab
