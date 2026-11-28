#pragma once
// (proteger-junto) Solo medicion, con KYTY_TECHO=1 (apagado = una lectura de un static). Las protecciones de pagina del
// hilo de render (PageManager::ProtectHost -> ProtectTransient: ~509 llamadas/fot, 1.36 ms, ~7 paginas por llamada,
// log 20261128-003521-239) contadas para saber cuanto se podria juntar SIN cambiar el estado final:
//   - fusionables: la llamada toca (contigua) a la anterior del hilo con la misma proteccion -> una sola VirtualProtect
//     si se aplicasen en la misma tanda (limite superior de "rangos tras fusionar" = llamadas - fusionables);
//   - cercanas: misma proteccion con un hueco de 1..8 paginas (NO fusionables sin tocar el hueco: solo informativo);
//   - repetidas: mismo rango y proteccion que una de las ultimas 64 del hilo (limite superior de las redundantes: otro
//     hilo pudo cambiarla entre medias);
//   - por origen (raiz.h: read-flush, write-release, reapply...).
// KYTY_PROTEGER_JUNTO: reservado. No hay mejora detras: dentro de una llamada los tramos ya son maximos (RunLocked /
// UpdateRegionWatchers / AddReadProtect ya juntan lo contiguo), y juntar ENTRE llamadas obliga a aplazar protecciones
// de seguimiento (solo lectura / sin acceso) mas alla del candado de la region: un hilo del juego podria escribir en ese
// hueco sin falta (escritura perdida). Con este contador se decide con datos si queda algo seguro que juntar.
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "local/techo.h"
#include "local/raiz.h"

namespace ProtegerJunto {

inline std::atomic<uint64_t> g_calls {0}, g_pages {0}, g_merge {0}, g_near {0}, g_repeat {0};
inline std::atomic<uint64_t> g_by_origin[Raiz::ProtOriginCount] {}, g_merge_origin[Raiz::ProtOriginCount] {};

// (Hilo de render, antes de la VirtualProtect.) vaddr/size alineados a pagina de 4 KiB.
inline void Note(uint64_t vaddr, uint64_t size, uint32_t protection) {
	struct Last { uint64_t b, e; uint32_t p; };
	thread_local Last     prev {0, 0, 0xffffffffu};
	thread_local Last     ring[64] {};
	thread_local uint32_t ri = 0;
	const uint64_t        e  = vaddr + size;
	const uint32_t        o  = Raiz::t_protect_origin < Raiz::ProtOriginCount ? Raiz::t_protect_origin : 0u;
	g_calls.fetch_add(1, std::memory_order_relaxed);
	g_pages.fetch_add(size >> 12u, std::memory_order_relaxed);
	g_by_origin[o].fetch_add(1, std::memory_order_relaxed);
	if (prev.p == protection) {
		if (prev.e == vaddr || e == prev.b) {
			g_merge.fetch_add(1, std::memory_order_relaxed);
			g_merge_origin[o].fetch_add(1, std::memory_order_relaxed);
		} else if ((vaddr > prev.e && vaddr - prev.e <= 0x8000u) || (prev.b > e && prev.b - e <= 0x8000u)) {
			g_near.fetch_add(1, std::memory_order_relaxed);
		}
	}
	for (const auto& r: ring)
		if (r.b == vaddr && r.e == e && r.p == protection && e != 0) {
			g_repeat.fetch_add(1, std::memory_order_relaxed);
			break;
		}
	ring[ri++ & 63u] = {vaddr, e, protection};
	prev             = {vaddr, e, protection};
}

// Una linea por ventana de [techo] (mismas ventanas de 5 s que los interruptores A/B).
inline void Window(double frames) {
	if (!Techo::On() || frames <= 0) return;
	const auto   x     = [frames](std::atomic<uint64_t>& a) { return static_cast<double>(a.exchange(0, std::memory_order_relaxed)) / frames; };
	const double calls = x(g_calls), pages = x(g_pages), merge = x(g_merge), near_ = x(g_near), rep = x(g_repeat);
	std::printf("[proteger-junto] hilo de render por fot: llamadas %.1f, paginas %.0f (%.1f/llam.), rangos tras fusionar "
	            "contiguas %.1f (fusionables %.1f = %.1f %%), cercanas (hueco 1-8 pag, no fusionables) %.1f, repetidas "
	            "(mismo rango+prot en las ultimas 64) %.1f | por origen (llamadas/fusionables):",
	            calls, pages, calls > 0 ? pages / calls : 0.0, calls - merge, merge, calls > 0 ? 100.0 * merge / calls : 0.0,
	            near_, rep);
	for (uint32_t i = 0; i < Raiz::ProtOriginCount; ++i) {
		const double c = x(g_by_origin[i]), m = x(g_merge_origin[i]);
		if (c >= 0.05) std::printf(" %s %.1f/%.1f", Raiz::ProtectOriginNames[i], c, m);
	}
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace ProtegerJunto
