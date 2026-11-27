#pragma once
// (techo) KYTY_TECHO=1 (apagado por defecto; apagado = una lectura de un static). Solo mide: el hilo de render, en
// ms por fotograma, repartido por las fases mas grandes, para ordenar el siguiente objetivo por coste real (no fps).
// Con KYTY_TECHO=1 se encienden tambien, sin mas variables, los censos que ya existen y que esto lee:
//   - KYTY_CENSO_DCB (ds-censo-dcb.h): tiempo de Process por DCB/ACB y su sitio del juego (gancho 0xa08ae0
//     KickoffSubmitJob): a09a7b profundidad+GBuffer+decal, a0ee77 sombras (Spot + CascadedShadows), a0ed3e ACB
//     VirtualGeometry::PreRenderJob, a150fa postproceso...; draws y dispatches por DCB (cabeceras PM4).
//   - KYTY_NATIVO_PASE_ESCENA=2 (ds-censo-escena.h): marcas CENS del gancho 0xab57d0 = tiempo por pase de listas de
//     mallas; sombras: cascada 0..3 del sol e "i+4" (spot/extra). Lo que la DCB de sombras gasta FUERA de esas marcas
//     son las otras vias del sol (VirtualGeo 0xa57af0 y 0xab6080) mas su PM4 suelto: fila "sombras otras vias".
//   - KYTY_RENDER_STATS (solo sus scopes: las fases del camino de tabla por draw y dispatch, live-census.h).
// Propio de techo: tiempo (rdtsc) de los cambios de proteccion (ProtectTransient: los ~98K protect_calls por ventana)
// y del manejador de faltas (GpuResourceManager::HandleFault), separado hilo de render / hilos del juego. El despacho
// de la excepcion de Windows (antes y despues del manejador) no se ve desde aqui: es un minimo.
// Salida: una linea "[techo] v" cada 5 s; cada 30 s y al salir la tabla acumulada (sin la 1a ventana, arranque):
// "[techo] fase X: Y ms/fot (Z %)" ordenada por bloques (pases / coste por draw / coherencia / esperas); Z = % del
// hilo de render ocupado. Los bloques se solapan entre si (un draw de la DCB de GBuffer esta en su pase Y en su
// coste por draw): dentro de cada bloque las filas no se solapan salvo lo indicado.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Techo {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* t = std::getenv("KYTY_TECHO");
		return t != nullptr && t[0] == '1' && t[1] == '\0';
	}();
	return on;
}

// Coherencia: [0] hilos del juego / otros, [1] hilo de render. Ciclos TSC y llamadas.
enum Coh : uint32_t { CProtect, CFault, Cohs };
inline std::atomic<uint64_t> g_coh_cyc[Cohs][2] {};
inline std::atomic<uint64_t> g_coh_n[Cohs][2] {};

struct CohScope {
	uint64_t t0;
	uint32_t k;
	bool     render;
	CohScope(Coh which, bool is_render): t0(On() ? __rdtsc() : 0), k(which), render(is_render) {}
	~CohScope() {
		if (t0 == 0) return;
		g_coh_cyc[k][render ? 1 : 0].fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		g_coh_n[k][render ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
	}
	CohScope(const CohScope&)            = delete;
	CohScope& operator=(const CohScope&) = delete;
};

// Una fila de la tabla: valor acumulado en ms (sumado por ventana) y su bloque.
constexpr uint32_t MaxRows = 96;
struct Row {
	char     name[96];
	uint32_t block;
	double   ms;    // suma de ms/fot * fotogramas (se divide al imprimir)
	double   extra; // draws o llamadas * fotogramas (idem)
};
inline Row      g_rows[MaxRows] {};
inline uint32_t g_row_count = 0;
inline double   g_frames_total = 0, g_busy_total = 0, g_frame_total = 0;

inline void Accum(uint32_t block, const char* name, double ms_per_frame, double extra_per_frame, double frames) {
	uint32_t i = 0;
	for (; i < g_row_count; ++i)
		if (g_rows[i].block == block && std::strcmp(g_rows[i].name, name) == 0) break;
	if (i == g_row_count) {
		if (g_row_count == MaxRows) return;
		std::snprintf(g_rows[i].name, sizeof(g_rows[i].name), "%s", name);
		g_rows[i].block = block;
		g_rows[i].ms = g_rows[i].extra = 0;
		++g_row_count;
	}
	g_rows[i].ms += ms_per_frame * frames;
	g_rows[i].extra += extra_per_frame * frames;
}

inline constexpr const char* kBlocks[] = {"PASES (por DCB/ACB del juego, tiempo de Process)", "DENTRO DE PASES (marcas 0xab57d0)",
                                          "COSTE POR DRAW/DISPATCH", "COHERENCIA DE MEMORIA", "ESPERAS (hilo de render)"};
constexpr uint32_t Blocks = sizeof(kBlocks) / sizeof(kBlocks[0]);
inline constexpr const char* kExtra[] = {"draws+disp/fot", "draws/fot", "llamadas/fot", "llamadas/fot", ""};

inline void PrintTable(const char* why) {
	if (g_frames_total <= 0) return;
	const double f = g_frames_total, busy = g_busy_total / f;
	std::printf("[techo] ===== TABLA (%s): %.0f fotogramas, fotograma %.2f ms, hilo de render ocupado %.2f ms/fot =====\n", why, f,
	            g_frame_total / f, busy);
	for (uint32_t b = 0; b < Blocks; ++b) {
		uint32_t idx[MaxRows], n = 0;
		for (uint32_t i = 0; i < g_row_count; ++i)
			if (g_rows[i].block == b) idx[n++] = i;
		std::sort(idx, idx + n, [](uint32_t x, uint32_t y) { return g_rows[x].ms > g_rows[y].ms; });
		std::printf("[techo] -- %s\n", kBlocks[b]);
		for (uint32_t k = 0; k < n; ++k) {
			const Row& r = g_rows[idx[k]];
			const double ms = r.ms / f;
			if (ms < 0.01 && r.extra / f < 0.5) continue;
			if (kExtra[b][0] != '\0' && r.extra > 0)
				std::printf("[techo] fase %s: %.2f ms/fot (%.1f %%) [%.0f %s]\n", r.name, ms, busy > 0 ? 100.0 * ms / busy : 0.0, r.extra / f,
				            kExtra[b]);
			else
				std::printf("[techo] fase %s: %.2f ms/fot (%.1f %%)\n", r.name, ms, busy > 0 ? 100.0 * ms / busy : 0.0);
		}
	}
	std::fflush(stdout);
}

} // namespace Techo
