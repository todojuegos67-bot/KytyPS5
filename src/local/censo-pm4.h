#pragma once
// (w18) KYTY_CENSO_PM4=1 (con KYTY_CENSO_DCB=1; apagado por defecto). Solo mide.
// Dentro de los trozos de los bloques mas caros del censo [censo-dcb], el tiempo del hilo GPU por tipo de paquete PM4
// (opcode; los NOP por su R_*: ACQUIRE_MEM, RELEASE_MEM, WRITE_DATA, DMA_DATA...) y por fase del trozo:
//   preambulo (ProcessCommands + escritura de feedback entre paquetes), barrido SweepCpuWrites, ejecutor de
//   DispatchDirect (RenderExecutor), envio a Vulkan (BufferFlush), resto del trozo.
// Atribucion plana con rdtsc: cada marca cierra el tramo anterior y abre el siguiente (los IB anidados no duplican).
// Linea [censo-pm4] cada 300 frames por bloque: us/f total, y los 14 tramos que mas pesan (n/f, us/f, us/paquete).
//
// Bloques (Ghidra projelf_par1; KickoffSubmitJob 0xa08ae0 = envia lo grabado antes de la llamada):
//   0 ACB #4 de a0ed3e  VirtualGeometryRendering::PreRenderJob (0xa5d4f0)
//   1 DCB #0 de a09a7b  RenderDepth (2a) + RenderGBuffer + RenderDecal (job 0xa04ed0 x3 antes del envio 0xa09a76)
//   2 DCB #1 de a09a7b  idem
//   3 DCB #0 de a150fa  postproceso (fn 0xa14886: pases 0xa158f0..0xa17ef0, vecina de PostProcessing::AsyncPreparePostFX)
//   4 DCB #1 de a150fa  idem, marcador "NASPSSCR"
//   5 DCB #0 de a0ee77  sombras: SpotShadows (0xa10e20) + CascadedShadows (0xa11220, RenderShadowCascade 0..3)
//   6 DCB #1 de a0ee77  idem
#include "techo.h" // (techo: KYTY_TECHO=1 lo enciende en TODAS las DCB/ACB, bloque 7 "resto")

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace CensoPm4 {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* t = std::getenv("KYTY_CENSO_PM4");
		const bool  o = (t != nullptr && t[0] == '1') || Techo::On();
		if (o) {
			std::printf("[censo-pm4] KYTY_CENSO_PM4=1: tiempo por paquete PM4 en 7 bloques (necesita KYTY_CENSO_DCB=1)\n");
			std::fflush(stdout);
		}
		return o;
	}();
	return on;
}

constexpr uint32_t KNop      = 256;       // 256 + R (0..63)
constexpr uint32_t KPre      = 320;       // vuelta del bucle + preambulo
constexpr uint32_t KWalk     = 321;       // entre preambulo y decodificar (comprobaciones del bucle)
constexpr uint32_t KExec     = 322;       // RenderExecutor::DispatchDirect
constexpr uint32_t KAfter    = 323;       // tras el ejecutor, hasta el siguiente paquete
constexpr uint32_t KSweep    = 324;       // SweepCpuWrites del primer trozo
constexpr uint32_t KFlush    = 325;       // BufferFlush / BufferFlushDeferrable
constexpr uint32_t KSlice    = 326;       // resto del trozo (Process fuera del bucle)
// (r0) Subdivisiones: el tramo abierto se reetiqueta (Retag) o se parte (Mark + vuelta) sin cambiar el total.
constexpr uint32_t KNopCopy   = 328; // NOP R_ZERO = anuncio de copia lineal (DsLinearCopyHook): la copia en CPU entera
constexpr uint32_t KNopHook   = 329; // NOP R_ZERO = otras marcas del emulador (port2, CENS, NATV, SSRClear)
constexpr uint32_t KNopMarker = 330; // NOP R_ZERO = marcador 'hu' del juego (CpOpMarker)
constexpr uint32_t KExecTable = 331; // ejecutor DispatchDirect: tomado por la tabla (TableDispatch)
constexpr uint32_t KExecCopy  = 332; // ejecutor DispatchDirect: copia lineal por registros (CPU)
constexpr uint32_t KEvBarrier = 333; // EVENT_WRITE que si grabo una barrera (EndRendering + pipelineBarrier2)
constexpr uint32_t KProtSync  = 334; // FlushDeferredProtection antes de un paquete (cualquiera)
constexpr uint32_t KProtAsync = 335; // FlushDeferredProtectionAsync antes de un paquete (cualquiera)
constexpr uint32_t Keys      = 336;
constexpr uint32_t Sites     = 8; // (7: techo, cualquier otro trozo: el recorrido PM4 entero queda repartido)

constexpr uint64_t KickTag = 0xD0000000ull, AcbTag = 0xA0000000ull;
inline constexpr uint64_t kSiteKey[Sites - 1] = {
    0xa0ed3eull | ((AcbTag | 4ull) << 32u),  0xa09a7bull | ((KickTag | 0ull) << 32u), 0xa09a7bull | ((KickTag | 1ull) << 32u),
    0xa150faull | ((KickTag | 0ull) << 32u), 0xa150faull | ((KickTag | 1ull) << 32u), 0xa0ee77ull | ((KickTag | 0ull) << 32u),
    0xa0ee77ull | ((KickTag | 1ull) << 32u)};
inline constexpr const char* kSiteName[Sites] = {
    "ACB #4 a0ed3e VirtualGeometry::PreRenderJob", "DCB #0 a09a7b RenderDepth2/GBuffer/Decal",
    "DCB #1 a09a7b RenderDepth2/GBuffer/Decal", "DCB #0 a150fa postproceso", "DCB #1 a150fa postproceso (NASPSSCR)",
    "DCB #0 a0ee77 sombras Spot/Cascadas", "DCB #1 a0ee77 sombras Spot/Cascadas", "resto de DCB/ACB (techo)"};

struct Cell { std::atomic<uint64_t> n {0}, cyc {0}; };
inline Cell                  g_cells[Sites][Keys];
inline std::atomic<uint64_t> g_slices[Sites] {};

inline thread_local int      t_site = -1;
inline thread_local uint32_t t_key  = KSlice;
inline thread_local uint64_t t_last = 0;

// Cierra el tramo abierto y abre `key`.
inline void Mark(uint32_t key) {
	if (t_site < 0) return;
	const uint64_t now = __rdtsc();
	g_cells[t_site][t_key].cyc.fetch_add(now - t_last, std::memory_order_relaxed);
	t_last = now;
	t_key  = key;
}
// (r0) El tramo abierto pasa a ser de `key` (desde que se abrio); cuenta uno en `key`.
inline void Retag(uint32_t key) {
	if (t_site < 0) return;
	t_key = key;
	g_cells[t_site][key].n.fetch_add(1, std::memory_order_relaxed);
}
// (r0) Reetiqueta solo si el tramo abierto es de `from`.
inline void RetagIf(uint32_t from, uint32_t key) {
	if (t_site >= 0 && t_key == from) Retag(key);
}
[[nodiscard]] inline uint32_t Current() { return t_key; }
// Un paquete de tipo `key` (cuenta y abre su tramo).
inline void Packet(uint32_t key) {
	if (t_site < 0) return;
	Mark(key);
	g_cells[t_site][key].n.fetch_add(1, std::memory_order_relaxed);
}
[[nodiscard]] inline uint32_t KeyOf(uint32_t header) {
	const uint32_t op = (header >> 8u) & 0xffu;
	return op == 0x10u ? KNop + ((header >> 2u) & 0x3fu) : op;
}

// Un trozo de submission (GuestGpu::Process): sitio por la clave del censo DCB.
struct SliceScope {
	int prev_site;
	explicit SliceScope(uint64_t site_key): prev_site(t_site) {
		t_site = -1;
		if (!On() || (site_key == 0 && !Techo::On())) return;
		for (uint32_t i = 0; i < Sites; ++i)
			if (i == Sites - 1 ? Techo::On() : kSiteKey[i] == site_key) {
				t_site = static_cast<int>(i);
				t_key  = KSlice;
				t_last = __rdtsc();
				g_slices[i].fetch_add(1, std::memory_order_relaxed);
				break;
			}
	}
	~SliceScope() {
		Mark(KSlice);
		t_site = prev_site;
	}
	SliceScope(const SliceScope&)            = delete;
	SliceScope& operator=(const SliceScope&) = delete;
};

[[nodiscard]] inline const char* NameOf(uint32_t key) {
	switch (key) {
		case KPre: return "preambulo+vuelta";
		case KWalk: return "comprobaciones bucle";
		case KExec: return "ejecutor DispatchDirect";
		case KAfter: return "tras ejecutor";
		case KSweep: return "barrido SweepCpuWrites";
		case KFlush: return "envio BufferFlush";
		case KSlice: return "resto del trozo";
		case KNopCopy: return "NOP copia lineal (anuncio)";
		case KNopHook: return "NOP marcas emulador";
		case KNopMarker: return "NOP marcador hu";
		case KExecTable: return "ejecutor tabla";
		case KExecCopy: return "ejecutor copia lineal";
		case KEvBarrier: return "EVENT_WRITE con barrera";
		case KProtSync: return "vaciar protecciones";
		case KProtAsync: return "vaciar protecciones async";
		case 0x11: return "SET_BASE";
		case 0x15: return "DISPATCH_DIRECT";
		case 0x16: return "DISPATCH_INDIRECT";
		case 0x24: return "DRAW_INDIRECT";
		case 0x2A: return "INDEX_TYPE";
		case 0x13: return "INDEX_BUFFER_SIZE";
		case 0x40: return "COPY_DATA";
		case 0x63: return "SET_SH_REG_INDIRECT";
		case 0x64: return "SET_UCONFIG_REG_INDIRECT";
		case 0x93: return "WAIT_REG_MEM_64";
		case 0x42: return "PFP_SYNC_ME";
		case KNop: return "NOP R_ZERO";
		case 0x25: return "DRAW_INDEX_INDIRECT";
		case 0x26: return "INDEX_BASE";
		case 0x27: return "DRAW_INDEX_2";
		case 0x2D: return "DRAW_INDEX_AUTO";
		case 0x2F: return "NUM_INSTANCES";
		case 0x35: return "DRAW_INDEX_OFFSET_2";
		case 0x37: return "WRITE_DATA";
		case 0x38: return "DRAW_INDEX_INDIRECT_MULTI";
		case 0x3C: return "WAIT_REG_MEM";
		case 0x3F: return "INDIRECT_BUFFER";
		case 0x46: return "EVENT_WRITE";
		case 0x47: return "EVENT_WRITE_EOP";
		case 0x49: return "RELEASE_MEM";
		case 0x50: return "DMA_DATA";
		case 0x58: return "ACQUIRE_MEM";
		case 0x68: return "SET_CONFIG_REG";
		case 0x69: return "SET_CONTEXT_REG";
		case 0x76: return "SET_SH_REG";
		case 0x79: return "SET_UCONFIG_REG";
		case 0x9F: return "SET_CONTEXT_REG_INDIRECT";
		case KNop + 0x05: return "NOP R_DRAW_RESET";
		case KNop + 0x09: return "NOP R_DISPATCH_RESET";
		case KNop + 0x0B: return "NOP R_PUSH_MARKER";
		case KNop + 0x0C: return "NOP R_POP_MARKER";
		case KNop + 0x14: return "NOP R_ACQUIRE_MEM";
		case KNop + 0x15: return "NOP R_WRITE_DATA";
		case KNop + 0x18: return "NOP R_RELEASE_MEM";
		case KNop + 0x19: return "NOP R_DMA_DATA";
		case KNop + 0x1A: return "NOP R_CONTEXT_STATE";
		default: return nullptr;
	}
}

// (techo) Draw y dispatch: su paquete lleva el draw/dispatch entero (sus fases de tabla); lo demas es la fila "pm4" de techo.
[[nodiscard]] inline bool IsWork(uint32_t k) {
	switch (k) {
		case 0x15: case 0x16: case 0x24: case 0x25: case 0x27: case 0x2C: case 0x2D: case 0x35: case 0x38: case KExec: case KAfter: case KExecTable: case KExecCopy:
			return true;
		default: return false;
	}
}
// (techo) Todas las DCB/ACB juntas: cada tipo de paquete y tramo, ms/fot, paquetes/fot, ns por paquete; y la suma de
// lo que no es draw ni dispatch (lo que techo llama "PM4/registros fuera de draws y dispatches").
inline void PrintAll(const uint64_t* n, const uint64_t* c, double f, double us_per_cycle) {
	uint32_t order[Keys], m = 0;
	uint64_t rest_c = 0, rest_n = 0, all_c = 0;
	for (uint32_t k = 0; k < Keys; ++k) {
		all_c += c[k];
		if (c[k] == 0 && n[k] == 0) continue;
		if (!IsWork(k)) {
			rest_c += c[k];
			rest_n += n[k];
		}
		order[m++] = k;
	}
	std::sort(order, order + m, [&](uint32_t a, uint32_t b) { return c[a] > c[b]; });
	std::printf("[techo] pm4 %.0f fot: Process %.2f ms/fot; fuera de draws y dispatches %.2f ms/fot en %.0f paquetes/fot | top",
	            f, static_cast<double>(all_c) * us_per_cycle / f / 1e3, static_cast<double>(rest_c) * us_per_cycle / f / 1e3,
	            static_cast<double>(rest_n) / f);
	for (uint32_t a = 0, shown = 0; a < m && shown < 24; ++a) {
		const uint32_t k  = order[a];
		const char*    nm = NameOf(k);
		char           buf[24];
		if (nm == nullptr) {
			if (k >= KNop && k < KNop + 64) std::snprintf(buf, sizeof(buf), "NOP R_0x%02x", k - KNop);
			else std::snprintf(buf, sizeof(buf), "op 0x%02x", k);
			nm = buf;
		}
		const double us = static_cast<double>(c[k]) * us_per_cycle;
		std::printf(" | %s%s %.3f ms %.0f/f", IsWork(k) ? "*" : "", nm, us / f / 1e3, static_cast<double>(n[k]) / f);
		if (n[k] != 0) std::printf(" %.0f ns", us * 1e3 / static_cast<double>(n[k]));
		++shown;
	}
	std::printf(" | (* = draw/dispatch, ya en sus filas)\n");
}

// Desde la frontera de frame (hilo GPU).
inline void OnFrame() {
	if (!On()) return;
	static uint32_t frames   = 0;
	static uint64_t tsc0     = __rdtsc();
	static auto     clock0   = std::chrono::steady_clock::now();
	if (++frames < 300) return;
	const uint64_t tsc   = __rdtsc();
	const auto     clock = std::chrono::steady_clock::now();
	const double   ns    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock - clock0).count());
	const double   us_per_cycle = tsc > tsc0 ? ns / 1000.0 / static_cast<double>(tsc - tsc0) : 0.0;
	const double   f            = frames;
	static uint64_t agg_n[Keys], agg_c[Keys];
	std::fill(std::begin(agg_n), std::end(agg_n), 0);
	std::fill(std::begin(agg_c), std::end(agg_c), 0);
	for (uint32_t s = 0; s < Sites; ++s) {
		struct R { uint32_t k; uint64_t n, c; };
		R        rows[Keys];
		uint32_t m = 0;
		uint64_t total = 0;
		for (uint32_t k = 0; k < Keys; ++k) {
			const uint64_t n = g_cells[s][k].n.exchange(0, std::memory_order_relaxed);
			const uint64_t c = g_cells[s][k].cyc.exchange(0, std::memory_order_relaxed);
			if (n == 0 && c == 0) continue;
			agg_n[k] += n;
			agg_c[k] += c;
			rows[m++] = {k, n, c};
			total += c;
		}
		const uint64_t slices = g_slices[s].exchange(0, std::memory_order_relaxed);
		if (m == 0) continue;
		for (uint32_t a = 0; a < m; ++a)
			for (uint32_t b = a + 1; b < m; ++b)
				if (rows[b].c > rows[a].c) { const R t = rows[a]; rows[a] = rows[b]; rows[b] = t; }
		std::printf("[censo-pm4] %s: %.0f us/f, %.1f trozos/f;", kSiteName[s], static_cast<double>(total) * us_per_cycle / f,
		            static_cast<double>(slices) / f);
		for (uint32_t a = 0; a < m && a < 14; ++a) {
			const char* nm = NameOf(rows[a].k);
			char        buf[24];
			if (nm == nullptr) {
				if (rows[a].k >= KNop && rows[a].k < KNop + 64) std::snprintf(buf, sizeof(buf), "NOP R_0x%02x", rows[a].k - KNop);
				else std::snprintf(buf, sizeof(buf), "op 0x%02x", rows[a].k);
				nm = buf;
			}
			const double us = static_cast<double>(rows[a].c) * us_per_cycle;
			std::printf(" | %s %.0f/f %.0f us/f", nm, static_cast<double>(rows[a].n) / f, us / f);
			if (rows[a].n != 0) std::printf(" (%.2f us/u)", us / static_cast<double>(rows[a].n));
		}
		std::printf("\n");
	}
	if (Techo::On()) PrintAll(agg_n, agg_c, f, us_per_cycle);
	std::fflush(stdout);
	frames = 0;
	tsc0   = tsc;
	clock0 = clock;
}

} // namespace CensoPm4
