#pragma once
// (w11) KYTY_NATIVO_VGEO: los dispatches de VirtualGeometryRendering::PreRenderJob (eboot 0xa5d4f0, la ACB #4 del envio
// de a0eb80 forward #1) por una via corta.
//
// Ghidra (eboot): 0xa5d4f0 recorre los objetos de VirtualGeometry cercanos y por cada uno graba
//   a5d930: AcbDispatchIndirect (j3EtxFkSIhQ), shader en el objeto global +0x1370
//   a5dc20: Dispatch(1,1,1)     (k3GhuSNmBLU), shader +0x1360 (prepara los argumentos del siguiente)
//   a5dd80: AcbDispatchIndirect por submalla,  shader +0x1380, argumentos en param_1+0x178 (escritos por la GPU)
// con el objeto global = *(*(*(*(eboot+0x2f0a5f8)+0x950)+0x20)+0x68), y el shader de cada uno en
// *(*(*(obj+off)+0x20)+0x60) -> +0x20 = ShaderRegister[] {offset, value} (FUN_01d134a0 -> AgcCbSetShRegistersDirect),
// de donde salen COMPUTE_PGM_LO/HI = la direccion del CS que ve el procesador de comandos.
//
// Que hace la via corta (sin traducir PM4 de mas, sin copias ni vigilancia):
//  1. DispatchIndirectAddress: para esos 3 CS, nunca baja los argumentos a la CPU (SyncGpuCleanBacking + memcpy):
//     Vulkan los lee de la GPU (vkCmdDispatchIndirect). Solo si el modo no pide dimensiones de hilo (bit 0x20).
//  2. RenderExecutor::DispatchDirect: para esos 3 CS, salta el prologo de diagnostico (trazas, LiveCounters,
//     SkipShader, catalogos de copia/limpieza) y va directo al camino de tabla (pipeline cacheado, set, push data,
//     dispatchIndirect). Si la tabla lo rechaza, sigue por el camino normal (contado aparte).
// KYTY_NATIVO_VGEO=1 encendido; =2 validacion: 1 de cada 8 frames va por el camino normal entero y la linea
// [nativo-vgeo] da los us/dispatch de cada camino (comparacion de coste; la imagen se compara a ojo: sin readback).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace VgeoNativo {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_NATIVO_VGEO");
		const int   m = v != nullptr ? std::atoi(v) : 0;
		std::printf("[nativo-vgeo] KYTY_NATIVO_VGEO=%d (%s)\n", m,
		            m == 1 ? "via corta" : m == 2 ? "via corta + validacion 1/8 frames por el camino normal"
		            : m == 3 ? "A/B cada 5 s (con KYTY_TECHO): solo tabla primero, sin la via de argumentos" : "apagado");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<uint64_t> g_cs[3] {}; // direcciones de CS (formato cs_regs.data_addr)
inline std::atomic<uint32_t> g_frame {0};
inline std::atomic<bool>     g_normal_frame {false}; // (=2) este frame por el camino normal
inline std::atomic<bool>     g_ab_on {false};        // (=3) esta ventana de 5 s con la tabla primero (B)

// Contadores del intervalo (hilo GPU).
inline std::atomic<uint64_t> g_fast {0}, g_fast_ns {0}, g_normal {0}, g_normal_ns {0}, g_val_normal {0},
    g_val_normal_ns {0}, g_args_skipped {0}, g_frames {0};

[[nodiscard]] inline bool Readable(uint64_t at, size_t bytes) {
#if defined(_WIN32)
	if (at < 0x10000 || (at & 7u) != 0) return false;
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
	if (info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) return false;
	return at + bytes <= reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
#else
	return at >= 0x10000;
#endif
}
[[nodiscard]] inline uint64_t Load(uint64_t at) {
	return Readable(at, 8) ? *reinterpret_cast<const volatile uint64_t*>(at) : 0;
}

// Una vez por frame (6 lecturas): los 3 CS desde los objetos del juego.
inline void Resolve() {
	static const uint64_t base = [] {
		const char* v = std::getenv("KYTY_NATIVO_VGEO_BASE");
		return v != nullptr ? std::strtoull(v, nullptr, 16) : 0x900000000ull;
	}();
	uint64_t p = Load(base + 0x2f0a5f8);
	p          = p ? Load(p + 0x950) : 0;
	p          = p ? Load(p + 0x20) : 0;
	const uint64_t obj = p ? Load(p + 0x68) : 0;
	static const uint32_t offs[3] {0x1360, 0x1370, 0x1380};
	for (int i = 0; i < 3; ++i) {
		uint64_t s = obj ? Load(obj + offs[i]) : 0;
		s          = s ? Load(s + 0x20) : 0;
		s          = s ? Load(s + 0x60) : 0;
		const uint64_t regs = s ? Load(s + 0x20) : 0;
		uint64_t       cs   = 0;
		if (regs != 0 && Readable(regs, 64 * 8)) {
			const auto* r  = reinterpret_cast<const volatile uint32_t*>(regs);
			uint32_t    lo = 0, hi = 0;
			bool        has_lo = false;
			for (int k = 0; k < 64; ++k) {
				const uint32_t off = r[k * 2] & 0xffffu, val = r[k * 2 + 1];
				if (off == 0x20c || off == 0x2e0c) { lo = val; has_lo = true; }
				if (off == 0x20d || off == 0x2e0d) hi = val;
			}
			if (has_lo) cs = (static_cast<uint64_t>(lo) << 8u) | (static_cast<uint64_t>(hi & 0xffu) << 40u);
		}
		if (cs != g_cs[i].load(std::memory_order_relaxed)) {
			g_cs[i].store(cs, std::memory_order_relaxed);
			static std::atomic<uint32_t> logs {0};
			if (logs.fetch_add(1) < 12) {
				std::printf("[nativo-vgeo] shader %s (+0x%x) -> CS 0x%llx\n", i == 0 ? "1x1x1" : i == 1 ? "indirecto objeto" : "indirecto submalla",
				            offs[i], static_cast<unsigned long long>(cs));
				std::fflush(stdout);
			}
		}
	}
}

// (w11b) Censo por CS dentro de la ACB #4 de a0ed3e (solo diagnostico; necesita KYTY_CENSO_DCB=1, que da el sitio).
// (w20) Filas en tabla hash de 1024 que NUNCA se vacian (la clave queda; solo los contadores van a 0 cada ventana):
// en w18 las 128 filas se vaciaban cada 300 frames y se llenaban con los primeros CS de la ventana -> 1125 dispatches/f
// "sin fila" (medido, log 20261125-131856). La via corta ya no depende de este censo (ver TableFirst).
inline thread_local bool t_in_target = false;
struct CsRow { std::atomic<uint64_t> cs {0}, direct {0}, indirect {0}, ns {0}; };
constexpr uint32_t CsRows = 1024; // potencia de 2
inline CsRow g_rows[CsRows];
inline std::atomic<uint64_t> g_target_other {0}; // (tabla llena)

inline void NoteTargetDispatch(uint64_t cs, bool indirect, uint64_t ns) {
	uint32_t h = static_cast<uint32_t>((cs >> 8u) * 0x9E3779B97F4A7C15ull >> 54u) & (CsRows - 1u);
	for (uint32_t n = 0; n < CsRows; ++n, h = (h + 1u) & (CsRows - 1u)) {
		uint64_t k = g_rows[h].cs.load(std::memory_order_relaxed);
		if (k == 0) { g_rows[h].cs.store(cs, std::memory_order_relaxed); k = cs; } // (un solo hilo GPU por cola)
		if (k == cs) {
			(indirect ? g_rows[h].indirect : g_rows[h].direct).fetch_add(1, std::memory_order_relaxed);
			g_rows[h].ns.fetch_add(ns, std::memory_order_relaxed);
			return;
		}
	}
	g_target_other.fetch_add(1, std::memory_order_relaxed);
}
// Ambito del trozo: el sitio del censo es la ACB #4 de a0ed3e.
struct TargetScope {
	bool prev = t_in_target;
	explicit TargetScope(uint64_t site_key) {
		t_in_target = site_key == (0xa0ed3eull | ((0xA0000000ull | 4ull) << 32u));
	}
	~TargetScope() { t_in_target = prev; }
	TargetScope(const TargetScope&) = delete;
	TargetScope& operator=(const TargetScope&) = delete;
};

// Los 3 CS de PreRenderJob resueltos desde el juego (para no bajar a la CPU sus argumentos indirectos).
[[nodiscard]] inline bool Is(uint64_t cs) {
	return cs != 0 && (cs == g_cs[0].load(std::memory_order_relaxed) || cs == g_cs[1].load(std::memory_order_relaxed) ||
	                   cs == g_cs[2].load(std::memory_order_relaxed));
}
[[nodiscard]] inline bool ValidationFrame() { return Mode() == 2 && g_normal_frame.load(std::memory_order_relaxed); }
// La via corta para este CS en este frame (argumentos indirectos en la GPU).
[[nodiscard]] inline bool Fast(uint64_t cs) { return (Mode() == 1 || Mode() == 2) && Is(cs) && !ValidationFrame(); }
// (w20) Camino de tabla PRIMERO para todo dispatch de compute (cualquier CS, cualquier cola): TableDispatch ya rechaza
// por si mismo lo que no puede ir por tabla (no visto 2 veces, copia lineal / limpieza marcadas "special", TAA de frame
// gen, HWRT, path tracing, SkipShader, captura). Sin aprendizaje ni lista de CS: no hay nada que vaciar.
// (vg1, =3) En la A/B solo esto cambia: el mismo TableDispatch con los mismos argumentos, antes o despues del prologo de
// diagnostico del camino normal (trazas, censos, LiveTrace, Envio1, comprobaciones de CS nulo / SkipShader, que la tabla
// ya excluye: CS visto 2 veces por el camino normal, sin SkipShader ni captura). Lo grabado en Vulkan es identico.
[[nodiscard]] inline bool TableFirst() {
	return Mode() == 3 ? g_ab_on.load(std::memory_order_relaxed) : Mode() != 0 && !ValidationFrame();
}
// (vg1) Ventana de 5 s de [techo] (hilo del informe): alterna A (prologo antes) / B (tabla primero) y compara el render
// ocupado de cada ventana con la media de sus dos vecinas del otro modo (deriva de escena interpolada). Sin la 1a ventana.
inline void AbWindow(double busy_ms, double fps, bool keep) {
	if (Mode() != 3) return;
	struct W { bool b; double busy; };
	static W      w[4096];
	static size_t n = 0;
	const bool    was = g_ab_on.load(std::memory_order_relaxed);
	if (keep && n < 4096) w[n++] = {was, busy_ms};
	double sum[2] {}, cnt[2] {};
	for (size_t i = 1; i + 1 < n; ++i)
		if (w[i - 1].b != w[i].b && w[i + 1].b != w[i].b) {
			const double d = w[i].busy - 0.5 * (w[i - 1].busy + w[i + 1].busy); // (este - vecinas)
			sum[w[i].b ? 1 : 0] += d, cnt[w[i].b ? 1 : 0] += 1;
		}
	// B-A: las ventanas B dan (B - A) y las A dan (A - B): delta = (mB - mA) / 2 con signo.
	const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
	const double delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	std::printf("[nativo-vgeo] A/B tabla primero: esta ventana %s, render ocupado %.2f ms, %.1f fps | ventanas A %.0f B %.0f | "
	            "delta interpolado B-A %+.3f ms/fot (negativo = B ahorra)\n",
	            was ? "B (tabla primero)" : "A (como antes)", busy_ms, fps, cnt[0], cnt[1], delta);
	std::fflush(stdout);
	g_ab_on.store(!was, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Now() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Desde el limite de frame (hilo GPU, CommandProcessor): resolver, elegir frame de validacion, informar cada 300.
inline void OnFrame(uint32_t frame) {
	if (Mode() == 0 || frame == g_frame.load(std::memory_order_relaxed)) return;
	g_frame.store(frame, std::memory_order_relaxed);
	Resolve();
	g_normal_frame.store(Mode() == 2 && (frame & 7u) == 0, std::memory_order_relaxed);
	if (g_frames.fetch_add(1) + 1 < 300) return;
	const double f   = static_cast<double>(g_frames.load());
	{
		struct R { uint64_t cs, d, i, ns; };
		static R rows[CsRows]; // (hilo GPU; 32 KiB fuera de la pila)
		uint32_t n = 0;
		uint64_t total = 0, total_ns = 0;
		for (auto& r: g_rows) {
			const uint64_t cs = r.cs.load(std::memory_order_relaxed); // (w20: la clave se queda)
			if (cs == 0) continue;
			const uint64_t d = r.direct.exchange(0), i = r.indirect.exchange(0), ns = r.ns.exchange(0);
			if (d + i == 0) continue;
			if (n < CsRows) rows[n++] = {cs, d, i, ns};
			total += d + i;
			total_ns += ns;
		}
		std::partial_sort(rows, rows + (n < 5u ? n : 5u), rows + n,
		                  [](const R& x, const R& y) { return x.d + x.i > y.d + y.i; });
		std::printf("[nativo-vgeo] censo ACB #4 a0ed3e: %.1f dispatches/f en %u CS distintos (tabla llena %.1f/f), %.0f us/f en "
		            "DispatchDirect; top 5:",
		            total / f, n, g_target_other.exchange(0) / f, total_ns / 1000.0 / f);
		for (uint32_t a = 0; a < n && a < 5; ++a)
			std::printf(" | CS 0x%llx %.1f/f (directos %.1f, indirectos %.1f) %.0f us/f%s", static_cast<unsigned long long>(rows[a].cs),
			            (rows[a].d + rows[a].i) / f, rows[a].d / f, rows[a].i / f, rows[a].ns / 1000.0 / f,
			            rows[a].cs == g_cs[0].load() ? " =+0x1360" : rows[a].cs == g_cs[1].load() ? " =+0x1370"
			                                         : rows[a].cs == g_cs[2].load() ? " =+0x1380" : "");
		std::printf("\n");
	}
	g_frames.store(0);
	const auto   fa  = g_fast.exchange(0), fan = g_fast_ns.exchange(0);
	const auto   no  = g_normal.exchange(0), non = g_normal_ns.exchange(0);
	const auto   vn  = g_val_normal.exchange(0), vnn = g_val_normal_ns.exchange(0);
	const auto   arg = g_args_skipped.exchange(0);
	std::printf("[nativo-vgeo] %.0f frames | via corta %.0f us/f, %.1f dispatches/f (%.2f us/dispatch) | rechazados por la "
	            "tabla al camino normal %.1f/f (%.0f us/f) | argumentos indirectos sin bajar a la CPU %.1f/f | validacion "
	            "(camino normal entero) %.1f dispatches/f de sus frames, %.2f us/dispatch\n",
	            f, fan / 1000.0 / f, fa / f, fa ? fan / 1000.0 / fa : 0.0, no / f, non / 1000.0 / f, arg / f,
	            vn ? vn / (f / 8.0) : 0.0, vn ? vnn / 1000.0 / vn : 0.0);
	std::fflush(stdout);
}

} // namespace VgeoNativo
