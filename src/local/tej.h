#pragma once
// (tej) KYTY_TEJ, tejido propio. FASE 1b "sombra paralela" (tools/plan-weave-propio.md). Apagado por defecto. SOLO MIDE: la
// imagen no puede cambiar (nada de lo que hacen los trabajadores se envia, se cose ni se sube a recursos del juego), y el
// hilo de render nunca espera a la sombra (ni cola de fotogramas extra ni retraso de la presentacion).
//   0 apagado (identico a antes: un static por gancho).
//   1 sombra encendida.
//   2 A/B cada ventana de 5 s de [techo] (KYTY_TECHO=1): B = sombra encendida; delta interpolado B-A.
// (tej-lag3) SEPARACION: lo marcado MEDIDA (HookTimer, RenderWait, F2p Real/Before, coste del registro de efectos,
// diagnosticos COS) solo corre con KYTY_TECHO=1; sin el, el render solo hace: epocas + validar regs + rangos (con filtro
// de regiones) por paquete, y lee slot.count solo al alcanzar lo visto. NECESARIO: proteccion de RAM (g_mem_lvl).
//   KYTY_TEJ_N=1..6: trabajadores (por defecto: nucleos de rendimiento de la L3 del render sin el suyo, tope 6).
// F1b (frente a F1, que tenia un solo hilo que llegaba tarde: 1.1 de 15 DCB con sombra, 'render ya paso' 9/fot):
// - Lector (Kyty.Tej.L) despertado en GuestGpu::Enqueue (hilo del juego: alli se COPIA la DCB): encadena el estado DCB a
//   DCB con una PASADA DE ESTADO (CommandProcessor::SetStateOnly, la de KYTY_SPECULATE: solo registros, las escrituras a
//   memoria a un Spec::State propio que nadie lee) en vez de traducir; asi el estado inicial de cada DCB esta listo en
//   cuanto llega y N trabajadores traducen DCB distintas EN PARALELO. Si la cadena se rompe, como F1: estado publicado
//   por el render. Cuenta DCB autocontenidas (reset o CLEAR_STATE antes del primer draw).
// - N trabajadores (Kyty.Tej.0..5), uno por DCB; el trabajador k solo toma trabajo si en la ventana anterior (100 ms)
//   hubo >= k+1 nucleos libres (con 1 libre, solo 1 trabaja; con 0, nada: coste 0). Prioridad por coste: con mas DCB en
//   cola que trabajadores libres, la mas grande primero. Afinidad: L3 del render (en X3D la de V-Cache, donde esta el
//   render) & nucleos de rendimiento (Intel hibrida: P-cores) sin el nucleo del render ni su hermano SMT.
// - Validacion en el render (primer draw de cada paquete de 64): epocas primero (baratas), registros (SameState) solo si
//   las epocas coinciden.
// - Modelo N=1,2,4,6 con los tiempos medidos de cada DCB (llegada, estado listo, render inicio/fin, coste por draw del
//   trabajador): % de draws en paquetes que llegarian a tiempo.
// - Latencia Enqueue->flip (primer envio grafico del fotograma hasta que el render procesa su flip) media y p99, y
//   fotogramas > 25 ms (flip a flip en el render), A frente a B.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/speculation.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "local/local-platform.h"
#include "local/speculation-state.h"
#include "local/techo.h"
#include "local/buffercache-par.h"
#include "local/tej-lag.h"
#include "local/tej-coser.h" // (tej-coser: KYTY_TEJ=3 etapa 1, solo mide)


namespace Tej {

using Libs::Graphics::CommandProcessor;
using Libs::Graphics::Pm4Execution;
using Libs::Graphics::Pm4ProcessResult;
using Libs::Graphics::RenderContext;
using Libs::Graphics::Speculation;

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TEJ");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m == 3) { // (tej-coser) etapa 1: como 2 + medida de cose; sin maintenance7, 2 sin mas
			std::printf("[tej] KYTY_TEJ=3: F2 coser etapa 1 %s\n", TejCoser::On() ? "(mide; imagen como KYTY_TEJ=2)"
			                                                                  : "NO disponible (sin VK_KHR_maintenance7): KYTY_TEJ=2");
			m = 2;
		}
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[tej] KYTY_TEJ=2 sin KYTY_TECHO=1: modo 1 (sombra siempre, sin linea)\n");
			m = 1;
		}
		if (m != 0) { // (tej-sinmutex) sin KYTY_TEJ_VAR: KYTY_TEJ_SINMUTEX (1 por defecto); con VAR solo su variante
			const char* sm  = std::getenv("KYTY_TEJ_SINMUTEX");
			const char* var = std::getenv("KYTY_TEJ_VAR");
			const bool  on  = (sm == nullptr || std::atoi(sm) != 0) && (var == nullptr || std::atoi(var) == 0);
			TejLag::g_sinmutex.store(on, std::memory_order_relaxed);
			std::printf("[tej] KYTY_TEJ_SINMUTEX: la sombra %s el RenderMutex%s\n", on ? "NO pide" : "pide",
			            var != nullptr && std::atoi(var) != 0 ? " (KYTY_TEJ_VAR: solo en la variante 'sin mutex')" : "");
		}
		if (m != 0)
			std::printf("[tej] KYTY_TEJ=%d (%s): F1b sombra paralela, SOLO MIDE (nada se cose ni se envia)\n", m,
			            m == 1 ? "sombra encendida" : "A/B cada 5 s, B = sombra encendida");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};
// (tej-var) KYTY_TEJ_VAR=1 (con KYTY_TEJ=2): cada ventana B es una VARIANTE distinta, en rotacion, siempre entre dos A:
//   0 normal | 1 sin traducir (la sombra recibe y prepara la DCB pero no la traduce: sin RenderMutex, LRU, cachés ni
//   objetos Vulkan; mide lo que cuesta el resto) | 2 N=1 (solo el trabajador 0) | 3 mismo CCD (trabajadores y lector en
//   la L3 del render) | 4 cortes (la sombra no pide el RenderMutex mientras el render esta dentro).
// Linea [tej] VARIANTES: delta interpolado B-A de cada una. Una sola ejecucion dice cual quita el lag.
//   5 sin mutex (KYTY_TEJ_SINMUTEX: la sombra nunca pide el RenderMutex; las variantes 0-4 lo piden, como antes).
//   6 N=1 otro CCD prioridad normal (tej-lag3: un solo trabajador, en la otra L3, a prioridad normal: termina antes y
//   compite menos tiempo con el render por memoria/driver?).
//   7 muestreo 1/8 (tej-lag4: sin mutex, la sombra solo recibe DCB 1 de cada 8 tramos de 40 ms).
inline constexpr int    VarN = 8;
inline constexpr const char* VarName[VarN] = {"normal", "sin traducir", "N=1", "mismo CCD", "cortes", "sin mutex",
                                              "N=1 otro CCD normal", "muestreo 1/8"};
[[nodiscard]] inline bool VarOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEJ_VAR");
		return v != nullptr && std::atoi(v) != 0;
	}();
	return on;
}
inline std::atomic<int> g_var {0}; // (variante de la ventana B actual; 0 sin KYTY_TEJ_VAR)
[[nodiscard]] inline int Var() { return g_var.load(std::memory_order_relaxed); }
// (tej-lag3) NECESARIO: proteccion de RAM (equipos de 16 GB). Nivel 0 normal; 1 (disponible < 1 GiB): un solo trabajador;
// 2 (< 0.5 GiB): sombra apagada (Active() falso: nada se copia ni se traduce). Histeresis: sube de nivel por debajo del
// umbral, baja por encima de 1.5x (0.75 / 1.5 GiB). Comprobacion: GlobalMemoryStatusEx 1 de cada 2048 envios de la cola 0 (~2 s), en el
// hilo del juego (OnEnqueue).
inline std::atomic<int> g_mem_lvl {0};
inline void             MemCheck() {
	static uint32_t k = 0;
	if ((++k & 2047u) != 1u) return;
	const uint64_t av = LocalPlatform::SystemMemoryStatus().available;
	if (av == 0) return;
	constexpr uint64_t G = 1ull << 30;
	const int          old = g_mem_lvl.load(std::memory_order_relaxed);
	const int          lvl = av < G / 2 || (old == 2 && av < G * 3 / 4) ? 2 : av < G || (old >= 1 && av < G * 3 / 2) ? 1 : 0;
	if (lvl != old) {
		g_mem_lvl.store(lvl, std::memory_order_relaxed);
		std::printf("[tej] RAM disponible %.2f GiB: %s\n", static_cast<double>(av) / G,
		            lvl == 2 ? "sombra APAGADA (< 0.5 GiB)" : lvl == 1 ? "un solo trabajador (< 1 GiB)" : "normal");
	}
}
[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return (m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed))) && g_mem_lvl.load(std::memory_order_relaxed) < 2;
}

// (tej F2p) Registro de efectos del render (escrituras GPU, escrituras de CPU que ven los barridos, sets retirados) con
// KYTY_TEJ: el mismo EffectsLog que CommitSegments lee (Spec::t_effects del hilo de render cuando no hay especulacion).
// Solo el render escribe y lee las entradas; el trabajador lee End() (atomico) al empezar a traducir.
// Dueno: el primer hilo que procesa una DCB de la cola 0 (GpuSlice), para siempre; los demas hilos no escriben.
inline Libs::Graphics::Spec::EffectsLog g_eff;
inline std::atomic<bool>                 g_eff_owned {false};
inline thread_local bool                 t_eff_owner = false;
// (tej-lag) Solo en B (Active): en A el render no escribe nada en el registro (lo que A no vio marca 'perdido').
[[nodiscard]] inline Libs::Graphics::Spec::EffectsLog* EffLog() { return Mode() != 0 && t_eff_owner && Active() ? &g_eff : nullptr; }
inline void ClaimEff() {
	if (t_eff_owner || g_eff_owned.load(std::memory_order_relaxed)) return;
	bool no = false;
	if (g_eff_owned.compare_exchange_strong(no, true)) t_eff_owner = true;
}
inline uint64_t g_eff_ticks = 0; // (tej-lag: TSC de EffectsLog::Add en el registro de tej; solo el hilo dueno)

inline constexpr uint32_t PacketDraws = 64, MaxPackets = 16, Slots = 16, MaxAhead = 12, MaxQueued = 24, MaxWorkers = 6;

enum C : uint32_t {
	// hilo de render
	GDraws, GDcbs, GDcbSlot, GPkOk, GDrawsOk, GHolesOk, GPkRegs, GDrawsRegsOk, GPkEpBuf, GPkEpImg, GPkEpMap, GPkLate,
	GPkSkip, GTailLate, GValTicks, GValN, GPublish,
	GDrawsOnTime, // draws de paquetes publicados antes de que el render llegara (traducidos a tiempo)
	GCalls, GPkAllHoles, GPkAssumed, GDrawsOkAssumed, GDrawsOnTimeAssumed, GRegDiffNames, GPkRegsSampled, GRgRaw,
	// (F1d) registros solo de draws; epocas por rangos de lo que el paquete leyo
	GPkCsIgnored, GPkCsDispatch, GDrawsOkGfx, GDrawsOkRanges, GPkRgBuf, GPkRgImg, GPkRgMap, GPkRgNoList, GRgEntries,
	GPkRgFailBuf, GPkRgFailImg, GPkRgFailMap,
	// lector
	LJobs, LSelf, LPassNs, LPassFail, LFromRender, LNoState, LBehind, LNoCpu, LOther, LDropFull, LAheadWaitNs, LCopyBytes,
	// (tej-enq) juego (Enqueue) y lector: coste del gancho en el hilo del juego, despertares, copia y pasada
	JEnqTicks, JEnqN, JWakes, LCopyNs, LCopyLate, LPassEndCopies, WOverlapNs,
	// trabajadores
	WJobs, WNs, WTransNs, WCycles, WDraws, WBehind, WFailed, WStops, WHolesDraw, WHolesOther, WOver, WRepins, WPassDiff,
	WIdleNoCpu, WBiggest,
	WCalls, WAssumed, WFixNs, WCallsMeasured, WTransMeasuredNs,
	WVarIdle, WVarSkip, // (tej-var)
	// (tej F2p) render: los cosibles F1d pasados por las comprobaciones de CommitSegments (solo lectura): paquetes por
	// primer motivo de rechazo (RNames), draws/llamadas que pasan, y la variante sin los efectos de la propia DCB
	GRk0, GRk1, GRk2, GRk3, GRk4, GRk5, GRk6, GRk7, GRk8, GRealDraws, GRealCalls, GRealPk, GRealDrawsPre, GRealPkPre,
	GEffEntries,
	// (tej F2p) render, 1 de cada BeforeEvery paquetes que pasan: lo que Before haria (consultas de solo lectura)
	GBfPk, GBfTicks, GBfRanges, GBfDirtyRanges, GBfDirtyBytes, GBfBda, GBfBdaAll, GBfInv, GBfImg,
	// (tej-cos) embudo en draws (render): a tiempo sin huecos, registros de draws, rangos por tipo; draws por motivo F2p;
	// buffers por rango EXACTO (anillo de BufferCache) frente a regiones de 4 MiB; imagenes solo sobre imagenes (Changed)
	GCosSewn, GCosGfxOk, GCosCsBad, GCosLostBuf, GCosLostImg, GCosLostMap, GCosBufExactOk, GCosBufExactFail,
	GCosBufExactLost, GCosBufRegionOk, GCosImgOnlyOk, GCosImgOverOther, GCosReadGpu, GCosReadHost, GCosRegSampled,
	GRkD0, GRkD1, GRkD2, GRkD3, GRkD4, GRkD5, GRkD6, GRkD7, GRkD8,
	// (tej-cos2) base de la muestra F2p (draws y paquetes F1d de las DCB muestreadas, SIN escalar: el % real es
	// F1d x pasa/base); imagenes y mapeos por rango exacto (anillos); metadatos borrados sobre imagen del paquete o no
	GRealBaseD, GRealBasePk, GCosImgExactOk, GCosImgExactFail, GCosImgExactLost, GCosMapExactOk, GCosMapExactFail,
	GCosMapExactLost, GCosMetaOnImg, GCosMetaOffImg, GCosCsSampled,
	// (tej-cosibles) lecturas limpias que el render escribio (CPU) validadas por VALOR (bytes que la traduccion leyo ==
	// memoria al llegar el render): pasan, distintas, sin bytes guardados; lecturas reenviadas de las DCB anteriores
	// (pasada del lector: la escritura del procesador aun no hecha) por paquete; paquetes con bytes guardados de mas
	GCosValOk, GCosValDiff, GCosValNone, GCosFwdReads, GCosFwdDcb, GCosFwdWrites, GCosSnapOver,
	NCount
};
// (tej-lag) Contadores por hilo, cada banco alineado a 64: el render no comparte linea de cache con la sombra (antes un
// solo array de atomicos: el render y los 6 trabajadores escribian las mismas lineas = false sharing).
struct alignas(64) Bank {
	std::atomic<uint64_t> c[NCount] {};
};
inline constexpr uint32_t Banks = 10; // 0 render, 1 lector, 2..7 trabajadores, 8 juego (Enqueue), 9 resto
inline Bank               g_bank[Banks];
inline thread_local uint32_t t_bank = 9;
inline void Add(C i, uint64_t v = 1) { g_bank[t_bank].c[i].fetch_add(v, std::memory_order_relaxed); }

// (tej-coste) Coste de CADA gancho de tej en el hilo de render (TSC), por lado (0 A, 1 B). Acumula en el hilo (sin
// atomicos por draw) y vuelca por DCB (GpuDone); la linea [tej] coste lo da en ms/fot. Ademas, por DCB de la cola 0, el
// tiempo de pared del render y sus ciclos de CPU (QueryThreadCycleTime): si en B sube la pared y no los ciclos, el render
// espera / es expulsado; si suben los ciclos, calcula mas o mas lento (cache/memoria).
enum H : uint32_t { HSlice, HDraw, HVal, HRanges, HRegDiff, HDispatch, HWait, HFlip, HDone, HPublish, HReal, HBefore, HEff, HN };
inline constexpr const char* HName[HN] = {"inicio DCB", "draw (todo)", "validar regs", "rangos", "DiffState 1/64",
                                          "dispatch", "espera WAIT_REG_MEM", "flip (latencia)", "fin DCB", "publicar estado",
                                          "F2p cosibles reales 1/32 DCB", "F2p Before 1/8",
                                          "registro de efectos (entradas)"};
struct alignas(64) HookAcc {
	std::atomic<uint64_t> t[2][HN] {}, n[2][HN] {};
	std::atomic<uint64_t> dcb_n[2] {}, dcb_wall[2] {}, dcb_cyc[2] {};
};
inline HookAcc g_hook;
struct HookLocal {
	uint64_t t[HN] {}, n[HN] {};
};
inline thread_local HookLocal t_hook;
inline bool                   g_hook_side = false; // (render: lado de la DCB actual)
inline uint64_t               g_dcb_tsc0 = 0, g_dcb_cyc0 = 0;
struct HookTimer {
	H        h;
	uint64_t t0;
	// (tej-lag3) MEDIDA: sin KYTY_TECHO=1 no lee el TSC (t0 = 0, nada que sumar).
	explicit HookTimer(H k): h(k), t0(Techo::On() ? __rdtsc() : 0) {}
	~HookTimer() {
		if (t0 != 0) t_hook.t[h] += __rdtsc() - t0, ++t_hook.n[h];
	}
	HookTimer(const HookTimer&)            = delete;
	HookTimer& operator=(const HookTimer&) = delete;
};
inline void RenderWaitsFlush(); // (tej-lag3, mas abajo)
inline void FlushHooks() {
	RenderWaitsFlush();
	const int sd = g_hook_side ? 1 : 0;
	for (uint32_t i = 0; i < HN; ++i)
		if (t_hook.n[i] != 0) {
			g_hook.t[sd][i].fetch_add(t_hook.t[i], std::memory_order_relaxed);
			g_hook.n[sd][i].fetch_add(t_hook.n[i], std::memory_order_relaxed);
			t_hook.t[i] = t_hook.n[i] = 0;
		}
}

// (tej-enq) Lector: su pasada de estado NO predice dispatches (Spec::t_record se tira: PredictDispatch solo pedia el
// CatalogRead y los IsRegionGpuModified de BufferCache, cerrojos que comparte con el render, para nada).
inline thread_local bool     t_no_predict = false;
inline std::atomic<bool>     g_render_in {false}; // (tej-enq: render dentro de una DCB de la cola 0, para el solape)
inline std::atomic<uint64_t> g_gpu_serial {0};
inline std::atomic<uint32_t> g_render_cpu {0xffffffffu};
inline std::atomic<int64_t>  g_free_x100 {-100};
inline std::atomic<uint64_t> g_free_sum {0}, g_free_n {0};

[[nodiscard]] inline uint64_t Now() { return Speculation::NowNs(); }

// Por serial (anillo): llegada, estado listo, coste por draw del trabajador; flips: primer envio del fotograma.
struct Rec {
	std::atomic<uint64_t> serial {0}, enq {0}, ready {0}, cw_ps {0}, frame_first {0};
	std::atomic<uint8_t>  kind {0};
};
inline constexpr uint32_t RecN = 512;
inline Rec                g_rec[RecN];
inline std::atomic<uint64_t> g_frame_first {0};

struct Packet {
	uint64_t                          key = 0;
	uint32_t                          draws = 0, holes = 0;
	uint64_t                          ep_buf = 0, ep_img = 0, ep_map = 0;
	uint64_t                          publish_tsc = 0;
	bool                              assumed = false; // (tej F1c: traducido tras una espera supuesta cumplida)
	// (tej F1d) dispatches traducidos dentro del paquete; rangos que leyo (buffers, lecturas limpias, imagenes: inicio,
	// fin) desde los 8 ultimos del segmento abierto al abrir (lo que Note no repite) hasta el cierre; lists: los rangos estan.
	uint32_t                                  dispatches = 0;
	bool                                      lists      = false;
	size_t                                    fi_buf = 0, fi_read = 0, fi_img = 0;
	std::vector<std::pair<uint64_t, uint64_t>> bufs, reads, imgs;
	// (tej-coste) Listas compactas que prepara el TRABAJADOR (el render solo las recorre): rb = buffers a regiones de 4 MiB
	// (la granularidad de BufferCache::RegisteredSince: misma respuesta), ordenados y fusionados; rx = buffers + lecturas
	// + imagenes ordenados y fusionados (solapados/contiguos: mismas direcciones, misma respuesta en imagenes y mapeos).
	std::vector<std::pair<uint64_t, uint64_t>> rb, rx;
	uint32_t                                   raw_ranges = 0;
	// (tej F2p) Lo demas que CommitSegments comprueba, copiado por el trabajador al cerrar (el estado de la traduccion
	// se reutiliza): copias de CPU, sets, lecturas de cadena, esperas, hechos de metadatos no borrados; y lo que Before
	// haria (rangos escritos y actuales; BDA, invalidaciones e imagenes de los segmentos: solo cuentas).
	size_t fi_wr = 0, fi_cur = 0, fi_bda = 0, fi_inv = 0, fi_copy = 0, fi_set = 0, fi_chain = 0, fi_wait = 0, seg_open = 0;
	std::vector<std::pair<uint64_t, uint64_t>>      wr, cur, copies;
	std::vector<uint64_t>                           sets;
	std::vector<Libs::Graphics::Spec::ChainRead>    chains;
	std::vector<Libs::Graphics::Spec::WaitCheck>    waits;
	std::vector<std::pair<uint64_t, uint32_t>>      metas;
	uint32_t n_bda = 0, n_inv = 0, n_img = 0;
	bool     bda_all = false;
	// (tej-cos) rbx: TODOS los rangos de buffers de la DCB hasta el cierre, exactos, ordenados y fusionados (sin el corte
	// de los 8 ultimos: una busqueda de antes reutilizada tambien cuenta); ep_buf0: epoca de registro al empezar la DCB
	// (mas estricta que la del paquete); rbx_ok: la lista esta entera. ri: solo imagenes (tocadas + usadas por los
	// segmentos), lo que Speculation::Changed comprueba de imagenes.
	std::vector<std::pair<uint64_t, uint64_t>> rbx, ri;
	// (tej-lag3) rbxr: rbx redondeado a regiones de 4 MiB y fusionado (filtro previo del render, misma prueba).
	std::vector<std::pair<uint64_t, uint64_t>> rbxr;
	uint64_t                                   ep_buf0 = 0;
	bool                                       rbx_ok  = false;
	TejCoser::PkInfo                           coser; // (tej-coser)
	// (tej-cosibles) bytes de las lecturas limpias tal como la traduccion los leyo (direccion, tamano; bytes seguidos)
	std::vector<std::pair<uint64_t, uint32_t>> rv;
	std::vector<uint8_t>                       rvb;
	std::unique_ptr<CommandProcessor> cp;
};
struct Slot {
	std::atomic<uint64_t> serial {0};
	std::atomic<uint64_t> eff_from {0}; // (tej F2p) fin del registro de efectos del render al empezar la traduccion
	std::atomic<uint64_t> ep_buf0 {0};  // (tej-cos) epoca de registro de buffers al empezar la traduccion
	std::atomic<uint32_t> count {0};
	std::atomic<bool>     done {false};
	Packet                pk[MaxPackets];
};

inline thread_local int t_side = 0; // 1 hilo de render (DCB con sombra encendida), 2 trabajador, 3 lector

[[nodiscard]] inline bool IsDrawHeader(uint32_t header) {
	switch ((header >> 8u) & 0xffu) {
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_2:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_AUTO:
		case Libs::Graphics::Pm4::IT_DRAW_INDIRECT:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_INDIRECT:
		case Libs::Graphics::Pm4::IT_DRAW_INDIRECT_MULTI:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

// Autocontenida: CLEAR_STATE en la DCB raiz antes de su primer draw (o reset del procesador).
[[nodiscard]] inline bool SelfContained(std::span<const uint32_t> cmds) {
	for (size_t at = 0; at < cmds.size();) {
		const uint32_t h = cmds[at];
		if ((h >> 30u) == 2u) { ++at; continue; }
		if ((h >> 30u) != 3u) return false;
		if (((h >> 8u) & 0xffu) == Libs::Graphics::Pm4::IT_CLEAR_STATE) return true;
		if (IsDrawHeader(h)) return false;
		at += ((h >> 16u) & 0x3fffu) + 2u;
	}
	return false;
}

[[nodiscard]] inline uint32_t CoresIn(uint64_t mask);
// (tej-lag) KYTY_TEJ_CCD: mismo/0 = L3 del render (como antes); otro/1 = OTRA L3 (en X3D el CCD sin V-Cache: el render se queda
// su L3 entera); por defecto 1 si la otra L3 tiene >= 2 nucleos de rendimiento en la mascara del proceso, si no 0.
[[nodiscard]] inline int CcdPref() {
	static const int v = [] {
		const char* e = std::getenv("KYTY_TEJ_CCD");
		if (e == nullptr) return -1;
		if (std::strcmp(e, "mismo") == 0) return 0; // (simula una CPU de una sola L3)
		if (std::strcmp(e, "otro") == 0) return 1;
		return std::atoi(e) != 0 ? 1 : 0;
	}();
	return v;
}
// (tej-var) La preferencia de CCD de la ventana: variante 3 = mismo.
[[nodiscard]] inline int CcdNow() { return Var() == 3 ? 0 : Var() == 6 ? 1 : CcdPref(); }
// Mascara de los trabajadores: L3 elegida (o KYTY_RENDER_CPUS) & nucleos de rendimiento, sin el nucleo del render.
[[nodiscard]] inline uint64_t WorkMask(uint32_t rc) {
	if (rc >= 64) return 0;
	uint64_t core = LocalPlatform::CoreMaskOfCpu(rc);
	if (core == 0) core = uint64_t {1} << rc;
	const uint64_t perf = LocalPlatform::PerformanceCpuMask();
	uint64_t       base = LocalPlatform::CpuListToMask(std::getenv("KYTY_RENDER_CPUS"));
	if (base == 0) {
		const uint64_t own = LocalPlatform::L3MaskOfCpu(rc);
		base               = own;
		if (CcdNow() != 0 && own != 0) {
			uint64_t idle = 0, proc = 0;
			if (!LocalPlatform::ProcessCpusIdleTime(&idle, &proc) || proc == 0) proc = ~uint64_t {0};
			uint64_t other = proc & ~own;
			if (perf != 0 && (other & perf) != 0) other &= perf;
			if (other != 0 && (CcdNow() == 1 || CoresIn(other) >= 2)) base = other;
		}
	}
	if (perf != 0 && (base & perf) != 0) base &= perf;
	return base & ~core;
}
[[nodiscard]] inline uint32_t CoresIn(uint64_t mask) {
	uint32_t n = 0;
	for (uint32_t c = 0; c < 64; ++c)
		if ((mask >> c) & 1u) {
			const uint64_t core = LocalPlatform::CoreMaskOfCpu(c);
			if (core != 0) mask &= ~core;
			++n;
		}
	return n;
}

struct Item {
	uint64_t                          serial = 0;
	std::vector<uint32_t>             cmds;
	std::unique_ptr<CommandProcessor> start, pass_end;
	bool                              pass_ok = false, has_end = false;
	// (tej-cosibles) escrituras pendientes del procesador de las DCB anteriores aun no procesadas por el render (sus
	// pasadas de estado): la traduccion las lee de aqui (lecturas de cadena, validadas por valor en el render)
	std::unique_ptr<Libs::Graphics::Spec::State> chain;
	bool                                         has_chain = false;
};

class Shadow;

// Un trabajador: su traductor (ranura de paquete y rol de catalogo propios), su foto por paquete.
class Worker {
public:
	Worker(Shadow& owner, RenderContext& r, uint32_t index, uint32_t number)
	    : m_owner(owner), m_r(r), m_index(index), m_spec(r, Speculation::ShadowTag {}, number, 16, 8),
	      m_end(std::make_unique<CommandProcessor>(r, 0)) {}
	void Run();
	void WorkerDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key);
	void Hole(const uint32_t* packet, const char* reason);
	bool Assume(uint64_t addr);
	void Dispatch() {
		++m_calls;
		if (m_slot != nullptr && m_open >= 0) ++m_slot->pk[m_open].dispatches;
	}
	void Close(const Libs::Graphics::Spec::State* st);
	// (tej-cosibles) una lectura limpia de la traduccion con sus bytes (Spec::t_clean_value_hook)
	void Snap(uint64_t address, uint64_t bytes, const void* data) {
		if (m_slot == nullptr) return;
		if (bytes > SnapMax || m_snap_bytes.size() + bytes > SnapDcbMax) {
			m_snap_over = true;
			return;
		}
		m_snaps.push_back({address, static_cast<uint32_t>(bytes), static_cast<uint32_t>(m_snap_bytes.size())});
		const auto* b = static_cast<const uint8_t*>(data);
		m_snap_bytes.insert(m_snap_bytes.end(), b, b + bytes);
	}
	static constexpr uint64_t SnapMax = 256, SnapDcbMax = 256 * 1024;
	std::thread                               thread;
	std::unordered_map<uint64_t, uint64_t>    waits; // (direccion de espera supuesta -> veces; bajo reason_mutex)
	std::mutex                                reason_mutex;
	std::unordered_map<const char*, uint64_t> holes, stops;

private:
	void Close() { Close(Libs::Graphics::Spec::t_state); }
	void Translate(Item& item);
	Shadow&                                   m_owner;
	RenderContext&                            m_r;
	uint32_t                                  m_index;
	Speculation                               m_spec;
	std::unique_ptr<CommandProcessor>         m_end;
	Slot*                                     m_slot = nullptr;
	int                                       m_open = -1;
	uint32_t                                  m_ord = 0, m_pinned_for = 0xffffffffu;
	int                                       m_ccd = -9; // (tej-var: CCD de la ultima afinidad)
	bool                                      m_prio_normal = false; // (tej-lag3)
	uint64_t                                  m_last_key = 0;
	uint32_t                                  m_calls = 0;
	bool                                      m_assumed = false;
	std::unordered_map<const char*, uint64_t> m_job_holes;
	std::unordered_map<uint64_t, uint64_t>    m_job_waits;
	// (tej-cosibles) lecturas limpias de la DCB con sus bytes; desde cual copia el paquete abierto
	struct SnapE {
		uint64_t address;
		uint32_t size, at;
	};
	std::vector<SnapE>   m_snaps;
	std::vector<uint8_t> m_snap_bytes;
	size_t               m_snap_open = 0;
	bool                 m_snap_over = false;
	// (tej-cos) rangos de buffers de la DCB acumulados (fusionados) hasta el ultimo cierre; de que estado y hasta donde
	std::vector<std::pair<uint64_t, uint64_t>> m_acc_bufs;
	const Libs::Graphics::Spec::State*         m_acc_st   = nullptr;
	size_t                                     m_acc_from = 0;
	bool                                       m_acc_ok   = true;
};
inline thread_local Worker* t_worker = nullptr;

class Shadow {
public:
	struct Job {
		enum Kind : uint8_t { Graphics, Flip, Other };
		uint64_t        serial = 0;
		const uint32_t* p      = nullptr; // (tej-enq) la DCB EN SITIO (memoria del juego): nada se copia en el hilo del juego
		uint32_t        n      = 0;
		Kind            kind   = Other;
		bool            reset  = false;
	};

	explicit Shadow(RenderContext& r)
	    : m_r(r), m_reader_spec(r, Speculation::ShadowTag {}, 0, 1, 1), m_chain(std::make_unique<CommandProcessor>(r, 0)),
	      m_published(std::make_unique<CommandProcessor>(r, 0)) {
		m_chain->SetStateOnly(true); // (tej-enq: la pasada corre sobre la cadena misma: 2 copias de estado menos por DCB)
		// N: KYTY_TEJ_N o nucleos de rendimiento de la L3 del render sin el suyo (el render aun no corre: el de este hilo).
		uint32_t n = 0;
		if (const char* v = std::getenv("KYTY_TEJ_N"); v != nullptr) n = static_cast<uint32_t>(std::max(0, std::atoi(v)));
		// (tej-lag) Por defecto 2 (modelo medido: N=2 87.8 %, N=6 88.0 %: N alto solo quema CPU y L3); CPUs modestas:
		// con < 4 nucleos libres en la mascara, 1.
		if (n == 0) {
			const uint32_t cores = CoresIn(WorkMask(LocalPlatform::CurrentCpu()));
			n                    = cores >= 4 ? 2 : 1;
		}
		// (ranuras de paquete: 0 el lector, 1..N los trabajadores; Spec::MaxPacketSlots = 8; con KYTY_SPECULATE se pisan)
		n          = std::clamp<uint32_t>(n, 1, std::min<uint32_t>(MaxWorkers, static_cast<uint32_t>(Libs::Graphics::Spec::MaxPacketSlots) - 1));
		m_n        = n;
		std::printf("[tej] F1b: lector + %u trabajadores (ranuras 0..%u), CCD %s, prioridad por debajo de normal\n", n, n,
		            CcdPref() == 0 ? "mismo, L3 del render (KYTY_TEJ_CCD=mismo)" : CcdPref() == 1 ? "otro (KYTY_TEJ_CCD=otro)" : "otro si hay (auto)");
		for (uint32_t i = 0; i < n; ++i) m_workers.push_back(std::make_unique<Worker>(*this, r, i, i + 1));
		m_reader = std::thread([this] { ReaderRun(); });
		for (auto& w: m_workers) w->thread = std::thread([p = w.get()] { p->Run(); });
	}
	~Shadow() {
		{
			std::lock_guard g(m_mutex);
			m_quit = true;
		}
		m_wake.fetch_add(1, std::memory_order_seq_cst);
		m_wake.notify_all();
		m_work_cv.notify_all();
		if (m_reader.joinable()) m_reader.join();
		for (auto& w: m_workers)
			if (w->thread.joinable()) w->thread.join();
	}
	Shadow(const Shadow&)            = delete;
	Shadow& operator=(const Shadow&) = delete;

	[[nodiscard]] uint32_t N() const { return m_n; }

	// Hilo del juego (Enqueue, bajo m_queue_mutex de GuestGpu: un solo productor a la vez). (tej-enq) Nada de copia, ni
	// cerrojo, ni condvar: un descriptor (puntero + tamano + serial) a un anillo SPSC y, solo si el lector duerme, un
	// despertar (WakeByAddressSingle). Ghidra: el juego no reescribe la DCB hasta que la GPU firma la etiqueta de su
	// fotograma (FUN_00904090), y esa etiqueta va en la ultima DCB del fotograma: mientras el render no haya EMPEZADO esta
	// DCB, su memoria es la que el juego envio. El lector la copia y luego comprueba eso (serial del render < el suyo).
	void Push(uint64_t serial, Job::Kind kind, bool reset, std::span<const uint32_t> commands) {
		t_bank            = 8;
		const uint64_t hd = m_head.load(std::memory_order_relaxed);
		if (hd - m_tail.load(std::memory_order_acquire) >= RingN) {
			Add(LDropFull);
			return;
		}
		Job& job   = m_ring[hd % RingN];
		job.serial = serial;
		job.kind   = kind;
		job.reset  = reset;
		job.p      = kind == Job::Graphics ? commands.data() : nullptr;
		job.n      = kind == Job::Graphics ? static_cast<uint32_t>(commands.size()) : 0u;
		m_head.store(hd + 1, std::memory_order_seq_cst);
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			m_wake.fetch_add(1, std::memory_order_seq_cst);
			m_wake.notify_one();
			Add(JWakes);
		}
	}

	// Hilo de render, al acabar un envio de la cola 0.
	void GpuDone(uint64_t serial, const CommandProcessor& cp) {
		if (m_want_serial.load(std::memory_order_acquire) != serial) return;
		// (tej-lag) El render NUNCA espera al lector: ocupado (copiando el publicado anterior) -> no publica (el lector
		// lo cuenta como 'sin estado' y la DCB va sin sombra).
		std::unique_lock g(m_pub_mutex, std::try_to_lock);
		if (!g.owns_lock()) {
			TejLag::g_acc[TejLag::Publish].n.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		const HookTimer tm(HPublish);
		m_published->CopyStateFrom(cp, false);
		m_pub_serial = serial;
		m_want_serial.store(0, std::memory_order_relaxed);
		Add(GPublish);
	}

	[[nodiscard]] Slot& SlotOf(uint64_t serial) { return m_slots[serial % Slots]; }

	// Trabajador k: el siguiente Item (la DCB mas grande si hay mas en cola que trabajadores; null: salir).
	std::unique_ptr<Item> Take(uint32_t k) {
		std::unique_lock lock(m_mutex);
		for (;;) {
			if (m_quit) return nullptr;
			// (descarta lo que el render ya empezo)
			const uint64_t gpu = g_gpu_serial.load(std::memory_order_acquire);
			for (auto it = m_items.begin(); it != m_items.end();)
				if ((*it)->serial <= gpu) {
					Add(WBehind);
					Recycle(std::move(*it));
					it = m_items.erase(it);
				} else {
					++it;
				}
			if (!m_items.empty()) {
				if (k >= 1 && (Var() == 2 || Var() == 6 || g_mem_lvl.load(std::memory_order_relaxed) >= 1)) {
					Add(WVarIdle); // (tej-var N=1: los demas no toman)
				} else if (static_cast<int64_t>(k + 1) * 100 > g_free_x100.load(std::memory_order_relaxed) &&
				    g_free_x100.load(std::memory_order_relaxed) >= 0) {
					Add(WIdleNoCpu); // (no hay nucleo libre para este trabajador: lo deja a otro / nadie)
				} else {
					auto best = m_items.begin();
					if (m_items.size() > 1) {
						for (auto it = m_items.begin(); it != m_items.end(); ++it)
							if ((*it)->cmds.size() > (*best)->cmds.size()) best = it;
						if (best != m_items.begin()) Add(WBiggest);
					}
					auto item = std::move(*best);
					m_items.erase(best);
					return item;
				}
			}
			m_work_cv.wait_for(lock, std::chrono::milliseconds(2));
		}
	}
	void Recycle(std::unique_ptr<Item> item) {
		if (item && m_spare.size() < MaxQueued) m_spare.push_back(std::move(item));
	}
	void Give(std::unique_ptr<Item> item) {
		std::lock_guard g(m_mutex);
		Recycle(std::move(item));
	}
	[[nodiscard]] bool Quit() {
		std::lock_guard g(m_mutex);
		return m_quit;
	}

	void TakeReasons(std::vector<std::pair<const char*, uint64_t>>& holes, std::vector<std::pair<const char*, uint64_t>>& stops,
	                 std::unordered_map<uint64_t, uint64_t>& waits) {
		std::unordered_map<const char*, uint64_t> h, s;
		for (auto& w: m_workers) {
			std::lock_guard g(w->reason_mutex);
			for (const auto& [k, v]: w->holes) h[k] += v;
			for (const auto& [k, v]: w->stops) s[k] += v;
			for (const auto& [k, v]: w->waits) waits[k] += v;
			w->holes.clear();
			w->stops.clear();
			w->waits.clear();
		}
		holes.assign(h.begin(), h.end());
		stops.assign(s.begin(), s.end());
	}

	// Muestra de CPU libre (lector, cada 100 ms).
	void SampleCpu() {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_sample_at < std::chrono::milliseconds(100)) return;
		uint64_t idle = 0, mask = 0;
		if (!LocalPlatform::ProcessCpusIdleTime(&idle, &mask)) {
			g_free_x100.store(-100, std::memory_order_relaxed); // (sin dato: no se limita)
			m_sample_at = now;
			return;
		}
		if (m_last_idle != 0 && idle >= m_last_idle) {
			const double wall_100ns = std::chrono::duration<double>(now - m_sample_at).count() * 1e7;
			const double free       = wall_100ns > 0 ? static_cast<double>(idle - m_last_idle) / wall_100ns : 0;
			g_free_x100.store(static_cast<int64_t>(free * 100), std::memory_order_relaxed);
			g_free_sum.fetch_add(static_cast<uint64_t>(free * 100), std::memory_order_relaxed);
			g_free_n.fetch_add(1, std::memory_order_relaxed);
		}
		m_last_idle = idle;
		m_sample_at = now;
	}

private:
	void ReaderRun() {
		LocalPlatform::SetThreadName("Kyty.Tej.L");
		LocalPlatform::MakeBackgroundThread(nullptr); // (tej-lag: por debajo del render, como los trabajadores)
		t_bank = 1;
		TejLag::t_shadow = true;
		m_reader_spec.ShadowThreadInit(); // (executors y ranura propios: la pasada de estado predice dispatches con el catalogo)
		t_side       = 3;
		t_no_predict = true;
		for (;;) {
			if (Quit()) break;
			const uint64_t tl = m_tail.load(std::memory_order_relaxed);
			if (m_head.load(std::memory_order_acquire) == tl) {
				// (sin trabajo: duerme hasta el siguiente Push; Dekker con m_sleeping/m_head, todo seq_cst)
				const uint32_t seq = m_wake.load(std::memory_order_seq_cst);
				m_sleeping.store(true, std::memory_order_seq_cst);
				if (m_head.load(std::memory_order_seq_cst) == tl && !Quit()) m_wake.wait(seq, std::memory_order_seq_cst);
				m_sleeping.store(false, std::memory_order_seq_cst);
				continue;
			}
			const Job job = m_ring[tl % RingN];
			m_tail.store(tl + 1, std::memory_order_release);
			SampleCpu();
			std::unique_ptr<Item> item;
			{
				std::lock_guard g(m_mutex);
				if (!m_spare.empty()) item = std::move(m_spare.back()), m_spare.pop_back();
			}
			if (!item) item = std::make_unique<Item>();
			if (Handle(job, *item)) {
				{
					std::lock_guard g(m_mutex);
					m_items.push_back(std::move(item));
				}
				m_work_cv.notify_all();
			} else {
				std::lock_guard g(m_mutex);
				Recycle(std::move(item));
			}
		}
		t_side = 0;
	}

	// Lector: el estado inicial de la DCB y la pasada de estado que deja el de la siguiente; true: a los trabajadores.
	bool Handle(const Job& job, Item& item) {
		const bool consecutive = m_chain_ok && m_chain_serial + 1 == job.serial;
		if (job.kind == Job::Flip) {
			if (consecutive) m_chain_serial = job.serial;
			else m_chain_ok = false;
			return false;
		}
		if (job.kind == Job::Other) {
			m_chain_ok = false;
			Add(LOther);
			return false;
		}
		Add(LJobs);
		// (tej-lag) El lector, con los trabajadores (fuera del nucleo del render y, en X3D, fuera de su CCD).
		if (const uint32_t rc = g_render_cpu.load(std::memory_order_relaxed);
		    rc < 64 && (rc != m_reader_pinned_for || CcdNow() != m_reader_ccd)) {
			m_reader_pinned_for = rc;
			m_reader_ccd        = CcdNow();
			if (const uint64_t m = WorkMask(rc); m != 0 && LocalPlatform::SetCurrentThreadMask(m)) Add(WRepins);
		}
		uint64_t gpu = g_gpu_serial.load(std::memory_order_acquire);
		if (job.serial <= gpu) {
			m_chain_ok = false;
			Add(LBehind);
			return false;
		}
		if (job.serial > gpu + MaxAhead) {
			const auto t0 = Now();
			while (job.serial > gpu + MaxAhead && !Quit()) {
				std::this_thread::sleep_for(std::chrono::microseconds(200));
				gpu = g_gpu_serial.load(std::memory_order_acquire);
			}
			Add(LAheadWaitNs, Now() - t0);
			if (Quit()) return false;
		}
		if (g_free_x100.load(std::memory_order_relaxed) >= 0 && g_free_x100.load(std::memory_order_relaxed) < 100) {
			m_chain_ok = false; // (ni un nucleo libre: nada, coste 0)
			Add(LNoCpu);
			return false;
		}
		// (tej-enq) La copia, aqui (lector), no en el hilo del juego; valida si el render aun no empezo esta DCB.
		{
			const uint64_t c0 = Now();
			item.cmds.assign(job.p, job.p + job.n);
			Add(LCopyNs, Now() - c0);
			Add(LCopyBytes, uint64_t {job.n} * 4u);
			if (g_gpu_serial.load(std::memory_order_seq_cst) >= job.serial) {
				m_chain_ok = false;
				Add(LCopyLate);
				return false;
			}
		}
		const bool self = job.reset || SelfContained(item.cmds);
		if (self) Add(LSelf);
		if (!consecutive && !self) {
			m_want_serial.store(job.serial - 1, std::memory_order_release);
			bool got = false;
			for (;;) {
				{
					std::lock_guard g(m_pub_mutex);
					if (m_pub_serial == job.serial - 1) {
						m_chain->CopyStateFrom(*m_published, false);
						got = true;
					}
				}
				if (got || Quit() || g_gpu_serial.load(std::memory_order_acquire) >= job.serial) break;
				std::this_thread::sleep_for(std::chrono::microseconds(100));
			}
			if (!got) {
				m_want_serial.store(0, std::memory_order_relaxed);
				m_chain_ok = false;
				Add(LNoState);
				return false;
			}
			Add(LFromRender);
		}
		if (job.reset) m_chain->Reset();
		m_chain->ResetDeCe();
		if (!item.start) item.start = std::make_unique<CommandProcessor>(m_r, 0);
		if (!item.pass_end) item.pass_end = std::make_unique<CommandProcessor>(m_r, 0);
		item.start->CopyStateFrom(*m_chain, false);
		// Pasada de estado (solo registros; escrituras a memoria a un registro propio que nadie lee). (tej-enq) Sobre la
		// cadena misma (antes: copia a m_pass y vuelta), sin predecir dispatches (t_no_predict).
		const auto t0 = Now();
		m_chain->SetFlip({});
		m_chain->SetChunking(0);
		m_record.Clear();
		Libs::Graphics::Spec::t_record = &m_record;
		Pm4Execution execution;
		const auto   res               = m_chain->Process(execution, std::span<const uint32_t>(item.cmds.data(), item.cmds.size()));
		Libs::Graphics::Spec::t_record = nullptr;
		Add(LPassNs, Now() - t0);
		item.pass_ok  = res == Pm4ProcessResult::Complete;
		item.has_end  = false;
		ChainFor(job.serial, item); // (tej-cosibles: antes de guardar las de esta, solo las anteriores)
		KeepWrites(job.serial);
		if (item.pass_ok) {
			// (el estado final de la pasada solo para el diagnostico 'estado distinto de la pasada': 1 de cada 16 DCB)
			if ((++m_end_k & 15u) == 0) {
				item.pass_end->CopyStateFrom(*m_chain, false);
				item.has_end = true;
				Add(LPassEndCopies);
			}
			m_chain_serial = job.serial;
			m_chain_ok     = true;
		} else {
			m_chain_ok = false; // (la siguiente pide el estado al render)
			Add(LPassFail);
		}
		item.serial = job.serial;
		if (auto& rec = g_rec[job.serial % RecN]; rec.serial.load(std::memory_order_relaxed) == job.serial)
			rec.ready.store(Now(), std::memory_order_relaxed);
		return true;
	}

	// (tej-cosibles) Escrituras del procesador de cada pasada (WRITE_DATA, rellenos pequenos: no copias), por serial;
	// solo tramos contiguos >= 64 bytes (las etiquetas de fin de tuberia, 4-8 bytes, se escriben al acabar la GPU y no al
	// procesar: reenviarlas haria fallar la validacion por valor). 3184 bytes medidos (COS, 0x269c83000).
	struct PassW {
		uint64_t                                     serial = 0;
		std::vector<Libs::Graphics::Spec::MemoryWrite> w;
	};
	static constexpr uint32_t PassWN = 16, FwdMinBytes = 64;
	PassW                     m_passw[PassWN];
	void KeepWrites(uint64_t serial) {
		auto& k  = m_passw[serial % PassWN];
		k.serial = serial;
		k.w.clear();
		const auto& w = m_record.writes;
		for (size_t i = 0; i < w.size();) {
			size_t   j     = i + 1;
			uint64_t bytes = w[i].copy ? 0 : w[i].size;
			while (j < w.size() && !w[j].copy && !w[i].copy && w[j].address == w[j - 1].address + w[j - 1].size) bytes += w[j++].size;
			if (!w[i].copy && bytes >= FwdMinBytes) k.w.insert(k.w.end(), w.begin() + i, w.begin() + j);
			i = j;
		}
	}
	void ChainFor(uint64_t serial, Item& item) {
		item.has_chain = false;
		const uint64_t gpu = g_gpu_serial.load(std::memory_order_acquire);
		uint64_t       n   = 0;
		for (uint64_t s = std::max(gpu + 1, serial > PassWN ? serial - PassWN + 1 : 1); s < serial; ++s) {
			const auto& k = m_passw[s % PassWN];
			if (k.serial != s || k.w.empty()) continue;
			if (!item.has_chain) {
				if (!item.chain) item.chain = std::make_unique<Libs::Graphics::Spec::State>();
				item.chain->Clear();
				item.has_chain = true;
				Add(GCosFwdDcb);
			}
			for (const auto& x: k.w) item.chain->Journal(x.address, x.value, x.size);
			n += k.w.size();
		}
		Add(GCosFwdWrites, n);
	}
	RenderContext&                        m_r;
	Speculation                           m_reader_spec;
	std::unique_ptr<CommandProcessor>     m_chain, m_published;
	Libs::Graphics::Spec::State           m_record;
	uint32_t                              m_n = 1;
	std::vector<std::unique_ptr<Worker>>  m_workers;
	std::thread                           m_reader;
	std::mutex                            m_mutex;
	std::condition_variable               m_work_cv;
	// (tej-enq) anillo SPSC juego -> lector (productor bajo m_queue_mutex del GuestGpu), lineas de cache separadas
	static constexpr uint64_t             RingN = 64;
	Job                                   m_ring[RingN];
	alignas(64) std::atomic<uint64_t>     m_head {0};
	alignas(64) std::atomic<uint64_t>     m_tail {0};
	alignas(64) std::atomic<bool>         m_sleeping {false};
	std::atomic<uint32_t>                 m_wake {0};
	uint32_t                              m_end_k = 0;
	std::deque<std::unique_ptr<Item>>     m_items;
	std::vector<std::unique_ptr<Item>>    m_spare;
	bool                                  m_quit = false;
	bool                                  m_chain_ok = false;
	uint64_t                              m_chain_serial = 0, m_last_idle = 0;
	uint32_t                              m_reader_pinned_for = 0xffffffffu;
	int                                   m_reader_ccd        = -9;
	std::chrono::steady_clock::time_point m_sample_at {};
	std::mutex                            m_pub_mutex;
	uint64_t                              m_pub_serial = 0;
	std::atomic<uint64_t>                 m_want_serial {0};
	Slot                                  m_slots[Slots];
};

inline std::atomic<Shadow*> g_shadow {nullptr};

inline void Worker::Run() {
	char name[32];
	std::snprintf(name, sizeof(name), "Kyty.Tej.%u", m_index);
	LocalPlatform::SetThreadName(name);
	LocalPlatform::MakeBackgroundThread(nullptr); // (por debajo del render)
	m_spec.ShadowThreadInit();
	TejLag::t_shadow = true;
	t_bank = 2 + (m_index % 6u);
	Libs::Graphics::Spec::t_hole_hook = [](const uint32_t* packet, const char* reason) {
		if (t_worker != nullptr && t_side == 2) t_worker->Hole(packet, reason);
	};
	Libs::Graphics::Spec::t_shadow_end_hook = [](const Libs::Graphics::Spec::State* st) {
		if (t_worker != nullptr && t_side == 2) t_worker->Close(st); // (F1d: el ultimo paquete con sus rangos)
	};
	Libs::Graphics::Spec::t_clean_value_hook = [](uint64_t address, uint64_t bytes, const void* data) {
		if (t_worker != nullptr && t_side == 2) t_worker->Snap(address, bytes, data); // (tej-cosibles)
	};
	t_side   = 2;
	t_worker = this;
	while (auto item = m_owner.Take(m_index)) {
		Translate(*item);
		m_owner.Give(std::move(item));
	}
	t_side   = 0;
	t_worker = nullptr;
}

inline void Worker::Translate(Item& item) {
	if (const uint32_t rc = g_render_cpu.load(std::memory_order_relaxed); rc < 64 && (rc != m_pinned_for || CcdNow() != m_ccd)) {
		m_pinned_for = rc;
		m_ccd        = CcdNow();
		if (const uint64_t m = WorkMask(rc); m != 0 && LocalPlatform::SetCurrentThreadMask(m)) Add(WRepins);
	}
	// (tej-lag3) variante 6: prioridad normal mientras dure (por debajo de normal en las demas).
	if (const bool normal = Var() == 6; normal != m_prio_normal) {
		m_prio_normal = normal;
		LocalPlatform::SetThreadNormalPriority(LocalPlatform::ThreadId(), normal);
	}
	auto& slot = m_owner.SlotOf(item.serial);
	slot.serial.store(0, std::memory_order_release);
	slot.count.store(0, std::memory_order_release);
	slot.done.store(false, std::memory_order_release);
	slot.eff_from.store(g_eff.End(), std::memory_order_relaxed);
	slot.ep_buf0.store(m_r.GetBufferCache().RegistrationEpoch(), std::memory_order_relaxed);
	m_acc_bufs.clear(), m_acc_st = nullptr, m_acc_from = 0, m_acc_ok = true;
	slot.serial.store(item.serial, std::memory_order_release);
	m_slot     = &slot;
	m_open     = -1;
	m_ord      = 0;
	m_last_key = ~uint64_t {0};
	m_calls    = 0;
	m_assumed  = false;
	m_job_holes.clear();
	m_job_waits.clear();
	m_snaps.clear(), m_snap_bytes.clear(), m_snap_open = 0, m_snap_over = false; // (tej-cosibles)
	if (Var() == 1) { // (tej-var sin traducir: la DCB llega y se descarta; el render no ve paquetes)
		slot.done.store(true, std::memory_order_release);
		m_slot = nullptr;
		Add(WVarSkip);
		return;
	}
	const uint64_t c0  = LocalPlatform::CurrentThreadCycles();
	const auto     t0  = Now();
	const bool     render_in0 = g_render_in.load(std::memory_order_relaxed);
	TejLag::t_translating = true; // (tej-sinmutex: aqui una RecorderRefusal 'sin mutex' es hueco)
	const auto     out = m_spec.Shadow(*item.start, std::span<const uint32_t>(item.cmds.data(), item.cmds.size()), *m_end,
	                                   item.has_chain ? item.chain.get() : nullptr); // (tej-cosibles)
	TejLag::t_translating = false;
	const auto     t1  = Now();
	// (tej-enq) solape con el render: render dentro de una DCB al empezar y al acabar la traduccion (estimacion)
	if (render_in0 && g_render_in.load(std::memory_order_relaxed)) Add(WOverlapNs, t1 - t0);
	Close();
	slot.done.store(true, std::memory_order_release);
	m_slot = nullptr;
	Add(WCycles, LocalPlatform::CurrentThreadCycles() - c0);
	Add(WJobs);
	Add(WNs, t1 - t0);
	Add(WTransNs, out.translate_ns);
	Add(WDraws, m_ord);
	// (tej F1c) Llamadas = draws + dispatches: el coste por llamada es el del render (1.8 us/llamada con dispatches);
	// fijo = Shadow menos Translate (copia de estado final y Finish), una vez por DCB.
	const uint32_t calls = m_ord + m_calls;
	Add(WCalls, calls);
	Add(WFixNs, (t1 - t0) - out.translate_ns);
	if (calls >= 32) Add(WCallsMeasured, calls), Add(WTransMeasuredNs, out.translate_ns);
	if (out.failed) Add(WFailed);
	if (out.stop != nullptr) Add(WStops);
	// (la pasada de estado del lector frente a la traduccion: el estado que deja cada una)
	if (item.pass_ok && item.has_end && !out.failed && out.stop == nullptr && !m_end->SameState(*item.pass_end)) Add(WPassDiff);
	// (tej F1c) Coste por LLAMADA traducida (antes: tiempo total, con el fijo, entre draws solo: con la DCB cortada en
	// la espera del principio daba 19.8 us/draw). Solo DCB con >= 32 llamadas (las cortas son casi todo fijo).
	if (calls >= 32)
		if (auto& rec = g_rec[item.serial % RecN]; rec.serial.load(std::memory_order_relaxed) == item.serial)
			rec.cw_ps.store(out.translate_ns * 1000u / calls, std::memory_order_relaxed);
	std::lock_guard g(reason_mutex);
	for (const auto& [why, n]: m_job_holes) holes[why] += n;
	for (const auto& [a, n]: m_job_waits) waits[a] += n;
	if (out.stop != nullptr) ++stops[out.stop];
}

// (tej-coste) Trabajador: ordena y fusiona (rangos vacios fuera). gran: alinear a esa granularidad antes (0: exacto).
inline void Merge(std::vector<std::pair<uint64_t, uint64_t>>& v, uint64_t gran) {
	size_t k = 0;
	for (auto [a, e]: v) {
		if (e <= a) continue;
		if (gran != 0) a -= a % gran, e = (e + gran - 1) / gran * gran;
		v[k++] = {a, e};
	}
	v.resize(k);
	std::sort(v.begin(), v.end());
	k = 0;
	for (size_t i = 0; i < v.size(); ++i) {
		if (k != 0 && v[i].first <= v[k - 1].second) v[k - 1].second = std::max(v[k - 1].second, v[i].second);
		else v[k++] = v[i];
	}
	v.resize(k);
}
inline void Compact(Packet& p) {
	p.raw_ranges = static_cast<uint32_t>(p.bufs.size() + p.reads.size() + p.imgs.size());
	p.rb.assign(p.bufs.begin(), p.bufs.end());
	Merge(p.rb, Libs::Graphics::TRACKER_REGION_SIZE);
	p.rx.clear();
	p.rx.insert(p.rx.end(), p.bufs.begin(), p.bufs.end());
	p.rx.insert(p.rx.end(), p.reads.begin(), p.reads.end());
	p.rx.insert(p.rx.end(), p.imgs.begin(), p.imgs.end());
	Merge(p.rx, 0);
	p.ri.assign(p.imgs.begin(), p.imgs.end()); // (tej-cos)
	Merge(p.ri, 0);
}

inline void Worker::Close(const Libs::Graphics::Spec::State* st) {
	if (m_slot == nullptr || m_open < 0) return;
	TejCoser::WorkerClose(m_slot->pk[m_open].coser); // (tej-coser: fin de la sonda del paquete)
	{
		// (F1d) Lo que el paquete leyo: rangos de buffers, lecturas limpias e imagenes tocadas. Sin estado: sin lista.
		auto& p = m_slot->pk[m_open];
		p.bufs.clear(), p.reads.clear(), p.imgs.clear();
		p.lists = st != nullptr && p.fi_buf != ~size_t {0};
		if (p.lists) {
			for (size_t i = std::min(p.fi_buf, st->buffer_ranges.size()); i < st->buffer_ranges.size(); ++i)
				p.bufs.emplace_back(st->buffer_ranges[i].begin, st->buffer_ranges[i].end);
			for (size_t i = std::min(p.fi_read, st->clean_reads.size()); i < st->clean_reads.size(); ++i)
				p.reads.emplace_back(st->clean_reads[i].begin, st->clean_reads[i].end);
			// (tej-cosibles) sus bytes (si alguno no cupo, ninguno: el render no valida por valor)
			p.rv.clear(), p.rvb.clear();
			if (!m_snap_over && !p.reads.empty())
				for (size_t i = std::min(m_snap_open, m_snaps.size()); i < m_snaps.size(); ++i) {
					const auto& x = m_snaps[i];
					p.rv.emplace_back(x.address, x.size);
					p.rvb.insert(p.rvb.end(), m_snap_bytes.begin() + x.at, m_snap_bytes.begin() + x.at + x.size);
				}
			if (m_snap_over) Add(GCosSnapOver);
			for (size_t i = std::min(p.fi_img, st->touched_images.size()); i < st->touched_images.size(); ++i) {
				const auto* im = st->touched_images[i].first;
				if (im != nullptr) p.imgs.emplace_back(im->info.data.address, im->info.data.address + im->info.data.size);
			}
			// (F2p) Lo demas de CommitSegments / Before (desde lo que habia al abrir; Note no repite los 8 ultimos)
			const auto cp2 = [](auto& out, const auto& in, size_t from, auto f) {
				out.clear();
				for (size_t i = std::min(from, in.size()); i < in.size(); ++i) out.push_back(f(in[i]));
			};
			const auto rg = [](const auto& x) { return std::pair<uint64_t, uint64_t> {x.begin, x.end}; };
			cp2(p.wr, st->written, p.fi_wr, rg);
			cp2(p.cur, st->current, p.fi_cur, rg);
			cp2(p.copies, st->cpu_copies, p.fi_copy, rg);
			cp2(p.sets, st->sets, p.fi_set, [](uint64_t x) { return x; });
			cp2(p.chains, st->chain_reads, p.fi_chain, [](const auto& x) { return x; });
			if (!p.chains.empty()) Add(GCosFwdReads, p.chains.size()); // (tej-cosibles)
			cp2(p.waits, st->waits, p.fi_wait, [](const auto& x) { return x; });
			p.n_bda = static_cast<uint32_t>(st->bda.size() - std::min(p.fi_bda, st->bda.size()));
			p.n_inv = static_cast<uint32_t>(st->invalidated.size() - std::min(p.fi_inv, st->invalidated.size()));
			p.bda_all = st->bda_all;
			p.n_img   = 0;
			for (size_t k = std::min(p.seg_open, st->closed.size()); k < st->closed.size(); ++k) {
				p.n_img += static_cast<uint32_t>(st->closed[k].images.size());
				p.bda_all = p.bda_all || st->closed[k].bda_all;
			}
			p.n_img += static_cast<uint32_t>(st->used.size()); // (las del segmento abierto)
			// (tej-cos) las imagenes que usan los segmentos del paquete (Changed: StillResolved) y las del abierto
			for (size_t k = std::min(p.seg_open, st->closed.size()); k < st->closed.size(); ++k)
				for (const auto& use: st->closed[k].images)
					if (use.image != nullptr)
						p.imgs.emplace_back(use.image->info.data.address, use.image->info.data.address + use.image->info.data.size);
			for (const auto idx: st->used)
				if (idx < st->overlay.size() && st->overlay[idx].image != nullptr) {
					const auto* im = st->overlay[idx].image;
					p.imgs.emplace_back(im->info.data.address, im->info.data.address + im->info.data.size);
				}
			// (tej-cos) todos los buffers de la DCB hasta aqui (mismo estado desde el primer paquete; si no, sin lista)
			if (m_acc_st == nullptr) m_acc_st = st;
			if (m_acc_st != st || m_acc_from > st->buffer_ranges.size()) m_acc_ok = false;
			if (m_acc_ok) {
				for (size_t i = m_acc_from; i < st->buffer_ranges.size(); ++i)
					m_acc_bufs.emplace_back(st->buffer_ranges[i].begin, st->buffer_ranges[i].end);
				m_acc_from = st->buffer_ranges.size();
				Merge(m_acc_bufs, 0);
			}
			p.rbx_ok = m_acc_ok;
			if (m_acc_ok) {
				p.rbx.assign(m_acc_bufs.begin(), m_acc_bufs.end());
				p.rbxr.assign(m_acc_bufs.begin(), m_acc_bufs.end()); // (tej-lag3: sus regiones de 4 MiB)
				Merge(p.rbxr, Libs::Graphics::TRACKER_REGION_SIZE);
			} else {
				p.rbx.clear(), p.rbxr.clear();
			}
			p.ep_buf0 = m_slot->ep_buf0.load(std::memory_order_relaxed);
			const size_t seg_close = st->closed.size();
			p.metas.clear();
			for (const auto& fact: st->meta_facts)
				if (!fact.cleared && fact.first <= seg_close && fact.last >= p.seg_open) p.metas.emplace_back(fact.address, fact.slice);
		} else {
			p.wr.clear(), p.cur.clear(), p.copies.clear(), p.sets.clear(), p.chains.clear(), p.waits.clear(), p.metas.clear();
			p.n_bda = p.n_inv = p.n_img = 0;
			p.rbx.clear(), p.rbxr.clear(), p.rbx_ok = false;
		}
		Compact(p);
	}
	m_slot->pk[m_open].publish_tsc = __rdtsc();
	m_slot->count.store(static_cast<uint32_t>(m_open) + 1, std::memory_order_release);
	m_open = -1;
}

inline void Worker::WorkerDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
	if (m_slot == nullptr || key == m_last_key) return;
	m_last_key     = key;
	const auto ord = m_ord++;
	if (ord % PacketDraws == 0) {
		if (m_slot != nullptr && m_open >= 0) m_slot->pk[m_open].coser.end_key = key; // (tej-coser)
		Close();
		const uint32_t j = ord / PacketDraws;
		if (j >= MaxPackets) {
			Add(WOver);
			return;
		}
		auto& p = m_slot->pk[j];
		if (!p.cp) p.cp = std::make_unique<CommandProcessor>(m_r, 0);
		p.cp->CopyStateFrom(cp, false);
		p.key    = key;
		p.draws  = 0;
		p.holes  = 0;
		p.ep_buf = r.GetBufferCache().RegistrationEpoch();
		p.ep_img = r.GetTextureCache().ResolutionEpoch();
		p.ep_map  = r.GetGpuResources().MappingEpoch();
		p.assumed = m_assumed;
		p.dispatches = 0;
		p.lists      = false;
		if (const auto* st = Libs::Graphics::Spec::t_state; st != nullptr) {
			// (Note no repite un rango que ya esta entre los 8 ultimos del segmento abierto: desde ahi)
			const auto sealed = st->Sealed();
			const auto back8  = [](size_t size, size_t seal) { return std::max<size_t>(seal, size >= 8 ? size - 8 : 0); };
			p.fi_buf  = back8(st->buffer_ranges.size(), sealed.buffer_ranges);
			p.fi_read = back8(st->clean_reads.size(), sealed.clean_reads);
			p.fi_img  = back8(st->touched_images.size(), sealed.touched_images);
			// (F2p) los que Note deduplica, igual; los que se apilan sin mas, exactos
			p.fi_wr    = back8(st->written.size(), sealed.written);
			p.fi_cur   = back8(st->current.size(), sealed.current);
			p.fi_inv   = back8(st->invalidated.size(), sealed.invalidated);
			p.fi_copy  = back8(st->cpu_copies.size(), sealed.cpu_copies);
			p.fi_set   = st->sets.empty() ? 0 : st->sets.size() - 1;
			p.fi_bda   = st->bda.size();
			p.fi_chain = st->chain_reads.size();
			p.fi_wait  = st->waits.size();
			p.seg_open = st->closed.size();
		} else {
			p.fi_buf = p.fi_read = p.fi_img = ~size_t {0};
		}
		m_open    = static_cast<int>(j);
		// (tej-cosibles) los bytes desde 64 lecturas antes (las 8 ultimas de Note que el paquete hereda, de sobra)
		m_snap_open = m_snaps.size() > 64 ? m_snaps.size() - 64 : 0;
		TejCoser::WorkerOpen(p.coser); // (tej-coser: herencia de rendering + sonda desde aqui)
	}
	if (m_open >= 0) ++m_slot->pk[m_open].draws;
}

inline bool Worker::Assume(uint64_t addr) {
	if (m_slot == nullptr) return false;
	m_assumed = true;
	if (m_open >= 0) m_slot->pk[m_open].assumed = true;
	++m_job_waits[addr];
	Add(WAssumed);
	return true;
}

inline void Worker::Hole(const uint32_t* packet, const char* reason) {
	const bool draw = packet != nullptr && IsDrawHeader(*packet);
	if (draw && m_open >= 0 && m_slot != nullptr) ++m_slot->pk[m_open].holes;
	Add(draw ? WHolesDraw : WHolesOther);
	if (draw) ++m_job_holes[reason != nullptr ? reason : "?"];
}

inline void Window(double busy_ms, double fps, double frames, bool keep);

inline void Start(RenderContext& r) {
	if (Mode() == 0 || g_shadow.load(std::memory_order_relaxed) != nullptr) return;
	// (tej-lag) CPUs modestas: con menos de 2 nucleos fuera del render (y su hermano SMT), la sombra no arranca.
	if (std::getenv("KYTY_TEJ_N") == nullptr && CoresIn(WorkMask(LocalPlatform::CurrentCpu())) < 2) {
		std::printf("[tej] sombra NO arranca: menos de 2 nucleos libres fuera del render (coste 0)\n");
		std::fflush(stdout);
		return;
	}
	g_shadow.store(new Shadow(r), std::memory_order_release);
	Techo::g_tej_window = &Window;
}
inline void Stop() {
	Techo::g_tej_window = nullptr;
	if (auto* s = g_shadow.exchange(nullptr, std::memory_order_acq_rel)) delete s;
}

// Latencia y picos (hilo de render), por lado A/B: histograma en pasos de 0.1 ms hasta 100 ms.
struct LatAcc {
	uint64_t n = 0, sum_ns = 0, frames = 0, over25 = 0;
	uint32_t hist[1001] {};
	void     Lat(uint64_t ns) {
        ++n, sum_ns += ns;
        ++hist[std::min<uint64_t>(ns / 100000u, 1000u)];
	}
	[[nodiscard]] double P99() const {
		uint64_t want = n - n / 100, acc = 0;
		for (uint32_t i = 0; i <= 1000; ++i)
			if ((acc += hist[i]) >= want && n != 0) return (i + 1) * 0.1;
		return 0;
	}
};
inline std::mutex g_lat_mutex;
inline LatAcc     g_lat[2];        // total de la partida (A, B)
inline uint64_t   g_last_flip_ns = 0;

// Registros por DCB (render) para el modelo de N trabajadores.
struct DcbRec {
	uint64_t ready, rs, re, cw_ps;
	uint32_t draws, calls;
};
inline std::mutex          g_dcb_mutex;
inline std::vector<DcbRec> g_dcbs;

// Hilo del juego: un envio de la cola 0.
inline void OnEnqueue(uint32_t queue_id, uint64_t serial, Shadow::Job::Kind kind, bool reset, std::span<const uint32_t> commands) {
	if (queue_id != 0 || Mode() == 0) return;
	MemCheck(); // (tej-lag3)
	// (tej-enq) Coste del gancho en el hilo del juego (dentro de SubmitJob FUN_00a18840, bajo m_queue_mutex), A y B.
	struct EnqTimer {
		uint64_t t0 = __rdtsc();
		~EnqTimer() { g_bank[8].c[JEnqTicks].fetch_add(__rdtsc() - t0, std::memory_order_relaxed), g_bank[8].c[JEnqN].fetch_add(1, std::memory_order_relaxed); }
	} enq_timer;
	auto& rec = g_rec[serial % RecN];
	const uint64_t now = Now();
	rec.serial.store(serial, std::memory_order_relaxed);
	rec.enq.store(now, std::memory_order_relaxed);
	rec.ready.store(0, std::memory_order_relaxed);
	rec.cw_ps.store(0, std::memory_order_relaxed);
	rec.kind.store(static_cast<uint8_t>(kind), std::memory_order_relaxed);
	if (kind == Shadow::Job::Graphics) {
		uint64_t zero = 0;
		(void)g_frame_first.compare_exchange_strong(zero, now, std::memory_order_relaxed);
	} else if (kind == Shadow::Job::Flip) {
		rec.frame_first.store(g_frame_first.exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
	}
	auto* s = g_shadow.load(std::memory_order_acquire);
	if (s == nullptr || !Active()) return; // (ventana A: nada se copia)
	// (tej-lag4) muestreo: fuera del tramo, como en A (nada se copia; la sombra duerme).
	if (TejLag::SampleOff(now, VarOn() ? (Var() == 7 ? 8u : 1u) : TejLag::SampleDiv())) return;
	s->Push(serial, kind, reset, commands);
}

struct GpuDcb {
	uint64_t serial = 0, prev_tsc = 0, draws = 0, rs = 0, calls = 0;
	Slot*    slot = nullptr;
	uint32_t next = 0;
	uint32_t n_seen = 0; // (tej-lag3) ultimo slot.count leido
	bool     on   = false;
};
inline GpuDcb g_gpu;


inline void EffDcbStart(); // (tej F2p, mas abajo)
struct GpuSlice {
	bool set = false;
	GpuSlice(bool graphics_q0, uint64_t serial, bool first_slice) {
		if (Mode() == 0 || !graphics_q0) return;
		if (first_slice) {
			// (tej-lag4) sonda de frecuencia real / migraciones del render (solo KYTY_TECHO=1), antes de medir la DCB.
			TejLag::g_measure.store(Techo::On(), std::memory_order_relaxed);
			TejLag::g_side.store(Active() ? 1 : 0, std::memory_order_relaxed);
			TejLag::RenderDcbStart(LocalPlatform::CurrentCpu());
			const uint64_t t0 = __rdtsc();
			FlushHooks(); // (lo de antes de esta DCB: flips y esperas sueltas, a su lado)
			g_hook_side = Active();
			g_dcb_cyc0  = LocalPlatform::CurrentThreadCycles();
			g_dcb_tsc0  = t0;
			TejLag::t_render = true;
			t_bank           = 0;
			g_gpu_serial.store(serial, std::memory_order_release);
			g_render_in.store(true, std::memory_order_relaxed);
			g_render_cpu.store(LocalPlatform::CurrentCpu(), std::memory_order_relaxed);
			g_gpu        = {};
			g_gpu.serial = serial;
			g_gpu.rs     = Now();
			EffDcbStart(); // (tej F2p)
			g_gpu.on     = Active();
			if (g_gpu.on) {
				Add(GDcbs);
				if (auto* s = g_shadow.load(std::memory_order_acquire); s != nullptr) {
					auto& sl = s->SlotOf(serial);
					if (sl.serial.load(std::memory_order_acquire) == serial) {
						g_gpu.slot = &sl;
						Add(GDcbSlot);
					}
				}
			}
			t_hook.t[HSlice] += __rdtsc() - t0, ++t_hook.n[HSlice];
		}
		if (g_gpu.serial != serial || !g_gpu.on) return;
		t_side = 1;
		set    = true;
	}
	~GpuSlice() {
		if (!set) return;
		t_side = 0;
	}
	GpuSlice(const GpuSlice&)            = delete;
	GpuSlice& operator=(const GpuSlice&) = delete;
};

// (tej F1c) Registros distintos en el punto de cose (solo paquetes que fallan por registros, render): por nombre,
// "aparece" y "unico" (es lo UNICO distinto: arreglarlo cose ese paquete).
struct RegDiff {
	std::mutex                                mutex;
	std::unordered_map<const char*, uint64_t> any, sole;
	const char*                               names[64] {};
	uint32_t                                  n = 0;
};
inline RegDiff  g_regdiff;
inline uint32_t g_regdiff_k = 0;
inline constexpr uint32_t RegDiffEvery = 64; // (tej-coste: 1 de cada 64 paquetes que fallan por registros; la linea x64)
inline void    NoteRegDiff(const CommandProcessor& render, const CommandProcessor& worker) {
	if (g_regdiff_k++ % RegDiffEvery != 0) return;
	const HookTimer tm(HRegDiff);
	Add(GPkRegsSampled);
	auto& d = g_regdiff;
	d.n     = 0;
	render.DiffState(worker, [](const char* name) {
		if (g_regdiff.n < 64) g_regdiff.names[g_regdiff.n++] = name;
	});
	std::lock_guard g(d.mutex);
	for (uint32_t i = 0; i < d.n; ++i) ++d.any[d.names[i]];
	if (d.n == 1) ++d.sole[d.names[0]];
	Add(GRegDiffNames, d.n);
}

// (tej-cos) Diagnostico del embudo (render; solo en fallos): registros de DRAWS distintos por nombre (sin m_cs, 1 de
// cada 16), lecturas limpias escritas (rango leido, quien escribio: GPU/CPU y su rango), metadatos borrados (superficie,
// slice, origen del borrado: TextureCache::MetaClearOrigin).
struct CosDiag {
	std::mutex                                mutex;
	std::unordered_map<const char*, uint64_t> regs;
	struct Rd {
		uint64_t n = 0, size = 0, w_addr = 0, w_size = 0;
		uint8_t  kind = 0, source = 0;
	};
	std::unordered_map<uint64_t, Rd>                        reads;
	std::unordered_map<uint64_t, std::array<uint64_t, 16>> metas; // (por superficie: veces por origen)
	// (tej-cos2) los cambios que tumban: imagen (begin -> end, alta/baja, veces) y desmapeo (begin -> end, origen, veces)
	struct Hit {
		uint64_t end = 0, n = 0;
		uint8_t  tag = 0;
	};
	std::unordered_map<uint64_t, Hit> img_hits, map_hits;
	const char*                                             names[64] {};
	uint32_t                                                n = 0;
};
inline CosDiag              g_cos;
inline uint32_t             g_cos_k = 0;
inline constexpr uint32_t   CosRegEvery = 16;
inline void CosHit(std::unordered_map<uint64_t, CosDiag::Hit>& m, uint64_t b, uint64_t e, uint8_t tag) {
	std::lock_guard g(g_cos.mutex);
	if (m.size() >= 4096 && !m.contains(b)) return;
	auto& h = m[b];
	h.end = e, h.tag = tag, ++h.n;
}
// (tej-cos2) Con dispatches y CS distinto (la perdida 'CS con dispatches'): solo los m_cs, mismo mapa.
inline uint32_t g_cos_cs_k = 0;
inline void     NoteCsDiff(const CommandProcessor& render, const CommandProcessor& worker) {
	if (g_cos_cs_k++ % CosRegEvery != 0) return;
	const HookTimer tm(HRegDiff);
	Add(GCosCsSampled);
	auto& d = g_cos;
	d.n     = 0;
	render.DiffState(worker, [](const char* name) {
		if (g_cos.n < 64 && (std::strncmp(name, "m_cs", 4) == 0 || std::strncmp(name, "m_dispatch", 10) == 0))
			g_cos.names[g_cos.n++] = name;
	});
	std::lock_guard g(d.mutex);
	for (uint32_t i = 0; i < d.n; ++i) ++d.regs[d.names[i]];
}
inline void NoteGfxDiff(const CommandProcessor& render, const CommandProcessor& worker) {
	if (g_cos_k++ % CosRegEvery != 0) return;
	const HookTimer tm(HRegDiff);
	Add(GCosRegSampled);
	auto& d = g_cos;
	d.n     = 0;
	render.DiffState(worker, [](const char* name) {
		if (g_cos.n < 64 && std::strncmp(name, "m_cs", 4) != 0) g_cos.names[g_cos.n++] = name;
	});
	std::lock_guard g(d.mutex);
	for (uint32_t i = 0; i < d.n; ++i) ++d.regs[d.names[i]];
}

// (tej F1c) WAIT_REG_MEM en memoria que no es etiqueta conocida: la sombra la supone cumplida y sigue
// (KYTY_TEJ_ESPERA=0: se para como antes). El render cuenta, por direccion, si al llegar ya estaba cumplida.
[[nodiscard]] inline bool AssumeOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEJ_ESPERA");
		return v == nullptr || std::atoi(v) != 0;
	}();
	return on;
}
[[nodiscard]] inline bool AssumeWait(uint64_t addr, uint64_t /*value*/, uint64_t /*ref*/, uint32_t /*func*/) {
	return AssumeOn() && t_worker != nullptr && t_worker->Assume(addr);
}
struct RenderWaits {
	std::mutex                                                  mutex;
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_addr; // (cumplida al llegar, tuvo que esperar)
};
inline RenderWaits g_rwaits;
// (tej-lag3) MEDIDA (solo KYTY_TECHO=1). Antes: mutex + unordered_map por espera (228/fot, 0.031 ms/fot en B). Ahora: tabla
// plana del hilo de render (64 direcciones, sondeo lineal, sin cerrojo ni asignacion); se vuelca al mapa una vez por DCB
// (RenderWaitsFlush, desde FlushHooks).
struct RwLocal {
	uint64_t addr[64] {}, sat[64] {}, wait[64] {};
	uint32_t used = 0;
};
inline RwLocal g_rwl; // (solo el hilo de render)
inline void    RenderWaitsFlush() {
	if (g_rwl.used == 0) return;
	std::lock_guard g(g_rwaits.mutex);
	for (uint32_t i = 0; i < 64; ++i)
		if (g_rwl.addr[i] != 0) {
			if (g_rwaits.by_addr.size() <= 4096 || g_rwaits.by_addr.contains(g_rwl.addr[i])) {
				auto& e = g_rwaits.by_addr[g_rwl.addr[i]];
				e.first += g_rwl.sat[i], e.second += g_rwl.wait[i];
			}
			g_rwl.addr[i] = g_rwl.sat[i] = g_rwl.wait[i] = 0;
		}
	g_rwl.used = 0;
}
inline void RenderWait(uint64_t addr, bool satisfied) {
	if (!Techo::On() || addr == 0) return;
	const HookTimer tm(HWait);
	uint32_t        i = static_cast<uint32_t>((addr >> 3) * 0x9E3779B97F4A7C15ull >> 58);
	for (uint32_t k = 0; k < 64; ++k, i = (i + 1) & 63u) {
		if (g_rwl.addr[i] == addr || g_rwl.addr[i] == 0) {
			if (g_rwl.addr[i] == 0) g_rwl.addr[i] = addr, ++g_rwl.used;
			++(satisfied ? g_rwl.sat[i] : g_rwl.wait[i]);
			return;
		}
	}
}

// (tej F2p) COSIBLES REALES: un paquete que cose por registros y epocas/rangos (F1d) pasa ademas, en solo lectura, lo
// que CommitSegments comprueba antes de Before (sin Before/After, sin enviar), contra los efectos del render desde que
// el trabajador empezo a traducir la DCB (EffectsLog propio, g_eff). Primer motivo, en el orden de CommitSegments:
inline constexpr uint32_t RN = 9;
inline constexpr const char* RNames[RN] = {"pasa", "registro de efectos perdido", "CPU escribio una imagen (barrido)",
                                           "metadatos borrados", "escrito lo que leyo (clean_reads)", "set retirado",
                                           "GPU escribio lo que copia", "cadena distinta", "espera no cumplida"};
struct EffAcc {
	Libs::Graphics::RangeSet gw, hw;
	std::vector<uint64_t>    sets;
	uint64_t                 next = 0;
	void Reset(uint64_t from) { gw.Clear(), hw.Clear(), sets.clear(), next = from; }
	void TakeTo(const Libs::Graphics::Spec::EffectsLog& log, uint64_t limit) {
		using K = Libs::Graphics::Spec::EffectsLog::Kind;
		limit   = std::min<uint64_t>(limit, log.base + log.entries.size());
		for (next = std::max(next, log.base); next < limit; ++next) {
			const auto& e = log.entries[next - log.base];
			if (e.kind == K::GpuWrite) gw.Add(e.address, e.size);
			else if (e.kind == K::HostWrite) hw.Add(e.address, e.size);
			else sets.push_back(e.address);
		}
	}
};
struct EffCheck {
	EffAcc   all, pre;            // (hasta ahora; hasta el inicio de la DCB que el render procesa)
	uint64_t from = ~uint64_t {0}, serial = 0, dcb_end = 0;
	bool     lost = false;
	uint64_t ring[32] {};
	uint32_t ring_k = 0;
	uint32_t before_k = 0;
	uint64_t gap_end  = 0;     // (tej-lag) fin del registro en la ultima DCB de A (lo de antes no se registro)
	bool     sample   = false; // (tej-lag) esta DCB se comprueba (1 de cada RealEvery)
	uint64_t ticks0   = 0;
};
inline EffCheck g_ec;
// (tej-lag3) MEDIDA: F2p 1/32 DCB (antes 1/8: 0.232 ms/fot en B, 12 paquetes/fot a ~19 us cada uno, casi todo
// RangeSet::Add de los efectos acumulados); cuentas x RealEvery, misma estimacion con 4x menos coste.
inline constexpr uint32_t BeforeEvery = 8, RealEvery = 32;
// (F2p) Render, inicio de DCB de la cola 0: fin del registro y poda de lo anterior a 32 DCB atras.
inline void EffDcbStart() {
	ClaimEff();
	if (!t_eff_owner) return;
	auto& e   = g_ec;
	g_eff.cost_ticks = Techo::On() ? &g_eff_ticks : nullptr; // (tej-lag3: MEDIDA)
	e.dcb_end = g_eff.End();
	// (tej-lag) Poda fija a 32 DCB atras (antes: tambien el inicio del ultimo paquete comprobado, que podia quedarse
	// viejo y no podar nunca). Un paquete que empezo antes sale como 'registro de efectos perdido'.
	const uint64_t old = e.ring[e.ring_k % 32];
	e.ring[e.ring_k++ % 32] = e.dcb_end;
	if (old != 0) g_eff.DropBefore(old);
	if (!g_hook_side) e.gap_end = e.dcb_end;
	// (tej-rep) ARREGLO de la medida rota en 29a8bea (PASA/F2p/COS3 0.0 %, muestra x0.00): 'serial % 32' no caia
	// nunca en 0 (los seriales de la cola 0 no recorren todos los restos), asi que Real() no entraba y sc = 0. Ahora:
	// contador propio de DCB del lado del gancho, mezclado (no siempre las mismas posiciones del fotograma), 1/RealEvery.
	static uint32_t dcb_k = 0;
	e.sample = g_hook_side && ((static_cast<uint32_t>(dcb_k++) * 0x9E3779B1u) >> 27) == 0;
	static uint64_t last = 0;
	Add(GEffEntries, e.dcb_end - last);
	t_hook.t[HEff] += g_eff_ticks - e.ticks0, t_hook.n[HEff] += e.dcb_end - last;
	e.ticks0 = g_eff_ticks;
	last = e.dcb_end;
}
// (tej-cosibles) Una lectura limpia [b, e) que el render escribio en la CPU (WRITE_DATA etc., ya en memoria al llegar el
// render a este paquete): la traduccion es la misma si TODOS los bytes que leyo ahi son los que hay ahora (lo que la
// traduccion normal leeria ahora). Cubierta entera por los bytes guardados (p.rv) y todos iguales; si no, falla.
// 0 pasa por valor, 1 distinta, 2 sin bytes guardados.
[[nodiscard]] inline int ReadByValue(const Packet& p, uint64_t b, uint64_t e) {
	if (e <= b || e - b > Worker::SnapMax) return 2;
	uint8_t  cov[Worker::SnapMax] {};
	size_t   at  = 0;
	for (const auto& [a, n]: p.rv) {
		const uint8_t* v = p.rvb.data() + at;
		at += n;
		if (a >= e || a + n <= b) continue;
		if (std::memcmp(v, reinterpret_cast<const void*>(a), n) != 0) return 1;
		for (uint64_t x = std::max(a, b); x < std::min(a + n, e); ++x) cov[x - b] = 1;
	}
	for (uint64_t x = 0; x < e - b; ++x)
		if (cov[x] == 0) return 2;
	return 0;
}
// Lecturas limpias: GPU encima = falla; CPU (render) encima = por valor. count: contadores (solo la pasada 'all').
[[nodiscard]] inline bool ReadsFail(const Packet& p, const EffAcc& a, bool count) {
	if (a.gw.Empty() && a.hw.Empty()) return false;
	for (const auto& [b, e]: p.reads) {
		if (e <= b) continue;
		if (!a.gw.Empty() && a.gw.Intersects(b, e - b)) return true;
		if (a.hw.Empty() || !a.hw.Intersects(b, e - b)) continue;
		const int v = ReadByValue(p, b, e);
		if (count) Add(v == 0 ? GCosValOk : v == 1 ? GCosValDiff : GCosValNone);
		if (v != 0) return true;
	}
	return false;
}
[[nodiscard]] inline uint32_t RealWhy(const Packet& p, RenderContext& r, const EffAcc& a, bool lost, bool count = false) {
	if (lost) return 1;
	const auto in = [](const Libs::Graphics::RangeSet& rs, const auto& v) {
		if (rs.Empty()) return false;
		for (const auto& [b, e]: v)
			if (e > b && rs.Intersects(b, e - b)) return true;
		return false;
	};
	if (in(a.hw, p.imgs)) return 2;
	auto& tc = r.GetTextureCache();
	for (const auto& [addr, slice]: p.metas)
		if (tc.IsMetaCleared(addr, slice)) return 3;
	if (ReadsFail(p, a, count)) return 4; // (tej-cosibles: antes cualquier escritura del render encima)
	for (const auto set: a.sets)
		if (std::find(p.sets.begin(), p.sets.end(), set) != p.sets.end()) return 5;
	if (in(a.gw, p.copies)) return 6;
	for (const auto& c: p.chains)
		if (*reinterpret_cast<const uint32_t*>(c.address) != c.value) return 7;
	for (const auto& w: p.waits) {
		uint64_t value = 0;
		std::memcpy(&value, reinterpret_cast<const void*>(w.address), w.bytes);
		if (!Libs::Graphics::TestWaitRegMemValue(value, w.reference, w.mask, w.function)) return 8;
	}
	return 0;
}
// (tej-cos) El primer culpable del paquete: la lectura escrita (y su escritor) o la superficie borrada.
inline void CosWhy(const Packet& p, RenderContext& r, const EffAcc& a, uint64_t from, uint32_t why) {
	auto& d = g_cos;
	if (why == 4) {
		for (const auto& [b, en]: p.reads) {
			if (en <= b) continue;
			const bool gw = !a.gw.Empty() && a.gw.Intersects(b, en - b);
			const bool hw = !gw && !a.hw.Empty() && a.hw.Intersects(b, en - b) && ReadByValue(p, b, en) != 0;
			if (!gw && !hw) continue;
			Add(gw ? GCosReadGpu : GCosReadHost); // (muestra: sin escalar; la linea escala por F1d/base)
			const auto*     w = g_eff.FirstWriteOver(from, b, en - b);
			std::lock_guard g(d.mutex);
			if (d.reads.size() >= 4096 && !d.reads.contains(b)) return;
			auto& x = d.reads[b];
			x.n += 1, x.size = en - b, x.kind = gw ? 1 : 2;
			if (w != nullptr) x.w_addr = w->address, x.w_size = w->size, x.source = static_cast<uint8_t>(w->source);
			return;
		}
		return;
	}
	auto& tc = r.GetTextureCache();
	for (const auto& [addr, slice]: p.metas)
		if (tc.IsMetaCleared(addr, slice)) {
			const uint8_t o = tc.MetaClearOrigin(addr) & 15u;
			// (tej-cos2) la superficie dentro de una imagen del paquete (p.ri, ordenado) o fuera
			const auto it  = std::upper_bound(p.ri.begin(), p.ri.end(), std::pair<uint64_t, uint64_t> {addr, ~uint64_t {0}});
			const bool on  = it != p.ri.begin() && std::prev(it)->second > addr;
			Add(on ? GCosMetaOnImg : GCosMetaOffImg);
			std::lock_guard g(d.mutex);
			if (d.metas.size() >= 4096 && !d.metas.contains(addr)) return;
			d.metas[addr][o] += 1;
			return;
		}
}
inline void Real(const Packet& p, RenderContext& r, uint32_t sewn, Slot& s) {
	auto& e = g_ec;
	if (!e.sample || !Techo::On()) return; // (tej-lag: 1 de cada RealEvery DCB; tej-lag3: MEDIDA, solo KYTY_TECHO=1)
	{
		const HookTimer tm(HReal);
		const uint64_t  from = s.eff_from.load(std::memory_order_relaxed);
		if (!t_eff_owner) return Add(GRk1);
		if (e.from != from || e.serial != g_gpu_serial.load(std::memory_order_relaxed)) {
			e.from   = from;
			e.serial = g_gpu_serial.load(std::memory_order_relaxed);
			e.lost   = from < g_eff.base || from <= e.gap_end;
			e.all.Reset(from);
			e.pre.Reset(from);
		}
		e.all.TakeTo(g_eff, ~uint64_t {0});
		const uint32_t why = RealWhy(p, r, e.all, e.lost, true);
		if (why != 0) e.pre.TakeTo(g_eff, e.dcb_end); // (tej-lag3: 'pre' solo hace falta si 'all' falla)
		// (tej-cos2) SIN escalar: antes x RealEvery, pero la DCB muestreada (serial % 8) cae siempre en las mismas
		// posiciones del fotograma (las grandes) y salia 171 %. Ahora: base de la muestra y % = F1d x pasa / base.
		constexpr uint64_t k = 1;
		Add(GRealBaseD, sewn), Add(GRealBasePk);
		Add(static_cast<C>(GRk0 + why), k);
		Add(static_cast<C>(GRkD0 + why), sewn * k); // (tej-cos)
		if (why == 3 || why == 4) CosWhy(p, r, e.all, e.from, why);
		if (why == 0) Add(GRealPk, k), Add(GRealDraws, sewn * k), Add(GRealCalls, (sewn + p.dispatches) * k);
		if (why == 0 || RealWhy(p, r, e.pre, e.lost) == 0) Add(GRealPkPre, k), Add(GRealDrawsPre, sewn * k);
		if (why != 0) return;
	}
	// (F2p) Lo que Before haria, 1 de cada BeforeEvery paquetes que pasan: solo las consultas de solo lectura del hilo
	// de render (registrado + sucio de CPU por rango escrito/actual: el camino rapido de ObtainBuffer /
	// EnsureBufferContents) y los bytes sucios que subiria; BDA, invalidaciones e imagenes: cuentas (no se tocan).
	if (e.before_k++ % BeforeEvery != 0) return;
	const HookTimer tb(HBefore);
	const uint64_t  t0 = tb.t0;
	auto&           bc = r.GetBufferCache();
	uint64_t        dirty_n = 0, dirty_b = 0;
	for (const auto* v: {&p.wr, &p.cur})
		for (const auto& [b, en]: *v) {
			if (en <= b) continue;
			(void)bc.RegisteredBuffer(b, en - b);
			if (bc.IsRegionCpuDirty(b, en - b)) ++dirty_n, dirty_b += en - b;
		}
	Add(GBfTicks, __rdtsc() - t0);
	Add(GBfPk);
	Add(GBfRanges, p.wr.size() + p.cur.size());
	Add(GBfDirtyRanges, dirty_n);
	Add(GBfDirtyBytes, dirty_b);
	Add(GBfBda, p.n_bda);
	if (p.bda_all) Add(GBfBdaAll);
	Add(GBfInv, p.n_inv);
	Add(GBfImg, p.n_img);
}

// Hilo de render: primer draw de cada paquete; epocas primero (baratas), registros solo si coinciden.
inline void GpuDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
	auto& c = g_gpu;
	++c.draws;
	TejCoser::RenderTick(key); // (tej-coser)
	Slot* const s = c.slot;
	if (s == nullptr) return;
	// (tej-lag3) slot.count es la linea que el trabajador escribe al publicar (en el otro CCD): antes se leia en CADA draw
	// (3258/fot; cada lectura tras una publicacion = fallo de cache entre CCD). Ahora solo cuando el render alcanza lo ya
	// visto; si el siguiente paquete conocido es de un draw posterior, sale sin tocar nada compartido.
	if (c.next < c.n_seen && s->pk[c.next].key > key) return;
	const HookTimer tm(HDraw);
	const uint64_t  now = tm.t0;
	uint32_t        n   = c.n_seen;
	for (;;) {
		if (c.next >= n) {
			const uint32_t n2 = s->count.load(std::memory_order_acquire);
			if (n2 <= n) break;
			n = c.n_seen = n2;
		}
		const Packet& p = s->pk[c.next];
		if (p.key > key) break;
		++c.next;
		if (p.key < key) {
			Add(p.publish_tsc > c.prev_tsc ? GPkLate : GPkSkip);
			continue;
		}
		Add(GDrawsOnTime, p.draws);
		if (p.assumed) Add(GDrawsOnTimeAssumed, p.draws);
		const uint64_t t0   = __rdtsc();
		auto&          bc   = r.GetBufferCache();
		auto&          tc   = r.GetTextureCache();
		auto&          gr   = r.GetGpuResources();
		const bool     eb   = bc.RegistrationEpoch() == p.ep_buf;
		const bool     ei0  = tc.ResolutionEpoch() == p.ep_img;
		const bool     em0  = gr.MappingEpoch() == p.ep_map;
		const bool     ei   = eb && ei0;
		const bool     em   = ei && em0;
		const uint32_t sewn = p.draws > p.holes ? p.draws - p.holes : 0;
		// (tej-coste) Dos comparaciones en vez de tres: SameState == SameGraphicsState && SameComputeState (Shader:
		// m_vs, m_ps, m_cs; SameDrawState: los de draws + base de dispatch indirecto). Mismo resultado.
		bool gfx = false, cseq = false;
		{
			const HookTimer tv(HVal);
			gfx  = sewn != 0 && cp.SameGraphicsState(*p.cp);
			cseq = gfx && cp.SameComputeState(*p.cp); // (solo cuenta con gfx)
		}
		const bool same = em && gfx && cseq;
		const bool csok = p.dispatches == 0 || cseq;
		if (gfx && !csok) Add(GPkCsDispatch);
		if (gfx && csok && !cseq) Add(GPkCsIgnored);
		// (F1d) Epocas por rangos: solo si la global cambio y los registros cosen, lo que el paquete leyo; listas
		// compactas del trabajador (rb: regiones de 4 MiB; rx: todo fusionado).
		// (tej-cos) Buffers por rango EXACTO (anillo de BufferCache: ningun buffer vino ni se fue sobre los bytes que la
		// DCB busco desde que empezo a traducirse; misma prueba que las regiones de 4 MiB, a nivel de byte); si el anillo
		// no llega, las regiones como antes. Imagenes: solo sobre las imagenes del paquete (lo que Changed mira); mapeos:
		// todo lo leido (rx), como Changed.
		const bool eb0 = bc.RegistrationEpoch() == p.ep_buf0;
		bool       rb = p.rbx_ok ? eb0 : eb, ri = ei0, rm = em0;
		if ((!rb || !ri || !rm) && !p.lists) {
			Add(GPkRgNoList);
		} else if ((!rb || !ri || !rm) && gfx && csok) {
			const HookTimer tr(HRanges);
			const bool      b0 = rb, i0 = ri, m0 = rm;
			uint64_t        k  = 0;
			if (!b0) {
				bool known = false;
				// (tej-lag3) Filtro previo EQUIVALENTE: las regiones de 4 MiB de rbx (rbxr) desde ep_buf0. Si ninguna region
				// cambio, ningun registro/desregistro toco esos bytes desde ep_buf0: el anillo exacto daria 'sin choque'
				// seguro, y no se recorre (hasta 4096 epocas con busqueda binaria: ~5.5 us/paquete, 0.094 ms/fot en B).
				// Si alguna region cambio, el exacto como antes. Mismo resultado, mismas cosibles.
				if (p.rbx_ok && !p.rbxr.empty() && p.rbxr.back().second <= Libs::Graphics::TRACKER_ADDRESS_SIZE) {
					bool reg = true;
					for (const auto& [a, e]: p.rbxr) {
						++k;
						if (bc.RegisteredSince(a, e - a, p.ep_buf0)) { reg = false; break; }
					}
					if (reg) known = true, rb = true, Add(GCosBufExactOk), Add(GCosBufRegionOk);
				}
				if (!known && p.rbx_ok) {
					const bool hit = bc.RegisteredOverlapSince(p.rbx.data(), p.rbx.size(), p.ep_buf0, known);
					k += p.rbx.size();
					if (known) rb = !hit, Add(hit ? GCosBufExactFail : GCosBufExactOk);
				}
				if (!known) {
					Add(GCosBufExactLost);
					rb = eb;
					if (!rb) {
						rb = true;
						for (const auto& [a, e]: p.rb) {
							++k;
							if (bc.RegisteredSince(a, e - a, p.ep_buf)) { rb = false; break; }
						}
					}
				}
			}
			if (rb && !i0) {
				// (tej-cos2) Rango EXACTO (anillo de TextureCache, a paginas de 4 KiB) sobre las imagenes del paquete;
				// antes paginas de 1 MiB (m_registration_pages): un alta de otra textura en el mismo MiB lo tumbaba.
				bool                    known = false;
				Libs::Graphics::TextureCache::RegHit h {};
				const bool hit = tc.RegisteredOverlapSince(p.ri.data(), p.ri.size(), p.ep_img, known, &h);
				k += p.ri.size();
				if (known) {
					ri = !hit;
					Add(hit ? GCosImgExactFail : GCosImgExactOk);
					if (hit && Techo::On()) CosHit(g_cos.img_hits, h.begin, h.end, h.reg ? 1 : 0);
				} else {
					Add(GCosImgExactLost);
					for (const auto& [a, e]: p.ri) {
						++k;
						if (tc.RegisteredOverSince(a, e - a, p.ep_img)) { ri = false; break; }
					}
				}
				if (ri && Techo::On()) Add(GCosImgOnlyOk); // (tej-lag3: diagnostico = MEDIDA)
			}
			if (rb && ri && !m0) {
				// (tej-cos2) Desmapeos a nivel de byte (anillo de GpuResourceManager) sobre lo leido; antes granulos de 16 KiB.
				bool                             known = false;
				Libs::Graphics::GpuResourceManager::UnmapHit h {};
				const bool hit = gr.UnmappedOverlapSince(p.rx.data(), p.rx.size(), p.ep_map, known, &h);
				k += p.rx.size();
				if (known) {
					rm = !hit;
					Add(hit ? GCosMapExactFail : GCosMapExactOk);
					if (hit && Techo::On()) CosHit(g_cos.map_hits, h.begin, h.end, h.origin);
				} else {
					Add(GCosMapExactLost);
					for (const auto& [a, e]: p.rx) {
						++k;
						if (gr.UnmappedSince(p.ep_map, a, e - a)) { rm = false; break; }
					}
				}
			}
			Add(GRgEntries, k);
			Add(GRgRaw, p.raw_ranges);
			if (!b0) Add(rb ? GPkRgBuf : GPkRgFailBuf);
			if (!i0 && rb) Add(ri ? GPkRgImg : GPkRgFailImg);
			if (!m0 && rb && ri) Add(rm ? GPkRgMap : GPkRgFailMap);
		}
		TejCoser::RenderOpen(p.coser, p.draws, gfx && csok && rb && ri && rm, p.holes == 0 && sewn != 0, r); // (tej-coser)
		Add(GCosSewn, sewn);
		if (gfx && !csok) {
			Add(GCosCsBad, sewn);
			if (Techo::On()) NoteCsDiff(cp, *p.cp);
		}
		if (sewn != 0 && !gfx) NoteGfxDiff(cp, *p.cp);
		if (gfx && csok) {
			Add(GCosGfxOk, sewn);
			if (eb && ei && em) Add(GDrawsOkGfx, sewn);
			if (!rb) Add(GCosLostBuf, sewn);
			else if (!ri) Add(GCosLostImg, sewn);
			else if (!rm) Add(GCosLostMap, sewn);
			if (rb && ri && rm) {
				Add(GDrawsOkRanges, sewn);
				if (p.lists) Real(p, r, sewn, *s); // (tej F2p)
			}
		}
		Add(GValTicks, __rdtsc() - t0);
		Add(GValN);
		if (!eb) Add(GPkEpBuf);
		else if (!ei) Add(GPkEpImg);
		else if (!em) Add(GPkEpMap);
		else if (sewn == 0) Add(GPkAllHoles); // (tej F1c: antes contado como 'registros')
		else if (!same) {
			Add(GPkRegs);
			NoteRegDiff(cp, *p.cp);
		} else {
			Add(GDrawsRegsOk, sewn);
			Add(GPkOk);
			Add(GDrawsOk, sewn);
			Add(GHolesOk, p.holes);
			if (p.assumed) Add(GPkAssumed), Add(GDrawsOkAssumed, sewn);
		}
		break;
	}
	c.prev_tsc = now;
}

inline void OnDispatch() {
	if (t_side == 1) {
		const HookTimer tm(HDispatch);
		++g_gpu.calls;
	}
	else if (t_side == 2 && t_worker != nullptr) t_worker->Dispatch();
}

// (tej F1c) Latencia: el render procesa el flip (CommandProcessor::Flip, dentro de la ultima DCB del fotograma: el
// juego hace submit+flip en FUN_00911d60; FlipPreparation sola no llega en partida: por eso antes 0 muestras).
// Inicio = Enqueue de la DCB siguiente a la del flip anterior (la primera del fotograma). Lado A/B = Active().
inline uint64_t g_last_flip_serial = 0;
inline void     OnRenderFlip() {
	const HookTimer tm(HFlip);
	const uint64_t  now    = Now();
	const uint64_t  serial = g_gpu_serial.load(std::memory_order_relaxed);
	const int       side   = Active() ? 1 : 0;
	const auto&     first  = g_rec[(g_last_flip_serial + 1) % RecN];
	std::lock_guard g(g_lat_mutex);
	if (g_last_flip_serial != 0 && serial > g_last_flip_serial &&
	    first.serial.load(std::memory_order_relaxed) == g_last_flip_serial + 1) {
		const uint64_t enq = first.enq.load(std::memory_order_relaxed);
		if (enq != 0 && now > enq) g_lat[side].Lat(now - enq);
	}
	if (g_last_flip_ns != 0) {
		++g_lat[side].frames;
		if (now - g_last_flip_ns > 25000000u) ++g_lat[side].over25;
	}
	g_last_flip_ns     = now;
	g_last_flip_serial = serial;
}

inline void OnDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
	if (t_side == 1) GpuDraw(cp, r, key);
	else if (t_side == 2 && t_worker != nullptr) t_worker->WorkerDraw(cp, r, key);
}

// Hilo de render: un envio de la cola 0 acabado.
inline void GpuDone(uint64_t serial, const CommandProcessor& cp) {
	if (Mode() == 0) return;
	g_render_in.store(false, std::memory_order_relaxed);
	TejCoser::RenderDcbEnd(); // (tej-coser)
	const uint64_t t_in = __rdtsc();
	const uint64_t now = Now();
	const auto&    rec = g_rec[serial % RecN];
	const bool     mine = rec.serial.load(std::memory_order_relaxed) == serial;
	if (g_gpu.serial == serial) {
		if (g_gpu.slot != nullptr && !g_gpu.slot->done.load(std::memory_order_acquire)) Add(GTailLate);
		g_gpu.slot = nullptr;
		if (g_gpu.on) {
			Add(GDraws, g_gpu.draws);
			Add(GCalls, g_gpu.draws + g_gpu.calls);
			if (mine && g_gpu.draws != 0) {
				std::lock_guard g(g_dcb_mutex);
				if (g_dcbs.size() < 65536)
					g_dcbs.push_back({rec.ready.load(std::memory_order_relaxed) != 0 ? rec.ready.load(std::memory_order_relaxed)
					                                                                 : rec.enq.load(std::memory_order_relaxed),
					                  g_gpu.rs, now, rec.cw_ps.load(std::memory_order_relaxed),
					                  static_cast<uint32_t>(g_gpu.draws), static_cast<uint32_t>(g_gpu.draws + g_gpu.calls)});
			}
		}
		g_gpu.on = false;
	}
	if (auto* s = g_shadow.load(std::memory_order_acquire); s != nullptr) s->GpuDone(serial, cp);
	t_hook.t[HDone] += __rdtsc() - t_in, ++t_hook.n[HDone];
	if (g_gpu.serial == serial && g_dcb_tsc0 != 0) {
		const int sd = g_hook_side ? 1 : 0;
		g_hook.dcb_n[sd].fetch_add(1, std::memory_order_relaxed);
		g_hook.dcb_wall[sd].fetch_add(__rdtsc() - g_dcb_tsc0, std::memory_order_relaxed);
		g_hook.dcb_cyc[sd].fetch_add(LocalPlatform::CurrentThreadCycles() - g_dcb_cyc0, std::memory_order_relaxed);
		g_dcb_tsc0 = 0;
	}
	FlushHooks();
}

// Modelo: % de draws en paquetes de 64 que estarian publicados antes de que el render llegue, con n trabajadores
// (cola por llegada, coste por draw medido de cada DCB o la media), sobre las DCB de la ventana.
// (tej F1c) Trabajador = llamadas (draws + dispatches) x coste por llamada + fijo por DCB; el paquete j esta listo
// cuando el trabajador lleva esa fraccion de llamadas; el render llega a la misma fraccion de su tiempo de DCB.
// Antes: tiempo total / draws (con DCB cortadas en la espera del principio: 19.8 us/draw) por TODOS los draws.
[[nodiscard]] inline double Model(const std::vector<DcbRec>& dcbs, uint32_t n, double avg_cw_ns, double fix_ns) {
	std::vector<double> free_at(n, 0.0);
	double              ok = 0, all = 0;
	for (const auto& d: dcbs) {
		all += d.draws;
		const double cw = d.cw_ps != 0 ? d.cw_ps / 1000.0 : avg_cw_ns;
		if (cw <= 0 || d.draws == 0) continue;
		auto         w  = std::min_element(free_at.begin(), free_at.end());
		const double w0 = std::max(*w, static_cast<double>(d.ready));
		const double rs = static_cast<double>(d.rs);
		if (w0 >= rs) continue; // (el render ya empezo: no se toma)
		const double calls = std::max<double>(d.calls, d.draws);
		const double span  = static_cast<double>(d.re - d.rs);
		for (uint32_t j = 0; j * PacketDraws < d.draws && j < MaxPackets; ++j) {
			const uint32_t last  = std::min<uint32_t>((j + 1) * PacketDraws, d.draws);
			const double   ready = w0 + calls * cw * last / d.draws;
			if (ready <= rs + span * (j * PacketDraws) / d.draws) ok += last - j * PacketDraws;
		}
		*w = w0 + calls * cw + fix_ns; // (el fijo despues: Finish va tras publicar el ultimo paquete)
	}
	return all > 0 ? 100.0 * ok / all : 0;
}

inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	const int  var   = Var();
	double     delta = 0, cnt[2] {};
	// (tej-var) Por variante: B entre dos A, B - media de sus A; ms/fot del lag del render y cerrojo/IPI.
	static double vsum[VarN] {}, vcnt[VarN] {}, vmin[VarN] {}, vmax[VarN] {};
	if (m == 2) {
		struct W { bool b; double busy; int var; };
		static W      w[4096];
		static size_t k = 0;
		if (keep && k < 4096) w[k++] = {was, busy_ms, var};
		// (la B de en medio de las tres ultimas, ya con su A siguiente)
		if (VarOn() && keep && k >= 3 && w[k - 2].b && !w[k - 1].b && !w[k - 3].b) {
			const auto&  b = w[k - 2];
			const double d = b.busy - 0.5 * (w[k - 3].busy + w[k - 1].busy);
			if (vcnt[b.var] == 0 || d < vmin[b.var]) vmin[b.var] = d;
			if (vcnt[b.var] == 0 || d > vmax[b.var]) vmax[b.var] = d;
			vsum[b.var] += d, vcnt[b.var] += 1;
		}
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (w[i - 1].b != w[i].b && w[i + 1].b != w[i].b) {
				const double d = w[i].busy - 0.5 * (w[i - 1].busy + w[i + 1].busy);
				sum[w[i].b ? 1 : 0] += d, cnt[w[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	}
	if (m != 2 || was) TejCoser::Print(frames); // (tej-coser: solo ventanas con sombra)
	else TejCoser::Print(0);
	uint64_t v[NCount] {};
	for (auto& b: g_bank)
		for (uint32_t i = 0; i < NCount; ++i) v[i] += b.c[i].exchange(0, std::memory_order_relaxed);
	static uint64_t last_tsc = 0;
	static auto     last_t   = std::chrono::steady_clock::now();
	const uint64_t  tsc      = __rdtsc();
	const auto      t        = std::chrono::steady_clock::now();
	const double    ns_per_tick =
        last_tsc != 0 && tsc > last_tsc ? std::chrono::duration<double, std::nano>(t - last_t).count() / static_cast<double>(tsc - last_tsc) : 0.25;
	last_tsc = tsc, last_t = t;
	const double f       = frames;
	const double draws   = static_cast<double>(v[GDraws]);
	const auto   pct     = [&](uint64_t x) { return draws > 0 ? 100.0 * static_cast<double>(x) / draws : 0.0; };
	const double on_time = static_cast<double>(v[GPkOk] + v[GPkRegs] + v[GPkEpBuf] + v[GPkEpImg] + v[GPkEpMap]);
	const double seen_pk = on_time + static_cast<double>(v[GPkLate]);
	const double val_ns  = v[GValN] != 0 ? static_cast<double>(v[GValTicks]) * ns_per_tick / static_cast<double>(v[GValN]) : 0;
	const double val_per_sewn =
	    v[GDrawsOk] != 0 ? static_cast<double>(v[GValTicks]) * ns_per_tick / static_cast<double>(v[GDrawsOk]) : 0;
	const double w_us   = v[WDraws] != 0 ? static_cast<double>(v[WNs]) / 1e3 / static_cast<double>(v[WDraws]) : 0;
	const double wt_us  = v[WDraws] != 0 ? static_cast<double>(v[WTransNs]) / 1e3 / static_cast<double>(v[WDraws]) : 0;
	const double ghz    = v[WNs] != 0 ? static_cast<double>(v[WCycles]) / static_cast<double>(v[WNs]) : 0;
	const uint64_t fn = g_free_n.exchange(0, std::memory_order_relaxed), fs = g_free_sum.exchange(0, std::memory_order_relaxed);
	auto*          sh = g_shadow.load(std::memory_order_acquire);
	std::printf("[tej] ventana %s: render ocupado %.2f ms, %.1f fps | N %u trabajadores | por fot: DCB %.1f (con sombra %.1f),"
	            " draws %.0f | TRADUCIDOS A TIEMPO %.1f %% | COSIBLES %.1f %% (%.0f/fot; cota solo registros %.1f %%) |"
	            " paquetes a tiempo %.1f %% (%.1f/fot validados, tarde %.1f, saltados %.1f, DCB acabadas antes que la sombra %.1f)"
	            " | fallo de paquete por fot: registros %.1f, epoca buffers %.1f, imagenes %.1f, mapeos %.1f | validacion %.0f"
	            " ns/paquete = %.0f ns/llamada cosida | trabajador %.2f us/llamada (traducir %.2f, %.2f GHz efectivos: <4 ="
	            " expulsado/esperas), %.0f draws/fot, %.2f ms/fot, huecos draw %.1f/fot (otros %.1f), paradas %.1f, fallos %.1f,"
	            " +64*%d %.1f, estado distinto de la pasada %.1f, la mas grande primero %.1f, sin nucleo para el k %.1f |"
	            " lector: DCB %.1f/fot, autocontenidas %.1f, pasada de estado %.1f us/DCB (falla %.1f), estado del render %.1f"
	            " (publicados %.1f), sin estado %.1f, render ya paso %.1f, sin nucleo libre %.1f, otro tipo %.1f, cola llena %.1f,"
	            " espera adelanto %.2f ms/fot; copia %.0f KiB/fot; reafinidad %.1f/fot | CPU libre %.2f nucleos",
	            m == 2 ? (was ? "B (sombra)" : "A (apagada)") : "sombra", busy_ms, fps, sh != nullptr ? sh->N() : 0u, v[GDcbs] / f,
	            v[GDcbSlot] / f, draws / f, pct(v[GDrawsOnTime]), pct(v[GDrawsOk]), v[GDrawsOk] / f, pct(v[GDrawsRegsOk]),
	            seen_pk > 0 ? 100.0 * on_time / seen_pk : 0.0, on_time / f, v[GPkLate] / f, v[GPkSkip] / f, v[GTailLate] / f,
	            v[GPkRegs] / f, v[GPkEpBuf] / f, v[GPkEpImg] / f, v[GPkEpMap] / f, val_ns, val_per_sewn, w_us, wt_us, ghz,
	            v[WDraws] / f, v[WNs] / 1e6 / f, v[WHolesDraw] / f, v[WHolesOther] / f, v[WStops] / f, v[WFailed] / f,
	            static_cast<int>(MaxPackets), v[WOver] / f, v[WPassDiff] / f, v[WBiggest] / f, v[WIdleNoCpu] / f, v[LJobs] / f,
	            v[LSelf] / f, v[LJobs] != 0 ? v[LPassNs] / 1e3 / static_cast<double>(v[LJobs]) : 0.0, v[LPassFail] / f,
	            v[LFromRender] / f, v[GPublish] / f, v[LNoState] / f, v[LBehind] / f + v[WBehind] / f, v[LNoCpu] / f,
	            v[LOther] / f, v[LDropFull] / f, v[LAheadWaitNs] / 1e6 / f, v[LCopyBytes] / 1024.0 / f, v[WRepins] / f,
	            fn != 0 ? static_cast<double>(fs) / 100.0 / static_cast<double>(fn) : -1.0);
	// (tej-enq) Coste fijo fuera del render: gancho en el hilo del juego (lo que alarga SubmitJob FUN_00a18840 y retiene
	// m_queue_mutex), despertares del lector, copia del lector y la pasada (sin prediccion de dispatches).
	std::printf(" | ENQ juego %.3f ms/fot (%.1f envios/fot, %.2f us/envio; alarga SubmitJob y m_queue_mutex) despertares %.1f/fot |"
	            " lector copia %.3f ms/fot (%.1f us/DCB), copia tarde (render ya empezo) %.1f/fot, pasada %.3f ms/fot de CPU,"
	            " copias de estado final 1/16 %.1f/fot | solape trabajador-render %.2f ms/fot",
	            v[JEnqTicks] * ns_per_tick / 1e6 / f, v[JEnqN] / f, v[JEnqN] != 0 ? v[JEnqTicks] * ns_per_tick / 1e3 / v[JEnqN] : 0.0,
	            v[JWakes] / f, v[LCopyNs] / 1e6 / f, v[LJobs] != 0 ? v[LCopyNs] / 1e3 / v[LJobs] : 0.0, v[LCopyLate] / f,
	            v[LPassNs] / 1e6 / f, v[LPassEndCopies] / f, v[WOverlapNs] / 1e6 / f);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (objetivo 0 +-0.2)%s%s", cnt[0], cnt[1], delta,
		            VarOn() && was ? " | variante " : "", VarOn() && was ? VarName[var] : "");
	{
		// (tej-lag) Lo que el render espero por cerrojos compartidos con la sombra (solo si los encontro ocupados) y el
		// coste fijo de los cerrojos de escritor de BufferCache que KYTY_TEJ enciende TAMBIEN en A.
		double wn[TejLag::Count], wms[TejLag::Count];
		for (uint32_t k = 0; k < TejLag::Count; ++k) {
			wn[k]  = static_cast<double>(TejLag::g_acc[k].n.exchange(0, std::memory_order_relaxed)) / f;
			wms[k] = static_cast<double>(TejLag::g_acc[k].ticks.exchange(0, std::memory_order_relaxed)) * ns_per_tick / 1e6 / f;
		}
		double bl = 0, bt = 0;
		if (BcPar::Mode() == 0) { // (con KYTY_BC_PAR los cuenta y vacia su propia linea)
			bl = static_cast<double>(BcPar::g_write_locks.exchange(0, std::memory_order_relaxed)) / f;
			bt = static_cast<double>(BcPar::g_write_ns.exchange(0, std::memory_order_relaxed)) * ns_per_tick / 1e6 / f;
		}
		// (tej-coste) Coste de cada gancho de tej en el render, ms/fot (lado de esta ventana) y pared/ciclos por DCB.
		const int sd = was ? 1 : 0;
		// (tej-lag) A y B en la misma linea: el lado de esta ventana y el ultimo valor del otro (su ventana anterior).
		static double last_ms[2][HN] {}, last_n[2][HN] {}, last_tot[2] {};
		double        tot = 0;
		for (uint32_t k = 0; k < HN; ++k) {
			last_ms[sd][k] = static_cast<double>(g_hook.t[sd][k].exchange(0, std::memory_order_relaxed)) * ns_per_tick / 1e6 / f;
			last_n[sd][k]  = static_cast<double>(g_hook.n[sd][k].exchange(0, std::memory_order_relaxed)) / f;
			if (k != HVal && k != HRanges && k != HRegDiff && k != HPublish && k != HReal && k != HBefore) tot += last_ms[sd][k]; // (esos van dentro de draw / fin DCB)
		}
		last_tot[sd] = tot;
		std::printf(" | COSTE tej en el render ms/fot (A / B; esta ventana %s):", m == 2 ? (was ? "B" : "A") : "sombra");
		for (uint32_t k = 0; k < HN; ++k)
			std::printf(" [%s] %.3f / %.3f (%.0f / %.0f/fot)", HName[k], last_ms[0][k], last_ms[1][k], last_n[0][k], last_n[1][k]);
		std::printf(" = TOTAL A %.3f / B %.3f ms/fot (objetivo B-A <= 0.2) | registro de efectos A %.3f / B %.3f ms/fot",
		            last_tot[0], last_tot[1], last_ms[0][HEff], last_ms[1][HEff]);
		for (int q = 0; q < 2; ++q) {
			const double dn = static_cast<double>(g_hook.dcb_n[q].exchange(0, std::memory_order_relaxed));
			const double dw = static_cast<double>(g_hook.dcb_wall[q].exchange(0, std::memory_order_relaxed)) * ns_per_tick / 1e6 / f;
			const double dc = static_cast<double>(g_hook.dcb_cyc[q].exchange(0, std::memory_order_relaxed)) / 1e6 / f;
			if (q == sd) // (Mciclos/ms = GHz)
				std::printf(" | DCB cola 0 del render: %.1f/fot, pared %.2f ms/fot (con esperas), CPU %.2f Mciclos/fot (%.2f GHz"
				            " efectivos; B-A: sube pared y no ciclos = esperas/expulsion, suben ciclos = calcula mas o cache)",
				            dn / f, dw, dc, dw > 0 ? dc / dw : 0.0);
		}
		{ // (tej-lag4) frecuencia REAL del nucleo del render, migraciones y VirtualProtect, A / B (ultima ventana de cada lado)
			static double gq[2] {}, gmax[2] {}, mig[2] {}, pn[2] {}, pt[2] {};
			auto& P = TejLag::g_probe;
			for (int q = 0; q < 2; ++q) {
				const double n = static_cast<double>(P.fq_n[q].exchange(0, std::memory_order_relaxed));
				const double t = static_cast<double>(P.fq_ticks[q].exchange(0, std::memory_order_relaxed)) * ns_per_tick;
				const double mn = static_cast<double>(P.fq_min[q].exchange(0, std::memory_order_relaxed)) * ns_per_tick;
				const double dc = static_cast<double>(P.dcb[q].exchange(0, std::memory_order_relaxed));
				const double mg = static_cast<double>(P.mig[q].exchange(0, std::memory_order_relaxed));
				const double prn = static_cast<double>(P.prot_n[q].exchange(0, std::memory_order_relaxed));
				const double prt = static_cast<double>(P.prot_t[q].exchange(0, std::memory_order_relaxed)) * ns_per_tick / 1e6;
				if (dc == 0 && prn == 0) continue;
				const double cyc = static_cast<double>(TejLag::FreqIters * TejLag::FreqCyclesPerIter);
				gq[q] = n > 0 && t > 0 ? cyc * n / t : 0.0, gmax[q] = mn > 0 ? cyc / mn : 0.0;
				mig[q] = mg / f, pn[q] = prn / f, pt[q] = prt / f;
			}
			const uint32_t rc = P.cpu.load(std::memory_order_relaxed);
			std::printf(" | LAG4 nucleo del render %d: GHz REALES (cadena imul) A %.2f (max %.2f) / B %.2f (max %.2f) = B/A %.3f"
			            " (<0.98: boost/frecuencia) | migraciones del render A %.2f / B %.2f /fot | VirtualProtect del render A %.0f/fot"
			            " %.3f ms/fot (%.2f us) / B %.0f/fot %.3f ms/fot (%.2f us) (B sube por llamada = TLB shootdown a mas nucleos)",
			            rc < 1024 ? static_cast<int>(rc) : -1, gq[0], gmax[0], gq[1], gmax[1], gq[0] > 0 ? gq[1] / gq[0] : 0.0, mig[0],
			            mig[1], pn[0], pt[0], pn[0] > 0 ? pt[0] * 1e3 / pn[0] : 0.0, pn[1], pt[1], pn[1] > 0 ? pt[1] * 1e3 / pn[1] : 0.0);
			if (!VarOn() && TejLag::SampleDiv() > 1) std::printf(" | MUESTREO 1/%u (KYTY_TEJ_MUESTREO)", TejLag::SampleDiv());
		}
		std::printf(" | CCD trabajadores %s", CcdNow() == 0 ? "mismo (L3 del render)" : CcdNow() == 1 ? "otro" : "auto");
		std::printf(" | RAM proteccion %s", g_mem_lvl.load() == 2 ? "SOMBRA APAGADA" : g_mem_lvl.load() == 1 ? "N=1" : "no"); // (tej-lag3)
		std::printf(" | LAG render espera por la sombra: cerrojo buffers %.1f veces %.3f ms/fot, etiquetas %.1f %.3f ms/fot,"
		            " publicar estado saltado %.1f/fot | cerrojos escritor buffers (tambien en A) %.1f/fot %.3f ms/fot",
		            wn[TejLag::BcWrite], wms[TejLag::BcWrite], wn[TejLag::Labels], wms[TejLag::Labels], wn[TejLag::Publish], bl, bt);
		// (tej-var) RenderMutex: el render esperando a un hilo que lo pidio (OwnerLockSlow) y las peticiones de la sombra
		// (cada una un FlushProcessWriteBuffers: IPI a todos los nucleos, el del render incluido).
		std::printf(" | RENDERMUTEX render espera %.1f veces %.3f ms/fot | sombra pide %.1f/fot (IPI), espera %.3f ms/fot |"
		            " var: no toman (N=1) %.1f/fot, sin traducir %.1f DCB/fot",
		            wn[TejLag::RmOwner], wms[TejLag::RmOwner], wn[TejLag::RmReq], wms[TejLag::RmReq], v[WVarIdle] / f, v[WVarSkip] / f);
		// (tej-sinmutex) Con KYTY_TEJ_SINMUTEX: 'sombra pide' debe ser 0; evitadas = huecos 'sin mutex' (en huecos por motivo).
		std::printf(" | SINMUTEX %s: peticiones evitadas %.1f/fot (huecos 'sin mutex'), barreras sin cerrojo %.1f/fot | render espera"
		            " catalogo (CatalogWrite) %.1f veces %.3f ms/fot",
		            TejLag::g_sinmutex.load(std::memory_order_relaxed) ? "si" : "no", wn[TejLag::RmSkip], wn[TejLag::BarFree],
		            wn[TejLag::CatW], wms[TejLag::CatW]);
	}
	std::printf("\n");
	// Modelo N=1,2,4,6 con los tiempos de esta ventana (solo B / modo 1: con draws contados).
	std::vector<DcbRec> dcbs;
	{
		std::lock_guard g(g_dcb_mutex);
		dcbs.swap(g_dcbs);
	}
	if (!dcbs.empty()) {
		// (F1c) coste por LLAMADA traducida (solo DCB con >= 32), media ponderada por llamadas; fijo por DCB aparte.
		const double avg = v[WCallsMeasured] != 0 ? static_cast<double>(v[WTransMeasuredNs]) / static_cast<double>(v[WCallsMeasured])
		                                          : 0.0;
		const double fix = v[WJobs] != 0 ? static_cast<double>(v[WFixNs]) / static_cast<double>(v[WJobs]) : 0.0;
		double       rc = 0, rn = 0; // (coste del render por llamada en estas DCB: su tiempo de DCB entre sus llamadas)
		for (const auto& d: dcbs) rc += static_cast<double>(d.re - d.rs), rn += std::max(d.calls, d.draws);
		const double render_cw = rn > 0 ? rc / rn : 0.0;
		std::printf("[tej] modelo (%zu DCB, trabajador %.2f us/llamada + fijo %.1f us/DCB; render %.2f us/llamada): traducidos a"
		            " tiempo con N=1 %.1f %%, N=2 %.1f %%, N=4 %.1f %%, N=6 %.1f %% (N actual %u: modelo %.1f %% frente a medido %.1f %%)",
		            dcbs.size(), avg / 1e3, fix / 1e3, render_cw / 1e3, Model(dcbs, 1, avg, fix), Model(dcbs, 2, avg, fix),
		            Model(dcbs, 4, avg, fix), Model(dcbs, 6, avg, fix), sh != nullptr ? sh->N() : 0u,
		            Model(dcbs, sh != nullptr ? sh->N() : 1u, avg, fix), pct(v[GDrawsOnTime]));
		const auto at = [&](double cw_ns, uint32_t n) {
			auto c = dcbs;
			for (auto& d: c) d.cw_ps = static_cast<uint64_t>(cw_ns * 1000.0);
			return Model(c, n, cw_ns, fix);
		};
		std::printf(" | con trabajador al coste del render (%.2f us/llamada): N=1 %.1f %%, N=2 %.1f %%, N=4 %.1f %%\n", render_cw / 1e3,
		            at(render_cw, 1), at(render_cw, 2), at(render_cw, 4));
	}
	{
		std::lock_guard g(g_lat_mutex);
		const auto& a = g_lat[0];
		const auto& b = g_lat[1];
		std::printf("[tej] latencia Enqueue->flip (partida): A media %.2f ms p99 %.1f ms, >25 ms %.2f %% (%llu fot) | B media"
		            " %.2f ms p99 %.1f ms, >25 ms %.2f %% (%llu fot) | B-A media %+.2f ms (no pasa si > +0.2 o picos suben)\n",
		            a.n ? a.sum_ns / 1e6 / a.n : 0.0, a.P99(), a.frames ? 100.0 * a.over25 / a.frames : 0.0,
		            static_cast<unsigned long long>(a.frames), b.n ? b.sum_ns / 1e6 / b.n : 0.0, b.P99(),
		            b.frames ? 100.0 * b.over25 / b.frames : 0.0, static_cast<unsigned long long>(b.frames),
		            (b.n ? b.sum_ns / 1e6 / b.n : 0.0) - (a.n ? a.sum_ns / 1e6 / a.n : 0.0));
	}
	if (sh != nullptr) {
		std::vector<std::pair<const char*, uint64_t>> holes, stops;
		std::unordered_map<uint64_t, uint64_t>        waits;
		sh->TakeReasons(holes, stops, waits);
		const auto top = [&](const char* title, std::vector<std::pair<const char*, uint64_t>>& list) {
			std::ranges::sort(list, [](const auto& a, const auto& b) { return a.second > b.second; });
			std::printf(" %s:", title);
			for (size_t i = 0; i < list.size() && i < 8; ++i) std::printf(" [%s] %.1f", list[i].first, list[i].second / f);
		};
		if (!holes.empty() || !stops.empty()) {
			std::printf("[tej] por fot");
			top("huecos (draws) por motivo", holes);
			top("| paradas", stops);
			std::printf("\n");
		}
		// (F1c) Esperas supuestas: direccion, veces/fot que el trabajador la supuso, y en el render: cumplida al llegar
		// frente a tuvo que esperar, y si ahora es etiqueta escrita por un procesador (Spec::IsLabel).
		std::vector<std::pair<uint64_t, uint64_t>> wl(waits.begin(), waits.end());
		std::ranges::sort(wl, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> rw;
		{
			std::lock_guard g(g_rwaits.mutex);
			rw.swap(g_rwaits.by_addr);
		}
		std::printf("[tej] F1c esperas supuestas %.1f/fot (%s), direcciones %zu:", v[WAssumed] / f,
		            AssumeOn() ? "KYTY_TEJ_ESPERA=1, sigue" : "KYTY_TEJ_ESPERA=0, para", wl.size());
		for (size_t i = 0; i < wl.size() && i < 6; ++i) {
			const auto it = rw.find(wl[i].first);
			std::printf(" [0x%llx %.1f/fot, render: cumplida %llu, espero %llu, etiqueta %s]",
			            static_cast<unsigned long long>(wl[i].first), wl[i].second / f,
			            static_cast<unsigned long long>(it != rw.end() ? it->second.first : 0),
			            static_cast<unsigned long long>(it != rw.end() ? it->second.second : 0),
			            Libs::Graphics::Spec::IsLabel(wl[i].first) ? "si" : "no");
		}
		std::printf("\n");
	}
	{
		// (F1c) Registros: lo que difiere en el punto de cose, top 10 (aparece / unico).
		std::vector<std::pair<const char*, uint64_t>> any, sole;
		{
			std::lock_guard g(g_regdiff.mutex);
			any.assign(g_regdiff.any.begin(), g_regdiff.any.end());
			sole.assign(g_regdiff.sole.begin(), g_regdiff.sole.end());
			g_regdiff.any.clear();
			g_regdiff.sole.clear();
		}
		std::ranges::sort(any, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::printf("[tej] F1c registros distintos (paquetes/fot: %.1f, %.1f nombres/paquete; muestra 1/%u, cuentas x%u): aparece",
		            v[GPkRegs] / f,
		            v[GPkRegsSampled] != 0 ? static_cast<double>(v[GRegDiffNames]) / static_cast<double>(v[GPkRegsSampled]) : 0.0,
		            RegDiffEvery, RegDiffEvery);
		for (size_t i = 0; i < any.size() && i < 10; ++i) std::printf(" [%s] %.1f", any[i].first, static_cast<double>(any[i].second) * RegDiffEvery / f);
		std::ranges::sort(sole, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::printf(" | unico (arreglarlo cose el paquete):");
		for (size_t i = 0; i < sole.size() && i < 10; ++i) std::printf(" [%s] %.1f", sole[i].first, static_cast<double>(sole[i].second) * RegDiffEvery / f);
		std::printf("\n");
		// (F1c) Resumen por llamada y cosibles con/sin espera supuesta.
		std::printf("[tej] F1c llamadas render %.0f/fot (draws %.0f) | trabajador %.0f llamadas/fot, traducir %.2f us/llamada"
		            " (DCB >=32: %.2f), fijo %.1f us/DCB | paquetes todo huecos %.1f/fot | COSIBLES sin espera supuesta %.1f %%,"
		            " con espera supuesta %.1f %% (a tiempo tras espera %.1f %%)\n",
		            v[GCalls] / f, draws / f, v[WCalls] / f,
		            v[WCalls] != 0 ? static_cast<double>(v[WTransNs]) / 1e3 / static_cast<double>(v[WCalls]) : 0.0,
		            v[WCallsMeasured] != 0 ? static_cast<double>(v[WTransMeasuredNs]) / 1e3 / static_cast<double>(v[WCallsMeasured]) : 0.0,
		            v[WJobs] != 0 ? static_cast<double>(v[WFixNs]) / 1e3 / static_cast<double>(v[WJobs]) : 0.0, v[GPkAllHoles] / f,
		            pct(v[GDrawsOk] - v[GDrawsOkAssumed]), pct(v[GDrawsOkAssumed]), pct(v[GDrawsOnTimeAssumed]));
		// (F1d) Cosibles con registros solo de draws, y ademas con epocas por rangos.
		std::printf("[tej] F1d COSIBLES regs de draws %.1f %% | + epocas por rangos %.1f %% (antes %.1f %%) | CS distinto ignorado"
		            " (paquete sin dispatches) %.1f/fot, CS distinto con dispatches %.1f/fot | rangos salvan: buffers %.1f"
		            " (fallan %.1f), imagenes %.1f (fallan %.1f), mapeos %.1f (fallan %.1f), sin lista %.1f /fot, %.1f rangos/paquete"
		            " comprobados (de %.1f sin compactar; solo paquetes con registros que cosen)"
		            " | validacion %.0f ns/paquete\n",
		            pct(v[GDrawsOkGfx]), pct(v[GDrawsOkRanges]), pct(v[GDrawsOk]), v[GPkCsIgnored] / f, v[GPkCsDispatch] / f,
		            v[GPkRgBuf] / f, v[GPkRgFailBuf] / f, v[GPkRgImg] / f, v[GPkRgFailImg] / f, v[GPkRgMap] / f,
		            v[GPkRgFailMap] / f, v[GPkRgNoList] / f,
		            v[GValN] != 0 ? static_cast<double>(v[GRgEntries]) / static_cast<double>(v[GValN]) : 0.0,
		            v[GValN] != 0 ? static_cast<double>(v[GRgRaw]) / static_cast<double>(v[GValN]) : 0.0, val_ns);
		// (tej F2p) Cosibles reales: los F1d pasados por las comprobaciones de CommitSegments; Before estimado; prevision.
		// (tej-cos2) La muestra F2p sin escalar: sc = draws F1d de todo / draws F1d de las DCB muestreadas.
		const double sc = v[GRealBaseD] != 0 ? static_cast<double>(v[GDrawsOkRanges]) / static_cast<double>(v[GRealBaseD]) : 0.0;
		const auto   pr = [&](uint64_t x) { return pct(static_cast<uint64_t>(static_cast<double>(x) * sc + 0.5)); };
		uint64_t rk_all = 0;
		for (uint32_t k = 0; k < RN; ++k) rk_all += v[GRk0 + k];
		std::printf("[tej] F2p COSIBLES REALES %.1f %% (F1d %.1f %%; sin efectos de la propia DCB %.1f %%) | paquetes F1d"
		            " comprobados %.1f/fot:",
		            pr(v[GRealDraws]), pct(v[GDrawsOkRanges]), pr(v[GRealDrawsPre]), rk_all * sc / f);
		for (uint32_t k = 0; k < RN; ++k) std::printf(" [%s] %.1f", RNames[k], v[GRk0 + k] * sc / f);
		std::printf(" | muestra %.0f draws/fot de %.0f (x%.2f)", v[GRealBaseD] / f, v[GDrawsOkRanges] / f, sc);
		const double bf_us  = v[GBfPk] != 0 ? static_cast<double>(v[GBfTicks]) * ns_per_tick / 1e3 / static_cast<double>(v[GBfPk]) : 0.0;
		const double bpk    = v[GBfPk] != 0 ? static_cast<double>(v[GBfPk]) : 1.0;
		std::printf(" | efectos del render %.0f entradas/fot | Before (muestra 1/%u de las DCB 1/%u, %.1f paquetes/fot): consultas %.2f us/paquete,"
		            " %.1f rangos/paquete, sucios %.1f (%.1f KiB a subir), BDA %.1f rangos/paquete (todo %.1f %%), invalidaciones"
		            " %.1f, imagenes %.1f (BDA/imagenes/subidas sin coste medido: no se ejecutan)\n",
		            v[GEffEntries] / f, BeforeEvery, RealEvery, v[GBfPk] / f, bf_us, v[GBfRanges] / bpk, v[GBfDirtyRanges] / bpk,
		            static_cast<double>(v[GBfDirtyBytes]) / 1024.0 / bpk, v[GBfBda] / bpk, 100.0 * v[GBfBdaAll] / bpk,
		            v[GBfInv] / bpk, v[GBfImg] / bpk);
		const double calls_f = v[GRealCalls] * sc / f, pk_f = v[GRealPk] * sc / f;
		std::printf("[tej] F2-prevision cosibles reales %.1f %% | llamadas cosibles reales %.0f/fot (%.1f paquetes/fot) | Before"
		            " estimado %.2f us/paquete (solo consultas) | ahorro neto en el render = %.0f x 1.15 us - %.1f x %.2f us ="
		            " %+.2f ms/fot (sin submit: con secundarios desaparece; sin el coste de subidas/BDA/imagenes)\n",
		            pr(v[GRealDraws]), calls_f, pk_f, bf_us, calls_f, pk_f, bf_us, (calls_f * 1.15 - pk_f * bf_us) / 1e3);
		// (tej-cos) Embudo en % de todos los draws: cada paso y su caida, con el motivo principal.
		const auto     scl     = [&](uint64_t x) { return static_cast<uint64_t>(static_cast<double>(x) * sc + 0.5); };
		const uint64_t c_meta  = v[GDrawsOkRanges] - std::min(v[GDrawsOkRanges], scl(v[GRkD1] + v[GRkD2] + v[GRkD3]));
		const uint64_t c_reads = c_meta - std::min(c_meta, scl(v[GRkD4]));
		std::printf("[tej] COS embudo: draws %.0f/fot -> a tiempo %.1f %% -> sin huecos %.1f %% -> regs draws %.1f %% (CS con"
		            " dispatches %.1f) -> rangos %.1f %% (pierde buffers %.1f, imagenes %.1f, mapeos %.1f) -> metadatos %.1f %% ->"
		            " clean_reads %.1f %% -> PASA %.1f %% | buffers exactos (de ellos por filtro de regiones %.1f): salvan %.1f pk/fot, fallan %.1f, sin anillo %.1f |"
		            " imagenes solo sobre imagenes salvan %.1f pk/fot (de ellas tocarian buffers/lecturas %.1f) | clean_reads"
		            " escritos por GPU %.1f pk/fot, por CPU %.1f\n",
		            draws / f, pct(v[GDrawsOnTime]), pct(v[GCosSewn]), pct(v[GCosGfxOk]), pct(v[GCosCsBad]), pct(v[GDrawsOkRanges]),
		            pct(v[GCosLostBuf]), pct(v[GCosLostImg]), pct(v[GCosLostMap]), pct(c_meta), pct(c_reads), pr(v[GRkD0]),
		            v[GCosBufRegionOk] / f, v[GCosBufExactOk] / f, v[GCosBufExactFail] / f, v[GCosBufExactLost] / f, v[GCosImgOnlyOk] / f,
		            v[GCosImgOverOther] / f, v[GCosReadGpu] * sc / f, v[GCosReadHost] * sc / f);
		// (tej-cosibles) validacion por valor de las lecturas limpias escritas por el render y reenvio de las DCB anteriores
		std::printf("[tej] COS3 por valor: lecturas escritas por el render que pasan %.1f/fot, distintas %.1f, sin bytes %.1f"
		            " (muestra x%.2f) | reenvio: DCB con escrituras de antes %.1f/fot (%.0f escrituras/fot), lecturas de cadena"
		            " %.1f/fot | bytes de mas %.1f DCB/fot\n",
		            v[GCosValOk] * sc / f, v[GCosValDiff] * sc / f, v[GCosValNone] * sc / f, sc, v[GCosFwdDcb] / f,
		            v[GCosFwdWrites] / f, v[GCosFwdReads] / f, v[GCosSnapOver] / f);
		std::printf("[tej] COS2 exactos: imagenes (4 KiB, anillo) pasan %.1f pk/fot, fallan %.1f, sin anillo %.1f | desmapeos (bytes,"
		            " anillo) pasan %.1f, fallan %.1f, sin anillo %.1f | metadatos borrados sobre imagen del paquete %.1f, fuera %.1f"
		            " pk/fot | CS distinto muestreado %.1f/fot\n",
		            v[GCosImgExactOk] / f, v[GCosImgExactFail] / f, v[GCosImgExactLost] / f, v[GCosMapExactOk] / f,
		            v[GCosMapExactFail] / f, v[GCosMapExactLost] / f, v[GCosMetaOnImg] * sc / f, v[GCosMetaOffImg] * sc / f,
		            v[GCosCsSampled] / f);
		std::vector<std::pair<const char*, uint64_t>>  regs;
		std::vector<std::pair<uint64_t, CosDiag::Rd>> reads;
		std::vector<std::pair<uint64_t, std::array<uint64_t, 16>>> metas;
		std::vector<std::pair<uint64_t, CosDiag::Hit>>              ihits, mhits;
		{
			std::lock_guard g(g_cos.mutex);
			ihits.assign(g_cos.img_hits.begin(), g_cos.img_hits.end());
			mhits.assign(g_cos.map_hits.begin(), g_cos.map_hits.end());
			g_cos.img_hits.clear(), g_cos.map_hits.clear();
			regs.assign(g_cos.regs.begin(), g_cos.regs.end());
			reads.assign(g_cos.reads.begin(), g_cos.reads.end());
			metas.assign(g_cos.metas.begin(), g_cos.metas.end());
			g_cos.regs.clear(), g_cos.reads.clear(), g_cos.metas.clear();
		}
		std::ranges::sort(regs, [](const auto& x, const auto& y) { return x.second > y.second; });
		std::ranges::sort(reads, [](const auto& x, const auto& y) { return x.second.n > y.second.n; });
		const auto msum = [](const std::array<uint64_t, 16>& o) {
			uint64_t t = 0;
			for (const auto x: o) t += x;
			return t;
		};
		std::ranges::sort(metas, [&](const auto& x, const auto& y) { return msum(x.second) > msum(y.second); });
		std::printf("[tej] COS regs de draws distintos (muestra 1/%u, x%u, pk/fot):", CosRegEvery, CosRegEvery);
		for (size_t i = 0; i < regs.size() && i < 10; ++i)
			std::printf(" [%s] %.1f", regs[i].first, static_cast<double>(regs[i].second) * CosRegEvery / f);
		std::printf(" | clean_reads escritos (leido -> escritor, pk/fot):");
		static constexpr const char* src_name[3] = {"normal", "hueco", "commit"};
		for (size_t i = 0; i < reads.size() && i < 8; ++i) {
			const auto& x = reads[i].second;
			std::printf(" [0x%llx +%llu <- %s %s 0x%llx +%llu] %.2f", static_cast<unsigned long long>(reads[i].first),
			            static_cast<unsigned long long>(x.size), x.kind == 1 ? "GPU" : "CPU", src_name[x.source % 3],
			            static_cast<unsigned long long>(x.w_addr), static_cast<unsigned long long>(x.w_size),
			            static_cast<double>(x.n) * sc / f);
		}
		std::printf(" | metadatos borrados (superficie: origen=veces, pk/fot):");
		static constexpr const char* org[8] = {"otro", "compute clear", "depth clear draw", "CP fill", "DCC fill",
		                                       "depth bind record", "?6", "?7"};
		for (size_t i = 0; i < metas.size() && i < 8; ++i) {
			std::printf(" [0x%llx %.2f:", static_cast<unsigned long long>(metas[i].first), static_cast<double>(msum(metas[i].second)) * sc / f);
			for (uint32_t o = 0; o < 16; ++o)
				if (metas[i].second[o] != 0) std::printf(" %s%s=%llu", org[o & 7u], o >= 8 ? " en hueco" : "",
				                                          static_cast<unsigned long long>(metas[i].second[o]));
			std::printf("]");
		}
		std::printf("\n");
		// (tej-cos2) Quien tumba imagenes y mapeos: el cambio exacto (direccion, tamano, alta/baja u origen), pk/fot.
		const auto by_n = [](const auto& x, const auto& y) { return x.second.n > y.second.n; };
		std::ranges::sort(ihits, by_n), std::ranges::sort(mhits, by_n);
		static constexpr const char* uorg[11] = {"otro", "MapNamedFlexible", "SetPrtAperture", "munmap", "ReleaseDirect",
		                                         "MapDirect", "ReplaceFixed", "ReserveVirtualRange", "AllocGuestRuntime",
		                                         "FreeGuest", "PoolDecommit"};
		std::printf("[tej] COS2 imagenes que tumban (%zu distintas):", ihits.size());
		for (size_t i = 0; i < ihits.size() && i < 10; ++i)
			std::printf(" [0x%llx +0x%llx %s] %.2f", static_cast<unsigned long long>(ihits[i].first),
			            static_cast<unsigned long long>(ihits[i].second.end - ihits[i].first), ihits[i].second.tag ? "alta" : "baja",
			            ihits[i].second.n / f);
		std::printf(" | desmapeos que tumban (%zu distintos):", mhits.size());
		for (size_t i = 0; i < mhits.size() && i < 10; ++i)
			std::printf(" [0x%llx +0x%llx %s] %.2f", static_cast<unsigned long long>(mhits[i].first),
			            static_cast<unsigned long long>(mhits[i].second.end - mhits[i].first), uorg[mhits[i].second.tag % 11],
			            mhits[i].second.n / f);
		std::printf("\n");
	}
	if (m == 2 && VarOn()) {
		std::printf("[tej] VARIANTES delta interpolado B-A del render ms/fot (media [min..max] n):");
		for (int i = 0; i < VarN; ++i)
			std::printf(" | %s %+.3f [%+.2f..%+.2f] %.0f", VarName[i], vcnt[i] > 0 ? vsum[i] / vcnt[i] : 0.0, vmin[i], vmax[i], vcnt[i]);
		std::printf(" | objetivo <= +0.2\n");
	}
	std::fflush(stdout);
	if (m == 2) {
		// (tej-var) Al pasar de A a B: la siguiente variante.
		static int next = 0;
		if (!was && VarOn()) {
			g_var.store(next, std::memory_order_relaxed);
			TejLag::g_polite.store(next == 4, std::memory_order_relaxed);
			TejLag::g_sinmutex.store(next == 5 || next == 6 || next == 7, std::memory_order_relaxed); // (tej-sinmutex; tej-lag3: 6 tambien)
			next = (next + 1) % VarN;
		}
		g_ab_on.store(!was, std::memory_order_relaxed);
	}
}

} // namespace Tej
