#pragma once
// (VRAM-TOPE, 11-25) KYTY_VRAM_TOPE=1 (por defecto; =0 lo apaga y todo vuelve a VRAM_TECHO / VRAM_DRV2 como antes).
// Pedido del usuario: la VRAM nunca se desborda ni da texturas corruptas; cerca del tope el juego baja calidad
// PROPORCIONALMENTE, como un juego de PC, y la recupera poco a poco al bajar. Un unico tope:
//   tope tabla (uso del PROCESO) por tamano de tarjeta: 8 GB -> 7,2 GB; 10 -> 8,5; 12 -> 10; 16 -> 14,5 (interpolado;
//   por debajo de 8 GB el 90 %, por encima de 16 GB el 90,6 %). Con KYTY_SIMULATE_VRAM_MB cuenta la tarjeta simulada.
//   tope efectivo = min(tope tabla, presupuesto del driver para el proceso - 128 MiB). Log 20261125-072323 (8 GB
//   simulados): driver 5625 MiB porque otros procesos tienen ~1,7 GB de la tarjeta: con la tabla sola (7373) el driver
//   paginaria. La linea [vram-tope] dice cual de los dos manda.
// Escalones (uso del proceso / tope efectivo):
//   nivel 1 >= 90 %: VRAM_TECHO (vram-juego.h): el juego suelta los mips que el mismo ya no pide (pasada 1 de 0xac9d80;
//     lo cercano conserva su mip). Se quita bajo el 80 %.
// (TOPE3, 11-25, decision del usuario) NIVELES NUEVOS: 1 = igual; 2 = bufferes frios a la RAM del sistema (BUF-RAM,
//   >= 95 %); el pool del juego ya NO se baja nunca; 3 = solo emergencia, hueco del driver < 128 MiB, limites DRV2.
//   Lo de abajo (nivel 2 pool) es historia.
//   (antiguo) nivel 2 >= 95 %: se baja el pool de streaming del juego (*0x2f69898) lo que sobra sobre el 92 % (cada 3 s, minimo
//     el 50 % del original). Ghidra 01.007.000 (abajo): es seguro en caliente y el recorte es proporcional. Bajo el
//     85 % se devuelve +128 MiB cada 2 s hasta el original.
//   (TOPE2, 11-25) Log 20261125-074401 (resolucion alta + FSR 4, 8 GB simulados): con el pool al 50 % (1536) el
//     presupuesto del juego (mgr+0x286038) quedo en 0-200 MiB: NO es un puntero malo (con el pool en 3072/2440/1838
//     daba 2293/2162/1883, sigue al pool). Ghidra 0xac9d80: presupuesto = pool<<20 - fijo (mgr+0x286048, lo que el juego
//     no puede bajar), y si *0x2f69998 != 0, min con el libre de su monton (0x80d540: paginas libres de 0x2ee4d60 *
//     64 KiB - *0x2f69998 MiB - *0x2f69a98 MiB). Con presupuesto <= 0 la pasada 2 deja TODO lo no fijado en el mip
//     minimo: texturas borrosas, y la VRAM no bajo (lo streameado por el juego, "nativas", era solo 130-170 MiB).
//     Ahora: el pool nunca baja si el presupuesto del juego queda bajo PresMinMiB (256) y se sube +128 si cae por debajo
//     de PresMinMiB/2. Bajar mas el suelo (25 %) no se hace: con presupuesto 0 ya no queda nada del juego que soltar.
//   nivel 3 >= 98 %: expulsion de imagenes sin usar (recolector VRAM_DRV2 de textureCache.cpp, con su histeresis, su
//     limite de MiB/s y su proteccion de re-subidas) hasta el 93 %. Nunca lo usado en los ultimos InUseTicks (8)
//     fotogramas: nada del fotograma actual.
//   (TOPE2) Log 074401: el nivel 3 expulso 36 GB en una partida con "sin usar >30 s" = 0 MiB en [vram-desg]: todo lo
//     expulsado se usaba y se volvia a subir (texturas mal). Ahora con el tope el nivel 3 va SIEMPRE con los limites de
//     VRAM_DRV2 (32 MiB/s, re-subidas protegidas 60 s, >= 300 fotogramas sin usar) y solo si el nivel 2 ya no puede
//     bajar mas; si en 5 s lo re-subido llega a la mitad de lo expulsado, o el uso no baja la mitad de lo expulsado,
//     es inutil (lo que llena la VRAM son RT/bufferes/driver): se bloquea 30 s. La linea [vram-tope] da el reparto.
//
// Ghidra (eboot PPSA01341 01.007.000, base 0x900000000):
//  - Lectores de *0x2f69898 (pool): 0xac9f50 en 0xac9d80 (gestor) y 0xac7caa/0xac80c9/0xac841f/0xac8523 en 0xac7c76
//    (solo el texto de depuracion "Texture Pool: Base ... Budget"); escritor unico 0xad4a3b en 0xad3070 (config). No
//    dimensiona ninguna reserva: cambiarlo en caliente es seguro.
//  - 0xac9d80 lo lee UNA vez por pasada (iVar3) y calcula presupuesto = pool<<20 - base (mgr+0x286048), guardado en
//    mgr+0x286038. mgr = *(0x2f6a138) (0x903f84 lea rax,[0x2f6a138]; 0x903f97 jmp 0xac9d80). Si lo pedido no cabe:
//    pasada 1 (0xacafef): ordena (0xacc070 modo 3) y baja al deseado solo las que tienen mas mip del pedido, y PARA en
//    cuanto cabe; pasada 2 (0xacb432): solo si aun falta, en el mismo orden pone deseado 0 textura a textura y PARA en
//    cuanto cabe (uStack_1b0 >= 0). Asi el recorte es proporcional a lo que se baja el pool: lo menos importante
//    (no muestreado, mas texeles de mas) primero; lo fijado (tex+0x6c > 0) no.
//  - Cambio de mundo: FUN_002f0fd0 (0x2f0feb mov rcx,[rip->0x27bf378]; 0x2f0ff4 mov eax,[rcx+0xc]) guarda el mapa
//    pedido en *(*(0x27bf378)+0xc); lo llaman la carga de area (FUN_00781b70: "map:/m%02d_%02d_00_00/....arealoadlist"
//    con area = id>>24), MoveMapStep (FUN_00793b70 "Loading world %d-%d (area:%d block:%d)") y los saltos de mapa.
//    0x63646464 = ninguno. Aqui solo se LEE cada segundo: si cambia el mundo (id>>24; id completo si >= 0x8000000), a
//    los MundoEsperaMs (15 s: ya cargado y dibujando el nuevo) el recolector expulsa durante MundoVentanaMs las
//    imagenes que no se han usado desde el cambio (las del mundo anterior), sin limite de MiB/s; lo que el mundo nuevo
//    usa tiene marca posterior y no se toca. Asi la VRAM no se vuelve a llenar al cambiar de mundo.
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gestor-memoria.h" // (GESTOR-MEM) misma tabla de topes que el gestor
#include "vram-drv.h"
#include "vram-juego.h"

namespace VramTope {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_TOPE");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

constexpr uint64_t MiB            = uint64_t {1} << 20u;
constexpr uint64_t DrvMarginMiB   = 128;
constexpr uint64_t N1Pct = 90, N1OffPct = 80;   // VRAM_TECHO (lejanas)
constexpr uint64_t N2Pct = 95, N2GoalPct = 92, N2UpPct = 85;
constexpr uint64_t N3Pct = 98, N3LowPct = 93;   // expulsion (VRAM_DRV2)
constexpr int      PoolFloorPct   = 50;
constexpr int64_t  PresMinMiB     = 256;   // (TOPE2) presupuesto minimo que se le deja al juego (mgr+0x286038)
constexpr uint64_t N3BloqueoMs    = 30000; // (TOPE2) nivel 3 bloqueado tras una ventana inutil
constexpr uint64_t N3MinEvalMiB   = 128;   // (TOPE2) expulsado minimo en 5 s para juzgar la ventana
constexpr int      PoolStepUpMiB  = 128;
constexpr uint64_t PoolDownMs     = 3000, PoolUpMs = 2000;
constexpr uint64_t MundoEsperaMs  = 15000, MundoVentanaMs = 10000;

constexpr uint64_t kMapPtrRva   = 0x27bf378; // puntero a la estructura con el mapa pedido en +0xc
constexpr uint64_t kMapRdRva    = 0x2f0feb;  // mov rcx, [rip->0x27bf378] en FUN_002f0fd0
constexpr uint64_t kMgrPtrRva   = 0x2f6a138; // puntero al gestor de streaming (param_1 de 0xac9d80)
constexpr uint64_t kMgrLeaRva   = 0x903f84;  // lea rax, [rip->0x2f6a138]
constexpr int      kMapNone     = 0x63646464;

// Tope de la tabla del usuario (MiB) para una tarjeta de card_mib.
inline uint64_t TablaMiB(uint64_t card_mib) {
	struct P { double card, tope; };
	const P t[] = {{8192, (GestorMem::On() ? 6.3 : 7.2) * 1024}, {10240, (GestorMem::On() ? 7.5 : 8.5) * 1024}, {12288, 10.0 * 1024}, {16384, 14.5 * 1024}};
	const double c = static_cast<double>(card_mib);
	if (c <= t[0].card) return static_cast<uint64_t>(c * t[0].tope / t[0].card);
	for (int i = 0; i + 1 < 4; i++) {
		if (c <= t[i + 1].card) {
			const double f = (c - t[i].card) / (t[i + 1].card - t[i].card);
			return static_cast<uint64_t>(t[i].tope + f * (t[i + 1].tope - t[i].tope));
		}
	}
	return static_cast<uint64_t>(c * t[3].tope / t[3].card);
}

struct Estado {
	// medidas del ultimo tick (MiB)
	uint64_t card = 0, tabla = 0, drv = 0, tope = 0, uso = 0;
	int      nivel = 0;
	bool     manda_driver = false;
	// pool del juego (nivel 2)
	bool          pool_checked = false, pool_valid = false;
	volatile int* pool = nullptr;
	int           pool_orig = 0, pool_floor = 0, pool_written = 0;
	uint64_t      pool_last_ms = 0, pool_bajadas = 0;
	// cambio de mundo
	bool     map_checked = false, map_valid = false;
	int      map_key = -1, map_id = 0;
	uint64_t mundo_ms = 0, mundo_tick = 0, mundos = 0;
	// log
	uint64_t last_log_ms = 0, ev0 = 0, freed0 = 0, mundo_bytes0 = 0;
	// (TOPE2) nivel 3
	uint64_t reup0 = 0, uso0 = 0, n3_bloqueo_hasta = 0, n3_bloqueos = 0;
	// (N3-REAL)
	bool     n3_on = false;
	uint64_t n3_seguidos = 0;
	int64_t  hueco = 0;
	uint64_t evt0 = 0, evn0 = 0, evo0 = 0, prot0 = 0;
	bool     pool_al_suelo = false;
};
inline Estado& E() {
	static Estado e;
	return e;
}

// Lo leen el recolector (hilo de render) y el log.
inline std::atomic<uint64_t> g_tope_bytes {0};       // tope efectivo (bytes); 0 = aun sin medir
inline std::atomic<uint64_t> g_mundo_tick {0};       // marca del recolector en el cambio de mundo
inline std::atomic<bool>     g_mundo_activo {false}; // ventana de expulsion del mundo anterior
inline std::atomic<uint64_t> g_mundo_n {0}, g_mundo_bytes {0}; // expulsado por cambio de mundo
inline std::atomic<bool>     g_n3_permitido {false}; // (TOPE2) el recolector solo expulsa por el tope si es true
// (N3-BLOQ, 11-25) Nivel 3 bloqueado por inutil: NINGUN camino del recolector expulsa (ni el del hueco del driver, ni
// ReclaimImageMemory sin `all`, ni el antiguo de vram8); solo el cambio de mundo (lo no usado desde el cambio).
inline std::atomic<bool>     g_n3_bloqueado {false};
// (N3-BLOQ) Nativas (texturas dispersas de CTexture): nunca se expulsa una usada en los ultimos NatUsoTicks fotogramas
// (~30 s) fuera del cambio de mundo. Log 20261125-104537: 1796 nativas expulsadas en 30 s (248 MB ligados) y 943
// re-subidas (3058 MiB): lo expulsado de 5 s volvia enseguida.
constexpr uint64_t NatUsoTicks = 1800;
// (TOPE2) Reparto medido en el recorrido de [vram-desg] (cada 30 s, textureCache.cpp), bytes.
inline std::atomic<uint64_t> g_rep_tex {0}, g_rep_rt {0}, g_rep_gpu {0}, g_rep_otras {0}, g_rep_buf {0}, g_rep_stale {0};

// (BUF-RAM, 11-25) Nivel 2 nuevo: bufferes frios de la cache (sin uso de la GPU en >= 5 s, no bda_read) pasan de VRAM a
// memoria del sistema (VMA, tipo sin DEVICE_LOCAL: la GPU los lee por PCIe), copia GPU->GPU y la copia de VRAM se suelta
// (BufferCache::BufRamPass, bufferCache.cpp). Vuelven a VRAM si se usan y hay sitio (histeresis). KYTY_VRAM_BUF_RAM=0 lo apaga.
inline bool BufRamOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_BUF_RAM");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return On() && on;
}
// (W17) KYTY_VRAM_MITAD=1: objetivo = mitad del tope efectivo; los bufferes migran sin limite de MiB, por ultimo uso.
inline bool Mitad() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_MITAD");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
constexpr uint64_t BufRamPct     = 95; // nivel 2: migrar desde el 95 % del tope
constexpr uint64_t BufRamBackPct = 90; // volver a VRAM solo si uso + tamano < 90 % (histeresis 5 puntos)
constexpr uint64_t N3HuecoMiB    = 128; // (TOPE3) nivel 3 (expulsar imagenes) solo si el hueco del driver < 128 MiB
// (N3-REAL, 11-25) Histeresis real: el nivel 3 se enciende con hueco < 128 MiB (o por encima del presupuesto: hueco
// negativo) y se apaga solo con hueco >= 384 MiB. Cada ventana inutil seguida dobla el bloqueo (30, 60, 120, 240, 300 s).
constexpr int64_t  N3HuecoOffMiB = 384;
constexpr uint64_t N3BloqueoMaxMs = 300000;
inline std::atomic<bool>     g_bufram_permitido {false}; // migrar a RAM permitido (lo pone Tick)
inline std::atomic<uint64_t> g_uso_bytes {0};            // uso del proceso del ultimo Tick
inline std::atomic<uint64_t> g_bufram_bytes {0}, g_bufram_n {0};          // bufferes ahora en RAM
inline std::atomic<uint64_t> g_bufram_out {0}, g_bufram_back {0};         // acumulado migrado / devuelto (bytes)
inline std::atomic<uint64_t> g_bufram_calientes {0};     // en RAM, usados, esperando sitio en VRAM (bytes)

inline bool CheckLea(uint64_t rva, uint64_t target) {
#if KYTY_VRAMJUEGO_OK
	const auto* at = reinterpret_cast<const uint8_t*>(VramJuego::kEbootBase + rva);
	if (IsBadReadPtr(at, 7) != 0) return false;
	if (at[0] != 0x48 || at[1] != 0x8d || at[2] != 0x05) return false;
	int32_t disp = 0;
	std::memcpy(&disp, at + 3, 4);
	return rva + 7 + static_cast<int64_t>(disp) == target;
#else
	(void)rva;
	(void)target;
	return false;
#endif
}

// Campo del gestor de streaming (mgr = *(0x2f6a138)), MiB; INT64_MIN si no se puede leer. 0x286038: presupuesto que
// 0xac9d80 calculo en su ultima pasada; (TOPE2) 0x286048: lo fijo (base) que resta del pool.
inline int64_t CampoMgrMiB(uint32_t off) {
#if KYTY_VRAMJUEGO_OK
	static const bool ok = CheckLea(kMgrLeaRva, kMgrPtrRva);
	if (!ok) return INT64_MIN;
	auto* const* pp = reinterpret_cast<uint8_t* const*>(VramJuego::kEbootBase + kMgrPtrRva);
	if (IsBadReadPtr(pp, 8) != 0 || *pp == nullptr) return INT64_MIN;
	const uint8_t* f = *pp + off;
	if (IsBadReadPtr(f, 8) != 0) return INT64_MIN;
	int64_t v = 0;
	std::memcpy(&v, f, 8);
	return v / static_cast<int64_t>(MiB);
#else
	(void)off;
	return INT64_MIN;
#endif
}
inline int64_t PresupuestoJuegoMiB() { return CampoMgrMiB(0x286038); }
inline int64_t FijoJuegoMiB() { return CampoMgrMiB(0x286048); }

inline void Pool(Estado& e, uint64_t now_ms) {
#if KYTY_VRAMJUEGO_OK
	if (!e.pool_checked) {
		e.pool_checked = true;
		e.pool_valid   = VramJuego::CheckEboot();
		if (!e.pool_valid) {
			std::printf("[vram-tope] eboot no es 01.007.000 (0xac9f50 no lee 0x2f69898): sin nivel 2 (pool)\n");
			std::fflush(stdout);
		}
	}
	if (!e.pool_valid) return;
	if (e.pool == nullptr) {
		auto* const* pp = reinterpret_cast<int* const*>(VramJuego::kEbootBase + VramJuego::kPoolPtrRva);
		if (IsBadReadPtr(pp, 8) != 0) return;
		int* p = *pp;
		if (p == nullptr || IsBadReadPtr(p, 4) != 0 || IsBadWritePtr(p, 4) != 0) return; // opcion aun sin crear
		const int v = *reinterpret_cast<volatile int*>(p);
		if (v < 64 || v > 65536) return;
		e.pool       = p;
		e.pool_orig  = v;
		e.pool_floor = v * PoolFloorPct / 100;
		std::printf("[vram-tope] pool de streaming del juego (*0x2f69898) %d MiB; minimo %d MiB (%d %%)\n", v, e.pool_floor,
		            PoolFloorPct);
		std::fflush(stdout);
	}
	int cur = *e.pool;
	if (e.pool_written != 0 && cur != e.pool_written) { // lo cambio el juego (config): nuevo original
		e.pool_orig    = cur;
		e.pool_floor   = cur * PoolFloorPct / 100;
		e.pool_written = 0;
	}
	const uint64_t tope = e.tope;
	int            next = cur;
	// (TOPE3, 11-25) Decision del usuario: el nivel 2 antiguo (bajar el pool) queda APAGADO. Con 1440p + FSR 4 dejo el
	// presupuesto del juego en 0 (texturas borrosas/negras) y la VRAM no bajo. El pool se deja SIEMPRE en su original:
	// si alguna vez se escribio, se restaura.
	const int64_t pres = PresupuestoJuegoMiB();
	e.pool_al_suelo    = true;
	if (e.pool_written != 0 && cur < e.pool_orig) next = e.pool_orig;
	if (next != cur) {
		*e.pool        = next;
		e.pool_written = next == e.pool_orig ? 0 : next;
		if (next < cur) e.pool_bajadas++;
		std::printf("[vram-tope] pool del juego %d -> %d MiB (original %d); uso %llu / tope %llu MiB; presupuesto del juego "
		            "%lld, fijo %lld MiB\n", cur, next, e.pool_orig, static_cast<unsigned long long>(e.uso),
		            static_cast<unsigned long long>(tope), static_cast<long long>(pres), static_cast<long long>(FijoJuegoMiB()));
		std::fflush(stdout);
	}
#else
	(void)e;
	(void)now_ms;
#endif
}

inline void Mundo(Estado& e, uint64_t now_ms, uint64_t gc_tick) {
#if KYTY_VRAMJUEGO_OK
	if (!e.map_checked) {
		e.map_checked = true;
		e.map_valid   = VramJuego::CheckRipRcx(kMapRdRva, kMapPtrRva);
		if (!e.map_valid) {
			std::printf("[vram-tope] eboot no es 01.007.000 (0x2f0feb no lee 0x27bf378): sin deteccion de cambio de mundo\n");
			std::fflush(stdout);
		}
	}
	if (e.map_valid) {
		auto* const* pp = reinterpret_cast<uint8_t* const*>(VramJuego::kEbootBase + kMapPtrRva);
		if (IsBadReadPtr(pp, 8) == 0 && *pp != nullptr && IsBadReadPtr(*pp + 0xc, 4) == 0) {
			int id = 0;
			std::memcpy(&id, *pp + 0xc, 4);
			if (id != kMapNone && id > 0) {
				const int key = static_cast<uint32_t>(id) < 0x8000000u ? (id >> 24) : id;
				if (e.map_key < 0) {
					e.map_key = key;
					e.map_id  = id;
					std::printf("[vram-tope] mundo inicial: mapa 0x%08x (m%02d)\n", id, id >> 24);
					std::fflush(stdout);
				} else if (key != e.map_key) {
					std::printf("[vram-tope] CAMBIO DE MUNDO: mapa 0x%08x (m%02d) -> 0x%08x (m%02d); en %llu s se suelta lo "
					            "que no se use desde ahora (el mundo anterior)\n",
					            e.map_id, e.map_id >> 24, id, id >> 24,
					            static_cast<unsigned long long>(MundoEsperaMs / 1000));
					std::fflush(stdout);
					e.map_key    = key;
					e.map_id     = id;
					e.mundo_ms   = now_ms;
					e.mundo_tick = gc_tick;
					e.mundos++;
					g_mundo_activo.store(false);
				}
			}
		}
	}
	if (e.mundo_ms != 0) {
		const uint64_t dt = now_ms - e.mundo_ms;
		if (dt >= MundoEsperaMs && dt < MundoEsperaMs + MundoVentanaMs) {
			if (!g_mundo_activo.load()) {
				g_mundo_tick.store(e.mundo_tick);
				g_mundo_activo.store(true);
				e.mundo_bytes0 = g_mundo_bytes.load();
			}
		} else if (dt >= MundoEsperaMs + MundoVentanaMs) {
			g_mundo_activo.store(false);
			e.mundo_ms = 0;
			std::printf("[vram-tope] cambio de mundo: soltado lo del mundo anterior %llu MiB (total %llu imagenes, %llu MiB)\n",
			            static_cast<unsigned long long>((g_mundo_bytes.load() - e.mundo_bytes0) >> 20u),
			            static_cast<unsigned long long>(g_mundo_n.load()),
			            static_cast<unsigned long long>(g_mundo_bytes.load() >> 20u));
			std::fflush(stdout);
		}
	}
#else
	(void)e;
	(void)now_ms;
	(void)gc_tick;
#endif
}

// Cada ~1 s desde el hilo de render (RunGarbageCollector). Bytes; now_ms monotono; gc_tick = marca del recolector.
// evicted_bytes / freed_mips_bytes: contadores acumulados (expulsado por cualquier recolector / mips soltados por el
// juego). drv_activo: nivel 3 (VRAM_DRV2) activo. Devuelve el tope efectivo en bytes (0 si apagado).
inline uint64_t Tick(uint64_t card_bytes, uint64_t uso_bytes, uint64_t drv_budget_bytes, uint64_t now_ms,
                     uint64_t gc_tick, uint64_t evicted_bytes, uint64_t freed_mips_bytes, bool drv_activo,
                     uint64_t reup_bytes = 0) {
	if (!On() || card_bytes == 0 || drv_budget_bytes == 0) return 0;
	auto& e        = E();
	e.card         = card_bytes / MiB;
	e.tabla        = TablaMiB(e.card);
	const uint64_t drv_cap = drv_budget_bytes / MiB > DrvMarginMiB ? drv_budget_bytes / MiB - DrvMarginMiB : 0;
	e.drv          = drv_budget_bytes / MiB;
	e.manda_driver = drv_cap < e.tabla;
	e.tope         = e.manda_driver ? drv_cap : e.tabla;
	e.uso          = uso_bytes / MiB;
	g_tope_bytes.store(e.tope * MiB, std::memory_order_relaxed);
	Pool(e, now_ms);
	Mundo(e, now_ms, gc_tick);
	const bool n1   = VramJuego::T().activo;
	g_uso_bytes.store(uso_bytes, std::memory_order_relaxed);
	// (TOPE3) Nivel 2 = bufferes frios a la RAM del sistema, desde el 95 % del tope.
	// (M7) Con el gestor central, el nivel 2 (bufferes frios a RAM) lo decide su presion de VRAM (>= 95 % de su presupuesto),
	// no este tope propio: BufferCache::BufRamPass ya lee GestorMem::Presion directamente; aqui solo para el nivel y el log.
	const bool n2   = GestorMem::On() ? GestorMem::Presion(GestorMem::Vram) >= static_cast<double>(BufRamPct) / 100.0
	                                  : BufRamOn() && (Mitad() ? e.uso * 2 > e.tope : e.uso * 100 >= e.tope * BufRamPct); // (W17) mitad
	g_bufram_permitido.store(n2, std::memory_order_relaxed);
	e.nivel         = drv_activo ? 3 : (n2 || g_bufram_bytes.load() != 0) ? 2 : n1 ? 1 : 0;
	// (TOPE3) Nivel 3 (expulsion de imagenes) solo como emergencia: hueco del driver < 128 MiB (si no, pagina).
	// (N3-REAL) Hueco con signo (negativo: el proceso pasa el presupuesto del driver y Windows pagina) e histeresis.
	e.hueco = (static_cast<int64_t>(drv_budget_bytes) - static_cast<int64_t>(uso_bytes)) / static_cast<int64_t>(MiB);
	if (!e.n3_on && e.hueco < static_cast<int64_t>(N3HuecoMiB)) e.n3_on = true;
	else if (e.n3_on && e.hueco >= N3HuecoOffMiB) e.n3_on = false;
	g_n3_permitido.store(e.n3_on && now_ms >= e.n3_bloqueo_hasta, std::memory_order_relaxed);
	if (e.last_log_ms == 0) {
		e.last_log_ms = now_ms;
		e.ev0         = evicted_bytes;
		e.freed0      = freed_mips_bytes;
		e.reup0       = reup_bytes;
		e.uso0        = e.uso;
		e.evt0 = VramDrv::g_ev_tex.load(); e.evn0 = VramDrv::g_ev_nat.load(); e.evo0 = VramDrv::g_ev_offl.load();
		e.prot0 = VramDrv::g_prot_bytes.load();
	}
	if (now_ms - e.last_log_ms >= 5000) {
		// (TOPE2) Ventana de 5 s del nivel 3: inutil si lo re-subido llega a la mitad de lo expulsado o el uso no bajo
		// la mitad de lo expulsado (fuera del cambio de mundo).
		const uint64_t ev_mib   = (evicted_bytes - e.ev0) >> 20u;
		const uint64_t reup_mib = (reup_bytes - e.reup0) >> 20u;
		const int64_t  bajo_mib = static_cast<int64_t>(e.uso0) - static_cast<int64_t>(e.uso);
		// (N3-BLOQ) Log 20261125-104537: las nativas expulsadas son pocos MiB ligados (2-43 MiB en 5 s) y lo re-subido
		// 51-1867 MiB: con el minimo de 128 MiB expulsados la ventana nunca se juzgaba y el nivel 3 nunca se bloqueaba.
		// Ahora tambien se juzga con >= 128 MiB re-subidos (y algo expulsado).
		const bool juzgar = ev_mib >= N3MinEvalMiB || (reup_mib >= N3MinEvalMiB && ev_mib > 0);
		if (!g_mundo_activo.load() && juzgar &&
		    (reup_mib * 2 >= ev_mib || bajo_mib * 2 < static_cast<int64_t>(ev_mib))) {
			// (N3-REAL) Bloqueo creciente: 30 s, 60, 120, 240, 300 (tope) mientras siga siendo inutil.
			const uint64_t sh  = std::min<uint64_t>(e.n3_seguidos, 4);
			// (VG-AB, 11-25) Log 102943: dos ventanas, 2338 y 1343 MiB expulsados (nativas), uso -309/+16 MiB: nunca
			// util; log 074401: 36 GB en vano. Una ventana inutil bloquea el nivel 3 el resto de la sesion.
			(void)sh;
			const uint64_t blq = uint64_t {1} << 40u;
			e.n3_seguidos++;
			e.n3_bloqueo_hasta = now_ms + blq;
			e.n3_bloqueos++;
			g_n3_permitido.store(false, std::memory_order_relaxed);
			g_n3_bloqueado.store(true, std::memory_order_relaxed);
			std::printf("[vram-tope] nivel 3 inutil en 5 s: expulsado %llu MiB, re-subido %llu, el uso bajo %lld MiB: lo que "
			            "llena la VRAM no son texturas sin usar; sin expulsar %llu s (bloqueos %llu)\n",
			            static_cast<unsigned long long>(ev_mib), static_cast<unsigned long long>(reup_mib),
			            static_cast<long long>(bajo_mib), static_cast<unsigned long long>(blq >= (uint64_t {1} << 40u) ? 0 : (e.n3_bloqueo_hasta - now_ms) / 1000),
			            static_cast<unsigned long long>(e.n3_bloqueos));
		} else if (ev_mib >= N3MinEvalMiB) {
			e.n3_seguidos = 0; // ventana util: el bloqueo vuelve a empezar en 30 s
		}
		{
			const uint64_t t = VramDrv::g_ev_tex.load(), n = VramDrv::g_ev_nat.load(), o = VramDrv::g_ev_offl.load(),
			               pb = VramDrv::g_prot_bytes.load();
			std::printf("[vram-tope] hueco del driver %lld MiB (presupuesto real %llu, uso %llu; nivel 3 %s, se apaga con >= %lld); "
			            "expulsado en 5 s por tipo: texturas %llu MiB, nativas %llu MiB, con bytes fuera de la RAM %llu MiB; "
			            "protegidas en la ultima pasada (RT, profundidad, escritas por la GPU: nunca se expulsan) %llu MiB; "
			            "escala de resolucion interna 100 %% (fija: el juego no tiene cambio en caliente)\n",
			            static_cast<long long>(e.hueco), static_cast<unsigned long long>(drv_budget_bytes / MiB),
			            static_cast<unsigned long long>(e.uso), e.n3_on ? "encendido" : "apagado",
			            static_cast<long long>(N3HuecoOffMiB), static_cast<unsigned long long>((t - e.evt0) >> 20u),
			            static_cast<unsigned long long>((n - e.evn0) >> 20u), static_cast<unsigned long long>((o - e.evo0) >> 20u),
			            static_cast<unsigned long long>(pb >> 20u));
			e.evt0 = t; e.evn0 = n; e.evo0 = o; e.prot0 = pb;
		}
		static const char* const nombre[4] = {"0 (calidad maxima)", "1 (mips lejanos)", "2 (pool del juego bajado)",
		                                      "3 (expulsion de imagenes sin usar)"};
		const int pool_now = e.pool != nullptr ? *e.pool : 0;
		std::printf("[vram-tope] tarjeta %llu MiB; tope tabla %llu; driver %llu (-%llu = %llu); tope efectivo %llu (manda %s); "
		            "uso del proceso %llu (%llu %%); nivel de calidad %s; pool del juego %d/%d MiB (presupuesto del juego "
		            "%lld, fijo %lld); liberado en 5 s: mips %llu MiB, expulsado %llu MiB, re-subido %llu MiB; nivel 3 %s; "
		            "reparto (MiB, recorrido de hace <30 s): texturas %llu, RT/profundidad %llu, escritas por la GPU %llu, "
		            "otras imagenes %llu, bufferes de la cache en VRAM %llu, sin usar >30 s %llu, resto (driver, FSR, bloques) %lld; "
		            "cambios de mundo %llu%s\n",
		            static_cast<unsigned long long>(e.card), static_cast<unsigned long long>(e.tabla),
		            static_cast<unsigned long long>(e.drv), static_cast<unsigned long long>(DrvMarginMiB),
		            static_cast<unsigned long long>(drv_cap), static_cast<unsigned long long>(e.tope),
		            e.manda_driver ? "el driver: otros procesos usan la tarjeta" : "la tabla",
		            static_cast<unsigned long long>(e.uso),
		            static_cast<unsigned long long>(e.tope ? e.uso * 100 / e.tope : 0), nombre[e.nivel], pool_now,
		            e.pool_orig, static_cast<long long>(PresupuestoJuegoMiB()), static_cast<long long>(FijoJuegoMiB()),
		            static_cast<unsigned long long>((freed_mips_bytes - e.freed0) >> 20u),
		            static_cast<unsigned long long>(ev_mib), static_cast<unsigned long long>(reup_mib),
		            now_ms < e.n3_bloqueo_hasta ? "bloqueado (inutil)"
		            : g_n3_permitido.load() ? "permitido" : "en espera (el nivel 2 aun puede bajar)",
		            static_cast<unsigned long long>(g_rep_tex.load() >> 20u),
		            static_cast<unsigned long long>(g_rep_rt.load() >> 20u),
		            static_cast<unsigned long long>(g_rep_gpu.load() >> 20u),
		            static_cast<unsigned long long>(g_rep_otras.load() >> 20u),
		            static_cast<unsigned long long>(g_rep_buf.load() >> 20u),
		            static_cast<unsigned long long>(g_rep_stale.load() >> 20u),
		            static_cast<long long>(e.uso) - static_cast<long long>((g_rep_tex.load() + g_rep_rt.load() +
		                                       g_rep_gpu.load() + g_rep_otras.load() + g_rep_buf.load() -
			                                       std::min(g_rep_buf.load(), g_bufram_bytes.load())) >> 20u),
		            static_cast<unsigned long long>(e.mundos), g_mundo_activo.load() ? " (soltando el mundo anterior)" : "");
		std::fflush(stdout);
		e.last_log_ms = now_ms;
		e.ev0         = evicted_bytes;
		e.freed0      = freed_mips_bytes;
		e.reup0       = reup_bytes;
		e.uso0        = e.uso;
	}
	return e.tope * MiB;
}

} // namespace VramTope
