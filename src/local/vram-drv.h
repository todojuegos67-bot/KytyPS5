#pragma once
// (VRAM-DRV, 11-25) KYTY_VRAM_DRV=1 (por defecto; =0 lo apaga).
// Log 20261125-055740: proceso 5936 MiB de VRAM con presupuesto 5394 ([vram]) / 5650 (driver); "fuera de las caches"
// 1359-1602 MiB sin desglose (vma.cpp: uso del proceso - VMA device-local - plain). Aqui:
//  1) MEDIR: todo vkAllocateMemory pasa por el gancho de vma.cpp (TaggedAllocateMemory, ahora siempre puesto); cada
//     punto de llamada marca su categoria (Scope) y se suman bytes vivos por categoria y por tipo (device-local o no).
//     Lo que queda del uso del proceso sin ninguna vkAllocateMemory propia es memoria interna del driver (pipelines,
//     descriptores, swapchain, colas). Linea "[vram-drv]" cada 5 s.
//  2) TECHO PROPIO: el recolector de texturas (textureCache.cpp RunGarbageCollector) actua ya al 92 % del presupuesto
//     contando el uso REAL del proceso (no solo VMA), y por encima puede expulsar imagenes sin usar desde 300 fotogramas
//     (~5 s) o 120 (~2 s) si falta >= 512 MiB, en vez de solo las de 600-1800. Nunca una usada en los ultimos
//     InUseTicks fotogramas (la del fotograma actual nunca: eso es el fallo grafico).
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace VramDrv {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_DRV");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

enum Cat : int {
	CatOtro = 0,     // sin marcar: cualquier vkAllocateMemory fuera de VMA no conocido
	CatVma,          // bloques y dedicadas de VMA (texturas, RT, bufferes del juego, FSR4/FG, anillos)
	CatSparseImg,    // texturas nativas (sparseImage.cpp: mips ligados y su pool)
	CatSparseBuf,    // bufferes sparse (streamBuffer.cpp)
	CatVramStore,    // vramStore.cpp
	CatHostImport,   // bufferCache.cpp: memoria del juego importada (puntero de host; no es VRAM)
	CatLastre,       // KYTY_SIMULATE_VRAM_MB
	CatCount
};

inline const char* Name(int c) {
	static const char* const n[CatCount] = {"otro", "VMA", "texturas nativas", "bufferes sparse", "vramStore",
	                                        "importada del juego", "lastre"};
	return c >= 0 && c < CatCount ? n[c] : "?";
}

inline thread_local int t_cat = CatOtro;

struct Scope {
	int prev;
	explicit Scope(int c): prev(t_cat) { t_cat = c; }
	~Scope() { t_cat = prev; }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};

// [cat][0 = device-local, 1 = otro tipo]: bytes vivos, objetos vivos, picos.
inline std::atomic<int64_t> g_bytes[CatCount][2];
inline std::atomic<int64_t> g_count[CatCount][2];
inline std::atomic<int64_t> g_peak[CatCount][2];

inline void Add(int cat, bool local, int64_t bytes, int64_t count) {
	if (cat < 0 || cat >= CatCount) cat = CatOtro;
	const int k   = local ? 0 : 1;
	const auto nb = g_bytes[cat][k].fetch_add(bytes, std::memory_order_relaxed) + bytes;
	g_count[cat][k].fetch_add(count, std::memory_order_relaxed);
	auto pk = g_peak[cat][k].load(std::memory_order_relaxed);
	while (nb > pk && !g_peak[cat][k].compare_exchange_weak(pk, nb, std::memory_order_relaxed)) {
	}
}

// Recolector: edad minima (fotogramas sin usarse) para expulsar con el techo propio; nunca por debajo de InUseTicks.
constexpr uint64_t InUseTicks = 8;   // ni lo del fotograma actual ni lo de las tablas (TouchTicks 8)
constexpr uint64_t TechoPct   = 92;  // % del presupuesto ([vram]) sobre el uso real del proceso: empieza a soltar
// (VRAM-DRV2, 11-25) Log 20261125-072323: el techo del 92 % (4940 MiB) quedaba por debajo de lo que el juego usa de
// verdad con VRAM_TECHO activo (5050-5320 MiB, 94-99 %): el recolector corria cada fotograma (7063 pasadas en ~2 min),
// expulso 5778 imagenes (20025 MiB, ~3.5 MiB c/u: texturas) y [vram-reparto] midio 19.8-23.9 re-subidas/s
// (73.8-85.7 MB/s), un 40-60 % de lo expulsado. Ahora:
//  - histeresis: empieza por encima del TechoPct, sigue hasta TechoBajoPct y no vuelve hasta pasar otra vez TechoPct;
//  - lo que se ha vuelto a subir tras expulsarse en los ultimos ReupProtegeMs no se expulsa (salvo lo critico);
//  - como mucho MaxMiBps expulsados por segundo (los mips lejanos de VRAM_TECHO van primero: ~30 MiB/s medidos);
//  - entre TechoBajoPct y TechoPct solo lo que lleva >= 1800 fotogramas (~30 s) sin usarse.
// Lo critico (hueco bajo el presupuesto del driver < 128 MiB) sigue como antes, sin estos limites.
constexpr uint64_t TechoBajoPct   = 85;
constexpr uint64_t ReupProtegeMs  = 60000;
constexpr uint64_t MaxMiBps       = 32;
inline std::atomic<uint64_t> g_gc_runs {0}, g_gc_need_max {0};
inline std::atomic<uint64_t> g_reup_n {0}, g_reup_bytes {0};  // re-subidas tras expulsion (cualquier recolector)
inline std::atomic<uint64_t> g_skip_reup {0}, g_cut_rate {0}; // saltadas por re-subida reciente / pasadas cortadas por MiB/s
inline std::atomic<uint64_t> g_activaciones {0};
// (VRAM-REPOSO, 11-25) Tras una pasada del techo propio sin nada que expulsar, tantos fotogramas sin volver a pasar.
constexpr uint64_t           ReposoTicks = 60;
inline std::atomic<uint64_t> g_reposo_vacias {0}, g_reposo_saltadas {0};
inline std::atomic<bool>     g_activo {false};
// (N3-REAL, 11-25) Presupuesto REAL del driver para el proceso (bytes, en las unidades de GetDeviceMemoryUsage: sin el
// lastre de KYTY_SIMULATE_VRAM_MB; con la tarjeta simulada / pequena, su tope). Lo escribe GetVideoMemoryRoom (vma.cpp).
// Log 20261125-092605: vram-tope recibia uso + hueco, y el hueco se recorta a 0 al pasar el presupuesto: "driver" = uso
// (5863 = 5863), uso siempre al 102 % del tope y el nivel 3 activo sin parar. 0 = aun sin medir.
inline std::atomic<uint64_t> g_drv_budget_bytes {0};
// (N3-REAL) Lo expulsado por el recolector por tipo (bytes acumulados) y lo pintado por la GPU que se dejo por protegido.
inline std::atomic<uint64_t> g_ev_tex {0}, g_ev_nat {0}, g_ev_offl {0}, g_prot_bytes {0}, g_prot_n {0};

} // namespace VramDrv
