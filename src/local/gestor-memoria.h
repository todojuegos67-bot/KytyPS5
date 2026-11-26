#pragma once
// (GESTOR-MEM, 11-25) KYTY_GESTOR_MEM=1 (por defecto; =0 lo apaga y todo queda como antes).
// Un unico arbitro de residencia para RAM y VRAM, como el gestor de un juego de PC: presupuestos calculados en caliente,
// registro de cada recurso y una sola politica de expulsion/degradacion por prioridad.
//
// Por que (logs 20261125-133607-868 / -132442-921, 8 GB simulados, PC de 16 GB):
//  - RAM: emulador commit 23,03 GB en un PC de 15,56 GB; Windows "disponible 0,47 GB" y 1,49 GB comprimidos: pagina.
//    [ram19]: juego commit 11,40 (en RAM 5,80; de ello 4,21 sin copia en GPU) + emulador fuera del juego commit 11,39
//    (en RAM 2,57, 8,79 fuera de RAM pero comprometido). Ghidra 0x80c200 (PlatformPageAllocator::Init): el juego pide
//    min(memoria directa disponible, 0x300000000 = 12 GiB) con sceKernelAllocateDirectMemory + MapDirectMemory de UNA
//    vez (memorySizeToPreAllocate) y reparte paginas el mismo; el resto queda "physicalMemoryToKeepUnmapped". El
//    emulador refleja esos 12 GiB enteros como commit del proceso: es una reserva FIJA que nadie arbitra.
//  - VRAM: 5521 MiB usados con presupuesto 5344 (por encima todo el rato). Reparto: imagenes 2572, pool de bufferes del
//    juego 974 (bloques 1043), driver 945, libre dentro de bloques VMA 122. Nadie sabe que recurso es cual ni cuando se
//    uso: cada modulo (textureCache VRAM_DRV2, bufferCache BUF-RAM, vram-tope niveles 1-3, vram-juego) tiene su propio
//    umbral y su propia histeresis, y el nivel 3 se bloquea el resto de la sesion tras una ventana inutil
//    (vram-tope.h, blq = 1<<40 ms). RT/profundidad/escritas por la GPU se marcan "protegidas" y no tienen respaldo, asi
//    que lo unico expulsable eran texturas que se volvian a subir (re-subidas 7,56/s).
// Aqui: cada modulo REGISTRA sus recursos (tipo, bytes, lugar, si tiene respaldo en la RAM del juego, si es solo-GPU),
// los TOCA al usarlos, y registra un EXPULSOR por tipo. El gestor mide (VK_EXT_memory_budget via VMA para VRAM;
// GlobalMemoryStatusEx/ProcessMemory para RAM), calcula la presion y, si se pasa del presupuesto, entrega a cada
// expulsor la lista de candidatos ya ordenada (con respaldo y menos usado primero; RT/profundidad de los ultimos
// InUseFrames fotogramas nunca; fijados nunca) con los bytes que faltan.
//
// Hilos: Tick() se llama una vez por fotograma desde GraphicContext::RefreshMemoryBudget (vma.cpp, hilo de render).
// Los expulsores se llaman desde ahi, SIN ningun lock del gestor tomado: deben tomar su propio lock (mejor try_lock;
// si esta ocupado devuelven 0 y se reintenta en el siguiente Tick). Registrar/Tocar/Liberar/CambiarTamano son seguros
// desde cualquier hilo; Tocar no toma locks (un store atomico).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace GestorMem {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_GESTOR_MEM");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

constexpr uint64_t MiB = uint64_t {1} << 20u;

enum Lugar : int { Vram = 0, Ram = 1, LugarN };

enum Tipo : int {
	TipoTextura = 0,  // textura muestreada (normalmente con respaldo en la RAM del juego)
	TipoNativa,       // textura nativa / sparse (mips ligados)
	TipoRenderTarget, // RT de color (escrito por la GPU)
	TipoProfundidad,  // profundidad/stencil
	TipoBuffer,       // buffer de GPU (vertices, indices, constantes)
	TipoBufferSubida, // anillos / staging
	TipoPipeline,     // pipelines y codigo de shaders en el driver
	TipoCopiaRam,     // copia en RAM del emulador de algo que ya esta en la GPU o en la RAM del juego
	TipoOtro,
	TipoN
};

inline const char* NombreTipo(int t) {
	static const char* const n[TipoN] = {"texturas", "nativas", "RT", "profundidad", "bufferes", "subida",
	                                     "pipelines", "copias RAM", "otros"};
	return t >= 0 && t < TipoN ? n[t] : "?";
}

enum Flag : uint32_t {
	FlagRespaldo = 1u << 0u, // el contenido se puede rehacer desde la RAM del juego (expulsar = barato)
	FlagSoloGpu  = 1u << 1u, // solo existe en la GPU (escrito por la GPU): expulsar = perderlo; solo degradar/mover
	FlagFijado   = 1u << 2u, // nunca expulsar (lo pone el modulo mientras lo necesita)
};

// Lo que el gestor pide al expulsor para cada candidato.
enum Accion : int {
	AccionExpulsar = 0, // soltar entero (se rehara desde el respaldo al volver a usarse)
	AccionDegradar = 1, // bajar calidad/mover (p. ej. buffer frio a RAM del sistema, soltar mips lejanos)
};

struct Candidato {
	uint64_t id;
	void*    dueno;      // lo que paso el modulo al registrar (su puntero a la imagen/buffer)
	uint64_t bytes;
	uint32_t edad;       // fotogramas sin tocarse
	int      tipo;
	int      accion;
};

// Devuelve los bytes que ha soltado de verdad. Por cada recurso soltado debe llamar Liberar(id) (o CambiarTamano).
using Expulsor = uint64_t (*)(void* ctx, const Candidato* c, size_t n, uint64_t objetivo_bytes);

constexpr uint32_t InUseFrames   = 8;    // lo usado en los ultimos 8 fotogramas nunca (fotograma actual y en vuelo)
constexpr uint32_t EdadNormal    = 300;  // ~5 s sin usarse
constexpr uint32_t EdadCritica   = 120;  // ~2 s, solo con presion critica
constexpr uint32_t EdadMaxAdapt  = 3600; // si lo expulsado vuelve (re-subidas), la edad minima sube hasta aqui
constexpr uint64_t DrvMarginMiB  = 128;
constexpr uint64_t RamReservaMiB = 1536; // lo que se deja libre al resto del sistema
constexpr uint64_t RamCriticaMiB = 512;  // disponible por debajo: presion critica (como LowMemoryResourceNotification)
constexpr int      HistOnPct = 100, HistOffPct = 95;

// Tope de VRAM del PROCESO por tamano de tarjeta (tabla del usuario 11-25): 8 GB -> 6,2; 10 -> 7,5; 12 -> 10; 16 -> 14,5.
inline uint64_t TablaVramMiB(uint64_t card_mib) {
	struct P { double card, tope; };
	static constexpr P t[] = {{8192, 6.3 * 1024}, {10240, 7.5 * 1024}, {12288, 10.0 * 1024}, {16384, 14.5 * 1024}};
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

// Tope de RAM residente del proceso: 70 % de la RAM fisica (16 GB -> 10,9 GB; 32 GB -> 21,8 GB).
inline uint64_t TablaRamMiB(uint64_t phys_mib) { return phys_mib * 70 / 100; }

// ---------------------------------------------------------------------------------------------------------------------
// Registro: tabla por trozos (sin realloc: Tocar puede leer un slot sin lock mientras otro hilo registra).
struct Slot {
	std::atomic<uint64_t> bytes {0};
	std::atomic<uint32_t> ultimo {0}; // fotograma del ultimo uso
	std::atomic<uint32_t> gen {0};    // impar = vivo
	std::atomic<uint32_t> flags {0};
	int                   tipo  = TipoOtro;
	int                   lugar = Vram;
	void*                 dueno = nullptr;
};
constexpr uint32_t kTrozo  = 4096;
constexpr uint32_t kTrozos = 1024; // 4 M recursos

struct ExpulsorReg {
	Expulsor fn  = nullptr;
	void*    ctx = nullptr;
};

struct Estado {
	std::atomic<Slot*>    trozos[kTrozos] {};
	std::mutex            m; // registro/liberacion y expulsores (nunca tomado al llamar a un expulsor)
	std::vector<uint32_t> libres;
	uint32_t              siguiente = 0;
	ExpulsorReg           exp[TipoN][LugarN] {};
	// uso registrado por lugar y tipo (bytes)
	std::atomic<int64_t> uso[LugarN][TipoN] {};
	std::atomic<int64_t> vivos[LugarN][TipoN] {};
	// medidas del ultimo Tick (bytes)
	std::atomic<uint64_t> pres[LugarN] {}, real[LugarN] {};
	std::atomic<uint32_t> presion_pm[LugarN] {}; // uso/presupuesto en por mil
	std::atomic<bool>     critica[LugarN] {};
	std::atomic<int64_t>  pedido[LugarN] {};     // PedirEspacio que no cupo (bytes), lo atiende el siguiente Tick
	std::atomic<uint64_t> liberado[LugarN] {}, resubido[LugarN] {}, expulsados_n {0};
	// politica
	bool     activo[LugarN] {};
	uint32_t edad_min = EdadNormal;
	// log
	uint64_t last_log_ms = 0, lib0[LugarN] {}, res0[LugarN] {}, n0 = 0, pasadas = 0, pasadas0 = 0;
	uint64_t card_mib = 0, tabla_vram = 0, drv_vram = 0, phys_mib = 0, avail_mib = 0, ws_mib = 0, priv_mib = 0;
};
inline Estado& S() {
	static Estado s;
	return s;
}
inline std::atomic<uint32_t> g_frame {1};

inline Slot* SlotDe(uint64_t id, uint32_t* gen_out = nullptr) {
	const uint32_t idx = static_cast<uint32_t>(id & 0xffffffffu);
	if (idx == 0) return nullptr;
	const uint32_t i = idx - 1;
	if (i / kTrozo >= kTrozos) return nullptr;
	Slot* t = S().trozos[i / kTrozo].load(std::memory_order_acquire);
	if (t == nullptr) return nullptr;
	Slot* s = &t[i % kTrozo];
	if (gen_out) *gen_out = static_cast<uint32_t>(id >> 32u);
	return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// API para los modulos

// Registra un recurso. Devuelve su id (0 si el gestor esta apagado o lleno: entonces Tocar/Liberar no hacen nada).
inline uint64_t Registrar(int tipo, int lugar, uint64_t bytes, uint32_t flags, void* dueno) {
	if (!On() || tipo < 0 || tipo >= TipoN || lugar < 0 || lugar >= LugarN) return 0;
	auto&    s = S();
	uint32_t i;
	{
		std::lock_guard lk(s.m);
		if (!s.libres.empty()) {
			i = s.libres.back();
			s.libres.pop_back();
		} else {
			if (s.siguiente >= kTrozo * kTrozos) return 0;
			i = s.siguiente++;
			if (s.trozos[i / kTrozo].load(std::memory_order_relaxed) == nullptr)
				s.trozos[i / kTrozo].store(new Slot[kTrozo], std::memory_order_release);
		}
		Slot* sl  = &s.trozos[i / kTrozo].load(std::memory_order_relaxed)[i % kTrozo];
		sl->tipo  = tipo;
		sl->lugar = lugar;
		sl->dueno = dueno;
		sl->bytes.store(bytes, std::memory_order_relaxed);
		sl->flags.store(flags, std::memory_order_relaxed);
		sl->ultimo.store(g_frame.load(std::memory_order_relaxed), std::memory_order_relaxed);
		const uint32_t g = sl->gen.load(std::memory_order_relaxed) + 1; // impar: vivo
		sl->gen.store(g, std::memory_order_release);
		s.uso[lugar][tipo].fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
		s.vivos[lugar][tipo].fetch_add(1, std::memory_order_relaxed);
		return (static_cast<uint64_t>(g) << 32u) | (i + 1);
	}
}

// Marca uso en el fotograma actual. Sin locks.
inline void Tocar(uint64_t id) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl != nullptr && sl->gen.load(std::memory_order_relaxed) == g)
		sl->ultimo.store(g_frame.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

inline void Fijar(uint64_t id, bool fijado) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl == nullptr || sl->gen.load(std::memory_order_relaxed) != g) return;
	if (fijado) sl->flags.fetch_or(FlagFijado, std::memory_order_relaxed);
	else sl->flags.fetch_and(~static_cast<uint32_t>(FlagFijado), std::memory_order_relaxed);
}

// El recurso cambia de tamano (mips ligados/soltados, buffer movido a RAM: CambiarLugar).
inline void CambiarTamano(uint64_t id, uint64_t bytes) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl == nullptr) return;
	auto& s = S();
	std::lock_guard lk(s.m);
	if (sl->gen.load(std::memory_order_relaxed) != g) return;
	const uint64_t antes = sl->bytes.exchange(bytes, std::memory_order_relaxed);
	s.uso[sl->lugar][sl->tipo].fetch_add(static_cast<int64_t>(bytes) - static_cast<int64_t>(antes), std::memory_order_relaxed);
	if (bytes < antes) s.liberado[sl->lugar].fetch_add(antes - bytes, std::memory_order_relaxed);
}

inline void CambiarLugar(uint64_t id, int lugar) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl == nullptr || lugar < 0 || lugar >= LugarN) return;
	auto& s = S();
	std::lock_guard lk(s.m);
	if (sl->gen.load(std::memory_order_relaxed) != g || sl->lugar == lugar) return;
	const int64_t b = static_cast<int64_t>(sl->bytes.load(std::memory_order_relaxed));
	s.uso[sl->lugar][sl->tipo].fetch_sub(b, std::memory_order_relaxed);
	s.vivos[sl->lugar][sl->tipo].fetch_sub(1, std::memory_order_relaxed);
	s.liberado[sl->lugar].fetch_add(static_cast<uint64_t>(b), std::memory_order_relaxed);
	sl->lugar = lugar;
	s.uso[lugar][sl->tipo].fetch_add(b, std::memory_order_relaxed);
	s.vivos[lugar][sl->tipo].fetch_add(1, std::memory_order_relaxed);
}

// El recurso se ha destruido (por el modulo o por un expulsor).
inline void Liberar(uint64_t id) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl == nullptr) return;
	auto& s = S();
	std::lock_guard lk(s.m);
	if (sl->gen.load(std::memory_order_relaxed) != g) return;
	const int64_t b = static_cast<int64_t>(sl->bytes.exchange(0, std::memory_order_relaxed));
	s.uso[sl->lugar][sl->tipo].fetch_sub(b, std::memory_order_relaxed);
	s.vivos[sl->lugar][sl->tipo].fetch_sub(1, std::memory_order_relaxed);
	s.liberado[sl->lugar].fetch_add(static_cast<uint64_t>(b), std::memory_order_relaxed);
	sl->dueno = nullptr;
	sl->gen.store(g + 1, std::memory_order_release); // par: libre
	s.libres.push_back(static_cast<uint32_t>((id & 0xffffffffu) - 1));
}

// (M7) El recurso cambia de tipo y/o flags (una textura que pasa a RT o a escribirse por la GPU: pierde su respaldo).
inline void CambiarTipo(uint64_t id, int tipo, uint32_t flags) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	if (sl == nullptr || tipo < 0 || tipo >= TipoN) return;
	auto& s = S();
	std::lock_guard lk(s.m);
	if (sl->gen.load(std::memory_order_relaxed) != g) return;
	sl->flags.store(flags, std::memory_order_relaxed);
	if (sl->tipo == tipo) return;
	const int64_t b = static_cast<int64_t>(sl->bytes.load(std::memory_order_relaxed));
	s.uso[sl->lugar][sl->tipo].fetch_sub(b, std::memory_order_relaxed);
	s.vivos[sl->lugar][sl->tipo].fetch_sub(1, std::memory_order_relaxed);
	sl->tipo = tipo;
	s.uso[sl->lugar][tipo].fetch_add(b, std::memory_order_relaxed);
	s.vivos[sl->lugar][tipo].fetch_add(1, std::memory_order_relaxed);
}

// (M7) El id sigue vivo (no liberado ni reutilizado). Sin locks.
inline bool Vivo(uint64_t id) {
	uint32_t g;
	Slot*    sl = SlotDe(id, &g);
	return sl != nullptr && (g & 1u) != 0 && sl->gen.load(std::memory_order_acquire) == g;
}

// Un recurso expulsado ha tenido que volver (re-subida): el gestor sube la edad minima si pasa a menudo.
inline void Resubido(int lugar, uint64_t bytes) {
	if (lugar >= 0 && lugar < LugarN) S().resubido[lugar].fetch_add(bytes, std::memory_order_relaxed);
}

// Un modulo va a crear algo de `bytes`. true = cabe en el presupuesto. false = no cabe: se apunta el deficit y el
// siguiente Tick libera (no bloquea ni llama a expulsores desde aqui: se puede llamar con locks tomados).
inline bool PedirEspacio(int lugar, uint64_t bytes) {
	if (!On() || lugar < 0 || lugar >= LugarN) return true;
	auto&          s  = S();
	const uint64_t p  = s.pres[lugar].load(std::memory_order_relaxed);
	const uint64_t u  = s.real[lugar].load(std::memory_order_relaxed);
	if (p == 0 || u + bytes <= p) return true;
	s.pedido[lugar].fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
	return false;
}

// Uso/presupuesto (1.0 = justo en el presupuesto). 0 si aun sin medir.
inline double Presion(int lugar) {
	if (lugar < 0 || lugar >= LugarN) return 0.0;
	return static_cast<double>(S().presion_pm[lugar].load(std::memory_order_relaxed)) / 1000.0;
}
inline uint64_t Presupuesto(int lugar) { return lugar >= 0 && lugar < LugarN ? S().pres[lugar].load() : 0; }
inline uint64_t Uso(int lugar) { return lugar >= 0 && lugar < LugarN ? S().real[lugar].load() : 0; } // (M7) medido
inline bool     Critica(int lugar) { return lugar >= 0 && lugar < LugarN && S().critica[lugar].load(); }

// Cada modulo registra un expulsor por (tipo, lugar). nullptr lo quita.
inline void RegistrarExpulsor(int tipo, int lugar, Expulsor fn, void* ctx) {
	if (tipo < 0 || tipo >= TipoN || lugar < 0 || lugar >= LugarN) return;
	auto&           s = S();
	std::lock_guard lk(s.m);
	s.exp[tipo][lugar] = ExpulsorReg {fn, ctx};
}


// ---------------------------------------------------------------------------------------------------------------------
// Adaptador simple para modulos que liberan por su cuenta (p. ej. buf-mem.h de bufferCache: BufMem::Presion(bytes)).
// Detectable con __has_include("gestor-memoria.h") y KYTY_GESTOR_MEM_API >= 1.
//   GestorMem::RegistrarPresion(GestorMem::TipoBuffer, GestorMem::Vram, &BufMem::Presion);
// Con presion, el gestor llama fn(bytes_que_faltan) despues de los candidatos registrados de tipos con respaldo; el
// modulo libera lo que pueda (su propio orden) y lo cuenta con Liberar/CambiarTamano/CambiarLugar (o Soltado(bytes)).
#define KYTY_GESTOR_MEM_API 1
using PresionFn = void (*)(uint64_t bytes);
struct PresionReg {
	PresionFn fn    = nullptr;
	int       lugar = Vram;
};
inline PresionReg g_presion[TipoN] {};
inline void       RegistrarPresion(int tipo, int lugar, PresionFn fn) {
	if (tipo >= 0 && tipo < TipoN && lugar >= 0 && lugar < LugarN) g_presion[tipo] = PresionReg {fn, lugar};
}
// Bytes soltados por un modulo que no registra cada recurso (solo para el log y la cuenta de lo liberado).
inline void Soltado(int lugar, uint64_t bytes) {
	if (lugar >= 0 && lugar < LugarN) S().liberado[lugar].fetch_add(bytes, std::memory_order_relaxed);
}
// (m2/m3) Callbacks con nombre: uint64_t fn(uint64_t bytes_pedidos) -> bytes liberados de verdad (NO llamar Soltado).
using LiberarFn = uint64_t (*)(uint64_t bytes_pedidos);
struct LibReg {
	const char*           nombre = nullptr;
	LiberarFn             fn     = nullptr;
	int                   lugar  = Vram;
	std::atomic<uint64_t> total {0};
};
constexpr int          kMaxLib = 16;
inline LibReg          g_lib[kMaxLib];
inline std::atomic<int> g_nlib {0};
inline bool RegistrarLiberador(const char* nombre, LiberarFn fn, int lugar = Vram) {
	static std::mutex m;
	std::lock_guard lk(m);
	const int n = g_nlib.load();
	for (int i = 0; i < n; i++)
		if (g_lib[i].fn == fn) return true;
	if (n >= kMaxLib || fn == nullptr) return false;
	g_lib[n].nombre = nombre;
	g_lib[n].fn     = fn;
	g_lib[n].lugar  = lugar;
	g_nlib.store(n + 1, std::memory_order_release);
	return true;
}

inline uint64_t PedirPresion(int lugar, uint64_t falta) {
	auto&          s     = S();
	const uint64_t antes = s.liberado[lugar].load();
	for (int t = 0; t < TipoN; t++) {
		const PresionReg r = g_presion[t];
		if (r.fn == nullptr || r.lugar != lugar) continue;
		const uint64_t hecho = s.liberado[lugar].load() - antes;
		if (hecho >= falta) break;
		r.fn(falta - hecho);
	}
	// Callbacks con nombre (GestorMemoria::Registrar): devuelven lo liberado.
	uint64_t hecho = s.liberado[lugar].load() - antes;
	const int nl = g_nlib.load(std::memory_order_acquire);
	for (int i = 0; i < nl && hecho < falta; i++) {
		if (g_lib[i].lugar != lugar || g_lib[i].fn == nullptr) continue;
		const uint64_t b = g_lib[i].fn(falta - hecho);
		g_lib[i].total.fetch_add(b, std::memory_order_relaxed);
		s.liberado[lugar].fetch_add(b, std::memory_order_relaxed);
		hecho += b;
	}
	return hecho;
}

// ---------------------------------------------------------------------------------------------------------------------
// Politica (solo desde Tick)

inline uint64_t Expulsar(int lugar, uint64_t falta, bool critica) {
	auto&          s     = S();
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	const uint32_t edad_min =
	    std::max<uint32_t>(InUseFrames, critica ? std::min<uint32_t>(s.edad_min, EdadCritica) : s.edad_min);
	ExpulsorReg exp[TipoN];
	{
		std::lock_guard lk(s.m);
		for (int t = 0; t < TipoN; t++) exp[t] = s.exp[t][lugar];
	}
	struct C {
		Candidato c;
		int       clase;
	};
	static thread_local std::vector<C> cand;
	cand.clear();
	const uint32_t total = [&] {
		std::lock_guard lk(s.m);
		return s.siguiente;
	}();
	for (uint32_t i = 0; i < total; i++) {
		Slot* t = s.trozos[i / kTrozo].load(std::memory_order_acquire);
		if (t == nullptr) { i += kTrozo - 1; continue; }
		Slot&          sl = t[i % kTrozo];
		const uint32_t g  = sl.gen.load(std::memory_order_acquire);
		if ((g & 1u) == 0 || sl.lugar != lugar || exp[sl.tipo].fn == nullptr) continue;
		const uint32_t f = sl.flags.load(std::memory_order_relaxed);
		if (f & FlagFijado) continue;
		const uint32_t edad = frame - sl.ultimo.load(std::memory_order_relaxed);
		if (edad < edad_min) continue; // nunca nada del fotograma actual / en vuelo
		// RT y profundidad: solo-GPU por naturaleza; nunca si se usaron en los ultimos InUseFrames (ya filtrado), y
		// ademas solo con presion critica.
		if ((sl.tipo == TipoRenderTarget || sl.tipo == TipoProfundidad) && !critica) continue;
		const bool respaldo = (f & FlagRespaldo) != 0 && (f & FlagSoloGpu) == 0;
		C c;
		c.c     = Candidato {(static_cast<uint64_t>(g) << 32u) | (i + 1), sl.dueno, sl.bytes.load(std::memory_order_relaxed),
		                     edad, sl.tipo, respaldo ? AccionExpulsar : AccionDegradar};
		c.clase = respaldo ? 0 : (f & FlagSoloGpu) ? 2 : 1;
		if (c.c.bytes != 0) cand.push_back(c);
	}
	if (cand.empty()) return 0;
	std::sort(cand.begin(), cand.end(), [](const C& a, const C& b) {
		if (a.clase != b.clase) return a.clase < b.clase;
		if (a.c.edad != b.c.edad) return a.c.edad > b.c.edad;
		return a.c.bytes > b.c.bytes;
	});
	// Se corta la lista a lo que hace falta (+25 % por lo que el expulsor no pueda soltar) y se reparte por tipo en el
	// mismo orden: cada expulsor recibe sus candidatos de una vez.
	uint64_t sum = 0;
	size_t   n   = 0;
	while (n < cand.size() && sum < falta + falta / 4) sum += cand[n++].c.bytes;
	static thread_local std::vector<Candidato> por_tipo[TipoN];
	for (auto& v : por_tipo) v.clear();
	int orden[TipoN], no = 0;
	for (size_t k = 0; k < n; k++) {
		const int t = cand[k].c.tipo;
		if (por_tipo[t].empty()) orden[no++] = t;
		por_tipo[t].push_back(cand[k].c);
	}
	uint64_t liberado = 0;
	for (int k = 0; k < no && liberado < falta; k++) {
		const int t = orden[k];
		liberado += exp[t].fn(exp[t].ctx, por_tipo[t].data(), por_tipo[t].size(), falta - liberado);
	}
	s.expulsados_n.fetch_add(n, std::memory_order_relaxed);
	return liberado;
}

// Medidas que le pasa vma.cpp cada fotograma (bytes; 0 = desconocido).
struct Medidas {
	uint64_t card_bytes  = 0; // tarjeta (o simulada)
	uint64_t vram_uso    = 0; // uso real del proceso (VK_EXT_memory_budget, sin lastre)
	uint64_t vram_driver = 0; // presupuesto del driver para el proceso
	uint64_t ram_fisica  = 0; // GlobalMemoryStatusEx.ullTotalPhys
	uint64_t ram_libre   = 0; // ullAvailPhys
	uint64_t ram_ws      = 0; // working set del proceso
	uint64_t ram_privada = 0; // commit privado del proceso
	bool     ram_baja    = false; // QueryMemoryResourceNotification(LowMemory)
	uint64_t now_ms      = 0;
};

inline void Log(Estado& s, const Medidas& m);

// Una vez por fotograma (vma.cpp RefreshMemoryBudget).
inline void Tick(const Medidas& m) {
	if (!On()) return;
	auto& s = S();
	g_frame.fetch_add(1, std::memory_order_relaxed);
	// VRAM: presupuesto = min(tabla por tarjeta, driver - 128 MiB).
	if (m.card_bytes != 0 && m.vram_driver != 0) {
		s.card_mib       = m.card_bytes / MiB;
		s.tabla_vram     = TablaVramMiB(s.card_mib);
		s.drv_vram       = m.vram_driver / MiB;
		const uint64_t d = s.drv_vram > DrvMarginMiB ? s.drv_vram - DrvMarginMiB : 0;
		s.pres[Vram].store(std::min(s.tabla_vram, d) * MiB, std::memory_order_relaxed);
		s.real[Vram].store(m.vram_uso, std::memory_order_relaxed);
		s.critica[Vram].store(m.vram_uso + DrvMarginMiB * MiB > m.vram_driver, std::memory_order_relaxed);
	}
	// RAM: presupuesto = min(70 % de la fisica, working set + disponible - reserva). Uso = working set.
	if (m.ram_fisica != 0) {
		s.phys_mib = m.ram_fisica / MiB;
		s.avail_mib = m.ram_libre / MiB;
		s.ws_mib    = m.ram_ws / MiB;
		s.priv_mib  = m.ram_privada / MiB;
		const uint64_t dinamico = s.ws_mib + s.avail_mib > RamReservaMiB ? s.ws_mib + s.avail_mib - RamReservaMiB : 0;
		s.pres[Ram].store(std::min(TablaRamMiB(s.phys_mib), dinamico) * MiB, std::memory_order_relaxed);
		s.real[Ram].store(m.ram_ws, std::memory_order_relaxed);
		s.critica[Ram].store(m.ram_baja || s.avail_mib < RamCriticaMiB, std::memory_order_relaxed);
	}
	for (int l = 0; l < LugarN; l++) {
		const uint64_t p = s.pres[l].load(), u = s.real[l].load();
		s.presion_pm[l].store(p ? static_cast<uint32_t>(std::min<uint64_t>(u * 1000 / p, 100000)) : 0);
		if (p == 0) continue;
		const int64_t pedido = s.pedido[l].exchange(0, std::memory_order_relaxed);
		const uint64_t obj   = u + static_cast<uint64_t>(std::max<int64_t>(pedido, 0));
		// Histeresis: empieza al pasar el 100 % (o con un pedido que no cabe), sigue hasta el 95 %.
		if (!s.activo[l] && (obj > p * HistOnPct / 100 || s.critica[l].load())) s.activo[l] = true;
		if (s.activo[l] && obj <= p * HistOffPct / 100 && !s.critica[l].load()) s.activo[l] = false;
		if (!s.activo[l]) continue;
		const uint64_t meta  = p * HistOffPct / 100;
		const uint64_t falta = obj > meta ? obj - meta : 0;
		if (falta == 0) continue;
		// Como mucho una pasada cada 15 fotogramas (el registro puede ser grande) salvo critica.
		if ((g_frame.load() % (s.critica[l].load() ? 4u : 15u)) != 0) continue; // critica: cada 4
		s.pasadas++;
		const uint64_t hecho = Expulsar(l, falta, s.critica[l].load());
		if (hecho < falta) PedirPresion(l, falta - hecho);
	}
	if (s.last_log_ms == 0) s.last_log_ms = m.now_ms;
	if (m.now_ms - s.last_log_ms >= 5000) {
		// Adaptacion: si en 5 s volvio la mitad o mas de lo liberado (en VRAM), lo expulsado se usaba: se espera el doble.
		const uint64_t lib = s.liberado[Vram].load() - s.lib0[Vram], res = s.resubido[Vram].load() - s.res0[Vram];
		if (lib >= 64 * MiB && res * 2 >= lib) s.edad_min = std::min<uint32_t>(s.edad_min * 2, EdadMaxAdapt);
		else if (res * 8 < lib && s.edad_min > EdadNormal) s.edad_min = std::max<uint32_t>(s.edad_min / 2, EdadNormal);
		Log(s, m);
		s.last_log_ms = m.now_ms;
		for (int l = 0; l < LugarN; l++) { s.lib0[l] = s.liberado[l].load(); s.res0[l] = s.resubido[l].load(); }
		s.n0       = s.expulsados_n.load();
		s.pasadas0 = s.pasadas;
	}
}

// Uso de la VRAM por categoria de vkAllocateMemory (vram-drv.h), para el log. Lo rellena vma.cpp.
struct CatsVram {
	uint64_t vma = 0, nativas = 0, sparse_buf = 0, store = 0, otro = 0;
};
inline CatsVram g_cats {};

inline void Log(Estado& s, const Medidas& m) {
	char tv[512], tr[256];
	int  o = 0;
	for (int t = 0; t < TipoN && o < 480; t++) {
		const int64_t b = s.uso[Vram][t].load();
		if (b != 0 || s.vivos[Vram][t].load() != 0)
			o += std::snprintf(tv + o, sizeof(tv) - o, " %s %lld (%lld)", NombreTipo(t), static_cast<long long>(b >> 20u),
			                   static_cast<long long>(s.vivos[Vram][t].load()));
	}
	if (o == 0) std::snprintf(tv, sizeof(tv), " (ningun modulo registrado aun)");
	o = 0;
	tr[0] = 0;
	for (int t = 0; t < TipoN && o < 230; t++) {
		const int64_t b = s.uso[Ram][t].load();
		if (b != 0) o += std::snprintf(tr + o, sizeof(tr) - o, " %s %lld", NombreTipo(t), static_cast<long long>(b >> 20u));
	}
	int nexp = 0;
	{
		std::lock_guard lk(s.m);
		for (auto& a : s.exp)
			for (auto& b : a) nexp += b.fn != nullptr;
		for (auto& p : g_presion) nexp += p.fn != nullptr;
		nexp += g_nlib.load();
	}
	const auto pr = [&](int l) { return static_cast<unsigned>(s.presion_pm[l].load() / 10); };
	// (M7) Comprobacion: lo registrado en VRAM contra lo medido por vkAllocateMemory (las mismas cifras que [vram-drv]:
	// VMA + nativas + bufferes sparse + vramStore + otro). La diferencia es lo que ningun modulo registra (anillos,
	// pipelines internos de VMA, FSR/FG, hueco de los bloques de VMA).
	int64_t reg_vram = 0;
	for (int t = 0; t < TipoN; t++) reg_vram += s.uso[Vram][t].load();
	const int64_t medido = static_cast<int64_t>(g_cats.vma + g_cats.nativas + g_cats.sparse_buf + g_cats.store + g_cats.otro);
	std::printf("[gestor-mem] cuadre VRAM (MiB): registrado %lld, medido por categorias %lld (VMA %llu + nativas %llu + "
	            "bufferes sparse %llu + vramStore %llu + otro %llu; igual que [vram-drv]), sin registrar %lld (%lld %%); uso del "
	            "proceso %llu\n",
	            static_cast<long long>(reg_vram >> 20), static_cast<long long>(medido >> 20),
	            static_cast<unsigned long long>(g_cats.vma >> 20u), static_cast<unsigned long long>(g_cats.nativas >> 20u),
	            static_cast<unsigned long long>(g_cats.sparse_buf >> 20u), static_cast<unsigned long long>(g_cats.store >> 20u),
	            static_cast<unsigned long long>(g_cats.otro >> 20u), static_cast<long long>((medido - reg_vram) >> 20),
	            static_cast<long long>(medido > 0 ? (medido - reg_vram) * 100 / medido : 0),
	            static_cast<unsigned long long>(m.vram_uso >> 20u));
	std::printf("[gestor-mem] VRAM: presupuesto %llu MiB (tabla %llu para tarjeta %llu, driver %llu-%llu), uso %llu (%u %%)%s, "
	            "por categoria: VMA %llu, nativas %llu, bufferes sparse %llu, vramStore %llu, otro %llu; registrado (MiB, "
	            "recursos):%s | RAM: presupuesto %llu MiB (70 %% de %llu, o working set + disponible %llu - %llu), working "
	            "set %llu (%u %%), privado %llu%s; registrado (MiB):%s | liberado en 5 s: VRAM %llu MiB, RAM %llu MiB "
	            "(%llu recursos, %llu pasadas); re-subido VRAM %llu MiB; edad minima %u fotogramas; expulsores %d; %s\n",
	            static_cast<unsigned long long>(s.pres[Vram].load() >> 20u), static_cast<unsigned long long>(s.tabla_vram),
	            static_cast<unsigned long long>(s.card_mib), static_cast<unsigned long long>(s.drv_vram),
	            static_cast<unsigned long long>(DrvMarginMiB), static_cast<unsigned long long>(m.vram_uso >> 20u), pr(Vram),
	            s.critica[Vram].load() ? " CRITICA (driver sin hueco)" : "", static_cast<unsigned long long>(g_cats.vma >> 20u),
	            static_cast<unsigned long long>(g_cats.nativas >> 20u), static_cast<unsigned long long>(g_cats.sparse_buf >> 20u),
	            static_cast<unsigned long long>(g_cats.store >> 20u), static_cast<unsigned long long>(g_cats.otro >> 20u), tv,
	            static_cast<unsigned long long>(s.pres[Ram].load() >> 20u), static_cast<unsigned long long>(s.phys_mib),
	            static_cast<unsigned long long>(s.avail_mib), static_cast<unsigned long long>(RamReservaMiB),
	            static_cast<unsigned long long>(s.ws_mib), pr(Ram), static_cast<unsigned long long>(s.priv_mib),
	            s.critica[Ram].load() ? " CRITICA (Windows sin RAM libre: pagina)" : "", tr[0] ? tr : " (nada)",
	            static_cast<unsigned long long>((s.liberado[Vram].load() - s.lib0[Vram]) >> 20u),
	            static_cast<unsigned long long>((s.liberado[Ram].load() - s.lib0[Ram]) >> 20u),
	            static_cast<unsigned long long>(s.expulsados_n.load() - s.n0),
	            static_cast<unsigned long long>(s.pasadas - s.pasadas0),
	            static_cast<unsigned long long>((s.resubido[Vram].load() - s.res0[Vram]) >> 20u), s.edad_min, nexp,
	            s.activo[Vram] || s.activo[Ram] ? "liberando" : "en reposo");
	std::fflush(stdout);
}

} // namespace GestorMem

// (m2/m3) Nombre de API que esperan tex-stream (texturas) y buf-mem.h (bufferes): __has_include("gestor-memoria.h") +
// KYTY_GESTOR_MEMORIA_API. GestorMemoria::Registrar("texturas", &Fn) con uint64_t Fn(uint64_t bytes_pedidos).
#define KYTY_GESTOR_MEMORIA_API 1
namespace GestorMemoria {
inline bool On() { return GestorMem::On(); }
inline bool Registrar(const char* nombre, uint64_t (*fn)(uint64_t), int lugar = GestorMem::Vram) {
	return GestorMem::RegistrarLiberador(nombre, fn, lugar);
}
inline double   Presion(int lugar = GestorMem::Vram) { return GestorMem::Presion(lugar); }
inline bool     PedirEspacio(uint64_t bytes, int lugar = GestorMem::Vram) { return GestorMem::PedirEspacio(lugar, bytes); }
inline uint64_t Presupuesto(int lugar = GestorMem::Vram) { return GestorMem::Presupuesto(lugar); }
} // namespace GestorMemoria
