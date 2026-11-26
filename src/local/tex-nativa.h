#pragma once
// (N1, 11-24) KYTY_TEX_NATIVA=1: gestor de texturas nativo, etapa N1. APAGADO por defecto.
//
// Ghidra (eboot 01.007.000, proyecto r19g):
//  - 0x8fe750 crea la CTexture: T# en this+0xbc (base = ((T#[1]&0xff)<<32 | T#[0]) << 8), dims en +0x264..+0x274.
//  - 0x901030 (op 0 sobre 0x7f9f70) compromete cada mip (o capa) de la textura; 0x901110 (op 2) lo libera. Cada mip es un
//    rango contiguo de VirtualMemo (layout FUN_01d1c060/FUN_01d1bee0). Esas operaciones llegan al emulador como
//    map/unmap -> TextureCache::MapMemory/UnmapMemory -> Image::SparseUpdate.
//  - El T# que ve el draw empieza en el primer mip cargado; el emulador creaba la imagen DESDE ese mip (HoldFromBaseLevel):
//    cada mip mayor que llegaba caia fuera y forzaba otra imagen (la causa de la cobertura del 9 % de la 2.1).
//
// N1: cada CTexture tiene UNA imagen Vulkan con toda su cadena de mips, dispersa (sparse residency): cada mip tiene
// memoria de video solo mientras el juego lo tiene comprometido (sparseImage.cpp, SparseUpdateLevels), como en la PS5.
// El draw la ve con una vista desde el mip base del T#.
// N2+N3: cada trozo que el juego copia (memcpy 0x901459) a un mip de una textura nativa ya creada va DIRECTO a la GPU
// (anillo de subida fijo, sin memoria temporal) y NO se escribe en la RAM emulada: esas paginas nunca se tocan (RAM real libre). Lo que no
// se puede subir asi (sin imagen, imagen no apta) se escribe en la RAM como siempre ("devueltas a la RAM").
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nativo-cubierto.h" // (11-25) rangos "solo en la GPU" fuera de la vigilancia del emulador

namespace TexNativa {

constexpr uint64_t kVmBegin = 0x1000000000ull;
constexpr uint64_t kVmEnd   = 0x1400000000ull;

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA");
		const bool  r = v != nullptr && v[0] == '1';
		if (r) {
			std::printf("[tex-nativa] KYTY_TEX_NATIVA=1: N1, una imagen Vulkan dispersa con todos sus mips por CTexture; "
			            "memoria de video por mip segun lo que el juego compromete (0x901030/0x901110)\n");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}

// (N2+N3f) Coste de las partes nuevas: microsegundos y llamadas acumulados (atomicos relajados; solo con On()).
enum CosteId : int { kCDrain = 0, kCUpload, kCRing, kCLevels, kCBind, kCCopy, kCQuery, kCosteN };
inline std::atomic<uint64_t> c_coste_us[kCosteN] {}, c_coste_n[kCosteN] {};
inline std::atomic<uint64_t> c_copy_bytes {0};
// Cronometro de ambito: suma el tiempo al salir. Apagado (sin On()) no lee el reloj.
struct Coste {
	int                                   id;
	bool                                  on;
	std::chrono::steady_clock::time_point t0;
	explicit Coste(int i): id(i), on(On()) {
		if (on) t0 = std::chrono::steady_clock::now();
	}
	~Coste() {
		if (!on) return;
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
		c_coste_us[id].fetch_add(static_cast<uint64_t>(us), std::memory_order_relaxed);
		c_coste_n[id].fetch_add(1, std::memory_order_relaxed);
	}
	Coste(const Coste&)            = delete;
	Coste& operator=(const Coste&) = delete;
};

struct Entry {
	bool    min_set   = false; // (NEGRAS) min_level fijado por el primer descriptor: ya no cambia
	uint8_t min_level = 0; // mips por debajo no se ligan (KYTY_TEXTURE_REDUCE: los que el emulador deja fuera)
};
inline std::mutex& Mutex() {
	static std::mutex m;
	return m;
}
inline std::unordered_map<uint64_t, Entry>& Table() {
	static std::unordered_map<uint64_t, Entry> t;
	return t;
}

// (N2+N3) Rangos de VirtualMemo cuyos bytes SOLO estan en la GPU (el trozo no se escribio en la RAM emulada). El
// refresco del emulador no los sube nunca desde la RAM (TextureCache::NativeGuardDirty) y su imagen no se expulsa
// (VRAM8_GC). Salen al liberar el juego el mip (unmap).
inline std::atomic<bool> g_any_owned {false}; // (N2+N3e) algun rango "solo en la GPU"
inline std::mutex& OwnedMutex() {
	static std::mutex m;
	return m;
}
inline std::map<uint64_t, uint64_t>& Owned() { // inicio -> fin, disjuntos
	static std::map<uint64_t, uint64_t> m;
	return m;
}
inline void AddOwned(uint64_t b, uint64_t e) {
	std::lock_guard<std::mutex> lock(OwnedMutex());
	auto&                       m  = Owned();
	auto                        it = m.upper_bound(b);
	if (it != m.begin() && std::prev(it)->second >= b) --it;
	while (it != m.end() && it->first <= e) {
		b  = std::min(b, it->first);
		e  = std::max(e, it->second);
		it = m.erase(it);
	}
	m.emplace(b, e);
	g_any_owned.store(true, std::memory_order_relaxed);
	// (11-25) Estas paginas solo las sirve la ruta nativa (anillo -> mip de la GPU): fuera de la vigilancia del emulador.
	NativoCubierto::MarcarCubierto(b, e - b, NativoCubierto::DTexturas);
}
inline uint64_t ForgetReleased(uint64_t b, uint64_t e, bool rebacked);
inline void RemoveOwned(uint64_t b, uint64_t e) {
	(void)ForgetReleased(b, e, false); // (N5) quien re-respalda llama antes ForgetReleased(b, e, true)
	std::lock_guard<std::mutex> lock(OwnedMutex());
	auto&                       m  = Owned();
	auto                        it = m.upper_bound(b);
	if (it != m.begin() && std::prev(it)->second > b) --it;
	std::vector<std::pair<uint64_t, uint64_t>> keep;
	while (it != m.end() && it->first < e) {
		if (it->first < b) keep.emplace_back(it->first, b);
		if (it->second > e) keep.emplace_back(e, it->second);
		it = m.erase(it);
	}
	for (const auto& k: keep) m.emplace(k.first, k.second);
	g_any_owned.store(!m.empty(), std::memory_order_relaxed);
	// (11-25) Vuelven a la vigilancia (Descubrir redondea hacia fuera: lo que sigue siendo nativo, otra vez cubierto).
	NativoCubierto::Descubrir(b, e - b, NativoCubierto::DTexturas);
	for (const auto& k: keep) NativoCubierto::MarcarCubierto(k.first, k.second - k.first, NativoCubierto::DTexturas);
}
inline void OwnedIn(uint64_t b, uint64_t e, std::vector<std::pair<uint64_t, uint64_t>>& out) {
	Coste coste(kCQuery); // (N2+N3f)
	out.clear();
	std::lock_guard<std::mutex> lock(OwnedMutex());
	auto&                       m  = Owned();
	auto                        it = m.upper_bound(b);
	if (it != m.begin() && std::prev(it)->second > b) --it;
	for (; it != m.end() && it->first < e; ++it)
		if (it->second > b) out.emplace_back(std::max(b, it->first), std::min(e, it->second));
}
// (N2+N3e) Sin reservar memoria: los recolectores la llaman por cada imagen candidata, cada fotograma.
inline bool AnyOwned(uint64_t b, uint64_t e) {
	if (!On() || !g_any_owned.load(std::memory_order_relaxed)) return false;
	Coste                       coste(kCQuery); // (N2+N3f) solo cuando hay rangos que mirar
	std::lock_guard<std::mutex> lock(OwnedMutex());
	const auto&                 m  = Owned();
	auto                        it = m.lower_bound(e); // primer rango que empieza en e o despues
	if (it == m.begin()) return false;
	--it;                                              // el ultimo que empieza antes de e
	return it->second > b;
}

// (N2+N3c) Trozos copiados por el juego (memcpy 0x901459) que van directos a la GPU. El juego copia los bytes de origen
// DIRECTAMENTE a un bufer de subida de la GPU de tamano fijo (anillo de 64 MiB, memoria de subida de Vulkan, creado por
// TextureCache en el hilo de render): ninguna memoria temporal. El hilo de render los destila a su mip desde ahi; el
// trozo del anillo se reutiliza cuando la GPU termino con el (scheduler.IsFree). Anillo lleno: a la RAM como siempre.
struct Pending {
	uint64_t dest = 0, size = 0, ring_offset = 0, seq = 0;
	uint64_t base = 0; // (11-25c) base del T# de su CTexture (tl_chunk_base): su imagen nativa sin FindImagesInRegion
};
// (11-25d) La espera en el anillo de los trozos sin imagen (t_us, kWaitNoImageUs) se quito: desde la 11-25b NativeChunkCopy
// los manda a la RAM antes de encolar, t_us era siempre 0 (log 075201: "subidos al crearse la imagen 0 MB"). Ahora el
// trozo sin imagen pide SU creacion al hilo de render y espera solo a esa (Urgente, abajo).
inline uint64_t NowUsN() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
struct Span {
	uint64_t offset = 0, size = 0, seq = 0, tick = 0;
	bool     done   = false;
};
inline std::mutex&          PMutex() {
	static std::mutex m;
	return m;
}
inline std::deque<Pending>& PQueue() {
	static std::deque<Pending> q;
	return q;
}
inline std::atomic<bool> g_pending {false};
constexpr uint64_t       kRingBytes = uint64_t {64} << 20u;
constexpr uint64_t       kMinChunk  = 64 * 1024;
constexpr uint64_t       kRamWrite  = ~uint64_t {0}; // (N2+N3d) aviso: el trozo fue a la RAM
struct Ring {
	std::atomic<uint8_t*> base {nullptr};
	uint64_t              size = 0, head = 0, used = 0, next_seq = 1;
	std::deque<Span>      spans; // en orden de reserva
};
inline Ring& TheRing() {
	static Ring r;
	return r;
}
// Hilo del juego: hueco contiguo de n bytes (alineado a 256) o false. Bajo PMutex.
inline bool RingAlloc(uint64_t n, uint64_t& offset, uint64_t& seq) {
	auto& r = TheRing();
	if (r.base.load(std::memory_order_acquire) == nullptr) return false;
	n = (n + 255) & ~uint64_t {255};
	if (n > r.size / 4) return false;
	if (r.spans.empty()) r.head = 0;
	const uint64_t tail = r.spans.empty() ? 0 : r.spans.front().offset;
	if (r.spans.empty() || r.head > tail || (r.head == tail && r.used == 0)) {
		if (r.head + n <= r.size) offset = r.head;
		else if (n <= tail) offset = 0;
		else return false;
	} else {
		if (r.head + n > tail) return false;
		offset = r.head;
	}
	seq = r.next_seq++;
	r.spans.push_back({offset, n, seq, 0, false});
	r.head = offset + n;
	r.used += n;
	return true;
}
// Hilo de render: el trozo `seq` ya grabado en el comando `tick` (0: libre ya).
inline void RingDone(uint64_t seq, uint64_t tick) {
	std::lock_guard<std::mutex> lock(PMutex());
	for (auto& sp: TheRing().spans)
		if (sp.seq == seq) {
			sp.tick = tick;
			sp.done = true;
			break;
		}
}
// Hilo de render: los primeros trozos cuyo trabajo de GPU acabo, libres.
template <typename IsFree>
inline void RingCollect(IsFree&& is_free) {
	std::lock_guard<std::mutex> lock(PMutex());
	auto&                       r = TheRing();
	while (!r.spans.empty() && r.spans.front().done && (r.spans.front().tick == 0 || is_free(r.spans.front().tick))) {
		r.used -= r.spans.front().size;
		r.spans.pop_front();
	}
}

inline std::atomic<uint64_t> c_soltado_bytes {0}, c_soltado_fallo {0};
// (11-25d) Paginas "solo en la GPU" que NativeDropRam encontro sin comprometer: con KYTY_COMMIT_REAL=1 la memoria del juego
// se compromete al tocarla, y el trozo directo no la toca nunca: esa RAM no llego a usarse (log 075201: "soltada 0 MB,
// rechazadas 0 MB" con 1053 MB directos = NativeOffloadDrop no encontro nada comprometido que soltar).
inline std::atomic<uint64_t> c_sin_commit_bytes {0};
inline std::atomic<uint64_t> c_decommit_bytes {0}; // (tex-decommit) MEM_DECOMMIT: compromiso privado devuelto (acumulado)
// (11-25d) Imagenes con bytes "solo en la GPU" borradas/aparcadas desde el grabador: recuperadas despues (textureCache.cpp).
inline std::atomic<uint64_t> c_later_n {0}, c_later_ok {0};

// (N5) KYTY_TEX_NATIVA_LIBERAR (por defecto encendido con KYTY_TEX_NATIVA=1; =0 lo apaga): las paginas enteras de un
// rango "solo en la GPU" pierden su respaldo fisico (MEM_RESET por la vista de respaldo + fuera del working set de LAS
// DOS vistas: la del juego, en la direccion invitada, y la del emulador). Registro de lo soltado: al salir un rango de
// "solo en la GPU" con escritura de sus bytes (restaurado de la VRAM, o el juego lo escribio en la RAM) cuenta como
// re-respaldado; si sale porque el juego libero el mip, solo se olvida.
// (tex2) Gancho de entrada de free del monton del juego (0x802f90, tex-anticipada.h): instalado = true. Sin el,
// NativeDropRam no suelta nada (el juego reutiliza los bloques liberados para sus nodos del arbol libre).
inline std::atomic<bool> g_free_hook_ok {false};
inline std::atomic<uint64_t> c_free_hits {0}, c_free_bytes {0};
// (tex2) KYTY_TEX_NATIVA_LIBERAR: 1 = encendido, 0 = apagado, sin poner = -1 (decide NativeDropRam: encendido solo con
// KYTY_NATIVE_MEMORY=1, la unica memoria donde MEM_RESET + VirtualUnlock baja de verdad el working set).
inline int LiberarPedido() {
	static const int r = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_LIBERAR");
		if (v == nullptr) v = std::getenv("KYTY_TEX_NATIVA_SOLTAR"); // (nombre anterior, mismo efecto)
		return (v == nullptr || v[0] == 0) ? -1 : (v[0] == '1' ? 1 : 0);
	}();
	return r;
}
inline bool Liberar(bool native_memory) {
	static const bool on = [native_memory] {
		const int  p = LiberarPedido();
		const bool r = On() && (p == 1 || (p == -1 && native_memory)); // (log 044349: en modo seccion no baja la RAM: solo con memoria nativa) // (log 042242: cuelgue al morir sin gancho de free)
		if (On()) {
			std::printf("[tex-nativa] KYTY_TEX_NATIVA_LIBERAR=%d (%s; memoria %s): paginas de texturas solo en la GPU sin "
			            "respaldo en la RAM\n",
			            r ? 1 : 0, p == -1 ? "por defecto" : "puesto", native_memory ? "nativa (KYTY_NATIVE_MEMORY=1)" : "de seccion");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}
inline std::atomic<uint64_t> c_liberada_bytes {0}, c_rerespaldada_bytes {0}, c_liberada_alias {0};
inline std::mutex& ReleasedMutex() {
	static std::mutex m;
	return m;
}
inline std::map<uint64_t, uint64_t>& Released() { // inicio -> fin, disjuntos, paginas enteras
	static std::map<uint64_t, uint64_t> m;
	return m;
}
inline void AddReleased(uint64_t b, uint64_t e) {
	std::lock_guard<std::mutex> lock(ReleasedMutex());
	auto& m  = Released();
	auto  it = m.upper_bound(b);
	if (it != m.begin() && std::prev(it)->second >= b) --it;
	while (it != m.end() && it->first <= e) {
		b  = std::min(b, it->first);
		e  = std::max(e, it->second);
		it = m.erase(it);
	}
	m.emplace(b, e);
}
// Quita [b, e) del registro; devuelve los bytes que estaban soltados (y los cuenta como re-respaldados si rebacked).
inline uint64_t ForgetReleased(uint64_t b, uint64_t e, bool rebacked) {
	std::lock_guard<std::mutex> lock(ReleasedMutex());
	auto&    m  = Released();
	if (m.empty()) return 0;
	auto     it = m.upper_bound(b);
	if (it != m.begin() && std::prev(it)->second > b) --it;
	uint64_t n = 0;
	std::vector<std::pair<uint64_t, uint64_t>> keep;
	while (it != m.end() && it->first < e) {
		n += std::min(e, it->second) - std::max(b, it->first);
		if (it->first < b) keep.emplace_back(it->first, b);
		if (it->second > e) keep.emplace_back(e, it->second);
		it = m.erase(it);
	}
	for (const auto& k: keep) m.emplace(k.first, k.second);
	if (rebacked) c_rerespaldada_bytes.fetch_add(n, std::memory_order_relaxed);
	return n;
}
inline std::atomic<uint64_t> c_direct_bytes {0}, c_fallback_bytes {0}, c_lost_bytes {0}, c_restored_bytes {0};
inline std::atomic<uint64_t> c_unbound_fallback {0}; // (NEGRAS) trozos a un mip sin ligar: a la RAM en vez de perderse
// (11-25) Trozos de 0x901330 que llegan a NativeChunkCopy: todos / CTexture no registrada / sin imagen todavia.
inline std::atomic<uint64_t> c_chunk_all_bytes {0}, c_chunk_nonative_bytes {0}, c_chunk_noimage_bytes {0};

inline std::atomic<uint64_t> c_registered {0}, c_images {0}, c_refused {0}, c_levels_bound {0}, c_levels_unbound {0},
    c_bound_bytes {0}, c_alloc_failed {0}, c_freed_bytes {0}, c_views {0};

// (P2) Por que una CTexture registrada no tiene imagen nativa (queda densa, con sus bytes en la RAM como siempre).
// Los de "fuera" no cuentan en c_refused (no se intentaron); los demas si.
enum Rechazo : int {
	kRCapas,      // fuera: array/pool (layers != 1): lo lleva KYTY_RAM_NATIVE_TEX por capas
	kRMips,       // fuera: mips del T# fuera de la tabla de mips
	kRAdjunto,    // fuera: uso de render target (color/profundidad)
	kRForma,      // fuera: no 2D color simple (volumen, MSAA, profundidad, stencil, cubo...)
	kRHilo,       // hilo que no es el de la GPU / grabacion / diferida / worker
	kRDisp,       // dispositivo sin sparseResidencyImage2D
	kRFormato,    // formato sin soporte disperso
	kRFormatoUso, // formato+uso sin imagen dispersa
	kRCrear,      // vkCreateImage disperso fallo
	kRReqs,       // requisitos dispersos incompletos / metadatos
	kRDoble,      // capa dispersa > 2x la densa (texturas pequenas)
	kRMemoria,    // sin tipo de memoria de video o semaforo
	kRN
};
inline std::atomic<uint64_t> c_rechazo[kRN] {};
inline std::atomic<uint64_t> c_un_mip {0}; // (P2) imagenes nativas de un solo mip

// (P2) KYTY_TEX_NATIVA_1MIP=0: las de un solo mip densas como antes (por defecto, nativas).
inline bool UnMip() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_1MIP");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

// (11-25d)
inline std::unordered_set<uint64_t>& BadBases() { // bases cuya imagen no se pudo crear (no esperar otra vez). Bajo Mutex().
	static std::unordered_set<uint64_t> s;
	return s;
}
// Salida de 0x8fe750 (tex-anticipada.h): la base del T# de la CTexture.
inline void Register(uint64_t base) {
	if (!On() || base < kVmBegin || base >= kVmEnd) return;
	std::lock_guard<std::mutex> lock(Mutex());
	if (Table().emplace(base, Entry {}).second) c_registered.fetch_add(1, std::memory_order_relaxed);
	BadBases().erase(base); // (11-25d) CTexture nueva en esa base: se vuelve a intentar
}

// (11-25c) Imagen nativa creada YA al crear la CTexture. Ghidra 01.007.000: CTexture::FTextureFactory (0x901240) llama
// a 0x8fe750 y en seguida pide la carga de cada trozo de su tabla (this+0x70..+0x78, 0x48 bytes, recurso +0x38:
// 0x8b13f0/0x8b1680); el gestor de streaming (0xac5da0/0xac6083/0xac61d3) llama a 0x901330 en cuanto un recurso llega.
// El trozo llega a los pocos ms; la cola de TexAnt (>= 20 ms de edad, 1 vez cada 16 ms, tope de MB/s) la creaba de
// media ~1 s tarde (log 072323: "espera media 1015 ms", 5828 MB a la RAM "sin imagen al llegar"). Las CTexture
// nativas van a esta cola propia, sin edad minima ni tope de bytes (la imagen es dispersa: crearla no liga memoria),
// y el hilo de render la vacia antes de cada draw (TexAnticipadaDrain).
inline std::atomic<bool> g_create_pending {false};
inline std::mutex& CreateMutex() {
	static std::mutex m;
	return m;
}
inline std::vector<uint64_t>& CreateQueue() { // CTexture (puntero del juego)
	static std::vector<uint64_t> q;
	return q;
}
inline std::atomic<uint64_t> c_pre_queued {0}, c_pre_made {0}, c_pre_native {0}, c_pre_bad {0}, c_pre_us {0},
    c_pre_max_us {0}, c_by_base_hit {0};
// (cov1, 11-25) Estado de cada CTexture nativa desde 0x8fe750: hora de encolado (us) y si su imagen ya se creo. Sirve para
// saber POR QUE un trozo llega sin imagen (log 090852: 8756 MB sin imagen, 11774 peticiones para 4273 CTexture):
//  - "aun en cola": la creacion no llego a tiempo (latencia del hilo de render);
//  - "ya creada": la imagen existia y el trozo no la encontro (clave/rango distinto, o la imagen ya se borro).
// Y los bytes que de verdad fueron directos a una imagen pedida por un trozo anterior (c_urg_ok_bytes: antes nunca se sumaba).
inline std::atomic<uint64_t> c_urg_ok_bytes {0};
struct Estado {
	int64_t t_us = 0;
	bool    made = false, urgent = false;
};
inline std::unordered_map<uint64_t, Estado>& States() { // bajo CreateMutex
	static std::unordered_map<uint64_t, Estado> m;
	return m;
}
inline std::atomic<uint64_t> c_ni_queued_n {0}, c_ni_queued_bytes {0}, c_ni_made_n {0}, c_ni_made_bytes {0},
    c_ni_unknown_bytes {0}, c_ni_age_us {0}, c_create_lat_us {0}, c_create_lat_n {0}, c_create_lat_max {0};
inline int64_t NowUsS() { return static_cast<int64_t>(NowUsN()); }
// Hilo de render: la imagen de `tex` ya existe.
inline void MarkMade(uint64_t tex) {
	std::lock_guard<std::mutex> lock(CreateMutex());
	auto it = States().find(tex);
	if (it == States().end() || it->second.made) return;
	it->second.made = true;
	const auto lat  = static_cast<uint64_t>(std::max<int64_t>(0, NowUsS() - it->second.t_us));
	c_create_lat_us.fetch_add(lat, std::memory_order_relaxed);
	c_create_lat_n.fetch_add(1, std::memory_order_relaxed);
	if (lat > c_create_lat_max.load(std::memory_order_relaxed)) c_create_lat_max.store(lat, std::memory_order_relaxed);
}
// Hilo del juego: trozo de `tex` sin imagen al llegar. Devuelve true si ya hay una peticion pendiente (no pedir otra).
inline bool NoteNoImage(uint64_t tex, uint64_t bytes) {
	std::lock_guard<std::mutex> lock(CreateMutex());
	auto it = States().find(tex);
	if (it == States().end()) {
		c_ni_unknown_bytes.fetch_add(bytes, std::memory_order_relaxed);
		return false;
	}
	c_ni_age_us.fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, NowUsS() - it->second.t_us)), std::memory_order_relaxed);
	if (it->second.made) {
		c_ni_made_n.fetch_add(1, std::memory_order_relaxed);
		c_ni_made_bytes.fetch_add(bytes, std::memory_order_relaxed);
		it->second.made   = false; // se vuelve a pedir (la imagen pudo borrarse)
		it->second.urgent = true;
		return false;
	}
	c_ni_queued_n.fetch_add(1, std::memory_order_relaxed);
	c_ni_queued_bytes.fetch_add(bytes, std::memory_order_relaxed);
	const bool already = it->second.urgent;
	it->second.urgent  = true;
	return already;
}
// Hilo del juego: trozo directo; si su CTexture se pidio con urgencia, cuenta.
inline void NoteDirect(uint64_t tex, uint64_t bytes) {
	if (tex == 0) return;
	std::lock_guard<std::mutex> lock(CreateMutex());
	auto it = States().find(tex);
	if (it != States().end() && it->second.urgent) c_urg_ok_bytes.fetch_add(bytes, std::memory_order_relaxed);
}
inline void EnqueueCreate(uint64_t tex) {
	std::lock_guard<std::mutex> lock(CreateMutex());
	if (CreateQueue().size() >= 65536) return;
	if (States().size() > 200000) States().clear();
	States()[tex] = Estado {NowUsS(), false, false};
	CreateQueue().push_back(tex);
	c_pre_queued.fetch_add(1, std::memory_order_relaxed);
	g_create_pending.store(true, std::memory_order_release);
}

// (11-25d) Creacion URGENTE: un trozo de 0x901330 llega y su CTexture nativa aun no tiene imagen (la cola de arriba se
// vacia cada >= 1 ms en el draw; FTextureFactory 0x901240 pide los trozos justo despues de 0x8fe750: llegan antes). El hilo
// del juego (fuera de todo candado del emulador: CoherentMove) pide la creacion de ESA CTexture y espera solo a ella; el
// hilo de render la atiende en la siguiente llamada a TexAnticipadaDrain (antes de cada draw, sin puerta de tiempo) y
// avisa. Seguridad anti-bloqueo (no es la espera normal): si el hilo de render no llega en kUrgentGuardUs (sin draws: p. ej.
// el hilo principal esperando al de streaming), el trozo va a la RAM y durante 500 ms no se espera.
struct Urgente {
	uint64_t tex = 0, base = 0;
	bool     done = false, ok = false;
};
constexpr int64_t     kUrgentGuardUs = 50000;
inline std::atomic<bool>    g_urgent {false};
inline std::atomic<int64_t> g_urgent_off_until {0};
inline std::mutex&          UMutex() {
	static std::mutex m;
	return m;
}
inline std::condition_variable& UCv() {
	static std::condition_variable cv;
	return cv;
}
inline std::vector<std::shared_ptr<Urgente>>& UQueue() {
	static std::vector<std::shared_ptr<Urgente>> q;
	return q;
}
inline std::atomic<uint64_t> c_urg_n {0}, c_urg_ok {0}, c_urg_fail {0}, c_urg_guard {0}, c_urg_us {0}, c_urg_max_us {0},
    c_urg_bad {0}, c_urg_off {0};
inline bool IsBadBase(uint64_t base) {
	std::lock_guard<std::mutex> lock(Mutex());
	return BadBases().count(base) != 0;
}
// Hilo de render: las peticiones pendientes (se responden con Responder).
inline void TakeUrgent(std::vector<std::shared_ptr<Urgente>>& out) {
	out.clear();
	if (!g_urgent.load(std::memory_order_acquire)) return;
	std::lock_guard<std::mutex> lock(UMutex());
	out.swap(UQueue());
	g_urgent.store(false, std::memory_order_release);
}
inline void Responder(const std::shared_ptr<Urgente>& r, bool ok) {
	if (!ok && r->base != 0) {
		std::lock_guard<std::mutex> lock(Mutex());
		BadBases().insert(r->base);
	}
	{
		std::lock_guard<std::mutex> lock(UMutex());
		r->done = true;
		r->ok   = ok;
	}
	if (ok) c_urg_ok.fetch_add(1, std::memory_order_relaxed); // (11-25e) imagen lista (nadie espera ya)
	UCv().notify_all();
}

inline bool IsNative(uint64_t base) {
	if (!On() || base < kVmBegin || base >= kVmEnd) return false;
	std::lock_guard<std::mutex> lock(Mutex());
	return Table().count(base) != 0;
}

// (NEGRAS) El primer descriptor fija el nivel; despues no cambia (devuelve el fijado). Antes cambiaba con cada T#: el
// `drop` de KYTY_TEXTURE_REDUCE depende de IsRegionGpuModified/HasGpuWrittenImageOverlap (cambian al reciclarse la
// memoria), y una vista que pasaba a empezar en el mip 0 leia un mip que nunca se ligo (sparse sin ligar = negro).
inline uint32_t SetMinLevel(uint64_t base, uint32_t level) {
	std::lock_guard<std::mutex> lock(Mutex());
	auto                        it = Table().find(base);
	if (it == Table().end()) return level;
	if (!it->second.min_set) {
		it->second.min_level = static_cast<uint8_t>(level);
		it->second.min_set   = true;
	}
	return it->second.min_level;
}

inline uint32_t MinLevel(uint64_t base) {
	std::lock_guard<std::mutex> lock(Mutex());
	auto                        it = Table().find(base);
	return it != Table().end() ? it->second.min_level : 0u;
}

inline void MaybeLog() {
	if (!On()) return;
	static std::atomic<int64_t> next {0};
	const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int64_t       due = next.load(std::memory_order_relaxed);
	if (now < due || !next.compare_exchange_strong(due, now + 30)) return;
	std::printf("[tex-nativa] RAM emulada sin escribir (texturas solo en la GPU) %llu MB; a la RAM como siempre %llu MB; "
	            "recuperadas de la VRAM antes de borrar %llu MB; perdidas %llu MB; a un mip sin ligar (a la RAM) %llu MB\n",
	            static_cast<unsigned long long>(c_direct_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_fallback_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_restored_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_lost_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_unbound_fallback.load() >> 20u));
	{
		const uint64_t all = c_chunk_all_bytes.load(), dir = c_direct_bytes.load();
		std::printf("[tex-nativa] cobertura directa a VRAM %.1f %% (%llu de %llu MB de trozos de 0x901459 sin pasar por la "
		            "RAM emulada); a la RAM: sin CTexture nativa %llu MB, sin imagen al llegar %llu MB, subida no posible %llu MB\n",
		            all ? 100.0 * static_cast<double>(dir) / static_cast<double>(all) : 0.0,
		            static_cast<unsigned long long>(dir >> 20u), static_cast<unsigned long long>(all >> 20u),
		            static_cast<unsigned long long>(c_chunk_nonative_bytes.load() >> 20u),
		            static_cast<unsigned long long>(c_chunk_noimage_bytes.load() >> 20u),
		            static_cast<unsigned long long>(c_fallback_bytes.load() >> 20u));
		const uint64_t un = c_urg_n.load();
		std::printf("[tex-nativa] trozos sin imagen al llegar: creacion urgente pedida %llu, imagen lista %llu (%llu MB de "
		            "trozos directos gracias a ello), no creable %llu, seguridad anti-bloqueo %llu (pausas %llu), base ya "
		            "fallida %llu; espera media %.2f ms (max %.1f ms); paginas fuera de la vigilancia (cubiertas) %lld\n",
		            static_cast<unsigned long long>(un), static_cast<unsigned long long>(c_urg_ok.load()),
		            static_cast<unsigned long long>(c_urg_ok_bytes.load() >> 20u),
		            static_cast<unsigned long long>(c_urg_fail.load()), static_cast<unsigned long long>(c_urg_guard.load()),
		            static_cast<unsigned long long>(c_urg_off.load()), static_cast<unsigned long long>(c_urg_bad.load()),
		            un ? static_cast<double>(c_urg_us.load()) / 1000.0 / static_cast<double>(un) : 0.0,
		            static_cast<double>(c_urg_max_us.load()) / 1000.0,
		            static_cast<long long>(NativoCubierto::g_pages[NativoCubierto::DTexturas].load()));
	}
	{
		const uint64_t ln = c_create_lat_n.load(), nq = c_ni_queued_n.load(), nm = c_ni_made_n.load();
		std::printf("[tex-nativa] (cov1) por que llega sin imagen: aun en cola de creacion %llu trozos %llu MB, imagen ya creada "
		            "(no encontrada o borrada) %llu trozos %llu MB, CTexture desconocida %llu MB; edad media del trozo desde "
		            "0x8fe750 %.1f ms; latencia de creacion media %.2f ms (max %.1f ms, %llu)\n",
		            static_cast<unsigned long long>(nq), static_cast<unsigned long long>(c_ni_queued_bytes.load() >> 20u),
		            static_cast<unsigned long long>(nm), static_cast<unsigned long long>(c_ni_made_bytes.load() >> 20u),
		            static_cast<unsigned long long>(c_ni_unknown_bytes.load() >> 20u),
		            (nq + nm) ? static_cast<double>(c_ni_age_us.load()) / 1000.0 / static_cast<double>(nq + nm) : 0.0,
		            ln ? static_cast<double>(c_create_lat_us.load()) / 1000.0 / static_cast<double>(ln) : 0.0,
		            static_cast<double>(c_create_lat_max.load()) / 1000.0, static_cast<unsigned long long>(ln));
	}
	std::printf("[tex-nativa] RAM reservada liberada %llu MB; re-respaldada %llu MB (alias saltados %llu MB); borradas desde "
	            "el grabador con bytes solo en la GPU %llu (recuperadas despues %llu)\n",
	            static_cast<unsigned long long>(c_liberada_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_rerespaldada_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_liberada_alias.load() >> 20u),
	            static_cast<unsigned long long>(c_later_n.load()), static_cast<unsigned long long>(c_later_ok.load()));
	std::printf("[tex-nativa] RAM real soltada (paginas de texturas solo en la GPU devueltas a Windows) %llu MB; rechazadas %llu MB; "
	            "nunca comprometidas (KYTY_COMMIT_REAL: el trozo directo no las toco, RAM que no llego a usarse) %llu MB; de ello descomprometidas (MEM_DECOMMIT, KYTY_TEX_NATIVA_DECOMMIT) %llu MB\n",
	            static_cast<unsigned long long>(c_soltado_bytes.load() >> 20u), static_cast<unsigned long long>(c_soltado_fallo.load() >> 20u),
	            static_cast<unsigned long long>(c_sin_commit_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_decommit_bytes.load() >> 20u));
	std::printf("[tex-nativa] gancho de free 0x802f90 %s: bloques solo-GPU liberados por el juego %llu (%llu MB, sacados del registro antes de que escriba)\n",
	            g_free_hook_ok.load() ? "instalado" : "NO instalado", static_cast<unsigned long long>(c_free_hits.load()),
	            static_cast<unsigned long long>(c_free_bytes.load() >> 20u));
	std::printf("[tex-nativa] CTexture registradas %llu; imagenes nativas %llu (rechazadas %llu); vistas del draw %llu; mips "
	            "ligados %llu, desligados %llu; VRAM ligada %llu MB, devuelta %llu MB; sin memoria %llu\n",
	            static_cast<unsigned long long>(c_registered.load()), static_cast<unsigned long long>(c_images.load()),
	            static_cast<unsigned long long>(c_refused.load()), static_cast<unsigned long long>(c_views.load()),
	            static_cast<unsigned long long>(c_levels_bound.load()), static_cast<unsigned long long>(c_levels_unbound.load()),
	            static_cast<unsigned long long>(c_bound_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_freed_bytes.load() >> 20u),
	            static_cast<unsigned long long>(c_alloc_failed.load()));
	std::printf("[tex-nativa] imagen al crear la CTexture: encoladas %llu, resueltas %llu (nativas dispersas %llu), no "
	            "resolubles %llu; render %.1f ms (max %.2f ms una vuelta); trozos con su imagen por la base (sin buscar) %llu\n",
	            static_cast<unsigned long long>(c_pre_queued.load()), static_cast<unsigned long long>(c_pre_made.load()),
	            static_cast<unsigned long long>(c_pre_native.load()), static_cast<unsigned long long>(c_pre_bad.load()),
	            static_cast<double>(c_pre_us.load()) / 1000.0, static_cast<double>(c_pre_max_us.load()) / 1000.0,
	            static_cast<unsigned long long>(c_by_base_hit.load()));
	// (P2) Motivos de rechazo (acumulados) e imagenes de un solo mip.
	std::printf("[tex-nativa] rechazos: capas %llu mips %llu adjunto %llu forma %llu hilo %llu disp %llu formato %llu "
	            "formato+uso %llu crear %llu reqs %llu doble %llu memoria %llu; de un mip %llu\n",
	            static_cast<unsigned long long>(c_rechazo[kRCapas].load()), static_cast<unsigned long long>(c_rechazo[kRMips].load()),
	            static_cast<unsigned long long>(c_rechazo[kRAdjunto].load()), static_cast<unsigned long long>(c_rechazo[kRForma].load()),
	            static_cast<unsigned long long>(c_rechazo[kRHilo].load()), static_cast<unsigned long long>(c_rechazo[kRDisp].load()),
	            static_cast<unsigned long long>(c_rechazo[kRFormato].load()),
	            static_cast<unsigned long long>(c_rechazo[kRFormatoUso].load()),
	            static_cast<unsigned long long>(c_rechazo[kRCrear].load()), static_cast<unsigned long long>(c_rechazo[kRReqs].load()),
	            static_cast<unsigned long long>(c_rechazo[kRDoble].load()), static_cast<unsigned long long>(c_rechazo[kRMemoria].load()),
	            static_cast<unsigned long long>(c_un_mip.load()));
	// (N2+N3f) Coste de las partes nuevas en los ultimos 30 s (diferencia con la impresion anterior).
	static uint64_t prev_us[kCosteN] {}, prev_n[kCosteN] {}, prev_bytes = 0;
	uint64_t        d_us[kCosteN], d_n[kCosteN];
	for (int i = 0; i < kCosteN; i++) {
		const uint64_t u = c_coste_us[i].load(std::memory_order_relaxed), n = c_coste_n[i].load(std::memory_order_relaxed);
		d_us[i] = u - prev_us[i], d_n[i] = n - prev_n[i];
		prev_us[i] = u, prev_n[i] = n;
	}
	const uint64_t bytes = c_copy_bytes.load(std::memory_order_relaxed), d_bytes = bytes - prev_bytes;
	prev_bytes           = bytes;
	std::printf("[tex-nativa-coste] 30 s: drenaje %.1f ms (subidas %.1f ms en %llu trozos, anillo %.1f ms), mips "
	            "ligados/desligados %.1f ms en %llu llamadas (vkQueueBindSparse %.1f ms), copia del juego %.1f ms en %llu "
	            "trozos (%llu MB), consultas de rangos %.1f ms en %llu llamadas\n",
	            d_us[kCDrain] / 1000.0, d_us[kCUpload] / 1000.0, static_cast<unsigned long long>(d_n[kCUpload]),
	            d_us[kCRing] / 1000.0, d_us[kCLevels] / 1000.0, static_cast<unsigned long long>(d_n[kCLevels]),
	            d_us[kCBind] / 1000.0, d_us[kCCopy] / 1000.0, static_cast<unsigned long long>(d_n[kCCopy]),
	            static_cast<unsigned long long>(d_bytes >> 20u), d_us[kCQuery] / 1000.0,
	            static_cast<unsigned long long>(d_n[kCQuery]));
	std::fflush(stdout);
}

} // namespace TexNativa
