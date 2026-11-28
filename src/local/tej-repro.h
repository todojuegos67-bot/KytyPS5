#pragma once
// (tej-repro) KYTY_TEJ=3, FASE 2 PASO A: captura de los draws del trabajador y REPRODUCCION EN SOMBRA en el render.
// Riesgo de imagen CERO: nada de esto llega a un command buffer real. La reproduccion usa un dispatcher de DESCARTE
// (LocalVulkanRecording::TejStubDispatch: todas las vkCmd* son funciones que solo hashean) y nunca el driver.
//  - Rec: una llamada de draw o de bind (pipeline, sets/push descriptors, push constants, VB/IB, estado dinamico vkCmdSet*,
//    vkCmdDraw*/ClearAttachments) con sus argumentos ya resueltos (POD + handles), en un buffer plano de palabras de 64 bits
//    [id, n, cuerpo...] (t_cap, solo el trabajador) y/o aplicada al seguidor de estado (t_acc).
//  - Acc: estado EFECTIVO de GRAFICOS por ranura y, en cada draw, el hash del estado vigente de cada categoria: binds
//    redundantes o elididos no cuentan, solo lo que cada draw ve. Dos variantes: exacta y "sin handles" (el trabajador usa
//    sus propios sets/anillos: sin handles ni offsets del anillo de subida en los descriptores de buffer).
// (paso A2) Ranura = el estado que Vulkan guarda, no la llamada: el estado dinamico por comando (+cara/indice), nunca por
//    valor (antes vkCmdSetDepthTestEnable(1) y (0) eran dos ranuras: el estado acumulaba historia y se llenaban 96
//    ranuras); VB por binding (Replay los liga 0..n de una vez, el camino directo solo los que cambian); IB solo cuenta en
//    draws indexados; push constants por offset (el layout/stages no cambian lo que el shader lee); el trabajo de COMPUTE
//    (pipeline/sets/push del punto compute: copias, detile, lod del propio emulador en el render) no es estado de draw.
// Solo el generador (vulkan-recording) y vulkan-recording.cpp escriben aqui; tej-coser.h compara y cuenta.
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <nmmintrin.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "local/tej-coser-sonda.h"

namespace TejRepro {

enum Cat : uint32_t { CPipe = 0, CSet = 1, CPush = 2, CVbib = 3, CDyn = 4, CDraw = 5, CN = 6 };
constexpr uint32_t kState = CDraw; // categorias de estado: 0..4
// Como forma la ranura cada comando (lo elige el generador por nombre).
enum Mode : uint32_t {
	MNone = 0, // solo el comando (estado dinamico global)
	MK0,       // comando + primer argumento (cara de stencil, firstViewport/firstAttachment...)
	MPipe,     // punto de bind (compute: fuera)
	MSets,     // punto de bind + firstSet (compute: fuera)
	MPush,     // offset (stages solo compute: fuera)
	MVb,       // por binding
	MIb,       // un solo IB, cuenta solo en draws indexados
	MDraw,     // draw no indexado
	MDrawI,    // draw indexado
};

using TejCoser::Id;
using TejCoser::Mix;
constexpr uint64_t kBasis = 0xcbf29ce484222325ull;
constexpr uint32_t kPushQ = 16; // palabras de 64 bits de las push constants que se siguen por posicion (128 bytes)

// (diag) llamadas no capturables por nombre (trabajador) y paquetes propios del emulador vistos en el render.
struct NameCount {
	std::atomic<const char*> name {nullptr};
	std::atomic<uint64_t>    n {0};
};
inline NameCount             g_bad[24];
inline std::atomic<uint64_t> g_bad_other {0};
inline std::atomic<uint64_t> g_compute {0}; // binds/push de compute fuera del estado de draw
inline void                  CountBad(const char* name) {
	for (auto& e : g_bad) {
		const char* cur = e.name.load(std::memory_order_relaxed);
		if (cur == nullptr && e.name.compare_exchange_strong(cur, name)) cur = name;
		if (cur == name) {
			e.n.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	g_bad_other.fetch_add(1, std::memory_order_relaxed);
}

// (paso A3) CONTENIDO de los anillos de subida. El push de graficos del camino de tabla lleva, por etapa, la direccion de
// dispositivo de su bloque (table-xpr.inc TableBlocks: blocks[st] = TableUploadBuffer.BDA + Copy(bloque), 64 bits en
// TableBlockDword) y el trabajador sube a SU anillo (Spec table_upload/shader_upload/draw): mismo bloque, otra direccion.
// StreamBuffer::Commit (solo con t_rec: render en seguimiento o trabajador capturando) apunta aqui direccion -> hash de
// los bytes subidos, y (handle, offset) -> hash para VB/IB. La variante "sin handles" (vn) compara ese hash en lugar
// del valor: igual = el shader lee los MISMOS datos. Tabla global (el render busca direcciones del trabajador), sin
// cerrojo: clave guardada como clave^hash; una lectura rota no casa y cuenta como distinta (nunca igual por error).
[[nodiscard]] inline bool t_cap_on() noexcept;
struct Content {
	static constexpr uint32_t kN = 1u << 14; // (A3b) 256 KiB, cabe en L2: ~2 fotogramas de subidas de ambos hilos
	struct E {
		std::atomic<uint64_t> kx {0}, h {0};
	};
	E e[kN];
};
inline Content                g_content;
inline std::atomic<uint64_t>  g_content_n {0}, g_content_big {0};
// (A3b) solo se apunta lo que pasa por StreamBuffer::Copy, hasheando el ORIGEN en memoria del host (cacheable); nunca se
// leen los bytes ya escritos en el anillo mapeado (memoria de GPU combinada en escritura: leerla costaba ~90 ms/fot en B).
constexpr uint64_t            kContentMax = 4 * 1024; // subidas mayores: no se apuntan (distintas por valor)
// (A3b, diag) coste de apuntar/sustituir, ticks de rdtsc: [0] hilo de render (seguidor), [1] trabajador (captura)
inline std::atomic<uint64_t> g_cost_note[2] {}, g_cost_subst[2] {};
inline thread_local uint32_t t_cost_tick = 0;
[[nodiscard]] inline uint64_t ContentSlot(uint64_t key) noexcept { return (key ^ (key >> 23) ^ (key >> 41)) * 0x9e3779b97f4a7c15ull; }
// (A3b) rango de direcciones de anillo apuntadas: filtro antes de buscar en la tabla (un push tiene 31 pares de dwords y casi
// ninguno es direccion; sin filtro cada par era una falta de cache en la tabla de 1 MiB)
inline std::atomic<uint64_t> g_bda_lo {~0ull}, g_bda_hi {0};
inline void BdaRange(uint64_t a, uint64_t size) noexcept {
	if (a < g_bda_lo.load(std::memory_order_relaxed)) g_bda_lo.store(a, std::memory_order_relaxed);
	if (a + size > g_bda_hi.load(std::memory_order_relaxed)) g_bda_hi.store(a + size, std::memory_order_relaxed);
}
inline void ContentPut(uint64_t key, uint64_t hash) noexcept {
	auto& s = g_content.e[ContentSlot(key) >> 50];
	s.kx.store(0, std::memory_order_relaxed);
	s.h.store(hash, std::memory_order_relaxed);
	s.kx.store(key ^ hash, std::memory_order_release);
}
[[nodiscard]] inline bool ContentGet(uint64_t key, uint64_t& hash) noexcept {
	if (key == 0) return false;
	auto&          s  = g_content.e[ContentSlot(key) >> 50];
	const uint64_t kx = s.kx.load(std::memory_order_acquire);
	const uint64_t h  = s.h.load(std::memory_order_relaxed);
	if (kx == 0 || (kx ^ h) != key || s.kx.load(std::memory_order_relaxed) != kx) return false;
	hash = h;
	return true;
}
[[nodiscard]] inline uint64_t HandleKey(uint64_t handle, uint64_t offset) noexcept {
	return TejCoser::Mix(TejCoser::Mix(0x5a17ull, handle), offset) | 1;
}
// (A3b) hash de lo subido: 3 carriles de CRC32C (1 instruccion por 8 bytes y carril, ~3x el multiplicador; SSE4.2, que
// tiene toda CPU capaz de correr el juego). Es comparacion de igualdad, no criptografia.
#ifdef _MSC_VER
#define KYTY_TEJ_CRC
#else
#define KYTY_TEJ_CRC __attribute__((target("sse4.2")))
#endif
KYTY_TEJ_CRC [[nodiscard]] inline uint64_t ContentHash(const uint8_t* d, uint64_t n) noexcept {
	uint64_t a = 0x243f6a8885a308d3ull ^ n, b = 0x13198a2e03707344ull, c = 0xa4093822299f31d0ull, i = 0;
	for (; i + 24 <= n; i += 24) {
		uint64_t w[3];
		std::memcpy(w, d + i, 24);
		a = _mm_crc32_u64(a, w[0]), b = _mm_crc32_u64(b, w[1]), c = _mm_crc32_u64(c, w[2]);
	}
	for (; i + 8 <= n; i += 8) {
		uint64_t w;
		std::memcpy(&w, d + i, 8);
		a = _mm_crc32_u64(a, w);
	}
	if (i < n) {
		uint64_t w = 0;
		std::memcpy(&w, d + i, n - i);
		b = _mm_crc32_u64(b, w);
	}
	return a ^ (b << 21 | b >> 43) ^ (c << 42 | c >> 22) ^ (n * 0x9e3779b97f4a7c15ull);
}
// Gancho de StreamBuffer::Commit: [data, data+size) recien subido en (handle, offset), direccion bda (0: sin BDA).
inline void ContentNote(uint64_t handle, uint64_t bda, uint64_t offset, const uint8_t* data, uint64_t size) noexcept {
	if (data == nullptr || size == 0) return;
	struct T { // (A3b) cronometra 1 de cada 16 llamadas (x16): rdtsc cuesta ~10-20 ns, tanto como una subida pequena
		uint64_t t0 = (++t_cost_tick & 15) == 0 ? __rdtsc() : 0;
		~T() {
			if (t0 != 0) g_cost_note[t_cap_on() ? 1 : 0].fetch_add((__rdtsc() - t0) * 16, std::memory_order_relaxed);
		}
	} timer;
	if (size > kContentMax) {
		g_content_big.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const uint64_t h0 = ContentHash(data, size);
	uint64_t h = TejCoser::Mix(0x5eedull, h0);
	h |= 1;
	if (bda != 0) BdaRange(bda + offset, size), ContentPut(bda + offset, h);
	ContentPut(HandleKey(handle, offset), h);
	g_content_n.fetch_add(1, std::memory_order_relaxed);
}
// Push por contenido: cada par de dwords (cualquier alineacion: TableBlockDword puede ser impar) que es una direccion
// apuntada se sustituye por el hash de lo que apunta. Devuelve cuantas sustituyo.
inline uint32_t PushByContent(uint32_t* d, size_t dwords) noexcept {
	const uint64_t t0 = (++t_cost_tick & 15) == 0 ? __rdtsc() : 0;
	uint32_t       n  = 0;
	const uint64_t lo = g_bda_lo.load(std::memory_order_relaxed), hi = g_bda_hi.load(std::memory_order_relaxed);
	for (size_t i = 0; i + 1 < dwords; ++i) {
		const uint64_t a = d[i] | (uint64_t {d[i + 1]} << 32);
		if (a < lo || a >= hi || a < 0x10000) continue;
		uint64_t h;
		if (!ContentGet(a, h)) continue;
		d[i] = static_cast<uint32_t>(h), d[i + 1] = static_cast<uint32_t>(h >> 32);
		++i, ++n;
	}
	if (t0 != 0) g_cost_subst[t_cap_on() ? 1 : 0].fetch_add((__rdtsc() - t0) * 16, std::memory_order_relaxed);
	return n;
}
inline std::atomic<uint64_t> g_push_subst {0}; // (diag) direcciones de push sustituidas por contenido
// (A4, alineacion) Cada draw lleva un ID ESTABLE = (clave PM4 del draw = posicion en la DCB raiz << 32 | en el bufer
// actual, la misma en el render y en la copia del trabajador; sub = n-esimo vkCmdDraw* desde ese draw PM4). Antes se
// comparaba el draw i con el draw i: un hueco del trabajador o un draw de mas del emulador desalineaba todo lo posterior.
struct DrawRec {
	uint64_t key = 0;
	uint32_t sub = 0, indexed = 0;
	uint64_t x[kState] {}, xn[kState] {}; // estado vigente por categoria (exacto / sin handles) en ESTE draw
	uint64_t v = 0, vn = 0;               // argumentos del draw
	uint64_t pw[kPushQ] {};               // push de graficos (offset 0) por palabra, por contenido
};
inline thread_local uint64_t t_key = 0; // clave PM4 del draw en curso (render: OnDraw; sombra: marcador @draw)
inline thread_local uint32_t t_sub = 0;
inline thread_local uint64_t t_done = 0; // (A4, diag) llamadas aplicadas al seguidor en este hilo
inline thread_local bool     t_note = false; // (A4) apuntar subidas aunque no se siga (muestreo: los bloques se reutilizan)
inline thread_local std::vector<uint64_t>* t_holes = nullptr; // (A4) sombra: claves de draws que el trabajador dejo en hueco
inline std::atomic<uint64_t> g_vb_subst {0};   // (diag) VB/IB en anillo sustituidos por contenido

struct Acc {
	static constexpr uint32_t kSlots = 256; // tabla abierta (potencia de 2)
	struct Slot {
		uint64_t key = 0, v = 0, vn = 0;
		uint32_t cat = 0;
	};
	Slot     slot[kSlots] {};
	uint32_t used = 0;
	uint64_t s[kState] {}, sn[kState] {}; // estado vigente por categoria (suma de Mix(clave, valor) de sus ranuras)
	uint64_t ib = 0, ibn = 0;             // IB vigente (cuenta solo en draws indexados)
	// Por draw (desde ResetDraws): hash del estado vigente en cada draw, exacto y sin handles; argumentos del draw.
	uint64_t h[kState] {}, hn[kState] {};
	uint64_t hd = kBasis, hdn = kBasis;
	uint64_t pq[kPushQ] {}; // (diag) secuencia de cada palabra de las push constants de graficos (offset 0) en los draws
	uint64_t pcur[kPushQ] {};
	uint32_t draws = 0;
	bool     overflow = false;
	std::vector<DrawRec>* dl = nullptr; // (A4) lista por draw con su ID (render: g_rdl, sombra: g_sdl)

	void ResetDraws() {
		for (uint32_t c = 0; c < kState; ++c) h[c] = hn[c] = kBasis;
		for (uint32_t i = 0; i < kPushQ; ++i) pq[i] = 0;
		hd = hdn = kBasis, draws = 0;
		if (dl != nullptr) dl->clear();
	}
	void ResetAll() {
		for (auto& e : slot) e.key = 0;
		used = 0, overflow = false, ib = ibn = 0;
		for (uint32_t c = 0; c < kState; ++c) s[c] = sn[c] = 0;
		for (uint32_t i = 0; i < kPushQ; ++i) pcur[i] = 0;
		ResetDraws();
	}
	Acc() { ResetDraws(); }
	void Draw(bool indexed, uint64_t v, uint64_t vn) {
		++draws;
		DrawRec* r = nullptr;
		if (dl != nullptr) {
			r = &dl->emplace_back();
			r->key = t_key, r->sub = t_sub++, r->indexed = indexed ? 1 : 0, r->v = v, r->vn = vn;
			std::memcpy(r->pw, pcur, sizeof(pcur));
		}
		for (uint32_t c = 0; c < kState; ++c) {
			uint64_t x = s[c], xn = sn[c];
			if (c == CVbib && indexed) x += ib, xn += ibn;
			h[c] = Mix(h[c], x), hn[c] = Mix(hn[c], xn);
			if (r != nullptr) r->x[c] = x, r->xn[c] = xn;
		}
		for (uint32_t i = 0; i < kPushQ; ++i) pq[i] = ((pq[i] << 7) | (pq[i] >> 57)) ^ pcur[i];
		hd = Mix(hd, v), hdn = Mix(hdn, vn);
	}
	void Apply(uint32_t cat, uint64_t key, uint64_t v, uint64_t vn) {
		key |= 1; // 0 = ranura libre
		uint32_t i = static_cast<uint32_t>(key ^ (key >> 29)) & (kSlots - 1);
		for (uint32_t probe = 0; probe < 16; ++probe, i = (i + 1) & (kSlots - 1)) {
			auto& e = slot[i];
			if (e.key == key) {
				s[cat] += Mix(key, v) - Mix(key, e.v);
				sn[cat] += Mix(key, vn) - Mix(key, e.vn);
				e.v = v, e.vn = vn;
				return;
			}
			if (e.key == 0) {
				e = {key, v, vn, cat};
				++used;
				s[cat] += Mix(key, v), sn[cat] += Mix(key, vn);
				return;
			}
		}
		overflow = true;
	}
};

inline thread_local Acc*                                     t_acc = nullptr; // seguidor de estado (render y sombra)
inline thread_local std::vector<uint64_t>*                   t_cap = nullptr; // captura (trabajador)
inline thread_local std::vector<std::shared_ptr<const void>>* t_own = nullptr; // retencion de duenos de paquetes
inline thread_local bool                                     t_rec = false;   // t_acc || t_cap (un solo TLS en cada vkCmd)
inline thread_local bool                                     t_bad = false;   // llamada no capturable en el paquete

[[nodiscard]] inline bool t_cap_on() noexcept { return t_cap != nullptr; }
inline void Sync() { t_rec = t_acc != nullptr || t_cap != nullptr; }

template <typename T>
[[nodiscard]] inline uint64_t W(const T& v) noexcept {
	if constexpr (std::is_pointer_v<T>) return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v));
	else if constexpr (std::is_enum_v<T>) return static_cast<uint64_t>(static_cast<std::underlying_type_t<T>>(v));
	else if constexpr (std::is_integral_v<T>) return static_cast<uint64_t>(v);
	else {
		static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 8);
		uint64_t r = 0;
		std::memcpy(&r, &v, sizeof(T));
		return r;
	}
}
template <typename T>
[[nodiscard]] inline T From(uint64_t w) noexcept {
	if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(static_cast<uintptr_t>(w));
	else if constexpr (std::is_enum_v<T>) return static_cast<T>(static_cast<std::underlying_type_t<T>>(w));
	else if constexpr (std::is_integral_v<T>) return static_cast<T>(w);
	else {
		static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 8);
		T r;
		std::memcpy(&r, &w, sizeof(T));
		return r;
	}
}
// Formato de un arreglo capturado: [bytes (~0 = puntero nulo), palabras...].
template <typename T>
[[nodiscard]] inline const T* Arr(const uint64_t*& p, const uint64_t* e) noexcept {
	if (p >= e) return nullptr;
	const uint64_t bytes = *p++;
	if (bytes == ~uint64_t {0}) return nullptr;
	const auto* r = reinterpret_cast<const T*>(p);
	p += (bytes + 7) / 8;
	return bytes == 0 ? nullptr : r;
}

constexpr uint32_t kPacket = Id("@paquete");
constexpr uint32_t kVb     = Id("@vb");
constexpr uint32_t kDrawKey = Id("@draw");  // (A4) [id, 1, clave PM4]: los vkCmd siguientes son de ese draw
constexpr uint32_t kHoleKey = Id("@hueco"); // (A4) [id, 1, clave PM4]: el trabajador dejo ese draw en hueco
// (A4) Un draw PM4 empieza (render: Tej::OnDraw; trabajador: lo apunta en su lista para que la sombra lo vea).
inline void MarkDraw(uint64_t key) noexcept {
	t_key = key, t_sub = 0;
	if (t_cap != nullptr) t_cap->push_back(kDrawKey), t_cap->push_back(1), t_cap->push_back(key);
}
inline void MarkHole() {
	if (t_cap != nullptr) t_cap->push_back(kHoleKey), t_cap->push_back(1), t_cap->push_back(t_key);
}
constexpr uint64_t kGraphicsStages = 0x1f | 0x40 | 0x80; // VS..FS, task, mesh (VkShaderStageFlagBits)

// Una llamada: argumentos en orden (V valor, A arreglo); Done la aplica al seguidor y cierra la captura.
struct Rec {
	uint32_t               id, cat, mode;
	std::vector<uint64_t>* out;
	size_t                 hdr = 0;
	uint64_t               k[3] {};
	uint32_t               nv = 0, na = 0;
	const void*            ap[4] {};
	size_t                 ab[4] {};
	uint64_t               vh = kBasis, vn = kBasis;
	uint64_t               pc[kPushQ] {}; // (paso A3) push con las direcciones de anillo sustituidas por contenido
	size_t                 pcb = 0;
	bool                   bad = false;
	Rec(uint32_t id_, uint32_t cat_, bool capture, uint32_t mode_ = MNone)
	    : id(id_), cat(cat_), mode(mode_), out(capture ? t_cap : nullptr) {
		if (out != nullptr) {
			hdr = out->size();
			out->push_back(id);
			out->push_back(0);
		}
	}
	void V(uint64_t w, bool handle) {
		if (nv < 3) k[nv] = w;
		++nv;
		vh = Mix(vh, w), vn = Mix(vn, handle ? 0 : w);
		if (out != nullptr) out->push_back(w);
	}
	void A(const void* p, size_t bytes, bool handle) {
		if (p == nullptr) bytes = 0;
		if (na < 4) ap[na] = p, ab[na] = bytes;
		++na;
		vh = Mix(vh, bytes), vn = Mix(vn, bytes);
		const size_t words = (bytes + 7) / 8;
		size_t       at    = 0;
		if (out != nullptr) {
			out->push_back(p == nullptr ? ~uint64_t {0} : bytes);
			at = out->size();
			out->resize(at + words, 0);
			if (bytes != 0) std::memcpy(out->data() + at, p, bytes);
		}
		if (t_acc == nullptr || mode == MVb) return; // (VB: Done los mezcla por binding)
		// (paso A3) push: la variante sin handles compara por CONTENIDO (direcciones de anillo -> hash de lo apuntado)
		const bool push = mode == MPush && !handle && na == 1 && bytes != 0 && bytes <= sizeof(pc);
		if (push) {
			std::memset(pc, 0, sizeof(pc));
			std::memcpy(pc, p, bytes);
			const uint32_t n = PushByContent(reinterpret_cast<uint32_t*>(pc), bytes / 4);
			if (n != 0) g_push_subst.fetch_add(n, std::memory_order_relaxed);
			pcb = bytes;
		}
		for (size_t i = 0; i < words; ++i) {
			uint64_t w = 0;
			if (out != nullptr) w = (*out)[at + i];
			else std::memcpy(&w, static_cast<const uint8_t*>(p) + i * 8, i * 8 + 8 <= bytes ? 8 : bytes - i * 8);
			vh = Mix(vh, w);
			if (!handle) vn = Mix(vn, push ? pc[i] : w);
		}
	}
	void Vb(Acc& a) const {
		// (firstBinding, bindingCount, pBuffers, pOffsets[, pSizes, pStrides])
		const auto  first = static_cast<uint32_t>(k[0]), count = static_cast<uint32_t>(k[1]);
		const auto* buf   = static_cast<const uint64_t*>(ap[0]);
		const auto* off   = static_cast<const uint64_t*>(ap[1]);
		const auto* size  = na > 2 ? static_cast<const uint64_t*>(ap[2]) : nullptr;
		const auto* strd  = na > 3 ? static_cast<const uint64_t*>(ap[3]) : nullptr;
		for (uint32_t i = 0; i < count; ++i) {
			const uint64_t b = buf ? buf[i] : 0, o = off ? off[i] : 0, z = size ? size[i] : ~uint64_t {0};
			const uint64_t st = strd ? strd[i] : 0;
			const uint64_t vv = Mix(Mix(Mix(kBasis, o), z), st);
			// (paso A3) VB en un anillo de subida: el contenido subido en lugar del offset
			uint64_t ch = 0, vc = vv;
			if (ContentGet(HandleKey(b, o), ch)) vc = Mix(Mix(Mix(kBasis, ch), z), st), g_vb_subst.fetch_add(1, std::memory_order_relaxed);
			a.Apply(CVbib, Mix(Mix(kBasis, kVb), first + i), Mix(vv, b), vc);
		}
	}
	void Done() {
		if (out != nullptr) (*out)[hdr + 1] = out->size() - hdr - 2;
		if (bad) t_bad = true;
		if (t_acc == nullptr) return;
		++t_done;
		auto&    a   = *t_acc;
		uint64_t key = Mix(kBasis, id);
		switch (mode) {
			case MDraw:
			case MDrawI: a.Draw(mode == MDrawI, Mix(vh, id), Mix(vn, id)); return;
			case MVb: Vb(a); return;
			case MIb: {
				a.ib = Mix(vh, id), a.ibn = Mix(vn, id);
				uint64_t ch = 0; // (paso A3) IB en un anillo de subida: (buffer, offset, tipo) -> contenido
				if (ContentGet(HandleKey(k[0], k[1]), ch))
					a.ibn = Mix(Mix(Mix(kBasis, ch), k[2]), id), g_vb_subst.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			case MPipe:
			case MSets:
				if (k[0] != 0) { // VK_PIPELINE_BIND_POINT_GRAPHICS = 0
					g_compute.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				key = Mix(Mix(kBasis, cat), mode == MSets ? k[2] : 0); // (Sets/Sets2/PushDescriptor: el mismo estado)
				break;
			case MPush: {
				if ((k[1] & kGraphicsStages) == 0) {
					g_compute.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				key = Mix(Mix(kBasis, CPush), k[2]);
				// (diag) contenido por posicion: offset 0 (los 128 bytes del PushData del emulador)
				if (k[2] == 0 && na > 0 && ap[0] != nullptr) {
					const size_t q = ab[0] / 8 < kPushQ ? ab[0] / 8 : kPushQ;
					std::memcpy(a.pcur, pcb != 0 ? static_cast<const void*>(pc) : ap[0], q * 8);
				}
				break;
			}
			case MK0: key = Mix(key, k[0]); break;
			default: break;
		}
		a.Apply(cat, key, Mix(vh, id), Mix(vn, id));
	}
};

// Llamada de una categoria que no se puede capturar (punteros anidados, longitudes complejas): el paquete no es
// reproducible; en el seguidor cambia su ranura con un valor unico (nunca da "igual" por error).
inline void Bad(uint32_t id, uint32_t cat, const char* name) {
	t_bad = true;
	if (t_cap != nullptr) CountBad(name);
	if (t_acc != nullptr) {
		static thread_local uint64_t n = 0;
		++n;
		if (cat >= CDraw) t_acc->Draw(false, Mix(id, n), Mix(id, n));
		else t_acc->Apply(cat, Mix(Mix(kBasis, id), 1), Mix(id, n), Mix(id, n));
	}
}
// Paquete que no se puede capturar en el trabajador (no es de draws/descriptores, sin dueno...): solo marca y cuenta.
inline void BadPacket(const char* why) {
	t_bad = true;
	CountBad(why);
}

} // namespace TejRepro
