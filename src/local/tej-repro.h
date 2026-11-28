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

	void ResetDraws() {
		for (uint32_t c = 0; c < kState; ++c) h[c] = hn[c] = kBasis;
		for (uint32_t i = 0; i < kPushQ; ++i) pq[i] = 0;
		hd = hdn = kBasis, draws = 0;
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
		for (uint32_t c = 0; c < kState; ++c) {
			uint64_t x = s[c], xn = sn[c];
			if (c == CVbib && indexed) x += ib, xn += ibn;
			h[c] = Mix(h[c], x), hn[c] = Mix(hn[c], xn);
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
		for (size_t i = 0; i < words; ++i) {
			uint64_t w = 0;
			if (out != nullptr) w = (*out)[at + i];
			else std::memcpy(&w, static_cast<const uint8_t*>(p) + i * 8, i * 8 + 8 <= bytes ? 8 : bytes - i * 8);
			vh = Mix(vh, w);
			if (!handle) vn = Mix(vn, w);
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
			a.Apply(CVbib, Mix(Mix(kBasis, kVb), first + i), Mix(vv, b), vv);
		}
	}
	void Done() {
		if (out != nullptr) (*out)[hdr + 1] = out->size() - hdr - 2;
		if (bad) t_bad = true;
		if (t_acc == nullptr) return;
		auto&    a   = *t_acc;
		uint64_t key = Mix(kBasis, id);
		switch (mode) {
			case MDraw:
			case MDrawI: a.Draw(mode == MDrawI, Mix(vh, id), Mix(vn, id)); return;
			case MVb: Vb(a); return;
			case MIb: a.ib = Mix(vh, id), a.ibn = Mix(vn, id); return;
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
					std::memcpy(a.pcur, ap[0], q * 8);
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
