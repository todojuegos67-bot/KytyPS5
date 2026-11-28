#pragma once
// (tej-repro) KYTY_TEJ=3, FASE 2 PASO A: captura de los draws del trabajador y REPRODUCCION EN SOMBRA en el render.
// Riesgo de imagen CERO: nada de esto llega a un command buffer real. La reproduccion usa un dispatcher de DESCARTE
// (LocalVulkanRecording::TejStubDispatch: todas las vkCmd* son funciones que solo hashean) y nunca el driver.
//  - Rec: una llamada de draw o de bind (pipeline, sets/push descriptors, push constants, VB/IB, estado dinamico vkCmdSet*,
//    vkCmdDraw*/ClearAttachments) con sus argumentos ya resueltos (POD + handles), en un buffer plano de palabras de 64 bits
//    [id, n, cuerpo...] (t_cap, solo el trabajador) y/o aplicada al seguidor de estado (t_acc).
//  - Acc: estado EFECTIVO por ranura (clave = comando + indice: punto de bind/firstSet, stage/offset, firstViewport...)
//    y, en cada draw, el hash del estado vigente de cada categoria: binds redundantes o elididos no cuentan, solo lo que
//    cada draw ve. Dos variantes por categoria: exacta y "sin handles" (el trabajador usa sus propios sets/anillos).
// Solo el generador (vulkan-recording) y vulkan-recording.cpp escriben aqui; tej-coser.h compara y cuenta.
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

using TejCoser::Id;
using TejCoser::Mix;
constexpr uint64_t kBasis = 0xcbf29ce484222325ull;

struct Acc {
	static constexpr uint32_t kSlots = 96;
	struct Slot {
		uint64_t key = 0, v = 0, vn = 0;
		uint32_t cat = 0;
	};
	Slot     slot[kSlots] {};
	uint32_t used = 0;
	uint64_t s[kState] {}, sn[kState] {}; // estado vigente por categoria (suma de Mix(clave, valor) de sus ranuras)
	// Por draw (desde ResetDraws): hash del estado vigente en cada draw, exacto y sin handles; argumentos del draw.
	uint64_t h[kState] {}, hn[kState] {};
	uint64_t hd = kBasis, hdn = kBasis;
	uint32_t draws = 0;
	bool     overflow = false;

	void ResetDraws() {
		for (uint32_t c = 0; c < kState; ++c) h[c] = hn[c] = kBasis;
		hd = hdn = kBasis, draws = 0;
	}
	void ResetAll() {
		used = 0, overflow = false;
		for (uint32_t c = 0; c < kState; ++c) s[c] = sn[c] = 0;
		ResetDraws();
	}
	Acc() { ResetDraws(); }
	void Apply(uint32_t cat, uint64_t key, uint64_t v, uint64_t vn) {
		if (cat >= CDraw) {
			++draws;
			for (uint32_t c = 0; c < kState; ++c) h[c] = Mix(h[c], s[c]), hn[c] = Mix(hn[c], sn[c]);
			hd = Mix(hd, v), hdn = Mix(hdn, vn);
			return;
		}
		for (uint32_t i = 0; i < used; ++i) {
			auto& e = slot[i];
			if (e.key != key) continue;
			s[cat] += Mix(key, v) - Mix(key, e.v);
			sn[cat] += Mix(key, vn) - Mix(key, e.vn);
			e.v = v, e.vn = vn;
			return;
		}
		if (used == kSlots) {
			overflow = true;
			return;
		}
		slot[used++] = {key, v, vn, cat};
		s[cat] += Mix(key, v), sn[cat] += Mix(key, vn);
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

// Una llamada: argumentos en orden (V valor, A arreglo); Done la aplica al seguidor y cierra la captura.
struct Rec {
	uint32_t               id, cat;
	std::vector<uint64_t>* out;
	size_t                 hdr = 0;
	uint64_t               k[3] {};
	uint32_t               nv = 0;
	uint64_t               vh = kBasis, vn = kBasis;
	bool                   bad = false;
	Rec(uint32_t id_, uint32_t cat_, bool capture) : id(id_), cat(cat_), out(capture ? t_cap : nullptr) {
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
		vh = Mix(vh, bytes), vn = Mix(vn, bytes);
		const size_t words = (bytes + 7) / 8;
		size_t       at    = 0;
		if (out != nullptr) {
			out->push_back(p == nullptr ? ~uint64_t {0} : bytes);
			at = out->size();
			out->resize(at + words, 0);
			if (bytes != 0) std::memcpy(out->data() + at, p, bytes);
		}
		for (size_t i = 0; i < words; ++i) {
			uint64_t w = 0;
			if (out != nullptr) w = (*out)[at + i];
			else std::memcpy(&w, static_cast<const uint8_t*>(p) + i * 8, i * 8 + 8 <= bytes ? 8 : bytes - i * 8);
			vh = Mix(vh, w);
			if (!handle) vn = Mix(vn, w);
		}
	}
	void Done() {
		if (out != nullptr) (*out)[hdr + 1] = out->size() - hdr - 2;
		if (bad) t_bad = true;
		if (t_acc == nullptr) return;
		// Clave de la ranura: el comando y su indice (punto de bind + firstSet/set; stage + offset en push constants).
		uint64_t key = Mix(kBasis, id);
		if (id == Id("vkCmdPushConstants")) key = Mix(Mix(key, k[1]), k[2]);
		else if (id == Id("vkCmdBindDescriptorSets") || id == Id("vkCmdPushDescriptorSetKHR") || id == Id("vkCmdPushDescriptorSet"))
			key = Mix(Mix(key, k[0]), k[2]);
		else if (nv != 0) key = Mix(key, k[0]);
		t_acc->Apply(cat, key, Mix(vh, id), Mix(vn, id));
	}
};

// Llamada de una categoria que no se puede capturar (punteros anidados, longitudes complejas): el paquete no es
// reproducible; en el seguidor cambia su ranura con un valor unico (nunca da "igual" por error).
inline void Bad(uint32_t id, uint32_t cat) {
	t_bad = true;
	if (t_acc != nullptr) {
		static thread_local uint64_t n = 0;
		++n;
		t_acc->Apply(cat, Mix(kBasis, id), Mix(id, n), Mix(id, n));
	}
}

} // namespace TejRepro
