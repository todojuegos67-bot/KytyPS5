#pragma once
// (buf7, objdump 01.007.000) La malla es duena de sus dos bloques: 0xaa06a0 en 0xaa12f2..0xaa131d hace
// malla+0x298 = *(&bloque) (bloque de vertices de 0xaa1f70/0x7f9cf0: tabla de secciones al principio, 0x18 bytes cada
// una, +0x10 offset -> puntero en 0xaa2000) y malla+0x2a0 = indices+8 (0xac26c0/0x7f9cf0), y pone a 0 los origenes: los
// 0xaa1e50/0xaa1ea0/0x7fa380 de MeshFactory liberan NULL. 0xaa1d10 los libera en diferido (0x903460) y pone +0x298/+0x2a0
// a 0. Los V# (flags 0x800) de los CBuffer apuntan DENTRO de esos bloques. Causa del bucle (073717/074903) y del cuelgue
// en dlmalloc 0x803528 (075619): OnMeshFree NO olvidaba lo tomado; liberado el bloque, dlmalloc/otros objetos (la propia
// malla: 0xbcaede lee malla+0x1c0) vivian en paginas aun vigiladas/soltadas y el primer toque metia bytes viejos o
// fallaba sin fin. buf7: se toma el bloque entero de +0x298 y el de +0x2a0 (sin cabecera de dlmalloc ni tabla de
// secciones: 64 KiB de margen, paginas enteras) y en 0xaa1d10, ANTES del free, se olvida (esperando a RAM18 si lo escribe).
// buf6: los CBuffer de malla son externos (flags 0x800, datos en el V#), no copias; buf4 (Ghidra): solo los 15 CBuffer de la malla con memoria propia, verificados contra el memcpy; 64 KiB de margen;
// al destruir la malla NO se pone nada a cero (causa del cuelgue de buf3 en 0x8e4f46).
// (BUF-NATIVO) KYTY_BUF_NATIVO=1 (encendido por defecto; =0 lo apaga): las mallas del juego (vertices e indices) solo en la
// VRAM, fuera de la RAM emulada.
//
// Ghidra 01.007.000:
//  - CMesh::MeshFactory 0xac0410 lee el flujo: 0xaa1f70 (bloque de vertices: 0x7f9cf0 + lectura 0x813130) y 0xac26c0
//    (indices), y llama a 0xaa06a0(malla, &bloque, &indices, ...). 0xaa06a0 busca POS0/COL*/TEX*/QUT0/BONI/BONW/SAT_
//    (0xaa1f50) y crea un CBuffer por seccion y el de "kIndices" con CBuffer::Create 0x909460, que copia los datos con
//    un solo memcpy (llamada 0x90954b, vuelve a 0x909550).
//  - Justo despues MeshFactory LIBERA el bloque y los indices del flujo (0x7fa380, 0xaa1e50/0xaa1ea0): esos rangos son
//    temporales y NO se registran. Lo que vive (y lee la GPU) es la memoria de los CBuffer: el destino de esos memcpy.
//  - El destructor de la malla es 0xaa1d10(malla): destruye sus CBuffer (0x9093c0 -> 0x909410) y libera en diferido
//    (0x903460). Aqui se olvidan sus rangos ANTES de eso, sin escribir nada en la RAM del juego.
//
// Flujo: el gancho de 0xaa06a0 marca el hilo con la malla; el memcpy de 0x909550 en ese hilo registra el destino; tras
// KYTY_BUF_NATIVO_DELAY_S (10 s: la cache de buferes ya lo subio a la VRAM) el hilo de render toma las paginas enteras
// del rango (sin la primera/ultima parcial) como rangos frios del offload nativo (texture-offload.h, MeshTakeNow): el
// hilo RAM18 copia los bytes al almacen (VRAM con mem1 etapa 2; si no, el archivo RAM17) y suelta la RAM. Cualquier
// toque (CPU del juego, ventana del emulador, subida de la cache de buferes) los lee de vuelta antes, y ese rango queda
// en RAM para siempre (contado en la auditoria). Al destruir la malla se olvidan (sin restaurar, sin escribir).
//
// Necesita la memoria nativa (KYTY_NATIVE_MEMORY / KYTY_RAIZ) y KYTY_RAM17=1 (el perfil de poca RAM los pone): si no,
// se queda inerte y lo dice en el log. Linea "[buf-nativo]" cada 30 s.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <vector>

namespace BufNativo {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BUF_NATIVO");
		return v != nullptr && v[0] == '1'; // (apagado: seguia colgando 11-24 09:08; =1 lo enciende)
	}();
	return on;
}

inline int DelaySeconds() {
	static const int s = [] {
		const char* v = std::getenv("KYTY_BUF_NATIVO_DELAY_S");
		const long  a = v != nullptr && v[0] != '\0' ? std::strtol(v, nullptr, 10) : 10;
		return static_cast<int>(a < 2 ? 2 : a > 600 ? 600 : a);
	}();
	return s;
}

using ForgetFn = void (*)(const std::vector<std::pair<uint64_t, uint64_t>>& chunks);

constexpr uint64_t Page   = 4096;
constexpr uint64_t Chunk  = uint64_t {1} << 20u; // (el maximo de un rango frio: texture-offload.h FileChunk)
constexpr uint64_t Margin = uint64_t {64} << 10u; // (buf4: 64 KiB sin tocar a cada lado del bloque del CBuffer)

// (buf7) Offsets en la malla (0x2a8 bytes, 0x80a330): 15 CBuffer en +8 + i*0x28 (+0x18 flags, +0x20 bytes), y los
// dos bloques que posee.
constexpr uint64_t SlotFirst = 0x8, SlotStride = 0x28, SlotCount = 15;
constexpr uint64_t VertexBlock = 0x298, IndexBlock = 0x2a0;
constexpr uint32_t ExternFlag = 0x800u, AliasFlag = 0x400u;
constexpr uint64_t MaxBlock   = uint64_t {256} << 20u; // (cordura: un V# a mas de esto del bloque no es suyo)
inline uint64_t VsharpBase(uint64_t slot) {
	return uint64_t {*reinterpret_cast<const uint32_t*>(slot)} | (uint64_t {*reinterpret_cast<const uint16_t*>(slot + 4)} << 32u);
}
inline uint64_t ReadU64(uint64_t a) { return *reinterpret_cast<const uint64_t*>(a); }

// (buf7) El final de los datos de un bloque: el mayor base+bytes de los CBuffer 0x800 propios (sin 0x400) cuyo V# cae
// dentro de [bloque, bloque + MaxBlock). 0 si ninguno.
inline uint64_t BlockEnd(uint64_t mesh, uint64_t block) {
	uint64_t end = 0;
	for (uint64_t i = 0; i < SlotCount; i++) {
		const uint64_t slot  = mesh + SlotFirst + i * SlotStride;
		const uint32_t flags = *reinterpret_cast<const uint32_t*>(slot + 0x18);
		const uint32_t size  = *reinterpret_cast<const uint32_t*>(slot + 0x20);
		if ((flags & ExternFlag) == 0 || (flags & AliasFlag) != 0 || size == 0) continue;
		const uint64_t base = VsharpBase(slot);
		if (base < block || base - block >= MaxBlock) continue;
		end = std::max(end, base + size);
	}
	return end;
}

struct Entry {
	uint64_t                                   end   = 0;
	uint64_t                                   owner = 0; // (la malla: el this de 0xaa06a0 / 0xaa1d10)
	uint64_t                                   slot  = 0; // (buf7: malla+0x298 o malla+0x2a0)
	uint64_t                                   data  = 0; // (el puntero al bloque al registrar)
	uint32_t                                   size  = 0; // (no usado)
	std::chrono::steady_clock::time_point      when;
	bool                                       taken = false;
	std::vector<std::pair<uint64_t, uint64_t>> chunks; // (los propuestos al offload)
};

struct Registry {
	std::mutex                                           mutex;
	std::mutex                                           take_mutex; // (buf7: Due..MeshTakeNow frente a OnMeshFree)
	std::map<uint64_t, Entry>                            ranges;   // inicio -> rango (disjuntos)
	std::unordered_map<uint64_t, std::vector<uint64_t>> by_owner; // malla -> inicios
	uint64_t                                             live_bytes = 0;
	std::atomic<ForgetFn>                                forget {nullptr}; // (buf4: ya no se usa; ver OnMeshFree)
	std::atomic<bool>                                    hooked {false};
	// Contadores (todo el juego).
	std::atomic<uint64_t> meshes {0}, copies {0}, copy_bytes {0}, freed {0}, forgot_chunks {0}, replaced {0};
	std::atomic<uint64_t> rejected {0}, stale {0}; // (buf4: CBuffer no validos al registrar / cambiados antes de tomar)
};

inline Registry& Get() {
	static auto* r = new Registry(); // (nunca se destruye: hilos del juego hasta el final)
	return *r;
}

inline thread_local uint64_t                                  t_owner = 0;
inline thread_local std::vector<std::pair<uint64_t, uint64_t>> t_copies; // (destino, tamano) de los memcpy de 0x909550

// (hilo del juego, gancho de 0xaa06a0) La malla que este hilo construye; 0 al salir.
inline uint64_t Enter(uint64_t owner) {
	const uint64_t prev = t_owner;
	t_owner             = owner;
	if (prev == 0) t_copies.clear();
	Get().meshes.fetch_add(1, std::memory_order_relaxed);
	return prev;
}

// (state.mutex held) Quita del registro los rangos que pisan [b, e).
inline void EraseOverLocked(Registry& r, uint64_t b, uint64_t e) {
	auto it = r.ranges.upper_bound(b);
	if (it != r.ranges.begin() && std::prev(it)->second.end > b) it = std::prev(it);
	while (it != r.ranges.end() && it->first < e) {
		auto& v = r.by_owner[it->second.owner];
		std::erase(v, it->first);
		if (v.empty()) r.by_owner.erase(it->second.owner);
		r.live_bytes -= std::min(r.live_bytes, it->second.end - it->first);
		r.replaced.fetch_add(1, std::memory_order_relaxed);
		it = r.ranges.erase(it);
	}
}

// (memcpy del juego, antes de copiar) Solo se anota: no registra nada por si solo (buf3 registraba cualquier destino).
inline void OnMeshCopy(uint64_t dst, uint64_t size) {
	if (t_owner == 0 || size == 0) return;
	t_copies.emplace_back(dst, size);
}

// La parte del bloque que se puede soltar: paginas enteras a >= 64 KiB de cada borde (la cabecera del heap y lo que
// el heap ponga al lado quedan fuera). 0 si no hay al menos 64 KiB utiles.
inline std::pair<uint64_t, uint64_t> Inner(uint64_t data, uint64_t size) {
	const uint64_t lo = (data + Margin + Margin - 1) & ~(Margin - 1);
	const uint64_t e  = data + size;
	if (e < Margin) return {0, 0};
	const uint64_t hi = (e - Margin) & ~(Margin - 1);
	if (hi <= lo || hi - lo < Margin) return {0, 0};
	return {lo, hi};
}

// (hilo del juego, gancho de 0xaa06a0, al volver) buf7: registra los dos bloques que la malla posee ya (+0x298, +0x2a0).
inline void Leave(uint64_t prev) {
	const uint64_t owner = t_owner;
	t_owner              = prev;
	if (owner == 0 || prev != 0) return; // (anidado: lo registra el de fuera)
	auto&           r = Get();
	std::lock_guard lock(r.mutex);
	for (const uint64_t off: {VertexBlock, IndexBlock}) {
		const uint64_t block = ReadU64(owner + off);
		if (block == 0) continue;
		const uint64_t end = BlockEnd(owner, block);
		if (end <= block) {
			r.rejected.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		const auto [lo, hi] = Inner(block, end - block);
		if (lo == 0) continue; // (pequeno: no merece la pena ni es seguro)
		EraseOverLocked(r, lo, hi);
		Entry e;
		e.end   = hi;
		e.owner = owner;
		e.slot  = owner + off;
		e.data  = block;
		e.when  = std::chrono::steady_clock::now();
		r.ranges.emplace(lo, std::move(e));
		r.by_owner[owner].push_back(lo);
		r.live_bytes += hi - lo;
		r.copies.fetch_add(1, std::memory_order_relaxed);
		r.copy_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
	}
	t_copies.clear();
}

// (hilo del juego, gancho de 0xaa1d10, ANTES del destructor: los bloques aun son de la malla, nadie los reutiliza)
// buf7: los trozos ya tomados se OLVIDAN (forget = TextureOffload::MeshForget: espera a RAM18 si los escribe, quita los
// vigilantes, libera su copia; sin restaurar). Sus bytes ya no importan: 0x903460 los libera. La cabecera de dlmalloc y
// la tabla de secciones quedan fuera (margen). take_mutex: el hilo de render no puede tomar entre Due y MeshTakeNow.
inline void OnMeshFree(uint64_t owner) {
	auto&                                      r = Get();
	std::vector<std::pair<uint64_t, uint64_t>> chunks;
	std::lock_guard                            take(r.take_mutex);
	{
		std::lock_guard lock(r.mutex);
		auto            o = r.by_owner.find(owner);
		if (o == r.by_owner.end()) return;
		for (const auto begin: o->second) {
			auto it = r.ranges.find(begin);
			if (it == r.ranges.end()) continue;
			chunks.insert(chunks.end(), it->second.chunks.begin(), it->second.chunks.end());
			r.live_bytes -= std::min(r.live_bytes, it->second.end - begin);
			r.ranges.erase(it);
		}
		r.by_owner.erase(o);
		r.freed.fetch_add(1, std::memory_order_relaxed);
	}
	if (!chunks.empty()) {
		if (const auto fn = r.forget.load(std::memory_order_acquire); fn != nullptr) {
			fn(chunks);
			r.forgot_chunks.fetch_add(chunks.size(), std::memory_order_relaxed);
		}
	}
}

// (hilo de render) Los rangos con mas de DelaySeconds desde su copia, en trozos de <= 1 MiB. Antes de tomar se
// comprueba otra vez que el CBuffer sigue igual (mismo puntero, tamano y flags buenos): si no, se descarta. La malla
// no puede liberarse mientras se mira: su 0xaa1d10 espera a este cerrojo.
inline std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> Due(uint8_t kind, uint64_t* pending_bytes) {
	std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> out;
	auto&                                                r     = Get();
	const auto                                           limit = std::chrono::steady_clock::now() - std::chrono::seconds(DelaySeconds());
	std::lock_guard                                      lock(r.mutex);
	uint64_t                                             pending = 0;
	for (auto it = r.ranges.begin(); it != r.ranges.end();) {
		auto& [begin, e] = *it;
		if (e.taken) { ++it; continue; }
		if (e.when > limit) {
			pending += e.end - begin;
			++it;
			continue;
		}
		if (ReadU64(e.slot) != e.data) { // (buf7: la malla ya no posee ese bloque)
			r.stale.fetch_add(1, std::memory_order_relaxed);
			auto& v = r.by_owner[e.owner];
			std::erase(v, begin);
			if (v.empty()) r.by_owner.erase(e.owner);
			r.live_bytes -= std::min(r.live_bytes, e.end - begin);
			it = r.ranges.erase(it);
			continue;
		}
		e.taken = true;
		for (uint64_t at = begin; at < e.end; at += Chunk) {
			const uint64_t stop = std::min(e.end, at + Chunk);
			e.chunks.emplace_back(at, stop - at);
			out.emplace_back(at, stop - at, kind);
		}
		++it;
		if (out.size() >= 4096) break; // (el resto en la pasada siguiente)
	}
	if (pending_bytes != nullptr) *pending_bytes = pending;
	return out;
}

inline uint64_t LiveBytes() {
	auto&           r = Get();
	std::lock_guard lock(r.mutex);
	return r.live_bytes;
}

} // namespace BufNativo
