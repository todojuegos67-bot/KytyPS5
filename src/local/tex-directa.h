#pragma once
// (etapa 2.1, 11-23) KYTY_TEX_DIRECTA: ESCRITURA DIRECTA A LA GPU de los trozos del streaming de texturas, manteniendo
// todavia la copia en RAM. APAGADO por defecto (KYTY_TEX_DIRECTA=1 lo enciende).
//
// Ghidra (proyecto r19g, eboot 01.007.000):
//  - El memcpy del juego es el import Q3VBxCXhUHs -> LibcInternal::memcpy (libs/libC.cpp). CTexture::
//    SetCurrentStreamingChunk (0x901330) copia cada trozo con la llamada en eboot+0x901459 al destino
//    base(this+0x98)+desplazamiento(trozo+8), tamano (trozo+4), dentro de VirtualMemo [0x10_0000_0000, 0x14_0000_0000).
//  - Reserva/liberacion de cada mip/capa: 0x901030 (op 0) / 0x901110 (op 2) sobre 0x7f9f70; direccion =
//    ((T#[0xc0]&0xff)<<32 | T#[0xbc])*0x100 + mip*caras*tamano_slice, tamano = caras*tamano_slice (layout Gnm,
//    FUN_01d1c060/FUN_01d1bee0). Es decir: cada trozo cae exactamente sobre los subrecursos (mip/capa) de su imagen.
//  - La CPU del juego no lee VirtualMemo (0 lecturas medidas).
//
// Que hace (solo con KYTY_TEX_DIRECTA=1):
//  1. En LibcInternal::memcpy, si el destino cae en VirtualMemo (trozos >= 4 KiB): se calcula un hash de los bytes de
//     ORIGEN y el trozo se encola. La copia a la RAM se hace igual que siempre (la RAM se mantiene en esta etapa).
//  2. En el hilo de render (antes de cada draw, RenderExecutor::TexAnticipadaDrain), cada trozo encolado se sube YA a
//     su imagen de la GPU, en los subrecursos (mip/capa) que cubre: TextureCache::DirectUploadRange ->
//     UploadImageSubresources, el mismo plan de tiles/detiling (tiler.cpp, BuildTextureTransfer) que usa la subida
//     normal. No se reimplementa el layout Gnm: el mapeo direccion->mip/capa es el del emulador, que coincide con el
//     de Ghidra (cada mip/capa es un rango contiguo de VirtualMemo).
//  3. Validacion: antes de subir se compara el hash de los bytes de origen (memcpy) con el de los bytes que la subida
//     normal leeria de la RAM para ese rango. Igual: "coincide". Distinto: "difiere" (se guarda formato/tile/offset del
//     primer caso) y ESE trozo NO se sube directo (ruta alternativa: lo sube el camino normal como siempre).
//  4. Ruta alternativa (sin subida directa): sin imagen en ese rango, imagen no apta (render target, storage, profundidad,
//     stencil, metadatos, escrita por la GPU), trozo fuera de los limites de la imagen (alias/vista parcial), imagen
//     de mas de 64 MiB (pools: la subida parcial toma un buffer de la imagen entera), plan de subida no valido, paginas
//     no legibles, cola llena o tiempo agotado.
// No cambia nada mas: ni la proteccion de paginas, ni el hash/refresco del emulador, ni VRAM8_GC, ni la RAM.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "tex-nativa.h"

namespace TexDirecta {
constexpr uint64_t kVmBegin  = 0x1000000000ull;
constexpr uint64_t kVmEnd    = 0x1400000000ull;
constexpr size_t   kMinChunk = 4096;
constexpr size_t   kMaxQueue = 16384;

// Resultado de TextureCache::DirectUploadRange.
enum Result : int {
	kOk           = 0, // subido directo y coincide
	kMismatch     = 1, // los bytes de la RAM ya no son los de origen: no se sube directo
	kNoImage      = 2, // ninguna imagen en ese rango todavia
	kUnsafe       = 3, // imagen no apta (RT, storage, profundidad, stencil, metadatos, escrita por la GPU)
	kOutside      = 4, // el trozo sale de los limites de la imagen (alias / vista parcial)
	kUploadFailed = 5, // plan de subida no valido para esta imagen
	kInvalid      = 6, // rango invalido o paginas no legibles ahora
	kLarge        = 7, // imagen > 64 MiB (pools): no se sube directo
	kClean        = 8, // (2.1d) la imagen ya tiene esos bytes en la GPU (p. ej. la creo la etapa 1 despues del trozo)
	kFmtAlt       = 9, // (2.1d) formato 181 / tile 9: siempre ruta alternativa (los 35-39 "difieren" de la 2.1c)
	kNotPartial   = 10, // (2.1d) la imagen no admite subida parcial (pendiente de su primera subida entera)
	kReasons      = 11
};

inline bool On() {
	static const bool on = [] {
		const char* e = std::getenv("KYTY_TEX_DIRECTA");
		const bool  v = e != nullptr && e[0] == '1';
		if (v) {
			std::printf("[tex-directa] KYTY_TEX_DIRECTA=1: etapa 2.1, los trozos del streaming se suben directos a su "
			            "imagen de la GPU (mip/capa) al copiarlos; la copia en RAM se mantiene\n");
			std::fflush(stdout);
		}
		return v;
	}();
	return on;
}

// Hash rapido de 64 bits (palabras de 8 bytes, mezcla tipo wyhash). Mismo calculo en el memcpy y en la validacion.
inline uint64_t Hash64(const void* data, size_t n) {
	const auto* p = static_cast<const unsigned char*>(data);
	uint64_t    h = 0x9E3779B97F4A7C15ull ^ (n * 0xff51afd7ed558ccdull);
	size_t      i = 0;
	for (; i + 8 <= n; i += 8) {
		uint64_t w;
		std::memcpy(&w, p + i, 8);
		h ^= w * 0x87c37b91114253d5ull;
		h = (h << 31u | h >> 33u) * 0x4cf5ad432745937full;
	}
	uint64_t t = 0;
	for (size_t k = 0; i < n; i++, k++) t |= static_cast<uint64_t>(p[i]) << (8u * (k & 7u));
	h ^= t * 0x87c37b91114253d5ull;
	h ^= h >> 33u;
	h *= 0xff51afd7ed558ccdull;
	h ^= h >> 33u;
	return h;
}

struct Item {
	uint64_t dest;
	uint64_t size;
	uint64_t t_us;      // (2.1d) cuando llego el trozo (memcpy)
	bool     had_image; // (2.1d) la imagen de la GPU ya existia al llegar el trozo
};
inline uint64_t NowUs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
constexpr uint64_t kWaitImageUs = 5000000; // (2.1e) un trozo sin imagen espera hasta 5 s a que la etapa 1 la cree (desde la salida de 0x901330)

inline std::mutex& QMutex() {
	static std::mutex m;
	return m;
}
inline std::vector<Item>& Queue() {
	static std::vector<Item> q;
	return q;
}
inline std::atomic<bool> g_pending {false};
// (2.1f) > 0 mientras el hilo esta dentro de CTexture::SetCurrentStreamingChunk (0x901330, ganchos de entrada/salida en
// tex-anticipada.h): solo esos memcpy son trozos de textura. La 2.1d/e contaba TODO memcpy >= 4 KiB a VirtualMemo
// (mallas, buffers...): 16,7 GB "copiados" frente a los trozos reales, y la cobertura salia falseada.
inline thread_local int tl_in_chunk = 0;
inline thread_local uint64_t tl_chunk_base = 0; // (N2+N3) base del T# de la CTexture de 0x901330 en curso
inline thread_local uint64_t tl_chunk_tex  = 0; // (11-25d) la CTexture de 0x901330 en curso (creacion urgente)
inline std::atomic<uint64_t> c_other {0}, c_other_bytes {0};

// (etapa 2.1c) Mapa de granulos de 64 KiB de VirtualMemo con imagen viva en la textureCache (RegisterImage /
// UnregisterImage lo mantienen). OnMemcpy lo consulta sin bloqueo: si el destino no tiene imagen, sale sin hash ni
// cola (era el 94 % de los candidatos y el coste: 10,9 GB hasheados en el hilo del juego en 2 min, log 221057).
constexpr unsigned kGranShift = 16u;
constexpr size_t   kGranCount = static_cast<size_t>((kVmEnd - kVmBegin) >> kGranShift);
inline std::atomic<uint16_t>* GranMap() {
	static std::atomic<uint16_t>* m = new std::atomic<uint16_t>[kGranCount]();
	return m;
}
inline void MarkImage(uint64_t addr, uint64_t size, int delta) {
	if ((!On() && !TexNativa::On()) || size == 0) return;
	uint64_t b = addr, e = addr + size;
	if (e <= kVmBegin || b >= kVmEnd) return;
	if (b < kVmBegin) b = kVmBegin;
	if (e > kVmEnd) e = kVmEnd;
	auto* m = GranMap();
	for (uint64_t g = (b - kVmBegin) >> kGranShift; g <= ((e - 1 - kVmBegin) >> kGranShift); g++) {
		if (delta > 0) m[g].fetch_add(1, std::memory_order_release);
		else if (m[g].load(std::memory_order_relaxed) > 0) m[g].fetch_sub(1, std::memory_order_release);
	}
}
inline bool HasImage(uint64_t d, uint64_t n) {
	auto* m = GranMap();
	return m[(d - kVmBegin) >> kGranShift].load(std::memory_order_acquire) != 0 ||
	       m[(d + n - 1 - kVmBegin) >> kGranShift].load(std::memory_order_acquire) != 0;
}
inline std::atomic<uint64_t> c_noimg_early {0}, c_noimg_bytes {0}, c_deferred {0};
// (2.1d) Cobertura: trozos con imagen ya creada al llegar / creada despues (esperando <= 3 s) / nunca (camino del draw);
// bytes que acaban en la GPU sin el camino del draw (subidos directos + ya en la GPU).
inline std::atomic<uint64_t> c_arrived {0}, c_arrived_bytes {0}, c_img_before {0}, c_img_after {0}, c_img_never {0},
    c_clean {0}, c_clean_bytes {0}, c_render_us {0}, c_after_wait_us {0};

inline std::atomic<uint64_t> c_seen {0}, c_seen_vm {0}; // (etapa2.1b) memcpy que pasan por el gancho / hacia VirtualMemo
inline std::atomic<uint64_t> c_cand {0}, c_cand_bytes {0}, c_direct {0}, c_direct_bytes {0}, c_match {0}, c_diff {0},
    c_queue_full {0}, c_time {0}, c_last_us {0};
inline std::atomic<uint64_t> c_reason[kReasons] {};
inline std::atomic<bool>     g_first_diff_set {false};
inline uint32_t              g_first_fmt = 0, g_first_tile = 0;
inline uint64_t              g_first_off = 0, g_first_size = 0;

inline void NoteFirstDiff(uint32_t fmt, uint32_t tile, uint64_t off, uint64_t size) {
	bool expected = false;
	if (g_first_diff_set.compare_exchange_strong(expected, true)) {
		g_first_fmt  = fmt;
		g_first_tile = tile;
		g_first_off  = off;
		g_first_size = size;
	}
}

inline void MaybeLog() {
	if (!On()) return;
	const uint64_t now  = NowUs();
	uint64_t       last = c_last_us.load(std::memory_order_relaxed);
	if (now - last < 30000000ull || !c_last_us.compare_exchange_strong(last, now)) return;
	const auto u   = [](const std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.load()); };
	const auto mb  = [](const std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.load() >> 20u); };
	const auto r   = [](int i) { return static_cast<unsigned long long>(c_reason[i].load()); };
	const double arrived = static_cast<double>(c_arrived_bytes.load());
	const double on_gpu  = static_cast<double>(c_direct_bytes.load() + c_clean_bytes.load());
	const double pct     = arrived > 0 ? 100.0 * on_gpu / arrived : 0.0;
	const double pct_img = c_arrived.load() > 0 ? 100.0 * static_cast<double>(c_img_before.load()) /
	                                                  static_cast<double>(c_arrived.load())
	                                            : 0.0;
	// (2.1d) Una linea: cobertura (cuanto de lo que el juego copia acaba en la GPU sin pasar por el draw), si la imagen
	// existia antes del trozo, la ruta alternativa por motivo y el coste en el hilo de render.
	std::printf("[tex-directa] (solo trozos de 0x901330; otras copias a VirtualMemo %llu, %llu MB) cobertura %.1f %% (%llu de %llu MB en la GPU sin el draw: subidos directos %llu trozos %llu MB, "
	            "ya en la GPU %llu trozos %llu MB); imagen creada ANTES del trozo %.1f %% (%llu), creada despues %llu, nunca "
	            "(camino del draw) %llu; ruta alternativa: 181/tile9 %llu, imagen no apta %llu, sin subida parcial %llu, fuera "
	            "%llu, grande %llu, plan %llu, rango %llu, cola llena %llu; aplazados %llu; hilo de render %llu ms; espera media "
	            "de los creados despues %llu ms\n",
	            u(c_other), mb(c_other_bytes), pct, static_cast<unsigned long long>(on_gpu / 1048576.0), mb(c_arrived_bytes), u(c_direct), mb(c_direct_bytes),
	            u(c_clean), mb(c_clean_bytes), pct_img, u(c_img_before), u(c_img_after), u(c_img_never), r(kFmtAlt),
	            r(kUnsafe), r(kNotPartial), r(kOutside), r(kLarge), r(kUploadFailed), r(kInvalid), u(c_queue_full),
	            u(c_deferred), static_cast<unsigned long long>(c_render_us.load() / 1000u),
	            static_cast<unsigned long long>(c_img_after.load() ? c_after_wait_us.load() / c_img_after.load() / 1000u : 0));
	std::fflush(stdout);
}

// Desde LibcInternal::memcpy, ANTES de copiar (los bytes de origen). Devuelve false: la copia a la RAM la hace siempre
// el llamador (en esta etapa la RAM se mantiene).
inline bool OnMemcpy(void* dest, const void* src, size_t n) {
	// (2.1d) Una resta sin signo fuera de VirtualMemo; dentro, sin hash (la 2.1c hasheaba cada trozo en el hilo del juego):
	// solo se anota direccion, tamano, hora y si la imagen ya existia. La subida lee la RAM vigilada, como el camino normal.
	const uint64_t d = reinterpret_cast<uint64_t>(dest);
	if (d - kVmBegin >= kVmEnd - kVmBegin) return false;
	if (!On()) return false;
	if (n < kMinChunk || src == nullptr || d + n > kVmEnd) return false;
	if (tl_in_chunk <= 0) { // no es un trozo de textura (0x901459)
		c_other.fetch_add(1, std::memory_order_relaxed);
		c_other_bytes.fetch_add(n, std::memory_order_relaxed);
		return false;
	}
	c_seen_vm.fetch_add(1, std::memory_order_relaxed);
	const bool img = HasImage(d, n);
	c_arrived.fetch_add(1, std::memory_order_relaxed);
	c_arrived_bytes.fetch_add(n, std::memory_order_relaxed);
	if (img) {
		c_cand.fetch_add(1, std::memory_order_relaxed);
		c_cand_bytes.fetch_add(n, std::memory_order_relaxed);
	} else {
		c_noimg_early.fetch_add(1, std::memory_order_relaxed);
		c_noimg_bytes.fetch_add(n, std::memory_order_relaxed);
	}
	const Item it {d, static_cast<uint64_t>(n), NowUs(), img};
	{
		std::lock_guard<std::mutex> lock(QMutex());
		auto&                       q = Queue();
		if (q.size() >= kMaxQueue) {
			c_queue_full.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		q.push_back(it);
	}
	g_pending.store(true, std::memory_order_release);
	return false;
}

// Hilo de render: hasta `max_items` trozos encolados.
inline void Take(std::vector<Item>& out, size_t max_items) {
	out.clear();
	if (!g_pending.load(std::memory_order_acquire)) return;
	std::lock_guard<std::mutex> lock(QMutex());
	auto&                       q = Queue();
	const size_t                n = q.size() < max_items ? q.size() : max_items;
	out.assign(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(n));
	q.erase(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(n));
	g_pending.store(!q.empty(), std::memory_order_release);
}

// (etapa 2.1c) Lo que no cupo en ~1 ms vuelve al principio de la cola para el siguiente fotograma (antes se descartaba:
// "tiempo 519").
inline void Requeue(const std::vector<Item>& items, size_t from) {
	if (from >= items.size()) return;
	std::lock_guard<std::mutex> lock(QMutex());
	auto&                       q = Queue();
	q.insert(q.begin(), items.begin() + static_cast<std::ptrdiff_t>(from), items.end());
	c_deferred.fetch_add(items.size() - from, std::memory_order_relaxed);
	g_pending.store(true, std::memory_order_release);
}

// (2.1d) Trozos cuya imagen aun no existe: al final de la cola (no bloquean a los demas) hasta kWaitImageUs.
inline void RequeueBack(const std::vector<Item>& items) {
	if (items.empty()) return;
	std::lock_guard<std::mutex> lock(QMutex());
	auto&                       q = Queue();
	for (const auto& it: items) {
		if (q.size() >= kMaxQueue) {
			c_queue_full.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		q.push_back(it);
	}
	g_pending.store(true, std::memory_order_release);
}

inline void Account(int result, uint64_t size) {
	if (result == kClean) {
		c_clean.fetch_add(1, std::memory_order_relaxed);
		c_clean_bytes.fetch_add(size, std::memory_order_relaxed);
		return;
	}
	if (result == kOk) {
		c_direct.fetch_add(1, std::memory_order_relaxed);
		c_direct_bytes.fetch_add(size, std::memory_order_relaxed);
		c_match.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (result == kMismatch) c_diff.fetch_add(1, std::memory_order_relaxed);
	if (result >= 0 && result < kReasons) c_reason[result].fetch_add(1, std::memory_order_relaxed);
}
// (11-25e) Hilo del juego, dentro de 0x901330: pide la imagen de la CTexture en curso al hilo de render y NO espera
// nunca (log 20261125-085841-927: la espera bloqueaba el hilo del juego 7.93 ms de media, max 51 ms, 117 pausas = lag).
// Este trozo va a la RAM al instante; los siguientes trozos de esa CTexture ya encuentran su imagen y van directos.
// Crear la imagen aqui no es posible: ResolveTextureUncached/PrePut registran en el TextureCache (solo hilo de render).
inline void PedirImagen() {
	const uint64_t tex = tl_chunk_tex, base = tl_chunk_base;
	if (tex == 0 || (tex & 7) != 0 || base == 0) return;
	if (TexNativa::IsBadBase(base)) {
		TexNativa::c_urg_bad.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	std::lock_guard<std::mutex> lock(TexNativa::UMutex()); // solo protege la cola (el render lo tiene lo que dura un swap)
	auto& q = TexNativa::UQueue();
	for (const auto& r: q)
		if (r->tex == tex) return; // ya pedida: una peticion por CTexture
	if (q.size() >= 4096) return;
	auto req  = std::make_shared<TexNativa::Urgente>();
	req->tex  = tex;
	req->base = base;
	q.push_back(std::move(req));
	TexNativa::c_urg_n.fetch_add(1, std::memory_order_relaxed);
	TexNativa::g_urgent.store(true, std::memory_order_release);
}

// (N2+N3, KYTY_TEX_NATIVA) Desde CoherentMove (el memcpy del juego), en el hilo del juego. Un trozo de 0x901330 a un mip
// de una textura nativa cuya imagen ya existe: los bytes de origen se guardan para el hilo de render y la copia a la RAM
// emulada NO se hace (devuelve true). El rango queda marcado "solo en la GPU" desde ya: ningun refresco lo sube de la RAM.
inline bool NativeChunkCopy(void* dest, const void* src, size_t n) {
	if (!TexNativa::On() || tl_in_chunk <= 0 || src == nullptr || n < TexNativa::kMinChunk) return false;
	const uint64_t d = reinterpret_cast<uint64_t>(dest);
	if (d - kVmBegin >= kVmEnd - kVmBegin || d + n > kVmEnd) return false;
	// (11-25) Cobertura real: todo trozo de 0x901330 (memcpy 0x901459) cuenta aqui, vaya directo o no.
	TexNativa::c_chunk_all_bytes.fetch_add(n, std::memory_order_relaxed);
	if (!TexNativa::IsNative(tl_chunk_base)) {
		TexNativa::c_chunk_nonative_bytes.fetch_add(n, std::memory_order_relaxed);
		return false;
	}
	// (11-25b) Sin imagen todavia: a la RAM al instante (como en el log 061906). La espera en el anillo (<= 133 ms) se
	// quita: log 071332, de 3179 MB que esperaron solo 416 MB llegaron a su imagen (13 %); 2763 MB acabaron igual en la
	// RAM, pero escritos por el hilo de render tras reintentar en cada drenaje (m_lock + FindImagesInRegion): drenaje
	// 67,6 -> 3050,7 ms/30 s, 2433 -> 16306 "trozos" (reintentos), fps 69 -> 51-55, y AddOwned/RemoveOwned de cada
	// uno (cubrir y volver a vigilar sus paginas) con FAULT 582 -> 1270/s.
	// (11-25d) Sin imagen todavia: creacion urgente de ESTA CTexture en el hilo de render y espera solo a ella (TexNativa::
	// Urgente). Si no se puede crear, o la seguridad anti-bloqueo salta, a la RAM como siempre.
	if (!HasImage(d, n)) {
		// (cov1) Una sola peticion por CTexture mientras siga pendiente (antes una por trozo: 11774 para 4273 CTexture).
		if (!TexNativa::NoteNoImage(tl_chunk_tex, n)) PedirImagen(); // (11-25e) sin espera: este trozo a la RAM
		TexNativa::c_chunk_noimage_bytes.fetch_add(n, std::memory_order_relaxed);
		// (NEGRAS) El juego escribe este trozo en la RAM: si el rango seguia "solo en la GPU" (imagen borrada/aparcada a
		// la espera de recuperarse), fuera ya, bajo PMutex como el anillo lleno; si no, la recuperacion posterior pisaria
		// estos bytes nuevos con los viejos de la VRAM y el refresco nunca los leeria de la RAM.
		if (TexNativa::AnyOwned(d, d + n)) {
			std::lock_guard<std::mutex> lock(TexNativa::PMutex());
			TexNativa::RemoveOwned(d, d + n);
			TexNativa::PQueue().push_back({d, static_cast<uint64_t>(n), TexNativa::kRamWrite, 0, 0});
			TexNativa::g_pending.store(true, std::memory_order_release);
		}
		return false;
	}
	TexNativa::NoteDirect(tl_chunk_tex, n); // (cov1) bytes directos gracias a una creacion urgente
	uint64_t           offset = 0, seq = 0;
	TexNativa::Coste   coste(TexNativa::kCCopy); // (N2+N3f) incluye la espera de PMutex y la copia al anillo
	TexNativa::c_copy_bytes.fetch_add(n, std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex> lock(TexNativa::PMutex());
		if (!TexNativa::RingAlloc(n, offset, seq)) {
			// Anillo lleno o sin crear: a la RAM como siempre (lo copia el llamador), con un aviso al hilo de render para
			// que la imagen lo vuelva a leer de ahi (sus paginas pueden no estar vigiladas).
			// Ya no es "solo en la GPU": fuera antes de que el juego escriba la RAM (bajo PMutex: un soltado de paginas
			// en curso acaba antes, y uno posterior ve el rango sin dueno y no toca lo que el juego escribe).
			TexNativa::RemoveOwned(d, d + n);
			TexNativa::PQueue().push_back({d, static_cast<uint64_t>(n), TexNativa::kRamWrite, 0, 0});
			TexNativa::g_pending.store(true, std::memory_order_release);
			return false;
		}
		// (11-25d, revision 5) "Solo en la GPU" YA, en el mismo candado que la reserva y ANTES de copiar: mientras dura la
		// copia (fuera del candado) NativeGuardDirty no puede refrescar ese rango desde la RAM vieja (la RAM no recibira
		// estos bytes). Antes AddOwned iba despues del memcpy.
		TexNativa::AddOwned(d, d + n);
	}
	// (11-25, congelacion con KYTY_NATIVO_PASE_SOMBRAS=1, blackbox 20261125-024655-421) La copia va FUERA del candado: el
	// origen es memoria del juego y un fallo de pagina (HandleFault -> BufferCache::ReadMemory -> SendCommandSync) espera
	// a la GPU, mientras el hilo de la GPU espera PMutex (NativeRingCollect). El trozo ya esta reservado y sin terminar
	// (RingCollect solo suelta trozos con done).
	std::memcpy(TexNativa::TheRing().base.load(std::memory_order_relaxed) + offset, src, n); // origen -> memoria de subida
	{
		std::lock_guard<std::mutex> lock(TexNativa::PMutex());
		TexNativa::PQueue().push_back({d, static_cast<uint64_t>(n), offset, seq, tl_chunk_base});
	}
	TexNativa::g_pending.store(true, std::memory_order_release);
	return true;
}

} // namespace TexDirecta
