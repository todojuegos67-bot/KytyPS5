#pragma once
// (etapa1, 11-23) KYTY_TEX_ANTICIPADA: crear las texturas del juego cuando el juego las crea, no en su primer draw.
//
// Ghidra (eboot 01.007.000, proyecto r19g):
//  - eboot+0x8fe750: fabrica de CTexture (PS5CTexture.cpp). Devuelve la CTexture (rax). Lee del stream el T# de 32 bytes a
//    this+0xbc (FUN_00813130(stream, this+0xbc, 0x20)), deriva ancho/alto/capas/mips a this+0x264/+0x268/+0x26c/+0x274,
//    reserva la memoria de la textura en VirtualMemo (FUN_007f9f70 op 0/1, o FUN_007f9cf0), lee los mips base del stream
//    (0x8fed14) y al final escribe la base en el propio T#: this+0xbc = base>>8, byte this+0xc0 = base>>40.
//  - eboot+0x901330: CTexture::SetCurrentStreamingChunk(this, chunk). Tabla de trozos en this+0x70 (0x48 bytes: tamano +4,
//    desplazamiento +8, recurso +0x38), base en this+0x98; compromete (0x7f9f70 op 0), copia el trozo (memcpy 0x901459) y
//    ajusta los mips residentes del T# (0x901540, palabra this+0xc8).
//  Ningun lector de CPU del rango conocido (medido: 0 lecturas del juego). El pool de sombras de 360 capas NO pasa por
//  CTexture (no tratado en esta etapa).
//
// Que hace: un gancho a la entrada de 0x8fe750 cambia su direccion de retorno por un trampolin que recibe la CTexture
// devuelta; un gancho a la entrada de 0x901330 recibe la CTexture. Las CTexture van a una cola. El hilo de render, antes de
// cada draw (RenderExecutor::DrawIndex -> TexAnticipadaDrain), resuelve su T# por el MISMO camino que un draw
// (RenderExecutor::ResolveTextureUncached: misma clave, formato, tiling, mips), con un presupuesto de tiempo, asi la imagen
// existe y esta subida antes del primer draw que la usa. La copia en RAM y todo el camino actual se mantienen (etapa 1).
// La cache de texturas no es de varios hilos: por eso se crea en el hilo de render, no en el del juego.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "tex-directa.h"
#include "tex-nativa.h"

#if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
extern "C" __declspec(dllimport) int __stdcall VirtualProtect(void*, unsigned long long, unsigned long, unsigned long*);
extern "C" __declspec(dllimport) void* __stdcall VirtualAlloc(void*, unsigned long long, unsigned long, unsigned long);
extern "C" __declspec(dllimport) int __stdcall FlushInstructionCache(void*, const void*, unsigned long long);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall IsBadReadPtr(const void*, unsigned long long);
#define KYTY_TEXANT_HOOKS 1
#else
#define KYTY_TEXANT_HOOKS 0
#endif

namespace TexAnt {

constexpr uint64_t kEbootBase  = 0x900000000ull;
constexpr uint64_t kCreateRva  = 0x8fe750; // fabrica de CTexture
constexpr uint64_t kChunkRva   = 0x901330; // CTexture::SetCurrentStreamingChunk
constexpr uint64_t kVmBegin    = 0x1000000000ull; // VirtualMemo
constexpr uint64_t kVmEnd      = 0x1400000000ull;

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_ANTICIPADA");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

struct Item {
	uint64_t tex;
	int64_t  t_us;
};
inline std::mutex&        QMutex() { static std::mutex m; return m; }
inline std::vector<Item>& Queue() { static std::vector<Item> q; return q; }
inline std::atomic<bool>  g_pending {false};
inline std::atomic<bool>  g_disabled {false};

// Contadores (los escribe el hilo de render salvo encoladas).
inline std::atomic<uint64_t> c_queued {0}, c_created {0}, c_created_bytes {0}, c_used {0}, c_dup {0}, c_discard {0},
    c_draw_inits {0}, c_draw_init_us {0}, c_already {0}, c_bad {0},
    c_calls_create {0}, c_calls_chunk {0}, // (etapa1b) llamadas a cada gancho
    c_render_us {0}, c_skip_same {0}, c_chunk_enq {0}; // (2.1e) encoladas a la salida de 0x901330 // (etapa1d) coste en el render y CTexture ya creadas con el mismo T#


// (etapa2, 11-23) KYTY_TEX_SOLO_GPU (encendido por defecto; =0 lo apaga): las texturas de VirtualMemo que la etapa 1 creo de
// antemano y que el draw USO con la misma clave (misma imagen) se quedan solo en la GPU: su copia en RAM sale por la ruta de
// offload nativa ya existente (TextureCache, TextureOffload: solo tras comprobar la subida, y con toda lectura posterior de
// la CPU o del emulador servida descargando de la GPU: "re-descargas"), sin el minimo de 1-4 MiB que dejaba fuera a las del
// streaming (0,7-1,4 MiB). Ghidra 01.007.000: la CPU del juego no lee VirtualMemo (escriben 0x8fe750/0x8fed14 y el memcpy
// 0x901459; 0x901330 libera trozos con op 2 = unmap, que la ruta de offload suelta sin descargar). Las que no cumplen
// (no creadas de antemano, descartadas, no usadas) siguen el camino de siempre = ruta alternativa con RAM.
inline bool SoloGpu() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_SOLO_GPU");
		const bool  r = (v != nullptr && v[0] == '1'); // (etapa2c) APAGADO por defecto: ruta parcial con re-descarga desde la GPU, rechazada
		std::printf("[solo-gpu] %s (KYTY_TEX_SOLO_GPU=0 lo apaga)\n", r ? "encendido: texturas creadas de antemano y usadas por el draw, solo en la GPU" : "apagado");
		return r;
	}();
	return on && On();
}
inline std::mutex& UsedMutex() { static std::mutex m; return m; }
inline std::vector<uint64_t>& UsedSorted() { static std::vector<uint64_t> v; return v; }
inline std::atomic<uint64_t> c_solo_marked {0}, c_solo_checked {0}, c_solo_yes {0};
// Lo llama el draw cuando encuentra la imagen creada de antemano (descriptors.cpp, c_used).
inline void MarkUsed(uint64_t addr) {
	if (addr < kVmBegin || addr >= kVmEnd || !SoloGpu()) return; // (etapa2c) sin marcas si esta apagado
	std::lock_guard<std::mutex> lock(UsedMutex());
	auto& v  = UsedSorted();
	auto  it = std::lower_bound(v.begin(), v.end(), addr);
	if (it != v.end() && *it == addr) return;
	v.insert(it, addr);
	c_solo_marked.fetch_add(1, std::memory_order_relaxed);
}
// La consulta TextureCache (paso de offload): la textura empieza en una direccion marcada.
inline bool IsUsed(uint64_t addr) {
	if (addr < kVmBegin || addr >= kVmEnd || !SoloGpu()) return false; // (etapa2c) tex_min normal si esta apagado
	c_solo_checked.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard<std::mutex> lock(UsedMutex());
	const auto& v = UsedSorted();
	const bool  r = std::binary_search(v.begin(), v.end(), addr);
	if (r) c_solo_yes.fetch_add(1, std::memory_order_relaxed);
	return r;
}

inline int64_t NowUs() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void Enqueue(uint64_t tex) {
	if (tex < 0x10000ull || tex >= 0x0000800000000000ull || (tex & 7) != 0 || g_disabled.load(std::memory_order_relaxed)) return;
	std::lock_guard<std::mutex> lock(QMutex());
	auto& q = Queue();
	if (q.size() >= 65536) return;
	for (size_t i = q.size() > 64 ? q.size() - 64 : 0; i < q.size(); i++)
		if (q[i].tex == tex) { q[i].t_us = NowUs(); return; }
	q.push_back({tex, NowUs()});
	c_queued.fetch_add(1, std::memory_order_relaxed);
	g_pending.store(true, std::memory_order_release);
}

// Lee el T# (8 dwords) de una CTexture si la memoria es legible y su base cae en VirtualMemo.
inline bool ReadTSharp(uint64_t tex, uint32_t out[8]) {
#if KYTY_TEXANT_HOOKS
	if (IsBadReadPtr(reinterpret_cast<const void*>(tex + 0xbc), 0x20) != 0) return false;
#endif
	std::memcpy(out, reinterpret_cast<const void*>(tex + 0xbc), 32);
	const uint64_t base = ((static_cast<uint64_t>(out[1] & 0xffu) << 32u) | out[0]) << 8u;
	return base >= kVmBegin && base < kVmEnd;
}

#if KYTY_TEXANT_HOOKS
inline std::vector<uint64_t>& RetStack() {
	static thread_local std::vector<uint64_t> s;
	return s;
}
inline uint64_t g_ret_tramp = 0;
inline uint64_t g_chunk_tramp = 0; // (2.1e) retorno de 0x901330
// (2.1e) Pila por hilo de (retorno original, CTexture) de 0x901330.
inline std::vector<std::pair<uint64_t, uint64_t>>& ChunkStack() {
	static thread_local std::vector<std::pair<uint64_t, uint64_t>> s;
	return s;
}
// (2.1f) Crear la imagen a la salida de 0x901330: APAGADO (la 2.1e creo 9 GB en 6 min, CPU 8,6 -> 10,3 ms, sin subir la
// cobertura). KYTY_TEX_CHUNK_CREATE=1 lo vuelve a encender. El gancho de salida sigue para marcar los trozos reales.
inline bool ChunkCreateEnq() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_CHUNK_CREATE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
inline bool ChunkCreate() { // (2.1e) ganchos de salida con la subida directa o (N2+N3) las texturas nativas
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_DIRECTA");
		return (v != nullptr && v[0] == '1') || TexNativa::On();
	}();
	return on;
}

// Retorno de 0x8fe750: rax = la CTexture. Devuelve la direccion de retorno original.
__attribute__((sysv_abi, noinline)) inline uint64_t OnCreateReturn(uint64_t tex) {
	auto&          s   = RetStack();
	const uint64_t ret = s.back();
	s.pop_back();
	if (tex != 0) {
		// (N1, KYTY_TEX_NATIVA) La CTexture queda registrada por la base de su T# (escrita al final de 0x8fe750).
		// (11-25c) Y si es nativa, su imagen se crea YA (cola propia, TexNativa::EnqueueCreate), no tras la cola de TexAnt.
		uint32_t t[8];
		bool     native = false;
		if (TexNativa::On() && ReadTSharp(tex, t)) {
			const uint64_t base = ((static_cast<uint64_t>(t[1] & 0xffu) << 32u) | t[0]) << 8u;
			TexNativa::Register(base);
			native = TexNativa::IsNative(base);
		}
		if (native && (tex & 7) == 0) TexNativa::EnqueueCreate(tex);
		else Enqueue(tex);
	}
	return ret;
}
// Entrada de 0x8fe750: guarda el retorno y lo cambia por el trampolin (hueco de registros: retorno en +0x248).
__attribute__((sysv_abi, noinline)) inline void OnCreateEntry(uint64_t, uint64_t, uint64_t, uint64_t ret, uint64_t regs) {
	c_calls_create.fetch_add(1, std::memory_order_relaxed);
	RetStack().push_back(ret);
	*reinterpret_cast<uint64_t*>(regs + 0x248) = g_ret_tramp;
}
// Entrada de 0x901330: rdi = la CTexture.
// (etapa1d) En la ENTRADA no se encola: el T# aun tiene los mips de antes (los cambia 0x901540 dentro); lo creado no
// coincidia con el draw (log 210923: 4751 duplicadas). (2.1e, con KYTY_TEX_DIRECTA=1) Se encola a la SALIDA: el trozo ya
// esta copiado (memcpy 0x901459) y el T# ya tiene sus mips nuevos, los que usara el draw. La imagen se crea en el hilo de
// render y el trozo, que espera en la cola de tex-directa, la encuentra ya creada.
__attribute__((sysv_abi, noinline)) inline uint64_t OnChunkReturn(uint64_t) {
	auto&          s  = ChunkStack();
	const auto     e  = s.back();
	s.pop_back();
	TexDirecta::tl_in_chunk--;
	TexDirecta::tl_chunk_base = 0;
	TexDirecta::tl_chunk_tex  = 0; // (11-25d)
	if (e.second != 0 && ChunkCreateEnq()) {
		Enqueue(e.second);
		c_chunk_enq.fetch_add(1, std::memory_order_relaxed);
	}
	return e.first;
}
__attribute__((sysv_abi, noinline)) inline void OnChunkEntry(uint64_t tex, uint64_t, uint64_t, uint64_t ret, uint64_t regs) {
	c_calls_chunk.fetch_add(1, std::memory_order_relaxed);
	if (!ChunkCreate() || g_chunk_tramp == 0) return;
	ChunkStack().push_back({ret, tex});
	TexDirecta::tl_in_chunk++;
	TexDirecta::tl_chunk_tex = tex; // (11-25d) para la creacion urgente de su imagen si un trozo llega antes que ella
	uint32_t t[8];
	TexDirecta::tl_chunk_base =
	    TexNativa::On() && ReadTSharp(tex, t) ? ((static_cast<uint64_t>(t[1] & 0xffu) << 32u) | t[0]) << 8u : 0;
	*reinterpret_cast<uint64_t*>(regs + 0x248) = g_chunk_tramp;
}

inline uint8_t* Exec(const std::vector<uint8_t>& c) {
	auto* p = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, 0x3000, 0x40));
	if (p == nullptr) return nullptr;
	std::memcpy(p, c.data(), c.size());
	FlushInstructionCache(GetCurrentProcess(), p, c.size());
	return p;
}
inline void Imm64(std::vector<uint8_t>& c, uint64_t v) {
	for (int i = 0; i < 8; i++) c.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

// Igual que el gancho de entrada del randomizer (rando-msb.h InstallStub): guarda registros y estado FP, llama a
// fn(rdi, rsi, rdx, retorno, hueco) con ABI sysv, restaura, ejecuta el prologo movido (12 bytes) y salta a entrada+12.
// (tex2) fast_flag != null: antes de guardar nada, si *fast_flag (un byte, std::atomic<bool>) es 0 se salta directo al
// cuerpo original (mov rax, flag; cmp byte [rax], 0; je original). rax es libre a la entrada (no es argumento).
inline bool InstallEntry(uint64_t rva, uint64_t fn, const char* what, const void* fast_flag = nullptr) {
	static const uint8_t expect[12] {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54};
	const uint64_t       entry = kEbootBase + rva;
	auto*                at    = reinterpret_cast<uint8_t*>(entry);
	if (IsBadReadPtr(at, 12) != 0 || std::memcmp(at, expect, sizeof(expect)) != 0) {
		// (etapa1b) Los bytes reales, para saber si es otro build o si otro gancho del emulador ya parcheo la entrada.
		char hex[3 * 16 + 1] = {};
		if (IsBadReadPtr(at, 16) == 0)
			for (int i = 0; i < 16; i++) std::snprintf(hex + 3 * i, 4, "%02x ", at[i]);
		std::printf("[anticipada] %s: eboot+0x%llx no es la de 01.007.000 (bytes: %s); no se instala\n", what,
		            static_cast<unsigned long long>(rva), hex);
		return false;
	}
	std::vector<uint8_t> c;
	auto b = [&](std::initializer_list<uint8_t> x) { c.insert(c.end(), x.begin(), x.end()); };
	if (fast_flag != nullptr) {
		b({0x48, 0xb8}); // mov rax, flag
		Imm64(c, reinterpret_cast<uint64_t>(fast_flag));
		b({0x80, 0x38, 0x00}); // cmp byte [rax], 0
		b({0x75, 0x18});       // jne +24 (camino lento: 12 + 10 + 2 bytes de abajo)
		c.insert(c.end(), expect, expect + sizeof(expect)); // 12: prologo original
		b({0x48, 0xb8});                                    // mov rax, entrada+12
		Imm64(c, entry + sizeof(expect));
		b({0xff, 0xe0}); // jmp rax
	}
	b({0x9c, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53});
	b({0x48, 0x81, 0xec, 0x00, 0x02, 0x00, 0x00});
	b({0x48, 0x0f, 0xae, 0x04, 0x24});
	b({0x48, 0x8b, 0x8c, 0x24, 0x48, 0x02, 0x00, 0x00});
	b({0x49, 0x89, 0xe0});
	b({0x48, 0xb8});
	Imm64(c, fn);
	b({0xff, 0xd0});
	b({0x48, 0x0f, 0xae, 0x0c, 0x24});
	b({0x48, 0x81, 0xc4, 0x00, 0x02, 0x00, 0x00});
	b({0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x9d});
	c.insert(c.end(), expect, expect + sizeof(expect));
	b({0x48, 0xb8});
	Imm64(c, entry + sizeof(expect));
	b({0xff, 0xe0});
	uint8_t* stub = Exec(c);
	if (stub == nullptr) return false;
	uint8_t patch[12];
	patch[0] = 0x48;
	patch[1] = 0xb8;
	const uint64_t target = reinterpret_cast<uint64_t>(stub);
	std::memcpy(patch + 2, &target, 8);
	patch[10] = 0xff;
	patch[11] = 0xe0;
	unsigned long old = 0;
	if (VirtualProtect(at, sizeof(patch), 0x40, &old) == 0) return false;
	std::memcpy(at, patch, sizeof(patch));
	unsigned long dummy = 0;
	VirtualProtect(at, sizeof(patch), old, &dummy);
	FlushInstructionCache(GetCurrentProcess(), at, sizeof(patch));
	std::printf("[anticipada] %s instalado en eboot+0x%llx\n", what, static_cast<unsigned long long>(rva));
	return true;
}

// Trampolin de retorno de 0x8fe750: guarda rax/rdx y el estado FP, llama a OnCreateReturn(rax) (sysv), salta al retorno
// original que devuelve. Los demas registros volatiles no hace falta guardarlos: el llamador ya los da por perdidos.
inline uint64_t BuildTrampolineFor(uint64_t fn) {
	std::vector<uint8_t> c;
	auto b = [&](std::initializer_list<uint8_t> x) { c.insert(c.end(), x.begin(), x.end()); };
	b({0x50, 0x52});                                     // push rax; push rdx
	b({0x48, 0x81, 0xec, 0x00, 0x02, 0x00, 0x00});       // sub rsp, 0x200
	b({0x48, 0x0f, 0xae, 0x04, 0x24});                   // fxsave64 [rsp]
	b({0x48, 0x8b, 0xbc, 0x24, 0x08, 0x02, 0x00, 0x00}); // mov rdi, [rsp+0x208] (rax)
	b({0x48, 0xb8});                                     // mov rax, OnCreateReturn
	Imm64(c, fn);
	b({0xff, 0xd0});                                     // call rax
	b({0x49, 0x89, 0xc3});                               // mov r11, rax (retorno original)
	b({0x48, 0x0f, 0xae, 0x0c, 0x24});                   // fxrstor64 [rsp]
	b({0x48, 0x81, 0xc4, 0x00, 0x02, 0x00, 0x00});       // add rsp, 0x200
	b({0x5a, 0x58});                                     // pop rdx; pop rax
	b({0x41, 0xff, 0xe3});                               // jmp r11
	uint8_t* p = Exec(c);
	return reinterpret_cast<uint64_t>(p);
}
inline bool BuildReturnTrampoline() {
	g_ret_tramp = BuildTrampolineFor(reinterpret_cast<uint64_t>(&OnCreateReturn));
	if (ChunkCreate()) g_chunk_tramp = BuildTrampolineFor(reinterpret_cast<uint64_t>(&OnChunkReturn));
	return g_ret_tramp != 0;
}
#endif

#if KYTY_TEXANT_HOOKS
// (tex2) free del monton del juego (0x802f90; rdi = monton, rsi = ptr; cabecera en ptr-16, tamano *(ptr-8) & ~7,
// estilo dlmalloc: Ghidra, FUN_00802f90 y FUN_008032b0 que fusiona con los vecinos). El juego escribe en el bloque
// liberado (nodos del arbol libre) y en la cabecera del siguiente. Si algo de [ptr-16, ptr+tamano+16) es "solo en la
// GPU", sale del registro ANTES (y de lo soltado), bajo PMutex: NativeDropRam no puede resetearlo despues ni a medias.
// Via rapida en el propio stub: sin rangos "solo en la GPU" (g_any_owned = 0) no se guarda nada ni se llama aqui.
// malloc (0x802c40) no necesita gancho: solo devuelve bloques que estaban libres (ya fuera del registro por este gancho)
// o memoria nueva del segmento (nunca registrada sin pasar por la textura que la usa).
constexpr uint64_t kFreeRva = 0x802f90;
__attribute__((sysv_abi, noinline)) inline void OnFreeEntry(uint64_t, uint64_t ptr, uint64_t, uint64_t, uint64_t) {
	if (ptr < 16) return;
	const uint64_t size = *reinterpret_cast<const uint64_t*>(ptr - 8) & ~uint64_t {7};
	const uint64_t b = ptr - 16, e = ptr + size + 16;
	if (!TexNativa::AnyOwned(b, e)) return;
	std::lock_guard<std::mutex> lock(TexNativa::PMutex());
	TexNativa::RemoveOwned(b, e);
	TexNativa::c_free_hits.fetch_add(1, std::memory_order_relaxed);
	TexNativa::c_free_bytes.fetch_add(size, std::memory_order_relaxed);
}
#endif
inline void InstallFreeHookOnce() {
	static std::once_flag once;
	std::call_once(once, [] {
#if KYTY_TEXANT_HOOKS
		if (!TexNativa::On()) return;
		const bool ok = InstallEntry(kFreeRva, reinterpret_cast<uint64_t>(&OnFreeEntry), "gancho de free del monton (tex2)",
		                             &TexNativa::g_any_owned);
		TexNativa::g_free_hook_ok.store(ok, std::memory_order_release);
		if (!ok) std::printf("[tex-nativa] sin gancho de free: KYTY_TEX_NATIVA_LIBERAR no suelta nada\n");
		std::fflush(stdout);
#endif
	});
}

inline void InstallOnce() {
	InstallFreeHookOnce(); // (tex2) aunque KYTY_TEX_ANTICIPADA=0
	static std::once_flag once;
	std::call_once(once, [] {
		if (!On()) {
			std::printf("[anticipada] KYTY_TEX_ANTICIPADA=0: apagado\n");
			std::fflush(stdout);
			return;
		}
#if KYTY_TEXANT_HOOKS
		// (etapa1b) Antes un fallo del gancho de 0x901330 desactivaba TODO (g_disabled) y la cola descartaba tambien lo
		// del gancho de creacion: "encoladas 0 [desactivada]". Ahora solo desactiva si falla el gancho de creacion; el de
		// los trozos es un extra (cambios de mips).
		const bool ok_create = BuildReturnTrampoline() &&
		    InstallEntry(kCreateRva, reinterpret_cast<uint64_t>(&OnCreateEntry), "gancho de creacion de CTexture");
		InstallEntry(kChunkRva, reinterpret_cast<uint64_t>(&OnChunkEntry), "gancho de SetCurrentStreamingChunk");
		if (!ok_create) g_disabled.store(true);
#endif
		std::fflush(stdout);
	});
}

inline void MaybeLog() {
	static std::atomic<int64_t> next {0};
	const int64_t               now = NowUs() / 1000000;
	int64_t                     due = next.load(std::memory_order_relaxed);
	if (due == 0) {
		next.compare_exchange_strong(due, now + 30);
		return;
	}
	if (now < due || !next.compare_exchange_strong(due, now + 30)) return;
	std::printf("[anticipada] creadas %llu (%.0f MB), usadas por el draw %llu, duplicadas %llu, descartadas por formato/mips "
	            "%llu, creaciones en el draw que quedan %llu (%.0f ms); encoladas %llu, no legibles %llu; llamadas a 0x8fe750 "
	            "%llu, a 0x901330 %llu (encoladas a su salida %llu); ya creadas con el mismo T# %llu; ms en el render %.0f%s\n",
	            static_cast<unsigned long long>(c_created.load()), static_cast<double>(c_created_bytes.load()) / 1048576.0,
	            static_cast<unsigned long long>(c_used.load()), static_cast<unsigned long long>(c_dup.load()),
	            static_cast<unsigned long long>(c_discard.load()), static_cast<unsigned long long>(c_draw_inits.load()),
	            static_cast<double>(c_draw_init_us.load()) / 1000.0, static_cast<unsigned long long>(c_queued.load()),
	            static_cast<unsigned long long>(c_bad.load()), static_cast<unsigned long long>(c_calls_create.load()),
	            static_cast<unsigned long long>(c_calls_chunk.load()), static_cast<unsigned long long>(c_chunk_enq.load()),
	            static_cast<unsigned long long>(c_skip_same.load()),
	            static_cast<double>(c_render_us.load()) / 1000.0, g_disabled.load() ? " [desactivada]" : "");
	std::printf("[solo-gpu] texturas marcadas (creadas de antemano y usadas por el draw) %llu; consultas del offload %llu, aceptadas %llu (RAM liberada y re-descargas: linea \"Native texture offload (30 s\")%s\n",
	            static_cast<unsigned long long>(c_solo_marked.load()), static_cast<unsigned long long>(c_solo_checked.load()),
	            static_cast<unsigned long long>(c_solo_yes.load()), SoloGpu() ? "" : " [apagado]");
	std::fflush(stdout);
}

} // namespace TexAnt
