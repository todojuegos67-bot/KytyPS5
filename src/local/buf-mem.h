#pragma once
// (BUF-MEM, 11-25) Memoria de bufferes y geometria como en un juego de PC: cada bufer del juego vive en UN sitio.
//   - Bufer de la cache usado por la GPU en los ultimos ColdFrames fotogramas: VRAM (pool de bufferes del juego).
//   - Sin uso de la GPU en ColdFrames fotogramas: memoria del sistema visible por la GPU (copia GPU->GPU, la de VRAM fuera;
//     nunca dos copias en la GPU). Vuelve a VRAM al usarse (BufferCache::BufRamPass, ya existente; antes solo con el
//     nivel 2 de vram-tope.h, que en el log 20261125-133607 nunca se activo: "en RAM 0 MiB" todo el log).
//   - Tras sacar >= 32 MiB se fuerza una compactacion del pool (vmaBeginDefragmentation, BufferCache::Defragment): los
//     bloques que quedan vacios los libera VMA (vkFreeMemory): VRAM devuelta al driver de verdad.
//   - Mallas (malla-nativa.h): deduplicadas por hash de contenido de sus paginas (mismo VkBuffer para dos bloques iguales,
//     con cuenta de referencias) y sacadas de VRAM tras StreamOutSeconds sin uso (sus bytes vuelven a la RAM del juego,
//     copia unica; el siguiente uso las prepara otra vez en el hilo Preparer, asincrono: ese uso va por el camino normal).
// KYTY_BUF_MEM=0 lo apaga (encendido por defecto).
//
// Gestor central (src/local/gestor-memoria.h, de otro agente): cuando exista, en BufMem::Enlazar() se rellenan los
// punteros de abajo con su API (Registrar/Tocar/Liberar/PedirEspacio/Presion). Mientras no exista, el adaptador minimo de
// este archivo hace lo mismo localmente: Presion(bytes) pide a la cache que saque esos bytes a RAM en la siguiente pasada.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>

#if defined(__has_include)
#if __has_include("gestor-memoria.h")
#include "gestor-memoria.h"
#define KYTY_BUF_MEM_GESTOR 1
#endif
#endif

namespace BufMem {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BUF_MEM");
		return v == nullptr || *v != '0';
	}();
	return on;
}

constexpr uint64_t ColdFrames       = 600;                  // (~10 s a 60 fps) sin uso de la GPU: a RAM
constexpr uint64_t RateBytes        = uint64_t {128} << 20u; // por segundo, cada sentido
constexpr uint64_t MaxRamBytes      = uint64_t {3} << 30u;   // bufferes de la cache en RAM como mucho
constexpr uint64_t DefragAfterBytes = uint64_t {32} << 20u;  // sacados que fuerzan compactacion
constexpr int      StreamOutSeconds = 120;                   // mallas sin uso: fuera de VRAM

// Contadores (bytes).
inline std::atomic<uint64_t> g_out {0}, g_back {0};            // cache: VRAM->RAM, RAM->VRAM (acumulado)
inline std::atomic<uint64_t> g_out_since_defrag {0};
inline std::atomic<uint64_t> g_returned {0};                   // bloques del pool devueltos al driver (acumulado)
inline std::atomic<uint64_t> g_dedup_bytes {0}, g_dedup_n {0}; // mallas: VRAM ahorrada ahora por compartir
inline std::atomic<uint64_t> g_mesh_out {0}, g_mesh_out_n {0}; // mallas sacadas de VRAM (acumulado)
inline std::atomic<uint64_t> g_pressure_want {0};              // pedido por el gestor (bytes a sacar)

// ---- Adaptador del gestor central ----
struct Gestor {
	std::function<void(const char* tipo, uint64_t id, uint64_t bytes, bool vram)> registrar;
	std::function<void(uint64_t id)>                                              tocar;
	std::function<void(uint64_t id)>                                              liberar;
	std::function<bool(uint64_t bytes)>                                           pedir_espacio;
};
inline Gestor& G() {
	static Gestor g;
	return g;
}
// El gestor llama a esto cuando necesita VRAM: la cache saca bufferes frios a RAM hasta `bytes`.
inline void Presion(uint64_t bytes) {
	uint64_t cur = g_pressure_want.load(std::memory_order_relaxed);
	while (cur < (uint64_t {1} << 30u) &&
	       !g_pressure_want.compare_exchange_weak(cur, cur + bytes, std::memory_order_relaxed)) {
	}
}
inline uint64_t TomarPresion() { return g_pressure_want.exchange(0, std::memory_order_relaxed); }

// Enlace con gestor-memoria.h: aqui se conectan sus llamadas cuando exista (ver comentario de cabecera). Sin el, nada.
// (M7) Conectado: cada bufer de la cache se registra en BufferCache (ChangeRegister: TipoBuffer, Vram o Ram segun donde
// vive; Tocar en TouchBuffer; CambiarLugar al migrar en BufRamPass; Liberar al quitarlo). Aqui se registra Presion como
// callback de presion de VRAM: el gestor la llama con los bytes que faltan y la siguiente BufRamPass saca frios a RAM.
inline void Enlazar() {
#if defined(KYTY_BUF_MEM_GESTOR)
	static std::atomic<bool> hecho {false};
	if (hecho.exchange(true)) return;
	GestorMem::RegistrarPresion(GestorMem::TipoBuffer, GestorMem::Vram, &Presion);
#endif
}
inline void Registrar(const char* tipo, uint64_t id, uint64_t bytes, bool vram) {
	if (G().registrar) G().registrar(tipo, id, bytes, vram);
}
inline void Liberar(uint64_t id) {
	if (G().liberar) G().liberar(id);
}

// ---- Linea [buf-mem] (cada 5 s, la escribe BufferCache) ----
struct Censo {
	uint64_t vram_static = 0, vram_gpu = 0, vram_bda = 0; // bufferes de la cache en VRAM
	uint64_t ram_static = 0, ram_gpu = 0, ram_bda = 0;    // en memoria del sistema visible por la GPU
	uint64_t mesh_vram = 0, mesh_ram = 0;                 // mallas nativas
	uint64_t pool_blocks = 0, pool_used = 0;
};
inline void Log(const Censo& c) {
	static uint64_t last_out = 0, last_back = 0, last_ret = 0;
	const auto      mib      = [](uint64_t b) { return static_cast<unsigned long long>(b >> 20u); };
	const uint64_t  out = g_out.load(), back = g_back.load(), ret = g_returned.load();
	std::printf("[buf-mem] 5 s: VRAM bufferes %llu MiB (estaticos %llu, escritos GPU %llu, bda %llu) + mallas %llu | RAM-GPU "
	            "bufferes %llu MiB (estaticos %llu, escritos GPU %llu, bda %llu) + mallas %llu | pool %llu MiB (hueco %llu) | "
	            "movidos a RAM %llu MiB, a VRAM %llu MiB (total %llu/%llu) | devueltos al driver %llu MiB (total %llu) | "
	            "dedup mallas %llu MiB (%llu) | mallas fuera de VRAM %llu MiB (%llu)\n",
	            mib(c.vram_static + c.vram_gpu + c.vram_bda), mib(c.vram_static), mib(c.vram_gpu), mib(c.vram_bda),
	            mib(c.mesh_vram), mib(c.ram_static + c.ram_gpu + c.ram_bda), mib(c.ram_static), mib(c.ram_gpu),
	            mib(c.ram_bda), mib(c.mesh_ram), mib(c.pool_blocks), mib(c.pool_blocks - std::min(c.pool_blocks, c.pool_used)),
	            mib(out - last_out), mib(back - last_back), mib(out), mib(back), mib(ret - last_ret), mib(ret),
	            mib(g_dedup_bytes.load()), static_cast<unsigned long long>(g_dedup_n.load()), mib(g_mesh_out.load()),
	            static_cast<unsigned long long>(g_mesh_out_n.load()));
	std::fflush(stdout);
	last_out = out, last_back = back, last_ret = ret;
}

} // namespace BufMem
