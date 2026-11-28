#pragma once
// (faltas-emu) KYTY_FALTAS_EMU: sin variable o 0 apagado (como antes); 1 encendido; 2 A/B dentro de la partida (cambia
// cada ventana de 5 s de KYTY_TECHO, con los demas interruptores).
// Log T72 ([techo-faltas] '| quien-?', ancla Report=0x140D24960, exe en 0x140000000): faltas de pagina del hilo de envio
// cuyo rip es del propio emulador (0x1419a9aca, 6/fot, escrituras <=64 B) o de memcpy/memmove de una DLL del sistema
// (copias <=4 KiB, ~1 ms/fot), con 0x2f65470 en la pila: no es codigo, es el segmento RW del eboot (LOAD 0x26dc000..,
// Ghidra: sin funcion ni xrefs; 'bzQExy..' basura de datos) = el CommandBuffer del juego (`this` de AllocateDW, que vive
// en .bss del eboot). Es decir: las funciones Agc del emulador (sceAgcDcb*/Cb*: AgcCbNop memset, SET_*_REG memcpy de
// valores, cmd[0..n] = ...) escriben los paquetes en el bufer de comandos del juego, que el seguimiento tiene protegido,
// y se provocan su propia falta en cada pagina nueva.
// Arreglo nativo: CommandBuffer::AllocateDW, justo antes de devolver el hueco que la funcion va a escribir, abre las
// paginas de [ptr, ptr + bytes) que el seguimiento tiene protegidas llamando EL MISMO manejador que la falta
// (GpuResourceManager::HandleFault(Write, direccion), mismo hilo): la pagina queda marcada como escrita por la CPU igual
// que ahora (buffers/imagenes invalidados, lectura de vuelta si era de la GPU, proteccion quitada), solo que sin la
// excepcion (trampa del kernel + KiUserExceptionDispatcher + VEH + NtContinue). Pagina por pagina, solo las que su
// estado dice protegidas (PageManager::WatchedSpan, sin llamada al sistema); por hilo se recuerda la ultima pagina
// abierta para no mirarla otra vez en cada paquete. Si otra cosa la protege despues, la escritura falla y va por el
// camino de siempre. Con KYTY_NATIVE_MEMORY sin proteccion de escritura no hace nada (alli no hay faltas de escritura).
// Seguro por construccion: es la misma funcion, en el mismo hilo, sobre la misma pagina, un instante antes; lo unico
// distinto es que si la funcion Agc no llegara a escribir algun byte del hueco la pagina se da igualmente por escrita
// (invalidacion de mas: conservador).
#include "local/techo.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::LibKernel::Memory {
// memory.cpp: las paginas de [vaddr, vaddr + size) con proteccion del seguimiento, abiertas con HandleFault(Write).
// Devuelve cuantas.
uint32_t OpenWatchedForEmulatorWrite(uint64_t vaddr, uint64_t size) noexcept;
} // namespace Libs::LibKernel::Memory

namespace FaltasEmu {

inline std::atomic<int>      g_mode {-1};
inline std::atomic<bool>     g_now {false};
inline std::atomic<uint64_t> g_pages {0}, g_cyc {0}, g_calls {0};

[[nodiscard]] inline int Mode() noexcept {
	int m = g_mode.load(std::memory_order_relaxed);
	if (m >= 0) return m;
	const char* v = std::getenv("KYTY_FALTAS_EMU");
	m             = (v != nullptr && (v[0] == '1' || v[0] == '2')) ? v[0] - '0' : 0;
	g_now.store(m == 1, std::memory_order_relaxed);
	g_mode.store(m, std::memory_order_relaxed);
	std::printf("[faltas-emu] KYTY_FALTAS_EMU=%d: %s\n", m,
	            m == 1   ? "encendido (Agc abre antes sus paginas protegidas del bufer de comandos: sin falta propia)"
	            : m == 2 ? "A/B cada 5 s"
	                     : "apagado (como antes)");
	std::fflush(stdout);
	return m;
}

inline thread_local uint64_t t_last_page = 0;

// CommandBuffer::AllocateDW: [ptr, ptr + bytes) va a escribirlo la funcion Agc del emulador.
inline void BeforeWrite(const void* ptr, uint64_t bytes) noexcept {
	if (Mode() == 0 || !g_now.load(std::memory_order_relaxed) || bytes == 0) return;
	const uint64_t a = reinterpret_cast<uint64_t>(ptr), e = a + bytes;
	const uint64_t first = a & ~uint64_t {4095}, last = (e - 1) & ~uint64_t {4095};
	if (first == last && first == t_last_page) return; // ya abierta por este hilo (si se protegio otra vez: falta normal)
	const bool     t   = Techo::On();
	const uint64_t t0  = t ? __rdtsc() : 0;
	const uint32_t n   = Libs::LibKernel::Memory::OpenWatchedForEmulatorWrite(a, bytes);
	t_last_page        = last;
	if (n != 0) g_pages.fetch_add(n, std::memory_order_relaxed);
	if (t) {
		g_cyc.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		g_calls.fetch_add(1, std::memory_order_relaxed);
	}
}

// Desde TechoFaltas::Report (cada 5 s). `faltas_envio_ms`: las faltas del hilo de envio de esta ventana (ms/fot).
inline void Report(double frames, double ms_cyc, double faltas_envio_ms) noexcept {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const uint64_t p = g_pages.exchange(0), c = g_cyc.exchange(0), k = g_calls.exchange(0);
	const double   ms = static_cast<double>(c) * ms_cyc / frames, total = faltas_envio_ms + ms;
	const bool     was = g_now.load(std::memory_order_relaxed);
	std::printf("[faltas-emu] 5 s (%s): faltas evitadas %.1f/fot (paginas abiertas antes por Agc), su manejador %.3f ms/fot en %.0f"
	            " miradas/fot; faltas del hilo de envio %.2f ms/fot; envio total (faltas + abiertas) %.2f ms/fot",
	            was ? "B nuevo" : "A antes", static_cast<double>(p) / frames, ms, static_cast<double>(k) / frames, faltas_envio_ms, total);
	if (m == 2) {
		static double sum[2] {}, cnt[2] {}, ev[2] {};
		sum[was ? 1 : 0] += total, cnt[was ? 1 : 0] += 1, ev[was ? 1 : 0] += static_cast<double>(p) / frames;
		const double a = cnt[0] > 0 ? sum[0] / cnt[0] : 0, b = cnt[1] > 0 ? sum[1] / cnt[1] : 0;
		std::printf(" | A/B: A %.2f ms (%.0f v) B %.2f ms (%.0f v, %.1f evitadas/fot) -> %+.2f ms/fot (sin contar la excepcion"
		            " de Windows que B se ahorra: no se mide aqui)",
		            a, cnt[0], b, cnt[1], cnt[1] > 0 ? ev[1] / cnt[1] : 0.0, (cnt[0] > 0 && cnt[1] > 0) ? b - a : 0.0);
		g_now.store(!was, std::memory_order_relaxed);
	}
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace FaltasEmu
