#pragma once
// (tej-lag) Lo que el HILO DE RENDER espera por la sombra (KYTY_TEJ): cerrojos compartidos con ella. Solo cuenta cuando el
// cerrojo esta ocupado (try_lock primero: sin espera no hay coste extra ni contador). Contadores alineados a 64 (el render
// no comparte linea de cache con nadie). Linea en [tej] (tej.h).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

namespace TejLag {

// (tej-var) RmOwner: el render encontro una peticion del RenderMutex (OwnerLockSlow) y espero el cerrojo (ticks);
// RmReq: peticiones de hilos de la sombra (cada una = FlushProcessWriteBuffers = IPI a todos los nucleos, tambien al
// del render; ticks: lo que la sombra espero a que el render saliera).
// (tej-sinmutex) RmSkip: peticiones de la sombra evitadas (KYTY_TEJ_SINMUTEX: el draw normal que la pediria = hueco
// 'sin mutex'); BarFree: barreras / cortes de cadena compute grabados por la sombra en SU buffer sin el RenderMutex;
// CatW: el render esperando el catalogo de tablas (CatalogWrite) porque la sombra lo lee (CatalogRead).
enum Kind : uint32_t { BcWrite, Labels, Publish, RmOwner, RmReq, RmSkip, BarFree, CatW, Count };
struct alignas(64) Acc {
	std::atomic<uint64_t> n {0}, ticks {0}; // (veces que el render encontro el cerrojo ocupado, ticks TSC esperando)
};
inline Acc                g_acc[Count];
inline thread_local bool  t_render = false; // (lo pone tej.h en el hilo de render)
inline thread_local bool  t_shadow = false; // (tej-var: lector y trabajadores de la sombra)
// (tej-var) Variante 4 'cortes': la sombra no pide el RenderMutex mientras el render esta dentro (sin IPI ni espera del
// render); espera fuera, como mucho ~50 us, y luego pide como siempre. Solo retrasa a la sombra.
inline std::atomic<bool>  g_polite {false};
// (tej-sinmutex) KYTY_TEJ_SINMUTEX (por defecto 1 con KYTY_TEJ>0): la sombra nunca pide el RenderMutex. t_translating:
// dentro de Speculation::Shadow de un trabajador (alli una RecorderRefusal se convierte en hueco contado por motivo).
inline std::atomic<bool>  g_sinmutex {false};
inline thread_local bool  t_translating = false;
[[nodiscard]] inline bool NoMutex() { return t_shadow && t_translating && g_sinmutex.load(std::memory_order_relaxed); }
inline void Note(Kind k) { g_acc[k].n.fetch_add(1, std::memory_order_relaxed); }

// Toma `m` (lock()); si el render lo encuentra ocupado, cuenta el tiempo de espera.
template <class M>
inline void Lock(M& m, Kind k) {
	if (m.try_lock()) return;
	if (!t_render) {
		m.lock();
		return;
	}
	const uint64_t t0 = __rdtsc();
	m.lock();
	g_acc[k].n.fetch_add(1, std::memory_order_relaxed);
	g_acc[k].ticks.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
}
template <class M>
struct Guard {
	M& m;
	Guard(M& mm, Kind k): m(mm) { Lock(m, k); }
	~Guard() { m.unlock(); }
	Guard(const Guard&)            = delete;
	Guard& operator=(const Guard&) = delete;
};

// (tej-lag4) CAUSA del lag en escena ligera, medida en el propio hilo de render (solo con KYTY_TECHO=1 y KYTY_TEJ>0;
// g_measure lo pone tej.h). Por lado (0 A, 1 B) por ventana:
//  - FRECUENCIA REAL del nucleo del render: cadena dependiente de 2048 multiplicaciones (imul: 3 ciclos de nucleo en
//    Zen 4/5, sin depender de memoria) medida con el TSC (frecuencia fija) 1 de cada 4 DCB: GHz = 6144 ciclos / tiempo.
//    La 'GHz efectivos' de la linea DCB cola 0 es TSC/pared (QueryThreadCycleTime cuenta en ticks del TSC), NO el reloj
//    del nucleo: no sirve para ver el boost. Esta si. min = la mas rapida (sin expulsion), media = tipica.
//  - nucleo del render y migraciones (otro nucleo al empezar una DCB que en la anterior).
//  - VirtualProtect del render (ProtectHost): llamadas y tiempo; con mas nucleos ejecutando hilos del proceso cada
//    proteccion que quita permisos hace un TLB shootdown (IPI) a todos ellos y espera su respuesta.
inline std::atomic<bool> g_measure {false};
inline std::atomic<int>  g_side {0};
struct alignas(64) Probe {
	std::atomic<uint64_t> fq_n[2] {}, fq_ticks[2] {}, fq_min[2] {}, mig[2] {}, dcb[2] {}, prot_n[2] {}, prot_t[2] {};
	std::atomic<uint32_t> cpu {0xffffffffu};
};
inline Probe g_probe;
inline constexpr uint64_t FreqIters = 2048, FreqCyclesPerIter = 3;
inline volatile uint64_t  g_freq_k = 0x9E3779B97F4A7C15ull, g_freq_sink = 0;
// Una vez por DCB de la cola 0 (en GpuSlice, primer trozo).
inline void RenderDcbStart(uint32_t cpu) {
	if (!g_measure.load(std::memory_order_relaxed)) return;
	const int sd = g_side.load(std::memory_order_relaxed);
	g_probe.dcb[sd].fetch_add(1, std::memory_order_relaxed);
	const uint32_t last = g_probe.cpu.exchange(cpu, std::memory_order_relaxed);
	if (last != 0xffffffffu && last != cpu) g_probe.mig[sd].fetch_add(1, std::memory_order_relaxed);
	static thread_local uint32_t k4 = 0;
	if ((++k4 & 3u) != 0) return;
	uint64_t       x = g_freq_sink | 1u;
	const uint64_t m = g_freq_k;
	const uint64_t t0 = __rdtsc();
	for (uint64_t i = 0; i < FreqIters; ++i) {
		x *= m;
#if defined(__clang__) || defined(__GNUC__)
		asm volatile("" : "+r"(x));
#endif
	}
	const uint64_t dt = __rdtsc() - t0;
	g_freq_sink = x;
	g_probe.fq_n[sd].fetch_add(1, std::memory_order_relaxed);
	g_probe.fq_ticks[sd].fetch_add(dt, std::memory_order_relaxed);
	uint64_t cur = g_probe.fq_min[sd].load(std::memory_order_relaxed);
	while ((cur == 0 || dt < cur) && !g_probe.fq_min[sd].compare_exchange_weak(cur, dt, std::memory_order_relaxed)) {
	}
}
// VirtualProtect del hilo de render (pageManager ProtectHost).
struct ProtTimer {
	uint64_t t0;
	ProtTimer(): t0(t_render && g_measure.load(std::memory_order_relaxed) ? __rdtsc() : 0) {}
	~ProtTimer() {
		if (t0 == 0) return;
		const int sd = g_side.load(std::memory_order_relaxed);
		g_probe.prot_n[sd].fetch_add(1, std::memory_order_relaxed);
		g_probe.prot_t[sd].fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
	}
	ProtTimer(const ProtTimer&)            = delete;
	ProtTimer& operator=(const ProtTimer&) = delete;
};
// (tej-lag4) Variante 7 'muestreo 1/8': la sombra solo toma DCB 1 de cada 8 tramos de 40 ms (12.5 % del tiempo); el
// resto del tiempo lector y trabajadores duermen. Si el lag es proporcional al trabajo de la sombra (sin traducir ~0),
// baja a 1/8. KYTY_TEJ_MUESTREO=8 lo aplica tambien fuera de las variantes (1 = siempre, como antes).
[[nodiscard]] inline uint32_t SampleDiv() {
	static const uint32_t d = [] {
		const char* v = std::getenv("KYTY_TEJ_MUESTREO");
		const int   n = v != nullptr ? std::atoi(v) : 1;
		return static_cast<uint32_t>(n < 1 ? 1 : n > 64 ? 64 : n);
	}();
	return d;
}
[[nodiscard]] inline bool SampleOff(uint64_t now_ns, uint32_t div) {
	return div > 1 && (now_ns / 40000000ull) % div != 0;
}

} // namespace TejLag
