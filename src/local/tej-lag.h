#pragma once
// (tej-lag) Lo que el HILO DE RENDER espera por la sombra (KYTY_TEJ): cerrojos compartidos con ella. Solo cuenta cuando el
// cerrojo esta ocupado (try_lock primero: sin espera no hay coste extra ni contador). Contadores alineados a 64 (el render
// no comparte linea de cache con nadie). Linea en [tej] (tej.h).
#include <atomic>
#include <cstdint>
#include <immintrin.h>

namespace TejLag {

enum Kind : uint32_t { BcWrite, Labels, Publish, Count };
struct alignas(64) Acc {
	std::atomic<uint64_t> n {0}, ticks {0}; // (veces que el render encontro el cerrojo ocupado, ticks TSC esperando)
};
inline Acc                g_acc[Count];
inline thread_local bool  t_render = false; // (lo pone tej.h en el hilo de render)

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

} // namespace TejLag
