// (tej-libre) Banco offline: coste para el "render" de su cerrojo de TextureCache (spinlock con dueno + secuencia, como
// TrackingSpinLock) cuando un "trabajador" de la sombra lo toma tambien (antes) frente a cuando solo lee las epocas
// atomicas que el render escribe rara vez (KYTY_TEJ_LIBRE=1). Ademas comprueba la logica de la prueba sin cerrojo:
// lectura del slot + re-comprobacion de liveness/epocas frente a un escritor que borra y registra a la vez.
// g++ -O2 -std=c++20 -pthread -I src src/local/bench/tej-libre-bench.cpp -o /tmp/tlib && /tmp/tlib [cpu_render cpu_trab]
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <immintrin.h>

static void Pin(int cpu) {
#if defined(__linux__)
	if (cpu < 0) return;
	cpu_set_t s;
	CPU_ZERO(&s);
	CPU_SET(cpu, &s);
	pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#endif
}

struct alignas(64) Lock {
	std::atomic_flag      f;
	std::atomic<uint32_t> owner {0};
	std::atomic<uint32_t> seq {0};
	void lock(uint32_t me) {
		while (f.test_and_set(std::memory_order_acquire))
			while (f.test(std::memory_order_relaxed)) _mm_pause();
		owner.store(me, std::memory_order_relaxed);
		seq.store(seq.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	}
	void unlock() {
		seq.store(seq.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		owner.store(0, std::memory_order_relaxed);
		f.clear(std::memory_order_release);
	}
};
struct alignas(64) Epochs {
	std::atomic<uint64_t> resolution {1}, erase {1};
};
struct alignas(64) SlotX {
	std::atomic<uint32_t> gen {1};
	std::atomic<bool>     alive {true};
	uint64_t              a = 0, b = 0; // (campos del objeto: a == b siempre que no este a medio escribir)
};

static Lock             g_lock;
static Epochs           g_ep;
static SlotX            g_slot;
static std::atomic<int> g_mode {0}, g_stop {0};
static std::atomic<uint64_t> g_bad {0}, g_hits {0}, g_fb {0};

// Trabajador: modo 1 = toma el cerrojo como hoy (~5170/fot repartidos), modo 2 = libre (solo lecturas + re-comprobacion).
static void Worker(int cpu) {
	Pin(cpu);
	uint64_t local_hits = 0, local_fb = 0, local_bad = 0;
	while (!g_stop.load(std::memory_order_relaxed)) {
		const int m = g_mode.load(std::memory_order_relaxed);
		if (m == 1) {
			g_lock.lock(2);
			volatile uint64_t x = g_slot.a;
			(void)x;
			g_lock.unlock();
		} else if (m == 2) {
			const uint64_t e0 = g_ep.erase.load(std::memory_order_acquire), r0 = g_ep.resolution.load(std::memory_order_acquire);
			const uint32_t gen = g_slot.gen.load(std::memory_order_acquire);
			const bool     al  = g_slot.alive.load(std::memory_order_acquire);
			const uint64_t a = g_slot.a, b = g_slot.b;
			std::atomic_thread_fence(std::memory_order_acquire);
			if (!al || g_slot.alive.load(std::memory_order_acquire) != al || g_slot.gen.load(std::memory_order_acquire) != gen ||
			    g_ep.erase.load(std::memory_order_acquire) != e0 || g_ep.resolution.load(std::memory_order_acquire) != r0) {
				++local_fb;
			} else {
				++local_hits;
				if (a != b) ++local_bad; // (aceptado algo leido a medio borrar: debe ser 0)
			}
		}
		for (int i = 0; i < 200; ++i) _mm_pause(); // (~el resto del trabajo de un draw de la sombra)
	}
	g_hits += local_hits;
	g_fb += local_fb;
	g_bad += local_bad;
}

int main(int argc, char** argv) {
	const int cr = argc > 1 ? std::atoi(argv[1]) : 0, cw = argc > 2 ? std::atoi(argv[2]) : 1;
	std::thread w(Worker, cw);
	Pin(cr);
	for (int mode = 0; mode <= 2; ++mode) {
		g_mode.store(mode);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		const auto     t0 = std::chrono::steady_clock::now();
		constexpr int  N  = 2000000;
		uint64_t       sum = 0;
		for (int i = 0; i < N; ++i) {
			g_lock.lock(1);
			sum += g_slot.a;
			if ((i & 1023) == 0) { // (rara vez: borrar y volver a registrar, como EraseRetired / RegisterImage)
				g_slot.alive.store(false, std::memory_order_release);
				g_slot.gen.store(g_slot.gen.load(std::memory_order_relaxed) + 1, std::memory_order_release);
				g_slot.a = i; // (a medio escribir: a != b)
				std::atomic_thread_fence(std::memory_order_seq_cst);
				g_slot.b = i;
				g_ep.erase.fetch_add(1, std::memory_order_release);
				g_slot.alive.store(true, std::memory_order_release);
				g_ep.resolution.fetch_add(1, std::memory_order_release);
			}
			g_lock.unlock();
		}
		const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / N;
		std::printf("modo %d (%s): render %.1f ns por cerrojo (sum %llu)\n", mode,
		            mode == 0 ? "sin sombra" : mode == 1 ? "sombra con cerrojo" : "sombra libre", ns,
		            static_cast<unsigned long long>(sum & 1));
	}
	g_stop.store(1);
	w.join();
	std::printf("libre: aceptadas %llu, a reserva %llu, aceptadas a medio borrar %llu (debe ser 0)\n",
	            static_cast<unsigned long long>(g_hits.load()), static_cast<unsigned long long>(g_fb.load()),
	            static_cast<unsigned long long>(g_bad.load()));
	return g_bad.load() == 0 ? 0 : 1;
}
