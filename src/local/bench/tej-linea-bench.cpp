// (tej-linea) Banco offline: coste para el "render" de leer datos que comparten linea de cache con lo que escribe un
// "trabajador" en otro nucleo, y el del catalogo con std::shared_mutex frente a TejLinea::BrLock.
// g++ -O2 -std=c++20 -pthread -I src src/local/bench/tej-linea-bench.cpp -o /tmp/tlb && /tmp/tlb [cpu_render cpu_trab]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <shared_mutex>
#include <thread>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "local/tej-linea.h"

static void Pin(int cpu) {
#if defined(__linux__)
	if (cpu < 0) return;
	cpu_set_t s;
	CPU_ZERO(&s);
	CPU_SET(cpu, &s);
	pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#endif
}

// Como RenderExecutor: campos que el render lee en cada draw y, al lado (o no), el cerrojo que el trabajador toma.
struct Juntos {
	std::shared_mutex     mutex; // (lo escribe el trabajador)
	void*                 catalog = nullptr;
	uint32_t              role    = 0;
	std::atomic<uint32_t> readers {0};
};
struct Separados {
	TejLinea::BrLock      mutex;
	alignas(64) void*     catalog = nullptr;
	uint32_t              role    = 0;
	std::atomic<uint32_t> readers {0};
};

template <class S, class Lock>
static double Run(S& s, Lock lock, bool worker_on, int cpu_r, int cpu_w) {
	std::atomic<bool> stop {false};
	std::thread       w([&] {
        Pin(cpu_w);
        while (!stop.load(std::memory_order_relaxed))
            if (worker_on) lock(s);
    });
	Pin(cpu_r);
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	constexpr int N   = 20'000'000;
	uint64_t      acc = 0;
	const auto    t0  = std::chrono::steady_clock::now();
	for (int i = 0; i < N; ++i) {
		acc += reinterpret_cast<uintptr_t>(*static_cast<void* volatile*>(&s.catalog)) + *static_cast<volatile uint32_t*>(&s.role);
		// (trabajo del draw entre lecturas: ~20 ns)
		for (int k = 0; k < 16; ++k) asm volatile("" : "+r"(acc));
	}
	const auto t1 = std::chrono::steady_clock::now();
	stop = true;
	w.join();
	if (acc == 42) std::printf("x");
	return std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
}

int main(int argc, char** argv) {
	const int cr = argc > 2 ? std::atoi(argv[1]) : 0, cw = argc > 2 ? std::atoi(argv[2]) : 1;
	Juntos    j;
	Separados p;
	const auto sm = [](Juntos& x) { x.mutex.lock_shared(), x.mutex.unlock_shared(); };
	const auto br = [](Separados& x) { x.mutex.lock_shared(), x.mutex.unlock_shared(); };
	for (int r = 0; r < 3; ++r) {
		const double bj  = Run(j, sm, false, cr, cw), jun = Run(j, sm, true, cr, cw);
		const double bp  = Run(p, br, false, cr, cw), sep = Run(p, br, true, cr, cw);
		std::printf("[tej-linea-bench] render cpu %d, trabajador cpu %d, ns por draw del render: shared_mutex en la misma"
		            " linea %.2f -> %.2f (+%.2f) | BrLock en su linea %.2f -> %.2f (+%.2f)\n",
		            cr, cw, bj, jun, jun - bj, bp, sep, sep - bp);
	}
	return 0;
}
