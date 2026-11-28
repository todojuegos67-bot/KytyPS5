// Banco offline (tej-var): coste en el render del RenderMutex asimetrico cuando la sombra traduce draws (cada draw de la
// sombra = RequesterLock: mutex + FlushProcessWriteBuffers (aqui membarrier, el IPI de Linux) + esperar a que el render
// salga; el render que ve la peticion espera el mutex entero). Render: 5700 llamadas/fot de ~1.6 us, ~1.0 dentro del
// cerrojo (log a280f07 / plan-weave §4). Uso: g++ -O2 -pthread tej-var-bench.cpp && ./a.out
#include <x86intrin.h>
#include <linux/membarrier.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
static void Spin(double us) { const auto e = std::chrono::steady_clock::now() + std::chrono::nanoseconds(int64_t(us * 1000)); while (std::chrono::steady_clock::now() < e) _mm_pause(); }
static void Ipi() { syscall(__NR_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0); }
std::mutex mu; alignas(64) std::atomic<int> busy {0}; alignas(64) std::atomic<int> req {0}; std::atomic<bool> quit {false}, polite {false};
thread_local bool locked = false;
static void OwnerLock() { busy.store(1, std::memory_order_relaxed); std::atomic_signal_fence(std::memory_order_seq_cst);
	if (req.load(std::memory_order_acquire)) { busy.store(0); mu.lock(); locked = true; } }
static void OwnerUnlock() { if (locked) { locked = false; mu.unlock(); } else busy.store(0, std::memory_order_release); }
static void ReqLock() { if (polite) for (int i = 0; i < 2000 && busy.load(std::memory_order_relaxed); ++i) _mm_pause();
	mu.lock(); req.store(1); std::atomic_thread_fence(std::memory_order_seq_cst); Ipi(); while (busy.load(std::memory_order_acquire)) std::this_thread::yield(); }
static void ReqUnlock() { req.store(0, std::memory_order_release); mu.unlock(); }
int main() {
	syscall(__NR_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0);
	struct Cfg { int n; bool pol; bool work; const char* name; } cfgs[] = {
	    {0, false, false, "A (sin sombra)"}, {2, false, true, "B N=2"}, {1, false, true, "B N=1"},
	    {2, true, true, "B N=2 cortes"}, {2, false, false, "B N=2 sin traducir (sin cerrojo)"}};
	const int frames = 60, calls = 5700;
	for (auto& c: cfgs) {
		quit = false; polite = c.pol;
		std::vector<std::thread> ws;
		for (int i = 0; i < c.n; ++i) ws.emplace_back([&] { while (!quit) { if (c.work) { ReqLock(); Spin(1.0); ReqUnlock(); Spin(0.6); } else Spin(1.6); } });
		double tot = 0;
		for (int f = 0; f < frames; ++f) { const auto t0 = std::chrono::steady_clock::now();
			for (int k = 0; k < calls; ++k) { OwnerLock(); Spin(1.0); OwnerUnlock(); Spin(0.6); }
			tot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
		quit = true; for (auto& t: ws) t.join();
		std::printf("%-36s render %.2f ms/fot\n", c.name, tot / frames);
	}
}
