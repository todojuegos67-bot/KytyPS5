// Banco offline (tej-sinmutex): render 3300 llamadas/fot (~1.0 us dentro del cerrojo + 0.6 fuera, RenderMutex asimetrico)
// con 2 trabajadores de sombra. Con mutex: 764 peticiones/fot de la sombra (dato real 2d48357), cada una membarrier (IPI)
// y ~2.7 us dentro. Sin mutex: la sombra trabaja igual pero no pide. Uso: g++ -O2 -pthread tej-sinmutex-bench.cpp
#include <x86intrin.h>
#include <linux/membarrier.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
static void Spin(double us) { const auto e = std::chrono::steady_clock::now() + std::chrono::nanoseconds(int64_t(us * 1000)); while (std::chrono::steady_clock::now() < e) _mm_pause(); }
std::mutex mu; alignas(64) std::atomic<int> busy {0}; alignas(64) std::atomic<int> req {0}; std::atomic<bool> quit {false};
thread_local bool locked = false;
static void OL() { busy.store(1, std::memory_order_relaxed); std::atomic_signal_fence(std::memory_order_seq_cst); if (req.load(std::memory_order_acquire)) { busy.store(0); mu.lock(); locked = true; } }
static void OU() { if (locked) { locked = false; mu.unlock(); } else busy.store(0, std::memory_order_release); }
static void RL() { mu.lock(); req.store(1); std::atomic_thread_fence(std::memory_order_seq_cst); syscall(__NR_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0); while (busy.load(std::memory_order_acquire)) std::this_thread::yield(); }
static void RU() { req.store(0, std::memory_order_release); mu.unlock(); }
int main() {
	syscall(__NR_membarrier, MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0);
	const int frames = 60, calls = 3300; const double frame_us = 16000;
	for (int cfg = 0; cfg < 3; ++cfg) { // 0 A, 1 B con mutex, 2 B sin mutex
		quit = false; std::vector<std::thread> ws;
		for (int i = 0; cfg && i < 2; ++i) ws.emplace_back([&] { const double gap = 2 * frame_us / 764.0; // 382 pet./fot cada uno
			while (!quit) { if (cfg == 1) { RL(); Spin(2.7); RU(); } else Spin(2.7); Spin(gap - 2.7); } });
		double tot = 0;
		for (int f = 0; f < frames; ++f) { const auto t0 = std::chrono::steady_clock::now();
			for (int k = 0; k < calls; ++k) { OL(); Spin(1.0); OU(); Spin(0.6); }
			tot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
		quit = true; for (auto& t: ws) t.join();
		std::printf("%-22s render %.2f ms/fot\n", cfg == 0 ? "A" : cfg == 1 ? "B con mutex" : "B sin mutex", tot / frames);
	}
}
