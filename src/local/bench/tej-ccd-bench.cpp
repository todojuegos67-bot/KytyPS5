// (tej-ccd) Banco offline: coste de una linea compartida render<->sombra (ping-pong de un atomico entre dos hilos) y
// estimacion del lag por fotograma con los contadores medidos (build f0256d4): cerrojos de seguimiento de la sombra
// 5170/fot; cada uno = la linea del cerrojo + la de los datos que el render lee despues (~2-3 traspasos por cerrojo).
// Sin dos CCD aqui: el traspaso medido es el de la misma L3; el de otro CCD se toma 100-200 ns (Zen 5, Infinity Fabric).
// Uso: g++ -O2 -std=c++20 -pthread tej-ccd-bench.cpp && ./a.out
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
int main() {
	alignas(64) static std::atomic<uint32_t> line {0};
	constexpr uint32_t N = 2000000;
	std::thread t([] {
		for (uint32_t i = 1; i < 2 * N; i += 2) {
			while (line.load(std::memory_order_acquire) != i) {}
			line.store(i + 1, std::memory_order_release);
		}
	});
	const auto t0 = std::chrono::steady_clock::now();
	for (uint32_t i = 0; i < 2 * N; i += 2) {
		while (line.load(std::memory_order_acquire) != i) {}
		line.store(i + 1, std::memory_order_release);
	}
	t.join();
	const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (2.0 * N);
	std::printf("traspaso de linea medido aqui (misma L3): %.0f ns\n", ns);
	const double locks = 5170, per[2] = {2, 3}, inter[2] = {100, 200}, intra = 45;
	for (int a = 0; a < 2; ++a)
		std::printf("%g traspasos/cerrojo: otro CCD %.2f-%.2f ms/fot, mismo CCD %.2f ms/fot -> ahorro %.2f-%.2f ms\n", per[a],
		            locks * per[a] * inter[0] * 1e-6, locks * per[a] * inter[1] * 1e-6, locks * per[a] * intra * 1e-6,
		            locks * per[a] * (inter[0] - intra) * 1e-6, locks * per[a] * (inter[1] - intra) * 1e-6);
}
