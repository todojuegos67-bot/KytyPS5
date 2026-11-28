// Banco offline (tej-lagmedida): coste en el HILO DE RENDER de la ventana B de KYTY_TEJ=3, cargas del log
// 20261128-122630-433 (3299 draws/fot, 2 trabajadores, subidas apuntadas 37746/fot, push sustituidas 6377/fot).
//  1) contencion: el render lee por draw 3 lineas compartidas (set, estado de draw, par) mientras un trabajador usa los
//     mismos objetos. ANTES el trabajador escribia used_frame/variant_hint en cada draw; AHORA solo si cambia.
//  2) medida del paso A en el render: registro por contenido (ContentNote real de tej-repro.h) y sustitucion de push,
//     ANTES en toda la ventana B, AHORA solo en los paquetes muestreados (1/16).
// Falla (salida 1) si el render B-A AHORA pasa de +1 ms/fot.
// g++ -std=c++20 -O2 -pthread -I src src/local/bench/tej-lag-bench.cpp -o /tmp/tlb/b && /tmp/tlb/b
#include "local/tej-repro.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
struct alignas(64) Obj {
	std::atomic<uint64_t> used_frame {0};
	uint32_t              hint = 0;
	uint32_t              words[12] {};
};
constexpr int kDraws = 3299, kObjs = 600, kFrames = 120;
static Obj g_obj[3][kObjs];
static std::atomic<bool> g_stop {false};
static std::atomic<int>  g_mode {0}; // 0 sin trabajador, 1 escribe siempre (antes), 2 escribe si cambia (ahora)
static std::atomic<uint64_t> g_frame {1};
static void Worker() {
	uint32_t i = 0;
	while (!g_stop.load(std::memory_order_relaxed)) {
		const int m = g_mode.load(std::memory_order_relaxed);
		if (m == 0) { std::this_thread::yield(); continue; }
		const uint64_t fr = g_frame.load(std::memory_order_relaxed);
		for (int k = 0; k < 3; ++k) {
			auto& o = g_obj[k][(i * 7 + k) % kObjs];
			if (m == 1) o.used_frame.store(fr, std::memory_order_relaxed), o.hint = i & 3;
			else {
				if (o.used_frame.load(std::memory_order_relaxed) != fr) o.used_frame.store(fr, std::memory_order_relaxed);
				if (o.hint != 1) o.hint = 1;
			}
		}
		++i;
		for (int s = 0; s < 40; ++s) asm volatile(""); // ~traduccion entre draws (el trabajador va a ~5 us/llamada: cota alta)
	}
}
static double RenderFrame(bool measure_all, std::vector<uint8_t>& src, uint64_t& sink) {
	using namespace TejRepro;
	const auto t0 = std::chrono::steady_clock::now();
	const uint64_t fr = g_frame.fetch_add(1) + 1;
	static uint64_t off = 0;
	const int notes = measure_all ? 37746 : 37746 / 16, pushes = measure_all ? 6377 : 6377 / 16;
	for (int d = 0; d < kDraws; ++d) {
		for (int k = 0; k < 3; ++k) {
			auto& o = g_obj[k][(d * 5 + k) % kObjs];
			sink += o.words[d & 7] + o.hint;
			if (o.used_frame.load(std::memory_order_relaxed) != fr) o.used_frame.store(fr, std::memory_order_relaxed);
		}
		for (int n = d * notes / kDraws; n < (d + 1) * notes / kDraws; ++n) {
			ContentNote(0x1234, 0x7f0000000000ull, off, src.data() + (n * 64) % 32768, 64 + (n * 37) % 448);
			off = (off + 512) & 0xffffff;
		}
		for (int n = d * pushes / kDraws; n < (d + 1) * pushes / kDraws; ++n) {
			uint32_t p[32];
			std::memcpy(p, src.data() + (n % 512) * 64, sizeof(p));
			const uint64_t a = 0x7f0000000000ull + ((n * 512ull) & 0xffffff);
			p[2] = static_cast<uint32_t>(a), p[3] = static_cast<uint32_t>(a >> 32);
			sink += PushByContent(p, 32);
		}
	}
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
static double Run(int mode, bool measure_all, std::vector<uint8_t>& src, uint64_t& sink) {
	g_mode = mode;
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	double best = 1e9, sum = 0;
	for (int f = 0; f < kFrames; ++f) {
		const double t = RenderFrame(measure_all, src, sink);
		sum += t, best = std::min(best, t);
	}
	return sum / kFrames;
}
int main() {
	std::vector<uint8_t> src(64 * 1024);
	std::mt19937_64 rng(5);
	for (auto& b : src) b = static_cast<uint8_t>(rng());
	std::thread w1(Worker), w2(Worker);
	uint64_t sink = 0;
	// (A: sin sombra ni medida; el render sigue tocando used_frame una vez por fotograma)
	double a = 0, before = 0, now = 0, only_w = 0, only_m = 0;
	(void)Run(0, false, src, sink); // calentamiento
	for (int r = 0; r < 3; ++r) { // alterna para que la frecuencia no sesgue
		a += Run(0, false, src, sink);
		before += Run(1, true, src, sink);
		only_w += Run(2, true, src, sink);  // solo escritura si cambia
		only_m += Run(1, false, src, sink); // solo medida 1/16
		now += Run(2, false, src, sink);
	}
	a /= 3, before /= 3, now /= 3, only_w /= 3, only_m /= 3;
	g_stop = true;
	w1.join(), w2.join();
	const double d_now = now - a;
	std::printf("tej-lag-bench (%u nucleos): render A %.3f ms/fot | B ANTES %+.3f | solo escritura-si-cambia %+.3f | solo medida 1/16"
	            " %+.3f | B AHORA (las dos) %+.3f ms (limite +1.0) sink %llu %s\n",
	            std::thread::hardware_concurrency(), a, before - a, only_w - a, only_m - a, d_now,
	            static_cast<unsigned long long>(sink & 1), d_now <= 1.0 ? "OK" : "FALLA");
	return d_now <= 1.0 ? 0 : 1;
}
