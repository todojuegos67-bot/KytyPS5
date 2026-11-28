// Banco offline (tej-repro A3b): registro por contenido (ContentNote sobre el ORIGEN del host), sustitucion de push
// (PushByContent) y busqueda VB/IB (ContentGet) con el codigo real de tej-repro.h. Cargas por fotograma (cota alta del
// log 20261128-114203: subidas apuntadas 3855-4219/fot, push sustituidas <= 2929/fot): 4500 subidas por Copy (bloques de
// tabla 64 B..2 KiB, 9 % de 2..4 KiB, 1 % > 4 KiB sin apuntar), 4000 push de 128 B, 8000 busquedas VB/IB. Falla (salida 1) si > 0.5 ms/fot.
// g++ -std=c++20 -O2 -I src src/local/bench/tej-contenido-bench.cpp -o /tmp/tcb && /tmp/tcb
#include "local/tej-repro.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <cstring>
#include <vector>
int main() {
	using namespace TejRepro;
	std::mt19937_64 rng(7);
	const int FR = 200, NUP = 4500, NPUSH = 4000, NVB = 8000;
	std::vector<uint8_t> src(64 * 1024);
	for (auto& b : src) b = static_cast<uint8_t>(rng());
	std::vector<uint64_t> sz(NUP), addr(NUP);
	for (int i = 0; i < NUP; ++i) {
		const int r = static_cast<int>(rng() % 100);
		sz[i]       = r == 0 ? 32768 : r < 10 ? 2048 + rng() % 2048 : 64 + rng() % 1984;
	}
	static uint32_t base[NPUSH][32], push[NPUSH][32];
	std::vector<uint32_t> pick(NPUSH + NVB);
	for (auto& r : base) for (auto& w : r) w = static_cast<uint32_t>(rng());
	for (auto& p : pick) p = static_cast<uint32_t>(rng() % NUP);
	uint64_t sink = 0, off = 0;
	const uint64_t bda = 0x7f0000000000ull, handle = 0x1234;
	const auto t0 = std::chrono::steady_clock::now();
	double tn = 0;
	for (int f = 0; f < FR; ++f) {
		const auto q0 = std::chrono::steady_clock::now();
		for (int i = 0; i < NUP; ++i) {
			addr[i] = off;
			ContentNote(handle, bda, off, src.data() + (i * 64) % 32768, sz[i]);
			off = (off + sz[i] + 255) & ~255ull;
		}
		tn += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q0).count();
		for (int i = 0; i < NPUSH; ++i) {
			std::memcpy(push[i], base[i], sizeof(push[i]));
			const uint64_t a = bda + addr[pick[i]]; // 2 direcciones de bloque por push
			push[i][2] = static_cast<uint32_t>(a), push[i][3] = static_cast<uint32_t>(a >> 32);
			push[i][9] = static_cast<uint32_t>(a), push[i][10] = static_cast<uint32_t>(a >> 32);
			sink += PushByContent(push[i], 32);
		}
		for (int i = 0; i < NVB; ++i) {
			uint64_t h = 0;
			sink += ContentGet(HandleKey(handle, addr[pick[NPUSH + i]]), h) ? h & 1 : 0;
		}
	}
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / FR;
	std::printf("tej-contenido-bench: %.3f ms/fot simulado (registro %.3f) (limite 0.5) sustituidas %llu %s\n", ms, tn / FR,
	            static_cast<unsigned long long>(sink), ms <= 0.5 ? "OK" : "FALLA");
	return ms <= 0.5 ? 0 : 1;
}
