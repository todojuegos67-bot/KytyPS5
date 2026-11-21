#pragma once
// (tiron8, KYTY_TIRON8=1; off by default) Texture upload copies over pages RAIZ made no-access: one window per copy (at most
// 64 MiB) and the eight copy threads inside it (async-upload.h, Worker::Tiron8Take), instead of a window - the protection
// gate taken alone - per 256 KiB piece. "[tiron8]" line every 5 s.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Tiron8 {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_TIRON8");
		const bool  set = v == nullptr || v[0] != '0'; // (picos) on by default
		std::printf("[tiron8] %s: copias de textura contiguas de 1 MiB o mas en una ventana por tramo (max 64 MiB), 8 hilos "
		            "copiando dentro%s\n",
		            set ? "ON (KYTY_TIRON8=1)" : "off", set ? "" : " - KYTY_TIRON8=1 lo enciende");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

inline std::atomic<uint64_t> g_windows {0}, g_bytes {0}, g_us {0};

// Upload worker: the "[tiron8]" line every 5 s (per second: windows, MB copied, ms with windows open; GB/s inside them).
inline void Report() {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	static uint64_t          prev[3] {};
	const auto               now  = Clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	const double   secs = std::chrono::duration<double>(now - last).count();
	const uint64_t cur[3] {g_windows.load(), g_bytes.load(), g_us.load()};
	const double   windows = (cur[0] - prev[0]) / secs, mb = (cur[1] - prev[1]) / 1e6 / secs, ms = (cur[2] - prev[2]) / 1e3 / secs;
	std::printf("[tiron8] (%.1f s) ventanas %.1f/s, copiado %.1f MB/s, ventana abierta %.1f ms/s, %.2f GB/s dentro\n", secs,
	            windows, mb, ms, ms > 0.0 ? mb / ms : 0.0);
	std::fflush(stdout);
	for (int i = 0; i < 3; ++i) prev[i] = cur[i];
	last = now;
}

} // namespace Tiron8
