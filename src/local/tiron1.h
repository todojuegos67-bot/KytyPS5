#pragma once
// (tiron1) KYTY_TIRON1=1 (off by default; only "1" turns it on): the hitches (frames over 25 ms) of the 16 GB / 8 GB
// test profile, by cause. Logs 20261116-093502-352, -110146-363, -183301-339 (KYTY_RAIZ=1): 408 frames over 25 ms, of
// which 230 the table path warming up at a zone's entry (TABLE_STORE_BUDGET=8: 8 stores a frame for ~1000 new pairs,
// ~2,000-3,000 store-deferred draws a frame for ~5 s, their draws on the normal path: draw=9-17 ms a frame), 83 compute
// pipelines and programs made on the render thread (the pipeline warmup is skipped under 30 GB of RAM), 43 loading
// screens and menus, 18 new images' initialisation, 11 upload waits.
//   1. Table stores by time, not by count (table-xpr.inc TableTry): past the 8 a frame, more while the stores granted
//      this frame took less than KYTY_TIRON1_STORE_MS (default 2) of the render thread; the draws of a pair waiting
//      for its store (compiling, stored this frame, past the budget) take the native records they have, and store no new
//      ones (native-xpr.inc NativeXprTry: those records were replaced by the table variant a frame or two later).
//   2. Compute pipelines before play on a PC under 30 GB of RAM (pipelineCache.cpp Tiron1WarmCompute): the recording's
//      compute recipes (newest first, at most KYTY_TIRON1_COMPUTE_MAX, default 1500, and the warmup's video memory cap),
//      their programs translated and their pipelines made from the static store's binaries beside the game, once the
//      shader prefetch is done. The graphics ones stay where they were (their workers already make them beside the
//      render thread). The same programs and pipelines the game makes: nothing is skipped, the image is the same.
//   3. A [tiron1] line every 5 s: the frames over 25 ms by cause (the classification of tools tiron1/tirones.py) and
//      what 1 and 2 did.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace Tiron1 {

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON1");
		const bool  is    = value == nullptr || std::string_view(value) != "0"; // (picos) on by default
		if (is) {
			std::printf("Tiron1: on (KYTY_TIRON1=1): table stores by time, compute pipelines before play on a low-RAM PC, "
			            "a [tiron1] line every 5 s\n");
			std::fflush(stdout);
		}
		return is;
	}();
	return on;
}

inline double EnvNumber(const char* name, double fallback, double low, double high) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const double number = std::atof(value);
	return number < low ? low : number > high ? high : number;
}

// The render thread's time a frame for table stores past the count budget (microseconds).
inline int64_t StoreBudgetUs() {
	static const int64_t us = static_cast<int64_t>(EnvNumber("KYTY_TIRON1_STORE_MS", 2.0, 0.0, 50.0) * 1000.0);
	return us;
}

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// What 1 and 2 did (totals; the line prints the 5 s differences).
inline std::atomic<uint64_t> g_stores {0}, g_stores_by_time {0}, g_store_us {0}, g_native_kept {0}, g_native_store_skipped {0};
inline std::atomic<uint64_t> g_warm_programs {0}, g_warm_compute {0}, g_warm_ms {0}, g_warm_wanted {0};
inline std::atomic<uint32_t> g_warm_state {0}; // 0 off, 1 waiting for the prefetch, 2 working, 3 done, 4 nothing recorded

// The draw TableTry last sent to the native records is a pair's waiting for its table store: NativeXprTry stores no
// native record for it (the thread's own: TableTry runs right before, on the same thread).
inline thread_local bool t_no_native_store = false;

enum Cause : uint32_t { Tables, Pipeline, Program, Upload, GpuWait, Images, Loading, Other, Causes };
inline constexpr const char* CauseNames[Causes] = {"tablas", "pipelines", "programas", "subidas", "esperas_gpu",
                                                   "imagenes", "carga/juego", "otros"};

// A frame's render-thread costs (ms), from the SLOW Frame line's sources (live-control.h Flip).
struct FrameCost {
	double ms = 0, idle = 0, draw = 0, xpr = 0, dispatch_table = 0, compile = 0, programs = 0, upload_wait = 0,
	       gpu_waits = 0, image_init = 0;
	uint64_t store_deferred = 0, xpr_stores = 0;
};

// The cause that alone takes the frame under 25 ms (the largest such), else the largest of at least 3 ms.
inline Cause Classify(const FrameCost& c) {
	if (c.idle > c.ms * 0.5 || c.ms > 200.0) return Loading;
	std::array<double, Causes> v {};
	if (c.store_deferred >= 200 || c.xpr_stores >= 64)
		v[Tables] = c.draw + (c.xpr > 6.0 ? c.xpr - 6.0 : 0.0) + (c.dispatch_table > 3.0 ? c.dispatch_table - 3.0 : 0.0);
	v[Pipeline] = c.compile;
	v[Program]  = c.programs;
	v[Upload]   = c.upload_wait;
	v[GpuWait]  = c.gpu_waits;
	v[Images]   = c.image_init;
	v[Loading]  = c.idle;
	int alone = -1, largest = -1;
	for (int i = 0; i < static_cast<int>(Other); ++i) {
		if (v[i] > 0 && c.ms - v[i] <= 25.0 && (alone < 0 || v[i] > v[alone])) alone = i;
		if (largest < 0 || v[i] > v[largest]) largest = i;
	}
	if (alone >= 0) return static_cast<Cause>(alone);
	return largest >= 0 && v[largest] >= 3.0 ? static_cast<Cause>(largest) : Other;
}

// Render thread, every frame (flip to flip): counted, and a line every 5 s.
inline void NoteFrame(const FrameCost& c, std::chrono::steady_clock::time_point now) {
	static std::chrono::steady_clock::time_point start {};
	static uint32_t                              frames = 0, slow = 0;
	static std::array<uint32_t, Causes>          causes {};
	static std::array<uint64_t, 9>               last {};
	if (start == std::chrono::steady_clock::time_point {}) start = now;
	frames++;
	if (c.ms > 25.0) {
		slow++;
		causes[Classify(c)]++;
	}
	if (now - start < std::chrono::seconds(5)) return;
	const std::array<uint64_t, 9> totals {g_stores.load(), g_stores_by_time.load(), g_store_us.load(), g_native_kept.load(),
	                                      g_native_store_skipped.load(), g_warm_programs.load(), g_warm_compute.load(),
	                                      g_warm_ms.load(), g_warm_wanted.load()};
	std::printf("[tiron1] 5 s: %u fotogramas, %u de mas de 25 ms (%.2f %%):", frames, slow,
	            frames != 0 ? 100.0 * slow / frames : 0.0);
	for (uint32_t i = 0; i < Causes; ++i) std::printf(" %s %u", CauseNames[i], causes[i]);
	static constexpr const char* states[] = {"apagado", "esperando al prefetch", "en marcha", "hecho", "nada grabado"};
	const auto state = g_warm_state.load(std::memory_order_relaxed);
	std::printf(" | tablas: stores %llu (%llu por tiempo, %.1f ms), draws de pares en espera a records nativos %llu, "
	            "records nativos no guardados %llu | compute previo (%s): %llu programas, %llu de %llu pipelines, %llu ms\n",
	            static_cast<unsigned long long>(totals[0] - last[0]), static_cast<unsigned long long>(totals[1] - last[1]),
	            static_cast<double>(totals[2] - last[2]) / 1000.0, static_cast<unsigned long long>(totals[3] - last[3]),
	            static_cast<unsigned long long>(totals[4] - last[4]), states[state < 5 ? state : 0],
	            static_cast<unsigned long long>(totals[5]), static_cast<unsigned long long>(totals[6]),
	            static_cast<unsigned long long>(totals[8]), static_cast<unsigned long long>(totals[7]));
	std::fflush(stdout);
	last   = totals;
	start  = now;
	frames = slow = 0;
	causes = {};
}

} // namespace Tiron1
