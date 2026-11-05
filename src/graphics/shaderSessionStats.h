#ifndef EMULATOR_SRC_GRAPHICS_SHADER_SESSION_STATS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_SESSION_STATS_H_
// ShaderSession: one cumulative line per 30 s (and at a fatal GPU error) with what the shader
// path cost this session, so a run can be compared with another from the log alone:
//   startup  = compute pipelines rebuilt by the prewarm replay + translations found on disk
//   in play  = guest shaders translated (first sight / new permutation of a known shader)
//   GPU      = GpuWaitTimeout (2 s slices) and ErrorDeviceLost events
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace ShaderSessionStats {

inline std::atomic<uint64_t> g_prewarm_compute {0};   // compute pipelines in the prewarm seed
inline std::atomic<uint64_t> g_xlat_loaded {0};       // TranslationCache records on disk at start
inline std::atomic<uint64_t> g_xlat_hits {0};
inline std::atomic<uint64_t> g_cs_first {0};          // CS translated, first sight of the guest shader
inline std::atomic<uint64_t> g_cs_perm {0};           // CS translated again for a new permutation
inline std::atomic<uint64_t> g_gfx_translations {0};  // VS/PS/... translations
inline std::atomic<uint64_t> g_cs_ns {0};
inline std::atomic<uint64_t> g_cs_max_ns {0};
inline std::atomic<uint64_t> g_gfx_ns {0};
inline std::atomic<uint64_t> g_runtime_stride_buffers {0}; // direct raw buffers given a runtime stride
inline std::atomic<uint64_t> g_wait_timeouts {0};
inline std::atomic<uint64_t> g_device_lost {0};

inline uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count());
}

// "Translate (5 s)": what translating guest shaders cost per 5 s window, printed from the translation
// itself (the gpuprof line stops when the game stalls at 0.1 fps, which is when it matters).
//   n/ms/max = shaders translated in the window; hits = translations served from the disk cache
//   instead (no translation); share = translate ms / window ms (>100% = several threads).
inline std::atomic<uint64_t> g_win_n {0}, g_win_ns {0}, g_win_max_ns {0}, g_win_hits {0}, g_win_cs {0};
inline std::atomic<uint64_t> g_win_next {0};
inline std::atomic<uint64_t> g_win_start {0};
inline std::atomic<uint64_t> g_total_windows {0};

inline void MaybePrintWindow() {
	const auto now = NowNs();
	auto due = g_win_next.load(std::memory_order_relaxed);
	if (due == 0) {
		if (g_win_next.compare_exchange_strong(due, now + 5'000'000'000ull)) g_win_start.store(now);
		return;
	}
	if (now < due || !g_win_next.compare_exchange_strong(due, now + 5'000'000'000ull)) return;
	const auto start  = g_win_start.exchange(now);
	const auto n      = g_win_n.exchange(0);
	const auto ns     = g_win_ns.exchange(0);
	const auto max_ns = g_win_max_ns.exchange(0);
	const auto hits   = g_win_hits.exchange(0);
	const auto cs     = g_win_cs.exchange(0);
	if (n == 0 && hits == 0) return;
	const double wall_ms = static_cast<double>(now - start) / 1e6;
	std::printf("Translate (5 s): %llu shaders translated (%llu compute) in %.0f ms (worst %.0f ms, %.0f%% of %.0f ms wall), "
	            "%llu served from the disk cache | total: %llu translated, %llu disk hits\n",
	            static_cast<unsigned long long>(n), static_cast<unsigned long long>(cs), ns / 1e6, max_ns / 1e6,
	            wall_ms > 0 ? 100.0 * (ns / 1e6) / wall_ms : 0.0, wall_ms, static_cast<unsigned long long>(hits),
	            static_cast<unsigned long long>(g_cs_first.load() + g_cs_perm.load() + g_gfx_translations.load()),
	            static_cast<unsigned long long>(g_xlat_hits.load()));
	std::fflush(stdout);
}

inline void NoteDiskHit() {
	g_win_hits.fetch_add(1, std::memory_order_relaxed);
	MaybePrintWindow();
}

inline void NoteTranslation(bool compute, bool first_sight, uint64_t ns) {
	g_win_n.fetch_add(1, std::memory_order_relaxed);
	g_win_ns.fetch_add(ns, std::memory_order_relaxed);
	if (compute) g_win_cs.fetch_add(1, std::memory_order_relaxed);
	{
		auto max = g_win_max_ns.load(std::memory_order_relaxed);
		while (ns > max && !g_win_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
		}
	}
	MaybePrintWindow();
	if (compute) {
		(first_sight ? g_cs_first : g_cs_perm).fetch_add(1, std::memory_order_relaxed);
		g_cs_ns.fetch_add(ns, std::memory_order_relaxed);
		auto max = g_cs_max_ns.load(std::memory_order_relaxed);
		while (ns > max && !g_cs_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
		}
	} else {
		g_gfx_translations.fetch_add(1, std::memory_order_relaxed);
		g_gfx_ns.fetch_add(ns, std::memory_order_relaxed);
	}
}

inline void Print(const char* why) {
	std::printf("ShaderSession (%s): startup: %llu compute pipelines prewarmed, %llu translations on disk | "
	            "in play: CS translated %llu (%llu first sight, %llu new permutation), CS translate %.0f ms total, "
	            "max %.0f ms; gfx translated %llu (%.0f ms); disk hits %llu; runtime-stride buffers %llu | "
	            "GpuWaitTimeout %llu, ErrorDeviceLost %llu\n",
	            why, static_cast<unsigned long long>(g_prewarm_compute.load()),
	            static_cast<unsigned long long>(g_xlat_loaded.load()),
	            static_cast<unsigned long long>(g_cs_first.load() + g_cs_perm.load()),
	            static_cast<unsigned long long>(g_cs_first.load()), static_cast<unsigned long long>(g_cs_perm.load()),
	            g_cs_ns.load() / 1e6, g_cs_max_ns.load() / 1e6,
	            static_cast<unsigned long long>(g_gfx_translations.load()), g_gfx_ns.load() / 1e6,
	            static_cast<unsigned long long>(g_xlat_hits.load()),
	            static_cast<unsigned long long>(g_runtime_stride_buffers.load()),
	            static_cast<unsigned long long>(g_wait_timeouts.load()),
	            static_cast<unsigned long long>(g_device_lost.load()));
	std::fflush(stdout);
}

// Called often (every gpuprof report); prints at most once per 30 s.
inline void MaybePrint() {
	static std::atomic<uint64_t> next {0};
	const auto now = NowNs();
	auto due = next.load(std::memory_order_relaxed);
	if (due == 0) {
		next.compare_exchange_strong(due, now + 30'000'000'000ull);
		return;
	}
	if (now < due || !next.compare_exchange_strong(due, now + 30'000'000'000ull)) return;
	Print("30 s");
}

inline void NoteDeviceLost() {
	g_device_lost.fetch_add(1, std::memory_order_relaxed);
	Print("device lost");
}

} // namespace ShaderSessionStats

#endif
