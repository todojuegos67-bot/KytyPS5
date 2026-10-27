#ifndef EMULATOR_SRC_GRAPHICS_GPU_PROF_H_
#define EMULATOR_SRC_GRAPHICS_GPU_PROF_H_
// [gpuprof]: always-on, cheap host-side split of Thread_Gpu work (Yotei: Thread_Gpu at 100% CPU
// with the GPU at ~2%). Every 5 s prints the top phases with ms per flip. KYTY_GPUPROF=0 = off.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace GpuProf {

enum Phase : int { CsProgram, CsPipeline, CsRecord, DrawPrograms, DrawPipeline, DrawRecord, Count };
inline const char* const kNames[Count] = {"cs.program(translate/materialize)", "cs.pipeline", "cs.record",
                                          "draw.programs", "draw.pipeline", "draw.record"};
inline std::atomic<uint64_t> g_ns[Count] {};
inline std::atomic<uint64_t> g_calls[Count] {};
inline std::atomic<uint64_t> g_flips {0};

inline bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_GPUPROF");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

inline uint64_t Now() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void Report() {
	static std::atomic<uint64_t> last {0};
	const auto now = Now();
	auto       prev = last.load(std::memory_order_relaxed);
	if (prev == 0) {
		last.store(now);
		return;
	}
	if (now - prev < 5'000'000'000ull || !last.compare_exchange_strong(prev, now)) return;
	std::array<std::pair<uint64_t, int>, Count> v {};
	for (int i = 0; i < Count; ++i) v[i] = {g_ns[i].exchange(0), i};
	std::array<uint64_t, Count> calls {};
	for (int i = 0; i < Count; ++i) calls[i] = g_calls[i].exchange(0);
	const auto flips = std::max<uint64_t>(1, g_flips.exchange(0));
	std::sort(v.begin(), v.end(), [](auto a, auto b) { return a.first > b.first; });
	std::printf("[gpuprof] 5 s, %llu flips:", static_cast<unsigned long long>(flips));
	for (int k = 0; k < 5; ++k) {
		const auto [ns, i] = v[k];
		std::printf(" %s=%.0f ms (%.1f ms/flip, %llu calls)", kNames[i], ns / 1e6, ns / 1e6 / flips,
		            static_cast<unsigned long long>(calls[i]));
	}
	std::printf("\n");
	std::fflush(stdout);
}

struct Scope {
	explicit Scope(Phase p): phase(p), begin(Enabled() ? Now() : 0) {}
	void Switch(Phase p) {
		if (begin == 0) return;
		const auto now = Now();
		g_ns[phase].fetch_add(now - begin, std::memory_order_relaxed);
		g_calls[phase].fetch_add(1, std::memory_order_relaxed);
		phase = p;
		begin = now;
	}
	~Scope() {
		if (begin == 0) return;
		g_ns[phase].fetch_add(Now() - begin, std::memory_order_relaxed);
		g_calls[phase].fetch_add(1, std::memory_order_relaxed);
		Report();
	}
	Phase    phase;
	uint64_t begin;
};

inline void NoteFlip() { g_flips.fetch_add(1, std::memory_order_relaxed); }

} // namespace GpuProf

#endif
