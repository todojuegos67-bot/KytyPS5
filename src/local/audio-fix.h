#pragma once
// [audio] (always): what cuts the sound for a moment, every 5 s and only when something happened:
//   underruns (the device found its queue empty), cap clears (latency cap restarted the queue: drops what was queued),
//   drain clears (queue dropped after 200 ms without draining: the device stalled), and the longest gap between two
//   outputs of a main port (>= 100 ms: the game's pump thread came late, i.e. starved or blocked).
// KYTY_AUDIO_FIX=1 (default off): the thread that feeds a main port runs at THREAD_PRIORITY_HIGHEST (it sleeps between
//   grains, as ncaPumpThread does on the console), and the two clears drop only the newest grain / stop waiting instead
//   of throwing away the whole queue (no ~0.1-1 s hole of silence).
#include "local/local-platform.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace AudioFix {

inline bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_AUDIO_FIX");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

inline std::atomic<uint32_t> g_underruns {0}, g_cap_clears {0}, g_drain_clears {0}, g_cap_kept {0}, g_drain_kept {0};
inline std::atomic<uint64_t> g_max_gap_us {0};

inline void NoteGap(uint64_t gap_us) {
	uint64_t old = g_max_gap_us.load(std::memory_order_relaxed);
	while (gap_us > old && !g_max_gap_us.compare_exchange_weak(old, gap_us, std::memory_order_relaxed)) {
	}
}

// Called by the thread that feeds a main port, once per output.
inline void OnMainOutput(uint64_t now_us) {
	if (Enabled()) {
		thread_local bool raised = false;
		if (!raised) {
			raised = true;
			LocalPlatform::MakeCriticalThread();
			std::printf("[audio] KYTY_AUDIO_FIX=1: hilo de audio en prioridad HIGHEST\n");
			std::fflush(stdout);
		}
	}
	static std::atomic<uint64_t> since {0};
	uint64_t                     start = since.load(std::memory_order_relaxed);
	if (start == 0) {
		since.store(now_us, std::memory_order_relaxed);
		return;
	}
	if (now_us - start < 5000000 || !since.compare_exchange_strong(start, now_us, std::memory_order_relaxed)) return;
	const uint32_t u = g_underruns.exchange(0), c = g_cap_clears.exchange(0), d = g_drain_clears.exchange(0);
	const uint32_t ck = g_cap_kept.exchange(0), dk = g_drain_kept.exchange(0);
	const uint64_t gap = g_max_gap_us.exchange(0);
	if (u == 0 && c == 0 && d == 0 && ck == 0 && dk == 0 && gap < 100000) return;
	std::printf("[audio] %.1f s: underruns %u, cola reiniciada por tope %u (evitado %u), vaciada sin drenar %u (evitado %u), "
	            "hueco max entre salidas %.0f ms, fix %s\n",
	            static_cast<double>(now_us) / 1e6, u, c, ck, d, dk, static_cast<double>(gap) / 1000.0,
	            Enabled() ? "on" : "off");
	std::fflush(stdout);
}

} // namespace AudioFix
