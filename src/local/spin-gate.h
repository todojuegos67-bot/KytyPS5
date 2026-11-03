#pragma once
// Whether the hand-off workers (Kyty.Record, Kyty.Upload) may spin for new work before they block. A spin spares the
// producer, the render thread, a wake-up system call per hand-off (at the 1-1 bench 7 recording-worker sleeps a frame
// instead of ~240: render thread ~0.25 ms a frame, 1-2% fps on the test PC's 22 CPUs) and burns ~1-1.3 CPUs between
// hand-offs. Where the game's threads want the CPUs, that burn is taken from them, the more so in the bursts of its
// job system: held to 6 of the test PC's cores, the 1-1 bench kept its fps but its 1% lows fell from 53-59 to 33-49 fps
// spinning; held to 4, 43-50 fps with lows of 3-5 spinning, 57-63 not. So the workers spin while the processors this
// process may run on keep three CPUs idle on average (measured every 250 ms, spinning included), and again from five
// (more than the spins' own burn: no flapping). kyty_local_spin_gate (KYTY_SPIN_GATE): 1 so, 0 always spin, 2 never.

#include "live-counters.h"
#include "local-platform.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>

extern "C" {
inline volatile std::atomic<uint32_t> kyty_local_spin_gate {0}; // (DeS port: 0 = always spin, the behaviour before the gate; upstream 1)
}

namespace SpinGate {

inline std::atomic<bool>    g_open {true};
inline std::atomic<int64_t> g_next_ns {0};
inline std::mutex           g_measuring;
// The last measurement (under g_measuring).
inline uint64_t g_last_idle = 0, g_last_cpus = 0;
inline int64_t  g_last_ns     = 0;
inline uint32_t g_transitions = 0;

inline void Measure(int64_t now) {
	uint64_t idle = 0, cpus = 0;
	if (!LocalPlatform::ProcessCpusIdleTime(&idle, &cpus)) return; // (unknown: stays open)
	if (g_last_ns != 0 && cpus == g_last_cpus && idle >= g_last_idle && now > g_last_ns) {
		const double idle_cpus = static_cast<double>(idle - g_last_idle) * 100.0 / static_cast<double>(now - g_last_ns);
		const bool   was_open  = g_open.load(std::memory_order_relaxed);
		if (const bool open = was_open ? idle_cpus >= 3.0 : idle_cpus >= 5.0; open != was_open) {
			g_open.store(open, std::memory_order_relaxed);
			if (!open) LiveCounters::Add(LiveCounters::SpinGateShuts);
			if (++g_transitions <= 16)
				std::printf("Spin gate: the hand-off workers %s (%.1f of %d CPUs idle)\n",
				            open ? "spin again" : "stop spinning", idle_cpus, std::popcount(cpus));
		}
	}
	g_last_idle = idle;
	g_last_cpus = cpus;
	g_last_ns   = now;
}

// A worker about to spin: whether it may (the recording worker measures: `measure`).
inline bool Open(bool measure) {
	if (const auto mode = kyty_local_spin_gate.load(std::memory_order_relaxed); mode != 1) return mode == 0;
	if (!measure) return g_open.load(std::memory_order_relaxed);
	const int64_t now =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	if (now >= g_next_ns.load(std::memory_order_relaxed) && g_measuring.try_lock()) {
		g_next_ns.store(now + 250'000'000, std::memory_order_relaxed);
		Measure(now);
		g_measuring.unlock();
	}
	return g_open.load(std::memory_order_relaxed);
}

} // namespace SpinGate
