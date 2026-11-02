#pragma once
// KYTY_EARLY_SUBMIT=1 (A/B, off by default): the GPU work that writes the memory guest threads wait for most in read
// faults ([rfaults]' top 2 MiB regions, learned every 5 s) is handed to the driver at the GPU thread's next point
// between packets, not when the command buffer fills or the frame ends: their readback then waits for that work only,
// not for it to be submitted too. Only when work is sent changes (a Flush, as a guest readback makes at the same
// point), never what is recorded or read.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace EarlySubmit {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_EARLY_SUBMIT");
		const bool  enabled = v != nullptr && v[0] == '1';
		if (enabled) std::printf("GPU submits: work writing the memory the game waits for most is sent at once (KYTY_EARLY_SUBMIT=1)\n");
		return enabled;
	}();
	return on;
}

// The hot 2 MiB regions (address >> 21, + 1; 0 none), written by the [rfaults] report.
inline std::atomic<uint64_t> g_hot[8] {};
inline std::atomic<bool>     g_due {false};
inline std::atomic<int64_t>  g_last_ns {0};

[[nodiscard]] inline bool Hot(uint64_t vaddr, uint64_t size) {
	const uint64_t first = (vaddr >> 21u) + 1u, last = ((vaddr + (size != 0 ? size - 1 : 0)) >> 21u) + 1u;
	for (const auto& h: g_hot) {
		const uint64_t r = h.load(std::memory_order_relaxed);
		if (r != 0 && r >= first && r <= last) return true;
	}
	return false;
}

// GPU thread, between packets: true when a flush is due (at most one per 250 us).
[[nodiscard]] inline bool TakeDue() {
	if (!g_due.load(std::memory_order_relaxed)) return false;
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	if (now - g_last_ns.load(std::memory_order_relaxed) < 250000) return false;
	g_due.store(false, std::memory_order_relaxed);
	g_last_ns.store(now, std::memory_order_relaxed);
	return true;
}

// The current command buffer's start (CommandScheduler::BeginNext), for the readbacks' producer age.
inline std::atomic<int64_t> g_open_since_ns {0};

} // namespace EarlySubmit
