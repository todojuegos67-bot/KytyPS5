#pragma once
// (A0b) Diagnostic instrumentation of the translation queues. Off unless KYTY_AUDIT is set.
//   KYTY_AUDIT=1: on in every 5 s window.
//   KYTY_AUDIT=2: on and off alternate by window (window 0 is on, window 1 off, ...), in one process, so the off
//                 windows are the control of the same run. Each report line says which state its window was in.
//   Off (default): each hook is one relaxed load and a branch that is not taken.
// What it measures (counts exact, times sampled):
//   side 0: the GPU thread (DCB) and side 1: the ACB thread (queue_id != 0), as recorded by the command encoder
//           (Writer::Command in vulkan-recording.cpp): calls, sampled time per call.
//   side 2: the recording worker: replays (the driver calls, one per recorded command), sampled time per replay,
//           and its idle time (the bursts it waits for a published chunk).
//   lock:   the queue lock's spin and blocking time per side, and its contended acquisitions (par-queues.h).
// Nothing here locks, waits, reorders or changes what is encoded, replayed or submitted. A sampled time costs two
// clock reads; one call in kSample takes them. Estimated totals: calls x (sampled ns / samples), reported per window.
//
// Window accounting: the counters are monotonic; a window is the difference between two snapshots taken on the
// render thread at a flip (OnFlip). An event racing a snapshot can land in the next window: a bound of a few events.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ParAudit {

constexpr uint32_t kSample = 64; // one timed call in kSample
constexpr int      kSides  = 3;  // 0 GPU thread (DCB), 1 ACB thread, 2 recording worker
enum Stat : int { Calls, Samples, SampledNs, SpinNs, BlockNs, Waits, WaitNs, StatCount };

[[nodiscard]] inline int ModeFromEnv() noexcept {
	const char* value = std::getenv("KYTY_AUDIT");
	return value != nullptr && (value[0] == '1' || value[0] == '2') ? value[0] - '0' : 0;
}
inline const int                 g_mode   = ModeFromEnv();
inline std::atomic<bool>         g_active {g_mode != 0};
// One cache line per side (8 counters of 8 bytes): the GPU and ACB threads never share a line.
alignas(64) inline std::atomic<uint64_t> g_stat[kSides][8] {};

[[nodiscard]] inline bool Active() noexcept { return g_active.load(std::memory_order_relaxed); }

[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline thread_local uint32_t t_seq[kSides] {}; // this thread's timed-call sequence per kind (only while on)

// One call of a timed kind (encoder: side 0 or 1; replay: side 2). Counts it, times one call in kSample.
class Timed {
public:
	explicit Timed(int side) noexcept : m_side(side) {
		if (!Active()) return;
		g_stat[m_side][Calls].fetch_add(1, std::memory_order_relaxed);
		if ((++t_seq[m_side] % kSample) == 0) m_t0 = NowNs();
	}
	~Timed() {
		if (m_t0 == 0) return;
		g_stat[m_side][Samples].fetch_add(1, std::memory_order_relaxed);
		g_stat[m_side][SampledNs].fetch_add(NowNs() - m_t0, std::memory_order_relaxed);
	}
	Timed(const Timed&)            = delete;
	Timed& operator=(const Timed&) = delete;

private:
	int      m_side;
	uint64_t m_t0 = 0;
};

// The recording worker's wait for a published chunk: counted only when it found nothing to run at entry.
class Idle {
public:
	explicit Idle(bool empty) noexcept {
		if (empty && Active()) m_t0 = NowNs();
	}
	~Idle() {
		if (m_t0 == 0) return;
		g_stat[2][Waits].fetch_add(1, std::memory_order_relaxed);
		g_stat[2][WaitNs].fetch_add(NowNs() - m_t0, std::memory_order_relaxed);
	}
	Idle(const Idle&)            = delete;
	Idle& operator=(const Idle&) = delete;

private:
	uint64_t m_t0 = 0;
};

// A contended queue-lock acquisition on `side`: spin from `start` to `block_start` (or to `end` if it never slept),
// block from `block_start` to `end`. Called only on the contended path, with the caller's own clock reads.
inline void LockWait(int side, uint64_t start, uint64_t block_start, uint64_t end) noexcept {
	if (!Active() || side < 0 || side > 1) return;
	const uint64_t spin_end = block_start != 0 ? block_start : end;
	g_stat[side][SpinNs].fetch_add(spin_end - start, std::memory_order_relaxed);
	if (block_start != 0) g_stat[side][BlockNs].fetch_add(end - block_start, std::memory_order_relaxed);
	g_stat[side][Waits].fetch_add(1, std::memory_order_relaxed);
}

// The window clock, called once per flip on the render thread (the only caller).
namespace Window {
inline uint64_t g_start = 0;
inline uint64_t g_last_flip = 0;
inline uint64_t g_max_flip_ns = 0;
inline uint64_t g_frames = 0;
inline int      g_index = 0;
inline uint64_t g_prev[kSides][StatCount] {};
}

inline void OnFlip() noexcept {
	if (g_mode == 0) return;
	const uint64_t now = NowNs();
	if (Window::g_start == 0) { // the first flip: the baseline, so nothing before it counts in window 0
		Window::g_start     = now;
		Window::g_last_flip = now;
		for (int s = 0; s < kSides; ++s)
			for (int i = 0; i < StatCount; ++i) Window::g_prev[s][i] = g_stat[s][i].load(std::memory_order_relaxed);
		return;
	}
	++Window::g_frames;
	if (Window::g_last_flip != 0 && now - Window::g_last_flip > Window::g_max_flip_ns) Window::g_max_flip_ns = now - Window::g_last_flip;
	Window::g_last_flip = now;
	if (now - Window::g_start < 5'000'000'000ull) return;

	uint64_t cur[kSides][StatCount];
	for (int s = 0; s < kSides; ++s)
		for (int i = 0; i < StatCount; ++i) cur[s][i] = g_stat[s][i].load(std::memory_order_relaxed);
	auto d = [&](int s, int i) { return static_cast<double>(cur[s][i] - Window::g_prev[s][i]); };
	auto mean_ns = [&](int s) {
		const double n = d(s, Samples);
		return n > 0 ? d(s, SampledNs) / n : 0.0;
	};
	auto est_ms = [&](int s, int calls_stat) { return d(s, calls_stat) * mean_ns(s) / 1e6; };

	const bool on = Window::g_index % 2 == 0 || g_mode == 1;
	const double secs = static_cast<double>(now - Window::g_start) / 1e9;
	std::printf("[audit] win=%d state=%s t=%.1fs frames=%llu max_flip_ms=%.1f | "
	            "dcb enc_calls=%.0f enc_samples=%.0f enc_mean_ns=%.0f enc_est_ms=%.2f lock_spin_ms=%.2f lock_block_ms=%.2f lock_waits=%.0f | "
	            "acb enc_calls=%.0f enc_samples=%.0f enc_mean_ns=%.0f enc_est_ms=%.2f lock_spin_ms=%.2f lock_block_ms=%.2f lock_waits=%.0f | "
	            "worker replay_calls=%.0f replay_samples=%.0f replay_mean_ns=%.0f replay_est_ms=%.2f idle_ms=%.2f idle_waits=%.0f\n",
	            Window::g_index, on ? "ON" : "OFF", secs, static_cast<unsigned long long>(Window::g_frames),
	            static_cast<double>(Window::g_max_flip_ns) / 1e6,
	            d(0, Calls), d(0, Samples), mean_ns(0), est_ms(0, Calls), d(0, SpinNs) / 1e6, d(0, BlockNs) / 1e6, d(0, Waits),
	            d(1, Calls), d(1, Samples), mean_ns(1), est_ms(1, Calls), d(1, SpinNs) / 1e6, d(1, BlockNs) / 1e6, d(1, Waits),
	            d(2, Calls), d(2, Samples), mean_ns(2), est_ms(2, Calls), d(2, WaitNs) / 1e6, d(2, Waits));
	std::fflush(stdout);

	for (int s = 0; s < kSides; ++s)
		for (int i = 0; i < StatCount; ++i) Window::g_prev[s][i] = cur[s][i];
	Window::g_start      = now;
	Window::g_frames     = 0;
	Window::g_max_flip_ns = 0;
	++Window::g_index;
	if (g_mode == 2) g_active.store(Window::g_index % 2 == 0, std::memory_order_relaxed);
}

} // namespace ParAudit
