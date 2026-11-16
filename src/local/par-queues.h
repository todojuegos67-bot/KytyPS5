#pragma once
// (par1) KYTY_PAR_QUEUES=1: the game's compute queues (ACB, GuestGpu queue ids 1..56) are translated on a thread of
// their own ("Kyty.Acb"), beside the GPU thread that translates the graphics queue (DCB, queue 0), the guest commands,
// the flips and the frame boundaries. 0 / unset: everything as before (one thread, this header does nothing).
//
// What is shared, and how (INFORME.md of par1 has the inventory):
// - Everything the translation of a packet touches outside the command processor's own registers (buffer cache,
//   texture cache, pipeline cache, GPU resource manager, command scheduler and its one open command buffer, the
//   Vulkan recording stream, the executors, the census) is used only while the thread holds the queue lock (Lock
//   below). Both threads hold it all the time they work, except across the register lane's runs
//   (graphicsRun.cpp, KYTY_REG_LANE): SET_SH_REG / SET_CONTEXT_REG / SET_UCONFIG_REG / INDEX_* / NUM_INSTANCES /
//   SET_BASE, whose handlers write only their own processor's registers (and atomics, thread-local shadows).
// - So the GPU sees the same thing as with one thread: both queues record into the one open command buffer, in the
//   order the lock gave, each packet whole. The game orders its queues with labels only (eboot: WRITE_DATA /
//   RELEASE_MEM of 1 into an 8-byte label, then WAIT_REG_MEM ==1 on the other queue): a label is written at
//   translation, under the lock, after the packets before it were recorded; the waiting queue records its packets
//   after it read the label. Same order as the one-thread interleaving, at a finer grain.
// - The render mutex (RenderMutex) gets two owners; only the queue lock's holder is inside it.
// - A queue blocked at a WAIT_REG_MEM waits for a label write of the other thread (WakeEpoch), not 100 us.
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace ParQueues {

// 0 off, 1 on (read once).
[[nodiscard]] inline int Mode() noexcept {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_PAR_QUEUES");
		return value != nullptr && value[0] == '1' ? 1 : 0;
	}();
	return mode;
}
[[nodiscard]] inline bool On() noexcept { return Mode() != 0; }

[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// The two translating threads: 0 the GPU thread (DCB), 1 the ACB thread. -1 any other thread.
inline thread_local int t_side = -1;

struct alignas(64) SideStats {
	std::atomic<uint64_t> work_ns {0};       // slices and guest commands, wall (lock waits and unlocked runs included)
	std::atomic<uint64_t> lock_wait_ns {0};  // waiting for the queue lock
	std::atomic<uint64_t> acquires {0};      // queue lock acquisitions
	std::atomic<uint64_t> contended {0};     // ... that found it taken
	std::atomic<uint64_t> unlocked_ns {0};   // register lane runs made without the lock (what may overlap)
	std::atomic<uint64_t> unlocked_runs {0};
	std::atomic<uint64_t> label_wait_ns {0}; // idle because its queue waits for a label (the other side's, mostly)
	std::atomic<uint64_t> label_waits {0};
	std::atomic<uint64_t> slices {0};
	std::atomic<uint64_t> cpu_handle {0};    // LocalPlatform::CurrentThreadHandle, for the CPU time
};
inline SideStats g_side[2];

// How long a waiter for the queue lock spins before it sleeps (KYTY_PAR_SPIN_US, default 20 us: a dispatch or a draw
// holds it ~1-3 us; a sleep costs a wake-up of ~5-20 us on Windows).
[[nodiscard]] inline uint64_t SpinNs() noexcept {
	static const uint64_t ns = [] {
		const char* value = std::getenv("KYTY_PAR_SPIN_US");
		const long  us    = value != nullptr ? std::strtol(value, nullptr, 10) : 20;
		return static_cast<uint64_t>(us < 0 ? 0 : us > 10000 ? 10000 : us) * 1000u;
	}();
	return ns;
}

// The queue lock: a ticket lock (FIFO: with two threads each waiter gets it next), spinning SpinNs, then
// atomic::wait (WaitOnAddress on Windows). Not recursive: Hold/Drop keep a thread-local flag.
class Lock {
public:
	void Acquire(int side) noexcept {
		const uint32_t ticket = m_next.fetch_add(1, std::memory_order_relaxed);
		uint32_t       serving = m_serving.load(std::memory_order_acquire);
		if (side >= 0) g_side[side].acquires.fetch_add(1, std::memory_order_relaxed);
		if (serving == ticket) return;
		const uint64_t start = NowNs();
		const uint64_t spin_until = start + SpinNs();
		bool           spinning   = true;
		if (side >= 0) g_side[side].contended.fetch_add(1, std::memory_order_relaxed);
		for (uint32_t spin = 1; serving != ticket; ++spin) {
			if (spinning && (spin & 63u) == 0 && NowNs() >= spin_until) spinning = false;
			if (spinning) {
				_mm_pause();
			} else {
				m_waiters.fetch_add(1, std::memory_order_seq_cst);
				serving = m_serving.load(std::memory_order_seq_cst);
				if (serving != ticket) m_serving.wait(serving, std::memory_order_seq_cst);
				m_waiters.fetch_sub(1, std::memory_order_relaxed);
			}
			serving = m_serving.load(std::memory_order_acquire);
		}
		if (side >= 0) g_side[side].lock_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	}
	void Release() noexcept {
		m_serving.fetch_add(1, std::memory_order_seq_cst);
		if (m_waiters.load(std::memory_order_seq_cst) != 0) m_serving.notify_all();
	}

private:
	alignas(64) std::atomic<uint32_t> m_next {0};
	alignas(64) std::atomic<uint32_t> m_serving {0};
	std::atomic<uint32_t>             m_waiters {0};
};
inline Lock             g_lock;
inline thread_local bool t_held = false;

[[nodiscard]] inline bool Held() noexcept { return t_held; }
// The calling translating thread takes the queue lock (nothing if off, not a translating thread, or held).
inline void Hold() noexcept {
	if (t_side < 0 || t_held) return;
	g_lock.Acquire(t_side);
	t_held = true;
}
inline void Drop() noexcept {
	if (t_side < 0 || !t_held) return;
	t_held = false;
	g_lock.Release();
}
// Held for a scope (a slice, a guest command), and its time counted as work.
class WorkScope {
public:
	WorkScope() noexcept: m_on(t_side >= 0), m_start(m_on ? NowNs() : 0) {
		if (m_on) Hold();
	}
	~WorkScope() {
		if (!m_on) return;
		Drop();
		g_side[t_side].work_ns.fetch_add(NowNs() - m_start, std::memory_order_relaxed);
	}
	WorkScope(const WorkScope&)            = delete;
	WorkScope& operator=(const WorkScope&) = delete;

private:
	bool     m_on;
	uint64_t m_start;
};
// Not held for a scope (a register lane run), held again after it.
class UnlockedScope {
public:
	UnlockedScope() noexcept: m_was(t_side >= 0 && t_held), m_start(m_was ? NowNs() : 0) {
		if (m_was) Drop();
	}
	~UnlockedScope() {
		if (!m_was) return;
		const uint64_t now = NowNs();
		g_side[t_side].unlocked_ns.fetch_add(now - m_start, std::memory_order_relaxed);
		g_side[t_side].unlocked_runs.fetch_add(1, std::memory_order_relaxed);
		Hold();
	}
	UnlockedScope(const UnlockedScope&)            = delete;
	UnlockedScope& operator=(const UnlockedScope&) = delete;

private:
	bool     m_was;
	uint64_t m_start;
};

// A translating thread's code that must hold the lock found it not held (a path this design did not foresee): told
// once on the console and the log, not fatal (KYTY_PAR_STRICT=1: fatal).
inline std::atomic<uint32_t> g_violations {0};
inline void CheckHeld(const char* where) noexcept {
	if (t_side < 0 || t_held) return;
	if (g_violations.fetch_add(1, std::memory_order_relaxed) == 0) {
		std::printf("par1: KYTY_PAR_QUEUES: %s without the queue lock on the %s thread (report this line)\n", where,
		            t_side == 0 ? "GPU" : "ACB");
		std::fflush(stdout);
	}
	static const bool strict = [] {
		const char* value = std::getenv("KYTY_PAR_STRICT");
		return value != nullptr && value[0] == '1';
	}();
	if (strict) std::abort();
}

// Label writes of a translating thread (CommandProcessor's guest memory writes): a queue waiting at a WAIT_REG_MEM
// checks again when this moves.
inline std::atomic<uint64_t> g_wake_epoch {0};
inline std::atomic<uint32_t> g_sleepers {0}; // threads waiting in the GuestGpu queue's condition variable for it
[[nodiscard]] inline uint64_t WakeEpoch() noexcept { return g_wake_epoch.load(std::memory_order_seq_cst); }
// Returns whether a thread sleeps (the caller then signals the condition variable under its mutex).
[[nodiscard]] inline bool NoteLabelWrite() noexcept {
	g_wake_epoch.fetch_add(1, std::memory_order_seq_cst);
	return g_sleepers.load(std::memory_order_seq_cst) != 0;
}

// The [par] line, every ~5 s (the GPU thread, after a slice). `frames`: the guest frames done so far; `cpu`: a thread's
// CPU seconds (LocalPlatform::ThreadCpuSeconds).
struct Snapshot {
	uint64_t values[2][9] {};
	double   cpu[2] {};
	uint64_t frames = 0;
	uint64_t ns     = 0;
};
inline Snapshot g_last;
template <typename CpuSeconds>
inline void MaybeReport(uint64_t frames, CpuSeconds cpu_seconds) {
	if (!On() || t_side != 0) return;
	const uint64_t now = NowNs();
	if (g_last.ns == 0) {
		g_last.ns     = now;
		g_last.frames = frames;
		return;
	}
	if (now - g_last.ns < 5'000'000'000ull) return;
	Snapshot current;
	current.ns     = now;
	current.frames = frames;
	for (int side = 0; side < 2; ++side) {
		const auto& s             = g_side[side];
		current.values[side][0] = s.work_ns.load(std::memory_order_relaxed);
		current.values[side][1] = s.lock_wait_ns.load(std::memory_order_relaxed);
		current.values[side][2] = s.acquires.load(std::memory_order_relaxed);
		current.values[side][3] = s.contended.load(std::memory_order_relaxed);
		current.values[side][4] = s.unlocked_ns.load(std::memory_order_relaxed);
		current.values[side][5] = s.unlocked_runs.load(std::memory_order_relaxed);
		current.values[side][6] = s.label_wait_ns.load(std::memory_order_relaxed);
		current.values[side][7] = s.label_waits.load(std::memory_order_relaxed);
		current.values[side][8] = s.slices.load(std::memory_order_relaxed);
		const uint64_t handle   = s.cpu_handle.load(std::memory_order_relaxed);
		current.cpu[side]       = handle != 0 ? cpu_seconds(handle) : 0.0;
	}
	const double f = current.frames > g_last.frames ? static_cast<double>(current.frames - g_last.frames) : 1.0;
	auto ms = [&](int side, int i) { return static_cast<double>(current.values[side][i] - g_last.values[side][i]) / 1e6 / f; };
	auto per = [&](int side, int i) { return static_cast<double>(current.values[side][i] - g_last.values[side][i]) / f; };
	auto cpu = [&](int side) { return (current.cpu[side] - g_last.cpu[side]) * 1e3 / f; };
	std::printf("[par] %.1f s, %" PRIu64 " frames | gfx work %.2f ms/frame (lock wait %.2f, %.0f acquires, %.0f contended; "
	            "unlocked %.2f in %.0f runs; label wait %.2f, %.0f waits; %.0f slices) | acb work %.2f (lock wait %.2f, "
	            "%.0f acquires, %.0f contended; unlocked %.2f in %.0f runs; label wait %.2f, %.0f waits; %.0f slices) | "
	            "cpu gfx %.2f acb %.2f sum %.2f ms/frame | violations %u\n",
	            static_cast<double>(now - g_last.ns) / 1e9, current.frames - g_last.frames, ms(0, 0), ms(0, 1), per(0, 2),
	            per(0, 3), ms(0, 4), per(0, 5), ms(0, 6), per(0, 7), per(0, 8), ms(1, 0), ms(1, 1), per(1, 2), per(1, 3),
	            ms(1, 4), per(1, 5), ms(1, 6), per(1, 7), per(1, 8), cpu(0), cpu(1), cpu(0) + cpu(1),
	            g_violations.load(std::memory_order_relaxed));
	std::fflush(stdout);
	g_last = current;
}

} // namespace ParQueues
