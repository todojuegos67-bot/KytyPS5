#ifndef EMULATOR_SRC_GRAPHICS_PHASE_STATS_H_
#define EMULATOR_SRC_GRAPHICS_PHASE_STATS_H_
// Diagnostic: KYTY_PHASE_STATS=1 (live, default 0). Times the phases of Thread_Gpu and the waits of
// guest threads on the emulator, and prints "Phases (5 s)" every 5 s: total ms, % of wall time,
// calls and us per call. Off, every scope costs one relaxed load. Changes no behaviour.
//   gpu.*    : Thread_Gpu. idle = waiting for work; cmd = SendCommandSync/SendCommand work run for
//              other threads (readbacks, unmaps, ...); gfx/compute = PM4 processing of a submission
//              (includes the translation, the submits and the GPU waits below); flush_slices;
//              wait = blocking GPU waits on Thread_Gpu; submit = vkQueueSubmit on any thread.
//   render.* : the guest render thread (KYTY_PROF_GUEST, default "Render Thread"); other.* : every
//              other guest thread. sync = round trip of a command to Thread_Gpu (queue + run);
//              fault = a page fault on GPU-written memory served by the buffer cache (whole fault);
//              gpuwait = blocking GPU waits on that thread.
#include "common/liveSwitches.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace PhaseStats {

enum Phase : int {
	GpuIdle,
	GpuCommand,
	GpuGraphics,
	GpuCompute,
	GpuFlushSlices,
	GpuWait,
	QueueSubmit,
	RenderSync,
	RenderFault,
	RenderGpuWait,
	OtherSync,
	OtherFault,
	OtherGpuWait,
	DrawTotal,
	DrawLock,
	DrawState,
	DrawBindings,
	DrawVertex,
	DrawPipeline,
	DrawTargets,
	DrawCommit,
	DispatchTotal,
	DispatchBindings,
	Count
};

inline const char* const kNames[Count] = {"gpu.idle",       "gpu.cmd",       "gpu.gfx",        "gpu.compute",
                                          "gpu.flush_slices", "gpu.wait",    "submit",         "render.sync",
                                          "render.fault",   "render.gpuwait", "other.sync",    "other.fault",
                                          "other.gpuwait",
                                          "draw.total", "draw.lock", "draw.state", "draw.bindings", "draw.vertex",
                                          "draw.pipeline", "draw.targets", "draw.commit", "dispatch.total", "dispatch.bindings"};

inline std::atomic<uint64_t> g_us[Count] {};
inline std::atomic<uint64_t> g_calls[Count] {};
inline thread_local int      t_role = 0; // 0 other guest/host thread, 1 Thread_Gpu, 2 guest render thread

inline bool Enabled() {
	static auto& on = Common::LiveSwitches::Get("KYTY_PHASE_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}
inline void MarkGpuThread() { t_role = 1; }
inline void MarkGuestRenderThread() { t_role = 2; }
inline bool IsGpuThread() { return t_role == 1; }
inline bool IsRenderThread() { return t_role == 2; }

// A guest-side phase picked by the calling thread's role; Count (not timed) on Thread_Gpu.
inline Phase GuestPhase(Phase render, Phase other) {
	return IsGpuThread() ? Count : (IsRenderThread() ? render : other);
}

inline uint64_t NowUs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Scope {
	explicit Scope(Phase p): phase(p), begin(p != Count && Enabled() ? NowUs() : 0) {}
	~Scope() {
		if (begin != 0) {
			g_us[phase].fetch_add(NowUs() - begin, std::memory_order_relaxed);
			g_calls[phase].fetch_add(1, std::memory_order_relaxed);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
	Phase    phase;
	uint64_t begin;
};

// Called from Thread_Gpu's loop; prints at most every 5 s.
inline void MaybeReport() {
	if (!Enabled()) {
		return;
	}
	static uint64_t window = 0;
	const auto      now    = NowUs();
	if (window == 0) {
		window = now;
		return;
	}
	if (now - window < 5'000'000) {
		return;
	}
	const double wall_us = static_cast<double>(now - window);
	window               = now;
	::printf("Phases (5 s): ms, %% of wall, calls, us/call\n");
	for (int i = 0; i < Count; i++) {
		const auto us    = g_us[i].exchange(0, std::memory_order_relaxed);
		const auto calls = g_calls[i].exchange(0, std::memory_order_relaxed);
		if (calls == 0) {
			continue;
		}
		::printf("  %-17s %9.1f ms %6.1f%% %8llu %9.1f\n", kNames[i], static_cast<double>(us) / 1000.0,
		         100.0 * static_cast<double>(us) / wall_us, static_cast<unsigned long long>(calls),
		         static_cast<double>(us) / static_cast<double>(calls));
	}
	std::fflush(stdout);
}

} // namespace PhaseStats

#endif
