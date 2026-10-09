#pragma once
// Local diagnostic: render-thread time per call kind and shader (live `census`).
// A scope costs one relaxed load while the census is off. The table is written only
// by the render thread; the live thread reads it after switching the census off.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <x86intrin.h>

namespace LiveCensus {

enum Kind : uint32_t {
	Dispatch      = 0, // a: compute shader, b: indirect
	Draw          = 1, // a: vertex (GS) shader, b: pixel shader
	NativeXpr     = 2,
	DispatchPhase = 3, // a: compute shader, b: phase
	GpuWait       = 4, // a, b: return addresses (caller, its caller)
	ReadbackWait  = 5, // a, b: return addresses
	SyncDownload  = 6, // a: 1 MiB granule, b: write access
	GraphicsPrograms = 7, // program lookup and SRT evaluation of a draw
	NativeGather  = 8, // native XPR record word gather
	QueueRun      = 9, // a: queue (interrupt event id; 0 graphics), whole PM4 run
	SrtInterpreter = 10, // a: shader hash, b: 1 no linear plan, 2 other sources, 3 other clean slots
	CommandSync   = 11, // GPU-written command memory check before parsing (indirect tables)
	DrawPhase     = 12, // a: pixel shader, b: phase (LogDrawPhase marks; DrawPhases below)
	Barrier       = 13, // a: return address of the recorded vkCmdPipelineBarrier(2), b: 1 for the 2 form
	GuestCommand  = 14, // a host command run for a guest thread (GuestGpu::ThreadRun)
	Submission    = 15, // one submission processed (GuestGpu::Process), whole
	Kinds         = 16
};

// Always on, render thread only: cycles and calls per kind since start (the slow-frame lines print the
// frame's share: where a busy frame went, without a live session).
inline uint64_t g_kind_cycles[Kinds] {};
inline uint64_t g_kind_calls[Kinds] {};
inline constexpr const char* KindNames[Kinds] = {"dispatch", "draw", "xpr", "dispatch_phase", "gpu_wait_c", "readback_wait_c",
                                                 "sync_download", "programs", "gather", "queue_run", "srt", "command_sync",
                                                 "draw_phase", "barrier", "guest_command", "submission"};

struct Entry {
	uint64_t a = 0, b = 0, calls = 0, cycles = 0;
	uint32_t kind = 0;
	bool     used = false;
};

constexpr size_t      TableSize = 32768;
inline std::atomic_bool g_on {false};
// Set on the render thread: other threads never touch the table.
inline thread_local bool g_render = false;
// The queue the render thread is running, ORed into every entry's b: 0 inside the
// graphics queue, QueueAsync inside an async compute queue, QueueNone outside both.
constexpr uint64_t      QueueAsync = 1ull << 63, QueueNone = 1ull << 62;
inline thread_local uint64_t g_queue = QueueNone;
inline Entry            g_table[TableSize];

inline void Add(uint32_t kind, uint64_t a, uint64_t b, uint64_t cycles) {
	uint64_t hash = (a * 0x9e3779b97f4a7c15ull) ^ (b * 0xc2b2ae3d27d4eb4full) ^ kind;
	for (size_t probe = 0; probe < TableSize; ++probe) {
		auto& entry = g_table[(hash + probe) & (TableSize - 1)];
		if (!entry.used) {
			entry = {a, b, 0, 0, kind, true};
		} else if (entry.a != a || entry.b != b || entry.kind != kind) {
			continue;
		}
		entry.calls += 1;
		entry.cycles += cycles;
		return;
	}
}

class Scope {
public:
	Scope(uint32_t kind, uint64_t a, uint64_t b = 0): m_on(g_render) {
		if (m_on) {
			m_kind  = kind;
			m_a     = a;
			m_b     = b | g_queue;
			m_table = g_on.load(std::memory_order_relaxed);
			m_start = __rdtsc();
		}
	}
	~Scope() {
		if (m_on) {
			const uint64_t cycles = __rdtsc() - m_start;
			g_kind_cycles[m_kind] += cycles;
			g_kind_calls[m_kind] += 1;
			if (m_table) Add(m_kind, m_a, m_b, cycles);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool     m_on;
	bool     m_table = false;
	uint32_t m_kind  = 0;
	uint64_t m_a     = 0, m_b = 0, m_start = 0;
};

// Render-thread waits, always summed (the slow-frame lines, LiveControl::Flip): for GPU work, for a guest
// readback's copy, in a synchronous download, in shader translation and pipeline creation. Steady-clock ns.
enum Wait : uint32_t { WaitGpu, WaitReadback, WaitDownload, WaitCompile, Waits };
inline int64_t g_waits_ns[Waits] {};
class WaitScope {
public:
	explicit WaitScope(Wait wait): m_wait(wait), m_on(g_render) {
		if (m_on) m_start = std::chrono::steady_clock::now();
	}
	~WaitScope() {
		if (m_on) g_waits_ns[m_wait] += (std::chrono::steady_clock::now() - m_start).count();
	}
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	Wait                                  m_wait;
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};

// Phases of one normal-path draw: the draw's scope starts phase 0, every LogDrawPhase marker
// (renderer/debug.cpp) closes the running phase and starts the next, the scope closes the last.
inline thread_local bool     g_draw_phases  = false;
inline thread_local uint64_t g_draw_shader  = 0, g_draw_phase_start = 0;
inline thread_local uint32_t g_draw_phase   = 0;
inline void MarkDrawPhase(uint32_t next) {
	if (!g_draw_phases) return;
	const auto now = __rdtsc();
	Add(DrawPhase, g_draw_shader, g_draw_phase | g_queue, now - g_draw_phase_start);
	g_draw_phase       = next;
	g_draw_phase_start = now;
}
class DrawPhases {
public:
	explicit DrawPhases(uint64_t shader): m_on(g_on.load(std::memory_order_relaxed) && g_render && !g_draw_phases) {
		if (m_on) {
			g_draw_phases      = true;
			g_draw_shader      = shader;
			g_draw_phase       = 0;
			g_draw_phase_start = __rdtsc();
		}
	}
	~DrawPhases() {
		if (m_on) {
			MarkDrawPhase(0);
			g_draw_phases = false;
		}
	}
	DrawPhases(const DrawPhases&)            = delete;
	DrawPhases& operator=(const DrawPhases&) = delete;

private:
	bool m_on;
};

} // namespace LiveCensus
