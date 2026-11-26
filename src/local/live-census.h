#pragma once
// Local diagnostic: render-thread time per call kind and shader (live `census`).
// A scope costs one relaxed load while the census is off. The table is written only
// by the render thread; the live thread reads it after switching the census off.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <x86intrin.h>

#include "blackbox.h"
#include "otros1.h"
#include "render5.h"
#include "sync-stats.h"
#include "native/g1-probe.h" // (g1: tq6 by pass)

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
// ... and the dispatch phases (renderCompute.cpp, phase = b & 15) and draw phases (LogDrawPhase marks).
constexpr size_t Phases = 16;
inline uint64_t g_dispatch_phase_cycles[Phases] {};
inline uint64_t g_draw_phase_cycles[Phases] {};
inline uint64_t g_xpr_cycles[4] {}; // NativeXpr scopes by their a (graphicsRun.cpp: 0 try/emit, 1 store, 2 direct)
// (render2) ... and the table path's phases (table-xpr.inc: DrawPhase scopes of a = 0x7ab1e for draws, 0x7ab1f for
// dispatches, by their b), for the [render] line (KYTY_RENDER_STATS, live-control.h).
inline uint64_t g_table_phase_cycles[2][Phases] {};
inline constexpr const char* DispatchPhaseNames[Phases] = {"program", "consume", "pipeline", "bindings", "find_buffers", "bda",
                                                           "rebind_buffers", "rebind_images", "commit", "barrier", "table", "linear_copy",
                                                           "lock", "p13", "p14", "p15"};
inline constexpr const char* DrawPhaseNames[Phases] = {"start", "color_target", "depth_target", "programs", "bindings",
                                                       "vertex_buffers", "pipeline", "begin_rendering", "complete", "p9",
                                                       "p10", "p11", "p12", "p13", "p14", "p15"};
inline constexpr const char* KindNames[Kinds] = {"dispatch", "draw", "xpr", "dispatch_phase", "gpu_wait_c", "readback_wait_c",
                                                 "sync_download", "programs", "gather", "queue_run", "srt", "command_sync",
                                                 "draw_phase", "barrier", "guest_command", "submission"};

// (render3) KYTY_RENDER_STATS=1 (the [render] line, live-control.h): read once, at its first use on the render thread
// (the port sets its settings' variables after the process started, so not at static initialization). The scopes and
// counters below that only that line prints are taken only with it (StatScope, Pm4Clock): nothing but a predicted
// branch otherwise.
[[nodiscard]] inline bool RenderStatsOn() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_RENDER_STATS");
		return text != nullptr && text[0] != '\0' && !(text[0] == '0' && text[1] == '\0');
	}();
	return on;
}
// (render3) GpuResourceManager::PrepareBdaReadRanges' memo (KYTY_BDA_MEMO): calls it answered, calls that did the work.
// GPU thread only.
inline uint64_t g_bda_memo[2] {};
// (render3) Write windows opened in the big-window granules (KYTY_BIG_WINDOW; GpuResourceManager's fault handler, any
// thread): flat 1 MiB windows, grown windows (KYTY_BIG_WINDOW_GROW) and the grown windows' pages.
inline std::atomic<uint64_t> g_big_window[3] {};
// (render3) The PM4 walk by packet class (CommandProcessor::ProcessPm4, render thread, KYTY_RENDER_STATS only): the
// cycles from a packet's start to the next packet's (its draw, dispatch or handler included) and the packets.
enum Pm4Class : uint32_t {
	Pm4Draw, Pm4Dispatch, Pm4ShReg, Pm4ContextReg, Pm4UconfigReg, Pm4Nop, Pm4IndirectBuffer, Pm4Other, Pm4Classes
};
inline uint64_t g_pm4_cycles[Pm4Classes] {};
inline uint64_t g_pm4_packets[Pm4Classes] {};
inline constexpr const char* Pm4ClassNames[Pm4Classes] = {"draws", "dispatches", "sh_reg", "context_reg", "uconfig_reg",
                                                          "nop", "indirect_buffer", "rest"};
// (render4) KYTY_RENDER_STATS only, render thread: the dispatches (the Dispatch scopes) and the whole PM4 runs (the
// QueueRun scopes) by queue, 0 the graphics queue (DCB) and 1 an async compute queue (ACB), calls and cycles; and the
// dispatches by compute shader. The shader is the program's hash when the dispatch's path knows it (g_dispatch_shader,
// set by the table path and the normal path once they have the program), LinearCopyShader for a dispatch the linear
// copy's register check took (no program there), else the shader's address.
inline uint64_t g_queue_dispatches[2][2] {}; // [queue][0 calls, 1 cycles]
inline uint64_t g_queue_runs[2][2] {};
inline uint64_t g_dispatch_shader = 0;
constexpr uint64_t LinearCopyShader = 1;
// (The render thread's dispatches only: g_render is declared below.)
inline void SetDispatchShader(uint64_t shader);
struct ShaderStat {
	uint64_t shader = 0, calls = 0, cycles = 0;
};
constexpr size_t  ShaderStats = 2048; // (open addressing, 16 probes; a 5 s window sees a few hundred shaders)
inline ShaderStat g_shader_stats[ShaderStats] {};
inline uint64_t   g_shader_stats_dropped = 0;
inline void NoteDispatchShader(uint64_t shader, uint64_t cycles) {
	const uint64_t hash = shader * 0x9e3779b97f4a7c15ull >> 32u;
	for (size_t probe = 0; probe < 16; ++probe) {
		auto& entry = g_shader_stats[(hash + probe) & (ShaderStats - 1)];
		if (entry.calls != 0 && entry.shader != shader) continue;
		entry.shader = shader;
		entry.calls += 1;
		entry.cycles += cycles;
		return;
	}
	++g_shader_stats_dropped;
}
// (render4) KYTY_RENDER_STATS only, render thread: what the table path's blocks (RenderExecutor::TableBlocks, the
// table dispatches' phase 6 and the table draws' phase 13) did with the V#s: 0 written ones, 1 those of them that went
// through BufferCache::ObtainBuffer (no memo), 2 read-only ones, 3 those of them resolved again (not held), 4 those
// whose range was proved clean again (another page of its regions written).
inline uint64_t g_table_blocks[5] {};
// (thread1) KYTY_RENDER_STATS only, render thread: what a frame's draws are. The NativeXpr scopes' calls by their a
// (g_xpr_cycles has their time): 0 runs of DRAW_INDEX_INDIRECT and multi draws, 1 direct indexed draws, 2 draws of
// vertices. The calls a clean continuation took (RenderExecutor::TableContinue): 0 runs, 1 direct draws. And the table
// draws by what they draw into (RenderExecutor::TableDraw), calls and draws: a depth image of 4 layers or more alone
// (the sun's cascades), another depth image alone (depth passes, other shadows), anything with a color target.
inline uint64_t g_xpr_calls[4] {};
inline uint64_t g_continued_calls[2] {};
enum DrawTarget : uint32_t { DrawDepthLayered, DrawDepthOnly, DrawColor, DrawTargets };
inline uint64_t g_draw_targets[DrawTargets][2] {}; // [target][0 calls, 1 draws]
inline constexpr const char* DrawTargetNames[DrawTargets] = {"depth_layered", "depth", "color"};

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
inline void SetDispatchShader(uint64_t shader) {
	if (g_render) g_dispatch_shader = shader;
}
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

// (thread1) KYTY_RENDER_STATS only, render thread, measurement only (nothing is run differently): what the frame's
// translation would take with the async compute queues' submissions (ACB) on a thread of their own and the graphics
// queue's (DCB) on another, at no cost for the hand-over - the ceiling of that split. Each side has a virtual clock
// that advances by the time its slices take here (GuestGpu::ThreadRun, around GuestGpu::Process). The game orders the
// queues with labels only (a pass writes its label at its end, the passes that need it start with WAIT_REG_MEM on it:
// eboot 0x9151d0 / 0x912890 write, sceAgcAcbWaitRegMem / sceAgcDcbWaitRegMem wait): a memory write of a command
// processor notes its side's virtual time there (Write), and a wait found satisfied (Waited) holds its side back to
// the time of the last write at its address when that is later - the other side had not got there yet. The flip
// brings both clocks together (Frame). The [render] line prints, a frame: the slices' time (serial), each side's,
// the virtual time both sides took (two_threads) and the time a side was held back for the other (stalled).
// Not modelled (so the real split would gain less): when the game submits each command buffer, the frame boundary's
// order (GuestGpu::CanProcessSubmission), the caches both sides share, and what the game reads back between them.
namespace Par2 {
struct Label {
	uint64_t address = 0, time = 0;
};
constexpr size_t Labels = 4096; // (a slot an address: a collision forgets the older one, whose waits then hold nothing)
inline Label     g_labels[Labels] {};
inline bool      g_active = false;
inline uint32_t  g_side   = 0;
inline uint64_t  g_real_start = 0, g_virtual_start = 0;
inline uint64_t  g_clock[2] {};       // each side's virtual clock (TSC cycles)
inline uint64_t  g_side_cycles[2] {}; // each side's slices, cycles
inline uint64_t  g_slices[2][2] {};   // [side][0 slices, 1 those that stopped at a wait]
inline uint64_t  g_waits[2] {};       // waits found satisfied at an address written here: 0 all, 1 that held their side back
inline uint64_t  g_stalled = 0;       // cycles a side was held back
inline uint64_t  g_elapsed = 0;       // virtual cycles of the frames ended so far (both sides)
inline uint64_t  g_frame_start = 0;   // both clocks at the last flip
[[nodiscard]] inline size_t SlotOf(uint64_t address) {
	return static_cast<size_t>(((address >> 2u) * 0x9e3779b97f4a7c15ull) >> 52u) & (Labels - 1);
}
// A slice of a submission begins: 0 the graphics queue, 1 an async compute queue.
inline void Begin(uint32_t side) {
	if (!g_render || !RenderStatsOn()) return;
	g_active        = true;
	g_side          = side & 1u;
	g_virtual_start = g_clock[g_side];
	g_real_start    = __rdtsc();
}
// ... and ends, done or stopped at a wait.
inline void End(bool complete) {
	if (!g_render || !g_active) return;
	const uint64_t cycles = __rdtsc() - g_real_start;
	g_clock[g_side]       = g_virtual_start + cycles;
	g_side_cycles[g_side] += cycles;
	g_slices[g_side][0] += 1;
	if (!complete) g_slices[g_side][1] += 1;
	g_active = false;
}
// A command processor wrote guest memory at `address` (a label, an end-of-pipe value).
inline void Write(uint64_t address) {
	if (!g_render || !g_active) return;
	g_labels[SlotOf(address)] = {address & ~uint64_t {3}, g_virtual_start + (__rdtsc() - g_real_start)};
}
// A WAIT_REG_MEM at `address` was found satisfied.
inline void Waited(uint64_t address) {
	if (!g_render || !g_active) return;
	const auto& label = g_labels[SlotOf(address)];
	if (label.address != (address & ~uint64_t {3}) || label.time == 0) return;
	g_waits[0] += 1;
	const uint64_t now = g_virtual_start + (__rdtsc() - g_real_start);
	if (label.time <= now) return;
	g_waits[1] += 1;
	g_stalled += label.time - now;
	g_virtual_start += label.time - now;
}
// (trust1) The game's flip (CommandProcessor::Flip -> LiveControl::Flip), noted while its slice still runs: the slice's
// end closes the frame (TakeFlip, GuestGpu::ThreadRun). Frame() was only called for FlipPreparation submissions,
// which a game session never makes: two_threads stayed 0 and the clocks never met again (stalled ~ the whole frame).
inline bool g_flip_pending = false;
inline void Flipped() {
	if (g_render && RenderStatsOn()) g_flip_pending = true;
}
[[nodiscard]] inline bool TakeFlip() {
	const bool flipped = g_flip_pending;
	g_flip_pending     = false;
	return flipped;
}
// The flip: both sides are done with the frame.
inline void Frame() {
	if (!g_render || !RenderStatsOn()) return;
	const uint64_t end = g_clock[0] > g_clock[1] ? g_clock[0] : g_clock[1];
	g_elapsed += end - g_frame_start;
	g_clock[0] = g_clock[1] = g_frame_start = end;
}
} // namespace Par2

// (render5) KYTY_CENSUS_LIGHT (render5.h). The scopes of these kinds come once or several times a draw or dispatch (at
// Boletaria ~40 000 a frame: NativeXpr, the table path's DrawPhase scopes 0-4 and 9, Dispatch, DispatchPhase, the
// blocks' resolve and reprove scopes...); what they sum is read only by the [render] line (KYTY_RENDER_STATS), the
// live census (g_on) and the SLOW Frame lines' breakdown. With 1 they are not taken while neither of the first two
// reads them (the SLOW Frame lines then show no time for these kinds); the waits, queue runs, submissions and guest
// commands are always taken. With 2 every scope is taken and those 1 would skip are counted, with a scope's cost
// measured once at start (the [render5] line: what 1 would save). Nothing but these sums changes: no scope decides
// anything that is drawn. Render thread only (a scope is taken only there, g_render).
namespace Light {
constexpr uint32_t LightKinds = (1u << Dispatch) | (1u << Draw) | (1u << NativeXpr) | (1u << DispatchPhase) |
                           (1u << GraphicsPrograms) | (1u << NativeGather) | (1u << SrtInterpreter) | (1u << CommandSync) |
                           (1u << DrawPhase) | (1u << Barrier);
// -1 not read yet, 0 off (or KYTY_RENDER_STATS on with 1: its line needs every scope), 1 skip, 2 count.
inline int32_t g_state = -1;
// A scope's own cost, as Scope takes it (thread-local flags, two TSC reads, the sums): TSC cycles each, the mean of
// 64 rounds of 64.
[[nodiscard]] inline uint64_t MeasureScopeCycles() {
	static uint64_t sink_cycles[16] {}, sink_calls[16] {};
	uint64_t        best = UINT64_MAX;
	for (uint32_t round = 0; round < 64; ++round) {
		const uint64_t start = __rdtsc();
		for (uint32_t i = 0; i < 64; ++i) {
			const bool     on    = g_render;
			const uint64_t queue = g_queue;
			const uint64_t begin = __rdtsc();
			const uint32_t kind  = (i & 7u) + (on ? 0u : 8u);
			const uint64_t spent = __rdtsc() - begin + (queue & 1u);
			sink_cycles[kind & 15u] += spent;
			sink_calls[kind & 15u] += 1;
		}
		const uint64_t cycles = (__rdtsc() - start) / 64u;
		if (cycles < best) best = cycles;
	}
	return best;
}
inline int32_t Init() {
	const auto mode = Render5::Mode(Render5::CensusLight);
	int32_t    state = static_cast<int32_t>(mode);
	if (mode == 1 && RenderStatsOn()) {
		state = 0;
		std::printf("render5: KYTY_CENSUS_LIGHT=1 does nothing while KYTY_RENDER_STATS is on (its [render] line reads every "
		            "scope); compare the [fps] lines' cpu without KYTY_RENDER_STATS\n");
		std::fflush(stdout);
	}
	if (mode == 2) {
		Render5::g_census_scope_cycles = MeasureScopeCycles();
		std::printf("render5: KYTY_CENSUS_LIGHT=2: a census scope costs ~%" PRIu64 " TSC cycles on this thread\n",
		            Render5::g_census_scope_cycles);
		std::fflush(stdout);
	}
	g_state = state;
	return state;
}
// A scope of `kind` is not taken (counted with 2).
[[nodiscard]] inline bool Skips(uint32_t kind) {
	if (kind >= 32u || ((LightKinds >> kind) & 1u) == 0) return false;
	int32_t state = g_state;
	if (state < 0) state = Init();
	if (state == 0) return false;
	if (state == 2) {
		++Render5::g_census[Render5::CensusWouldSkip];
		return false;
	}
	if (g_on.load(std::memory_order_relaxed)) return false; // (the live census table wants them all)
	++Render5::g_census[Render5::CensusSkipped];
	return true;
}
} // namespace Light

class Scope {
public:
	Scope(uint32_t kind, uint64_t a, uint64_t b = 0): m_on(g_render) {
		if (m_on) {
			if (Light::Skips(kind)) { // (render5, KYTY_CENSUS_LIGHT)
				m_on = false;
				return;
			}
			m_kind  = kind;
			m_a     = a;
			m_b     = b | g_queue;
			m_table = g_on.load(std::memory_order_relaxed);
			// (render4) The dispatch's or the PM4 run's share by queue, and the dispatch's shader.
			m_stats = (kind == Dispatch || kind == QueueRun) && RenderStatsOn();
			if (m_stats && kind == Dispatch) g_dispatch_shader = 0;
			m_start = __rdtsc();
		}
	}
	~Scope() {
		if (m_on) {
			const uint64_t cycles = __rdtsc() - m_start;
			g_kind_cycles[m_kind] += cycles;
			g_kind_calls[m_kind] += 1;
			if (m_kind == DispatchPhase) g_dispatch_phase_cycles[m_b & (Phases - 1)] += cycles;
			if (m_kind == NativeXpr) {
				g_xpr_cycles[m_a & 3] += cycles;
				g_xpr_calls[m_a & 3] += 1; // (thread1)
			}
			if (m_kind == DrawPhase && (m_a | 1u) == 0x7ab1fu) {
				g_table_phase_cycles[m_a & 1u][m_b & (Phases - 1)] += cycles;
				if (m_a == 0x7ab1fu && (m_b & (Phases - 1)) == 6) G1::NoteTq6(cycles); // (g1: tq6 by pass)
			}
			if (m_stats) {
				auto& by_queue = m_kind == Dispatch ? g_queue_dispatches : g_queue_runs;
				by_queue[(m_b & QueueAsync) != 0 ? 1 : 0][0] += 1;
				by_queue[(m_b & QueueAsync) != 0 ? 1 : 0][1] += cycles;
				if (m_kind == Dispatch) NoteDispatchShader(g_dispatch_shader != 0 ? g_dispatch_shader : m_a, cycles);
			}
			if (m_table) Add(m_kind, m_a, m_b, cycles);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool     m_on;
	bool     m_table = false;
	bool     m_stats = false;
	uint32_t m_kind  = 0;
	uint64_t m_a     = 0, m_b = 0, m_start = 0;
};

// (render3) As Scope for the table path's phases (a = 0x7ab1e draws, 0x7ab1f dispatches; b the phase), but taken only
// with KYTY_RENDER_STATS: the finer phases the [render] line prints (table-xpr.inc).
class StatScope {
public:
	// (render5: with KYTY_CENSUS_LIGHT=1 taking effect KYTY_RENDER_STATS is off: no need to ask it.)
	StatScope(uint32_t table, uint32_t phase): m_on(g_render && Light::g_state != 1 && RenderStatsOn()) {
		if (m_on) {
			m_table = table & 1u;
			m_phase = phase & (Phases - 1);
			m_start = __rdtsc();
		}
	}
	~StatScope() {
		if (m_on) g_table_phase_cycles[m_table][m_phase] += __rdtsc() - m_start;
	}
	StatScope(const StatScope&)            = delete;
	StatScope& operator=(const StatScope&) = delete;

private:
	bool     m_on;
	uint32_t m_table = 0, m_phase = 0;
	uint64_t m_start = 0;
};

// (render3) The PM4 walk's clock (see g_pm4_cycles). One per CommandProcessor::ProcessPm4 call; a nested call's time is
// its own (the outer one's clock stops meanwhile).
class Pm4Clock;
inline thread_local Pm4Clock* t_pm4_clock = nullptr;
class Pm4Clock {
public:
	Pm4Clock(): m_on(g_render && RenderStatsOn()) {
		if (m_on) {
			m_parent = t_pm4_clock;
			m_start  = __rdtsc();
			if (m_parent != nullptr) m_parent->Mark(m_start);
			t_pm4_clock = this;
		}
	}
	~Pm4Clock() {
		if (m_on) {
			const uint64_t now = __rdtsc();
			Mark(now);
			t_pm4_clock = m_parent;
			if (m_parent != nullptr) m_parent->m_start = now;
		}
	}
	Pm4Clock(const Pm4Clock&)            = delete;
	Pm4Clock& operator=(const Pm4Clock&) = delete;
	// A packet begins: the time since the last one began was that one's.
	void Next() {
		if (m_on) {
			Mark(__rdtsc());
			m_class = Pm4Other;
		}
	}
	// The class of the packet that began.
	void Class(uint32_t packet_class) {
		if (m_on) {
			m_class = packet_class < Pm4Classes ? packet_class : static_cast<uint32_t>(Pm4Other);
			g_pm4_packets[m_class] += 1;
		}
	}
	// (otros1, KYTY_OTROS1) The packet's kind (Otros1::KeyOf its header), after Class.
	void Key(uint32_t header) {
		if (m_on && m_otros) {
			m_key = Otros1::KeyOf(header);
			Otros1::g_packets[m_key] += 1;
		}
	}

private:
	void Mark(uint64_t now) {
		g_pm4_cycles[m_class] += now - m_start;
		if (m_otros) {
			Otros1::g_cycles[m_key] += now - m_start;
			m_key = Otros1::Unkeyed; // (until Key names the next packet)
		}
		m_start = now;
	}
	bool      m_on;
	bool      m_otros  = m_on && Otros1::Mode() != 0;
	uint32_t  m_key    = Otros1::Unkeyed;
	uint32_t  m_class  = Pm4Other;
	uint64_t  m_start  = 0;
	Pm4Clock* m_parent = nullptr;
};

// Render-thread waits, always summed (the slow-frame lines, LiveControl::Flip): for GPU work, for a guest
// readback's copy, in a synchronous download, in shader translation and pipeline creation, for the recording
// worker (a full stream or a drain). Steady-clock ns.
enum Wait : uint32_t { WaitGpu, WaitReadback, WaitDownload, WaitCompile, WaitRecord, Waits };
inline int64_t g_waits_ns[Waits] {};
// Any thread's waits for the upload worker's copies (the recording worker's submissions above all), steady-clock ns.
inline std::atomic<int64_t> g_upload_wait_ns {0};
class WaitScope {
public:
	explicit WaitScope(Wait wait): m_wait(wait), m_on(g_render) {
		if (m_on) {
			m_start = std::chrono::steady_clock::now();
			// (The black box: what the render thread waits for, and since when. Nested waits: the outer one is kept.)
			m_outer = Blackbox::g_render_wait.load(std::memory_order_relaxed) == 0;
			if (m_outer) {
				Blackbox::g_render_wait_since.store(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(m_start.time_since_epoch()).count(),
				    std::memory_order_relaxed);
				Blackbox::g_render_wait.store(static_cast<uint32_t>(wait) + 1u, std::memory_order_relaxed);
			}
		}
	}
	~WaitScope() {
		if (m_on) {
			if (m_outer) Blackbox::g_render_wait.store(0, std::memory_order_relaxed);
			const int64_t ns = (std::chrono::steady_clock::now() - m_start).count();
			g_waits_ns[m_wait] += ns;
			// KYTY_SYNC_STATS: the stacks the GPU waits and synchronous downloads come from.
			if ((m_wait == WaitGpu || m_wait == WaitDownload) && ns > 20000 && SyncStats::Enabled())
				SyncStats::Note(m_wait == WaitGpu ? 0u : 1u, ns);
		}
	}
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	Wait                                  m_wait;
	bool                                  m_on;
	bool                                  m_outer = false;
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
	g_draw_phase_cycles[g_draw_phase & (Phases - 1)] += now - g_draw_phase_start;
	if (g_on.load(std::memory_order_relaxed)) Add(DrawPhase, g_draw_shader, g_draw_phase | g_queue, now - g_draw_phase_start);
	g_draw_phase       = next;
	g_draw_phase_start = now;
}
class DrawPhases {
public:
	explicit DrawPhases(uint64_t shader): m_on(g_render && !g_draw_phases) {
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

// (mdraw C0/C1) KYTY_MDRAW_STATS=1 / KYTY_MDRAW_SKIP=1|2 (off by default; local diagnostic; render thread only, the
// main executor's table draws: RenderExecutor::TableDraw in table-xpr.inc). Off: one relaxed load a table draw.
// C0 (STATS): is a table call (TableTry or TableContinue -> TableDraw) a member of a multi-draw group with the call
// before it: both draw indexed indirect, in the same command buffer and graphics generation, inside the same render
// pass (no pass begun, no copy run or compute barrier pending), the same pair, variant (pipeline) and draw state, the
// same index buffer, the previous call wrote no buffer (no barrier after it), the same user data apart from the vertex
// program's per-draw words (start instance SGPR, vertex/instance offset SGPRs) and, after the preparation, the same
// image set. Groups by size, the calls a group would drop (members: every call but the first), their cycles (whole
// call from TableTry/TableContinue's entry; emission = from phase 4 on), why a group ended and what differs inside one.
// C1 (SKIP): a member found before its preparation is neither prepared nor emitted (image broken: diagnostic); 2
// alternates 5 s ON / 5 s OFF windows in one process. The [mdraw] line (live-control.h MdrawReport) every 5 s.
namespace Mdraw {
// Why a call is not a member of the group before it (the first test that failed); Breaks: a member.
enum Break : uint32_t { BFirst, BDirect, BExcluded, BCommand, BPass, BPair, BState, BWrites, BIndex, BUdVs, BUdPs, BSet, Breaks };
inline constexpr const char* BreakNames[Breaks] = {"first", "not_indirect", "excluded", "command", "pass", "pair",
                                                   "state", "writes", "index", "ud_vs", "ud_ps", "set"};
// What differs between a member and the call before it (counted per member).
enum Field : uint32_t { FArgsContig, FArgsJump, FIndexCount, FInstanceCount, FFirstIndex, FVertexOffset, FFirstInstance,
                        FArgsGpu, FBlocks, Fields };
inline constexpr const char* FieldNames[Fields] = {"args_contiguous", "args_jump", "index_count", "instance_count",
                                                   "first_index", "vertex_offset", "first_instance", "args_gpu_written",
                                                   "blocks"};
constexpr uint32_t Buckets = 6;
inline constexpr const char* BucketNames[Buckets] = {"1", "2", "3-4", "5-8", "9-16", "17+"};
[[nodiscard]] inline uint32_t Bucket(uint64_t calls) {
	return calls <= 1 ? 0u : calls == 2 ? 1u : calls <= 4 ? 2u : calls <= 8 ? 3u : calls <= 16 ? 4u : 5u;
}
// -1 not read yet; else bit 0 statistics, bits 1-2 the skip mode (1 always, 2 alternating windows).
inline std::atomic<int32_t> g_mode {-1};
[[nodiscard]] inline int32_t EnvNumber(const char* name) {
	const char* text = std::getenv(name);
	return text == nullptr || *text == '\0' ? 0 : std::atoi(text);
}
inline int32_t Init() {
	const int32_t stats = EnvNumber("KYTY_MDRAW_STATS") != 0 ? 1 : 0;
	int32_t       skip  = EnvNumber("KYTY_MDRAW_SKIP");
	if (skip < 0 || skip > 2) skip = 0;
	const int32_t mode = stats | (skip << 1);
	g_mode.store(mode, std::memory_order_relaxed);
	if (mode != 0) {
		std::printf("mdraw: KYTY_MDRAW_STATS=%d KYTY_MDRAW_SKIP=%d (diagnostic; SKIP draws a broken image)\n", stats, skip);
		std::fflush(stdout);
	}
	return mode;
}
[[nodiscard]] inline int32_t Mode() {
	const int32_t mode = g_mode.load(std::memory_order_relaxed);
	return mode >= 0 ? mode : Init();
}
[[nodiscard]] inline bool     Active() { return Mode() != 0; }
[[nodiscard]] inline uint32_t SkipMode() { return (static_cast<uint32_t>(Mode()) >> 1u) & 3u; }
inline bool g_window_on = true; // (skip mode 2: this window skips; MdrawReport flips it every 5 s)
[[nodiscard]] inline bool SkipNow() {
	const uint32_t skip = SkipMode();
	return skip == 1 || (skip == 2 && g_window_on);
}
// TSC at the entry of the table call (TableTry, TableContinue) that reaches TableDraw (0: TableDraw's own entry).
inline thread_local uint64_t t_entry = 0;
struct Counters {
	uint64_t calls = 0, draws = 0;           // table calls drawn (C0: measured) and their draws
	uint64_t groups = 0;                     // groups closed
	uint64_t hist[Buckets] {};               // groups by size (calls)
	uint64_t members = 0, member_draws = 0;  // calls a group would drop (every call but its first), their draws
	uint64_t member_cycles = 0, member_emit = 0, member_prep = 0; // their TSC cycles: whole, emission, preparation
	uint64_t all_cycles = 0, all_emit = 0;   // every measured call's (reference)
	uint64_t breaks[Breaks] {};
	uint64_t fields[Fields] {};
	uint64_t ud_member[2][32] {};            // members: user data words that differ (only the per-draw words can)
	uint64_t ud_break[2][32] {};             // groups ended by user data: the words that differed
	uint64_t skipped = 0, skipped_draws = 0; // C1: calls neither prepared nor emitted, their draws
};
inline Counters g_c;
inline uint64_t g_group_calls = 0; // the open group's calls
inline void Close() {
	if (g_group_calls == 0) return;
	++g_c.groups;
	++g_c.hist[Bucket(g_group_calls)];
	g_group_calls = 0;
}
// The last table call (TableDraw), what the next one is compared with.
struct Last {
	bool                                 valid = false, writes = false, args_known = false;
	uint64_t                             pair = 0, variant = 0, state = 0, set = 0, command = 0, generation = 0, passes = 0;
	uint64_t                             index_base = 0, index_bytes = 0, last_arg = 0;
	uint32_t                             index_type = 0;
	std::array<std::array<uint32_t, 32>, 2> ud {};
	std::array<uint32_t, 5>              args {};
	std::array<std::vector<uint32_t>, 2> blocks;
};
inline Last g_last;
} // namespace Mdraw

} // namespace LiveCensus
