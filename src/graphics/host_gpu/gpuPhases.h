#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPHASES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPHASES_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <unordered_map>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// Diagnostics: where the GPU thread's time goes, by step of a draw and of a dispatch. Each step is
// timed with the processor's time-stamp counter on the GPU thread only (a scope on another thread,
// the draw-prep scanner's among them, does nothing), and every 10 s a "GPU phases" line gives the
// microseconds each step takes per draw or dispatch, with counts (gpuPhases.cpp). The scopes nest:
// a step's time includes the steps inside it. On by default; KYTY_GPU_PHASES=0 turns it off.
//
// Everything a scope needs is inline here, so files built into the unit tests (pageManager.cpp)
// can use one without the report's file.
namespace Libs::Graphics::GpuPhases {

enum class Phase : uint32_t {
	// Whole operations on the GPU thread.
	Process,  // a slice of a guest submission (GuestGpu::Process)
	Command,  // a command run on the GPU thread (a game thread's read of GPU memory, ...)
	Draw,     // a draw (DrawIndex, DrawAuto)
	Dispatch, // a dispatch (DispatchDirect, DispatchIndirect)

	// Steps of a draw.
	DrawPending,   // the deferred operations it runs first (PopPendingOperations)
	DrawConsume,   // checks for a draw that is a copy, resolve or metadata operation instead
	DrawPrograms,  // its shaders: GetGraphicsPrograms (Describe and ProgramGet inside)
	DrawTargets,   // its render targets' images (ResolveRenderColorTarget / DepthTarget)
	DrawTextures,  // PrepareBindings of its stages: textures, samplers, bindless heaps
	DrawFind,      // FindBuffers: the cached buffers of its buffer bindings
	DrawBda,       // PrepareBda: CPU-dirty buffers synchronized for memory-reading shaders
	DrawImages,    // RebindImages and its render targets again: image views
	DrawBuffers,   // RebindBuffers: buffer views, synchronization, the uploaded SRT and data
	DrawVertex,    // vertex and index buffers
	DrawPipeline,  // TryGetGraphicsPipeline
	DrawRendering, // AcquireRenderTargets: attachments, clears, layouts
	DrawCommit,    // CommitBindings: layouts, descriptors, push data
	DrawEmit,      // the rest of ExecutePreparedDraw: dynamic state, the draw, its barriers

	// Steps of a dispatch.
	DispatchPending,
	DispatchProgram,  // GetComputeProgram (ProgramGet inside)
	DispatchClears,   // TryConsumeComputeMetaClear / ImageClear
	DispatchFill,     // ResolveDispatchFill and fills of mapped parts
	DispatchPipeline, // GetComputePipeline
	DispatchTextures, // PrepareBindings
	DispatchFind,     // FindBuffers
	DispatchBda,      // PrepareBda
	DispatchImages,   // RebindImages
	DispatchBuffers,  // RebindBuffers
	DispatchCommit,   // CommitBindings

	// Shared steps, inside the ones above.
	Describe,       // DescribeGraphicsStages (PrepareProgram of each stage)
	ProgramGet,     // ProgramCache::Get of one stage (draw or dispatch)
	ProgramKey,     // its key and lookup
	ProgramScanned, // its take of the draw-prep scanner's result (with the check of its bytes)
	ProgramWalk,    // its own descriptor walk (MaterializeResources), when it had no result
	BufferSync,     // BufferCache::SynchronizeBuffer
	Protect,        // page protection changes (PageManager)
	Flush,          // a command buffer ended and a new one begun (CommandScheduler::Flush)
	BdaWalk,        // PrepareBda's walk of CPU-dirty buffers (at most once a guest submission)
	BdaRegions,     // (a count only) the tracker regions those walks visited
	BdaProtects,    // (a count only) the page protection changes those walks made
	SpecCheck,      // CommitBindings' check of specialization reads against the draw's writes
	TessDraw,       // a tessellated draw, whole (inside Draw; round 78)
	TessPrograms,   // its shaders (inside DrawPrograms)
	// Round 86: the GPU thread's shader and pipeline work (ProgramCache::Get, PipelineCache).
	ShaderDisk,      // a shader's records read from the shader disk cache (and decoded)
	ShaderCompile,   // a translation and SPIR-V compile done on the GPU thread
	PipelineWait,    // a wait for a pipeline compiling in the background
	PipelineCompile, // a pipeline compiled on the GPU thread (small ones, compute ones)
	TranslationSkip, // (a count only) a draw skipped while its shader translated
	PipelineSkip,    // (a count only) a draw skipped while its pipeline compiled
	// Round 87: new pipelines looked up in the driver's caches alone before a background compile.
	PipelineProbe,     // a lookup
	PipelineProbeHit,  // (a count only) one the driver had: created at once
	PipelineProbeMiss, // (a count only) one it did not have: compiled in the background
	// Round 90: the walk for memory-reading shaders and the waits on memory (PrepareBda).
	BdaAfterWait, // (a count only) a walk made again in a submission, after a wait passed in it
	BdaHeldBack,  // (a count only) a memory-reading draw or dispatch after pages became CPU-dirty,
	              // with no walk: its submission had one, and no wait passed since or the game
	              // wrote nothing since (the GPU thread's own stores alone)
	WaitPassed,   // (a count only) a WAIT_REG_MEM that passed
	// Round 92.
	BufferCreate, // a Vulkan buffer made with its memory (Buffer's constructor)
	// Round 130: the driver's caches are asked on worker threads.
	PipelineLookupSkip, // (a count only) a draw skipped while its pipeline was looked up
	// Round 132.
	CleanStore, // (a count only) a GPU-thread store to watched CPU-clean pages done on both copies
	// Round 134: shader disk cache files read on worker threads.
	DiskReadSkip, // (a count only) a draw skipped while its shader's records were read from disk
	// gt7-prog-1112 (KYTY_PROG_MEMO): ProgramCache::Get's per-stage memo of its last lookup.
	ProgramMemoHit,  // (a count only) a get whose entry the memo gave, without hashing the key
	ProgramMemoMiss, // (a count only) a get the memo could not serve (the map was searched)
	ProgramMatch,    // a get's search of its entry's permutations (after the key and the refresh)
	// gt7-fastdraw-1112 (KYTY_FASTDRAW_TARGETS): ResolveRenderColorTarget's per-slot memo.
	TargetMemoHit,  // (a count only) a color target whose description the memo gave
	TargetMemoMiss, // (a count only) one computed again (registers changed, or first use)
	Count
};

constexpr size_t PhaseCount = static_cast<size_t>(Phase::Count);

struct Totals {
	std::array<uint64_t, PhaseCount> cycles {};
	std::array<uint64_t, PhaseCount> count {};
	// Round 131: the scopes of each phase that were timed, which `cycles` add up (all of them for
	// a phase timed in every frame, AlwaysTimed; those of the timed frames for the others). The
	// report takes a phase's time as its timed scopes' average times its count.
	std::array<uint64_t, PhaseCount> timed {};
	// The longest single scope of each phase since the last report (round 86; the report resets
	// it).
	std::array<uint64_t, PhaseCount> longest {};
};

namespace Detail {
// Set on the GPU thread while the phases are on (Start).
inline thread_local bool t_on = false;
// Round 131: set on the GPU thread in the frames whose draws and dispatches are timed, one in
// KYTY_GPU_PHASES_EVERY (4 by default; 1 times every frame, as before). Two reads of the
// processor's clock for each of the ~30 steps of a draw took about 0.4 us of its 10 (a clock read
// takes ~40 cycles on these processors). The steps are counted in every frame.
inline thread_local bool t_timed = false;
// The frames timed so far (GPU thread), for the report's per-frame figures of what only timed
// frames measure (the slices by queue).
inline uint64_t g_timed_frames = 0;
// Phases timed in every frame: each takes long enough that the clock reads cost nothing beside
// it, or is rare and wanted in each report as it happened (a compile, a lookup, a wait).
[[nodiscard]] constexpr bool AlwaysTimed(Phase phase) noexcept {
	switch (phase) {
		case Phase::Process:
		case Phase::Command:
		case Phase::Protect:
		case Phase::Flush:
		case Phase::BdaWalk:
		case Phase::SpecCheck:
		case Phase::ShaderDisk:
		case Phase::ShaderCompile:
		case Phase::PipelineWait:
		case Phase::PipelineCompile:
		case Phase::PipelineProbe:
		case Phase::BufferCreate: return true;
		default: return false;
	}
}
// Written by the GPU thread only.
inline Totals g_totals;
// The step a page protection change is charged to: the innermost open scope of a whole
// operation (a slice, a command, a draw, a dispatch, a BDA walk) or of a draw's or a dispatch's
// steps; Count outside them. And the protection changes by that step (GPU thread only).
inline thread_local Phase                                          t_step = Phase::Count;
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_protect_cycles {};
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_protect_count {};
// Round 80: the pages those changes covered, and what they set the pages to (no access,
// read-only, read-write), by step.
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_protect_pages {};
inline std::array<std::array<uint64_t, 3>, static_cast<size_t>(Phase::Count) + 1>
                             g_protect_modes {};
// Uploads of guest bytes to the GPU by the step they were made in (round 81): bytes, uploads and
// the copies they were made of.
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_upload_bytes {};
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_upload_count {};
inline std::array<uint64_t, static_cast<size_t>(Phase::Count) + 1> g_upload_copies {};
[[nodiscard]] constexpr bool ChargesProtects(Phase phase) noexcept {
	return phase == Phase::Process || phase == Phase::Command || phase == Phase::Draw ||
	       phase == Phase::Dispatch || phase == Phase::BdaWalk ||
	       (phase >= Phase::DrawPending && phase <= Phase::DispatchCommit);
}
} // namespace Detail

[[nodiscard]] inline uint64_t Now() noexcept {
	return __rdtsc();
}

inline void Add(Phase phase, uint64_t cycles) noexcept {
	const auto index = static_cast<size_t>(phase);
	Detail::g_totals.cycles[index] += cycles;
	Detail::g_totals.count[index]++;
	Detail::g_totals.timed[index]++;
	if (cycles > Detail::g_totals.longest[index]) {
		Detail::g_totals.longest[index] = cycles;
	}
}

// Round 131: a scope of `phase` in a frame not timed: counted only.
inline void AddUntimed(Phase phase) noexcept {
	Detail::g_totals.count[static_cast<size_t>(phase)]++;
}

// Adds `n` to the count of `phase` without time (a count-only phase), on the GPU thread.
inline void Count(Phase phase, uint64_t n) noexcept {
	if (Detail::t_on) {
		Detail::g_totals.count[static_cast<size_t>(phase)] += n;
	}
}

// Round 130: the pipeline cache lookups made on worker threads (any thread adds; the report takes
// them every 10 s).
struct WorkerLookups {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> us {0};
	std::atomic<uint64_t> longest_us {0};
	std::atomic<uint64_t> found {0};
	// Round 132: by how many lookups ran at once when one began (1, 2-3, 4-7, 8 or more): each
	// one's time grows with them when the driver serializes them.
	std::array<std::atomic<uint64_t>, 4> by_concurrency_count {};
	std::array<std::atomic<uint64_t>, 4> by_concurrency_us {};
	std::atomic<uint32_t>                running {0};
};
inline WorkerLookups g_worker_lookups;

[[nodiscard]] constexpr size_t ConcurrencyBucket(uint32_t running) noexcept {
	return running <= 1 ? 0u : running <= 3 ? 1u : running <= 7 ? 2u : 3u;
}

inline void AddWorkerLookup(uint64_t us, bool found, uint32_t running = 1) noexcept {
	const auto bucket = ConcurrencyBucket(running);
	g_worker_lookups.by_concurrency_count[bucket].fetch_add(1, std::memory_order_relaxed);
	g_worker_lookups.by_concurrency_us[bucket].fetch_add(us, std::memory_order_relaxed);
	g_worker_lookups.count.fetch_add(1, std::memory_order_relaxed);
	g_worker_lookups.us.fetch_add(us, std::memory_order_relaxed);
	if (found) {
		g_worker_lookups.found.fetch_add(1, std::memory_order_relaxed);
	}
	auto longest = g_worker_lookups.longest_us.load(std::memory_order_relaxed);
	while (us > longest && !g_worker_lookups.longest_us.compare_exchange_weak(
	                           longest, us, std::memory_order_relaxed)) {
	}
}

// Round 134: the shader disk cache reads made on worker threads (any thread adds; the report
// takes them every 10 s).
struct WorkerDiskReads {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> us {0};
	std::atomic<uint64_t> longest_us {0};
	std::atomic<uint64_t> found {0};
};
inline WorkerDiskReads g_worker_disk_reads;

inline void AddWorkerDiskRead(uint64_t us, bool found) noexcept {
	g_worker_disk_reads.count.fetch_add(1, std::memory_order_relaxed);
	g_worker_disk_reads.us.fetch_add(us, std::memory_order_relaxed);
	if (found) {
		g_worker_disk_reads.found.fetch_add(1, std::memory_order_relaxed);
	}
	auto longest = g_worker_disk_reads.longest_us.load(std::memory_order_relaxed);
	while (us > longest && !g_worker_disk_reads.longest_us.compare_exchange_weak(
	                           longest, us, std::memory_order_relaxed)) {
	}
}

// Round 133: the draws left out while their pipelines were looked up (round 130) since the last
// flip of the main display, for that flip (TakeFrameHold).
inline std::atomic<uint32_t> g_frame_lookup_skips {0};
inline void                  NoteLookupSkip() noexcept {
    g_frame_lookup_skips.fetch_add(1, std::memory_order_relaxed);
}
// The draws left out while their pipelines compiled in the background, since the same flip. They
// only count towards a hold with KYTY_HOLD_COMPILING=1 (TakeFrameHold).
inline std::atomic<uint32_t> g_frame_compile_skips {0};
inline void                  NoteCompileSkip() noexcept {
    g_frame_compile_skips.fetch_add(1, std::memory_order_relaxed);
}

// GPU thread, at a flip of the main display: whether the screen keeps the frame it shows instead
// of this one, which left out KYTY_HOLD_SKIPPED_DRAWS draws or more (100 by default; 0: never)
// while their pipelines were looked up. A burst of lookups at a race start left out most of the
// track for a frame or two; that frame is not shown, at most two in a row, then eight frames
// shown before the next. Notes the frame's left-out draws for the 10 s "Pipeline lookups" line.
// KYTY_HOLD_COMPILING=1: draws left out while their pipelines compiled count too, and frames are
// held in a row for up to KYTY_HOLD_MAX_MS (5000 by default), so the screen shows only complete
// frames while a burst of new pipelines compiles in parallel; one frame is shown after the cap.
[[nodiscard]] bool TakeFrameHold() noexcept;

namespace Detail {
// Round 108: page protection changes by 64 KB block since the last report, by mode (GPU thread):
// the blocks whose pages are protected over and over (the GPU writes them and the CPU touches
// them, or the CPU writes them and the GPU reads them, every frame). At most 65536 blocks.
inline std::unordered_map<uint64_t, std::array<uint32_t, 3>> g_protect_blocks;
inline void NoteProtectBlock(uint64_t vaddr, uint32_t mode) noexcept {
	const auto block = vaddr >> 16u;
	auto       it    = g_protect_blocks.find(block);
	if (it == g_protect_blocks.end()) {
		if (g_protect_blocks.size() >= 65536u) {
			return;
		}
		it = g_protect_blocks.emplace(block, std::array<uint32_t, 3> {}).first;
	}
	it->second[mode < 2u ? mode : 2u]++;
}
// Round 108: graphics streams stopped for compute work, and the packets they ran first.
inline uint64_t g_yields        = 0;
inline uint64_t g_yield_packets = 0;
} // namespace Detail

// Round 108: a graphics stream stopped for compute work after `packets` packets (the report's
// "compute yields"; GPU thread).
inline void NoteYield(uint32_t packets) noexcept {
	if (Detail::t_on) {
		Detail::g_yields++;
		Detail::g_yield_packets += packets;
	}
}

// A page protection change of `bytes` at `vaddr` to `mode` (0 no access, 1 read-only, 2
// read-write), charged to the step a Protect scope would charge it to, on the GPU thread (round
// 80).
inline void NoteProtect(uint64_t vaddr, uint64_t bytes, uint32_t mode) noexcept {
	if (!Detail::t_on) {
		return;
	}
	const auto step = static_cast<size_t>(Detail::t_step);
	Detail::g_protect_pages[step] += (bytes + 4095u) >> 12u;
	Detail::g_protect_modes[step][mode < 2u ? mode : 2u]++;
	Detail::NoteProtectBlock(vaddr, mode);
}

// An upload of `bytes` of guest memory in `copies` copies, charged to the step as a page
// protection change is, on the GPU thread (round 81).
inline void NoteUpload(uint64_t bytes, uint64_t copies) noexcept {
	if (!Detail::t_on) {
		return;
	}
	const auto step = static_cast<size_t>(Detail::t_step);
	Detail::g_upload_bytes[step] += bytes;
	Detail::g_upload_count[step]++;
	Detail::g_upload_copies[step] += copies;
}

// The count of `phase` so far (GPU thread).
[[nodiscard]] inline uint64_t CountOf(Phase phase) noexcept {
	return Detail::g_totals.count[static_cast<size_t>(phase)];
}

namespace Detail {
// Round 113: the GPU thread's slices by the kind of queue they ran (0 graphics, 1 compute):
// cycles, slices, draws, dispatches, the draws' cycles, the dispatches' cycles (GPU thread).
inline std::array<std::array<uint64_t, 6>, 2> g_queue_stats {};
} // namespace Detail

// Round 113: a slice of a guest submission, timed by the kind of queue it ran (the report's "by
// queue"): what a second GPU thread for the compute queues would take off this one.
class QueueSlice {
public:
	explicit QueueSlice(bool compute) noexcept: m_compute(compute) {
		// What the draws and dispatches took is known in timed frames only (round 131).
		if (!Detail::t_on || !Detail::t_timed) {
			return;
		}
		m_draws           = CountOf(Phase::Draw);
		m_dispatches      = CountOf(Phase::Dispatch);
		m_draw_cycles     = Detail::g_totals.cycles[static_cast<size_t>(Phase::Draw)];
		m_dispatch_cycles = Detail::g_totals.cycles[static_cast<size_t>(Phase::Dispatch)];
		m_start           = Now();
	}
	~QueueSlice() {
		if (m_start == 0) {
			return;
		}
		auto& stats = Detail::g_queue_stats[m_compute ? 1u : 0u];
		stats[0] += Now() - m_start;
		stats[1]++;
		stats[2] += CountOf(Phase::Draw) - m_draws;
		stats[3] += CountOf(Phase::Dispatch) - m_dispatches;
		stats[4] += Detail::g_totals.cycles[static_cast<size_t>(Phase::Draw)] - m_draw_cycles;
		stats[5] +=
		    Detail::g_totals.cycles[static_cast<size_t>(Phase::Dispatch)] - m_dispatch_cycles;
	}
	QueueSlice(const QueueSlice&)            = delete;
	QueueSlice& operator=(const QueueSlice&) = delete;

private:
	bool     m_compute;
	uint64_t m_start           = 0;
	uint64_t m_draws           = 0;
	uint64_t m_dispatches      = 0;
	uint64_t m_draw_cycles     = 0;
	uint64_t m_dispatch_cycles = 0;
};

// Times its block on the GPU thread (in timed frames, or always for an AlwaysTimed phase; it is
// counted in every frame).
class Scope {
public:
	explicit Scope(Phase phase) noexcept: m_phase(phase) { Begin(); }
	~Scope() { End(); }
	// Ends the timing before the block does (and its charge of protection changes).
	void End() noexcept {
		if (m_active) {
			if (m_start != 0) {
				const auto cycles = Now() - m_start;
				Add(m_phase, cycles);
				if (m_phase == Phase::Protect) {
					Detail::g_protect_cycles[static_cast<size_t>(m_step)] += cycles;
					Detail::g_protect_count[static_cast<size_t>(m_step)]++;
				}
			} else {
				AddUntimed(m_phase);
			}
			m_active = false;
			m_start  = 0;
		}
		RestoreStep();
	}
	// Drops the timing: the block is not counted.
	void Cancel() noexcept {
		m_active = false;
		m_start  = 0;
		RestoreStep();
	}
	// Times the same phase again from here (after an End); a second count.
	void Restart() noexcept {
		End();
		Begin();
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	void Begin() noexcept {
		if (!Detail::t_on) {
			return;
		}
		m_active = true;
		m_start  = Detail::t_timed || Detail::AlwaysTimed(m_phase) ? Now() : 0;
		m_step   = Detail::t_step;
		if (Detail::ChargesProtects(m_phase)) {
			Detail::t_step = m_phase;
			m_set_step     = true;
		}
	}
	void RestoreStep() noexcept {
		if (m_set_step) {
			Detail::t_step = m_step;
			m_set_step     = false;
		}
	}

	Phase    m_phase;
	uint64_t m_start    = 0;
	Phase    m_step     = Phase::Count; // t_step when it began
	bool     m_active   = false;        // counted (the phases were on when it began)
	bool     m_set_step = false;        // it set t_step (to restore when it ends)
};

// KYTY_OP_STATS=1 (round 84): the GPU thread's time by dispatch shader and by draw target. An
// OpStats scope times one draw or dispatch whole, with what its BDA walk and its page protection
// changes took; what the operation is grouped by (its shader for a dispatch, its first color
// target, or depth target when it has none, for a draw) is set with SetOpKey once it is known,
// and an operation that never sets it is not counted. The "GPU phases" report adds a "Dispatch
// shaders" and a "Draw targets" line: the groups that took the most of the GPU thread, with their
// operations a frame and microseconds each.
namespace Detail {
inline thread_local uint64_t t_op_key   = 0;
inline thread_local uint64_t t_op_extra = 0;
} // namespace Detail

[[nodiscard]] bool OpStatsOn() noexcept;
void NoteOp(bool dispatch, uint64_t key, uint64_t extra, uint64_t cycles, uint64_t bda_cycles,
            uint64_t protect_cycles) noexcept;

// Names what the operation being timed is grouped by (GPU thread): a dispatch's shader hash and
// its thread groups; a draw's target address (bit 63 set for a depth target) and its extent
// (width | height << 16 | layers << 32).
inline void SetOpKey(uint64_t key, uint64_t extra) noexcept {
	Detail::t_op_key   = key;
	Detail::t_op_extra = extra;
}

class OpStats {
public:
	explicit OpStats(bool dispatch) noexcept: m_dispatch(dispatch) {
		if (!Detail::t_on || !Detail::t_timed || !OpStatsOn()) {
			return;
		}
		Detail::t_op_key = 0;
		m_bda            = BdaCycles();
		m_protect        = Detail::g_totals.cycles[static_cast<size_t>(Phase::Protect)];
		m_start          = Now();
	}
	~OpStats() {
		if (m_start == 0 || Detail::t_op_key == 0) {
			return;
		}
		NoteOp(m_dispatch, Detail::t_op_key, Detail::t_op_extra, Now() - m_start,
		       BdaCycles() - m_bda,
		       Detail::g_totals.cycles[static_cast<size_t>(Phase::Protect)] - m_protect);
		Detail::t_op_key = 0;
	}
	OpStats(const OpStats&)            = delete;
	OpStats& operator=(const OpStats&) = delete;

private:
	[[nodiscard]] static uint64_t BdaCycles() noexcept {
		return Detail::g_totals.cycles[static_cast<size_t>(Phase::DrawBda)] +
		       Detail::g_totals.cycles[static_cast<size_t>(Phase::DispatchBda)];
	}

	bool     m_dispatch;
	uint64_t m_start   = 0;
	uint64_t m_bda     = 0;
	uint64_t m_protect = 0;
};

// GPU thread, at its start: turns the phases on there (KYTY_GPU_PHASES=0 leaves them off).
void Start();
// GPU thread: logs the "GPU phases" line every 10 s; `frames`: the frames the guest has finished.
void ReportIfDue(uint64_t frames);

} // namespace Libs::Graphics::GpuPhases

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUPHASES_H_
