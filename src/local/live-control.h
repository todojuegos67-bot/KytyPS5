#pragma once
// Local measurement hook (KYTY_LIVE_FILE): one boot, many same-process A/B experiments.
// A side thread polls the command file every 100 ms and runs its commands once per new
// "id N" first line, printing LIVE_* lines to stdout (tools/local/live-bench.py).
//   poke32 <host address> <hex>   write a switch (live-bench resolves `sym NAME VALUE`)
//   peek <host address> <bytes>   hex dump (<= 256 bytes)
//   measure <seconds> <label>     presented frames, fps, per-frame CPU of the render and
//                                 recording threads
//   prof <seconds> <path>         4 kHz samples of render-thread CPU time
//   profw <seconds> <path>        4 kHz samples of render-thread wall time (waits included)
//   profp <seconds> <path>        4 kHz samples of process CPU time (with thread ids)
//   proft <tid> <seconds> <path>  Windows: samples of another thread (e.g. the guest main thread)
//   pinthread <tid> <cpus>        Windows: set any thread's affinity
// A sample is 16 words: pc, the word at rsp, then up to 14 frame-pointer return addresses
// (process mode: pc, thread id | 1 << 63, zeros).
//   timecensus <seconds>          time-related HLE calls per guest caller (time-census.h)
//   glide <x> <y> <z> <speed>     Demon's Souls 01.007.000: the player walks there (warp requests, no input)
//   pad <lx> <ly> <rx> <ry> <buttons> <seconds>   scripted pad input (sticks 0..255, -1 the host's; buttons hex)
//   sleep <seconds>
// Without KYTY_LIVE_FILE nothing runs; the flip hook is one relaxed increment.

#include "local/skip-shader.h"
#include "common/assert.h"
#include "frame-capture.h"
#include "par-audit.h" // (A0b) diagnostic only: KYTY_AUDIT off by default
#include "kernel/memory.h"
#include "ds-linear-copy-hook.h"
#include "copia1.h" // (copia1)
#include "live-census.h"
#include "otros2.h" // (otros2)
#include "gancho2.h" // (gancho2)
#include "port1.h" // (port1)
#include "port2-report.h" // (port2)
#include "p7-report.h" // (p7, KYTY_P7)
#include "f5p1-techo.h" // (f5p1 techo, KYTY_F5P1_TECHO)
#include "f0-measure.h" // (f0)
#include "live-counters.h"
#include "tiron1.h" // (tiron1)
#include "tiron3.h" // (lab1 tiron3, KYTY_TIRON3)
#include "lane-sync.h"
#include "live-trace.h"
#include "libs/controller.h"
#include "graphics/presentation/videoOut.h"
#include "loader/demonsSoulsWarp.h"
#include "local-platform.h"
#include "perf-overlay.h"
#include "slow-log.h"
#include "time-census.h"
#include "techo-report.h" // (techo, KYTY_TECHO)
#include "ds-sombras-ab.h" // (sombras-ab, KYTY_NATIVO_SOMBRAS_AB)
#include "r0-ab.h" // (r0, KYTY_R0_AB)
#include "trusted-depth.h" // (trust1)
#include "native/g1-probe.h" // (g1, KYTY_G1_PROBE / KYTY_G1_STATE)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <ctime>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#include <pthread.h>
#include <ucontext.h>
#endif

#if defined(KYTY_PGO_GENERATE)
// compiler-rt profile runtime (instrumented builds only).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

namespace Loader {
std::string DescribeGuestAddressForDiagnostics(uint64_t vaddr);
} // namespace Loader

namespace LiveControl {

inline std::atomic_uint64_t g_flips {0};
// The steady-clock time (ns) of flip n at [n % FlipTimes], for frame-time percentiles (Measure).
constexpr size_t             FlipTimes = 8192;
inline std::atomic<int64_t>  g_flip_times[FlipTimes] {};
inline std::atomic_bool     g_render_known {false};
// pthread_t on Linux, a thread HANDLE on Windows (LocalPlatform::CurrentThreadHandle).
inline uint64_t              g_render_thread {};
inline std::atomic<uint32_t> g_render_tid {0};
// Render-thread stack bounds for the frame-pointer walk.
inline uint64_t             g_stack_low = 0, g_stack_high = 0;
constexpr size_t             SampleWords = 16;
constexpr size_t             SampleSlots = size_t {1} << 23u; // words of g_samples
// The samples (64 MiB) and their TSCs (4 MiB): allocated by the first prof (EnsureSamples) and kept, not static arrays:
// static ones were 68 MiB of the executable's writable image, charged to commit in every session that never profiles.
inline uint64_t*             g_samples    = nullptr;
inline uint64_t*             g_sample_tsc = nullptr; // Windows: each sample's TSC (SampleSlots / SampleWords)
inline bool                  EnsureSamples() {
	if (g_samples == nullptr) g_samples = static_cast<uint64_t*>(std::calloc(SampleSlots, sizeof(uint64_t)));
	if (g_sample_tsc == nullptr) g_sample_tsc = static_cast<uint64_t*>(std::calloc(SampleSlots / SampleWords, sizeof(uint64_t)));
	return g_samples != nullptr && g_sample_tsc != nullptr;
}
inline std::atomic<uint32_t> g_sample_count {0};
inline std::atomic_bool      g_process {false};

// The render thread's time waiting for guest work (GuestGpu::ThreadRun), in steady-clock ns.
inline int64_t g_render_idle_ns = 0;
// When the render thread began waiting for the guest's next submission (steady-clock ns; 0 while it has one): its
// idle waits set it once, a submission taken clears it (guest commands do not: a guest thread may wait on them).
inline std::atomic<int64_t> g_render_idle_since {0};
inline void NoteRenderIdle(int64_t now_ns) {
	int64_t expected = 0;
	(void)g_render_idle_since.compare_exchange_strong(expected, now_ns, std::memory_order_relaxed);
}

// (render2) KYTY_RENDER_STATS=1 (off by default): a [render] line every 5 s with the render thread's mean time per
// frame over ALL the frames of those 5 s, by the same call kinds and phases as the SLOW Frame lines (which only show
// frames of 20 ms or more: the frames where something new is stored or compiled, not the steady ones the frame rate
// is made of). "other" is the submissions' time outside draws, dispatches and native draws: the PM4 walk, register
// writes and what has no scope of its own. Costs two clock reads a frame and one line every 5 s. Render thread.
inline void RenderStats(int64_t flip_ns) {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_RENDER_STATS");
		return text != nullptr && *text != '\0' && std::string_view(text) != "0";
	}();
	if (!on) return;
	using C = LiveCounters::Id;
	static constexpr std::array counted {C::TableDraws,      C::TableContinued,   C::TableDispatches,  C::TableEvaluations,
	                                     C::TableStores,     C::TableStoreDeferred, C::TableDrawNative, C::XprTries,
	                                     C::XprHits,         C::XprStores,        C::DirectDraws,      C::RenderPasses,
	                                     C::ComputePipelines, C::GraphicsPipelines, C::AsyncDrawSkips,  C::ProtectCallsRender,
	                                     C::UploadBytes,
	                                     // (render3) what the uploads are made of: the write windows and their pages,
	                                     // the pages protected again, the copies, the BDA preparations and their region
	                                     // synchronizations (microseconds on this thread), the upload worker's time.
	                                     C::UploadCopies,    C::InOrderUploads,   C::WindowFault,      C::WindowPages,
	                                     C::WriteFault,      C::UnprotectPages,   C::ReprotectPages,   C::RegionSyncs,
	                                     C::RegionSkips,     C::RegionSyncUs,     C::BufferSyncUs,     C::BdaRangeCalls,
	                                     C::BdaRanges,       C::BdaRangeMiB,      C::BdaFullSyncs,     C::BdaRebuilds,
	                                     C::BufferRegistrations, C::UploadBusyUs, C::UploadCallUs,     C::Submissions,
	                                     C::TableWrittenMemo, C::TableRingCopies, C::TableRingBytes};
	struct Snapshot {
		int64_t                                         time_ns = 0, idle_ns = 0, upload_wait_ns = 0;
		// (render3) the PM4 walk by packet class, the BDA memo's hits and misses, the big-window windows.
		std::array<uint64_t, LiveCensus::Pm4Classes>    pm4_cycles {}, pm4_packets {};
		std::array<uint64_t, 2>                         bda_memo {};
		std::array<uint64_t, 3>                         big_window {};
		// (render4) the dispatches and PM4 runs by queue, the blocks' V#s, the linear copy hook's counters.
		std::array<std::array<uint64_t, 2>, 2>          queue_dispatches {}, queue_runs {};
		std::array<uint64_t, 5>                         table_blocks {};
		std::array<uint64_t, DsLinearCopyHook::Counters> copy_hook {};
		// (thread1) the draw calls by kind, those continued, the table draws by target; the two-thread what-if
		// (LiveCensus::Par2): each side's cycles, slices, waits, the cycles held back and the virtual cycles.
		std::array<uint64_t, 4>                         xpr_calls {};
		std::array<uint64_t, 2>                         continued_calls {};
		std::array<std::array<uint64_t, 2>, LiveCensus::DrawTargets> draw_targets {};
		std::array<uint64_t, 2>                         par2_side {}, par2_waits {};
		std::array<std::array<uint64_t, 2>, 2>          par2_slices {};
		uint64_t                                        par2_stalled = 0, par2_elapsed = 0;
		uint64_t                                        tsc = 0, thread_cycles = 0, flips = 0;
		std::array<uint64_t, LiveCensus::Kinds>         kinds {};
		std::array<uint64_t, LiveCensus::Kinds>         calls {};
		std::array<uint64_t, LiveCensus::Phases>        dispatch_phases {}, draw_phases {};
		std::array<std::array<uint64_t, LiveCensus::Phases>, 2> table_phases {};
		std::array<uint64_t, 4>                         xpr {};
		std::array<int64_t, LiveCensus::Waits>          waits {};
		std::array<uint64_t, counted.size()>            counts {};
	};
	static Snapshot last;
	static uint64_t frames = 0;
	++frames;
	if (last.time_ns != 0 && flip_ns - last.time_ns < 5'000'000'000ll) return;
	Snapshot now;
	now.time_ns        = flip_ns;
	now.idle_ns        = g_render_idle_ns;
	now.upload_wait_ns = LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed);
	now.tsc            = __rdtsc();
	now.thread_cycles  = LocalPlatform::CurrentThreadCycles();
	for (size_t i = 0; i < LiveCensus::Kinds; ++i) {
		now.kinds[i] = LiveCensus::g_kind_cycles[i];
		now.calls[i] = LiveCensus::g_kind_calls[i];
	}
	for (size_t i = 0; i < LiveCensus::Phases; ++i) {
		now.dispatch_phases[i] = LiveCensus::g_dispatch_phase_cycles[i];
		now.draw_phases[i]     = LiveCensus::g_draw_phase_cycles[i];
		now.table_phases[0][i] = LiveCensus::g_table_phase_cycles[0][i];
		now.table_phases[1][i] = LiveCensus::g_table_phase_cycles[1][i];
	}
	for (size_t i = 0; i < 4; ++i) now.xpr[i] = LiveCensus::g_xpr_cycles[i];
	for (size_t i = 0; i < LiveCensus::Waits; ++i) now.waits[i] = LiveCensus::g_waits_ns[i];
	for (size_t i = 0; i < counted.size(); ++i) now.counts[i] = LiveCounters::Value(counted[i]);
	for (size_t i = 0; i < LiveCensus::Pm4Classes; ++i) {
		now.pm4_cycles[i]  = LiveCensus::g_pm4_cycles[i];
		now.pm4_packets[i] = LiveCensus::g_pm4_packets[i];
	}
	for (size_t i = 0; i < now.bda_memo.size(); ++i) now.bda_memo[i] = LiveCensus::g_bda_memo[i];
	for (size_t i = 0; i < now.big_window.size(); ++i)
		now.big_window[i] = LiveCensus::g_big_window[i].load(std::memory_order_relaxed);
	for (size_t q = 0; q < 2; ++q)
		for (size_t i = 0; i < 2; ++i) {
			now.queue_dispatches[q][i] = LiveCensus::g_queue_dispatches[q][i];
			now.queue_runs[q][i]       = LiveCensus::g_queue_runs[q][i];
		}
	for (size_t i = 0; i < now.table_blocks.size(); ++i) now.table_blocks[i] = LiveCensus::g_table_blocks[i];
	for (size_t i = 0; i < now.copy_hook.size(); ++i)
		now.copy_hook[i] = DsLinearCopyHook::g_counters[i].load(std::memory_order_relaxed);
	// (thread1)
	for (size_t i = 0; i < now.xpr_calls.size(); ++i) now.xpr_calls[i] = LiveCensus::g_xpr_calls[i];
	for (size_t i = 0; i < now.continued_calls.size(); ++i) now.continued_calls[i] = LiveCensus::g_continued_calls[i];
	for (size_t t = 0; t < LiveCensus::DrawTargets; ++t)
		for (size_t i = 0; i < 2; ++i) now.draw_targets[t][i] = LiveCensus::g_draw_targets[t][i];
	for (size_t q = 0; q < 2; ++q) {
		now.par2_side[q]  = LiveCensus::Par2::g_side_cycles[q];
		now.par2_waits[q] = LiveCensus::Par2::g_waits[q];
		for (size_t i = 0; i < 2; ++i) now.par2_slices[q][i] = LiveCensus::Par2::g_slices[q][i];
	}
	now.par2_stalled = LiveCensus::Par2::g_stalled;
	now.par2_elapsed = LiveCensus::Par2::g_elapsed;
	if (last.time_ns != 0 && frames != 0 && now.tsc > last.tsc) {
		const double n       = static_cast<double>(frames);
		const double wall_ms = static_cast<double>(now.time_ns - last.time_ns) / 1e6;
		// (ms per TSC cycle over these 5 s; a frame's share of each kind.)
		const double per_cycle = wall_ms / static_cast<double>(now.tsc - last.tsc) / n;
		const auto   kind      = [&](size_t i) { return static_cast<double>(now.kinds[i] - last.kinds[i]) * per_cycle; };
		const auto   each      = [&](size_t i) { return static_cast<double>(now.calls[i] - last.calls[i]) / n; };
		const double submission = kind(LiveCensus::Submission), dispatch = kind(LiveCensus::Dispatch);
		const double draw = kind(LiveCensus::Draw), xpr = kind(LiveCensus::NativeXpr);
		std::printf("[render] %.1f s, %llu frames (%.1f fps): frame %.2f ms, cpu %.2f, idle %.2f | submissions %.2f = native "
		            "draws %.2f (%.0f calls) + dispatches %.2f (%.0f) + normal draws %.2f (%.0f) + other %.2f; guest "
		            "commands %.2f, command sync %.2f",
		            wall_ms / 1e3, static_cast<unsigned long long>(frames), n * 1e3 / wall_ms, wall_ms / n,
		            now.thread_cycles != 0 && last.thread_cycles != 0
		                ? static_cast<double>(now.thread_cycles - last.thread_cycles) * per_cycle
		                : 0.0,
		            static_cast<double>(now.idle_ns - last.idle_ns) / 1e6 / n, submission, xpr, each(LiveCensus::NativeXpr),
		            dispatch, each(LiveCensus::Dispatch), draw, each(LiveCensus::Draw), submission - xpr - dispatch - draw,
		            kind(LiveCensus::GuestCommand), kind(LiveCensus::CommandSync));
		std::printf(" | xpr");
		for (size_t i = 0; i < 4; ++i)
			if (const double ms = static_cast<double>(now.xpr[i] - last.xpr[i]) * per_cycle; ms >= 0.005)
				std::printf(" %zu=%.2f", i, ms);
		std::printf(" | dispatch");
		for (size_t i = 0; i < LiveCensus::Phases; ++i)
			if (const double ms = static_cast<double>(now.dispatch_phases[i] - last.dispatch_phases[i]) * per_cycle; ms >= 0.005)
				std::printf(" %s=%.2f", LiveCensus::DispatchPhaseNames[i], ms);
		std::printf(" | draw");
		for (size_t i = 0; i < LiveCensus::Phases; ++i)
			if (const double ms = static_cast<double>(now.draw_phases[i] - last.draw_phases[i]) * per_cycle; ms >= 0.005)
				std::printf(" %s=%.2f", LiveCensus::DrawPhaseNames[i], ms);
		// (table-xpr.inc's scopes: 0 the whole try; draws: 1 evaluate and gather, 2 set, 3 buffers and blocks, 9 and 4
		// emission, 5 to 7 set proofs, 10 and 11 held-buffer misses; dispatches: 1 evaluate and gather, 2 set, 3 blocks,
		// 4 emission. Nested scopes count in both. render3, inside 3: draws 12 the BDA preparation (its uploads), 13 the
		// blocks; dispatches 5 the BDA preparation, 6 the blocks, 7 the indirect arguments. thread1, draws: 14 what
		// NativeXprTry does before the table try (pending operations, lock, sets retired, state key), 15 the table
		// try's lookups up to TableDraw (inside 0).)
		static constexpr const char* table_names[2] = {"table draws", "table dispatches"};
		for (size_t t = 0; t < 2; ++t) {
			std::printf(" | %s", table_names[t]);
			for (size_t i = 0; i < LiveCensus::Phases; ++i)
				if (const double ms = static_cast<double>(now.table_phases[t][i] - last.table_phases[t][i]) * per_cycle; ms >= 0.005)
					std::printf(" %zu=%.2f", i, ms);
		}
		static constexpr const char* waits[] = {"gpu", "readback", "download", "compile", "record"};
		std::printf(" | waits");
		for (size_t i = 0; i < LiveCensus::Waits; ++i)
			if (const double ms = static_cast<double>(now.waits[i] - last.waits[i]) / 1e6 / n; ms >= 0.005)
				std::printf(" %s=%.2f", waits[i], ms);
		std::printf(" upload=%.2f | per frame", static_cast<double>(now.upload_wait_ns - last.upload_wait_ns) / 1e6 / n);
		for (size_t i = 0; i < counted.size(); ++i)
			if (const auto delta = now.counts[i] - last.counts[i]; delta != 0)
				std::printf(" %s=%.1f", LiveCounters::Names[counted[i]], static_cast<double>(delta) / n);
		// (render3) Where "other" goes: the PM4 walk by packet class, ms a frame from a packet's start to the next one's
		// (so "draws" and "dispatches" hold the native draws, normal draws and dispatches above plus their packets'
		// own walk; the register packets, NOPs and the rest are "other") and the packets a frame.
		std::printf(" | pm4");
		for (size_t i = 0; i < LiveCensus::Pm4Classes; ++i)
			if (const auto packets = now.pm4_packets[i] - last.pm4_packets[i], cycles = now.pm4_cycles[i] - last.pm4_cycles[i];
			    packets != 0 || cycles != 0)
				std::printf(" %s=%.2f (%.0f)", LiveCensus::Pm4ClassNames[i], static_cast<double>(cycles) * per_cycle,
				            static_cast<double>(packets) / n);
		// (render3) KYTY_BDA_MEMO's hits and misses, and the big-window granules' windows (flat 1 MiB ones; grown ones
		// and their pages, KYTY_BIG_WINDOW_GROW), a frame.
		std::printf(" | bda_memo hits=%.1f misses=%.1f | big_window flat=%.1f grown=%.1f grown_pages=%.1f",
		            static_cast<double>(now.bda_memo[0] - last.bda_memo[0]) / n,
		            static_cast<double>(now.bda_memo[1] - last.bda_memo[1]) / n,
		            static_cast<double>(now.big_window[0] - last.big_window[0]) / n,
		            static_cast<double>(now.big_window[1] - last.big_window[1]) / n,
		            static_cast<double>(now.big_window[2] - last.big_window[2]) / n);
		// (render4) The dispatches by queue: a frame's dispatches and their ms in the graphics queue (dcb) and in the
		// async compute queues (acb), and the whole PM4 runs of each (packets, draws and dispatches: what a queue costs
		// this thread).
		std::printf(" | queues");
		static constexpr const char* queue_names[2] = {"dcb", "acb"};
		for (size_t q = 0; q < 2; ++q)
			std::printf(" %s dispatches=%.0f (%.2f ms) run=%.2f ms", queue_names[q],
			            static_cast<double>(now.queue_dispatches[q][0] - last.queue_dispatches[q][0]) / n,
			            static_cast<double>(now.queue_dispatches[q][1] - last.queue_dispatches[q][1]) * per_cycle,
			            static_cast<double>(now.queue_runs[q][1] - last.queue_runs[q][1]) * per_cycle);
		// (render4) The 8 compute shaders with most dispatches: the program's hash (or @ its address when the dispatch's
		// path had no program; linear_copy: the linear copy's register check), dispatches a frame and their ms. The
		// table is these 5 s' and is emptied here (this thread is its only writer).
		{
			std::array<LiveCensus::ShaderStat, 8> top {};
			size_t                                shaders = 0;
			for (auto& entry: LiveCensus::g_shader_stats) {
				if (entry.calls == 0) continue;
				++shaders;
				auto candidate = entry;
				for (auto& slot: top)
					if (candidate.calls > slot.calls) std::swap(candidate, slot);
				entry = {};
			}
			std::printf(" | cs %zu shaders, top", shaders);
			for (const auto& slot: top) {
				if (slot.calls == 0) break;
				const double dispatches = static_cast<double>(slot.calls) / n;
				const double ms         = static_cast<double>(slot.cycles) * per_cycle;
				if (slot.shader == LiveCensus::LinearCopyShader) std::printf(" linear_copy=%.0f (%.2f)", dispatches, ms);
				else if ((slot.shader >> 48u) == 0)
					std::printf(" @%llx=%.0f (%.2f)", static_cast<unsigned long long>(slot.shader), dispatches, ms);
				else std::printf(" %016llx=%.0f (%.2f)", static_cast<unsigned long long>(slot.shader), dispatches, ms);
			}
			if (LiveCensus::g_shader_stats_dropped != 0)
				std::printf(" dropped=%llu", static_cast<unsigned long long>(LiveCensus::g_shader_stats_dropped));
			LiveCensus::g_shader_stats_dropped = 0;
		}
		// (render4) The blocks' V#s a frame (table dispatches' phase 6, table draws' phase 13): written ones and those
		// of them that went through ObtainBuffer; read-only ones, those resolved again and those proved clean again.
		{
			const auto blocks = [&](size_t i) { return static_cast<double>(now.table_blocks[i] - last.table_blocks[i]) / n; };
			std::printf(" | blocks written=%.0f obtained=%.0f read=%.0f resolved=%.0f reproved=%.0f", blocks(0), blocks(1),
			            blocks(2), blocks(3), blocks(4));
		}
		// (render4) KYTY_DS_LINEAR_COPY_HOOK, a frame: the game's linear copies (calls), the packets sealed, the copies
		// made from them (taken) and those left to the game's packets (passed); with any, why packets were not sealed and
		// the verify mode's results.
		if (now.copy_hook[DsLinearCopyHook::Calls] != 0) {
			using H = DsLinearCopyHook::Counter;
			const auto hook = [&](H i) { return static_cast<double>(now.copy_hook[i] - last.copy_hook[i]) / n; };
			std::printf(" | ds_copy_hook calls=%.1f sealed=%.1f taken=%.1f passed=%.1f", hook(H::Calls), hook(H::Sealed),
			            hook(H::Taken), hook(H::Passed));
			static constexpr std::array<std::pair<H, const char*>, 6> rest {{{H::NotPlain, "not_plain"},
			                                                                {H::NoRoom, "no_room"},
			                                                                {H::OtherPackets, "other_packets"},
			                                                                {H::VerifySame, "verify_same"},
			                                                                {H::VerifyDifferent, "verify_different"},
			                                                                {H::VerifyRefused, "verify_refused"}}};
			for (const auto& [id, name]: rest)
				if (now.copy_hook[id] != last.copy_hook[id]) std::printf(" %s=%.1f", name, hook(id));
		}
		// (thread1) What a frame's draws are: the draw calls by kind (runs of DRAW_INDEX_INDIRECT and multi draws,
		// direct indexed draws, draws of vertices: the "xpr" times above are theirs, in that order) and how many of
		// them a clean continuation took; the table draws by what they draw into, calls and draws (depth_layered: a
		// depth image of 4 layers or more alone - the sun's cascades; depth: another depth image alone; color: the rest).
		{
			const auto calls = [&](size_t i) { return static_cast<double>(now.xpr_calls[i] - last.xpr_calls[i]) / n; };
			const auto continued = [&](size_t i) {
				return static_cast<double>(now.continued_calls[i] - last.continued_calls[i]) / n;
			};
			std::printf(" | draw calls runs=%.0f (continued %.0f) direct=%.0f (continued %.0f) vertices=%.0f | draws into",
			            calls(0), continued(0), calls(1), continued(1), calls(2));
			for (size_t t = 0; t < LiveCensus::DrawTargets; ++t)
				std::printf(" %s=%.0f (%.0f)", LiveCensus::DrawTargetNames[t],
				            static_cast<double>(now.draw_targets[t][0] - last.draw_targets[t][0]) / n,
				            static_cast<double>(now.draw_targets[t][1] - last.draw_targets[t][1]) / n);
		}
		// (thread1) The two-thread what-if (LiveCensus::Par2; nothing runs differently): a frame's slices by side (dcb
		// the graphics queue, acb the async compute queues), ms and slices (and those that stopped at a wait); the ms
		// both sides would take on a thread each with the game's labels kept in order and a free hand-over
		// (two_threads: the split's ceiling is serial - two_threads); the ms a side was held back waiting for a label of
		// the other (stalled); the waits satisfied by a label written here and those of them that held their side back.
		{
			const auto side = [&](size_t q) { return static_cast<double>(now.par2_side[q] - last.par2_side[q]) * per_cycle; };
			const auto slices = [&](size_t q, size_t i) {
				return static_cast<double>(now.par2_slices[q][i] - last.par2_slices[q][i]) / n;
			};
			std::printf(" | par2 serial=%.2f ms (dcb %.2f, %.0f slices, %.0f at a wait; acb %.2f, %.0f slices, %.0f at a wait) "
			            "two_threads=%.2f ms stalled=%.2f ms label_waits=%.0f held=%.0f",
			            side(0) + side(1), side(0), slices(0, 0), slices(0, 1), side(1), slices(1, 0), slices(1, 1),
			            static_cast<double>(now.par2_elapsed - last.par2_elapsed) * per_cycle,
			            static_cast<double>(now.par2_stalled - last.par2_stalled) * per_cycle,
			            static_cast<double>(now.par2_waits[0] - last.par2_waits[0]) / n,
			            static_cast<double>(now.par2_waits[1] - last.par2_waits[1]) / n);
		}
		// (render5) The switches' modes and their verifications' differences since start (each must stay 0; the
		// [render5] line has them a frame).
		std::printf(" | render5 census_light=%u reg_lane=%u (different %llu) draw_flags=%u (mismatches %llu)",
		            Render5::Mode(Render5::CensusLight), Render5::Mode(Render5::RegLane),
		            static_cast<unsigned long long>(Render5::g_lane[Render5::LaneDifferent]), Render5::Mode(Render5::DrawFlags),
		            static_cast<unsigned long long>(Render5::g_flags[Render5::FlagMismatches].load(std::memory_order_relaxed)));
		std::printf("\n");
		std::fflush(stdout);
	} else {
		// (render4) No line (the first window): the shaders' table starts with the next.
		for (auto& entry: LiveCensus::g_shader_stats) entry = {};
		LiveCensus::g_shader_stats_dropped = 0;
	}
	last   = now;
	frames = 0;
}

// (render5) A [render5] line every 5 s while any of KYTY_CENSUS_LIGHT, KYTY_REG_LANE, KYTY_DRAW_FLAGS is on (render5.h),
// whatever KYTY_RENDER_STATS says, a frame's share: the census scopes skipped (1) or that 1 would skip (2, with what
// they cost at the scope cost measured at start); the register lane's runs and packets, and with 2 the guest commands
// and feedback write-backs it would postpone and its differences from the walk (must be 0); the draw flags' checks and
// mismatches (2; must be 0). Render thread.
inline void Render5Report(int64_t flip_ns) {
	static int64_t  last_ns   = 0;
	static uint64_t last_tsc  = 0, frames = 0;
	static std::array<uint64_t, Render5::LaneCounters>   last_lane {};
	static std::array<uint64_t, Render5::CensusCounters> last_census {};
	static std::array<uint64_t, Render5::FlagCounters>   last_flags {};
	++frames;
	if (last_ns != 0 && flip_ns - last_ns < 5'000'000'000ll) return;
	const uint32_t census = Render5::Mode(Render5::CensusLight), lane = Render5::Mode(Render5::RegLane),
	               flags = Render5::Mode(Render5::DrawFlags);
	const uint64_t tsc = __rdtsc();
	if (last_ns != 0 && (census | lane | flags) != 0 && frames != 0 && tsc > last_tsc) {
		const double n          = static_cast<double>(frames);
		const double ms_per_tsc = static_cast<double>(flip_ns - last_ns) / 1e6 / static_cast<double>(tsc - last_tsc);
		const auto   lane_d     = [&](uint32_t i) { return static_cast<double>(Render5::g_lane[i] - last_lane[i]) / n; };
		const auto   census_d   = [&](uint32_t i) { return static_cast<double>(Render5::g_census[i] - last_census[i]) / n; };
		const auto   flags_d    = [&](uint32_t i) {
			return static_cast<double>(Render5::g_flags[i].load(std::memory_order_relaxed) - last_flags[i]) / n;
		};
		std::printf("[render5] %.1f s, %llu frames (%.1f fps) | census_light=%u skipped=%.0f would_skip=%.0f",
		            static_cast<double>(flip_ns - last_ns) / 1e9, static_cast<unsigned long long>(frames), n * 1e9 / static_cast<double>(flip_ns - last_ns), census,
		            census_d(Render5::CensusSkipped), census_d(Render5::CensusWouldSkip));
		if (Render5::g_census_scope_cycles != 0)
			std::printf(" (~%.2f ms a frame at %llu cycles a scope)",
			            census_d(Render5::CensusWouldSkip) * static_cast<double>(Render5::g_census_scope_cycles) * ms_per_tsc,
			            static_cast<unsigned long long>(Render5::g_census_scope_cycles));
		std::printf(" | reg_lane=%u runs=%.0f packets=%.0f postponed_commands=%.1f postponed_feedback=%.1f different=%.0f", lane,
		            lane_d(Render5::LaneRuns), lane_d(Render5::LanePackets), lane_d(Render5::LanePostponed),
		            lane_d(Render5::LaneFeedback), lane_d(Render5::LaneDifferent));
		std::printf(" | draw_flags=%u checks=%.0f mismatches=%.0f\n", flags, flags_d(Render5::FlagChecks),
		            flags_d(Render5::FlagMismatches));
		std::fflush(stdout);
	}
	for (uint32_t i = 0; i < Render5::LaneCounters; ++i) last_lane[i] = Render5::g_lane[i];
	for (uint32_t i = 0; i < Render5::CensusCounters; ++i) last_census[i] = Render5::g_census[i];
	for (uint32_t i = 0; i < Render5::FlagCounters; ++i) last_flags[i] = Render5::g_flags[i].load(std::memory_order_relaxed);
	last_ns  = flip_ns;
	last_tsc = tsc;
	frames   = 0;
}

// (mdraw C0/C1) KYTY_MDRAW_STATS=1 / KYTY_MDRAW_SKIP=1|2 (LiveCensus::Mdraw in live-census.h, measured in table-xpr.inc
// TableDraw): an [mdraw] line every 5 s, per frame over the window: the render thread's frame, cpu, idle and
// submissions (as the [render] line takes them: with SKIP=2 the ON and OFF windows' cpu compared is the real saving),
// the table calls, the multi-draw groups by size (calls), the members (calls a group would drop) and their measured
// cycles (whole call, emission, preparation), why groups ended, what differs inside them. Render thread (Flip).
inline void MdrawReport(int64_t flip_ns) {
	if (!LiveCensus::Mdraw::Active()) return;
	namespace Md = LiveCensus::Mdraw;
	static int64_t  last_ns = 0, last_idle = 0;
	static uint64_t last_tsc = 0, last_thread = 0, last_submission = 0, frames = 0;
	++frames;
	if (last_ns != 0 && flip_ns - last_ns < 5'000'000'000ll) return;
	const uint64_t tsc        = __rdtsc();
	const uint64_t thread     = LocalPlatform::CurrentThreadCycles();
	const uint64_t submission = LiveCensus::g_kind_cycles[LiveCensus::Submission];
	Md::Close(); // (the open group, counted in this window)
	auto&          c    = Md::g_c;
	const uint32_t skip = Md::SkipMode();
	if (last_ns != 0 && frames != 0 && tsc > last_tsc) {
		const double n          = static_cast<double>(frames);
		const double wall_ms    = static_cast<double>(flip_ns - last_ns) / 1e6;
		const double ms_per_tsc = wall_ms / static_cast<double>(tsc - last_tsc);
		const auto   each       = [&](uint64_t value) { return static_cast<double>(value) / n; };
		const auto   ms         = [&](uint64_t cycles) { return static_cast<double>(cycles) * ms_per_tsc / n; };
		const char*  window     = skip == 0 ? "stats" : skip == 1 ? "SKIP" : Md::g_window_on ? "ON" : "OFF";
		const double us_member  = c.members != 0 ? static_cast<double>(c.member_cycles) * ms_per_tsc * 1e3 / static_cast<double>(c.members) : 0.0;
		const double us_call    = c.calls != 0 ? static_cast<double>(c.all_cycles) * ms_per_tsc * 1e3 / static_cast<double>(c.calls) : 0.0;
		std::printf("[mdraw] %s %.1f s, %llu frames (%.1f fps): frame %.2f ms, cpu %.2f, idle %.2f, submissions %.2f | calls %.0f "
		            "draws %.0f groups %.1f members %.1f (draws %.1f) skipped %.1f (draws %.1f) | hist",
		            window, wall_ms / 1e3, static_cast<unsigned long long>(frames), n * 1e3 / wall_ms, wall_ms / n,
		            last_thread != 0 && thread != 0 ? static_cast<double>(thread - last_thread) * ms_per_tsc / n : 0.0,
		            static_cast<double>(g_render_idle_ns - last_idle) / 1e6 / n, ms(submission - last_submission),
		            each(c.calls), each(c.draws), each(c.groups), each(c.members), each(c.member_draws), each(c.skipped),
		            each(c.skipped_draws));
		for (uint32_t i = 0; i < Md::Buckets; ++i) std::printf(" %s=%.1f", Md::BucketNames[i], each(c.hist[i]));
		std::printf(" | t_total_candidatos %.3f ms t_emision %.3f t_preparacion_por_draw %.3f | us_member %.3f us_call %.3f "
		            "emision_call %.3f ms | ahorro_potencial_ms %.3f = %.1f eliminables x %.3f us | breaks",
		            ms(c.member_cycles), ms(c.member_emit), ms(c.member_prep), us_member, us_call, ms(c.all_emit),
		            each(c.members) * us_member / 1e3, each(c.members), us_member);
		for (uint32_t i = 0; i < Md::Breaks; ++i)
			if (c.breaks[i] != 0) std::printf(" %s=%.1f", Md::BreakNames[i], each(c.breaks[i]));
		std::printf(" | fields");
		for (uint32_t i = 0; i < Md::Fields; ++i)
			if (c.fields[i] != 0) std::printf(" %s=%.1f", Md::FieldNames[i], each(c.fields[i]));
		std::printf(" | ud_member");
		for (uint32_t i = 0; i < 32; ++i)
			if (c.ud_member[0][i] != 0) std::printf(" vs%u=%.1f", i, each(c.ud_member[0][i]));
		std::printf(" | ud_break");
		for (uint32_t st = 0; st < 2; ++st)
			for (uint32_t i = 0; i < 32; ++i)
				if (c.ud_break[st][i] != 0) std::printf(" %s%u=%.1f", st == 0 ? "vs" : "ps", i, each(c.ud_break[st][i]));
		std::printf("\n");
		std::fflush(stdout);
		if (skip == 2) Md::g_window_on = !Md::g_window_on; // (the next window the other way)
	}
	c                  = Md::Counters {};
	Md::g_last.valid   = false; // (groups do not cross windows)
	last_ns            = flip_ns;
	last_idle          = g_render_idle_ns;
	last_tsc           = tsc;
	last_thread        = thread;
	last_submission    = submission;
	frames             = 0;
}

// Render thread. With KYTY_HITCH_LOG_MS (or KYTY_SLOW_LOG_MS), a frame (flip to flip) that long and
// at least 20 ms is a SLOW line too, on the same TSC timeline: the hitch the slow calls before it explain.
// The line also has the frame's render-thread wait for guest work, what some counters did in it and its
// wall-clock time.
inline void Flip() {
	const auto flip    = g_flips.load(std::memory_order_relaxed);
	const auto flip_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                         std::chrono::steady_clock::now().time_since_epoch())
	                         .count();
	g_flip_times[flip % FlipTimes].store(flip_ns, std::memory_order_relaxed);
	g_flips.store(flip + 1, std::memory_order_release);
	// (KYTY_FPS_HUD's frame-time graph: the frame and the render thread's busy part of it.)
	PerfOverlay::NoteFlip(flip_ns, g_render_idle_ns);
	RenderStats(flip_ns); // (render2, KYTY_RENDER_STATS)
	ParAudit::OnFlip(); // (A0b) window clock and report, KYTY_AUDIT only
	MdrawReport(flip_ns); // (mdraw C0/C1, KYTY_MDRAW_STATS / KYTY_MDRAW_SKIP; one relaxed load when off)
	if (LiveCensus::RenderStatsOn()) Otros1::Report(flip_ns, 0.0); // (otros1, KYTY_OTROS1)
	if (LiveCensus::RenderStatsOn()) Otros2::Report(flip_ns);      // (otros2, KYTY_OTROS2)
	if (LiveCensus::RenderStatsOn()) Gancho2::Report(flip_ns);     // (gancho2, KYTY_GANCHO2)
	Port1::Report(flip_ns);                                         // (port1, KYTY_PORT1; one load when off)
	Port2::Report(flip_ns);                                         // (port2, KYTY_PORT2; one load when off)
	P7::Report(flip_ns);                                            // (p7, KYTY_P7; one load when off)
	Tiron3::OnFlip(flip_ns, g_render_tid.load(std::memory_order_relaxed), g_stack_low, g_stack_high); // (lab1 tiron3)
	G1::Flip(flip_ns);                                              // (g1, KYTY_G1_PROBE / KYTY_G1_STATE; one load when off)
	if (F5p1Techo::On()) // (f5p1 techo, KYTY_F5P1_TECHO)
		F5p1Techo::Report(flip_ns, g_render_idle_ns, LiveCensus::g_kind_cycles[LiveCensus::Submission]);
	if (F0Measure::On()) // (f0, KYTY_F0)
		F0Measure::Flip(flip_ns, g_render_idle_ns,
		                LiveCensus::g_waits_ns[LiveCensus::WaitGpu] + LiveCensus::g_waits_ns[LiveCensus::WaitReadback] +
		                    LiveCensus::g_waits_ns[LiveCensus::WaitDownload],
		                Port2::Mode() == 1 ? (Port2::g_window_on.load(std::memory_order_relaxed) ? 1 : 2) : G1::AbWindow()); // (g1: STATE=1's windows)
	LiveCensus::Par2::Flipped(); // (trust1: the two-thread what-if's frame ends at this flip)
	TrustedDepth::OnFlip(flip_ns); // (trust1: the [trust1] line, KYTY_RENDER_STATS / KYTY_DS_TRUSTED_DEPTH)
	Render5Report(flip_ns); // (render5)
	Techo::Report(flip_ns, g_render_idle_ns); // (techo; nada si apagado)
	DsSombrasAb::OnFlip(flip_ns, g_render_idle_ns, LiveCensus::g_waits_ns[LiveCensus::WaitCompile]); // (sombras-ab; nada si apagado)
	R0Ab::OnFlip(flip_ns, g_render_idle_ns); // (r0, KYTY_R0_AB; nada si apagado)
	Copia1::NoteFrame();    // (copia1: the [copia1] line's frames)
	if (SlowLog::HitchThreshold() > 0.0) {
		using C = LiveCounters::Id;
		static constexpr std::array counted {C::XprStores, C::XprStored, C::XprStorePending, C::XprRefuseBind,
		                                     C::XprMissRefused, C::XprMissValidate, C::XprMissUnseen, C::XprMissBudget,
		                                     C::UploadBytes, C::AsyncImageBytes, C::FullUploadBytes,
		                                     C::UploadWaitUs, C::UploadBusyUs, C::UploadCallUs, C::UploadSharedRuns, C::UploadGuardStops, C::SparseUploadPieces,
		                                     C::BufferRegistrations, C::RegionSyncs, C::SyncDownloads, C::AsyncReadbacks,
		                                     C::ReadbackParts, C::ReadbackRegions, C::GuestCommands, C::TextureUnmaps, C::AsyncPipelines,
		                                     C::ImageInitUs, C::BufferSyncUs, C::DrawUs, C::TextureUnmapUs, C::BackingReadUs, C::RegionSyncUs, C::AsyncDrawWaitUs, C::AsyncDrawSkips, C::QueueLockUs,
		                                     C::TranslateUs, C::Translates, C::ComputePipelineUs, C::ComputePipelines,
		                                     C::GraphicsPipelineUs, C::GraphicsPipelines, C::FlipSlotWaitUs,
		                                     C::TableDraws, C::TableStores, C::TableStoreVariant, C::TableStoreTargets,
		                                     C::TableRefusedSets, C::TableNoSet, C::TableDrawNative, C::TableEvaluations,
		                                     C::TableStoreDeferred};
		static constexpr const char* waits[] = {"gpu_wait", "readback_wait", "download_wait", "compile", "record_wait"};
		static std::chrono::steady_clock::time_point     last {};
		static uint64_t                                  last_tsc = 0;
		static std::array<uint64_t, LiveCensus::Kinds>   last_kind_cycles {};
		static std::array<uint64_t, LiveCensus::Phases>  last_dispatch_phases {}, last_draw_phases {};
		static std::array<uint64_t, 4>                   last_xpr {};
		static int64_t                                   last_idle = 0;
		static int64_t                                   last_upload_wait = 0;
		static std::array<int64_t, LiveCensus::Waits>    last_waits {};
		static std::array<uint64_t, counted.size()>      last_counts {};
		// The render thread's run time (TSC ticks) and a TSC rate from the first flip on: busy wall time
		// (the frame less idle) well over cpu is time the thread was ready but not running (preempted).
		static const uint64_t                            first_tsc  = __rdtsc();
		static const auto                                first_time = std::chrono::steady_clock::now();
		static uint64_t                                  last_cycles = 0;
		const uint64_t                                   cycles      = LocalPlatform::CurrentThreadCycles();
		const auto                                       now = std::chrono::steady_clock::now();
		const uint64_t                                   tsc = __rdtsc();
		const double ms = std::chrono::duration<double, std::milli>(now - last).count();
		// KYTY_FPS_LOG=1: a line every second with the frames of that second (count, mean and longest), the render
		// thread's mean run time, idle and waits per frame, and how many frames passed 17 ms (missed 60 fps): what a
		// frame rate that fluctuates under 60 is made of, below the SLOW Frame threshold.
		static const bool fps_log = [] {
			const char* text = std::getenv("KYTY_FPS_LOG");
			return text != nullptr && *text != '\0' && std::string_view(text) != "0";
		}();
		if (fps_log && last != std::chrono::steady_clock::time_point {}) {
			struct Second {
				std::chrono::steady_clock::time_point start {};
				uint32_t frames = 0, missed = 0, slow = 0;
				double   sum_ms = 0, max_ms = 0, sum_cpu = 0, sum_idle = 0;
				std::array<double, LiveCensus::Waits> sum_waits {};
			};
			static Second second;
			if (second.start == std::chrono::steady_clock::time_point {}) second.start = now;
			const double since_first = std::chrono::duration<double, std::milli>(now - first_time).count();
			const double cpu_ms      = cycles != 0 && last_cycles != 0 && since_first > 1000.0
			                               ? static_cast<double>(cycles - last_cycles) * since_first /
			                                     static_cast<double>(__rdtsc() - first_tsc)
			                               : 0.0;
			second.frames++;
			second.sum_ms += ms;
			second.max_ms = std::max(second.max_ms, ms);
			second.sum_cpu += cpu_ms;
			second.sum_idle += static_cast<double>(g_render_idle_ns - last_idle) / 1e6;
			for (size_t i = 0; i < LiveCensus::Waits; ++i)
				second.sum_waits[i] += static_cast<double>(LiveCensus::g_waits_ns[i] - last_waits[i]) / 1e6;
			if (ms > 17.0) second.missed++;
			if (ms > 25.0) second.slow++;
			if (now - second.start >= std::chrono::seconds(1)) {
				const double n = static_cast<double>(second.frames);
				double fp_mean = 0, fp_sd = 0, pi_mean = 0, pi_sd = 0;
				(void)Libs::VideoOut::TakePresentTiming(fp_mean, fp_sd, pi_mean, pi_sd);
				std::printf("[fps] %u frames: mean %.1f ms, max %.1f, cpu %.1f, idle %.1f, gpu_wait %.2f, readback %.2f, download %.2f, "
				            "compile %.2f, record %.2f; over 17 ms: %u, over 25 ms: %u; flip->present mean %.2f sd %.2f ms, "
				            "present interval mean %.2f sd %.2f ms\n",
				            second.frames, second.sum_ms / n, second.max_ms, second.sum_cpu / n, second.sum_idle / n,
				            second.sum_waits[0] / n, second.sum_waits[1] / n, second.sum_waits[2] / n, second.sum_waits[3] / n,
				            second.sum_waits[4] / n, second.missed, second.slow, fp_mean, fp_sd, pi_mean, pi_sd);
				// The guest threads' tax that second: their write and read faults, protection re-arms and
				// synchronous GPU reads (what the game's own thread spends outside its code).
				using G = LiveCounters::Id;
				static constexpr std::array guest {G::WindowFault, G::WriteFault, G::ReadFault, G::Reprotect, G::Unprotect,
				                                   G::ProtectCalls, G::SyncReadsGuest, G::SyncDownloads, G::GuestCommands,
				                                   G::BackingLockWaitUs, G::BackingWriteWaitUs, G::UploadWaitUs,
				                                   G::ProtectCallsRender, G::ProtectCallsGfx, G::RenderReadFaults, G::SyncReadsRender,
				                                   G::AddressSpaceWaitKtRender, G::TimelineQueriesRender, G::QueueLockUs, G::TranslateUs};
				static std::array<uint64_t, guest.size()> last_guest {};
				std::printf("[fps+]");
				for (size_t i = 0; i < guest.size(); ++i) {
					const auto value = LiveCounters::Value(guest[i]);
					std::printf(" %s=%llu", LiveCounters::Names[guest[i]], static_cast<unsigned long long>(value - last_guest[i]));
					last_guest[i] = value;
				}
				std::printf("\n");
				if (LaneSync::g_on) {
					static uint64_t last_acq[4] {};
					const uint64_t  now_acq[4] {LaneSync::g_submits[0].load(), LaneSync::g_submits[1].load(),
					                            LaneSync::g_waits[0].load(), LaneSync::g_waits[1].load()};
					std::printf("[acq] submits graphics %llu compute %llu; cross waits on graphics %llu on compute %llu\n",
					            static_cast<unsigned long long>(now_acq[0] - last_acq[0]), static_cast<unsigned long long>(now_acq[1] - last_acq[1]),
					            static_cast<unsigned long long>(now_acq[2] - last_acq[2]), static_cast<unsigned long long>(now_acq[3] - last_acq[3]));
					for (int n = 0; n < 4; ++n) last_acq[n] = now_acq[n];
					static uint64_t last_why[6] {};
					std::printf("[acq] compute waited for graphics by: image read %llu, image write %llu, range read %llu, range write %llu, reuse %llu\n",
					            static_cast<unsigned long long>(LaneSync::g_why[0] - last_why[0]), static_cast<unsigned long long>(LaneSync::g_why[1] - last_why[1]),
					            static_cast<unsigned long long>(LaneSync::g_why[2] - last_why[2]), static_cast<unsigned long long>(LaneSync::g_why[3] - last_why[3]),
					            static_cast<unsigned long long>(LaneSync::g_why[4] - last_why[4]));
					for (int n = 0; n < 6; ++n) last_why[n] = LaneSync::g_why[n];
				}
				std::fflush(stdout);
				second       = {};
				second.start = now;
			}
		}
		if (last != std::chrono::steady_clock::time_point {} && ms >= std::max(20.0, SlowLog::HitchThreshold())) {
			std::printf("[tsc %llu] SLOW Frame %.1f ms idle=%.1f", static_cast<unsigned long long>(tsc), ms,
			            static_cast<double>(g_render_idle_ns - last_idle) / 1e6);
			if (const double since = std::chrono::duration<double, std::milli>(now - first_time).count();
			    cycles != 0 && last_cycles != 0 && since > 1000.0)
				std::printf(" cpu=%.1f", static_cast<double>(cycles - last_cycles) * since /
				                             static_cast<double>(__rdtsc() - first_tsc));
			// The render thread's time by call kind (ms), from the frame's own TSC rate.
			const double ms_per_cycle = tsc > last_tsc ? ms / static_cast<double>(tsc - last_tsc) : 0.0;
			for (size_t i = 0; i < LiveCensus::Kinds; ++i) {
				const double kind_ms = static_cast<double>(LiveCensus::g_kind_cycles[i] - last_kind_cycles[i]) * ms_per_cycle;
				if (kind_ms >= 0.05) std::printf(" %s=%.1f", LiveCensus::KindNames[i], kind_ms);
			}
			for (size_t i = 0; i < LiveCensus::Phases; ++i) {
				const double phase_ms = static_cast<double>(LiveCensus::g_dispatch_phase_cycles[i] - last_dispatch_phases[i]) * ms_per_cycle;
				if (phase_ms >= 0.5) std::printf(" dispatch.%s=%.1f", LiveCensus::DispatchPhaseNames[i], phase_ms);
			}
			for (size_t i = 0; i < LiveCensus::Phases; ++i) {
				const double phase_ms = static_cast<double>(LiveCensus::g_draw_phase_cycles[i] - last_draw_phases[i]) * ms_per_cycle;
				if (phase_ms >= 0.5) std::printf(" draw.%s=%.1f", LiveCensus::DrawPhaseNames[i], phase_ms);
			}
			for (size_t i = 0; i < 4; ++i) {
				const double xpr_ms = static_cast<double>(LiveCensus::g_xpr_cycles[i] - last_xpr[i]) * ms_per_cycle;
				if (xpr_ms >= 0.5) std::printf(" xpr.%zu=%.1f", i, xpr_ms);
			}
			for (size_t i = 0; i < LiveCensus::Waits; ++i)
				if (const auto ns = LiveCensus::g_waits_ns[i] - last_waits[i]; ns != 0)
					std::printf(" %s=%.1f", waits[i], static_cast<double>(ns) / 1e6);
			if (const auto ns = LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed) - last_upload_wait; ns != 0)
				std::printf(" upload_wait=%.1f", static_cast<double>(ns) / 1e6);
			for (size_t i = 0; i < counted.size(); ++i) {
				const auto delta = LiveCounters::Value(counted[i]) - last_counts[i];
				if (delta != 0) std::printf(" %s=%llu", LiveCounters::Names[counted[i]], static_cast<unsigned long long>(delta));
			}
			// Wall-clock time, to find the frame in a screen capture.
			std::printf(" unix_ms=%lld\n", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
			                                   std::chrono::system_clock::now().time_since_epoch())
			                                   .count()));
			std::fflush(stdout);
		}
		// (tiron1, KYTY_TIRON1=1) The frame by cause, from the SLOW Frame line's sources; a [tiron1] line every 5 s.
		if (Tiron1::On() && last != std::chrono::steady_clock::time_point {}) {
			Tiron1::FrameCost cost;
			cost.ms = ms;
			if (ms > 25.0) {
				const double per_cycle = tsc > last_tsc ? ms / static_cast<double>(tsc - last_tsc) : 0.0;
				const auto   kind_ms   = [&](size_t kind) {
                    return static_cast<double>(LiveCensus::g_kind_cycles[kind] - last_kind_cycles[kind]) * per_cycle;
				};
				const auto count = [&](C id) -> uint64_t {
					for (size_t i = 0; i < counted.size(); ++i)
						if (counted[i] == id) return LiveCounters::Value(id) - last_counts[i];
					return 0;
				};
				const auto wait_ms = [&](size_t wait) { return static_cast<double>(LiveCensus::g_waits_ns[wait] - last_waits[wait]) / 1e6; };
				cost.idle           = static_cast<double>(g_render_idle_ns - last_idle) / 1e6;
				cost.draw           = kind_ms(LiveCensus::Draw);
				cost.xpr            = kind_ms(LiveCensus::NativeXpr);
				cost.programs       = kind_ms(LiveCensus::GraphicsPrograms) +
				                static_cast<double>(LiveCensus::g_dispatch_phase_cycles[0] - last_dispatch_phases[0]) * per_cycle;
				cost.dispatch_table = static_cast<double>(LiveCensus::g_dispatch_phase_cycles[10] - last_dispatch_phases[10]) * per_cycle;
				cost.compile        = wait_ms(LiveCensus::WaitCompile);
				cost.gpu_waits      = wait_ms(LiveCensus::WaitGpu) + wait_ms(LiveCensus::WaitReadback) +
				                 wait_ms(LiveCensus::WaitDownload) + wait_ms(LiveCensus::WaitRecord);
				cost.upload_wait    = static_cast<double>(LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed) - last_upload_wait) / 1e6;
				cost.image_init     = static_cast<double>(count(C::ImageInitUs)) / 1000.0;
				cost.store_deferred = count(C::TableStoreDeferred);
				cost.xpr_stores     = count(C::XprStores);
			}
			Tiron1::NoteFrame(cost, now);
		}
		last      = now;
		last_tsc  = tsc;
		last_idle = g_render_idle_ns;
		last_upload_wait = LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed);
		for (size_t i = 0; i < LiveCensus::Kinds; ++i) last_kind_cycles[i] = LiveCensus::g_kind_cycles[i];
		for (size_t i = 0; i < LiveCensus::Phases; ++i) {
			last_dispatch_phases[i] = LiveCensus::g_dispatch_phase_cycles[i];
			last_draw_phases[i]     = LiveCensus::g_draw_phase_cycles[i];
		}
		for (size_t i = 0; i < 4; ++i) last_xpr[i] = LiveCensus::g_xpr_cycles[i];
		last_cycles = cycles;
		for (size_t i = 0; i < LiveCensus::Waits; ++i) last_waits[i] = LiveCensus::g_waits_ns[i];
		for (size_t i = 0; i < counted.size(); ++i) last_counts[i] = LiveCounters::Value(counted[i]);
	}
	// (agente2) Standard output is fully buffered (ds-main.cpp): what any thread printed since the last flip reaches
	// the log now, in one write (a crash loses at most one frame of lines). Nothing buffered: a lock and a test.
	std::fflush(stdout);
}

inline double ThreadCpuSeconds(uint64_t thread) {
	return LocalPlatform::ThreadCpuSeconds(thread);
}

inline double NamedThreadsCpuSeconds(const char* name) {
	return LocalPlatform::NamedThreadsCpuSeconds(name);
}

#if !defined(_WIN32)
inline void ProfSignal(int /*signal*/, siginfo_t* /*info*/, void* context) {
	const auto* uc  = static_cast<const ucontext_t*>(context);
	const auto  pc  = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
	const auto  rsp = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
	const auto  idx = g_sample_count.fetch_add(1, std::memory_order_relaxed);
	if (g_samples != nullptr && SampleWords * (size_t(idx) + 1) <= SampleSlots) {
		auto* record = g_samples + SampleWords * size_t(idx);
		record[0]    = pc;
		if (g_process.load(std::memory_order_relaxed)) {
			record[1] = static_cast<uint64_t>(LocalPlatform::ThreadId()) | (1ull << 63);
			for (size_t i = 2; i < SampleWords; ++i) record[i] = 0;
		} else {
			// Word at rsp (the caller of a leaf routine), then return addresses from the
			// frame-pointer chain, bounded to this stack.
			record[1]    = *reinterpret_cast<const uint64_t*>(rsp);
			auto rbp     = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RBP]);
			for (size_t i = 2; i < SampleWords; ++i) {
				record[i] = 0;
				if (rbp < rsp || rbp >= g_stack_high || g_stack_high - rbp < 16 || (rbp & 7u) != 0) continue;
				const auto* frame = reinterpret_cast<const uint64_t*>(rbp);
				record[i]         = frame[1];
				if (frame[0] <= rbp) {
					rbp = 0;
					continue;
				}
				rbp = frame[0];
			}
		}
	}
}

inline void Profile(uint64_t id, double seconds, const char* path, bool process, bool wall = false) {
	if (g_render_tid.load() == 0) return;
	g_process.store(process);
	struct sigaction action {};
	action.sa_sigaction = ProfSignal;
	action.sa_flags     = SA_SIGINFO | SA_RESTART;
	sigemptyset(&action.sa_mask);
	sigaction(SIGPROF, &action, nullptr);
	clockid_t clock = CLOCK_PROCESS_CPUTIME_ID;
	if (wall) {
		clock = CLOCK_MONOTONIC;
	} else if (!process && pthread_getcpuclockid(static_cast<pthread_t>(g_render_thread), &clock) != 0) {
		return;
	}
	sigevent event {};
	event.sigev_signo = SIGPROF;
	if (process) {
		event.sigev_notify = SIGEV_SIGNAL;
	} else {
		event.sigev_notify   = SIGEV_THREAD_ID;
		event._sigev_un._tid = g_render_tid.load();
	}
	if (!EnsureSamples()) return;
	timer_t timer {};
	if (timer_create(clock, &event, &timer) != 0) return;
	g_sample_count.store(0);
	itimerspec spec {};
	spec.it_interval.tv_nsec = 1000000000L / 4000;
	spec.it_value            = spec.it_interval;
	timer_settime(timer, 0, &spec, nullptr);
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	timer_delete(timer);
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	const auto count = std::min<uint32_t>(g_sample_count.load(), SampleSlots / SampleWords);
	if (auto* out = std::fopen(path, "wb")) {
		std::fwrite(g_samples, sizeof(uint64_t), SampleWords * size_t(count), out);
		std::fclose(out);
	}
	const std::string maps_path = std::string(path) + ".maps";
	if (auto* in = std::fopen("/proc/self/maps", "re")) {
		if (auto* out = std::fopen(maps_path.c_str(), "we")) {
			char buffer[65536];
			for (size_t got; (got = std::fread(buffer, 1, sizeof(buffer), in)) > 0;) std::fwrite(buffer, 1, got, out);
			std::fclose(out);
		}
		std::fclose(in);
	}
	std::printf("LIVE_PROF id=%" PRIu64 " samples=%u path=%s\n", id, count, path);
}
#else
// Windows: no per-thread CPU-time timers; prof and profw both sample the render thread at
// 4 kHz of wall time by suspending it. Process-wide sampling (profp) is not available.
// proft <tid>: another thread (a guest thread on its own stack) instead of the render thread.
inline void Profile(uint64_t id, double seconds, const char* path, bool process, bool /*wall*/ = false,
                    uint32_t tid = 0) {
	if (g_render_tid.load() == 0 || g_render_thread == 0) return;
	if (process) {
		std::printf("LIVE_ERROR id=%" PRIu64 " line=profp (not supported on Windows)\n", id);
		return;
	}
	if (!EnsureSamples()) {
		std::printf("LIVE_ERROR id=%" PRIu64 " line=prof (no memory for the samples)\n", id);
		return;
	}
	uint64_t thread = g_render_thread, stack_low = g_stack_low, stack_high = g_stack_high;
	if (tid != 0) {
		thread    = LocalPlatform::OpenThreadForSampling(tid);
		stack_low = stack_high = 0;
		if (thread == 0) {
			std::printf("LIVE_ERROR id=%" PRIu64 " line=proft %u (cannot open the thread)\n", id, tid);
			return;
		}
	}
	g_sample_count.store(0);
	LocalPlatform::PrepareSampling();
	// KYTY_PROF_HZ (default 4000): fewer suspensions distort lock and page-fault timing less.
	static const long hz = [] {
		const char* value = std::getenv("KYTY_PROF_HZ");
		const long  rate  = value != nullptr ? std::strtol(value, nullptr, 10) : 4000;
		return rate >= 100 && rate <= 20000 ? rate : 4000;
	}();
	const auto period = std::chrono::nanoseconds(1000000000L / hz);
	const auto begin  = std::chrono::steady_clock::now();
	const auto end    = begin + std::chrono::duration<double>(seconds);
	const auto tsc0   = __rdtsc();
	auto       next   = begin;
	while (next < end) {
		const auto idx = g_sample_count.load(std::memory_order_relaxed);
		if (SampleWords * (size_t(idx) + 1) > SampleSlots) break;
		if (LocalPlatform::SampleThread(thread, stack_low, stack_high, g_samples + SampleWords * size_t(idx),
		                                SampleWords)) {
			g_sample_tsc[idx] = __rdtsc();
			g_sample_count.store(idx + 1, std::memory_order_relaxed);
		}
		next += period;
		while (std::chrono::steady_clock::now() < next) _mm_pause();
	}
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const double tsc_hz  = static_cast<double>(__rdtsc() - tsc0) / elapsed;
	if (tid != 0) LocalPlatform::CloseThreadForSampling(thread);
	const auto count = std::min<uint32_t>(g_sample_count.load(), SampleSlots / SampleWords);
	if (auto* out = std::fopen(path, "wb")) {
		std::fwrite(g_samples, sizeof(uint64_t), SampleWords * size_t(count), out);
		std::fclose(out);
	}
	// <path>.tsc: each sample's TSC (slow-frame lines carry TSC stamps: hitch-prof.py picks their samples).
	if (auto* out = std::fopen((std::string(path) + ".tsc").c_str(), "wb")) {
		std::fwrite(g_sample_tsc, sizeof(uint64_t), size_t(count), out);
		std::fclose(out);
	}
	std::printf("LIVE_PROF id=%" PRIu64 " samples=%u path=%s tsc_hz=%.0f\n", id, count, path, tsc_hz);
}
#endif

inline void Measure(uint64_t id, double seconds, const char* label) {
	std::array<uint64_t, LiveCounters::Count> counters0 {};
	for (size_t i = 0; i < counters0.size(); ++i) counters0[i] = LiveCounters::Value(i);
	const auto   flips0  = g_flips.load();
	const double render0 = g_render_known ? ThreadCpuSeconds(g_render_thread) : 0;
	const double record0 = NamedThreadsCpuSeconds("Kyty.Record");
	const auto   begin   = std::chrono::steady_clock::now();
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const auto   frames  = g_flips.load() - flips0;
	const double render  = (g_render_known ? ThreadCpuSeconds(g_render_thread) : 0) - render0;
	const double record  = NamedThreadsCpuSeconds("Kyty.Record") - record0;
	const double per     = frames ? 1000.0 / static_cast<double>(frames) : 0;
	// Frame times (flip to flip) of the window: the worst 1% as fps (1% low), the 99th percentile, the
	// longest and the frames of 50 ms or more and of 100 ms or more.
	std::vector<double> times;
	for (auto flip = flips0 + 1; flip < flips0 + frames && flip + FlipTimes > flips0 + frames; ++flip)
		times.push_back(static_cast<double>(g_flip_times[flip % FlipTimes].load(std::memory_order_relaxed) -
		                                    g_flip_times[(flip - 1) % FlipTimes].load(std::memory_order_relaxed)) / 1e6);
	std::sort(times.begin(), times.end(), std::greater<>());
	const size_t worst = (times.size() + 99) / 100;
	double       sum   = 0;
	for (size_t i = 0; i < worst; ++i) sum += times[i];
	const double low1 = worst ? 1000.0 * static_cast<double>(worst) / sum : 0;
	const double p99  = times.empty() ? 0 : times[std::min(times.size() - 1, times.size() / 100)];
	const auto   over = [&](double ms) { return std::ranges::count_if(times, [&](double t) { return t >= ms; }); };
	std::printf("LIVE_MEASURE id=%" PRIu64 " label=%s seconds=%.2f frames=%" PRIu64
	            " fps=%.2f render_ms=%.2f record_ms=%.2f render_busy=%.1f%% low1=%.2f p99_ms=%.1f max_ms=%.1f"
	            " over50=%lld over100=%lld\n",
	            id, label, elapsed, frames, static_cast<double>(frames) / elapsed, render * per, record * per,
	            100.0 * render / elapsed, low1, p99, times.empty() ? 0 : times.front(),
	            static_cast<long long>(over(50)), static_cast<long long>(over(100)));
	std::string counters;
	for (size_t i = 0; i < counters0.size(); ++i) {
		char text[96];
		std::snprintf(text, sizeof(text), " %s=%.1f", LiveCounters::Names[i],
		              static_cast<double>(LiveCounters::Value(i) - counters0[i]) / (frames ? frames : 1));
		counters += text;
	}
	std::printf("LIVE_COUNTERS id=%" PRIu64 " label=%s%s\n", id, label, counters.c_str());
}

inline void Run(uint64_t id, const std::string& line) {
	char      command[16] {}, arg1[256] {}, arg2[256] {};
	const int n = std::sscanf(line.c_str(), "%15s %255s %255s", command, arg1, arg2);
	if (n < 1) return;
	const std::string_view cmd = command;
	if ((cmd == "poke32" || cmd == "peek") && n == 3 && std::strtoull(arg1, nullptr, 16) < 0x100000000ull) {
		// Host addresses of switches are far above 4 GiB; refuse guest-looking values.
		std::printf("LIVE_ERROR id=%" PRIu64 " line=%s (address below 4 GiB)\n", id, line.c_str());
	} else if (cmd == "poke32" && n == 3) {
		auto*      target = reinterpret_cast<void*>(std::strtoull(arg1, nullptr, 16));
		const auto value  = static_cast<uint32_t>(std::strtoul(arg2, nullptr, 16));
		std::memcpy(target, &value, 4);
	} else if (cmd == "peek" && n == 3) {
		const auto* data  = reinterpret_cast<const uint8_t*>(std::strtoull(arg1, nullptr, 16));
		auto        bytes = std::min<size_t>(std::strtoul(arg2, nullptr, 0), 256);
		// Only committed, readable memory (an unmapped address closed the game, 13-10).
		bytes = LocalPlatform::ReadableBytes(data, bytes);
		std::string hex;
		for (size_t i = 0; i < bytes; ++i) {
			char b[4];
			std::snprintf(b, sizeof(b), "%02x", data[i]);
			hex += b;
		}
		std::printf("LIVE_PEEK id=%" PRIu64 " va=%s bytes=%s\n", id, arg1, hex.c_str());
	} else if (cmd == "measure" && n >= 2) {
		Measure(id, std::strtod(arg1, nullptr), n == 3 ? arg2 : "-");
	} else if (cmd == "flips" && n == 3) {
		// flips <count> <path>: the last flips' intervals (us), oldest first, one a line (periodic slow frames).
		const auto last  = g_flips.load();
		const auto count = std::min<uint64_t>({std::strtoull(arg1, nullptr, 10), FlipTimes - 1, last ? last - 1 : 0});
		if (auto* out = std::fopen(arg2, "w")) {
			for (auto flip = last - count; flip < last; ++flip)
				std::fprintf(out, "%lld\n", static_cast<long long>((g_flip_times[flip % FlipTimes].load() -
				                                                    g_flip_times[(flip - 1) % FlipTimes].load()) / 1000));
			std::fclose(out);
		}
		std::printf("LIVE_FLIPS id=%" PRIu64 " count=%llu path=%s\n", id, static_cast<unsigned long long>(count), arg2);
	} else if ((cmd == "prof" || cmd == "profp" || cmd == "profw") && n == 3) {
		Profile(id, std::strtod(arg1, nullptr), arg2, cmd == "profp", cmd == "profw");
	} else if (cmd == "profbg" && n == 3) {
		// profbg <seconds> <path>: prof on a thread of its own, while the next commands run (a trace of the same
		// seconds: the samples' TSC stamps place them in its frames).
		std::thread([id, seconds = std::strtod(arg1, nullptr), path = std::string(arg2)] {
			Profile(id, seconds, path.c_str(), false, false);
		}).detach();
#if defined(_WIN32)
	} else if (cmd == "pinthread" && n == 3) {
		// pinthread <tid> <cpus>: any thread's affinity (placement A/B of the worker threads).
		const auto handle = LocalPlatform::OpenThreadForSampling(static_cast<uint32_t>(std::strtoul(arg1, nullptr, 10)));
		const bool ok     = handle != 0 && LocalPlatform::PinThreadHandleToCpuList(handle, arg2);
		LocalPlatform::CloseThreadForSampling(handle);
		std::printf("LIVE_PIN id=%" PRIu64 " thread=%s cpus=%s ok=%d\n", id, arg1, arg2, ok ? 1 : 0);
	} else if (cmd == "proft" && n == 3) {
		// proft <tid|name> <seconds> <path>: sample that thread (the file name follows the seconds); a name takes the
		// first thread of that name (e.g. Kyty.Record).
		char path[256] {};
		auto tid = static_cast<uint32_t>(std::strtoul(arg1, nullptr, 10));
		if (tid == 0)
			for (const auto& [thread, name]: LocalPlatform::ProcessThreads())
				if (name == arg1) {
					tid = thread;
					break;
				}
		if (tid == 0)
			std::printf("LIVE_ERROR id=%" PRIu64 " line=proft %s (no such thread)\n", id, arg1);
		else if (std::sscanf(line.c_str(), "%*s %*s %*s %255s", path) == 1)
			Profile(id, std::strtod(arg2, nullptr), path, false, false, tid);
#endif
	} else if ((cmd == "trace" || cmd == "tracew" || cmd == "tracem") && n == 3) {
		LiveTrace::g_count.store(0);
		LiveTrace::g_writes_on.store(cmd == "tracew");
		LiveTrace::g_mark_next.store(0);
		LiveTrace::g_marks_on.store(cmd == "tracem");
		const auto tsc0   = __rdtsc();
		const auto clock0 = std::chrono::steady_clock::now();
		if (!LiveTrace::EnsureRecords()) std::printf("LIVE_ERROR id=%" PRIu64 " line=trace (no memory for the records)\n", id);
		LiveTrace::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveTrace::g_on.store(false);
		LiveTrace::g_writes_on.store(false);
		LiveTrace::g_marks_on.store(false);
		const auto tsc1   = __rdtsc();
		const auto clock1 = std::chrono::steady_clock::now();
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		if (cmd == "tracem" && LiveTrace::g_read_marks)
			LiveTrace::g_read_marks(std::min(LiveTrace::g_mark_next.load(), LiveTrace::MarkSlots));
		LiveTrace::Dump(arg2);
		const double seconds = std::chrono::duration<double>(clock1 - clock0).count();
		std::printf("LIVE_TRACE id=%" PRIu64 " records=%" PRIu64 " tsc_hz=%.0f path=%s\n", id,
		            std::min<uint64_t>(LiveTrace::g_count.load(), LiveTrace::Capacity),
		            static_cast<double>(tsc1 - tsc0) / seconds, arg2);
	} else if (cmd == "census" && n >= 2) {
		// census <seconds>: render-thread time per call kind and shader over the window.
		LiveCensus::g_on.store(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		for (auto& entry: LiveCensus::g_table) entry = {};
		const auto flips0 = g_flips.load();
		const auto tsc0   = __rdtsc();
		const auto t0     = std::chrono::steady_clock::now();
		LiveCensus::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveCensus::g_on.store(false);
		const auto   frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		const double ns_per_cycle =
		    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
		    static_cast<double>(__rdtsc() - tsc0);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
#if defined(_WIN32)
		uint64_t base = 0;
		char     exe[512] {};
		if (!LocalPlatform::ModuleOf(reinterpret_cast<const void*>(&Flip), &base, exe, sizeof(exe))) exe[0] = '\0';
		std::printf("LIVE_CENSUS id=%" PRIu64 " frames=%" PRIu64 " base=%016" PRIx64 " exe=%s\n", id, frames, base,
		            exe[0] != '\0' ? exe : "?");
#else
		Dl_info self {};
		dladdr(reinterpret_cast<void*>(&Flip), &self);
		std::printf("LIVE_CENSUS id=%" PRIu64 " frames=%" PRIu64 " base=%016" PRIx64 " exe=%s\n", id, frames,
		            reinterpret_cast<uint64_t>(self.dli_fbase), self.dli_fname ? self.dli_fname : "?");
#endif
		for (const auto& entry: LiveCensus::g_table) {
			if (!entry.used) continue;
			std::printf("LIVE_CENSUS_ENTRY id=%" PRIu64 " kind=%u a=%016" PRIx64 " b=%016" PRIx64
			            " calls_per_frame=%.2f ms_per_frame=%.4f\n",
			            id, entry.kind, entry.a, entry.b, static_cast<double>(entry.calls) / frames,
			            static_cast<double>(entry.cycles) * ns_per_cycle / 1e6 / frames);
		}
	} else if (cmd == "timecensus" && n >= 2) {
		// timecensus <seconds>: time-related HLE calls per caller (guest return address).
		TimeCensus::g_on.store(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		for (auto& entry: TimeCensus::g_table) {
			entry.key.store(0);
			entry.calls.store(0);
			entry.arg_sum.store(0);
			entry.arg_last.store(0);
		}
		TimeCensus::g_overflow.store(0);
		const auto flips0   = g_flips.load();
		const auto vblanks0 = LiveCounters::Value(LiveCounters::Vblanks);
		const auto t0       = std::chrono::steady_clock::now();
		TimeCensus::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		TimeCensus::g_on.store(false);
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		const auto   frames  = g_flips.load() - flips0;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		const auto   vblanks = LiveCounters::Value(LiveCounters::Vblanks) - vblanks0;
		std::printf("LIVE_TIMECENSUS id=%" PRIu64 " seconds=%.2f frames=%" PRIu64 " vblanks=%" PRIu64
		            " overflow=%" PRIu64 "\n",
		            id, seconds, frames, vblanks, TimeCensus::g_overflow.load());
		for (const auto& entry: TimeCensus::g_table) {
			const auto key = entry.key.load();
			if (key == 0) continue;
			const auto calls = entry.calls.load();
			std::printf("LIVE_TIMECENSUS_ENTRY id=%" PRIu64 " source=%u caller=%012" PRIx64
			            " per_s=%.1f per_frame=%.2f arg_avg=%.1f arg_last=%" PRIu64 "\n",
			            id, static_cast<unsigned>(key & 0xffu), key >> 8u, static_cast<double>(calls) / seconds,
			            static_cast<double>(calls) / static_cast<double>(frames ? frames : 1),
			            calls ? static_cast<double>(entry.arg_sum.load()) / static_cast<double>(calls) : 0.0,
			            entry.arg_last.load());
		}
	} else if (cmd == "granules" && n >= 2) {
		// granules <seconds>: per 1 MiB granule, per frame: window pages, re-armed pages,
		// upload bytes and upload copies.
		for (auto& granule: LiveCounters::g_granules) {
			granule.tag.store(0);
			for (auto& value: granule.values) value.store(0);
		}
		const auto flips0 = g_flips.load();
		LiveCounters::g_granules_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveCounters::g_granules_on.store(false);
		const auto frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		for (const auto& granule: LiveCounters::g_granules) {
			const auto tag = granule.tag.load();
			if (tag == 0) continue;
			std::printf("LIVE_GRANULE id=%" PRIu64 " address=%09" PRIx64 " window_pages=%.1f reprotect_pages=%.1f"
			            " upload_kb=%.1f upload_calls=%.1f gpu_writes=%.2f gpu_write_kb=%.1f\n",
			            id, (tag - 1) << 20u, double(granule.values[0].load()) / frames,
			            double(granule.values[1].load()) / frames, double(granule.values[2].load()) / 1024.0 / frames,
			            double(granule.values[3].load()) / frames, double(granule.values[4].load()) / frames,
			            double(granule.values[5].load()) / 1024.0 / frames);
		}
		std::printf("LIVE_GRANULE_OVERFLOW id=%" PRIu64 " count=%" PRIu64 "\n", id, LiveCounters::g_granule_overflow.load());
	} else if (cmd == "pm4" && n >= 2) {
		// pm4 <seconds>: PM4 packets per frame by opcode.
		std::array<uint64_t, 256> before {};
		for (size_t i = 0; i < before.size(); ++i) before[i] = LiveCounters::g_pm4[i].load();
		const auto flips0 = g_flips.load();
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		const auto frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		for (size_t i = 0; i < before.size(); ++i) {
			const auto count = LiveCounters::g_pm4[i].load() - before[i];
			if (count != 0)
				std::printf("LIVE_PM4 id=%" PRIu64 " opcode=0x%02zx per_frame=%.2f\n", id, i,
				            static_cast<double>(count) / static_cast<double>(frames));
		}
	} else if (cmd == "pin" && n >= 2) {
		// pin <cpus>: the render thread's affinity (for same-process placement A/B).
		const bool ok = LocalPlatform::PinThreadHandleToCpuList(g_render_thread, arg1);
		std::printf("LIVE_PIN id=%" PRIu64 " cpus=%s ok=%d\n", id, arg1, ok ? 1 : 0);
	} else if (cmd == "cpuset" && n >= 2) {
		// cpuset <cpus|all>: the default CPU set of threads without their own (Windows only).
		const bool ok = LocalPlatform::SetProcessDefaultCpuList(std::string_view(arg1) == "all" ? "" : arg1);
		std::printf("LIVE_CPUSET id=%" PRIu64 " cpus=%s ok=%d\n", id, arg1, ok ? 1 : 0);
#if defined(KYTY_PGO_GENERATE)
	} else if (cmd == "pgo" && n >= 2) {
		// pgo <file>: write the instrumentation profile now (the emulator exits with quick_exit).
		__llvm_profile_set_filename(arg1);
		const int result = __llvm_profile_write_file();
		std::printf("LIVE_PGO id=%" PRIu64 " path=%s result=%d\n", id, arg1, result);
#endif
	} else if (cmd == "vma" && n >= 2) {
		// vma <path>: the GPU allocator's detailed statistics (JSON: every allocation's type and size).
		if (LiveCounters::g_vma_report != nullptr) LiveCounters::g_vma_report(arg1);
		std::printf("LIVE_VMA id=%" PRIu64 " path=%s written=%d\n", id, arg1, LiveCounters::g_vma_report != nullptr ? 1 : 0);
	} else if (cmd == "images" && n >= 2) {
		// images <path>: the texture cache's images, one tab-separated row each (what the video memory holds).
		if (LiveCounters::g_image_report != nullptr) LiveCounters::g_image_report(arg1);
		std::printf("LIVE_IMAGES id=%" PRIu64 " path=%s written=%d\n", id, arg1, LiveCounters::g_image_report != nullptr ? 1 : 0);
	} else if (cmd == "ranges" && n >= 2) {
		// ranges <path>: the guest's virtual ranges with the names the game gave them (Memory::WriteGuestRanges).
		const bool ok = Libs::LibKernel::Memory::WriteGuestRanges(arg1);
		std::printf("LIVE_RANGES id=%" PRIu64 " path=%s written=%d\n", id, arg1, ok ? 1 : 0);
	} else if (cmd == "buffers" && n >= 2) {
		// buffers <path>: the buffer cache's buffers, one tab-separated row each (what stays registered).
		if (LiveCounters::g_buffer_report != nullptr) LiveCounters::g_buffer_report(arg1);
		std::printf("LIVE_BUFFERS id=%" PRIu64 " path=%s written=%d\n", id, arg1, LiveCounters::g_buffer_report != nullptr ? 1 : 0);
	} else if (cmd == "heapstats" && n >= 2) {
		// heapstats <path>: the process heaps' busy blocks by size (LocalPlatform::WriteHeapHistogram).
		const auto begin = std::chrono::steady_clock::now();
		const bool ok    = LocalPlatform::WriteHeapHistogram(arg1);
		std::printf("LIVE_HEAPSTATS id=%" PRIu64 " path=%s written=%d ms=%.0f\n", id, arg1, ok ? 1 : 0,
		            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
	} else if (cmd == "alloctrace" && n >= 2) {
		// alloctrace start <sizes,...> | dump <path>: live heap blocks of those sizes by stack (LocalPlatform::StartAllocTrace).
		const bool start = std::string_view(arg1) == "start";
		const bool ok    = n >= 3 && (start ? LocalPlatform::StartAllocTrace(arg2) : LocalPlatform::WriteAllocTrace(arg2));
		std::printf("LIVE_ALLOCTRACE id=%" PRIu64 " %s ok=%d\n", id, start ? "start" : "dump", ok ? 1 : 0);
	} else if (cmd == "objects") {
		// objects: what the pipeline cache keeps (pipelines, unoptimized ones replaced, programs) and modules alive.
		if (LiveCounters::g_objects_report != nullptr) LiveCounters::g_objects_report(id);
		std::printf("LIVE_OBJECTS id=%" PRIu64 " shader_modules=%" PRId64 "\n", id, LiveCounters::g_shader_modules.load());
	} else if (cmd == "sync") {
		// sync: GPU timelines, deferred submissions and pending guest readbacks (a stalled GPU thread).
		if (LiveCounters::g_sync_report != nullptr) LiveCounters::g_sync_report();
		std::printf("LIVE_SYNC id=%" PRIu64 "\n", id);
	} else if (cmd == "gamecmd" && n >= 2) {
		// gamecmd <text>: an engine console command on the game's pad-reading thread (loader/demonsSoulsWarp.h), e.g.
		// gamecmd +cp11_unlockAllArchstones, gamecmd +cp11_loadLevel 2 1 (world, level: what archstone travel runs).
		auto text = line.substr(line.find("gamecmd") + 7);
		text.erase(0, text.find_first_not_of(' '));
		while (!text.empty() && (text.back() == '\r' || text.back() == ' ')) text.pop_back();
		const bool ok = Loader::DemonsSoulsWarp::QueueCommand(text);
		std::printf("LIVE_GAMECMD id=%" PRIu64 " queued=%d text=%s\n", id, ok ? 1 : 0, text.c_str());
	} else if (cmd == "gamekill" && n >= 2) {
		// gamekill <radius m>: the characters within it of the player die (loader/demonsSoulsWarp.h; log "Game kill: ...").
		using Mode    = Loader::DemonsSoulsWarp::KillMode;
		const auto mode = n == 3 && std::string_view(arg2) == "hp"      ? Mode::HpZero
		                  : n == 3 && std::string_view(arg2) == "death" ? Mode::Death
		                                                                : Mode::Suicide;
		const bool ok   = Loader::DemonsSoulsWarp::QueueKill(std::strtof(arg1, nullptr), mode);
		std::printf("LIVE_GAMEKILL id=%" PRIu64 " queued=%d radius=%s\n", id, ok ? 1 : 0, arg1);
	} else if (cmd == "gamestrike" && n >= 2) {
		// gamestrike <radius m>: the characters within it killed by the player's own (deadly) hits, nearest first; blocks
		// until done (loader/demonsSoulsWarp.h).
		const auto report = Loader::DemonsSoulsWarp::Strike(std::strtof(arg1, nullptr));
		std::printf("LIVE_GAMESTRIKE id=%" PRIu64 " %s\n", id, report.c_str());
	} else if (cmd == "gamesweep" && n >= 2) {
		// gamesweep <seconds> [radius m]: the nearest living character killed, again and again (loader/demonsSoulsWarp.h).
		const auto report = Loader::DemonsSoulsWarp::Sweep(std::strtod(arg1, nullptr), n == 3 ? std::strtof(arg2, nullptr) : 1000.0f);
		std::printf("LIVE_GAMESWEEP id=%" PRIu64 " %s\n", id, report.c_str());
	} else if (cmd == "warp" && n >= 2) {
		// warp <map> <spawn> | warp off: the debug warp (loader/demonsSoulsWarp.h).
		const bool off = std::string_view(arg1) == "off";
		const bool ok  = off ? (Loader::DemonsSoulsWarp::Disarm(), true)
		                     : n == 3 && Loader::DemonsSoulsWarp::Arm(arg1, arg2);
		std::printf("LIVE_WARP id=%" PRIu64 " map=%s spawn=%s ok=%d spawns=%zu\n", id, arg1, n == 3 ? arg2 : "-",
		            ok ? 1 : 0, Loader::DemonsSoulsWarp::Spawns().size());
	} else if (cmd == "glide") {
		// glide X Y Z SPEED: the player walked there in a straight line (loader/demonsSoulsWarp.h), at once; seconds = the
		// time it takes (-1: no player).
		float      x = 0, y = 0, z = 0, speed = 0;
		const bool parsed  = std::sscanf(line.c_str(), "%*s %f %f %f %f", &x, &y, &z, &speed) == 4;
		const auto seconds = parsed ? Loader::DemonsSoulsWarp::Glide(x, y, z, speed) : -1.0;
		std::printf("LIVE_GLIDE id=%" PRIu64 " seconds=%.2f\n", id, seconds);
	} else if (cmd == "pad") {
		// pad LX LY RX RY BUTTONS SECONDS: what the game reads from the pad for SECONDS (libs/controller.h).
		int          axes[4] {-1, -1, -1, -1};
		unsigned int buttons = 0;
		double       seconds = 0;
		const bool   parsed  = std::sscanf(line.c_str(), "%*s %d %d %d %d %x %lf", &axes[0], &axes[1], &axes[2], &axes[3],
		                                   &buttons, &seconds) == 6;
		if (parsed) Libs::Controller::SetScriptedInput(axes[0], axes[1], axes[2], axes[3], buttons, seconds);
		std::printf("LIVE_PAD id=%" PRIu64 " ok=%d\n", id, parsed ? 1 : 0);
	} else if (cmd == "fatal") {
		// fatal: a fatal error on this thread, the path a crash takes (log, then the process ends).
		EXIT("live fatal: a test of the fatal-error exit\n");
	} else if (cmd == "sleep" && n >= 2) {
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
	} else if (cmd == "skipcs" && n >= 2) {
		SkipShader::Toggle(std::strtoull(arg1, nullptr, 16));
		std::printf("LIVE_SKIPCS id=%" PRIu64 " cs=%s\n", id, arg1);
	} else if (cmd == "capture" && n >= 2) {
		// capture <dir> [frames] [hash,hash,...]: every draw/dispatch of the next frames, with
		// the constants of the listed shaders (frame-capture.h).
		char hashes[1024] {};
		int  frames = 2;
		std::sscanf(line.c_str(), "%*s %*s %d %1023s", &frames, hashes);
		frames        = std::max(1, frames);
		const bool ok = FrameCapture::Arm(arg1, frames, hashes, 120.0);
		std::printf("LIVE_CAPTURE id=%" PRIu64 " frames=%d dir=%s data=%s ok=%d\n", id, frames, arg1,
		            hashes[0] != '\0' ? hashes : "-", ok ? 1 : 0);
	} else {
		std::printf("LIVE_ERROR id=%" PRIu64 " line=%s\n", id, line.c_str());
	}
	std::fflush(stdout);
}

// Called on the render thread before it consumes commands.
#if defined(_WIN32)
// KYTY_STALL_DUMP_MS: the game stopped submitting work for that long in the middle of play (the 60 frames before the
// wait averaged under 50 ms): every thread but the emulator's own (Kyty.*) and SDL's audio threads is sampled, three times 150 ms apart,
// to show what the game waits on. STALL lines: the frames of each thread's stack, guest code as module+offset, host
// code as addresses (symbolize with the build's map). Not with a live `prof` (both prepare the unwind tables).
inline void StallWatch(int64_t threshold_ns) {
	LocalPlatform::SetThreadName("Kyty.StallWatch");
	int64_t stall = 0;
	int     dumps = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		const int64_t since = g_render_idle_since.load(std::memory_order_relaxed);
		if (since == 0) continue;
		const int64_t idle = std::chrono::steady_clock::now().time_since_epoch().count() - since;
		if (since != stall) {
			stall = since;
			dumps = 0;
		}
		if (dumps >= 3 || idle < threshold_ns + dumps * int64_t {150'000'000}) continue;
		const auto flips = g_flips.load(std::memory_order_acquire);
		if (flips < 120) continue;
		const int64_t last  = g_flip_times[(flips - 1) % FlipTimes].load(std::memory_order_relaxed);
		const int64_t first = g_flip_times[(flips - 61) % FlipTimes].load(std::memory_order_relaxed);
		if ((last - first) / 60 > 50'000'000 || since - last > 50'000'000) {
			dumps = 3; // (a loading screen or a pause: not play)
			continue;
		}
		++dumps;
		LocalPlatform::PrepareSampling();
		const auto self = LocalPlatform::ThreadId();
		std::string out;
		char        line[160];
		std::snprintf(line, sizeof(line), "STALL idle=%.0f ms (dump %d), render thread waiting since %.0f ms after the last flip\n",
		              static_cast<double>(idle) / 1e6, dumps, static_cast<double>(since - last) / 1e6);
		out += line;
		for (const auto& [tid, name]: LocalPlatform::ProcessThreads()) {
			if (tid == self || (std::getenv("KYTY_STALL_DUMP_ALL") == nullptr && name.rfind("Kyty.", 0) == 0) || name.rfind("SDLAudio", 0) == 0) continue;
			const auto handle = LocalPlatform::OpenThreadForSampling(tid);
			if (handle == 0) continue;
			uint64_t words[SampleWords] {};
			const bool sampled = LocalPlatform::SampleThread(handle, 0, 0, words, SampleWords);
			LocalPlatform::CloseThreadForSampling(handle);
			if (!sampled) continue;
			std::snprintf(line, sizeof(line), "STALL   %u %s:", tid, name.empty() ? "-" : name.c_str());
			out += line;
			for (const auto word: words) {
				if (word == 0) break;
				const auto guest = Loader::DescribeGuestAddressForDiagnostics(word);
				if (!guest.empty()) out += " " + guest;
				else {
					std::snprintf(line, sizeof(line), " %llx", static_cast<unsigned long long>(word));
					out += line;
				}
			}
			out += "\n";
		}
		std::fputs(out.c_str(), stdout);
		std::fflush(stdout);
	}
}
#endif

inline void Start() {
	g_render_thread = LocalPlatform::CurrentThreadHandle();
	g_render_tid.store(LocalPlatform::ThreadId());
	LiveCensus::g_render = true;
	LiveCounters::g_single_writer = true;
	(void)LocalPlatform::CurrentThreadStack(&g_stack_low, &g_stack_high);
	g_render_known.store(true);
#if defined(_WIN32)
	if (const char* stall = std::getenv("KYTY_STALL_DUMP_MS"); stall != nullptr && std::atoi(stall) > 0)
		std::thread(StallWatch, int64_t {std::atoi(stall)} * 1'000'000).detach();
#endif
	static const char* const path = std::getenv("KYTY_LIVE_FILE");
	if (path == nullptr) return;
	std::thread([] {
		LocalPlatform::SetThreadName("Kyty.Live");
		// Commands already in the file belong to an earlier process: only newer ids run.
		uint64_t last = 0;
		if (std::FILE* file = std::fopen(path, "rb")) {
			unsigned long long id = 0;
			if (std::fscanf(file, "id %llu", &id) == 1) last = id;
			std::fclose(file);
		}
		for (;;) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			std::FILE* file = std::fopen(path, "rb");
			if (file == nullptr) continue;
			std::string text;
			char        buffer[4096];
			for (size_t got; (got = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) text.append(buffer, got);
			std::fclose(file);
			unsigned long long id = 0;
			if (std::sscanf(text.c_str(), "id %llu", &id) != 1 || id == last) continue;
			last         = id;
			size_t begin = text.find('\n');
			while (begin != std::string::npos && begin + 1 < text.size()) {
				const size_t end = text.find('\n', begin + 1);
				Run(id, text.substr(begin + 1, end == std::string::npos ? std::string::npos : end - begin - 1));
				begin = end;
			}
			std::printf("LIVE_DONE id=%llu\n", id);
			std::fflush(stdout);
		}
	}).detach();
}

} // namespace LiveControl
