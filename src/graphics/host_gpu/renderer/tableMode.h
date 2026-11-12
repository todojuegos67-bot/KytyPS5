#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEMODE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEMODE_H_

#include "common/logging/log.h"
#include "graphics/host_gpu/gpuPhases.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

// KYTY_TABLE (gt7-table-1112; default 0): the draw path's table mode, after chenxiao07's table
// draws (experiment/perf-40fps-20260926: 5c5c8cef1 "table draws in place", the perf(table) series
// of 10-11 October, 17542a12f KYTY_TABLE_MEMO). There a draw that repeats a pipeline and resources
// already validated skips the normal draw path; here, on GT7's path, two of its steps are taken
// from tables while what they read is proven unchanged:
//
// - programs (PipelineCache::GetGraphicsPrograms: describe, the programs' lookups, permutations):
//   a table keyed by the draw's registers (the shader registers, the context, the user config,
//   the export mapping), valid while the program cache made no permutation and the shader map
//   registered nothing since, and while the bytes every stage's descriptor walk read hold, checked
//   through the KYTY_MT_PROG GPU-write seal of 36f7cc513 (ReadSet::ValidateFast) as the draw-prep
//   results are. Only a draw whose every stage was served by the draw-prep scanner (so has a read
//   set) is stored, and only when its key comes a second time.
// - bindings (tableBindings.h): per stage slot, the PreparedBindings of the slot's last draw are
//   used again (no PrepareBindings, RebindImages, FindBuffers nor RebindBuffers; the stage's SRT
//   and shader data are uploaded again) when the draw's programs came from the table and the
//   stage repeats the program and the image, sampler and buffer descriptors, with the image and
//   buffer set generations unchanged and every image and buffer still clean (ReuseTextureBinding,
//   BufferCache::ReuseReadBinding). CommitBindings (layout transitions, writes) runs as before.
// - targets (ResolveRenderColorTarget / ResolveRenderDepthTarget): per slot, the description and
//   the image of the target registers seen last, while the texture cache's image set generation
//   holds (as the texture memo: FindImage then finds the same image); the lookup's side effects
//   (touch, uint array description, color clear materialization) still run.
//
// KYTY_TABLE=1 uses the tables. KYTY_TABLE=2 (verifier) looks up the tables, runs the normal path
// too, draws with the normal result and logs every difference ("Table verify:" lines, the first
// 32; counted in the 10 s line). Every 10 s a "Table:" line (GPU thread): hits / misses (and why)
// of each table and the time per draw of the steps they replace, on hits and on misses.
namespace Libs::Graphics::TableMode {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_TABLE");
		if (value == nullptr || value[0] == '\0' || value[0] == '0') {
			return 0;
		}
		return value[0] == '2' ? 2 : 1;
	}();
	return mode;
}

[[nodiscard]] inline bool Enabled() {
	return Mode() != 0;
}

[[nodiscard]] inline bool Verifying() {
	return Mode() == 2;
}

enum class Counter : uint32_t {
	ProgramHits,
	ProgramMisses,
	ProgramMissNew,       // key not in the table (or another key in its slot)
	ProgramMissStale,     // a permutation or a shader registration since it was stored
	ProgramMissReads,     // a stage's recorded bytes changed (or are no longer clean)
	ProgramNotStored,     // a stage walked on the GPU thread (no read set), or vertex tables
	ProgramFetchTables,   // ... of those: a vertex stage that reads vertex tables in describe
	ProgramStored,
	ProgramVerifyChecks,
	ProgramVerifyMismatches,
	TargetHits,
	TargetMisses,
	TargetVerifyChecks,
	TargetVerifyMismatches,
	// Bindings (tableBindings.h), per stage of a draw.
	BindingHits,
	BindingMisses,
	BindingMissProgram,   // the draw's programs did not come from the table
	BindingMissKey,       // another program or resource snapshot than the slot's last
	BindingMissStale,     // image or buffer set generation moved, or an image or buffer refused
	BindingNotStored,     // a stage the table does not keep (storage, written, bindless, DMA...)
	BindingVerifyChecks,
	BindingVerifyMismatches,
	Count,
};

namespace Detail {
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(Counter::Count)> g_counters {};
// Cycles (GpuPhases::Now) of the steps the tables replace, per draw: programs + targets.
inline std::atomic<uint64_t> g_hit_cycles {0};
inline std::atomic<uint64_t> g_hit_draws {0};
inline std::atomic<uint64_t> g_miss_cycles {0};
inline std::atomic<uint64_t> g_miss_draws {0};
inline std::atomic<uint32_t> g_verify_logged {0};
// GPU thread: whether the last GetGraphicsPrograms found its draw in the programs table.
inline thread_local bool t_program_hit = false;
} // namespace Detail

inline void Add(Counter counter, uint64_t value = 1) {
	Detail::g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

// GPU thread: whether the draw's programs came from the table (see AddDraw).
[[nodiscard]] inline bool& ProgramHitFlag() {
	return Detail::t_program_hit;
}

// GPU thread: a draw's programs and targets took `cycles`; `hit`: its programs came from the
// table.
inline void AddDraw(bool hit, uint64_t cycles) {
	if (hit) {
		Detail::g_hit_cycles.fetch_add(cycles, std::memory_order_relaxed);
		Detail::g_hit_draws.fetch_add(1, std::memory_order_relaxed);
	} else {
		Detail::g_miss_cycles.fetch_add(cycles, std::memory_order_relaxed);
		Detail::g_miss_draws.fetch_add(1, std::memory_order_relaxed);
	}
}

// Whether one more "Table verify:" line may be logged.
[[nodiscard]] inline bool VerifyLogAllowed() {
	return Detail::g_verify_logged.fetch_add(1, std::memory_order_relaxed) < 32;
}

// GPU thread, once a draw: the "Table:" line every 10 s.
inline void LogIfDue() {
	struct Reporter {
		uint64_t                              next_check = 0;
		uint64_t                              last_tsc   = 0;
		std::chrono::steady_clock::time_point last_time {};
		std::array<uint64_t, static_cast<size_t>(Counter::Count)> last {};
		uint64_t last_hit_cycles  = 0;
		uint64_t last_hit_draws   = 0;
		uint64_t last_miss_cycles = 0;
		uint64_t last_miss_draws  = 0;
	};
	static Reporter r;
	const auto      tsc = GpuPhases::Now();
	if (r.last_tsc == 0) {
		r.last_tsc   = tsc;
		r.last_time  = std::chrono::steady_clock::now();
		r.next_check = tsc + 50'000'000u;
		LOGF("Table: KYTY_TABLE=%d (%s): programs, render targets and bindings from tables; a \"Table:\" "
		     "line every 10 s\n",
		     Mode(), Verifying() ? "verifier: the normal path draws, differences logged" : "on");
		return;
	}
	if (tsc < r.next_check) {
		return;
	}
	r.next_check   = tsc + 50'000'000u;
	const auto now = std::chrono::steady_clock::now();
	if (now - r.last_time < std::chrono::seconds(10)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - r.last_time).count();
	const double cycles_per_us =
	    seconds > 0.0 ? static_cast<double>(tsc - r.last_tsc) / (seconds * 1'000'000.0) : 0.0;
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> d {};
	for (size_t i = 0; i < d.size(); i++) {
		const auto value = Detail::g_counters[i].load(std::memory_order_relaxed);
		d[i]             = value - r.last[i];
		r.last[i]        = value;
	}
	const auto hit_cycles  = Detail::g_hit_cycles.load(std::memory_order_relaxed);
	const auto hit_draws   = Detail::g_hit_draws.load(std::memory_order_relaxed);
	const auto miss_cycles = Detail::g_miss_cycles.load(std::memory_order_relaxed);
	const auto miss_draws  = Detail::g_miss_draws.load(std::memory_order_relaxed);
	const auto dh_cycles   = hit_cycles - r.last_hit_cycles;
	const auto dh_draws    = hit_draws - r.last_hit_draws;
	const auto dm_cycles   = miss_cycles - r.last_miss_cycles;
	const auto dm_draws    = miss_draws - r.last_miss_draws;
	r.last_hit_cycles      = hit_cycles;
	r.last_hit_draws       = hit_draws;
	r.last_miss_cycles     = miss_cycles;
	r.last_miss_draws      = miss_draws;
	const auto us_per = [&](uint64_t cycles, uint64_t draws) {
		return draws != 0 && cycles_per_us > 0.0
		           ? static_cast<double>(cycles) / cycles_per_us / static_cast<double>(draws)
		           : 0.0;
	};
	const auto c = [&](Counter counter) {
		return static_cast<unsigned long long>(d[static_cast<size_t>(counter)]);
	};
	const auto all_draws = dh_draws + dm_draws;
	const auto all_us    = us_per(dh_cycles + dm_cycles, all_draws);
	LOGF("Table (%.0f s): hits %llu / misses %llu / %.2f us per draw (programs+targets; hit "
	     "%.2f, miss %.2f) | programs: misses new %llu, stale %llu, reads %llu, not stored %llu "
	     "(vertex tables %llu), stored %llu | targets: hits %llu / misses %llu | verify: programs "
	     "%llu checks, %llu mismatches; targets %llu checks, %llu mismatches | bindings: hits %llu / "
	     "misses %llu (program %llu, key %llu, stale %llu, not stored %llu), verify %llu checks, "
	     "%llu mismatches\n",
	     seconds, c(Counter::ProgramHits), c(Counter::ProgramMisses), all_us,
	     us_per(dh_cycles, dh_draws), us_per(dm_cycles, dm_draws), c(Counter::ProgramMissNew),
	     c(Counter::ProgramMissStale), c(Counter::ProgramMissReads), c(Counter::ProgramNotStored),
	     c(Counter::ProgramFetchTables), c(Counter::ProgramStored), c(Counter::TargetHits),
	     c(Counter::TargetMisses), c(Counter::ProgramVerifyChecks),
	     c(Counter::ProgramVerifyMismatches), c(Counter::TargetVerifyChecks),
	     c(Counter::TargetVerifyMismatches), c(Counter::BindingHits), c(Counter::BindingMisses),
	     c(Counter::BindingMissProgram), c(Counter::BindingMissKey), c(Counter::BindingMissStale),
	     c(Counter::BindingNotStored), c(Counter::BindingVerifyChecks),
	     c(Counter::BindingVerifyMismatches));
	r.last_tsc  = tsc;
	r.last_time = now;
}

} // namespace Libs::Graphics::TableMode

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEMODE_H_
