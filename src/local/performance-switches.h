#pragma once
// Runtime switches of the performance paths. Their defaults are the release settings (the values the variables
// define them with); an environment variable sets one for an A/B or a workaround (0 turns a path off). The GPU
// thread applies them once, before it consumes commands. Values outside a switch's range abort.

#include "native-buffer-residency.h"
#include "local-platform.h"
#include "mode-toggle.h"
#include "native-resource-state.h"
#include "speculation-ram-gate.h"
#include "spin-gate.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_frame_pipeline_mode;
extern volatile std::atomic_uint32_t kyty_local_image_pool_mode;
extern volatile std::atomic_uint32_t kyty_local_backing_read_mode;
extern volatile std::atomic<uint32_t> kyty_local_move_compare;
extern volatile std::atomic_uint32_t kyty_local_stream_upload_mode;
extern volatile std::atomic_uint32_t kyty_local_binding_scratch_mode;
extern volatile std::atomic_uint32_t kyty_local_draw_run_ranges_mode;
extern volatile std::atomic_uint32_t kyty_local_image_barrier_dedupe;
extern volatile std::atomic_uint32_t kyty_local_draw_read_after_write;
extern volatile std::atomic_uint32_t kyty_local_pending_drain_mode;
extern volatile std::atomic_uint32_t kyty_local_write_guard_mode;
extern volatile std::atomic_uint32_t kyty_local_flush_defer;
extern volatile std::atomic_uint32_t kyty_local_framegen_sleep;
extern volatile std::atomic_uint32_t kyty_local_dispatch_batch;
extern volatile std::atomic_uint32_t kyty_local_draw_batch;
extern volatile std::atomic<uint32_t> kyty_local_upload_prologue;
extern volatile std::atomic_uint32_t kyty_local_async_lod_stats_mode;
extern volatile std::atomic<uint32_t> kyty_local_buffer_reclaim_mode;
extern volatile std::atomic<uint32_t> kyty_local_unmap_protect_skip_mode;
extern volatile std::atomic<uint32_t> kyty_local_pm4_regfile;
extern volatile std::atomic<uint32_t> kyty_local_upload_lanes;
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
extern volatile std::atomic_uint32_t kyty_local_vulkan_recording_mode;
extern volatile std::atomic_uint32_t kyty_local_deferred_submit_mode;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_mode;
extern volatile std::atomic_uint32_t kyty_local_draw_packets_mode;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_predict_mode;
extern volatile std::atomic_uint32_t kyty_local_native_image_proof_mode;
extern volatile std::atomic_uint32_t kyty_local_async_xpr_pipelines_mode;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_relocate_mode;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_store_budget;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_keep_frames;
extern volatile std::atomic_uint32_t kyty_local_native_xpr_instance_mode;
extern volatile std::atomic_uint32_t kyty_local_table_xpr_mode;
extern volatile std::atomic_uint32_t kyty_local_table_dispatch_mode;
extern volatile std::atomic_uint32_t kyty_local_table_indirect_mode;
extern volatile std::atomic_uint32_t kyty_local_table_store_budget;
extern volatile std::atomic_uint32_t kyty_local_table_set_proofs;
extern volatile std::atomic_uint32_t kyty_local_table_mru;
extern volatile std::atomic_uint32_t kyty_local_table_target_read;
extern volatile std::atomic<uint32_t> kyty_local_table_gather_memo; // KYTY_TABLE_GATHER_MEMO (local/table-xpr.inc TableGather)
#endif
extern volatile std::atomic_uint32_t kyty_local_scout_mode;
// Speculative translation of graphics command buffers (src/graphics/guest_gpu/speculation.h).
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
extern volatile std::atomic_uint32_t kyty_local_speculate_threads;
}

inline void InitializePerformanceSwitches() {
	struct Switch {
		const char*                     environment;
		volatile std::atomic_uint32_t* value;
		uint32_t                        minimum = 0;
		uint32_t                        maximum = 1;
	};
	const std::array switches {
	    // Shader resource preparation.
	    Switch {"KYTY_SRT_PREDICATES", &kyty_local_srt_predicate_mode},
	    Switch {"KYTY_PREPARATION_SCRATCH", &kyty_local_preparation_scratch_mode},
	    Switch {"KYTY_PREPARATION_LOOKUP", &kyty_local_preparation_lookup_mode},
	    Switch {"KYTY_PREPARATION_TRIM", &kyty_local_preparation_trim_mode},
	    Switch {"KYTY_SPECIALIZATION_GUARD", &kyty_local_specialization_guard_mode},
	    Switch {"KYTY_PIPELINE_INDEX", &kyty_local_pipeline_index_mode},
	    Switch {"KYTY_BINDING_SCRATCH", &kyty_local_binding_scratch_mode},
	    Switch {"KYTY_DRAW_RUN_RANGES", &kyty_local_draw_run_ranges_mode},
	    // Buffers and guest memory.
	    Switch {"KYTY_BUFFER_RESIDENCY", &kyty_local_buffer_residency_mode},
	    Switch {"KYTY_COPY_FEEDBACK", &kyty_local_copy_feedback_mode},
	    Switch {"KYTY_ASYNC_LOD_STATS", &kyty_local_async_lod_stats_mode},
	    Switch {"KYTY_BACKING_READ", &kyty_local_backing_read_mode},
	    // The game's large memmoves write only the pages whose bytes differ (default on; 0 for an A/B).
	    Switch {"KYTY_GUEST_MOVE_COMPARE", &kyty_local_move_compare},
	    Switch {"KYTY_STREAM_UPLOAD", &kyty_local_stream_upload_mode},
	    Switch {"KYTY_FRAME_PIPELINE", &kyty_local_frame_pipeline_mode},
	    Switch {"KYTY_WRITE_WINDOW_HANDOFF", &kyty_local_write_window_handoff_mode},
	    Switch {"KYTY_READBACK_DETACH", &kyty_local_readback_detach_mode},
	    Switch {"KYTY_ASYNC_WRITE_READBACK", &kyty_local_async_write_readback_mode},
	    Switch {"KYTY_READBACK_SLOTS", &kyty_local_readback_slots_mode},
	    // 1: buffer upload copies on the upload worker; 2: image staging copies as well.
	    Switch {"KYTY_ASYNC_UPLOAD", &kyty_local_async_upload_mode, 0, 2},
	    // The upload and recording workers spin for new work only while this process's CPUs keep three idle
	    // (spin-gate.h); 0 always, 2 never.
	    Switch {"KYTY_SPIN_GATE", &kyty_local_spin_gate, 0, 2},
	    Switch {"KYTY_BUFFER_RECLAIM", &kyty_local_buffer_reclaim_mode},
	    Switch {"KYTY_UNMAP_PROTECT_SKIP", &kyty_local_unmap_protect_skip_mode},
	    // The indirect register tables' plain stores without their calls (pm4Handlers.cpp Pm4Regfile; 2 = verify). Read
	    // once a packet; the descriptors are proven once, at the first packet with it on, whatever the value.
	    Switch {"KYTY_PM4_REGFILE", &kyty_local_pm4_regfile, 0, 2},
	    // The upload worker's write protections over two lanes (async-upload.h; 1 one, 2 two, 3 two with the order checks).
	    // Read once a batch; the second thread, made at the first split, sleeps whatever the value.
	    Switch {"KYTY_UPLOAD_LANES", &kyty_local_upload_lanes, 1, 3},
	    Switch {"KYTY_BDA_DIRTY_REGIONS", &kyty_local_bda_dirty_regions_mode},
	    Switch {"KYTY_GLOBAL_BARRIER_DEDUPE", &kyty_local_global_barrier_dedupe},
	    Switch {"KYTY_RANGE_SET_FAST", &kyty_local_range_set_fast_mode},
	    Switch {"KYTY_IMAGE_GRANULES", &kyty_local_image_granules_mode, 0, 2},
	    Switch {"KYTY_TEXTURE_RESOLVE_PAGES", &kyty_local_texture_resolve_pages_mode},
	    Switch {"KYTY_PARTIAL_IMAGE_DIRTY", &kyty_local_partial_image_dirty_mode},
	    Switch {"KYTY_PARTIAL_ROW_BANDS", &kyty_local_partial_row_bands_mode},
	    // Sampled views refresh only their levels of a streamed texture (default on; 0 for an A/B).
	    Switch {"KYTY_TEXTURE_LEVELS", &kyty_local_texture_levels_mode},
	    Switch {"KYTY_ASYNC_REPROTECT", &kyty_local_async_reprotect_mode},
	    Switch {"KYTY_READBACK_NARROW", &kyty_local_readback_narrow_mode},
	    // Also creates the transfer queue at device creation (vulkanWindow.cpp).
	    Switch {"KYTY_READBACK_QUEUE", &kyty_local_readback_queue_mode},
	    // Default 2: on unless read guards are on (0 for an A/B). 1 also spans with
	    // KYTY_READ_GUARD=1 (not tested).
	    Switch {"KYTY_READBACK_SPAN", &kyty_local_readback_span_mode, 0, 2},
	    // Images and command submission.
	    Switch {"KYTY_IMAGE_BARRIER_DEDUPE", &kyty_local_image_barrier_dedupe},
	    // 1: a draw that reads an image it writes (the deferred decals' G-buffer) waits for the last draw's write: no
	    // 2x2 holes in overlapping decals (issue #315), up to ~15% slower where there are hundreds of them. Default 0.
	    Switch {"KYTY_DRAW_READ_AFTER_WRITE", &kyty_local_draw_read_after_write},
	    Switch {"KYTY_IMAGE_POOL", &kyty_local_image_pool_mode},
	    Switch {"KYTY_PENDING_DRAIN", &kyty_local_pending_drain_mode},
	    // 1: a guest write of the command processor lands at once, hidden from the game until the older upload jobs are
	    // done (local/write-guard.h); 0: it waits for them (per-page protection churn on the mapped views is behind
	    // MEMORY_MANAGEMENT blue screens on small-memory Windows PCs, #5, #50). Unset: on under KYTY_SPECULATE=4 only.
	    Switch {"KYTY_WRITE_GUARD", &kyty_local_write_guard_mode},
	    // Submissions the GPU has still to run for a flush nothing waits on to be deferred (0: never).
	    Switch {"KYTY_FLUSH_DEFER", &kyty_local_flush_defer, 0, 64},
	    // DLSS-G frames' Reflex sleep (0: none).
	    Switch {"KYTY_FRAMEGEN_SLEEP", &kyty_local_framegen_sleep},
	    // Dispatches recorded per submission.
	    Switch {"KYTY_DISPATCH_BATCH", &kyty_local_dispatch_batch, 1, 65536},
	    // Draws recorded per submission (0: no limit).
	    Switch {"KYTY_DRAW_BATCH", &kyty_local_draw_batch, 0, 65536},
	    // Buffer uploads of pages dirty since before the open command buffer in its upload prologue (on by default).
	    Switch {"KYTY_UPLOAD_PROLOGUE", &kyty_local_upload_prologue},
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	    Switch {"KYTY_VULKAN_RECORDING", &kyty_local_vulkan_recording_mode},
	    Switch {"KYTY_DEFERRED_SUBMIT", &kyty_local_deferred_submit_mode},
	    // 1: native XPR draws; 2: verification (the normal path draws and is compared).
	    Switch {"KYTY_NATIVE_XPR", &kyty_local_native_xpr_mode, 0, 2},
	    Switch {"KYTY_DRAW_PACKETS", &kyty_local_draw_packets_mode},
	    Switch {"KYTY_NATIVE_XPR_PREDICT", &kyty_local_native_xpr_predict_mode},
	    Switch {"KYTY_NATIVE_IMAGE_PROOF", &kyty_local_native_image_proof_mode},
	    Switch {"KYTY_ASYNC_XPR_PIPELINES", &kyty_local_async_xpr_pipelines_mode},
	    Switch {"KYTY_NATIVE_XPR_RELOCATE", &kyty_local_native_xpr_relocate_mode},
	    // Native XPR store requests per frame (0: no limit).
	    Switch {"KYTY_NATIVE_XPR_STORE_BUDGET", &kyty_local_native_xpr_store_budget, 0, 65536},
	    // Frames an unused native XPR record stays (at least 2).
	    Switch {"KYTY_NATIVE_XPR_KEEP_FRAMES", &kyty_local_native_xpr_keep_frames, 0, 1000000},
	    Switch {"KYTY_NATIVE_XPR_INSTANCES", &kyty_local_native_xpr_instance_mode},
	    // Table draws (src/local/table-xpr.inc), with native XPR draws; 2 also continues clean runs.
	    Switch {"KYTY_TABLE_XPR", &kyty_local_table_xpr_mode, 0, 2},
	    // The scout thread (graphics/guest_gpu/scout.h; default 1, Steam Deck builds 0): 0 off, 1 tickets used, 2
	    // verifier (tickets compared). It stands down while the spin gate is shut (KYTY_SPIN_GATE).
	    Switch {"KYTY_SCOUT", &kyty_local_scout_mode, 0, 2},
	    // Table dispatches (src/local/table-xpr.inc).
	    Switch {"KYTY_TABLE_DISPATCH", &kyty_local_table_dispatch_mode},
	    // The table path's gather: a V# unchanged at its index since the last draw keeps its resolved entry (no hash;
	    // default on, 0 for an A/B).
	    Switch {"KYTY_TABLE_GATHER_MEMO", &kyty_local_table_gather_memo},
	    // Table dispatches of programs with an address probe's indirect image or reads by device address at offsets no
	    // slot fixes (the light loops; default on, 0 for an A/B).
	    Switch {"KYTY_TABLE_INDIRECT", &kyty_local_table_indirect_mode},
	    // Table store requests per frame (0: no limit).
	    Switch {"KYTY_TABLE_STORE_BUDGET", &kyty_local_table_store_budget, 0, 65536},
	    // Table set image uses checked again only when they changed (default on; 0 checks all, 2 verifies).
	    Switch {"KYTY_TABLE_SET_PROOFS", &kyty_local_table_set_proofs, 0, 2},
	    // Table lookups that ask the last answer first (bits: pair cache size, variant, image set).
	    Switch {"KYTY_TABLE_MRU", &kyty_local_table_mru, 0, 7},
	    // Table image sets that hold a target the shaders sample (on by default under KYTY_SPECULATE=4, set below;
	    // local/native-xpr.inc).
	    Switch {"KYTY_TABLE_TARGET_READ", &kyty_local_table_target_read},
#endif
	    Switch {"KYTY_SPECULATE", &kyty_local_speculate_mode, 0, 4},
	    Switch {"KYTY_SPECULATE_THREADS", &kyty_local_speculate_threads, 1, 12},
	};
	std::string enabled;
	for (const auto& setting: switches) {
		const auto* text = std::getenv(setting.environment);
		if (text == nullptr || *text == '\0') { // (an empty value keeps the default, as run_ds.sh's ${NAME:-default} did)
			continue;
		}
		char*      end   = nullptr;
		const auto value = std::strtoul(text, &end, 10);
		if (*text == '\0' || *end != '\0' || value < setting.minimum || value > setting.maximum) {
			std::fprintf(stderr, "%s must be a value from %u to %u\n", setting.environment,
			             setting.minimum, setting.maximum);
			std::abort();
		}
		// (KYTY_SPECULATE: 0 or 4, the modes measured and kept.)
		if (setting.value == &kyty_local_speculate_mode && value != 0 && value != 4) {
			std::fprintf(stderr, "%s must be 0 or 4\n", setting.environment);
			std::abort();
		}
		setting.value->store(static_cast<uint32_t>(value), std::memory_order_relaxed);
		enabled += std::string(enabled.empty() ? "" : " ") + setting.environment + "=" + text;
	}
	// KYTY_SPECULATE=4 on a PC with less memory than KYTY_SPECULATE_RAM_GB (default 28 GiB; 0: no gate): mode 4 stands
	// down and the run stays in mode 0 (speculation-ram-gate.h; main.cpp's heap choice agrees).
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 && SpeculationRamGate::StandsDown()) {
		kyty_local_speculate_mode.store(0, std::memory_order_relaxed);
		std::printf("Speculation: KYTY_SPECULATE=4 stands down: %.1f GiB of physical memory, under the %u GiB it needs "
		            "(KYTY_SPECULATE_RAM_GB; 0 lifts the gate). This run stays in mode 0.\n",
		            static_cast<double>(LocalPlatform::PhysicalMemory()) / static_cast<double>(1ull << 30u),
		            SpeculationRamGate::ThresholdGib());
		enabled += std::string(enabled.empty() ? "" : " ") + "(KYTY_SPECULATE=4 stood down: memory)";
	}
	// KYTY_SPECULATE=4 on a CPU whose performance cores can't hold its threads: the GPU, recording and upload threads, the
	// workers (KYTY_SPECULATE_THREADS) and the async queue's 2. With 3 performance cores (6 logical CPUs) the default
	// placement put 4 workers on CPUs the render threads used, and mode 4 lost 6 % at Latria; on the efficiency cores it
	// lost half (Chenxiao, 9 Nov). Mode 4 stands down and the run stays in mode 0. KYTY_SPECULATE_CPUS (the workers' CPUs
	// named) lifts the check; the efficiency cores are never taken by default. Not known (Linux): no check.
	if (const char* named = std::getenv("KYTY_SPECULATE_CPUS");
	    kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 && (named == nullptr || *named == '\0')) {
		const uint32_t cpus    = LocalPlatform::PerformanceLogicalCpus();
		const uint32_t workers = kyty_local_speculate_threads.load(std::memory_order_relaxed);
		const uint32_t need    = 3u + workers + 2u;
		if (cpus != 0 && cpus < need) {
			kyty_local_speculate_mode.store(0, std::memory_order_relaxed);
			std::printf("Speculation: KYTY_SPECULATE=4 stands down: %u logical CPUs on the performance cores, under the %u its "
			            "threads need (3 render threads, %u workers, 2 async; KYTY_SPECULATE_CPUS lifts the check). This run "
			            "stays in mode 0.\n",
			            cpus, need, workers);
			enabled += std::string(enabled.empty() ? "" : " ") + "(KYTY_SPECULATE=4 stood down: CPU)";
		}
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	// KYTY_TABLE_TARGET_READ, unset or empty: on under KYTY_SPECULATE=4 (every mode-4 measure ran with it; mode 0 measured
	// neutral, 1 Nov).
	if (const char* text = std::getenv("KYTY_TABLE_TARGET_READ");
	    (text == nullptr || *text == '\0') && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4)
		kyty_local_table_target_read.store(ModeToggle::TableTargetRead4, std::memory_order_relaxed);
#endif
	// External tools find the render thread by this name (for example to pin it).
	LocalPlatform::SetThreadName("Kyty.Gpu");
	// KYTY_RENDER_CPUS (e.g. "1,2,3,6,7"): the CPUs this thread may run on; other threads are
	// not kept off them. On Windows the P-cores without CPU 0 (which takes most interrupts)
	// measured best; a dedicated core made the whole process slower there.
	if (const char* cpus = std::getenv("KYTY_RENDER_CPUS"); cpus != nullptr && *cpus != '\0') {
		LocalPlatform::PinThreadToCpuList(cpus);
		enabled += std::string(enabled.empty() ? "" : " ") + "KYTY_RENDER_CPUS=" + cpus;
	}
	if (const int core = LocalPlatform::KeepRenderCoreAlone(); core >= 0)
		enabled += std::string(enabled.empty() ? "" : " ") + "KYTY_RENDER_CORE=" + std::to_string(core);
	// Above the game's threads (KYTY_RENDER_PRIORITY=0: normal). Walking into a new area, the game's threads
	// streaming it in held the render CPUs for whole time slices: frames of 35-50 ms where the render thread
	// ran 10-14 ms, its samples parked for 20-26 ms at one instruction.
	if (const char* priority = std::getenv("KYTY_RENDER_PRIORITY"); priority == nullptr || std::strcmp(priority, "0") != 0)
		LocalPlatform::MakeCriticalThread();
	else
		enabled += std::string(enabled.empty() ? "" : " ") + "KYTY_RENDER_PRIORITY=0";
	if (!enabled.empty()) {
		std::printf("Performance switches: %s\n", enabled.c_str());
		std::fflush(stdout);
	}
}
