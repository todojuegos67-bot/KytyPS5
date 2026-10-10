#pragma once
// Runtime switches of the performance paths. Every path keeps the original
// behaviour unless its environment variable is set; the GPU thread applies them
// once, before it consumes commands. Values outside a switch's range abort.

#include "native-buffer-residency.h"
#include "local-platform.h"
#include "native-resource-state.h"

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
extern volatile std::atomic_uint32_t kyty_local_stream_upload_mode;
extern volatile std::atomic_uint32_t kyty_local_binding_scratch_mode;
extern volatile std::atomic_uint32_t kyty_local_draw_run_ranges_mode;
extern volatile std::atomic_uint32_t kyty_local_image_barrier_dedupe;
extern volatile std::atomic_uint32_t kyty_local_pending_drain_mode;
extern volatile std::atomic_uint32_t kyty_local_dispatch_batch;
extern volatile std::atomic_uint32_t kyty_local_draw_batch;
extern volatile std::atomic<uint32_t> kyty_local_upload_prologue;
extern volatile std::atomic_uint32_t kyty_local_async_lod_stats_mode;
extern volatile std::atomic<uint32_t> kyty_local_buffer_reclaim_mode;
extern volatile std::atomic<uint32_t> kyty_local_unmap_protect_skip_mode;
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
extern volatile std::atomic_uint32_t kyty_local_native_xpr_diag;
extern volatile std::atomic_uint32_t kyty_local_table_xpr_mode;
extern volatile std::atomic_uint32_t kyty_local_table_dispatch_mode;
extern volatile std::atomic_uint32_t kyty_local_table_indirect_mode;
extern volatile std::atomic_uint32_t kyty_local_table_store_budget;
#endif
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
	    Switch {"KYTY_SRT_NATIVE", &kyty_local_srt_native_mode},
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
	    Switch {"KYTY_STREAM_UPLOAD", &kyty_local_stream_upload_mode},
	    Switch {"KYTY_FRAME_PIPELINE", &kyty_local_frame_pipeline_mode},
	    Switch {"KYTY_WRITE_WINDOW_HANDOFF", &kyty_local_write_window_handoff_mode},
	    Switch {"KYTY_READBACK_DETACH", &kyty_local_readback_detach_mode},
	    Switch {"KYTY_ASYNC_WRITE_READBACK", &kyty_local_async_write_readback_mode},
	    Switch {"KYTY_READBACK_SLOTS", &kyty_local_readback_slots_mode},
	    // 1: buffer upload copies on the upload worker; 2: image staging copies as well.
	    Switch {"KYTY_ASYNC_UPLOAD", &kyty_local_async_upload_mode, 0, 2},
	    Switch {"KYTY_BUFFER_RECLAIM", &kyty_local_buffer_reclaim_mode},
	    Switch {"KYTY_UNMAP_PROTECT_SKIP", &kyty_local_unmap_protect_skip_mode},
	    Switch {"KYTY_BDA_DIRTY_REGIONS", &kyty_local_bda_dirty_regions_mode},
	    Switch {"KYTY_GLOBAL_BARRIER_DEDUPE", &kyty_local_global_barrier_dedupe},
	    Switch {"KYTY_RANGE_SET_FAST", &kyty_local_range_set_fast_mode},
	    Switch {"KYTY_IMAGE_GRANULES", &kyty_local_image_granules_mode, 0, 2},
	    Switch {"KYTY_TEXTURE_RESOLVE_PAGES", &kyty_local_texture_resolve_pages_mode},
	    Switch {"KYTY_PARTIAL_IMAGE_DIRTY", &kyty_local_partial_image_dirty_mode, 0, 2},
	    Switch {"KYTY_PARTIAL_ROW_BANDS", &kyty_local_partial_row_bands_mode},
	    // Sampled views refresh only their levels of a streamed texture (default on; 0 for an A/B).
	    Switch {"KYTY_TEXTURE_LEVELS", &kyty_local_texture_levels_mode},
	    Switch {"KYTY_ASYNC_REPROTECT", &kyty_local_async_reprotect_mode},
	    Switch {"KYTY_READBACK_NARROW", &kyty_local_readback_narrow_mode},
	    // Also creates the transfer queue at device creation (vulkanWindow.cpp).
	    Switch {"KYTY_READBACK_QUEUE", &kyty_local_readback_queue_mode, 0, 3},
	    // Images and command submission.
	    Switch {"KYTY_IMAGE_BARRIER_DEDUPE", &kyty_local_image_barrier_dedupe},
	    Switch {"KYTY_IMAGE_POOL", &kyty_local_image_pool_mode},
	    Switch {"KYTY_PENDING_DRAIN", &kyty_local_pending_drain_mode},
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
	    // Diagnosis bits (native-xpr.inc, kyty_local_native_xpr_diag).
	    Switch {"KYTY_NATIVE_XPR_DIAG", &kyty_local_native_xpr_diag, 0, 31},
	    // Table draws (src/local/table-xpr.inc), with native XPR draws; 2 also continues clean runs.
	    Switch {"KYTY_TABLE_XPR", &kyty_local_table_xpr_mode, 0, 2},
	    // Table dispatches (src/local/table-xpr.inc).
	    Switch {"KYTY_TABLE_DISPATCH", &kyty_local_table_dispatch_mode},
	    // Table dispatches of programs with an address probe's indirect image or reads by device address at offsets no
	    // slot fixes (the light loops; default on, 0 for an A/B).
	    Switch {"KYTY_TABLE_INDIRECT", &kyty_local_table_indirect_mode},
	    // Table store requests per frame (0: no limit).
	    Switch {"KYTY_TABLE_STORE_BUDGET", &kyty_local_table_store_budget, 0, 65536},
#endif
	    Switch {"KYTY_SPECULATE", &kyty_local_speculate_mode, 0, 4},
	    Switch {"KYTY_SPECULATE_THREADS", &kyty_local_speculate_threads, 1, 8},
	};
	std::string enabled;
	for (const auto& setting: switches) {
		const auto* text = std::getenv(setting.environment);
		if (text == nullptr) {
			continue;
		}
		char*      end   = nullptr;
		const auto value = std::strtoul(text, &end, 10);
		if (*text == '\0' || *end != '\0' || value < setting.minimum || value > setting.maximum) {
			std::fprintf(stderr, "%s must be a value from %u to %u\n", setting.environment,
			             setting.minimum, setting.maximum);
			std::abort();
		}
		setting.value->store(static_cast<uint32_t>(value), std::memory_order_relaxed);
		enabled += std::string(enabled.empty() ? "" : " ") + setting.environment + "=" + text;
	}
	// External tools find the render thread by this name (for example to pin it).
	LocalPlatform::SetThreadName("Kyty.Gpu");
	// KYTY_RENDER_CPUS (e.g. "1,2,3,6,7"): the CPUs this thread may run on; other threads are
	// not kept off them. On Windows the P-cores without CPU 0 (which takes most interrupts)
	// measured best; a dedicated core made the whole process slower there.
	if (const char* cpus = std::getenv("KYTY_RENDER_CPUS"); cpus != nullptr && *cpus != '\0') {
		LocalPlatform::PinThreadToCpuList(cpus);
		enabled += std::string(enabled.empty() ? "" : " ") + "KYTY_RENDER_CPUS=" + cpus;
	}
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
