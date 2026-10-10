#pragma once
// Switches of the native resource preparation paths; performance-switches.h
// sets them from the environment. All default to 0, the original behaviour.
#include <atomic>
#include <cstdint>

extern "C" {
// 1: SRT outputs come from the ahead-of-time compiled linear plan.
extern volatile std::atomic<uint32_t> kyty_local_srt_native_mode;
// 1: a plan with one control-flow condition selects its variant through the
// compiled predicate instead of the interpreter.
extern volatile std::atomic<uint32_t> kyty_local_srt_predicate_mode;
// 1: reusable preparation storage; contents are always fresh.
extern volatile std::atomic<uint32_t> kyty_local_preparation_scratch_mode;
// 1: bounded lookup shortcuts (the first image page, single-mapping backing copies).
extern volatile std::atomic<uint32_t> kyty_local_preparation_lookup_mode;
// 1: remove redundant attachment clearing and the temporary snapshot remap index.
extern volatile std::atomic<uint32_t> kyty_local_preparation_trim_mode;
// 1: semantic shape guard for resource specialization, with reusable miss
// storage and the guarded compiled-permutation shortcut.
extern volatile std::atomic<uint32_t> kyty_local_specialization_guard_mode;
// 1: graphics pipelines through a block-hashed index of the full key.
extern volatile std::atomic<uint32_t> kyty_local_pipeline_index_mode;
// 1: exact copy readbacks served from completed GPU mirrors.
extern volatile std::atomic<uint32_t> kyty_local_copy_feedback_mode;
// 1: a CPU write-window fault hands the image lock over once it holds the region lock,
// so the page protection change runs outside the image lock.
extern volatile std::atomic<uint32_t> kyty_local_write_window_handoff_mode;
// 1: a GPU write over a guest readback whose copy has not started detaches it
// instead of waiting for the reader (the reader faults again for newer bytes).
extern volatile std::atomic<uint32_t> kyty_local_readback_detach_mode;
// 1: a guest write to GPU-owned memory takes the asynchronous readback (the writer
// waits for the copy) instead of a GPU-thread download that drains the GPU.
extern volatile std::atomic<uint32_t> kyty_local_async_write_readback_mode;
// 1: 32 guest readback slots instead of 8, so a burst of readbacks does not make the
// render thread wait for the oldest pending copy.
extern volatile std::atomic<uint32_t> kyty_local_readback_slots_mode;
// 1: buffer upload copies run on a worker; submissions wait for the copies they carry.
extern volatile std::atomic<uint32_t> kyty_local_async_upload_mode;
// 1: a bounded BDA preparation visits only regions whose CPU state may have changed.
extern volatile std::atomic<uint32_t> kyty_local_bda_dirty_regions_mode;
extern volatile std::atomic<uint32_t> kyty_local_global_barrier_dedupe;
// 1: RangeSet fast paths for the GPU-modified range set (rangeSet.h).
extern volatile std::atomic<uint32_t> kyty_local_range_set_fast_mode;
// 1: region image queries first test a 64 KiB granule bitmap (textureCache.cpp);
// 2: image-start queries as well.
extern volatile std::atomic<uint32_t> kyty_local_image_granules_mode;
// 1: cached texture resolutions survive registrations away from their range (textureCache.cpp).
extern volatile std::atomic<uint32_t> kyty_local_texture_resolve_pages_mode;
// 1: CPU writes into large images release and re-upload only the written part (textureCache.cpp);
// 2: also checks every partial upload against guest data hashes of the untouched part.
extern volatile std::atomic<uint32_t> kyty_local_partial_image_dirty_mode;
extern volatile std::atomic<uint32_t> kyty_local_partial_row_bands_mode;
extern volatile std::atomic<uint32_t> kyty_local_texture_levels_mode;
// 1: write protection after uploads runs on the upload worker (memoryTracker.h).
extern volatile std::atomic<uint32_t> kyty_local_async_reprotect_mode;
// 1: guest readbacks narrow a window that meets an image to the request's pages (bufferCache.cpp).
extern volatile std::atomic<uint32_t> kyty_local_readback_narrow_mode;
extern volatile std::atomic<uint32_t> kyty_local_readback_queue_mode;
}
