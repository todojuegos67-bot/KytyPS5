#pragma once
// Local diagnostic event counters (printed per frame by the live `measure` command).
// Relaxed increments from any thread.

#include <atomic>
#include <cstdint>
#include <chrono>
#include <cstdlib>

namespace LiveCounters {

enum Id : uint32_t {
	WindowFault,        // guest write faults served by the CPU write window
	WindowPages,        // pages those windows made writable
	WriteFault,         // other guest write faults (buffer/texture invalidation)
	ReadFault,          // guest read faults (GPU-modified pages)
	Reprotect,          // region CPU-tracking re-arms that changed page state
	ReprotectPages,     // pages re-armed
	Unprotect,          // region CPU-tracking releases that changed page state
	UnprotectPages,     // pages released
	ProtectCalls,       // address-space protection changes (mprotect)
	ProtectCallsRender, // ... issued by the render thread
	UploadCopies,       // buffer upload copies
	UploadBytes,        // buffer upload bytes
	SyncDownloads,      // synchronous GPU downloads for guest access
	AsyncReadbacks,     // asynchronous guest readbacks started
	ReadbackDetaches,   // pending guest readbacks detached by a GPU write
	ReadbackEvictions,  // readbacks that waited for the oldest pending slot
	DispatchAfterDispatch, // dispatches whose previous draw/dispatch was a dispatch
	DispatchSameShader,    // ... of the same compute shader
	Pm4Suspends,           // PM4 executions suspended (WAIT_REG_MEM and similar)
	SubmissionRequeues,    // submissions put back blocked
	GuestCommands,         // host commands run for guest threads (SendCommand)
	RenderReadFaults,      // read faults taken by the render thread itself
	SrtWatchedReads,       // SRT word reads on watched (GPU-owned neighbour) pages
	Submissions,           // submissions processed (slices)
	BufferRegistrations,   // buffer cache registrations and unregistrations
	BdaRebuilds,           // BDA region list rebuilds (mapping or registration changed)
	RegionSyncs,           // BDA region requests that re-synchronized their buffers
	RegionSkips,           // ... proven unchanged by their epochs
	BdaFullSyncs,          // unbounded BDA preparations (every region)
	BdaRangeCalls,         // bounded BDA preparations
	BdaRanges,             // ... their ranges
	BdaRangeMiB,           // ... their total size in MiB
	ProtectCallsGfx,       // protection changes by the render thread inside the graphics queue
	SyncReadsGuest,        // synchronous GPU reads for guest accesses (no asynchronous readback)
	SyncReadsRender,       // ... for the render thread's own reads (indirect arguments, SRT words)
	RbRejectSize,          // guest readbacks refused: larger than a window
	RbRejectBacking,       // ... unmapped or aliased backing
	RbRejectImage,         // ... an image overlaps the window
	RbRejectCapacity,      // ... more than the download capacity
	RbQueueDone,           // guest readbacks on the copy engine: no GPU write of the bytes in flight
	RbQueueInflight,       // ... waiting only for their last writer (KYTY_READBACK_QUEUE=2)
	RbQueueVerified,       // copy-engine copies compared with a graphics-queue copy (mode 3)
	RbQueueMismatch,       // ... that differed
	IndirectTables,        // indirect register tables / command buffers synchronized before parsing
	DirectDraws,           // prepared draws recorded directly (not as a recording packet)
	MeshDraws,             // ... of them, mesh-shader draws
	BackingReadBytes,      // bytes of backing reads of 64 KiB and more (texture and buffer initialization)
	BackingReadUs,         // ... their copy time in microseconds (page faults of the backing view)
	XprTries,              // native XPR lookups (NativeXprTry)
	XprHits,               // ... emitted natively
	XprMissKey,            // ... no record for the draw's key
	XprMissState,          // ... no captured draw state / pipeline for the state key
	XprMissTarget,         // ... a target image changed or is also sampled
	XprMissValidate,       // ... the record's words or resources failed validation (record dropped)
	XprStores,             // records stored after a normal-path draw
	Vblanks,               // video-out vblanks signalled (present thread)
	AsyncImageBytes,       // image staging bytes copied by the upload worker (KYTY_ASYNC_UPLOAD=2)
	BackingLockWaits,      // render-thread waits for the guest backing store mutex
	BackingLockWaitUs,     // ... their time in microseconds
	BackingHoldZeroUs,     // backing store mutex held to zero new allocations (any thread), microseconds
	BackingHoldWriteUs,    // ... held for backing writes (GPU readback results)
	BackingHoldMapUs,      // ... held for mapping changes
	AliasRebuilds,         // rebuilds of the guest backing alias intervals (IsUniqueGuestBackingRange)
	AliasRebuildUs,        // ... their time in microseconds
	AliasMaps,             // ... guest mappings summed over the rebuilds
	UnmapFinishes,         // guest unmaps that waited for a download of an image over the range (GpuResourceManager::UnmapMemory)
	UnmapFinishUs,         // ... their wait in microseconds
	PartialDirtyFaults,    // CPU writes into large images recorded as dirty ranges (KYTY_PARTIAL_IMAGE_DIRTY)
	PartialUploads,        // ... refreshes that uploaded only the dirty subresources
	PartialUploadBytes,    // ... guest bytes those refreshes staged
	PartialFallbacks,      // ... refreshes that fell back to the whole image
	PartialUnmaps,         // unmaps inside a large image that kept it (only the unmapped span dirty)
	StaleProtectRepairs,   // write faults on unwatched pages given the guest's protection back
	ReadbackParts,         // GPU-modified ranges a finished guest readback subtracted
	GpuRangeEntries,       // entries of the GPU-modified range set, summed at each finished readback
	TextureUnmaps,         // TextureCache::UnmapMemory calls
	TextureUnmapUs,        // ... their time in microseconds
	TextureUnmapDeletes,   // ... images they deleted
	FullUploads,           // whole-image uploads (InitializeImage)
	FullUploadBytes,       // ... their guest bytes
	AsyncPipelines,        // pipelines handed to the compile workers (KYTY_ASYNC_XPR_PIPELINES)
	XprStorePending,       // native XPR stores skipped while their pipeline variant compiles
	DispatchKeyNew,        // dispatches whose (shader, user data, groups) last frame did not dispatch (KYTY_DISPATCH_KEYS)
	DispatchKeySame,       // ... repeated from last frame with the same materialized resources
	DispatchKeyChanged,    // ... repeated with other resources
	DrawKeyNew,            // KYTY_DISPATCH_KEYS: a DrawIndex object key (native XPR key) not drawn last frame
	DrawKeySame,           // ... drawn last frame too
	XprDirectTries,        // native records tried for a direct indexed draw (DRAW_INDEX_2 / _OFFSET_2)
	XprDirectHits,         // ... emitted from the record
	XprStored,             // native records inserted by a store
	XprRefuseProgram,      // stores refused: stage layout, DMA/GDS or written resources
	XprRefuseReads,        // ... the SRT read log (alignment, span or total size)
	XprRefuseBind,         // ... resource binding for the record's set
	XprRefuseDraw,         // store requests the normal path did not store (mesh, non-indexed, vertex buffers)
	XprRelocated,          // uses of relocatable native records (KYTY_NATIVE_XPR_RELOCATE)
	XprRelocatedStored,    // ... relocated records stored
	DepthOverlaps,         // images recreated between depth and colour use (ResolveDepthOverlap)
	DepthOverlapBytes,     // ... their guest bytes
	XprMissBlocked,        // native record misses (xpr_miss_key) of a pair that never stores (blocked, unstorable)
	XprMissRefused,        // ... of a relocated key whose store was refused lately
	XprMissUnseen,         // ... of a key no earlier frame drew (no store yet)
	XprMissBudget,         // native record store requests past the frame's budget (KYTY_NATIVE_XPR_STORE_BUDGET)
	ReadbackRegions,       // copy regions of the guest readbacks started (their parts' envelopes merged)
	TableDraws,            // draws the table path emitted (src/local/table-xpr.inc)
	TableEvaluations,      // table image sets made (T#/S# words not seen in the last second)
	TableStores,           // normal-path draws the table path asked to store
	TableContinued,        // ... of them drawn as a clean continuation of the previous one (TableContinue)
	TableStoreVariant,     // table store requests for a pair or draw state not compiled yet
	TableStoreTargets,     // ... for a draw state whose targets changed
	TableRefusedSets,      // table draws whose image set is refused (another specialization, a target image)
	TableDispatches,       // dispatches the table path emitted
	TableRegistered,       // table draws and dispatches left to the normal path: a buffer lookup registered a buffer
	RenderPasses,          // render passes begun (CommandBuffer::BeginRendering)
	WriteBackSkips,        // bytes of GPU write-backs not written: their pages were CPU-owned (BufferCache::WriteBackGpuOwned)
	BackingWriteWaits,     // render-thread waits for the backing store's data mutex to write (backing_lock_waits: the others)
	BackingWriteWaitUs,    // ... their time in microseconds
	GcImageDeletes,        // images the texture cache's garbage collection deleted (video memory pressure)
	GcBufferDeletes,       // buffers the buffer cache's garbage collection deleted or retired
	ImageRefills,          // whole-image refreshes that refilled the staging copy of an upload still unsubmitted
	ImageRefillBytes,      // ... their guest bytes (InitializeImage)
	ImageInitUs,           // render thread: TextureCache::InitializeImage (uploads), microseconds
	BufferSyncUs,          // render thread: BufferCache::SynchronizeBuffer past its clean check, microseconds
	DrawUs,                // render thread: RenderExecutor::ExecutePreparedDraw, microseconds
	RegionSyncUs,          // render thread: BufferCache::SynchronizeRegionRequest past its skip check, microseconds
	AsyncDrawWaitUs,       // render thread: draws waiting for a pipeline a worker compiles (KYTY_ASYNC_DRAW_PIPELINES), microseconds
	AsyncDrawSkips,        // ... draws skipped because their pipeline was still compiling
	QueueLockUs,           // render thread: waits for the submission queue's lock (GuestGpu::ThreadRun), microseconds
	TranslateUs,           // render thread: shader translations it waited for (PipelineCache, WaitCompile), microseconds
	Translates,            // ... their count
	ComputePipelineUs,     // render thread: compute pipelines created synchronously, microseconds
	ComputePipelines,      // ... their count
	GraphicsPipelineUs,    // render thread: graphics pipelines created synchronously, microseconds
	GraphicsPipelines,     // ... their count
	Count
};

inline constexpr const char* Names[Count] = {
    "window_faults", "window_pages",     "write_faults",     "read_faults",   "reprotects",
    "reprotect_pages", "unprotects",     "unprotect_pages",  "protect_calls", "protect_calls_render",
    "upload_copies", "upload_bytes",     "sync_downloads",   "async_readbacks", "readback_detaches", "readback_evictions", "dispatch_after_dispatch", "dispatch_same_shader", "pm4_suspends", "submission_requeues", "guest_commands", "render_read_faults", "srt_watched_reads", "submission_slices", "buffer_registrations", "bda_rebuilds", "region_syncs", "region_skips", "bda_full_syncs", "bda_range_calls", "bda_ranges", "bda_range_mib", "protect_calls_gfx", "sync_reads_guest", "sync_reads_render", "rb_reject_size", "rb_reject_backing", "rb_reject_image", "rb_reject_capacity", "rb_queue_done", "rb_queue_inflight", "rb_queue_verified", "rb_queue_mismatch", "indirect_tables", "direct_draws", "mesh_draws", "backing_read_bytes", "backing_read_us", "xpr_tries", "xpr_hits", "xpr_miss_key", "xpr_miss_state", "xpr_miss_target", "xpr_miss_validate", "xpr_stores", "vblanks", "async_image_bytes", "backing_lock_waits", "backing_lock_wait_us", "backing_hold_zero_us", "backing_hold_write_us", "backing_hold_map_us", "alias_rebuilds", "alias_rebuild_us", "alias_maps", "unmap_finishes", "unmap_finish_us", "partial_dirty_faults", "partial_uploads", "partial_upload_bytes", "partial_fallbacks", "partial_unmaps", "stale_protect_repairs", "readback_parts", "gpu_range_entries", "texture_unmaps", "texture_unmap_us", "texture_unmap_deletes", "full_uploads", "full_upload_bytes", "async_pipelines", "xpr_store_pending", "dispatch_key_new", "dispatch_key_same", "dispatch_key_changed", "draw_key_new", "draw_key_same", "xpr_direct_tries", "xpr_direct_hits", "xpr_stored", "xpr_refuse_program", "xpr_refuse_reads", "xpr_refuse_bind", "xpr_refuse_draw", "xpr_relocated", "xpr_relocated_stored", "depth_overlaps", "depth_overlap_bytes", "xpr_miss_blocked", "xpr_miss_refused", "xpr_miss_unseen", "xpr_miss_budget", "readback_regions", "table_draws", "table_evals", "table_stores", "table_continued", "table_store_variant", "table_store_targets", "table_refused_sets", "table_dispatches", "table_registered", "render_passes", "writeback_skips", "backing_write_waits", "backing_write_wait_us", "gc_image_deletes", "gc_buffer_deletes", "image_refills", "image_refill_bytes", "image_init_us", "buffer_sync_us", "draw_us", "region_sync_us", "async_draw_wait_us", "async_draw_skips", "queue_lock_us", "translate_us", "translates", "compute_pipeline_us", "compute_pipelines", "graphics_pipeline_us", "graphics_pipelines"};

inline std::atomic<uint64_t> g_values[Count];
// The render thread's counts: it is their only writer, so an increment needs no locked
// instruction (a locked add also drains the store buffer). Readers add both (Value).
inline std::atomic<uint64_t> g_render_values[Count];
inline thread_local bool     g_single_writer = false; // set on the render thread (live-control.h Start)
// Local diagnostic (KYTY_DISPATCH_KEYS): classify every dispatch against last frame's (renderCompute.cpp).
inline std::atomic_bool g_dispatch_keys_on {std::getenv("KYTY_DISPATCH_KEYS") != nullptr};
// Live `vma <path>`: writes the GPU allocator's detailed statistics (set by vma.cpp).
inline void (*g_vma_report)(const char* path) = nullptr;
// Live "images <path>": the texture cache's images as a tab-separated table (TextureCache::WriteReport).
inline void (*g_image_report)(const char* path) = nullptr;
// Live "sync": the GPU timelines, deferred submissions and pending guest readbacks (BufferCache::PrintSyncState):
// where a stalled GPU thread waits.
inline void (*g_sync_report)() = nullptr;
// Render thread: the last draw (0) or dispatch shader address.
inline thread_local uint64_t g_last_dispatch_shader = 0;

inline void Add(Id id, uint64_t n = 1) {
	if (g_single_writer) {
		auto& value = g_render_values[id];
		value.store(value.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
		return;
	}
	g_values[id].fetch_add(n, std::memory_order_relaxed);
}
// Adds the scope's duration in microseconds to a counter (the SLOW Frame line's phases).
struct ScopedUs {
	Id                                    id;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	explicit ScopedUs(Id id): id(id) {}
	~ScopedUs() {
		Add(id, static_cast<uint64_t>(
		            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
	}
};
[[nodiscard]] inline uint64_t Value(size_t id) {
	return g_values[id].load(std::memory_order_relaxed) + g_render_values[id].load(std::memory_order_relaxed);
}

// Per 1 MiB granule (hashed, tagged): what cycles where.
enum GranuleField : uint32_t { GWindowPages, GReprotectPages, GUploadBytes, GUploadCalls, GGpuWrites, GGpuWriteBytes, GFields };
struct Granule {
	std::atomic<uint64_t> tag {0};
	std::atomic<uint64_t> values[GFields] {};
};
inline std::atomic_bool g_granules_on {false};
inline Granule          g_granules[4096];
inline std::atomic<uint64_t> g_granule_overflow {0};

inline void AddGranule(uint64_t address, GranuleField field, uint64_t n) {
	if (!g_granules_on.load(std::memory_order_relaxed)) return;
	const uint64_t key = (address >> 20u) + 1;
	for (uint64_t probe = 0; probe < 8; ++probe) {
		auto& granule = g_granules[(key * 0x9e3779b97f4a7c15ull >> 52u) + probe & 4095u];
		auto  tag     = granule.tag.load(std::memory_order_relaxed);
		if (tag == 0 && granule.tag.compare_exchange_strong(tag, key)) tag = key;
		if (tag != key) continue;
		granule.values[field].fetch_add(n, std::memory_order_relaxed);
		return;
	}
	g_granule_overflow.fetch_add(1, std::memory_order_relaxed);
}

// PM4 packets per opcode (render thread; relaxed atomics so the live thread can read).
inline std::atomic<uint64_t> g_pm4[256];
inline void AddPm4(uint32_t opcode) {
	// Only the render thread counts: no locked add.
	auto& value = g_pm4[opcode & 0xffu];
	value.store(value.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

} // namespace LiveCounters
