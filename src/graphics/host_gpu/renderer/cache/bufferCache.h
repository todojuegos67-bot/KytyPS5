#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "lane-sync.h"
#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "loader/demonsSoulsWriteAudit.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "local/buffercache-par.h" // (buffercache-par, KYTY_BC_PAR)
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "speculation-state.h"

#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ReadbackQueue { class Queue; }
namespace GpuResidency {
struct Range;
} // namespace GpuResidency

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;
class GpuResourceManager;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache, GpuResourceManager* resources = nullptr);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Before a host write of all of [vaddr, vaddr + size) (any thread): as InvalidateMemory, but the GPU-written bytes the
	// write replaces are dropped, not read back (only those of its first and last page outside it are).
	void                   InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	// Live "sync" (any thread, racy by design: for a stalled GPU thread).
	void                   PrintSyncState();
	// Live "buffers <path>": the cached buffers as a tab-separated table (on the GPU thread).
	void                   WriteBufferReport(const char* path);
	// (Diagnostics, KYTY_GPU_ONLY_OFFLOAD's breakdown; the GPU thread) the cached buffers' bytes: all of them, those with a
	// page the GPU wrote, those the GPU wrote all of, those with a page the CPU wrote; and their count.
	void                   GuestUseBreakdown(uint64_t& total, uint64_t& gpu_any, uint64_t& gpu_full, uint64_t& cpu_any,
	                                         uint64_t& count);
	// KYTY_GPU_RESIDENCY (gpu-residency.h; the GPU thread): the registered buffers' guest ranges appended to `out`, a
	// buffer with a page the GPU wrote as GpuWrittenBuffer; their number in `count`.
	void                   CollectResidencyRanges(std::vector<GpuResidency::Range>& out, uint64_t& count);
	[[nodiscard]] bool TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// (parpadeo7, KYTY_PARPADEO7=1; the GPU thread, between packets: GpuResourceManager::PrefetchGuestReads) The
	// asynchronous guest readback a read fault of [vaddr, vaddr + size) would begin, begun now; a helper thread copies it
	// into guest memory as soon as the GPU is done and this thread retires it then (a command). Never waits: nothing when
	// a pending readback overlaps the range (it is in flight already), when no slot is free (retiring one waits for its
	// copy) or when the readback refuses it. Returns a Parpadeo7::PrefetchResult (local/parpadeo7.h).
	[[nodiscard]] uint32_t PrefetchGuestRead(uint64_t vaddr, uint64_t size, int64_t label_ns);
	// (parpadeo7) The helper stopped (its queued copies made first); later prefetches begin nothing. Before the GPU thread
	// stops (GuestGpu::Shutdown); again at destruction. Any thread but the GPU thread and the helper.
	void StopPrefetchHelper();
	// KYTY_NATIVE_MEMORY=1 (GPU thread; GpuResourceManager::NativeFlushGpuWrites): whether some range holds bytes GPU work
	// wrote that guest memory does not have yet (the GPU-modified ranges), and every such range of a registered buffer
	// downloaded to guest memory now, whatever its size (ReadMemoryOnGpu, as a game read's fault did; the pages then
	// GPU-clean). Returns the ranges downloaded and adds their bytes to `bytes`.
	[[nodiscard]] bool     NativeHasGpuWrites() const noexcept { return !m_gpu_modified_ranges.Empty(); }
	uint64_t               NativeWriteBackAll(uint64_t* bytes);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] Buffer* GetBufferIfLive(BufferId id) {
		auto* buffer = m_slot_buffers.try_get(id);
		if (buffer == nullptr || buffer->is_deleted) return nullptr;
		// KYTY_ASYNC_COMPUTE: the callers reuse a buffer they obtained before (memos, held V#s): a use of it.
		LaneSync::NoteReuse(buffer->lane_marks);
		return buffer;
	}
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// FindBuffer's registered buffer over the range, none registered: the null id (a speculative translation's
	// question whether ObtainBuffer gave the range's buffer).
	[[nodiscard]] BufferId RegisteredBuffer(uint64_t vaddr, uint64_t size) {
		if (BcPar::t_reader) return RegisteredBufferShared(vaddr, size); // (buffercache-par: hilo de traduccion)
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		return owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size) ? *owner : BufferId {};
	}
	[[nodiscard]] BufferId RegisteredBufferShared(uint64_t vaddr, uint64_t size);
	void                   EnsureBufferContents(uint64_t vaddr, uint64_t size);
	// The start of each range shaders read through pointers held by a buffer with the guest's bytes (see the definition).
	void                   EnsureBdaReadable(std::span<const GuestRange> ranges);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// (F5 pase 2, sombras nativas) Solo MALLA-NATIVA (Lookup, como ObtainBuffer): el VkBuffer del bloque de malla que
	// contiene [vaddr, vaddr + size) y el desplazamiento; {nullptr, 0} si no lo hay ahora (nada por el camino normal).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainMallaNativa(uint64_t vaddr, uint64_t size);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return DeviceUtilityBuffer(); // (VRAM-BUF: creado al primer uso)
			case MemoryUsage::SparseDeviceLocal: break;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// CPU-written snapshots consumed as shader storage buffers. Each selected
	// ring keeps its own GPU retirement watches across runtime mode changes.
	[[nodiscard]] StreamBuffer& GetShaderUploadBuffer() noexcept;
	// The per-draw blocks of table draws (src/local/table-xpr.inc), read by their device addresses.
	[[nodiscard]] StreamBuffer& GetTableUploadBuffer() noexcept;
	[[nodiscard]] const Buffer* GetLodStatsBuffer() const noexcept { return &m_lod_stats_buffer; }
	void ReportLodStats(void* dst, uint32_t size, bool reset);
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	// The BDA page table descriptor writes take (the newest: GrowBdaTable). A set made with it records
	// BdaTableGeneration() read before it, and is made again once that moved.
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return m_bda_table.load(std::memory_order_acquire); }
	[[nodiscard]] static uint64_t BdaTableGeneration() noexcept {
		return s_bda_table_generation.load(std::memory_order_acquire);
	}
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	// `refillable`: set when the bytes went to the staging ring by the upload worker (RefillImageStaging may copy them
	// again while the upload is unsubmitted).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size,
	                                                               bool* refillable = nullptr);
	// KYTY_ASYNC_REPROTECT: the write protection of the pages an image starts watching again (TextureCache's
	// DeferredImageProtects sets the page manager's deferred write-protect sink to this list around TrackImage), applied
	// by the upload worker ahead of an upload copy that reads those pages whole (the whole image: PushImageStagingCopies;
	// parts of it: StageImagePieces), else before any other read of them (ApplyImageProtects, and what is left when the
	// texture cache's upload returns): a CPU write before the protection is in the copy, one after it faults.
	[[nodiscard]] std::vector<PageManager::DeferredRange>* ImageProtectSink(PageManager& manager) noexcept {
		m_image_protect_manager = &manager;
		return &m_image_protects;
	}
	void ApplyImageProtects();
	// The guest bytes of [vaddr, vaddr + size) copied again by the upload worker to a staging copy ObtainBufferForImage
	// made (`ring` at `offset`): whose GPU copies are recorded but not submitted, so they upload the new bytes. False,
	// with nothing done, where ObtainBufferForImage would not stage them that way now.
	[[nodiscard]] bool RefillImageStaging(const Buffer& ring, uint64_t offset, uint64_t vaddr, uint64_t size);
	// Guest ranges staged one after another in one staging allocation of `total` bytes, each
	// at its `offset`. Returns nullptr (nothing staged) when a range holds GPU-written pages: the
	// caller then uploads the whole image instead.
	struct StagingPiece {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t offset = 0;
	};
	// (`why`: what refused, when nothing is staged - "gpu" or "map".)
	[[nodiscard]] std::pair<Buffer*, uint64_t> StageImagePieces(const std::vector<StagingPiece>& pieces,
	                                                            uint64_t total, const char** why = nullptr);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// CopyBuffer's copy on the CPU: only the destination pages whose bytes differ are written.
	void CopyGuestMemory(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size);
	// Before a CPU write of [vaddr, vaddr+size) through memory mapped at other addresses too (the aliases): false when
	// an alias holds GPU-written bytes (a download of them would overwrite the write), else every alias is made
	// CPU-dirty (as a write fault through it would), its images invalidated and its pending readbacks finished.
	[[nodiscard]] bool PrepareAliasesForCpuWrite(uint64_t vaddr, uint64_t size);
	// Called after an identified small GPU copy; it never submits or publishes guest data.
	void ScheduleCopyFeedback(uint64_t vaddr, uint64_t size);
	// KYTY_EAGER_FEEDBACK (live eager_feedback, off by default): GPU thread, the copy-feedback
	// snapshots whose tick has completed are written back to guest memory now (TryGuestReadFromFeedback, as a fault's
	// read would), before a guest read faults on them. `may_wait`: between guest packets, where a synchronous download
	// may run (the verify's).
	void WriteBackCompletedFeedback(bool may_wait);
	// (w20) Whether WriteBackCompletedFeedback could do anything (a copy-feedback snapshot or a native write queued): the
	// PM4 walk's per-packet preamble skips the call otherwise. GPU thread, as the queues' other readers.
	[[nodiscard]] bool FeedbackPending() const noexcept { return !m_eager_feedback.empty() || !m_native_writes.empty(); }
	// (recorrido-rapido, KYTY_RECORRIDO_RAPIDO) Whether it would do anything NOW: a queued snapshot or native write whose
	// tick has completed (the oldest of each is the first to complete; both write-backs return at once otherwise).
	[[nodiscard]] bool FeedbackReady() const noexcept;
	// GPU thread: retire a pending guest read before touching overlapping backing.
	// With no range, drain before mapping changes, unknown BDA access, or shutdown.
	// `gpu_write`: the range is about to be written by the GPU again; a read whose
	// copy has not started may be detached instead of awaited (KYTY_READBACK_DETACH).
	void DrainGuestReadback(uint64_t address = 0, uint64_t size = UINT64_MAX,
	                        bool gpu_read_only = false, bool gpu_write = false);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	// After a speculative translation's work was submitted (its commit obtained the ranges it writes as written
	// before it): the writes are that work's, which readbacks of the range wait for.
	void NoteSpeculativeWrite(uint64_t vaddr, uint64_t size) {
		InvalidateCopyFeedback(vaddr, size);
		NoteGpuWrite(vaddr, size);
	}
	// The tracker's answer whatever a speculative translation will upload (what a cache may keep).
	[[nodiscard]] bool IsRegionCpuDirty(uint64_t vaddr, uint64_t size) {
		PolicyBeforeUse(vaddr, size, false);
		return m_memory_tracker.IsRegionCpuModified(vaddr, size);
	}
	// ---- (policy1) KYTY_DS_NEIGHBOR_SPLIT (loader/demonsSoulsWriteAudit.h). One relaxed load each when off. --------
	// The fault handler: the page of a CPU write fault stops being write-protected (MemoryTracker::SetNoWatchPage).
	// (anuncio1) `why`: why not (MemoryTracker::SetNoWatchPage).
	[[nodiscard]] bool PolicySetNoWatch(uint64_t vaddr, int* why = nullptr) { return m_memory_tracker.SetNoWatchPage(vaddr, why); }
	// (anuncio1, KYTY_RAIZ_ANNOUNCE) A refused frame page handed out again since its refusal: refused no more, no-watch now.
	[[nodiscard]] bool PolicyUnblockSetNoWatch(uint64_t vaddr) { return m_memory_tracker.UnblockSetNoWatchPage(vaddr); }
	// GpuResourceManager::SweepCpuWrites (the GPU thread, before a submission's work reads guest memory): a hook of the
	// game announced a write to the range / a range no hook covers appeared there.
	uint64_t PolicyMarkDirty(uint64_t vaddr, uint64_t size) { return m_memory_tracker.MarkNoWatchDirty(vaddr, size); }
	uint64_t PolicyRevert(uint64_t vaddr, uint64_t size) { return m_memory_tracker.ClearNoWatch(vaddr, size, true); }
	uint64_t PolicyMarkAllDirty() { return m_memory_tracker.MarkAllNoWatchDirty(); }
	uint64_t PolicyRevertAll() { return m_memory_tracker.ClearAllNoWatch(); } // (policy4)
	// (policy4) A shader reads [vaddr, vaddr + size) through the BDA page table (a buffer marked bda_read): it may read
	// any byte of it, so its no-watch pages are trusted only when every byte of the range lies in ranges the hooks
	// announce; else they are watched again for good. One relaxed load when the policy is off.
	// (nativo2) KYTY_DS_NATIVE_HEAP=1: no-watch pages GPU work reads outside the hooks' rules stay no-watch as "native"
	// pages (RegionManager::KeepNative) instead of being watched again for good; before every submission their OS write
	// watch is read and reset (PolicyNativeSweep) and the written ones made CPU-dirty. 0: off (one cached load).
	[[nodiscard]] static bool NativeHeapOn() noexcept;
	// (nativo3) KYTY_DS_NATIVE_ALL=1: NativeHeapOn, and the sweep reads the OS write watch of EVERY no-watch page (frame
	// memory too), so a write no hook announced is still seen before the submission's work uses it (exact by
	// construction); the frame policy's marks are then not applied (the sweep covers them). 0: off (one cached load).
	[[nodiscard]] static bool NativeAllOn() noexcept;
	// (HEAP-NATIVO) KYTY_HEAP_NATIVO=1: cpuheap/cpubkt/unlabelled pages GPU work reads outside the hooks' rules are copied
	// at each use (CPU-dirty then) and stay writable for good: no veto, no write fault, no sweep. 0: off (one cached load).
	[[nodiscard]] static bool HeapNativoOn() noexcept;
	uint64_t                  HeapNativoTake(uint64_t vaddr, uint64_t size, bool bda) const;
	void                      HeapNativoReport();
	uint64_t                  PolicyKeepNative(uint64_t vaddr, uint64_t size) { return m_memory_tracker.KeepNative(vaddr, size); }
	// GpuResourceManager::SweepCpuWrites (the GPU thread, before a submission's work reads guest memory).
	void PolicyNativeSweep();
	void PolicyBdaRead(uint64_t vaddr, uint64_t size) const {
		if (::Loader::DemonsSoulsWriteAudit::SplitOn()) PolicyBdaReadSlow(vaddr, size);
	}
	// Before anything trusts the tracker about [vaddr, vaddr + size) (a buffer obtained for a draw, a proof by the CPU
	// epoch or the dirty bits): no-watch pages there are trusted only when every byte asked for lies in ranges whose
	// writes the game's hooks announce; else they are watched again, CPU-dirty, at once (so the answer that follows is
	// "modified"). `is_written`: GPU work will write the range (never left no-watch).
	void PolicyBeforeUse(uint64_t vaddr, uint64_t size, bool is_written) const {
		if (::Loader::DemonsSoulsWriteAudit::SplitOn()) PolicyBeforeUseSlow(vaddr, size, is_written);
	}
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	struct SyncRegionRequest {
		uint64_t address = 0, size = 0, cpu_epoch = 0, registration_epoch = 0;
	};
	void SynchronizeRegionRequest(SyncRegionRequest& request);
	[[nodiscard]] uint64_t RegistrationEpoch() const { return m_registration_epoch.load(std::memory_order_acquire); }
	// Moves whenever something ObtainBuffer of a written range does or looks at could differ from its last call over
	// that range within one tick: a guest readback queued or finished (DrainGuestReadback), a GPU-modified range
	// subtracted (its pages given back, a download, a discard), a copy feedback snapshot recorded
	// (InvalidateCopyFeedback), a CPU write window invalidated. With it, the tick, the registration epoch and the
	// range's CPU modification epoch unchanged, a written V# resolved once in the tick resolves to the same buffer and
	// needs no second ObtainBuffer (TableBlocks' memo).
	[[nodiscard]] uint64_t WrittenObtainEpoch() const { return m_written_obtain_epoch.load(std::memory_order_acquire); }
	void                   MoveWrittenObtainEpoch() { m_written_obtain_epoch.fetch_add(1, std::memory_order_acq_rel); }
	// (trad1, KYTY_TRAD1; local/trad1.h) Moves whenever m_gpu_modified_ranges loses bytes (m_gpu_coverage_epoch). GPU thread.
	[[nodiscard]] uint64_t GpuCoverageEpoch() const { return m_gpu_coverage_epoch; }
	// (trad1b) Whether m_gpu_modified_ranges may have lost bytes of [vaddr, vaddr + size) since `epoch` (a value of
	// GpuCoverageEpoch): the loss stamps of its 4 MiB tracker regions (Trad1StampCoverage), kept once a caller asked
	// (Trad1EnableCoverage; every region counts as lost before that). Conservative: a loss elsewhere in a region counts.
	[[nodiscard]] bool Trad1CoverageMoved(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		if (m_trad1_coverage == nullptr || m_trad1_coverage_far > epoch || size == 0) return true;
		const auto first = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		if (last >= Trad1CoverageRegions || last - first > 64) return true;
		for (auto region = first; region <= last; ++region)
			if (m_trad1_coverage[region] > epoch) return true;
		return false;
	}
	void Trad1EnableCoverage() {
		if (m_trad1_coverage != nullptr) return;
		m_trad1_coverage      = std::make_unique<uint64_t[]>(Trad1CoverageRegions);
		m_trad1_coverage_far  = m_gpu_coverage_epoch; // (losses before now were not stamped)
	}
	// (trad1) A range ObtainBuffer obtained as written into buffer `id` (`expect`), at registration epoch `registration`
	// and coverage epoch `coverage`, written again: what ObtainBuffer's write memo path does (the policy, the readback
	// drain, the LRU touch, the copy feedback invalidated, the GPU and native write notes) when, after the drain, those
	// epochs still stand and the buffer is live and holds the range (its pages still GPU-owned and the range still in the
	// GPU-modified set: SynchronizeBuffer and the Add would find nothing to do). False: nothing noted (the drain may have
	// run); the caller obtains it as before. Not texel, not speculative.
	[[nodiscard]] bool Trad1RenoteWritten(BufferId id, const Buffer* expect, uint64_t vaddr, uint64_t size,
	                                      uint64_t registration, uint64_t coverage);
	// (trad1, KYTY_TRAD1=2) Whether SynchronizeBuffer of the range as written would upload anything (some page of it not
	// GPU-owned), as ObtainBuffer would find it now.
	[[nodiscard]] bool Trad1SyncWouldUpload(uint64_t vaddr, uint64_t size);
	// (trad1-prueba, KYTY_TRAD1_PRUEBA) La prueba exacta de cobertura del rango escrito: todas sus paginas GPU-owned y el
	// rango entero en m_gpu_modified_ranges (SynchronizeBuffer no subiria nada, el Add no anadiria nada).
	[[nodiscard]] bool Trad1CoverageExact(uint64_t vaddr, uint64_t size) {
		return size != 0 && m_gpu_modified_ranges.Contains(vaddr, size) && m_memory_tracker.IsRegionFullyGpuModified(vaddr, size);
	}
	// Moves whenever a buffer is unregistered (deleted, or joined into a new one): Vulkan buffers a binding resolved
	// before it is unchanged are still the owners of their bytes. A join copies the old buffer's bytes into the new
	// one in the open command buffer, so a binding to the old one GPU work records after it reads or writes bytes
	// nothing reads again (the old buffer lives until the tick). The render thread's.
	[[nodiscard]] uint64_t UnregisterEpoch() const { return m_unregister_epoch; }
	// A buffer registered or unregistered in the tracker regions of the range since `epoch`.
	[[nodiscard]] bool RegisteredSince(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		for (auto region = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE; region <= last; ++region)
			if (RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) > epoch) return true;
		return false;
	}
	// (tej-cos) A buffer registered or unregistered over the EXACT bytes [vaddr, vaddr+size) since `epoch` (the region test
	// above at byte granularity: a lookup of the range finds the same buffers while no buffer covering any of its bytes
	// came or went; a join unregisters the old buffer over its whole span, so it is seen). From a ring of the last
	// RegLogN changes, each tagged by its epoch (written last); `known` false: the ring no longer reaches back (or an entry
	// was being rewritten): the caller must fall back to RegisteredSince.
	// `ranges`: [begin, end) sorted by begin, not overlapping (any of them).
	[[nodiscard]] bool RegisteredOverlapSince(const std::pair<uint64_t, uint64_t>* ranges, size_t n, uint64_t epoch,
	                                          bool& known) const {
		const uint64_t now = m_registration_epoch.load(std::memory_order_acquire);
		known              = now <= epoch || now - epoch <= RegLogN;
		if (!known || now <= epoch || n == 0) return !known;
		for (uint64_t e = epoch + 1; e <= now; ++e) {
			const auto& slot = m_reg_log[e % RegLogN];
			if (slot.epoch.load(std::memory_order_acquire) != e) return known = false, true;
			const uint64_t b = slot.begin.load(std::memory_order_relaxed), en = slot.end.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (slot.epoch.load(std::memory_order_relaxed) != e) return known = false, true;
			// (the first range ending after b: overlaps iff it begins before en)
			size_t lo = 0, hi = n;
			while (lo < hi) {
				const size_t mid = (lo + hi) / 2;
				if (ranges[mid].second <= b) lo = mid + 1;
				else hi = mid;
			}
			if (lo < n && ranges[lo].first < en) return true;
		}
		return false;
	}
	[[nodiscard]] bool RegisteredOverlapSince(uint64_t vaddr, uint64_t size, uint64_t epoch, bool& known) const {
		const std::pair<uint64_t, uint64_t> one {vaddr, vaddr + size};
		return RegisteredOverlapSince(&one, 1, epoch, known);
	}
	// The registration epoch of the last change of a buffer covering the tracker region of `vaddr`: a region
	// request's proof only depends on the buffers of its own region.
	[[nodiscard]] uint64_t RegionRegistrationEpoch(uint64_t vaddr) const {
		const auto region = vaddr / TRACKER_REGION_SIZE;
		return region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE ? m_region_registrations[region].load(std::memory_order_acquire)
		                                                           : 0;
	}
	// Proofs that an earlier resolution still holds. Pure queries; TouchLiveBuffer
	// only refreshes the LRU slot exactly as ObtainBuffer's TouchBuffer would.
	// Zero means "cannot prove anything" (see MemoryTracker::CpuModificationEpoch).
	// A range spanning several 4 MiB tracker regions returns the sum of their
	// epochs: region managers are never destroyed and each epoch only grows, so
	// an unchanged sum proves that no region saw a new CPU-dirty mark.
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const {
		return m_memory_tracker.DeferredProtectionPending(vaddr);
	}
	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const {
		if (!GuestRange {vaddr, size}.Valid()) return 0;
		PolicyBeforeUse(vaddr, size, false); // (policy1)
		constexpr uint64_t MaxRegions = 64;
		const uint64_t     first      = vaddr / TRACKER_REGION_SIZE;
		const uint64_t     last       = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		if (last - first >= MaxRegions) return 0;
		uint64_t sum = 0;
		for (uint64_t region = first; region <= last; ++region) {
			const uint64_t begin = std::max(vaddr, region * TRACKER_REGION_SIZE);
			const uint64_t end   = std::min(vaddr + size, (region + 1) * TRACKER_REGION_SIZE);
			const uint64_t epoch = m_memory_tracker.CpuModificationEpoch(begin, end - begin);
			if (epoch == 0) return 0;
			sum += epoch;
		}
		return sum;
	}
	// A speculative translation's use of the buffer, when it is committed.
	void TouchSpeculated(BufferId id) {
		if (auto* buffer = m_slot_buffers.try_get(id)) TouchBuffer(*buffer);
	}
	// True when `id` names a live (registered, not deleted) game buffer that
	// covers the range and whose Vulkan handle is `handle`; refreshes its LRU slot.
	[[nodiscard]] bool TouchLiveBuffer(BufferId id, vk::Buffer handle, uint64_t vaddr,
	                                   uint64_t size) {
		auto* buffer = m_slot_buffers.try_get(id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size) ||
		    buffer->Handle() != handle) {
			return false;
		}
		TouchBuffer(*buffer);
		return true;
	}
	// The buffers no binding used for a while go (RunGarbageCollector, once a flip).
	void RetireStaleBuffers(uint64_t tick);
	// The buffers [vaddr, vaddr + size) holds whole go, after the guest unmapped it (GpuResourceManager::UnmapMemory).
	void DeleteUnmappedBuffers(uint64_t vaddr, uint64_t size);
	// Then a buffer the range holds in part, mapped in no more than half of it now, is cut to its mapped parts.
	void SplitUnmappedBuffers(uint64_t vaddr, uint64_t size);
	// The mapped parts of registered buffers in [begin, end), adjacent ones merged.
	void CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
	                                   std::vector<RangeSet::Range>& ranges) const;
	// The spans of the buffers registered or unregistered since the last call, appended to `spans`; false when
	// more changed than it keeps (then anything may have).
	[[nodiscard]] bool TakeRegistrationSpans(std::vector<GuestRange>& spans);

	void               RunGarbageCollector(bool collect = true);
	void               UpdateGcThresholds();
	[[nodiscard]] uint64_t CacheBytes() const noexcept { return m_cache_bytes; }
	// The video memory the game buffers in use that got system memory need back, with headroom (0: none): the texture
	// collector makes this room too (TextureCache::RunGarbageCollector). Updated every 60 flips.
	[[nodiscard]] uint64_t SpilledRoom() const noexcept { return m_spilled_room; }

private:
	void PolicyBeforeUseSlow(uint64_t vaddr, uint64_t size, bool is_written) const; // (policy1)
	void PolicyBdaReadSlow(uint64_t vaddr, uint64_t size) const;                    // (policy4)
	void PolicyCheckAnnounced(uint64_t vaddr, uint64_t size, bool wide) const;      // (anuncio1)
	void PolicyGpuWrite(uint64_t vaddr, uint64_t size) const;                       // (anuncio2)
	void PolicyCheckStale(uint64_t vaddr, uint64_t size, bool wide) const;          // (policy6; policy7: wide)
	friend struct BufferCacheTestAccess;

	// A game buffer in use made in system memory (Buffer::IsInSystemMemory: video memory over budget then) goes back to
	// video memory once there is room for it, one a call (RunGarbageCollector, every 60 flips). True when one moved.
	bool MoveSpilledBufferBack();
	// (BUF-RAM, local/vram-tope.h nivel 2) Bufferes frios de VRAM a memoria del sistema y vuelta si se usan. Cada flip.
	void BufRamPass(uint64_t tick);
	void BufMemCensus(); // (BUF-MEM) linea [buf-mem]
	// The game buffers' pool compacted (GraphicContext::game_buffer_pool): a pass of moves at a time, every flip while
	// one runs (RunGarbageCollector).
	void Defragment(uint64_t tick);
	void EndDefragmentation();
	struct DefragMove {
		BufferId source, destination;
	};
	VmaDefragmentationContext      m_defrag = nullptr;
	VmaDefragmentationPassMoveInfo m_defrag_pass {};
	std::vector<DefragMove>        m_defrag_moves; // (a pass's, by its moves' order; none: ignored)
	bool                           m_defrag_pass_open = false;
	std::future<VkResult>          m_defrag_ending;     // (the pass ending on a worker thread)
	std::vector<VmaAllocation>     m_defrag_free_later; // (moved allocations of new buffers destroyed meanwhile)
	uint64_t                       m_defrag_next_check = 0;
	uint64_t                       m_defrag_free_after = 0; // the pool's free bytes after the last compaction
	uint32_t                       m_defrag_passes     = 0;
	uint32_t                       m_defrag_ignored    = 0; // (moves of allocations no buffer holds: Defragment)
	double                         m_defrag_ms = 0, m_defrag_max_ms = 0; // the GPU thread's time in its passes

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	struct DownloadCopy;
	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	static constexpr uint64_t               DOWNLOAD_ALIGNMENT = 64;
	[[nodiscard]] static constexpr uint64_t AlignDownload(uint64_t size) noexcept {
		return (size + DOWNLOAD_ALIGNMENT - 1) & ~(DOWNLOAD_ALIGNMENT - 1);
	}
	[[nodiscard]] static std::pair<uint64_t, uint64_t> DownloadEnvelope(const DownloadCopy& copy);
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// The BDA page tables' entries [first, first + count): written (`entries`) or cleared (null), in every table that
	// holds them (m_bda_tables). GrowBdaTable: a table with `entries` entries at least replaces the newest.
	void WriteBdaEntries(uint64_t first, uint64_t count, const vk::DeviceAddress* entries);
	void GrowBdaTable(uint64_t entries);
	// (FRAME-NATIVO2, KYTY_FRAME_NATIVO2: local/frame-nativo2.h) A read inside an imported DefaultTagg arena: its imported
	// buffer (null: not imported / not inside / GPU-written). The arenas' BDA entries pointed at their buffers.
	[[nodiscard]] std::pair<Buffer*, uint64_t> FrameNativo2Obtain(uint64_t vaddr, uint64_t size);
	void FrameNativo2WriteBda(int only, uint64_t first = 0, uint64_t count = 0);
	void FrameNativo2BdaToArena(uint64_t first, uint64_t count); // (FRAME-NATIVO2 v13, =6)
	void FrameNativo2Native6();                                  // (FRAME-NATIVO2 v13, =6)
	// GPU thread: InvalidateMemoryForWrite's work where GPU-written bytes are in the range.
	void DiscardGpuWrites(uint64_t vaddr, uint64_t size);
	void FrameNativo2DropGpu(uint64_t vaddr, uint64_t size); // (FRAME-NATIVO2 v10)
	void TouchBuffer(const Buffer& buffer);
	void TouchBufferNow(const Buffer& buffer); // (par2 fase 2: el toque en si; TouchBuffer lo difiere en un carril)
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Deleted buffers a speculation's packet may still read (Spec::PacketsPassed), destroyed once it passed.
	std::vector<std::pair<BufferId, Spec::PacketMarks>> m_retired;
	void                                                EraseRetired();
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void UploadDirtyRanges(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size, const Buffer** ring = nullptr);
	// The open upload prologue's copies by device address (VK_NV_copy_memory_indirect): one command as it closes.
	std::vector<VkCopyMemoryIndirectCommandNV> m_prologue_copies;
	void                                       FlushPrologueCopies(vk::CommandBuffer command);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void DownloadBufferMemory(std::span<const DownloadCopy> copies);
	void WriteBackGpuOwned(uint64_t address, const uint8_t* data, uint64_t size, const char* source);
	void ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write);
	void FinishWriteReadback(uint64_t vaddr, uint64_t size);
	struct GuestReadback;
	std::shared_ptr<GuestReadback> BeginGuestReadback(uint64_t address, uint64_t size,
	                                                  bool* completed = nullptr);
	void CopyGuestReadback(const std::shared_ptr<GuestReadback>& request);
	void FinishGuestReadback(size_t slot, bool detach = false);
	static constexpr size_t GuestReadbackSlots = 32;
	std::array<std::shared_ptr<GuestReadback>, GuestReadbackSlots> m_guest_readbacks {};
	std::array<std::unique_ptr<Buffer>, GuestReadbackSlots> m_guest_downloads {};
	uint32_t m_active_guest_readbacks = 0;
	struct CopyFeedback;
	void InvalidateCopyFeedback(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write);
	std::unique_ptr<CopyFeedback> m_copy_feedback;
	// KYTY_EAGER_FEEDBACK: the snapshots scheduled, in tick order.
	struct EagerFeedback {
		size_t   slot;
		uint64_t address, size, tick;
	};
	std::deque<EagerFeedback> m_eager_feedback;
	// KYTY_NATIVE_MEMORY: GPU buffer writes queued for write-back to guest memory once their tick completes (no read
	// watcher catches the game's read). NoteNativeWrite / WriteBackNativeWrites.
	struct NativeWrite {
		uint64_t address, size, tick;
	};
	std::deque<NativeWrite> m_native_writes;
	// 64 KiB granules some CPU-side read has ever pulled back (ReadMemory): the bound on large eager write-backs.
	std::unordered_set<uint64_t> m_native_read_granules;
	std::mutex                   m_native_read_mutex;
	void                         NoteNativeWrite(uint64_t vaddr, uint64_t size);
	void                         WriteBackNativeWrites();
	uint64_t                  m_eager_verify_count = 0;
	[[nodiscard]] bool        VerifyEagerFeedback(const EagerFeedback& entry);
	// KYTY_FEEDBACK_BATCH: the open buffer's copy-feedback snapshots, recorded together as it closes (FlushFeedbackBatch).
	struct FeedbackCopy {
		vk::Buffer source;
		uint64_t   source_offset, destination_offset, size;
	};
	std::vector<FeedbackCopy> m_feedback_batch;
	void                      FlushFeedbackBatch(CommandBuffer& command);
	bool TryReportLodStatsOnGpu(uint64_t address, bool reset);
	vk::Pipeline m_lod_pack_pipeline = nullptr;
	vk::PipelineLayout m_lod_pack_layout = nullptr;
	vk::DescriptorSetLayout m_lod_pack_descriptors = nullptr;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_lod_stats_buffer;
	// The BDA page tables: the newest last (m_bda_table), the ones it replaced kept and written alike - sets made before
	// it grew may still read them, never a freed buffer or a stale entry (GrowBdaTable).
	std::vector<std::unique_ptr<Buffer>>              m_bda_tables;
	std::atomic<Buffer*>                              m_bda_table {nullptr};
	bool                                              m_bda_zeroed = false; // the first table (WriteBdaEntries)
	static inline std::atomic<uint64_t>               s_bda_table_generation {0};
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	// (Read by a speculative translation's thread: src/graphics/guest_gpu/speculation.h.)
	std::atomic<uint64_t> m_registration_epoch {1};
	uint64_t              m_unregister_epoch {1};
	// Per tracker region, see RegionRegistrationEpoch.
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_registrations =
	    std::make_unique<std::atomic<uint64_t>[]>(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE);
	std::vector<GuestRange> m_registration_spans; // see TakeRegistrationSpans
	bool                    m_registration_spans_lost = true;
	// (tej-cos) RegisteredOverlapSince: the last RegLogN registrations/unregistrations, by epoch % RegLogN.
	static constexpr uint64_t RegLogN = 4096;
	struct RegLogEntry {
		std::atomic<uint64_t> epoch {0}, begin {0}, end {0};
	};
	std::unique_ptr<RegLogEntry[]> m_reg_log = std::make_unique<RegLogEntry[]>(RegLogN);
	struct SyncBuffer {
		uint64_t start;
		uint64_t end;
		Buffer* buffer;
		BufferId id; // with its generation: a later buffer in the same slot is another buffer
	};
	// GPU-thread-only derived index. SlotVector preserves addresses until erase;
	// every registration change invalidates this view before it can be reused.
	std::vector<SyncBuffer>                            m_sync_buffers;
	bool                                              m_sync_buffers_valid = false;
	struct SyncStamp { uint64_t begin = 0, end = 0, epoch = 0; };
	std::vector<SyncStamp> m_sync_stamps;
	std::vector<SyncBuffer> m_old_sync_buffers; // rebuild scratch
	std::vector<SyncStamp>  m_old_sync_stamps;
	// The registration changes since the index was current, applied to it in order (ApplySyncChanges) instead of a
	// rebuild from every registered buffer (a walk of m_buffers' nodes and slots: ~5 a frame at 1-1, 2% of the GPU
	// thread). Lost: too many, or the index was never built; the next use rebuilds it.
	struct SyncChange {
		BufferId id;
		uint64_t address = 0, size = 0;
		bool     insert  = false;
	};
	std::vector<SyncChange> m_sync_changes;
	bool                    m_sync_changes_lost = true;
	[[nodiscard]] bool      ApplySyncChanges();
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	std::shared_mutex                                 m_gpu_modified_mutex; // (its changes: HasGpuDirtyBytes)
	std::atomic<uint64_t>                             m_written_obtain_epoch {1};
	// Moves whenever m_gpu_modified_ranges loses bytes (readbacks, copy feedback, downloads, discards: CPU writes and
	// unmaps go through those first). GPU thread.
	uint64_t m_gpu_coverage_epoch = 1;
	// (trad1b) Per tracker region, the coverage epoch of its last loss (null until Trad1EnableCoverage: nothing stamped);
	// `far`: a loss not stamped by region (too wide, out of the table, before it existed).
	static constexpr uint64_t   Trad1CoverageRegions = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	std::unique_ptr<uint64_t[]> m_trad1_coverage;
	uint64_t                    m_trad1_coverage_far = 0;
	void Trad1StampCoverage(uint64_t vaddr, uint64_t size) {
		if (m_trad1_coverage == nullptr || size == 0) return;
		const auto first = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		if (last >= Trad1CoverageRegions || last - first > 4096) {
			m_trad1_coverage_far = m_gpu_coverage_epoch;
			return;
		}
		for (auto region = first; region <= last; ++region) m_trad1_coverage[region] = m_gpu_coverage_epoch;
	}
	template <typename Ranges>
	void Trad1StampCoverageAll(const Ranges& ranges) {
		if (m_trad1_coverage == nullptr) return;
		for (const auto& range: ranges) Trad1StampCoverage(range.address, range.size);
	}
	// The ranges ObtainBuffer made GPU-written lately (GPU thread): one written again while no buffer was registered
	// or unregistered and no GPU-written byte was given back since is the same buffer's, its pages still GPU-owned
	// and its bytes still in m_gpu_modified_ranges (what SynchronizeBuffer and the Add would find).
	struct WriteMemo {
		uint64_t vaddr = 0, size = 0, registration = 0, coverage = 0;
		BufferId id;
		bool     texel = false;
	};
	std::array<WriteMemo, 4096> m_write_memo {};
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	// ObtainBufferForImage (KYTY_ASYNC_UPLOAD=2): the backing pieces of an image upload.
	std::vector<std::pair<const uint8_t*, uint64_t>>  m_backing_pieces;
	// The upload worker's copies of guest [vaddr, vaddr + size) to `staging`; false when it reads the mapped parts
	// only (a partly unmapped range: the rest is zeros).
	bool PushImageStagingCopies(uint8_t* staging, uint64_t vaddr, uint64_t size);
	// ImageProtectSink's protections: those of pages a copy of [vaddr, vaddr + size) the upload worker makes next reads
	// whole go ahead of it, the others are applied here. `whole_image`: the range is the image's (bytes of its first and
	// last pages outside it are no other watcher's, or their pages would have been watched already).
	std::vector<PageManager::DeferredRange> m_image_protects;
	PageManager*                            m_image_protect_manager = nullptr;
	void QueueImageProtects(std::span<const StagingPiece> copied, bool whole_image);
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_host_shader_upload;
	StreamBuffer                                      m_table_upload;
	// Buffer uploads stage here while it has room (UploadCopies): device-local host-visible memory, whose copies read
	// video memory instead of system memory across the bus.
	StreamBuffer                                      m_staging_device;
	StreamBuffer                                      m_download_buffer;
	// (VRAM-BUF, KYTY_VRAM_BUF=1 por defecto) Anillo de 128 MiB de VRAM solo para copias imagen->imagen de formato distinto
	// (TextureCache::CopyImage -> CopyImageWithBuffer); se crea al primer uso en vez de al arrancar. =0: al arrancar, como antes.
	std::unique_ptr<StreamBuffer>                     m_device_buffer;
	std::mutex                                        m_device_buffer_mutex;
	std::atomic<bool>                                 m_device_buffer_made {false};
	StreamBuffer&                                     DeviceUtilityBuffer();
	void                                              VramBufReport();
	TextureCache&                                     m_texture_cache;
	GpuResourceManager*                               m_resources = nullptr;

public:
	// The guest range is mapped for the GPU (not unmapped since): deferred work checks this
	// before it creates buffers over ranges recorded earlier.
	[[nodiscard]] bool IsGpuMapped(uint64_t vaddr, uint64_t size) const noexcept;

private:
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_cache_bytes        = 0; // the registered buffers' bytes
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	// When each recent collector tick began (steady clock, by tick % size): RetireStaleBuffers counts idleness in time.
	std::array<int64_t, 8192> m_tick_times {};
	bool                      m_retire_pressure = false; // the video memory in use at the critical level
	// The registered game buffers in system memory and the room those in use need (MoveSpilledBufferBack).
	std::vector<BufferId> m_spilled;
	uint64_t              m_spilled_room = 0;

	// KYTY_READBACK_QUEUE: GPU writes that may still be in flight, oldest first. An entry's
	// tick is set at the next point between commands (every write noted by then is recorded
	// at or before the current tick: m_gpu_write_stamps); entries the GPU completed are dropped
	// from the front.
	struct GpuWrite {
		uint64_t begin, end;
		bool     big = false; // spans more granules than GpuWriteSpan: in m_gpu_big_writes, not m_gpu_write_last
		// KYTY_SYNC_STATS: the compute shader being translated when it was noted, and the noting code's return address.
		uint64_t shader = 0, caller = 0;
	};
	// KYTY_SYNC_STATS: the latest logged write over [begin, end) into SyncStats' writer (none: zeros).
	void NoteWriterOf(uint64_t begin, uint64_t end);
	// The ticks of the stamped writes, oldest first: (end, tick) dates the writes numbered from the previous entry's end
	// (the head's number for the first) up to `end`. (A tick in every entry: each was written once and read once more,
	// a pass over thousands of cold entries a frame.)
	std::deque<std::pair<uint64_t, uint64_t>> m_gpu_write_stamps;
	[[nodiscard]] uint64_t                    GpuWriteTick(size_t index) const;
	// Per 64 KiB granule (hashed into the counters), 1 + the number (m_gpu_writes_base + index) of the last small write
	// there; a write over more than GpuWriteSpan granules is in m_gpu_big_writes instead. The in-flight writes are the
	// log's tail: a range whose granules' last write completed overlaps no in-flight small write, and the walk for one
	// that may starts at that last write, not at the log's end (thousands of entries a frame at 1-1). (Counting the
	// writes per granule, up at each write and down at its completion, cost a random read-modify-write per granule.)
	static constexpr uint32_t   GpuWriteGranuleBits = 16, GpuWriteCounters = 1u << 16u, GpuWriteSpan = 64;
	std::unique_ptr<uint64_t[]> m_gpu_write_last = std::make_unique<uint64_t[]>(GpuWriteCounters);
	// The numbers of the big writes, oldest first (the completed ones dropped at queries): a query looks for an
	// overlapping one in them (one in flight, a 4-8 MiB copy every frame, made every query walk the log from its end).
	std::deque<uint64_t>        m_gpu_big_writes;
	uint64_t                    m_gpu_writes_base = 0; // the number of m_gpu_writes[0]
	[[nodiscard]] static uint32_t GpuWriteCounter(uint64_t granule) {
		return static_cast<uint32_t>((granule * 0x9e3779b97f4a7c15ull) >> (64u - 16u));
	}
	void ResetGpuWrites();
	void                  NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t caller = 0);
	[[nodiscard]] uint64_t InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed,
	                                         bool between_commands);
	// The graphics tick a transfer-queue copy of `ranges` ({begin, end}) waits for; false when
	// the copy must drain the graphics queue as before.
	[[nodiscard]] bool ReadbackQueueReady(std::span<const std::pair<uint64_t, uint64_t>> ranges,
	                                      bool between_commands, uint64_t& wait_tick);
	std::vector<GpuWrite> m_gpu_writes;
	size_t                m_gpu_writes_head    = 0;
	size_t                m_gpu_writes_stamped = 0;
	bool                  m_gpu_writes_on      = false;
	// Writes before this tick were not noted (UINT64_MAX: set at the next stamp).
	uint64_t              m_gpu_writes_from    = UINT64_MAX;
	// Last copy-engine value, and last graphics tick, that wrote each download slot: a detached
	// request's copy may still be pending on either queue when its slot is reused.
	std::array<uint64_t, GuestReadbackSlots> m_download_queue_values {};
	std::array<uint64_t, GuestReadbackSlots> m_download_ticks {};
	// Mode 3 (verification): the same bytes copied on the graphics queue, compared on use.
	std::array<std::unique_ptr<Buffer>, GuestReadbackSlots> m_verify_downloads {};
	void VerifyReadback(uint64_t address, const uint8_t* fast, const uint8_t* reference, uint64_t size,
	                    const char* path, uint64_t waited);

	// (parpadeo7) The helper thread of PrefetchGuestRead: copies the prefetched readbacks in, asks the GPU thread to retire
	// them and applies their pages' deferred releases. Created at the first prefetch.
	struct PrefetchHelper;
	std::unique_ptr<PrefetchHelper> m_prefetch_helper;
	std::mutex                      m_prefetch_helper_mutex; // (its creation and stop)
	void                            PrefetchHelperRun();
	void                            QueuePrefetchApply(const std::shared_ptr<GuestReadback>& request);

	// Last member: destroyed (queue idle) before the buffers its copies use.
	std::unique_ptr<ReadbackQueue::Queue> m_readback_queue;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
