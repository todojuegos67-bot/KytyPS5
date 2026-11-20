#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class GuestGpu;

class GpuResourceManager {
public:
	GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler);
	~GpuResourceManager();
	KYTY_CLASS_NO_COPY(GpuResourceManager);

	[[nodiscard]] BufferCache&  GetBufferCache() { return m_buffer_cache; }
	[[nodiscard]] TextureCache& GetTextureCache() { return m_texture_cache; }
	[[nodiscard]] bool          HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
        return m_page_manager.HasReadWatchers(vaddr, size);
	}
	// (ww2, KYTY_WRITE_WATCH_MODE=protect; false otherwise: PageManager::HasWriteProtection.)
	[[nodiscard]] bool HasWriteProtection(uint64_t vaddr, uint64_t size) const noexcept {
		return m_page_manager.HasWriteProtection(vaddr, size);
	}
	[[nodiscard]] const std::atomic<uint64_t>* ReadGranules() const noexcept { return m_page_manager.ReadGranules(); }
	void                        SetGpu(GuestGpu* gpu) noexcept { m_gpu = gpu; }
	// After a guest protection change: tracked pages get the tracker's protection back.
	void ReapplyProtection(uint64_t vaddr, uint64_t size) { m_page_manager.ReapplyProtection(vaddr, size); }
	// The pages of a range the trackers protect (PageManager::WatchedSpan).
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
		return m_page_manager.WatchedSpan(vaddr, size, first, end);
	}
	void SyncProtection(uint64_t vaddr, uint64_t size) { m_page_manager.SyncProtection(vaddr, size); }
	// (PageManager::WatchedRuns: for a holder of the guest address-space lock.)
	void WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<PageManager::WatchedRun>* runs) const {
		m_page_manager.WatchedRuns(vaddr, size, runs);
	}

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// (policy2, KYTY_DS_READBACK_HOOK) HandleFault's read branch for a whole range (a guest thread about to read all
	// of it): one readback instead of one per faulting page. False: the range is not all mapped (nothing done).
	[[nodiscard]] bool ReadbackRange(uint64_t vaddr, uint64_t size) noexcept;
	// (parpadeo7, KYTY_PARPADEO7=1; GPU thread, a command queued at an XPR label) Each range ({address, bytes}, sorted)
	// that holds GPU-written pages begun as an asynchronous guest readback (BufferCache::PrefetchGuestRead), never
	// waited for here. `label_ns`: the label's time (local/parpadeo7.h, for its line).
	void PrefetchGuestReads(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, int64_t label_ns) noexcept;
	// KYTY_NATIVE_MEMORY=1 only (nothing otherwise). GPU thread, at the points where guest writes become visible to
	// the GPU work after them - the start of each submission (GuestGpu::Process) and each satisfied WAIT_REG_MEM, the
	// points BdaDirtyRegions::Publish marks -, with no cache lock held: every page the game wrote since the last sweep
	// that a watcher holds (PageManager::SweepCpuWrites) is invalidated as its write fault did (HandleFault: the
	// buffers, a GPU-written page read back first, then the images), before any resource is used.
	// (ww2, KYTY_WRITE_WATCH_MODE=poll; any other mode sweeps whatever the reason, as before) Why the caller is at a
	// hand-over: Handover sweeps; CoveredSubmission (a submission queued before the last sweep began,
	// PageManager::SweptSince) and ProcessorWait (a WAIT_REG_MEM satisfied by the value a command processor wrote
	// itself) carry no write that sweep did not read, and only count.
	enum class SweepReason { Handover, CoveredSubmission, ProcessorWait };
	void SweepCpuWrites(SweepReason reason = SweepReason::Handover);
	// (ww2) A flip, for the "Write watch+" line's per-frame figures (GPU thread; nothing without KYTY_WRITE_WATCH).
	void NoteFlipForWatchStats();
	// KYTY_PROTECT_PAUSE: the pages whose write protection a pause skipped, invalidated as written (and protected
	// again when it ends). GPU thread; called from SweepCpuWrites.
	void SyncProtectPause();
	// KYTY_NATIVE_MEMORY=1 only (nothing otherwise). GPU thread, right before the game can see that GPU work it submitted
	// is done - an end-of-pipe label or event (WRITE_EOP / RELEASE_MEM), an end-of-pipe interrupt, a flip -: everything
	// the GPU wrote to guest memory is written there first, with the GPU waited for (no read fault serves the game's
	// reads): every GPU-written range of the buffers (BufferCache::NativeWriteBackAll) and every image GPU work wrote
	// since the last call (TextureCache::NativeDownloadWritten; KYTY_NATIVE_IMAGE_DOWNLOAD=0 leaves them out). Slow.
	void NativeFlushGpuWrites();
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Before a host write of the whole range: GPU-written bytes it replaces are dropped (BufferCache::InvalidateMemoryForWrite).
	[[nodiscard]] bool InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	// Before a host write of the range: the images over it only (their bytes are rewritten, as InvalidateMemory does).
	[[nodiscard]] bool InvalidateImages(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// The mapped parts of [vaddr, vaddr + size), added to `out`.
	void               MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	// `releasing`: the caller unmaps or decommits the range right after (munmap, free,
	// memory pool decommit), not a range about to receive a mapping.
	void               UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing = false);
	void               PrepareBda();
	void BeginAliasWrite() noexcept;
	void EndAliasWrite() noexcept;
	[[nodiscard]] uint64_t PreparationAliasEpoch() const noexcept;
	// Written only under m_mapped_ranges_mutex; readers compare it for change only,
	// so an acquire load needs no lock.
	[[nodiscard]] uint64_t MappingEpoch() const noexcept {
		return m_mapping_epoch.load(std::memory_order_acquire);
	}
	// Whether memory in [vaddr, vaddr + size) was unmapped after mapping epoch `epoch` (by 16 KiB
	// granule). GPU thread, as UnmapMemory.
	[[nodiscard]] bool UnmappedSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept;

	bool PrepareBdaReadRanges(std::span<const GuestRange> ranges);
	// After each completed submission: its shaders' fault records and the image downloads.
	void               EndSubmission();
	// A flip the GPU makes (an end-of-pipe flip, all of Demon's Souls' frames): the collectors run at the end of its
	// submission (EndSubmission), as AdvanceFrame runs them for a CPU flip. Since collection moved to the flips
	// (81cd311) they never ran in the game, and a GPU under 32 GB filled up. (The texture cache's frame, which
	// ResolveOverlap's stale-image test counts, still advances with CPU flips only: in this game it never did.)
	void               NoteGpuFlip() { m_gpu_flip_pending = true; }
	// Once per CPU flip: the frame count and the garbage collection (its ages count frames).
	void               AdvanceFrame();

private:
	void                      RunCollectors();
	// SweepCpuWrites: one run of written pages, all mapped.
	void                      InvalidateCpuWrittenPages(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool        TryInvalidateCpuWriteWindow(uint64_t fault);
	// One window of the fault's mapped range: [window_begin, window_begin + window_size) within it.
	[[nodiscard]] bool        TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t window_begin, uint64_t window_size);
	void                      SynchronizeDirtyBdaRegions(GuestRange range);
	void RefreshBdaRanges();
	// Fresh memory (src/local/fresh-memory.h): the never-written granules of a mapping the CPU and the GPU may use,
	// write-watched until their first write (MapMemory), unwatched when unmapped (UnmapMemory: `mark` counts them as
	// written, as when the mapping stays without the watch), a first write's (HandleFault: true when it was one) and a
	// host overwrite's (InvalidateMemoryForWrite) granules written. Each watched part is one granule's bytes in one
	// mapping, by its guest address; the parts of a granule (all its mappings) go together.
	void FreshWatch(uint64_t vaddr, uint64_t size);
	void FreshUnwatch(uint64_t vaddr, uint64_t size, bool mark);
	bool FreshFirstWrite(uint64_t vaddr);
	void FreshOverwrite(uint64_t vaddr, uint64_t size);
	void FreshUnwatchGranule(uint64_t granule); // (m_fresh_mutex held)
	struct FreshPart {
		uint64_t end     = 0;
		uint64_t granule = 0;
	};
	std::mutex                                          m_fresh_mutex;
	std::map<uint64_t, FreshPart>                       m_fresh_parts;    // by guest address
	std::unordered_map<uint64_t, std::vector<uint64_t>> m_fresh_granules; // the addresses of the parts of a granule
	uint64_t                                            m_fresh_report_flip = 0;
	PageManager               m_page_manager;
	CommandScheduler&         m_scheduler;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::atomic<uint64_t> m_mapping_epoch {1};
	// The mapping epoch of each 16 KiB granule's latest unmap, in 1 GiB leaves made at its first unmap
	// (written by UnmapMemory's GPU-thread part). The latest 64 unmaps before: streaming textures in
	// unmaps that many 64 KiB pool layers within frames, so a proof a few frames old counted as stale.
	static constexpr uint64_t UnmapGranuleBits = 14, UnmapLeafBits = 16;
	std::unordered_map<uint64_t, std::unique_ptr<uint64_t[]>> m_unmap_epochs;
	uint64_t m_unmap_latest = 0; // the latest unmap's mapping epoch
	// The spans whose part of RefreshBdaRanges' list (registered buffers' mapped parts) may have changed since it
	// last ran: mapping changes over a registered buffer and buffers (un)registered. Only their tracker regions are
	// collected again: the whole list (0.4 ms over ~2200 buffers) was, several times a frame while the game streams
	// in a new area. `m_bda_rebuild`: everything (at first, or after more changes than are kept). GPU thread.
	void                    NoteBdaSpan(uint64_t vaddr, uint64_t size);
	std::vector<GuestRange> m_bda_spans;
	bool                    m_bda_rebuild = true;
	std::vector<BufferCache::SyncRegionRequest> m_bda_region_requests, m_fresh_bda_region_requests;
	// (render3) KYTY_BDA_MEMO=1: the tracker regions (inclusive runs) whose dirty bits PrepareBdaReadRanges consumed
	// while no bit was set (BdaDirtyRegions::Generation), no buffer was (un)registered and no mapping changed since
	// `generation`, `registration` and `mapping` were read: a preparation whose ranges lie in them has nothing to do.
	// GPU thread.
	struct BdaMemo {
		static constexpr size_t                           Capacity = 8;
		uint64_t                                          generation = 0, registration = 0, mapping = 0;
		size_t                                            count = 0;
		std::array<std::pair<uint64_t, uint64_t>, Capacity> regions {};
	};
	BdaMemo m_bda_memo;
	GuestGpu*                 m_gpu = nullptr;
	std::atomic<uint64_t> m_preparation_alias_epoch {uint64_t{1} << 32};
	bool                      m_fault_process_pending = false;
	bool                      m_bda_used              = false;
	bool                      m_gpu_flip_pending      = false;
	GraphicContext&           m_graphics;
	uint64_t                  m_collections           = 0; // RunCollectors calls (one a flip)
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
