#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <array>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

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
	void                        SetGpu(GuestGpu* gpu) noexcept { m_gpu = gpu; }
	// After a guest protection change: tracked pages get the tracker's protection back.
	void ReapplyProtection(uint64_t vaddr, uint64_t size) { m_page_manager.ReapplyProtection(vaddr, size); }
	void SyncProtection(uint64_t vaddr, uint64_t size) { m_page_manager.SyncProtection(vaddr, size); }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
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
	// Once per flip: the frame count and the garbage collection (its ages count frames).
	void               AdvanceFrame();
	// A flip the GPU itself submits (EOP flip, the game's normal frames): the next completed submission
	// advances the frame. Only the CPU's flips (menus, loading screens) did, so in play the caches were
	// never collected and video memory filled to the driver's limit.
	void               NoteGpuFlip() noexcept { m_gpu_flip_pending.store(true, std::memory_order_release); }

private:
	[[nodiscard]] bool        TryInvalidateCpuWriteWindow(uint64_t fault);
	void                      SynchronizeDirtyBdaRegions(GuestRange range);
	void RefreshBdaRanges();
	std::atomic<bool>         m_gpu_flip_pending {false};
	uint64_t                  m_frames = 0;
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
	GuestGpu*                 m_gpu = nullptr;
	std::atomic<uint64_t> m_preparation_alias_epoch {uint64_t{1} << 32};
	bool                      m_fault_process_pending = false;
	bool                      m_bda_used              = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
