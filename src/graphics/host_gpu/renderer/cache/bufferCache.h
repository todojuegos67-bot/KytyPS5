#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/preflushTable.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer holds three bitmaps of CACHING_NUMPAGES bits: pages that device-address
	// accesses found without a cached buffer, pages that runtime V# stores wrote, and pages where
	// such a store found no cached buffer and was lost.
	static constexpr uint64_t WRITE_BITMAP_OFFSET   = CACHING_NUMPAGES / 8;
	static constexpr uint64_t DROPPED_BITMAP_OFFSET = 2 * WRITE_BITMAP_OFFSET;
	static constexpr uint64_t FAULT_BUFFER_SIZE     = 3 * WRITE_BITMAP_OFFSET;
	// Tags the BDA page-table entries of pages a DMA shader accessed; an access to another page
	// records a fault.
	static constexpr vk::DeviceAddress BDA_ACCESSED = 1;

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// Of a written range, only [vaddr + written_offset, + written_size) becomes GPU-written. A tracked
	// one (a shader that records its writes, BufferResource::tracked) is followed by TrackWrites.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {},
	                                                        uint64_t written_offset  = 0,
	                                                        uint64_t written_size    = UINT64_MAX,
	                                                        std::span<const std::pair<uint64_t, uint64_t>> written_ranges = {},
	                                                        bool tracked = false);
	// The device address of a zeroed bitmap, a bit for each 4 KiB from `base`, in which the dispatch
	// being recorded sets the pages of [vaddr, vaddr + size) it writes; once it has run, the others
	// stop being GPU-written. 0 when the range is not tracked: all of it stays GPU-written.
	[[nodiscard]] uint64_t TrackWrites(const Buffer& buffer, uint64_t base, uint64_t vaddr, uint64_t size);
	// After the dispatch whose bitmaps TrackWrites gave: copies them for the host.
	void CloseTrackedWrites();
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	// A PM4 packet that may write guest memory or wait on it: no small stream copy made before it is reused after it.
	void NewStreamEpoch() noexcept { m_stream_epoch++; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	// How many joins there have been: a binding taken before a join may name the old buffer, so a draw's or a dispatch's
	// acquisitions repeat while this moves, at most kJoinPasses times.
	[[nodiscard]] uint64_t Joins() const noexcept { return m_joins; }
	static constexpr uint32_t kJoinPasses = 4;
	// A shader's own read-only words (CompiledShaderInfo::program_constants), written once into a buffer the GPU
	// reads, kept for the run.
	[[nodiscard]] const Buffer& GetProgramConstants(const std::shared_ptr<const std::vector<uint32_t>>& words);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// Guest memory nothing has written since its first mapping reads as zeros: uploads fill it instead of
	// reading it (MemoryTracker::MarkRegionAsZero). Unmark when that stops being certain (an alias, a
	// write that bypasses the page protection).
	// A read fault on a page a tracked dispatch did not write (kept read-protected): lifts the kept pages of its
	// 256 KB, on the faulting thread; false if the page is not kept.
	[[nodiscard]] bool ReleaseKept(uint64_t vaddr);
	void MarkMemoryZero(uint64_t vaddr, uint64_t size);
	void UnmarkMemoryZero(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// Copies guest bytes for the host's own use without tripping the page protection. A page is
	// protected once a shader wrote any byte of it, and what the host wants is often that byte's
	// neighbour (dispatch arguments the game wrote next to a compute shader's output): only bytes
	// a shader did write are read back from the GPU first.
	void ReadGuestBytes(uint64_t vaddr, void* data, uint64_t size);
	// Its twin for the host's own stores (the command processor's labels): on a page a shader wrote,
	// a store would read the page back. The bytes go to the backing and, in command order, to the
	// cached buffer, where a read back and a store would leave them. False on any other page: the
	// caller stores as usual.
	[[nodiscard]] bool WriteGuestBytes(uint64_t vaddr, const void* data, uint64_t size);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Any part of the range in a cached buffer; from any thread (IsRegionRegistered needs the GPU thread).
	[[nodiscard]] bool HoldsRegion(uint64_t vaddr, uint64_t size) const;
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               PrepareFaultBuffer() { m_fault_manager.PrepareFaultBuffer(); }
	void               ProcessFaultBuffer();
	[[nodiscard]] uint64_t LastFaultTick() const noexcept { return m_fault_manager.LastPendingTick(); }
	// Tags fault-scan pages as DMA-accessed and queues them; creates buffers for unbacked ones.
	void MarkDmaAccess(const RangeSet& pages);
	// Uploads the CPU-dirty bytes of the queued ranges `mapped` covers behind one barrier pair.
	void SynchronizeBda(const RangeSet& mapped);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	// A write queues only the pages a DMA shader accessed.
	void               QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write = false);
	// A guest map (any thread): sparse buffers back its pages at their next synchronization.
	void NoteMapped(uint64_t vaddr, uint64_t size);
	void ProcessWriteBuffer(FaultManager::PagesHandler&& handler) {
		m_fault_manager.ProcessWriteBuffer(std::move(handler));
	}
	void ProcessDroppedStores(FaultManager::PagesHandler&& handler) {
		m_fault_manager.ProcessDroppedStores(std::move(handler));
	}
	// Records stores that a shader made through device addresses, known once it completed: the
	// bytes of [vaddr, vaddr + size) that a cached buffer holds become GPU modified, except on
	// pages the CPU dirtied since, whose upload replaces them.
	void NoteGpuWrites(uint64_t vaddr, uint64_t size);
	// Calls func(address, size) for the runs of [vaddr, vaddr + size) the CPU may have written
	// since the previous call over them (see MemoryTracker::ConsumeCpuWrites).
	template <typename Func>
	void ConsumeCpuWrites(uint64_t vaddr, uint64_t size, Func&& func) {
		m_memory_tracker.ConsumeCpuWrites(vaddr, size, std::forward<Func>(func));
	}
	// Calls func(address, size) for the ranges whose buffers the garbage collector released since
	// the previous call, and forgets them; the first call starts recording them. Merging buffers
	// keeps every address cached.
	template <typename Func>
	void ConsumeReleasedRanges(Func&& func) {
		m_released_ranges.ForEach(func);
		m_released_ranges.Clear();
		m_record_released = true;
	}
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	// Copies [previous upload's end, end) go to `buffer`.
	struct BufferUpload {
		Buffer* buffer;
		size_t  end;
	};
	// A known-zero run, filled on the GPU instead of read from guest memory.
	struct BufferFill {
		Buffer*  buffer;
		uint64_t offset;
		uint64_t bytes;
	};
	// What one barrier pair uploads.
	struct UploadBatch {
		std::vector<vk::BufferCopy> copies;
		std::vector<BufferUpload>   uploads;
		std::vector<BufferFill>     fills;
		uint64_t                    total_size = 0;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// A sparse buffer's pages under [vaddr, vaddr + size) that hold guest memory get memory of their own.
	void BackCommitted(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void BackPendingMaps();
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	// Adds the range's CPU-dirty runs to batch. With source, they are read from guest memory inside the
	// tracker walk (a written range's state changes after it); without, the caller reads them later.
	void CollectUploads(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written, UploadBatch& batch,
	                    vk::Buffer* source);
	[[nodiscard]] vk::Buffer UploadCopies(std::span<const BufferUpload> uploads,
	                                      std::span<vk::BufferCopy> copies, uint64_t total_size);
	void                     RecordUploads(vk::Buffer source, const UploadBatch& batch);
	// Uploads `copies` through the staging ring in parts, recording each part's copy as it goes.
	void UploadInParts(Buffer& buffer, std::span<const vk::BufferCopy> copies);
	void UploadInParts(const UploadBatch& batch);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse. All of
	// `ranges` go in one copy (its barriers cost the GPU more than the copy); ranges with no GPU-written
	// bytes are removed from the list.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, std::vector<std::pair<uint64_t, uint64_t>>& ranges);
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges {{vaddr, size}};
		return DownloadBufferMemory<async>(buffer, ranges);
	}
	// Applies the bitmaps of the tracked dispatches that have run (TrackWrites).
	void ResolveTrackedWrites();
	// The pre-flush (PreflushTable): a GPU write of [vaddr, vaddr + size) in the command buffer being recorded;
	// at its end, the copies of the pages written in it into their slots.
	void PreflushWrite(uint64_t vaddr, uint64_t size);
	void RecordPreflushCopies();
	// Publishes the GPU-written bytes of [vaddr, vaddr + size) from the pages' slots, as a download would, and
	// unmarks them; false, with nothing done, unless every such page's slot holds its last GPU write.
	[[nodiscard]] bool PublishPreflushed(uint64_t vaddr, uint64_t size);

	struct TrackedWrite {
		uint64_t base   = 0; // the guest address of the bitmap's first page
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint32_t offset = 0; // in dwords, in m_track_device and m_track_host
		uint32_t words  = 0;
		uint64_t tick   = 0; // 0 until the dispatch is recorded (CloseTrackedWrites)
	};
	static constexpr uint32_t TRACK_WORDS = 1024 * 1024; // bitmaps of 128 GB of bindings, resolved at reads and frames
	void ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	mutable std::mutex                                m_registered_mutex;
	RangeSet                                          m_registered; // m_buffers' ranges, under m_registered_mutex
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	RangeSet                                          m_released_ranges;
	bool                                              m_record_released = false;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	std::mutex                                        m_pending_maps_mutex;
	RangeSet                                          m_pending_maps; // NoteMapped, under m_pending_maps_mutex
	std::atomic<bool>                                 m_has_pending_maps {false};
	std::vector<BufferId>                             m_sparse_buffers; // GPU thread
	// Pages DMA shaders accessed, guarded by m_bda_sync_mutex.
	RangeSet                                          m_dma_accessed;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	// Small stream copies (ObtainBuffer) by (vaddr, size): reused within the command buffer and epoch they were made in.
	struct StreamCopy {
		uint64_t vaddr = 0, size = 0, tick = 0, epoch = 0, offset = 0;
	};
	std::array<StreamCopy, 1024> m_stream_copies {}; // GPU thread
	uint64_t                     m_stream_epoch = 1; // GPU thread
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	Buffer                                            m_track_device;
	Buffer                                            m_track_host;
	// FIFO over the two buffers above; m_track_head is the next free dword.
	std::deque<TrackedWrite>                          m_tracked_writes;
	uint32_t                                          m_track_head = 0;
	// The GPU-written bytes a writer is known to have written: one that does not record its writes,
	// or a tracked one whose bitmap showed them. Only the other GPU-written bytes can be cleared by a
	// bitmap (ResolveTrackedWrites).
	RangeSet                                          m_known_written;
	// GetProgramConstants' buffers, by the words' address; each keeps its words alive, so no other words can
	// take that address.
	std::unordered_map<const std::vector<uint32_t>*,
	                   std::pair<std::shared_ptr<const std::vector<uint32_t>>, std::unique_ptr<Buffer>>>
	                                                  m_program_constants;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// Only from here a buffer a shader wrote is read back to be deleted (the memory budget itself).
	uint64_t m_budget_gc_memory   = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_joins              = 0;
	// Read backs of this frame and the previous one: tracker page, the byte read there.
	std::unordered_map<uint64_t, uint64_t> m_read_backs;
	std::unordered_map<uint64_t, uint64_t> m_previous_read_backs;
	bool                                   m_read_ahead_done = false;
	// Pre-flush slots: a tracker page each, in chunks of host-visible memory allocated as the history grows
	// (64 MB at most; Marvel's Wolverine's titles read back 1-12 MB of pages in two frames).
	static constexpr uint32_t PREFLUSH_CHUNK_SLOTS = 1024;
	static constexpr uint32_t PREFLUSH_SLOTS       = 16 * PREFLUSH_CHUNK_SLOTS;
	// 8 copies of a page between two read backs, not the table's 2: at Marvel's Wolverine's Trask stand 25 of the 28
	// misses a frame without a slot were pages dropped at 2 copies, and each such miss is a download that waits for the GPU.
	PreflushTable                          m_preflush {PREFLUSH_SLOTS, 8};
	std::vector<std::unique_ptr<Buffer>>   m_preflush_chunks;
	bool                                   m_in_read_back = false; // ReadMemory's downloads enter the pre-flush
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
