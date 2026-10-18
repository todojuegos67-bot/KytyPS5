#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
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
	// The BDA page table is two-level: a directory with one entry per BDA_CHUNK_PAGES guest pages
	// (64 MiB of guest space), holding 0 or the element index of a chunk of per-page buffer
	// addresses. Chunks come from a fixed pool behind the directory, taken when a buffer first
	// covers their region and returned when the last one leaves. The flat table was 768 MiB of
	// device memory (8 bytes per 16 KiB page of the 1.5 TiB guest space), almost all zeros.
	static constexpr uint32_t BDA_CHUNK_BITS        = 12;
	static constexpr uint64_t BDA_CHUNK_PAGES       = uint64_t {1} << BDA_CHUNK_BITS;
	static constexpr uint64_t BDA_DIRECTORY_ENTRIES = CACHING_NUMPAGES >> BDA_CHUNK_BITS;
	static constexpr uint64_t BDA_CHUNK_COUNT       = 2048;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    (BDA_DIRECTORY_ENTRIES + BDA_CHUNK_COUNT * BDA_CHUNK_PAGES) * sizeof(vk::DeviceAddress);
	static_assert((CACHING_NUMPAGES & (BDA_CHUNK_PAGES - 1)) == 0);
	static_assert(BDA_DIRECTORY_ENTRIES + BDA_CHUNK_COUNT * BDA_CHUNK_PAGES < (uint64_t {1} << 32u));

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
	// needs_device_address: the caller reads the data through a buffer device address, which
	// the stream buffer used for small CPU-written reads does not have.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer      = false,
	                                                        BufferId id                   = {},
	                                                        bool     needs_device_address = false);
	// A written binding of which the shader's stores reach only [written_vaddr, written_vaddr +
	// written_size): the whole range is synchronized, only that part becomes GPU-written.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferWritten(uint64_t vaddr, uint64_t size,
	                                                               uint64_t written_vaddr,
	                                                               uint64_t written_size,
	                                                               BufferId id = {});
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
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// GPU thread: a cached buffer, a pending asynchronous readback or an in-flight download
	// shares bytes with [vaddr, vaddr + size) (KYTY_UNMAP_SKIP_IDLE). Conservative.
	[[nodiscard]] bool              MayReferenceRange(uint64_t vaddr, uint64_t size);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void                            ProcessFaultBuffer();
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	[[nodiscard]] bool              WriteClean(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	void                            SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	// Synchronizes the buffers of the ranges queued by QueueBdaSync, within the mapped ranges.
	void                            SynchronizeBda(const RangeSet& mapped);
	void                            QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	void                            RunGarbageCollector();
	// Bytes of the cached buffers.
	[[nodiscard]] uint64_t UsedMemory() const noexcept { return m_total_used_memory; }

	// Diagnostics: the guest shader whose bindings are being prepared on this thread, if any.
	inline static thread_local uint64_t s_diag_shader_hash = 0;
	// KYTY_READBACK_STATS: the shader whose resources are bound next (GPU thread).
	static void SetReadbackStatsShader(uint64_t hash);
	// Device-loss triage: the buffer (and its device address) the host page table holds for each
	// guest page of the range, i.e. what the BDA page table should contain.
	void DescribeGuestPages(uint64_t address, uint64_t size);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// Records bytes a binding makes GPU-written (after SynchronizeBuffer marked their pages).
	void MarkGpuWritten(uint64_t vaddr, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                                        uint64_t total_size);

	// GPU thread: a guest fault on GPU-written memory, resolved synchronously.
	void ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write);
	// GPU thread: KYTY_DIRECT_READBACK (see bufferCache.cpp).
	[[nodiscard]] bool TryDirectReadback(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_write);
	// GPU thread: KYTY_COPY_QUEUE_READBACK (see bufferCache.cpp). Downloads the window's
	// GPU-written bytes on the readback queue when every GPU write to the buffer has executed.
	[[nodiscard]] bool TryCopyQueueReadback(Buffer& buffer, uint64_t window_begin,
	                                        uint64_t window_end);
	struct CopyQueueReadback {
		vk::CommandPool         pool    = nullptr;
		vk::CommandBuffer       command = nullptr;
		vk::Fence               fence   = nullptr;
		std::unique_ptr<Buffer> staging;
	};
	CopyQueueReadback m_copy_queue_readback;
	// KYTY_ASYNC_WRITE_READBACK (see ReadMemory). A window whose download is submitted but not
	// yet published to guest memory; its pages stay GPU-owned until the readback is completed.
	// Only the GPU thread touches the list.
	struct PendingWriteReadback {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	// The asynchronous readback of a guest thread's fault (KYTY_ASYNC_WRITE_READBACK for writes,
	// KYTY_ASYNC_READBACK for reads).
	[[nodiscard]] uint64_t BeginWriteReadback(uint64_t vaddr, uint64_t size, bool is_write,
	                                          uint64_t& window_begin, uint64_t& window_end);
	// False when `snapshot` and the GPU wrote the page again: the caller reads the downloaded
	// bytes.
	bool FinishWriteReadback(uint64_t vaddr, uint64_t size, bool is_write, uint64_t window_begin,
	                         uint64_t window_end, uint64_t tick, bool snapshot = false);
	// Reads served from a download the GPU overwrote afterwards (KYTY_ASYNC_READ_SNAPSHOT).
	uint64_t               m_snapshot_reads = 0;
	void                   CompletePendingWriteReadback(size_t index);
	void                   CompletePendingWriteReadbacks(uint64_t begin, uint64_t end);
	[[nodiscard]] bool     OverlapsPendingWriteReadback(uint64_t begin, uint64_t end) const;

	// KYTY_BDA_VERIFY (research): compares the GPU's BDA page table with the host's expectation.
	struct BdaVerifyRun {
		uint64_t          guest_address   = 0;
		uint64_t          table_offset    = 0;
		uint64_t          download_offset = 0;
		vk::DeviceAddress first_address   = 0;
		uint64_t          pages           = 0;
	};
	void VerifyBdaPageTable();
	void AuditBuffers();
	// Writes the BDA entries of `pages` pages from packed page `first_page`: consecutive
	// addresses from `first_address`, or zeros (0), allocating and releasing chunks.
	void WriteBdaEntries(uint64_t first_page, uint64_t pages, vk::DeviceAddress first_address);
	std::vector<uint32_t> m_bda_directory;   // per directory entry: chunk slot + 1, 0 = none
	std::vector<uint32_t> m_bda_chunk_live;  // per chunk slot: pages with an address
	std::vector<uint32_t> m_bda_free_chunks; // chunk slots, taken from the back
	uint32_t              m_bda_chunks_peak = 0;

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	std::unique_ptr<Buffer>                            m_bda_verify_download;
	std::vector<BdaVerifyRun>                          m_bda_verify_runs;
	std::vector<vk::DeviceAddress>                     m_bda_verify_expected;
	// Guest ranges any buffer ever covered (KYTY_BDA_VERIFY checks their pages for stale entries).
	RangeSet                                           m_bda_ever_registered;
	uint64_t                                           m_bda_verify_tick = 0;
	std::chrono::steady_clock::time_point              m_bda_verify_last {};
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	std::vector<PendingWriteReadback>                  m_pending_write_readbacks;
	uint64_t                                           m_direct_readbacks = 0;
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	StreamBuffer                                       m_stream_buffer;
	StreamBuffer                                       m_download_buffer;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	uint64_t                                           m_total_used_memory = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// Buffers CreateBuffer made so far (KYTY_BUFFER_STATS).
	uint64_t               m_buffers_created    = 0;
	uint64_t m_gc_tick            = 0;
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
