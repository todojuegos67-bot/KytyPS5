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
#include "speculation-state.h"

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace ReadbackQueue { class Queue; }

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
	[[nodiscard]] bool TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] Buffer* GetBufferIfLive(BufferId id) {
		auto* buffer = m_slot_buffers.try_get(id);
		return buffer != nullptr && !buffer->is_deleted ? buffer : nullptr;
	}
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// FindBuffer's registered buffer over the range, none registered: the null id (a speculative translation's
	// question whether ObtainBuffer gave the range's buffer).
	[[nodiscard]] BufferId RegisteredBuffer(uint64_t vaddr, uint64_t size) {
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		return owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size) ? *owner : BufferId {};
	}
	void                   EnsureBufferContents(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
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
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// Guest ranges staged one after another in one staging allocation of `total` bytes, each
	// at its `offset`. Returns nullptr (nothing staged) when a range starts in a cached buffer
	// or holds GPU-written pages: the caller then uploads the whole image instead.
	struct StagingPiece {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t offset = 0;
	};
	[[nodiscard]] std::pair<Buffer*, uint64_t> StageImagePieces(const std::vector<StagingPiece>& pieces,
	                                                            uint64_t total);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// CopyBuffer's copy on the CPU: only the destination pages whose bytes differ are written.
	void CopyGuestMemory(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size);
	// Called after an identified small GPU copy; it never submits or publishes guest data.
	void ScheduleCopyFeedback(uint64_t vaddr, uint64_t size);
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
		return m_memory_tracker.IsRegionCpuModified(vaddr, size);
	}
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	struct SyncRegionRequest {
		uint64_t address = 0, size = 0, cpu_epoch = 0, registration_epoch = 0;
	};
	void SynchronizeRegionRequest(SyncRegionRequest& request);
	[[nodiscard]] uint64_t RegistrationEpoch() const { return m_registration_epoch.load(std::memory_order_acquire); }
	// A buffer registered or unregistered in the tracker regions of the range since `epoch`.
	[[nodiscard]] bool RegisteredSince(uint64_t vaddr, uint64_t size, uint64_t epoch) const {
		for (auto region = vaddr / TRACKER_REGION_SIZE, last = (vaddr + size - 1) / TRACKER_REGION_SIZE; region <= last; ++region)
			if (RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) > epoch) return true;
		return false;
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
	// The mapped parts of registered buffers in [begin, end), adjacent ones merged.
	void CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
	                                   std::vector<RangeSet::Range>& ranges) const;
	// The spans of the buffers registered or unregistered since the last call, appended to `spans`; false when
	// more changed than it keeps (then anything may have).
	[[nodiscard]] bool TakeRegistrationSpans(std::vector<GuestRange>& spans);

	void               RunGarbageCollector(bool collect = true);
	void               UpdateGcThresholds();
	[[nodiscard]] uint64_t CacheBytes() const noexcept { return m_cache_bytes; }

private:
	friend struct BufferCacheTestAccess;

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
	void TouchBuffer(const Buffer& buffer);
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
	bool TryReportLodStatsOnGpu(uint64_t address, bool reset);
	vk::Pipeline m_lod_pack_pipeline = nullptr;
	vk::PipelineLayout m_lod_pack_layout = nullptr;
	vk::DescriptorSetLayout m_lod_pack_descriptors = nullptr;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_lod_stats_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	// (Read by a speculative translation's thread: src/graphics/guest_gpu/speculation.h.)
	std::atomic<uint64_t> m_registration_epoch {1};
	// Per tracker region, see RegionRegistrationEpoch.
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_registrations =
	    std::make_unique<std::atomic<uint64_t>[]>(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE);
	std::vector<GuestRange> m_registration_spans; // see TakeRegistrationSpans
	bool                    m_registration_spans_lost = true;
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
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	std::shared_mutex                                 m_gpu_modified_mutex; // (its changes: HasGpuDirtyBytes)
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	// ObtainBufferForImage (KYTY_ASYNC_UPLOAD=2): the backing pieces of an image upload.
	std::vector<std::pair<const uint8_t*, uint64_t>>  m_backing_pieces;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_host_shader_upload;
	StreamBuffer                                      m_table_upload;
	// Buffer uploads stage here while it has room (UploadCopies): device-local host-visible memory, whose copies read
	// video memory instead of system memory across the bus.
	StreamBuffer                                      m_staging_device;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
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

	// KYTY_READBACK_QUEUE: GPU writes that may still be in flight, oldest first. An entry's
	// tick is set at the next point between commands (every write noted by then is recorded
	// at or before the current tick); entries the GPU completed are dropped from the front.
	struct GpuWrite {
		uint64_t begin, end, tick;
		bool     big = false; // spans more granules than GpuWriteSpan: not counted in m_gpu_write_granules
	};
	// The writes from m_gpu_writes_head on, per 64 KiB granule (hashed into the counters; a write over more than
	// GpuWriteSpan granules counts in m_gpu_writes_big instead): a range whose granules count none (and with no big
	// write) overlaps no in-flight write, without a walk of the log (thousands of entries a frame at 1-1).
	static constexpr uint32_t   GpuWriteGranuleBits = 16, GpuWriteCounters = 1u << 16u, GpuWriteSpan = 64;
	std::unique_ptr<uint32_t[]> m_gpu_write_granules = std::make_unique<uint32_t[]>(GpuWriteCounters);
	size_t                      m_gpu_writes_big     = 0;
	// Per counter, 1 + the number (m_gpu_writes_base + index) of the last write counted there: no later write touches
	// those granules, so the walk for a range starts at its granules' last one, not at the log's end.
	std::unique_ptr<uint64_t[]> m_gpu_write_last = std::make_unique<uint64_t[]>(GpuWriteCounters);
	uint64_t                    m_gpu_writes_base = 0; // the number of m_gpu_writes[0]
	[[nodiscard]] static uint32_t GpuWriteCounter(uint64_t granule) {
		return static_cast<uint32_t>((granule * 0x9e3779b97f4a7c15ull) >> (64u - 16u));
	}
	void CountGpuWrite(const GpuWrite& write, int32_t delta);
	void ResetGpuWrites();
	void                  NoteGpuWrite(uint64_t vaddr, uint64_t size);
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

	// Last member: destroyed (queue idle) before the buffers its copies use.
	std::unique_ptr<ReadbackQueue::Queue> m_readback_queue;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
