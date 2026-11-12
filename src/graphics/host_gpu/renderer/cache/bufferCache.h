#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/cache/bufferCachePages.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/pagedRangeSet.h"
#include "graphics/host_gpu/renderer/cache/readbackQueue.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/writeTicks.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
#include <string>
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
	static_assert(CACHING_PAGEBITS == BufferCachePages::kPageBits &&
	              CACHING_NUMPAGES == BufferCachePages::kNumPages &&
	              LOWER_ADDRESS_SIZE == BufferCachePages::kLowerAddressSize &&
	              LibKernel::Memory::kExtendedMemoryBase == BufferCachePages::kExtendedMemoryBase,
	              "bufferCachePages.h is out of date");
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

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
	// The guest unmaps this range (after InvalidateMemory wrote back what it could): GPU-written
	// bytes still marked in it are never downloaded, there being no memory to put them in.
	void                   ForgetUnmapped(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// GPU thread, between commands (round 80): while one lives, its ReadMemory calls copy bytes
	// whose GPU writers are done on the readback queue (FastReadback), as a game thread's do,
	// instead of waiting for everything queued on the GPU. Only where no command's bindings are
	// being prepared (their writes are not recorded yet): a flip's read of the display buffer's
	// compression metadata, where the check on the GPU cannot be made (TextureCache::
	// MaterializeColorClear). KYTY_BISECT's flipcheck:old waits as before.
	class GpuThreadFastReads final {
	public:
		explicit GpuThreadFastReads(BufferCache& cache) noexcept
		    : m_cache(cache), m_previous(cache.m_gpu_thread_fast_reads) {
			cache.m_gpu_thread_fast_reads = true;
		}
		~GpuThreadFastReads() { m_cache.m_gpu_thread_fast_reads = m_previous; }
		GpuThreadFastReads(const GpuThreadFastReads&)            = delete;
		GpuThreadFastReads& operator=(const GpuThreadFastReads&) = delete;

	private:
		BufferCache& m_cache;
		bool         m_previous;
	};
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
			case MemoryUsage::Stream: return ActiveStream();
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// The ring per-draw copies go to: in device memory, or for the step stream:host the one in
	// cached host memory (round 92).
	[[nodiscard]] StreamBuffer& ActiveStream() noexcept {
		return m_stream_other != nullptr && Bisect::StreamHost() ? *m_stream_other
		                                                         : m_stream_buffer;
	}
	// The tiler's: the usual ring whatever step runs (one taken at its start would stay).
	[[nodiscard]] StreamBuffer& GetStreamRing() noexcept { return m_stream_buffer; }
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// The guest bytes of a range the GPU did not write, staged for an image upload whatever
	// buffer covers them (ObtainBufferForImage copies from such a buffer instead).
	[[nodiscard]] std::pair<Buffer*, uint64_t> StageGuestBytesForImage(uint64_t vaddr,
	                                                                   uint64_t size);
	// GPU write bitmap (KYTY_WRITE_BITMAP=1, rework.md Phase 2 first step): one bit per 4 KiB
	// guest page below 1 TiB, set by shaders through bindings whose writes the host cannot bound.
	// Its device address, or 0 when it is off or unavailable.
	[[nodiscard]] vk::DeviceAddress WriteBitmapAddress();
	// ObtainBuffer for such a written binding: its pages become bitmap-tracked, so a guest read
	// of a page the GPU did not write needs no download (ReadMemoryStep).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferTracked(uint64_t vaddr, uint64_t size,
	                                                               BufferId id);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// GPU thread: a GPU fill of a range with holes in its mappings (a compute shader's fill of
	// sparse memory) sets the mapped `spans` of it on the GPU, as the hardware does, and leaves the
	// holes, where the hardware drops the writes.
	void FillMappedSpans(std::span<const std::pair<uint64_t, uint64_t>> spans, uint32_t value);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// GPU thread, where every GPU write marked so far has its command in a command buffer: the end
	// of a draw or a dispatch, the start of each PM4 packet, and before a fast readback. A write
	// is marked when its binding is prepared, and the scheduler can submit before its command is
	// recorded (a stream buffer that wraps): the ranges whose tick went by since take the current
	// one (FastReadback copies a range once its tick is done). `address_writes`: the command (of
	// `shader`) also stored through shader addresses, which mark no range; fast readbacks wait for
	// its tick.
	void WritesRecorded(bool address_writes = false, uint64_t shader = 0);
	// GPU thread, for the render targets' fast clears decided on the GPU (TextureCache::
	// MaterializeColorClear): such a check rewrites the color metadata it reads where the GPU
	// holds it, so what it found holds until something writes those bytes again. WatchMetadata
	// starts watching [vaddr, vaddr + size) as written by nothing since; MetadataUnwritten says
	// whether it is still watched as such. A write is any GPU write marked (a fill, a copy, a
	// shader's store, another check, a command storing through shader addresses, which marks all
	// of them), a CPU store to the bytes while GPU-written (done in place, or after a download), an
	// upload of CPU-written bytes into it, an image's download into the cached buffer over it, and
	// an unmap. (CPU-dirty bytes are the caller's to look for: IsRegionCpuModified.)
	void               WatchMetadata(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool MetadataUnwritten(uint64_t vaddr, uint64_t size) const;
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// The same, under the dirty-range lock: for a GPU-read delegate (Memory::SetGpuReadDelegate).
	[[nodiscard]] bool              HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	// Round 152, GPU thread, without the dirty-range lock: the fast readbacks' copies of bytes in
	// [vaddr, vaddr + size) are put in guest memory now (HasGpuDirtyBytes takes them as
	// GPU-written until then), for a read of guest memory that cannot wait for a fault's path.
	void FinishFastReadbacks(uint64_t vaddr, uint64_t size);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A CPU write fault's whole block (RenderContext::HandleFault): its pages become CPU-dirty and
	// writable at once, unless one of them holds GPU-written bytes (false: nothing changes, and the
	// fault takes its own page). The known fills over it are forgotten, as the CPU's later stores
	// there do not fault.
	[[nodiscard]] bool ReleaseBlockForCpuWrite(uint64_t block, uint64_t size);
	// Hot pages (MemoryTracker::MarkHot, RenderContext::HandleFault): the pages of the range that
	// hold no GPU-written bytes stay writable, and are uploaded again after each RedirtyHotPages
	// whether the CPU wrote them or not; how many became hot. The known fills over them are
	// forgotten, as the CPU's later stores there do not fault.
	size_t               MarkHotPages(uint64_t vaddr, uint64_t size);
	size_t               RedirtyHotPages() { return m_memory_tracker.RedirtyHot(); }
	// A store to guest memory that does not go through a fault (the GPU thread's own): the hot
	// pages of the range become CPU-dirty, as their write fault would have made them.
	void DirtyHotPagesIn(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.DirtyHotIn(vaddr, size);
	}
	void                 ClearHotPages() { m_memory_tracker.ClearHot(); }
	[[nodiscard]] size_t HotPages() const noexcept { return m_memory_tracker.HotPages(); }
	// Whether a shader has stored through shader addresses: those stores mark no range, so no
	// block can be proven free of GPU-written bytes.
	[[nodiscard]] bool AddressWritesSeen() const noexcept {
		return m_address_writes_seen.load(std::memory_order_relaxed);
	}
	// GPU thread: a command that stores through shader addresses may still be on the GPU, so guest
	// memory may not hold its stores yet (they mark no range to download).
	[[nodiscard]] bool              AddressWritesInFlight();
	void                            ProcessFaultBuffer();
	void                            NoteDmaShader(uint64_t hash, std::string label) {
        m_fault_manager.NoteDmaShader(hash, std::move(label));
	}
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	// Round 132: also on CPU-clean pages watched for the CPU's writes (read-only), which a cached
	// buffer mirrors: both copies are written and the pages stay watched. Before, the store made
	// them CPU-dirty and writable, and the next sync uploaded them and made them read-only again
	// (about 330 times a frame in a race, mostly labels). KYTY_STORE_CLEAN=0 or F8's
	// storeclean:off: not.
	// `guest_store`: a WRITE_DATA of the guest's command stream, which with the guest's sync
	// points waits for no GPU work before it (copies:lean, round 167). Labels, the CPU's stores
	// and the emulator's own wait for it.
	[[nodiscard]] bool WriteClean(uint64_t vaddr, const void* data, uint64_t size,
	                              bool guest_store = false);
	// Round 132: told of each WriteClean store to CPU-clean watched pages (RenderContext: the
	// bindless table's watches, which a CPU write fault tells otherwise).
	void SetCleanStoreHook(void (*hook)(void* user, uint64_t vaddr, uint64_t size), void* user) {
		m_clean_store_hook = hook;
		m_clean_store_user = user;
	}
	// Round 111 (Bisect::AsyncMemorySync): returns once the page protections and upload reads
	// left to the page manager's worker are done. The GPU thread calls it before its own stores
	// to guest memory: a game thread that sees one must fault on what the GPU writes, and the
	// uploads recorded before it must not read what follows it.
	// Before a GPU-thread store to guest memory, which holds no lock here: a batch of deferred
	// work the worker has not started is done right here (round 156).
	void FinishDeferredMemoryWork() {
		m_memory_tracker.Pages().FinishDeferred(!Bisect::Cuts156Off());
	}
	// The dword a GPU fill (FillBuffer) left in all of [vaddr, vaddr + size), when no other GPU
	// write and no CPU store touched those bytes since: a reader that only needs that value can
	// skip downloading the range, which drains the GPU.
	[[nodiscard]] bool              KnownFill(uint64_t vaddr, uint64_t size, uint32_t& value) const;
	// GPU thread, for an image made over [vaddr, vaddr + size) (TextureCache::
	// ClearsTargetsFromFills): the dword of the fill that holds all of it, when the bytes of it
	// the fill wrote on the GPU are all still GPU-written (a collected buffer or a CPU-path write
	// leaves them not); the bytes that were not mapped then are taken as the fill. `take` removes
	// the range from the fill, which keeps the rest of it for other images.
	[[nodiscard]] bool FillOfNewImage(uint64_t vaddr, uint64_t size, uint32_t& value, bool take);
	// GPU thread: a GPU fill set [vaddr, vaddr + size) to `value` (FillBuffer, or a compute
	// shader's fill once its dispatch is recorded); the first `gpu_bytes` of it were written on
	// the GPU (the rest was not mapped then). KnownFill answers for it until another GPU write or
	// a CPU store seen here touches it. `merge`: a fill that continues one of the same value
	// joins it (a range filled in pieces), when both were written whole on the GPU.
	void NoteFill(uint64_t vaddr, uint64_t size, uint32_t value, uint64_t gpu_bytes, bool merge);
	// GPU thread: a write over [vaddr, vaddr + size) the other paths do not see (an image's GPU
	// write, a CPU-path fill or copy, WriteClean), newer than the fills there.
	void ForgetFillsUnder(uint64_t vaddr, uint64_t size);
	// Diagnostics: names the GPU writes that mark bytes of [vaddr, vaddr + size) GPU-written from
	// now on, as KYTY_WATCH_GPU_WRITE does (a few ranges, the first write per range and shader).
	void                            WatchGpuWrites(uint64_t vaddr, uint64_t size, const char* label);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	void                            SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// GPU thread. A shader that reads guest memory through its page table (DMA) reads a page no
	// cached buffer covers as zeros, and only records the fault; a dispatch that writes images is
	// not run again after one (DispatchBufferRecovery). The guest-mapped pages of
	// [vaddr, vaddr + size) without a cached buffer get one, filled from guest memory. Returns how
	// many pages had none.
	uint32_t MakeResident(uint64_t vaddr, uint64_t size);
	// Diagnostics (the frame graph's verification, FrameGraph::ClaimVerify): a range a shader
	// reads that the GPU did not write, synchronized as a binding would synchronize it, against
	// its guest bytes in every cached buffer over it ("FG <op> stale ..." when they differ).
	void VerifyWithGuest(uint64_t vaddr, uint64_t size, uint32_t op, const std::string& what);
	// Diagnostics (KYTY_TRACE_DISPATCHES): up to 4 KB of a range as the cached buffer over it
	// holds it once the commands recorded so far have run, logged as "<prefix> ..." when the copy
	// lands: its words (the first 64), and how many of them are not finite as floats.
	void TraceContents(uint64_t vaddr, uint64_t size, const std::string& prefix);
	// Diagnostics, GPU thread (drains the GPU): the whole 16 KB pages of [vaddr, vaddr + size), at
	// most 64 MB, as the cached buffers over them hold them, against guest memory, before anything
	// is synchronized; logged as "<label> ..." by runs of pages in the same state: the tracker's
	// (CPU-dirty, GPU-written, hot, pending for the BDA walk), whether `mapped` holds the page,
	// the buffer over it and whether the page table and the BDA page table name that buffer, and
	// whether the GPU's bytes are guest memory's (or all zero). With `heal`, the pages whose
	// bytes differ while the tracker holds no GPU-written byte of them are uploaded again from
	// guest memory. Then the latest creations and collections of the buffers over the range.
	// Returns how many pages differ that way. `quiet`: only the runs of pages that differ (or have
	// no cached buffer, or no readable guest memory) are logged, and the history only when some
	// differ.
	uint64_t CompareWithGuest(uint64_t vaddr, uint64_t size, const std::string& label, bool heal,
	                          const std::function<bool(uint64_t, uint64_t)>& mapped,
	                          bool                                           quiet = false);
	// Diagnostics, GPU thread (drains the GPU; a second or more): every cached buffer as the GPU
	// holds it against guest memory ("Buffer verify:"). Stale pages: those whose bytes differ
	// while the tracker holds them clean (no 4 KB page of them CPU-dirty, GPU-written or hot),
	// which no synchronization corrects, or while they lie in no tracker region, which only a
	// binding's synchronization uploads (the BDA walk visits regions that exist). Lost pages:
	// CPU-dirty ones inside `mapped` in tracker regions the BDA walk is not pending for, which it
	// does not upload. With `heal`, the stale pages are uploaded again and the lost ones made
	// pending. Returns the stale pages.
	uint64_t VerifyCacheWithGuest(bool heal, const std::function<bool(uint64_t, uint64_t)>& mapped);
	// Memory mapped again (RenderContext::MapMemory, KYTY_MAP_REDIRTY): its clean pages become
	// CPU-dirty, and its tracker regions with CPU-dirty pages pending for the next BDA walk
	// (MemoryTracker::RedirtyForMap). Any thread. How many pages became CPU-dirty.
	size_t RedirtyMapped(uint64_t vaddr, uint64_t size);
	// Memory mapped again (RenderContext::MapMemory): its tracker regions are pending for the next
	// BDA walk, and the walk runs. Any thread, lock-free.
	void MarkMappedPending(uint64_t vaddr, uint64_t size) noexcept;
	// Whether a buffer registered since the last call created tracker regions (ChangeRegister),
	// which only a BDA walk uploads: RenderContext::PrepareBda then walks again within a
	// submission. GPU thread.
	[[nodiscard]] bool TakeNewRegions() noexcept { return std::exchange(m_new_regions, false); }
	// KYTY_TABLE's bindings table (tableBindings.h): moves whenever a cached buffer is registered
	// or unregistered (ChangeRegister), so a BufferId and handle kept since it stays name the same
	// buffer over the same range. GPU thread.
	[[nodiscard]] uint64_t BufferSetGeneration() const noexcept { return m_buffer_set_generation; }
	// KYTY_TABLE: whether a read-only, non-texel binding that ObtainBuffer gave earlier (buffer
	// `id`, its `handle`, guest range [vaddr, vaddr + size)) can be used again as it was: the
	// buffer is alive with that handle and covers the range, and no page of it is CPU-dirty, so
	// SynchronizeBuffer would copy nothing and the stream path would not be taken. It is then
	// touched and its last use is this tick, as ObtainBuffer does. GPU thread.
	[[nodiscard]] bool ReuseReadBinding(BufferId id, vk::Buffer handle, uint64_t vaddr,
	                                    uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	// The BDA walk (RenderContext::PrepareBda): the cached buffers over the CPU-dirty pages of the
	// tracker regions that are pending since the last walk (MemoryTracker::TakeBdaPending), in
	// their parts inside `mapped`. What a walk of every buffer in every region with a CPU-dirty
	// page synchronizes, without visiting the buffers of regions where nothing happened.
	// Returns the regions visited.
	uint64_t                        SynchronizeBdaPending(const RangeSet& mapped);
	void                            RunGarbageCollector();
	// Diagnostics, GPU thread: makes guest memory current for [vaddr, vaddr + size) by
	// downloading what the GPU wrote there (drains the GPU).
	void                            DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size);
	// The same for many ranges (address, size), with one wait.
	void DownloadRangesForDiagnostics(std::span<const std::pair<uint64_t, uint64_t>> ranges);
	// Diagnostics, GPU thread: waits until the GPU is idle and every download it queued (images
	// read back for the CPU too) has landed in guest memory.
	void WaitForDiagnostics();
	// Round 105, diagnostics (an F9 capture), GPU thread: the GPU's copy of [vaddr, vaddr + size)
	// as it stands at this point of the command stream (before the dispatch being prepared runs),
	// copied into a buffer of its own; guest memory as it is now when no cached buffer holds all
	// of it. TakeDiagnosticSnapshots, after the frame, waits for the GPU and returns them.
	struct DiagnosticSnapshot {
		uint64_t              address  = 0;
		uint32_t              op       = 0;
		bool                  from_gpu = false;
		// Round 106: taken after the dispatch (what it wrote), not before it.
		bool                  after = false;
		std::string           label;
		std::vector<uint32_t> words;
	};
	void SnapshotForDiagnostics(uint64_t vaddr, uint64_t size, uint32_t op, std::string label,
	                            bool after_dispatch = false);
	[[nodiscard]] std::vector<DiagnosticSnapshot> TakeDiagnosticSnapshots();

	// Diagnostics: the guest shader whose bindings are being prepared on this thread, if any.
	inline static thread_local uint64_t s_diag_shader_hash = 0;
	// Diagnostics: the last writer of a GPU-written range (with the tick it was recorded in), and
	// game-thread downloads by writer: how long after the write, and how often the GPU had already
	// run it (`done_tick`: the last tick known done), which is what readahead can save.
	static void RecordGpuWriter(uint64_t vaddr, uint64_t size, uint64_t shader, uint64_t tick);
	static void ReportGameThreadDownload(uint64_t vaddr, uint64_t bytes, uint64_t done_tick,
	                                     uint64_t now_ns);

	// Readahead (KYTY_READAHEAD=1, or KYTY_BISECT's readahead): the ranges game threads had to
	// download (ReadMemoryStep) are learned in 64 KB blocks, and when the GPU writes into them
	// again their download is recorded at the end of that command buffer (the scheduler's
	// before-flush hook). The game thread's read that follows then finds the bytes in guest memory
	// and only lifts the page's protection, instead of waiting for the GPU to catch up and
	// download them (about a millisecond each, 30-40 times a frame in GT7's races). A block game
	// threads stop reading ages out after 4 to 8 seconds.
	[[nodiscard]] static bool ReadaheadEnabled();
	// Diagnostics: the shaders (hex, "copy" for copies, fills and label writes) whose last
	// recorded writes overlap [vaddr, vaddr + size), at most four; empty when none.
	[[nodiscard]] static std::string GpuWritersOf(uint64_t vaddr, uint64_t size);

private:
	friend struct BufferCacheTestAccess;

	void ReleaseBuffer(BufferId id);
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
	// `guest_store`: a WRITE_DATA of the guest's command stream (WriteClean), which with the
	// guest's sync points waits for no GPU work before it (copies:lean, round 167).
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size,
	                     bool guest_store = false);
	// Records bytes a binding makes GPU-written (after SynchronizeBuffer marked their pages).
	void MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked = false);
	// ReadMemoryStep: drops the pages of the readback window the bitmap shows the GPU did not
	// write. True with wait_tick set when an asynchronous caller must first wait for that tick.
	bool RefineWithWriteBitmap(const Buffer& buffer, uint64_t vaddr, uint64_t size, bool async,
	                           uint64_t& wait_tick);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	// Diagnostics (CompareWithGuest): the latest creations of cached buffers, and their deletions
	// as joined into a newer one or collected, GPU thread only.
	enum class BufferEventKind : uint8_t { Created, Joined, Collected };
	struct BufferEvent {
		uint64_t        frame        = 0;
		uint64_t        address      = 0;
		uint64_t        size         = 0;
		uint64_t        request      = 0; // created: the range asked for
		uint64_t        request_size = 0;
		uint32_t        absorbed     = 0; // created: the buffers it took over
		BufferEventKind kind         = BufferEventKind::Created;
	};
	void NoteBufferEvent(BufferEventKind kind, const Buffer& buffer, uint64_t request = 0,
	                     uint64_t request_size = 0, uint32_t absorbed = 0);
	std::vector<BufferEvent> m_buffer_events;
	size_t                   m_buffer_events_next  = 0;
	uint64_t                 m_buffer_events_total = 0;
	// The creation of every live buffer, by its address.
	std::map<uint64_t, BufferEvent> m_buffer_origins;
	bool                            m_new_regions = false; // TakeNewRegions
	uint64_t                        m_buffer_set_generation = 1; // BufferSetGeneration
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	// Records the copies of an upload UploadCopies staged from `source` into `buffer`, with their
	// barriers (and the metadata watches' notes of the bytes).
	void RecordUpload(Buffer& buffer, std::span<const vk::BufferCopy> copies, uint64_t total_size,
	                  vk::Buffer source);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// One readback attempt on the GPU thread: 0 when done, else the tick to wait for.
	// `writer_wait`: the caller may wait for an unfinished writer's command buffer instead of a
	// download (readwait:writer, once a read).
	[[nodiscard]] uint64_t ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
	                                      bool from_gpu_thread, bool async, bool writer_wait);
	// A game thread's read (ReadMemoryStep, GPU thread) of [window_begin, window_end): the
	// GPU-written bytes of the window from ticks the GPU has finished are copied on the readback
	// queue. With `wait_here` they are waited for there, then put in guest memory, and the
	// window's pages that hold no GPU-written bytes any more are lifted; without it (round 152),
	// the copy is left to the reading thread (m_step_ticket), whose next step lifts the pages.
	// Bytes of an unfinished tick, or that a download or an image write-back on its way also holds,
	// stay for a later read. False when the faulting pages hold such bytes, a command storing
	// through shader addresses is unfinished, or nothing could be copied (the old download then
	// runs).
	[[nodiscard]] bool FastReadback(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                uint64_t window_begin, uint64_t window_end, bool is_write,
	                                bool wait_here);
	// Round 152, any thread: the fast readback's copy `ticket` is in guest memory when this returns
	// (waited for and put there by this thread, or by the one that got to it first). Who, for the
	// counts: the reading thread, the GPU thread for its own read (or with KYTY_FAST_READBACK=gpu),
	// the GPU thread for another thread's read, or the priority thread.
	enum class FastCompleter : uint8_t { Reader, Inline, GpuThread, Priority };
	void CompleteFastReadback(ReadbackQueue::Ticket ticket, FastCompleter who);
	// The pages of the readback window around vaddr that hold no GPU-written bytes, nor bytes
	// still being downloaded, lose their GPU-written protection (a run of pages at a time).
	void LiftCurrentPages(const Buffer& buffer, uint64_t vaddr);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	// `downloaded`, when given, is increased by the bytes copied.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                        uint64_t* downloaded = nullptr);
	template <bool async>
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                        uint64_t* downloaded = nullptr);
	// Readahead (ReadaheadEnabled), GPU thread: a game thread had to download [begin, end), or
	// found bytes of it already downloaded (`refresh`: only keeps a block it learned before);
	// the GPU marks [vaddr, vaddr + size) written; the scheduler is about to submit.
	void        LearnReadahead(uint64_t begin, uint64_t end, bool refresh);
	void        QueueReadahead(uint64_t vaddr, uint64_t size);
	void        IssueReadahead();
	static void IssueReadaheadHook(void* cache);
	// Every 4 s: the learned blocks become the older generation (and a summary is logged).
	void        RotateReadahead();
	template <bool async>
	void DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                          uint64_t total_size);

	struct PendingSnapshot {
		DiagnosticSnapshot      snapshot;
		uint64_t                size = 0;
		std::unique_ptr<Buffer> copy;
	};
	std::vector<PendingSnapshot> m_diagnostic_snapshots;

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	// With a page bitmap (round 80): HasGpuDirtyBytes answers most "no"s without the tree.
	PagedRangeSet m_gpu_modified_ranges;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Every in-flight download's ranges, one entry each: m_downloading_ranges is their union. A
	// publication removes only its own entries (ISSUES #26: the first of two downloads of one
	// window cleared the range while the second was in flight; a game thread then took the
	// page as current, wrote it, and the second publication put older GPU bytes over that).
	std::vector<std::pair<uint64_t, uint64_t>>         m_downloading_list;
	// Round 152: the bytes of fast readbacks' copies not yet in guest memory (they are in
	// m_downloading_ranges too), for HasGpuDirtyBytesShared; the GPU thread asks the readback
	// queue instead (HasGpuDirtyBytes), without the lock.
	RangeSet m_fast_pending_ranges;
	// Write bitmap (WriteBitmapAddress): the buffer, the whole pages whose GPU writes all came
	// through tracked bindings, and per 4 MiB region the last tick a tracked binding wrote it.
	// Guarded by m_dirty_ranges_mutex, as m_gpu_modified_ranges.
	std::unique_ptr<Buffer>                            m_write_bitmap;
	bool                                               m_write_bitmap_failed = false;
	RangeSet                                           m_bitmap_tracked;
	std::unordered_map<uint64_t, uint64_t>             m_bitmap_ticks;
	uint64_t                                           m_bitmap_dropped_pages = 0;
	uint64_t                                           m_bitmap_written_pages = 0;
	// The tick of the latest asynchronous readback (KYTY_ASYNC_READBACK), or readahead download;
	// GPU thread only.
	uint64_t                                           m_last_async_download_tick = 0;
	// Round 153: each asynchronous download's publication tick and the bounds of its bytes (one a
	// DownloadBufferCopies), under m_dirty_ranges_mutex, so a read of pages a download is still
	// bringing in waits for the downloads that cover them; and the batches' ids (GPU thread).
	struct DownloadBatch {
		uint64_t id    = 0;
		uint64_t tick  = 0;
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	std::vector<DownloadBatch> m_download_batches;
	uint64_t                   m_download_batch_ids = 0;
	// The tick of the last GPU write recorded to each range (MarkGpuWritten), under
	// m_dirty_ranges_mutex; and the fast readback's queue (FastReadback).
	WriteTicks                     m_write_ticks;
	std::unique_ptr<ReadbackQueue> m_readback_queue;
	// The writes marked since the last WritesRecorded(), with the tick they were marked in (GPU
	// thread, under m_dirty_ranges_mutex); and the last tick a command that stores through shader
	// addresses went into (GPU thread).
	struct UnrecordedWrite {
		uint64_t begin   = 0;
		uint64_t end     = 0;
		uint64_t tick    = 0;
		bool     tracked = false; // through a write-bitmap binding (m_bitmap_ticks)
	};
	std::vector<UnrecordedWrite> m_unrecorded_writes;
	uint64_t                     m_address_write_tick = 0;
	std::atomic<bool>            m_address_writes_seen {false};
	// The metadata ranges WatchMetadata watches, by first byte: whether something wrote any of
	// their bytes since. Under m_dirty_ranges_mutex; m_metadata_watch_count mirrors their number
	// for the paths that look without the lock.
	struct MetadataWatch {
		uint64_t end     = 0;
		bool     written = false;
	};
	std::map<uint64_t, MetadataWatch> m_metadata_watches;
	uint64_t                          m_metadata_watch_span = 0; // the longest watched range
	std::atomic<size_t>               m_metadata_watch_count {0};
	// Marks the watches over [vaddr, vaddr + size) written; m_dirty_ranges_mutex held.
	void NoteMetadataWrite(uint64_t vaddr, uint64_t size);
	// Set by a fast readback that let the access go on (GPU thread), for ReadMemory's counts.
	bool m_fast_step = false;
	// Round 135: what a ReadMemoryStep did, read back by ReadMemory's command on the GPU thread
	// (the counts by instruction and the wait to make), and why the last FastReadback copied
	// nothing (with the tick of the faulting pages' writer when the GPU was still running it).
	enum class ReadOutcome : uint8_t {
		None,
		Direct,     // nothing to do (a write to memory no buffer covers)
		Bitmap,     // the write bitmap's wait (RefineWithWriteBitmap)
		Lifted,     // the pages held no GPU-written bytes: lifted
		LiftWaited, // the same, once a download of them on its way lands
		Clean,      // the pages were current already: another thread's read lifted them
		Fast,       // copied on the readback queue (FastReadback)
		WaitWriter, // waits for its writer's command buffer, then asks again (readwait:writer)
		Download,   // downloaded behind the renderer's queue
		Nothing,    // went to download, and nothing was left to download
		FastWait,   // waited for another read's copy of its pages to land, then asks again
		Count
	};
	enum class FastRefusal : uint8_t {
		None,
		Unavailable, // no readback queue, or turned off
		Downloading, // a download or an image write-back of the faulting pages on its way
		Address,     // a command storing through shader addresses is still on the GPU
		Recording,   // the faulting pages' writer is in the command buffer being recorded
		InFlight,    // ... in one submitted and not finished
		Other,       // outside the buffer, the write bitmap's, or the copy could not be made
		Nothing,     // no GPU-written bytes it could copy in the window
		TooBig,
		Count
	};
	ReadOutcome m_step_outcome     = ReadOutcome::None;
	FastRefusal m_fast_refusal     = FastRefusal::None;
	uint64_t    m_fast_writer_tick = 0; // InFlight with no other refusal on the faulting pages
	uint32_t    m_fast_left        = 0; // Nothing: ranges elsewhere in the window it left
	// The ticks around the last refusal: the writer's (the latest of the faulting pages'), the one
	// being recorded and the last one known done, for the counts (GPU thread).
	uint64_t m_fast_refused_tick = 0;
	uint64_t m_fast_current_tick = 0;
	uint64_t m_fast_known_tick   = 0;
	// The bytes the last ReadMemoryStep recorded a download of (GPU thread).
	uint64_t m_step_bytes = 0;
	// Round 152: the fast readback copy the reading thread waits for, then asks again (GPU
	// thread, read back by ReadMemory's command).
	ReadbackQueue::Ticket m_step_ticket;
	// GpuThreadFastReads is alive (GPU thread only).
	bool m_gpu_thread_fast_reads = false;
	// Readahead (ReadaheadEnabled), GPU thread only: the blocks learned in this generation and
	// the one before, the GPU-written pieces of them waiting for the next flush, and what it did.
	static constexpr uint64_t                          ReadaheadBlock      = 64 * 1024;
	static constexpr uint64_t                          ReadaheadMaxLearned = 512ull << 20u;
	static constexpr size_t                            ReadaheadMaxQueue   = 4096;
	RangeSet                                           m_readahead_current;
	RangeSet                                           m_readahead_previous;
	uint64_t                                           m_readahead_learned_bytes   = 0;
	uint64_t                                           m_readahead_generation_ns   = 0;
	std::vector<std::pair<uint64_t, uint64_t>>         m_readahead_queue;
	uint64_t                                           m_readahead_downloads       = 0;
	uint64_t                                           m_readahead_bytes           = 0;
	uint64_t                                           m_readahead_found           = 0;
	uint64_t                                           m_readahead_missed          = 0;
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	// GPU fills whose bytes nothing has written since (KnownFill); under m_dirty_ranges_mutex.
	struct KnownFillRecord {
		uint64_t address   = 0;
		uint64_t size      = 0;
		uint32_t value     = 0;
		uint64_t gpu_bytes = 0; // [address, address + gpu_bytes) written on the GPU
	};
	std::vector<KnownFillRecord>                       m_known_fills;
	// m_known_fills.size(), read without the mutex.
	std::atomic<size_t> m_known_fill_count {0};
	void ForgetFills(uint64_t vaddr, uint64_t size); // m_dirty_ranges_mutex held
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	// Guest reads UploadCopies leaves to the recording thread: staging destination, guest
	// address, bytes.
	struct UploadRead {
		uint8_t* destination;
		uint64_t address;
		uint64_t size;
	};
	std::vector<UploadRead> m_upload_reads;
	StreamBuffer                                       m_stream_buffer;
	// For the step stream:host (made only when a step has it).
	std::unique_ptr<StreamBuffer>                      m_stream_other;
	StreamBuffer                                       m_download_buffer;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	void (*m_clean_store_hook)(void* user, uint64_t vaddr, uint64_t size)   = nullptr;
	void*                                              m_clean_store_user   = nullptr;
	uint64_t                                           m_total_used_memory = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// As configured; the ones above come down while memory is short (as in the texture cache).
	uint64_t m_base_trigger_gc_memory     = 0;
	uint64_t m_base_critical_gc_memory    = 0;
	uint64_t m_applied_trigger_gc_memory  = UINT64_MAX; // as in the texture cache
	uint64_t m_applied_critical_gc_memory = UINT64_MAX;
	uint64_t m_gc_tick            = 0;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
