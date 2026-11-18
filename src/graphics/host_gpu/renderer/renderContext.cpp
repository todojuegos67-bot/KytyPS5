#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/mtGpuStageA.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/cache/bufferAddressLog.h"
#include "graphics/host_gpu/renderer/faultPages.h"
#include "graphics/host_gpu/renderer/frameGraphQueries.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

// KYTY_FAULT_BLOCK_KB (default 64; a power of two from 8 to 1024, 0 turns it off; the F8 step
// "faultblock:off" too): a CPU write fault in a block of that size that also faulted less than
// 100 ms before releases the whole aligned block from the buffer cache's write tracking, instead
// of the faulting page alone, when no byte of it is GPU-written (checked in the same lock hold as
// the release) and no image lies in it (round 79: when one does, the block's pages it does not lie
// on; ReleaseAroundImages). The game rewrites some heaps every frame a page at a time,
// and each of those pages faulted on its own (a signal, a protection change and a TLB flush on
// every processor): 40,000 faults a second at Tokyo. Its other pages are uploaded at the block's
// next use, changed or not. Not once a shader stores through shader addresses (those stores mark
// no range). After netwarex's "release 64 KiB blocks on guest write faults" (KytyPS5 PR 1297), here
// for blocks that fault again only.
uint64_t FaultBlockBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_FAULT_BLOCK_KB");
		uint64_t    kb    = 64;
		if (value != nullptr && *value != '\0') {
			kb = std::strtoull(value, nullptr, 10);
			if (kb != 0 && (kb < 8 || kb > 1024 || !std::has_single_bit(kb))) {
				kb = 64;
			}
		}
		LOGF("Write faults: %s\n",
		     kb == 0 ? "each releases its own page (KYTY_FAULT_BLOCK_KB=0)"
		             : fmt::format("a block of {} KB that faults again within 100 ms is released "
		                           "whole (KYTY_FAULT_BLOCK_KB)",
		                           kb)
		                   .c_str());
		return kb * 1024u;
	}();
	if (Bisect::FaultBlocksOff()) {
		return 0;
	}
	// Round 155, KYTY_BISECT's faultblock:<kb>: another block size, for an A/B within one run.
	if (const auto kb = Bisect::FaultBlockKB(); kb > 0) {
		return static_cast<uint64_t>(kb) * 1024u;
	}
	return bytes;
}

// Round 90: whether a WAIT_REG_MEM that passed lets the walk for memory-reading shaders run again
// in the same submission (PrepareBda). KYTY_BDA_WAIT_SYNC=0 (or the F8 step "bdasync:submission")
// walks once a submission, as before.
bool BdaSyncAtWaits() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BDA_WAIT_SYNC");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("Memory-reading shaders: %s\n",
		     on ? "the CPU's writes are synchronized again after each wait on memory that passes "
		          "(KYTY_BDA_WAIT_SYNC=0: once a submission)"
		        : "the CPU's writes are synchronized once a submission (KYTY_BDA_WAIT_SYNC=0)");
		return on;
	}();
	return enabled && !Bisect::BdaSyncSubmission();
}

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Whether the block faulted less than 100 ms before; notes this fault. A hint: threads race on
// the slots, and a block that shares one with another may be missed.
bool BlockFaultedRecently(uint64_t block_index, uint64_t now_ms) {
	constexpr uint64_t                             WindowMs = 100;
	constexpr uint64_t                             TimeBits = 24;
	constexpr uint64_t                             TimeMask = (uint64_t {1} << TimeBits) - 1;
	static std::array<std::atomic<uint64_t>, 4096> recent {};
	auto&          slot     = recent[(block_index ^ (block_index >> 12u)) % recent.size()];
	const uint64_t entry    = (block_index << TimeBits) | (now_ms & TimeMask);
	const uint64_t previous = slot.exchange(entry, std::memory_order_relaxed);
	return previous != 0 && (previous >> TimeBits) == block_index &&
	       ((now_ms - previous) & TimeMask) < WindowMs;
}

// The blocks released (and the faults that did not release one) in the last 10 s or so, logged.
std::atomic<uint64_t> g_released_blocks {0};
std::atomic<uint64_t> g_page_faults {0};
std::atomic<uint64_t> g_around_images {0};
std::atomic<uint64_t> g_fault_report_ms {0};

void NoteWriteFault(bool released, bool around_image, uint64_t now_ms) {
	(around_image ? g_around_images
	 : released   ? g_released_blocks
	              : g_page_faults)
	    .fetch_add(1, std::memory_order_relaxed);
	auto last = g_fault_report_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		g_fault_report_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed);
		return;
	}
	if (now_ms - last < 10000 ||
	    !g_fault_report_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
		return;
	}
	const auto blocks = g_released_blocks.exchange(0, std::memory_order_relaxed);
	const auto pages  = g_page_faults.exchange(0, std::memory_order_relaxed);
	const auto around = g_around_images.exchange(0, std::memory_order_relaxed);
	LOGF("Write faults, %.0f s: %" PRIu64 " released a whole block, %" PRIu64
	     " the pages of their block around an image, %" PRIu64 " their own page\n",
	     static_cast<double>(now_ms - last) / 1000.0, blocks, around, pages);
}

// Round 79: a write fault in a block that faulted again and that an image lies in releases the
// runs of the block's pages no image lies on (each run on its own, so one holding GPU-written
// bytes stays tracked), as a fault in a block without images releases the whole block. The pages
// an image lies on keep faulting one by one (the texture cache's own tracking of them is what
// those faults are for). Returns the pages released; fault_released tells whether the faulting
// page is among them.
uint64_t ReleaseAroundImages(BufferCache& cache, uint64_t block, uint64_t block_bytes,
                             uint64_t image_pages, uint64_t fault_vaddr, bool& fault_released) {
	fault_released       = false;
	const uint64_t pages = block_bytes / TRACKER_PAGE_SIZE;
	if (pages == 0 || pages > 64 || block_bytes % TRACKER_PAGE_SIZE != 0) {
		return 0;
	}
	const uint64_t all       = pages == 64 ? ~uint64_t {0} : (uint64_t {1} << pages) - 1u;
	const uint64_t imageless = ~image_pages & all;
	if (imageless == 0) {
		return 0;
	}
	const auto fault_page = (fault_vaddr - block) / TRACKER_PAGE_SIZE;
	uint64_t   released   = 0;
	for (uint64_t first = 0; first < pages;) {
		if ((imageless >> first & 1u) == 0) {
			first++;
			continue;
		}
		uint64_t last = first;
		while (last + 1 < pages && (imageless >> (last + 1) & 1u) != 0) {
			last++;
		}
		const auto begin = block + first * TRACKER_PAGE_SIZE;
		const auto bytes = (last - first + 1) * TRACKER_PAGE_SIZE;
		if (cache.ReleaseBlockForCpuWrite(begin, bytes)) {
			LibKernel::Memory::JournalWrite(begin, bytes,
			                                LibKernel::Memory::WriteSource::CpuWriteFault);
			released += last - first + 1;
			fault_released = fault_released || (fault_page >= first && fault_page <= last);
		}
		first = last + 1;
	}
	return released;
}

// Hot pages (Bisect::HotPages): whether the page (or block) faulted within 250 ms of its fault
// before, three times running; notes this fault. A hint, as BlockFaultedRecently: a key that shares
// a slot with another may be missed.
bool FaultRecurs(uint64_t key, uint64_t now_ms) {
	constexpr uint64_t                              WindowMs  = 250;
	constexpr uint64_t                              CountBits = 8;
	constexpr uint64_t                              TimeBits  = 20;
	constexpr uint64_t                              KeyBits   = 64 - CountBits - TimeBits;
	constexpr uint64_t                              TimeMask  = (uint64_t {1} << TimeBits) - 1;
	constexpr uint64_t                              CountMask = (uint64_t {1} << CountBits) - 1;
	constexpr uint64_t                              KeyMask   = (uint64_t {1} << KeyBits) - 1;
	constexpr uint64_t                              Needed    = 3;
	static std::array<std::atomic<uint64_t>, 16384> slots {};
	auto&          slot     = slots[(key ^ (key >> 14u) ^ (key >> 28u)) % slots.size()];
	const uint64_t previous = slot.load(std::memory_order_relaxed);
	const uint64_t time     = (previous >> CountBits) & TimeMask;
	uint64_t       count    = 1;
	if (previous != 0 && (previous >> (CountBits + TimeBits)) == (key & KeyMask) &&
	    ((now_ms - time) & TimeMask) < WindowMs) {
		count = std::min(CountMask, (previous & CountMask) + 1u);
	}
	slot.store(((key & KeyMask) << (CountBits + TimeBits)) | ((now_ms & TimeMask) << CountBits) |
	               count,
	           std::memory_order_relaxed);
	return count >= Needed;
}

} // namespace

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// MT GPU stage C: the GPU thread's binding flags, read by the cache (TextureCache::BindFlags).
	m_texture_cache.RegisterBindingTable(&m_render_executor.MainTranslation().images);
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
	// Round 132: a GPU-thread store the buffer cache makes on watched CPU-clean pages
	// (BufferCache::WriteClean) is a write the bindless table's watches must hear of.
	m_buffer_cache.SetCleanStoreHook(
	    [](void* user, uint64_t vaddr, uint64_t size) {
		    static_cast<BindlessTable*>(user)->OnCpuWrite(vaddr, size);
	    },
	    &m_bindless_table);
	m_texture_cache.on_image_release = [this](const Image& image) {
		if (!m_bindless_table.Enabled()) {
			return false;
		}
		std::vector<vk::ImageView> views;
		views.reserve(image.views.size());
		for (const auto& cached: image.views) {
			views.push_back(cached.view);
		}
		return m_bindless_table.ReleaseViews(views);
	};
	// vkQueueSubmit, and the queue lock that present also holds, runs on a queue thread instead
	// of the GPU thread. Frames reach the presenter only after their flip tick completed, and the
	// presenter's own scheduler stays synchronous. KYTY_ASYNC_SUBMIT=0 submits on the GPU thread.
	// KYTY_RECORD_THREAD=0: no recording thread (CommandScheduler::EnableRecordingThread), which
	// submits too.
	const char* record_thread = std::getenv("KYTY_RECORD_THREAD");
	const char* async_submit  = std::getenv("KYTY_ASYNC_SUBMIT");
	if (record_thread == nullptr || std::strcmp(record_thread, "0") != 0) {
		m_command_scheduler.EnableRecordingThread();
	} else if (async_submit == nullptr || std::strcmp(async_submit, "0") != 0) {
		m_command_scheduler.EnableAsyncSubmit();
	}
	LOGF("Recording thread: %s, async queue submit: %s\n",
	     m_command_scheduler.Threaded() ? "on" : "off",
	     m_command_scheduler.AsyncSubmit() ? "on" : "off");
	// Reads KYTY_FAULT_BLOCK_KB (and logs it) here rather than in the first fault's handler.
	(void)FaultBlockBytes();
	// Round 111: submissions wait for the page protections deferred before them.
	m_command_scheduler.SetSubmitBarrier(&m_page_manager);
}

RenderContext::~RenderContext() {
	FaultPages::Stop();
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	// Round 111: the page manager's worker is idle before the staging buffer its upload reads
	// write into goes (the buffer cache, after the page manager), and the page manager goes
	// before the scheduler that waits for it at each submission.
	m_page_manager.FinishDeferred();
	m_command_scheduler.SetSubmitBarrier(nullptr);
	// The buffers destroyed from here on may outlive the scheduler their log entries ask.
	BufferAddressLog::Stop();
	FrameGraphQueries::Shutdown();
	OcclusionQueries::Shutdown();
}

void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& scheduler);

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	WarmUpLocalMemory(m_graphics, m_command_scheduler);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		using FaultPages::Outcome;
		const auto frame        = m_graphics.presented_frames.load(std::memory_order_relaxed);
		const bool hot_pages    = Bisect::HotPages() && !m_buffer_cache.AddressWritesSeen();
		bool       texture_done = false;
		auto       outcome      = Outcome::BlocksOff;
		if (const auto block_bytes = FaultBlockBytes(); block_bytes != 0) {
			const auto now      = NowMs();
			const auto block    = Common::AlignDown(fault_vaddr, block_bytes);
			bool       released = false;
			// The pages released around an image (round 79), the faulting one among them or not.
			uint64_t around = 0;
			if (!BlockFaultedRecently(block / block_bytes, now)) {
				outcome = Outcome::NotRecent;
			} else if (m_buffer_cache.AddressWritesSeen()) {
				outcome = Outcome::AddressWrites;
			} else if (!IsMapped(block, block_bytes)) {
				outcome = Outcome::BlockUnmapped;
			} else {
				// The texture cache first: an image in the block keeps the pages it lies on to
				// their own faults (before round 79, the whole block: 88 % of a race's write
				// faults were in blocks of GT7's per-frame memory that an image lies in).
				texture_done         = true;
				uint64_t image_pages = 0;
				if (m_texture_cache.InvalidateMemoryNear(fault_vaddr, fault_size, block,
				                                         block_bytes, &image_pages)) {
					outcome = Outcome::ImageInBlock;
					if (!Bisect::FaultBlockAroundImagesOff()) {
						around = ReleaseAroundImages(m_buffer_cache, block, block_bytes,
						                             image_pages, fault_vaddr, released);
						if (around != 0) {
							outcome = Outcome::AroundImage;
						}
					}
				} else if (!m_buffer_cache.ReleaseBlockForCpuWrite(block, block_bytes)) {
					outcome = Outcome::GpuBytes;
				} else {
					outcome  = Outcome::Released;
					released = true;
				}
			}
			NoteWriteFault(released, around != 0, now);
			FaultPages::Note(fault_vaddr, frame, outcome);
			if (released && MtGpu::WorkersMode()) {
				// KYTY_MT_GPU=1/4: the block is writable from now on without faults.
				MtGpu::NoteWrite(MtGpu::WriteKind::Host, block, block_bytes);
			}
			if (released && around != 0) {
				// The released runs were journaled as they were released.
				m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
				return true;
			}
			if (released) {
				LibKernel::Memory::JournalWrite(block, block_bytes,
				                                LibKernel::Memory::WriteSource::CpuWriteFault);
				m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
				// A block released again and again stays writable (hot pages).
				if (hot_pages && FaultRecurs((block / block_bytes) | (uint64_t {1} << 35u), now)) {
					FaultPages::NoteHotMarked(m_buffer_cache.MarkHotPages(block, block_bytes));
				}
				return true;
			}
		} else {
			FaultPages::Note(fault_vaddr, frame, outcome);
		}
		const auto page = Common::AlignDown(fault_vaddr, TRACKER_PAGE_SIZE);
		LibKernel::Memory::JournalWrite(page, TRACKER_PAGE_SIZE,
		                                LibKernel::Memory::WriteSource::CpuWriteFault);
		MtGpu::NoteWrite(MtGpu::WriteKind::Host, fault_vaddr, fault_size);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		if (!texture_done) {
			m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
		if (MtGpu::WorkersMode()) {
			// Stamped again once the caches have seen it: a worker that took its epoch between
			// the first stamp and the invalidation read the caches before it.
			MtGpu::NoteWrite(MtGpu::WriteKind::Host, fault_vaddr, fault_size);
		}
		m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
		// A page written again and again stays writable (hot pages).
		if (hot_pages && FaultRecurs(page / TRACKER_PAGE_SIZE, NowMs())) {
			FaultPages::NoteHotMarked(m_buffer_cache.MarkHotPages(page, TRACKER_PAGE_SIZE));
		}
	} else {
		// KYTY_TIMELINE / F6: a game thread's read of GPU-written memory, and when it went on.
		Timeline::Mark("fault-read", fault_vaddr);
		FaultPages::Note(fault_vaddr, m_graphics.presented_frames.load(std::memory_order_relaxed),
		                 FaultPages::Outcome::Read);
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
		Timeline::Mark("fault-read-end", fault_vaddr);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
}

namespace {

// Stores done in place: each is a round trip to the GPU thread that leaves the page protected. A
// page the CPU writes over many times (a structure filled field by field) costs more that way than
// one download, after which it is writable: after PageBudget of them within WindowMs, the page's
// next store takes the download. A store that spends the budget of BulkPages pages within a second
// is copying in bulk (a memcpy into GPU-written memory): it takes the download for BulkMs.
constexpr uint32_t PageBudget = 8;
constexpr uint64_t WindowMs   = 100;
constexpr uint32_t BulkPages  = 2;
constexpr uint64_t BulkMs     = 2000;

bool TakeStoreBudget(uint64_t pc, uint64_t page) {
	struct PageEntry {
		uint64_t page     = UINT64_MAX;
		uint64_t since_ms = 0;
		uint32_t count    = 0;
	};
	struct SiteEntry {
		uint64_t pc            = 0;
		uint64_t since_ms      = 0;
		uint32_t spent         = 0; // pages whose budget it spent since since_ms
		uint64_t bulk_until_ms = 0;
	};
	static std::array<PageEntry, 1024> pages {};
	static std::array<SiteEntry, 256>  sites {};
	static std::mutex                  mutex;
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	std::scoped_lock lock(mutex);
	auto&            site = sites[(pc ^ (pc >> 8u)) % sites.size()];
	if (site.pc != pc) {
		site = {pc, now, 0, 0};
	}
	if (now < site.bulk_until_ms) {
		return false;
	}
	auto& entry = pages[(page / TRACKER_PAGE_SIZE) % pages.size()];
	if (entry.page != page || now - entry.since_ms >= WindowMs) {
		entry = {page, now, 0};
	}
	if (entry.count >= PageBudget) {
		// This store takes the download, which leaves the page writable.
		if (now - site.since_ms >= 1000) {
			site.since_ms = now;
			site.spent    = 0;
		}
		if (++site.spent >= BulkPages) {
			site.bulk_until_ms = now + BulkMs;
			site.spent         = 0;
		}
		return false;
	}
	entry.count++;
	return true;
}

} // namespace

bool RenderContext::IsGpuWrittenPage(uint64_t fault_vaddr) noexcept {
	if (!IsMapped(fault_vaddr, 1)) {
		return false;
	}
	const auto page = fault_vaddr & ~(TRACKER_PAGE_SIZE - 1);
	return m_buffer_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
}

std::string RenderContext::DescribeGpuBytes(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !IsMapped(vaddr, size)) {
		return "not mapped for the GPU";
	}
	const auto  page        = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const bool  page_gpu    = m_buffer_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
	const bool  bytes_dirty = m_buffer_cache.HasGpuDirtyBytesShared(vaddr, size);
	const bool  clean       = m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
	const char* watched     = m_page_manager.IsReadWatched(vaddr)    ? "reads and writes fault"
	                          : m_page_manager.IsWriteWatched(vaddr) ? "writes fault"
	                                                                 : "no access faults";
	return fmt::format("page GPU-written {}, these bytes GPU-written {}{}, the page: {}",
	                   page_gpu ? "yes" : "no", bytes_dirty ? "yes" : "no",
	                   !bytes_dirty && !clean ? " (being downloaded)" : "", watched);
}

bool RenderContext::TryWriteFaulting(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr,
                                     const void* data, uint64_t size) noexcept {
	if (m_gpu == nullptr || GuestGpu::IsGpuThread() || size == 0 || !IsMapped(vaddr, size)) {
		return false;
	}
	// A page with no GPU-written bytes left (they were downloaded with another page) only has its
	// protection lifted on the normal path, without a download: the stores after this one are then
	// free, where done in place each would fault again.
	const auto page = fault_vaddr & ~(TRACKER_PAGE_SIZE - 1);
	if (m_buffer_cache.IsCleanForConcurrentRead(page, TRACKER_PAGE_SIZE)) {
		return false;
	}
	if (!TakeStoreBudget(pc, page)) {
		Common::PerfStats::AddDeclinedStore();
		return false;
	}
	bool written = false;
	if (!m_gpu->TrySendCommandSync([this, vaddr, data, size, &written] {
		    written = m_buffer_cache.WriteClean(vaddr, data, size);
	    })) {
		return false;
	}
	if (!written) {
		Common::PerfStats::AddDeclinedStore();
		return false;
	}
	m_bindless_table.OnCpuWrite(vaddr, size);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	MtGpu::NoteWrite(MtGpu::WriteKind::Host, vaddr, size);
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	if (MtGpu::WorkersMode()) {
		MtGpu::NoteWrite(MtGpu::WriteKind::Host, vaddr, size); // as in the fault path
	}
	m_bindless_table.OnCpuWrite(vaddr, size);
	return true;
}

void RenderContext::PrepareGpuThreadWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// Round 111: the store below relies on the protections the GPU thread deferred.
	m_buffer_cache.FinishDeferredMemoryWork();
	// A hot page is writable without a fault, and nothing else would tell the tracker.
	m_buffer_cache.DirtyHotPagesIn(vaddr, size);
	if (Bisect::GpuWritesFault()) {
		return;
	}
	for (uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1); page < vaddr + size;
	     page += TRACKER_PAGE_SIZE) {
		if (m_page_manager.IsWriteWatched(page)) {
			(void)InvalidateMemory(vaddr, size);
			return;
		}
	}
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

// KYTY_MAP_REDIRTY (round 103): what MapMemory does for the range it maps for the GPU. By default
// its tracker regions become pending for the BDA walk again (BufferCache::MarkMappedPending): a
// walk while the range was unmapped (between the unmap that precedes a map and the map) can take
// their pending bit without synchronizing them, and their CPU-dirty pages stay writable, so no
// write fault marks them again. "1" also makes its clean pages CPU-dirty
// (BufferCache::RedirtyMapped), for bytes a mapping brings that no write fault announced (a new
// view of direct memory with data in it, a buffer that covered the range while it was unmapped and
// read zeros there). "0" does neither, as before round 103.
enum class MapRedirtyMode : uint8_t { Off, Pending, Full };
static MapRedirtyMode MapRedirty() {
	static const MapRedirtyMode mode = [] {
		const char* value = std::getenv("KYTY_MAP_REDIRTY");
		if (value == nullptr || value[0] == '\0') {
			return MapRedirtyMode::Pending;
		}
		return value[0] == '1'   ? MapRedirtyMode::Full
		       : value[0] == '0' ? MapRedirtyMode::Off
		                         : MapRedirtyMode::Pending;
	}();
	return mode;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
	}
	uint64_t   redirtied = 0;
	const auto mode      = MapRedirty();
	if (mode == MapRedirtyMode::Pending && GuestRange {vaddr, size}.Valid()) {
		m_buffer_cache.MarkMappedPending(vaddr, size);
	} else if (mode == MapRedirtyMode::Full && GuestRange {vaddr, size}.Valid()) {
		redirtied = m_buffer_cache.RedirtyMapped(vaddr, size);
		m_buffer_cache.MarkMappedPending(vaddr, size);
		static std::atomic<uint64_t> maps {0};
		static std::atomic<uint64_t> pages {0};
		const auto                   n = maps.fetch_add(1, std::memory_order_relaxed) + 1;
		const auto total = pages.fetch_add(redirtied, std::memory_order_relaxed) + redirtied;
		if (std::has_single_bit(n)) {
			LOGF("Map redirty: %" PRIu64 " maps so far made %" PRIu64
			     " clean pages CPU-dirty (KYTY_MAP_REDIRTY); the latest 0x%016" PRIx64 "+0x%" PRIx64
			     ", %" PRIu64 " pages\n",
			     n, total, vaddr, size, redirtied);
		}
	}
	NoteMapEvent(MapEventKind::Map, vaddr, size, redirtied);
}

void RenderContext::NoteMapEvent(MapEventKind kind, uint64_t vaddr, uint64_t size,
                                 uint64_t redirtied) {
	constexpr size_t Capacity = 65536;
	const MapEvent   event {.frame = m_graphics.presented_frames.load(std::memory_order_relaxed),
	                        .vaddr = vaddr,
	                        .size  = size,
	                        .redirtied = redirtied,
	                        .kind      = kind};
	std::lock_guard  lock(m_map_events_mutex);
	if (m_map_events.size() < Capacity) {
		m_map_events.push_back(event);
	} else {
		m_map_events[m_map_events_next] = event;
	}
	m_map_events_next = (m_map_events_next + 1) % Capacity;
	m_map_events_total++;
}

std::vector<std::string> RenderContext::DescribeMappings(uint64_t vaddr, uint64_t size,
                                                         size_t count) const {
	constexpr size_t         Capacity = 65536;
	std::vector<std::string> lines;
	std::lock_guard          lock(m_map_events_mutex);
	size_t                   matching = 0;
	for (size_t k = m_map_events.size(); k-- > 0;) {
		const auto  index = m_map_events.size() < Capacity ? k : (m_map_events_next + k) % Capacity;
		const auto& event = m_map_events[index];
		if (event.vaddr >= vaddr + size || event.vaddr + event.size <= vaddr) {
			continue;
		}
		if (matching++ < count) {
			lines.insert(
			    lines.begin(),
			    fmt::format("frame {} {} 0x{:012x}+0x{:x}{}", event.frame,
			                event.kind == MapEventKind::Map ? "map" : "unmap", event.vaddr,
			                event.size,
			                event.redirtied != 0
			                    ? fmt::format(" ({} pages made CPU-dirty)", event.redirtied)
			                    : std::string {}));
		}
	}
	lines.push_back(fmt::format("{} map and unmap calls over the range among the last {} of {}",
	                            matching, m_map_events.size(), m_map_events_total));
	return lines;
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The kernel unmaps every free range it is about to (re)map, most of which were never GPU
	// mapped. Buffers, images and GPU-dirty pages are only tracked inside mapped ranges (faults
	// and invalidations outside them are ignored), so such an unmap has nothing to invalidate:
	// skip the GPU-thread round trip and the full drain it forces. KYTY_UNMAP_SKIP_UNMAPPED=0
	// sends every unmap to the GPU thread again.
	static const bool skip_unmapped = [] {
		const auto* value = std::getenv("KYTY_UNMAP_SKIP_UNMAPPED");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (skip_unmapped && GuestRange {vaddr, size}.Valid()) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	NoteMapEvent(MapEventKind::Unmap, vaddr, size, 0);
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_buffer_cache.ForgetUnmapped(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_bindless_table.UnwatchRange(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Every cached buffer is synchronized so a global-memory shader sees the CPU's writes.
	// That walk touched every buffer per dispatch (a quarter of the GPU thread in the
	// world); it only has to repeat once something became CPU-dirty or a buffer appeared.
	// The walk runs at most once per guest submission: what the game wrote before submitting is
	// visible to all its commands, as on the console, and the GPU thread runs behind the game
	// anyway. Per draw, the walk and the re-protection it triggers took ~38 % of the GPU thread
	// on the title menu (21 -> 32 fps).
	// Round 90: and once more after each wait on memory that passed in it (WAIT_REG_MEM): a game
	// may write data after it submits and let the GPU go on past a wait once it is done (a
	// crowd's vertex tables). Read through addresses from before the wait, such draws find what
	// the memory held frames before (other draws' tables: objects where none are), and a colour
	// pass misses its depth pass (people drawn black or see-through).
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		const auto submission = g_guest_submission_seq.load(std::memory_order_relaxed);
		const auto waits      = g_guest_wait_seq.load(std::memory_order_relaxed);
		const auto writes     = g_cpu_write_epoch.load(std::memory_order_acquire);
		if (submission == m_bda_synced_submission) {
			// Again in the same submission only when a wait passed since the last walk and the
			// game wrote since then (not the GPU thread alone: its labels, as one queue waiting
			// for another's, would bring back a walk a draw).
			const bool reopened =
			    BdaSyncAtWaits() && waits != m_bda_synced_waits && writes != m_bda_synced_writes;
			// Round 103: and when a new buffer brought tracker regions (BufferCache::
			// TakeNewRegions): what it covers there is uploaded by a walk alone, and the shaders
			// of this submission read it from now on.
			if (!reopened && !m_buffer_cache.TakeNewRegions()) {
				GpuPhases::Count(GpuPhases::Phase::BdaHeldBack, 1);
				m_fault_process_pending = true;
				return;
			}
			GpuPhases::Count(GpuPhases::Phase::BdaAfterWait, 1);
		}
		m_bda_synced_submission = submission;
		m_bda_synced_waits      = waits;
		m_bda_synced_writes     = writes;
	}
	if (epoch != m_bda_synced_epoch) {
		// The guest writes somewhere nearly all the time, so the epoch moves between most
		// dispatches: walk only the regions where pages became CPU-dirty or a buffer appeared
		// since the last walk (BufferCache::SynchronizeBdaPending). KYTY_BISECT's bdawalk:all
		// walks every region holding a CPU-dirty page and every buffer in it, as before round 69.
		const GpuPhases::Scope walk_phase(GpuPhases::Phase::BdaWalk);
		const auto             protects = GpuPhases::CountOf(GpuPhases::Phase::Protect);
		std::shared_lock       lock(m_mapped_ranges_mutex);
		if (Bisect::BdaWalkAll()) {
			m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(start, end - start);
			});
		} else {
			GpuPhases::Count(GpuPhases::Phase::BdaRegions,
			                 m_buffer_cache.SynchronizeBdaPending(m_mapped_ranges));
		}
		GpuPhases::Count(GpuPhases::Phase::BdaProtects,
		                 GpuPhases::CountOf(GpuPhases::Phase::Protect) - protects);
		(void)m_buffer_cache.TakeNewRegions(); // this walk visits them
		m_bda_synced_epoch = epoch;
	}
	m_fault_process_pending = true;
}

void RenderContext::RefreshHotPages() {
	FaultPages::StartIfActive();
	// Hot pages stay writable to the CPU and are uploaded again after each refresh, so a shader
	// store through an address (which marks no range) there would be overwritten: none once one
	// was seen.
	if (!Bisect::HotPages() || m_buffer_cache.AddressWritesSeen()) {
		if (m_buffer_cache.HotPages() != 0) {
			m_buffer_cache.ClearHotPages();
			FaultPages::NoteHotCleared();
		}
		return;
	}
	// A page the game stopped writing would be uploaded at each refresh for nothing: every 4 s
	// none is hot, and the pages still written fault their way back in (three faults each).
	constexpr uint64_t DecayMs = 4000;
	const auto         now     = NowMs();
	if (now - m_hot_cleared_ms >= DecayMs) {
		m_hot_cleared_ms = now;
		if (m_buffer_cache.HotPages() != 0) {
			m_buffer_cache.ClearHotPages();
			FaultPages::NoteHotCleared();
		}
		return;
	}
	const auto pages = m_buffer_cache.RedirtyHotPages();
	FaultPages::NoteHotRefresh(pages, m_buffer_cache.HotPages());
}

// Diagnostics: KYTY_DUMP_GUEST=<frame>:<address>+<size>[,<address>+<size>...] (frame decimal,
// the rest hex) writes guest memory, with the GPU's writes downloaded first, to
// dump_<frame>_<address>.bin once that many frames have been presented. A frame written +<n>
// counts from the first use of the KYTY_WATCH_SHADER shader.
static void DumpGuestMemory(GraphicContext& graphics, BufferCache& cache) {
	struct Request {
		uint64_t                                   frame    = 0;
		bool                                       relative = false;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	static const Request request = [] {
		Request     result;
		const char* value = std::getenv("KYTY_DUMP_GUEST");
		if (value == nullptr) {
			return result;
		}
		char* end       = nullptr;
		result.relative = *value == '+';
		result.frame    = std::strtoull(result.relative ? value + 1 : value, &end, 10);
		value        = end != nullptr && *end == ':' ? end + 1 : nullptr;
		while (value != nullptr && *value != '\0') {
			const auto address = std::strtoull(value, &end, 16);
			uint64_t   size    = 4;
			if (end != nullptr && *end == '+') {
				size = std::strtoull(end + 1, &end, 16);
			}
			result.ranges.emplace_back(address, size);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static bool done = false;
	if (done || request.ranges.empty()) {
		return;
	}
	const auto first = WatchedShaderFirstFrame();
	if (request.relative && first == UINT64_MAX) {
		return;
	}
	const auto frame = request.relative ? first + request.frame : request.frame;
	if (graphics.presented_frames.load(std::memory_order_relaxed) < frame) {
		return;
	}
	done = true;
	for (const auto& [address, size]: request.ranges) {
		cache.DownloadRangeForDiagnostics(address, size);
		std::vector<uint8_t> bytes(size);
		const bool read = LibKernel::Memory::TryReadBacking(address, bytes.data(), size);
		const auto name = "dump_" + std::to_string(frame) + "_" +
		                  std::to_string(address) + ".bin";
		if (read) {
			if (auto* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
				std::fwrite(bytes.data(), 1, bytes.size(), file);
				std::fclose(file);
			}
		}
		LOGF("DumpGuest: 0x%016" PRIx64 "+0x%" PRIx64 " -> %s (%s)\n", address, size, name.c_str(),
		     read ? "written" : "not readable");
	}
}

// Diagnostics: KYTY_PROBE_LIST=<lo>,<hi>,<records>,<start>,<next>,<word0>,<word1> (indices into
// the values of KYTY_PROBE <hash>@stuck samples) reads the list a stuck wave walks in guest
// memory, with the GPU's writes downloaded first: a buffer whose V# has its base in values
// <lo> and <hi> (the low 16 bits; the stride in bits 16-29 of <hi>) and its record count in
// <records>, the list starting at the byte offset <start> (its low 3 bits cleared), and the wave
// at the record before index <next>, which it read as <word0> and <word1>. It logs the list's
// first records, whether guest memory holds what the wave read, and where the first record whose
// second word has bit 31 set (the end of the list) is ("Stuck list:").
//
// Round 103: first, for each of up to 8 such buffers (the V#'s whole range), its pages as the GPU
// holds them against guest memory, with the tracker's state of each, its page table entries and
// the history of its buffers and mappings (BufferCache::CompareWithGuest, "Stuck buffer"). With
// KYTY_PROBE_LIST_HEAL=1 the pages that differ while the GPU wrote none of them are uploaded
// again from guest memory, and the buffer is compared again 10 s later ("Stuck buffer ...
// recheck").
static uint64_t CompareStuckBuffer(RenderContext& context, BufferCache& cache, uint64_t base,
                                   uint64_t size, const std::string& label, bool heal,
                                   bool quiet = false) {
	const auto stale = cache.CompareWithGuest(
	    base, size, label, heal,
	    [&context](uint64_t vaddr, uint64_t bytes) { return context.IsMapped(vaddr, bytes); },
	    quiet);
	if (!quiet || stale != 0) {
		for (const auto& line: context.DescribeMappings(base, size, 24)) {
			LOGF("%s: mappings: %s\n", label.c_str(), line.c_str());
		}
	}
	return stale;
}

static void AnalyzeStuckSamples(RenderContext& context, BufferCache& cache) {
	static const std::vector<uint32_t> list = [] {
		std::vector<uint32_t> indices;
		const char*           value = std::getenv("KYTY_PROBE_LIST");
		while (value != nullptr && *value != '\0' && indices.size() < 7) {
			char* end = nullptr;
			indices.push_back(static_cast<uint32_t>(std::strtoul(value, &end, 10)));
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return indices;
	}();
	if (list.size() != 7) return;
	// Round 104: a buffer is compared again when its waves are stuck again 3 s or more after its
	// last comparison (an episode, such as one a change of view brings), up to 12 times.
	struct Checked {
		uint64_t base       = 0;
		uint64_t size       = 0;
		uint64_t recheck_ms = 0;
		uint64_t last_ms    = 0;
		uint32_t compares   = 1;
	};
	static std::vector<Checked> checked;
	static const bool           heal = [] {
        const char* value = std::getenv("KYTY_PROBE_LIST_HEAL");
        return value != nullptr && value[0] == '1';
	}();
	for (auto& entry: checked) {
		if (entry.recheck_ms != 0 && NowMs() >= entry.recheck_ms) {
			entry.recheck_ms = 0;
			(void)CompareStuckBuffer(context, cache, entry.base, entry.size,
			                         fmt::format("Stuck buffer 0x{:012x} recheck", entry.base),
			                         false);
		}
	}
	auto samples = TakeStuckProbeSamples();
	if (samples.empty()) return;
	static std::vector<std::pair<uint64_t, uint64_t>> seen;
	for (const auto& sample: samples) {
		if (std::ranges::any_of(list, [&](uint32_t index) { return index >= sample.count; })) {
			continue;
		}
		const auto value  = [&](uint32_t which) { return sample.values[list[which]]; };
		const auto base   = value(0) | (uint64_t {value(1) & 0xffffu} << 32u);
		const auto stride = std::max<uint32_t>((value(1) >> 16u) & 0x3fffu, 1u);
		const auto start  = uint64_t {value(3) & ~7u};
		const auto next   = uint64_t {value(4)};
		auto       known =
		    std::ranges::find_if(checked, [&](const Checked& entry) { return entry.base == base; });
		if (known == checked.end() && checked.size() < 8) {
			// Before anything below downloads or synchronizes the range.
			const auto size  = uint64_t {stride} * value(2);
			const auto stale = CompareStuckBuffer(
			    context, cache, base, size, fmt::format("Stuck buffer 0x{:012x}", base), heal);
			checked.push_back({base, size, heal && stale != 0 ? NowMs() + 10000u : 0u, NowMs()});
		} else if (known != checked.end() && known->compares < 12 &&
		           NowMs() >= known->last_ms + 3000u) {
			known->compares++;
			known->last_ms   = NowMs();
			const auto stale = CompareStuckBuffer(
			    context, cache, known->base, known->size,
			    fmt::format("Stuck buffer 0x{:012x} again ({})", known->base, known->compares),
			    heal, true);
			if (heal && stale != 0) {
				known->recheck_ms = NowMs() + 10000u;
			}
		}
		if (seen.size() >= 24) continue;
		if (std::ranges::find(seen, std::pair {base, start}) != seen.end()) continue;
		seen.emplace_back(base, start);
		// The list from its start to two records past the wave's, at most 1 MB.
		const auto end  = std::max(start + 8u * stride, (next + 2u) * stride);
		const auto size = std::min<uint64_t>(end > start ? end - start : 8u * stride, 1u << 20u);
		std::vector<uint32_t> words((size + 3u) / 4u);
		cache.DownloadRangeForDiagnostics(base + start, size);
		const bool  read = LibKernel::Memory::TryReadBacking(base + start, words.data(), size);
		std::string first;
		for (uint64_t record = 0; read && record < 8u && (record * stride + 8u) <= size; record++) {
			const auto at = record * stride / 4u;
			first += fmt::format(" ({:08x} {:08x})", words[at], words[at + 1u]);
		}
		std::string ends = "none in the records read";
		for (uint64_t record = 0; read && (record * stride + 8u) <= size; record++) {
			if ((words[record * stride / 4u + 1u] & 0x80000000u) != 0u) {
				ends = fmt::format("record {} (index {})", record, start / stride + record);
				break;
			}
		}
		std::string held = "outside the bytes read";
		if (read && next >= 1u && (next - 1u) * stride >= start &&
		    (next - 1u) * stride + 8u - start <= size) {
			const auto at = ((next - 1u) * stride - start) / 4u;
			held =
			    fmt::format("{:08x} {:08x}{}", words[at], words[at + 1u],
			                words[at] == value(5) && words[at + 1u] == value(6) ? " (the same)"
			                                                                    : " (different)");
		}
		LOGF(
		    "Stuck list: shader %08x%08x loop 0x%04x: buffer 0x%012" PRIx64
		    " stride %u records %u, list at +0x%" PRIx64 " (index %" PRIu64
		    "), the wave at index %" PRIu64
		    " read %08x %08x; guest memory%s: there %s; the first records:%s; the list's end: %s\n",
		    sample.shader_hash_high, sample.shader_hash_low, sample.loop_pc, base, stride, value(2),
		    start, start / stride, next >= 1u ? next - 1u : 0u, value(5), value(6),
		    read ? "" : " not readable", held.c_str(), first.c_str(), ends.c_str());
	}
}

// Diagnostics: KYTY_VERIFY_BUFFERS=<seconds> (at least 10) compares every cached buffer with guest
// memory that often, the first time that long after the first frames ("Buffer verify:",
// BufferCache::VerifyCacheWithGuest): pages the GPU holds otherwise while the tracker holds them
// clean, and CPU-dirty pages the BDA walk will not upload. KYTY_VERIFY_BUFFERS_HEAL=1 uploads
// them again.
static void VerifyBuffers(RenderContext& context, BufferCache& cache) {
	static const uint64_t period_ms = [] {
		const char* value   = std::getenv("KYTY_VERIFY_BUFFERS");
		const auto  seconds = value != nullptr ? std::strtoull(value, nullptr, 10) : 0u;
		return seconds == 0 ? uint64_t {0} : std::max<uint64_t>(seconds, 10u) * 1000u;
	}();
	if (period_ms == 0) {
		return;
	}
	static const bool heal = [] {
		const char* value = std::getenv("KYTY_VERIFY_BUFFERS_HEAL");
		return value != nullptr && value[0] == '1';
	}();
	static uint64_t next_ms = 0;
	const auto      now     = NowMs();
	if (next_ms == 0) {
		next_ms = now + period_ms;
		return;
	}
	if (now < next_ms) {
		return;
	}
	(void)cache.VerifyCacheWithGuest(heal, [&context](uint64_t vaddr, uint64_t bytes) {
		return context.IsMapped(vaddr, bytes);
	});
	next_ms = NowMs() + period_ms;
}

void RenderContext::RunGarbageCollector() {
	DumpGuestMemory(GetGraphics(), m_buffer_cache);
	VerifyBuffers(*this, m_buffer_cache);
	AnalyzeStuckSamples(*this, m_buffer_cache);
	LogWatchedRereads(m_buffer_cache, GetGraphics().presented_frames.load(std::memory_order_relaxed));
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
