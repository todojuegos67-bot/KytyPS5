#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics, true),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()), m_guest_image_heap(graphics, m_page_manager),
      // The guest image heap (--bindless) owns descriptor set 1; the resident image heap then stays unused.
      m_pipeline_cache(graphics, graphics.guest_image_heap_enabled ? nullptr : &m_image_heap),
      m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	ShaderRecompiler::IR::SetSrtCleanReader([](uint64_t address, std::span<uint32_t> values) {
		return Libs::LibKernel::Memory::TryReadGpuCleanBytes(address, values.data(), values.size_bytes());
	});
	m_guest_image_heap.UseScheduler(m_command_scheduler);
}

RenderContext::~RenderContext() {
	ShaderRecompiler::IR::SetSrtCleanReader(nullptr);
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	m_command_scheduler.AdoptRecording();
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
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		// A game writes its frame's data a page after the other, each page a fault (Marvel's
		// Wolverine: 137,000 a second at its title, most of it a ring of skinning data that its job
		// threads copy ~19 MB a frame into). One write fault opens the 512 KB around it for the
		// buffer cache, else the 256 KB, else the 64 KB, unless a shader wrote in there (that would be
		// a read back of all of it); a page opened without a write is uploaded again unchanged (512 KB:
		// a quarter fewer faults for a fifth more upload at the cabin title; a megabyte cost frames).
		// The texture cache keeps the byte.
		auto begin = fault_vaddr, size = fault_size;
		for (const uint64_t window: {uint64_t {512} << 10u, uint64_t {256} << 10u, uint64_t {64} << 10u}) {
			const auto start = fault_vaddr & ~(window - 1u);
			if (IsMapped(start, window) && !m_buffer_cache.IsRegionGpuModified(start, window)) {
				begin = start;
				size  = window;
				break;
			}
		}
		m_buffer_cache.InvalidateMemory(begin, size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
		m_guest_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else if (!m_buffer_cache.ReleaseKept(fault_vaddr) && m_buffer_cache.IsRegionGpuModified(fault_vaddr, fault_size)) {
		// A page neither kept nor GPU-written skips this: another thread lifted it while this one faulted, so the
		// instruction runs again with no trip through the GPU thread, and faults again if it's still protected.
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	m_guest_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t RenderContext::MappedRangesVersion() const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges_version;
}

uint64_t RenderContext::MappedEnd(uint64_t vaddr, uint64_t size) const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	uint64_t         end = vaddr;
	m_mapped_ranges.ForEachInRange(vaddr, size, [&](uint64_t begin, uint64_t finish) {
		if (begin == end) {
			end = finish;
		}
	});
	return end;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
		m_buffer_cache.QueueBdaSync(vaddr, size);
		m_mapped_ranges_version++;
	}
	m_buffer_cache.NoteMapped(vaddr, size); // before the invalidation below: no upload may meet a page not backed yet
	// A buffer can already span this range from when it was reserved (a finite view over unmapped pages): it holds the
	// zeros of then, and no write protection watches the new pages. Mark them CPU-written so the next use uploads them.
	if (m_buffer_cache.HoldsRegion(vaddr, size)) {
		m_buffer_cache.InvalidateMemory(vaddr, size);
	}
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// No part of the range mapped for the GPU, no guest-memory completion pending, and no cache holding
	// anything there (a buffer outlives its range's unmap, and can stretch over unmapped neighbours): the
	// GPU thread would only invalidate the range's pages, which a fault does from any thread too. Mapping
	// over reserved memory (Marvel's Wolverine's automatic memory: ~1,200 a second in the white transition)
	// then costs no round trip to the GPU thread.
	bool mapped = false;
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		mapped = m_mapped_ranges.Intersects(vaddr, size);
	}
	if (!mapped && !m_command_scheduler.HasPendingPriorityOperations() && !m_buffer_cache.HoldsRegion(vaddr, size) &&
	    !m_texture_cache.HoldsRegion(vaddr, size) && m_image_heap.Empty()) {
		m_buffer_cache.InvalidateMemory(vaddr, size);
		return;
	}
	const auto unmap = [this, vaddr, size] {
		// A streamed texture's dropped levels give their memory back first, with no wait: the
		// game raised its min LOD frames ago, and the unbind waits behind the work recorded so far.
		m_texture_cache.ReleaseLevels(vaddr, size);
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			// Only the guest-memory completions (priority queue) and the fault scan's buffer lookups touch guest
			// memory from the GPU's side; buffers, images and heap views free themselves at their own tick. So
			// the wait is for the newest of those, not for all the work recorded so far (Marvel's Wolverine's
			// white transition: 160 unmaps a second, each a full drain), unless it is in the command buffer
			// being recorded.
			const auto tick   = m_command_scheduler.CurrentTick();
			const auto needed = std::max(m_command_scheduler.LastPriorityTick(), m_buffer_cache.LastFaultTick());
			if (needed >= tick) {
				m_command_scheduler.Finish();
			} else if (needed != 0) {
				m_command_scheduler.Wait(needed);
				m_command_scheduler.PopPendingOperations();
			}
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_ranges_version++;
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
	m_buffer_cache.PrepareFaultBuffer();
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	if (m_indirect_write_tables.TakePendingWrites()) {
		m_buffer_cache.ProcessWriteBuffer([this](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteWrites(pages);
		});
		m_buffer_cache.ProcessDroppedStores([this](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteDroppedStores(pages);
		});
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_guest_image_heap.NextFrame();
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
