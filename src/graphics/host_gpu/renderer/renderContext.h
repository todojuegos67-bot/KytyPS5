#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/rendererLock.h"
#include "kernel/eventQueue.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	// The renderer's locks and their order: rendererLock.h. Take them with RendererLock.
	RendererLocks&      GetLocks() { return m_locks; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	BindlessTable&      GetBindlessTable() { return m_bindless_table; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// Any thread: a read of these bytes faulted on a page protected because the GPU wrote to it,
	// but none of them is GPU-written, so guest memory already holds their value.
	[[nodiscard]] bool CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
	                                     uint64_t size) const noexcept;
	// A game thread's write fault: whether its page is protected because the GPU wrote to it.
	[[nodiscard]] bool IsGpuWrittenPage(uint64_t fault_vaddr) noexcept;
	// Round 133, for a crash report (LibKernel::Memory::DescribeGpuBytes). Any thread.
	[[nodiscard]] std::string DescribeGpuBytes(uint64_t vaddr, uint64_t size);
	// A game thread's store at `pc` that faulted at fault_vaddr on such a page: written to both
	// copies by the GPU thread (BufferCache::WriteClean), the page left protected. False to
	// decline.
	[[nodiscard]] bool TryWriteFaulting(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr,
	                                    const void* data, uint64_t size) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// GPU thread, before it stores to guest memory itself (a label, WRITE_DATA): a page protected
	// for tracking gets what its write fault would have done (InvalidateMemory), so the store
	// does not fault. A fault on the GPU thread cost an exception dispatch each (round 68).
	void               PrepareGpuThreadWrite(uint64_t vaddr, uint64_t size);
	// GPU thread, where it may next read what the CPU wrote since it last looked (a new slice of
	// a submission, a wait on memory that passes): the hot pages (Bisect::HotPages) become
	// CPU-dirty again, as their writes do not fault. Every few seconds no page is hot any more
	// (each has to fault its way back in), and none is while hot pages are off.
	void               RefreshHotPages();
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	// The host protection of the range was replaced behind the page tracker
	// (PageManager::ResyncHostProtection): the watched pages get the tracker's protection back.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ResyncHostProtection(vaddr, size);
	}
	void               PrepareBda();
	void               RunGarbageCollector();
	// Diagnostics: the latest changes to the mapped ranges over [vaddr, vaddr + size) (at most
	// `count`, the latest last), one line each, with the frame they happened in.
	// Round 124, diagnostics: the texture cache's images over the range (DescribeImagesInRange).
	[[nodiscard]] std::vector<std::string> DescribeImages(uint64_t vaddr, uint64_t size,
	                                                      size_t count) {
		return m_texture_cache.DescribeImagesInRange(vaddr, size, count);
	}
	[[nodiscard]] std::vector<std::string> DescribeMappings(uint64_t vaddr, uint64_t size,
	                                                        size_t count) const;

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	RendererLocks             m_locks;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	BindlessTable             m_bindless_table;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	// Diagnostics (DescribeMappings): the latest MapMemory and UnmapMemory calls.
	enum class MapEventKind : uint8_t { Map, Unmap };
	struct MapEvent {
		uint64_t     frame     = 0;
		uint64_t     vaddr     = 0;
		uint64_t     size      = 0;
		uint64_t     redirtied = 0; // KYTY_MAP_REDIRTY: pages the map made CPU-dirty
		MapEventKind kind      = MapEventKind::Map;
	};
	void NoteMapEvent(MapEventKind kind, uint64_t vaddr, uint64_t size, uint64_t redirtied);
	mutable std::mutex        m_map_events_mutex;
	std::vector<MapEvent>     m_map_events;
	size_t                    m_map_events_next  = 0;
	uint64_t                  m_map_events_total = 0;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;
	uint64_t                  m_bda_synced_epoch      = 0;
	uint64_t                  m_bda_synced_submission = 0;
	uint64_t                  m_bda_synced_waits      = 0; // g_guest_wait_seq at that walk
	uint64_t                  m_bda_synced_writes     = 0; // g_cpu_write_epoch at that walk
	uint64_t                  m_hot_cleared_ms        = 0;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
