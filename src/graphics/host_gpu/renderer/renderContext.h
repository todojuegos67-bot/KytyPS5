#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <memory>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

// The renderer's lock (recursive, as Common::Mutex). The GPU thread takes it around every draw and dispatch
// (thousands a frame), other threads (presentation, a capture) a few times a frame: an interlocked acquisition and
// release per draw stalled the GPU thread on its store buffer (~5% of its graphics queue samples at Latria). So its
// acquisition is a store and a load (an asymmetric Dekker lock): another thread takes the mutex, raises `requested`,
// makes the GPU thread's stores visible and its later loads see the request (LocalPlatform::FlushProcessWriteBuffers)
// and waits until the GPU thread is outside (`busy` clear); the GPU thread that finds a request takes the mutex.
class RenderMutex {
public:
	RenderMutex()  = default;
	~RenderMutex() = default;
	KYTY_CLASS_NO_COPY(RenderMutex);

	// The calling thread (the GPU thread) takes the owner's side from now on, until it resigns (outside any lock).
	void BecomeOwner();
	void ResignOwner();

	void Lock() {
		if (!t_owner) return RequesterLock();
		if (m_owner_depth++ != 0) return;
		m_busy.store(1, std::memory_order_relaxed);
		// (The store before the load in program order: the requester's flush is what orders them on the CPU.)
		std::atomic_signal_fence(std::memory_order_seq_cst);
		if (!m_asymmetric || m_requested.load(std::memory_order_acquire) != 0) OwnerLockSlow();
	}
	void Unlock() {
		if (!t_owner) return RequesterUnlock();
		if (--m_owner_depth != 0) return;
		if (m_owner_locked) return OwnerUnlockSlow();
		m_busy.store(0, std::memory_order_release);
	}

private:
	void OwnerLockSlow();
	void OwnerUnlockSlow();
	void RequesterLock();
	void RequesterUnlock();

	Common::Mutex m_mutex;
	// The owner's (written at each of its acquisitions): it is inside, and its depth.
	alignas(64) std::atomic<uint32_t> m_busy {0};
	uint32_t                          m_owner_depth  = 0;
	bool                              m_owner_locked = false; // (its outermost acquisition took the mutex)
	// The requesters' (written under the mutex): one holds the mutex and waits for the owner to leave; an owner exists.
	alignas(64) std::atomic<uint32_t> m_requested {0};
	bool                              m_owner_exists = false;
	bool                              m_asymmetric   = false; // (the host can flush other processors' stores)
	inline static thread_local bool     t_owner           = false;
	inline static thread_local uint32_t t_requester_depth = 0;
};

class RenderLockGuard {
public:
	// NOLINTNEXTLINE(google-runtime-references)
	explicit RenderLockGuard(RenderMutex& mutex): m_mutex(mutex) { m_mutex.Lock(); }
	~RenderLockGuard() { m_mutex.Unlock(); }
	KYTY_CLASS_NO_COPY(RenderLockGuard);

private:
	RenderMutex& m_mutex;
};

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	// The guest frame number, 0 without a guest GPU (offline tools).
	[[nodiscard]] uint64_t                  FrameNumber() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	RenderMutex&        GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	GpuResourceManager& GetGpuResources() { return m_gpu_resources; }
	BufferCache&        GetBufferCache() { return m_gpu_resources.GetBufferCache(); }
	TextureCache&       GetTextureCache() { return m_gpu_resources.GetTextureCache(); }
	RenderExecutor&     GetRenderExecutor() { return t_executors != nullptr ? *t_executors->draw : m_render_executor; }
	// Dispatches use their own executor, so draws do not displace their
	// per-operation scratch and texture resolutions.
	RenderExecutor&     GetComputeRenderExecutor() {
		return t_executors != nullptr ? *t_executors->compute : m_compute_render_executor;
	}
	// The executors above whatever the calling thread's are (what a speculation's read: RenderExecutor::ReadCatalogOf).
	RenderExecutor& DefaultRenderExecutor() { return m_render_executor; }
	RenderExecutor& DefaultComputeRenderExecutor() { return m_compute_render_executor; }
	// The calling thread's executors instead (a speculative translation's, with their own caches); null: these.
	struct Executors {
		RenderExecutor* draw    = nullptr;
		RenderExecutor* compute = nullptr;
	};
	static void SetThreadExecutors(const Executors* executors) noexcept { t_executors = executors; }

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	inline static thread_local const Executors* t_executors = nullptr;

	GraphicContext&           m_graphics;
	RenderMutex               m_mutex;
	RenderExecutor            m_render_executor;
	RenderExecutor            m_compute_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	GpuResourceManager        m_gpu_resources;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
