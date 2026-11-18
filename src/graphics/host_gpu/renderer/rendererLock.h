#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERERLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERERLOCK_H_

#include "common/common.h"
#include "common/perfStats.h"
#include "common/threads.h"

#include <array>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

// MT GPU stage C (gt7-mt-1117): the renderer's one lock (was RenderContext::m_mutex) split in
// four, one per shared part:
//   pipeline  - PipelineCache (lookups, creation, seeds, Save)
//   buffer    - BufferCache
//   texture   - TextureCache (and the binding tables registered in it), BindlessTable
//   scheduler - CommandScheduler and the command buffer being recorded
// ORDER (fixed, to rule out deadlocks): pipeline -> buffer -> texture -> scheduler, and the
// graphics queue mutex (GraphicContext::queue_mutex) after all of them. A thread takes the ones
// it needs at once, through RendererLock, which always locks in that order; nothing takes one
// of them while holding a later one. The mutexes are recursive (Common::Mutex), so a nested
// RendererLock for a subset of what the thread already holds is fine.
// Who takes what:
//   GPU thread, a draw or a dispatch (renderDraw.cpp, renderCompute.cpp): all four. Until stage
//     d splits a draw into translation and emission on different threads, nothing narrower is
//     safe there, and with all four the GPU thread behaves as under the one lock.
//   GPU thread, a global barrier (CommandProcessor::EmitGlobalBarrier): scheduler.
//   presenter, a guest frame (Presenter::PrepareFrame, PresentLastFrame): buffer, texture,
//     scheduler (ResolveSurface may reach the buffer cache through the texture cache).
//   presenter, a blank frame (PrepareBlankFrame): scheduler.
//   window thread, pipeline seeds and their Save (window.cpp): pipeline.
//   RenderDoc capture boundaries (renderDoc.cpp): all four, then the queue.
enum RendererLockSet : uint32_t {
	kRendererLockPipeline  = 1u << 0,
	kRendererLockBuffer    = 1u << 1,
	kRendererLockTexture   = 1u << 2,
	kRendererLockScheduler = 1u << 3,
	kRendererLockAll       = 0xfu,
};

struct RendererLocks {
	// Indexed by the bit's position: the locking order.
	std::array<Common::Mutex, 4> mutexes;
};

// The renderer's locks a thread needs, taken in the fixed order and released in reverse. With
// `account` (the GPU thread's draws and dispatches), how long it waited goes to the "Perf
// locks:" line ("renderer"), as with the one lock before (round 128).
class RendererLock {
public:
	explicit RendererLock(RendererLocks& locks, uint32_t set = kRendererLockAll,
	                      bool account = true)
	    : m_locks(locks), m_set(set & kRendererLockAll) {
		uint64_t waited = 0;
		for (uint32_t i = 0; i < m_locks.mutexes.size(); i++) {
			if ((m_set & (1u << i)) == 0) continue;
			auto& mutex = m_locks.mutexes[i];
			if (!mutex.TryLock()) {
				const auto begin = std::chrono::steady_clock::now();
				mutex.Lock();
				waited += static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - begin)
				        .count());
			}
		}
		if (account && waited != 0) {
			Common::PerfStats::AddLockWait(Common::PerfStats::Lock::Renderer, waited);
		}
	}
	~RendererLock() {
		for (uint32_t i = static_cast<uint32_t>(m_locks.mutexes.size()); i-- > 0;) {
			if ((m_set & (1u << i)) != 0) m_locks.mutexes[i].Unlock();
		}
	}

	KYTY_CLASS_NO_COPY(RendererLock);

private:
	RendererLocks& m_locks;
	uint32_t       m_set;
};

// Round 128: how long the presenter holds the renderer's lock, from its construction to its end
// (made just after the lock is taken): the "Perf locks:" line's "renderer held by the
// presenter".
class RendererHoldTimer {
public:
	RendererHoldTimer() = default;
	~RendererHoldTimer() {
		Common::PerfStats::AddLockWait(
		    Common::PerfStats::Lock::RendererHeldByPresenter,
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_begin)
		                              .count()));
	}

	KYTY_CLASS_NO_COPY(RendererHoldTimer);

private:
	std::chrono::steady_clock::time_point m_begin = std::chrono::steady_clock::now();
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERERLOCK_H_
