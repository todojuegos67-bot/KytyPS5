#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"
#include "local-platform.h"

#include <algorithm>
#include <thread>

namespace Libs::Graphics {

void RenderMutex::BecomeOwner() {
	EXIT_IF(t_owner);
	m_mutex.Lock(); // (no requester inside: a later one finds the owner)
	m_owner_exists = true;
	m_asymmetric   = LocalPlatform::FlushProcessWriteBuffers();
	m_owner_depth  = 0;
	m_owner_locked = false;
	t_owner        = true;
	m_mutex.Unlock();
}

void RenderMutex::ResignOwner() {
	EXIT_IF(!t_owner || m_owner_depth != 0);
	m_mutex.Lock();
	t_owner        = false;
	m_owner_exists = false;
	m_mutex.Unlock();
}

void RenderMutex::OwnerLockSlow() {
	// A requester holds the mutex or is about to: out, and in after it (or the host has no flush: the mutex always).
	m_busy.store(0, std::memory_order_release);
	m_mutex.Lock();
	m_owner_locked = true;
}

void RenderMutex::OwnerUnlockSlow() {
	m_owner_locked = false;
	m_mutex.Unlock();
}

void RenderMutex::RequesterLock() {
	if (t_requester_depth++ != 0) return;
	m_mutex.Lock();
	if (!m_owner_exists || !m_asymmetric) return;
	m_requested.store(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	// The owner's `busy` store is visible now if it made it, and its next acquisition sees the request.
	(void)LocalPlatform::FlushProcessWriteBuffers();
	while (m_busy.load(std::memory_order_acquire) != 0) std::this_thread::yield();
}

void RenderMutex::RequesterUnlock() {
	if (--t_requester_depth != 0) return;
	m_requested.store(0, std::memory_order_release);
	m_mutex.Unlock();
}

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_compute_render_executor(*this),
      m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_gpu_resources(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
	m_gpu_resources.SetGpu(m_gpu.get());
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu_resources.SetGpu(nullptr);
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

uint64_t RenderContext::FrameNumber() const {
	return m_gpu != nullptr ? static_cast<uint64_t>(m_gpu->GetFrameNum()) : 0;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
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
