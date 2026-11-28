#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"
#include "local-platform.h"
#include "local/serie1.h" // (serie1: KYTY_SERIE1)
#include "local/tej-lag.h" // (tej-var)

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace Libs::Graphics {

void RenderMutex::BecomeOwner() {
	EXIT_IF(t_owner);
	m_mutex.Lock(); // (no requester inside: a later one finds the owner)
	// (par1) A second owner (KYTY_PAR_QUEUES=1: the ACB thread) joins the first: the fields are the first one's.
	EXIT_IF(m_owner_count != 0 && !ParQueues::On());
	if (m_owner_count++ == 0) {
		m_owner_exists = true;
		m_asymmetric   = LocalPlatform::FlushProcessWriteBuffers();
		m_owner_depth  = 0;
		m_owner_locked = false;
	}
	t_owner = true;
	m_mutex.Unlock();
}

void RenderMutex::ResignOwner() {
	EXIT_IF(!t_owner || m_owner_depth != 0);
	m_mutex.Lock();
	t_owner = false;
	EXIT_IF(m_owner_count == 0);
	if (--m_owner_count == 0) m_owner_exists = false;
	m_mutex.Unlock();
}

void RenderMutex::OwnerLockSlow() {
	// A requester holds the mutex or is about to: out, and in after it (or the host has no flush: the mutex always).
	m_busy.store(0, std::memory_order_release);
	if (TejLag::t_render) { // (tej-var: lo que el render espera a un hilo que pidio el cerrojo)
		const uint64_t t0 = __rdtsc();
		m_mutex.Lock();
		TejLag::g_acc[TejLag::RmOwner].n.fetch_add(1, std::memory_order_relaxed);
		TejLag::g_acc[TejLag::RmOwner].ticks.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
	} else {
		m_mutex.Lock();
	}
	m_owner_locked = true;
}

void RenderMutex::OwnerUnlockSlow() {
	m_owner_locked = false;
	m_mutex.Unlock();
}

void RenderMutex::RequesterLock() {
	// (tej-sinmutex) La sombra no pide el cerrojo (ni IPI ni espera del render): lo que lo necesitaria es hueco 'sin mutex'
	// (Speculation::Translate la recoge como las demas RecorderRefusal: hueco por paquete o parada contada).
	if (t_requester_depth == 0 && TejLag::NoMutex()) {
		TejLag::Note(TejLag::RmSkip);
		throw CommandScheduler::RecorderRefusal {"sin mutex"};
	}
	if (t_requester_depth++ != 0) return;
	// (tej-var) Variante cortes: la sombra espera fuera a que el render salga (lectura relajada, sin IPI), como mucho ~50 us.
	if (TejLag::t_shadow && TejLag::g_polite.load(std::memory_order_relaxed))
		for (uint32_t i = 0; i < 2000 && m_busy.load(std::memory_order_relaxed) != 0; ++i) _mm_pause();
	m_mutex.Lock();
	if (!m_owner_exists || !m_asymmetric) return;
	const uint64_t t0 = TejLag::t_shadow ? __rdtsc() : 0;
	m_requested.store(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	// The owner's `busy` store is visible now if it made it, and its next acquisition sees the request.
	(void)LocalPlatform::FlushProcessWriteBuffers();
	while (m_busy.load(std::memory_order_acquire) != 0) std::this_thread::yield();
	if (TejLag::t_shadow) {
		TejLag::g_acc[TejLag::RmReq].n.fetch_add(1, std::memory_order_relaxed);
		TejLag::g_acc[TejLag::RmReq].ticks.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
	}
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
	// (serie1) KYTY_SERIE1=1: the lanes for the XPR culling's command buffer (local/serie1.h).
	if (const char* async = std::getenv("KYTY_ASYNC_COMPUTE");
	    (async != nullptr && (*async == '1' || *async == '2')) || Serie1::WantsLanes())
		m_command_scheduler.EnableLanes();
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
