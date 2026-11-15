#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "common/upstreamPaths.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit(uint64_t tick) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick, tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = tick;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = tick;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics, bool record_thread)
    : m_master(graphics), m_sparse(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	if (graphics.sparse_queue) {
		m_sparse_thread = std::jthread([this](std::stop_token stop) { SparseThread(stop); });
	}
	// The Vulkan calls of the GPU thread's command buffers go to a recording thread, on before the first command buffer
	// begins; KYTY_RECORD_THREAD=0 keeps them on the GPU thread. KYTY_RECORD_VERIFY=1 also checks every packet.
	const auto* thread = std::getenv("KYTY_RECORD_THREAD");
	if (record_thread && (thread == nullptr || std::strcmp(thread, "0") != 0)) {
		const auto* verify = std::getenv("KYTY_RECORD_VERIFY");
		EnableRecordingThread(verify != nullptr && std::strcmp(verify, "0") != 0);
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	m_sparse.Wait(m_sparse.CurrentTick() - 1);
	FreeSparse();
}

// Pending binds go out at the next submit, binds first: their batch is the next value of
// m_sparse, the unbinds' the one after at most.
uint64_t CommandScheduler::BindSparse(vk::Image image, const vk::SparseImageMemoryBind& level, bool deferred) {
	// Binds go before unbinds: an unbind of this level still waiting would undo this bind.
	std::erase_if(m_sparse_unbinds, [&](const SparseBind& unbind) {
		if (unbind.image != image || unbind.level.subresource.mipLevel != level.subresource.mipLevel) return false;
		m_sparse_retired.emplace_back(m_sparse.CurrentTick(), unbind.unbound); // free once rebound
		return true;
	});
	(deferred ? m_sparse_grows : m_sparse_binds).push_back({.image = image, .level = level});
	return m_sparse.CurrentTick() + 1;
}

uint64_t CommandScheduler::SendSparseGrows() {
	if (m_sparse_grows.empty()) return 0u;
	if (!m_graphics.sparse_queue) {
		// No bind thread: they go out before the next submit, as other binds (0: no wait needed).
		m_sparse_binds.insert(m_sparse_binds.end(), m_sparse_grows.begin(), m_sparse_grows.end());
		m_sparse_grows.clear();
		return 0u;
	}
	if (Threaded()) {
		// The value now, in order; the batch goes to the bind thread after the submits queued before it, as it
		// would without the recording thread.
		struct P {
			CommandScheduler* self;
			SparseBatch*      batch;
		};
		auto*      batch  = new SparseBatch(MakeSparse(m_sparse_grows, nullptr, 0));
		const auto signal = batch->signal;
		Cmd(*m_stream).PutBack(P {this, batch}, [](CmdBack&, const P& p) {
			std::unique_ptr<SparseBatch> owned(p.batch);
			p.self->DispatchSparse(std::move(*owned));
		});
		m_stream->Publish();
		return signal;
	}
	return SendSparse(m_sparse_grows, nullptr, 0);
}

uint64_t CommandScheduler::BindSparseTail(vk::Image image, const vk::SparseMemoryBind& tail) {
	m_sparse_binds.push_back({.image = image, .tail = tail, .is_tail = true});
	return m_sparse.CurrentTick() + 1;
}

uint64_t CommandScheduler::BindSparseBuffer(vk::Buffer buffer, const vk::SparseMemoryBind& page) {
	m_sparse_binds.push_back({.tail = page, .buffer = buffer});
	return m_sparse.CurrentTick() + 1;
}

uint64_t CommandScheduler::UnbindSparse(vk::Image image, const vk::SparseImageMemoryBind& level,
                                        VmaAllocation memory) {
	m_sparse_unbinds.push_back({.image = image, .level = level, .unbound = memory});
	return m_sparse.CurrentTick() + 1;
}

void CommandScheduler::ForgetSparse(vk::Image image, uint64_t batch) {
	std::erase_if(m_sparse_binds, [image](const SparseBind& bind) { return bind.image == image; });
	std::erase_if(m_sparse_grows, [image](const SparseBind& bind) { return bind.image == image; });
	std::erase_if(m_sparse_unbinds, [&](const SparseBind& bind) {
		if (bind.image != image) return false;
		m_sparse_retired.emplace_back(0, bind.unbound); // never sent: goes with the image
		return true;
	});
	// The image goes frames after its last use, behind which its unbinds waited: this is done.
	m_sparse.Wait(std::min(batch, m_sparse.CurrentTick() - 1));
}

uint64_t CommandScheduler::SendSparse(std::vector<SparseBind>& binds, vk::Semaphore wait,
                                      uint64_t wait_value) {
	auto       batch  = MakeSparse(binds, wait, wait_value);
	const auto signal = batch.signal;
	DispatchSparse(std::move(batch));
	return signal;
}

CommandScheduler::SparseBatch CommandScheduler::MakeSparse(std::vector<SparseBind>& binds, vk::Semaphore wait,
                                                           uint64_t wait_value) {
	// Values are given here, in order: a submit may wait for a batch the bind thread has not sent
	// yet (a timeline semaphore allows the wait before the signal).
	SparseBatch batch {.binds      = std::move(binds),
	                   .wait       = wait,
	                   .wait_value = wait_value,
	                   .previous   = m_sparse.CurrentTick() - 1};
	batch.signal = m_sparse.NextTick();
	binds.clear();
	for (const auto& bind: batch.binds) {
		if (bind.unbound != nullptr) m_sparse_retired.emplace_back(batch.signal, bind.unbound);
	}
	return batch;
}

void CommandScheduler::DispatchSparse(SparseBatch&& batch) {
	const auto signal = batch.signal;
	if (m_graphics.sparse_queue) {
		{
			std::lock_guard lock(m_sparse_mutex);
			m_sparse_batches.push_back(std::move(batch));
		}
		m_sparse_available.notify_one();
	} else {
		RunSparse(m_graphics.queue, batch);
		m_sparse_sent.store(signal);
	}
}

void CommandScheduler::SparseThread(std::stop_token stop) {
	for (;;) {
		SparseBatch batch;
		{
			std::unique_lock lock(m_sparse_mutex);
			m_sparse_available.wait(lock, stop, [this] { return !m_sparse_batches.empty(); });
			// Every batch handed over is sent, also at shutdown: submits may wait for it.
			if (m_sparse_batches.empty()) return;
			batch = std::move(m_sparse_batches.front());
			m_sparse_batches.pop_front();
		}
		RunSparse(m_graphics.sparse_queue, batch);
		m_sparse_sent.store(batch.signal);
		m_sparse_sent.notify_all();
	}
}

void CommandScheduler::RunSparse(vk::Queue queue, const SparseBatch& batch) {
	const auto&                                      binds = batch.binds;
	std::vector<vk::SparseImageMemoryBindInfo>       levels;
	std::vector<vk::SparseImageOpaqueMemoryBindInfo> tails;
	std::vector<vk::SparseBufferMemoryBindInfo>      buffers;
	for (const auto& bind: binds) {
		if (bind.buffer) {
			buffers.push_back({bind.buffer, 1, &bind.tail});
		} else if (bind.is_tail) {
			tails.push_back({bind.image, 1, &bind.tail});
		} else {
			levels.push_back({bind.image, 1, &bind.level});
		}
	}
	// Batches may finish out of order: each waits for the one before, so that a level unbound and
	// bound again ends up bound.
	std::array<vk::Semaphore, 2> waits {};
	std::array<uint64_t, 2>      wait_values {};
	uint32_t                     wait_count = 0;
	if (batch.previous != 0) {
		waits[wait_count]         = m_sparse.Handle();
		wait_values[wait_count++] = batch.previous;
	}
	if (batch.wait) {
		waits[wait_count]         = batch.wait;
		wait_values[wait_count++] = batch.wait_value;
	}
	const auto                      signal = batch.signal;
	const auto                      handle = m_sparse.Handle();
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount   = wait_count;
	timeline.pWaitSemaphoreValues      = wait_values.data();
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &signal;
	vk::BindSparseInfo info {};
	info.pNext                = &timeline;
	info.waitSemaphoreCount   = wait_count;
	info.pWaitSemaphores      = waits.data();
	info.imageBindCount       = static_cast<uint32_t>(levels.size());
	info.pImageBinds          = levels.data();
	info.imageOpaqueBindCount = static_cast<uint32_t>(tails.size());
	info.pImageOpaqueBinds    = tails.data();
	info.bufferBindCount      = static_cast<uint32_t>(buffers.size());
	info.pBufferBinds         = buffers.data();
	info.signalSemaphoreCount = 1;
	info.pSignalSemaphores    = &handle;
	const auto result         = queue.bindSparse(1, &info, nullptr);
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueBindSparse", result, signal, 0, 0, 0, 0, 0, 0, 0);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::FreeSparse() {
	if (m_sparse_retired.empty()) return;
	m_sparse.Refresh();
	std::erase_if(m_sparse_retired, [this](const auto& retired) {
		if (!m_sparse.IsFree(retired.first)) return false;
		vmaFreeMemory(m_graphics.allocator, retired.second);
		return true;
	});
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
	if (m_stream != nullptr) {
		m_stream->Stop();
	}
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		WaitSent(tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		WaitSent(tick);
		m_master.Wait(tick);
	}
}

// A driver wait for a value no submit has sent yet is legal and silent: it would park behind the recording thread with
// nothing in a stack to say so. Wait for the send first; a blocked recording thread then shows here.
void CommandScheduler::WaitSent(uint64_t tick) {
	for (auto sent = m_sent_tick.load(std::memory_order_acquire); sent < tick;
	     sent      = m_sent_tick.load(std::memory_order_acquire)) {
		m_sent_tick.wait(sent, std::memory_order_acquire);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Thousands of calls a frame, nearly all with nothing pending: no lock, and only every eighth call asks the driver
	// for the tick (IsFree asks for itself when it needs a newer one). An operation queued by now waits for a later tick.
	if (m_pending_count.load(std::memory_order_acquire) == 0u) {
		thread_local uint32_t calls = 0;
		if ((++calls & 7u) == 0u) m_master.Refresh();
		return;
	}
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_count.fetch_sub(1u, std::memory_order_relaxed);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		if (!priority) m_pending_count.fetch_add(1u, std::memory_order_release);
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

uint64_t CommandScheduler::LastPriorityTick() {
	std::lock_guard lock(m_operation_mutex);
	uint64_t tick = m_priority_active ? m_priority_active_tick : 0;
	if (!m_priority_operations.empty()) {
		tick = std::max(tick, m_priority_operations.back().tick);
	}
	return tick;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	if (m_stream != nullptr && m_threaded_wanted != Threaded()) SwitchRecording();
	// Ticks go out in submit order: this command buffer is submitted with the current one.
	const auto tick  = m_master.CurrentTick();
	m_command.m_open = true;
	if (Threaded()) {
		struct P {
			CommandScheduler* self;
			uint64_t          tick;
		};
		Cmd(*m_stream).PutBack(P {this, tick}, [](CmdBack& back, const P& p) {
			back.command = p.self->m_command_pool.Commit(p.tick);
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(back.command.begin(&begin_info) != vk::Result::eSuccess);
		});
	} else {
		m_command.m_buffer = m_command_pool.Commit(tick);
		m_command.Begin();
	}
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF((submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	         submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores) &&
	        !KYTY_UPSTREAM_PATH("45b35fafb"));

	if (m_end_hook) {
		m_end_hook();
	}
	m_recorded_work = 0;
	m_command.End();
	if (Threaded()) {
		return SubmitThreaded(submit);
	}
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		if (!m_sparse_binds.empty()) {
			// Levels this command buffer uploads to have memory before it runs.
			RequireSparse(SendSparse(m_sparse_binds, nullptr, 0));
		}
		if (m_sparse_required != 0) {
			// The queue waits only for a batch already sent: NVIDIA's bind waits for this queue in turn
			// (a submit waiting for an unsent bind deadlocked).
			for (auto sent = m_sparse_sent.load(); sent < m_sparse_required; sent = m_sparse_sent.load()) {
				m_sparse_sent.wait(sent);
			}
			submit.AddWait(m_sparse.Handle(), std::exchange(m_sparse_required, 0));
		}
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
		if (result == vk::Result::eSuccess && !m_sparse_unbinds.empty()) {
			// Levels draws up to here may sample keep their memory until those draws are done.
			(void)SendSparse(m_sparse_unbinds, m_master.Handle(), tick);
		}
	}
	m_sent_tick.store(tick, std::memory_order_release);
	m_sent_tick.notify_all();
	FreeSparse();
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_command.m_buffer = nullptr;
	m_command.m_open   = false;
	return tick;
}

uint64_t CommandScheduler::SubmitThreaded(SubmitInfo& submit) {
	// Everything the direct Submit decides, decided here in the same order; the recording thread only sends.
	auto job = std::make_unique<SubmitJob>();
	if (!m_sparse_binds.empty()) {
		job->binds = std::make_unique<SparseBatch>(MakeSparse(m_sparse_binds, nullptr, 0));
		RequireSparse(job->binds->signal);
	}
	if (m_sparse_required != 0) {
		job->sparse_required = std::exchange(m_sparse_required, 0);
		submit.AddWait(m_sparse.Handle(), job->sparse_required);
	}
	job->tick = m_master.NextTick();
	submit.AddSignal(m_master.Handle(), job->tick);
	if (!m_sparse_unbinds.empty()) {
		job->unbinds = std::make_unique<SparseBatch>(MakeSparse(m_sparse_unbinds, m_master.Handle(), job->tick));
	}
	job->submit          = submit;
	job->debug_op        = m_command.m_debug_op;
	job->debug_submit_id = m_command.m_debug_submit_id;
	job->debug_arg0      = m_command.m_debug_arg0;
	job->debug_arg1      = m_command.m_debug_arg1;
	job->debug_arg2      = m_command.m_debug_arg2;
	job->debug_arg3      = m_command.m_debug_arg3;
	job->debug_arg4      = m_command.m_debug_arg4;
	const auto tick      = job->tick;

	struct P {
		CommandScheduler* self;
		SubmitJob*        job;
	};
	Cmd(*m_stream).PutBack(P {this, job.release()}, [](CmdBack& back, const P& p) {
		std::unique_ptr<SubmitJob> owned(p.job);
		auto&                      self     = *p.self;
		auto&                      job      = *owned;
		auto&                      graphics = self.m_graphics;
		EXIT_NOT_IMPLEMENTED(back.command.end() != vk::Result::eSuccess);
		vk::Result result;
		{
			Common::LockGuard lock(graphics.queue_mutex);
			if (job.binds != nullptr) {
				// Levels this command buffer uploads to have memory before it runs.
				self.DispatchSparse(std::move(*job.binds));
			}
			if (job.sparse_required != 0) {
				// The queue waits only for a batch already sent (see Submit).
				for (auto sent = self.m_sparse_sent.load(); sent < job.sparse_required;
				     sent      = self.m_sparse_sent.load()) {
					self.m_sparse_sent.wait(sent);
				}
			}
			const auto&                     submit = job.submit;
			vk::TimelineSemaphoreSubmitInfo timeline_info {};
			timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
			timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
			timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
			timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();
			vk::SubmitInfo submit_info {};
			submit_info.pNext                = &timeline_info;
			submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
			submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
			submit_info.pWaitDstStageMask    = submit.wait_stages.data();
			submit_info.commandBufferCount   = 1;
			submit_info.pCommandBuffers      = &back.command;
			submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
			submit_info.pSignalSemaphores    = submit.signal_semaphores.data();
			result                           = graphics.queue.submit(1, &submit_info, nullptr);
			if (result == vk::Result::eSuccess && job.unbinds != nullptr) {
				// Levels draws up to here may sample keep their memory until those draws are done.
				self.DispatchSparse(std::move(*job.unbinds));
			}
		}
		self.m_sent_tick.store(job.tick, std::memory_order_release);
		self.m_sent_tick.notify_all();
		Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
		if (result != vk::Result::eSuccess) {
			ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit_id, job.debug_arg0,
			                  job.debug_arg1, job.debug_arg2, job.debug_arg3, job.debug_arg4);
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		back.command = nullptr;
	});
	m_stream->Publish();
	FreeSparse();
	m_command.m_open = false;

	// At most four command buffers wait for the recording thread: it bounds the memory the stream holds and how far
	// a wait for a tick lags behind its submit.
	m_submit_marks.push_back(m_stream->Appended());
	while (!m_submit_marks.empty() && m_submit_marks.front() <= m_stream->Executed()) m_submit_marks.pop_front();
	if (m_submit_marks.size() > 4) {
		m_backlog_waits++;
		m_stream->WaitExecuted(m_submit_marks.front());
		m_submit_marks.pop_front();
	}
	ReportRecording();
	return tick;
}

// Between command buffers only (BeginCommand), so each one is recorded one way from begin to submit.
void CommandScheduler::SwitchRecording() {
	if (Threaded()) {
		// Everything queued, the last submit and heap writes included, goes out before direct recording starts.
		m_stream->Drain();
		m_submit_marks.clear();
	}
	m_threaded.store(m_threaded_wanted, std::memory_order_relaxed);
	m_recording_switches++;
	if (m_recording_switches <= 4) LOGF("Recording thread: %s\n", m_threaded_wanted ? "on" : "off");
}

void CommandScheduler::EnableRecordingThread(bool verify, bool on) {
	EXIT_IF(m_stream != nullptr || !m_command.IsInvalid());
	m_back.device    = m_graphics.device;
	m_back.scheduler = this;
	m_stream         = std::make_unique<Common::PacketStream>(
        &m_back, verify, 1024, [](void* /*context*/, Common::PacketStream::Run run, size_t bytes) {
            static std::atomic<uint32_t> logged = 0;
            if (logged.fetch_add(1) < 32) {
                LOGF("Recording thread: a packet changed between queueing and recording (run %p, %zu bytes)\n",
                     reinterpret_cast<void*>(run), bytes);
            }
        });
	m_stream->Start();
	struct P {};
	Cmd(*m_stream).PutBack(P {}, [](CmdBack&, const P&) { KYTY_PROFILER_THREAD("Thread_GpuRecord"); });
	m_stream->Publish();
	m_threaded.store(on, std::memory_order_relaxed);
	m_threaded_wanted = on;
	LOGF("Recording thread: started, %s%s\n", on ? "on" : "off", verify ? ", verify" : "");
}

Cmd CommandScheduler::Recorder() {
	return Threaded() ? Cmd(*m_stream) : Cmd(m_command.m_buffer);
}

vk::CommandBuffer CommandScheduler::DrainRecording(const char* file, uint32_t line) {
	EXIT_IF(!Threaded());
	m_drains++;
	std::string site = file;
	if (const auto slash = site.find_last_of("/\\"); slash != std::string::npos) site.erase(0, slash + 1);
	m_drain_sites[site + ":" + std::to_string(line)]++;
	m_stream->Drain();
	return m_back.command;
}

// Every 5 s with the recording thread: command buffers, packets, drains (raw Handle() uses, where the GPU thread
// waited for the recording thread) by site, waits for a backlog or a block, and packets that changed (verify).
void CommandScheduler::ReportRecording() {
	m_report_buffers++;
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count());
	if (m_report_time == 0) m_report_time = now;
	if (now - m_report_time < 5000) return;
	std::vector<std::pair<uint64_t, std::string>> sites;
	for (const auto& [site, count]: m_drain_sites) sites.emplace_back(count, site);
	std::sort(sites.rbegin(), sites.rend());
	std::string top;
	for (size_t i = 0; i < sites.size() && i < 6; i++) {
		top += " " + sites[i].second + "=" + std::to_string(sites[i].first);
	}
	const auto packets = m_stream->Appended();
	LOGF("Recording thread (5 s): %" PRIu64 " command buffers, %" PRIu64 " packets, %" PRIu64
	     " drains%s, %" PRIu64 " backlog waits, %" PRIu64 " block waits, %" PRIu64 " changed, %" PRIu64
	     " queued from other threads\n",
	     m_report_buffers, packets - m_report_packets, m_drains - m_report_drains, top.c_str(),
	     m_backlog_waits - m_report_waits, m_stream->BackpressureWaits(), m_stream->Mismatches(),
	     m_stream->ForeignAppends());
	m_drain_sites.clear();
	m_report_buffers = 0;
	m_report_packets = packets;
	m_report_drains  = m_drains;
	m_report_waits   = m_backlog_waits;
	m_report_time    = now;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
