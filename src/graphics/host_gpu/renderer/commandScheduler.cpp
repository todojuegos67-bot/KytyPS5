#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "live-switch.h"
#include "mode-toggle.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionManager.h"
#include "anti-lag.h"
#include "async-upload.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#include "device-fault.h"
#endif
#include "live-trace-gpu.h"
#include "local-platform.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <optional>

extern "C" {
// Dispatches recorded per submission (KYTY_DISPATCH_BATCH).
volatile std::atomic_uint32_t kyty_local_dispatch_batch {128};
// Draws recorded per submission (KYTY_DRAW_BATCH, 0: no limit): about 0.5 ms of translated draws.
volatile std::atomic_uint32_t kyty_local_draw_batch {256};
// Buffer uploads of pages dirty since before the open command buffer began go into its upload prologue
// (CommandScheduler::UploadPrologue).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_upload_prologue {1};
// 0: refresh the master timeline before every drain attempt.
// 1: prove the pending-operation queue empty before the timeline query and the
//    operation lock. Draws and dispatches call this with nothing to retire.
volatile std::atomic<uint32_t> kyty_local_pending_drain_mode {1};
// Flush points nothing waits on (a label without an interrupt, the end of a queue's slice) leave the open buffer open
// while the GPU still runs work it was handed (CommandScheduler::DeferrableFlush; KYTY_FLUSH_DEFER=0: flush at once).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_flush_defer {2};
}

extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (graphics/guest_gpu/speculation.cpp)
}

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;
static thread_local CommandScheduler::Recorder* t_recorder           = nullptr;
// (KYTY_SPECULATE unset or 0: no recorder on any thread. The global is read first, so mode 0's per-draw and per-bind
// callers read no thread-local, as the base.)
static CommandScheduler::Recorder* Rec() noexcept {
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 ? t_recorder : nullptr;
}

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
	if (result == vk::Result::eErrorDeviceLost) DeviceFault::Report();
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

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
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
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this), m_prologue(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_tick_monitor([this](std::stop_token stop) {
	      LocalPlatform::SetThreadName("Kyty.TickMon");
	      {
		      vk::QueryPoolCreateInfo info {};
		      info.queryType  = vk::QueryType::eTimestamp;
		      info.queryCount = LiveTrace::TimestampSlots * 2u;
		      vk::QueryPool pool {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &pool) == vk::Result::eSuccess) {
			      LiveTrace::g_timestamp_period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
			      LiveTrace::g_timestamp_pool   = static_cast<VkQueryPool>(pool);
		      }
		      // Per-draw marks (`tracem`): reset on the host, read after tracing stops.
		      info.queryCount = LiveTrace::MarkSlots;
		      vk::QueryPool marks {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &marks) == vk::Result::eSuccess) {
			      m_graphics.device.resetQueryPool(marks, 0, LiveTrace::MarkSlots);
			      LiveTrace::g_mark_pool = static_cast<VkQueryPool>(marks);
			      static vk::Device device = m_graphics.device;
			      LiveTrace::g_read_marks = [](uint32_t count) {
				      const auto            pool = static_cast<VkQueryPool>(LiveTrace::g_mark_pool);
				      std::vector<uint64_t> values(count);
				      if (count != 0 &&
				          device.getQueryPoolResults(pool, 0, count, count * sizeof(uint64_t), values.data(),
				                                     sizeof(uint64_t),
				                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
				              vk::Result::eSuccess) {
					      for (uint32_t i = 0; i < count; ++i)
						      LiveTrace::Append(LiveTrace::GpuMarkValue, i,
						                        static_cast<uint64_t>(static_cast<double>(values[i]) *
						                                              LiveTrace::g_timestamp_period));
				      }
				      device.resetQueryPool(pool, 0, LiveTrace::MarkSlots);
			      };
		      }
	      }
	      uint64_t next = 0;
	      while (!stop.stop_requested()) {
		      if (!LiveTrace::g_on.load(std::memory_order_relaxed)) {
			      next = 0;
			      std::this_thread::sleep_for(std::chrono::milliseconds(5));
			      continue;
		      }
		      if (next == 0) next = m_master.KnownGpuTick() + 1;
		      vk::SemaphoreWaitInfo wait {};
		      const auto            semaphore = m_master.Handle();
		      wait.semaphoreCount           = 1;
		      wait.pSemaphores              = &semaphore;
		      wait.pValues                  = &next;
		      if (m_graphics.device.waitSemaphores(&wait, 2000000) == vk::Result::eSuccess) {
			      LiveTrace::Event(LiveTrace::TickDone, next);
			      auto& mapped = LiveTrace::g_tick_slot[next % LiveTrace::TimestampSlots];
			      if (const auto slot = mapped.exchange(0); slot != 0 && LiveTrace::g_timestamp_pool != nullptr) {
				      uint64_t times[2] {};
				      if (m_graphics.device.getQueryPoolResults(
				              static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool), (slot - 1u) * 2u, 2u,
				              sizeof(times), times, sizeof(uint64_t), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
					      LiveTrace::Event(LiveTrace::GpuSpan, next);
					      LiveTrace::Event(LiveTrace::GpuSpanNs,
					                       static_cast<uint64_t>(static_cast<double>(times[0]) * LiveTrace::g_timestamp_period),
					                       static_cast<uint64_t>(static_cast<double>(times[1]) * LiveTrace::g_timestamp_period));
				      }
			      }
			      ++next;
		      }
	      }
      }) {}

void CommandScheduler::CompleteDispatch() {
	// Expose command-processor/GPU overlap without a host fence. Bound the
	// recording batch; resource retirement still uses the submission timeline.
	// The boundary costs one all-commands dependency, which lets no part of the
	// next batch begin before this one drains, so its size trades that drain
	// against how early the queue receives work. Dispatches inside a batch keep
	// the same chain barriers at any size. (A recorder's buffers are submitted later, whole.)
	if (Rec() != nullptr) return;
	if (FlushedDeferred(4)) return;
	const auto batch = std::clamp(kyty_local_dispatch_batch.load(std::memory_order_relaxed), 1u, 65536u);
	if (++m_recorded_dispatches < batch) {
		return;
	}
	// While the GPU still runs work handed to it the batch goes on (GpuBusy, asked at the batch's end and every 4
	// dispatches after it), up to 8 batches.
	if (m_recorded_dispatches < 8 * batch &&
	    ((m_recorded_dispatches != batch && (m_recorded_dispatches & 3u) != 0) || GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// As CompleteDispatch for draws: a draw-heavy guest command buffer reaches the GPU in parts while it is still being
// translated. Whole, the GPU idled through its translation (3-4.5 ms for the big passes) and ran it after, late into the
// next frame, where the next frame's culling readbacks waited for it.
void CommandScheduler::CompleteDraws(uint32_t draws) {
	if (Rec() == nullptr && FlushedDeferred(draws)) return;
	const auto batch = kyty_local_draw_batch.load(std::memory_order_relaxed);
	if (Rec() != nullptr || batch == 0 || (m_recorded_draws += draws) < batch) {
		return;
	}
	// As CompleteDispatch's: on while the GPU is busy (asked at the batch's end and about every 16 draws after it), up
	// to 8 batches.
	const bool end = m_recorded_draws - draws < batch;
	if (m_recorded_draws < 8 * batch && ((!end && (m_recorded_draws & 15u) >= draws && draws < 16) || GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// A batch boundary gives the GPU work early, where it would idle through the batch's translation; while it still runs
// work it was handed, the boundary gives it nothing it lacks and costs a drain on the GPU (~8 us between back-to-back
// command buffers) and a submission on the recording worker (~13 us, half of it in the kernel; at 1-1 standing 141
// submissions a frame were ~1 ms of the GPU's and ~1.8 ms of the worker's, whose lag left the GPU idle ~1.35 ms a frame
// with work already submitted). The batch then goes on until the GPU has run what it was handed. Same-process A/B, 3-4
// rounds each: 1-1 standing 60.2 -> 62.4 fps, low1 52.8 -> 56.0 (a boot paced at 60: low1 54.2 -> 55.6); Latria spin
// 74.2 -> 75.9, low1 53.9 -> 55.4; Shrine standing unchanged (75.3 / 75.3).
bool CommandScheduler::GpuBusy() {
	const auto handed = m_driver_tick.load(std::memory_order_acquire);
	if (m_master.IsFree(handed)) return false;
	m_master.Refresh();
	return !m_master.IsFree(handed);
}

// A flush point nothing waits on (a label without an interrupt, the end of a queue's slice: CommandProcessor::
// BufferFlushDeferrable) while the GPU has kyty_local_flush_defer (or more) submissions still to run: the buffer stays
// open, and goes with the next flush, or once the GPU is down to its last submission (FlushedDeferred: asked every 16
// draws or 4 dispatches, so it never runs dry waiting for this one), or before the GPU thread waits for work
// (FlushDeferred). Each submission costs the GPU a drain between back-to-back command buffers (~8 us) and the recording
// worker a submission; ~118 a frame at 1-1, mostly labels and the async queues' slices. A flush while the GPU merely
// still ran something (GpuBusy, as CompleteDraws asks) halved the submissions but starved the GPU after short ones.
// (Ended as a submission ends it: the render pass and a compute chain's pending barrier, so the commands after it are
// recorded as they would be into the next buffer.)
void CommandScheduler::DeferrableFlush() {
	if (const auto backlog = kyty_local_flush_defer.load(std::memory_order_relaxed);
	    backlog != 0 && Rec() == nullptr && !m_command.IsInvalid() && BacklogAtLeast(backlog)) {
		m_command.EndRendering();
		(void)m_command.Handle();
		m_flush_deferred = true;
		return;
	}
	Flush();
}

void CommandScheduler::FlushDeferred() {
	if (m_flush_deferred) Flush();
}

// While a flush is deferred, every 16 draws (or 4 dispatches: `weight`) whether the GPU is down to its last submission:
// then the open buffer is submitted (true).
bool CommandScheduler::FlushedDeferred(uint32_t weight) {
	if (!m_flush_deferred) return false;
	if ((m_deferred_checks += weight) < 16) return false;
	m_deferred_checks = 0;
	if (BacklogAtLeast(2)) return false;
	Flush();
	return true;
}

// The GPU's tick as last read is never ahead of it: the backlog it gives is never short. Under `count` that way, under
// `count` for sure, without a timeline read (a system call on Linux' RADV, ~9 us; these polls asked ~400 times a frame
// at 1-1, issue #80).
bool CommandScheduler::BacklogAtLeast(uint64_t count) {
	const auto backlog = [this] {
		const auto submitted = m_master.CurrentTick() - 1;
		const auto done      = m_master.KnownGpuTick();
		return submitted > done ? submitted - done : 0;
	};
	if (backlog() < count) return false;
	m_master.Refresh();
	return backlog() >= count;
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
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
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	if (Rec() != nullptr) {
		BeginRecording(*Rec(), registers, user_config, shaders);
		return;
	}
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
	auto& command = Rec() != nullptr ? *Rec()->command : m_command;
	if (Active() && !command.IsInvalid()) {
		command.EndRendering();
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
	if (Rec() != nullptr) throw RecorderRefusal {"FlushAndWait"};
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	if (Rec() != nullptr) throw RecorderRefusal {"Finish"};
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	if (Rec() != nullptr) {
		// (A recorder's own work is not submitted: waiting for it would never end.)
		if (tick >= PendingTick) throw RecorderRefusal {"Wait"};
		m_master.Wait(tick);
		return;
	}
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	if (Rec() != nullptr) return; // (the owning thread retires them)
	// The pending queue is empty for the large majority of draws and dispatches.
	// Refresh() issues a Vulkan timeline query and the drain takes the operation
	// mutex, so let an atomic count prove there is nothing to retire first. The
	// count is only a hint: the queue is still re-checked under the lock.
	if (kyty_local_pending_drain_mode.load(std::memory_order_relaxed) != 0) {
		if (m_pending_operation_count.load(std::memory_order_acquire) == 0) return;
		// Nor while the oldest waits for a submission the driver has not been handed yet (the open buffer, or one still
		// with the recording worker: the GPU cannot have run it), and no timeline read when the GPU was seen past it
		// already (a system call each on Linux' RADV, issue #80).
		const auto front = m_pending_front_tick.load(std::memory_order_acquire);
		if (front > m_driver_tick.load(std::memory_order_acquire)) return;
		if (!m_master.IsFree(front)) m_master.Refresh();
	} else {
		m_master.Refresh();
	}
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
			m_pending_front_tick.store(m_pending_operations.empty() ? UINT64_MAX : m_pending_operations.front().tick,
			                           std::memory_order_relaxed);
			m_pending_operation_count.fetch_sub(1, std::memory_order_relaxed);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	if (Rec() != nullptr) throw RecorderRefusal {"DeferOperation"};
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		if (m_pending_operations.size() == 1) m_pending_front_tick.store(CurrentTick(), std::memory_order_relaxed);
		m_pending_operation_count.fetch_add(1, std::memory_order_release);
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	if (Rec() != nullptr) throw RecorderRefusal {"DeferPriorityOperation"};
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
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
		LiveTrace::Event(LiveTrace::GpuDone, operation.tick);
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

bool CommandScheduler::Active() const noexcept {
	return (Rec() != nullptr ? *Rec()->command : m_command).m_registers != nullptr;
}

uint64_t CommandScheduler::CurrentTick() const noexcept {
	return Rec() != nullptr ? Rec()->pending_tick : m_master.CurrentTick();
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return Rec() != nullptr ? *Rec()->command : m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid() || !m_prologue.IsInvalid());
	++m_command_serial;
	m_command_clock    = g_cpu_dirty_clock.load(std::memory_order_acquire);
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	LiveTrace::g_mark_command = static_cast<VkCommandBuffer>(m_command.m_buffer);
	return m_command;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace {
// vkEndCommandBuffer + vkQueueSubmit executed by the recording worker, in order with
// the commands it has replayed. The tick is assigned on the producer as before.
struct DeferredSubmit {
	uint32_t             ended_mask; // bit i: commands[i] recorded and ended on another thread
	uint32_t             count;
	VkQueue              queue;
	Common::Mutex*       queue_mutex;
	uint32_t             waits, signals;
	VkSemaphore          wait_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             wait_ticks[SubmitInfo::MaxSemaphores];
	VkPipelineStageFlags wait_stages[SubmitInfo::MaxSemaphores];
	VkSemaphore          signal_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             signal_ticks[SubmitInfo::MaxSemaphores];
	uint64_t             tick;
	uint64_t             upload_sequence; // KYTY_ASYNC_UPLOAD copies these buffers read
	uint32_t             timestamp_slot;  // live trace (the last buffer the stream ends)
	std::atomic<uint64_t>* driver_tick;   // CommandScheduler::m_driver_tick (GpuBusy)
	// Last: the stream copies count of them (offsetof + count entries), not MaxSubmitEntries.
	VkCommandBuffer      commands[CommandScheduler::MaxSubmitEntries];
};
static_assert(CommandScheduler::MaxSubmitEntries <= 32, "DeferredSubmit::ended_mask");
void ReplaySubmit(std::span<const LocalVulkanRecording::Segment> segments,
                  const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& submit = *static_cast<const DeferredSubmit*>(segments[0].data);
	AsyncUpload::Wait(submit.upload_sequence);
	LiveTrace::Event(LiveTrace::GpuSubmit, submit.tick, 1);
	for (uint32_t i = 0; i < submit.count; ++i) {
		if ((submit.ended_mask >> i) & 1u) continue;
		if (submit.timestamp_slot != UINT32_MAX && i + 1 == submit.count && LiveTrace::g_timestamp_pool != nullptr) {
			dispatch.vkCmdWriteTimestamp(submit.commands[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			                             static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool),
			                             submit.timestamp_slot * 2u + 1u);
			LiveTrace::g_tick_slot[submit.tick % LiveTrace::TimestampSlots].store(submit.timestamp_slot + 1u,
			                                                                        std::memory_order_release);
		}
		if (dispatch.vkEndCommandBuffer(submit.commands[i]) != VK_SUCCESS) {
			EXIT("deferred vkEndCommandBuffer failed, tick=%" PRIu64 "\n", submit.tick);
		}
	}
	VkTimelineSemaphoreSubmitInfo timeline {};
	timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timeline.waitSemaphoreValueCount   = submit.waits;
	timeline.pWaitSemaphoreValues      = submit.wait_ticks;
	timeline.signalSemaphoreValueCount = submit.signals;
	timeline.pSignalSemaphoreValues    = submit.signal_ticks;
	VkSubmitInfo info {};
	info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	info.pNext                = &timeline;
	info.waitSemaphoreCount   = submit.waits;
	info.pWaitSemaphores      = submit.wait_semaphores;
	info.pWaitDstStageMask    = submit.wait_stages;
	info.commandBufferCount   = submit.count;
	info.pCommandBuffers      = submit.commands;
	info.signalSemaphoreCount = submit.signals;
	info.pSignalSemaphores    = submit.signal_semaphores;
	for (uint32_t i = 0; i < submit.signals; ++i) {
		if (submit.signal_ticks[i] != submit.tick) continue;
		DeviceFault::NoteSubmit((uint64_t)(submit.signal_semaphores[i]), submit.tick, submit.commands, submit.count, // NOLINT
		                        submit.wait_semaphores, submit.wait_ticks, submit.waits);
		break;
	}
	VkResult result;
	{
		Common::LockGuard lock(*submit.queue_mutex);
		result = Libs::Graphics::AntiLag::QueueSubmit(submit.queue, info, dispatch.vkQueueSubmit, dispatch.vkQueueSubmit2);
	}
	if (result != VK_SUCCESS) {
		if (result == VK_ERROR_DEVICE_LOST) DeviceFault::Report();
		EXIT("deferred vkQueueSubmit failed: %d, tick=%" PRIu64 "\n", static_cast<int>(result), submit.tick);
	}
	submit.driver_tick->store(submit.tick, std::memory_order_release);
	LocalVulkanRecording::NoteDeferredSubmitDone();
}
// A recorded buffer dropped before submission, in order with the commands replayed into it.
void ReplayDiscard(std::span<const LocalVulkanRecording::Segment> segments,
                   const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto command = *static_cast<const VkCommandBuffer*>(segments[0].data);
	DeviceFault::NoteDiscard(command);
	if (dispatch.vkResetCommandBuffer(command, 0) != VK_SUCCESS) {
		EXIT("deferred vkResetCommandBuffer failed\n");
	}
}
} // namespace
#endif

vk::CommandBuffer CommandScheduler::UploadPrologue(uint64_t last_dirty, uint64_t written_serial) {
	if (Rec() != nullptr || kyty_local_upload_prologue.load(std::memory_order_relaxed) == 0 ||
	    m_command.IsInvalid() || last_dirty > m_command_clock || written_serial == m_command_serial)
		return nullptr;
	if (m_prologue.IsInvalid()) {
		// (Committed for the open buffer's submission: the pool reuses it once that completes.)
		m_prologue.m_buffer = m_command_pool.Commit();
		m_prologue.Begin();
		// The copies after everything submitted before (which may read or write their destinations).
		VulkanMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		                                    {}, 1, &before, 0, nullptr, 0, nullptr);
	}
	return m_prologue.Handle();
}

CommandScheduler::SubmitEntry CommandScheduler::ClosePrologue() {
	if (m_prologue_hook) m_prologue_hook(m_prologue.Handle());
	// Everything after the copies (the open buffer's commands) sees them.
	VulkanMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {},
	                                    1, &after, 0, nullptr, 0, nullptr);
	const SubmitEntry entry {m_prologue.m_buffer, false, UINT32_MAX};
	m_prologue.m_buffer = nullptr;
	return entry;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	if (Rec() != nullptr) throw RecorderRefusal {"Submit"};
	EXIT_IF(m_command.IsInvalid());
	m_command.EndRendering();
	if (m_epilogue_hook) m_epilogue_hook(m_command); // (KYTY_FEEDBACK_BATCH)
	// A compute chain's pending barrier closes this buffer (left pending, the next Begin recorded it into a buffer
	// that had not begun).
	(void)m_command.Handle();
	// (After the recorded buffers held for it: SubmitRecorded leaves room for these two.)
	std::array<SubmitEntry, MaxSubmitEntries> entries {};
	size_t                                    n = 0;
	for (size_t i = 0; i < m_batched; ++i) entries[n++] = m_batch[i];
	m_batched = 0;
	if (!m_prologue.IsInvalid()) entries[n++] = ClosePrologue();
	entries[n++]          = {m_command.m_buffer, false, m_command.m_timestamp_slot};
	const auto tick       = SubmitBuffers({entries.data(), n}, submit);
	m_command.m_buffer    = nullptr;
	m_recorded_dispatches = 0;
	m_recorded_draws      = 0;
	m_flush_deferred      = false;
	m_spliced             = false;
	return tick;
}

uint64_t CommandScheduler::SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit) {
	// The work may run once submitted: the pages it writes are read-protected first.
	PageManager::FlushDeferredProtection();
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	EXIT_IF(m_graphics.queue == nullptr || entries.empty() || entries.size() > MaxSubmitEntries);

#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (LocalVulkanRecording::DeferredSubmitEnabled()) {
		DeferredSubmit deferred {};
		// This thread alone allocates ticks and the worker submits them in stream order: no queue
		// lock here (it only waited for the worker's vkQueueSubmit, 1.2% of the render thread).
		deferred.tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), deferred.tick);
		deferred.count          = static_cast<uint32_t>(entries.size());
		deferred.timestamp_slot = UINT32_MAX;
		deferred.driver_tick    = &m_driver_tick;
		for (size_t i = 0; i < entries.size(); ++i) {
			deferred.commands[i] = entries[i].buffer;
			if (entries[i].ended) deferred.ended_mask |= 1u << i;
			if (!entries[i].ended) deferred.timestamp_slot = entries[i].timestamp_slot;
		}
		deferred.upload_sequence = AsyncUpload::SubmitSequence();
		deferred.queue           = m_graphics.queue;
		deferred.queue_mutex     = &m_graphics.queue_mutex;
		deferred.waits           = submit.num_wait_semaphores;
		deferred.signals         = submit.num_signal_semaphores;
		for (uint32_t i = 0; i < deferred.waits; ++i) {
			deferred.wait_semaphores[i] = submit.wait_semaphores[i];
			deferred.wait_ticks[i]      = submit.wait_ticks[i];
			deferred.wait_stages[i]     = static_cast<VkPipelineStageFlags>(submit.wait_stages[i]);
		}
		for (uint32_t i = 0; i < deferred.signals; ++i) {
			deferred.signal_semaphores[i] = submit.signal_semaphores[i];
			deferred.signal_ticks[i]      = submit.signal_ticks[i];
		}
		const LocalVulkanRecording::Segment segments[] {
		    {&deferred, offsetof(DeferredSubmit, commands) + deferred.count * sizeof(VkCommandBuffer)}};
		LocalVulkanRecording::NoteDeferredSubmitQueued();
		if (!LocalVulkanRecording::EnqueueDeferred(ReplaySubmit, segments, true)) {
			// Keep order: everything recorded so far, then this exact submit.
			LocalVulkanRecording::Drain();
			ReplaySubmit(segments, VULKAN_HPP_DEFAULT_DISPATCHER);
		}
		LiveTrace::Event(LiveTrace::GpuSubmit, deferred.tick);
		return deferred.tick;
	}
#endif
	std::array<vk::CommandBuffer, MaxSubmitEntries> buffers {};
	for (size_t i = 0; i < entries.size(); ++i) {
		buffers[i] = entries[i].buffer;
		if (!entries[i].ended) EXIT_NOT_IMPLEMENTED(buffers[i].end() != vk::Result::eSuccess);
	}
	auto& graphics = m_graphics;
	AsyncUpload::Drain();

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
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
		submit_info.commandBufferCount   = static_cast<uint32_t>(entries.size());
		submit_info.pCommandBuffers      = buffers.data();
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
		DeviceFault::NoteSubmit((uint64_t)(static_cast<VkSemaphore>(m_master.Handle())), tick, // NOLINT
		                        reinterpret_cast<const VkCommandBuffer*>(buffers.data()),
		                        static_cast<uint32_t>(entries.size()),
		                        reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()), submit.wait_ticks.data(),
		                        submit.num_wait_semaphores);
#endif

		result = static_cast<vk::Result>(Libs::Graphics::AntiLag::QueueSubmit(
		    static_cast<VkQueue>(graphics.queue), *reinterpret_cast<const VkSubmitInfo*>(&submit_info),
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit, VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2));
		m_driver_tick.store(tick, std::memory_order_release);
	}

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	return tick;
}

std::unique_ptr<CommandScheduler::Recorder> CommandScheduler::CreateRecorder() {
	auto                      recorder = std::make_unique<Recorder>();
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = m_graphics.queue_family;
	create.flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	EXIT_NOT_IMPLEMENTED(m_graphics.device.createCommandPool(&create, nullptr, &recorder->pool) != vk::Result::eSuccess);
	recorder->command = std::unique_ptr<CommandBuffer>(new CommandBuffer(*this));
	return recorder;
}

void CommandScheduler::DestroyRecorder(std::unique_ptr<Recorder> recorder) {
	if (!recorder) return;
	for (const auto& [buffer, tick]: recorder->buffers)
		if (tick != UINT64_MAX && tick != 0) m_master.Wait(tick);
	m_graphics.device.destroyCommandPool(recorder->pool, nullptr);
}

void CommandScheduler::SetThreadRecorder(Recorder* recorder) noexcept {
	t_recorder = recorder;
}

CommandScheduler::Recorder* CommandScheduler::ThreadRecorder() noexcept {
	return Rec();
}

void CommandScheduler::BeginRecording(Recorder& recorder, HW::Context& registers, HW::UserConfig& user_config,
                                      HW::Shader& shaders) {
	auto& command = *recorder.command;
	command.Bind(registers, user_config, shaders);
	if (!command.IsInvalid()) return;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (recorder.packets) {
		// Worker packets: no command buffer; the commands go to this thread's passive producer, against the placeholder.
		LocalVulkanRecording::BeginSegment();
		recorder.open    = Recorder::PacketsOpen;
		command.m_buffer = vk::CommandBuffer {static_cast<VkCommandBuffer>(LocalVulkanRecording::PlaceholderCommandBuffer())};
		command.Begin(true);
		recorder.begin_work = LocalVulkanRecording::RecordedWork();
		return;
	}
#endif
	// A buffer whose last submission completed, else a new one.
	size_t index = recorder.buffers.size();
	for (size_t i = 0; i < recorder.buffers.size(); ++i) {
		const auto tick = recorder.buffers[i].second;
		if (tick != UINT64_MAX && (tick == 0 || m_master.IsFree(tick))) {
			index = i;
			break;
		}
	}
	if (index == recorder.buffers.size()) {
		m_master.Refresh();
		for (size_t i = 0; i < recorder.buffers.size() && index == recorder.buffers.size(); ++i)
			if (recorder.buffers[i].second != UINT64_MAX && m_master.IsFree(recorder.buffers[i].second)) index = i;
	}
	if (index == recorder.buffers.size()) {
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = recorder.pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		vk::CommandBuffer buffer;
		EXIT_NOT_IMPLEMENTED(m_graphics.device.allocateCommandBuffers(&allocate, &buffer) != vk::Result::eSuccess);
		recorder.buffers.emplace_back(buffer, 0);
	}
	recorder.buffers[index].second = UINT64_MAX;
	recorder.open                  = index;
	command.m_buffer               = recorder.buffers[index].first;
	command.Begin();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	recorder.begin_work = LocalVulkanRecording::RecordedWork(); // (after its begin: what it records is counted from here)
#endif
}

void CommandScheduler::EndRecording(Recorder& recorder) {
	auto& command = *recorder.command;
	if (command.IsInvalid()) return;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (recorder.open == Recorder::PacketsOpen) {
		Recorder::Recorded recorded {.index = SIZE_MAX, .ended = true};
		recorded.packets = true;
		if (LocalVulkanRecording::Passive()) {
			const bool empty = command.m_handle_requests == 0 && !command.m_rendering && !command.m_compute_access_pending;
			command.EndRendering();
			(void)command.Handle(); // (a pending compute chain barrier)
			recorded.empty  = empty && LocalVulkanRecording::RecordedWork() == recorder.begin_work;
			recorded.chunks = LocalVulkanRecording::EndSegment();
		} else {
			// (Ended on another thread, to be dropped: what the speculation thread recorded stays in its producer and
			// goes at its next segment. Nothing is recorded here: this thread's stream knows no placeholder.)
			command.m_rendering              = false;
			command.m_compute_access_pending = false;
			command.m_copy_run.clear();
			command.m_run_tail.clear();
			recorded.chunks.failed = true;
		}
		recorder.recorded.push_back(std::move(recorded));
		recorder.open    = SIZE_MAX;
		command.m_buffer = nullptr;
		return;
	}
#endif
	// (Empty: nothing asked for its handle, so nothing recorded into it, nor through the recording stream.)
	const bool empty = command.m_handle_requests == 0 && !command.m_rendering && !command.m_compute_access_pending;
	command.EndRendering();
	(void)command.Handle(); // (a pending compute chain barrier)
	Recorder::Recorded recorded {.index = recorder.open, .timestamp_slot = command.m_timestamp_slot};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	recorded.empty = empty && LocalVulkanRecording::RecordedWork() == recorder.begin_work;
#else
	(void)empty;
#endif
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// Through a recording stream the commands are replayed later: the submission ends the buffer, in order.
	const bool streamed = LocalVulkanRecording::DeferredSubmitEnabled() && !recorder.own_thread;
#else
	const bool streamed = false;
#endif
	if (!streamed) {
		EXIT_NOT_IMPLEMENTED(command.m_buffer.end() != vk::Result::eSuccess);
		recorded.ended = true;
	}
	recorder.recorded.push_back(std::move(recorded));
	recorder.open    = SIZE_MAX;
	command.m_buffer = nullptr;
}

uint64_t CommandScheduler::SubmitRecorded(Recorder& recorder, size_t count) {
	EXIT_IF(Rec() != nullptr || !recorder.command->IsInvalid() || count > recorder.recorded.size());
	// This scheduler's open buffer first (its commands precede the recorded ones), its next one begun after them:
	// what is recorded into a buffer is fenced with the tick the buffer is submitted with (CurrentTick).
	CheckActive();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (count > 0 && recorder.recorded.front().packets) return SpliceRecorded(recorder, count);
#endif
	const bool open = !m_command.IsInvalid();
	// KYTY_SPECULATE_KEEP_PASS (live speculate_keep_pass, on by default under KYTY_SPECULATE=4): a segment that
	// recorded nothing (its buffers empty: two holes with only state between them) does not close the open buffer: its
	// buffers are dropped and the open buffer, with its render pass, goes on into the next hole. Nothing is submitted:
	// no tick (the dropped buffers read none of the translation's rings; its earlier segments' ticks cover those).
	static auto& keep_pass = LiveSwitch::Find(
	    "speculate_keep_pass", ModeToggle::Default("speculate_keep_pass"));
	if (keep_pass.load(std::memory_order_relaxed) != 0 && open && m_command.Rendering() && count > 0 &&
	    std::all_of(recorder.recorded.begin(), recorder.recorded.begin() + static_cast<ptrdiff_t>(count),
	                [](const Recorder::Recorded& recorded) { return recorded.empty; })) {
		for (size_t i = 0; i < count; ++i) DiscardOne(recorder, recorder.recorded[i]);
		recorder.recorded.erase(recorder.recorded.begin(), recorder.recorded.begin() + static_cast<ptrdiff_t>(count));
		LiveCounters::Add(LiveCounters::KeptPasses);
		return 0;
	}
	// (One submission for the open buffer and the recorded ones after it.)
	std::array<SubmitEntry, MaxSubmitEntries> entries {};
	size_t                                    n = 0;
	if (open) {
		m_command.EndRendering();
		if (m_epilogue_hook) m_epilogue_hook(m_command); // (KYTY_FEEDBACK_BATCH)
		(void)m_command.Handle(); // (a pending compute chain barrier closes it, as Submit's)
		if (!m_prologue.IsInvalid()) entries[n++] = ClosePrologue();
		entries[n++]          = {m_command.m_buffer, false, m_command.m_timestamp_slot};
		m_command.m_buffer    = nullptr;
		m_recorded_dispatches = 0;
		m_recorded_draws      = 0;
		m_flush_deferred      = false;
	}
	// Held for one submission with what follows (the next Submit; FlushRecorded): each buffer's tick is the one that
	// submission gets (CurrentTick). Only while this scheduler has an open buffer, which that submission includes:
	// the open buffer's work is fenced with CurrentTick, so held buffers are never submitted without it (a batch
	// submitted alone would move the tick under it). When the batch fills it is submitted alone, but only once the
	// closed open buffer is in it: everything fenced with that tick goes in that submission (split from the batch, the
	// open buffer went a tick late, its stream buffer and descriptor space free a tick early). The batch keeps room for
	// Submit's prologue and open buffer.
	static auto& batch = LiveSwitch::Find("speculate_batch_submits", 1);
	if (open && (batch.load(std::memory_order_relaxed) != 0 || m_batched != 0)) {
		EXIT_IF(m_batched + n > MaxSubmitEntries);
		for (size_t i = 0; i < n; ++i) m_batch[m_batched++] = entries[i];
		uint64_t tick = 0;
		for (size_t i = 0; i < count; ++i) {
			if (m_batched + 2 >= MaxSubmitEntries) {
				(void)SubmitBuffers({m_batch.data(), m_batched}, {});
				m_batched = 0;
			}
			const auto& recorded = recorder.recorded[i];
			m_batch[m_batched++] = {recorder.buffers[recorded.index].first, recorded.ended, recorded.timestamp_slot};
			tick = recorder.buffers[recorded.index].second = m_master.CurrentTick();
		}
		if (m_batched + 2 > MaxSubmitEntries) {
			(void)SubmitBuffers({m_batch.data(), m_batched}, {});
			m_batched = 0;
		}
		recorder.recorded.erase(recorder.recorded.begin(), recorder.recorded.begin() + static_cast<ptrdiff_t>(count));
		BeginNext();
		return tick;
	}
	uint64_t tick = 0;
	for (size_t i = 0, first = 0; i < count; ++i) {
		const auto& recorded = recorder.recorded[i];
		entries[n++]         = {recorder.buffers[recorded.index].first, recorded.ended, recorded.timestamp_slot};
		if (n == MaxSubmitEntries || i + 1 == count) {
			tick = SubmitBuffers({entries.data(), n}, {});
			for (size_t j = first; j <= i; ++j) recorder.buffers[recorder.recorded[j].index].second = tick;
			first = i + 1;
			n     = 0;
		}
	}
	if (n != 0) (void)SubmitBuffers({entries.data(), n}, {}); // (the open buffer alone: nothing recorded)
	recorder.recorded.erase(recorder.recorded.begin(), recorder.recorded.begin() + static_cast<ptrdiff_t>(count));
	if (open) BeginNext();
	return tick;
}

void CommandScheduler::FlushRecorded() {
	if (m_batched == 0 && !m_spliced) return;
	SubmitInfo submit;
	Flush(submit); // (with the open buffer: SubmitRecorded)
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
// Worker packets: the segments' packets spliced into the open buffer's recording stream, after everything recorded
// into it so far, so the recording worker replays them into it in order. What they record is fenced with the tick the
// open buffer is submitted with (CurrentTick), as a batched segment's buffer is. The open buffer goes on: what this
// thread knows of its state (the render pass, bound state, a copy run's barrier) is ended or forgotten here, as a new
// buffer would begin; the recording worker's own replay state goes on (the same buffer on the same thread).
uint64_t CommandScheduler::SpliceRecorded(Recorder& recorder, size_t count) {
	static auto& keep_pass = LiveSwitch::Find(
	    "speculate_keep_pass", ModeToggle::Default("speculate_keep_pass"));
	const auto first = recorder.recorded.begin(), last = first + static_cast<ptrdiff_t>(count);
	// (KEEP_PASS: segments that recorded nothing splice nothing; the open buffer and its render pass go on.)
	if (std::all_of(first, last, [](const Recorder::Recorded& recorded) {
		    return recorded.packets && !recorded.chunks.failed && recorded.chunks.empty();
	    })) {
		recorder.recorded.erase(first, last);
		if (keep_pass.load(std::memory_order_relaxed) != 0) LiveCounters::Add(LiveCounters::KeptPasses);
		return 0;
	}
	if (m_command.IsInvalid()) BeginNext();
	m_command.EndRendering();
	(void)m_command.Handle(); // (a pending compute chain barrier, a copy run's: before the spliced commands)
	const VkCommandBuffer target = m_command.m_buffer;
	for (auto it = first; it != last; ++it) {
		EXIT_IF(!it->packets);
		if (it->chunks.failed) { // (never committed: CommitSegments stops at a failed translation)
			LocalVulkanRecording::Discard(it->chunks);
			continue;
		}
		// (Only a thread with a recording stream commits worker packets: dropped, the work would be lost.)
		EXIT_IF(!LocalVulkanRecording::Splice(it->chunks, target));
	}
	recorder.recorded.erase(first, last);
	// (As a new buffer begins: no graphics state of this thread's, no global barrier assumed since.)
	m_command.InvalidateGraphicsState();
	m_command.m_barrier_work = UINT64_MAX;
	++m_command.m_handle_requests;
	// No upload made from now on goes to this buffer's prologue (UploadPrologue: ahead of all of the buffer): the
	// spliced commands were recorded on a speculation thread, under no clock or serial of this buffer (a copy they make
	// into a buffer carries another buffer's serial), so a later upload is recorded in order, after them.
	m_command_clock = 0;
	m_spliced = true;
	return m_master.CurrentTick();
}
#endif

void CommandScheduler::DiscardOne(Recorder& recorder, const Recorder::Recorded& recorded) {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (recorded.packets) return; // (its chunks go with it: ChunkList's destructor discards them)
#endif
	const VkCommandBuffer command = recorder.buffers[recorded.index].first;
	if (recorder.own_thread) { // (ended: begun again, the pool resets it)
		recorder.buffers[recorded.index].second = 0;
		return;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	const LocalVulkanRecording::Segment segments[] {{&command, sizeof(command)}};
	if (!recorded.ended && LocalVulkanRecording::EnqueueDeferred(ReplayDiscard, segments, false)) {
		recorder.buffers[recorded.index].second = 0; // (reset by the stream, before any later use of it there)
		return;
	}
	if (!recorded.ended) LocalVulkanRecording::Drain();
#endif
	EXIT_NOT_IMPLEMENTED(vk::CommandBuffer {command}.reset() != vk::Result::eSuccess);
	recorder.buffers[recorded.index].second = 0;
}

void CommandScheduler::DiscardRecorded(Recorder& recorder) {
	EndRecording(recorder);
	for (const auto& recorded: recorder.recorded) DiscardOne(recorder, recorded);
	recorder.recorded.clear();
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
