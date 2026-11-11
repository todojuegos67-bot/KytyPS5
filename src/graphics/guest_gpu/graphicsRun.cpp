#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/drawPrepScanner.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rendererLock.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// The Tracy source locations of the PM4 packets that get a command zone (Profiler::CommandZone):
// draws, dispatches, copies, memory writes and synchronization; not register writes and NOPs.
static const tracy::SourceLocationData* Pm4ZoneLocation(uint32_t opcode, uint32_t r) {
	struct Table {
		std::array<tracy::SourceLocationData, 256>        ops {};
		std::array<tracy::SourceLocationData, Pm4::R_NUM> custom {};
	};
	static const Table table = [] {
		Table      result;
		const auto op = [&](uint32_t code, const char* name) {
			result.ops[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		const auto custom = [&](uint32_t code, const char* name) {
			result.custom[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		op(Pm4::IT_DISPATCH_DIRECT, "PM4 DISPATCH_DIRECT");
		op(Pm4::IT_DISPATCH_INDIRECT, "PM4 DISPATCH_INDIRECT");
		op(Pm4::IT_DRAW_INDIRECT, "PM4 DRAW_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_INDIRECT, "PM4 DRAW_INDEX_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_2, "PM4 DRAW_INDEX_2");
		op(Pm4::IT_DRAW_INDIRECT_MULTI, "PM4 DRAW_INDIRECT_MULTI");
		op(Pm4::IT_DRAW_INDEX_AUTO, "PM4 DRAW_INDEX_AUTO");
		op(Pm4::IT_DRAW_INDEX_OFFSET_2, "PM4 DRAW_INDEX_OFFSET_2");
		op(Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, "PM4 DRAW_INDEX_INDIRECT_MULTI");
		op(Pm4::IT_DISPATCH_DRAW, "PM4 DISPATCH_DRAW");
		op(Pm4::IT_WRITE_DATA, "PM4 WRITE_DATA");
		op(Pm4::IT_MEM_SEMAPHORE, "PM4 MEM_SEMAPHORE");
		op(Pm4::IT_WAIT_REG_MEM, "PM4 WAIT_REG_MEM");
		op(Pm4::IT_WAIT_REG_MEM_64, "PM4 WAIT_REG_MEM_64");
		op(Pm4::IT_COPY_DATA, "PM4 COPY_DATA");
		op(Pm4::IT_CP_DMA, "PM4 CP_DMA");
		op(Pm4::IT_DMA_DATA, "PM4 DMA_DATA");
		op(Pm4::IT_SURFACE_SYNC, "PM4 SURFACE_SYNC");
		op(Pm4::IT_EVENT_WRITE, "PM4 EVENT_WRITE");
		op(Pm4::IT_EVENT_WRITE_EOP, "PM4 EVENT_WRITE_EOP");
		op(Pm4::IT_EVENT_WRITE_EOS, "PM4 EVENT_WRITE_EOS");
		op(Pm4::IT_RELEASE_MEM, "PM4 RELEASE_MEM");
		op(Pm4::IT_ACQUIRE_MEM, "PM4 ACQUIRE_MEM");
		op(Pm4::IT_DUMP_CONST_RAM, "PM4 DUMP_CONST_RAM");
		custom(Pm4::R_DRAW_RESET, "PM4 NOP DRAW_RESET");
		custom(Pm4::R_WAIT_FLIP_DONE, "PM4 NOP WAIT_FLIP_DONE");
		custom(Pm4::R_DISPATCH_RESET, "PM4 NOP DISPATCH_RESET");
		custom(Pm4::R_ACQUIRE_MEM, "PM4 NOP ACQUIRE_MEM");
		custom(Pm4::R_WRITE_DATA, "PM4 NOP WRITE_DATA");
		custom(Pm4::R_FLIP, "PM4 NOP FLIP");
		custom(Pm4::R_RELEASE_MEM, "PM4 NOP RELEASE_MEM");
		custom(Pm4::R_DMA_DATA, "PM4 NOP DMA_DATA");
		custom(Pm4::R_CONTEXT_STATE, "PM4 NOP CONTEXT_STATE");
		return result;
	}();
	const auto& location = opcode == Pm4::IT_NOP ? table.custom[r & (Pm4::R_NUM - 1u)]
	                                             : table.ops[opcode & 0xffu];
	return location.name != nullptr ? &location : nullptr;
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// KYTY_LOG_INDIRECT_DRAWS=<n>: log the arguments of the first n indirect draws, including
// multi-draws whose GPU-written count is zero. A diagnostic for draws that GPU work generation
// (culling passes) should have filled.
// KYTY_GPU_INDIRECT_DRAWS=0 reads every DRAW_INDIRECT's arguments on the CPU, as before.
static bool GpuIndirectDraws() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_INDIRECT_DRAWS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_GPU_INDEXED_INDIRECT=0 reads DRAW_INDEX_INDIRECT arguments on the CPU, as before. Read
// there, arguments a dispatch had just written drained the GPU: in GT7's races, 78 % of the GPU
// thread's waits (2026-10-10 profile).
static bool GpuIndexedIndirectDraws() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_INDEXED_INDIRECT");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && GpuIndirectDraws();
}

static bool LogIndirectDraw() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_INDIRECT_DRAWS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (budget == 0) {
		return false;
	}
	// With KYTY_LOG_DRAWS_TRIGGER=<file>, start once that file exists (checked every 256 draws).
	static const char*           trigger   = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	static std::atomic_bool      triggered = trigger == nullptr;
	static std::atomic<uint32_t> polls {0};
	if (!triggered) {
		if ((polls.fetch_add(1, std::memory_order_relaxed) & 255u) != 0u) {
			return false;
		}
		FILE* file = std::fopen(trigger, "rb");
		if (file == nullptr) {
			return false;
		}
		std::fclose(file);
		triggered = true;
	}
	static std::atomic<uint32_t> used {0};
	return used.fetch_add(1, std::memory_order_relaxed) < budget;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	if (DrawPrep::Enabled()) {
		m_draw_prep = std::make_unique<DrawPrepScanner>(renderer, "graphics");
		// Round 170: the compute queues' submissions on a scanner of their own
		// (KYTY_DRAW_PREP_SCANNERS=1, or KYTY_BISECT's scan:shared, as before). Behind a graphics
		// submission of thousands of draws, a compute queue's was often still waiting when the
		// GPU thread ran it.
		const char* scanners = std::getenv("KYTY_DRAW_PREP_SCANNERS");
		if (scanners == nullptr || std::strcmp(scanners, "1") != 0) {
			m_draw_prep_compute = std::make_unique<DrawPrepScanner>(renderer, "compute");
		}
		LOGF("DrawPrep: the scanner prepares draws ahead of the GPU thread (KYTY_DRAW_PREP=0 turns it "
		     "off; verify %d)\n",
		     DrawPrep::VerifyMode());
	}
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_draw_prep_compute.reset();
	m_draw_prep.reset();
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return true;
	}
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) {
		return false;
	}
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	return true;
}

namespace {
std::atomic<bool> g_labels_deferred {false};
} // namespace

bool GuestGpu::LabelsDeferred() noexcept {
	return g_labels_deferred.load(std::memory_order_relaxed);
}

bool GuestGpu::LabelsAtCompletion() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LABELS_AT_COMPLETION");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void GuestGpu::DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size) {
	EXIT_IF(!IsGpuThread() || (size != 4 && size != 8));
	g_labels_deferred.store(true, std::memory_order_relaxed);
	const auto sequence        = ++m_label_sequence;
	m_pending_labels[address] = {value, size, sequence};
	// The completion runs on the scheduler's priority thread, which must not fault on guest
	// memory the GPU wrote to; the write itself goes back to the GPU thread.
	m_renderer.GetCommandScheduler().DeferPriorityOperation([this, address, value, size,
	                                                         sequence] {
		(void)TrySendCommand([this, address, value, size, sequence] {
			Timeline::Mark("label", address, value);
			if (!m_renderer.GetBufferCache().WriteClean(address, &value, size)) {
				m_renderer.PrepareGpuThreadWrite(address, size);
				std::memcpy(reinterpret_cast<void*>(address), &value, size);
			}
			const auto pending = m_pending_labels.find(address);
			if (pending != m_pending_labels.end() && pending->second.sequence == sequence) {
				m_pending_labels.erase(pending);
			}
		});
	});
}

template <typename T>
T GuestGpu::ReadLabel(const volatile T* address) const {
	if (!m_pending_labels.empty()) {
		const auto pending = m_pending_labels.find(reinterpret_cast<uint64_t>(address));
		if (pending != m_pending_labels.end() && pending->second.size >= sizeof(T)) {
			return static_cast<T>(pending->second.value);
		}
	}
	return *address;
}

template uint32_t GuestGpu::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t GuestGpu::ReadLabel<uint64_t>(const volatile uint64_t*) const;

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

bool GuestGpu::TrySendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return true;
	}
	std::binary_semaphore done {0};
	if (!TrySendCommand([operation = std::move(command), &done]() mutable {
		    operation();
		    done.release();
	    })) {
		return false;
	}
	done.acquire();
	return true;
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Timeline::Mark("submit-gfx", draw_commands.size(), constant_commands.size());
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Timeline::Mark("submit-cq", compute_queue, commands.size());
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_args               = 0;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::CompleteReleaseMemWrite() {
	GetScheduler().CompleteReleaseMemWrite();
}

void CommandProcessor::CompleteReleaseMemInterrupt() {
	GetScheduler().CompleteReleaseMemInterrupt();
}

void CommandProcessor::CompleteDraw() {
	GetScheduler().CompleteDraw();
}

void CommandProcessor::CompleteDispatch() {
	GetScheduler().CompleteDispatch();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid, const uint32_t* packet) {
	const bool compute = m_interrupt_event_id != 0;
	if (valid) {
		if (compute && packet == m_rewind_packet && !m_rewind_released) {
			// The CPU released a REWIND this queue waited at: its rewinds work.
			m_rewind_released = true;
			LOGF("PM4: compute queue 0x%x: the CPU released the REWIND at %p it waited at "
			     "(round 124)\n",
			     static_cast<unsigned>(m_interrupt_event_id), static_cast<const void*>(packet));
		}
	} else if (!compute) {
		SuspendPm4();
		return;
	} else if (!m_rewind_waits_off) {
		// The time this packet was polled: a longer gap between two polls (the host busy
		// elsewhere, compiling pipelines or paused) counts as 250 ms.
		const auto now = std::chrono::steady_clock::now();
		if (packet != m_rewind_packet) {
			m_rewind_packet = packet;
			m_rewind_waited = {};
		} else {
			m_rewind_waited += std::min<std::chrono::steady_clock::duration>(
			    now - m_rewind_polled, std::chrono::milliseconds(250));
		}
		m_rewind_polled = now;
		if (m_rewind_waited < std::chrono::seconds(3)) {
			SuspendPm4();
			return;
		}
		const auto releases = Gen5::AcbRewindReleases();
		const bool released = m_rewind_released || releases != 0;
		if (released && ++m_rewind_timeouts < 8) {
			LOGF("PM4: compute queue 0x%x waited 3 s at the REWIND at %p: passed (timeout %u of "
			     "8; the CPU released %" PRIu64 " ACB rewinds so far, round 124)\n",
			     static_cast<unsigned>(m_interrupt_event_id), static_cast<const void*>(packet),
			     m_rewind_timeouts, releases);
		} else {
			m_rewind_waits_off = true;
			LOGF("PM4: compute queue 0x%x waited 3 s at the REWIND at %p, %s: it no longer waits "
			     "at REWINDs (round 124)\n",
			     static_cast<unsigned>(m_interrupt_event_id), static_cast<const void*>(packet),
			     released ? "its eighth timeout" : "and the CPU has released no rewind");
		}
	}
	m_rewind_packet = nullptr;
	// The CPU let the GPU go on, as a WAIT_REG_MEM that passes (round 90).
	g_guest_wait_seq.fetch_add(1, std::memory_order_relaxed);
	GpuPhases::Count(GpuPhases::Phase::WaitPassed, 1);
}

void CommandProcessor::BeforeGuestStore() {
	m_renderer.GetBufferCache().FinishDeferredMemoryWork();
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	BeforeGuestStore();
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
T CommandProcessor::ReadLabel(const volatile T* addr) const {
	return GuestGpu::LabelsDeferred() ? m_renderer.GetGpu().ReadLabel(addr) : *addr;
}

template uint32_t CommandProcessor::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t CommandProcessor::ReadLabel<uint64_t>(const volatile uint64_t*) const;

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(ReadLabel(addr), ref, mask, func)) {
		SuspendPm4();
		return;
	}
	// The wait passed: what the CPU wrote to its hot pages before it let the GPU go on (the
	// next slice does this for a wait that suspends), and for shaders that read memory through
	// addresses, a walk of what it wrote since the last one (round 90, RenderContext::PrepareBda).
	m_renderer.RefreshHotPages();
	g_guest_wait_seq.fetch_add(1, std::memory_order_relaxed);
	GpuPhases::Count(GpuPhases::Phase::WaitPassed, 1);
	// Round 166 (barriers:guest): what comes after the wait comes after what it waited for.
	if (GuestGpu::GuestPm4Barriers()) {
		EmitGlobalBarrier();
	}
	if (FrameGraph::Logging()) {
		FrameGraph::BeginOp();
		FrameGraph::Log("wait 0x%012" PRIx64 " passed: function %u reference 0x%" PRIx64
		                " mask 0x%" PRIx64 " (%u bytes)",
		                reinterpret_cast<uint64_t>(addr), func, static_cast<uint64_t>(ref),
		                static_cast<uint64_t>(mask), static_cast<uint32_t>(sizeof(T)));
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	if (FrameGraph::Logging() && dw_num <= 2) {
		// Round 107: the small writes a game thread may wait on (a sampled thread's "loads").
		char second[16] = "";
		if (dw_num > 1) {
			std::snprintf(second, sizeof(second), " 0x%08" PRIx32, src[1]);
		}
		FrameGraph::BeginOp();
		FrameGraph::Log("write data 0x%012" PRIx64 " <- 0x%08" PRIx32 "%s (%u dwords) queue %d",
		                reinterpret_cast<uint64_t>(dst), src[0], second, dw_num,
		                m_interrupt_event_id);
	}

	// Labels often share a page with bytes the GPU wrote. A plain write then faults and downloads
	// the page, draining the GPU; WriteClean writes both copies instead.
	auto& cache = m_renderer.GetBufferCache();
	if (write_one_address) {
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), &src[dw_num - 1], sizeof(uint32_t),
		                     true)) {
			return;
		}
		m_renderer.PrepareGpuThreadWrite(reinterpret_cast<uint64_t>(dst), sizeof(uint32_t));
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
		LibKernel::Memory::JournalWrite(reinterpret_cast<uint64_t>(dst), sizeof(uint32_t),
		                                LibKernel::Memory::WriteSource::GpuThreadStore);
	} else {
		const auto bytes = static_cast<size_t>(dw_num) * sizeof(uint32_t);
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), src, bytes, true)) {
			return;
		}
		m_renderer.PrepareGpuThreadWrite(reinterpret_cast<uint64_t>(dst), bytes);
		memcpy(dst, src, bytes);
		LibKernel::Memory::JournalWrite(reinterpret_cast<uint64_t>(dst), bytes,
		                                LibKernel::Memory::WriteSource::GpuThreadStore);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	BeforeGuestStore();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	if (m_draw_prep != nullptr) {
		// In queue order, as the GPU thread will run them.
		auto& scanner =
		    submission.queue_id != 0 && m_draw_prep_compute != nullptr && !Bisect::ScanShared()
		        ? *m_draw_prep_compute
		        : *m_draw_prep;
		if (submission.type == SubmissionType::Graphics ||
		    submission.type == SubmissionType::Compute) {
			submission.prep_seq = ++m_prep_seq;
			scanner.EnqueueSubmission(submission.queue_id, submission.prep_seq,
			                          submission.commands);
		} else if (submission.type == SubmissionType::SuspendPoint) {
			scanner.EnqueueReset(submission.queue_id);
		}
	}
	const bool compute = submission.type == SubmissionType::Compute;
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	if (compute) {
		m_compute_enqueued.fetch_add(1, std::memory_order_relaxed);
	}
	m_work_available.Signal();
}

uint32_t GuestGpu::ComputeYieldMicroseconds() {
	static const uint32_t microseconds = [] {
		constexpr uint32_t Default = 500;
		const char*        value   = std::getenv("KYTY_COMPUTE_YIELD");
		uint32_t           result  = Default;
		if (value != nullptr &&
		    (std::strcmp(value, "off") == 0 || std::strcmp(value, "false") == 0)) {
			result = 0;
		} else if (value != nullptr && value[0] >= '0' && value[0] <= '9') {
			result = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
		}
		if (result == 0) {
			LOGF("GPU thread: graphics streams do not stop for compute submissions "
			     "(KYTY_COMPUTE_YIELD)\n");
		} else {
			LOGF("GPU thread: graphics streams stop between packets for compute submissions, "
			     "after %u us (KYTY_COMPUTE_YIELD)\n",
			     result);
		}
		return result;
	}();
	if (Bisect::ComputeYieldOff()) {
		return 0;
	}
	const auto step = Bisect::ComputeYieldMicroseconds();
	return step >= 0 ? static_cast<uint32_t>(step) : microseconds;
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	Common::PreferPerformanceCores("GPU thread");
	Common::RegisterGpuSideThread("Thread_Gpu");
	Profiler::t_gpu_thread = true;
	Common::PerfStats::MarkGpuThread();
	t_guest_gpu_thread     = true;
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	Common::PerfStats::Start();
	Bisect::Start();
	GpuPhases::Start();
	// KYTY_PERF: how the GPU thread spends its time.
	using PerfPhase      = Common::PerfStats::GpuThread;
	const auto perf_time = [] {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	};

	for (;;) {
		GpuPhases::ReportIfDue(static_cast<uint64_t>(gpu->GetFrameNum()));
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				Timeline::Mark("gpu-idle");
				const auto idle_from = perf_time();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				Common::PerfStats::AddGpuThread(PerfPhase::Idle, perf_time() - idle_from);
				Timeline::Mark("gpu-wake");
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					Timeline::Mark("gpu-blocked");
					const auto blocked_from = perf_time();
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					Common::PerfStats::AddGpuThread(PerfPhase::Blocked, perf_time() - blocked_from);
					Timeline::Mark("gpu-unblocked");
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			const auto busy_from = perf_time();
			{
				const GpuPhases::Scope phase(GpuPhases::Phase::Command);
				command();
			}
			Common::PerfStats::AddGpuThread(PerfPhase::Busy, perf_time() - busy_from);

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const auto busy_from = perf_time();
		bool       complete  = false;
		{
			const GpuPhases::QueueSlice slice(submission.type == SubmissionType::Compute);
			const GpuPhases::Scope      phase(GpuPhases::Phase::Process);
			complete = gpu->Process(submission);
		}
		Common::PerfStats::AddGpuThread(PerfPhase::Busy, perf_time() - busy_from);

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			// Round 108: a slice that ran packets may have written what the other queues' blocked
			// fronts wait for (a label), so they are tried again before the all-blocked wait.
			if (submission.progressed) {
				for (auto& queue: gpu->m_queues) {
					if (!queue.empty()) {
						queue.front().blocked = false;
					}
				}
			}
			// A stream that stopped for compute work (round 108) goes on once that has run: the
			// queues after this one come first (m_next_queue).
			submission.blocked    = !submission.yielded;
			submission.yielded    = false;
			submission.progressed = false;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

// KYTY_FLUSH_BEFORE_GC=0 records the garbage collectors' work at the end of a submission into the
// command buffer of its last commands, as before. By default that command buffer is submitted
// first, readahead's downloads with it (the before-flush hook), so a game thread's read of what
// those commands wrote does not wait behind the collectors' image write-backs: in a GT7 race the
// 14 KB a compute dispatch wrote for a game thread came 70-310 ms after the dispatch, behind
// 10-180 MB of images written back and retiled in the same command buffer (2026-10-24).
static bool FlushBeforeGc() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FLUSH_BEFORE_GC");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);
	Timeline::Mark("proc-begin", submission.queue_id, static_cast<uint64_t>(submission.type));
	struct EndMark {
		const Submission& submission;
		~EndMark() {
			Timeline::Mark("proc-end", submission.queue_id,
			               submission.command_complete ? 1u : 0u);
		}
	} end_mark {submission};

	if (first_slice) {
		g_guest_submission_seq.fetch_add(1, std::memory_order_relaxed);
		// Texture-heap entries the guest wrote before submitting are visible to all of this
		// submission's draws, as on the console.
		m_renderer.GetRenderExecutor().SyncBindlessHeaps();
	}
	// What the CPU wrote to its hot pages since the last slice (their writes do not fault).
	m_renderer.RefreshHotPages();

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete = true;
	// Round 166 (barriers:guest): this queue's work comes after what the slices before it
	// recorded, other queues' included (their labels this one may have waited for).
	if (GuestPm4Barriers()) {
		cp.EmitGlobalBarrier();
	}

	switch (submission.type) {
		case SubmissionType::Graphics: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			submission.command_execution.AllowYield(ComputeYieldMicroseconds(), ComputeEnqueued());
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (!submission.command_complete && submission.command_execution.Yielded()) {
					submission.yielded = true;
					break;
				}
				if (complete || !round_progress) {
					break;
				}
			}
			submission.progressed = progressed;
			if (progressed) {
				if (complete) {
					if (FlushBeforeGc()) {
						cp.BufferFlush();
					}
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			submission.progressed = submission.command_execution.MadeProgress();
			if (submission.progressed) {
				if (complete) {
					if (FlushBeforeGc()) {
						cp.BufferFlush();
					}
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			FrameGraph::OnFrameBoundary(m_renderer.GetBufferCache());
			m_renderer.GetTextureCache().TraceFrameEnd();
			if (Pm4SyncBarriers()) {
				cp.EmitGlobalBarrier();
			}
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool draw_prep = execution.m_prep_seq != 0 && DrawPrep::Enabled();
	if (DrawPrep::Enabled() && !draw_prep) {
		DrawPrep::ClearCurrent();
	}
	while (!execution.m_buffer_stack.empty()) {
		// The packets before this one have recorded their commands: the GPU writes they marked
		// take the ticks those went into (before a game thread's read below copies any).
		m_renderer.GetBufferCache().WritesRecorded();
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
			// Round 108: compute work arrived while this graphics stream ran (AllowYield).
			constexpr uint32_t YieldAfterPackets = 32;
			if (execution.m_yield_us != 0 && execution.m_yield_packets >= YieldAfterPackets &&
			    g_gpu_state->ComputeEnqueued() != execution.m_yield_mark &&
			    execution.YieldTimeReached()) {
				execution.m_yielded = true;
				GpuPhases::NoteYield(execution.m_yield_packets);
				return;
			}
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if ((packet_header >> 30u) == 0u) {
			// A type-0 packet writes COUNT+1 registers from a base index; the command processor
			// ignores them, and games leave reserved dwords as zeros, which decode to this
			// (a zero header is one register write of 0 to index 0).
			const auto packet_dw = ((packet_header >> 16u) & 0x3fffu) + 2u;
			static std::atomic<uint32_t> type0_log_count {0};
			if (type0_log_count.fetch_add(1) < 16) {
				LOGF("CP: skipping type-0 packet header=0x%08" PRIx32 " len=%" PRIu32
				     " at offset=0x%05" PRIx32 "\n",
				     packet_header, packet_dw, total_dw - remaining_dw);
			}
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		// Draw prep: number the draw and dispatch packets as the scanner does (predicated ones too;
		// a packet suspended and run again keeps its number).
		if (draw_prep && DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			if (execution.m_prep_packet != packet) {
				execution.m_prep_packet = packet;
				execution.m_prep_ordinal++;
			}
			DrawPrep::SetCurrent({.queue   = execution.m_prep_queue,
			                      .seq     = execution.m_prep_seq,
			                      .ordinal = execution.m_prep_ordinal - 1u,
			                      .packet  = packet});
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		std::optional<Profiler::CommandZone> command_zone;
		if (Profiler::g_counters) {
			if (const auto* location = Pm4ZoneLocation(opcode, KYTY_PM4_R(packet_header));
			    location != nullptr) {
				command_zone.emplace(location);
			}
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		execution.m_yield_packets++;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_args = 0;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);

	if (!indexed && GpuIndirectDraws()) {
		// The GPU reads the counts (RenderExecutor::DrawAuto and ExecutePreparedDraw): fast-launch
		// mesh draws and vertex-pipeline draws; other mesh draws read them there on the CPU. The
		// arguments are usually written by a dispatch just before; reading them here waited for
		// the whole GPU (6.7 % of the GPU thread in the jungle for the fast-launch draws,
		// 2026-10-03, and 2.9 % for the rest, 2026-10-05).
		m_num_instances_args = reinterpret_cast<uint64_t>(args_addr);
		DrawIndexAuto({.offset_source    = DrawOffsetSource::IndirectArgs,
		               .gpu_args_address = reinterpret_cast<uint64_t>(args_addr)});
		return;
	}
	if (!indexed) {
		DrawIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		if (LogIndirectDraw()) {
			LOGF("Indirect draw: args=0x%010" PRIx64 " vertices=%u instances=%u first=%u\n",
			     reinterpret_cast<uint64_t>(args_addr), args.vertex_count_per_instance,
			     args.instance_count, args.start_vertex_location);
		}
		m_num_instances = args.instance_count;
		m_num_instances_args = 0;
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = args.start_instance_location,
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	if (GpuIndexedIndirectDraws() && m_index_buffer_size != 0 &&
	    (m_index_type_and_size == 0 || m_index_type_and_size == 1)) {
		// The GPU reads the arguments (RenderExecutor::DrawIndex, drawIndexedIndirect): the index
		// buffer is bound up to INDEX_BUFFER_SIZE, the start index applied there. Draws the GPU
		// cannot take that way read them on the CPU there.
		m_num_instances_args = reinterpret_cast<uint64_t>(args_addr);
		DrawIndex({.index_count      = m_index_buffer_size,
		           .index_addr       = reinterpret_cast<const void*>(m_index_base_addr),
		           .instance_count   = 1,
		           .offset_source    = DrawOffsetSource::IndirectArgs,
		           .gpu_args_address = reinterpret_cast<uint64_t>(args_addr)});
		return;
	}
	DrawIndexedIndirectArgs args {};
	std::memcpy(&args, args_addr, sizeof(args));
	if (LogIndirectDraw()) {
		LOGF("Indirect indexed draw: args=0x%010" PRIx64 " indices=%u instances=%u first=%u\n",
		     reinterpret_cast<uint64_t>(args_addr), args.index_count_per_instance,
		     args.instance_count, args.start_index_location);
	}

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
	m_num_instances_args = 0;
	DrawIndex({.index_count    = index_count,
	           .index_addr     = index_addr,
	           .instance_count = args.instance_count,
	           .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
	           .first_instance = args.start_instance_location,
	           .offset_source  = DrawOffsetSource::IndirectArgs});
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed, uint32_t draw_index_register) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}
	if (LogIndirectDraw()) {
		LOGF("Indirect multi draw: args=0x%010" PRIx64 " indexed=%d count=%u max=%u count_addr=%p "
		     "stride=%u\n",
		     m_draw_indirect_args_base_addr + data_offset, indexed ? 1 : 0, draw_count,
		     max_count_or_count, static_cast<const void*>(const_cast<const uint32_t*>(count_addr)),
		     stride_in_bytes);
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		// DRAW_INDIRECT_MULTI writes DrawIndex before each draw, including draws
		// with zero vertices/instances. It starts at zero for each packet.
		if (draw_index_register != Pm4::SH_NOP) {
			EXIT_NOT_IMPLEMENTED(draw_index_register >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[draw_index_register] == nullptr);
			SetUserDataMarker(HW::UserSgprType::Unknown);
			g_hw_sh_indirect_func[draw_index_register](*this, draw_index_register, i);
		}
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args      = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			m_num_instances = args->instance_count;
			m_num_instances_args = 0;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		m_num_instances_args = 0;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		FrameGraph::SetGuestQueue(m_interrupt_event_id);
		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode);
	}
	CompleteDispatch();
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	// Thread counts are converted on the GPU. Reading them here drained the GPU for every
	// dispatch, since the previous dispatch writes them; KYTY_INDIRECT_THREADS_ON_CPU=1 restores
	// that path for A/B runs.
	static const bool threads_on_cpu = std::getenv("KYTY_INDIRECT_THREADS_ON_CPU") != nullptr;
	if (threads_on_cpu && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	FrameGraph::SetGuestQueue(m_interrupt_event_id);
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode);
	CompleteDispatch();
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_args != 0) {
		DrawIndirectArgs args {};
		std::memcpy(&args, reinterpret_cast<const void*>(m_num_instances_args), sizeof(args));
		m_num_instances      = args.instance_count;
		m_num_instances_args = 0;
	}
	return m_num_instances;
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0 && args.gpu_args_address == 0) {
		if (m_num_instances_args != 0 && GpuIndirectDraws()) {
			// The instance count a DRAW_INDIRECT read on the GPU left: the GPU reads it too
			// (RenderExecutor::DrawAuto). Reading it here drained the GPU (2.35 % of the GPU
			// thread at the jungle's Leap Attack prompt, 2026-10-05). Read at this draw, not
			// latched at the indirect draw: the arguments are not rewritten in between.
			args.gpu_instance_count_address = m_num_instances_args + sizeof(uint32_t);
		} else {
			args.instance_count = NumInstances();
		}
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();

	if (FrameGraph::Logging()) {
		// Round 107: the labels a game thread may wait on (a sampled thread's "loads").
		FrameGraph::BeginOp();
		FrameGraph::Log("end of pipe 0x%012" PRIx64 " <- 0x%" PRIx64
		                " (%u bytes, source %u, event 0x%x, interrupt %u) queue %d",
		                reinterpret_cast<uint64_t>(dst_gpu_addr), static_cast<uint64_t>(value),
		                static_cast<uint32_t>(sizeof(T)), event_write_source, eop_event_type,
		                interrupt_selector, m_interrupt_event_id);
	}

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	// Round 121 (KYTY_READBACK_LABELS=1, off by default): while a read-back for the CPU is on its
	// way (a small linear image the GPU wrote, going back to guest memory), a label is written once
	// the GPU has run the work before it, as with KYTY_LABELS_AT_COMPLETION, and its read-back is
	// recorded first, so that it lands before the label: a game thread that waits for the label
	// and then reads the result takes this frame's. With KYTY_LABELS_AT_COMPLETION the read-backs
	// are recorded first as well.
	static const bool readback_labels = [] {
		const char* value = std::getenv("KYTY_READBACK_LABELS");
		const bool  on    = value != nullptr && value[0] == '1';
		LOGF("Read-backs for the CPU: %s (KYTY_READBACK_LABELS)\n",
		     on ? "a label written while one is on its way waits for it to land"
		        : "labels do not wait for them");
		return on;
	}();
	bool at_completion = !with_interrupt && GuestGpu::LabelsAtCompletion();
	if (!with_interrupt && (at_completion || readback_labels) &&
	    m_renderer.GetTextureCache().ReadbacksOnTheirWay()) {
		at_completion = true;
		m_renderer.GetTextureCache().ProcessDownloadImages();
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		if (at_completion) {
			m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst), data,
			                                    sizeof(data));
		} else {
			// As WRITE_DATA: a label sharing a page with GPU-written bytes faulted and drained
			// the GPU (twice a frame in GT7's races, 20 % of the GPU thread); WriteClean writes
			// both copies instead.
			if (!m_renderer.GetBufferCache().WriteClean(reinterpret_cast<uint64_t>(dst), &data,
			                                            sizeof(data))) {
				m_renderer.PrepareGpuThreadWrite(reinterpret_cast<uint64_t>(dst), sizeof(data));
				std::memcpy(dst, &data, sizeof(data));
			}
		}

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					if (at_completion) {
						m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst),
						                                    value, sizeof(value));
					} else if (!m_renderer.GetBufferCache().WriteClean(
					               reinterpret_cast<uint64_t>(dst), &value, sizeof(value))) {
						m_renderer.PrepareGpuThreadWrite(reinterpret_cast<uint64_t>(dst),
						                                 sizeof(value));
						std::memcpy(dst, &value, sizeof(value));
					}

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

bool GuestGpu::GlobalPm4Barriers() {
	static const bool whole_run = [] {
		const char* value    = std::getenv("KYTY_PM4_BARRIERS");
		const bool  on       = value != nullptr && std::strcmp(value, "global") == 0;
		const bool  renderer = value != nullptr && std::strcmp(value, "renderer") == 0;
		LOGF("PM4 barriers: %s (KYTY_PM4_BARRIERS=renderer, or KYTY_BISECT's barriers:renderer, "
		     "for a barrier before every writing dispatch and after every dispatch, as before "
		     "round 167; =global, or barriers:global, for the full barriers of before round "
		     "135)\n",
		     on         ? "a full barrier at each cache flush, RELEASE_MEM and suspend point"
		     : renderer ? "the renderer's resource barriers only"
		                : "the guest's sync points (round 166, the default since round 167)");
		return on;
	}();
	return whole_run || Bisect::GlobalPm4Barriers();
}

bool GuestGpu::GuestPm4Barriers() {
	// Round 167: on unless KYTY_PM4_BARRIERS names another kind (or a step asks for one). In a
	// race A/B the GPU thread was no longer held up waiting, about 8 % more frames.
	static const bool renderer_run = [] {
		const char* value = std::getenv("KYTY_PM4_BARRIERS");
		return value != nullptr && std::strcmp(value, "renderer") == 0;
	}();
	if (Bisect::GuestPm4Barriers()) {
		return true;
	}
	return !renderer_run && !GlobalPm4Barriers() && !Bisect::RendererPm4Barriers();
}

void CommandProcessor::EmitGlobalBarrier(const std::source_location& site) {
	const RendererLock lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Recorder().pipelineBarrier2(dependency, site);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order (mainline 70668fbd; KYTY_PM4_BARRIERS=global records the full
	// barrier of before).
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010:
			if (GuestGpu::Pm4SyncBarriers()) {
				EmitGlobalBarrier();
			}
			break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			if (GuestGpu::Pm4SyncBarriers()) {
				EmitGlobalBarrier();
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			// Round 119: the host GPU counts the samples (OcclusionQueries), and the counters are
			// written once it has run the work before the dump.
			if (OcclusionQueries::Enabled() && OcclusionQueries::Dump(m_renderer, event_address)) {
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready. Each dump advances every DB's counter by KYTY_OCCLUSION_SAMPLES (1 by
			// default), so a query between two dumps reads that many samples per DB: one sample a
			// DB is barely visible to a game that weighs visibility by the samples (an experiment:
			// GT7's tunnels and nights lack their lights).
			static const uint64_t step = [] {
				const char* text  = std::getenv("KYTY_OCCLUSION_SAMPLES");
				const auto  value = text != nullptr ? std::strtoull(text, nullptr, 10) : 0;
				return value != 0 ? value : uint64_t {1};
			}();
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			BeforeGuestStore(); // the game polls the ready bits
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + step) & counter_mask;
			// Diagnostics: how many dumps the game makes, at how many addresses, every ~5 s.
			static std::atomic<uint64_t>        dumps {0};
			static std::atomic<int64_t>         window_start {0};
			static std::mutex                   addresses_mutex;
			static std::unordered_set<uint64_t> addresses;
			const auto n   = dumps.fetch_add(1, std::memory_order_relaxed) + 1;
			const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
			                     std::chrono::steady_clock::now().time_since_epoch())
			                     .count();
			std::lock_guard lock(addresses_mutex);
			addresses.insert(event_address & ~uint64_t {0xff});
			auto start = window_start.load(std::memory_order_relaxed);
			if (start == 0) {
				window_start.store(now, std::memory_order_relaxed);
			} else if (now - start >= 5000) {
				window_start.store(now, std::memory_order_relaxed);
				LOGF("Occlusion queries: %" PRIu64 " counter dumps so far, %zu query blocks in the "
				     "last %" PRId64 " ms, %" PRIu64 " samples per DB a dump\n",
				     n, addresses.size(), now - start, step);
				addresses.clear();
			}
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	BeforeGuestStore();
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	BeforeGuestStore();
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
