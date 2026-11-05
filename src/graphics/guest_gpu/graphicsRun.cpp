#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/guest_gpu/speculation.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/renderer/pipeline/shaderReadObserver.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include "async-upload.h"
#include "live-trace.h"
#include "live-trace-gpu.h"
#include "live-census.h"
#include "live-switch.h"
#include "write-guard.h"
#include "draw-state-observer.h"
#include "frame-capture.h"
#include "hud-anchor.h"
#include "live-control.h"
#include "mode-toggle.h"
#include "performance-switches.h"
#include "speculation-state.h"
#include "xpr-capture.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <x86intrin.h>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <string>
#include <vector>

namespace Libs::Graphics {

extern "C" {
// 0: synchronous frame boundary. 1: the guest may prepare the following frame
// while the renderer consumes the previous one; guest reads of GPU-written
// memory are then served by asynchronous readbacks (BufferCache).
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_frame_pipeline_mode {1};
// 1: a full barrier with no recorded work since the previous one is skipped.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_global_barrier_dedupe {1};
// 1: a guest memory write of the command processor made while older upload jobs are pending lands at once and is hidden
// from the game until they are done (local/write-guard.h); 0: it waits for them. Off by default because each guarded
// write protects pages of a mapped view and back, and per-page protection churn on the views is behind
// MEMORY_MANAGEMENT blue screens on small-memory Windows PCs under memory pressure (#5, #50). Not set (2,
// KYTY_WRITE_GUARD unset): on under KYTY_SPECULATE=4 only, where a commit's writes need it (WriteGuard::Enabled).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_write_guard_mode {2};
}

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
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

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex): m_mutex(mutex) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_mutex.Lock();
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
	}

private:
	Common::Mutex& m_mutex;
};

// The configuration's, once the GPU exists (the command processor asks at every packet).
static bool g_graphics_run_debug_dump = false;

static bool GraphicsRunDebugDumpEnabled() {
	return g_graphics_run_debug_dump;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	g_graphics_run_debug_dump =
	    Config::GraphicsDebugDumpEnabled() && Config::GetPrintfDirection() != Config::OutputDirection::Silent;
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
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
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

// Every run of a guest command on the GPU thread, whichever path ran it (KYTY_SERVE_IN_WAITS: no command is served
// from inside another; the unmap command drains the upload worker over its range before the range goes).
static thread_local uint32_t t_command_depth = 0;
struct CommandScope {
	CommandScope() { ++t_command_depth; }
	~CommandScope() { --t_command_depth; }
	CommandScope(const CommandScope&)            = delete;
	CommandScope& operator=(const CommandScope&) = delete;
};

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		CommandScope scope;
		command();
		return;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// A caller may hand its current command buffer to the GPU thread (for example,
	// fault-driven readback). The receiver can drain only its own recording stream.
	// Finish recording the caller's commands before publishing that handoff.
	LocalVulkanRecording::Drain();
#endif
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		CommandScope scope;
		command();
		return true;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::Drain(); // (as SendCommand)
#endif
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) return false;
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	return true;
}

// Packets that only change state or record draws and dispatches.
static constexpr auto KeepsDeferredProtectionBits = [] {
	std::array<uint64_t, 4> bits {};
	for (const uint32_t opcode:
	     {Pm4::IT_DISPATCH_DIRECT, Pm4::IT_DISPATCH_INDIRECT, Pm4::IT_DRAW_INDIRECT, Pm4::IT_DRAW_INDEX_INDIRECT,
	      Pm4::IT_DRAW_INDEX_2, Pm4::IT_DRAW_INDIRECT_MULTI, Pm4::IT_DRAW_INDEX_AUTO, Pm4::IT_DRAW_INDEX_OFFSET_2,
	      Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, Pm4::IT_INDEX_BASE, Pm4::IT_INDEX_TYPE, Pm4::IT_INDEX_BUFFER_SIZE,
	      Pm4::IT_NUM_INSTANCES, Pm4::IT_SET_BASE, Pm4::IT_SET_SH_REG, Pm4::IT_SET_CONTEXT_REG, Pm4::IT_SET_UCONFIG_REG,
	      Pm4::IT_SET_UCONFIG_REG_INDEX, Pm4::IT_SET_SH_REG_INDIRECT, Pm4::IT_SET_CONTEXT_REG_INDIRECT,
	      Pm4::IT_SET_UCONFIG_REG_INDIRECT})
		bits[opcode / 64u] |= uint64_t {1} << (opcode % 64u);
	return bits;
}();
// (A bit per opcode: the switch it replaces cost 0.3% of the GPU thread, once per packet.)
static bool KeepsDeferredProtection(uint32_t opcode) {
	return ((KeepsDeferredProtectionBits[(opcode >> 6u) & 3u] >> (opcode & 63u)) & 1u) != 0;
}

static bool DrawsOrDispatches(uint32_t opcode) {
	return opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ||
	       opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
	       opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI || opcode == Pm4::IT_DISPATCH_DIRECT ||
	       opcode == Pm4::IT_DISPATCH_INDIRECT;
}
// The next draw's (or dispatch's) state, asked for while this one is made: it is read ~1 us from now and is cold (the
// game's job threads wrote it). Past this draw (and the draws right after it), up to the next one: the tables of the
// indirect register packets (the draw state observer and the register handlers read them first), and the first lines
// of what each 64-bit word of the SET_SH_REG packets (user data) would point at in the guest's range (the tables a
// draw's SRT evaluation reads first; a prefetch of an address nothing maps does nothing). Census at 1-1 standing, same
// process, 2 rounds: a table draw's evaluation 0.31 -> 0.28 us, the whole table draw 1.42 -> 1.37 us, the frame's PM4
// runs -0.24 ms.
static void PrefetchNextDraw(const uint32_t* packet, uint32_t remaining_dw) {
	bool     state = false;
	uint32_t at    = 0;
	for (uint32_t n = 0; n < 64 && at < remaining_dw; ++n) {
		const uint32_t header = packet[at];
		if (header == 0x80000000u) {
			++at;
			continue;
		}
		if ((header >> 30u) != 3u) return;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len > remaining_dw - at) return;
		const uint32_t  op = (header >> 8u) & 0xffu;
		const uint32_t* p  = packet + at;
		at += len;
		if (DrawsOrDispatches(op)) {
			if (state) return;
			continue;
		}
		state = true;
		if (op == Pm4::IT_SET_SH_REG_INDIRECT || op == Pm4::IT_SET_CONTEXT_REG_INDIRECT ||
		    op == Pm4::IT_SET_UCONFIG_REG_INDIRECT) {
			if (len < 5) continue;
			const auto*    block = reinterpret_cast<const char*>((uint64_t {p[1]} & 0xfffffffcu) | (uint64_t {p[2]} << 32u));
			const uint32_t bytes = std::min<uint32_t>((p[4] & 0x3fffu) * 8u, 512u);
			for (uint32_t offset = 0; offset < bytes; offset += 64) __builtin_prefetch(block + offset);
		} else if (op == Pm4::IT_SET_SH_REG) {
			for (uint32_t i = 2; i + 1 < len; ++i) {
				if (p[i + 1] - 1u >= 0x20u) continue; // (a guest address: 4 GiB to 128 GiB)
				const auto* table = reinterpret_cast<const char*>(uint64_t {p[i]} | (uint64_t {p[i + 1]} << 32u));
				__builtin_prefetch(table);
				__builtin_prefetch(table + 64);
			}
		}
	}
}

// A game thread's readback fault sends its command here and waits for it, and the GPU thread took commands only before
// each packet: a request queued ~0.1 ms, 11-15 ms of the game's threads' time a frame at Demon's Souls' stone arch.
// Run while the GPU thread waits for the upload worker as well, where it spun anyway (KYTY_SERVE_IN_WAITS).
uint32_t GuestGpu::ServeInWaits() noexcept {
	static auto& serve = LiveSwitch::Find("serve_in_waits", [] {
		const char* value = std::getenv("KYTY_SERVE_IN_WAITS");
		// (Under KYTY_SPECULATE=4, 7: also after each committed segment and in the chunk waits.)
		return value != nullptr ? std::atoll(value) : ModeToggle::Default("serve_in_waits");
	}());
	// (Any nonzero value serves in the upload waits, as upstream's 1 did: KYTY_SERVE_IN_WAITS=2 or 4 keeps them.)
	const auto value = static_cast<uint32_t>(serve.load(std::memory_order_relaxed));
	return value != 0 ? value | 1u : 0u;
}

// (Only the drains before a guest memory write and at the frame boundary serve (AsyncUpload::Drain(true)): a submit's
// drain has a command buffer ended and its tick not yet taken, and a command's own wait (the unmap's drain over its
// range) must not run the commands behind it inside itself, so inside any command (t_command_depth) nothing is
// served; those commands wait their turn, as before.)
void GuestGpu::ServePendingCommands() {
	if (t_command_depth != 0 || g_gpu_state == nullptr ||
	    g_gpu_state->m_pending_commands.load(std::memory_order_acquire) == 0)
		return;
	g_gpu_state->ProcessCommands();
}

bool GuestGpu::CommandsPending() noexcept {
	return t_command_depth == 0 && g_gpu_state != nullptr &&
	       g_gpu_state->m_pending_commands.load(std::memory_order_acquire) != 0;
}

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
		// Fault, readback and mapping callbacks cannot overtake a queued draw.
		PageManager::FlushDeferredProtection();
		LiveCounters::Add(LiveCounters::GuestCommands);
		LiveTrace::Event(LiveTrace::GuestCommand, 1, 0);
		{
			CommandScope scope;
			command();
		}
		LiveTrace::Event(LiveTrace::GuestCommand, 0, 0);
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		CommandScope scope;
		command();
		return;
	}
	// The caller spins a while (~20 us, about the time a guest readback command waits for its turn and runs) before it
	// sleeps: waking a sleeping caller cost the GPU thread a system call each, ~200 a frame at 1-1 (same process: GPU
	// thread 16.66 -> 16.38 ms a frame). 0 waiting, 1 done, 2 the caller sleeps.
	constexpr uint32_t    Spins = 500;
	std::atomic<uint32_t> state {0};
	SendCommand([operation = std::move(command), &state]() mutable {
		operation();
		if (state.exchange(1, std::memory_order_acq_rel) == 2) state.notify_one();
	});
	for (uint32_t spin = 0; spin < Spins; ++spin) {
		if (state.load(std::memory_order_acquire) == 1) return;
		_mm_pause();
	}
	uint32_t expected = 0;
	if (!state.compare_exchange_strong(expected, 2, std::memory_order_acq_rel)) return; // (done meanwhile)
	while (state.load(std::memory_order_acquire) != 1) state.wait(2, std::memory_order_acquire);
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	submission.reset_processor   = m_graphics_done;
	m_graphics_done              = false;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());
	GpuMutexLock lock(m_submission_mutex);

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

void GuestGpu::Done() {
	GpuMutexLock lock(m_submission_mutex);
	if (!IsGpuThread()) {
		if (kyty_local_frame_pipeline_mode.load(std::memory_order_relaxed) != 0) {
			// Permit CPU preparation of one following frame. Before publishing
			// another boundary, the preceding frame must have been consumed on
			// every PM4 queue. Host commands remain serviceable during this wait.
			{
				Common::LockGuard queue_lock(m_queue_mutex);
				while (m_submitted_frame != m_consumed_frame) {
					m_idle.Wait(&m_queue_mutex);
				}
			}
			LiveTrace::Event(LiveTrace::GuestDone, m_submitted_frame);
			Submission boundary;
			boundary.type = SubmissionType::FrameBoundary;
			Enqueue(std::move(boundary));
			++m_submitted_frame;
			m_graphics_done = true;
			return;
		}
		WaitForIdle();
	}
	m_graphics_done = true;
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
	m_draw_run_skip = 0;
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, size_t {m_const_ram_used} * sizeof(uint32_t));
	m_const_ram_used = 0;
	// Every register changed: the native path's clean-draw shadow no longer matches.
	DrawStateObserver::Invalidate();
}

const uint32_t* CommandProcessor::SuspendedPacket(const Pm4Execution& execution, uint32_t& dwords) {
	EXIT_IF(execution.m_buffer_stack.empty());
	const auto& cursor = execution.m_buffer_stack.back();
	EXIT_IF(cursor.offset_dw >= cursor.commands.size() || cursor.deferred_advance_dw != 0);
	const auto* packet = cursor.commands.data() + cursor.offset_dw;
	dwords             = *packet == 0x80000000u ? 1u : KYTY_PM4_LEN(*packet);
	EXIT_IF(dwords == 0 || dwords > cursor.commands.size() - cursor.offset_dw);
	return packet;
}

bool CommandProcessor::SamePosition(const Pm4Execution& a, const Pm4Execution& b) {
	// (The innermost first: most packets differ there. Not the deferred advances: an outer level's is set only when the
	// execution was suspended inside the buffer it called.)
	if (a.m_buffer_stack.size() != b.m_buffer_stack.size()) return false;
	for (size_t i = a.m_buffer_stack.size(); i-- > 0;)
		if (a.m_buffer_stack[i].Position() != b.m_buffer_stack[i].Position() ||
		    a.m_buffer_stack[i].offset_dw != b.m_buffer_stack[i].offset_dw)
			return false;
	return true;
}

// A draw or dispatch: what chunks of a command buffer are counted in.
static bool IsChunkWork(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return true;
		default: return false;
	}
}

bool CommandProcessor::AtWait(const Pm4Execution& execution) {
	if (execution.m_buffer_stack.empty()) return false;
	const auto& cursor = execution.m_buffer_stack.back();
	if (cursor.offset_dw >= cursor.commands.size() || cursor.deferred_advance_dw != 0) return false;
	const auto opcode = (cursor.commands[cursor.offset_dw] >> 8u) & 0xffu;
	return (cursor.commands[cursor.offset_dw] >> 30u) == 3u && (opcode == Pm4::IT_WAIT_REG_MEM || opcode == Pm4::IT_WAIT_REG_MEM_64);
}

void CommandProcessor::SkipSuspendedPacket(Pm4Execution& execution) {
	uint32_t dwords = 0;
	(void)SuspendedPacket(execution, dwords);
	execution.m_buffer_stack.back().offset_dw += dwords;
	execution.m_made_progress = true;
}

bool CommandProcessor::SameDrawState(const CommandProcessor& other) const {
	return m_context_state_pushed == other.m_context_state_pushed && m_user_data_marker == other.m_user_data_marker &&
	       m_index_type_and_size == other.m_index_type_and_size && m_index_buffer_size == other.m_index_buffer_size &&
	       m_index_base_addr == other.m_index_base_addr &&
	       m_draw_indirect_args_base_addr == other.m_draw_indirect_args_base_addr &&
	       m_dispatch_indirect_args_base_addr == other.m_dispatch_indirect_args_base_addr &&
	       m_num_instances == other.m_num_instances && m_de_count == other.m_de_count &&
	       m_ce_count == other.m_ce_count && m_synthetic_occlusion_counter == other.m_synthetic_occlusion_counter &&
	       m_predicate_skip == other.m_predicate_skip;
}

// A state pass's packets (CommandProcessor::SetStateOnly): those that change the processor's state run, and those whose
// guest memory writes and predicted clears its record keeps (Spec::t_record); the others are passed over.
static bool StatePassRuns(uint32_t opcode, uint32_t header) {
	switch (opcode) {
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_DMA_DATA:
		case Pm4::IT_COPY_DATA:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_INCREMENT_CE_COUNTER:
		case Pm4::IT_INCREMENT_DE_COUNTER:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_COND_EXEC:
		case Pm4::IT_SET_PREDICATION:
		case Pm4::IT_EVENT_WRITE: return true; // (an occlusion counter dump counts: TriggerEvent)
		case Pm4::IT_NOP: return KYTY_PM4_R(header) == Pm4::R_CONTEXT_STATE || KYTY_PM4_R(header) == Pm4::R_DISPATCH_RESET;
		default: return false;
	}
}

// A SET_PREDICATION's op (CpOpSetPredication's two layouts): nonzero reads the predicate from memory.
static bool StatePassPredicationReads(const uint32_t* packet) {
	const auto         payload_dw = KYTY_PM4_LEN(packet[0]) - 1u;
	constexpr uint32_t mask       = 0x00071100u;
	if (payload_dw < 2) return true;
	const bool     packed = payload_dw >= 3 && (packet[1] & ~mask) == 0 && packet[3] <= 0x0000ffffu;
	const uint32_t flags  = packed ? packet[1] : packet[2];
	return ((flags >> 16u) & 0x7u) != 0;
}

bool CommandProcessor::SameState(const CommandProcessor& other) const {
	// KYTY_SPEC_START_INSTANCE_FREE (live spec_start_instance_free, on by default under KYTY_SPECULATE=4): not the
	// start-instance user SGPR (HW::Shader::
	// SameButStartInstance): a draw's hole sets it where the state pass, which passes draws over, does not, and a chunk's
	// commit then stopped at "ahead: another start state" (~0.3 ms a frame of in-order translation with spec_predict_dcc).
	static auto& start_instance_free = LiveSwitch::Find("spec_start_instance_free", [] {
		const char* value = std::getenv("KYTY_SPEC_START_INSTANCE_FREE");
		return value != nullptr ? std::atoll(value) : ModeToggle::Default("spec_start_instance_free");
	}());
	const bool same_sh = start_instance_free.load(std::memory_order_relaxed) != 0 ? m_sh_ctx.SameButStartInstance(other.m_sh_ctx)
	                                                                                : m_sh_ctx == other.m_sh_ctx;
	return m_draw_run_skip == other.m_draw_run_skip && m_ctx == other.m_ctx && m_saved_ctx == other.m_saved_ctx &&
	       m_ucfg == other.m_ucfg && same_sh && SameDrawState(other);
}

void CommandProcessor::CopyStateFrom(const CommandProcessor& other, bool const_ram) {
	m_draw_run_skip                    = other.m_draw_run_skip;
	m_ctx                              = other.m_ctx;
	m_saved_ctx                        = other.m_saved_ctx;
	m_context_state_pushed             = other.m_context_state_pushed;
	m_ucfg                             = other.m_ucfg;
	m_sh_ctx                           = other.m_sh_ctx;
	m_user_data_marker                 = other.m_user_data_marker;
	m_index_type_and_size              = other.m_index_type_and_size;
	m_index_buffer_size                = other.m_index_buffer_size;
	m_index_base_addr                  = other.m_index_base_addr;
	m_draw_indirect_args_base_addr     = other.m_draw_indirect_args_base_addr;
	m_dispatch_indirect_args_base_addr = other.m_dispatch_indirect_args_base_addr;
	m_num_instances                    = other.m_num_instances;
	m_de_count                         = other.m_de_count;
	m_ce_count                         = other.m_ce_count;
	m_ce_complete                      = other.m_ce_complete;
	if (const_ram) {
		std::memcpy(m_const_ram, other.m_const_ram,
		            size_t {std::max(m_const_ram_used, other.m_const_ram_used)} * sizeof(uint32_t));
		m_const_ram_used = other.m_const_ram_used;
	}
	m_flip                        = other.m_flip;
	m_submit_id                   = other.m_submit_id;
	m_synthetic_occlusion_counter = other.m_synthetic_occlusion_counter;
	m_predicate_skip              = other.m_predicate_skip;
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
	// (A speculation's work goes to the GPU at its commit, before the guest memory writes that follow it.)
	if (Spec::Current() != nullptr) return;
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushDeferrable() {
	if (Spec::Current() != nullptr) return;
	GetScheduler().DeferrableFlush();
}

void CommandProcessor::BufferFlushAndWait() {
	GetScheduler().FlushAndWait();
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

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
	m_const_ram_used = std::min(std::max(m_const_ram_used, offset / 4 + dw_num), static_cast<uint32_t>(std::size(m_const_ram)));
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
	if (Scout::Running()) Scout::NoteHostWrite(reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * 4);
}

// KYTY_WRITE_GUARD (local/write-guard.h): an execution's command buffers over a guarded page read through the backing
// view, the same bytes (GNM allocates labels in its command buffers). ProcessPm4 takes each packet from its buffer's
// span anew, so the switch takes effect between packets.
static void RebaseGuardedCommands(Pm4Execution* execution) {
	if (execution == nullptr || WriteGuard::g_count.load(std::memory_order_acquire) == 0) return;
	execution->RebaseCommands([](std::span<const uint32_t> commands) {
		const auto* backing = commands.empty() ? nullptr
		                                       : WriteGuard::GuardedBacking(reinterpret_cast<uint64_t>(commands.data()),
		                                                                    commands.size_bytes());
		if (backing == nullptr) return commands;
		return std::span<const uint32_t>(static_cast<const uint32_t*>(backing), commands.size());
	});
}

// (KYTY_WRITE_GUARD.) The upload worker, behind the jobs pending at the write: the page's guard lifted.
static void LiftGuardCall(void* /*context*/, uint64_t page, uint64_t token) {
	WriteGuard::Lift(page, token);
}

// (KYTY_WRITE_GUARD.) Today's write-fault path for the page of `address`, on this thread, the guest's value unchanged:
// a locked add of 0 (an idempotent atomic RMW is compiled to a fenced load, which does not fault on a write-protected
// page).
static void TouchForWrite(uint64_t address) {
	auto* word = reinterpret_cast<uint32_t*>(address & ~uint64_t {3});
	asm volatile("lock addl $0, %0" : "+m"(*word) : : "memory");
}

// KYTY_WRITE_GUARD (local/write-guard.h): the write made at once and hidden from the game until the upload jobs pending
// now are done. False: the caller drains them and writes, as without it (no older job is pending, so there is nothing
// to hide the write behind; or the form does not apply).
static bool GuardedWrite(void* dst, const void* src, uint32_t size) {
	// (Not under a speculation's translation, whose writes are journaled, nor in mode 4, whose threads read the
	// processor's writes: KYTY_SPECULATE=0, the default, only.)
	if (!WriteGuard::Enabled() || size == 0 || !AsyncUpload::g_used.load(std::memory_order_relaxed) ||
	    !GuestGpu::IsGpuThread() ||
	    (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 && !WriteGuard::InSpeculation()))
		return false;
	auto&          worker  = AsyncUpload::Get();
	const uint64_t address = reinterpret_cast<uint64_t>(dst);
	if (worker.Pushed() <= worker.Done()) return false;
	if (Libs::LibKernel::Memory::TryGetBackingPointer(address, size) == nullptr) return false;
	// The bytes first (the packet's, in guest memory: a fault reading them is served here, not inside the backing
	// write below, whose mutation the fault's read-back protection would wait for: the first builds hung there at boot).
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	std::memcpy(bytes.data(), src, size);
	if (WriteGuard::g_upload_done == nullptr) WriteGuard::g_upload_done = [] { return AsyncUpload::Get().Done(); };
	// A pending copy of these bytes takes their old value, as it did with the queue drained before the write.
	AsyncUpload::DrainRange(address, size);
	if (worker.Pushed() <= worker.Done()) return false;
	const uint64_t first = address & ~(WriteGuard::Page - 1), end = address + size;
	// KYTY_WRITE_GUARD_FAST (local/write-guard.h): the lookups below without g_lock when the filter says no page of the
	// write is guarded, else under one shared take; the same answers: only this thread arms, so a page not guarded at
	// the probe stays so until the arm below (a lift meanwhile is the race the separate takes had too). Up to 64 pages.
	constexpr uint64_t FastPages = 64;
	const uint64_t     pages     = (end - first + WriteGuard::Page - 1) / WriteGuard::Page;
	const bool         fast      = pages <= FastPages && WriteGuard::Fast();
	if (fast) {
		uint64_t guarded = 0; // (bit i: the page first + i pages)
		if (WriteGuard::g_count.load(std::memory_order_acquire) != 0 && WriteGuard::MayBeGuarded(address, size)) {
			std::shared_lock lock(WriteGuard::g_lock);
			for (uint64_t i = 0; i < pages; ++i) {
				const auto it = WriteGuard::g_pages.find(first + i * WriteGuard::Page);
				if (it == WriteGuard::g_pages.end()) continue;
				// (A page already guarded whose own protection is not writable: the drain, as below.)
				if (it->second.saved != Common::VirtualMemory::Mode::ReadWrite &&
				    it->second.saved != Common::VirtualMemory::Mode::ExecuteReadWrite)
					return false;
				guarded |= uint64_t {1} << i;
			}
		}
		// (Today's write-fault path for each page not guarded, as below.)
		for (uint64_t i = 0; i < pages; ++i)
			if (((guarded >> i) & 1u) == 0) TouchForWrite(std::max(first + i * WriteGuard::Page, address));
	} else {
		// A page already guarded whose own protection is not writable needs today's write-fault path, which runs the
		// trackers under their locks and may write the guest view: not under a guard (it would wait for a lift behind
		// jobs that need those locks). The drain then, as without the guard.
		{
			std::shared_lock lock(WriteGuard::g_lock);
			for (uint64_t page = first; page < end; page += WriteGuard::Page)
				if (const auto it = WriteGuard::g_pages.find(page); it != WriteGuard::g_pages.end() &&
				                                                    it->second.saved != Common::VirtualMemory::Mode::ReadWrite &&
				                                                    it->second.saved != Common::VirtualMemory::Mode::ExecuteReadWrite)
					return false;
		}
		// Today's write-fault path for each page not guarded (the caches' dirty marks, the GPU's bytes read back first, a
		// fresh granule's first write), as a write through the guest mapping would take it on this thread.
		for (uint64_t page = first; page < end; page += WriteGuard::Page) {
			const uint64_t              at      = std::max(page, address);
			bool                        guarded = false;
			Common::VirtualMemory::Mode saved   = Common::VirtualMemory::Mode::NoAccess;
			{
				std::shared_lock lock(WriteGuard::g_lock);
				if (const auto it = WriteGuard::g_pages.find(page); it != WriteGuard::g_pages.end()) {
					guarded = true;
					saved   = it->second.saved;
				}
			}
			(void)saved;
			if (!guarded) TouchForWrite(at);
		}
	}
	// (KYTY_WRITE_GUARD_FAST: each page's backing view for this thread's own table, WriteGuard::OwnBacking: its entry's
	// while it holds, else looked up here, outside g_lock, which the address space's lock must not be taken under. No
	// unmap runs meanwhile: its Drop runs on this thread, GpuResourceManager::UnmapMemory.)
	std::array<const uint8_t*, FastPages> backings {};
	if (fast)
		for (uint64_t i = 0; i < pages; ++i) {
			const uint64_t page = first + i * WriteGuard::Page;
			backings[i]         = WriteGuard::OwnPage(page);
			if (backings[i] == nullptr) backings[i] = Libs::LibKernel::Memory::TryGetBackingPointer(page, WriteGuard::Page);
		}
	// The guards, the lift jobs behind the pending ones (their guest range [0, 1): over no written byte, so no DrainRange
	// waits for them), then the bytes through the backing view (the guest mapping of the pages has no access now; a
	// fault in the write itself waits for a lift already queued).
	uint64_t token = 0;
	bool     armed = true;
	{
		std::unique_lock lock(WriteGuard::g_lock);
		token = WriteGuard::NextToken();
		for (uint64_t page = first, i = 0; page < end; page += WriteGuard::Page, ++i) {
			const bool ok = WriteGuard::Arm(page, token);
			armed         = ok && armed;
			if (fast && ok) WriteGuard::OwnNote(page, backings[i]);
		}
	}
	for (uint64_t page = first; page < end; page += WriteGuard::Page)
		(void)worker.PushCall(LiftGuardCall, nullptr, page, token, 0, 1);
	worker.Kick();
	const bool written = armed && Libs::LibKernel::Memory::TryWriteBacking(address, bytes.data(), size);
	// (Not written: the caller's drain waits for the lifts too, then writes through the guest mapping.)
	if (!written) return false;
	RebaseGuardedCommands(g_current_execution);
	return true;
}

// KYTY_WRITE_GUARD_SPEC: a commit's guest memory write (Speculation::After) guarded as the normal path's are; false: the
// caller drains and writes, as before.
bool CommitGuardedWrite(void* dst, const void* src, uint32_t size) {
	if (!GuardedWrite(dst, src, size)) return false;
	if (Scout::Running()) Scout::NoteWrite(dst, size, src); // (the scout sees a guarded write as an unguarded one)
	return true;
}

// A guest memory write of the command processor: a label, or data the game or later work reads. The buffer copies
// recorded before it are made first (KYTY_ASYNC_UPLOAD): the game takes a label as the GPU being done and may rewrite
// the buffers those copies read at once, and a copy made after that took the new bytes for the earlier draws (issue
// #196: lit ground and water flickered white; the frame boundary's wait, 5c6f80fa, came after the labels). A
// speculative translation's writes are made when it is committed (speculation-state.h), in order with its other
// writes (Speculation::After drains the same way).
static void WriteGuestMemory(void* dst, const void* src, uint32_t size) {
	// (A state pass's: its record keeps it.)
	if (auto* record = Spec::Record()) {
		for (uint32_t offset = 0; offset < size; offset += 4) {
			uint32_t word = 0;
			std::memcpy(&word, static_cast<const uint8_t*>(src) + offset, 4);
			record->Journal(reinterpret_cast<uint64_t>(dst) + offset, word, 4);
		}
		return;
	}
	auto* spec = Spec::Current();
	if (spec == nullptr) {
		if (GuardedWrite(dst, src, size)) {
			// (Spec::HostWriteParts: the whole write, as without a speculation's effects.)
			Spec::HostWriteParts parts;
			parts.address = reinterpret_cast<uint64_t>(dst), parts.size = size;
			parts.Note();
			// (The scout sees a guarded write as an unguarded one: its bytes from `src`, guest memory may not show them yet.)
			if (Scout::Running()) Scout::NoteWrite(dst, size, src);
		} else {
			AsyncUpload::Drain(/*serve=*/true);
			// (Only what changes, Spec::HostWriteParts: compared where reading it faults nothing back from the GPU.)
			Spec::HostWriteParts parts;
			if (!Libs::LibKernel::Memory::ReadFaults(reinterpret_cast<uint64_t>(dst), size))
				parts.Find(reinterpret_cast<uint64_t>(dst), src, size);
			else
				parts.address = reinterpret_cast<uint64_t>(dst), parts.size = size;
			std::memcpy(dst, src, size);
			parts.Note();
			if (Scout::Running()) Scout::NoteWrite(dst, size); // (as upstream: the unguarded write)
		}
		// (The labels a speculation's waits check: none without speculation.)
		if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0)
			for (uint32_t offset = 0; offset < size; offset += 4) Spec::NoteLabel(reinterpret_cast<uint64_t>(dst) + offset);
		return;
	}
	for (uint32_t offset = 0; offset < size; offset += 4) {
		uint32_t word = 0;
		std::memcpy(&word, static_cast<const uint8_t*>(src) + offset, 4);
		spec->Journal(reinterpret_cast<uint64_t>(dst) + offset, word, 4);
	}
}

// Guest memory as the command processor reads it (a speculative translation sees its own pending writes).
template <typename T>
static T ReadGuestMemory(const T* address) {
	// (A speculation's thread does not read what the GPU wrote: only the GPU thread's read fault reads it back.)
	if (Spec::Current() != nullptr && !GuestGpu::IsGpuThread() &&
	    Libs::LibKernel::Memory::ReadFaults(reinterpret_cast<uint64_t>(address), sizeof(T)))
		return Spec::Refuse("GPU-written guest memory"), T {};
	auto* spec = Spec::Current();
	if (spec == nullptr) {
		// (KYTY_WRITE_GUARD: a page a pending guest write guards read through the backing view, the bytes this thread
		// wrote: WAIT_REG_MEM's label, local/write-guard.h.)
		if (const auto* backing = WriteGuard::GuardedBacking(reinterpret_cast<uint64_t>(address), sizeof(T))) {
			T value;
			std::memcpy(&value, backing, sizeof(T));
			return value;
		}
		return *address;
	}
	static_assert(sizeof(T) % 4 == 0);
	std::array<uint32_t, sizeof(T) / 4> words {};
	if (reinterpret_cast<uint64_t>(address) % 4 != 0 ||
	    !spec->ReadJournaled(reinterpret_cast<uint64_t>(address), static_cast<uint32_t>(words.size()), words.data(), false))
		return Spec::Refuse("memory the processor writes"), T {};
	T value;
	std::memcpy(&value, words.data(), sizeof(T));
	return value;
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
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	BreakComputeChain();
	const T value = ReadGuestMemory(addr);
	// (A speculation's, satisfied by what memory holds: satisfied at its commit too, Spec::WaitCheck. By its pending
	// writes: they come first; by a previous speculation's: its chain reads.)
	if (auto* spec = Spec::Current(); spec != nullptr && TestWaitRegMemValue(value, ref, mask, func) &&
	                                  !spec->JournalIntersects(reinterpret_cast<uint64_t>(addr), sizeof(T)))
		spec->waits.push_back({reinterpret_cast<uint64_t>(addr), ref, mask, func, static_cast<uint32_t>(sizeof(T))});
	if (!TestWaitRegMemValue(value, ref, mask, func)) {
		// A speculation's: a hole on a label the processor writes (the commit waits there; what work of the GPU queues
		// writes before the label is checked against what the speculation read). On memory only the CPU writes it
		// stops: what the guest writes before is for the commands after it (a new speculation starts at the wait when
		// it is satisfied: GuestGpu::Process).
		if (Spec::Current() != nullptr) {
			if (Spec::IsLabel(reinterpret_cast<uint64_t>(addr))) return Spec::Refuse("wait");
			return Spec::Stop("wait on memory the CPU writes");
		}
		LiveTrace::Event(LiveTrace::FrontSuspend, reinterpret_cast<uint64_t>(addr),
		                 (static_cast<uint64_t>(value) << 32u) | static_cast<uint32_t>(ref));
		SuspendPm4();
	} else if (Spec::Current() == nullptr) { // (a speculation's commit publishes them)
		BdaDirtyRegions::Publish();
		// KYTY_SCOUT=2: a wait the scout passes, on a word another writer changed (Scout::CheckWait).
		if (Scout::Mode() == 2 && !IsAsyncComputeQueue() && Scout::Running()) Scout::CheckWait(addr, sizeof(T));
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
	LiveTrace::Event(LiveTrace::LabelWrite, reinterpret_cast<uint64_t>(dst),
	                 src[write_one_address ? dw_num - 1 : 0] | uint64_t {2} << 56u);

	if (write_one_address) {
		WriteGuestMemory(dst, src + dw_num - 1, sizeof(uint32_t));
	} else {
		WriteGuestMemory(dst, src, dw_num * sizeof(uint32_t));
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	// (A state pass's record cannot tell the clock its commit reads.)
	if (auto* record = Spec::Record()) return record->NoteWritten(dst_address, num_bytes);
	const auto value = Sync::ReadReferenceClock();
	WriteGuestMemory(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64 " size=%u\n",
	     dst_address, value, num_bytes);
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
	auto& buffer_cache = GetGpuResources().GetBufferCache();
	if (auto* record = Spec::Record()) {
		// A state pass's record: a small fill's words; a copy between guest memory no GPU work wrote (made on the CPU at
		// the commit, as BufferCache::CopyBuffer's); else the destination as written.
		bool       src_gds = false;
		const auto dst     = dst_address_or_offset;
		const auto src     = src_address_or_offset_or_immediate;
		if (dst_gds) return;
		if (src_sel == 2 && num_bytes <= 64 && num_bytes % 4 == 0) {
			for (uint32_t at = 0; at < num_bytes; at += 4) record->Journal(dst + at, src & 0xffffffffu, 4);
		} else if (src_sel != 2 && decode_gds(src_sel, src_gds) && !src_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes) &&
		           !buffer_cache.IsRegionGpuModified(src, num_bytes)) {
			record->Journal(dst, src, num_bytes, true);
		} else {
			record->NoteWritten(dst, num_bytes);
		}
		return;
	}
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
	submission.frame_epoch = m_submitted_frame;
	// (The scout follows queue 0's graphics submissions, each with the reset the command processor makes before it.)
	if (submission.queue_id == 0 && submission.type == SubmissionType::Graphics) {
		submission.scout_sequence = ++m_scout_sequence;
		Scout::Feed(submission.scout_sequence, submission.commands, !submission.constant_commands.empty(),
		            submission.reset_processor || m_scout_reset);
		m_scout_reset = false;
	} else if (submission.type == SubmissionType::FlipPreparation && submission.reset_processor) {
		m_scout_reset = true;
	}
	submission.serial      = ++m_serials[submission.queue_id];
	// (KYTY_SPECULATE=4: the graphics queue's, translated ahead from now on.)
	if (auto* ahead = m_ahead.load(std::memory_order_acquire); ahead != nullptr && submission.queue_id == 0)
		ahead->Push({submission.serial, submission.commands, submission.reset_processor,
		             submission.type == SubmissionType::FlipPreparation || submission.type == SubmissionType::FrameBoundary
		                 ? Speculation::Job::Kind::Flip
		             : submission.type == SubmissionType::Graphics && submission.constant_commands.empty()
		                 ? Speculation::Job::Kind::Graphics
		                 : Speculation::Job::Kind::Other});
	// (KYTY_SPECULATE_ASYNC: that queue's command buffers, passed and translated as the graphics queue's are.)
	if (auto* ahead = m_ahead_async.load(std::memory_order_acquire);
	    ahead != nullptr && submission.queue_id == m_async_queue && submission.queue_id != 0)
		ahead->Push({submission.serial, submission.commands, submission.reset_processor,
		             submission.type == SubmissionType::Compute && submission.constant_commands.empty()
		                 ? Speculation::Job::Kind::Graphics
		                 : Speculation::Job::Kind::Other});
	LiveTrace::Event(LiveTrace::Submit, submission.queue_id | static_cast<uint32_t>(submission.type) << 8u,
	                 submission.commands.size());
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

bool GuestGpu::CanProcessSubmission(const Submission& submission) const {
	// Caller holds m_queue_mutex. Later-frame compute work must not overtake
	// the graphics boundary, nor may that boundary pass earlier compute work.
	if (submission.frame_epoch != m_consumed_frame) return false;
	if (submission.type == SubmissionType::FrameBoundary) {
		for (uint32_t id = 1; id < QueueCount; ++id) {
			if (!m_queues[id].empty() &&
			    m_queues[id].front().frame_epoch == submission.frame_epoch) return false;
		}
	}
	return true;
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	PageManager::DeferReadProtection(true);
	gpu->m_renderer.GetMutex().BecomeOwner();
	BdaDirtyRegions::g_translating = true;
	LiveTrace::g_mark_thread = true;
	InitializePerformanceSwitches();
#if defined(__linux__)
	static const auto check_write_guard_uffd = [] { // (static: apply_mode calls it too)
	// KYTY_WRITE_GUARD with the userfaultfd write watch (KYTY_UFFD_WP=1, the Steam Deck build's default unless 0;
	// sysLinuxVirtual.cpp WriteWatchInit): the GPU thread faulted forever on a WAIT_REG_MEM label read before the menu
	// (issue #289: a view page's protection unrecorded, see sysLinuxVirtual.cpp). The guard stays off with it unless
	// KYTY_WRITE_GUARD_UFFD=1.
	if (WriteGuard::Enabled()) {
		const char* uffd = std::getenv("KYTY_UFFD_WP");
#if defined(KYTY_STEAM_DECK)
		const bool watch = uffd == nullptr || uffd[0] == '\0' || uffd[0] == '1';
#else
		const bool watch = uffd != nullptr && uffd[0] == '1';
#endif
		// KYTY_WRITE_GUARD_UFFD=1 (default 0): the cause fixed (the views' protection recorded, sysLinuxVirtual.cpp), the
		// guard kept.
		const char* fixed = std::getenv("KYTY_WRITE_GUARD_UFFD");
		if (watch && fixed != nullptr && fixed[0] == '1') {
			std::printf("Write guard: on with the userfaultfd write watch (KYTY_WRITE_GUARD_UFFD=1, issue #289)\n");
		} else if (watch) {
			kyty_local_write_guard_mode.store(0, std::memory_order_relaxed);
			std::printf("Write guard: off with the userfaultfd write watch (KYTY_UFFD_WP, issue #289)\n");
		}
	}
	};
	check_write_guard_uffd();
#endif
	AsyncUpload::g_serve_hook = [](bool serve) {
		if (!serve) return (ServeInWaits() & 1u) != 0 && IsGpuThread();
		if (!IsGpuThread() || !CommandsPending()) return false;
		ServePendingCommands();
		return true;
	};
	// (Mode 4's machinery: at the start, and again on a live `mode 4` (mode-toggle.h, below).)
	const auto start_speculation = [gpu] {
		gpu->m_speculation = std::make_unique<Speculation>(gpu->m_renderer);
		gpu->m_ahead.store(gpu->m_speculation.get(), std::memory_order_release);
		// (The async queue's threads take the packet slots after the graphics queue's: none left, no async speculation.)
		// KYTY_SPECULATE_ASYNC unset: the compute queue 49 (Demon's Souls' async compute; a queue a game does not use
		// costs nothing) with 2 threads unless KYTY_SPECULATE_ASYNC_THREADS says otherwise; 0 none.
		const char* async_value = std::getenv("KYTY_SPECULATE_ASYNC");
		const int   async       = async_value != nullptr ? std::atoi(async_value) : 49;
		if (async > 0 && static_cast<uint32_t>(async) < QueueCount &&
		    kyty_local_speculate_threads.load(std::memory_order_relaxed) < Spec::MaxPacketSlots) {
			const auto  queue   = static_cast<uint32_t>(async);
			const char* threads = std::getenv("KYTY_SPECULATE_ASYNC_THREADS");
			const auto  base    = kyty_local_speculate_threads.load(std::memory_order_relaxed);
			gpu->m_speculation_async = std::make_unique<Speculation>(
			    gpu->m_renderer, static_cast<int>(ComputeQueueBase + queue - 1), base,
			    threads != nullptr && std::atoi(threads) > 0 ? static_cast<uint32_t>(std::atoi(threads))
			                                                 : async_value != nullptr ? 4u : 2u);
			gpu->m_async_queue = queue;
			gpu->m_ahead_async.store(gpu->m_speculation_async.get(), std::memory_order_release);
		}
	};
	// (A KYTY_SPECULATE=4 boot: the mode table's values set before the first packet, as a live `mode 4` sets them; the
	// switches whose Find keeps main's default, the four levers, take mode 4's here.)
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4) {
		ModeToggle::ApplyTable(4);
		start_speculation();
	}
	// (Tooling: the live mode toggle (mode-toggle.h), applied between submissions when no speculated queue has one started
	// and not done: a speculation's results and blocked commits belong to the submission they began. Under m_queue_mutex,
	// so Enqueue pushes to the old speculation or to the new one, never to one being destroyed.)
	const auto apply_mode = [gpu, &start_speculation] {
		const int want = ModeToggle::g_request.load(std::memory_order_acquire);
		if (want < 0) return;
		std::unique_ptr<Speculation> old, old_async;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			const auto started = [&](uint32_t id) {
				return id < QueueCount && !gpu->m_queues[id].empty() && gpu->m_queues[id].front().started;
			};
			if (started(0) || (gpu->m_speculation_async && started(gpu->m_async_queue))) return;
			ModeToggle::g_request.store(-1, std::memory_order_release);
			if (static_cast<int>(kyty_local_speculate_mode.load(std::memory_order_relaxed)) == want) return;
			ModeToggle::ApplyTable(want);
			if (want == 0) {
				gpu->m_ahead.store(nullptr, std::memory_order_release);
				gpu->m_ahead_async.store(nullptr, std::memory_order_release);
				old       = std::move(gpu->m_speculation);
				old_async = std::move(gpu->m_speculation_async);
			} else {
				kyty_local_speculate_mode.store(4, std::memory_order_relaxed);
#if defined(__linux__)
				check_write_guard_uffd(); // (the guard turns on with mode 4: #289's check, as at boot)
#endif
				start_speculation();
				// (The command buffers queued already, handed over as Enqueue hands them.)
				for (const auto& s: gpu->m_queues[0])
					gpu->m_speculation->Push(
					    {s.serial, s.commands, s.reset_processor,
					     s.type == SubmissionType::FlipPreparation || s.type == SubmissionType::FrameBoundary ? Speculation::Job::Kind::Flip
					     : s.type == SubmissionType::Graphics && s.constant_commands.empty()               ? Speculation::Job::Kind::Graphics
					                                                                                         : Speculation::Job::Kind::Other});
				if (gpu->m_speculation_async && gpu->m_async_queue != 0)
					for (const auto& s: gpu->m_queues[gpu->m_async_queue])
						gpu->m_speculation_async->Push({s.serial, s.commands, s.reset_processor,
						                                s.type == SubmissionType::Compute && s.constant_commands.empty()
						                                    ? Speculation::Job::Kind::Graphics
						                                    : Speculation::Job::Kind::Other});
			}
		}
		// (To 0: the threads joined and the unfinished results finished as lost ones (Speculation::Stop: a destroyed
		// result's recorder waited for ticks never submitted, a hang at the 5th flip), outside the lock.)
		if (old) old->Stop();
		if (old_async) old_async->Stop();
		old.reset();
		old_async.reset();
		// (Mode 0 once no speculation thread is left: Spec::Current(), the recorder and executor accessors read the global
		// first, so a thread still translating would otherwise take the normal path. No packet runs in between: this is
		// the GPU thread between submissions.)
		if (want == 0) kyty_local_speculate_mode.store(0, std::memory_order_relaxed);
		std::printf("Mode: KYTY_SPECULATE=%d (live)\n", want);
		std::fflush(stdout);
		Loader::DemonsSoulsWarp::ShowNotice(want == 4 ? "Mode 4 (speculation on)" : "Mode 0 (speculation off)", 3);
	};
	LiveControl::Start();
	XprCapture::Initialize();
	// KYTY_SCOUT: not while a capture sees every draw on the normal path.
	if (!XprCapture::Enabled()) Scout::Start(gpu->m_renderer.GetGpuResources());
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::ProducerScope recording;
#endif

	for (;;) {
		if (ModeToggle::g_request.load(std::memory_order_relaxed) >= 0) apply_mode();
		PageManager::FlushDeferredProtection(); // (before a guest command, an idle wait, another queue's turn)
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				// (A flush deferred while the GPU was busy goes before the thread waits: nothing else would submit it.)
				if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
					gpu->m_queue_mutex.Unlock();
					gpu->m_renderer.GetCommandScheduler().FlushDeferred();
					gpu->m_queue_mutex.Lock();
					continue;
				}
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				LiveTrace::Event(LiveTrace::RenderIdle, 1, 0);
				const auto idle = std::chrono::steady_clock::now();
				LiveControl::NoteRenderIdle(idle.time_since_epoch().count());
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				LiveControl::g_render_idle_ns += (std::chrono::steady_clock::now() - idle).count();
				LiveTrace::Event(LiveTrace::RenderIdle, 0, 0);
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
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked &&
					    gpu->CanProcessSubmission(gpu->m_queues[id].front())) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
						gpu->m_queue_mutex.Unlock();
						gpu->m_renderer.GetCommandScheduler().FlushDeferred();
						gpu->m_queue_mutex.Lock();
						continue;
					}
					gpu->m_processing = false;
					LiveTrace::Event(LiveTrace::RenderIdle, 1, 1);
					const auto idle = std::chrono::steady_clock::now();
					LiveControl::NoteRenderIdle(idle.time_since_epoch().count());
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					LiveControl::g_render_idle_ns += (std::chrono::steady_clock::now() - idle).count();
					LiveTrace::Event(LiveTrace::RenderIdle, 0, 1);
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
				LiveControl::g_render_idle_since.store(0, std::memory_order_relaxed);
			}
		}
		if (should_stop) {
			Scout::Stop();
			gpu->m_gfx_cp->BufferWait();
			gpu->m_renderer.GetMutex().ResignOwner();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			LiveCounters::Add(LiveCounters::GuestCommands);
			{
				CommandScope scope;
				command();
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u | 1u << 17u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);
		const bool complete = gpu->Process(submission);
		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u |
		                     static_cast<uint32_t>(complete) << 16u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);

		Common::LockGuard lock(gpu->m_queue_mutex);
		LiveCounters::Add(LiveCounters::Submissions);
		if (!complete) {
			LiveCounters::Add(LiveCounters::SubmissionRequeues);
			submission.blocked = true;
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

bool GuestGpu::Process(Submission& submission) {
	BdaDirtyRegions::Publish();
	if (submission.type == SubmissionType::FrameBoundary) {
		EXIT_IF(submission.started || submission.queue_id != 0);
		// The frame's upload copies taken first (KYTY_ASYNC_UPLOAD: the upload worker reads guest memory after the
		// render thread recorded them): past the boundary the game writes what that frame's work read (the next frame
		// but one, double buffered), and a copy still pending took those bytes - lit ground and water flickered white
		// for a frame on a fast PC with the game's async compute (issue #196). Without deferred submits each submission
		// drained them; the boundary waits here only for the copies recorded since the last one.
		AsyncUpload::Drain(/*serve=*/true);
		Common::LockGuard lock(m_queue_mutex);
		EXIT_IF(submission.frame_epoch != m_consumed_frame);
		++m_done_num;
		++m_consumed_frame;
		m_idle.SignalAll();
		return true;
	}
	const bool first_slice = !submission.started;
	auto& cp = GetProcessor(submission.queue_id);

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		if (submission.type == SubmissionType::Graphics && Scout::Running()) Scout::BeginSubmission(submission.scout_sequence);
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	// The graphics queue's command buffers, speculation on: speculatively first (speculation.h).
	const bool speculate = submission.type == SubmissionType::Graphics && submission.queue_id == 0 &&
	                       kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0;
	if (speculate && !m_speculation) m_speculation = std::make_unique<Speculation>(m_renderer);
	// (What the translation does a speculation not committed yet must see.)
	struct EffectsScope {
		EffectsScope(Speculation* speculation, Speculation* async) {
			Spec::t_effects       = speculation != nullptr ? &speculation->Log() : nullptr;
			Spec::t_effects_async = async != nullptr ? &async->Log() : nullptr;
		}
		~EffectsScope() {
			Spec::t_effects       = nullptr;
			Spec::t_effects_async = nullptr;
		}
		KYTY_CLASS_NO_COPY(EffectsScope);
	} effects_scope {m_speculation.get(), m_speculation_async.get()};

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			// (The speculation's report: the graphics queue's time while it is on.)
			struct QueueTime {
				Speculation* speculation;
				uint64_t     start = Speculation::NowNs();
				~QueueTime() {
					if (speculation != nullptr) speculation->NoteQueueTime(Speculation::NowNs() - start);
				}
			} queue_time {speculate ? m_speculation.get() : nullptr};
			// KYTY_SPECULATE=4: chunk by chunk, each committed as the speculation's threads translated it, or translated
			// the normal way up to where the next begins (Speculation::CommitReady).
			if (speculate && submission.constant_commands.empty() &&
			    kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4) {
				const Speculation::Job job {submission.serial, submission.commands};
				for (;;) {
					if (!submission.spec_normal) {
						const auto outcome = m_speculation->Blocked(submission.serial)
						                         ? m_speculation->Resume(cp, submission.command_execution)
						                         : m_speculation->CommitReady(cp, job, submission.spec_chunk,
						                                                      submission.command_execution, submission.spec_end);
						progressed |= outcome != Speculation::Outcome::None;
						if (outcome == Speculation::Outcome::Blocked) break;
						if (outcome == Speculation::Outcome::Whole) {
							submission.command_complete = true;
							break;
						}
						if (outcome == Speculation::Outcome::Chunk) {
							++submission.spec_chunk;
							if ((ServeInWaits() & 4u) != 0) ServePendingCommands();
							continue;
						}
						submission.spec_normal = true;
					}
					cp.SetStopAt(submission.spec_end ? &*submission.spec_end : nullptr);
					// (KYTY_SPEC_CHUNK_IN_ORDER: no end known, the chunk the pass will cut, Speculation::ChunkInOrder.)
					const bool chunked = !submission.spec_end && Speculation::ChunkInOrder();
					if (chunked) cp.SetChunking(Speculation::ChunkWork);
					const auto processed = cp.Process(submission.command_execution, submission.commands);
					cp.SetStopAt(nullptr);
					const bool at_boundary = chunked && cp.TakeBoundary();
					if (chunked) cp.SetChunking(0);
					progressed |= submission.command_execution.MadeProgress();
					if (cp.TakeStopReached() || at_boundary) {
						submission.spec_normal = false;
						submission.spec_end.reset();
						++submission.spec_chunk;
						continue;
					}
					submission.command_complete = processed == Pm4ProcessResult::Complete;
					break;
				}
				submission.constant_complete = true;
				complete                     = submission.command_complete;
				if (complete) m_renderer.GetGpuResources().EndSubmission();
				// (What a commit blocked at a wait committed before it is flushed too.)
				if (progressed || m_speculation->Blocked(submission.serial)) cp.BufferFlush();
				break;
			}
			// (From its start, or from a wait the normal translation stopped at, which a speculation may find satisfied
			// now; one that stopped at a wait goes on from there when the work it committed satisfied it.)
			auto outcome = Speculation::Outcome::None;
			if (speculate && m_speculation->Blocked(submission.serial)) {
				// (A commit waiting at a wait hole: on from there.)
				outcome = m_speculation->Resume(cp, submission.command_execution);
				progressed |= outcome == Speculation::Outcome::Partly;
				while (outcome == Speculation::Outcome::Partly && CommandProcessor::AtWait(submission.command_execution)) {
					outcome = m_speculation->TranslateAndCommit(cp, submission.serial, submission.commands,
					                                            submission.command_execution);
					progressed |= outcome == Speculation::Outcome::Partly;
				}
			} else if (speculate && submission.constant_commands.empty() &&
			           (first_slice || CommandProcessor::AtWait(submission.command_execution))) {
				const auto mode = kyty_local_speculate_mode.load(std::memory_order_relaxed);
				// KYTY_SPECULATE=3: from its start, translated ahead with the graphics queue's command buffers after it.
				if (first_slice && mode == 3) {
					std::vector<Speculation::Job> queued;
					{
						Common::LockGuard lock(m_queue_mutex);
						for (const auto& next: m_queues[0])
							queued.push_back({next.serial, next.commands, next.reset_processor,
							                  next.type == SubmissionType::FlipPreparation ? Speculation::Job::Kind::Flip
							                  : next.type == SubmissionType::Graphics && next.constant_commands.empty()
							                      ? Speculation::Job::Kind::Graphics
							                      : Speculation::Job::Kind::Other});
					}
					outcome = m_speculation->CommitAhead(cp, {submission.serial, submission.commands}, queued,
					                                     submission.command_execution);
				} else {
					outcome = m_speculation->TranslateAndCommit(cp, submission.serial, submission.commands,
					                                            submission.command_execution);
				}
				progressed |= outcome == Speculation::Outcome::Partly;
				while (outcome == Speculation::Outcome::Partly && CommandProcessor::AtWait(submission.command_execution)) {
					outcome = m_speculation->TranslateAndCommit(cp, submission.serial, submission.commands,
					                                            submission.command_execution);
					progressed |= outcome == Speculation::Outcome::Partly;
				}
			}
			if (outcome == Speculation::Outcome::Blocked) {
				// (What it committed before the wait is flushed; the submission waits as at a wait of its own.)
				cp.BufferFlush();
				complete = false;
				break;
			}
			if (outcome == Speculation::Outcome::Whole) {
				submission.command_complete  = true;
				submission.constant_complete = true;
				m_renderer.GetGpuResources().EndSubmission();
				cp.BufferFlush();
				break;
			}
			// The rest the normal way (what was committed is progress: its holes' work is flushed with it).
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
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.GetGpuResources().EndSubmission();
				}
				// (A slice stopped at a wait goes to the GPU at once: what it waits for may need it.)
				if (complete) cp.BufferFlushDeferrable();
				else cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().EndSubmission();
			}
			break;
		}
		case SubmissionType::Compute: {
			// KYTY_SPECULATE_ASYNC: that async queue's command buffers chunk by chunk, as the graphics queue's (above).
			if (m_speculation_async && submission.queue_id == m_async_queue && submission.constant_commands.empty()) {
				auto&                   speculation = *m_speculation_async;
				bool                    progressed  = false;
				const Speculation::Job  job {submission.serial, submission.commands};
				struct QueueTime {
					Speculation& speculation;
					uint64_t     start = Speculation::NowNs();
					~QueueTime() { speculation.NoteQueueTime(Speculation::NowNs() - start); }
				} queue_time {speculation};
				for (;;) {
					if (!submission.spec_normal) {
						const auto outcome = speculation.Blocked(submission.serial)
						                         ? speculation.Resume(cp, submission.command_execution)
						                         : speculation.CommitReady(cp, job, submission.spec_chunk,
						                                                   submission.command_execution, submission.spec_end);
						progressed |= outcome != Speculation::Outcome::None;
						if (outcome == Speculation::Outcome::Blocked) break;
						if (outcome == Speculation::Outcome::Whole) {
							submission.command_complete = true;
							break;
						}
						if (outcome == Speculation::Outcome::Chunk) {
							++submission.spec_chunk;
							continue;
						}
						submission.spec_normal = true;
					}
					cp.SetStopAt(submission.spec_end ? &*submission.spec_end : nullptr);
					const auto processed = cp.Process(submission.command_execution, submission.commands);
					cp.SetStopAt(nullptr);
					progressed |= submission.command_execution.MadeProgress();
					if (cp.TakeStopReached()) {
						submission.spec_normal = false;
						submission.spec_end.reset();
						++submission.spec_chunk;
						continue;
					}
					submission.command_complete = processed == Pm4ProcessResult::Complete;
					break;
				}
				complete = submission.command_complete;
				if (complete) m_renderer.GetGpuResources().EndSubmission();
				if (progressed || speculation.Blocked(submission.serial)) {
					if (complete) cp.BufferFlushDeferrable();
					else cp.BufferFlush();
				}
				break;
			}
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
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.GetGpuResources().EndSubmission();
				}
				if (complete) cp.BufferFlushDeferrable();
				else cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().EndSubmission();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.GetGpuResources().EndSubmission();
			m_renderer.GetGpuResources().AdvanceFrame();
			m_renderer.GetPipelineCache().PromoteFinished();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::FrameBoundary: EXIT("frame boundary already handled\n");
	}
	// (KYTY_SPECULATE=4: the state after the graphics queue's submission, for a chain that could not follow it.)
	if (complete && submission.queue_id == 0 && m_ahead.load(std::memory_order_relaxed) != nullptr)
		m_speculation->Done(submission.serial, cp);
	if (complete && submission.queue_id == m_async_queue && submission.queue_id != 0 &&
	    m_ahead_async.load(std::memory_order_relaxed) != nullptr)
		m_speculation_async->Done(submission.serial, cp);

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
	RebaseGuardedCommands(&execution); // (KYTY_WRITE_GUARD: a resumed execution's too)
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

	struct QueueScope {
		explicit QueueScope(uint64_t queue): previous(LiveCensus::g_queue) { LiveCensus::g_queue = queue; }
		~QueueScope() { LiveCensus::g_queue = previous; }
		uint64_t previous;
	} queue_scope(IsAsyncComputeQueue() ? LiveCensus::QueueAsync : 0);
	LiveCensus::Scope census(LiveCensus::QueueRun, static_cast<uint64_t>(m_interrupt_event_id));
	ProcessPm4(execution, 0);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands) {
	EXIT_IF(g_current_execution == nullptr);
	if (commands.empty()) {
		return;
	}
	auto&      execution  = *g_current_execution;
	const auto stop_depth = execution.m_buffer_stack.size();
	execution.m_buffer_stack.push_back({commands});
	RebaseGuardedCommands(&execution); // (KYTY_WRITE_GUARD: a buffer over a guarded page through the backing view)
	ProcessPm4(execution, stop_depth);
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	LiveCounters::Add(LiveCounters::Pm4Suspends);
	g_current_execution->m_suspended = true;
}

// KYTY_SPEC_DRAW_INDIRECT (live spec_draw_indirect, on by default under KYTY_SPECULATE=4): a DRAW_INDIRECT
// (non-indexed) as a table draw, its arguments read on the GPU as a DRAW_INDEX_INDIRECT table draw reads its own
// (CommandProcessor::TryNativeIndirectAutoDraw), on the GPU thread and in a speculative translation. Off: the packet
// is a speculation's hole, drawn the normal way.
static bool SpecDrawIndirect() {
	static auto& on = LiveSwitch::Find("spec_draw_indirect", ModeToggle::Default("spec_draw_indirect"));
	return on.load(std::memory_order_relaxed) != 0;
}
// KYTY_SPEC_DRAW_INDIRECT_VERIFY (live spec_draw_indirect_verify, default 0): every Nth of those table draws (on any
// thread) checked against the normal path's draw on the GPU thread (VerifyDrawIndirect); 0 none.
static int64_t SpecDrawIndirectVerifyEvery() {
	static auto& every = LiveSwitch::Find("spec_draw_indirect_verify", 0);
	return every.load(std::memory_order_relaxed);
}

// KYTY_SPEC_DRAW_INDIRECT_VERIFY: a DRAW_INDIRECT table draw (`entry`, noted where it was translated) against what the
// normal path (CpOpDrawIndirect -> DrawIndirect -> TryGpuIndirectDraw / its CPU read) would draw here, on the GPU
// thread where the arguments are current (a speculation's: after its segment's Before()). What differs sets a bit:
//   0x1   the packet's dwords in memory now are not those translated;
//   0x2   the arguments' address from them now (the base register then + data offset now) is not the one drawn from;
//   0x4   the draw is not one draw of a 16-byte stride (VkDrawIndirectCommand);
//   0x8   the primitive type is not one TryGpuIndirectDraw draws from GPU arguments;
//   0x10  drawn from a game buffer that is not the one registered over the arguments now, or at another offset;
//   0x20  an image GPU work wrote is over the arguments now (the normal path reads the image's bytes: SyncGpuCleanBacking);
//   0x40  the arguments' range is still CPU-written and not GPU-written (the buffer the GPU reads does not hold them);
//   0x80  drawn from a ring's copy whose bytes are not the guest's now, or the GPU wrote the range since (the normal
//         path would draw from the GPU's bytes);
//   0x100 the arguments are not mapped now (the normal path's read has nothing).
// (Neither the GPU buffer's bytes of a game buffer (device-local) nor the base register now (a speculation's packet is
// past on the GPU thread) can be read here: the buffer cache's dirty state stands for the first, the value then for
// the second.)
void VerifyDrawIndirect(RenderContext& renderer, const Spec::DrawIndirectVerify& entry) {
	using Prospero::PrimitiveType;
	constexpr uint64_t size = sizeof(DrawIndirectArgs);
	RenderLockGuard    lock(renderer.GetMutex());
	auto&              buffers   = renderer.GetBufferCache();
	auto&              textures  = renderer.GetTextureCache();
	auto&              resources = renderer.GetGpuResources();
	// (Memory as the commit's own checks read it: through the backing view of a page a pending guest write guards.)
	const auto read = [](uint64_t address, void* out, uint64_t bytes) {
		if (Libs::LibKernel::Memory::TryReadBackingToHost(address, out, bytes)) return;
		const void* from = reinterpret_cast<const void*>(address);
		if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
			if (const auto* backing = WriteGuard::GuardedBacking(address, bytes)) from = backing;
		std::memcpy(out, from, bytes);
	};
	uint32_t                mismatch = 0;
	std::array<uint32_t, 5> words {};
	read(entry.packet, words.data(), sizeof(words));
	if (words != entry.words) mismatch |= 0x1u;
	const uint64_t args = entry.base + words[1];
	if (args != entry.args) mismatch |= 0x2u;
	if (entry.stride != size || entry.count != 1) mismatch |= 0x4u;
	const auto primitive = static_cast<PrimitiveType>(entry.primitive);
	if (primitive != PrimitiveType::kTriList && primitive != PrimitiveType::kTriStrip && primitive != PrimitiveType::kTriFan &&
	    primitive != PrimitiveType::kLineList && primitive != PrimitiveType::kLineStrip && primitive != PrimitiveType::kPointList)
		mismatch |= 0x8u;
	const bool mapped = resources.IsMapped(entry.args, size);
	if (!mapped) mismatch |= 0x100u;
	const bool image       = mapped && textures.IsRegionGpuModified(entry.args, size);
	const bool gpu_written = mapped && buffers.IsRegionGpuModified(entry.args, size);
	const bool cpu_written = mapped && buffers.IsRegionCpuModified(entry.args, size);
	if (image) mismatch |= 0x20u;
	// (The normal path's choice: GPU arguments (TryGpuIndirectDraw) or its CPU read; reported, either draws the bytes.)
	const bool gpu_arguments = mapped && !image && buffers.HasGpuDirtyBytes(entry.args, size);
	std::array<uint32_t, 4> guest {};
	const bool guest_read = mapped && Libs::LibKernel::Memory::TryReadBackingToHost(entry.args, guest.data(), sizeof(guest));
	uint64_t   owner_offset = UINT64_MAX;
	if (entry.copy) {
		if (gpu_written || (guest_read && guest != entry.copied)) mismatch |= 0x80u;
	} else {
		const auto  id    = buffers.RegisteredBuffer(entry.args, size);
		const auto* owner = id ? buffers.GetBufferIfLive(id) : nullptr;
		if (owner != nullptr) owner_offset = owner->Offset(entry.args);
		if (owner == nullptr || static_cast<const void*>(owner) != entry.buffer || owner_offset != entry.offset)
			mismatch |= 0x10u;
		if (cpu_written && !gpu_written) mismatch |= 0x40u;
	}
	LiveCounters::Add(LiveCounters::DrawIndirectVerified);
	if (mismatch == 0) return;
	LiveCounters::Add(LiveCounters::DrawIndirectMismatch);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) >= 8) return;
	std::printf("DRAW_INDIRECT_VERIFY mismatch=0x%x packet=0x%llx words=%08x,%08x,%08x,%08x,%08x now=%08x,%08x,%08x,%08x,%08x "
	            "base=0x%llx args=0x%llx derived=0x%llx prim=%u stride=%u count=%u copy=%d offset=0x%llx owner_offset=0x%llx "
	            "image=%d gpu_written=%d cpu_written=%d gpu_arguments=%d taken=%s%08x,%08x,%08x,%08x guest=%s%08x,%08x,%08x,%08x "
	            "copied=%08x,%08x,%08x,%08x segment=%zu\n",
	            mismatch, static_cast<unsigned long long>(entry.packet), entry.words[0], entry.words[1], entry.words[2],
	            entry.words[3], entry.words[4], words[0], words[1], words[2], words[3], words[4],
	            static_cast<unsigned long long>(entry.base), static_cast<unsigned long long>(entry.args),
	            static_cast<unsigned long long>(args), entry.primitive, entry.stride, entry.count, entry.copy ? 1 : 0,
	            static_cast<unsigned long long>(entry.offset), static_cast<unsigned long long>(owner_offset), image ? 1 : 0,
	            gpu_written ? 1 : 0, cpu_written ? 1 : 0, gpu_arguments ? 1 : 0, entry.read ? "" : "(unread)", entry.taken[0],
	            entry.taken[1], entry.taken[2], entry.taken[3], guest_read ? "" : "(unread)", guest[0], guest[1], guest[2], guest[3], entry.copied[0],
	            entry.copied[1], entry.copied[2], entry.copied[3], entry.segment);
}

// The packets a speculative translation (speculation-state.h) takes: those whose work it records and whose effects
// it notes. At any other it stops, and the command buffer is translated as before.
static bool SpeculationTakes(uint32_t opcode, uint32_t header) {
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return true;
		case Pm4::IT_INDIRECT_BUFFER: return KYTY_PM4_LEN(header) == 4u; // (not a branch)
		case Pm4::IT_NOP: {
			const auto r = KYTY_PM4_R(header);
			return r == Pm4::R_ZERO || r == Pm4::R_RELEASE_MEM || r == Pm4::R_ACQUIRE_MEM ||
			       r == Pm4::R_CONTEXT_STATE || r == Pm4::R_DISPATCH_RESET || r == Pm4::R_PUSH_MARKER ||
			       r == Pm4::R_POP_MARKER;
		}
		// Draws only as table draws (TryNativeXprDraws, TryNativeDirectDraw, TryNativeAutoDraw, TryNativeIndirectAutoDraw).
		case Pm4::IT_DRAW_INDEX_AUTO: return header == 0xc0012d00u;
		case Pm4::IT_DRAW_INDIRECT: return header == 0xc0032400u && SpecDrawIndirect();
		case Pm4::IT_DRAW_INDEX_INDIRECT: return header == 0xc0032500u;
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return header == 0xc0083800u;
		case Pm4::IT_DRAW_INDEX_2: return header == 0xc0042700u;
		case Pm4::IT_DRAW_INDEX_OFFSET_2: return header == 0xc0033500u;
		default: return false;
	}
}

// The GS user SGPR at an SH register location, -1 for another register.
static int32_t GsUserSgprAt(uint32_t location) {
	const auto id = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
	return id < 32u ? static_cast<int32_t>(id) : -1;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution, size_t stop_depth) {
	// (KYTY_SPECULATE=4: the state pass, the speculation's threads and the GPU thread's chunked runs take
	// ProcessPm4Spec. Mode 0 runs main's loop: none of their per-packet checks. The mode is 4 from before the first
	// speculation until its results are discarded, so a run never changes copies midway.)
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) [[unlikely]]
		return ProcessPm4Spec(execution, stop_depth);
	while (execution.m_buffer_stack.size() > stop_depth) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
			// (KYTY_EAGER_FEEDBACK: completed copy-feedback snapshots written back between packets.)
			g_gpu_state->m_renderer.GetBufferCache().WriteBackCompletedFeedback(true);
		}
		const auto buffer_index = execution.m_buffer_stack.size() - 1;
		auto&      cursor       = execution.m_buffer_stack[buffer_index];
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.deferred_advance_dw != 0) {
			EXIT_IF(cursor.deferred_advance_dw > cursor.commands.size() - cursor.offset_dw);
			cursor.offset_dw += cursor.deferred_advance_dw;
			cursor.deferred_advance_dw = 0;
			execution.m_made_progress  = true;
			continue;
		}
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

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		// KYTY_SCOUT: queue 0's draws, numbered as the scout numbers them (skipped ones too).
		if (Scout::IsDraw(opcode) && !IsAsyncComputeQueue() && Scout::Running()) Scout::NoteDraw(packet);

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			if (kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0)
				DrawStateObserver::ObservePacket(opcode, packet, packet_dw, true);
#endif
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.load(std::memory_order_relaxed) < 2048 && skip_log_count.fetch_add(1, std::memory_order_relaxed) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.load(std::memory_order_relaxed) < 128 && log_count.fetch_add(1, std::memory_order_relaxed) < 128) {
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

		// The CP writes an indirect draw's start instance into the user SGPR the packet names
		// (START_INST_LOC); every path below translates the draw's shaders for it.
		switch (opcode) {
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
				m_sh_ctx.SetStartInstanceUserSgpr(GsUserSgprAt(packet[3] & 0xffffu));
				break;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DISPATCH_DRAW_PREAMBLE: m_sh_ctx.SetStartInstanceUserSgpr(-1); break;
			default: break;
		}

		// Only state, draws and dispatches keep the deferred read protections pending: before a label, a memory write,
		// a wait or anything else the pages GPU work writes are protected (PageManager::DeferReadProtection).
		if (!KeepsDeferredProtection(opcode)) PageManager::FlushDeferredProtection();
		auto handler = g_cp_op_func[opcode];
		LiveCounters::AddPm4(opcode);
		if (DrawsOrDispatches(opcode)) PrefetchNextDraw(packet, remaining_dw);
		if (XprCapture::Enabled()) {
			if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT) {
				XprCapture::ObserveDraw(m_draw_indirect_args_base_addr + packet[1]);
			} else if (opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
			           opcode == Pm4::IT_DRAW_INDEX_OFFSET_2) {
				XprCapture::ObserveDraw(0);
			}
		}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		// Native XPR draws; on refusal the normal path stores a record, with the
		// reads of its SRT evaluation logged until this packet completes.
		std::optional<ShaderReadObserver> native_reads;
		struct NativeStoreEnd {
			RenderExecutor* executor = nullptr;
			~NativeStoreEnd() {
				if (executor) executor->NativeXprEndPacket();
			}
		} native_store_end;
		if (const auto native = kyty_local_native_xpr_mode.load(std::memory_order_relaxed); native != 0) {
			DrawStateObserver::ObservePacket(opcode, packet, std::min(KYTY_PM4_LEN(packet_header), remaining_dw),
			                                 false);
			if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT &&
			    (packet_header != 0xc0032500u || GraphicsRunDebugDumpEnabled()))
				m_renderer.GetRenderExecutor().NativeXprForgetState();
			if (packet_header == 0xc0032500u && !GraphicsRunDebugDumpEnabled()) {
				const auto consumed =
				    TryNativeXprDraws({packet, remaining_dw}, DrawStateObserver::g_shadow.last_clean);
				if (consumed) {
					m_draw_run_skip = 0;
					if (Scout::Running() && !IsAsyncComputeQueue()) Scout::SkipDraws(consumed / 5u - 1u);
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(consumed / 5u);
					continue;
				}
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			} else if (packet_header == 0xc0083800u && !GraphicsRunDebugDumpEnabled()) {
				// Several indexed indirect draws of consecutive arguments, as a run of DRAW_INDEX_INDIRECT.
				const auto consumed = TryNativeMultiDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(packet[5]);
					continue;
				}
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			} else if (packet_header == 0xc0012d00u && !GraphicsRunDebugDumpEnabled()) {
				// A draw of vertices as a table draw (the normal path stores the pair's state).
				const auto consumed = TryNativeAutoDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
			} else if ((packet_header == 0xc0042700u || packet_header == 0xc0033500u) &&
			           !GraphicsRunDebugDumpEnabled()) {
				// Direct indexed draws from the same records (the normal path stores them).
				const auto consumed = TryNativeDirectDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			}
		}
#endif
		if (opcode != Pm4::IT_DRAW_INDEX_INDIRECT) m_draw_run_skip = 0;
		if (packet_header == 0xc0032500u && !GraphicsRunDebugDumpEnabled()) {
			const auto consumed = TryDrawIndirectRun({packet, remaining_dw});
			if (consumed) {
				if (Scout::Running() && !IsAsyncComputeQueue()) Scout::SkipDraws(consumed / 5u - 1u);
				execution.m_buffer_stack[buffer_index].offset_dw += consumed;
				execution.m_made_progress = true;
				GetScheduler().CompleteDraws(consumed / 5u);
				continue;
			}
		}

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

		const uint32_t packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			if (execution.m_buffer_stack.size() > buffer_index + 1) {
				execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
			}
			return;
		}
		EXIT_IF(execution.m_buffer_stack.size() != buffer_index + 1);
		execution.m_buffer_stack[buffer_index].offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
		    opcode == Pm4::IT_DRAW_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
		    opcode == Pm4::IT_DRAW_INDIRECT_MULTI || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI)
			GetScheduler().CompleteDraws(1);
	}
}

// KYTY_SPECULATE=4's PM4 loop (ProcessPm4 chooses it): the state pass's and a speculation's checks, refusals and
// recoverable exits, and the GPU thread's chunk stops.
void CommandProcessor::ProcessPm4Spec(Pm4Execution& execution, size_t stop_depth) {
	// (Behind a plain global: KYTY_SPECULATE unset, no thread-local read.)
	auto* const spec = kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 ? Spec::Current() : nullptr;
	// A speculation's refused packet suspends the execution there (speculation.cpp goes on past it, or stops).
	const auto refuse = [&](const char* why) {
		Spec::Refuse(why);
		execution.m_suspended = true;
	};
	while (execution.m_buffer_stack.size() > stop_depth) {
		if (spec != nullptr && spec->refused != nullptr) return refuse(spec->refused);
		if (g_gpu_state != nullptr && spec == nullptr) { // (a speculation's thread runs no guest commands)
			g_gpu_state->ProcessCommands();
			// (KYTY_EAGER_FEEDBACK: completed copy-feedback snapshots written back between packets; its verify's
			// synchronous download only outside a recorder.)
			g_gpu_state->m_renderer.GetBufferCache().WriteBackCompletedFeedback(CommandScheduler::ThreadRecorder() ==
			                                                                    nullptr);
		}
		// (KYTY_WRITE_GUARD_SPEC: a page the GPU thread guarded since this thread's buffers were rebased: rebased again,
		// before the next packet is read through a mapping that has no access now. KYTY_WRITE_GUARD_LOCKFREE: only when a
		// page armed since lies in one of its buffers, or the ring of armed pages no longer holds them all.)
		if (spec != nullptr) {
			static thread_local uint64_t seen_arms = 0;
			if (const auto arms = WriteGuard::g_arms.load(std::memory_order_acquire); arms != seen_arms) {
				const uint64_t since = std::exchange(seen_arms, arms);
				if (!WriteGuard::LockFree() || WriteGuard::ArmedSinceIn(since, arms, [&](uint64_t page) {
					    for (const auto& buffer: execution.m_buffer_stack) {
						    const auto at = reinterpret_cast<uint64_t>(buffer.commands.data());
						    if (!buffer.commands.empty() && page < at + buffer.commands.size_bytes() && at < page + WriteGuard::Page)
							    return true;
					    }
					    return false;
				    }))
					RebaseGuardedCommands(&execution);
			}
		}
		const auto buffer_index = execution.m_buffer_stack.size() - 1;
		auto&      cursor       = execution.m_buffer_stack[buffer_index];
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.deferred_advance_dw != 0) {
			EXIT_IF(cursor.deferred_advance_dw > cursor.commands.size() - cursor.offset_dw);
			cursor.offset_dw += cursor.deferred_advance_dw;
			cursor.deferred_advance_dw = 0;
			execution.m_made_progress  = true;
			continue;
		}
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
		// A state pass or a speculation may pass a command buffer after the GPU thread translated it, and the game may
		// have written that memory again since (its labels are written then): a packet that does not fit ends a state
		// pass there (the rest is translated the normal way) and is refused by a speculation, instead of the exits
		// below (one ended a mode 4 run while the stone arch loaded, 10-31).
		if ((m_state_only || spec != nullptr) && packet_header != 0x80000000u &&
		    (remaining_dw < 2 || KYTY_PM4_LEN(packet_header) == 0 || KYTY_PM4_LEN(packet_header) > remaining_dw)) {
			if (spec != nullptr) return refuse("malformed packet");
			execution.m_suspended = true;
			return;
		}

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		// KYTY_SCOUT: queue 0's draws, numbered as the scout numbers them (skipped ones too).
		if (Scout::IsDraw(opcode) && !IsAsyncComputeQueue() && Scout::Running()) Scout::NoteDraw(packet);

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			if (kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0)
				DrawStateObserver::ObservePacket(opcode, packet, packet_dw, true);
#endif
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.load(std::memory_order_relaxed) < 2048 && skip_log_count.fetch_add(1, std::memory_order_relaxed) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.load(std::memory_order_relaxed) < 128 && log_count.fetch_add(1, std::memory_order_relaxed) < 128) {
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

		// A chunk's end (SetStopAt): before the packet the next begins at.
		if (m_stop_at != nullptr && SamePosition(execution, *m_stop_at)) {
			m_stop_reached        = true;
			execution.m_suspended = true;
			return;
		}
		// A state pass's chunk boundary (SetChunking): not within a run of indirect draws, which a translation takes
		// together (the packet runs when the pass resumes, counted again).
		if (m_chunk_every != 0 && IsChunkWork(opcode)) {
			if (m_chunk_work >= m_chunk_every &&
			    !(opcode == Pm4::IT_DRAW_INDEX_INDIRECT && m_chunk_last_opcode == Pm4::IT_DRAW_INDEX_INDIRECT)) {
				m_chunk_work          = 0;
				m_at_boundary         = true;
				execution.m_suspended = true;
				return;
			}
			++m_chunk_work;
		}
		m_chunk_last_opcode = opcode;
		if (spec != nullptr) {
			if (!GuestGpu::IsGpuThread()) Spec::EnterPacket();
			spec->Mark(LocalVulkanRecording::RecordedWrites());
			// (Between packets, nothing of this one recorded: what is translated so far is committed, the rest is taken the
			// normal way.)
			if (spec->cancel != nullptr && spec->cancel->load(std::memory_order_relaxed)) {
				Spec::Stop("cancelled: the GPU thread came for it");
				execution.m_suspended = true;
				return;
			}
			if (!SpeculationTakes(opcode, packet_header)) {
				// (By opcode in the speculation's report.)
				static const auto names = [] {
					std::array<std::string, 512> all; // (NOPs by their R type after the opcodes)
					for (uint32_t i = 0; i < 256; ++i) all[i] = "packet " + std::to_string(i);
					for (uint32_t i = 0; i < 256; ++i) all[256 + i] = "packet NOP R" + std::to_string(i);
					return all;
				}();
				return refuse(names[opcode == Pm4::IT_NOP ? 256u + (KYTY_PM4_R(packet_header) & 0xffu) : opcode].c_str());
			}
		}
		// The CP writes an indirect draw's start instance into the user SGPR the packet names
		// (START_INST_LOC); every path below translates the draw's shaders for it.
		switch (opcode) {
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
				m_sh_ctx.SetStartInstanceUserSgpr(GsUserSgprAt(packet[3] & 0xffffu));
				break;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DISPATCH_DRAW_PREAMBLE: m_sh_ctx.SetStartInstanceUserSgpr(-1); break;
			default: break;
		}
		// A state pass: what changes the processor's state only (a run of indirect draws ends within its packets).
		if (m_state_only) {
			if (opcode != Pm4::IT_DRAW_INDEX_INDIRECT) m_draw_run_skip = 0;
			// A SET_PREDICATION that reads its predicate (op != 0) ends the pass there: the value is the GPU's, current
			// only once the work before it is done (with wait_op the GPU thread flushes and waits first: SetPredication),
			// and its page may be read-protected (a plain read on this thread faults). The rest is translated the normal
			// way; a clear (op 0) is state only and runs.
			if (opcode == Pm4::IT_SET_PREDICATION && StatePassPredicationReads(packet)) {
				execution.m_suspended = true;
				return;
			}
			if (!StatePassRuns(opcode, packet_header)) {
				const auto packet_dw = KYTY_PM4_LEN(packet_header);
				EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
				cursor.offset_dw += packet_dw;
				execution.m_made_progress = true;
				continue;
			}
		}

		// Only state, draws and dispatches keep the deferred read protections pending: before a label, a memory write,
		// a wait or anything else the pages GPU work writes are protected (PageManager::DeferReadProtection).
		if (!KeepsDeferredProtection(opcode)) PageManager::FlushDeferredProtection();
		auto handler = g_cp_op_func[opcode];
		LiveCounters::AddPm4(opcode);
		if (DrawsOrDispatches(opcode)) PrefetchNextDraw(packet, remaining_dw);
		if (XprCapture::Enabled()) {
			if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT) {
				XprCapture::ObserveDraw(m_draw_indirect_args_base_addr + packet[1]);
			} else if (opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
			           opcode == Pm4::IT_DRAW_INDEX_OFFSET_2) {
				XprCapture::ObserveDraw(0);
			}
		}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		// Native XPR draws; on refusal the normal path stores a record, with the
		// reads of its SRT evaluation logged until this packet completes.
		std::optional<ShaderReadObserver> native_reads;
		struct NativeStoreEnd {
			RenderExecutor* executor = nullptr;
			~NativeStoreEnd() {
				if (executor) executor->NativeXprEndPacket();
			}
		} native_store_end;
		if (const auto native = kyty_local_native_xpr_mode.load(std::memory_order_relaxed); native != 0) {
			DrawStateObserver::ObservePacket(opcode, packet, std::min(KYTY_PM4_LEN(packet_header), remaining_dw),
			                                 false);
			if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT &&
			    (packet_header != 0xc0032500u || GraphicsRunDebugDumpEnabled()))
				m_renderer.GetRenderExecutor().NativeXprForgetState();
			if (packet_header == 0xc0032500u && !GraphicsRunDebugDumpEnabled()) {
				const auto consumed =
				    TryNativeXprDraws({packet, remaining_dw}, DrawStateObserver::g_shadow.last_clean);
				if (consumed) {
					m_draw_run_skip = 0;
					if (Scout::Running() && !IsAsyncComputeQueue()) Scout::SkipDraws(consumed / 5u - 1u);
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(consumed / 5u);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			} else if (packet_header == 0xc0083800u && !GraphicsRunDebugDumpEnabled()) {
				// Several indexed indirect draws of consecutive arguments, as a run of DRAW_INDEX_INDIRECT.
				const auto consumed = TryNativeMultiDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(packet[5]);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			} else if (packet_header == 0xc0012d00u && !GraphicsRunDebugDumpEnabled()) {
				// A draw of vertices as a table draw (the normal path stores the pair's state).
				const auto consumed = TryNativeAutoDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
			} else if (packet_header == 0xc0032400u && !GraphicsRunDebugDumpEnabled() && SpecDrawIndirect()) {
				// A draw of vertices of GPU-read arguments as a table draw (the normal path stores the pair's state).
				const auto consumed = TryNativeIndirectAutoDraw({packet, remaining_dw});
				if (consumed) {
					m_draw_run_skip = 0;
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
			} else if ((packet_header == 0xc0042700u || packet_header == 0xc0033500u) &&
			           !GraphicsRunDebugDumpEnabled()) {
				// Direct indexed draws from the same records (the normal path stores them).
				const auto consumed = TryNativeDirectDraw({packet, remaining_dw});
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
				auto& executor            = m_renderer.GetRenderExecutor();
				native_store_end.executor = &executor;
				if (auto* log = executor.NativeXprReadLog()) {
					native_reads.emplace(
					    [](void* userdata, uint64_t address, uint64_t size) {
						    static_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(userdata)->emplace_back(
						        address, size);
					    },
					    log);
				}
			}
		}
#endif
		if (opcode != Pm4::IT_DRAW_INDEX_INDIRECT) m_draw_run_skip = 0;
		if (packet_header == 0xc0032500u && !GraphicsRunDebugDumpEnabled()) {
			const auto consumed = TryDrawIndirectRun({packet, remaining_dw});
			if (consumed) {
				// (KYTY_SPEC_CHUNK_IN_ORDER: the run's other draws counted as the state pass counts them, one a packet, so
				// the in-order chunk ends where the pass cut it.)
				if (m_chunk_every != 0) m_chunk_work += consumed / 5u - 1u;
				if (Scout::Running() && !IsAsyncComputeQueue()) Scout::SkipDraws(consumed / 5u - 1u);
				execution.m_buffer_stack[buffer_index].offset_dw += consumed;
				execution.m_made_progress = true;
				GetScheduler().CompleteDraws(consumed / 5u);
				continue;
			}
		}

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

		if (spec != nullptr && (opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_2 ||
		                        opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
		                        opcode == Pm4::IT_DRAW_INDIRECT))
			return refuse("draw"); // (no table draw took it)
		uint32_t packet_dw = 0;
		// (Off the GPU thread a handler's exit, a packet that cannot be, is a worker reading command memory the game
		// reused or GPU work wrote: the speculation refuses it ("malformed packet", the spec report's stops) and a state
		// pass ends there, as 4c1e5d95's length check does; the render thread keeps the exits. A COPY_DATA of another
		// header ended a mode 4 run after a death, 5 Nov.)
		if (spec == nullptr && m_state_only && !GuestGpu::IsGpuThread()) {
			try {
				const Common::RecoverableExitScope recoverable;
				packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			} catch (const Common::RecoverableExit& malformed) {
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
					std::printf("Speculation: a state pass ended at a malformed packet (header 0x%08x): %s\n", packet_header,
					            malformed.message.c_str());
				execution.m_suspended = true;
				return;
			}
		} else if (spec == nullptr) {
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		} else {
			// What would submit or wait for the GPU refuses the packet.
			try {
				const Common::RecoverableExitScope recoverable;
				packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			} catch (const CommandScheduler::RecorderRefusal& refusal) {
				Spec::Refuse(refusal.what);
			} catch (const Common::RecoverableExit&) {
				Spec::Refuse("malformed packet");
			}
			if (spec->refused != nullptr) {
				// Refused in an indirect buffer it opened: this level goes on past it.
				if (execution.m_buffer_stack.size() > buffer_index + 1 && packet_dw != 0)
					execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
				return refuse(spec->refused);
			}
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			if (execution.m_buffer_stack.size() > buffer_index + 1) {
				execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
			}
			return;
		}
		EXIT_IF(execution.m_buffer_stack.size() != buffer_index + 1);
		execution.m_buffer_stack[buffer_index].offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
		    opcode == Pm4::IT_DRAW_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
		    opcode == Pm4::IT_DRAW_INDIRECT_MULTI || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI)
			GetScheduler().CompleteDraws(1);
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

	m_num_instances = num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	if (wait_op != 0 && !m_state_only) {
		BufferFlushAndWait();
	}

	(void)count_in_dwords;

	switch (op) {
		case 0x00: {
			m_predicate_skip = false;
		} break;
		case 0x03: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);

			auto value = *reinterpret_cast<const volatile uint64_t*>(address);

			switch (condition) {
				case 0x00: m_predicate_skip = (value != 0); break;
				case 0x01: m_predicate_skip = (value == 0); break;
				default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
			}
			static std::atomic<uint32_t> log_count {0};
			if (log_count.load(std::memory_order_relaxed) < 128 && log_count.fetch_add(1, std::memory_order_relaxed) < 128) {
				LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
				     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(address), value, condition,
				     m_predicate_skip ? 1u : 0u, wait_op);
			}
		} break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
}

bool CommandProcessor::DrawIndex(DrawIndexArgs args) {
	CheckBuffer();

	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (args.base_vertex != 0 || args.first_instance != 0) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	// KYTY_OUTPUT_HEIGHT: the interface's HD layout scaled to the output (hud-anchor.h).
	std::optional<HudAnchor::ScaleScope> scaled;
	if (HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs())) scaled.emplace(m_ctx, m_sh_ctx.GetPs());
	return m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

bool CommandProcessor::TryGpuIndirectDraw(uint64_t args, uint32_t count, uint32_t stride, bool indexed) {
	// Reading GPU-written arguments on the CPU waits for the GPU to finish all work queued so
	// far (0.1-0.3 ms). The draw uses them on the GPU instead, as native XPR runs do: an indexed
	// triangle list over the whole index buffer (primitive restart with a custom index scans the
	// indices a draw uses), or a non-indexed draw of a directly drawn type (a triangle strip here).
	using Prospero::PrimitiveType;
	const auto     primitive = m_ucfg.GetPrimType();
	const uint64_t size      = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	if (count == 0 || (args & 3u) != 0 || stride % 4u != 0 || stride < size ||
	    (indexed ? primitive != PrimitiveType::kTriList || m_index_type_and_size > 1 || !m_index_base_addr ||
	                   !m_index_buffer_size
	             : primitive != PrimitiveType::kTriList && primitive != PrimitiveType::kTriStrip &&
	                   primitive != PrimitiveType::kTriFan && primitive != PrimitiveType::kLineList &&
	                   primitive != PrimitiveType::kLineStrip && primitive != PrimitiveType::kPointList))
		return false;
	const uint64_t bytes     = uint64_t {count - 1u} * stride + size;
	auto&          resources = GetGpuResources();
	if (!resources.IsMapped(args, bytes) || !resources.GetBufferCache().HasGpuDirtyBytes(args, bytes) ||
	    resources.GetTextureCache().IsRegionGpuModified(args, bytes))
		return false;
	if (!indexed)
		return DrawIndexAuto({.vertex_count    = 1,
		                      .instance_count  = 1,
		                      .offset_source   = DrawOffsetSource::IndirectArgs,
		                      .gpu_args        = args,
		                      .gpu_args_count  = count,
		                      .gpu_args_stride = stride});
	return DrawIndex({.index_count     = m_index_buffer_size,
	                  .index_addr      = reinterpret_cast<const void*>(m_index_base_addr),
	                  .instance_count  = 1,
	                  .offset_source   = DrawOffsetSource::IndirectArgs,
	                  .gpu_args        = args,
	                  .gpu_args_count  = count,
	                  .gpu_args_stride = stride});
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

uint32_t CommandProcessor::TryNativeXprDraws(std::span<const uint32_t> packets, bool clean) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 0);
	LiveTrace::MarkAfter gpu_mark {[&] { return GetScheduler().Current().RawHandle(); }, LiveTrace::MarkNativeXpr,
	                               m_sh_ctx.GetPs().ps_regs.data_addr};
	LiveCounters::g_last_dispatch_shader = 0;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	auto& executor = m_renderer.GetRenderExecutor();
	if (FrameCapture::Active()) { // a recorded frame sees every draw on the normal path
		executor.NativeXprForgetState();
		return 0;
	}
	if (packets.size() < 5 || (packets[4] & ~0x20u) != 2u || m_index_type_and_size > 1 ||
	    !m_index_base_addr || !m_index_buffer_size || !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList ||
	    HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs())) {
		// The cached draw-state key covers only draws the native path saw.
		executor.NativeXprForgetState();
		return 0;
	}
	// A run: consecutive packets that differ only in the argument offset draw
	// the same object (no register write between them).
	constexpr uint32_t max_draws = 64;
	uint32_t           count     = 1;
	while (count < max_draws && (count + 1u) * 5u <= packets.size()) {
		const auto* next = packets.data() + count * 5u;
		if (next[0] != 0xc0032500u || next[2] != packets[2] || next[3] != packets[3] ||
		    next[4] != packets[4])
			break;
		++count;
	}
	std::array<uint64_t, max_draws> commands; // the first `count` are written, only they are read
	for (uint32_t i = 0; i < count; ++i) commands[i] = m_draw_indirect_args_base_addr + packets[i * 5u + 1u];
	CheckBuffer();
	const uint64_t element = m_index_type_and_size == 0 ? 2 : 4;
	if (clean && executor.TableContinue(CurrentBuffer(), {commands.data(), count}, m_index_base_addr,
	                                    uint64_t {m_index_buffer_size} * element,
	                                    m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
	                                    nullptr))
		return count * 5u;
	// On refusal NativeXprTry has asked the normal path to store when that helps.
	if (!executor.NativeXprTry(CurrentBuffer(), clean, {commands.data(), count}, m_index_base_addr,
	                           uint64_t {m_index_buffer_size} * element,
	                           m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32))
		return 0;
	return count * 5u;
#else
	(void)packets;
	(void)clean;
	return 0;
#endif
}

uint32_t CommandProcessor::TryNativeMultiDraw(std::span<const uint32_t> packet) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 0);
	LiveTrace::MarkAfter gpu_mark {[&] { return GetScheduler().Current().RawHandle(); }, LiveTrace::MarkNativeXpr,
	                               m_sh_ctx.GetPs().ps_regs.data_addr};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	auto&      executor = m_renderer.GetRenderExecutor();
	const auto skip     = [&] {
		executor.NativeXprForgetState();
		return 0u;
	};
	// What CpOpDrawIndirectMulti takes, of a count the packet holds (not read from memory), the arguments one after
	// another: the draws of as many DRAW_INDEX_INDIRECT (TryNativeXprDraws).
	constexpr uint32_t max_draws = 64;
	if (FrameCapture::Active() || packet.size() < 10 || ((packet[4] >> 30u) & 1u) != 0 || packet[5] == 0 ||
	    packet[5] > max_draws || packet[8] != sizeof(DrawIndexedIndirectArgs) || (packet[9] & ~0x20u) != 2u ||
	    m_index_type_and_size > 1 || !m_index_base_addr || !m_index_buffer_size || !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList || HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs()))
		return skip();
	const uint32_t                  count = packet[5];
	std::array<uint64_t, max_draws> commands; // the first `count` are written, only they are read
	for (uint32_t i = 0; i < count; ++i)
		commands[i] = m_draw_indirect_args_base_addr + packet[1] + uint64_t {i} * sizeof(DrawIndexedIndirectArgs);
	CheckBuffer();
	const uint64_t element    = m_index_type_and_size == 0 ? 2 : 4;
	const auto     index_type = m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	if (DrawStateObserver::g_shadow.last_clean &&
	    executor.TableContinue(CurrentBuffer(), {commands.data(), count}, m_index_base_addr,
	                           uint64_t {m_index_buffer_size} * element, index_type, nullptr))
		return 10;
	if (!executor.NativeXprTry(CurrentBuffer(), DrawStateObserver::g_shadow.last_clean, {commands.data(), count},
	                           m_index_base_addr, uint64_t {m_index_buffer_size} * element, index_type))
		return 0;
	return 10;
#else
	(void)packet;
	return 0;
#endif
}

uint32_t CommandProcessor::TryNativeDirectDraw(std::span<const uint32_t> packet) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 1);
	LiveTrace::MarkAfter gpu_mark {[&] { return GetScheduler().Current().RawHandle(); }, LiveTrace::MarkNativeXpr,
	                               m_sh_ctx.GetPs().ps_regs.data_addr};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	auto& executor = m_renderer.GetRenderExecutor();
	// A draw never tried leaves the cached draw-state key behind: the next clean draw would take it.
	const auto skip = [&] {
		executor.NativeXprForgetState();
		return 0u;
	};
	if (FrameCapture::Active() || XprCapture::Enabled() || m_index_type_and_size > 1 ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList || HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs()))
		return skip();
	// What CpOpDrawIndex / CpOpDrawIndexOffset accept (anything else takes them).
	const uint64_t element = m_index_type_and_size == 0 ? 2 : 4;
	uint64_t       index_address = 0;
	uint32_t       index_count = 0, size = 0;
	if (packet.size() >= 6 && packet[0] == 0xc0042700u) {
		index_address = packet[2] | (uint64_t {packet[3]} << 32u);
		index_count   = packet[4];
		size          = 6;
		if (index_count > packet[1] || (packet[5] & ~0x20u) != 0) return skip();
	} else if (packet.size() >= 5 && packet[0] == 0xc0033500u && m_index_base_addr != 0) {
		index_address = m_index_base_addr + uint64_t {packet[2]} * element;
		index_count   = packet[3];
		size          = 5;
		if (index_count > packet[1] || (packet[4] & ~0x20u) != 0) return skip();
	} else {
		return skip();
	}
	// A zero-sized draw draws nothing on the normal path either (it returns first).
	if (index_count == 0 || index_address == 0 || index_address % element != 0) return skip();
	CheckBuffer();
	const RenderExecutor::NativeXprDirectDraw draw {index_address, index_count, m_num_instances};
	const auto index_type = m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	if (DrawStateObserver::g_shadow.last_clean &&
	    executor.TableContinue(CurrentBuffer(), {}, index_address, uint64_t {index_count} * element, index_type, &draw))
		return size;
	if (!executor.NativeXprTry(CurrentBuffer(), DrawStateObserver::g_shadow.last_clean, {}, index_address,
	                           uint64_t {index_count} * element,
	                           m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32, &draw))
		return 0;
	return size;
#else
	(void)packet;
	return 0;
#endif
}

uint32_t CommandProcessor::TryNativeAutoDraw(std::span<const uint32_t> packet) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 2);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	using Prospero::PrimitiveType;
	// What CpOpDrawIndexAuto accepts and draws (no vertices: nothing), of a type drawn as it is.
	const auto primitive = m_ucfg.GetPrimType();
	if (packet.size() < 3 || packet[0] != 0xc0012d00u || (packet[2] & ~0x22u) != 0 || packet[1] == 0 ||
	    FrameCapture::Active() || XprCapture::Enabled() ||
	    (primitive != PrimitiveType::kTriList && primitive != PrimitiveType::kTriStrip))
		return 0;
	CheckBuffer();
	// KYTY_RENDER_ASPECT: the HUD at its sides of the screen, unstretched (hud-anchor.h); a draw over several of its
	// areas goes the normal way, drawn once for each. A menu's own draw notes the menu first (its interface then goes
	// to the middle, KYTY_PRESENT_ASPECT=fit).
	HudAnchor::NoteMenuDraw(m_sh_ctx.GetVs(), m_sh_ctx.GetPs());
	const auto hud =
	    HudAnchor::Split(HudAnchor::Classify(m_ctx, m_sh_ctx.GetVs(), m_ucfg.GetIndexOffset(), packet[1], primitive));
	if (hud.Several()) return 0u;
	std::optional<HudAnchor::Scope>        anchor;
	std::optional<HudAnchor::StretchScope> widen;
	std::optional<HudAnchor::ScaleScope>   scaled;
	if (hud.Single()) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), static_cast<uint32_t>(std::countr_zero(hud.areas)), false);
	} else if (hud.marker) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), HudAnchor::MarkerArea, false);
	} else {
		HudAnchor::RestoreMask(m_sh_ctx.GetPs());
		if (HudAnchor::Stretched(m_ctx)) widen.emplace(m_ctx, m_sh_ctx.GetPs());
		else if (HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs())) scaled.emplace(m_ctx, m_sh_ctx.GetPs());
	}
	const RenderExecutor::NativeXprDirectDraw draw {0, packet[1], m_num_instances, false};
	return m_renderer.GetRenderExecutor().NativeXprTry(CurrentBuffer(), false, {}, 0, 0, vk::IndexType::eUint16, &draw)
	           ? 3u
	           : 0u;
#else
	(void)packet;
	return 0;
#endif
}

uint32_t CommandProcessor::TryNativeIndirectAutoDraw(std::span<const uint32_t> packet) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 3);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	using Prospero::PrimitiveType;
	// What CpOpDrawIndirect / DrawIndirect accept (anything else exits there), drawn as TryGpuIndirectDraw draws a draw of
	// vertices from arguments the GPU reads (one, DrawIndirectArgs): a type drawn as it is, the arguments aligned, mapped
	// and over no image GPU work wrote. (Not its GPU-dirty rule: that picks the normal path's CPU read of arguments the GPU
	// did not write; the table draw reads them on the GPU whatever wrote them, as DRAW_INDEX_INDIRECT's table draws do.)
	const auto primitive = m_ucfg.GetPrimType();
	if (packet.size() < 5 || packet[0] != 0xc0032400u || (packet[4] & ~0x20u) != 2u || m_draw_indirect_args_base_addr == 0 ||
	    FrameCapture::Active() || XprCapture::Enabled() ||
	    (primitive != PrimitiveType::kTriList && primitive != PrimitiveType::kTriStrip &&
	     primitive != PrimitiveType::kTriFan && primitive != PrimitiveType::kLineList &&
	     primitive != PrimitiveType::kLineStrip && primitive != PrimitiveType::kPointList))
		return 0;
	const uint64_t     args      = m_draw_indirect_args_base_addr + packet[1];
	constexpr uint64_t size      = sizeof(DrawIndirectArgs);
	auto&              resources = GetGpuResources();
	if ((args & 3u) != 0 || !resources.IsMapped(args, size) || resources.GetTextureCache().IsRegionGpuModified(args, size))
		return 0;
	CheckBuffer();
	// The HUD as DrawIndexAuto places a draw of GPU-read arguments (a count of 1: TryGpuIndirectDraw's); a draw over
	// several of its areas goes the normal way, drawn once for each.
	HudAnchor::NoteMenuDraw(m_sh_ctx.GetVs(), m_sh_ctx.GetPs());
	const auto hud = HudAnchor::Split(HudAnchor::Classify(m_ctx, m_sh_ctx.GetVs(), m_ucfg.GetIndexOffset(), 1, primitive));
	if (hud.Several()) return 0u;
	std::optional<HudAnchor::Scope>        anchor;
	std::optional<HudAnchor::StretchScope> widen;
	std::optional<HudAnchor::ScaleScope>   scaled;
	if (hud.Single()) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), static_cast<uint32_t>(std::countr_zero(hud.areas)), false);
	} else if (hud.marker) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), HudAnchor::MarkerArea, false);
	} else {
		HudAnchor::RestoreMask(m_sh_ctx.GetPs());
		if (HudAnchor::Stretched(m_ctx)) widen.emplace(m_ctx, m_sh_ctx.GetPs());
		else if (HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs())) scaled.emplace(m_ctx, m_sh_ctx.GetPs());
	}
	// The arguments as a DRAW_INDEX_INDIRECT table draw's: obtained as a buffer range (a speculation notes it: its commit
	// checks nothing registered over it since and makes it current first), read on the GPU.
	const std::array<uint64_t, 1>             commands {args};
	const RenderExecutor::NativeXprDirectDraw draw {0, 0, 1, false, true};
	auto&                                     executor = m_renderer.GetRenderExecutor();
	executor.m_table_args_emit                         = {};
	if (!executor.NativeXprTry(CurrentBuffer(), false, commands, 0, 0, vk::IndexType::eUint16, &draw)) return 0u;
	LiveCounters::Add(LiveCounters::DrawIndirectTables);
	// KYTY_SPEC_DRAW_INDIRECT_VERIFY: every Nth, what it was translated from and drew, checked on the GPU thread (at
	// once here, a speculation's at its segment's commit).
	if (const auto every = SpecDrawIndirectVerifyEvery(); every > 0) {
		static std::atomic<uint64_t> taken {0};
		if (taken.fetch_add(1, std::memory_order_relaxed) % static_cast<uint64_t>(every) == 0) {
			const auto&             emitted = executor.m_table_args_emit;
			Spec::DrawIndirectVerify entry;
			entry.packet = reinterpret_cast<uint64_t>(packet.data());
			std::memcpy(entry.words.data(), packet.data(), sizeof(entry.words));
			entry.base      = m_draw_indirect_args_base_addr;
			entry.args      = args;
			entry.primitive = static_cast<uint32_t>(primitive);
			entry.buffer    = emitted.buffer;
			entry.offset    = emitted.offset;
			entry.stride    = emitted.stride;
			entry.count     = emitted.count;
			entry.copy      = emitted.copy;
			entry.copied    = emitted.copied;
			entry.read      = Libs::LibKernel::Memory::TryReadBackingToHost(args, entry.taken.data(), sizeof(entry.taken));
			if (auto* spec = Spec::Current(); spec != nullptr) {
				entry.segment = spec->closed.size(); // (the open segment)
				spec->draw_verifies.push_back(entry);
			} else {
				VerifyDrawIndirect(m_renderer, entry);
			}
		}
	}
	return 5u;
#else
	(void)packet;
	return 0;
#endif
}

uint32_t CommandProcessor::TryDrawIndirectRun(std::span<const uint32_t> packets) {
	constexpr uint32_t max_draws = 64;
	if (FrameCapture::Active()) return 0; // one record per draw while a frame is recorded
	if (m_draw_run_skip) {
		--m_draw_run_skip;
		return 0;
	}
	if (packets.size() < 10 || m_index_type_and_size > 1 || !m_index_base_addr ||
	    !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList)
		return 0;
	uint32_t count = 1;
	while (count < max_draws && (count + 1u) * 5u <= packets.size()) {
		const auto* next = packets.data() + count * 5u;
		if (next[0] != 0xc0032500u || next[2] != packets[2] || next[3] != packets[3] ||
		    next[4] != packets[4])
			break;
		++count;
	}
	if (count < 2 || (packets[4] & ~0x20u) != 2u) return 0;
	// A rejected run must not be prepared N, N-1, ... times on fallback.
	m_draw_run_skip                                = count - 1;
	auto&                                resources = GetGpuResources();
	std::array<DrawIndexArgs, max_draws> draws {};
	std::array<uint64_t, max_draws>      argument_addresses {};
	const uint32_t                       element_size = m_index_type_and_size == 0 ? 2 : 4;
	uint32_t                             active = 0, last_instances = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const uint64_t address = m_draw_indirect_args_base_addr + packets[i * 5u + 1u];
		argument_addresses[i]  = address;
		if (address < m_draw_indirect_args_base_addr || (address & 3u) ||
		    !resources.IsMapped(address, sizeof(DrawIndexedIndirectArgs)) ||
		    !LibKernel::Memory::SyncGpuCleanBacking(address, sizeof(DrawIndexedIndirectArgs)))
			return 0;
		DrawIndexedIndirectArgs args {};
		std::memcpy(&args, reinterpret_cast<const void*>(address), sizeof(args));
		last_instances         = args.instance_count;
		const uint32_t indices = m_index_buffer_size
		                             ? std::min(args.index_count_per_instance, m_index_buffer_size)
		                             : args.index_count_per_instance;
		if (!indices || !args.instance_count) continue;
		const uint64_t index_address =
		    m_index_base_addr + uint64_t(args.start_index_location) * element_size;
		if (index_address < m_index_base_addr ||
		    !resources.IsMapped(index_address, uint64_t(indices) * element_size))
			return 0;
		draws[active++] = {.index_count         = indices,
		                   .index_addr          = reinterpret_cast<const void*>(index_address),
		                   .instance_count      = args.instance_count,
		                   .index_type_and_size = m_index_type_and_size,
		                   .base_vertex         = static_cast<int32_t>(args.base_vertex_location),
		                   .first_instance      = args.start_instance_location,
		                   .offset_source       = DrawOffsetSource::IndirectArgs};
	}
	if (active < 2) return 0;
	CheckBuffer();
	if (!m_renderer.GetRenderExecutor().TryDrawIndexRun(m_submit_id, CurrentBuffer(),
	                                                    {draws.data(), active},
	                                                    {argument_addresses.data(), count})) {
		return 0;
	}
	m_num_instances = last_instances;
	m_draw_run_skip = 0;
	return count * 5u;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);
	if (TryGpuIndirectDraw(m_draw_indirect_args_base_addr + data_offset, 1,
	                       indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs), indexed))
		return;
	if (!Libs::LibKernel::Memory::SyncGpuCleanBacking(
	        m_draw_indirect_args_base_addr + data_offset,
	        indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs))) {
		static std::atomic<uint32_t> sync_fallback_logs {0};
		if (sync_fallback_logs.load(std::memory_order_relaxed) < 16 && sync_fallback_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("DrawIndirect: failed to synchronise indirect arguments at 0x%016" PRIx64
			     " (image-owned range, reading guest memory)\n",
			     m_draw_indirect_args_base_addr + data_offset);
		}
	}

	if (!indexed) {
		DrawIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		m_num_instances = args.instance_count;
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = args.start_instance_location,
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	DrawIndexedIndirectArgs args {};
	std::memcpy(&args, args_addr, sizeof(args));

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
		if (log_count.load(std::memory_order_relaxed) < 64 && log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
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
                                         bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		if (!Libs::LibKernel::Memory::SyncGpuCleanBacking(reinterpret_cast<uint64_t>(count_addr),
		                                                  sizeof(uint32_t))) {
			static std::atomic<uint32_t> sync_fallback_logs {0};
			if (sync_fallback_logs.load(std::memory_order_relaxed) < 16 && sync_fallback_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("DrawIndirectMulti: failed to synchronise the draw count at 0x%016" PRIx64
				     " (image-owned range, reading guest memory)\n",
				     reinterpret_cast<uint64_t>(count_addr));
			}
		}
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);
	if (TryGpuIndirectDraw(m_draw_indirect_args_base_addr + data_offset, draw_count, stride_in_bytes, indexed))
		return;
	if (!Libs::LibKernel::Memory::SyncGpuCleanBacking(
	        m_draw_indirect_args_base_addr + data_offset,
	        static_cast<uint64_t>(draw_count - 1u) * stride_in_bytes + args_size)) {
		static std::atomic<uint32_t> sync_fallback_logs {0};
		if (sync_fallback_logs.load(std::memory_order_relaxed) < 16 && sync_fallback_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("DrawIndirectMulti: failed to synchronise indirect arguments at 0x%016" PRIx64
			     " (image-owned range, reading guest memory)\n",
			     m_draw_indirect_args_base_addr + data_offset);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			m_num_instances = args->instance_count;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);

		uint64_t index_size = 0;
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.load(std::memory_order_relaxed) < 64 && log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode,
                                      uint64_t indirect_args) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	if (m_state_only) {
		if (auto* record = Spec::Record(); record != nullptr && indirect_args == 0)
			m_renderer.GetComputeRenderExecutor().PredictDispatch(m_sh_ctx, thread_group_x, thread_group_y, thread_group_z,
			                                                      mode, *record);
		return;
	}

	uint32_t frame_num = 0;

	{
		CheckBuffer();
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.load(std::memory_order_relaxed) < 1024 && log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
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

		const auto& cs = m_sh_ctx.GetCs().cs_regs;
		if (cs.wave_size == 64u) {
			static std::atomic_bool logged_wave64_shader {false};
			if (!logged_wave64_shader.exchange(true, std::memory_order_relaxed)) {
				LOGF("warning: executing wave64 compute shader cs=0x%016" PRIx64 "\n",
				     cs.data_addr);
				std::printf("warning: executing wave64 compute shader cs=0x%016" PRIx64 "\n",
				            cs.data_addr);
				std::fflush(stdout);
			}
		}

		m_renderer.GetComputeRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                                     thread_group_y, thread_group_z, mode,
		                                                     indirect_args);
	}
}

void CommandProcessor::DispatchIndirect(uint32_t data_offset, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(m_dispatch_indirect_args_base_addr == 0);
	DispatchIndirectAddress(m_dispatch_indirect_args_base_addr + data_offset, mode);
}

void CommandProcessor::DispatchIndirectAddress(uint64_t args_addr, uint32_t mode) {
	// (A state pass: its state only.)
	if (m_state_only) return m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	struct DispatchIndirectArgs {
		uint32_t thread_group_x, thread_group_y, thread_group_z;
	};
	EXIT_NOT_IMPLEMENTED(args_addr == 0);
	// Workgroup dimensions are consumed by Vulkan. Only thread-dimension mode
	// needs CPU counts for specialization/conversion; do not download GPU-written
	// arguments solely to pass unused group counts back to the indirect command.
	if ((mode & 0x20u) == 0 && (args_addr & 3u) == 0 &&
	    GetGpuResources().IsMapped(args_addr, sizeof(DispatchIndirectArgs)) &&
	    !GetGpuResources().GetTextureCache().IsRegionGpuModified(args_addr,
	                                                             sizeof(DispatchIndirectArgs))) {
		DispatchDirect(1, 1, 1, mode, args_addr);
		return;
	}
	if (!Libs::LibKernel::Memory::SyncGpuCleanBacking(args_addr, sizeof(DispatchIndirectArgs))) {
		static std::atomic<uint32_t> sync_fallback_logs {0};
		if (sync_fallback_logs.load(std::memory_order_relaxed) < 16 && sync_fallback_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("DispatchIndirect: failed to synchronise indirect arguments at 0x%016" PRIx64
			     " (image-owned range, reading guest memory)\n",
			     args_addr);
		}
	}
	DispatchIndirectArgs args {};
	std::memcpy(&args, reinterpret_cast<const void*>(args_addr), sizeof(args));

	DispatchDirect(args.thread_group_x, args.thread_group_y, args.thread_group_z, mode, args_addr);
}

bool CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	CheckBuffer();

	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	// KYTY_RENDER_ASPECT: the HUD at its sides of the screen, unstretched (hud-anchor.h); a menu's own draw notes the
	// menu first.
	HudAnchor::NoteMenuDraw(m_sh_ctx.GetVs(), m_sh_ctx.GetPs());
	const auto hud = HudAnchor::Split(
	    HudAnchor::Classify(m_ctx, m_sh_ctx.GetVs(), m_ucfg.GetIndexOffset(), args.vertex_count, m_ucfg.GetPrimType()));
	if (hud.areas == 0 || hud.Single()) {
		std::optional<HudAnchor::Scope>        anchor;
		std::optional<HudAnchor::StretchScope> widen;
		std::optional<HudAnchor::ScaleScope>   scaled;
		if (hud.Single()) {
			anchor.emplace(m_ctx, m_sh_ctx.GetPs(), static_cast<uint32_t>(std::countr_zero(hud.areas)), false);
		} else if (hud.marker) {
			anchor.emplace(m_ctx, m_sh_ctx.GetPs(), HudAnchor::MarkerArea, false);
		} else {
			HudAnchor::RestoreMask(m_sh_ctx.GetPs());
			if (HudAnchor::Stretched(m_ctx)) widen.emplace(m_ctx, m_sh_ctx.GetPs());
			else if (HudAnchor::Scaled(m_ctx, m_sh_ctx.GetVs(), m_sh_ctx.GetPs())) scaled.emplace(m_ctx, m_sh_ctx.GetPs());
		}
		return m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
	}
	HudAnchor::RestoreMask(m_sh_ctx.GetPs());
	bool drawn = true;
	for (uint32_t area = 0; area < 32; ++area) {
		if ((hud.areas & (1u << area)) == 0) continue;
		HudAnchor::Scope anchor(m_ctx, m_sh_ctx.GetPs(), area, true);
		drawn = m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args) && drawn;
	}
	if (hud.outside) {
		std::array<HudAnchor::Rect, 32> rects {};
		const auto                      count = HudAnchor::Complement(m_ctx, rects);
		// (The wide output's: widened as the canvas's other draws, the rectangles with them.)
		const bool wide = HudAnchor::Stretched(m_ctx);
		for (uint32_t r = 0; r < count; ++r) {
			std::optional<HudAnchor::StretchScope> widen;
			if (wide) widen.emplace(m_ctx, m_sh_ctx.GetPs());
			HudAnchor::ClipScope clip(m_ctx, wide ? HudAnchor::Widened(rects[r]) : rects[r]);
			drawn = m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args) && drawn;
		}
	}
	return drawn;
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

	CheckBuffer();
	BreakComputeChain();

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
				Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		WriteGuestMemory(dst, &data, sizeof(data));

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, CurrentBuffer(), dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, CurrentBuffer(), dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, CurrentBuffer(), dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, CurrentBuffer(), dst, data);
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
					Sync::WriteAtEndOfPipeGds32(m_submit_id, CurrentBuffer(), dst, value & 0xffffu,
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
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					WriteGuestMemory(dst, &value, sizeof(value));

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, CurrentBuffer(), dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, CurrentBuffer(), dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, CurrentBuffer(), dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, CurrentBuffer(), dst, value);
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
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x2f:
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
								     event_index == 0x05 && !with_interrupt) ||
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
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint64_t)) {
				const auto clock = Sync::ReadReferenceClock();
				auto*      dst   = static_cast<uint64_t*>(dst_gpu_addr);
				WriteGuestMemory(dst, &clock, sizeof(clock));
				switch (cache_action) {
					case 0x00:
						if ((eop_event_type == 0x04 && event_index == 0x05) ||
						    (eop_event_type == 0x28 && event_index == 0x00)) {
							if (with_interrupt) {
								Sync::WriteAtEndOfPipeWithInterrupt64(
								    m_submit_id, CurrentBuffer(), dst, clock, m_interrupt_event_id,
								    interrupt_context_id);
							} else {
								Sync::WriteAtEndOfPipeClockCounter(m_submit_id, CurrentBuffer(),
								                                   dst, clock);
							}
							return;
						}
						break;
					case 0x38:
						if ((eop_event_type == 0x04 &&
						     (event_index == 0x00 || event_index == 0x05)) ||
						    (eop_event_type == 0x28 && event_index == 0x00)) {
							if (with_interrupt) {
								Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
								    m_submit_id, CurrentBuffer(), dst, clock, m_interrupt_event_id,
								    interrupt_context_id);
							} else {
								Sync::WriteAtEndOfPipeClockCounterWithWriteBack(
								    m_submit_id, CurrentBuffer(), dst, clock);
							}
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

void CommandProcessor::EmitGlobalBarrier() {
	CheckBuffer();
	// In a run of guest buffer copies (everything recorded since its barrier is its copies) the
	// barrier waits for the run's end. The run already orders what came before it ahead of its copies, its copies
	// among themselves where they touch the same bytes (CopyRunHandle starts a new run), and its end orders the copies
	// ahead of what follows: through them, what came before too (one chain of dependencies). The game's linear copies
	// (cs_memset32, ~70 at the end of a frame) each followed by a compute partial flush made a run of their own each:
	// ~5 barriers a copy, drains in the GPU's frame tail (Latria turning in place, same process: 69.8 -> 70.8 fps).
	if (CurrentBuffer().InCopyRun()) return;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	{
		// KYTY_GLOBAL_BARRIER_DEDUPE: a barrier orders all earlier work in submission order, so
		// one with nothing recorded since the previous barrier of the same command buffer orders nothing new (draws
		// recorded as packets are work too).
		auto&      last_work = CurrentBuffer().BarrierWork();
		const auto work      = LocalVulkanRecording::RecordedWork();
		if (work == last_work && kyty_local_global_barrier_dedupe.load(std::memory_order_relaxed) != 0) return;
		last_work = work + 1; // this barrier's own recorded call
	}
#endif

	// (A speculation's thread records into its own command buffer: nothing the render lock guards.)
	std::optional<RenderLockGuard> lock;
	if (CommandScheduler::ThreadRecorder() == nullptr) lock.emplace(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().HandleForFullBarrier().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	CheckBuffer();

	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::BreakComputeChain() {
	// Guest synchronization may be implemented without a host command.
	// It still ends the compatibility profile's independent dispatch group.
	if (!GetScheduler().Active()) return;
	std::optional<RenderLockGuard> lock; // (as EmitGlobalBarrier's)
	if (CommandScheduler::ThreadRecorder() == nullptr) lock.emplace(m_renderer.GetMutex());
	auto& buffer = CurrentBuffer();
	if (buffer.ComputeChainPending()) (void)buffer.Handle();
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	// (A state pass: the occlusion counter a dump moves, nothing written.)
	if (m_state_only) {
		if (event_type == 0x00000039 && event_index == 0x00000001 && event_address != 0)
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & ((1ull << 63u) - 1u);
		return;
	}
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
			LiveCounters::Add(LiveCounters::GuestBarrierCs);
			EmitGlobalBarrier();
			break;
		case 0x0000000f:
		case 0x00000010:
			LiveCounters::Add(LiveCounters::GuestBarrierGfx);
			EmitGlobalBarrier();
			break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			LiveCounters::Add(LiveCounters::GuestBarrierCache);
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			LiveCounters::Add(LiveCounters::GuestBarrierCache);
			EmitGlobalBarrier();
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
			BreakComputeChain();
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				WriteGuestMemory(results + db * 2u, &value, sizeof(value));
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	LiveControl::Flip();
	FrameCapture::OnFlip();
	CheckBuffer();
	// (As a CPU flip's preparation does: optimized builds in place, idle shader plans out of memory.)
	m_renderer.GetPipelineCache().PromoteFinished();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	HudAnchor::NoteFlip();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	LiveTrace::Event(LiveTrace::FlipSubmit, request);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	CheckBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	if (Scout::Running()) Scout::NoteHostWrite(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value));
	auto& command = CurrentBuffer();
	HudAnchor::NoteFlip();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	LiveTrace::Event(LiveTrace::FlipSubmit, request);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	CheckBuffer();

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
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	if (Scout::Running()) Scout::NoteHostWrite(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value));
	auto& command = CurrentBuffer();
	HudAnchor::NoteFlip();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	LiveTrace::Event(LiveTrace::FlipSubmit, request);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	CheckBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	LiveTrace::Event(LiveTrace::FlipSubmit, request_id, 1);
	HudAnchor::NoteFlip();
	m_renderer.GetVideoOut().PrepareFlip(request_id, CurrentBuffer());
	GetScheduler().Flush();
	m_renderer.GetVideoOut().CompleteFlip(request_id);
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

static thread_local bool g_speculation_thread = false;

bool GuestGpu::IsTranslatingThread() noexcept {
	return g_gpu_thread || g_speculation_thread;
}

void GuestGpu::SetSpeculationThread(bool speculation) noexcept {
	g_speculation_thread = speculation;
}

void GuestGpu::SetOfflineGpuThread(bool gpu_thread) noexcept {
	g_gpu_thread = gpu_thread;
}

} // namespace Libs::Graphics
