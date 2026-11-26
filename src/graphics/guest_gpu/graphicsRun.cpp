#include "local/skip-shader.h"
#include "local/vgeo-nativo.h" // (w11, KYTY_NATIVO_VGEO)
#include "local/censo-pm4.h" // (w18, KYTY_CENSO_PM4: [censo-pm4])
#include "local/envio1.h"
#include "local/envio2.h"
#include "local/envio3.h"
#include "local/sync-stats.h"
#include "local/copia1.h" // (copia1)
#include "graphics/guest_gpu/graphicsRun.h"
#include "local/anti-lag.h"

#include "common/assert.h"
#include "common/guardedCopy.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
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

#include "live-trace.h"
#include "live-trace-gpu.h"
#include "live-census.h"
#include "otros2.h" // (otros2)
#include "gancho2.h" // (gancho2)
#include "port1.h" // (port1, KYTY_PORT1)
#include "port2.h" // (port2, KYTY_PORT2)
#include "f0-measure.h" // (f0, KYTY_F0)
#include "native/native-render.h" // (f5, KYTY_NATIVE_RENDER)
#include "native/verify-gpu.h" // (f5b, KYTY_NATIVE_PASSES ..:2)
#include "native/g1-probe.h" // (g1, KYTY_G1_PROBE / KYTY_G1_STATE)
#include "local/pathtrace.h" // (pt1, BB_PATHTRACING)
#include "render5.h"
#include "draw-state-observer.h"
#include "frame-capture.h"
#include "hud-anchor.h"
#include "live-control.h"
#include "trusted-depth.h" // (trust1)
#include "ds-linear-copy-hook.h"
#include "local/ds-nativo-pase.h" // (F5 pase 1, KYTY_NATIVO_PASE_SSRCLEAR)
#include "local/ds-nativo-sombras.h" // (F5 pase 2, KYTY_NATIVO_PASE_SOMBRAS=2: volcado del VS de sombras)
#include "local/ds-censo-dcb.h" // (w7, KYTY_CENSO_DCB: [censo-dcb])
#include "local/ds-censo-escena.h" // (F5 pase 3, KYTY_NATIVO_PASE_ESCENA=2: censo por pase)
#include "render1.h" // (render1 F0+F1)
#include "f5p1-techo.h" // (f5p1 techo, KYTY_F5P1_TECHO)
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "blackbox-run.h"
#include "performance-switches.h"
#include "speculation-state.h"
#include "xpr-capture.h"
#include "par-queues.h" // (par1, KYTY_PAR_QUEUES)
#include "local-platform.h"
#include "parpadeo.h" // (parpadeo2)
#include "parpadeo7.h" // (parpadeo7)
#include "local/juego3.h" // (juego3, KYTY_JUEGO3)
#include "p7-compat.h" // (p7, KYTY_P7)
#include "local/serie1.h" // (serie1: KYTY_SERIE1)
#include "local/parpadeo8.h" // (parpadeo8, w21: el tick del ultimo cull para 0xa1f290)
#include "local/frame-nativo2.h" // (FRAME-NATIVO2 v13: g_guest_consumed)
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <bit>
#include <x86intrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <string>
#include <vector>

namespace Libs::Graphics {

extern "C" {
// 0: synchronous frame boundary. 1: the guest may prepare the following frame
// while the renderer consumes the previous one; guest reads of GPU-written
// memory are then served by asynchronous readbacks (BufferCache).
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_frame_pipeline_mode {0};
// 1: a full barrier with no recorded work since the previous one is skipped.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_global_barrier_dedupe {0};
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
	// (parpadeo7) Its helper first: it posts to this queue, and copies what it holds while the GPU thread still runs.
	if (Parpadeo7::On() || Juego3::On()) m_renderer.GetBufferCache().StopPrefetchHelper(); // (juego3: its prefetch too)
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

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
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
	// (par1: two threads wait on it; the GPU thread runs the guest commands)
	if (ParQueues::On()) m_work_available.SignalAll();
	else m_work_available.Signal();
}

// (parpadeo7) No Drain of the caller's recording (it hands none over) and no run in place on the GPU thread: the
// command may record and flush, which the packet being translated must not see.
bool GuestGpu::PostCommand(Common::UniqueFunction<void>&& command) {
	if (!command) return false;
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) return false;
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	if (ParQueues::On()) m_work_available.SignalAll();
	else m_work_available.Signal();
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

// (render3, KYTY_RENDER_STATS) A packet's class for the [render] line's "pm4" part (LiveCensus::Pm4Clock).
static uint32_t Pm4ClassOf(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2: return LiveCensus::Pm4Draw;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return LiveCensus::Pm4Dispatch;
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_SH_REG_INDIRECT: return LiveCensus::Pm4ShReg;
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: return LiveCensus::Pm4ContextReg;
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return LiveCensus::Pm4UconfigReg;
		case Pm4::IT_NOP: return LiveCensus::Pm4Nop;
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_INDIRECT_BUFFER_CNST: return LiveCensus::Pm4IndirectBuffer;
		default: return LiveCensus::Pm4Other;
	}
}

// (render5) KYTY_REG_LANE (render5.h): the packets the register lane takes - register writes and index / instance /
// indirect base state. Their handlers only set the command processor's registers and fields (never a memory access, a
// wait, a suspension or another buffer), and each keeps the deferred protections pending (KeepsDeferredProtection).
static constexpr auto RegLaneBits = [] {
	std::array<uint64_t, 4> bits {};
	for (const uint32_t opcode: {Pm4::IT_SET_SH_REG, Pm4::IT_SET_CONTEXT_REG, Pm4::IT_SET_UCONFIG_REG,
	                             Pm4::IT_SET_UCONFIG_REG_INDEX, Pm4::IT_INDEX_BASE, Pm4::IT_INDEX_BUFFER_SIZE,
	                             Pm4::IT_INDEX_TYPE, Pm4::IT_NUM_INSTANCES, Pm4::IT_SET_BASE})
		bits[opcode / 64u] |= uint64_t {1} << (opcode % 64u);
	return bits;
}();
static bool RegLaneOpcode(uint32_t opcode) {
	return ((RegLaneBits[(opcode >> 6u) & 3u] >> (opcode & 63u)) & 1u) != 0;
}
// A packet the lane takes at dword `at` of `total`: a whole type-3 lane packet without the predicate bit (a predicated
// one goes back to the walk, which decides whether it is skipped).
static bool RegLanePacket(const uint32_t* commands, uint32_t at, uint32_t total) {
	if (at >= total || total - at < 2) return false;
	const uint32_t header = commands[at];
	return (header >> 30u) == 3u && (header & 1u) == 0 && RegLaneOpcode((header >> 8u) & 0xffu) &&
	       KYTY_PM4_LEN(header) <= total - at;
}
// Packets a lane run takes at most before the walk's per-packet work runs again (the guest commands and the copy
// feedback between packets wait at most this long: ~64 register packets, a few microseconds).
constexpr uint32_t RegLaneMaxPackets = 64;

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

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	// (par1fix, KYTY_PAR_QUEUES=1) The guest commands are the GPU thread's only: the ACB thread runs none (its walk's
	// call returns here). Before, both threads ran them between packets: the ACB thread saw the pending count of a command
	// the GPU thread's loop (ThreadRun) then took, and found the queue empty under the mutex - the EXIT_IF below
	// (20261116-074057, loading). One consumer again, as with one thread.
	if (ParQueues::t_side == 1) return;
	Juego3::NoteGpuThread(); // (juego3: its CPU per window)
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			// (The count is read outside the mutex: with one consumer it cannot be ahead of the queue, but stay safe.)
			if (ParQueues::On() && m_commands.empty()) break;
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		// Fault, readback and mapping callbacks cannot overtake a queued draw.
		PageManager::FlushDeferredProtection();
		LiveCounters::Add(LiveCounters::GuestCommands);
		LiveTrace::Event(LiveTrace::GuestCommand, 1, 0);
		command();
		LiveTrace::Event(LiveTrace::GuestCommand, 0, 0);
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Juego3::Scope juego3_wait(Juego3::TagSendSync); // (juego3: the game thread waiting for the GPU thread's turn)
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
		FrameNativo2::g_guest_consumed.fetch_add(1, std::memory_order_acq_rel); // (FRAME-NATIVO2 v13: sin tuberia, todo traducido)
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
		if (a.m_buffer_stack[i].commands.data() != b.m_buffer_stack[i].commands.data() ||
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

bool CommandProcessor::SameState(const CommandProcessor& other) const {
	return m_draw_run_skip == other.m_draw_run_skip && m_ctx == other.m_ctx && m_saved_ctx == other.m_saved_ctx &&
	       m_ucfg == other.m_ucfg && m_sh_ctx == other.m_sh_ctx && SameDrawState(other);
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
}

// KYTY_NATIVE_MEMORY, at each flip (once a frame): what the guest wrote since the last sweep is seen
// (GpuResourceManager::SweepCpuWrites), and all the GPU wrote is put in guest memory before the game can learn the frame
// is done (GpuResourceManager::NativeFlushGpuWrites). Nothing otherwise.
static void NativeFlipSync(GpuResourceManager& resources, bool state_only) {
	if (state_only || !PageManager::NativeWriteWatch()) return;
	resources.NoteFlipForWatchStats(); // (ww2: the "Write watch+" line counts per frame)
	resources.SweepCpuWrites();
	resources.NativeFlushGpuWrites();
}

// (ww2, KYTY_WRITE_WATCH_MODE=poll) The last word each command processor wrote at an address (WriteGuestMemory: labels,
// end-of-pipe values), a slot an address (a collision forgets the older one). A WAIT_REG_MEM satisfied while memory
// holds exactly what a processor wrote there was satisfied by the GPU side - one queue's commands write a label,
// another's wait for it (Demon's Souls: the skinning and particle queues against the graphics queue) - and hands over
// nothing the game wrote: no sweep of the write watch after it (CommandProcessor::WaitRegMem). Anything else - a
// value the game wrote, an address never written here, one written in more than 64 bytes - sweeps as before.
// (The game writing, over a label of the GPU side, the very value last written there, as its own signal, would be
// missed: the same value says nothing new to a waiter on the console either.)
namespace {
struct ProcessorLabels {
	static constexpr size_t Slots = 4096;
	std::mutex              mutex;
	uint64_t                address[Slots] = {};
	uint32_t                word[Slots]    = {};

	static size_t SlotOf(uint64_t at) noexcept { return static_cast<size_t>((at >> 2u) * 0x9E3779B97F4A7C15ull >> 52u) % Slots; }

	void Note(uint64_t at, const void* src, uint32_t size) noexcept {
		if (size == 0 || size > 64 || (at & 3u) != 0 || (size & 3u) != 0) return;
		std::lock_guard lock(mutex);
		for (uint32_t offset = 0; offset < size; offset += 4) {
			const auto slot = SlotOf(at + offset);
			address[slot]   = at + offset;
			std::memcpy(&word[slot], static_cast<const uint8_t*>(src) + offset, 4);
		}
	}
	// Whether every word of [at, at + size) was last written here with the bytes of `value`.
	[[nodiscard]] bool Holds(uint64_t at, const void* value, uint32_t size) noexcept {
		if (size == 0 || (at & 3u) != 0 || (size & 3u) != 0) return false;
		std::lock_guard lock(mutex);
		for (uint32_t offset = 0; offset < size; offset += 4) {
			const auto slot = SlotOf(at + offset);
			if (address[slot] != at + offset || std::memcmp(&word[slot], static_cast<const uint8_t*>(value) + offset, 4) != 0)
				return false;
		}
		return true;
	}
};
ProcessorLabels g_processor_labels;

// (p7, C5: KYTY_P7_C5=1, local/p7-compat.h) Labels written when the GPU has run the work recorded before them: the
// write is a priority operation of the open command buffer's tick (CommandScheduler::DeferPriorityOperation: run in
// order, once that tick is complete), not a store at translation. A later immediate write of the command processor to
// an address with a label still pending goes the same way, behind it (the order of the two is kept). Only on the GPU
// thread, without speculation or KYTY_PAR_QUEUES (their own write logs) and without a recorder.
struct DeferredLabels {
	std::mutex                                                   mutex;
	std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> pending; // address -> (labels pending, bytes)
	std::atomic<uint32_t>                                        count {0};
	CommandScheduler*                                            scheduler = nullptr;

	[[nodiscard]] bool Overlaps(uint64_t at, uint32_t size) {
		std::lock_guard lock(mutex);
		for (const auto& [address, entry]: pending)
			if (at < address + entry.second && address < at + size) return true;
		return false;
	}
};
DeferredLabels g_deferred_labels;
} // namespace

static bool DeferLabelWrite(CommandScheduler& scheduler, void* dst, const void* src, uint32_t size) {
	if (size == 0 || size > 8 || Spec::t_record != nullptr || Spec::Current() != nullptr || ParQueues::On() ||
	    kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 || !GuestGpu::IsGpuThread() ||
	    CommandScheduler::ThreadRecorder() != nullptr || !scheduler.Active())
		return false;
	uint64_t value = 0;
	std::memcpy(&value, src, size);
	const auto at = reinterpret_cast<uint64_t>(dst);
	{
		std::lock_guard lock(g_deferred_labels.mutex);
		auto& entry  = g_deferred_labels.pending[at];
		entry.first += 1;
		entry.second = std::max(entry.second, size);
	}
	g_deferred_labels.scheduler = &scheduler;
	g_deferred_labels.count.fetch_add(1, std::memory_order_acq_rel);
	scheduler.DeferPriorityOperation([at, value, size] {
		std::memcpy(reinterpret_cast<void*>(at), &value, size);
		// (nativo4: also with KYTY_DS_NATIVE_ALL - a wait satisfied by a label the command processor wrote is no hand-over.)
		if (PageManager::WatchPolled() || Libs::Graphics::GpuResourceManager::NativeAllOn()) g_processor_labels.Note(at, &value, size);
		{
			std::lock_guard lock(g_deferred_labels.mutex);
			if (const auto found = g_deferred_labels.pending.find(at);
			    found != g_deferred_labels.pending.end() && --found->second.first == 0)
				g_deferred_labels.pending.erase(found);
		}
		g_deferred_labels.count.fetch_sub(1, std::memory_order_acq_rel);
	});
	return true;
}

// A guest memory write of the command processor: now, or when a speculative translation is committed
// (speculation-state.h), in order with its other writes.
static void WriteGuestMemory(void* dst, const void* src, uint32_t size) {
	// (A state pass's: its record keeps it.)
	if (auto* record = Spec::t_record) {
		for (uint32_t offset = 0; offset < size; offset += 4) {
			uint32_t word = 0;
			std::memcpy(&word, static_cast<const uint8_t*>(src) + offset, 4);
			record->Journal(reinterpret_cast<uint64_t>(dst) + offset, word, 4);
		}
		return;
	}
	auto* spec = Spec::Current();
	// (p7, C5: an address a label is still pending for - written at GPU completion -: this write behind it.)
	if (spec == nullptr && g_deferred_labels.count.load(std::memory_order_acquire) != 0 &&
	    g_deferred_labels.scheduler != nullptr && GuestGpu::IsGpuThread() &&
	    g_deferred_labels.Overlaps(reinterpret_cast<uint64_t>(dst), size) &&
	    DeferLabelWrite(*g_deferred_labels.scheduler, dst, src, size)) {
		P7::Add(P7::LabelsOrdered);
		return;
	}
	if (spec == nullptr) {
		// (KYTY_NATIVE_MEMORY: this store's write fault on a page a cache watches invalidated the caches there before the
		// bytes landed - a GPU-written page read back first -, ahead of the packets after it. No fault comes now, and the
		// next sweep may be after those packets: the same invalidation, made here.)
		// (raiz1, KYTY_WRITE_WATCH_MODE=protect - KYTY_RAIZ -: the store faults as without native memory; not here.)
		if (PageManager::NativeWriteWatch() && !PageManager::WatchProtects() && GuestGpu::IsGpuThread())
			(void)Libs::LibKernel::Memory::TryPrepareHostWrite(reinterpret_cast<uint64_t>(dst), size);
		std::memcpy(dst, src, size);
		LiveCensus::Par2::Write(reinterpret_cast<uint64_t>(dst)); // (thread1, KYTY_RENDER_STATS: measurement only)
		// (par1, KYTY_PAR_QUEUES=1: a queue of the other translating thread waiting at a WAIT_REG_MEM checks again.)
		if (ParQueues::On() && g_gpu_state != nullptr) g_gpu_state->ParWake();
		// (ww2: what a WAIT_REG_MEM there is satisfied by, g_processor_labels.)
		if (PageManager::WatchPolled() || Libs::Graphics::GpuResourceManager::NativeAllOn()) g_processor_labels.Note(reinterpret_cast<uint64_t>(dst), src, size);
		Spec::NoteHostWrite(reinterpret_cast<uint64_t>(dst), size);
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
		// (KYTY_NATIVE_MEMORY: a value GPU work wrote - a WAIT_REG_MEM on memory a shader or a copy writes - was read back
		// by this read's fault; no fault comes now: read back first, as the indirect arguments are.)
		// (raiz1, mode protect: the read faults as without native memory; not here.)
		if (PageManager::NativeWriteWatch() && !PageManager::WatchProtects() && GuestGpu::IsGpuThread())
			(void)Libs::LibKernel::Memory::SyncGpuCleanBacking(reinterpret_cast<uint64_t>(address), sizeof(T));
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
		LiveCensus::Par2::Waited(reinterpret_cast<uint64_t>(addr)); // (thread1, KYTY_RENDER_STATS: measurement only)
		// (KYTY_NATIVE_MEMORY: what the guest wrote before the label it waited for, seen before the packets after it.)
		// (ww2, KYTY_WRITE_WATCH_MODE=poll: not when the value that satisfies the wait is one a command processor wrote
		// there itself - the game handed nothing over through this label -, g_processor_labels.)
		// (nativo4, KYTY_DS_NATIVE_ALL: the same test - such a wait skips the native write-watch sweep.)
		const bool processor_label = (PageManager::WatchPolled() || Libs::Graphics::GpuResourceManager::NativeAllOn()) &&
		                             g_processor_labels.Holds(reinterpret_cast<uint64_t>(addr), &value, static_cast<uint32_t>(sizeof(T)));
		GetGpuResources().SweepCpuWrites(processor_label ? GpuResourceManager::SweepReason::ProcessorWait
		                                                 : GpuResourceManager::SweepReason::Handover);
		BdaDirtyRegions::Publish();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

static void PrefetchXprBits(uint64_t label, GpuResourceManager& resources); // (lab1 p7b2, below)

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
	// (lab1 p7b2) The XPR labels are a WRITE_DATA, not an end-of-pipe write: 0x912960 -> 0x1d18830(cb, 0, 0x53, 2,
	// label, 1) takes its "param_4 == 2" branch, DcbWriteData(cb, 4, 0, label, &1, 2 dwords) (Ghidra, eboot 01.007.000).
	// parpadeo7 only looked at WriteAtEndOfPipe, so it never met one (logs 054809 / 062037: "del XPR 0"). Same check
	// here, before the 1 is written: a 64-bit 1 to one address range.
	if (Parpadeo7::On() && !m_state_only && !write_one_address && dw_num == 2 && src[0] == 1 && src[1] == 0 &&
	    Spec::Current() == nullptr)
		PrefetchXprBits(reinterpret_cast<uint64_t>(dst), GetGpuResources());

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
	if (auto* record = Spec::t_record) return record->NoteWritten(dst_address, num_bytes);
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
	if (auto* record = Spec::t_record) {
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
	submission.serial      = ++m_serials[submission.queue_id];
	// (ww2, KYTY_WRITE_WATCH_MODE=poll: the game wrote all it hands over with this submission before it queued it; a sweep
	// of the write watch that begins after this needs no other at the submission's first slice, GuestGpu::Process.)
	submission.sweep_stamp = PageManager::SweepStamp();
	// (nativo3, KYTY_DS_NATIVE_ALL: without the poll mode its stamp never moves; the native sweep's instead.)
	if (!PageManager::WatchPolled() && Libs::Graphics::BufferCache::NativeAllOn())
		submission.sweep_stamp = Libs::Graphics::GpuResourceManager::NativeSweepStamp();
	// (serie1) KYTY_SERIE1: when the game queued it (the culling's: its passes wait for its label from about now).
	if (Serie1::Mode() != 0) submission.enqueue_ns = Serie1::NowNs();
	// (KYTY_SPECULATE=4: the graphics queue's, translated ahead from now on.)
	if (auto* ahead = m_ahead.load(std::memory_order_acquire); ahead != nullptr && submission.queue_id == 0)
		ahead->Push({submission.serial, submission.commands, submission.reset_processor,
		             submission.type == SubmissionType::FlipPreparation || submission.type == SubmissionType::FrameBoundary
		                 ? Speculation::Job::Kind::Flip
		             : submission.type == SubmissionType::Graphics && submission.constant_commands.empty()
		                 ? Speculation::Job::Kind::Graphics
		                 : Speculation::Job::Kind::Other});
	LiveTrace::Event(LiveTrace::Submit, submission.queue_id | static_cast<uint32_t>(submission.type) << 8u,
	                 submission.commands.size());
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	// (par1: the GPU thread and the ACB thread wait on it)
	if (ParQueues::On()) m_work_available.SignalAll();
	else m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || m_acb_processing || !m_commands.empty() || m_submission_count != 0) {
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
		// (par1: nor while the ACB thread runs one of its frame.)
		if (m_acb_processing && m_acb_epoch == submission.frame_epoch) return false;
	}
	return true;
}

bool GuestGpu::ParRetry(const Submission& submission) const {
	return ParQueues::On() && submission.blocked && submission.block_epoch != ParQueues::WakeEpoch();
}

void GuestGpu::ParWake() {
	if (!ParQueues::NoteLabelWrite()) return;
	Common::LockGuard lock(m_queue_mutex);
	m_work_available.SignalAll();
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
	// (par1) KYTY_PAR_QUEUES=1: this thread translates the graphics queue, the ACB thread (started below) the compute
	// queues. Not with what translates elsewhere or reads one thread's own recording: speculation, the global barrier
	// dedupe (it counts this thread's recorded work only), the second Vulkan queue (one open command buffer is shared).
	if (ParQueues::On()) {
		ParQueues::t_side = 0;
		ParQueues::g_side[0].cpu_handle.store(LocalPlatform::CurrentThreadHandle(), std::memory_order_relaxed);
		kyty_local_speculate_mode.store(0, std::memory_order_relaxed);
		kyty_local_global_barrier_dedupe.store(0, std::memory_order_relaxed);
		std::printf("par1: KYTY_PAR_QUEUES=1: compute queues (ACB) on their own thread, graphics (DCB) on this one; "
		            "KYTY_SPECULATE and the barrier dedupe off, KYTY_ASYNC_COMPUTE's second queue unused%s\n",
		            Render5::Mode(Render5::RegLane) == 1 ? "" : " (KYTY_REG_LANE is not 1: the threads never overlap)");
		std::fflush(stdout);
	}
	// (p7, KYTY_P7: the live fixes on - C1, C2, C5, C6, C7 -, after every switch above; local/p7-compat.h.)
	P7::ApplyAtGpuStart();
	if (kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 && !PageManager::NativeWriteWatch()) {
		gpu->m_speculation = std::make_unique<Speculation>(gpu->m_renderer);
		gpu->m_ahead.store(gpu->m_speculation.get(), std::memory_order_release);
	}
	// (envio2) KYTY_ENVIO2=1/2: the compute queues translated on their own thread (local/envio2.h). Not with what else
	// translates elsewhere or in another order (KYTY_PAR_QUEUES, KYTY_SPECULATE: one effects log per thread) nor with the
	// second Vulkan queue (its commits go to the one queue, lane 0).
	if (Envio2::Mode() != 0) {
		const bool lanes = gpu->m_renderer.GetCommandScheduler().GetMasterSemaphore().LanesOn();
		const bool speculate = kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 && !PageManager::NativeWriteWatch();
		if (ParQueues::On() || speculate || lanes) {
			std::printf("envio2: KYTY_ENVIO2 off: needs KYTY_PAR_QUEUES and KYTY_SPECULATE off and one Vulkan queue "
			            "(par %d, speculate %u, lanes %d)\n",
			            ParQueues::On() ? 1 : 0, kyty_local_speculate_mode.load(std::memory_order_relaxed), lanes ? 1 : 0);
		} else {
			gpu->m_acb_spec = std::make_unique<Speculation>(gpu->m_renderer, static_cast<int>(ComputeQueueBase), [gpu] {
				Common::LockGuard lock(gpu->m_queue_mutex);
				gpu->m_work_available.SignalAll();
			});
			std::printf("envio2: KYTY_ENVIO2=%u: compute queues (ACB) translated on Kyty.Acb2, committed in order by "
			            "this thread%s\n",
			            Envio2::Mode(), Envio2::Mode() == 2 ? " (verification: committed at once, no overlap)" : "");
			if (Envio3::Mode() != 0)
				std::printf("envio3: KYTY_ENVIO3=%u: up to %u compute submissions in flight, continuations after waits on "
				            "Kyty.Acb2, sweeps logged for their commits\n",
				            Envio3::Mode(), Envio3::MaxInFlight);
		}
		std::fflush(stdout);
	}
	LiveControl::Start();
	Blackbox::Start(); // (the crash / freeze black box: logs/blackbox-<stamp>.txt, local/blackbox-run.h)
	XprCapture::Initialize();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::ProducerScope recording;
	void* const recording_stream = LocalVulkanRecording::CurrentProducer();
#else
	void* const recording_stream = nullptr;
#endif
	if (ParQueues::On()) gpu->m_acb_thread = std::thread(AcbThreadRun, gpu, recording_stream);

	for (;;) {
		PageManager::FlushDeferredProtection(); // (before a guest command, an idle wait, another queue's turn)
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			const auto        lock_start = std::chrono::steady_clock::now();
			Common::LockGuard lock(gpu->m_queue_mutex);
			LiveCounters::Add(LiveCounters::QueueLockUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - lock_start)
			                                            .count()));
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				// (A flush deferred while the GPU was busy goes before the thread waits: nothing else would submit it.)
				// (par1: the scheduler is read only with the queue lock, never taken with the queue mutex held.)
				if (ParQueues::On()) {
					gpu->m_queue_mutex.Unlock();
					bool flushed = false;
					{
						ParQueues::WorkScope work;
						if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
							gpu->m_renderer.GetCommandScheduler().FlushDeferred();
							flushed = true;
						}
					}
					gpu->m_queue_mutex.Lock();
					if (flushed) continue;
					if (!(gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping)) break;
				} else if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
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
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_acb_processing) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
				gpu->m_acb_stop = true; // (par1)
				gpu->m_work_available.SignalAll();
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				// (par1, KYTY_PAR_QUEUES=1: the graphics queue only here, the compute queues are the ACB thread's; a
				// blocked submission is tried again once the other thread wrote a label since its slice began.)
				const bool     par        = ParQueues::On();
				const uint32_t candidates = par ? 1u : QueueCount;
				const uint64_t seen_epoch = par ? ParQueues::WakeEpoch() : 0;
				for (uint32_t offset = 0; offset < candidates; offset++) {
					const auto id = par ? 0u : (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() &&
					    (!gpu->m_queues[id].front().blocked || gpu->ParRetry(gpu->m_queues[id].front())) &&
					    gpu->CanProcessSubmission(gpu->m_queues[id].front())) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					if (par) {
						gpu->m_queue_mutex.Unlock();
						bool flushed = false;
						{
							ParQueues::WorkScope work;
							if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
								gpu->m_renderer.GetCommandScheduler().FlushDeferred();
								flushed = true;
							}
						}
						gpu->m_queue_mutex.Lock();
						if (flushed) continue;
					} else if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
						gpu->m_queue_mutex.Unlock();
						gpu->m_renderer.GetCommandScheduler().FlushDeferred();
						gpu->m_queue_mutex.Lock();
						continue;
					}
					gpu->m_processing = false;
					LiveTrace::Event(LiveTrace::RenderIdle, 1, 1);
					const auto idle = std::chrono::steady_clock::now();
					LiveControl::NoteRenderIdle(idle.time_since_epoch().count());
					if (par) {
						// (par1) Asleep until a label write of the ACB thread (ParWake), a submission, or 100 us.
						const bool waits = !gpu->m_queues[0].empty();
						ParQueues::g_sleepers.fetch_add(1, std::memory_order_seq_cst);
						if (ParQueues::WakeEpoch() == seen_epoch) gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
						ParQueues::g_sleepers.fetch_sub(1, std::memory_order_seq_cst);
						if (waits) {
							ParQueues::g_side[0].label_waits.fetch_add(1, std::memory_order_relaxed);
							ParQueues::g_side[0].label_wait_ns.fetch_add(
							    static_cast<uint64_t>((std::chrono::steady_clock::now() - idle).count()),
							    std::memory_order_relaxed);
						}
					} else {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					}
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
			// (par1) The ACB thread first: it records into this thread's stream, gone when this thread returns.
			if (gpu->m_acb_thread.joinable()) gpu->m_acb_thread.join();
			gpu->m_acb_spec.reset(); // (envio2: its thread; nothing is handed, the queues are empty)
			{
				ParQueues::WorkScope work;
				gpu->m_gfx_cp->BufferWait();
			}
			gpu->m_renderer.GetMutex().ResignOwner();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			LiveCounters::Add(LiveCounters::GuestCommands);
			{
				ParQueues::WorkScope work; // (par1: nothing unless KYTY_PAR_QUEUES=1)
				LiveCensus::Scope census(LiveCensus::GuestCommand, 0);
				// (envio2: what a guest command writes while a compute submission is handed is logged for its commit.)
				Spec::t_effects = gpu->m_acb_spec && gpu->m_acb_spec->Pending() ? &gpu->m_acb_spec->Log() : nullptr;
				command();
				Spec::t_effects = nullptr;
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_acb_processing) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u | 1u << 17u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);
		// (thread1, KYTY_RENDER_STATS: the slice's time on its side's virtual clock, LiveCensus::Par2; measurement only.)
		LiveCensus::Par2::Begin(submission.queue_id != 0 ? 1u : 0u);
		const uint64_t slice_epoch = ParQueues::On() ? ParQueues::WakeEpoch() : 0; // (par1)
		F0Measure::SliceBegin(submission.queue_id != 0); // (f0: a GPU stamp where the side changes)
		NativeRender::SliceBegin(submission.queue_id); // (f5, KYTY_NATIVE_RENDER=map)
		const bool complete = [&] {
			ParQueues::WorkScope work; // (par1: the queue lock for the slice; nothing unless KYTY_PAR_QUEUES=1)
			LiveCensus::Scope census(LiveCensus::Submission, submission.queue_id);
			return gpu->Process(submission);
		}();
		NativeRender::SliceEnd(); // (f5)
		F0Measure::SliceEnd(); // (f0)
		if (ParQueues::On()) {
			ParQueues::g_side[0].slices.fetch_add(1, std::memory_order_relaxed);
			ParQueues::MaybeReport(static_cast<uint64_t>(gpu->m_done_num.load(std::memory_order_relaxed)),
			                       [](uint64_t handle) { return LocalPlatform::ThreadCpuSeconds(handle); });
		}
		LiveCensus::Par2::End(complete);
		// (trust1) ... and at the game's flip too (CommandProcessor::Flip during this slice): FlipPreparation alone never
		// came in a game session.
		if (const bool flipped = LiveCensus::Par2::TakeFlip(); flipped || submission.type == SubmissionType::FlipPreparation)
			LiveCensus::Par2::Frame();
		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u |
		                     static_cast<uint32_t>(complete) << 16u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);

		Common::LockGuard lock(gpu->m_queue_mutex);
		LiveCounters::Add(LiveCounters::Submissions);
		if (!complete) {
			LiveCounters::Add(LiveCounters::SubmissionRequeues);
			submission.blocked     = true;
			submission.block_epoch = slice_epoch; // (par1)
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
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_acb_processing) {
			gpu->m_idle.SignalAll();
		}
	}
}

// (par1, KYTY_PAR_QUEUES=1) The ACB thread: the compute queues' submissions, round robin, as the GPU thread took them
// before (ThreadRun), each slice with the queue lock (par-queues.h); a submission stopped at a WAIT_REG_MEM is tried
// again when the GPU thread writes a label (ParWake) or after 100 us (memory the game's CPU writes). It runs no guest
// commands of its own accord (ProcessPm4 runs those queued, between packets, as on the GPU thread) and stops when the
// GPU thread stops (m_acb_stop) with its queues empty.
void GuestGpu::AcbThreadRun(GuestGpu* gpu, void* stream) {
	LocalPlatform::SetThreadName("Kyty.Acb");
	KYTY_PROFILER_THREAD("Thread_Acb");
	g_gpu_thread      = true;
	g_gpu_state       = gpu;
	ParQueues::t_side = 1;
	ParQueues::g_side[1].cpu_handle.store(LocalPlatform::CurrentThreadHandle(), std::memory_order_relaxed);
	PageManager::DeferReadProtection(true);
	gpu->m_renderer.GetMutex().BecomeOwner();
	BdaDirtyRegions::g_translating = true;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::AdoptProducer(stream);
#else
	(void)stream;
#endif
	constexpr uint32_t ComputeQueues = QueueCount - 1;
	for (;;) {
		PageManager::FlushDeferredProtection();
		Submission submission;
		bool       stop = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			for (;;) {
				const uint64_t seen     = ParQueues::WakeEpoch();
				int            selected = -1;
				bool           queued   = false;
				for (uint32_t offset = 0; offset < ComputeQueues; ++offset) {
					const auto id    = 1u + (gpu->m_acb_next_queue - 1u + offset) % ComputeQueues;
					auto&      queue = gpu->m_queues[id];
					if (queue.empty()) continue;
					queued = true;
					if ((!queue.front().blocked || gpu->ParRetry(queue.front())) && gpu->CanProcessSubmission(queue.front())) {
						selected = static_cast<int>(id);
						break;
					}
				}
				if (selected >= 0) {
					auto& queue = gpu->m_queues[static_cast<uint32_t>(selected)];
					submission  = std::move(queue.front());
					queue.pop_front();
					gpu->m_submission_count--;
					gpu->m_acb_next_queue = 1u + static_cast<uint32_t>(selected) % ComputeQueues;
					gpu->m_acb_processing = true;
					gpu->m_acb_epoch      = submission.frame_epoch;
					break;
				}
				if (gpu->m_acb_stop && !queued) {
					stop = true;
					break;
				}
				// (A flush either thread deferred goes before this one sleeps: the open command buffer is shared.)
				gpu->m_queue_mutex.Unlock();
				bool flushed = false;
				{
					ParQueues::WorkScope work;
					if (gpu->m_renderer.GetCommandScheduler().FlushPending()) {
						gpu->m_renderer.GetCommandScheduler().FlushDeferred();
						flushed = true;
					}
				}
				gpu->m_queue_mutex.Lock();
				if (flushed) continue;
				const uint64_t start = ParQueues::NowNs();
				ParQueues::g_sleepers.fetch_add(1, std::memory_order_seq_cst);
				if (ParQueues::WakeEpoch() == seen && !gpu->m_acb_stop) {
					if (queued) gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					else gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				}
				ParQueues::g_sleepers.fetch_sub(1, std::memory_order_seq_cst);
				if (queued) {
					ParQueues::g_side[1].label_waits.fetch_add(1, std::memory_order_relaxed);
					ParQueues::g_side[1].label_wait_ns.fetch_add(ParQueues::NowNs() - start, std::memory_order_relaxed);
					// (As the GPU thread's 100 us wait: a label the game's CPU writes moves no epoch.)
					for (uint32_t id = 1; id < QueueCount; ++id)
						if (!gpu->m_queues[id].empty()) gpu->m_queues[id].front().blocked = false;
				}
			}
		}
		if (stop) break;

		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u | 1u << 17u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);
		const uint64_t slice_epoch = ParQueues::WakeEpoch();
		const bool     complete    = [&] {
            ParQueues::WorkScope work;
            return gpu->Process(submission);
		}();
		ParQueues::g_side[1].slices.fetch_add(1, std::memory_order_relaxed);
		LiveTrace::Event(LiveTrace::RenderSlice,
		                 submission.queue_id | static_cast<uint32_t>(submission.type) << 8u |
		                     static_cast<uint32_t>(complete) << 16u,
		                 submission.frame_epoch | uint64_t {submission.commands.size()} << 32u);

		Common::LockGuard lock(gpu->m_queue_mutex);
		LiveCounters::Add(LiveCounters::Submissions);
		if (!complete) {
			LiveCounters::Add(LiveCounters::SubmissionRequeues);
			submission.blocked     = true;
			submission.block_epoch = slice_epoch;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (uint32_t id = 1; id < QueueCount; ++id)
				if (!gpu->m_queues[id].empty()) gpu->m_queues[id].front().blocked = false;
		}
		gpu->m_acb_processing = false;
		gpu->m_work_available.SignalAll(); // (the GPU thread's frame boundary, or its stop, may wait for this)
		if (!gpu->m_processing && gpu->m_commands.empty() && gpu->m_submission_count == 0) gpu->m_idle.SignalAll();
	}
	{
		ParQueues::WorkScope work; // (the other owner is outside the render mutex)
		gpu->m_renderer.GetMutex().ResignOwner();
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::AdoptProducer(nullptr);
#endif
	PageManager::FlushDeferredProtection();
	PageManager::DeferReadProtection(false);
	BdaDirtyRegions::g_translating = false;
	ParQueues::t_side              = -1;
	g_gpu_state                    = nullptr;
	g_gpu_thread                   = false;
}

bool GuestGpu::Process(Submission& submission) {
	BdaDirtyRegions::Publish();
	if (submission.type == SubmissionType::FrameBoundary) {
		EXIT_IF(submission.started || submission.queue_id != 0);
		Common::LockGuard lock(m_queue_mutex);
		EXIT_IF(submission.frame_epoch != m_consumed_frame);
		++m_done_num;
		++m_consumed_frame;
		FrameNativo2::g_guest_consumed.fetch_add(1, std::memory_order_acq_rel); // (FRAME-NATIVO2 v13: fotograma traducido)
		m_idle.SignalAll();
		if (Serie1::Mode() != 0) ++Serie1::g_c.frames; // (serie1; the GPU thread's, as every Serie1 counter)
		if (ParQueues::On()) m_work_available.SignalAll(); // (par1: the ACB thread waits for the next frame's)
		if (DsCensoDcb::On()) DsCensoDcb::OnFrameBoundary(); // (w7: [censo-dcb] cada 5 s)
		Envio1::NoteFrame(); // (envio1, KYTY_ENVIO1=1)
		CensoPm4::OnFrame(); // (w18)
		if (const double frames = Envio2::NoteFrame(); frames > 0) Envio3::Print(frames); // (envio2/envio3)
		return true;
	}
	const bool first_slice = !submission.started;
	// (w7) KYTY_CENSO_DCB=1: el tiempo de este trozo, al sitio del juego que envio la DCB (local/ds-censo-dcb.h).
	std::optional<DsCensoDcb::Slice> censo_dcb;
	if (DsCensoDcb::On())
		censo_dcb.emplace(submission.queue_id == 0 && submission.type == SubmissionType::Graphics, submission.commands.data());
	// (w11b) [nativo-vgeo]: dispatches por CS dentro de la ACB #4 de a0ed3e.
	const VgeoNativo::TargetScope vgeo_target(censo_dcb ? DsCensoDcb::g_sites[censo_dcb->site].key.load(std::memory_order_relaxed) : 0);
	// (w18) [censo-pm4]: tiempo por paquete PM4 y por fase en los bloques caros del censo.
	const CensoPm4::SliceScope censo_pm4(censo_dcb ? DsCensoDcb::g_sites[censo_dcb->site].key.load(std::memory_order_relaxed) : 0);
	// (envio1, KYTY_ENVIO1=1: a compute queue's submission, measured.)
	const Envio1::AcbScope envio1_acb(submission.queue_id != 0);
	const Envio2::GpuScope envio2_gpu(submission.queue_id != 0 && m_acb_spec != nullptr); // (envio2)
	auto& cp = GetProcessor(submission.queue_id);
	// (serie1) KYTY_SERIE1=1/2: a queue-0 command buffer of the XPR culling's shape (dispatches and a label, no draw, no
	// indirect buffer, no flip: local/serie1.h). =1: recorded into a command buffer of the second Vulkan queue (lane 1),
	// translated here as always (same processor, same order); its semaphore waits are the graphics ticks that last wrote
	// (or used, for its writes) what it touches, and every later graphics submission waits for it (lanes' safe mode).
	bool serie1_cull = false;
	if (Serie1::Mode() != 0 && submission.queue_id == 0 && submission.type == SubmissionType::Graphics &&
	    submission.constant_commands.empty()) {
		if (first_slice) {
			submission.serie1_kind = Serie1::Classify(submission.commands);
			++Serie1::g_c.classified[submission.serie1_kind & 7u];
		}
		serie1_cull = submission.serie1_kind == Serie1::Cull;
		if (first_slice && !serie1_cull) Serie1::NoteGraphicsStart();
		Serie1::MaybeReport(m_renderer.GetCommandScheduler().GetMasterSemaphore().LanesOn());
	}
	const bool serie1_lane = serie1_cull && Serie1::Mode() == 1 && !ParQueues::On() &&
	                         m_renderer.GetCommandScheduler().GetMasterSemaphore().LanesOn();
	if (serie1_cull && first_slice) ++(serie1_lane ? Serie1::g_c.culls_lane1 : Serie1::g_c.no_lane);
	// KYTY_ASYNC_COMPUTE: the game's compute queues on the second Vulkan queue (lane-sync.h).
	// (par1, KYTY_PAR_QUEUES=1: lane 0 always: both threads record into the one open command buffer.)
	// (serie1: KYTY_SERIE1=1 alone makes the second queue for the culling only: the compute queues stay on lane 0.)
	static const bool acb_lane = !Serie1::WantsLanes() || Serie1::AsyncComputeEnv();
	m_renderer.GetCommandScheduler().SetLane(serie1_lane || (submission.queue_id != 0 && !ParQueues::On() && acb_lane) ? 1 : 0);
	Serie1::g_open_cull = serie1_lane;
	// KYTY_NATIVE_MEMORY: what the guest wrote before this submission invalidated before the command processor reads any
	// of its memory (command buffers, descriptors, constants: the guest wrote them before submitting, and the queue's
	// lock orders those writes before this thread took the submission) and before any of its resources is used
	// (GpuResourceManager::SweepCpuWrites). At its first slice only: a slice resumed at a WAIT_REG_MEM sweeps when the
	// wait is satisfied (CommandProcessor::WaitRegMem), the only point where the guest hands memory over meanwhile; one
	// before each retry of an unsatisfied wait was most of the ~1100 sweeps a second. Nothing otherwise.
	// (ww2, KYTY_WRITE_WATCH_MODE=poll: no sweep when one began after the game queued this submission - it read every
	// write the game made before queueing it, PageManager::SweptSince; the queue was a few submissions deep.)
	if (first_slice) {
		// (envio3: what it finds CPU-written is logged for the compute submissions in flight, Speculation::Log.)
		Spec::t_effects = Envio3::Mode() != 0 && m_acb_spec && m_acb_spec->Pending() ? &m_acb_spec->Log() : nullptr;
		const uint64_t censo_sweep_t0 = DsCensoDcb::On() ? DsCensoDcb::NowNs() : 0; // (w7)
		CensoPm4::Mark(CensoPm4::KSweep); // (w18)
		m_renderer.GetGpuResources().SweepCpuWrites((PageManager::SweptSince(submission.sweep_stamp) ||
		                                             (!PageManager::WatchPolled() && GpuResourceManager::NativeSweptSince(submission.sweep_stamp)))
		                                                ? GpuResourceManager::SweepReason::CoveredSubmission
		                                                : GpuResourceManager::SweepReason::Handover);
		Spec::t_effects = nullptr;
		if (censo_sweep_t0 != 0) DsCensoDcb::g_sweep_ns.fetch_add(DsCensoDcb::NowNs() - censo_sweep_t0, std::memory_order_relaxed);
		CensoPm4::Mark(CensoPm4::KSlice); // (w18)
	}

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	// The graphics queue's command buffers, speculation on: speculatively first (speculation.h).
	// (Not with KYTY_NATIVE_MEMORY: a speculation translates on other threads, ahead of the sweeps of the write watch
	// that tell the caches what the guest wrote, GpuResourceManager::SweepCpuWrites.)
	const bool speculate = submission.type == SubmissionType::Graphics && submission.queue_id == 0 &&
	                       kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 && !PageManager::NativeWriteWatch();
	if (speculate && !m_speculation) m_speculation = std::make_unique<Speculation>(m_renderer);
	// (What the translation does a speculation not committed yet must see.)
	struct EffectsScope {
		explicit EffectsScope(Speculation* speculation) {
			Spec::t_effects = speculation != nullptr ? &speculation->Log() : nullptr;
		}
		~EffectsScope() { Spec::t_effects = nullptr; }
		KYTY_CLASS_NO_COPY(EffectsScope);
	} effects_scope {m_speculation ? m_speculation.get()
	                 : m_acb_spec && m_acb_spec->Pending() ? m_acb_spec.get() // (envio2: one handed or blocked)
	                                                        : nullptr};

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
			CensoPm4::Mark(CensoPm4::KFlush); // (w18)
			if (progressed) {
				if (complete) {
					m_renderer.GetGpuResources().EndSubmission();
				}
				// (A slice stopped at a wait goes to the GPU at once: what it waits for may need it.)
				// (serie1: the culling's lane-1 buffer at once too: the game reads its bits next.)
				if (complete && !serie1_lane) cp.BufferFlushDeferrable();
				else cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().EndSubmission();
			}
			break;
		}
		case SubmissionType::Compute: {
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
			// (envio2, KYTY_ENVIO2) Handed to the ACB thread at its first slice (when it has none): the GPU thread goes
			// on with the other queues and commits it, in this queue's order, when it takes it again (blocked meanwhile;
			// the ACB thread's end wakes it). Mode 2: committed at once. What it did not commit goes on the normal way
			// from where the commit left it (the safe fallback).
			// (envio3, KYTY_ENVIO3) Several in flight (one per compute queue head, Speculation::Full), and continuations:
			// a submission at a WAIT_REG_MEM (where a commit stopped, "wait on memory the CPU writes", or a slice resumed
			// there) has the wait run here - its sweep of CPU writes included, CommandProcessor::WaitRegMem - and the
			// rest handed to the ACB thread again. Not satisfied: the slice waits, as the normal way would.
			bool envio2_progress = false;
			bool envio2_stop     = false; // (the slice ends here: handed, blocked or whole)
			if (m_acb_spec) {
				const uint64_t spec_serial = uint64_t {submission.queue_id} << 48u | submission.serial;
				const bool     envio3      = Envio3::Mode() != 0;
				const auto     hand        = [&](bool continuation) {
					if (!m_acb_spec->Hand(spec_serial, cp, submission.commands, submission.command_execution)) return false;
					submission.envio2_handed = true;
					submission.envio3_cont   = continuation;
					if (continuation) Envio2::Add(Envio3::g_totals.continuations);
					return true;
				};
				if (first_slice) {
					Envio2::Add(Envio2::g_totals.submissions);
					if (hand(false)) {
						Envio2::Add(Envio2::g_totals.handed);
					} else {
						Envio2::Add(Envio2::g_totals.fallback[Envio2::Busy]);
					}
				} else if (!submission.envio2_handed && !m_acb_spec->BlockedHanded(spec_serial)) {
					// (A slice resumed the normal way: at a wait, envio3 tries a continuation below.)
					if (!envio3 || !CommandProcessor::AtWait(submission.command_execution))
						Envio2::Add(Envio2::g_totals.fallback[Envio2::NotFirst]);
				}
				for (bool first_round = true;; first_round = false) {
					auto outcome   = Speculation::Outcome::None;
					bool committed = false;
					if (submission.envio2_handed) {
						if (Envio2::Mode() == 1 && first_slice && first_round) {
							// (Mode 1: handed in this slice; committed when the GPU thread takes it again.)
							envio2_stop = true;
							complete                     = false;
							break;
						}
						submission.envio2_handed = false;
						uint64_t waited          = 0;
						outcome   = m_acb_spec->CommitHanded(spec_serial, cp, submission.command_execution, waited);
						committed = true;
						if (waited != 0) {
							Envio2::Add(Envio2::g_totals.forced);
							Envio2::Add(Envio2::g_totals.forced_ns, waited);
						}
						// (The order check: a queue's submissions committed in their order; its continuations after it.)
						if (submission.serial < m_envio2_last[submission.queue_id] ||
						    (submission.serial == m_envio2_last[submission.queue_id] && !submission.envio3_cont))
							Envio2::Add(Envio2::g_totals.order_errors);
						m_envio2_last[submission.queue_id] = submission.serial;
					} else if (m_acb_spec->BlockedHanded(spec_serial)) {
						outcome   = m_acb_spec->ResumeHanded(spec_serial, cp, submission.command_execution);
						committed = true;
					}
					if (committed) {
						switch (outcome) {
							case Speculation::Outcome::Whole: Envio2::Add(Envio2::g_totals.whole); break;
							case Speculation::Outcome::Blocked: Envio2::Add(Envio2::g_totals.fallback[Envio2::Blocked]); break;
							case Speculation::Outcome::None: Envio2::Add(Envio2::g_totals.fallback[Envio2::None]); break;
							default: Envio2::Add(Envio2::g_totals.fallback[Envio2::Partly]); break;
						}
					}
					if (outcome == Speculation::Outcome::Blocked) {
						// (What it committed before the wait goes to the GPU; the submission waits as at a wait of its own.)
						cp.BufferFlush();
						envio2_stop = true;
						complete    = false;
						break;
					}
					if (outcome == Speculation::Outcome::Whole) {
						envio2_stop = true;
						complete    = true;
						m_renderer.GetGpuResources().EndSubmission();
						cp.BufferFlushDeferrable();
						break;
					}
					envio2_progress |= outcome == Speculation::Outcome::Partly;
					// (envio3) A continuation: at a wait, after a commit that stopped there or a slice resumed there.
					if (!envio3 || !CommandProcessor::AtWait(submission.command_execution) ||
					    (!committed && first_slice))
						break;
					if (m_acb_spec->Full()) {
						Envio2::Add(Envio3::g_totals.cont_full);
						if (!committed) Envio2::Add(Envio2::g_totals.fallback[Envio2::NotFirst]);
						break;
					}
					uint32_t     dwords = 0;
					const auto*  packet = CommandProcessor::SuspendedPacket(submission.command_execution, dwords);
					Pm4Execution wait;
					if (cp.Process(wait, {packet, dwords}) != Pm4ProcessResult::Complete) {
						// (Not satisfied yet: the slice waits there, as the normal way's would; what was committed goes.)
						Envio2::Add(Envio3::g_totals.cont_unsatisfied);
						if (envio2_progress) cp.BufferFlush();
						envio2_stop = true;
						complete    = false;
						break;
					}
					CommandProcessor::SkipSuspendedPacket(submission.command_execution);
					envio2_progress = true;
					if (!hand(true)) break; // (the rest the normal way, after the wait)
					if (Envio2::Mode() == 1) {
						// (What was committed goes to the GPU now; the rest when the GPU thread takes it again.)
						cp.BufferFlush();
						envio2_stop = true;
						complete    = false;
						break;
					}
				}
			}
			if (envio2_stop) break;
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			CensoPm4::Mark(CensoPm4::KFlush); // (w18)
			if (submission.command_execution.MadeProgress() || envio2_progress) {
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
			if (AntiLag::OwnLowLatencyOn()) { // KYTY_LOWLAT: this frame's last host GPU tick (local/anti-lag.h)
				auto& master = m_renderer.GetCommandScheduler().GetMasterSemaphore();
				AntiLag::OnFrameSubmitted(static_cast<VkDevice>(m_renderer.GetGraphics().device),
				                          static_cast<VkSemaphore>(master.Handle(0)),
				                          master.LanesOn() ? master.LastTick(0) : master.CurrentTick() - 1);
			}
			m_renderer.GetGpuResources().AdvanceFrame();
			m_renderer.GetPipelineCache().PromoteFinished();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::FrameBoundary: EXIT("frame boundary already handled\n");
	}
	// (serie1) A culling's label is written (translated): the game's wait for it ends; the GPU thread's gap begins.
	if (serie1_cull && complete) Serie1::NoteLabel(submission.enqueue_ns);
	// (parpadeo8, w21) El tick que contendra este cull (el buffer abierto se cierra con CurrentTick() o uno mayor): el hilo
	// del juego sabe en 0xa1f290 si la GPU ya lo corrio (local/parpadeo8.h). Una lectura atomica y dos escrituras.
	if (serie1_cull && complete) {
		auto& master = m_renderer.GetCommandScheduler().GetMasterSemaphore();
		Parpadeo8::NoteCullLabel(
		    &master, [](const void* m, uint64_t tick) { return static_cast<const MasterSemaphore*>(m)->IsFree(tick); },
		    master.CurrentTick());
	}
	Serie1::g_open_cull = false;
	// (KYTY_SPECULATE=4: the state after the graphics queue's submission, for a chain that could not follow it.)
	if (complete && submission.queue_id == 0 && m_ahead.load(std::memory_order_relaxed) != nullptr)
		m_speculation->Done(submission.serial, cp);

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

	struct QueueScope {
		explicit QueueScope(uint64_t queue): previous(LiveCensus::g_queue) { LiveCensus::g_queue = queue; }
		~QueueScope() { LiveCensus::g_queue = previous; }
		uint64_t previous;
	} queue_scope(IsAsyncComputeQueue() ? LiveCensus::QueueAsync : 0);
	LiveCensus::Scope census(LiveCensus::QueueRun, static_cast<uint64_t>(m_interrupt_event_id));
	ProcessPm4(execution, 0);
	CensoPm4::Mark(CensoPm4::KSlice); // (w18)
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
	ProcessPm4(execution, stop_depth);
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	LiveCounters::Add(LiveCounters::Pm4Suspends);
	g_current_execution->m_suspended = true;
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
		// Draws only as table draws (TryNativeXprDraws, TryNativeDirectDraw, TryNativeAutoDraw).
		case Pm4::IT_DRAW_INDEX_AUTO: return header == 0xc0012d00u;
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

// (render1 F0+F1, KYTY_RENDER1=1; local/render1.h) The DRAW_INDEX_INDIRECT packets about to run (5 dwords each,
// `packets` of them from `packet`), paired with the hook's markers; then, every 5 s, the [render1] line. Observes only.
// (F2) Returns how many were paired with a marker of a native pass (p0/p1) and draw depth only; `all`: all of them.
static uint32_t Render1Observe(const HW::Context& ctx, const uint32_t* packet, uint32_t packets, bool& all) {
	const auto& depth = ctx.GetDepthControl();
	bool        color = false;
	const uint32_t mask = ctx.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8 && !color; ++slot)
		color = ((mask >> (slot * 4u)) & 0xfu) != 0 && ctx.GetRenderTarget(slot).base.addr != 0;
	const bool depth_only = depth.z_enable && depth.z_write_enable && !color;
	uint32_t   native     = 0;
	all                   = true;
	for (uint32_t i = 0; i < packets; ++i) {
		if (packet[i * 5u] != 0xc0032500u) break;
		const int pass = Render1::ObserveDraw(packet + i * 5u, depth_only);
		if (depth_only && pass >= 0 && Render1::NativePass(static_cast<uint32_t>(pass))) ++native;
		else all = false;
	}
	Render1::MaybeReport(LiveControl::g_flips.load(std::memory_order_relaxed));
	return native;
}

// The rest of a run drawn with the first packet: observed after it (F2: a draw of it not of p0/p1 spoils the batch's
// comparison; the batch has the whole run).
static void Render1ObserveRest(const HW::Context& ctx, RenderExecutor& executor, const uint32_t* packet, uint32_t packets,
                               bool in_spec) {
	// (F3) Paired already, before the run was drawn (TryNativeXprDraws).
	if (Render1::Mode() == 3 && !in_spec && Render1::t_current.rest_observed) return;
	bool       all    = false;
	const auto native = Render1Observe(ctx, packet, packets, all);
	if (Render1::Mode() < 2 || in_spec || !Render1::t_current.native) return;
	if (Render1::Mode() == 3) {
		Render1::t_step_draws += native;
		return;
	}
	Render1::t_step_draws += native;
	if (!all) executor.Render1Taint(Render1::RRunMixed);
}

// (render1 F3) The other draws of a run, by their packets (a run over gancho2's gaps is not 5 dwords a draw), paired
// before the run is drawn: how many are p0/p1 depth draws; `all`: every one.
static uint32_t Render1ObservePackets(const HW::Context& ctx, const uint32_t* const* packets, uint32_t count, bool& all) {
	const auto& depth = ctx.GetDepthControl();
	bool        color = false;
	const uint32_t mask = ctx.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8 && !color; ++slot)
		color = ((mask >> (slot * 4u)) & 0xfu) != 0 && ctx.GetRenderTarget(slot).base.addr != 0;
	const bool depth_only = depth.z_enable && depth.z_write_enable && !color;
	uint32_t   native     = 0;
	all                   = true;
	for (uint32_t i = 0; i < count; ++i) {
		const int pass = Render1::ObserveDraw(packets[i], depth_only);
		if (depth_only && pass >= 0 && Render1::NativePass(static_cast<uint32_t>(pass))) ++native;
		else all = false;
	}
	return native;
}

// (render1 F2, KYTY_RENDER1=2) One step of the walk that processes a p0/p1 draw packet (and the run it may start): its
// time, less the native batch's own work, is the normal path's for its draws; a draw the native batch did not take
// spoils the batch's comparison. Ends the step's state (Render1::t_current).
struct Render1Step {
	RenderExecutor* executor = nullptr;
	uint64_t        start    = 0;
	void Start(RenderExecutor& ex) {
		executor                  = &ex;
		Render1::t_native_cycles = 0;
		start                     = __rdtsc();
	}
	~Render1Step() {
		if (executor == nullptr) return;
		const uint64_t cycles = __rdtsc() - start;
		const uint64_t native = Render1::t_native_cycles;
		auto&          c      = Render1::t_current;
		if (Render1::Mode() == 3) {
			// (F3) The step's whole time, drawn natively or the normal way (by window).
			if (c.native) Render1::F3EndStep(cycles, Render1::t_step_draws);
		} else if (c.native) {
			Render1::g_f2.normal_cycles.fetch_add(cycles > native ? cycles - native : 0, std::memory_order_relaxed);
			Render1::g_f2.normal_draws.fetch_add(Render1::t_step_draws, std::memory_order_relaxed);
			if (!c.captured) {
				if (!c.noted) Render1::Note(Render1::RViaNormal, std::max<uint32_t>(1, Render1::t_step_draws));
				executor->Render1Taint(Render1::RViaNormal);
			}
		}
		Render1::t_current       = {};
		Render1::t_native_cycles = 0;
		Render1::t_step_draws    = 0;
	}
};

void CommandProcessor::ProcessPm4(Pm4Execution& execution, size_t stop_depth) {
	auto* const spec = Spec::Current();
	// A speculation's refused packet suspends the execution there (speculation.cpp goes on past it, or stops).
	const auto refuse = [&](const char* why) {
		Spec::Refuse(why);
		execution.m_suspended = true;
	};
	LiveCensus::Pm4Clock pm4_clock; // (render3, KYTY_RENDER_STATS: the walk's time by packet class; nothing otherwise)
	const uint32_t       lane_mode = Render5::Mode(Render5::RegLane); // (render5, KYTY_REG_LANE)
	// (otros1, KYTY_OTROS1: with KYTY_RENDER_STATS, the walk's time by packet kind, otros1.h)
	const bool otros1_on = Otros1::Mode() != 0 && LiveCensus::RenderStatsOn();
	// (otros2, KYTY_OTROS2: A, otros2.h) The preamble waits past steps that cannot need it; `pre_skips` steps since it
	// last ran. On any way out of the walk with steps pending, it runs once (the guest commands and write-backs then).
	const uint32_t otros2_mode = Otros2::Active() && spec == nullptr && g_gpu_state != nullptr ? Otros2::Mode() : 0u;
	uint32_t       pre_skips   = 0;
	auto* const    preamble_cache = g_gpu_state != nullptr ? &g_gpu_state->m_renderer.GetBufferCache() : nullptr;
	const auto     preamble    = [&] {
        // (w20) Between packets nothing is usually queued: two inline checks instead of two calls (each with its own
        // checks: GPU thread, par queues, juego3, native memory, the semaphore's tick) on every packet. ACB #4 a0ed3e
        // ~10700 packets/f, "preambulo+vuelta" 1026 us/f (measured with [censo-pm4], its marks included).
        if (Juego3::Mode() == 0 && !g_gpu_state->HasPendingCommands() && (preamble_cache == nullptr || !preamble_cache->FeedbackPending())) return;
        const uint64_t pre0 = otros1_on ? __rdtsc() : 0; // (otros1: the preamble's own time)
        g_gpu_state->ProcessCommands();
        // (KYTY_EAGER_FEEDBACK: completed copy-feedback snapshots written back between packets.)
        g_gpu_state->m_renderer.GetBufferCache().WriteBackCompletedFeedback(true);
        if (otros1_on) {
            Otros1::g_pre_cycles += __rdtsc() - pre0;
            ++Otros1::g_pre_calls;
        }
	};
	struct PreambleAtExit {
		const uint32_t&             skips;
		const decltype(preamble)&   run;
		~PreambleAtExit() {
			if (skips != 0) {
				++Otros2::g_pre_flush;
				run();
			}
		}
	} preamble_at_exit {pre_skips, preamble};
	while (execution.m_buffer_stack.size() > stop_depth) {
		CensoPm4::Mark(CensoPm4::KPre); // (w18)
		pm4_clock.Next();
		G1::WalkStep(); // (g1 NP1, KYTY_G1_PROBE: a draw packet's walk by stretch; one relaxed load when off)
		if (spec != nullptr && spec->refused != nullptr) return refuse(spec->refused);
		if (g_gpu_state != nullptr && spec == nullptr) { // (a speculation's thread runs no guest commands)
			bool can_wait = false;
			if (otros2_mode != 0 && pre_skips < Otros2::MaxSkips) {
				const auto& next = execution.m_buffer_stack.back();
				if (next.deferred_advance_dw != 0 || next.offset_dw >= next.commands.size()) {
					can_wait = true; // (no packet runs in this step)
				} else {
					const uint32_t header = next.commands[next.offset_dw];
					can_wait              = Otros2::PreambleCanWait(header, RegLaneOpcode((header >> 8u) & 0xffu)) &&
					           (header == 0x80000000u || KYTY_PM4_LEN(header) <= next.commands.size() - next.offset_dw);
				}
			}
			if (can_wait && otros2_mode == 2) {
				// (2) Run anyway; count the steps where it found work that 1 would have postponed.
				using C             = LiveCounters::Id;
				const auto before   = LiveCounters::Value(C::GuestCommands) + LiveCounters::Value(C::EagerWritten) +
				                    LiveCounters::Value(C::EagerStale) + LiveCounters::Value(C::EagerClean) +
				                    LiveCounters::Value(C::EagerRefused);
				preamble();
				const auto after = LiveCounters::Value(C::GuestCommands) + LiveCounters::Value(C::EagerWritten) +
				                   LiveCounters::Value(C::EagerStale) + LiveCounters::Value(C::EagerClean) +
				                   LiveCounters::Value(C::EagerRefused);
				if (after != before) ++Otros2::g_pre_would_postpone;
				++Otros2::g_pre_skipped;
			} else if (can_wait) {
				++pre_skips;
				++Otros2::g_pre_skipped;
			} else {
				pre_skips = 0;
				preamble();
			}
		}
		G1::WalkMark(G1::WPreamble); // (g1 NP1)
		CensoPm4::Mark(CensoPm4::KWalk); // (w18)
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
		pm4_clock.Class(Pm4ClassOf(opcode));
		pm4_clock.Key(packet_header); // (otros1)
		CensoPm4::Packet(CensoPm4::KeyOf(packet_header)); // (w18)
		G1::WalkDecoded(Pm4ClassOf(opcode) == LiveCensus::Pm4Draw); // (g1 NP1, KYTY_G1_PROBE: one relaxed load when off)
		if (DsNativoSombras::g_gpu_pending.load(std::memory_order_relaxed) != 0xFFFFFFFFu && DsNativoSombras::Mode() == 2)
			DsNativoSombras::GpuNotePacket(opcode, packet, remaining_dw, m_sh_ctx.GetCs().cs_regs.data_addr); // (F5 pase 2, =2)
		// (otros1, KYTY_OTROS1) The register pairs an indirect block carries (its handler and the draw state observer
		// each read the block).
		if (otros1_on && (opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT || opcode == Pm4::IT_SET_SH_REG_INDIRECT ||
		                  opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT) &&
		    remaining_dw >= 5) {
			const uint32_t space = opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? 0u : opcode == Pm4::IT_SET_SH_REG_INDIRECT ? 1u : 2u;
			Otros1::g_pairs[space] += packet[4] & 0x3fffu;
			++Otros1::g_indirect[space];
		}

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		// (render1 F2, KYTY_RENDER1=2) A packet that is not state (nor a draw, decided when paired) ends the native batch
		// first: it is drawn and compared as the pass left the depth.
		if (Render1::Mode() == 2 && spec == nullptr && !m_state_only) {
			auto& executor = m_renderer.GetRenderExecutor();
			if (executor.Render1BatchOpen() &&
			    !Render1::KeepsBatch(opcode, opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_ZERO &&
			                                     packet_header != DsLinearCopyHook::Header && packet_header != DsNativoPase::Header))
				executor.Render1EndBatch(CurrentBuffer(), Render1::EOther);
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			if (kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0 &&
			    (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 16u) == 0)
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
		// (render5) KYTY_REG_LANE (render5.h): this register / index state packet and the lane packets right after it in
		// this buffer (RegLanePacket, up to RegLaneMaxPackets) run here: the walk's checks below do nothing for them
		// outside a speculation, a state pass, a chunked pass or a stop (none here; the hook's header is a NOP, they keep
		// the deferred protections), so only the guest commands and the copy feedback the walk runs between packets wait
		// for the run's end. Each one is observed (draw state observer), counted and handled as the walk does. 2: the
		// same lane with the walk's per-packet work done between its packets, as the walk does it (what is drawn as with
		// 0); the guest commands and feedback write-backs that ran there (what 1 postpones) and any packet a check of the
		// walk would have acted on (a difference) are counted ([render5]).
		if (lane_mode != 0 && spec == nullptr && !m_state_only && m_chunk_every == 0 && m_stop_at == nullptr &&
		    g_gpu_state != nullptr && !GraphicsRunDebugDumpEnabled() &&
		    RegLanePacket(cursor.commands.data(), cursor.offset_dw, total_dw)) {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			const bool observe = kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0 &&
			                     (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 16u) == 0;
#else
			const bool observe = false;
#endif
			const bool verify = lane_mode == 2;
			// (2) The walk's work between two packets, and what it did.
			const auto between = [&] {
				using C                     = LiveCounters::Id;
				const auto commands         = LiveCounters::Value(C::GuestCommands);
				const auto feedback         = LiveCounters::Value(C::EagerWritten) + LiveCounters::Value(C::EagerStale) +
				                      LiveCounters::Value(C::EagerClean) + LiveCounters::Value(C::EagerRefused);
				g_gpu_state->ProcessCommands();
				g_gpu_state->m_renderer.GetBufferCache().WriteBackCompletedFeedback(true);
				Render5::g_lane[Render5::LanePostponed] += LiveCounters::Value(C::GuestCommands) - commands;
				Render5::g_lane[Render5::LaneFeedback] += LiveCounters::Value(C::EagerWritten) +
				                                         LiveCounters::Value(C::EagerStale) +
				                                         LiveCounters::Value(C::EagerClean) +
				                                         LiveCounters::Value(C::EagerRefused) - feedback;
			};
			uint32_t taken = 0;
			{
			// (par1, KYTY_PAR_QUEUES=1) The run's handlers write this processor's registers only (RegLaneBits): made
			// without the queue lock, beside the other translating thread's work (nothing unless on; not with 2, whose
			// walk work between the packets needs it).
			std::optional<ParQueues::UnlockedScope> unlocked;
			if (!verify && ParQueues::On()) unlocked.emplace();
			for (;;) {
				const uint32_t  at     = cursor.offset_dw;
				const uint32_t* lane   = cursor.commands.data() + at;
				const uint32_t  rest   = total_dw - at;
				const uint32_t  header = lane[0];
				const uint32_t  op     = (header >> 8u) & 0xffu;
				if (taken != 0) { // (the run's first packet had the walk's work and class already)
					pm4_clock.Next();
					if (verify) between();
					pm4_clock.Class(Pm4ClassOf(op));
					pm4_clock.Key(header); // (otros1)
					CensoPm4::Packet(CensoPm4::KeyOf(header)); // (w18)
				}
				const auto handler = g_cp_op_func[op];
				if (verify && (!KeepsDeferredProtection(op) || handler == nullptr || header == DsLinearCopyHook::Header))
					++Render5::g_lane[Render5::LaneDifferent];
				if (handler == nullptr) break; // (the walk reports it)
				m_chunk_last_opcode = op;
				LiveCounters::AddPm4(op);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
				if (observe)
					DrawStateObserver::ObservePacket(op, lane, KYTY_PM4_LEN(header) < rest ? KYTY_PM4_LEN(header) : rest, false);
#endif
				m_draw_run_skip           = 0;
				const uint32_t packet_dw = handler(*this, header & ~1u, lane + 1, rest, total_dw) + 1;
				EXIT_IF(packet_dw > rest);
				if (execution.m_suspended) {
					if (execution.m_buffer_stack.size() > buffer_index + 1)
						execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
					return;
				}
				EXIT_IF(execution.m_buffer_stack.size() != buffer_index + 1);
				cursor.offset_dw += packet_dw;
				execution.m_made_progress = true;
				++taken;
				// One-dword NOPs between them, as the walk passes them over.
				while (taken < RegLaneMaxPackets && cursor.offset_dw < total_dw &&
				       cursor.commands[cursor.offset_dw] == 0x80000000u) {
					pm4_clock.Next();
					if (verify) between();
					pm4_clock.Class(Pm4ClassOf(0));
					pm4_clock.Key(0x80000000u); // (otros1)
					cursor.offset_dw++;
					++taken;
				}
				if (taken >= RegLaneMaxPackets || !RegLanePacket(cursor.commands.data(), cursor.offset_dw, total_dw)) break;
			}
			} // (par1: the queue lock again)
			// (par1) The other thread may have begun its slice meanwhile: the open command buffer, shared, is bound to its
			// processor's registers then (CommandScheduler::Begin); bound to this one's again before any packet records.
			if (ParQueues::On() && !verify) {
				auto& scheduler = GetScheduler();
				if (!scheduler.Active() || &scheduler.Current().GetRegisters() != &m_ctx) BufferInit();
			}
			++Render5::g_lane[Render5::LaneRuns];
			Render5::g_lane[Render5::LanePackets] += taken;
			if (taken != 0) continue;
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
			if (!StatePassRuns(opcode, packet_header)) {
				const auto packet_dw = KYTY_PM4_LEN(packet_header);
				EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
				cursor.offset_dw += packet_dw;
				execution.m_made_progress = true;
				continue;
			}
		}

		// (port2) KYTY_PORT2 (local/port2.h): a marker the emulator's Agc wrote after a draw / dispatch. Sealed (the state
		// the binder and the game wrote for the next draw, recorded as a flow when that draw was recorded): applied from the
		// flow and its span passed over, the walk lands on the next draw; else (not sealed, a speculation, a state pass, a
		// chunk, a stop, a capture, the record gone) a plain NOP: the span after it is walked as always.
		if (packet_header == Port2::Header && Port2::Mode() != 0 && Port2::IsMarker(packet, remaining_dw)) {
			const bool can_apply = spec == nullptr && !m_state_only && m_stop_at == nullptr && m_chunk_every == 0 &&
			                       g_gpu_state != nullptr && GuestGpu::IsGpuThread() &&
			                       kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0 &&
			                       !GraphicsRunDebugDumpEnabled() && !Render1::On() && Port1::Mode() == 0 &&
			                       !FrameCapture::Active() && !XprCapture::Enabled();
			cursor.offset_dw += Port2Marker(packet, remaining_dw, can_apply);
			execution.m_made_progress = true;
			continue;
		}

		// (F5 pase 3, KYTY_NATIVO_PASE_ESCENA=2) The scene hook's CENS marks (6 dwords) around one pass call's draws: the
		// translation time between them is the pass's cost ([censo-escena]). Passed over.
		if (packet_header == DsCensoEscena::MarkHeader && DsCensoEscena::IsMark(packet, remaining_dw)) {
			if (spec == nullptr && !m_state_only && GuestGpu::IsGpuThread()) DsCensoEscena::GpuMark(packet);
			m_draw_run_skip = 0; // (como las marcas NATV)
			cursor.offset_dw += DsCensoEscena::MarkDwords;
			execution.m_made_progress = true;
			continue;
		}

		// (F5 pase 2, =3 / =1) The shadow hook's NATV marks (6 dwords) around one bucket's draws: the native cascade
		// (renderer/dsSombrasNativas.inc). Passed over; in a speculation or a state pass, nothing else.
		if (packet_header == DsNativoSombras::MarkHeader && DsNativoSombras::IsNatMark(packet, remaining_dw)) {
			if (spec == nullptr && !m_state_only && GuestGpu::IsGpuThread() &&
			    (DsNativoSombras::Mode() == 1 || DsNativoSombras::Mode() == 3)) {
				CheckBuffer();
				if (DsNativoSombras::NatDiag() && (packet[3] & 0xFFu) == DsNativoSombras::NatBegin)
					DsNativoSombras::NatScanDcb(packet, remaining_dw, packet[4]); // (diagnostico)
				if (DsNativoSombras::Mode() == 1 && DsNativoSombras::NatCascada() && (packet[3] & 0xFFu) == DsNativoSombras::NatBegin)
					DsNativoSombras::NatScanGroup(packet, remaining_dw, (packet[3] >> 8u) & 0xFFu, packet[4]); // (=1 + CASCADA)
				m_renderer.GetRenderExecutor().DsNatMark(CurrentBuffer(), packet[3], packet[4]);
			}
			m_draw_run_skip = 0;
			cursor.offset_dw += DsNativoSombras::MarkDwords;
			execution.m_made_progress = true;
			continue;
		}

		// (F5 pase 2) KYTY_NATIVO_PASE_SOMBRAS=2 (local/ds-nativo-sombras.h): the shadow hook's mark (a 4-dword NOP) before a
		// cascade's draws. Passed over; the next DrawIndex dumps the VS user data once per cascade.
		if (packet_header == DsNativoSombras::MarkHeader && DsNativoSombras::IsMark(packet, remaining_dw)) {
			if (spec == nullptr && !m_state_only && GuestGpu::IsGpuThread()) {
				DsNativoSombras::g_gpu_pending_index = packet[4] | (uint64_t {packet[5]} << 32u);
				DsNativoSombras::g_gpu_pending       = packet[3];
			}
			cursor.offset_dw += DsNativoSombras::MarkDwords;
			execution.m_made_progress = true;
			continue;
		}

		// (F5 pase 1) KYTY_NATIVO_PASE_SSRCLEAR (local/ds-nativo-pase.h): the game's SSRClear texture clear announced the
		// T# its packets after this one clear. 1: cleared natively and those packets passed over; 2: cleared natively and
		// those packets run too (the emulated image is the one shown). Refused, unsealed, a speculation, a state pass or a
		// chunked walk: the announcement alone is passed over and the game's packets run as always.
		if (packet_header == DsNativoPase::Header && DsNativoPase::IsPacket(packet, remaining_dw)) {
			uint32_t advance = DsNativoPase::PacketDwords;
			if (spec == nullptr && !m_state_only && m_stop_at == nullptr && m_chunk_every == 0 && g_gpu_state != nullptr &&
			    GuestGpu::IsGpuThread() && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0 &&
			    DsNativoPase::IsSealed(packet, remaining_dw)) {
				CheckBuffer();
				if (m_renderer.GetComputeRenderExecutor().DsNativeClear(CurrentBuffer(), packet + DsNativoPase::TSharpWord)) {
					if (DsNativoPase::Mode() == 1) {
						advance += packet[DsNativoPase::SkipWord]; // (up to the dispatch: its EVENT_WRITEs run)
						DsNativoPase::Count(DsNativoPase::Native);
					} else {
						DsNativoPase::Count(DsNativoPase::Verified);
					}
				} else {
					DsNativoPase::Count(DsNativoPase::Refused);
				}
				DsNativoPase::Report();
			}
			m_draw_run_skip = 0;
			cursor.offset_dw += advance;
			execution.m_made_progress = true;
			continue;
		}

		// (render4) KYTY_DS_LINEAR_COPY_HOOK (local/ds-linear-copy-hook.h): the game's linear copy function announced
		// the copy its packets after this one make. Made here from the announcement, those packets are passed over
		// (as a dispatch: the deferred protections stay pending); else the announcement alone is, as if it were not
		// there, and the game's packets run. Never made in a speculation or a state pass, nor where a chunk's end or
		// boundary could lie within the packets passed over, nor with KYTY_SPECULATE at all (its state passes run the
		// game's register packets, and a commit compares the processor's state with theirs).
		if (packet_header == DsLinearCopyHook::Header && DsLinearCopyHook::IsPacket(packet, remaining_dw)) {
			uint32_t advance = DsLinearCopyHook::PacketDwords;
			if (spec == nullptr && !m_state_only && m_stop_at == nullptr && m_chunk_every == 0 && g_gpu_state != nullptr &&
			    GuestGpu::IsGpuThread() && kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0 &&
			    DsLinearCopyHook::IsSealed(packet, remaining_dw)) {
				const auto     dst    = DsLinearCopyHook::Address(packet, DsLinearCopyHook::DstWord);
				const auto     src    = DsLinearCopyHook::Address(packet, DsLinearCopyHook::SrcWord);
				const uint32_t count  = packet[DsLinearCopyHook::CountWord];
				const uint32_t period = packet[DsLinearCopyHook::PeriodWord];
				const uint32_t span   = packet[DsLinearCopyHook::SpanWord];
				const auto     bytes  = uint64_t {DsLinearCopyHook::PacketDwords + span} * sizeof(uint32_t);
				if (DsLinearCopyHook::Mode() == 2) {
					// (Verify: the dispatch that follows compares its copy with the announcement's.)
					DemonsSouls::ExpectLinearCopy(dst, src, count, period, reinterpret_cast<uint64_t>(packet), bytes);
				} else if (Copia1::On()) {
					// (copia1, local/copia1.h) This announcement and those right after it (only one-dword NOPs between),
					// made in one run, in order; `ends[i]`: the offset past copy i's game packets and the NOPs after them.
					CheckBuffer();
					std::array<RenderExecutor::DsLinearCopyItem, Copia1::MaxRun> items;
					std::array<uint32_t, Copia1::MaxRun> starts {}, ends {};
					uint32_t       n   = 0;
					uint32_t       at  = cursor.offset_dw;
					Copia1::Flush  why = Copia1::Full;
					while (true) {
						const auto* p    = cursor.commands.data() + at;
						const auto  left = total_dw - at;
						if (n != 0) {
							if (left == 0) { why = Copia1::BufferEnd; break; }
							if (!DsLinearCopyHook::IsPacket(p, left)) { why = Copia1::OtherPacket; break; }
							if (!DsLinearCopyHook::IsSealed(p, left)) { why = Copia1::NotSealed; break; }
						}
						const uint32_t sp = p[DsLinearCopyHook::SpanWord];
						items[n]          = {.dst            = DsLinearCopyHook::Address(p, DsLinearCopyHook::DstWord),
						                     .src            = DsLinearCopyHook::Address(p, DsLinearCopyHook::SrcWord),
						                     .count          = p[DsLinearCopyHook::CountWord],
						                     .period         = p[DsLinearCopyHook::PeriodWord],
						                     .commands       = reinterpret_cast<uint64_t>(p),
						                     .commands_bytes = uint64_t {DsLinearCopyHook::PacketDwords + sp} * sizeof(uint32_t)};
						starts[n] = at;
						at += DsLinearCopyHook::PacketDwords + sp;
						while (at < total_dw && cursor.commands[at] == 0x80000000u) ++at;
						ends[n] = at;
						if (++n == Copia1::MaxRun) { why = Copia1::Full; break; }
					}
					const auto t0   = std::chrono::steady_clock::now();
					const auto made = m_renderer.GetComputeRenderExecutor().DsLinearCopyBatch(CurrentBuffer(), items.data(), n);
					const auto ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
					for (uint32_t i = 0; i < made; ++i) DsLinearCopyHook::Count(DsLinearCopyHook::Taken);
					if (made < n) {
						// (Copy `made` refused: its announcement alone is passed over, its game's packets run; the
						// announcements after it come to this branch again.)
						DsLinearCopyHook::Count(DsLinearCopyHook::Passed);
						why = Copia1::Refused;
						advance = starts[made] + DsLinearCopyHook::PacketDwords - cursor.offset_dw;
					} else {
						advance = ends[n - 1] - cursor.offset_dw;
					}
					Copia1::NoteRun(made, static_cast<uint64_t>(ns), why);
				} else {
					CheckBuffer();
					if (m_renderer.GetComputeRenderExecutor().DsLinearCopy(CurrentBuffer(), dst, src, count, period,
					                                                       reinterpret_cast<uint64_t>(packet), bytes)) {
						advance += span;
						DsLinearCopyHook::Count(DsLinearCopyHook::Taken);
					} else {
						DsLinearCopyHook::Count(DsLinearCopyHook::Passed);
					}
				}
			}
			m_draw_run_skip = 0;
			cursor.offset_dw += advance;
			execution.m_made_progress = true;
			continue;
		}

		// Only state, draws and dispatches keep the deferred read protections pending: before a label, a memory write,
		// a wait or anything else the pages GPU work writes are protected (PageManager::DeferReadProtection).
		if (!KeepsDeferredProtection(opcode)) {
			// (FLUSH-ASYNC, KYTY_FLUSH_ASYNC=1) Packets that write no memory, wait for nothing and signal nothing: the
			// pending no-access ranges go to a worker (joined before the next flush or protection of this thread).
			if (opcode == Pm4::IT_NOP || opcode == Pm4::IT_EVENT_WRITE || opcode == Pm4::IT_ACQUIRE_MEM ||
			    opcode == Pm4::IT_INDIRECT_BUFFER || opcode == Pm4::IT_INDIRECT_BUFFER_CNST || opcode == Pm4::IT_CONTEXT_CONTROL ||
			    opcode == Pm4::IT_CLEAR_STATE || opcode == Pm4::IT_PFP_SYNC_ME)
				PageManager::FlushDeferredProtectionAsync();
			else
				PageManager::FlushDeferredProtection();
		}
		// (f5p1 techo, KYTY_F5P1_TECHO=1; local/f5p1-techo.h) In an ON window, a draw of a verified pass 0/1 range of
		// 0xad8bc0 is passed over (nothing translated, recorded or drawn); everything else runs as always.
		if (F5p1Techo::On() && !m_state_only) {
			if (const uint32_t skip = F5p1Techo::SkipDraw(opcode, packet, remaining_dw, cursor.commands.data(), total_dw)) {
				m_draw_run_skip = 0;
				cursor.offset_dw += skip;
				execution.m_made_progress = true;
				continue;
			}
		}
		// (f5b) The pass cut (native/pass-cut.h): KYTY_NATIVE_TECHO passes over the draws / dispatches of the pass of its
		// ceiling window; KYTY_NATIVE_PASSES ..:2 snapshots verified passes (state packets, target copies at the scope
		// marks). One static load per packet when neither is set.
		// (pt1) BB_PATHTRACING's GPU work at the packet after a game scope mark (TLAS build after GBuffer, shadows before
		// RenderLights, reflections before RenderIBL). One static load per packet when off.
		if (PathTrace::Requested() && !m_state_only && spec == nullptr && PathTrace::WorkPending()) PathTrace::RunWork(CurrentBuffer());
		if (NativeRender::WalkOn() && !m_state_only && spec == nullptr) {
			if (NativeRender::GpuWorkPending()) NativeRender::RunGpuWork(CurrentBuffer());
			if (const uint32_t skip =
			        NativeRender::WalkPacketSlow(opcode, packet, remaining_dw, cursor.commands.data(), total_dw)) {
				m_draw_run_skip = 0;
				cursor.offset_dw += skip;
				execution.m_made_progress = true;
				continue;
			}
		}
		// (F5 pase 2, =3 / =1; renderer/dsSombrasNativas.inc) Cada draw directo indexado de un cubo de sombras marcado, ANTES
		// de cualquier camino que lo dibuje (tablas, XPR nativo o el handler): con =3, el primero copia la capa emulada; con
		// =1, en un cubo reemplazado, el draw de una entrada que dibuja el nativo se salta entero (solo este paquete: los
		// de estado siguen). Indices como CommandProcessor::DrawIndexOffset / DRAW_INDEX_2.
		if ((opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_2) &&
		    (DsNativoSombras::Mode() == 1 || DsNativoSombras::Mode() == 3) && (m_state_only || spec != nullptr || !GuestGpu::IsGpuThread()))
			DsNativoSombras::g_nat_draws_elsewhere[m_state_only ? 0 : spec != nullptr ? 1 : 2].fetch_add(1, std::memory_order_relaxed); // (diagnostico)
		if ((opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_2) && !m_state_only && spec == nullptr &&
		    (DsNativoSombras::Mode() == 1 || DsNativoSombras::Mode() == 3) && GuestGpu::IsGpuThread() &&
		    m_renderer.GetRenderExecutor().DsNatWants()) {
			const uint32_t packet_dw = KYTY_PM4_LEN(packet_header);
			if (packet_dw >= 5 && packet_dw <= remaining_dw) {
				const uint64_t isz        = m_index_type_and_size == 0 ? 2 : m_index_type_and_size == 1 ? 4 : 1;
				const bool     offset2    = opcode == Pm4::IT_DRAW_INDEX_OFFSET_2;
				const uint64_t index_addr = offset2 ? m_index_base_addr + uint64_t {packet[2]} * isz
				                                    : (uint64_t {packet[2]} | (uint64_t {packet[3] & 0xFFFFu} << 32u));
				const uint32_t count      = offset2 ? packet[3] : packet[4];
				CheckBuffer();
				DsNativoSombras::g_nat_cur_es = m_sh_ctx.GetVs().es_regs.data_addr; // (diagnostico: el VS de este draw)
				if (m_renderer.GetRenderExecutor().DsNatBeforeDraw(CurrentBuffer(), index_addr, count)) {
					m_draw_run_skip = 0;
					cursor.offset_dw += packet_dw;
					execution.m_made_progress = true;
					continue;
				}
			}
		}
		G1::WalkMark(G1::WChecked); // (g1 NP1: the walk's checks done)
		auto handler = g_cp_op_func[opcode];
		LiveCounters::AddPm4(opcode);
		if (Port1::Mode() != 0) Port1::OnPacket(opcode, packet, std::min(KYTY_PM4_LEN(packet_header), remaining_dw)); // (port1)
		// (render1) KYTY_RENDER1=1/2: this draw (a run that takes more packets below observes the rest).
		Render1Step render1_step; // (F2: the step of a p0/p1 draw; ends with this iteration)
		if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT && Render1::On() && !m_state_only && packet_header == 0xc0032500u) {
			bool       all    = false;
			const auto native = Render1Observe(m_ctx, packet, 1, all);
			if (Render1::Mode() >= 2 && spec == nullptr) {
				auto& executor     = m_renderer.GetRenderExecutor();
				Render1::t_current = {};
				// (F3) The window this step is drawn in, and its first draw's R.
				Render1::t_current.f3_on = Render1::g_f3_on.load(std::memory_order_relaxed);
				Render1::t_current.r     = native == 1 ? Render1::t_last_r : 0;
				if (native == 1) {
					Render1::t_current.native = true;
					Render1::t_current.args   = m_draw_indirect_args_base_addr + packet[1];
					Render1::t_step_draws     = 1;
					render1_step.Start(executor);
				} else if (Render1::Mode() == 2 && executor.Render1BatchOpen()) {
					executor.Render1EndBatch(CurrentBuffer(), Render1::EDrawOther);
				}
			}
		}
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
		G1::WalkMark(G1::WEntry); // (g1 NP1: the native XPR block begins)
		std::optional<ShaderReadObserver> native_reads;
		struct NativeStoreEnd {
			RenderExecutor* executor = nullptr;
			~NativeStoreEnd() {
				if (executor) executor->NativeXprEndPacket();
			}
		} native_store_end;
		if (const auto native = kyty_local_native_xpr_mode.load(std::memory_order_relaxed);
		    native != 0 && (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 16u) == 0) {
			DrawStateObserver::ObservePacket(opcode, packet, std::min(KYTY_PM4_LEN(packet_header), remaining_dw),
			                                 false);
			if (opcode == Pm4::IT_DRAW_INDEX_INDIRECT &&
			    (packet_header != 0xc0032500u || GraphicsRunDebugDumpEnabled()))
				m_renderer.GetRenderExecutor().NativeXprForgetState();
			if (packet_header == 0xc0032500u && !GraphicsRunDebugDumpEnabled()) {
				G1::WalkMark(G1::WObserved); // (g1 NP1: the observer done, the try begins)
				const auto consumed =
				    TryNativeXprDraws({packet, remaining_dw}, DrawStateObserver::g_shadow.last_clean);
				G1::WalkMark(G1::WTried); // (g1 NP1: the try returned)
				if (consumed) {
					if (Port1::Mode() != 0) Port1::OnRun(packet, std::min(consumed, remaining_dw)); // (port1: the run's other packets)
					if (Render1::On() && !m_state_only && consumed > 5) Render1ObserveRest(m_ctx, m_renderer.GetRenderExecutor(), packet + 5, consumed / 5u - 1u, spec != nullptr);
					m_draw_run_skip = 0;
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(Gancho2::t_last_draws); // (gancho2: gaps make it not consumed / 5)
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
				G1::WalkMark(G1::WObserved); // (g1 NP1)
				const auto consumed = TryNativeMultiDraw({packet, remaining_dw});
				G1::WalkMark(G1::WTried); // (g1 NP1)
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
				G1::WalkMark(G1::WObserved); // (g1 NP1)
				const auto consumed = TryNativeAutoDraw({packet, remaining_dw});
				G1::WalkMark(G1::WTried); // (g1 NP1)
				if (consumed) {
					execution.m_buffer_stack[buffer_index].offset_dw += consumed;
					execution.m_made_progress = true;
					GetScheduler().CompleteDraws(1);
					continue;
				}
				if (spec != nullptr) return refuse("draw");
			} else if ((packet_header == 0xc0042700u || packet_header == 0xc0033500u) &&
			           !GraphicsRunDebugDumpEnabled()) {
				// Direct indexed draws from the same records (the normal path stores them).
				G1::WalkMark(G1::WObserved); // (g1 NP1)
				const auto consumed = TryNativeDirectDraw({packet, remaining_dw});
				G1::WalkMark(G1::WTried); // (g1 NP1)
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
				if (Render1::On() && !m_state_only && consumed > 5) Render1ObserveRest(m_ctx, m_renderer.GetRenderExecutor(), packet + 5, consumed / 5u - 1u, spec != nullptr);
				execution.m_buffer_stack[buffer_index].offset_dw += consumed;
				execution.m_made_progress = true;
				GetScheduler().CompleteDraws(Port2::t_run_draws); // (port2: not consumed / 5 over markers)
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
		                        opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO))
			return refuse("draw"); // (no table draw took it)
		uint32_t packet_dw = 0;
		if (spec == nullptr) {
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		} else {
			// What would submit or wait for the GPU refuses the packet.
			try {
				packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			} catch (const CommandScheduler::RecorderRefusal& refusal) {
				Spec::Refuse(refusal.what);
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
	// (render1 F2) The whole command buffer walked: the native batch ends with it (before any submission).
	if (Render1::Mode() == 2 && spec == nullptr && stop_depth == 0 && !m_state_only) {
		auto& executor = m_renderer.GetRenderExecutor();
		if (executor.Render1BatchOpen()) executor.Render1EndBatch(CurrentBuffer(), Render1::EPm4End);
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

			// (KYTY_NATIVE_MEMORY: the predicate GPU work wrote was read back by this read's fault; no fault comes now.)
			// (raiz1, mode protect: the read faults; not here.)
			if (PageManager::NativeWriteWatch() && !PageManager::WatchProtects() && !m_state_only && Spec::Current() == nullptr &&
			    GuestGpu::IsGpuThread())
				(void)Libs::LibKernel::Memory::SyncGpuCleanBacking(reinterpret_cast<uint64_t>(address), sizeof(uint64_t));

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
	if (DsNativoSombras::Mode() == 2 && DsNativoSombras::g_gpu_pending.load(std::memory_order_relaxed) != 0xFFFFFFFFu) {
		// (F5 pase 2, =2: registros del VS de la cascada marcada y el estado de la GPU de este draw: viewport, recortes,
		// destino de profundidad y sesgo; solo registros del emulador, nada de memoria del juego en este hilo)
		char        state[512];
		const auto& vp = m_ctx.GetScreenViewport();
		const auto& v0 = vp.viewports[0];
		const auto& db = m_ctx.GetDepthRenderTarget();
		const auto& dv = m_ctx.GetDepthDepthView();
		const auto& sz = m_ctx.GetDepthDepthSizeXY();
		const auto& po = m_ctx.GetPolyOffset();
		if (db.z_write_base_addr != 0) DsNativoSombras::g_shadow_depth.store(db.z_write_base_addr, std::memory_order_relaxed);
		std::snprintf(state, sizeof(state),
		              "vp0 escala=(%g,%g,%g) desplaz=(%g,%g,%g) z=[%g,%g] recorte_vp=(%d,%d)-(%d,%d) pantalla=(%d,%d)-(%d,%d) "
		              "ventana=(%d,%d)-(%d,%d) | prof z_read=%llx z_write=%llx capas=%u..%u tam=%ux%u | sesgo escala=%g desplaz=%g "
		              "clamp=%g",
		              double(v0.xscale), double(v0.yscale), double(v0.zscale), double(v0.xoffset), double(v0.yoffset),
		              double(v0.zoffset), double(v0.zmin), double(v0.zmax), v0.viewport_scissor_left, v0.viewport_scissor_top,
		              v0.viewport_scissor_right, v0.viewport_scissor_bottom, vp.screen_scissor_left, vp.screen_scissor_top,
		              vp.screen_scissor_right, vp.screen_scissor_bottom, vp.window_scissor_left, vp.window_scissor_top,
		              vp.window_scissor_right, vp.window_scissor_bottom, static_cast<unsigned long long>(db.z_read_base_addr),
		              static_cast<unsigned long long>(db.z_write_base_addr), dv.slice_start, dv.slice_max,
		              static_cast<unsigned>(sz.x_max) + 1u, static_cast<unsigned>(sz.y_max) + 1u, double(po.front_scale),
		              double(po.front_offset), double(po.clamp));
		DsNativoSombras::GpuDrawDump(m_sh_ctx.GetVs().es_regs.data_addr, m_sh_ctx.GetVs().ls_regs.data_addr,
		                             m_sh_ctx.GetVs().hs_regs.data_addr, m_sh_ctx.GetVs().gs_regs.data_addr,
		                             m_sh_ctx.GetVs().hs_user_sgpr.value, m_sh_ctx.GetVs().hs_user_sgpr.count,
		                             m_sh_ctx.GetVs().gs_user_sgpr.value, m_sh_ctx.GetVs().gs_user_sgpr.count,
		                             reinterpret_cast<uint64_t>(args.index_addr), state);
	}
	// (F5 pase 2, =3 / =1) DsNatBeforeDraw ya no va aqui: va por paquete en ProcessPm4 (la mayoria de los draws de sombra
	// los hace el camino de tablas / XPR nativo y nunca llegan a esta funcion: log 20261125-024655-421, ~1 de cada 25).

	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (args.base_vertex != 0 || args.first_instance != 0) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
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

#ifdef KYTY_LOCAL_VULKAN_RECORDING
// KYTY_DECAL_SAFE (on unless 0; des-candle-1111): a blended draw that tests depth without writing it and is not an
// equal test (decals and mesh decals over the G-buffer: the wax runs and splashes under candle groups, blood, dirt;
// forward translucent meshes) is drawn by the normal path, never from a native XPR record, a table draw or a
// continued table run. Those paths key a draw by its pair, its draw state and its n-th use in the frame and patch only
// its data words; a decal pass draws many small objects of one pair in an order that changes from frame to frame, so a
// record could be replayed for another object or another blend state, and a whole decal came and went on a still
// camera (TAA then showed it half transparent). The normal path evaluates every draw from its registers and memory.
// The G-buffer itself (depth writes or an equal test) and everything without blending keep the fast paths.
static bool DecalSafeOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DECAL_SAFE");
		const bool  set   = value == nullptr || value[0] != '0';
		std::printf("Decal draws: KYTY_DECAL_SAFE=%d: blended draws that test depth without writing it (decals, wax, "
		            "translucent meshes) %s\n",
		            set ? 1 : 0, set ? "take the normal path (no native XPR record, table draw or continued run)"
		                         : "may take the native XPR / table paths (as before)");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

static bool DecalLikeDraw(const HW::Context& ctx) {
	const auto& depth = ctx.GetDepthControl();
	if (!depth.z_enable || depth.z_write_enable || depth.zfunc == static_cast<uint8_t>(vk::CompareOp::eEqual) ||
	    depth.zfunc == static_cast<uint8_t>(vk::CompareOp::eAlways))
		return false;
	const uint32_t mask = ctx.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8; ++slot) {
		if (((mask >> (slot * 4u)) & 0xfu) == 0) continue;
		const auto& target = ctx.GetRenderTarget(slot);
		if (target.base.addr == 0 || target.info.blend_bypass) continue;
		if (ctx.GetBlendControl(slot).enable) return true;
	}
	return false;
}

// The normal path takes the draw (KYTY_DECAL_SAFE): counted, the first ones and then every power of two printed.
static bool DecalTakesNormalPath(const HW::Context& ctx) {
	if (!DecalSafeOn() || !DecalLikeDraw(ctx)) return false;
	static std::atomic<uint64_t> routed {0};
	const uint64_t               n = routed.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 2 || (n & (n - 1)) == 0) {
		const auto& depth = ctx.GetDepthControl();
		std::printf("Decal draws: %llu drawn by the normal path (KYTY_DECAL_SAFE; last: depth func %u, targets mask "
		            "0x%08x, blend CB0 %d; KYTY_DECAL_SAFE=0: as before)\n",
		            static_cast<unsigned long long>(n), static_cast<unsigned>(depth.zfunc), ctx.GetRenderTargetMask(),
		            ctx.GetBlendControl(0).enable ? 1 : 0);
		std::fflush(stdout);
	}
	return true;
}

#endif // KYTY_LOCAL_VULKAN_RECORDING

// (parpadeo2, KYTY_PARPADEO; local/parpadeo.h) An indexed indirect draw run about to be drawn (`args` its argument
// addresses): what its arguments and indices are at this point. Not a speculation's (the callers check).
void CommandProcessor::ParpadeoNoteDraws(std::span<const uint64_t> args, uint64_t index_bytes, uint32_t path) {
	if (!GetScheduler().Active()) return;
	Parpadeo::OnDraws(GetGpuResources().GetBufferCache(), CurrentBuffer(), GetScheduler(), m_renderer.GetMutex(),
	                  m_draw_indirect_args_base_addr, args, sizeof(DrawIndexedIndirectArgs), m_index_base_addr, index_bytes,
	                  static_cast<Parpadeo::Path>(path));
}

uint32_t CommandProcessor::TryNativeXprDraws(std::span<const uint32_t> packets, bool clean) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 0);
	TrustedDepth::ClassClock class_clock(0); // (trust1, KYTY_RENDER_STATS: us per call by class)
	LiveTrace::MarkAfter gpu_mark {[&] { return GetScheduler().Current().RawHandle(); }, LiveTrace::MarkNativeXpr,
	                               m_sh_ctx.GetPs().ps_regs.data_addr};
	LiveCounters::g_last_dispatch_shader = 0;
	SyncStats::g_shader = 0;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	auto& executor = m_renderer.GetRenderExecutor();
	if (Gancho2::Mode() != 0) Gancho2::Note(Gancho2::RCalls);
	if ((FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed))) { // a recorded frame sees every draw on the normal path
		executor.NativeXprForgetState();
		if (Gancho2::Mode() != 0) Gancho2::Note(Gancho2::RExitEarly);
		return 0;
	}
	if (DecalTakesNormalPath(m_ctx)) {
		executor.NativeXprForgetState();
		if (Gancho2::Mode() != 0) Gancho2::Note(Gancho2::RExitEarly);
		return 0;
	}
	if (packets.size() < 5 || (packets[4] & ~0x20u) != 2u || m_index_type_and_size > 1 ||
	    !m_index_base_addr || !m_index_buffer_size || !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList) {
		// The cached draw-state key covers only draws the native path saw.
		executor.NativeXprForgetState();
		if (Gancho2::Mode() != 0) Gancho2::Note(Gancho2::RExitEarly);
		return 0;
	}
	// A run: consecutive packets that differ only in the argument offset draw
	// the same object (no register write between them).
	constexpr uint32_t max_draws = 64;
	uint32_t           count     = 1;
	std::array<uint64_t, max_draws> commands; // the first `count` are written, only they are read
	commands[0]       = m_draw_indirect_args_base_addr + packets[1];
	// (render1 F3) Each draw's packet (the run's other draws are paired before it is drawn).
	std::array<const uint32_t*, max_draws> draw_packets;
	draw_packets[0] = packets.data();
	uint32_t consumed = 5; // dwords the run takes
	uint32_t gap_draws = 0, gap_skipped = 0; // (gancho2, counted when the run is drawn)
	// (gancho2, KYTY_GANCHO2; local/gancho2.h) Also over a gap that rewrites what the registers hold (the game's binder
	// re-emits the user data before every draw, ad8bc0 -> 1d10840): fillers, the game's user-data markers (NOP R_ZERO
	// "hu", ids 0 / 4 / 0xd) and SET_SH_REG packets. A user-data SGPR is compared with the command processor's own
	// (value, the type the write would store - the marker then -, count); another SH register with the draw state
	// observer's shadow. The marker must end as it was. Not in a speculation, a state pass, a stop, a capture.
	using G2         = Gancho2::Reason;
	const auto mode2 = Gancho2::Mode();
	bool       gap_runs = false;
	if (mode2 != 0) {
		G2 off = G2::Reasons;
		if (!Gancho2::Active()) off = G2::ROffWindow;
		else if (Spec::Current() != nullptr) off = G2::ROffSpec;
		else if (m_state_only) off = G2::ROffState;
		else if (m_chunk_every != 0) off = G2::ROffChunk;
		else if (m_stop_at != nullptr) off = G2::ROffStop;
		else if (XprCapture::Enabled()) off = G2::ROffCapture;
		if (off != G2::Reasons) Gancho2::Note(off);
		gap_runs = off == G2::Reasons;
	}
	const auto same_draw = [&](const uint32_t* next) {
		return next[0] == 0xc0032500u && next[2] == packets[2] && next[3] == packets[3] && next[4] == packets[4];
	};
	// The user-data SGPRs of SH register `offset` (null: another register).
	const auto user_sgprs = [&](uint32_t offset, uint32_t& slot) -> const HW::UserSgprInfo* {
		if (offset >= Pm4::SPI_SHADER_USER_DATA_PS_0 && offset < Pm4::SPI_SHADER_USER_DATA_PS_0 + 32u) {
			slot = offset - Pm4::SPI_SHADER_USER_DATA_PS_0;
			return &m_sh_ctx.GetPs().ps_user_sgpr;
		}
		if (offset >= Pm4::SPI_SHADER_USER_DATA_GS_0 && offset < Pm4::SPI_SHADER_USER_DATA_GS_0 + 32u) {
			slot = offset - Pm4::SPI_SHADER_USER_DATA_GS_0;
			return &m_sh_ctx.GetVs().gs_user_sgpr;
		}
		if (offset >= Pm4::SPI_SHADER_USER_DATA_HS_0 && offset < Pm4::SPI_SHADER_USER_DATA_HS_0 + 32u) {
			slot = offset - Pm4::SPI_SHADER_USER_DATA_HS_0;
			return &m_sh_ctx.GetVs().hs_user_sgpr;
		}
		return nullptr;
	};
	G2 end = G2::REndMax;
	// (port2) Runs go on over the empty markers of KYTY_PORT2 (not with render1: its observers count 5 dwords a draw).
	const bool port2_runs = Port2::Mode() != 0 && !Render1::On();
	while (count < max_draws) {
		if (consumed + 5u > packets.size()) {
			end = G2::REndBuffer;
			break;
		}
		const auto* next = packets.data() + consumed;
		// (port2) An empty marker of the emulator's Agc between two draws of the run (a plain NOP): taken with them.
		if (port2_runs && consumed + Port2::MarkerDw + 5u <= packets.size() &&
		    Port2::IsEmptyMarker(next, packets.size() - consumed) && same_draw(next + Port2::MarkerDw)) {
			draw_packets[count] = next + Port2::MarkerDw;
			commands[count++]   = m_draw_indirect_args_base_addr + next[Port2::MarkerDw + 1u];
			consumed += Port2::MarkerDw + 5u;
			continue;
		}
		if (same_draw(next)) {
			draw_packets[count] = next;
			commands[count++]   = m_draw_indirect_args_base_addr + next[1];
			consumed += 5u;
			continue;
		}
		if (!gap_runs) {
			end = G2::Reasons;
			break;
		}
		uint32_t at = consumed, gap_packets = 0;
		auto     marker = m_user_data_marker;
		G2       why    = G2::Reasons; // (Reasons: the gap is a no-op so far)
		while (at < packets.size() && why == G2::Reasons) {
			const uint32_t header = packets[at];
			if (header == 0x80000000u) {
				++at;
				++gap_packets;
				continue;
			}
			const uint32_t op = (header >> 8u) & 0xffu;
			if ((header >> 30u) != 3u || (header & 1u) != 0) break;
			const uint32_t len = KYTY_PM4_LEN(header);
			if (len < 2u || len > packets.size() - at) {
				why = G2::RBadSh;
				break;
			}
			if (op == Pm4::IT_NOP && header != DsLinearCopyHook::Header && KYTY_PM4_R(header) == Pm4::R_ZERO &&
			    (packets[at + 1u] & 0xffff0000u) == 0x68750000u) {
				const uint32_t id = packets[at + 1u] & 0xfffu;
				if (id == 0x4u) marker = HW::UserSgprType::Vsharp;
				else if (id == 0xdu) marker = HW::UserSgprType::Region;
				else if (id != 0x0u) {
					++Gancho2::g_end_packet[512];
					why = G2::REndPacket;
					break;
				}
				at += len;
				++gap_packets;
				continue;
			}
			if (op != Pm4::IT_SET_SH_REG) break;
			const uint32_t offset = packets[at + 1u], values = len - 2u;
			if (values == 0 || offset == Pm4::SH_NOP || offset >= Pm4::SH_NUM || values > Pm4::SH_NUM - offset) {
				why = G2::RBadSh;
				break;
			}
			uint32_t    slot = 0;
			const auto* ud   = user_sgprs(offset, slot);
			if (ud != nullptr) {
				if (slot + values > 32u) {
					why = G2::RBadSh;
					break;
				}
				for (uint32_t i = 0; i < values && why == G2::Reasons; ++i) {
					const uint32_t v = packets[at + 2u + i];
					if (ud->value[slot + i] != v) why = G2::RDiffUdValue;
					else if (ud->type[slot + i] != marker) why = G2::RDiffUdType;
					else if (slot + i + 1u > ud->count) why = G2::RDiffUdCount;
					if (why != G2::Reasons) ++Gancho2::g_diff_offset[offset + i];
				}
				marker = HW::UserSgprType::Unknown; // (the user-data handlers clear it after the packet)
			} else {
				for (uint32_t i = 0; i < values && why == G2::Reasons; ++i)
					if (!DrawStateObserver::ShHolds(offset + i, packets[at + 2u + i])) {
						why = G2::RDiffOther;
						++Gancho2::g_diff_offset[offset + i];
					}
			}
			if (why != G2::Reasons) break;
			at += len;
			++gap_packets;
		}
		if (why == G2::Reasons) {
			if (at + 5u > packets.size()) {
				why = G2::REndBuffer;
			} else if (!same_draw(packets.data() + at)) {
				const uint32_t header = packets[at];
				const uint32_t op     = (header >> 8u) & 0xffu;
				if (at != consumed && op == Pm4::IT_DRAW_INDEX_INDIRECT) {
					why = G2::RNoDraw;
				} else {
					why = G2::REndPacket;
					++Gancho2::g_end_packet[op == Pm4::IT_NOP ? 256u + (KYTY_PM4_R(header) & 0xffu) : op];
				}
			} else if (marker != m_user_data_marker) {
				why = G2::RMarkerEnd;
			}
		}
		if (why == G2::Reasons && mode2 == 2) {
			// (2) The gap through its handlers: the register files and the marker must not change.
			const size_t         sizes[3] = {sizeof(HW::Context), sizeof(HW::Shader), sizeof(HW::UserConfig)};
			uint8_t* const       files[3] = {reinterpret_cast<uint8_t*>(&m_ctx), reinterpret_cast<uint8_t*>(&m_sh_ctx),
			                                 reinterpret_cast<uint8_t*>(&m_ucfg)};
			std::vector<uint8_t> before[3];
			for (int o = 0; o < 3; ++o) before[o].assign(files[o], files[o] + sizes[o]);
			const auto marker_before = m_user_data_marker;
			for (uint32_t g = consumed; g < at;) {
				const uint32_t header = packets[g];
				if (header == 0x80000000u) {
					++g;
					continue;
				}
				const auto rest = static_cast<uint32_t>(packets.size() - g);
				g += g_cp_op_func[(header >> 8u) & 0xffu](*this, header & ~1u, packets.data() + g + 1, rest, rest) + 1u;
			}
			bool same = m_user_data_marker == marker_before;
			for (int o = 0; o < 3; ++o)
				if (std::memcmp(files[o], before[o].data(), sizes[o]) != 0) {
					same = false;
					if (Gancho2::g_discrepancias++ < 8)
						std::printf("[gancho2] DISCREPANCIA gap of %u packets before draw %u: register file %d changed\n",
						            gap_packets, count, o);
				}
			if (!same) {
				for (int o = 0; o < 3; ++o) std::memcpy(files[o], before[o].data(), sizes[o]);
				m_user_data_marker = marker_before;
				why                = G2::RDiscrepancy;
			}
		}
		if (why != G2::Reasons) {
			end = why;
			break;
		}
		draw_packets[count] = packets.data() + at;
		commands[count++]   = m_draw_indirect_args_base_addr + packets[at + 1u];
		consumed            = at + 5u;
		++gap_draws;
		gap_skipped += gap_packets;
	}
	if (end != G2::Reasons && gap_runs) Gancho2::Note(end);
	Gancho2::t_last_draws = count;
	const auto drawn = [&] {
		Gancho2::g_draws += gap_draws;
		Gancho2::g_packets += gap_skipped;
		Gancho2::g_reason[G2::RGapsTaken] += gap_draws;
		return consumed;
	};
	CheckBuffer();
	const uint64_t element = m_index_type_and_size == 0 ? 2 : 4;
	if (Parpadeo::On() && Spec::Current() == nullptr) // (parpadeo2)
		ParpadeoNoteDraws({commands.data(), count}, uint64_t {m_index_buffer_size} * element, Parpadeo::PathNative);
	// (render1 F3, KYTY_RENDER1=3) A paired p0/p1 depth draw: its run's other draws paired now (in both windows, so the
	// two sides pay the same probe), then, in an "on" window, drawn ONLY the native way when every draw of the run is a
	// p0/p1 depth draw and the native way takes it (render1-native.inc Render1Draw); otherwise the normal path below.
	if (Render1::Mode() == 3 && Render1::t_current.native && Spec::Current() == nullptr && !m_state_only) {
		bool all = true;
		if (count > 1) Render1::t_step_draws += Render1ObservePackets(m_ctx, draw_packets.data() + 1, count - 1, all);
		Render1::t_current.rest_observed = true;
		if (Render1::t_current.f3_on) {
			if (!all) {
				Render1::F3Note(Render1::FRunMixed, count);
				Render1::t_current.f3_noted = true;
			} else if (executor.Render1Draw(CurrentBuffer(), clean, {commands.data(), count}, m_index_base_addr,
			                                uint64_t {m_index_buffer_size} * element,
			                                m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32)) {
				return drawn();
			}
		}
	}
	if (clean && executor.TableContinue(CurrentBuffer(), {commands.data(), count}, m_index_base_addr,
	                                    uint64_t {m_index_buffer_size} * element,
	                                    m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
	                                    nullptr))
		return drawn();
	// On refusal NativeXprTry has asked the normal path to store when that helps.
	if (!executor.NativeXprTry(CurrentBuffer(), clean, {commands.data(), count}, m_index_base_addr,
	                           uint64_t {m_index_buffer_size} * element,
	                           m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32)) {
		if (gap_draws != 0) Gancho2::Note(G2::RRefusedRun);
		return 0;
	}
	return drawn();
#else
	(void)packets;
	(void)clean;
	return 0;
#endif
}

uint32_t CommandProcessor::TryNativeMultiDraw(std::span<const uint32_t> packet) {
	LiveCensus::Scope census(LiveCensus::NativeXpr, 0);
	TrustedDepth::ClassClock class_clock(0); // (trust1, KYTY_RENDER_STATS: us per call by class)
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
	if ((FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed)) || packet.size() < 10 || ((packet[4] >> 30u) & 1u) != 0 || packet[5] == 0 ||
	    packet[5] > max_draws || packet[8] != sizeof(DrawIndexedIndirectArgs) || (packet[9] & ~0x20u) != 2u ||
	    m_index_type_and_size > 1 || !m_index_base_addr || !m_index_buffer_size || !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList)
		return skip();
	if (DecalTakesNormalPath(m_ctx)) return skip();
	const uint32_t                  count = packet[5];
	std::array<uint64_t, max_draws> commands; // the first `count` are written, only they are read
	for (uint32_t i = 0; i < count; ++i)
		commands[i] = m_draw_indirect_args_base_addr + packet[1] + uint64_t {i} * sizeof(DrawIndexedIndirectArgs);
	CheckBuffer();
	const uint64_t element    = m_index_type_and_size == 0 ? 2 : 4;
	const auto     index_type = m_index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	if (Parpadeo::On() && Spec::Current() == nullptr) // (parpadeo2)
		ParpadeoNoteDraws({commands.data(), count}, uint64_t {m_index_buffer_size} * element, Parpadeo::PathMulti);
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
	TrustedDepth::ClassClock class_clock(1); // (trust1, KYTY_RENDER_STATS: us per call by class)
	LiveTrace::MarkAfter gpu_mark {[&] { return GetScheduler().Current().RawHandle(); }, LiveTrace::MarkNativeXpr,
	                               m_sh_ctx.GetPs().ps_regs.data_addr};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	auto& executor = m_renderer.GetRenderExecutor();
	// A draw never tried leaves the cached draw-state key behind: the next clean draw would take it.
	const auto skip = [&] {
		executor.NativeXprForgetState();
		return 0u;
	};
	if ((FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed)) || XprCapture::Enabled() || m_index_type_and_size > 1 ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList)
		return skip();
	if (DecalTakesNormalPath(m_ctx)) return skip();
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
	TrustedDepth::ClassClock class_clock(2); // (trust1, KYTY_RENDER_STATS: us per call by class)
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	using Prospero::PrimitiveType;
	// What CpOpDrawIndexAuto accepts and draws (no vertices: nothing), of a type drawn as it is.
	const auto primitive = m_ucfg.GetPrimType();
	if (packet.size() < 3 || packet[0] != 0xc0012d00u || (packet[2] & ~0x22u) != 0 || packet[1] == 0 ||
	    (FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed)) || XprCapture::Enabled() ||
	    (primitive != PrimitiveType::kTriList && primitive != PrimitiveType::kTriStrip))
		return 0;
	if (DecalTakesNormalPath(m_ctx)) {
		m_renderer.GetRenderExecutor().NativeXprForgetState();
		return 0;
	}
	CheckBuffer();
	// KYTY_RENDER_ASPECT: the HUD at its sides of the screen, unstretched (hud-anchor.h); a draw over several of its
	// areas goes the normal way, drawn once for each.
	const auto hud = HudAnchor::Split(HudAnchor::Classify(m_ctx, m_sh_ctx.GetVs(), m_ucfg.GetIndexOffset(), packet[1]));
	if (hud.Several()) return 0u;
	std::optional<HudAnchor::Scope>        anchor;
	std::optional<HudAnchor::StretchScope> widen;
	if (hud.Single()) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), static_cast<uint32_t>(std::countr_zero(hud.areas)), false);
	} else if (hud.marker) {
		anchor.emplace(m_ctx, m_sh_ctx.GetPs(), HudAnchor::MarkerArea, false);
	} else {
		HudAnchor::RestoreMask(m_sh_ctx.GetPs());
		if (HudAnchor::Stretched(m_ctx)) widen.emplace(m_ctx, m_sh_ctx.GetPs());
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

uint32_t CommandProcessor::TryDrawIndirectRun(std::span<const uint32_t> packets) {
	constexpr uint32_t max_draws = 64;
	if ((FrameCapture::Active() || SkipShader::g_any.load(std::memory_order_relaxed))) return 0; // one record per draw while a frame is recorded
	if (m_draw_run_skip) {
		--m_draw_run_skip;
		return 0;
	}
	if (packets.size() < 10 || m_index_type_and_size > 1 || !m_index_base_addr ||
	    !m_draw_indirect_args_base_addr ||
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList)
		return 0;
	uint32_t count = 1, end = 5;
	// (port2) Each draw's packet: runs go on over the empty markers of KYTY_PORT2 (not with render1, whose observers
	// count 5 dwords a draw).
	std::array<uint32_t, max_draws> at_dw {};
	const bool                      port2_runs = Port2::Mode() != 0 && !Render1::On();
	while (count < max_draws) {
		uint32_t pos = end;
		if (port2_runs && pos + Port2::MarkerDw + 5u <= packets.size() &&
		    Port2::IsEmptyMarker(packets.data() + pos, packets.size() - pos))
			pos += Port2::MarkerDw;
		if (pos + 5u > packets.size()) break;
		const auto* next = packets.data() + pos;
		if (next[0] != 0xc0032500u || next[2] != packets[2] || next[3] != packets[3] ||
		    next[4] != packets[4])
			break;
		at_dw[count++] = pos;
		end            = pos + 5u;
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
		const uint64_t address = m_draw_indirect_args_base_addr + packets[at_dw[i] + 1u];
		argument_addresses[i]  = address;
		if (address < m_draw_indirect_args_base_addr || (address & 3u) ||
		    !resources.IsMapped(address, sizeof(DrawIndexedIndirectArgs)) ||
		    !LibKernel::Memory::SyncGpuCleanBacking(address, sizeof(DrawIndexedIndirectArgs)))
			return 0;
		DrawIndexedIndirectArgs args {};
		std::memcpy(&args, reinterpret_cast<const void*>(address), sizeof(args));
		if (Parpadeo::On() && Spec::Current() == nullptr) // (parpadeo2: the arguments read on the CPU)
			Parpadeo::OnCpuArgs(CurrentBuffer().GetContext().FrameNumber(), m_draw_indirect_args_base_addr, address,
			                    args.index_count_per_instance);
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
	Port2::t_run_draws = count; // (port2: the caller completes `count` draws)
	return end;
}

// (port2, KYTY_PORT2; local/port2.h) A marker of the emulator's Agc (written after a draw / dispatch; sealed at the next
// one on the same command buffer with the flow of what lies between). Returns the dwords to pass over.
uint32_t CommandProcessor::Port2Marker(const uint32_t* packet, uint32_t remaining_dw, bool can_apply) {
	using Port2::MarkerDw;
	if (packet[Port2::SpanWord] == 0) { // (not sealed: a plain NOP)
		Port2::g_gpu_empty.fetch_add(1, std::memory_order_relaxed);
		return MarkerDw;
	}
	if (!can_apply) {
		Port2::g_gpu_fall[Port2::GOff].fetch_add(1, std::memory_order_relaxed);
		return MarkerDw;
	}
	uint32_t      span = 0;
	Port2::GFall  why  = Port2::GFalls;
	const auto*   rec  = Port2::Find(packet, remaining_dw, span, why);
	if (rec == nullptr) {
		Port2::g_gpu_fall[why].fetch_add(1, std::memory_order_relaxed);
		return MarkerDw;
	}
	const uint32_t* const begin = packet + MarkerDw;
	const auto* const     ops   = reinterpret_cast<const uint32_t*>(rec + 1);
	const uint32_t        words = rec->words;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	const bool observe = kyty_local_native_xpr_mode.load(std::memory_order_relaxed) != 0 &&
	                     (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 16u) == 0;
#else
	const bool observe = false;
#endif
	// The flow, in order, with the walk's handlers. `length`: the span dwords the walk would have taken (a handler that
	// takes another length than its header's: the walk goes on from where it would be); `types`: operations by type.
	uint64_t   types[Port2::Types] {};
	const auto apply = [&](bool with_observer, uint32_t& length) {
		length = span;
		for (uint32_t w = 0; w < words;) {
			const uint32_t word = ops[w];
			switch (Port2::KindOf(word)) {
				case Port2::OUd: {
					const uint32_t stage = Port2::UdStage(word), slot = Port2::UdSlot(word), n = Port2::UdCount(word);
					const uint32_t* v      = ops + w + 1;
					const auto      marker = m_user_data_marker;
					switch (stage) {
						case Port2::SPs:
							for (uint32_t i = 0; i < n; ++i) m_sh_ctx.SetPsUserSgpr(slot + i, v[i], marker);
							break;
						case Port2::SGs:
							for (uint32_t i = 0; i < n; ++i) m_sh_ctx.SetGsUserSgpr(slot + i, v[i], marker);
							break;
						case Port2::SHs:
							for (uint32_t i = 0; i < n; ++i) m_sh_ctx.SetHsUserSgpr(slot + i, v[i], marker);
							break;
						default:
							for (uint32_t i = 0; i < n; ++i) m_sh_ctx.SetCsUserSgpr(slot + i, v[i], marker);
							break;
					}
					m_user_data_marker = HW::UserSgprType::Unknown; // (the user-data handlers clear it after the packet)
#ifdef KYTY_LOCAL_VULKAN_RECORDING
					if (with_observer)
						for (uint32_t i = 0; i < n; ++i) DrawStateObserver::WriteRegister(1, Port2::UdBase[stage] + slot + i, v[i]);
#endif
					++types[Port2::TUd];
					w += 1u + n;
					break;
				}
				case Port2::OMk: {
					const uint32_t id = Port2::MkId(word);
					if (id == 0x4u) m_user_data_marker = HW::UserSgprType::Vsharp;
					else if (id == 0xdu) m_user_data_marker = HW::UserSgprType::Region;
					++types[Port2::TMk];
					w += 1u;
					break;
				}
				case Port2::OGen: {
					const uint32_t        len  = Port2::GenLen(word);
					const uint32_t* const copy = ops + w + 1;
					const uint32_t        op   = (copy[0] >> 8u) & 0xffu;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
					if (with_observer) DrawStateObserver::ObservePacket(op, copy, len, false);
#endif
					const uint32_t taken = g_cp_op_func[op](*this, copy[0] & ~1u, copy + 1, len, len) + 1u;
					m_chunk_last_opcode  = op;
					++types[Port2::GenType(word) < Port2::Types ? Port2::GenType(word) : Port2::TSh];
					if (taken != len) {
						length = Port2::OffOf(word) + taken;
						return false;
					}
					w += 1u + len;
					break;
				}
				default: length = 0; return false; // (never written)
			}
		}
		return true;
	};
	m_draw_run_skip = 0; // (as the walk does at any packet that is not an indirect draw)
	if (Port2::Mode() == 2) {
		// Verification: the flow on a copy of the registers, then the span's own PM4 through the handlers; compared field
		// by field, the PM4's state kept.
		struct Snap {
			HW::Context      ctx;
			HW::Shader       sh;
			HW::UserConfig   uc;
			HW::UserSgprType marker = HW::UserSgprType::Unknown;
			uint32_t         index_type = 0, index_size = 0, instances = 0;
			uint64_t         index_base = 0, draw_args = 0, dispatch_args = 0;
		};
		thread_local std::unique_ptr<Snap> before_ptr, flow_ptr;
		if (!before_ptr) before_ptr = std::make_unique<Snap>();
		if (!flow_ptr) flow_ptr = std::make_unique<Snap>();
		Snap&      before = *before_ptr;
		Snap&      flow   = *flow_ptr;
		const auto take   = [&](Snap& s) {
            s.ctx           = m_ctx;
            s.sh            = m_sh_ctx;
            s.uc            = m_ucfg;
            s.marker        = m_user_data_marker;
            s.index_type    = m_index_type_and_size;
            s.index_size    = m_index_buffer_size;
            s.instances     = m_num_instances;
            s.index_base    = m_index_base_addr;
            s.draw_args     = m_draw_indirect_args_base_addr;
            s.dispatch_args = m_dispatch_indirect_args_base_addr;
		};
		take(before);
		uint32_t flow_length = 0;
		(void)apply(false, flow_length);
		take(flow);
		m_ctx                             = before.ctx;
		m_sh_ctx                          = before.sh;
		m_ucfg                            = before.uc;
		m_user_data_marker                = before.marker;
		m_index_type_and_size             = before.index_type;
		m_index_buffer_size               = before.index_size;
		m_num_instances                   = before.instances;
		m_index_base_addr                 = before.index_base;
		m_draw_indirect_args_base_addr    = before.draw_args;
		m_dispatch_indirect_args_base_addr = before.dispatch_args;
		// The span's PM4, as the walk takes it.
		uint32_t at = 0;
		while (at < span) {
			const uint32_t header = begin[at];
			if (header == 0x80000000u) {
				++at;
				continue;
			}
			const uint32_t op   = (header >> 8u) & 0xffu;
			const uint32_t rest = remaining_dw - MarkerDw - at;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
			if (observe) DrawStateObserver::ObservePacket(op, begin + at, std::min(KYTY_PM4_LEN(header), rest), false);
#endif
			EXIT_IF(g_cp_op_func[op] == nullptr);
			at += g_cp_op_func[op](*this, header & ~1u, begin + at + 1, rest, rest) + 1u;
			m_chunk_last_opcode = op;
		}
		// Field by field.
		bool       field[Port2::Fields] {};
		const auto& ps = m_sh_ctx.GetPs();
		const auto& vs = m_sh_ctx.GetVs();
		const auto& cs = m_sh_ctx.GetCs();
		const auto& fps = flow.sh.GetPs();
		const auto& fvs = flow.sh.GetVs();
		const auto& fcs = flow.sh.GetCs();
		field[Port2::VCx]    = !(flow.ctx == m_ctx);
		field[Port2::VUc]    = !(flow.uc == m_ucfg);
		field[Port2::VUdPs]  = !(fps.ps_user_sgpr == ps.ps_user_sgpr);
		field[Port2::VUdGs]  = !(fvs.gs_user_sgpr == vs.gs_user_sgpr);
		field[Port2::VUdHs]  = !(fvs.hs_user_sgpr == vs.hs_user_sgpr);
		field[Port2::VUdCs]  = !(fcs.cs_user_sgpr == cs.cs_user_sgpr);
		field[Port2::VShOther] = !(fps.ps_regs == ps.ps_regs) || !(fvs.es_regs == vs.es_regs) || !(fvs.ls_regs == vs.ls_regs) ||
		                         !(fvs.hs_regs == vs.hs_regs) || !(fvs.gs_regs == vs.gs_regs) ||
		                         fvs.start_instance_user_sgpr != vs.start_instance_user_sgpr || !(fcs.cs_regs == cs.cs_regs);
		if (!(flow.sh == m_sh_ctx) && !field[Port2::VUdPs] && !field[Port2::VUdGs] && !field[Port2::VUdHs] && !field[Port2::VUdCs])
			field[Port2::VShOther] = true;
		field[Port2::VMarker] = flow.marker != m_user_data_marker;
		field[Port2::VIndex]  = flow.index_type != m_index_type_and_size || flow.index_size != m_index_buffer_size ||
		                       flow.instances != m_num_instances || flow.index_base != m_index_base_addr ||
		                       flow.draw_args != m_draw_indirect_args_base_addr ||
		                       flow.dispatch_args != m_dispatch_indirect_args_base_addr;
		field[Port2::VLength] = flow_length != at;
		// The record against the PM4 as it is now (a write into the span after the seal).
		for (uint32_t w = 0; w < words && !field[Port2::VChanged];) {
			const uint32_t word = ops[w], off = Port2::OffOf(word);
			switch (Port2::KindOf(word)) {
				case Port2::OUd: {
					const uint32_t n = Port2::UdCount(word);
					field[Port2::VChanged] = off + 2u + n > span ||
					                         begin[off + 1u] != Port2::UdBase[Port2::UdStage(word)] + Port2::UdSlot(word) ||
					                         std::memcmp(begin + off + 2u, ops + w + 1, 4ull * n) != 0;
					w += 1u + n;
					break;
				}
				case Port2::OMk:
					field[Port2::VChanged] = off + 2u > span || (begin[off + 1u] & 0xfffu) != Port2::MkId(word);
					w += 1u;
					break;
				default: {
					const uint32_t len = Port2::GenLen(word);
					field[Port2::VChanged] = off + len > span || std::memcmp(begin + off, ops + w + 1, 4ull * len) != 0;
					w += 1u + len;
					break;
				}
			}
		}
		bool any = false;
		for (uint32_t f = 0; f < Port2::Fields; ++f)
			if (field[f]) {
				any = true;
				Port2::g_field[f].fetch_add(1, std::memory_order_relaxed);
			}
		Port2::g_verified.fetch_add(1, std::memory_order_relaxed);
		if (any) {
			const auto n = Port2::g_discrepancias.fetch_add(1, std::memory_order_relaxed);
			if (n < 8) {
				std::printf("[port2] DISCREPANCIA %llu: marcador 0x%llx, hueco %u dw, %u paquetes; campos:",
				            static_cast<unsigned long long>(n + 1), static_cast<unsigned long long>(reinterpret_cast<uint64_t>(packet)),
				            span, rec->packets);
				for (uint32_t f = 0; f < Port2::Fields; ++f)
					if (field[f]) std::printf(" %s", Port2::FieldNames[f]);
				// The first user data slot that differs.
				const HW::UserSgprInfo* pairs[4][2] = {{&fps.ps_user_sgpr, &ps.ps_user_sgpr}, {&fvs.gs_user_sgpr, &vs.gs_user_sgpr},
				                                       {&fvs.hs_user_sgpr, &vs.hs_user_sgpr}, {&fcs.cs_user_sgpr, &cs.cs_user_sgpr}};
				for (uint32_t s = 0; s < 4; ++s)
					for (int i = 0; i < HW::UserSgprInfo::SGPRS_MAX; ++i)
						if (pairs[s][0]->value[i] != pairs[s][1]->value[i] || pairs[s][0]->type[i] != pairs[s][1]->type[i]) {
							std::printf(" | etapa %u slot %d: flujo 0x%08x (tipo %d) pm4 0x%08x (tipo %d)", s, i, pairs[s][0]->value[i],
							            static_cast<int>(pairs[s][0]->type[i]), pairs[s][1]->value[i],
							            static_cast<int>(pairs[s][1]->type[i]));
							i = HW::UserSgprInfo::SGPRS_MAX;
							s = 4;
						}
				std::printf("\n");
				std::fflush(stdout);
			}
		}
		Port2::g_gpu_applied.fetch_add(1, std::memory_order_relaxed);
		Port2::g_gpu_packets.fetch_add(rec->packets, std::memory_order_relaxed);
		Port2::g_gpu_dw.fetch_add(span, std::memory_order_relaxed);
		for (uint32_t t = 0; t < Port2::Types; ++t)
			if (types[t] != 0) Port2::g_gpu_type[t].fetch_add(types[t], std::memory_order_relaxed);
		return MarkerDw + at; // (the PM4's own length)
	}
	static thread_local uint32_t tick = 0;
	const bool                   timed = (++tick & 15u) == 0;
	const uint64_t               t0    = timed ? __rdtsc() : 0;
	uint32_t                     length = span;
	if (!apply(observe, length)) {
		Port2::g_gpu_fall[Port2::GLength].fetch_add(1, std::memory_order_relaxed);
		if (length == 0 || length > remaining_dw - MarkerDw) length = span; // (never: register handlers take their header's)
	}
	// The record was written over meanwhile (16 MiB of newer records in the time of this apply: never expected).
	if (rec->abs != (packet[Port2::IdWord] | uint64_t {packet[Port2::IdWord + 1]} << 32u))
		Port2::g_gpu_fall[Port2::GRace].fetch_add(1, std::memory_order_relaxed);
	if (timed) {
		Port2::g_gpu_cycles.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		Port2::g_gpu_timed.fetch_add(1, std::memory_order_relaxed);
	}
	types[Port2::TNop] += rec->fillers;
	Port2::g_gpu_applied.fetch_add(1, std::memory_order_relaxed);
	Port2::g_gpu_packets.fetch_add(rec->packets, std::memory_order_relaxed);
	Port2::g_gpu_dw.fetch_add(span, std::memory_order_relaxed);
	for (uint32_t t = 0; t < Port2::Types; ++t)
		if (types[t] != 0) Port2::g_gpu_type[t].fetch_add(types[t], std::memory_order_relaxed);
	return MarkerDw + length;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);
	if (Parpadeo::On() && indexed && Spec::Current() == nullptr) { // (parpadeo2)
		const uint64_t at = m_draw_indirect_args_base_addr + data_offset;
		ParpadeoNoteDraws({&at, 1}, uint64_t {m_index_buffer_size} * (m_index_type_and_size == 0 ? 2u : 4u),
		                  Parpadeo::PathNormal);
	}
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
	if (Parpadeo::On() && Spec::Current() == nullptr) // (parpadeo2: the arguments read on the CPU)
		Parpadeo::OnCpuArgs(CurrentBuffer().GetContext().FrameNumber(), m_draw_indirect_args_base_addr,
		                    m_draw_indirect_args_base_addr + data_offset, args.index_count_per_instance);

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
	if (Parpadeo::On() && indexed && Spec::Current() == nullptr) { // (parpadeo2)
		std::array<uint64_t, 64> at {};
		const uint32_t           n = std::min<uint32_t>(draw_count, 64);
		for (uint32_t i = 0; i < n; ++i) at[i] = m_draw_indirect_args_base_addr + data_offset + uint64_t {i} * stride_in_bytes;
		ParpadeoNoteDraws({at.data(), n}, uint64_t {m_index_buffer_size} * (m_index_type_and_size == 0 ? 2u : 4u),
		                  Parpadeo::PathNormalMulti);
	}
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
		if (auto* record = Spec::t_record; record != nullptr && indirect_args == 0)
			m_renderer.GetComputeRenderExecutor().PredictDispatch(m_sh_ctx, thread_group_x, thread_group_y, thread_group_z,
			                                                      mode, *record);
		return;
	}

	uint32_t frame_num = 0;
	// uint32_t local_x   = 1;
	// uint32_t local_y   = 1;
	// uint32_t local_z   = 1;

	{
		CheckBuffer();
		frame_num = m_renderer.GetGpu().GetFrameNum();
		VgeoNativo::OnFrame(frame_num); // (w11)
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
		// local_x        = std::max(cs.num_thread_x, 1u);
		// local_y        = std::max(cs.num_thread_y, 1u);
		// local_z        = std::max(cs.num_thread_z, 1u);
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

		const uint64_t vgeo_t0 = VgeoNativo::t_in_target ? VgeoNativo::Now() : 0; // (w11b)
		CensoPm4::Mark(CensoPm4::KExec); // (w18)
		m_renderer.GetComputeRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                                     thread_group_y, thread_group_z, mode,
		                                                     indirect_args);
		if (vgeo_t0 != 0) VgeoNativo::NoteTargetDispatch(cs.data_addr, indirect_args != 0, VgeoNativo::Now() - vgeo_t0);
		CensoPm4::Mark(CensoPm4::KAfter); // (w18)
	}

	/*constexpr uint32_t DispatchInitiatorUseThreadDimensions = 1u << 5u;
	auto               group_count = [](uint32_t threads, uint32_t group_size) {
	    return (threads == 0
	                ? 0u
	                : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
	};

	auto groups_x = thread_group_x;
	auto groups_y = thread_group_y;
	auto groups_z = thread_group_z;
	if ((mode & DispatchInitiatorUseThreadDimensions) != 0) {
	    groups_x = group_count(thread_group_x, local_x);
	    groups_y = group_count(thread_group_y, local_y);
	    groups_z = group_count(thread_group_z, local_z);
	}

	const uint64_t invocations =
	    static_cast<uint64_t>(groups_x) * groups_y * groups_z * local_x * local_y * local_z;
	if (invocations != 0) {
	    BufferFlushAndWait();
	}*/
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
	// (w11, KYTY_NATIVO_VGEO) The PreRenderJob shaders: the GPU reads their GPU-written arguments, never the CPU.
	if ((mode & 0x20u) == 0 && (args_addr & 3u) == 0 && VgeoNativo::Fast(m_sh_ctx.GetCs().cs_regs.data_addr)) {
		VgeoNativo::g_args_skipped.fetch_add(1, std::memory_order_relaxed);
		DispatchDirect(1, 1, 1, mode, args_addr);
		return;
	}
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
	// KYTY_RENDER_ASPECT: the HUD at its sides of the screen, unstretched (hud-anchor.h).
	const auto hud = HudAnchor::Split(HudAnchor::Classify(m_ctx, m_sh_ctx.GetVs(), m_ucfg.GetIndexOffset(), args.vertex_count));
	if (hud.areas == 0 || hud.Single()) {
		std::optional<HudAnchor::Scope>        anchor;
		std::optional<HudAnchor::StretchScope> widen;
		if (hud.Single()) {
			anchor.emplace(m_ctx, m_sh_ctx.GetPs(), static_cast<uint32_t>(std::countr_zero(hud.areas)), false);
		} else if (hud.marker) {
			anchor.emplace(m_ctx, m_sh_ctx.GetPs(), HudAnchor::MarkerArea, false);
		} else {
			HudAnchor::RestoreMask(m_sh_ctx.GetPs());
			if (HudAnchor::Stretched(m_ctx)) widen.emplace(m_ctx, m_sh_ctx.GetPs());
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

// (parpadeo7) KYTY_PARPADEO7=1 (local/parpadeo7.h): a label written with 1 as data. When it is an XPR culling label of a
// view the game's reads taught (0xa1f290's hook), the chunk bits it hands to the game are read back from now on: one
// command queued for the GPU thread, run between packets (this packet records nothing for it). Only guarded reads of
// the view and its frame table here.
static void PrefetchXprBits(uint64_t label, GpuResourceManager& resources) {
	thread_local std::vector<std::pair<uint64_t, uint64_t>> ranges;
	const auto read = [](uint64_t address, uint64_t* out) {
		return address >= 0x10000 && Common::GuardedCopy(out, reinterpret_cast<const void*>(address), sizeof(*out)) == 0;
	};
	if (!Parpadeo7::MatchLabel(label, read, ranges)) return;
	Parpadeo7::JoinRanges(ranges);
	Parpadeo7::g_c.spans.fetch_add(ranges.size(), std::memory_order_relaxed);
	if (g_gpu_state == nullptr || ranges.empty()) return;
	const int64_t label_ns = Parpadeo7::NowNs();
	(void)g_gpu_state->PostCommand([&resources, spans = ranges, label_ns] { resources.PrefetchGuestReads(spans, label_ns); });
}

// (juego3) KYTY_JUEGO3's ON windows (local/juego3.h): the ranges the learned compute shaders wrote since the last label,
// read back from this label on - the same command parpadeo7 queues (run by the GPU thread between packets, never inside
// this one; the copy right behind the work translated so far, a helper copies the bytes in).
static void Juego3PrefetchAtLabel(GpuResourceManager& resources) {
	thread_local std::vector<std::pair<uint64_t, uint64_t>> ranges;
	if (g_gpu_state == nullptr || !Juego3::TakePending(ranges)) return;
	const int64_t label_ns = Parpadeo7::NowNs();
	(void)g_gpu_state->PostCommand([&resources, spans = ranges, label_ns] { resources.PrefetchGuestReads(spans, label_ns); });
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
	// KYTY_NATIVE_MEMORY: the label, the event and the interrupt below tell the game that the work before is done; all it
	// wrote goes to guest memory first (GpuResourceManager::NativeFlushGpuWrites). Nothing otherwise.
	if (!m_state_only) GetGpuResources().NativeFlushGpuWrites();
	// (parpadeo7) Before the label is written: the readbacks are queued ahead of any the game's reads after it ask.
	if (Parpadeo7::On() && !m_state_only && event_write_source == 0x02 && static_cast<uint64_t>(value) == 1 &&
	    Spec::Current() == nullptr)
		PrefetchXprBits(reinterpret_cast<uint64_t>(dst_gpu_addr), GetGpuResources());
	// (juego3) After the work that wrote them, before the game sees the label.
	if (Juego3::FixNow() && !m_state_only && Spec::Current() == nullptr) Juego3PrefetchAtLabel(GetGpuResources());

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

	// (p7, C5) A data label: written when the GPU has run the work before it (DeferLabelWrite). Its interrupt, if any, is
	// a priority operation queued after it (Sync::TriggerEopEventAtEndOfPipe): the game sees the label first, as before.
	const bool label_at_completion = P7::LabelsAtCompletion() && !m_state_only && event_write_source == 0x02;
	const auto write_label         = [&](void* dst, const void* src, uint32_t size) {
		if (label_at_completion) {
			if (DeferLabelWrite(GetScheduler(), dst, src, size)) {
				P7::Add(P7::LabelsDeferred);
				return;
			}
			P7::Add(P7::LabelsImmediate);
		}
		WriteGuestMemory(dst, src, size);
	};

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		write_label(dst, &data, sizeof(data));

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
					write_label(dst, &value, sizeof(value));

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
		if (work == last_work && kyty_local_global_barrier_dedupe.load(std::memory_order_relaxed) != 0) {
			if (P7::Tracking()) P7::Add(P7::GlobalDedupeDrops); // (p7b: counted for the [p7] line)
			return;
		}
		last_work = work + 1; // this barrier's own recorded call
	}
#endif

	RenderLockGuard lock(m_renderer.GetMutex());

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
	// (KYTY_NATIVE_MEMORY: the event tells the game the work before is done: GpuResourceManager::NativeFlushGpuWrites.)
	if (!m_state_only) GetGpuResources().NativeFlushGpuWrites();

	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::BreakComputeChain() {
	// Guest synchronization may be implemented without a host command.
	// It still ends the compatibility profile's independent dispatch group.
	if (!GetScheduler().Active()) return;
	RenderLockGuard lock(m_renderer.GetMutex());
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
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
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
	if (!m_state_only && DsCensoEscena::Mode() != 0) DsCensoEscena::OnFlip(); // (F5 pase 3: [censo-escena] cada 5 s)
	NativeFlipSync(GetGpuResources(), m_state_only);
	if (!m_state_only) // (f5, KYTY_NATIVE_RENDER=map; f5b: the per-pass ceiling's windows read the GPU thread's times)
		NativeRender::Flip(LiveControl::g_render_idle_ns, LiveCensus::g_kind_cycles[LiveCensus::Submission]);
	if (!m_state_only) PathTrace::Flip(); // (pt1, BB_PATHTRACING: its frame, calibrations and 5 s line; nothing when off)
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
	NativeFlipSync(GetGpuResources(), m_state_only);

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
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
	NativeFlipSync(GetGpuResources(), m_state_only);

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
	NativeFlipSync(GetGpuResources(), m_state_only);
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
