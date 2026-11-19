#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>

#include <queue>

#include <thread>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

namespace StallWatch {
class Timeline;
enum class Transfer : uint8_t;
} // namespace StallWatch

class PageManager;

// Round 167 (copies:lean): the batch of buffer copies the scheduler's command buffer is recording
// (CommandRecorder::copyBufferBatched), kept in the order of its commands. A batch ends with the
// command buffer (Submit).
struct CopyBatchState {
	struct Written {
		vk::Buffer buffer;
		uint64_t   begin = 0;
		uint64_t   end   = 0;
	};
	// Nothing but batched copies since a barrier that made them wait for all the GPU work before.
	bool fenced = false;
	// Batched copies that no barrier has made visible to the commands after them yet.
	bool after_pending = false;
	// Their destinations since the last barrier that ordered them.
	std::vector<Written> written;
};

// Round 172: the pipelines and the dynamic state the command buffer being recorded holds, as the
// recorder last set them (CommandRecorder), so a call that sets them to what they are can be left
// out. Kept in the order of the commands; reset with each command buffer.
struct DynamicStateShadow {
	enum Kind : uint32_t {
		Viewports,
		Scissors,
		LineWidth,
		BlendConstants,
		CullMode,
		FrontFace,
		DepthBoundsTest,
		DepthBounds,
		DepthTest,
		DepthWrite,
		DepthCompare,
		DepthBiasEnable,
		DepthBias,
		StencilTest,
		StencilOpFront,
		StencilOpBack,
		StencilCompareFront,
		StencilCompareBack,
		StencilWriteFront,
		StencilWriteBack,
		StencilReferenceFront,
		StencilReferenceBack,
		ColorWrite,
		FeedbackLoop,
		Count
	};
	static constexpr uint32_t MaxBytes = 16u * sizeof(vk::Viewport);
	struct Entry {
		bool                            valid = false;
		uint32_t                        size  = 0;
		std::array<std::byte, MaxBytes> bytes {};
	};
	std::array<Entry, Count> entries {};
	vk::Pipeline             graphics_pipeline;
	vk::Pipeline             compute_pipeline;
	// The counts of the "Recorded state" line, kept here (the recording thread's) and added to
	// RecordedCommands' at each command buffer: an atomic add a call cost more than most calls.
	uint64_t calls           = 0;
	uint64_t unchanged       = 0;
	uint64_t binds           = 0;
	uint64_t binds_unchanged = 0;
	uint64_t left_out        = 0;

	// Whether the state holds `size` bytes of `data` already; it does after.
	bool Same(Kind kind, const void* data, uint32_t size) noexcept {
		auto& entry = entries[kind];
		if (entry.valid && entry.size == size && std::memcmp(entry.bytes.data(), data, size) == 0) {
			return true;
		}
		entry.valid = size <= MaxBytes;
		entry.size  = size;
		if (entry.valid) {
			std::memcpy(entry.bytes.data(), data, size);
		}
		return false;
	}
	void Invalidate(Kind kind) noexcept { entries[kind].valid = false; }
	// gt7-replay-1119 (KYTY_REPLAY): the graphics descriptor set 0 and the push constants a
	// draw last recorded (RenderExecutor::CommitBindings), flattened, for the next draw to leave
	// out the same again. Any other set-0 update or push constant call, an emulator pipeline
	// bind and a new command buffer forget them.
	bool                     replay_set_valid = false;
	vk::PipelineLayout       replay_set_layout;
	std::vector<uint64_t>    replay_set_words;
	bool                     replay_constants_valid = false;
	vk::PipelineLayout       replay_constants_layout;
	uint32_t                 replay_constants_stages = 0;
	std::array<uint32_t, 32> replay_constants {};
	void InvalidateReplay() noexcept {
		replay_set_valid       = false;
		replay_constants_valid = false;
	}
	void InvalidateGraphics() noexcept {
		for (auto& entry: entries) {
			entry.valid = false;
		}
		graphics_pipeline = nullptr;
		InvalidateReplay();
	}
	void InvalidateAll() noexcept {
		InvalidateGraphics();
		compute_pipeline = nullptr;
	}
};

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	// Diagnostics (renderer/stallWatch.h, KYTY_STALL_WATCH): the name its stall reports give it
	// ("render" unless set, before the first command buffer), and what the command buffer being
	// recorded holds. The scheduler's own thread only.
	void SetName(const char* name);
	void NoteDraw(uint64_t shader, bool indirect);
	void NoteDispatch(uint64_t shader, bool indirect, uint64_t groups);
	void NoteTransfer(StallWatch::Transfer kind, uint64_t bytes);
	// Any thread that waited `waited_ns` for this scheduler's `tick` (`known`: the last tick seen
	// done when the wait began): a stall report when it waited long (StallWatch).
	void OnWaited(uint64_t tick, uint64_t known, uint64_t waited_ns);
	// Runs in Flush, on the thread that flushes, before the command buffer is submitted, for what
	// belongs at its end (the buffer cache's readahead downloads); not re-entered. Null for none.
	using BeforeFlush = void (*)(void* user);
	void SetBeforeFlush(BeforeFlush hook, void* user) noexcept {
		m_before_flush      = hook;
		m_before_flush_user = user;
	}
	// Round 111: a command buffer is submitted only once the page protections deferred before
	// its submission are made (PageManager::DeferScope): the GPU's writes it holds are then seen
	// by the CPU's reads, and the bytes it uploads were read after their write protection.
	void SetSubmitBarrier(PageManager* pages) noexcept { m_submit_pages = pages; }

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Bounds how many no-interrupt RELEASE_MEM fence writes accumulate in one command buffer
	// before it is submitted, so consecutive fence updates that nothing is blocked waiting on
	// don't each pay a host vkQueueSubmit. Only safe for a RELEASE_MEM whose guest-visible write
	// already happened synchronously and that scheduled no interrupt callback -- see the call
	// site in pm4Handlers.cpp CpOpReleaseMem for the exact condition. Never waits itself.
	void           CompleteReleaseMemWrite();
	// Same idea, but for a RELEASE_MEM that DOES request a guest interrupt/event (a guest thread
	// may be waiting on it via an event queue) -- deferring the flush delays real event delivery,
	// so this uses a much smaller batch bound than CompleteReleaseMemWrite as a hedge, trading
	// some of the possible win for less added latency. Never waits itself.
	void           CompleteReleaseMemInterrupt();
	// Called after every DrawIndex/DrawAuto. Long chains of draws with no intervening
	// RELEASE_MEM/wait can otherwise sit fully recorded but unsubmitted for a long time, leaving
	// the GPU idle until something else finally forces a flush -- this periodically calls Flush()
	// (non-blocking: Submit() + BeginNext(), same as CompleteReleaseMemWrite) every
	// KYTY_DRAW_FLUSH_INTERVAL draws to keep the queue fed instead. Submitting while the GPU is
	// still executing earlier work is always legal (vkQueueSubmit never waits on prior submissions
	// completing), so this never blocks the CPU. Defaults to 16, validated against real gameplay --
	// override via KYTY_DRAW_FLUSH_INTERVAL (0 disables) if a different workload needs retuning:
	// too small reintroduces per-submit overhead, too large leaves the same idle bubbles this
	// exists to remove.
	void           CompleteDraw();
	// The same after every dispatch, every KYTY_DISPATCH_FLUSH_INTERVAL dispatches (default 16; 0
	// turns it off, as before round 69). Before, only draws sent command buffers on: a compute
	// queue's hundreds of dispatches in a row (561 at the start of a race frame, 7 ms of the GPU
	// thread) went to the GPU at once at the end, which then took 9 ms over them while every game
	// thread waited for their results.
	void           CompleteDispatch();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Runs the deferred operations whose ticks the GPU has finished, in tick order, each after the
	// priority callbacks of its tick and before (WaitPriorityOperations). With `wait` false (the
	// call at each draw and dispatch), one whose priority callbacks are still queued or running
	// stays for a later call instead of making the GPU thread wait for the priority thread.
	void                      PopPendingOperations(bool wait = true);
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	// As DeferOperation, also between guest command buffers (no active command buffer): runs
	// once the GPU has finished the current tick, so work submitted earlier, still running, can
	// keep using what the operation releases.
	void                      DeferRelease(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Hands later submissions to a dedicated queue thread, which submits them in tick order, so
	// vkQueueSubmit (and the queue lock that present also takes) leaves the recording thread.
	// Submit() then returns once the tick is allocated; host waits on a timeline value may precede
	// its signal operation. Enable before the first Submit().
	void EnableAsyncSubmit();
	[[nodiscard]] bool AsyncSubmit() const noexcept { return m_async_submit; }
	// A Vulkan recording thread (KYTY_RECORD_THREAD=0 turns it off; the pattern and the
	// CommandRecorder are IDXTRI's, d9958e28). The GPU thread queues the current command
	// buffer's Vulkan commands (CommandBuffer::Recorder(), Record()); the recording thread begins,
	// records, ends and submits the command buffers in order, so the driver's recording cost
	// leaves the GPU thread. Ticks are still allocated here, in order, at Submit(). Code that
	// records through the raw CommandBuffer::Handle() first waits for the thread to catch up
	// (DrainRecording), so it stays correct. Replaces the queue thread (EnableAsyncSubmit).
	// Enable before the first command buffer.
	void EnableRecordingThread();
	// The current command buffer's commands go through the recording thread.
	[[nodiscard]] bool Threaded() const noexcept { return m_threaded; }
	// Queues fn(vk::CommandBuffer) for the recording thread (threaded only). Captures must be
	// values; arrays go through Stash().
	template <typename F>
	void Record(F&& fn) {
		using Command = TypedCommand<std::decay_t<F>>;
		void* memory  = Allocate(sizeof(Command), alignof(Command));
		auto* command = new (memory) Command(std::forward<F>(fn));
		if (m_chunk->last != nullptr) {
			m_chunk->last->next = command;
		} else {
			m_chunk->first = command;
		}
		m_chunk->last = command;
	}
	// Copies count elements into memory that stays valid until the recording thread has run the
	// next command recorded after this call (threaded only).
	template <typename T>
	[[nodiscard]] const T* Stash(const T* data, size_t count) {
		if (data == nullptr || count == 0) {
			return nullptr;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
		std::memcpy(static_cast<void*>(copy), data, sizeof(T) * count);
		return copy;
	}
	// Waits until the recording thread has run every queued command, and returns the command
	// buffer it records into, for code that records directly.
	vk::CommandBuffer DrainRecording();

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Round 167: the recording command buffer's batch of copies (the recording side's thread).
	[[nodiscard]] CopyBatchState& CopyBatch() noexcept { return m_copy_batch; }
	// Round 172: the recording command buffer's pipelines and dynamic state.
	[[nodiscard]] DynamicStateShadow& StateShadow() noexcept { return m_state_shadow; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		// tick: the tick the command buffer will be submitted with.
		vk::CommandBuffer Commit(uint64_t tick);

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// A finished command buffer and everything its vkQueueSubmit needs, including the recording
	// state that a fatal submit report prints.
	struct SubmitJob {
		vk::CommandBuffer buffer = nullptr;
		SubmitInfo        submit;
		uint64_t          tick         = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
		uint32_t          debug_arg0   = 0;
		uint32_t          debug_arg1   = 0;
		uint32_t          debug_arg2   = 0;
		uint32_t          debug_arg3   = 0;
		uint64_t          debug_arg4   = 0;
	};

	struct RecordedCommand {
		virtual ~RecordedCommand()                  = default;
		virtual void     Execute(vk::CommandBuffer) = 0;
		RecordedCommand* next                       = nullptr;
	};
	template <typename F>
	struct TypedCommand final: RecordedCommand {
		explicit TypedCommand(F&& function): fn(std::move(function)) {}
		explicit TypedCommand(const F& function): fn(function) {}
		void Execute(vk::CommandBuffer command) override { fn(command); }
		F    fn;
	};
	struct Chunk {
		static constexpr size_t Size = 256 * 1024;
		// Allocations above this get a heap block of their own, owned by the chunk and freed when
		// the recording thread recycles it, so one command's stashed arrays (a texture upload's
		// copy regions, a large barrier batch) are not limited by Size.
		static constexpr size_t LargeThreshold = Size / 4;
		alignas(64) std::byte storage[Size];
		size_t           used  = 0;
		RecordedCommand* first = nullptr;
		RecordedCommand* last  = nullptr;
		std::vector<std::unique_ptr<std::byte[]>> large;

		[[nodiscard]] bool Empty() const noexcept {
			return first == nullptr && used == 0 && large.empty();
		}
	};
	void* Allocate(size_t size, size_t alignment);
	void  DispatchChunk();
	void  RecordingThread(std::stop_token stop);
	void  StopRecordingThread();
	void  ReportRecording();

	void BeginNext();
	void QueueSubmit(SubmitJob& job);
	void SubmitThread(std::stop_token stop);
	void StopSubmitThread();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	uint32_t                     m_recorded_release_mem_writes     = 0;
	uint32_t                     m_recorded_release_mem_interrupts = 0;
	uint32_t                     m_recorded_draws                  = 0;
	uint32_t                     m_recorded_dispatches             = 0;
	// m_operation_mutex held: a priority callback of `tick` or before is queued or running.
	[[nodiscard]] bool PriorityPendingThrough(uint64_t tick) const;

	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	bool                         m_async_submit         = false;
	std::deque<SubmitJob>        m_submit_jobs;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_available;
	std::jthread                 m_submit_thread;
	// Recording thread. m_threaded and m_chunk belong to the producer (the GPU thread); the queue,
	// free list and counters to m_record_mutex; m_worker_buffer to the recording thread (read by
	// the producer only after a drain).
	bool                                m_threaded = false;
	std::unique_ptr<Chunk>              m_chunk;
	std::mutex                          m_record_mutex;
	std::condition_variable_any         m_record_available;
	std::condition_variable             m_record_executed;
	std::deque<std::unique_ptr<Chunk>>  m_record_queue;
	std::vector<std::unique_ptr<Chunk>> m_free_chunks;
	uint64_t                            m_chunks_dispatched = 0;
	uint64_t                            m_chunks_executed   = 0;
	uint64_t                            m_drains            = 0;
	vk::CommandBuffer                   m_worker_buffer     = nullptr;
	uint64_t                            m_record_report_buffers = 0;
	uint64_t                            m_record_report_chunks  = 0;
	uint64_t                            m_record_report_drains  = 0;
	std::chrono::steady_clock::time_point m_record_report_time {};
	std::jthread                        m_record_thread;
	// KYTY_STALL_WATCH: the command buffers' timeline (null when off), and the tick of the one
	// being recorded.
	std::unique_ptr<StallWatch::Timeline> m_timeline;
	uint64_t                              m_open_tick = 0;
	BeforeFlush                           m_before_flush      = nullptr;
	void*                                 m_before_flush_user = nullptr;
	bool                                  m_in_before_flush   = false;
	PageManager*                          m_submit_pages      = nullptr;
	CopyBatchState                        m_copy_batch;
	DynamicStateShadow                    m_state_shadow;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
