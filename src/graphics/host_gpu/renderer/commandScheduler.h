#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include <source_location>
#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
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

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush(std::source_location where = std::source_location::current());
	void           Flush(SubmitInfo& submit, std::source_location where = std::source_location::current());
	void           FlushAndWait(std::source_location where = std::source_location::current());
	void           Finish(std::source_location where = std::source_location::current());
	CommandBuffer& BeginCommand();
	// KYTY_SHADER_GPU_TIME: a timestamp pair around one draw or dispatch, tagged with the shader
	// hash. Called on the thread that records the command buffer (CommandRecorder forwards it).
	void ShaderTimeBegin(vk::CommandBuffer command, uint64_t hash, bool compute);
	void ShaderTimeEnd(vk::CommandBuffer command);
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick, std::source_location where = std::source_location::current());
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Steady-clock time of the last vkQueueSubmit, in microseconds.
	[[nodiscard]] int64_t LastSubmitUs() const noexcept {
		return m_last_submit_us.load(std::memory_order_relaxed);
	}

	// KYTY_RECORD_THREAD (see commandScheduler.cpp). Enabled for the renderer's scheduler only.
	void EnableRecordingThread();
	// The current command buffer's Vulkan commands are executed by the recording thread.
	[[nodiscard]] bool Threaded() const noexcept { return m_threaded; }
	// Queues fn(vk::CommandBuffer) for the recording thread. Threaded mode only; captures must
	// be values (arrays through Stash), as fn runs later on another thread.
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
	// Copies count elements into memory that stays valid until the recording thread has executed
	// the next command recorded after this call. Threaded mode only.
	template <typename T>
	[[nodiscard]] const T* Stash(const T* data, size_t count) {
		if (data == nullptr || count == 0) {
			return nullptr;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
		std::memcpy(static_cast<void*>(copy), data, sizeof(T) * count);
		return copy;
	}
	// Waits until the recording thread has executed every queued command and returns the command
	// buffer it records into, for code that records directly. Threaded mode only.
	vk::CommandBuffer DrainRecording();

private:
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
		alignas(64) std::byte storage[Size];
		size_t           used  = 0;
		RecordedCommand* first = nullptr;
		RecordedCommand* last  = nullptr;
	};
	struct SubmitDebug;
	void*    Allocate(size_t size, size_t alignment);
	void     DispatchChunk();
	void     RecordingThread(std::stop_token stop);
	void     StopRecordingThread();
	void     BeginThreadedCommand();
	uint64_t SubmitThreaded(SubmitInfo submit, const SubmitDebug& debug);
	void     QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
	                     const SubmitDebug& debug);

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

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	// KYTY_GPU_TIME (see commandScheduler.cpp).
	struct GpuTimer;
	std::unique_ptr<GpuTimer> m_gpu_timer;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	std::atomic<int64_t>         m_last_submit_us {0};

	// Recording thread state. m_chunk and m_threaded belong to the recording side's producer (the
	// thread that owns the scheduler); the queue, the free list and the counters are guarded by
	// m_record_mutex; m_worker_buffer belongs to the recording thread (read after a drain).
	bool                                m_threaded = false;
	std::unique_ptr<Chunk>              m_chunk;
	std::mutex                          m_record_mutex;
	std::condition_variable_any         m_record_available;
	std::condition_variable             m_record_executed;
	std::deque<std::unique_ptr<Chunk>>  m_record_queue;
	std::vector<std::unique_ptr<Chunk>> m_free_chunks;
	uint64_t                            m_chunks_dispatched = 0;
	uint64_t                            m_chunks_executed   = 0;
	vk::CommandBuffer                   m_worker_buffer     = nullptr;
	struct RecordStats {
		uint64_t                              buffers = 0;
		uint64_t                              chunks  = 0;
		uint64_t                              drains  = 0;
		std::chrono::steady_clock::time_point report  = std::chrono::steady_clock::now();
	};
	RecordStats  m_record_stats;
	std::jthread m_record_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
