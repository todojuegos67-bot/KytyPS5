#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/packetStream.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/cmdRecorder.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <thread>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

class CommandScheduler {
public:
	// record_thread: KYTY_RECORD_THREAD applies to this scheduler (the GPU thread's; the present scheduler stays direct).
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool record_thread = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Runs at the end of each command buffer, before it is submitted, with its tick still current: the last
	// commands it records run after all others (BufferCache's pre-flush). It must not submit or wait.
	void SetEndHook(std::function<void()> hook) { m_end_hook = std::move(hook); }
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Waits until the submit of `tick` has gone to the queue (the recording thread sends threaded ones).
	void                      WaitSent(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// The tick of the newest guest-memory completion queued or running, 0 if none.
	[[nodiscard]] uint64_t    LastPriorityTick();
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

	// Sparse binds of texture levels (Image), on the GPU thread. A bind goes to the queue before the
	// next submit, which waits for it. An unbind goes after it, behind all work recorded so far, and
	// its memory is freed once it has run. Each returns the batch by which it will have run.
	// `deferred`: a level the image does not show yet (a grow, TextureCache::HoldLevels); it goes
	// out with SendSparseGrows, and no submit waits for it until RequireSparse.
	uint64_t BindSparse(vk::Image image, const vk::SparseImageMemoryBind& level, bool deferred = false);
	// Sends the deferred binds now (to the bind thread where there is one); their batch value.
	uint64_t SendSparseGrows();
	// The bind thread has sent the batch: a submit may wait for it without waiting for the thread.
	[[nodiscard]] bool SparseSent(uint64_t batch) const { return m_sparse_sent.load() >= batch; }
	// Until the bind thread has sent the batch. It takes no lock the GPU thread holds, and waits only for work already submitted.
	void WaitSparseSent(uint64_t batch) {
		for (auto sent = m_sparse_sent.load(); sent < batch; sent = m_sparse_sent.load()) m_sparse_sent.wait(sent);
	}
	// The next submit waits for this batch (it uploads to levels the batch binds).
	void RequireSparse(uint64_t batch) { m_sparse_required = std::max(m_sparse_required, batch); }
	uint64_t BindSparseTail(vk::Image image, const vk::SparseMemoryBind& tail);
	uint64_t BindSparseBuffer(vk::Buffer buffer, const vk::SparseMemoryBind& page); // out before the next submit
	uint64_t UnbindSparse(vk::Image image, const vk::SparseImageMemoryBind& level, VmaAllocation memory);
	// The image is about to be destroyed: none of its binds, the last in `batch`, may still wait or run.
	void ForgetSparse(vk::Image image, uint64_t batch);

	// The recording thread (on by default, KYTY_RECORD_THREAD=0 off; record-thread design H1). The GPU thread decides every command
	// and keeps every tick, as before; the recording thread makes the Vulkan calls: it begins the command buffers,
	// records what Recorder() queued, ends them and submits them, in the same order. Code that records through a
	// raw CommandBuffer::Handle() waits for it to catch up first (DrainRecording). Enable before the first
	// command buffer. verify (KYTY_RECORD_VERIFY=1): every packet is hashed when queued and again when recorded.
	// on: whether the command buffers use it from the start.
	void EnableRecordingThread(bool verify, bool on = true);
	[[nodiscard]] bool Threaded() const noexcept { return m_threaded.load(std::memory_order_relaxed); }
	// GPU thread, once the recording thread is enabled: on or off from the next command buffer on. Off first waits until
	// it has recorded and submitted everything queued, so no command buffer and no heap write spans both ways.
	void RequestRecordingThread(bool on) noexcept { m_threaded_wanted = on; }
	// The current command buffer's recorder.
	[[nodiscard]] Cmd Recorder();
	// Waits until the recording thread has recorded everything queued; its command buffer, for direct recording.
	vk::CommandBuffer DrainRecording(const char* file, uint32_t line);
	// The calling thread queues from now on: the GPU thread at a submission (the constructing thread set up and began
	// the first command buffer), the owner once the GPU thread has ended.
	void AdoptRecording() {
		if (m_stream != nullptr) m_stream->Adopt();
	}

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		// tick: the tick the command buffer is submitted with.
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

	struct SparseBind {
		vk::Image                 image;
		vk::SparseImageMemoryBind level;
		vk::SparseMemoryBind      tail;
		bool                      is_tail = false;
		vk::Buffer                buffer  = nullptr; // a sparse buffer's page (tail is its bind)
		VmaAllocation             unbound = nullptr; // an unbind's memory
	};
	struct SparseBatch {
		std::vector<SparseBind> binds;
		vk::Semaphore           wait       = nullptr;
		uint64_t                wait_value = 0;
		uint64_t                previous   = 0; // the batch before, 0: none
		uint64_t                signal     = 0;
	};

	// A submit the GPU thread queued for the recording thread: the tick and semaphores it chose, and the sparse
	// batches that go to the queue right before and after it, their values given in order on the GPU thread.
	struct SubmitJob {
		SubmitInfo                 submit;
		uint64_t                   tick            = 0;
		uint64_t                   sparse_required = 0;
		std::unique_ptr<SparseBatch> binds;
		std::unique_ptr<SparseBatch> unbinds;
		uint32_t                   debug_op        = 0;
		uint64_t                   debug_submit_id = 0;
		uint32_t                   debug_arg0 = 0, debug_arg1 = 0, debug_arg2 = 0, debug_arg3 = 0;
		uint64_t                   debug_arg4      = 0;
	};

	void BeginNext();
	uint64_t SubmitThreaded(SubmitInfo& submit);
	void     SwitchRecording();
	void     ReportRecording();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	// Caller holds the queue mutex. Returns the batch's value of m_sparse.
	uint64_t SendSparse(std::vector<SparseBind>& binds, vk::Semaphore wait, uint64_t wait_value);
	// SendSparse in two halves: the batch with its value (GPU thread), and sending it (the caller holds the queue
	// mutex where there is no bind thread).
	SparseBatch MakeSparse(std::vector<SparseBind>& binds, vk::Semaphore wait, uint64_t wait_value);
	void        DispatchSparse(SparseBatch&& batch);
	void     FreeSparse();
	// With a queue of their own (GraphicContext::sparse_queue), batches are sent by a thread.
	void     SparseThread(std::stop_token stop);
	void     RunSparse(vk::Queue queue, const SparseBatch& batch);

	MasterSemaphore              m_master;
	MasterSemaphore              m_sparse; // one value a sparse batch, each batch after the last
	std::vector<SparseBind>      m_sparse_binds;
	std::vector<SparseBind>      m_sparse_grows;
	std::vector<SparseBind>      m_sparse_unbinds;
	std::atomic<uint64_t>        m_sparse_sent {0};
	uint64_t                     m_sparse_required = 0;
	std::vector<std::pair<uint64_t, VmaAllocation>> m_sparse_retired; // freed once that batch ran
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::function<void()>        m_end_hook;
	std::queue<PendingOperation> m_pending_operations;
	std::atomic_uint32_t         m_pending_count {0}; // of m_pending_operations, read without the lock
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	std::mutex                   m_sparse_mutex;
	std::condition_variable_any  m_sparse_available;
	std::deque<SparseBatch>      m_sparse_batches;
	std::jthread                 m_sparse_thread; // stops and sends what is left before the rest goes
	// Recording thread (Threaded()): m_back belongs to it, the rest to the GPU thread.
	CmdBack                      m_back;
	std::atomic<bool>            m_threaded {false}; // the current command buffer goes through m_stream
	std::atomic<uint64_t>        m_sent_tick {0};    // the last tick whose submit went to the queue
	bool                         m_threaded_wanted = false;
	uint64_t                     m_recording_switches = 0;
	std::deque<uint64_t>         m_submit_marks; // the packets queued up to each submit not yet recorded
	std::unordered_map<std::string, uint64_t> m_drain_sites;
	uint64_t                     m_drains           = 0;
	uint64_t                     m_report_buffers   = 0;
	uint64_t                     m_report_packets   = 0;
	uint64_t                     m_report_drains    = 0;
	uint64_t                     m_report_waits     = 0;
	uint64_t                     m_backlog_waits    = 0;
	uint64_t                     m_report_time      = 0;
	std::unique_ptr<Common::PacketStream> m_stream; // last: its thread stops before the others go
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
