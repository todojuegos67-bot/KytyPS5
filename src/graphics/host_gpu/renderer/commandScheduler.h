#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

#include <queue>

#include <thread>
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
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Finish a complete dispatch; may submit, but never waits for the GPU.
	void           CompleteDispatch();
	// Finish `draws` complete draws; may submit (KYTY_DRAW_BATCH), but never waits for the GPU.
	void           CompleteDraws(uint32_t draws);
	// A flush nothing waits on: deferred while the GPU has KYTY_FLUSH_DEFER submissions to run.
	void           DeferrableFlush();
	// The deferred flush, if one is pending (before the GPU thread waits for work).
	void           FlushDeferred();
	[[nodiscard]] bool FlushPending() const noexcept { return m_flush_deferred; }
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept;
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	// A buffer upload (BufferCache::UploadDirtyRanges) of pages whose regions were last made CPU-dirty at `last_dirty`
	// (g_cpu_dirty_clock), no later than the open buffer began: those pages were dirty all the time since, so no
	// command recorded into the open buffer read or wrote them (every use of a CPU-dirty range synchronizes it first),
	// and their copies may run before all of its commands. They go into this buffer, submitted ahead of the open one:
	// one dependency each way per submission, where an upload in order ended the render pass and drained the queue on
	// both sides. Not into a buffer the open buffer recorded a copy into (`written_serial`, Buffer::written_serial:
	// a buffer created now gets its old buffers' bytes by a copy, an image download writes more than the range synced).
	// A shader's write synchronized its pages first: those are not among the pages still CPU-dirty.
	// Null: the upload is recorded in order (pages dirtied since, switch off).
	[[nodiscard]] vk::CommandBuffer UploadPrologue(uint64_t last_dirty, uint64_t written_serial);
	// Called with the upload prologue's buffer as it closes, before its closing barrier (BufferCache: the prologue's
	// copies gathered as one indirect copy).
	void SetPrologueHook(std::function<void(vk::CommandBuffer)> hook) { m_prologue_hook = std::move(hook); }
	// Called as the open buffer closes for submission, outside a render pass: commands batched for its end
	// (KYTY_FEEDBACK_BATCH: BufferCache's copy-feedback snapshots).
	void SetEpilogueHook(std::function<void(CommandBuffer&)> hook) { m_epilogue_hook = std::move(hook); }
	// Advances with every command buffer this scheduler begins.
	[[nodiscard]] uint64_t CommandSerial() const noexcept { return m_command_serial; }
	[[nodiscard]] uint64_t         CurrentTick() const noexcept;
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

	static constexpr size_t MaxSubmitEntries = 8; // command buffers in one submission

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

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
	void RunOperation(Common::UniqueFunction<void>&& operation);
	// Submits the buffers, in order, in one submission, the next tick signalled.
	struct SubmitEntry {
		vk::CommandBuffer buffer;
		uint32_t          timestamp_slot = UINT32_MAX;
	};
	uint64_t SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit);

	// The open buffer's upload prologue (UploadPrologue), ended and put ahead of it in its submission.
	[[nodiscard]] SubmitEntry ClosePrologue();
	// The GPU still runs work of this scheduler's the driver has (CompleteDispatch, CompleteDraws).
	[[nodiscard]] bool GpuBusy();
	// A deferred flush done once the GPU is down to its last submission, asked every 16 units of `weight`
	// (DeferrableFlush).
	[[nodiscard]] bool     FlushedDeferred(uint32_t weight);
	// The submissions the GPU has still to finish is at least `count` (the recording worker's not handed yet included).
	[[nodiscard]] bool BacklogAtLeast(uint64_t count);
	std::function<void(vk::CommandBuffer)> m_prologue_hook;
	std::function<void(CommandBuffer&)>    m_epilogue_hook;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	CommandBuffer                m_prologue;
	uint64_t                     m_command_clock  = 0; // g_cpu_dirty_clock when m_command began
	uint64_t                     m_command_serial = 1;
	uint32_t                     m_recorded_dispatches = 0;
	uint32_t                     m_recorded_draws      = 0;
	bool                         m_flush_deferred      = false; // DeferrableFlush left the open buffer open
	uint32_t                     m_deferred_checks     = 0;
	// The last tick handed to the driver (its vkQueueSubmit returned: on the recording worker for a deferred one).
	std::atomic<uint64_t>        m_driver_tick {0};
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	// Count of live entries in m_pending_operations, updated under
	// m_operation_mutex and read lock-free as a "nothing to retire" hint.
	std::atomic<uint32_t> m_pending_operation_count {0};
	// The tick of m_pending_operations' oldest entry (UINT64_MAX: none), the same way.
	std::atomic<uint64_t> m_pending_front_tick {UINT64_MAX};
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	// Local diagnostic: records every tick's completion while the live trace runs.
	std::jthread                 m_tick_monitor;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
