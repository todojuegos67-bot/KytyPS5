#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <optional>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;
class Speculation;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// SendCommand, or false once the GPU thread takes no more (shutting down): for callers on guest threads that may still
	// run then.
	[[nodiscard]] bool TrySendCommand(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void              Submit(std::span<const uint32_t> draw_commands,
	                         std::span<const uint32_t> constant_commands);
	void              SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void              SubmitFlipPreparation(uint64_t request_id);
	void              Done();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	// KYTY_SERVE_IN_WAITS (live serve_in_waits, default 1): where the GPU thread runs the guest commands queued for it (a
	// game thread's readback fault among them) besides before each packet: 1 while it waits for the upload worker (only
	// from the drains told to serve, never inside a command), 2 after each segment a speculation commits, 4 while it
	// waits for a speculation's chunk and between chunks.
	[[nodiscard]] static uint32_t ServeInWaits() noexcept;
	// GPU thread: runs the guest commands queued for it, if any (ProcessCommands); not from inside one of them.
	static void               ServePendingCommands();
	[[nodiscard]] static bool CommandsPending() noexcept;
	// The GPU thread, or the speculation's thread while the GPU thread waits for its translation (KYTY_SPECULATE=2):
	// what reads the GPU thread's caches to translate (not what may fault: only the GPU thread reads back).
	[[nodiscard]] static bool IsTranslatingThread() noexcept;
	static void               SetSpeculationThread(bool speculation) noexcept;

	// Offline tools (xpr_replay_tests) run the renderer on their own thread.
	static void SetOfflineGpuThread(bool gpu_thread) noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, FrameBoundary };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      reset_processor   = false;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
		uint64_t                  frame_epoch       = 0;
		uint64_t                  scout_sequence    = 0; // queue 0's graphics submissions, numbered (scout.h)
		uint64_t                  serial            = 0; // its place in its queue (from 1)
		// KYTY_SPECULATE=4: the chunk committed next; whether the one it is in is translated the normal way, up to where
		// the next begins (none: to the end).
		uint32_t                    spec_chunk  = 0;
		bool                        spec_normal = false;
		std::optional<Pm4Execution> spec_end;
	};

	void              Enqueue(Submission submission);
	void              WaitForIdle();
	bool              CanProcessSubmission(const Submission& submission) const;
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_submission_mutex;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	std::array<uint64_t, QueueCount>               m_serials {};
	uint32_t                                       m_submission_count  = 0;
	// Producer epoch is owned by m_submission_mutex; consumed epoch by m_queue_mutex.
	uint64_t                                       m_submitted_frame   = 0;
	uint64_t                                       m_consumed_frame    = 0;
	bool                                           m_processing        = false;
	bool                                           m_graphics_done     = true;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Under m_queue_mutex: the last graphics submission's number, and a reset a flip preparation carried (the next one's).
	uint64_t                                       m_scout_sequence    = 0;
	bool                                           m_scout_reset       = false;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	// KYTY_SPECULATE (speculation.h): made at the first graphics command buffer it translates.
	std::unique_ptr<Speculation>                                     m_speculation;
	std::atomic<Speculation*>                                        m_ahead {nullptr}; // KYTY_SPECULATE=4's
	// KYTY_SPECULATE_ASYNC=<queue id> (mode 4): that async compute queue's command buffers speculated too, by
	// KYTY_SPECULATE_ASYNC_THREADS threads of their own (the culling chain's, 3.7 ms a frame on the GPU thread, 10-31).
	std::unique_ptr<Speculation>                                     m_speculation_async;
	std::atomic<Speculation*>                                        m_ahead_async {nullptr};
	uint32_t                                                         m_async_queue = 0; // (0: none)
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
