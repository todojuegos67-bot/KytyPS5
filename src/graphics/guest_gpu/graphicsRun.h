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
	// (parpadeo7) As SendCommand, but always queued, from the GPU thread too: run by the GPU thread between packets
	// (ProcessCommands) or by its loop, never inside the caller; the caller hands over no recording. False (nothing queued)
	// once the GPU no longer accepts commands.
	bool               PostCommand(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void              Submit(std::span<const uint32_t> draw_commands,
	                         std::span<const uint32_t> constant_commands);
	void              SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void              SubmitFlipPreparation(uint64_t request_id);
	void              Done();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	// (par1, KYTY_PAR_QUEUES=1) A command processor wrote guest memory (a label): a queue waiting at a WAIT_REG_MEM on
	// the other translating thread checks again now.
	void ParWake();
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
		uint64_t                  serial            = 0; // its place in its queue (from 1)
		uint64_t                  sweep_stamp       = 0; // (ww2) PageManager::SweepStamp when it was queued (Enqueue)
		uint64_t                  block_epoch       = 0; // (par1) ParQueues::WakeEpoch when its slice that blocked began
		int64_t                   enqueue_ns        = 0; // (serie1, KYTY_SERIE1) when it was queued (steady clock)
		uint8_t                   serie1_kind       = 0; // (serie1) Serie1::Kind of a queue-0 graphics one (first slice)
		// KYTY_SPECULATE=4: the chunk committed next; whether the one it is in is translated the normal way, up to where
		// the next begins (none: to the end).
		uint32_t                    spec_chunk  = 0;
		bool                        spec_normal = false;
		std::optional<Pm4Execution> spec_end;
		// (envio2, KYTY_ENVIO2) A compute one handed to the ACB thread, committed when the GPU thread takes it again.
		bool envio2_handed = false;
		bool envio3_cont   = false; // (envio3: the one handed is a continuation, after a wait)
	};

	void              Enqueue(Submission submission);
	void              WaitForIdle();
	bool              CanProcessSubmission(const Submission& submission) const;
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	// (par1, KYTY_PAR_QUEUES=1) The ACB thread: the compute queues' submissions (ids 1..), recording into the GPU
	// thread's Vulkan recording stream (`stream`).
	static void       AcbThreadRun(GuestGpu* gpu, void* stream);
	// (par1) Whether the blocked submission at the front of a queue may be tried again (caller holds m_queue_mutex).
	[[nodiscard]] bool ParRetry(const Submission& submission) const;
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

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	// KYTY_SPECULATE (speculation.h): made at the first graphics command buffer it translates.
	std::unique_ptr<Speculation>                                     m_speculation;
	std::atomic<Speculation*>                                        m_ahead {nullptr}; // KYTY_SPECULATE=4's
	// (envio2, KYTY_ENVIO2) The compute queues' translation on their own thread (Kyty.Acb2), the GPU thread's; the
	// last serial each compute queue committed (mode 2's order check).
	std::unique_ptr<Speculation>                                     m_acb_spec;
	std::array<uint64_t, QueueCount>                                 m_envio2_last {};
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;
	// (par1, KYTY_PAR_QUEUES=1) The ACB thread (started and joined by the GPU thread), its round robin and whether it
	// runs a submission now (of which frame): m_queue_mutex.
	std::thread     m_acb_thread;
	uint32_t        m_acb_next_queue  = 1;
	bool            m_acb_processing  = false;
	uint64_t        m_acb_epoch       = 0;
	bool            m_acb_stop        = false;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
