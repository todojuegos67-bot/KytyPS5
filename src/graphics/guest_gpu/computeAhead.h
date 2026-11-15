#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMPUTEAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMPUTEAHEAD_H_

#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

// Compute ahead (from the old fork's draw-prep scanner): a thread reads each queued submission
// before the GPU thread runs it, follows its register writes and hands every dispatch to
// PipelineCache::BuildComputeAhead, which builds the dispatch's pipeline on a worker. The GPU
// thread then finds that pipeline in the driver cache instead of compiling it while the game
// waits. Draws too: a draw's pixel shader is built with the states its vertex shader was built
// with before (PipelineCache::BuildGraphicsAhead). Nothing is skipped; a dispatch or draw the
// scanner gets wrong only costs a worker's time.
// KYTY_COMPUTE_AHEAD=0 turns it off.
class ComputeAheadScanner final {
public:
	explicit ComputeAheadScanner(RenderContext& renderer);
	~ComputeAheadScanner();
	ComputeAheadScanner(const ComputeAheadScanner&)            = delete;
	ComputeAheadScanner& operator=(const ComputeAheadScanner&) = delete;

	[[nodiscard]] static bool Enabled();

	// queue: 0 graphics, 1 + n compute queue n; seq: the submission's order, see Started.
	void Enqueue(uint32_t queue, uint64_t seq, std::span<const uint32_t> commands);
	// The GPU thread resets its graphics command processor at a suspend point.
	void Reset();
	// The GPU thread has begun submission seq of a queue: anything up to it there is no longer ahead.
	void Started(uint32_t queue, uint64_t seq) { m_started[queue % m_started.size()].store(seq, std::memory_order_relaxed); }

private:
	struct Item {
		uint32_t                  queue = 0;
		uint64_t                  seq   = 0; // 0: reset
		std::span<const uint32_t> commands;
	};
	struct QueueState {
		std::unique_ptr<CommandProcessor> cp;
		std::unique_ptr<HW::Context>      saved_ctx;
		bool                              pushed  = false;
		bool                              tainted = false; // the register copy may be wrong
	};

	void        Run();
	QueueState& State(uint32_t queue);
	void        Scan(const Item& item);
	void        ApplyContextState(QueueState& state, uint32_t operation);
	void        ScanDispatch(QueueState& state, const uint32_t* packet, uint32_t opcode);
	void        ScanDraw(QueueState& state, const uint32_t* packet, uint32_t opcode);
	// The GPU thread has begun a later submission of the scanned item's queue.
	[[nodiscard]] static bool Followable(QueueState& state, uint64_t address, uint64_t bytes);
	[[nodiscard]] bool Behind() const {
		return m_item->seq < m_started[m_item->queue % m_started.size()].load(std::memory_order_relaxed);
	}

	RenderContext&                           m_renderer;
	std::array<std::unique_ptr<QueueState>, 57> m_queues;
	std::mutex                               m_mutex;
	std::condition_variable                  m_work;
	std::deque<Item>                         m_items;
	bool                                     m_stop = false;
	std::array<std::atomic<uint64_t>, 57>    m_started {};
	std::thread                              m_thread;
	const Item*                              m_item = nullptr; // the item being scanned
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMPUTEAHEAD_H_
