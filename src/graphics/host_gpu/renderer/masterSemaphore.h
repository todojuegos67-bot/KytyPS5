#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <source_location>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	// Refresh for a non-blocking "is it done yet?" check: with KYTY_TICK_POLL_US=N (live,
	// default 0 = always) it asks the driver at most once every N microseconds and otherwise
	// keeps the last known tick (still correct, only possibly older). Wait() is unaffected.
	void Poll();
	// Blocking waits are timed per call site and printed every 5 s ("GPU waits (5 s)").
	void Wait(uint64_t tick, std::source_location where = std::source_location::current());
	// A blocking wait on another GPU semaphore (the copy-queue readback timeline), counted in the
	// same "GPU waits (5 s)" list by call site and thread.
	static void NoteBlockingWait(const std::source_location& where, int64_t us);

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::atomic<int64_t>  m_last_poll_us {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
