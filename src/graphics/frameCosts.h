#ifndef EMULATOR_SRC_GRAPHICS_FRAME_COSTS_H_
#define EMULATOR_SRC_GRAPHICS_FRAME_COSTS_H_
// Time spent per 5 s window (printed with the "Frames (5 s)" line): shader compilation on the GPU thread and draws
// waiting for a pipeline that compiles in the background.
#include <atomic>
#include <chrono>
#include <cstdint>

namespace FrameCosts {
inline std::atomic<uint64_t> g_shader_us {0};
inline std::atomic<uint64_t> g_shader_count {0};
inline std::atomic<uint64_t> g_pipeline_wait_us {0};
inline std::atomic<uint64_t> g_pipeline_waits {0};
inline std::atomic<uint64_t> g_gpu_wait_us {0};
inline std::atomic<uint64_t> g_gpu_waits {0};
// Blocking GPU waits of every other thread (guest threads' readbacks, the completion thread). Before
// wolverine-turbo3-1025 they were added to g_gpu_wait_us, which made the "Render thread" line's busy share negative.
inline std::atomic<uint64_t> g_other_gpu_wait_us {0};
inline std::atomic<uint64_t> g_other_gpu_waits {0};
inline std::atomic<uint64_t> g_idle_us {0};
inline std::atomic<uint64_t> g_idles {0};

struct Timer {
	std::atomic<uint64_t>&                sum;
	std::atomic<uint64_t>&                count;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	~Timer() {
		sum.fetch_add(static_cast<uint64_t>(
		                  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count()),
		              std::memory_order_relaxed);
		count.fetch_add(1, std::memory_order_relaxed);
	}
};
} // namespace FrameCosts

#endif
