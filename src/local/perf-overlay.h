#ifndef EMULATOR_SRC_LOCAL_PERF_OVERLAY_H_
#define EMULATOR_SRC_LOCAL_PERF_OVERLAY_H_
// The numbers of the frame rate panel (KYTY_FPS_HUD, drawn by SystemOverlay): the guest's frame times (flip to flip,
// with the render thread's busy part of each) and, in the detailed panel, the machine's: the GPU (NVIDIA: the
// driver's NVML), video memory, the processor's cores and memory.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace PerfOverlay {

// 0 off, 1 the frame rate with the frame-time graph, 2 detailed (GPU, processor, memory): KYTY_FPS_HUD at start-up
// (the launcher's Overlay), then F8 in the game (CycleLevel).
[[nodiscard]] int Level();
// The next level (off, frame rate, detailed, off...); returns it.
int CycleLevel();

// GPU thread, at each guest flip: the steady-clock time and the render thread's total wait for guest work (ns).
void NoteFlip(int64_t now_ns, int64_t idle_ns);

// The last `count` frames (fewer early on), oldest first: each one's time and (`busy_ms` not null) the render
// thread's busy part of it (the frame less its wait for guest work), in ms.
void RecentFrames(size_t count, std::vector<float>* frame_ms, std::vector<float>* busy_ms);

struct Summary {
	double fps = 0, low1 = 0, max_ms = 0; // low1: the worst 1% of the frame times as frames per second
};
// The frames that ended in the last `seconds`.
[[nodiscard]] Summary Summarize(double seconds);

struct Machine {
	bool        valid = false;
	std::string gpu_name, cpu_name;
	// NVIDIA GPUs (NVML): load %, clocks in MHz, temperature in C, power in W, the memory in use and its size.
	bool     gpu_valid = false;
	uint32_t gpu_load = 0, gpu_clock = 0, gpu_memory_clock = 0, gpu_temperature = 0, gpu_power = 0;
	uint64_t gpu_memory_used = 0, gpu_memory_total = 0;
	// The emulator's video memory and the budget Windows gives it (0: unknown).
	uint64_t video_usage = 0, video_budget = 0;
	// The logical processors' load (0..1) and efficiency class (a hybrid CPU's E cores have the lower one), their
	// clocks in MHz (Windows: effective, Linux: /proc/cpuinfo; empty: unknown); all of them together,
	// the emulator's CPU time (in cores' worth) and
	// its render thread's (0..1).
	std::vector<float>    core_load;
	std::vector<uint8_t>  core_class;
	std::vector<uint32_t> core_clock;
	float                 cpu_load = 0, process_cores = 0, render_thread = 0;
	// Memory: the machine's in use and in all, the emulator's in use (its working set).
	uint64_t ram_used = 0, ram_total = 0, process_memory = 0;
};
// Starts the sampling thread (twice a second) at the first call. `device_uuid`: the Vulkan device's (16 bytes,
// VkPhysicalDeviceIDProperties::deviceUUID), the GPU whose NVML figures are taken.
void StartMachine(const uint8_t* device_uuid, const std::string& device_name);
// The last sample (valid false before the first).
[[nodiscard]] Machine LatestMachine();

} // namespace PerfOverlay

#endif // EMULATOR_SRC_LOCAL_PERF_OVERLAY_H_
