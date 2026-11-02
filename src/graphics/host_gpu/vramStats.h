#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMSTATS_H_

#include <atomic>
#include <cstdint>

// Live device allocations by category (images, and buffers by MemoryUsage), for the 5 s
// "VRAM:" report next to the driver's heap usage/budget (VK_EXT_memory_budget via VMA).
namespace Libs::Graphics::VramStats {

enum Category : int { Images, BufDeviceLocal, BufUpload, BufDownload, BufStream, Count };
inline const char* const kNames[Count] = {"images", "buf.device", "buf.upload", "buf.download", "buf.stream"};

inline std::atomic<int64_t> g_bytes[Count] {};
inline std::atomic<int64_t> g_count[Count] {};
inline std::atomic<int64_t> g_images_outside_vram {0};

inline void Add(Category c, uint64_t bytes) {
	g_bytes[c].fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
	g_count[c].fetch_add(1, std::memory_order_relaxed);
}
inline void Remove(Category c, uint64_t bytes) {
	g_bytes[c].fetch_sub(static_cast<int64_t>(bytes), std::memory_order_relaxed);
	g_count[c].fetch_sub(1, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::VramStats

#endif
