#pragma once
// Local diagnostic: a GPU timestamp after each recorded draw, dispatch or native XPR run while
// a live `tracem` runs (live-trace.h). The timestamp is recorded in order with the command, so
// the recording worker replays it right after; it costs one relaxed load otherwise.

#include "live-trace.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace LiveTrace {

// MarkVulkan: an emulator-side command (copy, fill, clear, blit, resolve, barrier or a dispatch of
// its own), tagged with a VulkanMark id (generate-vulkan-recording.py).
enum MarkKind : uint64_t { MarkDraw = 1, MarkDispatch = 2, MarkNativeXpr = 3, MarkVulkan = 4 };

// Set on the GPU (render) thread: only its command buffers belong to ticks.
inline thread_local bool g_mark_thread = false;
// The scheduler's command buffer being recorded (CommandScheduler::BeginCommand): other command
// buffers the thread records (presentation, one-off transfers) get no marks.
inline thread_local VkCommandBuffer g_mark_command = VK_NULL_HANDLE;
// Nonzero while a guest draw or dispatch is recorded: its own dispatch carries the guest mark.
inline thread_local uint32_t g_mark_scope = 0;
// The emulator's own work an emulator-side command belongs to (bits 16-31 of its mark; live-trace.py names them).
enum MarkSiteId : uint32_t {
	SiteTilerDetile = 1, SiteTilerTile = 2, SiteTilerImage = 3, SiteTilerD16 = 4, SiteTilerBgra16 = 5,
	SiteFaultProcess = 6, SiteLodStats = 7,
};
inline thread_local uint32_t g_mark_site = 0;
struct MarkSite {
	uint32_t previous;
	explicit MarkSite(uint32_t site): previous(g_mark_site) { g_mark_site = site; }
	MarkSite(const MarkSite&)            = delete;
	MarkSite& operator=(const MarkSite&) = delete;
	~MarkSite() { g_mark_site = previous; }
};

inline uint32_t GpuMarkNow(vk::CommandBuffer command, MarkKind kind, uint64_t shader) {
	if (!g_marks_on.load(std::memory_order_relaxed) || g_mark_pool == nullptr || !command) return UINT32_MAX;
	const auto slot = g_mark_next.fetch_add(1, std::memory_order_relaxed);
	if (slot >= MarkSlots) return UINT32_MAX;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteTimestamp(static_cast<VkCommandBuffer>(command),
	                                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                                                  static_cast<VkQueryPool>(g_mark_pool), slot);
	Event(GpuMark, slot, (shader & 0x0fffffffffffffffull) | kind << 60u);
	return slot;
}

// Marks the command recorded while this object lives; `handle` yields the command buffer then.
template <typename Handle>
struct MarkAfter {
	Handle   handle;
	MarkKind kind;
	uint64_t shader;
	MarkAfter(Handle h, MarkKind k, uint64_t s): handle(h), kind(k), shader(s) { ++g_mark_scope; }
	MarkAfter(const MarkAfter&)            = delete;
	MarkAfter& operator=(const MarkAfter&) = delete;
	~MarkAfter() {
		--g_mark_scope;
		if (g_marks_on.load(std::memory_order_relaxed)) GpuMarkNow(handle(), kind, shader);
	}
};
template <typename Handle>
MarkAfter(Handle, MarkKind, uint64_t) -> MarkAfter<Handle>;

// generate-vulkan-recording.py: after an emulator-side work command on the GPU thread. `caller` (barriers): the code
// that recorded it (live-trace.py marks --exe names it).
inline void VulkanMark(VkCommandBuffer command, uint32_t id, bool dispatch, uint64_t caller = 0) {
	if (!g_marks_on.load(std::memory_order_relaxed) || !g_mark_thread || command != g_mark_command ||
	    (dispatch && g_mark_scope != 0))
		return;
	const auto slot = GpuMarkNow(vk::CommandBuffer(command), MarkVulkan, id | uint64_t {g_mark_site} << 16u);
	if (caller != 0 && slot != UINT32_MAX) Event(GpuMarkCaller, slot, caller);
}

} // namespace LiveTrace
