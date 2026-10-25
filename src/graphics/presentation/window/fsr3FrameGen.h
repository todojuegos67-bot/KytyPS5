#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_
// AMD FSR 3.1 frame generation for any game (KYTY_FSR3_FG=1), ported from the Demon's Souls fork
// (todojuegos67-bot/KytyPS5 src/local/fsr3-fg.cpp) without its game-specific inputs.
//
// The Demon's Souls version feeds FSR 3 the game's own depth and motion vectors (found by shader hash in
// its TAA pass) and a HUD-less image. This title-independent version has none of them: it hands FSR 3 a
// constant far depth and zero motion vectors, so the interpolation comes from FSR 3's own optical flow
// between the two real frames (as driver-level frame generation does). Expect more artifacts than with
// game vectors, mostly around the HUD and fast-moving edges.
//
// The FidelityFX API's signed DLL (amd_fidelityfx_vk.dll next to the emulator) does x2. A DLL built from
// the patched SDK (tools/local/fsr3-mfg, CI input ffx_mfg=true) also does x3/x4 with KYTY_FSR3_MFG=3 or 4.
//
// The frame generation swapchain replaces the window's swapchain: it paces the generated frames between the
// real ones on threads of its own, so the guest's clock (vblank, flips) and the present thread keep their rate.
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>

namespace Libs::Graphics::Fsr3 {

// KYTY_FSR3_FG=1 and the DLL loaded (process start, before the device).
[[nodiscard]] bool Requested();

// Device creation: the extra queues the frame generation swapchain needs (a present queue of the graphics
// family, an async compute and an image acquire queue of a compute family). `graphics_count` is the number
// of queues of the graphics family the emulator already creates. False: frame generation stays off.
struct QueuePlan {
	uint32_t graphics_index = 0;          // index of the FI present queue in the graphics family
	uint32_t compute_family = UINT32_MAX; // family of the async compute and image acquire queues
	uint32_t compute_count  = 0;          // queues to create in it
};
bool PlanQueues(vk::PhysicalDevice physical_device, uint32_t queue_family, uint32_t graphics_count, QueuePlan& plan);
// After vkCreateDevice: fetch the queues.
void OnDevice(GraphicContext& graphics, const QueuePlan& plan);

// The frame generation swapchain is in use.
[[nodiscard]] bool Active();

// Replaces vkCreateSwapchainKHR (false: frame generation off, create it the normal way). The others replace
// their Vulkan function while Active(). QueuePresent takes the graphics queue's lock itself (through the
// swapchain's submit callback): call it without holding GraphicContext::queue_mutex.
bool       CreateSwapchain(GraphicContext& graphics, const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out);
void       DestroySwapchain(GraphicContext& graphics, vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(GraphicContext& graphics, vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result AcquireNextImage(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index);
vk::Result QueuePresent(GraphicContext& graphics, const vk::PresentInfoKHR& info);

// Present thread, after the frame's commands were submitted, right before QueuePresent. `new_frame`: the
// guest flipped a new image (false: a repeat of the last one, a blank frame; no generation for it).
void BeforePresent(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer, vk::Format format,
                   bool new_frame);

} // namespace Libs::Graphics::Fsr3

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_
