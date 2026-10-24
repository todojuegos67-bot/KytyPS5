#ifndef EMULATOR_SRC_LOCAL_FSR3_FG_H_
#define EMULATOR_SRC_LOCAL_FSR3_FG_H_
// AMD FSR 3.1 frame generation (KYTY_FSR3_FG=1), through the FidelityFX API's signed DLL
// (amd_fidelityfx_vk.dll next to the emulator) on any Vulkan GPU. The frame generation swapchain
// replaces the window's swapchain: it paces one generated frame between two real ones on threads of
// its own, so the guest's clock (vblank, flips) and the present thread keep their rate. Generation is
// on only for the frames whose guest frame ran the game's TAA pass (its depth and motion vectors,
// FrameGen::FrameInputs); menus, loading screens and movies have none and are presented as they are
// (frame generation off: the swapchain passes them through and restarts generation after them).
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>

namespace Libs::Graphics {
namespace FrameGen {
struct FrameInputs;
struct Camera;
}
namespace Fsr3 {

// Process start (before the device): KYTY_FSR3_FG=1 and the DLL loaded.
[[nodiscard]] bool Requested();

// Device creation: the extra queues the frame generation swapchain needs (a present queue of the
// graphics family, an async compute queue and an image acquire queue). `queue_family` is the
// graphics family; the counts are added to `graphics_count` (the graphics family's queue count) and
// written for the compute family. False: they cannot be had (frame generation stays off).
struct QueuePlan {
	uint32_t graphics_index = 0;            // index of the FI present queue in the graphics family
	uint32_t compute_family = UINT32_MAX;   // family of the async compute and image acquire queues
	uint32_t compute_count  = 0;            // queues to create in it
};
bool PlanQueues(vk::PhysicalDevice physical_device, uint32_t queue_family, uint32_t graphics_count, QueuePlan& plan);
// After vkCreateDevice: fetch them.
void OnDevice(GraphicContext& graphics, const QueuePlan& plan);

// The swapchain. Create: replaces vkCreateSwapchainKHR (false: frame generation off, create it the
// normal way). The others replace their Vulkan function while Active().
[[nodiscard]] bool Active();
// The real frames' least interval (seconds) for the shown frames to stay under KYTY_VRR_CAP_FPS (the G-SYNC cap), 0
// when no cap applies. The flip queue holds the game's flips to it (videoOut.cpp), so no frame of the game is dropped.
[[nodiscard]] double RealFrameInterval();

// FSR 3.1 upscaling (KYTY_UPSCALE=fsr): the game's 3D image, at its render size, upscaled into `output` with the TAA
// pass's depth and motion vectors (FrameGen::UpscaleComposite records it right before the game's composite reads it).
// The images: color, depth and motion in SHADER_READ_ONLY_OPTIMAL, the output in GENERAL; recorded into `command`.
[[nodiscard]] bool UpscaleWanted();
// KYTY_UPSCALER=fsr: the game's PS5 Pro path upscaled by FSR 3.1 (FrameGen::Upscale).
[[nodiscard]] bool SrPathWanted();
// The FidelityFX API loaded and the device known (after OnDevice).
[[nodiscard]] bool SrReady();
struct UpscaleDesc {
	VkCommandBuffer          command = VK_NULL_HANDLE;
	const VulkanImage*       color   = nullptr;
	const VulkanImage*       depth   = nullptr;
	const VulkanImage*       motion  = nullptr;
	const VulkanImage*       output  = nullptr;
	vk::Extent2D             render {};
	vk::Extent2D             upscale {};
	const FrameGen::Camera*  camera   = nullptr;
	bool                     reset    = false;
	float                    delta_ms = 16.7f;
	float jitter[2] {0.0f, 0.0f}; // (pixels, as DLSS takes it: the raw jittered frame of KYTY_TAA_REPLACE=upscaler)
	float mv_scale[2] {0.0f, 0.0f}; // vectors * scale = pixels to the previous frame (0: the TAA's UV vectors)
	float fov = 0.0f, z_near = 0.0f; // the camera's (0: from `camera`)
};
bool Upscale(GraphicContext& graphics, const UpscaleDesc& desc);
bool               CreateSwapchain(GraphicContext& graphics, const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out);
void               DestroySwapchain(GraphicContext& graphics, vk::SwapchainKHR swapchain);
vk::Result         GetSwapchainImages(GraphicContext& graphics, vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result         AcquireNextImage(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Semaphore semaphore,
                                    uint32_t* index);
vk::Result         QueuePresent(GraphicContext& graphics, const vk::PresentInfoKHR& info);

// Present thread, in the present's command buffer right after the frame was scaled into the swapchain image
// (`backbuffer_image`, left in PRESENT_SRC; `region`: where; `final_image`: the frame, in TRANSFER_SRC): the HUD layer
// (F where it differs from the frame's HUD-less image H) for FSR 3 to composite over every frame shown, and H scaled
// into the backbuffer in place of F. KYTY_FSR3_HUD_LAYER=0: off (the HUD-less image goes to FSR 3 instead).
void RecordHudLayer(GraphicContext& graphics, vk::CommandBuffer command, vk::Image backbuffer_image, vk::Extent2D backbuffer,
                    vk::Rect2D region, VulkanImage& final_image, const FrameGen::FrameInputs* inputs);

// Present thread, after the frame's image is in the swapchain's backbuffer and its commands were
// submitted (and finished), before QueuePresent: the frame's generation inputs (null or invalid:
// none, no generation for this frame).
void BeforePresent(GraphicContext& graphics, const FrameGen::FrameInputs* inputs, vk::SwapchainKHR swapchain,
                   vk::Extent2D backbuffer, vk::Format format);

} // namespace Fsr3
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FSR3_FG_H_
