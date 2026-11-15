#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3FRAMEGEN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3FRAMEGEN_H_

#include "graphics/host_gpu/renderer/fsr3.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <functional>
#include <vector>

// AMD FidelityFX frame generation (FSR 3.1, SDK 1.1.4, MIT): one frame generated between each two
// frames PSSR upscaled with FSR, presented by FidelityFX's frame interpolation swapchain, which
// paces the presents itself. On by default in Pro mode with FSR 3.1 (Windows only); KYTY_FSR3_FG=0
// turns it off. PSSR's
// output is the scene before the game's tone mapping, so it can't serve as the HUD-less image; the
// presenter passes the game's flip buffer from before its HUD (WolverineHudless) when there is one,
// else the finished frame, HUD included, is interpolated (59-fsr-framegen.md).
namespace Libs::Graphics::Fsr3::FrameGen {

// Pro mode, FSR 3.1 upscaling, and KYTY_FSR3_FG not 0, on a build that has frame generation.
[[nodiscard]] bool Requested();
// Device creation: adds the present and image-acquire queues the swapchain needs, apart from the
// ones Kyty uses (already in `queues`). Without two spare queues frame generation stays off.
void AddQueues(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
               std::vector<vk::DeviceQueueCreateInfo>& queues);
// After the device exists: fetches those queues. The swapchain's submissions to graphics.queue
// hold queue_mutex, as Kyty's own do.
void Start(GraphicContext& graphics);
// Frame generation is on: Requested() and the device gave it its queues.
[[nodiscard]] bool Active();

// Renderer, after FSR upscaled a frame, in the same command buffer: the depth and motion vector
// inputs frame generation keeps for that frame.
void Prepare(vk::CommandBuffer command, const Pssr::Request& request, const Inputs& inputs, bool reset);
// The newest prepared frame; 0 before the first.
[[nodiscard]] uint64_t LatestFrame();
[[nodiscard]] uint64_t Presents(); // every present, real and generated (the window title's rate)

// Presenter: the swapchain, through FidelityFX's replacements. Present wants the image in
// SHADER_READ_ONLY_OPTIMAL, not PRESENT_SRC.
vk::Result CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* swapchain);
void       DestroySwapchain(vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index);
vk::Result Present(vk::Queue queue, const vk::PresentInfoKHR& present);
// Before each Present: `frame` is the presented frame's LatestFrame() when the flip was prepared,
// `rect` where the game image lies in the swapchain image. A frame equal to the previous present's
// is a repeat and gets no motion. `hudless`, when set, is the same image before the game drew its HUD
// (swapchain size and format, SHADER_READ_ONLY_OPTIMAL): the HUD then comes from the real frame, not
// interpolated. `dump`, when set, is called with the generated image (GENERAL
// layout) inside the command buffer that made it.
using DumpGenerated = std::function<void(vk::CommandBuffer command, vk::Image image, vk::Extent2D extent)>;
void Configure(vk::SwapchainKHR swapchain, uint64_t frame, vk::Rect2D rect, vk::Image hudless,
               const DumpGenerated& dump);

} // namespace Libs::Graphics::Fsr3::FrameGen

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3FRAMEGEN_H_
