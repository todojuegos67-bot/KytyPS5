// Live analyzer ("Analizar vectores", F1 menu): while the game runs it follows every image the game creates and
// every dynamic-rendering pass / layout barrier it submits, and scores, frame after frame:
//   - depth:   the depth attachment used by the most passes at the largest size (reversed or not: its clear value);
//   - motion:  a 2-channel float/snorm image (R16G16_SFLOAT, ...) written each frame at the depth's size;
//   - HUD:     the display-size image the frame ends in, and the pass after which only UI is drawn into it.
// Each gets a confidence (0..1: stability over frames, size and format match, profile match), shown in the menu
// with a live preview. Once confident, the selected depth / motion vectors are copied each frame into our own
// images and handed to FSR 3 frame generation (and, in phase 2, to the upscaler). The selection is written to
// injector_profiles\<exe>.ini as signatures (format, size, usage) so the next run locks on at once. Generic: no
// handle, address, shader hash or game name is ever used to choose.
#pragma once
#include "common.h"

#include <vector>

namespace inj::analyzer {

// Image creation: usage bits we need for copies added (TRANSFER_SRC) on candidate images.
void PatchImageCreate(VkImageCreateInfo& info);
void OnImageCreated(VkImage image, const VkImageCreateInfo& info);
void OnImageDestroyed(VkImage image);
void OnViewCreated(VkImageView view, const VkImageViewCreateInfo& info);
void OnViewDestroyed(VkImageView view);
void SetSwapchainImages(const VkImage* images, uint32_t count);

// Command recording (record order) and submission (execution order).
void OnBeginCommandBuffer(VkCommandBuffer cb);
void OnBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info);
void OnBarrier2(VkCommandBuffer cb, const VkDependencyInfo* info);
void OnBarrier(VkCommandBuffer cb, uint32_t count, const VkImageMemoryBarrier* barriers);
void OnSubmit(const VkCommandBuffer* cbs, uint32_t count);

// At each present: score this frame's passes. display: the swapchain size.
void EndFrame(VkExtent2D display);

// What frame generation gets (our copies, in SHADER_READ_ONLY_OPTIMAL after RecordCopies).
struct Inputs {
	bool       depth_ok = false, motion_ok = false;
	VkImage    depth = VK_NULL_HANDLE, motion = VK_NULL_HANDLE;
	VkFormat   depth_format = VK_FORMAT_UNDEFINED, motion_format = VK_FORMAT_UNDEFINED;
	VkExtent2D depth_extent {}, motion_extent {};
	bool       reversed_depth = false;
};
// Records the copies of the selected images into `cb` (the game queue, right before present).
Inputs RecordCopies(VkCommandBuffer cb);

// The menu section (inside an ImGui window).
void DrawMenu();
void Shutdown();

} // namespace inj::analyzer
