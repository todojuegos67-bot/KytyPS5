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
// What a pass draws into, for the upscaler's placement and the camera finder (record order).
struct PassInfo {
	VkImage       color0 = VK_NULL_HANDLE;    // its first color attachment
	VkImageLayout color0_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage       depth = VK_NULL_HANDLE;
	bool          main_depth = false;         // depth is the analyzer's pick (a scene pass)
	bool          swapchain = false;          // color0 is a swapchain image (DXVK's present blit: the frame ends)
	bool          backbuffer = false;         // color0 is the analyzer's backbuffer pick
};
PassInfo OnBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info);
// The analyzer runs (live analysis, game inputs for FG, or an upscaler on).
bool Active();
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

// What the upscaler needs, with each image's layout at this point of `cb`'s recording (the last barrier or attachment
// layout recorded for it in this command buffer, else the last one submitted).
struct Image {
	VkImage           image = VK_NULL_HANDLE;
	VkFormat          format = VK_FORMAT_UNDEFINED;
	VkExtent2D        extent {};
	VkImageUsageFlags usage = 0;
	VkImageLayout     layout = VK_IMAGE_LAYOUT_UNDEFINED;
	float             confidence = 0;
};
struct UpscaleView {
	Image backbuffer;  // display size: the composite and UI are drawn into it
	Image scene;       // render size: the last color image written before the composite (the game's post-TAA scene)
	Image pretaa;      // render size: the color image the game's TAA reads (found by its history ping-pong), if any
	Image depth, motion;
	bool  reversed_depth = false;
	int   hud_after = -1, ui_passes = 0;
	float hud_confidence = 0;
	bool  taa_found = false;
};
UpscaleView View(VkCommandBuffer cb);
VkImageLayout LayoutIn(VkCommandBuffer cb, VkImage image);

// The menu section (inside an ImGui window).
void DrawMenu();
void Shutdown();

} // namespace inj::analyzer
