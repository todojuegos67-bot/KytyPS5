// FSR 3.1 frame generation x2 through AMD's public FidelityFX API (amd_fidelityfx_vk.dll, the signed DLL of the
// FidelityFX SDK v1.1.4, MIT). Ported from KytyPS5's src/local/fsr3-fg.cpp: the frame interpolation swapchain
// replaces the game's swapchain (its acquire / present / images), the generation context is configured and its
// prepare pass dispatched on the game queue right before each real present, and the swapchain presents the
// generated frame between two real ones with its own pacing. Inputs: the live analyzer's depth and motion vectors
// once confident, else neutral ones (far depth, zero motion: FSR 3 then interpolates from its optical flow alone).
#pragma once
#include "analyzer.h"
#include "common.h"

namespace inj::framegen {

bool Available();  // the DLL loaded and the extra queues created
bool Owns(VkSwapchainKHR swapchain);

VkResult CreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info, VkSwapchainKHR* out);
void     DestroySwapchain(VkSwapchainKHR swapchain);
VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images);
VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence,
                          uint32_t* index);
void     SetHdrMetadata(VkDevice device, uint32_t count, const VkSwapchainKHR* swapchains, const VkHdrMetadataEXT* metadata);

// Before a real present on `swapchain`: configure generation (on / off from the menu) and record its prepare pass.
void     Record(VkCommandBuffer cb, VkSwapchainKHR swapchain, VkExtent2D display, VkFormat format, const analyzer::Inputs& in);
// The pacing wait (real frames held so real x 2 stays under the VRR cap), then the present.
VkResult Present(VkQueue queue, const VkPresentInfoKHR* info);

struct Stats {
	float real_fps = 0, shown_fps = 0, ratio = 0;
	uint32_t generated = 0;     // generation dispatches in the last second
	bool  generating = false;
	bool  game_inputs = false;  // the analyzer's depth + motion vectors (else neutral)
	float cap_fps = 0;
	const char* state = "off";
};
Stats LastStats();

} // namespace inj::framegen
