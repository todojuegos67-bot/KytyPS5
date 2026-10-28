// Upscaling (phase 2, menu > Reescalado; ab_key = A/B): DLSS (NVIDIA Streamline, the public SDK), FSR 4 (AMD's v07 INT8 model through
// KytyPS5's own runtime), FSR 3.1 (AMD FidelityFX API, amd_fidelityfx_vk.dll) and XeSS (Intel XeSS SDK, libxess.dll),
// each on Vulkan, inside DXVK's command stream.
//
// Render size: path (b). The game renders below the display through its own resolution-scale option; the analyzer then
// sees a main depth smaller than the swapchain, and the scene image at that size. Path (a), shrinking the game's targets,
// viewports and scissors underneath DXVK, was not taken: DXVK translates D3D11 one call at a time, and the game's shaders
// read its targets through constants and SV_Position math sized for the targets it asked for (half-resolution effects,
// full-screen passes, UV clamps), so shrinking them below DXVK breaks post-processing in ways no generic rule can find.
//
// Placement: at the HUD point. The analyzer's backbuffer receives the game's composite (its stretch of the scene to the
// display) as its first pass and the UI as its later passes; right before the second pass into it begins (the first UI
// pass), in the same command buffer, we copy the scene image, the depth and the motion vectors, run the upscaler into a
// display-size image of ours, sharpen it (RCAS, at the display size), and blit it over the game's composite. The UI is then
// drawn on top as the game drew it. With no UI pass (the HUD inside the composite) it is done right before DXVK's present
// blit instead (the HUD is then covered: said in the menu and log). DXVK's compute pipeline, compute descriptor sets and
// push constants bound at that point are recorded and re-bound after our work.
//
// Jitter: the camera finder's (camera.h). Into a post-TAA scene image the game's jitter is already resolved (0 is given)
// unless the game's TAA is not seen; pre-TAA input (experimental) takes the detected jitter; with the TAA off and no
// jitter of its own a Halton jitter can be injected into the game's projection.
#pragma once
#include "analyzer.h"
#include "common.h"

#include <vector>

namespace inj::upscale {

// Instance creation (before the game's instance exists): Streamline's slInit and XeSS's library, then the instance
// extensions DLSS / XeSS need (those the system has).
void BeforeCreateInstance();
void InstanceExtensions(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& extensions);

// Device creation: the extensions and features every backend needs (those the device has), patched into `info`
// (features ORed into the game's own structures, new ones chained in front). Restore() undoes the in-place changes if
// the device could not be made with them.
struct DevicePatch {
	std::vector<const char*>       extensions;
	std::vector<std::vector<uint8_t>> saved;  // the game's feature structures as they were
	std::vector<void*>             saved_at;
	std::vector<std::vector<uint8_t>> owned;  // structures we chained in
	void Restore();
};
void PatchDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo& in, VkDeviceCreateInfo& out, DevicePatch& patch);
void OnDeviceCreated(VkDevice device, const VkDeviceCreateInfo& made_with);  // the create info each device was made with
void OnActivate(VkDevice device);  // that device becomes the active one (the one with the swapchain)
void OnFirstPresent();  // the device that presents (DXVK's real one, not its probe): Streamline is told about it here
void OnDeviceDestroyed();
void SetDisplay(VkExtent2D display, VkFormat format);

// The state DXVK leaves bound in its command buffers (re-bound after our work).
void OnBeginCommandBuffer(VkCommandBuffer cb);
void OnBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline);
void OnBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count,
                          const VkDescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic);
void OnPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                     const void* data);

// Before the game's vkCmdBeginRendering (record order, DXVK's CS thread): the placement point and the frame boundary.
void BeforeRendering(VkCommandBuffer cb, const VkRenderingInfo* info, const analyzer::PassInfo& pass);
// At each present (DXVK's submission thread): readbacks, logs, retired objects.
void OnPresent();

void DrawMenu();
const char* OsdText();  // "" when off

} // namespace inj::upscale
