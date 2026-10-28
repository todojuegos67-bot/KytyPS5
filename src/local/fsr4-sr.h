#ifndef EMULATOR_SRC_LOCAL_FSR4_SR_H_
#define EMULATOR_SRC_LOCAL_FSR4_SR_H_
// AMD FSR 4 super resolution (KYTY_UPSCALER=fsr4) on the game's PS5 Pro path, as DLSS / XeSS / FSR 3.1 there
// (FrameGen::Upscale). The model is AMD's source-v07 INT8 / DOT4 network (MIT, fsr4_shaders/LICENSE-FSR4-v07.txt),
// shipped precompiled as SPIR-V next to the emulator in fsr4_shaders/: per preset a pre pass, 12 network passes and a
// post pass (1080 and 2160 variants), the preset's initializers.bin (network weights) and pre_weights.bin (the pre
// pass's convolution), and a shader_manifest.json listing each module (entry point, local size, capabilities,
// descriptors, size). Our own runtime: it reads the manifest, checks every module against it (its descriptors
// reflected from the SPIR-V), builds each pipeline's push-descriptor layout from that, and records the passes in
// order (the dispatch sizes and constant buffer of AMD's FidelityFX SDK, MIT), then RCAS sharpening (rcas.spv).
// Needs shaderIntegerDotProduct (Vulkan 1.3, VK_KHR_shader_integer_dot_product), shaderInt8 / shaderInt16 /
// shaderFloat16 and computeDerivativeGroupLinear (VK_KHR_compute_shader_derivatives or the NV extension): without
// any of them Ready() is false and the game keeps its own TAA (demonsSoulsSuperResolution.cpp says so).
//
// KYTY_FSR4_PRESET = native | quality (default) | balanced | performance | ultraperf | drs : the model (one per
//   preset, as AMD trains them; drs is the dynamic-resolution model, any ratio). auto: AMD's choice by the ratio.
// KYTY_FSR4_DIR = <dir> : the assets' folder (default: fsr4_shaders next to the emulator).
// KYTY_FSR4_OPT = 1 : the optimized pass 11 and post pass (fsr4_shaders/opt/; same interface).
// KYTY_FSR4_DEPTH_INVERT = 0 : the depth as it is (default 1: the game's reversed depth flipped for the model's
//   closest-depth motion vector dilation, which the shipped modules compile for a forward depth).
// KYTY_UPSCALE_SHARPNESS = <0..1> : RCAS (default 0.3; 0 off), as for FSR 3.1.
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics::Fsr4 {

// KYTY_UPSCALER=fsr4.
[[nodiscard]] bool SrPathWanted();
// Device creation (with SrPathWanted): the compute shader derivatives extension, when the device has it.
void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions);
// Device creation: the features the model needs, turned on where the device has them (in the caller's structs), and
// the derivatives feature chained before `next`; returns the new head of the chain (`next` when nothing was added).
void* ChainDeviceFeatures(vk::PhysicalDevice physical_device, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& features12, vk::PhysicalDeviceVulkan13Features& features13,
                          vk::PhysicalDeviceFeatures& features, void* next);
// The device has what the model needs and the preset's assets are there and match their manifest (after the device
// exists; checked once, logged).
[[nodiscard]] bool Ready();
// The preset's display name ("FSR 4 Quality", ...).
[[nodiscard]] const char* Name();

struct UpscaleDesc {
	vk::CommandBuffer command = nullptr;
	vk::ImageView     color_view = nullptr; // render size, color_layout
	vk::ImageLayout   color_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::ImageView     depth_view = nullptr; // render size, depth_layout
	vk::ImageLayout   depth_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::ImageView     motion_view = nullptr; // render size
	vk::ImageLayout   motion_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::Image         output = nullptr;      // upscale size, GENERAL
	vk::ImageView     output_view = nullptr;
	vk::Format        output_format = vk::Format::eUndefined;
	vk::ImageUsageFlags output_usage {};
	uint32_t          output_level = 0, output_layer = 0;
	vk::Extent2D      render {};
	vk::Extent2D      upscale {};
	bool              reset = false;
	float jitter[2] {0.0f, 0.0f};   // render pixels
	float mv_scale[2] {0.0f, 0.0f}; // vectors * scale = render pixels to the previous frame (0: UV vectors)
	// The game's exposure (applied after the upscale): its float at exposure_index of the range (none: 1).
	vk::Buffer     exposure_buffer = nullptr;
	vk::DeviceSize exposure_base = 0, exposure_range = 0;
	uint32_t       exposure_index = 0;
};
// Records the passes; false: nothing usable written (the caller stretches).
bool Upscale(GraphicContext& graphics, const UpscaleDesc& desc);

} // namespace Libs::Graphics::Fsr4

#endif
