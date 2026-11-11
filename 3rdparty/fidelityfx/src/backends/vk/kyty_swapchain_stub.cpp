// kyty: the frame interpolation swap chain of the SDK is not built (the emulator presents
// generated frames itself); the Vulkan backend still references its configuration entry point.
#include <FidelityFX/host/backends/vk/ffx_vk.h>

FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const* config) {
	(void)config;
	return FFX_ERROR_INVALID_ARGUMENT;
}
