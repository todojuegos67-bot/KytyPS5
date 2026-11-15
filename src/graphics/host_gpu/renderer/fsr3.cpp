#include "graphics/host_gpu/renderer/fsr3.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/shaderBindings.h"

#include <bit> // FidelityFX's ffx_util.h calls std::popcount without it
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

// The Vulkan functions FidelityFX's backend calls directly (3rdparty/FidelityFX-SDK/kyty/ffx_vk_loader.h).
extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL kyty_ffx_vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice device, const char* layer, uint32_t* count, VkExtensionProperties* properties) {
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateDeviceExtensionProperties(device, layer, count,
	                                                                          properties);
}
VKAPI_ATTR void VKAPI_CALL kyty_ffx_vkGetPhysicalDeviceFeatures(VkPhysicalDevice          device,
                                                                VkPhysicalDeviceFeatures* features) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures(device, features);
}
VKAPI_ATTR void VKAPI_CALL kyty_ffx_vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice device, VkPhysicalDeviceFeatures2* features) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2(device, features);
}
VKAPI_ATTR void VKAPI_CALL kyty_ffx_vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties(device, properties);
}
VKAPI_ATTR void VKAPI_CALL kyty_ffx_vkGetPhysicalDeviceProperties(
    VkPhysicalDevice device, VkPhysicalDeviceProperties* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties(device, properties);
}
VKAPI_ATTR void VKAPI_CALL kyty_ffx_vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice device, VkPhysicalDeviceProperties2* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(device, properties);
}
VKAPI_ATTR VkResult VKAPI_CALL kyty_ffx_vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* info,
                                                        const VkAllocationCallbacks* allocator,
                                                        VkBuffer*                    buffer) {
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer(device, info, allocator, buffer);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL kyty_ffx_vkGetDeviceProcAddr(VkDevice    device,
                                                                     const char* name) {
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, name);
}
}

#if !defined(_WIN32)
// The backend's interface points at frame generation's swapchain hook; the frame interpolation
// swapchain is built on Windows only (3rdparty/CMakeLists.txt).
FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const* /*config*/) {
	return FFX_ERROR_INVALID_ARGUMENT;
}
#endif

namespace Libs::Graphics::Fsr3 {

struct Context {
	FfxFsr3UpscalerContext fsr {};
	// The images FSR 3.1 shares with frame generation, which the caller makes; Kyty makes them in a
	// backend context of their own, as the SDK's FSR 3 does.
	FfxUInt32                          shared_id    = 0;
	bool                               has_shared   = false; // shared_id holds a backend context
	size_t                             shared_count = 0;     // of shared made
	std::array<FfxResourceInternal, 3> shared {}; // previous nearest depth, dilated depth and motion
	std::chrono::steady_clock::time_point last {};
};

namespace {

// The request flags a context fixes at creation.
constexpr uint32_t FixedFlags = Pssr::kReverseDepth | Pssr::kAutoExposure;
// Each context takes two of the backend's: FSR's and its shared images'. Four PSSR contexts (the
// library's most), with room for replaced ones whose release waits for the GPU.
constexpr size_t BackendContexts = 16;

std::mutex           g_mutex; // FidelityFX's backend isn't thread-safe; releases come from other threads
std::vector<uint8_t> g_scratch;
VkDeviceContext      g_device {}; // the interface keeps a pointer to it and reads it at each context creation
FfxInterface         g_interface {};

void Warn(std::string_view text, FfxErrorCode error) {
	Log::WriteToConsoleAndLog(
	    fmt::format("WARNING: FSR 3.1: {} (0x{:08x}); PSSR scales linearly.\n", text,
	                static_cast<uint32_t>(error)));
}

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FSR3");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return enabled;
}

FfxResource Resource(const Image& image, const wchar_t* name, bool write) {
	FfxResourceDescription description {};
	description.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
	description.format   = ffxGetSurfaceFormatVK(image.format);
	description.width    = image.width;
	description.height   = image.height;
	description.depth    = 1;
	description.mipCount = 1;
	description.flags    = FFX_RESOURCE_FLAGS_NONE;
	// A depth image's view takes its own format and the depth aspect; with stencil, its transitions
	// take both aspects (Kyty's change in the backend's addBarrier).
	description.usage = static_cast<FfxResourceUsage>(
	    write ? FFX_RESOURCE_USAGE_UAV
	          : (image.depth ? FFX_RESOURCE_USAGE_DEPTHTARGET |
	                               (image.stencil ? FFX_RESOURCE_USAGE_STENCILTARGET : 0)
	                         : FFX_RESOURCE_USAGE_READ_ONLY));
	// The caller leaves every image in the general layout, FidelityFX's unordered access state.
	return ffxGetResourceVK(reinterpret_cast<void*>(image.image), description, name,
	                        FFX_RESOURCE_STATE_UNORDERED_ACCESS);
}

void DestroyLocked(Context* context) {
	for (size_t i = 0; i < context->shared_count; ++i) {
		g_interface.fpDestroyResource(&g_interface, context->shared[i], context->shared_id);
	}
	if (context->has_shared) {
		g_interface.fpDestroyBackendContext(&g_interface, context->shared_id);
	}
	ffxFsr3UpscalerContextDestroy(&context->fsr);
	delete context;
}

} // namespace

bool Accepts(const Pssr::Request& request) {
	ShaderTextureResource depth {}, motion {};
	std::copy_n(request.depth, 8, depth.fields);
	std::copy_n(request.motion, 8, motion.fields);
	return Enabled() && (request.flags & Pssr::kSpatialOnly) == 0 && !depth.IsNull() &&
	       !motion.IsNull() && request.x == 0 && request.y == 0;
}

bool Supports(VkFormat format) {
	return ffxGetSurfaceFormatVK(format) != FFX_SURFACE_FORMAT_UNKNOWN;
}

Creation CreationFor(const Pssr::Request& request, const Inputs& inputs) {
	return {inputs.color.width, inputs.color.height, inputs.output.width, inputs.output.height,
	        request.flags & FixedFlags};
}

Context* Create(const GraphicContext& graphics, const Creation& creation) {
	std::lock_guard lock(g_mutex);
	const auto      start = std::chrono::steady_clock::now();
	if (g_scratch.empty()) {
		g_device = {static_cast<VkDevice>(graphics.device),
		            static_cast<VkPhysicalDevice>(graphics.physical_device),
		            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr};
		std::vector<uint8_t> scratch(
		    ffxGetScratchMemorySizeVK(g_device.vkPhysicalDevice, BackendContexts));
		if (const auto error = ffxGetInterfaceVK(&g_interface, ffxGetDeviceVK(&g_device),
		                                         scratch.data(), scratch.size(), BackendContexts);
		    error != FFX_OK) {
			Warn("the Vulkan backend didn't start", error);
			return nullptr;
		}
		g_scratch = std::move(scratch); // the vector's buffer stays where the interface points
	}
	auto context = std::make_unique<Context>();
	FfxFsr3UpscalerContextDescription description {};
	description.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE |
	                    FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION;
	if ((creation.flags & Pssr::kReverseDepth) != 0) {
		description.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
	}
	if ((creation.flags & Pssr::kAutoExposure) != 0) {
		description.flags |= FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE;
	}
	description.maxRenderSize    = {creation.max_width, creation.max_height};
	description.maxUpscaleSize   = {creation.output_width, creation.output_height};
	description.backendInterface = g_interface;
	if (const auto error = ffxFsr3UpscalerContextCreate(&context->fsr, &description);
	    error != FFX_OK) {
		Warn("the context wasn't created", error);
		return nullptr;
	}
	FfxFsr3UpscalerSharedResourceDescriptions shared {};
	auto error = ffxFsr3UpscalerGetSharedResourceDescriptions(&context->fsr, &shared);
	if (error == FFX_OK) {
		error = g_interface.fpCreateBackendContext(&g_interface, FFX_EFFECT_SHAREDRESOURCES,
		                                           nullptr, &context->shared_id);
		context->has_shared = error == FFX_OK;
	}
	const std::array descriptions {&shared.reconstructedPrevNearestDepth, &shared.dilatedDepth,
	                               &shared.dilatedMotionVectors};
	for (size_t i = 0; i < descriptions.size() && error == FFX_OK; ++i) {
		error = g_interface.fpCreateResource(&g_interface, descriptions[i], context->shared_id,
		                                     &context->shared[i]);
		context->shared_count += error == FFX_OK ? 1 : 0;
	}
	if (error != FFX_OK) {
		Warn("the shared images weren't created", error);
		DestroyLocked(context.release());
		return nullptr;
	}
	FfxEffectMemoryUsage own {}, common {};
	(void)ffxFsr3UpscalerContextGetGpuMemoryUsage(&context->fsr, &own);
	(void)g_interface.fpGetEffectGpuMemoryUsage(&g_interface, context->shared_id, &common);
	Log::WriteToConsoleAndLog(fmt::format(
	    "FSR 3.1: a context for {}x{} to {}x{}, flags 0x{:x}, built in {} ms, {} MB of video memory\n",
	    creation.max_width, creation.max_height, creation.output_width, creation.output_height,
	    creation.flags,
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
	                                                          start)
	        .count(),
	    (own.totalUsageInBytes + common.totalUsageInBytes) >> 20));
	return context.release();
}

bool Dispatch(Context* context, vk::CommandBuffer command, const Pssr::Request& request,
              const Inputs& inputs, bool reset) {
	std::lock_guard lock(g_mutex);
	const auto      now      = std::chrono::steady_clock::now();
	const float     frame_ms = context->last == std::chrono::steady_clock::time_point {}
	                               ? 1000.0f / 60.0f
	                               : std::clamp(std::chrono::duration<float, std::milli>(
	                                                now - context->last)
	                                                .count(),
	                                            1.0f, 200.0f);
	context->last = now;

	FfxFsr3UpscalerDispatchDescription description {};
	description.commandList   = ffxGetCommandListVK(static_cast<VkCommandBuffer>(command));
	description.color         = Resource(inputs.color, L"PSSR color", false);
	description.depth         = Resource(inputs.depth, L"PSSR depth", false);
	description.motionVectors = Resource(inputs.motion, L"PSSR motion", false);
	if (inputs.exposure != nullptr) {
		description.exposure = Resource(*inputs.exposure, L"PSSR exposure", false);
	}
	description.output = Resource(inputs.output, L"PSSR output", true);
	description.reconstructedPrevNearestDepth = g_interface.fpGetResource(&g_interface, context->shared[0]);
	description.dilatedDepth                   = g_interface.fpGetResource(&g_interface, context->shared[1]);
	description.dilatedMotionVectors           = g_interface.fpGetResource(&g_interface, context->shared[2]);
	// PSSR's jitter and motion vectors are in render pixels, y down, as FSR takes them.
	description.jitterOffset      = {request.jitter[0], request.jitter[1]};
	description.motionVectorScale = {request.motion_scale[0], request.motion_scale[1]};
	description.renderSize        = {request.width, request.height};
	description.upscaleSize       = {inputs.output.width, inputs.output.height};
	description.frameTimeDelta    = frame_ms;
	description.preExposure       = request.pre_exposure > 0.0f ? request.pre_exposure : 1.0f;
	description.reset             = reset;
	// shortcut: PSSR's request carries no camera. The game's depth range (0.1 to 10000, 57-pro.md
	// section 6) and a 60-degree vertical field of view stand in; FSR only weighs disocclusion by
	// view depth with them. Pass the game's own if the request ever carries them.
	description.cameraNear              = 0.1f;
	description.cameraFar               = 10000.0f;
	description.cameraFovAngleVertical  = 1.0471976f;
	description.viewSpaceToMetersFactor = 1.0f;
	if (const auto error = ffxFsr3UpscalerContextDispatch(&context->fsr, &description);
	    error != FFX_OK) {
		Warn("the dispatch failed", error);
		return false;
	}
	return true;
}

void Destroy(Context* context) {
	if (context != nullptr) {
		std::lock_guard lock(g_mutex);
		DestroyLocked(context);
	}
}

} // namespace Libs::Graphics::Fsr3
