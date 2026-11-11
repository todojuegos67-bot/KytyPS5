#include "local/fsr-native.h"

#include "graphics/host_gpu/graphicContext.h"
#include "local/live-switch.h"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <atomic>
#include <memory>
#include <mutex>
#include <string_view>

// The FidelityFX sources are built on Linux and Windows (3rdparty/fidelityfx; macOS: none). Windows' FSR frame generation
// stays AMD's runtime (frame-gen.cpp): InterpolationRequested below is Linux's alone.
#if defined(__linux__) || defined(_WIN32)
#define KYTY_FSR_NATIVE 1
#else
#define KYTY_FSR_NATIVE 0
#endif

#if KYTY_FSR_NATIVE
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/ffx_opticalflow.h>

// The FidelityFX Vulkan backend calls a few global Vulkan entry points. The emulator loads Vulkan at run time, so
// ffx_linux_compat.h routes them here, to the emulator's dispatcher.
extern "C" {
void kyty_ffx_vkGetPhysicalDeviceFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures* features) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures(device, features);
}
void kyty_ffx_vkGetPhysicalDeviceFeatures2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2* features) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2(device, features);
}
void kyty_ffx_vkGetPhysicalDeviceProperties(VkPhysicalDevice device, VkPhysicalDeviceProperties* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties(device, properties);
}
void kyty_ffx_vkGetPhysicalDeviceProperties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(device, properties);
}
void kyty_ffx_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties* properties) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties(device, properties);
}
VkResult kyty_ffx_vkEnumerateDeviceExtensionProperties(VkPhysicalDevice device, const char* layer, uint32_t* count,
                                                       VkExtensionProperties* properties) {
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateDeviceExtensionProperties(device, layer, count, properties);
}
PFN_vkVoidFunction kyty_ffx_vkGetDeviceProcAddr(VkDevice device, const char* name) {
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, name);
}
}
#endif

namespace Libs::Graphics::FsrNative {

namespace {
std::atomic<bool> g_instead_of_dlss {false}; // UpscaleInsteadOfDlss
} // namespace

bool UpscalerRequested() {
#if KYTY_FSR_NATIVE
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_UPSCALER");
		return value != nullptr && std::string_view(value) == "fsr";
	}();
	return requested || g_instead_of_dlss.load(std::memory_order_relaxed);
#else
	return false;
#endif
}

bool UpscaleInsteadOfDlss() {
#if KYTY_FSR_NATIVE
	g_instead_of_dlss.store(true, std::memory_order_relaxed);
	return true;
#else
	return false;
#endif
}

bool InterpolationRequested() {
#if defined(__linux__)
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_FRAMEGEN_BACKEND");
		return value != nullptr && std::string_view(value) == "fsr";
	}();
	return requested;
#else
	return false;
#endif
}

#if KYTY_FSR_NATIVE
namespace {

void Report(const char* format, auto... args) {
	std::printf(format, args...);
	std::fflush(stdout);
}

// The state FidelityFX is told an image is in: what its layout is (the backend puts it back there afterwards).
FfxResourceStates StateOf(vk::ImageLayout layout, bool written) {
	if (written) return FFX_RESOURCE_STATE_UNORDERED_ACCESS;
	return layout == vk::ImageLayout::eGeneral ? FFX_RESOURCE_STATE_COMMON : FFX_RESOURCE_STATE_COMPUTE_READ;
}

FfxResource ToResource(const FrameGen::UpscaleImage& image, const wchar_t* name, bool written) {
	VkImageCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType   = VK_IMAGE_TYPE_2D;
	// (FidelityFX makes its own views: of the image's own format, a depth one's with the depth aspect. The game's depth
	// is a D32S8 image the bindings view as R32, which as a color view of it reads nothing.)
	info.format = static_cast<VkFormat>(image.image_format != vk::Format::eUndefined ? image.image_format : image.format);
	info.extent      = {image.extent.width, image.extent.height, 1};
	info.mipLevels   = 1;
	info.arrayLayers = 1;
	info.samples     = VK_SAMPLE_COUNT_1_BIT;
	info.usage       = static_cast<VkImageUsageFlags>(image.usage);
	const auto description = ffxGetImageResourceDescriptionVK(static_cast<VkImage>(image.image), info,
	                                                          written ? FFX_RESOURCE_USAGE_UAV : FFX_RESOURCE_USAGE_READ_ONLY);
	return ffxGetResourceVK(static_cast<VkImage>(image.image), description, name, StateOf(image.layout, written));
}

// (Windows' C library has no aligned_alloc: its own aligned pair.)
void* AlignedAlloc(size_t alignment, size_t size) {
#if defined(_WIN32)
	return _aligned_malloc(size, alignment);
#else
	return std::aligned_alloc(alignment, size);
#endif
}

struct AlignedFree {
#if defined(_WIN32)
	void operator()(void* memory) const { _aligned_free(memory); }
#else
	void operator()(void* memory) const { std::free(memory); }
#endif
};

// The backend interface over `contexts` effect contexts, its scratch memory owned here.
struct Backend {
	VkDeviceContext                       device_context {};
	std::unique_ptr<uint8_t, AlignedFree> scratch;
	FfxInterface                          backend {};

	bool Create(GraphicContext& graphics, size_t contexts) {
		device_context.vkDevice         = static_cast<VkDevice>(graphics.device);
		device_context.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
		device_context.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
		// (The SDK keeps structures with SIMD members in its scratch memory: more than its declared alignment.)
		const size_t size = (ffxGetScratchMemorySizeVK(device_context.vkPhysicalDevice, contexts) + 63) & ~size_t {63};
		scratch.reset(static_cast<uint8_t*>(AlignedAlloc(64, size)));
		std::memset(scratch.get(), 0, size);
		return ffxGetInterfaceVK(&backend, ffxGetDeviceVK(&device_context), scratch.get(), size, contexts) == FFX_OK;
	}
};

// Destroying FidelityFX contexts needs the GPU done with what referenced them (a size change: rare).
void WaitIdle(GraphicContext& graphics) {
	Common::LockGuard lock(graphics.queue_mutex);
	(void)graphics.queue.waitIdle();
}

float ElapsedMs(std::chrono::steady_clock::time_point& last) {
	const auto now = std::chrono::steady_clock::now();
	const float ms = last.time_since_epoch().count() == 0 ? 16.7f : std::chrono::duration<float, std::milli>(now - last).count();
	last           = now;
	return ms > 0.0f && ms < 200.0f ? ms : 16.7f;
}

#define KYTY_FSR_CHECK(state, call)                                                                                    \
	do {                                                                                                               \
		if (const FfxErrorCode kyty_ffx_error = (call); kyty_ffx_error != FFX_OK) {                                   \
			Report("FSR 3: %s failed (0x%x); off\n", #call, static_cast<unsigned>(kyty_ffx_error));                  \
			(state).failed = true;                                                                                     \
			return false;                                                                                              \
		}                                                                                                              \
	} while (false)

// FSR 3 upscaling (KYTY_UPSCALER=fsr).
struct Upscaler {
	std::mutex             mutex;
	Backend                backend;
	alignas(64) FfxFsr3UpscalerContext context {};
	uint32_t               shared_context = 0;
	FfxResourceInternal    previous_depth {}, dilated_depth {}, dilated_motion {};
	bool                   created = false, failed = false, reset = true;
	vk::Extent2D           render {}, display {};
	uint32_t               flags = 0;
	std::chrono::steady_clock::time_point last;

	void Destroy(GraphicContext& graphics) {
		if (!created) return;
		WaitIdle(graphics);
		auto& b = backend.backend;
		b.fpDestroyResource(&b, previous_depth, shared_context);
		b.fpDestroyResource(&b, dilated_depth, shared_context);
		b.fpDestroyResource(&b, dilated_motion, shared_context);
		b.fpDestroyBackendContext(&b, shared_context);
		ffxFsr3UpscalerContextDestroy(&context);
		created = false;
	}

	bool Create(GraphicContext& graphics, vk::Extent2D render_size, vk::Extent2D display_size, uint32_t context_flags) {
		render  = render_size;
		display = display_size;
		flags   = context_flags;
		KYTY_FSR_CHECK(*this, backend.Create(graphics, 2) ? FFX_OK : FFX_ERROR_BACKEND_API_ERROR);
		FfxFsr3UpscalerContextDescription description {};
		// The game's linear HDR scene before its tone mapping; reversed-Z with an infinite far plane.
		description.flags            = flags;
		description.maxRenderSize    = {render.width, render.height};
		description.maxUpscaleSize   = {display.width, display.height};
		description.backendInterface = backend.backend;
		KYTY_FSR_CHECK(*this, ffxFsr3UpscalerContextCreate(&context, &description));
		auto& b = backend.backend;
		KYTY_FSR_CHECK(*this, b.fpCreateBackendContext(&b, FFX_EFFECT_SHAREDRESOURCES, nullptr, &shared_context));
		FfxFsr3UpscalerSharedResourceDescriptions shared {};
		KYTY_FSR_CHECK(*this, ffxFsr3UpscalerGetSharedResourceDescriptions(&context, &shared));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &shared.reconstructedPrevNearestDepth, shared_context, &previous_depth));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &shared.dilatedDepth, shared_context, &dilated_depth));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &shared.dilatedMotionVectors, shared_context, &dilated_motion));
		created = true;
		reset   = true;
		Report("Super resolution: FSR 3 %ux%u -> %ux%u (FidelityFX SDK 1.1.4, built in)\n", render.width, render.height,
		       display.width, display.height);
		return true;
	}

	bool Record(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs) {
		std::lock_guard lock(mutex);
		if (failed) return false;
		// (FidelityFX makes its own views of mip level 0 and layer 0.)
		for (const auto* image: {&inputs.color, &inputs.depth, &inputs.motion, &inputs.output}) {
			if (image->base_level != 0 || image->base_layer != 0) {
				static bool reported = false;
				if (!std::exchange(reported, true)) Report("Super resolution: FSR 3 needs whole images; stretched instead\n");
				return false;
			}
		}
		// The game's linear HDR scene before its tone mapping, reversed-Z (live switches: fsr_depth_infinite 0 gives FSR
		// the game's far plane, fsr_auto_exposure 0 its exposure of 1).
		static auto& infinite_switch = LiveSwitch::Find("fsr_depth_infinite", 1);
		static auto& exposure_switch = LiveSwitch::Find("fsr_auto_exposure", 1);
		const bool   infinite        = infinite_switch.load(std::memory_order_relaxed) != 0;
		const uint32_t context_flags =
		    FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED |
		    (infinite ? FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE : 0u) |
		    (exposure_switch.load(std::memory_order_relaxed) != 0 ? FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE : 0u);
		if (created && (render != inputs.color.extent || display != inputs.output.extent || flags != context_flags))
			Destroy(graphics);
		if (!created && !Create(graphics, inputs.color.extent, inputs.output.extent, context_flags)) return false;
		if (static int reports = 0; reports++ < 2) {
			Report("Super resolution: FSR 3 inputs: color %s %ux%u, depth %s, motion %s, output %s %ux%u; jitter %g %g, "
			       "motion scale %g %g, near %g far %g, projection y %g, reset %d\n",
			       vk::to_string(inputs.color.format).c_str(), inputs.color.extent.width, inputs.color.extent.height,
			       vk::to_string(inputs.depth.format).c_str(), vk::to_string(inputs.motion.format).c_str(),
			       vk::to_string(inputs.output.format).c_str(), inputs.output.extent.width, inputs.output.extent.height,
			       inputs.jitter_x, inputs.jitter_y, inputs.motion_scale_x, inputs.motion_scale_y, inputs.near_plane,
			       inputs.far_plane, inputs.view_to_clip[5], inputs.reset ? 1 : 0);
		}

		static const float sharpness = [] {
			const char* value = std::getenv("KYTY_FSR_SHARPNESS");
			return value != nullptr ? std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f) : 0.6f;
		}();
		// Live switches (live-switch.h; KYTY_FSR_JITTER_SIGN, KYTY_FSR_UPSCALE as their start values): fsr_jitter_sign -1
		// negates the jitter (a test of the convention), fsr_upscale 0 stretches instead (a comparison).
		// fsr_jitter_mode (a test of the convention): 0 the game's jitter as DLSS takes it, 1 negated, 2 y negated, 3 x
		// negated. fsr_sharpness_pct overrides KYTY_FSR_SHARPNESS (-1: not).
		static auto& jitter_switch    = LiveSwitch::Find("fsr_jitter_mode", 0);
		static auto& upscale_switch   = LiveSwitch::Find("fsr_upscale", 1);
		static auto& sharpness_switch = LiveSwitch::Find("fsr_sharpness_pct", -1);
		if (upscale_switch.load(std::memory_order_relaxed) == 0) return false;
		const auto  jitter_mode = jitter_switch.load(std::memory_order_relaxed);
		const float jitter_x    = (jitter_mode == 1 || jitter_mode == 3) ? -1.0f : 1.0f;
		const float jitter_y    = (jitter_mode == 1 || jitter_mode == 2) ? -1.0f : 1.0f;
		const auto  sharpness_pct = sharpness_switch.load(std::memory_order_relaxed);
		const float sharpen       = sharpness_pct >= 0 ? std::clamp(static_cast<float>(sharpness_pct) / 100.0f, 0.0f, 1.0f)
		                                               : sharpness;
		auto&      b      = backend.backend;
		const auto list   = ffxGetCommandListVK(static_cast<VkCommandBuffer>(command));
		FfxFsr3UpscalerDispatchDescription dispatch {};
		dispatch.commandList   = list;
		dispatch.color         = ToResource(inputs.color, L"KytyFsrColor", false);
		dispatch.depth         = ToResource(inputs.depth, L"KytyFsrDepth", false);
		dispatch.motionVectors = ToResource(inputs.motion, L"KytyFsrMotion", false);
		dispatch.output        = ToResource(inputs.output, L"KytyFsrOutput", true);
		// (The game's reactive mask: the pixels taken from this frame rather than the history, issue #232.)
		if (inputs.reactive.image) dispatch.reactive = ToResource(inputs.reactive, L"KytyFsrReactive", false);
		// (Render pixels, as DLSS takes them; the game's vectors times their scale: render pixels to the previous position.)
		dispatch.jitterOffset      = {jitter_x * inputs.jitter_x, jitter_y * inputs.jitter_y};
		// Live switch fsr_mv_mode (a diagnostic): 1 the vectors negated, 2 none (zero), 3 half the scale.
		static auto& mv_mode_switch = LiveSwitch::Find("fsr_mv_mode", 0);
		const auto   mv_mode        = mv_mode_switch.load(std::memory_order_relaxed);
		const float  mv_factor      = mv_mode == 1 ? -1.0f : mv_mode == 2 ? 0.0f : mv_mode == 3 ? 0.5f : 1.0f;
		dispatch.motionVectorScale  = {mv_factor * inputs.motion_scale_x, mv_factor * inputs.motion_scale_y};
		dispatch.renderSize        = {render.width, render.height};
		dispatch.upscaleSize       = {display.width, display.height};
		dispatch.enableSharpening  = sharpen > 0.0f;
		dispatch.sharpness         = sharpen;
		dispatch.frameTimeDelta    = ElapsedMs(last);
		dispatch.preExposure       = 1.0f;
		dispatch.reset             = reset || inputs.reset;
		dispatch.cameraNear        = inputs.near_plane > 0.0f ? inputs.near_plane : 0.1f;
		dispatch.cameraFar         = infinite ? FLT_MAX : inputs.far_plane;
		// (Row vectors: the projection's y scale is cot(fov / 2).)
		const float y_scale             = std::fabs(inputs.view_to_clip[5]);
		dispatch.cameraFovAngleVertical = y_scale > 0.0f ? 2.0f * std::atan(1.0f / y_scale) : 1.0f;
		dispatch.viewSpaceToMetersFactor       = 1.0f;
		dispatch.dilatedDepth                  = b.fpGetResource(&b, dilated_depth);
		dispatch.dilatedMotionVectors          = b.fpGetResource(&b, dilated_motion);
		dispatch.reconstructedPrevNearestDepth = b.fpGetResource(&b, previous_depth);
		// Live switch fsr_debug_view 1 (a diagnostic): FSR's own view of its inputs and history in the output.
		static auto& debug_switch = LiveSwitch::Find("fsr_debug_view", 0);
		if (debug_switch.load(std::memory_order_relaxed) != 0) dispatch.flags |= FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW;
		KYTY_FSR_CHECK(*this, ffxFsr3UpscalerContextDispatch(&context, &dispatch));
		reset = false;
		return true;
	}
};

// FSR 3 frame interpolation (KYTY_FRAMEGEN_BACKEND=fsr).
struct Interpolator {
	Backend                backend;
	alignas(64) FfxOpticalflowContext        optical_flow {};
	alignas(64) FfxFrameInterpolationContext interpolation {};
	uint32_t               shared_context = 0;
	FfxResourceInternal    flow_vectors {}, scene_change {}, dilated_depth {}, dilated_motion {}, previous_depth {};
	bool                   created = false, failed = false, reset = true;
	vk::Extent2D           display {}, render {};
	vk::Format             format = vk::Format::eUndefined;
	uint64_t               frame_id = 0;

	void Destroy(GraphicContext& graphics) {
		if (!created) return;
		WaitIdle(graphics);
		auto& b = backend.backend;
		for (auto* resource: {&flow_vectors, &scene_change, &dilated_depth, &dilated_motion, &previous_depth})
			b.fpDestroyResource(&b, *resource, shared_context);
		b.fpDestroyBackendContext(&b, shared_context);
		ffxFrameInterpolationContextDestroy(&interpolation);
		ffxOpticalflowContextDestroy(&optical_flow);
		created = false;
	}

	bool Create(GraphicContext& graphics, const InterpolationInputs& inputs) {
		display = inputs.backbuffer.extent;
		render  = inputs.motion.extent;
		format  = inputs.backbuffer.format;
		KYTY_FSR_CHECK(*this, backend.Create(graphics, 3) ? FFX_OK : FFX_ERROR_BACKEND_API_ERROR);
		auto& b = backend.backend;
		FfxOpticalflowContextDescription flow {};
		flow.backendInterface = b;
		flow.resolution       = {display.width, display.height};
		KYTY_FSR_CHECK(*this, ffxOpticalflowContextCreate(&optical_flow, &flow));
		FfxFrameInterpolationContextDescription description {};
		description.flags            = FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE;
		description.maxRenderSize    = {render.width, render.height};
		description.displaySize      = {display.width, display.height};
		description.backBufferFormat = ffxGetSurfaceFormatVK(static_cast<VkFormat>(format));
		description.previousInterpolationSourceFormat = description.backBufferFormat;
		description.backendInterface                  = b;
		KYTY_FSR_CHECK(*this, ffxFrameInterpolationContextCreate(&interpolation, &description));
		KYTY_FSR_CHECK(*this, b.fpCreateBackendContext(&b, FFX_EFFECT_SHAREDRESOURCES, nullptr, &shared_context));
		FfxOpticalflowSharedResourceDescriptions flow_resources {};
		KYTY_FSR_CHECK(*this, ffxOpticalflowGetSharedResourceDescriptions(&optical_flow, &flow_resources));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &flow_resources.opticalFlowVector, shared_context, &flow_vectors));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &flow_resources.opticalFlowSCD, shared_context, &scene_change));
		FfxFrameInterpolationSharedResourceDescriptions resources {};
		KYTY_FSR_CHECK(*this, ffxFrameInterpolationGetSharedResourceDescriptions(&interpolation, &resources));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &resources.dilatedDepth, shared_context, &dilated_depth));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &resources.dilatedMotionVectors, shared_context, &dilated_motion));
		KYTY_FSR_CHECK(*this, b.fpCreateResource(&b, &resources.reconstructedPrevNearestDepth, shared_context, &previous_depth));
		created = true;
		reset   = true;
		Report("Frame generation: FSR 3 %ux%u, depth and motion %ux%u, 2x (FidelityFX SDK 1.1.4, built in)\n", display.width,
		       display.height, render.width, render.height);
		return true;
	}

	bool Record(GraphicContext& graphics, vk::CommandBuffer command, const InterpolationInputs& inputs,
	            const FrameGen::UpscaleImage& output) {
		if (failed) return false;
		if (created && (display != inputs.backbuffer.extent || render != inputs.motion.extent || format != inputs.backbuffer.format))
			Destroy(graphics);
		if (!created && !Create(graphics, inputs)) return false;
		reset |= inputs.reset;
		// An HDR10 backbuffer is PQ: FSR takes it to linear light over a 1000-nit range (the guest's frames carry no
		// mastering metadata), as sRGB is taken over 0..1.
		const auto                 transfer  = inputs.pq ? FFX_BACKBUFFER_TRANSFER_FUNCTION_PQ : FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
		const std::array<float, 2> luminance = inputs.pq ? std::array<float, 2> {0.0f, 1000.0f} : std::array<float, 2> {0.0f, 1.0f};
		// KYTY_FRAMEGEN_MV_SIGN=1 (a test): the game's vectors as they are. They point along the motion in UV units;
		// FSR wants render pixels towards the previous position.
		static const float mv_sign = [] {
			const char* value = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return value != nullptr && std::atof(value) > 0 ? 1.0f : -1.0f;
		}();
		auto&      b    = backend.backend;
		const auto list = ffxGetCommandListVK(static_cast<VkCommandBuffer>(command));
		const auto color  = ToResource(inputs.backbuffer, L"KytyFgColor", false);
		// (With a HUD-less image FSR interpolates that and takes the pixels that differ, the HUD's, from this frame.)
		const bool has_hudless = inputs.hudless.image != nullptr;
		const auto hudless     = has_hudless ? ToResource(inputs.hudless, L"KytyFgHudless", false) : FfxResource {};
		const auto target = ToResource(output, L"KytyFgOutput", true);
		const auto flow_vector    = b.fpGetResource(&b, flow_vectors);
		const auto flow_scd       = b.fpGetResource(&b, scene_change);
		const auto depth_dilated  = b.fpGetResource(&b, dilated_depth);
		const auto motion_dilated = b.fpGetResource(&b, dilated_motion);
		const auto depth_previous = b.fpGetResource(&b, previous_depth);

		FfxFrameInterpolationPrepareDescription prepare {};
		prepare.commandList             = list;
		prepare.renderSize              = {render.width, render.height};
		prepare.jitterOffset            = {inputs.jitter_px[0], inputs.jitter_px[1]};
		prepare.motionVectorScale       = {mv_sign * static_cast<float>(render.width), mv_sign * static_cast<float>(render.height)};
		prepare.frameTimeDelta          = inputs.frame_ms;
		prepare.cameraNear              = inputs.near_plane;
		prepare.cameraFar               = FLT_MAX;
		prepare.viewSpaceToMetersFactor = 1.0f;
		prepare.cameraFovAngleVertical  = inputs.fov_y;
		prepare.depth                   = ToResource(inputs.depth, L"KytyFgDepth", false);
		prepare.motionVectors           = ToResource(inputs.motion, L"KytyFgMotion", false);
		prepare.frameID                 = frame_id;
		prepare.dilatedDepth            = depth_dilated;
		prepare.dilatedMotionVectors    = motion_dilated;
		prepare.reconstructedPrevDepth  = depth_previous;
		KYTY_FSR_CHECK(*this, ffxFrameInterpolationPrepare(&interpolation, &prepare));

		FfxOpticalflowDispatchDescription flow {};
		flow.commandList                = list;
		flow.color                      = has_hudless ? hudless : color;
		flow.opticalFlowVector          = flow_vector;
		flow.opticalFlowSCD             = flow_scd;
		flow.reset                      = reset;
		flow.backbufferTransferFunction = transfer;
		flow.minMaxLuminance            = {luminance[0], luminance[1]};
		KYTY_FSR_CHECK(*this, ffxOpticalflowContextDispatch(&optical_flow, &flow));

		FfxFrameInterpolationDispatchDescription dispatch {};
		dispatch.commandList                     = list;
		dispatch.displaySize                     = {display.width, display.height};
		dispatch.renderSize                      = {render.width, render.height};
		dispatch.currentBackBuffer               = color;
		dispatch.currentBackBuffer_HUDLess       = hudless;
		dispatch.output                          = target;
		dispatch.interpolationRect               = {inputs.region.offset.x, inputs.region.offset.y,
		                                            static_cast<int32_t>(inputs.region.extent.width),
		                                            static_cast<int32_t>(inputs.region.extent.height)};
		dispatch.opticalFlowVector               = flow_vector;
		dispatch.opticalFlowSceneChangeDetection = flow_scd;
		dispatch.opticalFlowBufferSize           = {flow_vector.description.width, flow_vector.description.height};
		dispatch.opticalFlowScale = {1.0f / static_cast<float>(display.width), 1.0f / static_cast<float>(display.height)};
		dispatch.opticalFlowBlockSize       = 8;
		dispatch.cameraNear                 = inputs.near_plane;
		dispatch.cameraFar                  = FLT_MAX;
		dispatch.cameraFovAngleVertical     = inputs.fov_y;
		dispatch.viewSpaceToMetersFactor    = 1.0f;
		dispatch.frameTimeDelta             = inputs.frame_ms;
		dispatch.reset                      = reset;
		dispatch.backBufferTransferFunction = transfer;
		dispatch.minMaxLuminance[0]         = luminance[0];
		dispatch.minMaxLuminance[1]         = luminance[1];
		dispatch.frameID                    = frame_id;
		dispatch.dilatedDepth               = depth_dilated;
		dispatch.dilatedMotionVectors       = motion_dilated;
		dispatch.reconstructedPrevDepth     = depth_previous;
		KYTY_FSR_CHECK(*this, ffxFrameInterpolationDispatch(&interpolation, &dispatch));
		++frame_id;
		const bool generated = !reset;
		reset                = false;
		return generated;
	}
};

Upscaler& TheUpscaler() {
	static auto* upscaler = new Upscaler; // (kept for the process: its contexts live as long as the device)
	return *upscaler;
}

Interpolator& TheInterpolator() {
	static auto* interpolator = new Interpolator;
	return *interpolator;
}

} // namespace

bool Upscale(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs) {
	// AMD's runtime where there is one (Windows, FSR SDK 2: FSR 4 on the Radeon cards AMD lists for it), else built in.
	return UpscalerRequested() && (FrameGen::UpscaleWithAmdRuntime(graphics, command, inputs) ||
	                               TheUpscaler().Record(graphics, command, inputs));
}

bool Interpolate(GraphicContext& graphics, vk::CommandBuffer command, const InterpolationInputs& inputs,
                 const FrameGen::UpscaleImage& output) {
	return InterpolationRequested() && TheInterpolator().Record(graphics, command, inputs, output);
}
#else
bool Upscale(GraphicContext& /*graphics*/, vk::CommandBuffer /*command*/, const FrameGen::UpscaleInputs& /*inputs*/) {
	return false;
}

bool Interpolate(GraphicContext& /*graphics*/, vk::CommandBuffer /*command*/, const InterpolationInputs& /*inputs*/,
                 const FrameGen::UpscaleImage& /*output*/) {
	return false;
}
#endif

} // namespace Libs::Graphics::FsrNative
