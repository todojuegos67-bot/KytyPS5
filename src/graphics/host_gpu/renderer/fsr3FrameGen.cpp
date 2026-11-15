#include "graphics/host_gpu/renderer/fsr3FrameGen.h"

#include "common/emulatorConfig.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"

#include <bit> // FidelityFX's ffx_util.h calls std::popcount without it
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include "ffx_vk_fi_functions.h" // 3rdparty/FidelityFX-SDK/kyty

// The swapchain's direct Vulkan calls (ffx_vk_fi_loader.h), filled in Start().
extern "C" {
#define KYTY_FFX_FI_DEFINE(name) PFN_##name kyty_fi_##name = nullptr;
KYTY_FFX_FI_VK_FUNCTIONS(KYTY_FFX_FI_DEFINE)
#undef KYTY_FFX_FI_DEFINE
}
#endif

namespace Libs::Graphics::Fsr3::FrameGen {

namespace {

struct QueueSlot {
	uint32_t family;
	uint32_t index;
};

// What one prepared frame hands the generation after it.
struct Prepared {
	uint64_t id = 0;
	uint32_t render_width = 0, render_height = 0;
	float    frame_ms = 0.0f;
	bool     reset    = false;
};

// One context pair for the swapchain's size and the largest render size seen.
struct Contexts {
	FfxFrameInterpolationContext fi {};
	FfxOpticalflowContext        of {};
	FfxUInt32                    shared_id = 0;
	// Two of each, by frame parity: a frame's are read at its present, after the next frame may
	// have been prepared. Then the optical flow vector and its scene change detection.
	std::array<FfxResourceInternal, 8> shared {};
	size_t                             shared_count = 0;
	bool                               has_fi = false, has_of = false, has_shared = false;
	FfxDimensions2D                    display {}, max_render {};
	FfxSurfaceFormat                   format    = FFX_SURFACE_FORMAT_UNKNOWN;
	bool                               inverted  = false;
};

// Frame interpolation, optical flow, the shared images; twice, as retired contexts live until the next present.
constexpr uint32_t BackendContexts = 6;

std::mutex                     g_mutex; // contexts, prepared frames, the backend
std::optional<QueueSlot>       g_present_slot, g_acquire_slot;
bool                           g_active = false;
GraphicContext*                g_graphics = nullptr;
VkDeviceContext                g_device {};
FfxInterface                   g_interface {};
std::vector<uint8_t>           g_scratch;
FfxSwapchainReplacementFunctions g_functions {};
VkFrameInterpolationInfoFFX    g_info {};
std::unique_ptr<Contexts>      g_contexts;
// Replaced contexts: a present's interpolation is recorded under g_mutex but submitted later inside queuePresentKHR,
// so they go only after the next present has returned (Present).
std::vector<std::unique_ptr<Contexts>> g_retired;
std::array<Prepared, 4>        g_prepared {};
uint64_t                       g_latest = 0;
FfxDimensions2D                g_display {}; // the swapchain's
FfxSurfaceFormat               g_format = FFX_SURFACE_FORMAT_UNKNOWN;
std::chrono::steady_clock::time_point g_last_prepare {};
uint64_t                       g_last_configured = 0;
const DumpGenerated*           g_dump = nullptr; // during Present only
vk::Image                      g_hudless;        // during Present only

// Presents counted on the swapchain's own present thread, for the 5 s log line.
std::mutex                            g_stats_mutex;
std::vector<float>                    g_intervals_ms;
std::chrono::steady_clock::time_point g_last_present {}, g_window_start {};
uint64_t                              g_real = 0, g_generated = 0, g_repeats = 0, g_with_hudless = 0;
std::atomic<uint64_t>                 g_presents {0};

void Warn(const char* what, FfxErrorCode error) {
	Log::WriteToConsoleAndLog(fmt::format("FSR 3.1 frame generation: {} (error 0x{:x})\n", what,
	                                      static_cast<uint32_t>(error)));
}

FfxResource Resource(const Image& image, const wchar_t* name) {
	FfxResourceDescription description {};
	description.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
	description.format   = ffxGetSurfaceFormatVK(image.format);
	description.width    = image.width;
	description.height   = image.height;
	description.depth    = 1;
	description.mipCount = 1;
	description.usage    = static_cast<FfxResourceUsage>(
        image.depth ? FFX_RESOURCE_USAGE_DEPTHTARGET | (image.stencil ? FFX_RESOURCE_USAGE_STENCILTARGET : 0)
                       : FFX_RESOURCE_USAGE_READ_ONLY);
	// As for the upscaler (fsr3.cpp): Kyty leaves the inputs in the general layout.
	return ffxGetResourceVK(reinterpret_cast<void*>(image.image), description, name,
	                        FFX_RESOURCE_STATE_UNORDERED_ACCESS);
}

FfxResource Shared(const Contexts& contexts, size_t index) {
	return g_interface.fpGetResource(&g_interface, contexts.shared[index]);
}

void DestroyLocked(std::unique_ptr<Contexts>& contexts) {
	if (!contexts) {
		return;
	}
	auto& c = *contexts;
	for (size_t i = 0; i < c.shared_count; ++i) {
		g_interface.fpDestroyResource(&g_interface, c.shared[i], c.shared_id);
	}
	if (c.has_shared) {
		g_interface.fpDestroyBackendContext(&g_interface, c.shared_id);
	}
	if (c.has_fi) {
		ffxFrameInterpolationContextDestroy(&c.fi);
	}
	if (c.has_of) {
		ffxOpticalflowContextDestroy(&c.of);
	}
	contexts.reset();
}

// Waits for the GPU before contexts it may still read go: rare (a resize, a larger render size).
void WaitIdleLocked() {
	Common::LockGuard lock(g_graphics->queue_mutex);
	(void)g_graphics->queue.waitIdle();
}

bool CreateLocked(FfxDimensions2D max_render, bool inverted) {
	const auto start = std::chrono::steady_clock::now();
	if (g_scratch.empty()) {
		std::vector<uint8_t> scratch(ffxGetScratchMemorySizeVK(g_device.vkPhysicalDevice, BackendContexts));
		if (const auto error = ffxGetInterfaceVK(&g_interface, ffxGetDeviceVK(&g_device), scratch.data(),
		                                         scratch.size(), BackendContexts);
		    error != FFX_OK) {
			Warn("the Vulkan backend didn't start", error);
			return false;
		}
		g_scratch = std::move(scratch);
	}
	g_contexts  = std::make_unique<Contexts>();
	auto& c      = *g_contexts;
	c.display    = g_display;
	c.max_render = max_render;
	c.format     = g_format;
	c.inverted   = inverted;

	FfxOpticalflowContextDescription of {};
	of.backendInterface = g_interface;
	of.resolution       = g_display;
	auto error          = ffxOpticalflowContextCreate(&c.of, &of);
	c.has_of            = error == FFX_OK;

	FfxFrameInterpolationContextDescription fi {};
	fi.backendInterface = g_interface;
	// PSSR's motion vectors are in render pixels, without jitter.
	fi.flags                             = inverted ? FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED : 0;
	fi.maxRenderSize                     = max_render;
	fi.displaySize                       = g_display;
	fi.backBufferFormat                  = g_format;
	fi.previousInterpolationSourceFormat = g_format;
	if (error == FFX_OK) {
		error    = ffxFrameInterpolationContextCreate(&c.fi, &fi);
		c.has_fi = error == FFX_OK;
	}
	if (error == FFX_OK) {
		error = g_interface.fpCreateBackendContext(&g_interface, FFX_EFFECT_SHAREDAPIBACKEND, nullptr, &c.shared_id);
		c.has_shared = error == FFX_OK;
	}
	FfxFrameInterpolationSharedResourceDescriptions fi_shared {};
	FfxOpticalflowSharedResourceDescriptions        of_shared {};
	if (error == FFX_OK) {
		error = ffxFrameInterpolationGetSharedResourceDescriptions(&c.fi, &fi_shared);
	}
	if (error == FFX_OK) {
		error = ffxOpticalflowGetSharedResourceDescriptions(&c.of, &of_shared);
	}
	const std::array descriptions {&fi_shared.dilatedDepth, &fi_shared.dilatedMotionVectors,
	                               &fi_shared.reconstructedPrevNearestDepth, &fi_shared.dilatedDepth,
	                               &fi_shared.dilatedMotionVectors, &fi_shared.reconstructedPrevNearestDepth,
	                               &of_shared.opticalFlowVector, &of_shared.opticalFlowSCD};
	for (size_t i = 0; i < descriptions.size() && error == FFX_OK; ++i) {
		error = g_interface.fpCreateResource(&g_interface, descriptions[i], c.shared_id, &c.shared[i]);
		c.shared_count += error == FFX_OK ? 1 : 0;
	}
	if (error != FFX_OK) {
		Warn("the contexts weren't created", error);
		DestroyLocked(g_contexts);
		return false;
	}
	FfxEffectMemoryUsage fi_use {}, of_use {}, shared_use {};
	(void)ffxFrameInterpolationContextGetGpuMemoryUsage(&c.fi, &fi_use);
	(void)ffxOpticalflowContextGetGpuMemoryUsage(&c.of, &of_use);
	(void)g_interface.fpGetEffectGpuMemoryUsage(&g_interface, c.shared_id, &shared_use);
	Log::WriteToConsoleAndLog(fmt::format(
	    "FSR 3.1 frame generation: contexts for {}x{} renders to {}x{}, built in {} ms, {} MB of video memory\n",
	    max_render.width, max_render.height, g_display.width, g_display.height,
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(),
	    (fi_use.totalUsageInBytes + of_use.totalUsageInBytes + shared_use.totalUsageInBytes) >> 20));
	return true;
}

// Records optical flow and interpolation for the frame the swapchain presents (its game thread,
// inside Present).
FfxErrorCode Generate(const FfxFrameGenerationDispatchDescription* desc, void* /*context*/) {
	std::lock_guard lock(g_mutex);
	const auto&     prepared = g_prepared[desc->frameID % g_prepared.size()];
	const bool      usable   = g_contexts && prepared.id == desc->frameID &&
	                    desc->presentColor.description.width == g_contexts->display.width &&
	                    desc->presentColor.description.height == g_contexts->display.height &&
	                    desc->presentColor.description.format == g_contexts->format;
	if (!usable) {
		// The swapchain presents the output only when a frame was generated.
		const_cast<FfxFrameGenerationDispatchDescription*>(desc)->numInterpolatedFrames = 0;
		return FFX_ERROR_INVALID_ARGUMENT;
	}
	auto&      c      = *g_contexts;
	const auto parity = static_cast<size_t>(desc->frameID & 1u) * 3u;
	const bool reset  = desc->reset || prepared.reset;

	// As FidelityFX's provider: with a HUD-less image, optical flow follows it and interpolation keeps the HUD from
	// the real frame.
	FfxResource hudless {};
	if (g_hudless) {
		FfxResourceDescription description {};
		description.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
		description.format   = c.format;
		description.width    = c.display.width;
		description.height   = c.display.height;
		description.depth    = 1;
		description.mipCount = 1;
		description.usage    = FFX_RESOURCE_USAGE_READ_ONLY;
		hudless = ffxGetResourceVK(reinterpret_cast<void*>(static_cast<VkImage>(g_hudless)), description, L"HUDLess",
		                           FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
	}
	FfxOpticalflowDispatchDescription of {};
	of.commandList                = desc->commandList;
	of.color                      = g_hudless ? hudless : desc->presentColor;
	of.reset                      = reset;
	of.backbufferTransferFunction = desc->backBufferTransferFunction;
	of.minMaxLuminance            = {desc->minMaxLuminance[0], desc->minMaxLuminance[1]};
	of.opticalFlowVector          = Shared(c, 6);
	of.opticalFlowSCD             = Shared(c, 7);
	if (const auto error = ffxOpticalflowContextDispatch(&c.of, &of); error != FFX_OK) {
		Warn("optical flow failed", error);
		const_cast<FfxFrameGenerationDispatchDescription*>(desc)->numInterpolatedFrames = 0;
		return error;
	}

	// As FidelityFX's own frame generation provider (ffx-api, ffx_provider_framegeneration.cpp).
	FfxFrameInterpolationDispatchDescription fi {};
	fi.commandList                     = desc->commandList;
	fi.displaySize                     = c.display;
	fi.renderSize                      = {prepared.render_width, prepared.render_height};
	fi.currentBackBuffer               = desc->presentColor;
	fi.currentBackBuffer_HUDLess       = hudless;
	fi.output                          = desc->outputs[0];
	fi.interpolationRect               = desc->interpolationRect;
	fi.opticalFlowVector               = Shared(c, 6);
	fi.opticalFlowSceneChangeDetection = Shared(c, 7);
	fi.opticalFlowBlockSize            = 8;
	fi.opticalFlowScale = {1.0f / static_cast<float>(c.display.width), 1.0f / static_cast<float>(c.display.height)};
	// shortcut: the same camera stand-ins as the upscaler (fsr3.cpp); pass the game's own if PSSR's
	// request ever carries them.
	fi.cameraNear                 = 0.1f;
	fi.cameraFar                  = 10000.0f;
	fi.cameraFovAngleVertical     = 1.0471976f;
	fi.viewSpaceToMetersFactor    = 1.0f;
	fi.frameTimeDelta             = prepared.frame_ms;
	fi.reset                      = reset;
	fi.backBufferTransferFunction = desc->backBufferTransferFunction;
	fi.minMaxLuminance[0]         = desc->minMaxLuminance[0];
	fi.minMaxLuminance[1]         = desc->minMaxLuminance[1];
	fi.frameID                    = desc->frameID;
	fi.dilatedDepth               = Shared(c, parity + 0);
	fi.dilatedMotionVectors       = Shared(c, parity + 1);
	fi.reconstructedPrevDepth     = Shared(c, parity + 2);
	if (const auto error = ffxFrameInterpolationDispatch(&c.fi, &fi); error != FFX_OK) {
		Warn("interpolation failed", error);
		const_cast<FfxFrameGenerationDispatchDescription*>(desc)->numInterpolatedFrames = 0;
		return error;
	}
	if (g_dump != nullptr && *g_dump) {
		(*g_dump)(vk::CommandBuffer(static_cast<VkCommandBuffer>(desc->commandList)),
		          vk::Image(static_cast<VkImage>(desc->outputs[0].resource)),
		          {desc->outputs[0].description.width, desc->outputs[0].description.height});
	}
	{
		std::lock_guard stats(g_stats_mutex);
		g_generated++;
		g_with_hudless += g_hudless ? 1 : 0;
	}
	return FFX_OK;
}

#if defined(_WIN32)
VkResult SubmitToGameQueue(uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	Common::LockGuard lock(g_graphics->queue_mutex);
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit(static_cast<VkQueue>(g_graphics->queue), count, submits,
	                                                    fence);
}

// The swapchain's idle waits need the game queue to themselves; they take queue_mutex only for the call. The swapchain's
// semaphore waits around them run without it, so the recording thread can still send what they wait for.
VKAPI_ATTR VkResult VKAPI_CALL LockedQueueWaitIdle(VkQueue queue) {
	if (queue != static_cast<VkQueue>(g_graphics->queue)) {
		return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueWaitIdle(queue);
	}
	Common::LockGuard lock(g_graphics->queue_mutex);
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueWaitIdle(queue);
}

VKAPI_ATTR VkResult VKAPI_CALL LockedDeviceWaitIdle(VkDevice device) {
	Common::LockGuard lock(g_graphics->queue_mutex);
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle(device);
}

// The swapchain's present thread: every real and generated present, for the pacing numbers.
VKAPI_ATTR VkResult VKAPI_CALL CountedPresent(VkQueue queue, const VkPresentInfoKHR* info) {
	const auto result = VULKAN_HPP_DEFAULT_DISPATCHER.vkQueuePresentKHR(queue, info);
	const auto now    = std::chrono::steady_clock::now();
	g_presents.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(g_stats_mutex);
	if (g_last_present != std::chrono::steady_clock::time_point {}) {
		g_intervals_ms.push_back(std::chrono::duration<float, std::milli>(now - g_last_present).count());
	}
	g_last_present = now;
	if (g_window_start == std::chrono::steady_clock::time_point {}) {
		g_window_start = now;
	}
	if (now - g_window_start >= std::chrono::seconds(5) && !g_intervals_ms.empty()) {
		auto        sorted  = g_intervals_ms;
		std::ranges::sort(sorted);
		float sum = 0.0f;
		for (const float ms: sorted) {
			sum += ms;
		}
		const float        seconds        = std::chrono::duration<float>(now - g_window_start).count();
		unsigned long long guard_waits    = 0;
		unsigned long long guard_timeouts = 0;
		kyty_fi_GetSendGuardStats(&guard_waits, &guard_timeouts);
		Log::WriteToConsoleAndLog(fmt::format(
		    "FSR 3.1 frame generation (5 s): {:.1f} presents/s, {:.1f} game frames/s, {:.1f} generated/s, {} repeats, "
		    "{} HUD-less; present interval mean {:.2f} ms, median {:.2f}, p99 {:.2f}, max {:.2f}; waits for an unsent "
		    "signal {} (gave up {}) since start\n",
		    static_cast<float>(sorted.size()) / seconds, static_cast<float>(g_real) / seconds,
		    static_cast<float>(g_generated) / seconds, g_repeats, g_with_hudless, sum / static_cast<float>(sorted.size()),
		    sorted[sorted.size() / 2], sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], sorted.back(),
		    guard_waits, guard_timeouts));
		g_intervals_ms.clear();
		g_window_start = now;
		g_real = g_generated = g_repeats = g_with_hudless = 0;
	}
	return result;
}
#endif

} // namespace

bool Requested() {
#if defined(_WIN32)
	// On by default where it has frames to work on: Pro mode with FSR 3.1 upscaling PSSR.
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_FSR3_FG");
		const char* fsr = std::getenv("KYTY_FSR3"); // fsr3.cpp's switch: 0 is linear PSSR, no frames to generate from
		// The built-in FSR 3.1 only: DLSS, linear PSSR and AMD's DLLs (--fsr-folder) give it no frames.
		const bool built_in = Config::GetPssrUpscaler() == Config::PssrUpscaler::FSR && Config::GetFsrFolder().empty();
		return Config::Ps5ProEnabled() && built_in && (fsr == nullptr || std::string_view(fsr) != "0") &&
		       (value == nullptr || std::string_view(value) != "0");
	}();
	return requested;
#else
	return false; // shortcut: FidelityFX's Vulkan swapchain is Windows-only code; port it for Linux
#endif
}

void AddQueues(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
               std::vector<vk::DeviceQueueCreateInfo>& queues) {
	if (!Requested()) {
		return;
	}
	uint32_t family_count = 0;
	physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	physical_device.getQueueFamilyProperties(&family_count, families.data());
	std::vector<uint32_t> used(family_count, 0);
	for (const auto& queue: queues) {
		used[queue.queueFamilyIndex] = std::max(used[queue.queueFamilyIndex], queue.queueCount);
	}
	// The next free queue of the first family that qualifies, the engine's own family first.
	const auto take = [&](auto qualifies) -> std::optional<QueueSlot> {
		for (uint32_t pass = 0; pass < 2; ++pass) {
			for (uint32_t family = 0; family < family_count; ++family) {
				const bool own = used[family] != 0;
				if ((pass == 0) != own || used[family] >= families[family].queueCount || !qualifies(family)) {
					continue;
				}
				return QueueSlot {family, used[family]++};
			}
		}
		return std::nullopt;
	};
	const auto present = take([&](uint32_t family) {
		vk::Bool32 supported = VK_FALSE;
		return physical_device.getSurfaceSupportKHR(family, surface, &supported) == vk::Result::eSuccess &&
		       supported == VK_TRUE &&
		       static_cast<bool>(families[family].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
		                                                        vk::QueueFlagBits::eTransfer));
	});
	const auto acquire = take([](uint32_t) { return true; });
	if (!present || !acquire) {
		Log::WriteToConsoleAndLog("FSR 3.1 frame generation: off, the device has no two spare queues for its swapchain\n");
		return;
	}
	static const std::array<float, 64> priorities = [] {
		std::array<float, 64> ones {};
		ones.fill(1.0f);
		return ones;
	}();
	for (const auto& slot: {*present, *acquire}) {
		auto found = std::ranges::find(queues, slot.family, &vk::DeviceQueueCreateInfo::queueFamilyIndex);
		if (found == queues.end()) {
			vk::DeviceQueueCreateInfo info {};
			info.queueFamilyIndex = slot.family;
			found                 = queues.insert(queues.end(), info);
		}
		found->queueCount       = std::max(found->queueCount, slot.index + 1u);
		found->pQueuePriorities = priorities.data();
	}
	g_present_slot = present;
	g_acquire_slot = acquire;
	Log::WriteToConsoleAndLog(fmt::format(
	    "FSR 3.1 frame generation: present queue {}.{}, image acquire queue {}.{}\n", present->family,
	    present->index, acquire->family, acquire->index));
}

void Start(GraphicContext& graphics) {
#if defined(_WIN32)
	if (!g_present_slot || !g_acquire_slot) {
		return;
	}
	g_graphics = &graphics;
	g_device   = {static_cast<VkDevice>(graphics.device), static_cast<VkPhysicalDevice>(graphics.physical_device),
                VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr};
#define KYTY_FFX_FI_LOAD(name) kyty_fi_##name = VULKAN_HPP_DEFAULT_DISPATCHER.name;
	KYTY_FFX_FI_VK_FUNCTIONS(KYTY_FFX_FI_LOAD)
#undef KYTY_FFX_FI_LOAD
	kyty_fi_vkQueuePresentKHR = CountedPresent;
	kyty_fi_vkQueueWaitIdle   = LockedQueueWaitIdle;
	kyty_fi_vkDeviceWaitIdle  = LockedDeviceWaitIdle;
	const auto queue          = [&](const QueueSlot& slot) {
        VkQueueInfoFFX info {};
        info.queue       = static_cast<VkQueue>(graphics.device.getQueue(slot.family, slot.index));
        info.familyIndex = slot.family;
        return info;
	};
	g_info.physicalDevice        = g_device.vkPhysicalDevice;
	g_info.device                = g_device.vkDevice;
	g_info.gameQueue             = {static_cast<VkQueue>(graphics.queue), graphics.queue_family, SubmitToGameQueue};
	g_info.presentQueue          = queue(*g_present_slot);
	g_info.imageAcquireQueue     = queue(*g_acquire_slot);
	g_info.compositionMode       = VK_COMPOSITION_MODE_NOT_FORCED_FFX;
	if (ffxGetSwapchainReplacementFunctionsVK(static_cast<FfxDevice>(g_device.vkDevice), &g_functions) != FFX_OK) {
		Log::WriteToConsoleAndLog("FSR 3.1 frame generation: off, no swapchain replacement functions\n");
		return;
	}
	g_active = true;
	Log::WriteToConsoleAndLog("FSR 3.1 frame generation: on\n");
#else
	(void)graphics;
#endif
}

bool Active() {
	return g_active;
}

void Prepare(vk::CommandBuffer command, const Pssr::Request& request, const Inputs& inputs, bool reset) {
	if (!g_active) {
		return;
	}
	std::lock_guard lock(g_mutex);
	if (g_display.width == 0) {
		return; // no swapchain yet
	}
	const bool            inverted = (request.flags & Pssr::kReverseDepth) != 0;
	const FfxDimensions2D render {inputs.color.width, inputs.color.height};
	if (!g_contexts || g_contexts->display.width != g_display.width || g_contexts->display.height != g_display.height ||
	    g_contexts->format != g_format || g_contexts->inverted != inverted ||
	    g_contexts->max_render.width < render.width || g_contexts->max_render.height < render.height) {
		FfxDimensions2D max_render = render;
		if (g_contexts) {
			max_render = {std::max(render.width, g_contexts->max_render.width),
			              std::max(render.height, g_contexts->max_render.height)};
			g_retired.push_back(std::move(g_contexts));
			g_prepared = {}; // prepared with the old contexts
		}
		if (!CreateLocked(max_render, inverted)) {
			return;
		}
		reset = true;
	}
	const auto now      = std::chrono::steady_clock::now();
	const float frame_ms = g_last_prepare == std::chrono::steady_clock::time_point {}
	                           ? 1000.0f / 30.0f
	                           : std::clamp(std::chrono::duration<float, std::milli>(now - g_last_prepare).count(), 1.0f,
	                                        200.0f);
	g_last_prepare = now;

	const uint64_t id     = g_latest + 1;
	const auto     parity = static_cast<size_t>(id & 1u) * 3u;
	auto&          c      = *g_contexts;
	FfxFrameInterpolationPrepareDescription description {};
	description.commandList             = ffxGetCommandListVK(static_cast<VkCommandBuffer>(command));
	description.renderSize              = {request.width, request.height};
	description.jitterOffset            = {request.jitter[0], request.jitter[1]};
	description.motionVectorScale       = {request.motion_scale[0], request.motion_scale[1]};
	description.frameTimeDelta          = frame_ms;
	description.cameraNear              = 0.1f; // the upscaler's stand-ins (fsr3.cpp)
	description.cameraFar               = 10000.0f;
	description.cameraFovAngleVertical  = 1.0471976f;
	description.viewSpaceToMetersFactor = 1.0f;
	description.depth                   = Resource(inputs.depth, L"PSSR depth");
	description.motionVectors           = Resource(inputs.motion, L"PSSR motion");
	description.frameID                 = id;
	description.dilatedDepth            = Shared(c, parity + 0);
	description.dilatedMotionVectors    = Shared(c, parity + 1);
	description.reconstructedPrevDepth  = Shared(c, parity + 2);
	if (const auto error = ffxFrameInterpolationPrepare(&c.fi, &description); error != FFX_OK) {
		Warn("the prepare pass failed", error);
		return;
	}
	g_prepared[id % g_prepared.size()] = {id, request.width, request.height, frame_ms, reset};
	g_latest                           = id;
}

uint64_t Presents() {
	return g_presents.load(std::memory_order_relaxed);
}

uint64_t LatestFrame() {
	std::lock_guard lock(g_mutex);
	return g_latest;
}

vk::Result CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* swapchain) {
	const auto raw = static_cast<VkSwapchainCreateInfoKHR>(info);
	VkSwapchainKHR handle = VK_NULL_HANDLE;
	const auto     result = g_functions.createSwapchainFFX(g_device.vkDevice, &raw, nullptr, &handle, &g_info);
	*swapchain            = handle;
	if (result == VK_SUCCESS) {
		std::lock_guard lock(g_mutex);
		g_display = {info.imageExtent.width, info.imageExtent.height};
		g_format  = ffxGetSurfaceFormatVK(static_cast<VkFormat>(info.imageFormat));
	}
	return static_cast<vk::Result>(result);
}

void DestroySwapchain(vk::SwapchainKHR swapchain) {
	// Not under queue_mutex: it waits for the swapchain's presents, which may need the recording thread's submits;
	// its idle waits lock the game queue themselves (LockedQueueWaitIdle).
	g_functions.destroySwapchainKHR(g_device.vkDevice, static_cast<VkSwapchainKHR>(swapchain), nullptr);
}

vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
	return static_cast<vk::Result>(g_functions.getSwapchainImagesKHR(
	    g_device.vkDevice, static_cast<VkSwapchainKHR>(swapchain), count, reinterpret_cast<VkImage*>(images)));
}

vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
	return static_cast<vk::Result>(g_functions.acquireNextImageKHR(g_device.vkDevice,
	                                                               static_cast<VkSwapchainKHR>(swapchain), UINT64_MAX,
	                                                               static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE,
	                                                               index));
}

vk::Result Present(vk::Queue queue, const vk::PresentInfoKHR& present) {
	// Not under queue_mutex: generation takes g_mutex inside, and Prepare may hold g_mutex while it
	// waits for the queue. The swapchain's submissions to the game queue lock it one by one.
	const auto raw    = static_cast<VkPresentInfoKHR>(present);
	const auto result = g_functions.queuePresentKHR(static_cast<VkQueue>(queue), &raw);
	std::lock_guard lock(g_mutex);
	g_dump    = nullptr;
	g_hudless = nullptr;
	if (!g_retired.empty()) {
		WaitIdleLocked();
		for (auto& contexts: g_retired) DestroyLocked(contexts);
		g_retired.clear();
	}
	return static_cast<vk::Result>(result);
}

void Configure(vk::SwapchainKHR swapchain, uint64_t frame, vk::Rect2D rect, vk::Image hudless,
               const DumpGenerated& dump) {
	FfxFrameGenerationConfig config {};
	{
		std::lock_guard lock(g_mutex);
		const bool repeat = frame == g_last_configured;
		if (frame != 0 && g_prepared[frame % g_prepared.size()].id == frame) {
			// A repeated frame has nothing new: generate from a reset, which copies it.
			g_prepared[frame % g_prepared.size()].reset |= repeat;
		}
		g_last_configured = frame;
		g_dump            = dump ? &dump : nullptr;
		g_hudless         = hudless;
		std::lock_guard stats(g_stats_mutex);
		(repeat ? g_repeats : g_real)++;
	}
	config.swapChain               = reinterpret_cast<FfxSwapchain>(static_cast<VkSwapchainKHR>(swapchain));
	config.frameGenerationCallback = Generate;
	config.frameGenerationEnabled  = frame != 0; // from the first frame PSSR upscaled
	config.interpolationRect       = {rect.offset.x, rect.offset.y, static_cast<int32_t>(rect.extent.width),
	                                  static_cast<int32_t>(rect.extent.height)};
	config.frameID                 = frame;
	// Not under queue_mutex: a change of the enable waits for the swapchain's presents, and the recording thread needs
	// queue_mutex to send the game's submits. Its idle waits lock the game queue themselves (LockedQueueWaitIdle).
	if (const auto error = ffxSetFrameGenerationConfigToSwapchainVK(&config); error != FFX_OK) {
		Warn("the swapchain refused its configuration", error);
	}
}

} // namespace Libs::Graphics::Fsr3::FrameGen
