#include "framegen.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <thread>

namespace inj::framegen {
namespace {

using Clock = std::chrono::steady_clock;

HMODULE      g_dll = nullptr;
ffxFunctions g_ffx {};
bool         g_load_tried = false;

ffxContext                                  g_sc_ctx = nullptr;  // the frame interpolation swapchain's context
ffxQueryDescSwapchainReplacementFunctionsVK g_fn {};
VkSwapchainKHR                              g_swapchain = VK_NULL_HANDLE;

ffxContext g_fg_ctx = nullptr;  // the generation context (display size, format, depth convention)
VkExtent2D g_fg_display {}, g_fg_max_render {};
VkFormat   g_fg_format = VK_FORMAT_UNDEFINED;
bool       g_fg_reversed = false;
bool       g_fg_failed = false;
uint64_t   g_frame_id = 0;

// Neutral inputs: far depth (R32_SFLOAT) and zero motion (R16G16_SFLOAT) at the display size.
struct Owned { VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkExtent2D extent {}; float value = -1; };
Owned g_neutral_depth, g_neutral_motion;

std::atomic<uint32_t> g_dispatches {0};
Clock::time_point g_last_real {}, g_window_start {}, g_next_real {};
uint64_t g_presents_at_start = 0;
uint32_t g_real = 0;
double   g_present_ms = 0, g_slept_ms = 0;
Stats    g_stats {};
bool     g_logged_on = false, g_logged_inputs_game = false, g_inputs_mode_known = false;

void Message(uint32_t type, const wchar_t* message) {
	static std::atomic<uint32_t> count {0};
	if (count.fetch_add(1) < 200) Log("FidelityFX message (%u): %ls", type, message);
}

VkResult GameQueueSubmit(uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	std::lock_guard lock(QueueLock());
	return R().vkQueueSubmit(Q().game, count, submits, fence);
}

bool Load() {
	if (g_load_tried) return g_dll != nullptr;
	g_load_tried = true;
	g_dll = LoadLibraryW((ExeDir() + L"amd_fidelityfx_vk.dll").c_str());
	if (g_dll == nullptr) {
		Log("[fg] amd_fidelityfx_vk.dll not found next to the exe: frame generation unavailable");
		return false;
	}
	ffxLoadFunctions(&g_ffx, g_dll);
	if (g_ffx.CreateContext == nullptr || g_ffx.Configure == nullptr || g_ffx.Dispatch == nullptr || g_ffx.Query == nullptr) {
		Log("[fg] amd_fidelityfx_vk.dll has no FidelityFX API");
		g_dll = nullptr;
		return false;
	}
	ffxConfigureDescGlobalDebug1 debug {};
	debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
	debug.debugLevel  = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
	debug.fpMessage   = &Message;
	g_ffx.Configure(nullptr, &debug.header);
	Log("[fg] FidelityFX API loaded (amd_fidelityfx_vk.dll)");
	return true;
}

FfxApiResource Resource(VkImage image, VkFormat format, VkExtent2D extent, bool depth) {
	FfxApiResource r {};
	r.resource           = reinterpret_cast<void*>(image);
	r.state              = FFX_API_RESOURCE_STATE_COMPUTE_READ;
	r.description.type   = FFX_API_RESOURCE_TYPE_TEXTURE2D;
	r.description.format = ffxApiGetSurfaceFormatVK(format);
	r.description.width  = extent.width;
	r.description.height = extent.height;
	r.description.depth  = 1;
	r.description.mipCount = 1;
	r.description.flags  = FFX_API_RESOURCE_FLAGS_NONE;
	r.description.usage  = depth ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : FFX_API_RESOURCE_USAGE_READ_ONLY;
	return r;
}

void DestroyFg() {
	if (g_fg_ctx == nullptr) return;
	ffxConfigureDescFrameGeneration off {};
	off.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	off.swapChain              = reinterpret_cast<void*>(g_swapchain);
	off.frameGenerationEnabled = false;
	g_ffx.Configure(&g_fg_ctx, &off.header);
	g_ffx.DestroyContext(&g_fg_ctx, nullptr);
	g_fg_ctx = nullptr;
}

bool CreateFg(VkExtent2D display, VkExtent2D render, VkFormat format, bool reversed) {
	DestroyFg();
	ffxCreateBackendVKDesc backend {};
	backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend.vkDevice         = R().device;
	backend.vkPhysicalDevice = R().physical;
	backend.vkDeviceProcAddr = R().gdpa;  // the driver's, never our hooks
	ffxCreateContextDescFrameGeneration create {};
	create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
	create.header.pNext = &backend.header;
	create.flags        = reversed ? (FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE) : 0u;
	if (format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
	    format == VK_FORMAT_R16G16B16A16_SFLOAT)
		create.flags |= FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
	create.displaySize      = {display.width, display.height};
	create.maxRenderSize    = {render.width, render.height};
	create.backBufferFormat = ffxApiGetSurfaceFormatVK(format);
	const auto result = g_ffx.CreateContext(&g_fg_ctx, &create.header, nullptr);
	if (result != FFX_API_RETURN_OK) {
		Log("[fg] generation context not created (%u): generation off", result);
		g_fg_ctx    = nullptr;
		g_fg_failed = true;
		return false;
	}
	g_fg_display = display;
	g_fg_max_render = render;
	g_fg_format = format;
	g_fg_reversed = reversed;
	Log("[fg] generation context %ux%u (render up to %ux%u, format %d, depth %s)", display.width, display.height, render.width,
	    render.height, format, reversed ? "reversed, infinite" : "normal");
	return true;
}

void DestroyOwned(Owned& o) {
	if (o.image) R().vkDestroyImage(R().device, o.image, nullptr);
	if (o.memory) R().vkFreeMemory(R().device, o.memory, nullptr);
	o = {};
}

// A neutral input image cleared to `value` (recorded into cb), left in SHADER_READ_ONLY_OPTIMAL.
bool EnsureNeutral(VkCommandBuffer cb, Owned& o, VkFormat format, VkExtent2D extent, float value) {
	if (o.image != VK_NULL_HANDLE && o.extent.width == extent.width && o.extent.height == extent.height && o.value == value)
		return true;
	DestroyOwned(o);
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType = VK_IMAGE_TYPE_2D;
	ci.format = format;
	ci.extent = {extent.width, extent.height, 1};
	ci.mipLevels = ci.arrayLayers = 1;
	ci.samples = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling = VK_IMAGE_TILING_OPTIMAL;
	ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	if (R().vkCreateImage(R().device, &ci, nullptr, &o.image) != VK_SUCCESS) { o.image = VK_NULL_HANDLE; return false; }
	VkMemoryRequirements req {};
	R().vkGetImageMemoryRequirements(R().device, o.image, &req);
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize = req.size;
	ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (R().vkAllocateMemory(R().device, &ai, nullptr, &o.memory) != VK_SUCCESS ||
	    R().vkBindImageMemory(R().device, o.image, o.memory, 0) != VK_SUCCESS) {
		DestroyOwned(o);
		return false;
	}
	VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image = o.image;
	b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	VkClearColorValue clear {};
	clear.float32[0] = value;
	R().vkCmdClearColorImage(cb, o.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &b.subresourceRange);
	b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	o.extent = extent;
	o.value = value;
	return true;
}

double CapFps() {
	const int cfg = Cfg().fg_cap_fps;
	if (cfg == 0) return 0.0;
	if (cfg > 0) return cfg;
	const double hz = DisplayRefreshHz();  // KytyPS5's VRR rule: refresh - refresh^2 / 3600 (240 Hz -> 224)
	return hz > 30 ? hz - hz * hz / 3600.0 : 0.0;
}

} // namespace

bool Available() {
	return Q().fg_queues_ok && Load();
}
bool Owns(VkSwapchainKHR swapchain) {
	return g_sc_ctx != nullptr && swapchain == g_swapchain;
}

VkResult CreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info, VkSwapchainKHR* out) {
	std::lock_guard lock(QueueLock());
	if (g_sc_ctx != nullptr) DestroySwapchain(g_swapchain);  // (one context per swapchain: KytyPS5's lesson)
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	ffxCreateContextDescFrameGenerationSwapChainVK create {};
	create.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
	create.physicalDevice = R().physical;
	create.device         = device;
	create.swapchain      = &swapchain;
	create.createInfo     = *info;
	create.createInfo.oldSwapchain = VK_NULL_HANDLE;
	create.createInfo.pNext = nullptr;  // (format lists / maintenance chains: not for the replacement swapchain)
	create.createInfo.flags &= ~VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
	if (Q().game == VK_NULL_HANDLE) {  // before the first present: the graphics family's first queue (DXVK's)
		R().vkGetDeviceQueue(device, Q().graphics_family, 0, &Q().game);
		Q().game_family = Q().graphics_family;
	}
	create.gameQueue         = {Q().game, Q().game_family, &GameQueueSubmit};
	create.asyncComputeQueue = {Q().async_compute, Q().compute_family, nullptr};
	create.presentQueue      = {Q().present, Q().graphics_family, nullptr};
	create.imageAcquireQueue = {Q().image_acquire, Q().compute_family, nullptr};
	const auto result = g_ffx.CreateContext(&g_sc_ctx, &create.header, nullptr);
	if (result != FFX_API_RETURN_OK || swapchain == VK_NULL_HANDLE) {
		Log("[fg] frame interpolation swapchain not created (%u): plain swapchain", result);
		g_sc_ctx = nullptr;
		return R().vkCreateSwapchainKHR(device, info, nullptr, out);
	}
	g_fn = {};
	g_fn.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
	g_ffx.Query(&g_sc_ctx, &g_fn.header);
	if (g_fn.pOutQueuePresentKHR == nullptr || g_fn.pOutAcquireNextImageKHR == nullptr || g_fn.pOutGetSwapchainImagesKHR == nullptr) {
		Log("[fg] no replacement swapchain functions: plain swapchain");
		g_ffx.DestroyContext(&g_sc_ctx, nullptr);
		g_sc_ctx = nullptr;
		return R().vkCreateSwapchainKHR(device, info, nullptr, out);
	}
	g_swapchain = swapchain;
	g_presents_at_start = 0;
	g_window_start = {};
	*out = swapchain;
	Log("[fg] frame interpolation swapchain %ux%u, format %d, present mode %d, %u images", info->imageExtent.width,
	    info->imageExtent.height, info->imageFormat, info->presentMode, info->minImageCount);
	return VK_SUCCESS;
}

void DestroySwapchain(VkSwapchainKHR swapchain) {
	std::lock_guard lock(QueueLock());
	if (!Owns(swapchain)) return;
	DestroyFg();
	g_ffx.DestroyContext(&g_sc_ctx, nullptr);  // destroys its swapchain too
	g_sc_ctx = nullptr;
	g_fn = {};
	g_swapchain = VK_NULL_HANDLE;
	g_logged_on = false;
	Log("[fg] frame interpolation swapchain released (resize, mode change or exit)");
}

VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images) {
	return g_fn.pOutGetSwapchainImagesKHR(device, swapchain, count, images);
}
VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence,
                          uint32_t* index) {
	return g_fn.pOutAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, index);
}
void SetHdrMetadata(VkDevice device, uint32_t count, const VkSwapchainKHR* swapchains, const VkHdrMetadataEXT* metadata) {
	if (g_fn.pOutSetHdrMetadataEXT != nullptr) g_fn.pOutSetHdrMetadataEXT(device, count, swapchains, metadata);
}

void Record(VkCommandBuffer cb, VkSwapchainKHR swapchain, VkExtent2D display, VkFormat format, const analyzer::Inputs& in) {
	if (!Owns(swapchain)) return;
	const auto now = Clock::now();
	const double delta_ms = g_last_real == Clock::time_point {} ? 16.7 : std::chrono::duration<double, std::milli>(now - g_last_real).count();
	g_last_real = now;

	// Inputs: the game's (the analyzer's copies) when both are confident and the same size, else neutral.
	const bool game = in.depth_ok && in.motion_ok && in.depth_extent.width == in.motion_extent.width &&
	                  in.depth_extent.height == in.motion_extent.height;
	const bool reversed = game ? in.reversed_depth : false;
	if (!g_inputs_mode_known || game != g_logged_inputs_game) {
		if (game)
			Log("[fg] inputs: the game's depth (fmt %d %ux%u, %s) and motion vectors (fmt %d) from the analyzer",
			    in.depth_format, in.depth_extent.width, in.depth_extent.height, reversed ? "reversed" : "normal", in.motion_format);
		else
			Log("[fg] inputs: neutral (far depth, zero motion): FSR 3 interpolates from optical flow only");
		g_logged_inputs_game = game;
		g_inputs_mode_known  = true;
	}
	const VkExtent2D render = game ? in.depth_extent : display;
	const bool want = Cfg().fg_enabled && !g_fg_failed;
	if (want && (g_fg_ctx == nullptr || g_fg_display.width != display.width || g_fg_display.height != display.height ||
	             g_fg_format != format || g_fg_reversed != reversed || render.width > g_fg_max_render.width ||
	             render.height > g_fg_max_render.height)) {
		CreateFg(display, {std::max(render.width, display.width), std::max(render.height, display.height)}, format, reversed);
	}
	g_stats.game_inputs = game;
	if (g_fg_ctx == nullptr) { g_stats.generating = false; g_stats.state = g_fg_failed ? "error" : "off"; return; }

	++g_frame_id;
	ffxConfigureDescFrameGeneration config {};
	config.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	config.swapChain   = reinterpret_cast<void*>(swapchain);
	config.frameGenerationCallback = [](ffxDispatchDescFrameGeneration* params, void* context) -> ffxReturnCode_t {
		g_dispatches.fetch_add(1, std::memory_order_relaxed);
		return g_ffx.Dispatch(reinterpret_cast<ffxContext*>(context), &params->header);
	};
	config.frameGenerationCallbackUserContext = &g_fg_ctx;
	config.frameGenerationEnabled = want;
	config.allowAsyncWorkloads    = false;
	config.flags                  = 0;
	config.onlyPresentGenerated   = false;
	config.generationRect         = {0, 0, static_cast<int32_t>(display.width), static_cast<int32_t>(display.height)};
	config.frameID                = g_frame_id;
	if (const auto r = g_ffx.Configure(&g_fg_ctx, &config.header); r != FFX_API_RETURN_OK) {
		static int logged = 0;
		if (logged++ < 8) Log("[fg] configure failed (%u)", r);
		return;
	}
	g_stats.generating = want;
	g_stats.state = want ? (game ? "on (game depth + MV)" : "on (optical flow only)") : "off (menu)";
	if (!want) { if (g_logged_on) Log("[fg] generation off (menu / F2)"); g_logged_on = false; return; }

	VkImage depth = in.depth, motion = in.motion;
	VkFormat depth_format = in.depth_format, motion_format = in.motion_format;
	if (!game) {
		if (!EnsureNeutral(cb, g_neutral_depth, VK_FORMAT_R32_SFLOAT, display, 1.0f) ||
		    !EnsureNeutral(cb, g_neutral_motion, VK_FORMAT_R16G16_SFLOAT, display, 0.0f))
			return;
		depth = g_neutral_depth.image; depth_format = VK_FORMAT_R32_SFLOAT;
		motion = g_neutral_motion.image; motion_format = VK_FORMAT_R16G16_SFLOAT;
	}
	ffxDispatchDescFrameGenerationPrepare prepare {};
	prepare.header.type  = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
	prepare.frameID      = g_frame_id;
	prepare.flags        = 0;
	prepare.commandList  = cb;
	prepare.renderSize   = {render.width, render.height};
	prepare.jitterOffset = {0, 0};
	const float sign = static_cast<float>(Cfg().mv_sign);
	prepare.motionVectorScale = Cfg().mv_units == 0 ? FfxApiFloatCoords2D {sign * render.width, sign * render.height}
	                                                : FfxApiFloatCoords2D {sign, sign};
	prepare.frameTimeDelta = static_cast<float>(std::clamp(delta_ms, 1.0, 100.0));
	prepare.cameraNear = reversed ? FLT_MAX : 0.1f;
	prepare.cameraFar  = reversed ? 0.1f : 1000.0f;
	prepare.cameraFovAngleVertical  = 1.0f;  // unknown generically (phase 2: from the profile)
	prepare.viewSpaceToMetersFactor = 1.0f;
	prepare.depth         = Resource(depth, depth_format, render, game);
	prepare.motionVectors = Resource(motion, motion_format, render, false);
	if (const auto r = g_ffx.Dispatch(&g_fg_ctx, &prepare.header); r != FFX_API_RETURN_OK) {
		static int logged = 0;
		if (logged++ < 8) Log("[fg] prepare failed (%u)", r);
		return;
	}
	if (!g_logged_on) Log("[fg] generation on: x2, render %ux%u -> display %ux%u", render.width, render.height, display.width, display.height);
	g_logged_on = true;
}

VkResult Present(VkQueue queue, const VkPresentInfoKHR* info) {
	// Pacing (KytyPS5's CapInterval): with a VRR display the shown rate (real x 2) is kept under the cap by holding
	// the real frame here, on DXVK's present thread; with dxgi.maxFrameLatency = 1 (dxvk.conf) the game's own frame
	// waits behind it instead of queueing (the anti-lag idea: no frames buffered in front of the cap).
	const double cap = g_stats.generating ? CapFps() : 0.0;
	g_stats.cap_fps = static_cast<float>(cap);
	if (cap > 0) {
		const auto step = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(2.0 / cap));
		auto now = Clock::now();
		if (g_next_real > now) {
			g_slept_ms += std::chrono::duration<double, std::milli>(g_next_real - now).count();
			std::this_thread::sleep_until(g_next_real - std::chrono::milliseconds(1));
			while (Clock::now() < g_next_real) std::this_thread::yield();
			now = g_next_real;
		}
		g_next_real = std::max(g_next_real, now - step) + step;
	}
	const auto begin = Clock::now();
	VkResult r = g_fn.pOutQueuePresentKHR(queue, info);
	const auto end = Clock::now();
	g_present_ms += std::chrono::duration<double, std::milli>(end - begin).count();
	++g_real;
	if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) r = VK_ERROR_OUT_OF_DATE_KHR;

	// Once a second: the proof. "shown" is the frame interpolation swapchain's own count of presents to the display
	// (real + generated); "generated" the generation dispatches FidelityFX asked us to run. x2 working: shown = 2 x real.
	if (g_window_start == Clock::time_point {}) {
		g_window_start = end;
		g_presents_at_start = g_fn.pOutGetLastPresentCountFFXAPI ? g_fn.pOutGetLastPresentCountFFXAPI(g_swapchain) : 0;
	}
	const double window = std::chrono::duration<double>(end - g_window_start).count();
	if (window >= 1.0) {
		const uint64_t presents = g_fn.pOutGetLastPresentCountFFXAPI ? g_fn.pOutGetLastPresentCountFFXAPI(g_swapchain) : 0;
		const uint64_t shown = presents >= g_presents_at_start ? presents - g_presents_at_start : 0;
		const uint32_t gen = g_dispatches.exchange(0);
		g_stats.real_fps  = static_cast<float>(g_real / window);
		g_stats.shown_fps = static_cast<float>(shown / window);
		g_stats.ratio     = g_real ? static_cast<float>(shown) / g_real : 0.0f;
		g_stats.generated = gen;
		Log("[fg] real %.1f fps -> shown %.1f fps (x%.2f) | generated %u | %s | present %.2f ms | cap %.0f fps, held %.1f ms",
		    g_stats.real_fps, g_stats.shown_fps, g_stats.ratio, gen, g_stats.state, g_real ? g_present_ms / g_real : 0.0, cap,
		    g_slept_ms);
		g_window_start = end;
		g_presents_at_start = presents;
		g_real = 0;
		g_present_ms = g_slept_ms = 0;
	}
	return r;
}

void OnDeviceDestroyed() {
	std::lock_guard lock(QueueLock());
	if (g_sc_ctx != nullptr) DestroySwapchain(g_swapchain);
	DestroyFg();
	DestroyOwned(g_neutral_depth);
	DestroyOwned(g_neutral_motion);
	g_fg_failed = false;
	g_inputs_mode_known = false;
	g_last_real = g_window_start = g_next_real = {};
	g_stats = {};
}

Stats LastStats() {
	return g_stats;
}

} // namespace inj::framegen
