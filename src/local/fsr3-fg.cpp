#include "local/port-live.h"
#include "fsr3-fg.h"

#include "frame-gen.h"
#include "graphics/presentation/videoOut.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>
#include "gpu_tiler_shaders/hud_split_spv.h"
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
#include "local/vulkan-recording.h"
#include "local/dyn-multiplier.h"
#endif
#endif

namespace Libs::Graphics::Fsr3 {

#if defined(_WIN32)
namespace {

bool         g_requested = false;
HMODULE      g_dll       = nullptr;
ffxFunctions g_ffx {};

// Queues (OnDevice).
GraphicContext* g_graphics       = nullptr;
VkQueue         g_present_queue  = VK_NULL_HANDLE; // FI present: graphics family
VkQueue         g_async_queue    = VK_NULL_HANDLE; // compute family, index 0
VkQueue         g_acquire_queue  = VK_NULL_HANDLE; // compute family, index 1
uint32_t        g_compute_family = UINT32_MAX;

// The driver's functions, not the recording layer's.
PFN_vkGetDeviceProcAddr g_get_device_proc = nullptr;
PFN_vkQueueSubmit       g_queue_submit    = nullptr;

// The swapchain context and its replacement functions.
ffxContext                                g_swapchain_context = nullptr;
ffxQueryDescSwapchainReplacementFunctionsVK g_functions {};

// The frame generation context, made for one display size and format.
ffxContext   g_fg_context = nullptr;
vk::Extent2D g_fg_display {};
vk::Extent2D g_fg_max_render {};
vk::Format   g_fg_format = vk::Format::eUndefined;
bool         g_fg_failed = false;

// The prepare pass's command buffers (a ring, each reused once its fence signaled).
struct Slot {
	VkCommandBuffer command = VK_NULL_HANDLE;
	VkFence         fence   = VK_NULL_HANDLE;
	bool            pending = false;
};
VkCommandPool         g_pool = VK_NULL_HANDLE;
std::array<Slot, 4>   g_slots {};
uint32_t              g_slot = 0;

// The HUD-less image at the backbuffer's size and format (the frame's copy scaled as the backbuffer's).
VulkanImage g_present_hudless;
bool        g_present_hudless_ready = false;

// The HUD layer (RecordHudLayer): the game's HUD as the real frame shows it, transparent elsewhere, at the backbuffer's
// size; FSR 3 composites it over every frame shown. The backbuffer then holds the frame without its HUD.
struct HudLayer {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             sampler    = nullptr;
	VulkanImage             image;
	vk::ImageView           view       = nullptr;
	bool                    failed     = false;
	bool                    written    = false; // holds a frame's HUD
	bool                    this_frame = false; // the backbuffer holds this frame's HUD-less image
	struct View {
		VkImage       image  = VK_NULL_HANDLE;
		vk::Format    format = vk::Format::eUndefined;
		vk::ImageView view   = nullptr;
	};
	std::array<View, 6> views {}; // the presenter's frame images
	uint32_t            next_view = 0;
	uint32_t            logs      = 0;
} g_hud;

bool HudLayerWanted() {
	static const bool wanted = [] {
		// (Off by default: FSR 3's own composition did not show the layer, the HUD was gone. The HUD-less image the
		// game's composite gives every frame keeps the HUD out of the generated frames: confirmed in play, 10-11.)
		const char* value = std::getenv("KYTY_FSR3_HUD_LAYER");
		return value != nullptr && *value == '1';
	}();
	return wanted;
}

uint64_t g_frame_id      = 0;
bool     g_generating    = false; // the last frame had generation on
int      g_state_logged  = -1;
std::chrono::steady_clock::time_point g_last_real {};

// Per-second statistics (the run log).
struct Stats {
	std::chrono::steady_clock::time_point start {};
	uint32_t real = 0, generated_on = 0, no_inputs = 0, movie = 0;
	uint64_t presents_at_start = 0;
	double   prepare_ms = 0, present_ms = 0, max_present_ms = 0;
} g_stats;

std::mutex g_mutex; // the contexts are not thread safe (presents, prepare, destruction)

bool EnvOn(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	return *value != '0';
}
float EnvFloat(const char* name, float fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? static_cast<float>(std::atof(value)) : fallback;
}

// AMD-side multi frame generation (patched amd_fidelityfx_vk.dll, tools/local/fsr3-mfg: frame k of N-1 interpolated
// at t = k / N, presents paced 1 / N of the real frame time). KYTY_FSR3_MFG=N (2..4) is the multiplier (it sets
// KYTY_FSR3_FG_FRAMES = N - 1 for the DLL, read when it makes the swapchain). KYTY_FSR3_DYNAMIC=1: the multiplier
// (2..max, max = KYTY_FSR3_MFG, default 4; the patched DLL holds 3 generated frames at most) picked once a second:
// the largest N with real fps x N <= the target (KYTY_FSR3_TARGET_FPS, default refresh - refresh^2 / 3600), with
// hysteresis (10-22, MfgUpdate; no flip floor); real frames limited to target / N.
// The DLL reads KYTY_FSR3_FG_ACTIVE (generated frames, 1..max) on every present. Both unset: nothing changes.
struct Mfg {
	bool   on = false, dynamic = false;
	int    max = 2;       // multiplier maximum (frames allocated in the DLL)
	int    current = 2;   // multiplier in use
	int    pending = 0, pending_n = 0;
	double target = 0.0;  // shown fps target (dynamic)
	double slept  = 0.0;  // seconds the limiter waited in this window
	double smooth = 0.0;  // 10-22: smoothed real fps
	int    up_n = 0, pinned_n = 0, noup_s = 0; // 10-22: hysteresis windows (as DLSS-G's dynamic multiplier)
	Kyty::Local::DynMultiplier dm; // 10-22: the free-rate multiplier choice
	int    flip_fps = 0;  // 10-21: the game's own flip floor for this multiplier (VideoOut::SetDynamicFlipFps)
} g_mfg;

// 10-21: the multiplier `n` at 60 real passing the target: the game's own flips held to the vblank's even reach rate
// (840 Hz: 56, every 15 vblanks) instead of our limiter (that beat against the game's 60: a 4 Hz judder).
void MfgApplyFlipFloor(int n) {
	const int want = 60.0 * n > g_mfg.target + 0.5 ? Libs::VideoOut::ReachFps(g_mfg.target, n) : 0;
	const int fps  = Libs::VideoOut::SetDynamicFlipFps(want >= 60 ? 0 : want);
	if (fps != g_mfg.flip_fps) {
		if (fps > 0) std::printf("[dyn] x%d: game flips at %d (gameplay)\n", n, fps);
		else std::printf("[dyn] x%d: game flips at 60\n", n);
		std::fflush(stdout);
	}
	g_mfg.flip_fps = fps;
}

double RefreshTarget() {
	DEVMODEA mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 30) {
		const double hz = mode.dmDisplayFrequency;
		return hz - hz * hz / 3600.0;
	}
	return 0.0;
}

void MfgSetActive(int multiplier) {
	char text[8];
	std::snprintf(text, sizeof(text), "%d", multiplier - 1);
	SetEnvironmentVariableA("KYTY_FSR3_FG_ACTIVE", text);
}

void MfgInit() {
	static bool done = false;
	if (done) return;
	done = true;
	const char* n   = std::getenv("KYTY_FSR3_MFG");
	const char* dyn = std::getenv("KYTY_FSR3_DYNAMIC");
	g_mfg.dynamic   = dyn != nullptr && *dyn == '1';
	if ((n == nullptr || *n == '\0') && !g_mfg.dynamic) return;
	g_mfg.on      = true;
	g_mfg.max     = std::clamp(n != nullptr && *n != '\0' ? std::atoi(n) : 4, 2, 4);
	g_mfg.current = g_mfg.max;
	char frames[8];
	std::snprintf(frames, sizeof(frames), "%d", g_mfg.max - 1);
	_putenv_s("KYTY_FSR3_FG_FRAMES", frames); // (CRT and process environment: the DLL reads the latter)
	if (g_mfg.dynamic) {
		const float target = EnvFloat("KYTY_FSR3_TARGET_FPS", 0.0f);
		g_mfg.target       = target > 0.0f ? target : RefreshTarget();
		if (g_mfg.target <= 0.0) g_mfg.target = 60.0 * g_mfg.max;
		g_mfg.current = 2;
	}
	MfgSetActive(g_mfg.current);
	std::printf("[fsr3-mfg] %s, up to x%d (patched FidelityFX DLL only; AMD's signed DLL stays x2)\n",
	            g_mfg.dynamic ? "dynamic multiplier" : "fixed multiplier", g_mfg.max);
	if (g_mfg.dynamic) std::printf("[fsr3-mfg] target %.0f fps shown\n", g_mfg.target);
	std::fflush(stdout);
}

// Once a second (dynamic): real = frames / window, free = frames / (window - limiter wait).
void MfgUpdate(uint32_t frames, double window, double shown) {
	if (!g_mfg.on) return;
	// (The PC port's menu: a fixed multiplier chosen while playing, up to the launch's.)
	if (const int live = PortLive::fg_generate.load(std::memory_order_relaxed); live > 0 && !g_mfg.dynamic) {
		const int want = std::clamp(live + 1, 2, g_mfg.max);
		if (want != g_mfg.current) {
			g_mfg.current = want;
			MfgSetActive(want);
		}
	}
	PortLive::fg_now.store(g_mfg.current, std::memory_order_relaxed);
	const double real = frames / window;
	const double busy = std::max(window - g_mfg.slept, 0.05);
	const double free = frames / busy;
	g_mfg.slept       = 0.0;
	if (g_mfg.dynamic && frames > 0) {
		// 10-22: no flip floor; the multiplier from the free rate (dyn-multiplier.h), real capped at target / m.
		const bool   cinematic = Libs::VideoOut::CinematicPlaying();
		const double T         = g_mfg.target;
		const int    maxm      = cinematic ? std::min(3, g_mfg.max) : g_mfg.max;
		g_mfg.smooth           = g_mfg.smooth <= 0.0 ? real : 0.5 * g_mfg.smooth + 0.5 * real;
		// 10-22: decided on the free rate (src/local/dyn-multiplier.h; see there for the rules).
		const int next = std::clamp(g_mfg.dm.Window(g_mfg.current, real, T, maxm,
		                                               cinematic || Libs::VideoOut::CutsceneHeld() || Libs::VideoOut::MoviePlaying(), free), 2, std::max(maxm, 2));
		(void)free;
		if (next != g_mfg.current) {
			const int from = g_mfg.current;
			g_mfg.current  = next; // (CapInterval reads it: the pacer's target / m changes with it)
			g_mfg.pinned_n = 0;
			MfgSetActive(next);
			if (g_mfg.flip_fps != 0) {
				Libs::VideoOut::SetDynamicFlipFps(0);
				g_mfg.flip_fps = 0;
			}
			std::printf("[dyn] x%d (from x%d): real %.1f fps, shown %.1f (target %.0f), free %.1f; pacer interval %.0f us for x%d\n",
			            next, from, real, std::min(real * from, T), T, g_mfg.dm.last_free, 1e6 * next / T, next);
		}
	}
	std::printf("[fsr3-mfg] x%d%s; real %.1f fps (free %.1f), shown %.1f fps\n", g_mfg.current, g_mfg.dynamic ? " dynamic" : "",
	            real, free, shown);
	if (g_mfg.dynamic) std::printf("[fsr3-mfg] target %.0f, real limit %.1f\n", g_mfg.target, g_mfg.target / g_mfg.current);
	std::fflush(stdout);
}

void Message(uint32_t type, const wchar_t* message) {
	static std::atomic<uint32_t> count {0};
	if (count.fetch_add(1, std::memory_order_relaxed) > 200) return;
	std::printf("FSR3 FG: FidelityFX message (%u): %ls\n", type, message);
	std::fflush(stdout);
}

VkResult GameQueueSubmit(uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	Common::LockGuard lock(g_graphics->queue_mutex);
	return g_queue_submit(static_cast<VkQueue>(g_graphics->queue), count, submits, fence);
}

FfxApiResource ImageResource(const VulkanImage& image, bool depth) {
	FfxApiResource resource {};
	resource.resource                = static_cast<void*>(static_cast<VkImage>(image.image));
	resource.state                   = FFX_API_RESOURCE_STATE_COMPUTE_READ;
	auto& desc                       = resource.description;
	desc.type                        = FFX_API_RESOURCE_TYPE_TEXTURE2D;
	desc.format                      = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
	desc.width                       = image.extent.width;
	desc.height                      = image.extent.height;
	desc.depth                       = 1;
	desc.mipCount                    = 1;
	desc.flags                       = FFX_API_RESOURCE_FLAGS_NONE;
	desc.usage                       = depth ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : FFX_API_RESOURCE_USAGE_READ_ONLY;
	return resource;
}

void DestroyFrameGeneration() {
	if (g_fg_context != nullptr) {
		// Off in the swapchain first (it calls the context while presenting).
		ffxConfigureDescFrameGeneration off {};
		off.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
		off.frameGenerationEnabled = false;
		g_ffx.Configure(&g_fg_context, &off.header);
		g_ffx.DestroyContext(&g_fg_context, nullptr);
		g_fg_context = nullptr;
	}
}

bool CreateFrameGeneration(GraphicContext& graphics, vk::Extent2D display, vk::Extent2D render, vk::Format format) {
	DestroyFrameGeneration();
	ffxCreateBackendVKDesc backend {};
	backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend.vkDevice         = static_cast<VkDevice>(graphics.device);
	backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
	backend.vkDeviceProcAddr = g_get_device_proc;

	ffxCreateContextDescFrameGeneration create {};
	create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
	create.header.pNext = &backend.header;
	// The game's depth is reversed (1 near, 0 far) with an infinite far plane (KYTY_FSR3_DEPTH=normal: as is).
	const char* depth   = std::getenv("KYTY_FSR3_DEPTH");
	const bool  normal  = depth != nullptr && std::string(depth) == "normal";
	create.flags        = normal ? 0u : (FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE);
	if (EnvOn("KYTY_FSR3_DEBUG", false)) create.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
	// --hdr: an HDR10 (PQ) swapchain's backbuffer holds high-dynamic-range color (the context is made again when the
	// swapchain's format changes between SDR and HDR10).
	if (format == vk::Format::eA2B10G10R10UnormPack32 || format == vk::Format::eA2R10G10B10UnormPack32)
		create.flags |= FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
	create.displaySize      = {display.width, display.height};
	create.maxRenderSize    = {render.width, render.height};
	create.backBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(format));
	const auto result       = g_ffx.CreateContext(&g_fg_context, &create.header, nullptr);
	if (result != FFX_API_RETURN_OK) {
		std::printf("FSR3 FG: frame generation context not created (%u): generation stays off\n", result);
		std::fflush(stdout);
		g_fg_context = nullptr;
		g_fg_failed  = true;
		return false;
	}
	g_fg_display    = display;
	g_fg_max_render = render;
	g_fg_format     = format;
	std::printf("FSR3 FG: frame generation context %ux%u (render up to %ux%u, depth %s)\n", display.width, display.height,
	            render.width, render.height, normal ? "as is" : "reversed, infinite");
	std::fflush(stdout);
	return true;
}

bool EnsurePresentHudless(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (g_present_hudless.image != nullptr && g_present_hudless.format == format &&
	    g_present_hudless.extent.width == extent.width && g_present_hudless.extent.height == extent.height)
		return true;
	if (g_present_hudless.image != nullptr) graphics.DeleteImage(g_present_hudless);
	g_present_hudless.image = nullptr;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_present_hudless)) {
		g_present_hudless.image = nullptr;
		return false;
	}
	return true;
}

// The frame's HUD-less copy scaled into g_present_hudless (as the presenter scales the frame into the backbuffer).
void RecordHudlessBlit(VkDevice device, VkCommandBuffer command, const VulkanImage& source) {
	static auto* barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(g_get_device_proc(device, "vkCmdPipelineBarrier"));
	static auto* blit    = reinterpret_cast<PFN_vkCmdBlitImage>(g_get_device_proc(device, "vkCmdBlitImage"));
	VkImageMemoryBarrier barriers[2] {};
	for (auto& b: barriers) {
		b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	}
	barriers[0].image         = static_cast<VkImage>(source.image);
	barriers[0].oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barriers[0].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barriers[1].image         = static_cast<VkImage>(g_present_hudless.image);
	barriers[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
	barriers[1].newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
	VkImageBlit region {};
	region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	region.srcOffsets[1]  = {static_cast<int32_t>(source.extent.width), static_cast<int32_t>(source.extent.height), 1};
	region.dstOffsets[1]  = {static_cast<int32_t>(g_present_hudless.extent.width),
	                         static_cast<int32_t>(g_present_hudless.extent.height), 1};
	blit(command, static_cast<VkImage>(source.image), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	     static_cast<VkImage>(g_present_hudless.image), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
	barriers[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barriers[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barriers[1].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barriers[1].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
}

bool EnsureHudPipeline(GraphicContext& graphics) {
	if (g_hud.pipeline != nullptr) return true;
	if (g_hud.failed) return false;
	g_hud.failed = true; // until it is made
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t i = 0; i < 3; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_hud.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 24};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_hud.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_hud.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(HUD_SPLIT_SPV);
	module_info.pCode    = HUD_SPLIT_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_hud.layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_hud.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_hud.pipeline = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eLinear;
	sampler.minFilter    = vk::Filter::eLinear;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = 0.0f;
	if (device.createSampler(&sampler, nullptr, &g_hud.sampler) != vk::Result::eSuccess) return false;
	g_hud.failed = false;
	return true;
}

bool EnsureHudImage(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_hud.image.image != nullptr && g_hud.image.extent.width == extent.width && g_hud.image.extent.height == extent.height)
		return true;
	if (g_hud.view != nullptr) graphics.device.destroyImageView(g_hud.view, nullptr);
	g_hud.view = nullptr;
	if (g_hud.image.image != nullptr) graphics.DeleteImage(g_hud.image);
	g_hud.image.image = nullptr;
	g_hud.written     = false;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = vk::Format::eR16G16B16A16Sfloat;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_hud.image)) {
		g_hud.image.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view {};
	view.image            = g_hud.image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = create.format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view, nullptr, &g_hud.view) != vk::Result::eSuccess) {
		graphics.DeleteImage(g_hud.image);
		g_hud.image.image = nullptr;
		g_hud.view        = nullptr;
		return false;
	}
	g_hud.image.state = {};
	return true;
}

vk::ImageView FrameImageView(GraphicContext& graphics, const VulkanImage& image) {
	const auto handle = static_cast<VkImage>(image.image);
	for (const auto& view: g_hud.views)
		if (view.image == handle && view.format == image.format && view.view != nullptr) return view.view;
	auto& slot = g_hud.views[g_hud.next_view];
	g_hud.next_view = (g_hud.next_view + 1) % static_cast<uint32_t>(g_hud.views.size());
	if (slot.view != nullptr) graphics.device.destroyImageView(slot.view, nullptr);
	slot = {};
	vk::ImageViewCreateInfo view {};
	view.image            = image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = image.format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageView created = nullptr;
	if (graphics.device.createImageView(&view, nullptr, &created) != vk::Result::eSuccess) return nullptr;
	slot = {handle, image.format, created};
	return created;
}

vk::ImageMemoryBarrier2 ImageBarrier(vk::Image image, vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags2 src_access,
                                     vk::AccessFlags2 dst_access) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	return barrier;
}
void Barriers(vk::CommandBuffer command, std::initializer_list<vk::ImageMemoryBarrier2> barriers) {
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.begin();
	command.pipelineBarrier2(dependency);
}

bool EnsureCommandBuffers(GraphicContext& graphics) {
	if (g_pool != VK_NULL_HANDLE) return true;
	const auto device = static_cast<VkDevice>(graphics.device);
	auto*      create_pool = reinterpret_cast<PFN_vkCreateCommandPool>(g_get_device_proc(device, "vkCreateCommandPool"));
	auto*      allocate    = reinterpret_cast<PFN_vkAllocateCommandBuffers>(g_get_device_proc(device, "vkAllocateCommandBuffers"));
	auto*      create_fence = reinterpret_cast<PFN_vkCreateFence>(g_get_device_proc(device, "vkCreateFence"));
	VkCommandPoolCreateInfo pool {};
	pool.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool.queueFamilyIndex = graphics.queue_family;
	if (create_pool(device, &pool, nullptr, &g_pool) != VK_SUCCESS) return false;
	for (auto& slot: g_slots) {
		VkCommandBufferAllocateInfo info {};
		info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool        = g_pool;
		info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		if (allocate(device, &info, &slot.command) != VK_SUCCESS) return false;
		VkFenceCreateInfo fence {};
		fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		if (create_fence(device, &fence, nullptr, &slot.fence) != VK_SUCCESS) return false;
	}
	return true;
}

// The camera's vertical field of view and near plane from the world->clip matrix (row-major, clip = M * world),
// and its axes and position (FSR 3.1.4's camera info). False: no camera this frame.
struct CameraValues {
	float fov = 1.0f, z_near = 0.1f;
	float position[3] {}, up[3] {0, 1, 0}, right[3] {1, 0, 0}, forward[3] {0, 0, 1};
	bool  valid = false;
};
CameraValues FromCamera(const FrameGen::Camera& camera) {
	CameraValues out;
	if (!camera.valid) return out;
	const auto& m   = camera.view_proj;
	const auto  row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto  r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return out;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf = r3[3] / flen;
	const auto split = [&](const std::array<double, 4>& r, double& s, double axis[3], double& t) {
		const double off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
		double v[3] {r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
		s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (s < 1e-9) return false;
		for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
		t = (r[3] / flen - off * tf) / s;
		return true;
	};
	double sx = 0, sy = 0, x[3] {}, y[3] {}, tx = 0, ty = 0;
	if (!split(r0, sx, x, tx) || !split(r1, sy, y, ty)) return out;
	const double up[3] {f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2], f[0] * x[1] - f[1] * x[0]};
	if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) ty = -ty;
	for (int i = 0; i < 3; ++i) {
		out.right[i]    = static_cast<float>(x[i]);
		out.up[i]       = static_cast<float>(up[i]);
		out.forward[i]  = static_cast<float>(f[i]);
		out.position[i] = static_cast<float>(-(tx * x[i] + ty * up[i] + tf * f[i]));
	}
	out.fov    = static_cast<float>(2.0 * std::atan(1.0 / std::fabs(sy)));
	const double z_near = std::fabs(r2[3] / flen);
	out.z_near = z_near > 1e-4 && z_near < 100.0 ? static_cast<float>(z_near) : 0.1f;
	out.valid  = std::isfinite(out.fov) && out.fov > 0.1f && out.fov < 3.0f;
	return out;
}

void LogState(int state, const char* text) {
	if (state == g_state_logged) return;
	g_state_logged = state;
	std::printf("FSR3 FG: %s\n", text);
	std::fflush(stdout);
}

} // namespace

// The FidelityFX API's DLL (amd_fidelityfx_vk.dll next to the emulator), loaded once: frame generation or upscaling.
static bool LoadApi() {
	static const bool loaded = [] {
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		std::wstring dir(exe);
		dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
		for (const auto* name: {L"amd_fidelityfx_vk.dll", L"fsr3\\amd_fidelityfx_vk.dll"}) {
			g_dll = LoadLibraryW((dir + name).c_str());
			if (g_dll != nullptr) break;
		}
		if (g_dll == nullptr) {
			std::printf("FSR3: amd_fidelityfx_vk.dll not found next to the emulator\n");
			std::fflush(stdout);
			return false;
		}
		ffxLoadFunctions(&g_ffx, g_dll);
		if (g_ffx.CreateContext == nullptr || g_ffx.Configure == nullptr || g_ffx.Dispatch == nullptr) {
			std::printf("FSR3: amd_fidelityfx_vk.dll has no FidelityFX API\n");
			std::fflush(stdout);
			return false;
		}
		return true;
	}();
	return loaded;
}

bool Requested() {
	static const bool requested = [] {
		if (!EnvOn("KYTY_FSR3_FG", false)) return false;
		if (!LoadApi()) {
			std::printf("FSR3 FG: frame generation off\n");
			std::fflush(stdout);
			return false;
		}
		std::printf("FSR3 FG: requested (KYTY_FSR3_FG), FidelityFX API loaded\n");
		std::fflush(stdout);
		return true;
	}();
	g_requested = requested;
	return requested;
}

bool PlanQueues(vk::PhysicalDevice physical_device, uint32_t queue_family, uint32_t graphics_count, QueuePlan& plan) {
	if (!Requested()) return false;
	uint32_t family_count = 0;
	physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	physical_device.getQueueFamilyProperties(&family_count, families.data());
	if (queue_family >= family_count || families[queue_family].queueCount < graphics_count + 1) {
		std::printf("FSR3 FG: no free graphics queue for its presents: frame generation off\n");
		std::fflush(stdout);
		g_requested = false;
		return false;
	}
	plan.graphics_index = graphics_count;
	for (uint32_t family = 0; family < family_count; ++family) {
		const auto flags = families[family].queueFlags;
		if (family != queue_family && (flags & vk::QueueFlagBits::eCompute) && !(flags & vk::QueueFlagBits::eGraphics) &&
		    families[family].queueCount >= 2) {
			plan.compute_family = family;
			plan.compute_count  = 2;
			break;
		}
	}
	if (plan.compute_family == UINT32_MAX) {
		std::printf("FSR3 FG: no async compute queue family: frame generation off\n");
		std::fflush(stdout);
		g_requested = false;
		return false;
	}
	return true;
}

void OnDevice(GraphicContext& graphics, const QueuePlan& plan) {
	if (g_requested || ((UpscaleWanted() || SrPathWanted()) && LoadApi())) {
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
		g_get_device_proc = LocalVulkanRecording::DirectDispatch().vkGetDeviceProcAddr;
#endif
		if (g_get_device_proc == nullptr) g_get_device_proc = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	}
	if (!g_requested || plan.compute_family == UINT32_MAX) return;
	g_graphics = &graphics;
	const auto device = static_cast<VkDevice>(graphics.device);
	g_queue_submit    = reinterpret_cast<PFN_vkQueueSubmit>(g_get_device_proc(device, "vkQueueSubmit"));
	auto* get_queue   = reinterpret_cast<PFN_vkGetDeviceQueue>(g_get_device_proc(device, "vkGetDeviceQueue"));
	get_queue(device, graphics.queue_family, plan.graphics_index, &g_present_queue);
	get_queue(device, plan.compute_family, 0, &g_async_queue);
	get_queue(device, plan.compute_family, 1, &g_acquire_queue);
	g_compute_family = plan.compute_family;
	if (g_present_queue == VK_NULL_HANDLE || g_async_queue == VK_NULL_HANDLE || g_acquire_queue == VK_NULL_HANDLE ||
	    g_queue_submit == nullptr) {
		std::printf("FSR3 FG: its queues were not created: frame generation off\n");
		std::fflush(stdout);
		g_requested = false;
		return;
	}
	ffxConfigureDescGlobalDebug1 debug {};
	debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
	debug.debugLevel  = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
	debug.fpMessage   = &Message;
	g_ffx.Configure(nullptr, &debug.header);
	std::printf("FSR3 FG: queues: present (graphics family %u, queue %u), async compute and image acquire (family %u)\n",
	            graphics.queue_family, plan.graphics_index, plan.compute_family);
	std::fflush(stdout);
}

bool Active() {
	return g_swapchain_context != nullptr;
}

bool CreateSwapchain(GraphicContext& graphics, const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out) {
	if (!g_requested || g_graphics == nullptr) return false;
	std::lock_guard lock(g_mutex);
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	// (A new context each time: one re-created through the context's own replacement function presented nothing,
	// a black screen after the window went fullscreen.)
	ffxCreateContextDescFrameGenerationSwapChainVK create {};
	create.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
	create.physicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
	create.device         = static_cast<VkDevice>(graphics.device);
	create.swapchain      = &swapchain;
	create.allocator      = nullptr;
	create.createInfo     = static_cast<VkSwapchainCreateInfoKHR>(info);
	create.gameQueue      = {static_cast<VkQueue>(graphics.queue), graphics.queue_family, &GameQueueSubmit};
	create.asyncComputeQueue = {g_async_queue, g_compute_family, nullptr};
	create.presentQueue      = {g_present_queue, graphics.queue_family, nullptr};
	create.imageAcquireQueue = {g_acquire_queue, g_compute_family, nullptr};
	MfgInit();
	{
		// Pacing floor (the patched amd_fidelityfx_vk.dll, tools/local/fsr3-mfg; AMD's signed DLL ignores it): no present
		// of the frame interpolation swapchain closer than the display's refresh period x KYTY_FSR3_PACING_FLOOR (default 1.0,
		// 0 off). Its pacer spaces presents by (real frame time) / (generated + 1) less the variance and a 0.1 ms margin:
		// at x4 from 59-60 real fps that is under 4.17 ms on a 240 Hz display, a present above the G-SYNC ceiling.
		static bool logged = false;
		double      factor = 1.0;
		if (const char* text = std::getenv("KYTY_FSR3_PACING_FLOOR"); text != nullptr && *text != '\0') factor = std::strtod(text, nullptr);
		DEVMODEA mode {};
		mode.dmSize = sizeof(mode);
		const bool refresh = EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 30;
		if (factor > 0.0 && refresh) {
			const double floor_us = 1000000.0 / mode.dmDisplayFrequency * factor;
			SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", std::to_string(floor_us).c_str());
			if (!logged) std::printf("FSR3 FG: present pacing floor %.3f ms (display %lu Hz x %.2f; KYTY_FSR3_PACING_FLOOR=0 off)\n",
			                         floor_us / 1000.0, static_cast<unsigned long>(mode.dmDisplayFrequency), factor);
		} else {
			SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", nullptr);
			if (!logged) std::printf("FSR3 FG: no present pacing floor (%s)\n", refresh ? "KYTY_FSR3_PACING_FLOOR=0" : "display refresh unknown");
		}
		logged = true;
		std::fflush(stdout);
	}
	const auto result        = g_ffx.CreateContext(&g_swapchain_context, &create.header, nullptr);
	if (result != FFX_API_RETURN_OK || swapchain == VK_NULL_HANDLE) {
		std::printf("FSR3 FG: frame generation swapchain not created (%u): frame generation off\n", result);
		std::fflush(stdout);
		g_swapchain_context = nullptr;
		g_requested         = false;
		return false;
	}
	{
		// x3 (KYTY_FSR3_FG_FRAMES=2): read by the patched amd_fidelityfx_vk.dll (fg/x3 builds, see
		// tools/local/fsr3-mfg) when it creates the swapchain; AMD's signed DLL ignores it and stays at x2.
		const char* frames = std::getenv("KYTY_FSR3_FG_FRAMES");
		if (frames != nullptr && std::atoi(frames) >= 2) {
			const int generated = std::atoi(frames) >= 3 ? 3 : 2;
			std::printf("FSR3 FG: KYTY_FSR3_FG_FRAMES=%s: %d generated frames per real frame (x%d, patched DLL only)\n", frames,
			            generated, generated + 1);
			std::fflush(stdout);
		}
	}
	g_functions.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
	g_ffx.Query(&g_swapchain_context, &g_functions.header);
	if (g_functions.pOutAcquireNextImageKHR == nullptr || g_functions.pOutQueuePresentKHR == nullptr ||
	    g_functions.pOutGetSwapchainImagesKHR == nullptr) {
		std::printf("FSR3 FG: no replacement swapchain functions: frame generation off\n");
		std::fflush(stdout);
		g_ffx.DestroyContext(&g_swapchain_context, nullptr);
		g_swapchain_context = nullptr;
		g_requested         = false;
		return false;
	}
	std::printf("FSR3 FG: frame generation swapchain %ux%u (present mode %d)\n", info.imageExtent.width,
	            info.imageExtent.height, static_cast<int>(info.presentMode));
	std::fflush(stdout);
	*out = swapchain;
	return true;
}

void DestroySwapchain(GraphicContext& /*graphics*/, vk::SwapchainKHR /*swapchain*/) {
	std::lock_guard lock(g_mutex);
	if (g_swapchain_context == nullptr) return;
	// The frame generation context first (the swapchain calls it while presenting); destroying the swapchain's
	// context destroys its swapchain. The next CreateSwapchain makes both again.
	DestroyFrameGeneration();
	g_ffx.DestroyContext(&g_swapchain_context, nullptr);
	g_swapchain_context = nullptr;
	g_functions         = {};
	g_generating        = false;
	g_state_logged      = -1;
	g_stats             = {};
	std::printf("FSR3 FG: frame generation swapchain released (resize, fullscreen or exit)\n");
	std::fflush(stdout);
}

vk::Result GetSwapchainImages(GraphicContext& graphics, vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
	if (g_functions.pOutGetSwapchainImagesKHR == nullptr) return vk::Result::eErrorOutOfDateKHR;
	return static_cast<vk::Result>(g_functions.pOutGetSwapchainImagesKHR(
	    static_cast<VkDevice>(graphics.device), static_cast<VkSwapchainKHR>(swapchain), count, reinterpret_cast<VkImage*>(images)));
}

vk::Result AcquireNextImage(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
	if (g_functions.pOutAcquireNextImageKHR == nullptr) return vk::Result::eErrorOutOfDateKHR;
	return static_cast<vk::Result>(g_functions.pOutAcquireNextImageKHR(static_cast<VkDevice>(graphics.device),
	                                                                   static_cast<VkSwapchainKHR>(swapchain), UINT64_MAX,
	                                                                   static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE, index));
}

vk::Result QueuePresent(GraphicContext& graphics, const vk::PresentInfoKHR& info) {
	std::lock_guard lock(g_mutex);
	// 10-25: the swapchain released (resize, fullscreen, exit) between the presenter's acquire and this present: no
	// call into a destroyed context; out of date makes the presenter create it again.
	if (g_swapchain_context == nullptr || g_functions.pOutQueuePresentKHR == nullptr) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("FSR3 FG: present without a frame generation swapchain: skipped, swapchain made again\n");
			std::fflush(stdout);
		}
		return vk::Result::eErrorOutOfDateKHR;
	}
	const auto      begin   = std::chrono::steady_clock::now();
	VkPresentInfoKHR present = info;
	present.pNext            = nullptr;
	// (10-25: the crash minutes into play here, a write to 0x38 in amd_fidelityfx_vk.dll, was its command list pool
	// running dry: VulkanCommandPool::get returned null with all 8 lists of the shared graphics family in flight.
	// Fixed in the DLL, tools/local/fsr3-mfg/ffx-sdk-v1.1.4-mfg.patch; AMD's signed DLL keeps the 8-list pool.)
	auto result = g_functions.pOutQueuePresentKHR(static_cast<VkQueue>(graphics.queue), &present);
	const double ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	g_stats.present_ms += ms;
	g_stats.max_present_ms = std::max(g_stats.max_present_ms, ms);
	if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
		// Out of memory inside FidelityFX: the swapchain (and with it the generation context) made again instead of
		// the presenter's exit on an unexpected result.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("FSR3 FG: frame generation present out of %s memory (%d): swapchain and generation context made again\n",
			            result == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "video" : "host", static_cast<int>(result));
			std::fflush(stdout);
		}
		result = VK_ERROR_OUT_OF_DATE_KHR;
	} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("FSR3 FG: frame generation present returned %d\n", static_cast<int>(result));
			std::fflush(stdout);
		}
	}
	return static_cast<vk::Result>(result);
}

void RecordHudLayer(GraphicContext& graphics, vk::CommandBuffer command, vk::Image backbuffer_image, vk::Extent2D backbuffer,
                    vk::Rect2D region, VulkanImage& final_image, const FrameGen::FrameInputs* inputs) {
	if (!Active() || !HudLayerWanted()) return;
	std::lock_guard lock(g_mutex);
	g_hud.this_frame = false;
	// Only a frame whose HUD-less image is the game's composite drawn without its HUD texture (the same pixels as the
	// frame but the HUD's), with generation possible.
	if (inputs == nullptr || !inputs->valid || !inputs->has_hudless || !inputs->hudless_composite ||
	    inputs->hudless.image == nullptr || inputs->hudless_view == nullptr || VideoOut::MoviePlaying() ||
	    final_image.image == nullptr || final_image.extent.width != inputs->hudless.extent.width ||
	    final_image.extent.height != inputs->hudless.extent.height)
		return;
	if (!EnsureHudPipeline(graphics) || !EnsureHudImage(graphics, backbuffer)) return;
	const auto final_view = FrameImageView(graphics, final_image);
	if (final_view == nullptr) return;

	// The layer: F where it differs from H.
	const auto layer_from = g_hud.written ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eUndefined;
	Barriers(command, {ImageBarrier(final_image.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	                                vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eShaderRead),
	                   ImageBarrier(g_hud.image.image, layer_from, vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
	                                vk::AccessFlagBits2::eShaderWrite)});
	vk::DescriptorImageInfo images[3] {};
	images[0] = {g_hud.sampler, final_view, vk::ImageLayout::eShaderReadOnlyOptimal};
	images[1] = {g_hud.sampler, inputs->hudless_view, vk::ImageLayout::eShaderReadOnlyOptimal};
	images[2] = {nullptr, g_hud.view, vk::ImageLayout::eGeneral};
	vk::WriteDescriptorSet writes[3] {};
	for (uint32_t i = 0; i < 3; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &images[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_hud.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_hud.layout, 0, 3, writes);
	struct {
		int32_t offset[2];
		int32_t size[2];
		float   threshold;
		int32_t radius;
	} push {{region.offset.x, region.offset.y},
	        {static_cast<int32_t>(region.extent.width), static_cast<int32_t>(region.extent.height)},
	        EnvFloat("KYTY_FSR3_HUD_THRESHOLD", 3.0f / 1023.0f), static_cast<int32_t>(EnvFloat("KYTY_FSR3_HUD_RADIUS", 2.0f))};
	command.pushConstants(g_hud.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch((backbuffer.width + 7) / 8, (backbuffer.height + 7) / 8, 1);

	// The backbuffer: H instead of F, scaled as the presenter scaled F (FSR 3 interpolates it: no HUD to drag).
	Barriers(command, {ImageBarrier(g_hud.image.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
	                                vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead),
	                   ImageBarrier(final_image.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
	                                vk::AccessFlagBits2::eShaderRead, vk::AccessFlagBits2::eTransferRead),
	                   ImageBarrier(inputs->hudless.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
	                                vk::AccessFlagBits2::eShaderRead, vk::AccessFlagBits2::eTransferRead),
	                   ImageBarrier(backbuffer_image, vk::ImageLayout::ePresentSrcKHR, vk::ImageLayout::eTransferDstOptimal,
	                                vk::AccessFlagBits2::eMemoryRead, vk::AccessFlagBits2::eTransferWrite)});
	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(inputs->hudless.extent.width), static_cast<int32_t>(inputs->hudless.extent.height), 1};
	blit.dstOffsets[0]  = vk::Offset3D {region.offset.x, region.offset.y, 0};
	blit.dstOffsets[1]  = vk::Offset3D {region.offset.x + static_cast<int32_t>(region.extent.width),
	                                    region.offset.y + static_cast<int32_t>(region.extent.height), 1};
	command.blitImage(inputs->hudless.image, vk::ImageLayout::eTransferSrcOptimal, backbuffer_image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
	Barriers(command, {ImageBarrier(inputs->hudless.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	                                vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eShaderRead),
	                   ImageBarrier(backbuffer_image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
	                                vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eMemoryRead)});
	g_hud.written    = true;
	g_hud.this_frame = true;
	if (g_hud.logs < 3) {
		++g_hud.logs;
		std::printf("FSR3 FG: HUD layer %ux%u from the frame and its HUD-less image (%ux%u): composited over every frame shown\n",
		            backbuffer.width, backbuffer.height, inputs->hudless.extent.width, inputs->hudless.extent.height);
		std::fflush(stdout);
	}
}

// ---- FSR 3.1 upscaling ----
namespace {
ffxContext   g_up_context = nullptr;
vk::Extent2D g_up_render {};
vk::Extent2D g_up_output {};
bool         g_up_failed = false;
uint32_t     g_up_errors = 0;
} // namespace

bool UpscaleWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_UPSCALE");
		return value != nullptr && std::string(value) == "fsr";
	}();
	return wanted;
}

bool SrPathWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_UPSCALER");
		return value != nullptr && std::string(value) == "fsr";
	}();
	return wanted;
}

bool SrReady() {
	return SrPathWanted() && !g_up_failed && LoadApi();
}

bool Upscale(GraphicContext& graphics, const UpscaleDesc& desc) {
	if (!(UpscaleWanted() || SrPathWanted()) || g_up_failed || !LoadApi() || g_get_device_proc == nullptr || desc.color == nullptr ||
	    desc.depth == nullptr || desc.motion == nullptr || desc.output == nullptr || desc.command == VK_NULL_HANDLE)
		return false;
	if (g_up_context == nullptr || desc.render.width > g_up_render.width || desc.render.height > g_up_render.height ||
	    desc.upscale.width != g_up_output.width || desc.upscale.height != g_up_output.height) {
		if (g_up_context != nullptr) {
			graphics.device.waitIdle(); // (a size change: the context's resources are released)
			g_ffx.DestroyContext(&g_up_context, nullptr);
			g_up_context = nullptr;
		}
		ffxCreateBackendVKDesc backend {};
		backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
		backend.vkDevice         = static_cast<VkDevice>(graphics.device);
		backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
		backend.vkDeviceProcAddr = g_get_device_proc;
		ffxCreateContextDescUpscale create {};
		create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		create.header.pNext = &backend.header;
		// The game's 3D image is linear HDR (RGBA16F) before its composite tonemaps it; its depth reversed, infinite.
		create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
		               FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
		if (EnvOn("KYTY_FSR3_DEBUG", false)) create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
		create.maxRenderSize  = {desc.render.width, desc.render.height};
		create.maxUpscaleSize = {desc.upscale.width, desc.upscale.height};
		create.fpMessage      = &Message;
		const auto result     = g_ffx.CreateContext(&g_up_context, &create.header, nullptr);
		if (result != FFX_API_RETURN_OK) {
			std::printf("FSR3 upscaling: context not created (%u): upscaling off\n", result);
			std::fflush(stdout);
			g_up_context = nullptr;
			g_up_failed  = true;
			return false;
		}
		g_up_render = desc.render;
		g_up_output = desc.upscale;
		std::printf("FSR3 upscaling: FSR 3.1 %ux%u -> %ux%u\n", desc.render.width, desc.render.height, desc.upscale.width,
		            desc.upscale.height);
		std::fflush(stdout);
	}
	const auto        camera  = desc.camera != nullptr ? FromCamera(*desc.camera) : CameraValues {};
	static const float mv_sign = EnvFloat("KYTY_FSR3_MV_SIGN", -1.0f) >= 0 ? 1.0f : -1.0f;
	ffxDispatchDescUpscale up {};
	up.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
	up.commandList   = desc.command;
	up.color         = ImageResource(*desc.color, false);
	up.depth         = ImageResource(*desc.depth, true);
	up.motionVectors = ImageResource(*desc.motion, false);
	up.output        = ImageResource(*desc.output, false);
	up.output.state  = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
	up.output.description.usage = FFX_API_RESOURCE_USAGE_UAV;
	// The game's TAA already resolved its jitter into the image (no jitter left); its vectors are in UV units along the
	// motion (FSR wants pixels towards the previous position, as for frame generation).
	up.jitterOffset      = {desc.jitter[0], desc.jitter[1]}; // (0 after the TAA; the camera's for the raw frame)
	up.motionVectorScale = {mv_sign * static_cast<float>(desc.render.width), mv_sign * static_cast<float>(desc.render.height)};
	// The PS5 Pro path's vectors: their own scale to pixels (KYTY_SR_MV_SIGN=-1 the other way round).
	if (desc.mv_scale[0] != 0.0f || desc.mv_scale[1] != 0.0f) {
		static const float sr_sign = EnvFloat("KYTY_SR_MV_SIGN", 1.0f) >= 0 ? 1.0f : -1.0f;
		up.motionVectorScale = {sr_sign * desc.mv_scale[0], sr_sign * desc.mv_scale[1]};
	}
	if (desc.reactive != nullptr && desc.reactive->image != nullptr) {
		// (rtparticles) The particle mask: reactive (favour the current frame there) and transparency-and-composition
		// (no locks / less history there), as FSR takes them for particles drawn without motion vectors.
		up.reactive                   = ImageResource(*desc.reactive, false);
		up.transparencyAndComposition = up.reactive;
	}
	up.renderSize        = {desc.render.width, desc.render.height};
	up.upscaleSize       = {desc.upscale.width, desc.upscale.height};
	const float live      = PortLive::sharpness.load(std::memory_order_relaxed);
	const float sharpness = live >= 0.0f ? live : EnvFloat("KYTY_UPSCALE_SHARPNESS", 0.3f);
	up.enableSharpening  = sharpness > 0.0f;
	up.sharpness         = std::clamp(sharpness, 0.0f, 1.0f);
	up.frameTimeDelta    = std::clamp(desc.delta_ms, 1.0f, 100.0f);
	up.preExposure       = 1.0f;
	up.reset             = desc.reset;
	const float z_near   = EnvFloat("KYTY_FSR3_NEAR", desc.z_near > 0.0f ? desc.z_near : camera.valid ? camera.z_near : 0.1f);
	up.cameraNear        = FLT_MAX; // reversed infinite depth: the SDK's convention (frame generation's too)
	up.cameraFar         = z_near;
	up.cameraFovAngleVertical  = EnvFloat("KYTY_FSR3_FOV", desc.fov > 0.0f ? desc.fov : camera.valid ? camera.fov : 1.0f);
	up.viewSpaceToMetersFactor = 1.0f;
	up.flags                   = 0;
	const auto result = g_ffx.Dispatch(&g_up_context, &up.header);
	if (result != FFX_API_RETURN_OK) {
		if (g_up_errors++ < 4) {
			std::printf("FSR3 upscaling: dispatch failed (%u)\n", result);
			std::fflush(stdout);
		}
		return false;
	}
	return true;
}

static double CapInterval() {
	static const double interval = [] {
		const char* text = std::getenv("KYTY_VRR_CAP_FPS");
		if (text == nullptr || *text == '\0') return 0.0;
		double cap = std::strtod(text, nullptr);
		if (std::string(text) == "auto") {
			cap = 0.0;
			DEVMODEA mode {};
			mode.dmSize = sizeof(mode);
			if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 30) {
				const double hz = mode.dmDisplayFrequency;
				cap             = hz - hz * hz / 3600.0;
			}
		}
		const char* frames    = std::getenv("KYTY_FSR3_FG_FRAMES");
		const int   value     = frames != nullptr ? std::atoi(frames) : 1;
		const int   generated = value >= 3 ? 3 : (value >= 2 ? 2 : 1);
		if (cap <= 0.0) return 0.0;
		std::printf("FSR3 FG: shown frames held under %.0f fps (real at most %.1f)\n", cap, cap / (1 + generated));
		std::fflush(stdout);
		return (1 + generated) / cap;
	}();
	// Dynamic multi frame generation: real frames at most target / current multiplier.
	if (g_mfg.on && g_mfg.dynamic) {
		// 10-21: an in-engine cinematic keeps 60 real: x3 at most and no flip floor, at once.
		if (Libs::VideoOut::CinematicPlaying() && (g_mfg.current > 3 || g_mfg.flip_fps != 0)) {
			g_mfg.current   = std::min(g_mfg.current, 3);
			g_mfg.pending_n = 0;
			MfgSetActive(g_mfg.current);
			Libs::VideoOut::SetDynamicFlipFps(0);
			g_mfg.flip_fps = 0;
			std::printf("[dyn] cinematic: back to 60, x%d\n", g_mfg.current);
			std::fflush(stdout);
		}
		// The game's flip floor holds the real rate: no limiter of ours on top (two limiters beat).
		if (g_mfg.flip_fps > 0) return 0.0;
		return g_mfg.current / g_mfg.target;
	}
	return interval;
}

double RealFrameInterval() {
	return Active() ? CapInterval() : 0.0;
}

void BeforePresent(GraphicContext& graphics, const FrameGen::FrameInputs* inputs, vk::SwapchainKHR swapchain,
                   vk::Extent2D backbuffer, vk::Format format) {
	if (!Active()) return;
	MfgInit();
	// G-SYNC: with KYTY_VRR_CAP_FPS (auto = refresh - refresh^2 / 3600; 240 Hz -> 224) the real frames wait just enough
	// for real x (1 + generated) to stay under the cap, as Reflex does for DLSS-G (x4 at 60 real sent 240 to a 224 cap:
	// 16 frames a second dropped or held, a stutter at a steady 60). Only while generating; movies and menus untouched.
	const double interval = CapInterval();
	if (interval > 0.0 && inputs != nullptr && inputs->valid && !VideoOut::MoviePlaying() && VideoOut::PlayerInputSeen()) {
		static std::chrono::steady_clock::time_point next {};
		const auto step = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(interval));
		auto       now  = std::chrono::steady_clock::now();
		if (next > now) {
			g_mfg.slept += std::chrono::duration<double>(next - now).count();
			std::this_thread::sleep_until(next - std::chrono::milliseconds(1));
			while (std::chrono::steady_clock::now() < next) std::this_thread::yield();
			now = next;
		}
		next = std::max(next, now - step) + step;
	}
	std::lock_guard lock(g_mutex);
	const auto now    = std::chrono::steady_clock::now();
	const bool movie  = VideoOut::MoviePlaying() || !VideoOut::PlayerInputSeen(); // (before the first button: the logo and intro)
	const bool usable = inputs != nullptr && inputs->valid && !movie && !g_fg_failed;

	// Per-second line: real frames, the frames shown (real and generated), how many had generation.
	if (g_stats.start == std::chrono::steady_clock::time_point {}) g_stats.start = now;
	const double window = std::chrono::duration<double>(now - g_stats.start).count();
	if (window >= 1.0) {
		const uint64_t presents =
		    g_functions.pOutGetLastPresentCountFFXAPI != nullptr
		        ? g_functions.pOutGetLastPresentCountFFXAPI(static_cast<VkSwapchainKHR>(swapchain))
		        : 0;
		const uint64_t shown = presents >= g_stats.presents_at_start ? presents - g_stats.presents_at_start : 0;
		std::printf("[fsr3] real %.1f fps, shown %.1f fps; generation on %u of %u frames (no TAA %u, movie %u); "
		            "prepare %.2f ms, present call %.2f ms (max %.1f)\n",
		            g_stats.real / window, static_cast<double>(shown) / window, g_stats.generated_on, g_stats.real,
		            g_stats.no_inputs, g_stats.movie, g_stats.real != 0 ? g_stats.prepare_ms / g_stats.real : 0.0,
		            g_stats.real != 0 ? g_stats.present_ms / g_stats.real : 0.0, g_stats.max_present_ms);
		std::fflush(stdout);
		MfgUpdate(g_stats.real, window, static_cast<double>(shown) / window);
		g_stats                   = {};
		g_stats.start             = now;
		g_stats.presents_at_start = presents;
	}
	g_stats.real++;
	if (movie) g_stats.movie++;
	else if (!usable) g_stats.no_inputs++;

	const double delta_ms = g_last_real == std::chrono::steady_clock::time_point {}
	                            ? 16.7
	                            : std::chrono::duration<double, std::milli>(now - g_last_real).count();
	g_last_real = now;

	if (usable && !g_fg_failed) {
		const vk::Extent2D render = inputs->render_extent;
		if (g_fg_context == nullptr || g_fg_display != backbuffer || g_fg_format != format ||
		    render.width > g_fg_max_render.width || render.height > g_fg_max_render.height) {
			CreateFrameGeneration(graphics, backbuffer,
			                      {std::max(render.width, g_fg_max_render.width), std::max(render.height, g_fg_max_render.height)},
			                      format);
		}
	}
	if (g_fg_context == nullptr) {
		LogState(0, movie ? "off (movie)" : "off (no TAA pass yet: menu, loading or movie)");
		return;
	}

	// (A camera cut in this frame: no frame generated between the two shots; chenxiao07 3e1e3250.)
	const bool generate = usable && g_fg_context != nullptr && !inputs->cut;
	++g_frame_id;
	ffxConfigureDescFrameGeneration config {};
	config.header.type             = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	config.swapChain               = static_cast<void*>(static_cast<VkSwapchainKHR>(swapchain));
	config.presentCallback         = nullptr;
	config.presentCallbackUserContext = nullptr;
	config.frameGenerationCallback = [](ffxDispatchDescFrameGeneration* params, void* context) -> ffxReturnCode_t {
		return g_ffx.Dispatch(reinterpret_cast<ffxContext*>(context), &params->header);
	};
	config.frameGenerationCallbackUserContext = &g_fg_context;
	config.frameGenerationEnabled             = generate;
	config.allowAsyncWorkloads                = EnvOn("KYTY_FSR3_ASYNC", false);
	// The HUD-less image (the frame's, scaled to the backbuffer below): FSR 3 keeps the HUD out of the generated frames.
	// The HUD layer (RecordHudLayer): registered while generating, the backbuffer then has no HUD (no HUD-less image to
	// give); off, none (menus are shown as they are).
	const bool layer = generate && HudLayerWanted() && g_hud.written && g_hud.view != nullptr &&
	                   g_hud.image.extent.width == backbuffer.width && g_hud.image.extent.height == backbuffer.height;
	{
		ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK ui {};
		ui.header.type = FFX_API_CONFIGURE_DESC_TYPE_FGSWAPCHAIN_REGISTERUIRESOURCE_VK;
		if (layer) {
			ui.uiResource       = ImageResource(g_hud.image, false);
			ui.uiResource.state = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
			ui.flags = FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA |
			           FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING;
		}
		if (const auto result = g_ffx.Configure(&g_swapchain_context, &ui.header); result != FFX_API_RETURN_OK) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 4) {
				std::printf("FSR3 FG: HUD layer not registered (%u)\n", result);
				std::fflush(stdout);
			}
		}
	}
	const bool hudless = generate && !layer && inputs->has_hudless && inputs->hudless.image != nullptr &&
	                     EnsurePresentHudless(graphics, backbuffer, format);
	config.HUDLessColor = FfxApiResource {};
	if (hudless) {
		auto& resource                   = config.HUDLessColor;
		resource.resource                = static_cast<void*>(static_cast<VkImage>(g_present_hudless.image));
		resource.state                   = FFX_API_RESOURCE_STATE_COMPUTE_READ;
		resource.description.type        = FFX_API_RESOURCE_TYPE_TEXTURE2D;
		resource.description.format      = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(format));
		resource.description.width       = backbuffer.width;
		resource.description.height      = backbuffer.height;
		resource.description.depth       = 1;
		resource.description.mipCount    = 1;
		resource.description.flags       = FFX_API_RESOURCE_FLAGS_NONE;
		resource.description.usage       = FFX_API_RESOURCE_USAGE_READ_ONLY;
	}
	config.flags = EnvOn("KYTY_FSR3_DEBUG_VIEW", false) ? static_cast<uint32_t>(FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW) : 0u;
	config.onlyPresentGenerated = false;
	config.generationRect       = {0, 0, static_cast<int32_t>(backbuffer.width), static_cast<int32_t>(backbuffer.height)};
	config.frameID              = g_frame_id;
	if (const auto result = g_ffx.Configure(&g_fg_context, &config.header); result != FFX_API_RETURN_OK) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			std::printf("FSR3 FG: configure failed (%u)\n", result);
			std::fflush(stdout);
		}
		return;
	}
	if (!generate) {
		g_generating = false;
		LogState(1, movie ? "off (movie)" : "off (no TAA pass: menu, loading or movie)");
		return;
	}

	// The prepare pass: the game's depth and motion vectors (copied at the guest flip) into the context.
	if (!EnsureCommandBuffers(graphics)) return;
	const auto device       = static_cast<VkDevice>(graphics.device);
	static auto* wait_fences  = reinterpret_cast<PFN_vkWaitForFences>(g_get_device_proc(device, "vkWaitForFences"));
	static auto* reset_fences = reinterpret_cast<PFN_vkResetFences>(g_get_device_proc(device, "vkResetFences"));
	static auto* begin_cb     = reinterpret_cast<PFN_vkBeginCommandBuffer>(g_get_device_proc(device, "vkBeginCommandBuffer"));
	static auto* end_cb       = reinterpret_cast<PFN_vkEndCommandBuffer>(g_get_device_proc(device, "vkEndCommandBuffer"));
	static auto* reset_cb     = reinterpret_cast<PFN_vkResetCommandBuffer>(g_get_device_proc(device, "vkResetCommandBuffer"));
	auto&        slot         = g_slots[g_slot];
	g_slot                    = (g_slot + 1) % static_cast<uint32_t>(g_slots.size());
	if (slot.pending) {
		wait_fences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
		reset_fences(device, 1, &slot.fence);
		slot.pending = false;
	}
	reset_cb(slot.command, 0);
	VkCommandBufferBeginInfo begin {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_cb(slot.command, &begin);

	if (hudless) RecordHudlessBlit(device, slot.command, inputs->hudless);
	const auto& camera = FromCamera(inputs->camera);
	static const float mv_sign = EnvFloat("KYTY_FSR3_MV_SIGN", -1.0f) >= 0 ? 1.0f : -1.0f;
	ffxDispatchDescFrameGenerationPrepare prepare {};
	prepare.header.type    = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
	prepare.frameID        = g_frame_id;
	prepare.flags          = config.flags;
	prepare.commandList    = slot.command;
	prepare.renderSize     = {inputs->render_extent.width, inputs->render_extent.height};
	prepare.jitterOffset   = {0.0f, 0.0f};
	// The game's vectors are in UV units along the motion; FSR wants pixels towards the previous position.
	prepare.motionVectorScale = {mv_sign * static_cast<float>(inputs->render_extent.width),
	                             mv_sign * static_cast<float>(inputs->render_extent.height)};
	prepare.frameTimeDelta = static_cast<float>(std::clamp(delta_ms, 1.0, 100.0));
	prepare.unused_reset   = false;
	// Reversed infinite depth: far is the near plane, near is infinite (the SDK's convention).
	const float z_near     = EnvFloat("KYTY_FSR3_NEAR", camera.valid ? camera.z_near : 0.1f);
	prepare.cameraNear     = FLT_MAX;
	prepare.cameraFar      = z_near;
	prepare.cameraFovAngleVertical  = EnvFloat("KYTY_FSR3_FOV", camera.valid ? camera.fov : 1.0f);
	prepare.viewSpaceToMetersFactor = 1.0f;
	prepare.depth                   = ImageResource(inputs->depth, true);
	prepare.motionVectors           = ImageResource(inputs->motion, false);
	ffxDispatchDescFrameGenerationPrepareCameraInfo camera_info {};
	camera_info.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
	if (camera.valid) {
		std::copy_n(camera.position, 3, camera_info.cameraPosition);
		std::copy_n(camera.up, 3, camera_info.cameraUp);
		std::copy_n(camera.right, 3, camera_info.cameraRight);
		std::copy_n(camera.forward, 3, camera_info.cameraForward);
		prepare.header.pNext = &camera_info.header;
	}
	const auto prepare_begin = std::chrono::steady_clock::now();
	const auto result        = g_ffx.Dispatch(&g_fg_context, &prepare.header);
	end_cb(slot.command);
	if (result != FFX_API_RETURN_OK) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			std::printf("FSR3 FG: prepare failed (%u)\n", result);
			std::fflush(stdout);
		}
		return;
	}
	VkSubmitInfo submit {};
	submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &slot.command;
	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		g_queue_submit(static_cast<VkQueue>(graphics.queue), 1, &submit, slot.fence);
	}
	slot.pending = true;
	g_stats.prepare_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_begin).count();
	g_stats.generated_on++;
	if (!g_generating) {
		std::printf("FSR3 FG: on (TAA pass: %ux%u depth and motion vectors, camera %s, HUD-less image %s)\n",
		            inputs->render_extent.width, inputs->render_extent.height, camera.valid ? "found" : "default",
		            hudless ? "yes" : "no");
		std::fflush(stdout);
		g_state_logged = 2;
	}
	g_generating = true;
}

#else // !_WIN32

bool Requested() {
	return false;
}
bool PlanQueues(vk::PhysicalDevice, uint32_t, uint32_t, QueuePlan&) {
	return false;
}
void OnDevice(GraphicContext&, const QueuePlan&) {}
double RealFrameInterval() {
	return 0.0;
}
bool UpscaleWanted() {
	return false;
}
bool SrPathWanted() {
	return false;
}
bool SrReady() {
	return false;
}
bool Upscale(GraphicContext&, const UpscaleDesc&) {
	return false;
}
bool Active() {
	return false;
}
bool CreateSwapchain(GraphicContext&, const vk::SwapchainCreateInfoKHR&, vk::SwapchainKHR*) {
	return false;
}
void       DestroySwapchain(GraphicContext&, vk::SwapchainKHR) {}
vk::Result GetSwapchainImages(GraphicContext&, vk::SwapchainKHR, uint32_t*, vk::Image*) {
	return vk::Result::eErrorFeatureNotPresent;
}
vk::Result AcquireNextImage(GraphicContext&, vk::SwapchainKHR, vk::Semaphore, uint32_t*) {
	return vk::Result::eErrorFeatureNotPresent;
}
vk::Result QueuePresent(GraphicContext&, const vk::PresentInfoKHR&) {
	return vk::Result::eErrorFeatureNotPresent;
}
void BeforePresent(GraphicContext&, const FrameGen::FrameInputs*, vk::SwapchainKHR, vk::Extent2D, vk::Format) {}
void RecordHudLayer(GraphicContext&, vk::CommandBuffer, vk::Image, vk::Extent2D, vk::Rect2D, VulkanImage&,
                    const FrameGen::FrameInputs*) {}

#endif

} // namespace Libs::Graphics::Fsr3
