#include "graphics/presentation/window/fsr3FrameGen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
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
#include <ffx_api/vk/ffx_api_vk.h>
#endif

namespace Libs::Graphics::Fsr3 {

#if defined(_WIN32)
namespace {

bool         g_requested = false;
HMODULE      g_dll       = nullptr;
ffxFunctions g_ffx {};

GraphicContext* g_graphics       = nullptr;
VkQueue         g_present_queue  = VK_NULL_HANDLE; // FI present: graphics family
VkQueue         g_async_queue    = VK_NULL_HANDLE; // compute family, index 0
VkQueue         g_acquire_queue  = VK_NULL_HANDLE; // compute family, index 1
uint32_t        g_compute_family = UINT32_MAX;

PFN_vkGetDeviceProcAddr g_get_device_proc = nullptr;
PFN_vkQueueSubmit       g_queue_submit    = nullptr;

ffxContext                                  g_swapchain_context = nullptr;
ffxQueryDescSwapchainReplacementFunctionsVK g_functions {};

// The frame generation context, made for one display size and format.
ffxContext   g_fg_context = nullptr;
vk::Extent2D g_fg_display {};
vk::Format   g_fg_format = vk::Format::eUndefined;
bool         g_fg_failed = false;

// The constant inputs (no game vectors): far depth and zero motion, at half the backbuffer's size.
VulkanImage  g_depth;
VulkanImage  g_motion;
vk::Extent2D g_inputs_extent {};
bool         g_inputs_cleared = false;

struct Slot {
	VkCommandBuffer command = VK_NULL_HANDLE;
	VkFence         fence   = VK_NULL_HANDLE;
	bool            pending = false;
};
VkCommandPool       g_pool = VK_NULL_HANDLE;
std::array<Slot, 4> g_slots {};
uint32_t            g_slot = 0;

uint64_t g_frame_id     = 0;
bool     g_generating   = false;
int      g_state_logged = -1;
std::chrono::steady_clock::time_point g_last_real {};

struct Stats {
	std::chrono::steady_clock::time_point start {};
	uint32_t real = 0, generated_on = 0, repeats = 0;
	uint64_t presents_at_start = 0;
	double   prepare_ms = 0, present_ms = 0, max_present_ms = 0;
} g_stats;

std::mutex g_mutex; // the contexts are not thread safe

bool EnvOn(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	return *value != '0';
}

void Log(const char* text) {
	std::printf("%s\n", text);
	std::fflush(stdout);
}

// KYTY_FSR3_MFG=N (3 or 4): N - 1 generated frames per real one, read by the patched DLL (tools/local/fsr3-mfg)
// when it creates the swapchain (KYTY_FSR3_FG_FRAMES) and on every present (KYTY_FSR3_FG_ACTIVE). AMD's signed
// DLL ignores both and stays at x2.
void MfgInit() {
	static bool done = false;
	if (done) return;
	done            = true;
	const char* n   = std::getenv("KYTY_FSR3_MFG");
	if (n == nullptr || *n == '\0') return;
	const int multiplier = std::clamp(std::atoi(n), 2, 4);
	char      frames[8];
	std::snprintf(frames, sizeof(frames), "%d", multiplier - 1);
	_putenv_s("KYTY_FSR3_FG_FRAMES", frames); // (CRT and process environment: the DLL reads the latter)
	SetEnvironmentVariableA("KYTY_FSR3_FG_FRAMES", frames);
	SetEnvironmentVariableA("KYTY_FSR3_FG_ACTIVE", frames);
	std::printf("FSR3 FG: KYTY_FSR3_MFG=%d: x%d (patched FidelityFX DLL only; AMD's signed DLL stays x2)\n", multiplier,
	            multiplier);
	std::fflush(stdout);
}

// Patched DLL: no present closer than the display's refresh period x KYTY_FSR3_PACING_FLOOR (default 1.0, 0 off).
void SetPacingFloor() {
	static bool logged = false;
	double      factor = 1.0;
	if (const char* text = std::getenv("KYTY_FSR3_PACING_FLOOR"); text != nullptr && *text != '\0') factor = std::strtod(text, nullptr);
	DEVMODEA mode {};
	mode.dmSize        = sizeof(mode);
	const bool refresh = EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 30;
	if (factor > 0.0 && refresh) {
		const double floor_us = 1000000.0 / mode.dmDisplayFrequency * factor;
		SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", std::to_string(floor_us).c_str());
		if (!logged)
			std::printf("FSR3 FG: present pacing floor %.3f ms (display %lu Hz)\n", floor_us / 1000.0,
			            static_cast<unsigned long>(mode.dmDisplayFrequency));
	} else {
		SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", nullptr);
	}
	logged = true;
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
	resource.resource      = static_cast<void*>(static_cast<VkImage>(image.image));
	resource.state         = FFX_API_RESOURCE_STATE_COMPUTE_READ;
	auto& desc             = resource.description;
	desc.type              = FFX_API_RESOURCE_TYPE_TEXTURE2D;
	desc.format            = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
	desc.width             = image.extent.width;
	desc.height            = image.extent.height;
	desc.depth             = 1;
	desc.mipCount          = 1;
	desc.flags             = FFX_API_RESOURCE_FLAGS_NONE;
	desc.usage             = depth ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : FFX_API_RESOURCE_USAGE_READ_ONLY;
	return resource;
}

void DestroyFrameGeneration() {
	if (g_fg_context != nullptr) {
		ffxConfigureDescFrameGeneration off {};
		off.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
		off.frameGenerationEnabled = false;
		g_ffx.Configure(&g_fg_context, &off.header);
		g_ffx.DestroyContext(&g_fg_context, nullptr);
		g_fg_context = nullptr;
	}
}

bool CreateInputImage(GraphicContext& graphics, VulkanImage& image, vk::Format format, vk::Extent2D extent) {
	if (image.image != nullptr) graphics.DeleteImage(image);
	image.image = nullptr;
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
	if (!graphics.CreateImage(create, image)) {
		image.image = nullptr;
		return false;
	}
	return true;
}

bool CreateFrameGeneration(GraphicContext& graphics, vk::Extent2D display, vk::Format format) {
	DestroyFrameGeneration();
	const vk::Extent2D render {std::max(display.width / 2, 1u), std::max(display.height / 2, 1u)};
	if (!CreateInputImage(graphics, g_depth, vk::Format::eR32Sfloat, render) ||
	    !CreateInputImage(graphics, g_motion, vk::Format::eR16G16Sfloat, render)) {
		Log("FSR3 FG: input images not created: generation stays off");
		g_fg_failed = true;
		return false;
	}
	g_inputs_extent  = render;
	g_inputs_cleared = false;

	ffxCreateBackendVKDesc backend {};
	backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend.vkDevice         = static_cast<VkDevice>(graphics.device);
	backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
	backend.vkDeviceProcAddr = g_get_device_proc;
	ffxCreateContextDescFrameGeneration create {};
	create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
	create.header.pNext = &backend.header;
	// The constant depth is 0 = far with reversed infinite depth.
	create.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
	if (EnvOn("KYTY_FSR3_DEBUG", false)) create.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
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
	g_fg_display = display;
	g_fg_format  = format;
	std::printf("FSR3 FG: frame generation context %ux%u (optical flow only: no game depth or motion vectors)\n",
	            display.width, display.height);
	std::fflush(stdout);
	return true;
}

bool EnsureCommandBuffers(GraphicContext& graphics) {
	if (g_pool != VK_NULL_HANDLE) return true;
	const auto device       = static_cast<VkDevice>(graphics.device);
	auto*      create_pool  = reinterpret_cast<PFN_vkCreateCommandPool>(g_get_device_proc(device, "vkCreateCommandPool"));
	auto*      allocate     = reinterpret_cast<PFN_vkAllocateCommandBuffers>(g_get_device_proc(device, "vkAllocateCommandBuffers"));
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

// The constant inputs, cleared once (depth 0 = far, motion 0) and left in SHADER_READ_ONLY_OPTIMAL.
void RecordClearInputs(VkDevice device, VkCommandBuffer command) {
	static auto* barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(g_get_device_proc(device, "vkCmdPipelineBarrier"));
	static auto* clear   = reinterpret_cast<PFN_vkCmdClearColorImage>(g_get_device_proc(device, "vkCmdClearColorImage"));
	VkImageMemoryBarrier barriers[2] {};
	VkImage              images[2] {static_cast<VkImage>(g_depth.image), static_cast<VkImage>(g_motion.image)};
	for (int i = 0; i < 2; ++i) {
		auto& b               = barriers[i];
		b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		b.image               = images[i];
		b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		b.srcAccessMask       = 0;
		b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	}
	barrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
	const VkClearColorValue       zero {};
	const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	for (auto* image: images) clear(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
	for (auto& b: barriers) {
		b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	}
	barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
}

void LogState(int state, const char* text) {
	if (state == g_state_logged) return;
	g_state_logged = state;
	std::printf("FSR3 FG: %s\n", text);
	std::fflush(stdout);
}

bool LoadApi() {
	static const bool loaded = [] {
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		std::wstring dir(exe);
		dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
		// KYTY_FSR3_DLL=<path>: another build of the DLL (the package's fsr3-mfg\ one does x3/x4).
		if (const char* path = std::getenv("KYTY_FSR3_DLL"); path != nullptr && *path != '\0') {
			g_dll = LoadLibraryA(path);
			std::printf("FSR3: KYTY_FSR3_DLL=%s %s\n", path, g_dll != nullptr ? "loaded" : "not loaded (using the default DLL)");
			std::fflush(stdout);
		}
		for (const auto* name: {L"amd_fidelityfx_vk.dll", L"fsr3\\amd_fidelityfx_vk.dll"}) {
			if (g_dll != nullptr) break;
			g_dll = LoadLibraryW((dir + name).c_str());
			if (g_dll != nullptr) break;
		}
		if (g_dll == nullptr) {
			Log("FSR3: amd_fidelityfx_vk.dll not found next to the emulator");
			return false;
		}
		ffxLoadFunctions(&g_ffx, g_dll);
		if (g_ffx.CreateContext == nullptr || g_ffx.Configure == nullptr || g_ffx.Dispatch == nullptr ||
		    g_ffx.Query == nullptr || g_ffx.DestroyContext == nullptr) {
			Log("FSR3: amd_fidelityfx_vk.dll has no FidelityFX API");
			return false;
		}
		return true;
	}();
	return loaded;
}

} // namespace

bool Requested() {
	static const bool requested = [] {
		if (!EnvOn("KYTY_FSR3_FG", false)) return false;
		if (!LoadApi()) {
			Log("FSR3 FG: frame generation off");
			return false;
		}
		Log("FSR3 FG: requested (KYTY_FSR3_FG), FidelityFX API loaded");
		return true;
	}();
	// g_requested starts as the request and turns false for good when a later step fails.
	static const bool init = (g_requested = requested, true);
	(void)init;
	return g_requested;
}

bool PlanQueues(vk::PhysicalDevice physical_device, uint32_t queue_family, uint32_t graphics_count, QueuePlan& plan) {
	if (!Requested()) return false;
	uint32_t family_count = 0;
	physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	physical_device.getQueueFamilyProperties(&family_count, families.data());
	if (queue_family >= family_count || families[queue_family].queueCount < graphics_count + 1) {
		Log("FSR3 FG: no free graphics queue for its presents: frame generation off");
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
		Log("FSR3 FG: no async compute queue family: frame generation off");
		g_requested = false;
		return false;
	}
	return true;
}

void OnDevice(GraphicContext& graphics, const QueuePlan& plan) {
	if (!Requested() || plan.compute_family == UINT32_MAX) {
		g_requested = false;
		return;
	}
	g_get_device_proc = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	g_graphics        = &graphics;
	const auto device = static_cast<VkDevice>(graphics.device);
	g_queue_submit    = reinterpret_cast<PFN_vkQueueSubmit>(g_get_device_proc(device, "vkQueueSubmit"));
	auto* get_queue   = reinterpret_cast<PFN_vkGetDeviceQueue>(g_get_device_proc(device, "vkGetDeviceQueue"));
	get_queue(device, graphics.queue_family, plan.graphics_index, &g_present_queue);
	get_queue(device, plan.compute_family, 0, &g_async_queue);
	get_queue(device, plan.compute_family, 1, &g_acquire_queue);
	g_compute_family = plan.compute_family;
	if (g_present_queue == VK_NULL_HANDLE || g_async_queue == VK_NULL_HANDLE || g_acquire_queue == VK_NULL_HANDLE ||
	    g_queue_submit == nullptr) {
		Log("FSR3 FG: its queues were not created: frame generation off");
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
	// a black screen after the window went fullscreen, in the Demon's Souls fork.)
	ffxCreateContextDescFrameGenerationSwapChainVK create {};
	create.header.type       = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
	create.physicalDevice    = static_cast<VkPhysicalDevice>(graphics.physical_device);
	create.device            = static_cast<VkDevice>(graphics.device);
	create.swapchain         = &swapchain;
	create.allocator         = nullptr;
	create.createInfo        = static_cast<VkSwapchainCreateInfoKHR>(info);
	create.gameQueue         = {static_cast<VkQueue>(graphics.queue), graphics.queue_family, &GameQueueSubmit};
	create.asyncComputeQueue = {g_async_queue, g_compute_family, nullptr};
	create.presentQueue      = {g_present_queue, graphics.queue_family, nullptr};
	create.imageAcquireQueue = {g_acquire_queue, g_compute_family, nullptr};
	MfgInit();
	SetPacingFloor();
	const auto result = g_ffx.CreateContext(&g_swapchain_context, &create.header, nullptr);
	if (result != FFX_API_RETURN_OK || swapchain == VK_NULL_HANDLE) {
		std::printf("FSR3 FG: frame generation swapchain not created (%u): frame generation off\n", result);
		std::fflush(stdout);
		g_swapchain_context = nullptr;
		g_requested         = false;
		return false;
	}
	g_functions             = {};
	g_functions.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
	g_ffx.Query(&g_swapchain_context, &g_functions.header);
	if (g_functions.pOutAcquireNextImageKHR == nullptr || g_functions.pOutQueuePresentKHR == nullptr ||
	    g_functions.pOutGetSwapchainImagesKHR == nullptr) {
		Log("FSR3 FG: no replacement swapchain functions: frame generation off");
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

void DestroySwapchain(GraphicContext& graphics, vk::SwapchainKHR /*swapchain*/) {
	std::lock_guard lock(g_mutex);
	if (g_swapchain_context == nullptr) return;
	// Our prepare command buffers may still read the input images.
	const auto device = static_cast<VkDevice>(graphics.device);
	if (g_get_device_proc != nullptr) {
		static auto* wait_fences = reinterpret_cast<PFN_vkWaitForFences>(g_get_device_proc(device, "vkWaitForFences"));
		static auto* reset_fences = reinterpret_cast<PFN_vkResetFences>(g_get_device_proc(device, "vkResetFences"));
		for (auto& slot: g_slots) {
			if (slot.pending) {
				wait_fences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
				reset_fences(device, 1, &slot.fence);
				slot.pending = false;
			}
		}
	}
	// The frame generation context first (the swapchain calls it while presenting); destroying the swapchain's
	// context destroys its swapchain. The next CreateSwapchain makes both again.
	DestroyFrameGeneration();
	g_ffx.DestroyContext(&g_swapchain_context, nullptr);
	g_swapchain_context = nullptr;
	g_functions         = {};
	g_generating        = false;
	g_state_logged      = -1;
	g_stats             = {};
	g_fg_display        = {};
	if (g_depth.image != nullptr) graphics.DeleteImage(g_depth);
	if (g_motion.image != nullptr) graphics.DeleteImage(g_motion);
	g_depth.image    = nullptr;
	g_motion.image   = nullptr;
	g_inputs_cleared = false;
	Log("FSR3 FG: frame generation swapchain released (resize, fullscreen or exit)");
}

vk::Result GetSwapchainImages(GraphicContext& graphics, vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
	return static_cast<vk::Result>(g_functions.pOutGetSwapchainImagesKHR(
	    static_cast<VkDevice>(graphics.device), static_cast<VkSwapchainKHR>(swapchain), count, reinterpret_cast<VkImage*>(images)));
}

vk::Result AcquireNextImage(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
	return static_cast<vk::Result>(g_functions.pOutAcquireNextImageKHR(static_cast<VkDevice>(graphics.device),
	                                                                   static_cast<VkSwapchainKHR>(swapchain), UINT64_MAX,
	                                                                   static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE, index));
}

vk::Result QueuePresent(GraphicContext& graphics, const vk::PresentInfoKHR& info) {
	std::lock_guard  lock(g_mutex);
	const auto       begin   = std::chrono::steady_clock::now();
	VkPresentInfoKHR present = info;
	present.pNext            = nullptr;
	const auto result        = g_functions.pOutQueuePresentKHR(static_cast<VkQueue>(graphics.queue), &present);
	const double ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	g_stats.present_ms += ms;
	g_stats.max_present_ms = std::max(g_stats.max_present_ms, ms);
	return static_cast<vk::Result>(result);
}

void BeforePresent(GraphicContext& graphics, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer, vk::Format format,
                   bool new_frame) {
	if (!Active()) return;
	std::lock_guard lock(g_mutex);
	const auto      now = std::chrono::steady_clock::now();

	// Once a second: real frames, frames shown (real and generated).
	if (g_stats.start == std::chrono::steady_clock::time_point {}) g_stats.start = now;
	const double window = std::chrono::duration<double>(now - g_stats.start).count();
	if (window >= 5.0) {
		const uint64_t presents = g_functions.pOutGetLastPresentCountFFXAPI != nullptr
		                              ? g_functions.pOutGetLastPresentCountFFXAPI(static_cast<VkSwapchainKHR>(swapchain))
		                              : 0;
		const uint64_t shown = presents >= g_stats.presents_at_start ? presents - g_stats.presents_at_start : 0;
		std::printf("[fsr3] real %.1f fps, shown %.1f fps; generation on %u of %u presents (repeats %u); prepare %.2f ms, "
		            "present call %.2f ms (max %.1f)\n",
		            g_stats.real / window, static_cast<double>(shown) / window, g_stats.generated_on, g_stats.real,
		            g_stats.repeats, g_stats.real != 0 ? g_stats.prepare_ms / g_stats.real : 0.0,
		            g_stats.real != 0 ? g_stats.present_ms / g_stats.real : 0.0, g_stats.max_present_ms);
		std::fflush(stdout);
		g_stats                   = {};
		g_stats.start             = now;
		g_stats.presents_at_start = presents;
	}
	g_stats.real++;
	if (!new_frame) g_stats.repeats++;

	const double delta_ms = g_last_real == std::chrono::steady_clock::time_point {}
	                            ? 16.7
	                            : std::chrono::duration<double, std::milli>(now - g_last_real).count();
	if (new_frame) g_last_real = now;

	if (!g_fg_failed && (g_fg_context == nullptr || g_fg_display != backbuffer || g_fg_format != format)) {
		CreateFrameGeneration(graphics, backbuffer, format);
	}
	if (g_fg_context == nullptr) {
		LogState(0, "off (no frame generation context)");
		return;
	}

	const bool generate = new_frame && !EnvOn("KYTY_FSR3_PAUSE", false);
	++g_frame_id;
	ffxConfigureDescFrameGeneration config {};
	config.header.type                = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	config.swapChain                  = static_cast<void*>(static_cast<VkSwapchainKHR>(swapchain));
	config.presentCallback            = nullptr;
	config.presentCallbackUserContext = nullptr;
	config.frameGenerationCallback    = [](ffxDispatchDescFrameGeneration* params, void* context) -> ffxReturnCode_t {
		return g_ffx.Dispatch(reinterpret_cast<ffxContext*>(context), &params->header);
	};
	config.frameGenerationCallbackUserContext = &g_fg_context;
	config.frameGenerationEnabled             = generate;
	config.allowAsyncWorkloads                = EnvOn("KYTY_FSR3_ASYNC", false);
	config.HUDLessColor                       = FfxApiResource {};
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
		return;
	}

	if (!EnsureCommandBuffers(graphics)) return;
	const auto   device       = static_cast<VkDevice>(graphics.device);
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
	if (!g_inputs_cleared) {
		RecordClearInputs(device, slot.command);
		g_inputs_cleared = true;
	}

	ffxDispatchDescFrameGenerationPrepare prepare {};
	prepare.header.type       = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
	prepare.frameID           = g_frame_id;
	prepare.flags             = config.flags;
	prepare.commandList       = slot.command;
	prepare.renderSize        = {g_inputs_extent.width, g_inputs_extent.height};
	prepare.jitterOffset      = {0.0f, 0.0f};
	prepare.motionVectorScale = {static_cast<float>(g_inputs_extent.width), static_cast<float>(g_inputs_extent.height)};
	prepare.frameTimeDelta    = static_cast<float>(std::clamp(delta_ms, 1.0, 100.0));
	prepare.unused_reset      = false;
	// Reversed infinite depth: far is the near plane, near is infinite (the SDK's convention).
	prepare.cameraNear              = FLT_MAX;
	prepare.cameraFar               = 0.1f;
	prepare.cameraFovAngleVertical  = 1.0f;
	prepare.viewSpaceToMetersFactor = 1.0f;
	prepare.depth                   = ImageResource(g_depth, true);
	prepare.motionVectors           = ImageResource(g_motion, false);
	const auto prepare_begin        = std::chrono::steady_clock::now();
	const auto result               = g_ffx.Dispatch(&g_fg_context, &prepare.header);
	end_cb(slot.command);
	if (result != FFX_API_RETURN_OK) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			std::printf("FSR3 FG: prepare failed (%u)\n", result);
			std::fflush(stdout);
		}
		// (The command buffer may hold the inputs' clear: submit it anyway.)
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
	if (result != FFX_API_RETURN_OK) return;
	g_stats.prepare_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_begin).count();
	g_stats.generated_on++;
	if (!g_generating) LogState(2, "on (optical flow only, no game depth or motion vectors)");
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
void BeforePresent(GraphicContext&, vk::SwapchainKHR, vk::Extent2D, vk::Format, bool) {}

#endif

} // namespace Libs::Graphics::Fsr3
