#include "xess-sr.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#if defined(_WIN32) && defined(KYTY_XESS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <xess/xess.h>
#include <xess/xess_vk.h>
#endif

namespace Libs::Graphics::Xess {

bool Wanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_UPSCALE");
		return value != nullptr && std::string(value) == "xess";
	}();
	return wanted;
}

bool SrPathWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_UPSCALER");
		return value != nullptr && std::string(value) == "xess";
	}();
	return wanted;
}

#if defined(_WIN32) && defined(KYTY_XESS)
namespace {

// libxess.dll's functions, loaded at run time (the emulator starts without it).
struct Api {
	decltype(&xessVKGetRequiredDeviceExtensions) get_device_extensions = nullptr;
	decltype(&xessVKCreateContext)               create_context        = nullptr;
	decltype(&xessVKInit)                        init                  = nullptr;
	decltype(&xessVKExecute)                     execute               = nullptr;
	decltype(&xessSetVelocityScale)              set_velocity_scale    = nullptr;
	decltype(&xessDestroyContext)                destroy_context       = nullptr;
	decltype(&xessGetVersion)                    get_version           = nullptr;
};
Api                   g_api;
bool                  g_extension   = false; // VK_EXT_mutable_descriptor_type added to the device
VkInstance            g_instance    = VK_NULL_HANDLE;
VkPhysicalDevice      g_physical    = VK_NULL_HANDLE;
VkDevice              g_device      = VK_NULL_HANDLE;
xess_context_handle_t g_context     = nullptr;
vk::Extent2D          g_output {};
vk::Extent2D          g_render {};
float                 g_scale[2] {};
bool                  g_failed      = false;
uint32_t              g_errors      = 0;
VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT g_mutable {};

bool Load() {
	static const bool loaded = [] {
		if (!Wanted()) return false;
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		std::wstring dir(exe);
		dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
		HMODULE dll = LoadLibraryW((dir + L"libxess.dll").c_str());
		if (dll == nullptr) {
			std::printf("XeSS: libxess.dll not found next to the emulator: upscaling off\n");
			std::fflush(stdout);
			return false;
		}
		const auto load = [&](auto& function, const char* name) {
			function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
			    reinterpret_cast<void*>(GetProcAddress(dll, name)));
			return function != nullptr;
		};
		if (!load(g_api.get_device_extensions, "xessVKGetRequiredDeviceExtensions") ||
		    !load(g_api.create_context, "xessVKCreateContext") || !load(g_api.init, "xessVKInit") ||
		    !load(g_api.execute, "xessVKExecute") || !load(g_api.set_velocity_scale, "xessSetVelocityScale") ||
		    !load(g_api.destroy_context, "xessDestroyContext") || !load(g_api.get_version, "xessGetVersion")) {
			std::printf("XeSS: libxess.dll lacks a Vulkan entry point: upscaling off\n");
			std::fflush(stdout);
			return false;
		}
		xess_version_t version {};
		g_api.get_version(&version);
		std::printf("XeSS: libxess.dll %u.%u.%u loaded\n", version.major, version.minor, version.patch);
		std::fflush(stdout);
		return true;
	}();
	return loaded;
}

xess_quality_settings_t QualityFor(double ratio) {
	// XeSS 1.3+ scaling: AA 1x, ultra quality plus 1.3x, ultra quality 1.5x, quality 1.7x, balanced 2x,
	// performance 2.3x, ultra performance 3x.
	if (ratio < 1.15) return XESS_QUALITY_SETTING_AA;
	if (ratio < 1.4) return XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS;
	if (ratio < 1.6) return XESS_QUALITY_SETTING_ULTRA_QUALITY;
	if (ratio < 1.85) return XESS_QUALITY_SETTING_QUALITY;
	if (ratio < 2.15) return XESS_QUALITY_SETTING_BALANCED;
	if (ratio < 2.65) return XESS_QUALITY_SETTING_PERFORMANCE;
	return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
}

xess_vk_image_view_info ViewInfo(const VulkanImage& image, vk::ImageView view, bool depth) {
	xess_vk_image_view_info info {};
	info.imageView        = static_cast<VkImageView>(view);
	info.image            = static_cast<VkImage>(image.image);
	info.subresourceRange = {depth ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT)
	                               : static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
	                         0, 1, 0, 1};
	info.format = static_cast<VkFormat>(image.format);
	info.width  = image.extent.width;
	info.height = image.extent.height;
	return info;
}

} // namespace

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical_device,
                         const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions) {
	if (!Load()) return;
	uint32_t           count = 0;
	const char* const* names = nullptr;
	if (g_api.get_device_extensions(static_cast<VkInstance>(instance), static_cast<VkPhysicalDevice>(physical_device),
	                                &count, &names) != XESS_RESULT_SUCCESS) {
		std::printf("XeSS: this GPU or driver is not supported: upscaling off\n");
		std::fflush(stdout);
		g_failed = true;
		return;
	}
	for (uint32_t i = 0; i < count; ++i) {
		const bool has = std::any_of(available.begin(), available.end(),
		                             [&](const auto& e) { return std::strcmp(e.extensionName, names[i]) == 0; });
		const bool added = std::any_of(extensions.begin(), extensions.end(),
		                               [&](const char* e) { return std::strcmp(e, names[i]) == 0; });
		if (!has) {
			std::printf("XeSS: the device lacks %s: upscaling off\n", names[i]);
			std::fflush(stdout);
			g_failed = true;
			return;
		}
		if (!added) extensions.push_back(names[i]);
	}
	g_extension = std::any_of(extensions.begin(), extensions.end(), [](const char* e) {
		return std::strcmp(e, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) == 0;
	});
}

void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next) {
	if (g_failed || !Load()) return next;
	const bool mutable_ext = std::any_of(extensions.begin(), extensions.end(), [](const char* e) {
		return std::strcmp(e, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) == 0;
	});
	if (!mutable_ext) return next;
	g_mutable                       = {};
	g_mutable.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
	g_mutable.mutableDescriptorType = VK_TRUE;
	g_mutable.pNext                 = next;
	return &g_mutable;
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	if (g_failed || !Load()) return;
	g_instance = static_cast<VkInstance>(instance);
	g_physical = static_cast<VkPhysicalDevice>(physical_device);
	g_device   = static_cast<VkDevice>(device);
}

bool Ready() {
	// (Asked when the game's eboot is patched, which may come before the device: libxess.dll loading is enough there;
	// without the device Upscale fails and the PS5 Pro path's stretch shows.)
	return !g_failed && Load();
}

bool Upscale(const UpscaleDesc& desc) {
	if (g_failed || !Load() || g_device == VK_NULL_HANDLE || desc.command == VK_NULL_HANDLE || desc.color == nullptr ||
	    desc.depth == nullptr || desc.motion == nullptr || desc.output == nullptr)
		return false;
	if (g_context == nullptr || g_output != desc.upscale || g_render != desc.render) {
		if (g_context != nullptr) {
			vk::Device(g_device).waitIdle(); // (a size change)
			g_api.destroy_context(g_context);
			g_context = nullptr;
		}
		if (g_api.create_context(g_instance, g_physical, g_device, &g_context) != XESS_RESULT_SUCCESS) {
			std::printf("XeSS: context not created: upscaling off\n");
			std::fflush(stdout);
			g_failed = true;
			return false;
		}
		xess_vk_init_params_t init {};
		init.outputResolution = {desc.upscale.width, desc.upscale.height};
		const double ratio    = static_cast<double>(desc.upscale.height) / desc.render.height;
		init.qualitySetting   = QualityFor(ratio);
		// The game's depth is reversed; its 3D image linear HDR (exposure computed by XeSS).
		init.initFlags = XESS_INIT_FLAG_INVERTED_DEPTH | XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
		if (const auto result = g_api.init(g_context, &init); result != XESS_RESULT_SUCCESS) {
			std::printf("XeSS: init failed (%d): upscaling off\n", static_cast<int>(result));
			std::fflush(stdout);
			g_api.destroy_context(g_context);
			g_context = nullptr;
			g_failed  = true;
			return false;
		}
		// The game's vectors are in UV units along the motion; XeSS wants pixels towards the previous frame.
		static const float sign = [] {
			const char* value = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return value != nullptr && std::atof(value) > 0 ? 1.0f : -1.0f;
		}();
		g_api.set_velocity_scale(g_context, sign * static_cast<float>(desc.render.width),
		                         sign * static_cast<float>(desc.render.height));
		g_scale[0] = g_scale[1] = 0.0f;
		g_output = desc.upscale;
		g_render = desc.render;
		std::printf("XeSS: %ux%u -> %ux%u, quality %d\n", desc.render.width, desc.render.height, desc.upscale.width,
		            desc.upscale.height, static_cast<int>(init.qualitySetting));
		std::fflush(stdout);
	}
	// The PS5 Pro path's vectors: their own scale to pixels (the game's constants; XeSS's own sign convention).
	if ((desc.mv_scale[0] != 0.0f || desc.mv_scale[1] != 0.0f) &&
	    (desc.mv_scale[0] != g_scale[0] || desc.mv_scale[1] != g_scale[1])) {
		g_scale[0] = desc.mv_scale[0];
		g_scale[1] = desc.mv_scale[1];
		static const float sign = [] {
			const char* value = std::getenv("KYTY_SR_MV_SIGN");
			return value != nullptr && std::atof(value) < 0 ? -1.0f : 1.0f;
		}();
		g_api.set_velocity_scale(g_context, sign * g_scale[0], sign * g_scale[1]);
	}
	xess_vk_execute_params_t exec {};
	exec.colorTexture    = ViewInfo(*desc.color, desc.color_view, false);
	exec.velocityTexture = ViewInfo(*desc.motion, desc.motion_view, false);
	exec.depthTexture    = ViewInfo(*desc.depth, desc.depth_view, true);
	exec.outputTexture   = ViewInfo(*desc.output, desc.output_view, false);
	// (After the game's TAA: 0, its jitter resolved. KYTY_TAA_REPLACE=upscaler: the camera's; KYTY_XESS_JITTER_SIGN=-1 the
	// other way round.)
	static const float jitter_sign = [] {
		const char* value = std::getenv("KYTY_XESS_JITTER_SIGN");
		return value != nullptr && std::atof(value) < 0 ? -1.0f : 1.0f;
	}();
	exec.jitterOffsetX   = jitter_sign * desc.jitter[0];
	exec.jitterOffsetY   = jitter_sign * desc.jitter[1];
	exec.exposureScale   = 1.0f;
	exec.resetHistory    = desc.reset ? 1u : 0u;
	exec.inputWidth      = desc.render.width;
	exec.inputHeight     = desc.render.height;
	if (const auto result = g_api.execute(g_context, desc.command, &exec); result != XESS_RESULT_SUCCESS) {
		if (g_errors++ < 4) {
			std::printf("XeSS: execute failed (%d)\n", static_cast<int>(result));
			std::fflush(stdout);
		}
		return false;
	}
	return true;
}

#else

void AddDeviceExtensions(vk::Instance, vk::PhysicalDevice, const std::vector<vk::ExtensionProperties>&,
                         std::vector<const char*>&) {}
bool Ready() {
	return false;
}
void* ChainDeviceFeatures(const std::vector<const char*>&, void* next) {
	return next;
}
void OnDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {}
bool Upscale(const UpscaleDesc&) {
	return false;
}

#endif

} // namespace Libs::Graphics::Xess
