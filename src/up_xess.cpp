// Intel XeSS super resolution through the XeSS SDK's libxess.dll (Intel's redistributable, next to the exe) on its Vulkan
// backend: any GPU with DP4a (the RTX 5080 included). Ported from KytyPS5's src/local/xess-sr.cpp: the same context
// handling, quality-by-ratio table and velocity / jitter conventions; plus what a Vulkan app must do for XeSS before
// its device exists (its instance extensions, device extensions and device features).
#include "upscale_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#ifdef INJ_XESS
#include <xess/xess.h>
#include <xess/xess_vk.h>
#endif

namespace inj::upscale::xess {

#ifdef INJ_XESS
namespace {

struct Api {
	decltype(&xessVKGetRequiredInstanceExtensions) get_instance_extensions = nullptr;
	decltype(&xessVKGetRequiredDeviceExtensions)   get_device_extensions = nullptr;
	decltype(&xessVKGetRequiredDeviceFeatures)     get_device_features = nullptr;
	decltype(&xessVKCreateContext)                 create_context = nullptr;
	decltype(&xessVKInit)                          init = nullptr;
	decltype(&xessVKExecute)                       execute = nullptr;
	decltype(&xessSetVelocityScale)                set_velocity_scale = nullptr;
	decltype(&xessDestroyContext)                  destroy_context = nullptr;
	decltype(&xessGetVersion)                      get_version = nullptr;
};
Api         g_api;
bool        g_tried = false, g_loaded = false;
std::string g_why = "libxess.dll no cargado";
std::vector<std::string> g_instance_ext, g_device_ext;
xess_version_t g_version {};

bool Load() {
	if (g_tried) return g_loaded;
	g_tried = true;
	HMODULE dll = LoadLibraryW((ExeDir() + L"libxess.dll").c_str());
	RedirectVulkanLoads(dll, "libxess.dll");  // ("vulkan-1.dll" -> this winevulkan.dll)
	if (dll == nullptr) {
		g_why = "libxess.dll no esta junto al exe";
		Log("[xess] libxess.dll not next to the exe: XeSS unavailable");
		return false;
	}
	const auto load = [&](auto& fn, const char* name) {
		fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(reinterpret_cast<void*>(GetProcAddress(dll, name)));
		return fn != nullptr;
	};
	if (!load(g_api.get_instance_extensions, "xessVKGetRequiredInstanceExtensions") ||
	    !load(g_api.get_device_extensions, "xessVKGetRequiredDeviceExtensions") ||
	    !load(g_api.get_device_features, "xessVKGetRequiredDeviceFeatures") || !load(g_api.create_context, "xessVKCreateContext") ||
	    !load(g_api.init, "xessVKInit") || !load(g_api.execute, "xessVKExecute") ||
	    !load(g_api.set_velocity_scale, "xessSetVelocityScale") || !load(g_api.destroy_context, "xessDestroyContext") ||
	    !load(g_api.get_version, "xessGetVersion")) {
		g_why = "libxess.dll sin las funciones Vulkan";
		Log("[xess] libxess.dll lacks a Vulkan entry point: XeSS unavailable");
		return false;
	}
	g_api.get_version(&g_version);
	uint32_t count = 0, min_api = 0;
	const char* const* names = nullptr;
	if (g_api.get_instance_extensions(&count, &names, &min_api) == XESS_RESULT_SUCCESS)
		for (uint32_t i = 0; i < count; ++i) g_instance_ext.emplace_back(names[i]);
	Log("[xess] libxess.dll %u.%u.%u loaded: %u instance extension(s), Vulkan %u.%u+", g_version.major, g_version.minor,
	    g_version.patch, count, VK_API_VERSION_MAJOR(min_api), VK_API_VERSION_MINOR(min_api));
	g_loaded = true;
	g_why.clear();
	return true;
}

xess_quality_settings_t QualityFor(int quality) {
	switch (quality) {  // (KytyPS5's table by ratio; here by the resolved mode)
	case 1: return XESS_QUALITY_SETTING_AA;
	case 2: return XESS_QUALITY_SETTING_QUALITY;
	case 3: return XESS_QUALITY_SETTING_BALANCED;
	case 4: return XESS_QUALITY_SETTING_PERFORMANCE;
	default: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
	}
}

xess_vk_image_view_info View(const OwnImage& img) {
	xess_vk_image_view_info info {};
	info.imageView        = img.view;
	info.image            = img.image;
	info.subresourceRange = {ViewAspect(img.format), 0, 1, 0, 1};
	info.format           = img.format;
	info.width            = img.extent.width;
	info.height           = img.extent.height;
	return info;
}

class Xess final : public Upscaler {
public:
	~Xess() override {
		if (ctx_ != nullptr) g_api.destroy_context(ctx_);
		for (auto c: old_) g_api.destroy_context(c);
	}
	const char* Name() const override { return "XeSS"; }
	std::string Detail() const override { return detail_; }

	bool Dispatch(const Frame& f) override {
		uint32_t flags = 0;
		if (f.reversed) flags |= XESS_INIT_FLAG_INVERTED_DEPTH;
		flags |= f.hdr ? XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE : XESS_INIT_FLAG_LDR_INPUT_COLOR;
		const xess_quality_settings_t quality = QualityFor(f.quality);
		if (ctx_ == nullptr || flags != flags_ || quality != quality_ || f.display.width != display_.width || f.display.height != display_.height) {
			if (ctx_ != nullptr) {  // (frames already recorded may still use it: destroyed with this upscaler, after the device idles)
				old_.push_back(ctx_);
				ctx_ = nullptr;
			}
			if (g_api.create_context(R().instance, R().physical, R().device, &ctx_) != XESS_RESULT_SUCCESS) {
				Log("[xess] context not created");
				ctx_ = nullptr;
				return false;
			}
			xess_vk_init_params_t init {};
			init.outputResolution = {f.display.width, f.display.height};
			init.qualitySetting   = quality;
			init.initFlags        = flags;
			if (const auto r = g_api.init(ctx_, &init); r != XESS_RESULT_SUCCESS) {
				Log("[xess] init failed (%d)", static_cast<int>(r));
				g_api.destroy_context(ctx_);
				ctx_ = nullptr;
				return false;
			}
			flags_ = flags;
			quality_ = quality;
			display_ = f.display;
			scale_[0] = scale_[1] = 0;
			char buf[96];
			std::snprintf(buf, sizeof buf, "libxess %u.%u.%u, quality setting %d, flags 0x%x", g_version.major, g_version.minor, g_version.patch,
			              static_cast<int>(quality), flags);
			detail_ = buf;
			Log("[xess] context: %ux%u output, %s", f.display.width, f.display.height, detail_.c_str());
		}
		// Velocity: pixels toward the previous frame (KytyPS5's convention; XeSS's own sign).
		const float sx = f.mv_scale[0] != 0 ? f.mv_scale[0] : 1.0f, sy = f.mv_scale[1] != 0 ? f.mv_scale[1] : 1.0f;
		if (sx != scale_[0] || sy != scale_[1]) {
			g_api.set_velocity_scale(ctx_, sx, sy);
			scale_[0] = sx;
			scale_[1] = sy;
		}
		xess_vk_execute_params_t exec {};
		exec.colorTexture    = View(*f.color);
		exec.velocityTexture = View(*f.motion);
		exec.depthTexture    = View(*f.depth);
		exec.outputTexture   = View(*f.output);
		exec.jitterOffsetX   = f.jitter[0];
		exec.jitterOffsetY   = f.jitter[1];
		exec.exposureScale   = 1.0f;
		exec.resetHistory    = f.reset ? 1u : 0u;
		exec.inputWidth      = f.render.width;
		exec.inputHeight     = f.render.height;
		if (const auto r = g_api.execute(ctx_, f.cb, &exec); r != XESS_RESULT_SUCCESS) {
			if (errors_++ < 8) Log("[xess] execute failed (%d)", static_cast<int>(r));
			return false;
		}
		if (frames_++ == 0)
			Log("[xess] first execute: %ux%u -> %ux%u, velocity scale (%.0f, %.0f), jitter (%.3f, %.3f)", f.render.width, f.render.height,
			    f.display.width, f.display.height, sx, sy, f.jitter[0], f.jitter[1]);
		return true;
	}

private:
	xess_context_handle_t   ctx_ = nullptr;
	std::vector<xess_context_handle_t> old_;
	uint32_t                flags_ = 0, errors_ = 0;
	xess_quality_settings_t quality_ = XESS_QUALITY_SETTING_QUALITY;
	VkExtent2D              display_ {};
	float                   scale_[2] {0, 0};
	uint64_t                frames_ = 0;
	std::string             detail_;
};

} // namespace

void BeforeCreateInstance() {
	Load();
}
void InstanceExtensions(std::vector<const char*>& names) {
	for (const auto& e: g_instance_ext) names.push_back(e.c_str());
}
void DeviceExtensions(VkPhysicalDevice physical, std::vector<const char*>& names) {
	if (!g_loaded || R().instance == VK_NULL_HANDLE) return;
	uint32_t count = 0;
	const char* const* list = nullptr;
	InternalScope scope;
	if (g_api.get_device_extensions(R().instance, physical, &count, &list) != XESS_RESULT_SUCCESS) {
		g_why = "XeSS no soporta esta GPU / driver";
		Log("[xess] this GPU or driver is not supported (xessVKGetRequiredDeviceExtensions failed)");
		return;
	}
	g_device_ext.clear();
	for (uint32_t i = 0; i < count; ++i) {
		g_device_ext.emplace_back(list[i]);
		names.push_back(list[i]);
	}
}
void* DeviceFeatureChain(VkPhysicalDevice physical) {
	if (!g_loaded || R().instance == VK_NULL_HANDLE) return nullptr;
	void* chain = nullptr;
	InternalScope scope;
	if (g_api.get_device_features(R().instance, physical, &chain) != XESS_RESULT_SUCCESS) return nullptr;
	return chain;
}
bool Ready(std::string& why) {
	if (!g_loaded) {
		why = g_why;
		return false;
	}
	for (const auto& e: g_device_ext) {
		if (!DeviceHas(e.c_str())) {
			why = "al dispositivo le falta " + e;
			return false;
		}
	}
	if (!g_why.empty()) {
		why = g_why;
		return false;
	}
	return true;
}
std::unique_ptr<Upscaler> Create(std::string& why) {
	if (!Ready(why)) return nullptr;
	return std::make_unique<Xess>();
}

#else  // built without the XeSS SDK headers

void BeforeCreateInstance() {}
void InstanceExtensions(std::vector<const char*>&) {}
void DeviceExtensions(VkPhysicalDevice, std::vector<const char*>&) {}
void* DeviceFeatureChain(VkPhysicalDevice) { return nullptr; }
bool Ready(std::string& why) {
	why = "esta build no incluye XeSS";
	return false;
}
std::unique_ptr<Upscaler> Create(std::string& why) {
	why = "this build has no XeSS";
	return nullptr;
}

#endif

} // namespace inj::upscale::xess
