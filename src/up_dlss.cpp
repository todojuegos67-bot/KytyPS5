// DLSS super resolution through NVIDIA Streamline (the public SDK, v2.14: sl.interposer.dll, sl.common.dll, sl.dlss.dll and
// nvngx_dlss.dll from NVIDIA's own release, in a "streamline" folder next to the exe; nothing else). The tagging,
// constants and evaluate follow KytyPS5's DLSS path (src/local/frame-gen.cpp).
//
// Streamline in "manual hooking" mode (sl::PreferenceFlags::eUseManualHooking): it does not become the game's Vulkan
// loader (we already are, and DXVK must keep loading us); we do what its proxies would: slInit before the game's
// vkCreateInstance, the instance / device extensions and Vulkan 1.2 / 1.3 features slGetFeatureRequirements lists added
// at creation, and slSetVulkanInfo with the device that presents (DXVK's real one, not its probe device).
#include "upscale_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#ifdef INJ_STREAMLINE
#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers.h>
#include <sl_helpers_vk.h>
#endif

namespace inj::upscale::dlss {

#ifdef INJ_STREAMLINE
namespace {

struct Api {
	PFun_slInit*                   init = nullptr;
	PFun_slShutdown*               shutdown = nullptr;
	PFun_slIsFeatureSupported*     is_feature_supported = nullptr;
	PFun_slIsFeatureLoaded*        is_feature_loaded = nullptr;
	PFun_slGetFeatureRequirements* get_requirements = nullptr;
	PFun_slGetFeatureFunction*     get_feature_function = nullptr;
	PFun_slGetNewFrameToken*       get_new_frame_token = nullptr;
	PFun_slSetConstants*           set_constants = nullptr;
	PFun_slSetTagForFrame*         set_tag_for_frame = nullptr;
	PFun_slEvaluateFeature*        evaluate = nullptr;
	PFun_slSetVulkanInfo*          set_vulkan_info = nullptr;
	PFun_slDLSSSetOptions*         set_options = nullptr;
	PFun_slDLSSGetOptimalSettings* get_optimal = nullptr;
};
Api         g_api;
bool        g_tried = false, g_inited = false, g_device_set = false, g_ready = false;
std::string g_why = "Streamline no cargado";
std::vector<std::string> g_instance_ext, g_device_ext, g_features12, g_features13;
VkDevice    g_device = VK_NULL_HANDLE;

// Streamline's plugins (loaded by slInit) and NGX: their "vulkan-1.dll" loads redirected to us too.
void PatchPlugins() {
	for (const wchar_t* name: {L"sl.interposer.dll", L"sl.common.dll", L"sl.dlss.dll", L"nvngx_dlss.dll"})
		if (HMODULE m = GetModuleHandleW(name)) {
			char who[64];
			std::snprintf(who, sizeof who, "%ls", name);
			RedirectVulkanLoads(m, who);
		}
}

void SlLog(sl::LogType type, const char* message) {
	static uint32_t count = 0;
	if (type == sl::LogType::eInfo && count > 40) return;
	if (count++ > 400) return;
	std::string m = message != nullptr ? message : "";
	while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
	Log("[sl] %s%s", type == sl::LogType::eError ? "ERROR " : type == sl::LogType::eWarn ? "warn " : "", m.c_str());
}

sl::DLSSMode ModeFor(int quality) {
	switch (quality) {
	case 1: return sl::DLSSMode::eDLAA;
	case 2: return sl::DLSSMode::eMaxQuality;
	case 3: return sl::DLSSMode::eBalanced;
	case 4: return sl::DLSSMode::eMaxPerformance;
	default: return sl::DLSSMode::eUltraPerformance;
	}
}
const char* ModeName(sl::DLSSMode m) {
	switch (m) {
	case sl::DLSSMode::eDLAA: return "DLAA";
	case sl::DLSSMode::eMaxQuality: return "Quality";
	case sl::DLSSMode::eBalanced: return "Balanced";
	case sl::DLSSMode::eMaxPerformance: return "Performance";
	case sl::DLSSMode::eUltraPerformance: return "Ultra Performance";
	case sl::DLSSMode::eUltraQuality: return "Ultra Quality";
	default: return "off";
	}
}

bool Invert4(const float* m, float* out) {
	float inv[16];
	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
	const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (std::fabs(det) < 1e-12f) return false;
	for (int i = 0; i < 16; ++i) out[i] = inv[i] / det;
	return true;
}

sl::float4x4 Matrix(const float* m) {
	sl::float4x4 r;
	for (uint32_t i = 0; i < 4; ++i) r.setRow(i, sl::float4(m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]));
	return r;
}

sl::Resource Res(const OwnImage& img, VkImageLayout layout) {
	sl::Resource r(sl::ResourceType::eTex2d, reinterpret_cast<void*>(img.image), nullptr, reinterpret_cast<void*>(img.view),
	               static_cast<uint32_t>(layout));
	r.width        = img.extent.width;
	r.height       = img.extent.height;
	r.nativeFormat = static_cast<uint32_t>(img.format);
	r.mipLevels    = 1;
	r.arrayLayers  = 1;
	r.flags        = 0;
	r.usage        = static_cast<uint32_t>(img.usage);
	return r;
}

class Dlss final : public Upscaler {
public:
	const char* Name() const override { return "DLSS"; }
	std::string Detail() const override { return detail_; }

	bool Dispatch(const Frame& f) override {
		const sl::ViewportHandle viewport {0};
		if (!options_set_ || f.quality != quality_ || f.hdr != hdr_ || f.render.width != render_.width || f.render.height != render_.height ||
		    f.display.width != display_.width || f.display.height != display_.height) {
			if (!SetOptions(viewport, f)) return false;
		}
		uint32_t index = static_cast<uint32_t>(f.index);
		sl::FrameToken* token = nullptr;
		if (g_api.get_new_frame_token(token, &index) != sl::Result::eOk || token == nullptr) {
			if (errors_++ < 8) Log("[dlss] slGetNewFrameToken failed");
			return false;
		}
		sl::Constants c {};
		float proj[16];
		if (f.cam.found) {
			std::memcpy(proj, f.cam.proj, sizeof(proj));
		} else {  // (unknown camera: a plain perspective at the render aspect; DLSS SR reads the motion vectors)
			std::memset(proj, 0, sizeof(proj));
			const float b = 1.0f / std::tan(0.5f), a = b * static_cast<float>(f.render.height) / static_cast<float>(f.render.width);
			proj[0] = a; proj[5] = b; proj[10] = f.reversed ? 0.0f : 1.0f; proj[11] = 1.0f; proj[14] = f.reversed ? 0.1f : -0.1f;
		}
		float inv[16];
		const float identity[16] {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		if (!Invert4(proj, inv)) std::memcpy(inv, identity, sizeof(inv));
		c.cameraViewToClip = Matrix(proj);
		c.clipToCameraView = Matrix(inv);
		c.clipToLensClip   = Matrix(identity);
		c.clipToPrevClip   = Matrix(identity);
		c.prevClipToClip   = Matrix(identity);
		c.jitterOffset     = sl::float2(f.jitter[0], f.jitter[1]);
		// (Streamline's vectors are render-size UV units: its scale makes them so from our pixels-toward-previous.)
		c.mvecScale = sl::float2(f.mv_scale[0] / static_cast<float>(f.render.width), f.mv_scale[1] / static_cast<float>(f.render.height));
		c.cameraPinholeOffset = sl::float2(0, 0);
		c.cameraPos   = sl::float3(0, 0, 0);
		c.cameraUp    = sl::float3(0, 1, 0);
		c.cameraRight = sl::float3(1, 0, 0);
		c.cameraFwd   = sl::float3(0, 0, 1);
		c.cameraNear  = f.cam.found ? f.cam.near_plane : 0.1f;
		c.cameraFar   = f.cam.found && std::isfinite(f.cam.far_plane) ? f.cam.far_plane : 100000.0f;
		c.cameraFOV   = f.cam.found ? f.cam.fov : 1.0f;
		c.cameraAspectRatio = static_cast<float>(f.render.width) / static_cast<float>(f.render.height);
		c.motionVectorsInvalidValue = sl::INVALID_FLOAT;
		c.depthInverted          = f.reversed ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		c.cameraMotionIncluded   = sl::Boolean::eTrue;
		c.motionVectors3D        = sl::Boolean::eFalse;
		c.reset                  = f.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		c.orthographicProjection = sl::Boolean::eFalse;
		c.motionVectorsDilated   = sl::Boolean::eFalse;
		c.motionVectorsJittered  = sl::Boolean::eFalse;
		if (const auto r = g_api.set_constants(c, *token, viewport); r != sl::Result::eOk) {
			if (errors_++ < 8) Log("[dlss] slSetConstants failed (%s)", sl::getResultAsStr(r));
			return false;
		}
		auto color  = Res(*f.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		auto depth  = Res(*f.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		auto motion = Res(*f.motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		auto output = Res(*f.output, VK_IMAGE_LAYOUT_GENERAL);
		const sl::Extent render {0, 0, f.render.width, f.render.height};
		const sl::Extent full {0, 0, f.display.width, f.display.height};
		sl::ResourceTag tags[4] {
		    sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
		    sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
		};
		auto* cmd = reinterpret_cast<sl::CommandBuffer*>(f.cb);
		if (const auto r = g_api.set_tag_for_frame(*token, viewport, tags, 4, cmd); r != sl::Result::eOk) {
			if (errors_++ < 8) Log("[dlss] slSetTagForFrame failed (%s)", sl::getResultAsStr(r));
			return false;
		}
		const sl::BaseStructure* inputs[] {&viewport};
		const auto r = g_api.evaluate(sl::kFeatureDLSS, *token, inputs, 1, cmd);
		if (r != sl::Result::eOk && r != sl::Result::eWarnOutOfVRAM) {
			if (errors_++ < 8) Log("[dlss] slEvaluateFeature failed (%s)", sl::getResultAsStr(r));
			return false;
		}
		if (frames_++ == 0)
			Log("[dlss] first evaluate: %ux%u -> %ux%u, %s, mvec scale (%.5f, %.5f), jitter (%.3f, %.3f), depth %s, camera %s", f.render.width,
			    f.render.height, f.display.width, f.display.height, detail_.c_str(), c.mvecScale.x, c.mvecScale.y, f.jitter[0], f.jitter[1],
			    f.reversed ? "inverted" : "forward", f.cam.found ? "from the game's projection" : "default");
		return true;
	}

private:
	bool SetOptions(const sl::ViewportHandle& viewport, const Frame& f) {
		sl::DLSSOptions o {};
		o.outputWidth     = f.display.width;
		o.outputHeight    = f.display.height;
		o.colorBuffersHDR = f.hdr ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		o.useAutoExposure = f.hdr ? sl::Boolean::eTrue : sl::Boolean::eFalse;
		// The asked mode if the game's render size is in its range, else the one whose range holds it (KytyPS5's search).
		sl::DLSSMode modes[] {ModeFor(f.quality), sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced, sl::DLSSMode::eMaxPerformance,
		                      sl::DLSSMode::eUltraPerformance, sl::DLSSMode::eUltraQuality, sl::DLSSMode::eDLAA};
		bool found = false;
		sl::DLSSOptimalSettings best {};
		for (auto m: modes) {
			o.mode = m;
			sl::DLSSOptimalSettings st {};
			if (g_api.get_optimal == nullptr || g_api.get_optimal(o, st) != sl::Result::eOk || st.renderWidthMax == 0) continue;
			if (f.render.width >= st.renderWidthMin && f.render.width <= st.renderWidthMax && f.render.height >= st.renderHeightMin &&
			    f.render.height <= st.renderHeightMax) {
				found = true;
				best  = st;
				break;
			}
		}
		if (!found) {
			if (errors_++ < 8)
				Log("[dlss] the game renders %ux%u: outside every DLSS mode's range for %ux%u (change the game's resolution scale)",
				    f.render.width, f.render.height, f.display.width, f.display.height);
			return false;
		}
		if (const auto r = g_api.set_options(viewport, o); r != sl::Result::eOk) {
			if (errors_++ < 8) Log("[dlss] slDLSSSetOptions failed (%s)", sl::getResultAsStr(r));
			return false;
		}
		options_set_ = true;
		quality_ = f.quality;
		hdr_     = f.hdr;
		render_  = f.render;
		display_ = f.display;
		char buf[160];
		std::snprintf(buf, sizeof buf, "DLSS mode %s (its range %ux%u-%ux%u), %s colour", ModeName(o.mode), best.renderWidthMin,
		              best.renderHeightMin, best.renderWidthMax, best.renderHeightMax, f.hdr ? "HDR" : "LDR");
		detail_ = buf;
		Log("[dlss] options: %ux%u -> %ux%u, %s", f.render.width, f.render.height, f.display.width, f.display.height, detail_.c_str());
		return true;
	}

	bool        options_set_ = false, hdr_ = false;
	int         quality_ = 0;
	VkExtent2D  render_ {}, display_ {};
	uint32_t    errors_ = 0;
	uint64_t    frames_ = 0;
	std::string detail_;
};

} // namespace

void BeforeCreateInstance() {
	if (g_tried) return;
	g_tried = true;
	std::wstring dir = ExeDir() + L"streamline\\";
	if (GetFileAttributesW((dir + L"sl.interposer.dll").c_str()) == INVALID_FILE_ATTRIBUTES) dir = ExeDir();
	// (Its own dependencies resolved against the exe's folder first; then "vulkan-1.dll" -> us in its imports.)
	HMODULE sl = LoadLibraryExW((dir + L"sl.interposer.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	RedirectVulkanLoads(sl, "sl.interposer.dll");
	if (sl == nullptr) {
		g_why = "no esta streamline\\sl.interposer.dll junto al exe";
		Log("[dlss] no streamline\\sl.interposer.dll next to the exe: DLSS unavailable");
		return;
	}
	const auto load = [&](auto& fn, const char* name) {
		fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(reinterpret_cast<void*>(GetProcAddress(sl, name)));
		return fn != nullptr;
	};
	if (!load(g_api.init, "slInit") || !load(g_api.shutdown, "slShutdown") || !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
	    !load(g_api.is_feature_loaded, "slIsFeatureLoaded") || !load(g_api.get_requirements, "slGetFeatureRequirements") ||
	    !load(g_api.get_feature_function, "slGetFeatureFunction") || !load(g_api.get_new_frame_token, "slGetNewFrameToken") ||
	    !load(g_api.set_constants, "slSetConstants") || !load(g_api.set_tag_for_frame, "slSetTagForFrame") ||
	    !load(g_api.evaluate, "slEvaluateFeature") || !load(g_api.set_vulkan_info, "slSetVulkanInfo")) {
		g_why = "sl.interposer.dll sin las funciones esperadas";
		Log("[dlss] sl.interposer.dll lacks an entry point: DLSS unavailable");
		return;
	}
	static std::wstring plugins;
	plugins = dir.substr(0, dir.size() - 1);
	static const wchar_t* paths[] {plugins.c_str()};
	static const sl::Feature features[] {sl::kFeatureDLSS};
	sl::Preferences p {};
	p.showConsole        = false;
	p.logLevel           = sl::LogLevel::eDefault;
	p.pathsToPlugins     = paths;
	p.numPathsToPlugins  = 1;
	p.logMessageCallback = &SlLog;
	p.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	p.featuresToLoad    = features;
	p.numFeaturesToLoad = 1;
	p.engine            = sl::EngineType::eCustom;
	p.engineVersion     = "1.0";
	p.projectId         = "3d4f6c2a-8e1b-4b7d-9a5e-2f6c1d8e0b47";
	p.renderAPI         = sl::RenderAPI::eVulkan;
	if (const auto r = g_api.init(p, sl::kSDKVersion); r != sl::Result::eOk) {
		g_why = std::string("slInit fallo (") + sl::getResultAsStr(r) + ")";
		Log("[dlss] slInit failed (%s): DLSS unavailable", sl::getResultAsStr(r));
		return;
	}
	g_inited = true;
	PatchPlugins();
	sl::FeatureRequirements req {};
	if (g_api.get_requirements(sl::kFeatureDLSS, req) == sl::Result::eOk) {
		for (uint32_t i = 0; i < req.vkNumInstanceExtensions; ++i) g_instance_ext.emplace_back(req.vkInstanceExtensions[i]);
		for (uint32_t i = 0; i < req.vkNumDeviceExtensions; ++i) g_device_ext.emplace_back(req.vkDeviceExtensions[i]);
		for (uint32_t i = 0; i < req.vkNumFeatures12; ++i) g_features12.emplace_back(req.vkFeatures12[i]);
		for (uint32_t i = 0; i < req.vkNumFeatures13; ++i) g_features13.emplace_back(req.vkFeatures13[i]);
		Log("[dlss] Streamline %d.%d.%d loaded from %ls (manual hooking): DLSS asks for %u instance / %u device extensions, %u + %u "
		    "Vulkan 1.2 / 1.3 features, %u graphics / %u compute queues",
		    SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH, dir.c_str(), req.vkNumInstanceExtensions, req.vkNumDeviceExtensions,
		    req.vkNumFeatures12, req.vkNumFeatures13, req.vkNumGraphicsQueuesRequired, req.vkNumComputeQueuesRequired);
	} else {
		Log("[dlss] slGetFeatureRequirements(DLSS) failed: the device is made without its extras");
	}
	g_why = "esperando al dispositivo";
}

void InstanceExtensions(std::vector<const char*>& names) {
	for (const auto& e: g_instance_ext) names.push_back(e.c_str());
}
void DeviceExtensions(std::vector<const char*>& names) {
	for (const auto& e: g_device_ext) names.push_back(e.c_str());
}
void DeviceFeatures(VkPhysicalDeviceVulkan12Features& f12, VkPhysicalDeviceVulkan13Features& f13) {
	if (!g_inited) return;
	std::vector<const char*> n12, n13;
	for (const auto& s: g_features12) n12.push_back(s.c_str());
	for (const auto& s: g_features13) n13.push_back(s.c_str());
	const VkPhysicalDeviceVulkan12Features w12 = sl::getVkPhysicalDeviceVulkan12Features(static_cast<uint32_t>(n12.size()), n12.data());
	const VkPhysicalDeviceVulkan13Features w13 = sl::getVkPhysicalDeviceVulkan13Features(static_cast<uint32_t>(n13.size()), n13.data());
	// OR (the caller ANDs with what the device has).
	const auto merge = [](void* dst, const void* src, size_t size) {
		for (size_t at = 16; at + 4 <= size; at += 4) {
			VkBool32 s = 0, d = 0;
			std::memcpy(&s, static_cast<const uint8_t*>(src) + at, 4);
			std::memcpy(&d, static_cast<uint8_t*>(dst) + at, 4);
			d = d || s;
			std::memcpy(static_cast<uint8_t*>(dst) + at, &d, 4);
		}
	};
	merge(&f12, &w12, sizeof(f12));
	merge(&f13, &w13, sizeof(f13));
}

void OnFirstPresent(VkDevice device) {
	if (!g_inited || (g_device_set && g_device == device)) return;
	if (InstanceOf(R().physical) == VK_NULL_HANDLE) { Log("[dlss] no live instance for the active device: Streamline not told"); return; }
	PatchPlugins();  // (before Streamline loads the Vulkan loader for this device: ours)
	sl::VulkanInfo info {};
	info.device             = device;
	info.instance           = InstanceOf(R().physical);  // (the live instance owning the active device's GPU)
	info.physicalDevice     = R().physical;
	info.graphicsQueueFamily = Q().graphics_family;
	info.graphicsQueueIndex  = QueueCount(Q().graphics_family);  // (where queues Streamline asks for would start: none for DLSS SR)
	info.computeQueueFamily  = Q().compute_family != UINT32_MAX ? Q().compute_family : Q().graphics_family;
	info.computeQueueIndex   = QueueCount(info.computeQueueFamily);
	if (const auto r = g_api.set_vulkan_info(info); r != sl::Result::eOk) {
		g_why = std::string("slSetVulkanInfo fallo (") + sl::getResultAsStr(r) + ")";
		Log("[dlss] slSetVulkanInfo failed (%s)", sl::getResultAsStr(r));
		return;
	}
	g_device_set = true;
	g_device = device;
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = R().physical;
	bool loaded = false;
	const auto supported = g_api.is_feature_supported(sl::kFeatureDLSS, adapter);
	if (supported != sl::Result::eOk || g_api.is_feature_loaded(sl::kFeatureDLSS, loaded) != sl::Result::eOk || !loaded) {
		g_why = std::string("DLSS no soportado en esta GPU (") + sl::getResultAsStr(supported) + ")";
		Log("[dlss] DLSS not supported on this device (%s, loaded %d)", sl::getResultAsStr(supported), loaded ? 1 : 0);
		return;
	}
	void* fn = nullptr;
	if (g_api.get_feature_function(sl::kFeatureDLSS, "slDLSSSetOptions", fn) == sl::Result::eOk) g_api.set_options = reinterpret_cast<PFun_slDLSSSetOptions*>(fn);
	fn = nullptr;
	if (g_api.get_feature_function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", fn) == sl::Result::eOk)
		g_api.get_optimal = reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(fn);
	if (g_api.set_options == nullptr) {
		g_why = "sl.dlss sin slDLSSSetOptions";
		Log("[dlss] no slDLSSSetOptions: DLSS unavailable");
		return;
	}
	for (const auto& e: g_device_ext) {
		if (!DeviceHas(e.c_str())) {
			Log("[dlss] the device was made without %s (asked by DLSS): it may fail", e.c_str());
		}
	}
	g_ready = true;
	g_why.clear();
	Log("[dlss] DLSS ready on this device (Streamline told about it with slSetVulkanInfo)");
}

bool Ready(std::string& why) {
	why = g_why;
	return g_ready;
}

std::unique_ptr<Upscaler> Create(std::string& why) {
	if (!g_ready) {
		why = g_why;
		return nullptr;
	}
	return std::make_unique<Dlss>();
}

void Shutdown() {
	if (g_inited && g_api.shutdown != nullptr) g_api.shutdown();
	g_inited = g_ready = g_device_set = false;
}

#else  // built without the Streamline SDK headers

void BeforeCreateInstance() {}
void InstanceExtensions(std::vector<const char*>&) {}
void DeviceExtensions(std::vector<const char*>&) {}
void DeviceFeatures(VkPhysicalDeviceVulkan12Features&, VkPhysicalDeviceVulkan13Features&) {}
void OnFirstPresent(VkDevice) {}
bool Ready(std::string& why) {
	why = "esta build no incluye Streamline";
	return false;
}
std::unique_ptr<Upscaler> Create(std::string& why) {
	why = "this build has no Streamline";
	return nullptr;
}
void Shutdown() {}

#endif

} // namespace inj::upscale::dlss
