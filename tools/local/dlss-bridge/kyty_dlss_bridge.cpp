// SPDX-FileCopyrightText: Copyright 2026 IFreemz (shadps4_dlss bridge), bbport contributors, KytyPS5 contributors
// SPDX-License-Identifier: MIT
//
// libkyty_dlss.so: the only part of KytyPS5 that uses NVIDIA's DLSS (NGX) SDK, on Linux (Windows has Streamline,
// src/local/frame-gen.cpp). Built on its own (build-dlss-linux.sh); the emulator loads it at run time
// (src/local/kyty-dlss-bridge.h). Adapted from bbport's bridge (spacepepperoni/uncharted_Linux, gpu/dlss_bridge),
// itself from IFreemz/shadPS4-Bloodborne-DLSS-FSR.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "kyty-dlss-bridge.h" // Vulkan first: the NGX headers expect its types

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

namespace {

constexpr char ProjectId[]     = "6a3f1d5e-4b7c-4f2a-9d8e-1c2b3a4d5e6f"; // (the emulator's Streamline project id)
constexpr char EngineVersion[] = "kytyps5-linux";
// A retired feature is released this many evaluations later: the GPU is done with the frames that used it.
constexpr uint64_t RetireEvaluations = 16;

struct Retired {
	NVSDK_NGX_Handle*    handle;
	NVSDK_NGX_Parameter* parameters;
	uint64_t             at; // evaluations when retired
};

struct State {
	KytyDlssLogFn                  log {};
	std::wstring                   library_directory;
	std::wstring                   data_directory;
	const wchar_t*                 library_path {};
	NVSDK_NGX_FeatureCommonInfo    common {};
	NVSDK_NGX_FeatureDiscoveryInfo discovery {};
	NVSDK_NGX_FeatureDiscoveryInfo discovery_fg {}; // frame generation
	NVSDK_NGX_Parameter*           capabilities {};
	NVSDK_NGX_Parameter*           parameters {};
	NVSDK_NGX_Handle*              feature {};
	KytyDlssFeature                feature_desc {};
	// Frame generation: its own parameter map (NVIDIA's advice with two features).
	NVSDK_NGX_Parameter* fg_parameters {};
	NVSDK_NGX_Handle*    fg_feature {};
	KytyDlssgFeature     fg_desc {};
	bool                 fg_device_supported {};
	int32_t              fg_max_count {};
	// The extensions of both features, without duplicates.
	std::vector<VkExtensionProperties> instance_extensions, device_extensions;
	std::vector<Retired>               retired;
	// DLSS runs on the renderer thread (or its recording worker), frame generation on the present thread: one NGX
	// call at a time.
	std::mutex                         mutex;
	uint64_t                           evaluations {};
	VkDevice                           device {};
	bool                               configured {};
	bool                               initialized {};
} state;

template <typename... Args>
void Log(int warning, const char* format, Args... args) {
	if (!state.log) return;
	std::array<char, 1024> text {};
	std::snprintf(text.data(), text.size(), format, args...);
	state.log(warning, text.data());
}

bool Check(const char* operation, NVSDK_NGX_Result result) {
	if (NVSDK_NGX_FAILED(result)) {
		Log(1, "%s failed: 0x%08x", operation, static_cast<unsigned>(result));
		return false;
	}
	return true;
}

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
	Log(0, "NGX: %s", message ? message : "");
}

// UTF-8 to wchar_t (NGX's paths; UTF-32 on Linux).
std::wstring Wide(const char* text) {
	std::wstring out;
	if (text == nullptr) return out;
	const auto* p = reinterpret_cast<const unsigned char*>(text);
	while (*p != 0) {
		uint32_t code = *p;
		int      more = 0;
		if (code >= 0xf0) {
			code &= 0x07;
			more = 3;
		} else if (code >= 0xe0) {
			code &= 0x0f;
			more = 2;
		} else if (code >= 0xc0) {
			code &= 0x1f;
			more = 1;
		}
		++p;
		for (; more > 0 && (*p & 0xc0) == 0x80; --more, ++p) code = (code << 6) | (*p & 0x3f);
		out.push_back(static_cast<wchar_t>(code));
	}
	return out;
}

int32_t Configure(const char* library_directory, const char* data_directory, KytyDlssLogFn log) {
	state.log               = log;
	state.library_directory = Wide(library_directory != nullptr ? library_directory : ".");
	state.data_directory    = Wide(data_directory != nullptr ? data_directory : ".");
	state.library_path      = state.library_directory.c_str();
	state.common.PathListInfo = {&state.library_path, 1};
	// KYTY_DLSS_NGX_LOG=1: NGX's own log (verbose) through the emulator's.
	const char* ngx_log       = std::getenv("KYTY_DLSS_NGX_LOG");
	state.common.LoggingInfo  = {NgxLog, ngx_log != nullptr && *ngx_log == '1' ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE
	                                                                          : NVSDK_NGX_LOGGING_LEVEL_OFF,
	                             false};
	state.discovery.SDKVersion                  = NVSDK_NGX_Version_API;
	state.discovery.FeatureID                   = NVSDK_NGX_Feature_SuperSampling;
	state.discovery.Identifier.IdentifierType   = NVSDK_NGX_Application_Identifier_Type_Project_Id;
	state.discovery.Identifier.v.ProjectDesc    = {ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion};
	state.discovery.ApplicationDataPath         = state.data_directory.c_str();
	state.discovery.FeatureInfo                 = &state.common;
	state.discovery_fg                          = state.discovery;
	state.discovery_fg.FeatureID                = NVSDK_NGX_Feature_FrameGeneration;
	state.configured                            = true;
	return 1;
}

void Merge(std::vector<VkExtensionProperties>& into, const VkExtensionProperties* add, uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		const auto same = std::find_if(into.begin(), into.end(),
		                               [&](const auto& have) { return std::strcmp(have.extensionName, add[i].extensionName) == 0; });
		if (same == into.end()) {
			into.push_back(add[i]);
		} else {
			same->specVersion = std::max(same->specVersion, add[i].specVersion);
		}
	}
}

int32_t InstanceExtensions(uint32_t* count, const VkExtensionProperties** extensions) {
	VkExtensionProperties* required {};
	uint32_t               required_count {};
	if (!state.configured || count == nullptr || extensions == nullptr ||
	    !Check("Instance extension query",
	           NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&state.discovery, &required_count, &required)))
		return 0;
	state.instance_extensions.clear();
	Merge(state.instance_extensions, required, required_count);
	// Frame generation's own needs; without them only DLSS is offered.
	if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&state.discovery_fg, &required_count,
	                                                                               &required)))
		Merge(state.instance_extensions, required, required_count);
	*count      = static_cast<uint32_t>(state.instance_extensions.size());
	*extensions = state.instance_extensions.data();
	return 1;
}

int32_t DeviceExtensions(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
                         const VkExtensionProperties** extensions) {
	if (!state.configured || count == nullptr || extensions == nullptr) return 0;
	NVSDK_NGX_FeatureRequirement support {};
	if (!Check("Feature requirement query",
	           NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.discovery, &support)))
		return 0;
	if (support.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
		Log(1, "DLSS is not supported on this GPU or driver (code 0x%x)", static_cast<unsigned>(support.FeatureSupported));
		return 0;
	}
	VkExtensionProperties* required {};
	uint32_t               required_count {};
	if (!Check("Device extension query", NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
	                                         instance, physical, &state.discovery, &required_count, &required)))
		return 0;
	state.device_extensions.clear();
	Merge(state.device_extensions, required, required_count);
	NVSDK_NGX_FeatureRequirement fg_support {};
	state.fg_device_supported =
	    NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.discovery_fg, &fg_support)) &&
	    fg_support.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported &&
	    NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physical, &state.discovery_fg,
	                                                                             &required_count, &required));
	if (state.fg_device_supported) {
		Merge(state.device_extensions, required, required_count);
	} else {
		Log(0, "frame generation is not supported on this GPU or driver (code 0x%x)",
		    static_cast<unsigned>(fg_support.FeatureSupported));
	}
	*count      = static_cast<uint32_t>(state.device_extensions.size());
	*extensions = state.device_extensions.data();
	return 1;
}

int32_t Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr get_instance_proc,
                   PFN_vkGetDeviceProcAddr get_device_proc) {
	if (!state.configured || state.initialized) return 0;
	if (!Check("NGX initialization",
	           NVSDK_NGX_VULKAN_Init_with_ProjectID(ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion,
	                                                state.data_directory.c_str(), instance, physical, device,
	                                                get_instance_proc, get_device_proc, &state.common)))
		return 0;
	state.initialized = true;
	state.device      = device;
	if (!Check("Capability query", NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.capabilities))) return 0;
	int available {}, needs_driver {};
	NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
	NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
	if (needs_driver) {
		Log(1, "DLSS needs a newer NVIDIA driver");
		return 0;
	}
	if (!available) {
		Log(1, "DLSS super resolution is not available on this system");
		return 0;
	}
	int      fg_available {}, fg_needs_driver {};
	unsigned fg_max {};
	NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_FrameGeneration_Available, &fg_available);
	NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &fg_needs_driver);
	if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetUI(state.capabilities, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &fg_max)))
		fg_max = 1;
	if (state.fg_device_supported && fg_available) {
		state.fg_max_count = static_cast<int32_t>(std::clamp(fg_max, 1u, 3u));
		Log(0, "frame generation available (up to %dx)", state.fg_max_count + 1);
	} else if (fg_needs_driver) {
		Log(1, "frame generation needs a newer NVIDIA driver");
	}
	return 1;
}

// Retired features: released once the GPU is done with them (all: at Shutdown, the caller waited).
void Collect(bool all) {
	auto keep = state.retired.begin();
	for (auto& entry: state.retired) {
		if (all || state.evaluations - entry.at >= RetireEvaluations) {
			if (entry.handle != nullptr) NVSDK_NGX_VULKAN_ReleaseFeature(entry.handle);
			if (entry.parameters != nullptr) NVSDK_NGX_VULKAN_DestroyParameters(entry.parameters);
		} else {
			*keep++ = entry;
		}
	}
	state.retired.erase(keep, state.retired.end());
}

void Retire(NVSDK_NGX_Handle*& handle, NVSDK_NGX_Parameter*& parameters) {
	if (handle != nullptr || parameters != nullptr) state.retired.push_back({handle, parameters, state.evaluations});
	handle     = nullptr;
	parameters = nullptr;
}

int32_t CreateFeature(VkCommandBuffer command, const KytyDlssFeature* desc) {
	const std::lock_guard lock(state.mutex);
	if (!state.initialized || state.capabilities == nullptr || desc == nullptr) return 0;
	Retire(state.feature, state.parameters);
	if (!Check("Parameter allocation", NVSDK_NGX_VULKAN_AllocateParameters(&state.parameters))) return 0;
	for (const char* key: {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
	                       NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
	                       NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
	                       NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
	                       NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
		NVSDK_NGX_Parameter_SetUI(state.parameters, key, desc->preset);
	static constexpr std::array qualities {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
	                                       NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
	                                       NVSDK_NGX_PerfQuality_Value_UltraPerformance};
	NVSDK_NGX_DLSS_Create_Params settings {};
	settings.Feature.InWidth            = desc->input_width;
	settings.Feature.InHeight           = desc->input_height;
	settings.Feature.InTargetWidth      = desc->output_width;
	settings.Feature.InTargetHeight     = desc->output_height;
	settings.Feature.InPerfQualityValue = qualities[std::clamp(desc->quality, 0, 4)];
	settings.InFeatureCreateFlags =
	    NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | (desc->depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) |
	    (desc->hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0);
	const auto result = NGX_VULKAN_CREATE_DLSS_EXT1(state.device, command, 1, 1, &state.feature, state.parameters, &settings);
	if (!Check("DLSS feature creation", result) || state.feature == nullptr) {
		state.feature = nullptr;
		return 0;
	}
	state.feature_desc = *desc;
	Log(0, "DLSS %ux%u -> %ux%u, quality %d, preset %u%s", desc->input_width, desc->input_height, desc->output_width,
	    desc->output_height, desc->quality, desc->preset, desc->hdr ? ", HDR input" : "");
	return 1;
}

NVSDK_NGX_Resource_VK Wrap(const KytyDlssImage& image, bool writable) {
	return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, image.range, image.format, image.width, image.height,
	                                              writable);
}

int32_t Evaluate(VkCommandBuffer command, const KytyDlssEvaluate* evaluate) {
	const std::lock_guard lock(state.mutex);
	if (state.feature == nullptr || state.parameters == nullptr || evaluate == nullptr) return 0;
	++state.evaluations;
	Collect(false);
	auto                          color  = Wrap(evaluate->color, false);
	auto                          depth  = Wrap(evaluate->depth, false);
	auto                          motion = Wrap(evaluate->motion, false);
	auto                          output = Wrap(evaluate->output, true);
	NVSDK_NGX_VK_DLSS_Eval_Params parameters {};
	parameters.Feature.pInColor       = &color;
	parameters.Feature.pInOutput      = &output;
	parameters.Feature.InSharpness    = evaluate->sharpness;
	parameters.pInDepth               = &depth;
	parameters.pInMotionVectors       = &motion;
	parameters.InRenderSubrectDimensions = {state.feature_desc.input_width, state.feature_desc.input_height};
	parameters.InJitterOffsetX        = evaluate->jitter_x;
	parameters.InJitterOffsetY        = evaluate->jitter_y;
	parameters.InReset                = evaluate->reset;
	parameters.InMVScaleX             = evaluate->mv_scale_x;
	parameters.InMVScaleY             = evaluate->mv_scale_y;
	parameters.InPreExposure          = 1.0f;
	parameters.InExposureScale        = 1.0f;
	parameters.InFrameTimeDeltaInMsec = evaluate->frame_ms;
	return Check("DLSS evaluation", NGX_VULKAN_EVALUATE_DLSS_EXT(command, state.feature, state.parameters, &parameters)) ? 1
	                                                                                                                     : 0;
}

int32_t FrameGenMaxCount() {
	return state.initialized ? state.fg_max_count : 0;
}

int32_t FrameGenCreate(VkCommandBuffer command, const KytyDlssgFeature* desc) {
	const std::lock_guard lock(state.mutex);
	if (!state.initialized || state.fg_max_count < 1 || desc == nullptr) return 0;
	Retire(state.fg_feature, state.fg_parameters);
	if (!Check("Frame generation parameter allocation", NVSDK_NGX_VULKAN_AllocateParameters(&state.fg_parameters))) return 0;
	// Depth and motion vectors are smaller than the backbuffer (the game's own size).
	NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Width, desc->width);
	NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Height, desc->height);
	NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalWidth, desc->render_width);
	NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalHeight, desc->render_height);
	NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_DynamicResolution, 0);
	NVSDK_NGX_DLSSG_Create_Params settings {};
	settings.Width                  = desc->width;
	settings.Height                 = desc->height;
	settings.NativeBackbufferFormat = static_cast<unsigned>(desc->format);
	settings.RenderWidth            = desc->render_width;
	settings.RenderHeight           = desc->render_height;
	const auto result               = NGX_VK_CREATE_DLSSG(command, 1, 1, &state.fg_feature, state.fg_parameters, &settings);
	if (!Check("Frame generation feature creation", result) || state.fg_feature == nullptr) {
		state.fg_feature = nullptr;
		return 0;
	}
	state.fg_desc = *desc;
	Log(0, "frame generation %ux%u (depth and motion %ux%u)", desc->width, desc->height, desc->render_width, desc->render_height);
	return 1;
}

void Matrix(float (&out)[4][4], const float (&in)[16]) {
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c) out[r][c] = in[r * 4 + c];
}

void Identity(float (&m)[4][4]) {
	std::memset(m, 0, sizeof(m));
	for (int i = 0; i < 4; ++i) m[i][i] = 1.0f;
}

int32_t FrameGenEvaluate(VkCommandBuffer command, const KytyDlssgEvaluate* evaluate) {
	const std::lock_guard lock(state.mutex);
	if (state.fg_feature == nullptr || state.fg_parameters == nullptr || evaluate == nullptr) return 0;
	if (evaluate->multi_frame_index == 1) {
		++state.evaluations;
		Collect(false);
	}
	auto                           backbuffer = Wrap(evaluate->backbuffer, false);
	auto                           depth      = Wrap(evaluate->depth, false);
	auto                           motion     = Wrap(evaluate->motion, false);
	auto                           output     = Wrap(evaluate->output, true);
	NVSDK_NGX_VK_DLSSG_Eval_Params resources {};
	resources.pBackbuffer        = &backbuffer;
	resources.pDepth             = &depth;
	resources.pMVecs             = &motion;
	resources.pOutputInterpFrame = &output;
	NVSDK_NGX_Resource_VK hudless {};
	if (evaluate->hudless.image != VK_NULL_HANDLE) {
		hudless            = Wrap(evaluate->hudless, false);
		resources.pHudless = &hudless;
	}
	NVSDK_NGX_DLSSG_Opt_Eval_Params options {};
	options.multiFrameCount = evaluate->multi_frame_count;
	options.multiFrameIndex = evaluate->multi_frame_index;
	Matrix(options.cameraViewToClip, evaluate->camera_view_to_clip);
	Matrix(options.clipToCameraView, evaluate->clip_to_camera_view);
	Identity(options.clipToLensClip);
	Matrix(options.clipToPrevClip, evaluate->clip_to_prev_clip);
	Matrix(options.prevClipToClip, evaluate->prev_clip_to_clip);
	options.jitterOffset[0] = evaluate->jitter_x;
	options.jitterOffset[1] = evaluate->jitter_y;
	options.mvecScale[0]    = evaluate->mv_scale_x;
	options.mvecScale[1]    = evaluate->mv_scale_y;
	for (int i = 0; i < 3; ++i) {
		options.cameraPos[i]   = evaluate->camera_pos[i];
		options.cameraUp[i]    = evaluate->camera_up[i];
		options.cameraRight[i] = evaluate->camera_right[i];
		options.cameraFwd[i]   = evaluate->camera_fwd[i];
	}
	options.cameraNear                = evaluate->camera_near;
	options.cameraFar                 = evaluate->camera_far;
	options.cameraFOV                 = evaluate->camera_fov;
	options.cameraAspectRatio         = evaluate->camera_aspect;
	options.depthInverted             = evaluate->depth_inverted != 0;
	options.cameraMotionIncluded      = true;
	options.motionVectorsInvalidValue = std::numeric_limits<float>::max();
	options.reset                     = evaluate->reset != 0 && evaluate->multi_frame_index == 1;
	if (evaluate->backbuffer_width != 0) {
		options.backbufferSubrectBase   = {evaluate->backbuffer_x, evaluate->backbuffer_y};
		options.backbufferSubrectSize   = {evaluate->backbuffer_width, evaluate->backbuffer_height};
		options.outputInterpSubrectBase = options.backbufferSubrectBase;
		options.outputInterpSubrectSize = options.backbufferSubrectSize;
		if (resources.pHudless != nullptr) {
			options.hudLessSubrectBase = options.backbufferSubrectBase;
			options.hudLessSubrectSize = options.backbufferSubrectSize;
		}
	}
	return Check("Frame generation evaluation",
	             NGX_VK_EVALUATE_DLSSG(command, state.fg_feature, state.fg_parameters, &resources, &options))
	           ? 1
	           : 0;
}

void Shutdown() {
	const std::lock_guard lock(state.mutex);
	Retire(state.fg_feature, state.fg_parameters);
	Retire(state.feature, state.parameters);
	Collect(true);
	if (state.capabilities != nullptr) {
		NVSDK_NGX_VULKAN_DestroyParameters(state.capabilities);
		state.capabilities = nullptr;
	}
	if (state.initialized) {
		NVSDK_NGX_VULKAN_Shutdown1(state.device);
		state.initialized = false;
		state.device      = VK_NULL_HANDLE;
	}
}

const KytyDlssApi api {KYTY_DLSS_BRIDGE_ABI, Configure, InstanceExtensions, DeviceExtensions, Initialize,
                       CreateFeature,        Evaluate,  Shutdown,           FrameGenMaxCount, FrameGenCreate,
                       FrameGenEvaluate};

} // namespace

extern "C" __attribute__((visibility("default"))) const KytyDlssApi* KytyDlssGetApi() {
	return &api;
}
