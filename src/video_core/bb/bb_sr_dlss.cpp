// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// DLSS super resolution / DLAA (BB_UPSCALER=dlss) through NVIDIA Streamline (MIT,
// LICENSES/Streamline-MIT.txt). Ported from the Demon's Souls fork's src/local/frame-gen.cpp
// (Streamline load, slInit, the DLSS options, constants, tags and evaluate). Streamline's
// sl.interposer.dll becomes the Vulkan loader (vk_platform.cpp), so instance and device creation go
// through it and it adds what DLSS needs. The DLSS-G, Reflex and PCL plugins load too (DLSS-G stays
// off): the Demon's Souls fork found the device creation failing with the DLSS plugin alone, and
// phase 2 (frame generation) needs them.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32) && defined(BB_STREAMLINE)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#endif

namespace BB::SR::Dlss {

#if defined(_WIN32) && defined(BB_STREAMLINE)
namespace {

struct Api {
    PFun_slInit* init = nullptr;
    PFun_slIsFeatureSupported* is_feature_supported = nullptr;
    PFun_slIsFeatureLoaded* is_feature_loaded = nullptr;
    PFun_slGetFeatureFunction* get_feature_function = nullptr;
    PFun_slGetNewFrameToken* get_new_frame_token = nullptr;
    PFun_slSetConstants* set_constants = nullptr;
    PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
    PFun_slEvaluateFeature* evaluate = nullptr;
    PFun_slDLSSSetOptions* dlss_set_options = nullptr;
    PFun_slDLSSGetOptimalSettings* dlss_get_optimal_settings = nullptr;
};
Api g_api;
bool g_loaded = false;
bool g_ready = false;
std::wstring g_interposer;
uint32_t g_refusals = 0;

void LogMessage(sl::LogType type, const char* message) {
    if (message == nullptr) {
        return;
    }
    if (type == sl::LogType::eError && std::strstr(message, "[sl.dlss]") != nullptr &&
        std::strstr(message, "evaluate feature failed") != nullptr) {
        ++g_refusals;
    }
    static const bool verbose = std::getenv("BB_DLSS_VERBOSE") != nullptr;
    if (type == sl::LogType::eError || verbose) {
        BBLogf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
    }
}

template <typename F>
bool FeatureFunction(sl::Feature feature, const char* name, F*& function) {
    void* pointer = nullptr;
    if (g_api.get_feature_function(feature, name, pointer) != sl::Result::eOk ||
        pointer == nullptr) {
        BBLogf("DLSS: %s is unavailable", name);
        return false;
    }
    function = reinterpret_cast<F*>(pointer);
    return true;
}

sl::DLSSMode ModeFor(Quality quality, double share) {
    switch (quality) {
    case Quality::Native:
        return sl::DLSSMode::eDLAA;
    case Quality::UltraQualityPlus:
    case Quality::UltraQuality:
    case Quality::Quality:
        return sl::DLSSMode::eMaxQuality;
    case Quality::Balanced:
        return sl::DLSSMode::eBalanced;
    case Quality::Performance:
        return sl::DLSSMode::eMaxPerformance;
    case Quality::UltraPerformance:
        return sl::DLSSMode::eUltraPerformance;
    default:
        break;
    }
    // The Demon's Souls choice by the render share of the output height.
    return share >= 0.95    ? sl::DLSSMode::eDLAA
           : share >= 0.62  ? sl::DLSSMode::eMaxQuality
           : share >= 0.55  ? sl::DLSSMode::eBalanced
           : share >= 0.495 ? sl::DLSSMode::eMaxPerformance
                            : sl::DLSSMode::eUltraPerformance;
}

sl::Resource ToResource(const Image& image) {
    sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
                   static_cast<VkImageView>(image.view),
                   static_cast<uint32_t>(static_cast<VkImageLayout>(image.layout)));
    r.width = image.extent.width;
    r.height = image.extent.height;
    r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
    return r;
}

sl::float4x4 Matrix(const float* m) {
    sl::float4x4 r;
    for (uint32_t i = 0; i < 4; ++i) {
        r[i] = sl::float4(m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
    }
    return r;
}

bool Inverted(const float* m, float* out) {
    double a[4][8]{};
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t pivot = c;
        for (uint32_t r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        }
        if (std::fabs(a[pivot][c]) < 1e-12) return false;
        for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double scale = 1.0 / a[c][c];
        for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
        for (uint32_t r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][4 + j]);
    }
    return true;
}

} // namespace

bool Preload() {
    static const bool loaded = [] {
        // BB_STREAMLINE_DIR, else a "streamline" folder next to the emulator, else the emulator's
        // own folder (the CI package puts the DLLs there).
        wchar_t exe[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        const auto exe_dir = std::filesystem::path(exe).parent_path();
        std::vector<std::filesystem::path> candidates;
        if (const wchar_t* env = _wgetenv(L"BB_STREAMLINE_DIR"); env != nullptr && *env != L'\0') {
            candidates.emplace_back(env);
        }
        candidates.push_back(exe_dir / "streamline");
        candidates.push_back(exe_dir);
        std::filesystem::path dir;
        for (const auto& candidate : candidates) {
            for (const auto& folder : {candidate, candidate / "bin" / "x64"}) {
                std::error_code error;
                if (dir.empty() && std::filesystem::exists(folder / "sl.interposer.dll", error)) {
                    dir = folder;
                }
            }
        }
        if (dir.empty()) {
            BBLogf("DLSS: no sl.interposer.dll (BB_STREAMLINE_DIR, a streamline folder or next to "
                   "the emulator): DLSS off");
            return false;
        }
        const auto interposer = dir / "sl.interposer.dll";
        BBLogf("DLSS: LoadLibrary %s (vulkan-1.dll already loaded: %d)", interposer.string().c_str(),
               GetModuleHandleA("vulkan-1.dll") != nullptr ? 1 : 0);
        HMODULE module = LoadLibraryW(interposer.c_str());
        if (module == nullptr) {
            BBLogf("DLSS: cannot load %s: DLSS off", interposer.string().c_str());
            return false;
        }
        const auto load = [&](auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                reinterpret_cast<void*>(GetProcAddress(module, name)));
            return function != nullptr;
        };
        if (!load(g_api.init, "slInit") ||
            !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
            !load(g_api.is_feature_loaded, "slIsFeatureLoaded") ||
            !load(g_api.get_feature_function, "slGetFeatureFunction") ||
            !load(g_api.get_new_frame_token, "slGetNewFrameToken") ||
            !load(g_api.set_constants, "slSetConstants") ||
            !load(g_api.set_tag_for_frame, "slSetTagForFrame") ||
            !load(g_api.evaluate, "slEvaluateFeature")) {
            BBLogf("DLSS: sl.interposer.dll lacks an entry point: DLSS off");
            return false;
        }
        static const std::wstring plugin_dir = dir.wstring();
        static const wchar_t* plugin_paths[]{plugin_dir.c_str()};
        static std::vector<sl::Feature> features{sl::kFeatureDLSS, sl::kFeatureDLSS_G,
                                                 sl::kFeatureReflex, sl::kFeaturePCL};
        sl::Preferences preferences{};
        preferences.showConsole = false;
        preferences.logLevel = std::getenv("BB_DLSS_VERBOSE") ? sl::LogLevel::eVerbose
                                                              : sl::LogLevel::eDefault;
        preferences.pathsToPlugins = plugin_paths;
        preferences.numPathsToPlugins = 1;
        preferences.pathToLogsAndData = nullptr;
        preferences.logMessageCallback = LogMessage;
        preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                            sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        preferences.featuresToLoad = features.data();
        preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
        preferences.engine = sl::EngineType::eCustom;
        preferences.engineVersion = "1.0";
        preferences.projectId = "6a3f1d5e-4b7c-4f2a-9d8e-1c2b3a4d5e6f";
        preferences.renderAPI = sl::RenderAPI::eVulkan;
        {
            std::error_code ec;
            BBLogf("DLSS: plugins present: sl.common.dll %d, sl.dlss.dll %d, nvngx_dlss.dll %d, "
                   "sl.dlss_g.dll %d, sl.reflex.dll %d, sl.pcl.dll %d",
                   std::filesystem::exists(dir / "sl.common.dll", ec) ? 1 : 0,
                   std::filesystem::exists(dir / "sl.dlss.dll", ec) ? 1 : 0,
                   std::filesystem::exists(dir / "nvngx_dlss.dll", ec) ? 1 : 0,
                   std::filesystem::exists(dir / "sl.dlss_g.dll", ec) ? 1 : 0,
                   std::filesystem::exists(dir / "sl.reflex.dll", ec) ? 1 : 0,
                   std::filesystem::exists(dir / "sl.pcl.dll", ec) ? 1 : 0);
        }
        BBLogf("DLSS: slInit (features DLSS, DLSS-G, Reflex, PCL; plugins in %s)",
               dir.string().c_str());
        const auto init_result = g_api.init(preferences, sl::kSDKVersion);
        BBLogf("DLSS: slInit returned %d", static_cast<int>(init_result));
        if (const auto result = init_result; result != sl::Result::eOk) {
            BBLogf("DLSS: slInit failed (%d): DLSS off", static_cast<int>(result));
            return false;
        }
        g_interposer = interposer.wstring();
        BBLogf("DLSS: Streamline loaded from %s (the Vulkan loader from here on)",
               dir.string().c_str());
        return true;
    }();
    g_loaded = loaded;
    return loaded;
}

const wchar_t* LoaderPath() {
    return g_loaded && !g_interposer.empty() ? g_interposer.c_str() : nullptr;
}

void OnDevice(vk::Instance, vk::PhysicalDevice physical, vk::Device) {
    if (!g_loaded) {
        return;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical);
    bool dlss_loaded = false;
    if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) == sl::Result::eOk &&
        g_api.is_feature_loaded(sl::kFeatureDLSS, dlss_loaded) == sl::Result::eOk && dlss_loaded &&
        FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options) &&
        FeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings",
                        g_api.dlss_get_optimal_settings)) {
        g_ready = true;
        BBLogf("DLSS: super resolution ready on this device");
    } else {
        BBLogf("DLSS: not supported on this device: DLSS off");
    }
}

bool Ready() {
    return g_ready;
}

bool Evaluate(const Desc& desc) {
    if (!g_ready || g_refusals >= 3) {
        static bool said = false;
        if (g_ready && !said) {
            said = true;
            BBLogf("DLSS: NGX refused it %u times (Streamline errors above): DLSS off", g_refusals);
        }
        return false;
    }
    const sl::ViewportHandle viewport{1};
    static vk::Extent2D options_render{}, options_output{};
    static Quality options_quality = Quality::Dynamic;
    static bool options_hdr = true;
    if (options_render != desc.render || options_output != desc.upscale ||
        options_quality != desc.quality || options_hdr != desc.hdr) {
        sl::DLSSOptions options{};
        options.mode = ModeFor(desc.quality, static_cast<double>(desc.render.height) /
                                                 static_cast<double>(desc.upscale.height));
        options.outputWidth = desc.upscale.width;
        options.outputHeight = desc.upscale.height;
        options.colorBuffersHDR = desc.hdr ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        options.useAutoExposure = sl::Boolean::eTrue;
        if (const char* preset = std::getenv("BB_DLSS_PRESET");
            preset != nullptr && *preset >= 'A' && *preset <= 'O') {
            const auto value = static_cast<sl::DLSSPreset>(*preset - 'A' + 1);
            options.dlaaPreset = options.qualityPreset = options.balancedPreset =
                options.performancePreset = options.ultraPerformancePreset =
                    options.ultraQualityPreset = value;
        }
        sl::DLSSOptimalSettings settings{};
        const bool queried =
            g_api.dlss_get_optimal_settings(options, settings) == sl::Result::eOk;
        if (const auto result = g_api.dlss_set_options(viewport, options);
            result != sl::Result::eOk) {
            BBLogf("DLSS: slDLSSSetOptions failed (%d)", static_cast<int>(result));
            return false;
        }
        BBLogf("DLSS: %ux%u -> %ux%u, mode %u, %s input (its range %ux%u-%ux%u%s)",
               desc.render.width, desc.render.height, desc.upscale.width, desc.upscale.height,
               static_cast<uint32_t>(options.mode), desc.hdr ? "HDR" : "LDR",
               settings.renderWidthMin, settings.renderHeightMin, settings.renderWidthMax,
               settings.renderHeightMax, queried ? "" : ", not queried");
        options_render = desc.render;
        options_output = desc.upscale;
        options_quality = desc.quality;
        options_hdr = desc.hdr;
    }
    static uint32_t frame = 0;
    ++frame;
    sl::FrameToken* token = nullptr;
    if (g_api.get_new_frame_token(token, &frame) != sl::Result::eOk || token == nullptr) {
        return false;
    }
    sl::Constants constants{};
    float identity[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float inverse[16]{};
    // Streamline's matrices are row-vector (p * M); ours are row major for column vectors, so
    // each is given transposed.
    const auto transposed = [](const float* m) {
        float t[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) t[j * 4 + i] = m[i * 4 + j];
        return Matrix(t);
    };
    constants.cameraViewToClip = transposed(desc.camera_valid ? desc.view_to_clip : identity);
    constants.clipToCameraView =
        desc.camera_valid && Inverted(desc.view_to_clip, inverse) ? transposed(inverse)
                                                                  : Matrix(identity);
    constants.clipToLensClip = Matrix(identity);
    constants.clipToPrevClip = transposed(desc.camera_valid ? desc.clip_to_prev_clip : identity);
    float prev_to_clip[16]{};
    constants.prevClipToClip = desc.camera_valid && Inverted(desc.clip_to_prev_clip, prev_to_clip)
                                   ? transposed(prev_to_clip)
                                   : Matrix(identity);
    constants.jitterOffset = sl::float2(desc.jitter[0], desc.jitter[1]);
    // Our vectors are render pixels; Streamline wants render-size UV.
    constants.mvecScale = sl::float2(1.0f / static_cast<float>(desc.render.width),
                                     1.0f / static_cast<float>(desc.render.height));
    constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    constants.cameraPos = sl::float3(0, 0, 0);
    constants.cameraUp = sl::float3(0, 1, 0);
    constants.cameraRight = sl::float3(1, 0, 0);
    constants.cameraFwd = sl::float3(0, 0, 1);
    constants.cameraNear = desc.z_near;
    constants.cameraFar = desc.z_far;
    constants.cameraFOV = desc.fov_y;
    constants.cameraAspectRatio =
        static_cast<float>(desc.render.width) / static_cast<float>(desc.render.height);
    constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
    constants.depthInverted = desc.depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = desc.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.orthographicProjection = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    if (const auto result = g_api.set_constants(constants, *token, viewport);
        result != sl::Result::eOk) {
        static uint32_t reports = 0;
        if (reports++ < 8) BBLogf("DLSS: slSetConstants failed (%d)", static_cast<int>(result));
        return false;
    }
    auto color = ToResource(desc.color);
    auto depth = ToResource(desc.depth);
    auto motion = ToResource(desc.motion);
    auto output = ToResource(desc.output);
    const sl::Extent render{0, 0, desc.render.width, desc.render.height};
    const sl::Extent full{0, 0, desc.upscale.width, desc.upscale.height};
    sl::ResourceTag tags[4]{
        sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &render),
        sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &full),
        sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate,
                        &render),
        sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &render),
    };
    auto* command_buffer =
        reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(desc.cmd));
    if (const auto result = g_api.set_tag_for_frame(*token, viewport, tags, 4, command_buffer);
        result != sl::Result::eOk) {
        static uint32_t reports = 0;
        if (reports++ < 8) BBLogf("DLSS: slSetTagForFrame failed (%d)", static_cast<int>(result));
        return false;
    }
    const sl::BaseStructure* inputs[]{&viewport};
    const auto result = g_api.evaluate(sl::kFeatureDLSS, *token, inputs, 1, command_buffer);
    if (result != sl::Result::eOk && result != sl::Result::eWarnOutOfVRAM) {
        static uint32_t reports = 0;
        if (reports++ < 8) BBLogf("DLSS: slEvaluateFeature failed (%d)", static_cast<int>(result));
        return false;
    }
    return true;
}

#else

bool Preload() {
    return false;
}
const wchar_t* LoaderPath() {
    return nullptr;
}
void OnDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {}
bool Ready() {
    return false;
}
bool Evaluate(const Desc&) {
    return false;
}

#endif

} // namespace BB::SR::Dlss
