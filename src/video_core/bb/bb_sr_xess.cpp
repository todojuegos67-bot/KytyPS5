// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Intel XeSS super resolution (BB_UPSCALER=xess) through the XeSS SDK's libxess.dll (next to the
// emulator) on its Vulkan backend. Ported from the Demon's Souls fork's src/local/xess-sr.cpp.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32) && defined(BB_XESS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <xess/xess.h>
#include <xess/xess_vk.h>
#endif

namespace BB::SR::Xess {

#if defined(_WIN32) && defined(BB_XESS)
namespace {

struct Api {
    decltype(&xessVKGetRequiredDeviceExtensions) get_device_extensions = nullptr;
    decltype(&xessVKCreateContext) create_context = nullptr;
    decltype(&xessVKInit) init = nullptr;
    decltype(&xessVKExecute) execute = nullptr;
    decltype(&xessSetVelocityScale) set_velocity_scale = nullptr;
    decltype(&xessDestroyContext) destroy_context = nullptr;
    decltype(&xessGetVersion) get_version = nullptr;
};
Api g_api;
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
xess_context_handle_t g_context = nullptr;
vk::Extent2D g_output{};
vk::Extent2D g_render{};
uint32_t g_flags = 0;
bool g_failed = false;
uint32_t g_errors = 0;
VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT g_mutable{};

xess_quality_settings_t QualityFor(Quality quality, double ratio) {
    switch (quality) {
    case Quality::Native:
        return XESS_QUALITY_SETTING_AA;
    case Quality::UltraQualityPlus:
        return XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS;
    case Quality::UltraQuality:
        return XESS_QUALITY_SETTING_ULTRA_QUALITY;
    case Quality::Quality:
        return XESS_QUALITY_SETTING_QUALITY;
    case Quality::Balanced:
        return XESS_QUALITY_SETTING_BALANCED;
    case Quality::Performance:
        return XESS_QUALITY_SETTING_PERFORMANCE;
    case Quality::UltraPerformance:
        return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
    default:
        break;
    }
    // The Demon's Souls choice by the ratio (XeSS 1.3+ scales).
    if (ratio < 1.15) return XESS_QUALITY_SETTING_AA;
    if (ratio < 1.4) return XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS;
    if (ratio < 1.6) return XESS_QUALITY_SETTING_ULTRA_QUALITY;
    if (ratio < 1.85) return XESS_QUALITY_SETTING_QUALITY;
    if (ratio < 2.15) return XESS_QUALITY_SETTING_BALANCED;
    if (ratio < 2.65) return XESS_QUALITY_SETTING_PERFORMANCE;
    return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
}

xess_vk_image_view_info ViewInfo(const Image& image, bool depth) {
    xess_vk_image_view_info info{};
    info.imageView = static_cast<VkImageView>(image.view);
    info.image = static_cast<VkImage>(image.image);
    info.subresourceRange = {depth ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT)
                                   : static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
                             0, 1, 0, 1};
    info.format = static_cast<VkFormat>(image.format);
    info.width = image.extent.width;
    info.height = image.extent.height;
    return info;
}

} // namespace

bool Load() {
    static const bool loaded = [] {
        wchar_t exe[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe);
        dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
        HMODULE dll = LoadLibraryW((dir + L"libxess.dll").c_str());
        if (dll == nullptr) {
            BBLogf("XeSS: libxess.dll not found next to the emulator: upscaling off");
            return false;
        }
        const auto load = [&](auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                reinterpret_cast<void*>(GetProcAddress(dll, name)));
            return function != nullptr;
        };
        if (!load(g_api.get_device_extensions, "xessVKGetRequiredDeviceExtensions") ||
            !load(g_api.create_context, "xessVKCreateContext") ||
            !load(g_api.init, "xessVKInit") || !load(g_api.execute, "xessVKExecute") ||
            !load(g_api.set_velocity_scale, "xessSetVelocityScale") ||
            !load(g_api.destroy_context, "xessDestroyContext") ||
            !load(g_api.get_version, "xessGetVersion")) {
            BBLogf("XeSS: libxess.dll lacks a Vulkan entry point: upscaling off");
            return false;
        }
        xess_version_t version{};
        g_api.get_version(&version);
        BBLogf("XeSS: libxess.dll %u.%u.%u loaded", version.major, version.minor, version.patch);
        return true;
    }();
    return loaded;
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    if (!Load()) {
        return;
    }
    uint32_t count = 0;
    const char* const* names = nullptr;
    if (g_api.get_device_extensions(static_cast<VkInstance>(instance),
                                    static_cast<VkPhysicalDevice>(physical), &count,
                                    &names) != XESS_RESULT_SUCCESS) {
        BBLogf("XeSS: this GPU or driver is not supported: upscaling off");
        g_failed = true;
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const bool has = std::find(available.begin(), available.end(), names[i]) != available.end();
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, names[i]) == 0; });
        if (!has) {
            BBLogf("XeSS: the device lacks %s: upscaling off", names[i]);
            g_failed = true;
            return;
        }
        if (!added) {
            extensions.push_back(names[i]);
        }
    }
}

void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next) {
    if (g_failed || !Load()) {
        return next;
    }
    const bool mutable_ext = std::any_of(extensions.begin(), extensions.end(), [](const char* e) {
        return std::strcmp(e, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) == 0;
    });
    if (!mutable_ext) {
        return next;
    }
    g_mutable = {};
    g_mutable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
    g_mutable.mutableDescriptorType = VK_TRUE;
    g_mutable.pNext = next;
    return &g_mutable;
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (g_failed || !Load()) {
        return;
    }
    g_instance = static_cast<VkInstance>(instance);
    g_physical = static_cast<VkPhysicalDevice>(physical);
    g_device = static_cast<VkDevice>(device);
}

bool Ready() {
    return !g_failed && Load() && g_device != VK_NULL_HANDLE;
}

bool Evaluate(const Desc& desc) {
    if (!Ready() || !desc.cmd || !desc.color.view || !desc.depth.view || !desc.motion.view ||
        !desc.output.view) {
        return false;
    }
    // Auto exposure only for HDR (linear) input: on tonemapped LDR input it shifts brightness.
    uint32_t flags = desc.hdr ? XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE : 0u;
    if (desc.depth_inverted) flags |= XESS_INIT_FLAG_INVERTED_DEPTH;
    if (!desc.hdr) flags |= XESS_INIT_FLAG_LDR_INPUT_COLOR;
    if (g_context == nullptr || g_output != desc.upscale || g_render != desc.render ||
        g_flags != flags) {
        if (g_context != nullptr) {
            (void)vk::Device(g_device).waitIdle();
            g_api.destroy_context(g_context);
            g_context = nullptr;
        }
        if (g_api.create_context(g_instance, g_physical, g_device, &g_context) !=
            XESS_RESULT_SUCCESS) {
            BBLogf("XeSS: context not created: upscaling off");
            g_failed = true;
            return false;
        }
        xess_vk_init_params_t init{};
        init.outputResolution = {desc.upscale.width, desc.upscale.height};
        const double ratio = static_cast<double>(desc.upscale.height) / desc.render.height;
        init.qualitySetting = QualityFor(desc.quality, ratio);
        init.initFlags = flags;
        if (const auto result = g_api.init(g_context, &init); result != XESS_RESULT_SUCCESS) {
            BBLogf("XeSS: init failed (%d): upscaling off", static_cast<int>(result));
            g_api.destroy_context(g_context);
            g_context = nullptr;
            g_failed = true;
            return false;
        }
        // Our vectors are in render pixels towards the previous frame (BB_XESS_MV_SIGN=-1 flips).
        const char* sign_env = std::getenv("BB_XESS_MV_SIGN");
        const float sign = sign_env != nullptr && std::atof(sign_env) < 0 ? -1.0f : 1.0f;
        g_api.set_velocity_scale(g_context, sign, sign);
        g_output = desc.upscale;
        g_render = desc.render;
        g_flags = flags;
        BBLogf("XeSS: %ux%u -> %ux%u, quality setting %d, %s input", desc.render.width,
               desc.render.height, desc.upscale.width, desc.upscale.height,
               static_cast<int>(init.qualitySetting), desc.hdr ? "HDR" : "LDR");
    }
    xess_vk_execute_params_t exec{};
    exec.colorTexture = ViewInfo(desc.color, false);
    exec.velocityTexture = ViewInfo(desc.motion, false);
    exec.depthTexture = ViewInfo(desc.depth, true);
    exec.outputTexture = ViewInfo(desc.output, false);
    exec.jitterOffsetX = desc.jitter[0];
    exec.jitterOffsetY = desc.jitter[1];
    exec.exposureScale = 1.0f;
    exec.resetHistory = desc.reset ? 1u : 0u;
    exec.inputWidth = desc.render.width;
    exec.inputHeight = desc.render.height;
    if (const auto result =
            g_api.execute(g_context, static_cast<VkCommandBuffer>(desc.cmd), &exec);
        result != XESS_RESULT_SUCCESS) {
        if (g_errors++ < 4) {
            BBLogf("XeSS: execute failed (%d)", static_cast<int>(result));
        }
        return false;
    }
    return true;
}

#else

bool Load() {
    return false;
}
void AddDeviceExtensions(vk::Instance, vk::PhysicalDevice, const std::vector<std::string>&,
                         std::vector<const char*>&) {}
void* ChainDeviceFeatures(const std::vector<const char*>&, void* next) {
    return next;
}
void OnDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {}
bool Ready() {
    return false;
}
bool Evaluate(const Desc&) {
    return false;
}

#endif

} // namespace BB::SR::Xess
