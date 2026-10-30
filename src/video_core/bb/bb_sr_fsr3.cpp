// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// AMD FSR 3.1 upscaling (BB_UPSCALER=fsr3) through the FidelityFX API's signed DLL
// (amd_fidelityfx_vk.dll next to the emulator). Ported from the Demon's Souls fork's
// src/local/fsr3-fg.cpp ("FSR 3.1 upscaling" section). Headers: externals/ffx_api (MIT).
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <string>
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>
#endif

namespace BB::SR::Fsr3 {

#if defined(_WIN32)
namespace {
HMODULE g_dll = nullptr;
ffxFunctions g_ffx{};
VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
PFN_vkGetDeviceProcAddr g_gdpa = nullptr;
PFN_vkGetDeviceProcAddr g_real_gdpa = nullptr;

// The FidelityFX backend asks for some entry points by their KHR / EXT names (null on a device
// without the extension even when the function is core): the core one is given instead, and any
// name still missing is logged, so no null function pointer is ever called unannounced.
PFN_vkVoidFunction VKAPI_CALL ProcAddr(VkDevice device, const char* name) {
    PFN_vkVoidFunction function = g_real_gdpa(device, name);
    if (function != nullptr || name == nullptr) {
        return function;
    }
    std::string core = name;
    for (const char* suffix : {"KHR", "EXT", "AMD", "NV"}) {
        const std::size_t n = std::strlen(suffix);
        if (core.size() > n && core.compare(core.size() - n, n, suffix) == 0) {
            core.resize(core.size() - n);
            function = g_real_gdpa(device, core.c_str());
            break;
        }
    }
    static uint32_t reports = 0;
    if (reports++ < 32) {
        BBLogf("FSR 3.1: device function %s %s", name,
               function != nullptr ? "given as its core alias" : "MISSING (null)");
    }
    return function;
}
ffxContext g_context = nullptr;
vk::Extent2D g_render{};
vk::Extent2D g_output{};
uint32_t g_flags = 0;
bool g_failed = false;
uint32_t g_errors = 0;

float EnvFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? static_cast<float>(std::atof(value)) : fallback;
}

void Message(uint32_t type, const wchar_t* message) {
    static uint32_t count = 0;
    if (count++ > 200) {
        return;
    }
    char text[512]{};
    std::wcstombs(text, message, sizeof(text) - 1);
    BBLogf("FSR 3.1: FidelityFX message (%u): %s", type, text);
}

FfxApiResource Resource(const Image& image, bool depth) {
    FfxApiResource resource{};
    resource.resource = static_cast<void*>(static_cast<VkImage>(image.image));
    resource.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    auto& desc = resource.description;
    desc.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    desc.format = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
    desc.width = image.extent.width;
    desc.height = image.extent.height;
    desc.depth = 1;
    desc.mipCount = 1;
    desc.flags = FFX_API_RESOURCE_FLAGS_NONE;
    desc.usage = depth ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : FFX_API_RESOURCE_USAGE_READ_ONLY;
    return resource;
}
} // namespace

bool Load() {
    static const bool loaded = [] {
        wchar_t exe[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe);
        dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
        for (const auto* name : {L"amd_fidelityfx_vk.dll", L"fsr3\\amd_fidelityfx_vk.dll"}) {
            g_dll = LoadLibraryW((dir + name).c_str());
            if (g_dll != nullptr) {
                break;
            }
        }
        if (g_dll == nullptr) {
            BBLogf("FSR 3.1: amd_fidelityfx_vk.dll not found next to the emulator: upscaling off");
            return false;
        }
        ffxLoadFunctions(&g_ffx, g_dll);
        if (g_ffx.CreateContext == nullptr || g_ffx.Configure == nullptr ||
            g_ffx.Dispatch == nullptr) {
            BBLogf("FSR 3.1: amd_fidelityfx_vk.dll has no FidelityFX API: upscaling off");
            return false;
        }
        BBLogf("FSR 3.1: FidelityFX API loaded");
        return true;
    }();
    return loaded;
}

void OnDevice(vk::PhysicalDevice physical, vk::Device device, PFN_vkGetDeviceProcAddr gdpa) {
    if (!Load()) {
        return;
    }
    g_physical = static_cast<VkPhysicalDevice>(physical);
    g_device = static_cast<VkDevice>(device);
    g_real_gdpa = gdpa;
    g_gdpa = gdpa != nullptr ? &ProcAddr : nullptr;
}

bool Ready() {
    return !g_failed && Load() && g_device != VK_NULL_HANDLE && g_gdpa != nullptr;
}

bool Evaluate(const Desc& desc) {
    if (!Ready() || !desc.cmd || !desc.color.image || !desc.depth.image || !desc.motion.image ||
        !desc.output.image) {
        return false;
    }
    uint32_t flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if (desc.hdr) flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    if (desc.depth_inverted) flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    // Infinite far always (the camera's far plane, when found, varies; a flag change recreates
    // the context and stalls the GPU).
    flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    if (g_context == nullptr || desc.render.width > g_render.width ||
        desc.render.height > g_render.height || desc.upscale != g_output || flags != g_flags) {
        static uint32_t creations = 0;
        if (++creations > 1) {
            BBLogf("FSR 3.1: context recreated (%u): render %ux%u output %ux%u flags 0x%x", creations,
                   desc.render.width, desc.render.height, desc.upscale.width, desc.upscale.height,
                   flags);
        }
        if (g_context != nullptr) {
            (void)vk::Device(g_device).waitIdle();
            g_ffx.DestroyContext(&g_context, nullptr);
            g_context = nullptr;
        }
        ffxCreateBackendVKDesc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.vkDevice = g_device;
        backend.vkPhysicalDevice = g_physical;
        backend.vkDeviceProcAddr = g_gdpa;
        ffxCreateContextDescUpscale create{};
        create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        create.header.pNext = &backend.header;
        create.flags = flags;
        create.maxRenderSize = {desc.render.width, desc.render.height};
        create.maxUpscaleSize = {desc.upscale.width, desc.upscale.height};
        create.fpMessage = &Message;
        if (const auto result = g_ffx.CreateContext(&g_context, &create.header, nullptr);
            result != FFX_API_RETURN_OK) {
            BBLogf("FSR 3.1: context not created (%u): upscaling off", result);
            g_context = nullptr;
            g_failed = true;
            return false;
        }
        g_render = desc.render;
        g_output = desc.upscale;
        g_flags = flags;
        BBLogf("FSR 3.1: %ux%u -> %ux%u, %s input, depth %s", desc.render.width,
               desc.render.height, desc.upscale.width, desc.upscale.height,
               desc.hdr ? "HDR" : "LDR", desc.depth_inverted ? "inverted" : "forward");
    }
    // Our vectors: render pixels towards the previous frame (BB_FSR3_MV_SIGN=-1 flips).
    static const float mv_sign = EnvFloat("BB_FSR3_MV_SIGN", 1.0f) >= 0 ? 1.0f : -1.0f;
    ffxDispatchDescUpscale up{};
    up.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    up.commandList = static_cast<VkCommandBuffer>(desc.cmd);
    up.color = Resource(desc.color, false);
    up.depth = Resource(desc.depth, true);
    up.motionVectors = Resource(desc.motion, false);
    up.output = Resource(desc.output, false);
    up.output.state = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
    up.output.description.usage = FFX_API_RESOURCE_USAGE_UAV;
    up.jitterOffset = {desc.jitter[0], desc.jitter[1]};
    up.motionVectorScale = {mv_sign, mv_sign};
    up.renderSize = {desc.render.width, desc.render.height};
    up.upscaleSize = {desc.upscale.width, desc.upscale.height};
    up.enableSharpening = desc.sharpness > 0.0f;
    up.sharpness = std::clamp(desc.sharpness, 0.0f, 1.0f);
    up.frameTimeDelta = std::clamp(desc.delta_ms, 1.0f, 100.0f);
    up.preExposure = 1.0f;
    up.reset = desc.reset;
    const float z_near = std::max(desc.z_near, 0.001f);
    const float z_far = FLT_MAX;
    // The SDK's convention: with inverted depth, near and far are given swapped.
    up.cameraNear = desc.depth_inverted ? z_far : z_near;
    up.cameraFar = desc.depth_inverted ? z_near : z_far;
    up.cameraFovAngleVertical = desc.fov_y > 0.0f ? desc.fov_y : 1.0f;
    up.viewSpaceToMetersFactor = 1.0f;
    up.flags = 0;
    if (const auto result = g_ffx.Dispatch(&g_context, &up.header); result != FFX_API_RETURN_OK) {
        if (g_errors++ < 4) {
            BBLogf("FSR 3.1: dispatch failed (%u)", result);
        }
        return false;
    }
    return true;
}

#else

bool Load() {
    return false;
}
void OnDevice(vk::PhysicalDevice, vk::Device, PFN_vkGetDeviceProcAddr) {}
bool Ready() {
    return false;
}
bool Evaluate(const Desc&) {
    return false;
}

#endif

} // namespace BB::SR::Fsr3
