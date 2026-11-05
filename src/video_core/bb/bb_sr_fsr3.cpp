// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// AMD FSR 3.1 upscaling (BB_UPSCALER=fsr3) through the FidelityFX API's signed DLL
// (amd_fidelityfx_vk.dll next to the emulator). Ported from the Demon's Souls fork's
// src/local/fsr3-fg.cpp ("FSR 3.1 upscaling" section). Headers: externals/ffx_api (MIT).
#include <algorithm>
#include <cfloat>
#include <cmath>
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

std::string FlagsText(uint32_t flags) {
    std::string text;
    const auto add = [&](uint32_t bit, const char* name) {
        if (flags & bit) {
            text += text.empty() ? name : std::string("|") + name;
        }
    };
    add(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, "HIGH_DYNAMIC_RANGE");
    add(FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS, "DISPLAY_RESOLUTION_MOTION_VECTORS");
    add(FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION, "MOTION_VECTORS_JITTER_CANCELLATION");
    add(FFX_UPSCALE_ENABLE_DEPTH_INVERTED, "DEPTH_INVERTED");
    add(FFX_UPSCALE_ENABLE_DEPTH_INFINITE, "DEPTH_INFINITE");
    add(FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, "AUTO_EXPOSURE");
    add(FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE, "NON_LINEAR_COLORSPACE");
    if (text.empty()) {
        text = "none";
    }
    text += " (vectors at render size, jitter removed: no DISPLAY_RESOLUTION_MOTION_VECTORS / "
            "JITTER_CANCELLATION)";
    return text;
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
    // Auto exposure only for HDR (linear) input: on tonemapped LDR input it shifts brightness.
    uint32_t flags = desc.hdr ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0u;
    if (desc.hdr) flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    if (desc.depth_inverted) flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    // 0.86: the AMD manual (Depth buffer configurations): the infinite flag says the depth buffer
    // uses an infinite far plane. Bloodborne's does not: its camera block has far = 3000
    // ("[dlssg] constants ... far 3000", z = 1.00002 in the projection), so with a camera found
    // the real far plane is given and DEPTH_INFINITE is left off (DLSS is told the same far).
    // Without a camera (zero vectors anyway) or with BB_FSR3_DEPTH_INFINITE=1 it stays infinite,
    // as before 0.86. Once a camera was found the flag stays off (one context re-creation at most).
    static const bool infinite_forced = [] {
        const char* v = std::getenv("BB_FSR3_DEPTH_INFINITE");
        return v != nullptr && v[0] == '1';
    }();
    static bool finite_seen = false;
    const bool far_ok = desc.camera_valid && std::isfinite(desc.z_far) &&
                        desc.z_far > 2.0f * std::max(desc.z_near, 0.001f) && desc.z_far < 1.0e7f;
    finite_seen = finite_seen || far_ok; // (sticky: a camera lost for a frame does not recreate)
    const bool finite_far = !infinite_forced && finite_seen;
    if (!finite_far) {
        flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    }
    // The AMD manual: FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE when the colour holds perceptual
    // (gamma-corrected) values, i.e. the sRGB-encoded LDR input of BB_SR_LDR_ENCODED=1.
    if (desc.encoded) flags |= FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
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
    // Our vectors: render pixels towards the previous frame. AMD's FSR 3 upscaler manual
    // (Providing motion vectors): the application provides them in the screen-space range
    // [<-width, -height> .. <width, height>] and they "encode the motion from a pixel in the
    // current frame to the position of that same pixel in the previous frame"; motionVectorScale
    // only adapts other spaces (NDC), so +1,+1. 0.86: a BB_FSR3_MV_SIGN=-1 of an old launcher is
    // ignored (SR::MvSign), BB_MV_SIGN_FORCE=1 applies it.
    static const float mv_sign = MvSign(Kind::Fsr3);
    {
        SdkInputs inputs;
        inputs.mv_scale[0] = inputs.mv_scale[1] = mv_sign;
        inputs.mv_sign[0] = inputs.mv_sign[1] = mv_sign;
        inputs.mv_units = "render pixels from the current frame to the previous one, jitter excluded";
        inputs.flags = flags;
        inputs.flags_text = &FlagsText;
        inputs.exposure = desc.hdr ? "auto (FFX_UPSCALE_ENABLE_AUTO_EXPOSURE), preExposure 1.0"
                                   : "none given, preExposure 1.0 (LDR input)";
        inputs.depth_note = (flags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE)
                                ? "infinite (cameraFar FLT_MAX; no camera found or BB_FSR3_DEPTH_INFINITE=1)"
                                : "finite, cameraFar = the camera's far plane";
        LogSdkInputs(Kind::Fsr3, desc, inputs);
    }
    ffxDispatchDescUpscale up{};
    up.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    up.commandList = static_cast<VkCommandBuffer>(desc.cmd);
    up.color = Resource(desc.color, false);
    up.depth = Resource(desc.depth, true);
    up.motionVectors = Resource(desc.motion, false);
    up.output = Resource(desc.output, false);
    if (desc.has_reactive) {
        up.reactive = Resource(desc.reactive, false);
        up.transparencyAndComposition = Resource(desc.reactive, false);
    }
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
    const float z_far = (flags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) ? FLT_MAX : desc.z_far;
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
