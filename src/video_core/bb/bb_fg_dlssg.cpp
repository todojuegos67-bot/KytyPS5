// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Supermedo (github.com/Supermedo/bloodborne_pc @115a0299a)
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-bbport: DLSS Frame Generation through NGX directly, as in Supermedo's Bloodborne
// port: bbport_dlss.dll (tools/bb/dlss_bridge, MIT, from IFreemz's shadps4_dlss bridge; the only
// part that links the NVIDIA DLSS SDK) loaded at run time, with nvngx_dlssg.dll from the
// package's streamline folder. The frames are made in the presenter from two finished frames
// (UI included) and shown before the game's own (vk_presenter.cpp: Supermedo's Present,
// EnsureGenerated, UpdateFrameGenLimit). This replaces the Streamline DLSS-G path.
// BB_FG=dlss-x2 | dlss-x3 | dlss-x4 (multiFrameCount 1..3) | dlss-dyn (as many as the display's
// refresh rate leaves room for), capped by NGX's MultiFrameCountMax.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include "../../../tools/bb/dlss_bridge/bbport_dlss_bridge.h"
#endif

namespace BB::FG::Dlss {

#if defined(_WIN32)
namespace {

const BbDlssApi* g_api = nullptr;
bool g_tried = false, g_ready = false, g_failed = false;
int g_max = 0;
BbDlssgFeature g_feature{};
bool g_has_feature = false;
std::wstring g_dll_dir, g_data_dir;

void BridgeLog(int warning, const char* message) {
    BBLogf("DLSS-G (NGX)%s: %s", warning ? " warning" : "", message ? message : "");
}

std::filesystem::path ExeDir() {
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

bool Load() {
    if (g_tried) return g_api != nullptr;
    g_tried = true;
    const auto dir = ExeDir();
    const auto bridge = dir / L"bbport_dlss.dll";
    BBLogf("DLSS-G start 1/6: loading %ls", bridge.c_str());
    if (!std::filesystem::exists(bridge)) {
        BBLogf("DLSS-G: %ls is missing: off", bridge.c_str());
        return false;
    }
    // Its dependencies from the executable's folder too (not the current directory).
    HMODULE module = LoadLibraryExW(bridge.c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) module = LoadLibraryW(bridge.c_str());
    if (!module) {
        BBLogf("DLSS-G: %ls not loaded (error %lu): off", bridge.c_str(), GetLastError());
        return false;
    }
    const auto get = reinterpret_cast<BbDlssGetApiFn>(GetProcAddress(module, "BbDlssGetApi"));
    BBLogf("DLSS-G start 2/6: bridge loaded, BbDlssGetApi %s", get ? "found" : "MISSING");
    const BbDlssApi* api = get ? get() : nullptr;
    if (!api || api->abi < 2 || !api->FrameGenCreate) {
        BBLogf("DLSS-G: bbport_dlss.dll has no Frame Generation (ABI %u): off",
               api ? api->abi : 0u);
        return false;
    }
    // nvngx_dlssg.dll (and nvngx_dlss.dll, which NGX's initialisation checks) are the package's
    // own, in its streamline folder; else next to the executable.
    g_dll_dir = std::filesystem::exists(dir / L"streamline" / L"nvngx_dlssg.dll")
                    ? (dir / L"streamline").wstring()
                    : dir.wstring();
    g_data_dir = dir.wstring();
    BBLogf("DLSS-G start 3/6: configure (nvngx_dlssg.dll from %ls, found %d)", g_dll_dir.c_str(),
           std::filesystem::exists(std::filesystem::path(g_dll_dir) / L"nvngx_dlssg.dll") ? 1 : 0);
    if (!api->Configure(g_dll_dir.c_str(), g_data_dir.c_str(), BridgeLog)) {
        BBLogf("DLSS-G: bridge configuration failed: off");
        return false;
    }
    g_api = api;
    BBLogf("DLSS-G: NGX bridge loaded (ABI %u), DLLs from %ls", api->abi, g_dll_dir.c_str());
    return true;
}

} // namespace

bool Requested() {
    return Cfg().kind == Kind::Dlss;
}

void AddInstanceExtensions(const std::vector<std::string>& available,
                           std::vector<const char*>& extensions) {
    if (!Requested() || !Load()) return;
    uint32_t count = 0;
    const VkExtensionProperties* list = nullptr;
    BBLogf("DLSS-G start 4/6: instance extension query");
    if (!g_api->InstanceExtensions(&count, &list)) {
        BBLogf("DLSS-G: instance extension query failed");
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const char* name = list[i].extensionName;
        const bool have = std::find(available.begin(), available.end(), name) != available.end();
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (have && !added) {
            extensions.push_back(name);
            BBLogf("DLSS-G: instance extension %s added", name);
        }
    }
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    if (!Requested() || !Load()) return;
    uint32_t count = 0;
    const VkExtensionProperties* list = nullptr;
    BBLogf("DLSS-G start 5/6: device extension query");
    if (!g_api->DeviceExtensions(instance, physical, &count, &list)) {
        BBLogf("DLSS-G: device extension query failed (FG needs DLSS support): off");
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const char* name = list[i].extensionName;
        const bool have = std::find(available.begin(), available.end(), name) != available.end();
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (have && !added) {
            extensions.push_back(name);
            BBLogf("DLSS-G: device extension %s added", name);
        }
    }
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!Requested() || !Load()) return;
    BBLogf("DLSS-G start 6/6: NGX initialisation on the device%s",
           BB::Upscale::Cfg().kind == BB::SR::Kind::Dlss
               ? " (Streamline DLSS SR already initialised NGX on it: a second init)"
               : "");
    if (!g_api->Initialize(instance, physical, device,
                           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr)) {
        BBLogf("DLSS-G: NGX initialisation failed: off");
        return;
    }
    g_max = g_api->FrameGenMaxCount();
    g_ready = g_max >= 1;
    BBLogf("DLSS-G start done: MultiFrameCountMax %d", g_max);
    BBLogf("DLSS-G: %s (MultiFrameCountMax %d)", g_ready ? "available" : "NOT available", g_max);
}

bool Ready() {
    return Requested() && g_ready && !g_failed;
}

uint32_t MaxCount() {
    return Ready() ? uint32_t(g_max) : 0u;
}

uint32_t WantedCount() {
    if (!Ready()) return 0;
    const auto& c = Cfg();
    // dlss-dyn: up to NGX's maximum; the presenter's refresh limit picks how many.
    const uint32_t want = c.dynamic ? 3u : uint32_t(std::clamp(c.multiplier, 2, 4) - 1);
    return std::min(want, uint32_t(g_max));
}

static BbDlssImage Wrap(const Image& img, vk::ImageAspectFlags aspect) {
    BbDlssImage out{};
    out.image = img.image;
    out.view = img.view;
    out.range = {static_cast<VkImageAspectFlags>(aspect), 0, 1, 0, 1};
    out.format = static_cast<VkFormat>(img.format);
    out.width = img.extent.width;
    out.height = img.extent.height;
    return out;
}

uint32_t Generate(vk::CommandBuffer cmd, const Image& backbuffer, const Inputs& in,
                  const Image* outputs, uint32_t count, vk::Device device) {
    if (!Ready() || count == 0 || !in.valid) return 0;
    const BbDlssgFeature want{backbuffer.extent.width, backbuffer.extent.height,
                              static_cast<VkFormat>(backbuffer.format), in.render.width,
                              in.render.height};
    bool reset = in.reset;
    if (!g_has_feature || std::memcmp(&want, &g_feature, sizeof(want)) != 0) {
        // The previous feature may still be used by submitted work (Supermedo: WaitSubmitted).
        if (g_has_feature) {
            (void)device.waitIdle();
            g_api->FrameGenRelease();
            g_has_feature = false;
        }
        BBLogf("DLSS-G: feature creation %ux%u (render %ux%u)", want.width, want.height,
               want.render_width, want.render_height);
        if (!g_api->FrameGenCreate(cmd, &want)) {
            BBLogf("DLSS-G: feature creation failed: off");
            g_failed = true;
            return 0;
        }
        g_feature = want;
        g_has_feature = true;
        reset = true;
    }
    BbDlssgEvaluate e{};
    e.backbuffer = Wrap(backbuffer, vk::ImageAspectFlagBits::eColor);
    e.depth = Wrap(in.depth, vk::ImageAspectFlagBits::eDepth);
    e.motion = Wrap(in.motion, vk::ImageAspectFlagBits::eColor);
    if (in.has_hudless) e.hudless = Wrap(in.hudless, vk::ImageAspectFlagBits::eColor);
    e.jitter_x = in.jitter[0];
    e.jitter_y = in.jitter[1];
    e.camera_near = in.z_near;
    e.camera_far = in.z_far;
    e.camera_fov = in.fov_y;
    e.depth_inverted = in.depth_inverted ? 1 : 0;
    e.reset = reset ? 1 : 0;
    e.has_matrices = in.camera_valid ? 1 : 0;
    // bb-gpu6 (BB_DLSSG_MATRIX): NGX DLSS-G takes row-vector matrices (clip = p * M, as
    // Streamline: bb_sr_dlss.cpp gives DLSS SR the same matrices transposed), and its
    // cameraViewToClip must be a projection (the bridge inverts it as one). Ours are row major for
    // column vectors and view_to_clip is the whole view-projection: before gpu6 DLSS-G got them
    // as they are (transposed clipToPrevClip, a view-projection read as a projection), so its
    // camera motion contradicted the vectors on every camera turn and the generated frames
    // smeared the background. BB_DLSSG_MATRIX=raw restores that.
    // bb-dlssg-proj: cameraViewToClip = the game's own projection (transposed to row vectors)
    // by default. The one rebuilt from fov / near / far assumed the render aspect and a depth
    // mapping that is not the game's, so DLSS-G placed the scene wrong, worst toward the screen
    // edges, and blurred them on camera turns (AMD FG does not use that matrix).
    // BB_DLSSG_MATRIX=fov restores the rebuilt one, raw the one before gpu6.
    static const int matrix_mode = [] {
        const char* v = std::getenv("BB_DLSSG_MATRIX");
        const int mode = v && std::strcmp(v, "raw") == 0 ? 0 : v && std::strcmp(v, "fov") == 0 ? 1 : 2;
        BBLogf("DLSS-G: camera matrices %s (BB_DLSSG_MATRIX)",
               mode == 0 ? "raw (as before gpu6)"
               : mode == 1 ? "row-vector: clipToPrevClip transposed, cameraViewToClip = projection "
                             "from fov/near/far"
                           : "row-vector: clipToPrevClip transposed, cameraViewToClip = the game's "
                             "projection transposed");
        return mode;
    }();
    const bool raw_matrices = matrix_mode == 0;
    float to_prev_rv[16];
    if (in.camera_valid) {
        if (raw_matrices) {
            std::memcpy(e.view_to_clip, in.view_to_clip, sizeof(e.view_to_clip));
            std::memcpy(to_prev_rv, in.clip_to_prev_clip, sizeof(to_prev_rv));
        } else {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) to_prev_rv[j * 4 + i] = in.clip_to_prev_clip[i * 4 + j];
            // The bridge's Perspective (row vectors, depth 0 at near).
            const float aspect = float(in.render.width) / float(std::max(in.render.height, 1u));
            const float y = 1.0f / std::tan(std::max(in.fov_y, 0.01f) * 0.5f);
            const float n = std::max(in.z_near, 1e-4f);
            const float f = std::max(in.z_far, n * 2.0f);
            float p[16]{};
            p[0] = y / aspect;
            p[5] = y;
            p[10] = f / (f - n);
            p[11] = 1.0f;
            p[14] = -n * f / (f - n);
            std::memcpy(e.view_to_clip, p, sizeof(e.view_to_clip));
            if (matrix_mode == 2 && in.proj_valid) {
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) e.view_to_clip[j * 4 + i] = in.projection[i * 4 + j];
            }
        }
        std::memcpy(e.clip_to_prev_clip, to_prev_rv, sizeof(e.clip_to_prev_clip));
        // prev_clip_to_clip: the inverse (general 4x4).
        const float* m = to_prev_rv;
        float inv[16];
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
                 m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
                 m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
                 m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
                  m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
                 m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
                 m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
                 m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
                  m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
                 m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
                 m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
                  m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
                  m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
                 m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
                 m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
                  m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
                  m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
        const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if (det != 0.0f) {
            for (float& v : inv) v /= det;
            std::memcpy(e.prev_clip_to_clip, inv, sizeof(e.prev_clip_to_clip));
        } else {
            e.has_matrices = 0;
        }
    }
    uint32_t made = 0;
    for (uint32_t index = 1; index <= count; ++index) {
        e.output = Wrap(outputs[index - 1], vk::ImageAspectFlagBits::eColor);
        e.multi_frame_count = count;
        e.multi_frame_index = index;
        if (!g_api->FrameGenEvaluate(cmd, &e)) {
            BBLogf("DLSS-G: evaluation failed: off");
            g_failed = true;
            return 0;
        }
        ++made;
    }
    static bool said = false;
    if (!said) {
        said = true;
        BBLogf("DLSS-G: first generated frames: %u per game frame, %ux%u (depth and motion "
               "%ux%u), camera %s, HUD-less %s",
               count, want.width, want.height, want.render_width, want.render_height,
               in.camera_valid ? "yes" : "no", in.has_hudless ? "yes" : "no");
    }
    return made;
}

#else // !_WIN32

bool Requested() {
    return false;
}
void AddInstanceExtensions(const std::vector<std::string>&, std::vector<const char*>&) {}
void AddDeviceExtensions(vk::Instance, vk::PhysicalDevice, const std::vector<std::string>&,
                         std::vector<const char*>&) {}
void OnDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {}
bool Ready() {
    return false;
}
uint32_t MaxCount() {
    return 0;
}
uint32_t WantedCount() {
    return 0;
}
uint32_t Generate(vk::CommandBuffer, const Image&, const Inputs&, const Image*, uint32_t,
                  vk::Device) {
    return 0;
}

#endif

} // namespace BB::FG::Dlss
