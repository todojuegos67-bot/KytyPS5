// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne frame generation (BB_FG), ported from the Demon's Souls fork's src/local:
//  - frame-gen.cpp: NVIDIA DLSS Frame Generation through Streamline 2.14.16 (sl.dlss_g, Reflex, PCL),
//    fixed x2..x4 (x5 / x6 where the GPU makes them) and the app-side dynamic multiplier
//    (bb_dyn_multiplier.h; Streamline's eDynamic is D3D12 only).
//  - fsr3-fg.cpp: AMD FSR 3 frame generation (the FidelityFX frame interpolation swapchain), x2 with
//    AMD's DLL, x3 / x4 and the dynamic multiplier with the patched one
//    (tools/bb/fsr3-mfg/ffx-sdk-v1.1.4-mfg.patch, with its command list pool fix), and its pacing.
//  - anti-lag.cpp: bb_antilag.h.
// Inputs (all from the upscaler, bb_upscale.h): the upscaled image before the HUD (the HUD-less
// colour, through the presenter's own post-processing pass so it matches the backbuffer), the
// scene depth, the camera-reconstructed motion vectors, the jitter and the camera matrices. Frames
// without them (menus, movies, loading screens: no 3D scene upscaled) are shown as they are.
//
// BB_FG=off | amd-x2 | amd-x3 | amd-x4 | amd-dyn | dlss-x2 | dlss-x3 | dlss-x4 | dlss-x5 | dlss-x6 |
//       dlss-dyn
// Every KYTY_* knob of the Demon's Souls code is read as BB_* first (BB_FSR3_TARGET_FPS,
// BB_DLSSG_TARGET_FPS, BB_VRR_CAP_FPS, BB_DLSSG_REFLEX, BB_ANTILAG, BB_LATENCY, ...).
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::FG {

// "KYTY_X": BB_X when set, else KYTY_X.
const char* Env(const char* kyty_name);
// The Demon's Souls lines ([dlssg], [fsr3], [fsr3-mfg], [dyn], [lat] ...): to the log and to the
// metrics file (BB_METRICS_OUT, read by metricas-bb.ps1).
void Printf(const char* format, ...);

enum class Kind : uint32_t { Off, Amd, Dlss };
struct Config {
    Kind kind = Kind::Off;
    int multiplier = 2; // the fixed multiplier, or the dynamic one's maximum
    bool dynamic = false;
};
const Config& Cfg();
const char* Name(); // "off", "dlss-x3", ...

bool StreamlineWanted();   // DLSS-G asked: Streamline must load before the Vulkan instance
bool StreamlineLoaded();   // Streamline is the Vulkan loader (its Reflex is the latency mode)
bool DlssgEnabled();       // DLSS-G ready on this device
int DlssgGenerated();      // generated frames per real frame in use (0: off)
bool Fsr3Requested();      // FSR 3 frame generation asked and the FidelityFX DLL loaded
bool Fsr3Active();         // the frame interpolation swapchain exists

// ---- Device creation ----
void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next);
// FSR 3: a second queue in the graphics family (its presents) and two in an async compute family.
struct QueuePlan {
    uint32_t graphics_count = 1;
    uint32_t compute_family = UINT32_MAX;
    uint32_t compute_count = 0;
};
QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              uint32_t graphics_family, vk::Queue graphics_queue, const QueuePlan& plan,
              std::mutex* submit_mutex);

// ---- Swapchain (FSR 3 replaces it; Reflex / Anti-Lag chain into it) ----
bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out);
void DestroySwapchain(vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index);
vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info);
void SwapchainChanging(); // DLSS-G off before a resize / full screen change
void OnSwapchain(vk::SwapchainKHR swapchain);
void* ChainSwapchain(void* next);

// ---- Per frame ----
struct Image {
    vk::Image image{};
    vk::ImageView view{};
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent{};
    vk::ImageUsageFlags usage{};
};
struct Inputs {
    bool valid = false; // a 3D frame (the upscaler ran with depth and vectors this frame)
    Image depth;        // render size, D32, SHADER_READ_ONLY_OPTIMAL
    Image motion;       // render size, RG16F pixels towards the previous frame, SHADER_READ_ONLY
    Image hudless;      // output size, the frame image's format, SHADER_READ_ONLY_OPTIMAL
    bool has_hudless = false;
    vk::Extent2D render{};
    float jitter[2]{};
    bool reset = false;
    bool depth_inverted = true;
    float fov_y = 1.0f, z_near = 0.1f, z_far = 1000.0f;
    bool camera_valid = false;
    float view_to_clip[16]{};
    float clip_to_prev_clip[16]{};
    uint64_t lat_index = 0; // BB::AntiLag's frame index (0: none)
};
// FG wants the inputs captured (any method asked and its back end ready).
bool WantInputs();
// Before the swapchain present (the present thread, the frame's commands submitted). region: the
// game image inside the backbuffer.
void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format, vk::Rect2D region, uint32_t image_count);
void AfterPresent();

} // namespace BB::FG
