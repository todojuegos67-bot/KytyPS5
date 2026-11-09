// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-fg: Bloodborne frame generation, the AMD FSR 3 part of the 1.59 build's BB_FG
// (bb_fg_fsr3.cpp: the FidelityFX frame interpolation swapchain, x2), without its extras (guest
// flip pacer, present ids / present wait, dynamic multiplier, Anti-Lag, metrics, object vectors).
// Inputs: the scene depth and camera motion vectors of the presented frame (bb_present_sr.cpp),
// with or without an upscaler. Frames without them (menus, movies, loading screens) are shown as
// they are.
//
// BB_FG=off (default: nothing changes) | amd-x2 (aliases fsr3-x2, amd-2, fsr3)
//       | dlss-x2 | dlss-x3 | dlss-x4 | dlss-dyn (bb-clean-021-bbport: NVIDIA DLSS Frame Generation
//         through NGX directly, Supermedo's path: bb_fg_dlssg.cpp and the presenter's loop)
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::FG {

// "KYTY_X": BB_X when set, else KYTY_X.
const char* Env(const char* kyty_name);
void Printf(const char* format, ...);
void NotePacing(double real_sd_ms, double shown_sd_ms);

enum class Kind : uint32_t { Off, Amd, Dlss };
struct Config {
    Kind kind = Kind::Off;
    int multiplier = 2;
    bool dynamic = false; // dlss-dyn
};
const Config& Cfg();
const char* Name(); // "off", "amd-x2", "dlss-x3", "dlss-dyn"
bool On();          // BB_FG asked (inputs captured even without an upscaler)

bool FgBlocked(const char** why);
bool Fsr3Requested(); // FSR 3 frame generation asked and the FidelityFX DLL loaded
bool Fsr3Active();    // the frame interpolation swapchain exists

// ---- Device creation ----
void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
// FSR 3: a second queue in the graphics family (its presents) and two in an async compute family.
struct QueuePlan {
    uint32_t graphics_count = 1;
    uint32_t compute_family = UINT32_MAX;
    uint32_t compute_count = 0;
};
QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family);
void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t graphics_family,
              vk::Queue graphics_queue, const QueuePlan& plan, std::mutex* submit_mutex);

// ---- Swapchain (FSR 3 replaces it) ----
bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out);
void DestroySwapchain(vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index);
vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info);

// ---- Per frame ----
struct Image {
    vk::Image image{};
    vk::ImageView view{};
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent{};
    vk::ImageUsageFlags usage{};
};
bool WantsUiLayer();
void SetUiLayer(const Image* ui);
struct Inputs {
    bool valid = false; // a 3D frame (depth and vectors captured this frame)
    Image depth;        // render size, the scene depth's format, SHADER_READ_ONLY_OPTIMAL
    Image motion;       // render size, RG16F pixels towards the previous frame, SHADER_READ_ONLY
    Image mask;
    bool has_mask = false;
    Image hudless; // output size, the backbuffer's format, SHADER_READ_ONLY_OPTIMAL
    bool has_hudless = false;
    vk::Extent2D render{};
    float jitter[2]{};
    bool reset = false;
    bool depth_inverted = true;
    float fov_y = 1.0f, z_near = 0.1f, z_far = 1000.0f;
    bool camera_valid = false;
    float view_to_clip[16]{};
    float clip_to_prev_clip[16]{};
    uint64_t color_frame = ~0ull, depth_frame = ~0ull, mv_frame = ~0ull, obj_frame = ~0ull,
             hudless_frame = ~0ull;
};
// FG wants the inputs captured (asked and its back end ready).
bool WantInputs();
// Before the swapchain present (the present thread, the frame's commands submitted).
void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format);

// ---- DLSS Frame Generation (NGX, bb_fg_dlssg.cpp) ----
namespace Dlss {
bool Requested(); // BB_FG=dlss-*
void AddInstanceExtensions(const std::vector<std::string>& available,
                           std::vector<const char*>& extensions);
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
bool Ready();          // NGX up with Frame Generation
uint32_t MaxCount();   // NGX's MultiFrameCountMax (1..3), 0 when unavailable
uint32_t WantedCount(); // frames per game frame asked (dlss-x2: 1 .. dlss-x4: 3, dyn: max)
// `backbuffer`: the finished frame (UI included, GENERAL); `outputs`: `count` storage images of
// its size and format (GENERAL). Records multiFrameIndex 1..count; returns how many were made.
uint32_t Generate(vk::CommandBuffer cmd, const Image& backbuffer, const Inputs& inputs,
                  const Image* outputs, uint32_t count, vk::Device device);
} // namespace Dlss

// F8 panel: FSR 3 FG's last per-second real / shown fps and its state ("on", or why off).
bool PanelInfo(float& real, float& shown, const char*& state);

} // namespace BB::FG
