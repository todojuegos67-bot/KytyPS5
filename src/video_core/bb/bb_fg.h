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
//       | dlss-x2 | dlss-x3 | dlss-x4 | dlss-x5 | dlss-x6 | dlss-dyn (bb-clean-021-dlssg: NVIDIA
//       DLSS Frame Generation through Streamline, bb_fg_dlssg.cpp; x5 / x6 trimmed to what the
//       GPU and nvngx_dlssg.dll make). Only one frame generation is ever active.
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
    int multiplier = 2; // fixed multiplier (DLSS-G dynamic: its maximum)
    bool dynamic = false;
};
const Config& Cfg();
const char* Name(); // "off", "amd-x2", "dlss-x3", "dlss-dyn" ...
bool On();          // BB_FG asked (inputs captured even without an upscaler)

bool FgBlocked(const char** why);
bool Fsr3Requested(); // FSR 3 frame generation asked and the FidelityFX DLL loaded
bool Fsr3Active();    // the frame interpolation swapchain exists
// DLSS-G (BB_FG=dlss-*): Streamline loads before the Vulkan instance with sl.interposer.dll as
// the loader (it is not otherwise: BB_FG off / amd-* keep the clean build's manual hooking).
bool DlssRequested();
bool DlssgEnabled(); // DLSS-G ready on this device
// After the dispatcher's init(device): DLSS-G's per-command entry points back to the system
// loader (bb_fg_dlssg.cpp). Nothing unless BB_FG=dlss-*.
void OnDispatch(vk::Device device);
// The swapchain's present mode with DLSS-G (no vsync on Vulkan); `current` otherwise.
vk::PresentModeKHR PreferredPresentMode(vk::PresentModeKHR current,
                                        const std::vector<vk::PresentModeKHR>& available);
// Before a swapchain is destroyed (the device idle): DLSS-G off and freed.
void SwapchainChanging();

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
// DLSS-G, the present thread: before the swapchain present (the frame's commands submitted;
// region: the game image in the backbuffer) and right after it.
void BeforePresentDlssg(const Inputs* inputs, vk::Extent2D backbuffer, vk::Format format,
                        vk::Rect2D region, uint32_t image_count);
void AfterPresentDlssg();

// F8 panel: FSR 3 FG's last per-second real / shown fps and its state ("on", or why off).
bool PanelInfo(float& real, float& shown, const char*& state);

} // namespace BB::FG
