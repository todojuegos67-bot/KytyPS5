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
    bool dynamic = false; // dlss-dyn, amd-dyn
    // bb-dlssg-sl: DLSS-G through Streamline (sl.dlss_g + Reflex / PCL) instead of the NGX
    // bridge: BB_FG=dlss-sl[-x2|-x3|-x4|-dyn], or BB_FG=dlss-* with BB_DLSSG_SL=1. Opt-in.
    bool streamline = false;
};
const Config& Cfg();
const char* Name(); // "off", "amd-x2", "dlss-x3", "dlss-dyn"
void SetAmdActive(int multiplier); // (amd-dyn) generated frames per real one + 1, on every present
// bb-fsr3-mfg-mult: AMD generated frames per real frame: AmdFrames() = the swapchain's (x3: 2,
// x4 / dynamic: 3), AmdActiveFrames() = this present's (1..AmdFrames()); ApplyAmdEnv() writes
// both to KYTY_FSR3_FG_FRAMES / _ACTIVE (the patched DLL reads them with GetEnvironmentVariableA).
int AmdFrames();
int AmdActiveFrames();
void ApplyAmdEnv();
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
// bb-dlssg-sl: DLSS Frame Generation through Streamline (bb_fg_dlssg_sl.cpp). Streamline's
// DLSS-G plugin generates and presents the frames inside its vkQueuePresentKHR proxy: the
// swapchain functions go through sl.interposer.dll's proxies (manual hooking), the frame's
// inputs are tagged on DLSS super resolution's frame token, and PCL markers + Reflex options
// tell the driver about the frame. Needs BB_UPSCALER=dlss (Streamline loaded). No pacing of ours.
namespace DlssSl {
bool Requested(); // BB_FG=dlss-sl* / BB_DLSSG_SL=1 (the NGX bridge is then not used)
void Queues(uint32_t* graphics, uint32_t* compute); // extra queues Streamline's features want
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
bool Hooked(); // the swapchain goes through Streamline's proxies
bool Ready();  // DLSS-G feature functions and proxies all present
void* SurfaceProc(vk::Instance instance); // vkCreateWin32SurfaceKHR proxy, or null
vk::Result CreateSwapchain(vk::Device device, const vk::SwapchainCreateInfoKHR& info,
                           vk::SwapchainKHR* out);
void DestroySwapchain(vk::Device device, vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain, uint32_t* count,
                              vk::Image* images);
vk::Result AcquireNextImage(vk::Device device, vk::SwapchainKHR swapchain, uint64_t timeout,
                            vk::Semaphore semaphore, uint32_t* index);
vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info);
vk::Result DeviceWaitIdle(vk::Device device);
void DestroySurface(vk::Instance instance, vk::SurfaceKHR surface);
// GPU thread, the game frame's command buffer: PCL simulation / render-submit-start markers,
// the DLSS-G tags (depth, motion vectors, HUD-less: copied now, eOnlyValidNow) and options.
// inputs null or !valid: DLSS-G off for this frame. Returns the Streamline frame index (0: none).
uint32_t TagFrame(vk::CommandBuffer cmd, const Inputs* inputs, vk::Extent2D backbuffer,
                  vk::Format format);
// Present thread, around every present through Streamline (frame: TagFrame's index, 0: an
// untagged present, which gets a fresh index and the full PCL marker set so Reflex never goes
// missing, with DLSS-G off). Returns the index to pass to AfterPresent. false from
// PresentAllowed: shutting down, skip the present.
uint32_t BeforePresent(uint32_t frame);
void AfterPresent(uint32_t frame, bool presented);
bool PresentAllowed();
// Exit: DLSS-G off, Reflex off, wait for an in-flight present (bounded), slShutdown. Any thread,
// once; bounded waits only.
void Shutdown();
// bb-sl-metrics: the last second's numbers (from AfterPresent's once-a-second slDLSSGGetState and
// [dlssg-sl] counters), for the F8 "Metricas" panel and the [bbport] line. Read only: calling it
// changes nothing in Streamline's path. False until Ready() and the first second has passed.
struct SecondStats {
    bool on = false;          // options mode on and Streamline presents generated frames
    float real_fps = 0;       // game frames presented through the proxy per second
    float shown_fps = 0;      // real_fps x presented per frame
    float generated_fps = 0;  // shown_fps - real_fps
    uint32_t presented_per_frame = 0; // DLSSGState::numFramesActuallyPresented (0: unknown)
    uint32_t multiplier = 1;  // options: numFramesToGenerate + 1 (1 while off)
    uint32_t status = 0;      // DLSSGState::status
    bool reflex = false;      // slReflexSetOptions(low latency) returned eOk
};
bool Stats(SecondStats& out);
} // namespace DlssSl

bool PanelInfo(float& real, float& shown, const char*& state);
// bb-menu-ui: the presenter's measured game / shown frame rates with DLSS-G (once a second), for
// the F8 panel (it showed "FSR 3 x2 · DLSS-G (NGX)" and no shown rate, as if FG were off).
void NoteDlssPanel(float real, float shown);
// The panel row's name: "DLSS-G" or "FSR 3 x2".
const char* PanelName();

} // namespace BB::FG
