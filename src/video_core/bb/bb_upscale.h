// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021: Bloodborne temporal upscaling on stock shadPS4 0.21.0, reduced to what the
// upscaler needs. The game renders at output / ratio (core/bb_respatch: the game's own size
// immediates), the presenter runs DLSS (Streamline) or FSR 4 on the presented image, with the
// scene depth and camera motion vectors (one compute pass from depth + the scene constant
// block's camera). No per-draw analysis: one size compare per depth target bind, one float
// compare per flattened user-data upload, and nothing at all when BB_UPSCALER is unset.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include "common/types.h"
#include "video_core/bb/bb_sr.h"

namespace BB::Upscale {

struct Config {
    SR::Kind kind = SR::Kind::Off;
    SR::Quality quality = SR::Quality::Quality;
    float sharpness = 0.3f;
    bool jitter = true;
};
const Config& Cfg();
bool Enabled();       // an upscaler (or native AA) was asked for
bool Temporal();      // DLSS or FSR 4 asked
// bb-clean-021-fg: the scene depth, camera and motion vectors are captured (an upscaler, or
// frame generation without one: BB_FG != off).
bool Capture();
// The output size the resolution patch divides by the ratio (the launcher's output resolution):
// the upscaler outputs this size, not the window's; the presenter then scales it to the window.
void SetOutputSize(u32 w, u32 h);
bool OutputSize(u32& w, u32& h);
bool BackendReady();  // ... and its back end is up
float LiveSharpness();
double Ratio();       // output / render asked by the mode
void Checkpoint(std::string_view text);

// Instance / device (as 1.59).
void PreInstance();
const wchar_t* VulkanLoaderOverride();
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family);
bool RunBackend(const SR::Desc& desc);

// The size the scene renders at (the last presented image's size; 0 until the first flip).
extern u32 g_scene_w, g_scene_h;
// Flattened user data of a stage (Bloodborne's 864-byte scene constant block carries the camera).
void OnConstants(const u32* data, size_t count);
// This frame's jitter in render pixels (zero when off).
void Jitter(float& x, float& y);

struct Camera {
    bool valid = false;
    float clip_to_prev_clip[16]{};
    float view_to_clip[16]{};
    float fov_y = 1.0f, z_near = 0.1f, z_far = 1000.0f;
    bool reset = false;
};
// At the flip: the frame's camera (and the motion to the previous one); starts the next frame.
Camera EndFrame(u32 flip_w, u32 flip_h);

// Once-per-second performance line and the F8 overlay values.
struct Stats {
    float fps = 0, frame_ms = 0, cpu_ms = 0, gpu_ms = 0, wait_ms = 0, sr_ms = 0, mv_ms = 0;
    u32 in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    bool upscaled = false;
};
void NoteFrame(double cpu_ms, double wait_ms, double gpu_ms, double sr_ms, double mv_ms,
               u32 in_w, u32 in_h, u32 out_w, u32 out_h, bool upscaled);
const Stats& LastStats();
bool OverlayShown();
void ToggleOverlay();
void DrawOverlay(); // ImGui, only while shown
void LoadOverlayFont(); // Consolas Bold from Windows fonts, at ImGui init (as Demon's Souls)

} // namespace BB::Upscale
