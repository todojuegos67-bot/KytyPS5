// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne upscaling (phase 1): DLSS / DLAA (Streamline), XeSS, FSR 3.1 and FSR 4 on the game's
// 3D image before the HUD, with the HUD left to the game at the output size.
//
// How the frame is read (all from the guest's own passes, nothing title specific is hard coded;
// BB_UPSCALE_* overrides exist for every guess):
//  - scene depth: the depth target of the frame's G-buffer pass (the pass with the most colour
//    targets, at least 3, and a depth target). Bloodborne: D32S8, 6-7 MRTs.
//  - scene colour: the last HDR colour target (RGBA16F / B10G11R11) drawn with that depth bound,
//    at its size: the lit, forward-shaded 3D image, before post-processing and the HUD.
//  - injection: when the game first samples the final scene colour (after its last write that
//    frame) the upscaler runs and the sample is redirected to the upscaled image ("scene" mode,
//    internal size == output size: DLAA / native AA). When the 3D renders below the output size the
//    injection moves to the first draw at the output size that samples a render-size image of the
//    scene chain ("composite" mode, LDR), so post-processing stays at the size the game made it for.
//    Everything the game draws after that (its HUD, into the output / flip images) never passes
//    through the upscaler.
//  - motion vectors: Bloodborne has no TAA and no motion vectors; camera motion vectors are made
//    from the scene depth and the view-projection found in the G-buffer draws' vertex constant
//    buffers (this frame's and the previous one's), BB_MV=zero turns them off.
//  - jitter: the scene passes' viewport is offset by a Halton(2,3) sub-pixel sequence while an
//    upscaler runs with reconstructed vectors (BB_UPSCALE_JITTER=0 turns it off).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include "common/types.h"
#include "video_core/bb/bb_sr.h"

namespace BB::Upscale {

struct Config {
    SR::Kind kind = SR::Kind::Off;
    SR::Quality quality = SR::Quality::Quality;
    float sharpness = 0.3f;
    bool jitter = true;
    bool mv_reconstruct = true;
    bool hudless_copy = false;
    int force_mode = 0; // 0 auto, 1 scene, 2 composite
    VAddr force_source = 0;
    u32 log_every = 600;
};
const Config& Cfg();
bool Enabled(); // an upscaler (or native) was asked for

// Instance / device creation.
void PreInstance();
const wchar_t* VulkanLoaderOverride(); // Streamline's interposer for DLSS
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family);
bool BackendReady();

// ---- Frame tracking (GPU command processor thread) ----
struct Target {
    VAddr addr = 0;
    u32 width = 0, height = 0;
    vk::Format format = vk::Format::eUndefined;
};
struct Viewport {
    float xscale = 1, xoffset = 0, yscale = 1, yoffset = 0, zscale = 1, zoffset = 0;
};
// Every draw, once its targets are known.
void OnDraw(std::span<const Target> colors, const Target* depth, bool depth_greater,
            const Viewport& viewport);
// Every dispatch.
void OnDispatch();
// The current draw is a scene draw whose vertex constant buffers should be scanned for the camera.
bool WantCameraScan();
void OnVertexConstants(const void* data, size_t size);
// A sampled colour image in the current draw/dispatch.
enum class Sample { Keep, Inject, Redirect };
Sample OnSample(VAddr addr, u32 width, u32 height, vk::Format format);
// The current draw is a scene draw to jitter: the offset in pixels.
bool Jitter(float& x, float& y);
// The guest flip (end of the frame).
void OnFlip(VAddr addr, u32 width, u32 height);
// The current draw renders into the presented image (HUD stage): true once per frame, before its
// first HUD draw, when the HUD-less copy should be taken.
bool WantHudlessCopy();

// ---- What an injection needs (filled by OnSample(...) == Inject) ----
struct Plan {
    bool scene_mode = true;
    VAddr color_addr = 0;
    VAddr depth_addr = 0;
    u32 render_w = 0, render_h = 0;
    u32 output_w = 0, output_h = 0;
    bool reset = false;
    bool hdr = true;
    bool depth_inverted = true;
    float jitter[2]{};
    float delta_ms = 16.7f;
    bool camera_valid = false;
    float view_to_clip[16]{};
    float clip_to_prev_clip[16]{};
    float fov_y = 1.0f, z_near = 0.1f, z_far = 1000.0f;
    Viewport viewport;
};
const Plan& CurrentPlan();
void OnInjected(bool ok, float mv_ms_unused);
void OnGpuTime(double ms);
void OnHudlessCopied();

} // namespace BB::Upscale
