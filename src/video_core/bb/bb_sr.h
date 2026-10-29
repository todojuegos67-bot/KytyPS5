// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Super resolution back ends for the Bloodborne port (BB_UPSCALER). Ported from the Demon's Souls
// fork's src/local (frame-gen.cpp DLSS through Streamline, xess-sr.cpp, fsr3-fg.cpp FSR 3.1
// upscaling, fsr4-sr.cpp FSR 4 v07 INT8) onto shadPS4's Vulkan renderer. Every back end records
// into the rasterizer's command buffer, outside a render pass, with the inputs in the layouts given.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace BB::SR {

enum class Kind : uint32_t { Off, Native, Dlss, Xess, Fsr3, Fsr4 };

// The Demon's Souls ratios (output / render, per axis).
enum class Quality : uint32_t {
    Native,           // DLAA / XeSS AA / FSR native AA: 1.0
    UltraQualityPlus, // XeSS only: 1.3
    UltraQuality,     // XeSS only: 1.5
    Quality,          // 1.5 (XeSS 1.7)
    Balanced,         // 1.72 (XeSS 2.0)
    Performance,      // 2.0 (XeSS 2.3)
    UltraPerformance, // 3.0
    Dynamic,          // FSR 4 drs model: any ratio
};

const char* KindName(Kind kind);
const char* QualityName(Quality quality);
double QualityRatio(Kind kind, Quality quality);

struct Image {
    vk::Image image{};
    vk::ImageView view{};
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::ImageUsageFlags usage{};
};

struct Desc {
    vk::CommandBuffer cmd{};
    Image color;  // render size, shader read
    Image depth;  // render size, depth read only
    Image motion; // render size, RG16F pixels towards the previous frame, shader read
    Image output; // upscale size, RGBA16F, GENERAL, storage
    vk::Extent2D render{};
    vk::Extent2D upscale{};
    float jitter[2]{}; // render pixels, the sub-pixel offset the scene was rasterised with
    bool reset = false;
    bool hdr = true;            // linear HDR input (scene mode) or tonemapped LDR (composite mode)
    bool depth_inverted = true; // greater-is-closer
    float delta_ms = 16.7f;
    float fov_y = 1.0f;
    float z_near = 0.1f;
    float z_far = 1000.0f;
    float sharpness = 0.3f;
    Quality quality = Quality::Quality;
    // Camera, row major (clip_i = dot(row_i, p)): this frame's view-projection and the transform
    // from this frame's clip space to the previous frame's (identity when unknown).
    float view_to_clip[16]{};
    float clip_to_prev_clip[16]{};
    bool camera_valid = false;
};

// Before the Vulkan instance: DLSS loads Streamline, whose interposer is then the Vulkan loader.
namespace Dlss {
bool Preload();                      // true when Streamline was loaded (BB_UPSCALER=dlss)
const wchar_t* LoaderPath();         // sl.interposer.dll, or null
bool Manual();                       // BB_SL_NO_INTERPOSER_LOADER=1: slSetVulkanInfo after the device
const std::vector<std::string>& RequiredDeviceExtensions();   // slGetFeatureRequirements
const std::vector<std::string>& RequiredInstanceExtensions();
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
bool Ready();
bool Evaluate(const Desc& desc);
} // namespace Dlss

namespace Xess {
bool Load();
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
bool Ready();
bool Evaluate(const Desc& desc);
} // namespace Xess

namespace Fsr3 {
bool Load();
void OnDevice(vk::PhysicalDevice physical, vk::Device device, PFN_vkGetDeviceProcAddr gdpa);
bool Ready();
bool Evaluate(const Desc& desc);
} // namespace Fsr3

namespace Fsr4 {
void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
// Turns on what the model needs in the caller's feature structs; returns the new pNext head.
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, uint32_t family);
bool Ready();
const char* Name();
bool Evaluate(const Desc& desc);
} // namespace Fsr4

} // namespace BB::SR
