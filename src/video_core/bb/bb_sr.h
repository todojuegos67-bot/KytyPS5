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
    Image reactive; // render size, R8 0..1: trust the current frame (particles, fire, fog)
    bool has_reactive = false;
    vk::Extent2D render{};
    vk::Extent2D upscale{};
    float jitter[2]{}; // render pixels, the sub-pixel offset the scene was rasterised with
    bool reset = false;
    bool hdr = true;            // linear HDR input (scene mode) or tonemapped LDR (composite mode)
    bool depth_inverted = true; // greater-is-closer
    // 0.86: the colour holds sRGB-encoded (perceptual) LDR values, not linear light (the window
    // path with BB_SR_LDR_ENCODED=1). False for every linear input (the default).
    bool encoded = false;
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

// ---- 0.86 [sdk-inputs]: what each SDK is told, in one comparable line ----
// The sign (+1 / -1) of the jitter handed to an SDK (BB_JITTER_SIGN_<DLSS|FSR3|FSR4|XESS>=-1
// flips only that SDK; default +1: all four take the offset the scene was rasterised with).
float JitterSign(Kind kind);
// The sign of the motion vectors handed to XeSS (BB_XESS_MV_SIGN) or FSR 3.1 (BB_FSR3_MV_SIGN).
// Intel's XeSS-SR guide and AMD's FSR 3 upscaler manual both define the vectors as screen-space
// pixels from the current frame to the previous one, which is what bb_mv.comp writes, so the
// correct sign is +1. A -1 in the environment (an old launcher setting) is ignored unless
// BB_MV_SIGN_FORCE=1 is also set (it is logged either way).
float MvSign(Kind kind);
struct SdkInputs {
    float mv_scale[2]{1.0f, 1.0f}; // the value given to the SDK's own vector scale parameter
    float mv_sign[2]{1.0f, 1.0f};  // the sign part of mv_scale
    const char* mv_units = "";     // what a vector means to the SDK once scaled
    uint32_t flags = 0;            // the SDK's flag word (or our own bits for DLSS)
    std::string (*flags_text)(uint32_t) = nullptr;
    const char* exposure = "";
    const char* depth_note = "";   // far plane handling
};
// Logs "[sdk-inputs] <SDK>: ..." at the first dispatch and again whenever one of its fields (not
// the per-frame jitter value) changes. Cheap to call every dispatch.
void LogSdkInputs(Kind kind, const Desc& desc, const SdkInputs& in);

// Before the Vulkan instance: DLSS loads Streamline, whose interposer is then the Vulkan loader.
namespace Dlss {
bool Preload();                      // true when Streamline was loaded (BB_UPSCALER=dlss)
const wchar_t* LoaderPath();         // sl.interposer.dll, or null
bool Manual();                       // BB_SL_NO_INTERPOSER_LOADER=1: slSetVulkanInfo after the device
bool Loaded();                       // slInit succeeded (DLSS or DLSS-G asked)
uint32_t NextFrameIndex();           // the one Streamline frame index stream (DLSS and DLSS-G)
uint32_t LastFrameIndex();           // the index of the last DLSS evaluate
void* Proc(const char* name);        // an sl.interposer.dll export
bool FeatureReady(uint32_t feature, vk::PhysicalDevice physical); // supported and loaded
void* FeatureProc(uint32_t feature, const char* name);            // slGetFeatureFunction
const std::vector<std::string>& RequiredDeviceExtensions();   // slGetFeatureRequirements
const std::vector<std::string>& RequiredInstanceExtensions();
const std::vector<std::string>& RequiredFeatures12();
const std::vector<std::string>& RequiredFeatures13();
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
