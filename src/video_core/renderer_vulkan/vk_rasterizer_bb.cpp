// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The rasterizer side of the Bloodborne upscaling (video_core/bb/bb_upscale.h): the upscaler's own
// images, the camera motion vector pass, the GPU timing, and the call into the selected back end,
// recorded in the rasterizer's command buffer at the injection point.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <stb_image_write.h>
#include <unordered_map>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/bb/bb_analyze.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_objmv.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/bb/bb_dlss5.h"
#include "video_core/host_shaders/bb_alpha_comp.h"
#include "video_core/host_shaders/bb_hud_comp.h"
#include "video_core/host_shaders/bb_hudless_lut_comp.h"
#include "video_core/host_shaders/bb_mv_comp.h"
#include "video_core/host_shaders/bb_rtrefl_comp.h"
#include "video_core/host_shaders/bb_vol_comp.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

struct OwnImage {
    vk::Image image{};
    VmaAllocation allocation{};
    vk::ImageView view{};
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent{};
    vk::ImageUsageFlags usage{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

struct BBVk {
    bool init = false;
    bool failed = false;
    vk::Device device{};
    VmaAllocator allocator{};
    vk::DescriptorSetLayout mv_set_layout{};
    vk::PipelineLayout mv_layout{};
    vk::Pipeline mv_pipeline{};
    OwnImage reactive{};
    std::array<OwnImage, 2> lum{};
    u32 lum_index = 0;
    vk::Sampler nearest{};
    OwnImage mv;
    OwnImage mv_zero; // BB_MV_UPSCALE=zero: the upscalers' motion vectors (cleared each frame)
    bool vol_init = false, vol_failed = false;
    vk::DescriptorSetLayout vol_set_layout{};
    vk::PipelineLayout vol_layout{};
    vk::Pipeline vol_pipeline{};
    OwnImage vol; // BB_VOL=1: the scene colour after the volumetric pass, the upscalers' input
    std::array<vk::Buffer, 2> vol_red{}; // the sky / sun reduction, ping-pong
    std::array<VmaAllocation, 2> vol_red_alloc{};
    u32 vol_index = 0;
    u64 vol_passes = 0;
    OwnImage output;
    OwnImage hudless;
    // Upscaling to the window size: the presented image (upscale + HUD) and its pass.
    OwnImage present_in; // RGBA16F copy of the presented image before the HUD
    OwnImage present;
    bool present_ready = false;
    VAddr present_flip = 0;
    const char* present_why = ""; // why the last window-size upscale / composite failed
    bool hud_init = false, hud_failed = false;
    vk::DescriptorSetLayout hud_set_layout{};
    vk::PipelineLayout hud_layout{};
    vk::Pipeline hud_pipeline{};
    vk::Sampler linear{};
    vk::QueryPool queries{};
    std::array<bool, 8> query_pending{};
    u32 query_slot = 0;
    double timestamp_period_ns = 1.0;
    // The alpha / non-finite fix-up of the output (bb_alpha.comp).
    bool alpha_init = false, alpha_failed = false;
    vk::DescriptorSetLayout alpha_set_layout{};
    vk::PipelineLayout alpha_layout{};
    vk::Pipeline alpha_pipeline{};
    // BB_RT_REFLECT=2: the reflection pass (bb_rtrefl.comp) and its image, which the upscaler reads
    // in place of the scene colour.
    bool refl_init = false, refl_failed = false;
    vk::DescriptorSetLayout refl_set_layout{};
    vk::PipelineLayout refl_layout{};
    vk::Pipeline refl_pipeline{};
    OwnImage refl_out{};
    u64 refl_frames = 0;
    bool refl_skip_logged = false;
    // The output self-check: a few texels read back from the first upscales.
    vk::Buffer check_buffer{};
    VmaAllocation check_allocation{};
    void* check_mapped = nullptr;
    u64 check_tick = 0;
    bool check_pending = false;
    u32 check_done = 0;
    u64 upscales = 0;
    std::string check_what;
    vk::Format present_src_format = vk::Format::eUndefined; // the flip image present_in copies
    vk::Format present_view_format = vk::Format::eUndefined; // the presenter's view of the flip
    // present_in holds sRGB-encoded (perceptual) LDR values: the upscaler is given and returns
    // encoded colour, the HUD composite decodes it (bb_hud.comp flag 1).
    bool present_encoded = false;
    u64 composites = 0;
    // The luminance self-check of the presented image (bb_hud.comp flag 2, frame 120).
    vk::Buffer lum_buffer{};
    VmaAllocation lum_allocation{};
    void* lum_mapped = nullptr;
    u64 lum_tick = 0;
    bool lum_pending = false;
    // Per-stage GPU timestamps of the window-size path: 4 frames x 8 stamps
    // (0 copy, 1 MV + upscaler, 2 upscaler end, 3 alpha end, 4 composite, 5 composite end,
    // 6 present pass end).
    vk::QueryPool stage_queries{};
    u32 stage_slot = 0;
    bool stage_active = false;
    std::array<bool, 4> stage_pending{};
    std::array<u8, 4> stage_written{}; // bit mask of the written stamps
    std::vector<std::pair<u64, vk::ImageView>> transient_views;
    // Frame generation (bb_fg.h): the last upscale's inputs, the HUD-less composite and the
    // per-frame copies of depth and motion vectors.
    bool fg_upscaled = false;
    BB::Upscale::Plan fg_plan{};
    OwnImage fg_present; // the composite without the HUD (bb_hud.comp flag 4)
    bool fg_present_ready = false;
    bool fg_raw_hudless_ready = false; // BBHudlessCopy ran this frame (the in-place path)
    OwnImage fg_hudless_view_copy; // that copy in the presenter's view format (raw bytes)
    // 0.39 (BB_FG_NATIVE): frame generation with no upscaler: the presented image before its HUD
    // (raw copy, its own format): the motion-vector pass's colour input and FG's HUD-less image.
    OwnImage fgn_src;
    OwnImage d5_work; // 0.90: DLSS 5 without an upscaler (RGBA16F copy of the native image)
    bool fgn_ready = false;
    // BB_HUD_DIAG: the presented image's address once its HUD-less copy is taken, frames seen.
    VAddr hud_flip = 0;
    u64 hud_frames = 0, hud_logged_frame = 0;
    u32 hud_draws_logged = 0;
    vk::Buffer diag_buffer{};
    VmaAllocation diag_allocation{};
    void* diag_mapped = nullptr;
    bool diag_pending = false;
    // The in-place HUD-less image through the game's curve (BBHudlessLut).
    std::vector<std::tuple<VAddr, u32, u32>> draw_samples; // this draw's sampled images
    std::unordered_map<VAddr, VAddr> sampled_by;            // target -> same-size image it sampled
    VAddr lut_a = 0;                    // the image the presented image is copied from
    bool lut_captured = false;
    VideoCore::ImageId lut_a_id{};
    OwnImage lut_s, lut_f, lut_out;
    vk::Pipeline lut_pipeline{};
    bool lut_failed_init = false, lut_disabled = false, lut_ready = false;
    std::array<vk::Buffer, 4> lut_buf{};
    std::array<VmaAllocation, 4> lut_alloc{};
    std::array<void*, 4> lut_mapped{};
    std::array<bool, 4> lut_pending{};
    u32 lut_slot = 0;
    // 0.23: the self-check runs on a sampled frame (every lut_every presents); the frames between
    // use the table of the last checked frame (lut_table_slot) when its verdict was good.
    u32 lut_table_slot = ~0u;
    u64 lut_recorded = 0;
    u32 lut_every = 0;
    bool lut_verdict_ok = false;
    u64 lut_frames = 0, lut_ok = 0, lut_conflict = 0, lut_mismatch = 0, lut_bad_streak = 0;
    u64 fg_hudless_frames = 0, fg_frames = 0;
    struct FgSlot {
        OwnImage depth, mv;
    };
    std::array<FgSlot, 8> fg_slots{};
};
BBVk g_bb;

void DestroyImage(OwnImage& img) {
    if (img.view) g_bb.device.destroyImageView(img.view);
    if (img.image) vmaDestroyImage(g_bb.allocator, static_cast<VkImage>(img.image), img.allocation);
    img = {};
}

bool EnsureImage(OwnImage& img, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
    if (img.image && img.extent == extent && img.format == format) {
        return true;
    }
    if (img.image) {
        (void)g_bb.device.waitIdle();
        DestroyImage(img);
    }
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(g_bb.allocator, &ci, &alloc, &image, &img.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    img.image = image;
    const auto [result, view] = g_bb.device.createImageView(vk::ImageViewCreateInfo{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        DestroyImage(img);
        return false;
    }
    img.view = view;
    img.format = format;
    img.extent = extent;
    img.usage = usage;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

void Barrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout,
             vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = img.layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    img.layout = layout;
}

bool InitOnce(const Instance& instance) {
    if (g_bb.init || g_bb.failed) {
        return g_bb.init;
    }
    g_bb.device = instance.GetDevice();
    g_bb.allocator = instance.GetAllocator();
    const auto device = g_bb.device;
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 5,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 112,
    };
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto module = CompileSPV(BB_MV_COMP, device);
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    vk::PipelineShaderStageCreateInfo{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = layout,
            });
    device.destroyShaderModule(module);
    auto [sampler_result, sampler] = device.createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    });
    auto [query_result, queries] = device.createQueryPool(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = 16,
    });
    if (set_result != vk::Result::eSuccess || layout_result != vk::Result::eSuccess ||
        pipe_result != vk::Result::eSuccess || sampler_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "[BB_UPSCALE] the motion vector pass was not created: off");
        g_bb.failed = true;
        return false;
    }
    g_bb.mv_set_layout = set_layout;
    g_bb.mv_layout = layout;
    g_bb.mv_pipeline = pipeline;
    g_bb.nearest = sampler;
    g_bb.queries = query_result == vk::Result::eSuccess ? queries : vk::QueryPool{};
    {
        auto [stage_result, stage_pool] = device.createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = 32,
        });
        g_bb.stage_queries = stage_result == vk::Result::eSuccess ? stage_pool : vk::QueryPool{};
    }
    g_bb.timestamp_period_ns =
        instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    g_bb.init = true;
    return true;
}

void Stamp(vk::CommandBuffer cmd, u32 index) {
    if (!g_bb.stage_active) {
        return;
    }
    cmd.writeTimestamp2(index == 0 ? vk::PipelineStageFlagBits2::eTopOfPipe
                                   : vk::PipelineStageFlagBits2::eBottomOfPipe,
                        g_bb.stage_queries, g_bb.stage_slot * 8 + index);
    g_bb.stage_written[g_bb.stage_slot] |= static_cast<u8>(1u << index);
}

void ReadStageTimestamps() {
    if (!g_bb.stage_queries) {
        return;
    }
    for (u32 slot = 0; slot < g_bb.stage_pending.size(); ++slot) {
        if (!g_bb.stage_pending[slot] || (g_bb.stage_active && slot == g_bb.stage_slot)) {
            continue;
        }
        std::array<u64, 14> data{};
        const auto result = g_bb.device.getQueryPoolResults(
            g_bb.stage_queries, slot * 8, 7, sizeof(data), data.data(), sizeof(u64) * 2,
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        const u8 mask = g_bb.stage_written[slot];
        bool ready = result == vk::Result::eSuccess || result == vk::Result::eNotReady;
        for (u32 i = 0; i < 7 && ready; ++i) {
            ready = !(mask & (1u << i)) || data[i * 2 + 1] != 0;
        }
        if (!ready) {
            continue;
        }
        g_bb.stage_pending[slot] = false;
        // Every stage needs its two ends; a missing stage counts 0 (alpha skipped, say).
        const auto t = [&](u32 a, u32 b) {
            if (!(mask & (1u << a)) || !(mask & (1u << b)) || data[b * 2] < data[a * 2]) {
                return 0.0;
            }
            return static_cast<double>(data[b * 2] - data[a * 2]) * g_bb.timestamp_period_ns /
                   1e6;
        };
        const double ms[BB::Upscale::kStages]{t(0, 1), t(1, 2), t(2, 3), t(4, 5), t(5, 6)};
        BB::Upscale::OnStageTimes(ms);
    }
}

void ReadTimestamps() {
    ReadStageTimestamps();
    if (!g_bb.queries) {
        return;
    }
    for (u32 slot = 0; slot < g_bb.query_pending.size(); ++slot) {
        if (!g_bb.query_pending[slot]) {
            continue;
        }
        std::array<u64, 4> data{};
        const auto result = g_bb.device.getQueryPoolResults(
            g_bb.queries, slot * 2, 2, sizeof(data), data.data(), sizeof(u64) * 2,
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (result == vk::Result::eSuccess && data[1] != 0 && data[3] != 0) {
            const double ms =
                static_cast<double>(data[2] - data[0]) * g_bb.timestamp_period_ns / 1e6;
            BB::Upscale::OnGpuTime(ms);
            g_bb.query_pending[slot] = false;
        }
    }
}

vk::ImageView TransientView(Scheduler& scheduler, vk::Image image, vk::Format format,
                            vk::ImageAspectFlags aspect, vk::ComponentMapping components = {}) {
    // Sampled only: a view in another format of the same image (sRGB of a UNORM image, say) need
    // not support the image's other usages (storage).
    const vk::ImageViewUsageCreateInfo usage{.usage = vk::ImageUsageFlagBits::eSampled};
    const auto [result, view] = g_bb.device.createImageView(vk::ImageViewCreateInfo{
        .pNext = &usage,
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = components,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        return {};
    }
    const vk::Device device = g_bb.device;
    scheduler.DeferOperation([device, view] { device.destroyImageView(view); });
    return view;
}

bool InitHud() {
    if (g_bb.hud_init || g_bb.hud_failed) {
        return g_bb.hud_init;
    }
    g_bb.hud_failed = true;
    const auto device = g_bb.device;
    std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = i,
            .descriptorType = i < 3    ? vk::DescriptorType::eCombinedImageSampler
                              : i == 3 ? vk::DescriptorType::eStorageImage
                                       : vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        };
    }
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    if (set_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.hud_set_layout = set_layout;
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 16,
    };
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &g_bb.hud_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    if (layout_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.hud_layout = layout;
    const auto module = CompileSPV(BB_HUD_COMP, device);
    if (!module) {
        return false;
    }
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    vk::PipelineShaderStageCreateInfo{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = g_bb.hud_layout,
            });
    device.destroyShaderModule(module);
    if (pipe_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.hud_pipeline = pipeline;
    auto [sampler_result, sampler] = device.createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    });
    if (sampler_result != vk::Result::eSuccess) {
        return false;
    }
    if (g_bb.linear) {
        device.destroySampler(sampler);
    } else {
        g_bb.linear = sampler;
    }
    g_bb.hud_failed = false;
    g_bb.hud_init = true;
    return true;
}

bool IsSrgb(vk::Format f) {
    switch (f) {
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return true;
    default:
        return false;
    }
}

// The UNORM twin of an sRGB format (and the format itself otherwise).
vk::Format Unorm(vk::Format f) {
    switch (f) {
    case vk::Format::eR8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Unorm;
    case vk::Format::eB8G8R8A8Srgb:
        return vk::Format::eB8G8R8A8Unorm;
    case vk::Format::eA8B8G8R8SrgbPack32:
        return vk::Format::eA8B8G8R8UnormPack32;
    default:
        return f;
    }
}

// The format the game's own texture view reads the image through, when it differs from the
// image's format only by its sRGB decode: the upscaler must read what the game's shader reads
// (an sRGB view of a UNORM render target decodes to linear, the raw UNORM view does not).
vk::Format SampledFormat(vk::Format image_format, vk::Format view_format) {
    return view_format != image_format && Unorm(view_format) == Unorm(image_format)
               ? view_format
               : image_format;
}

// The format the native presenter reads the flip through when it is the same 8-bit RGBA layout
// (R8G8B8A8 vs B8G8R8A8, sRGB kept as the flip's): else the flip's own format.
vk::Format PresentReadFormat(vk::Format flip) {
    const vk::Format view = g_bb.present_view_format;
    const auto is8 = [](vk::Format f) {
        const vk::Format u = Unorm(f);
        return u == vk::Format::eR8G8B8A8Unorm || u == vk::Format::eB8G8R8A8Unorm;
    };
    if (view == vk::Format::eUndefined || !is8(view) || !is8(flip)) {
        return flip;
    }
    return IsSrgb(flip) ? (Unorm(view) == vk::Format::eB8G8R8A8Unorm ? vk::Format::eB8G8R8A8Srgb
                                                                     : vk::Format::eR8G8B8A8Srgb)
                        : Unorm(view);
}

const char* ColourSpace(vk::Format f) {
    switch (f) {
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR32G32B32A32Sfloat:
        return "linear HDR";
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return "sRGB, decoded to linear on read";
    default:
        return "UNORM, gamma-encoded LDR";
    }
}

bool InitAlpha() {
    if (g_bb.alpha_init || g_bb.alpha_failed) {
        return g_bb.alpha_init;
    }
    g_bb.alpha_failed = true;
    const auto device = g_bb.device;
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    if (set_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.alpha_set_layout = set_layout;
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 16,
    };
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &g_bb.alpha_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    if (layout_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.alpha_layout = layout;
    const auto module = CompileSPV(BB_ALPHA_COMP, device);
    if (!module) {
        return false;
    }
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    vk::PipelineShaderStageCreateInfo{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = g_bb.alpha_layout,
            });
    device.destroyShaderModule(module);
    if (pipe_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.alpha_pipeline = pipeline;
    if (!g_bb.linear) {
        auto [sampler_result, sampler] = device.createSampler(vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        });
        if (sampler_result != vk::Result::eSuccess) {
            return false;
        }
        g_bb.linear = sampler;
    }
    g_bb.alpha_failed = false;
    g_bb.alpha_init = true;
    return true;
}

constexpr float kVolHazeDensity = 1.5f;
constexpr float kVolShaftGain = 0.5f;

bool VolEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_VOL");
        return v && v[0] == '1';
    }();
    return on;
}

bool InitVol() {
    if (g_bb.vol_init || g_bb.vol_failed) {
        return g_bb.vol_init;
    }
    g_bb.vol_failed = true;
    const auto device = g_bb.device;
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    if (set_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.vol_set_layout = set_layout;
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 32,
    };
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &g_bb.vol_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    if (layout_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.vol_layout = layout;
    const auto module = CompileSPV(BB_VOL_COMP, device);
    if (!module) {
        return false;
    }
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    vk::PipelineShaderStageCreateInfo{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = g_bb.vol_layout,
            });
    device.destroyShaderModule(module);
    if (pipe_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.vol_pipeline = pipeline;
    g_bb.vol_failed = false;
    g_bb.vol_init = true;
    LOG_INFO(Render_Vulkan, "BB_VOL=1: volumetric pass (god rays / fog) active");
    return true;
}

bool EnsureVolBuffers() {
    if (g_bb.vol_red[0]) {
        return true;
    }
    for (u32 i = 0; i < g_bb.vol_red.size(); ++i) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = 32,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eTransferDst,
        };
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBuffer buffer{};
        if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.vol_red_alloc[i],
                            nullptr) != VK_SUCCESS) {
            return false;
        }
        g_bb.vol_red[i] = buffer;
    }
    return true;
}

// The volumetric pass over the render-size scene colour (bb_vol.comp), before the upscaler reads
// it. Returns false, recording nothing, when the pass is not available this frame.
bool RecordVol(vk::CommandBuffer cmd, vk::ImageView depth_view, vk::ImageView color_view,
               vk::Extent2D render, const BB::Upscale::Plan& plan) {
    if (!InitVol() ||
        !EnsureImage(g_bb.vol, render, vk::Format::eR16G16B16A16Sfloat,
                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)) {
        return false;
    }
    const bool fresh = !g_bb.vol_red[0];
    if (!EnsureVolBuffers()) {
        return false;
    }
    Barrier(cmd, g_bb.vol, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);
    const vk::Buffer cur = g_bb.vol_red[g_bb.vol_index];
    const vk::Buffer prev = g_bb.vol_red[g_bb.vol_index ^ 1];
    const vk::MemoryBarrier2 sync{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &sync});
    if (fresh) {
        cmd.fillBuffer(prev, 0, 32, 0);
    }
    cmd.fillBuffer(cur, 0, 32, 0);
    const vk::MemoryBarrier2 fill{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &fill});

    struct {
        u32 size[4];
        float planes[4];
    } push{{render.width, render.height, plan.depth_inverted ? 1u : 0u, 0u},
           {plan.z_near, plan.z_far, kVolHazeDensity, kVolShaftGain}};
    static_assert(sizeof(push) == 32);
    const vk::DescriptorImageInfo depth_info{g_bb.nearest, depth_view,
                                             vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const vk::DescriptorImageInfo color_info{g_bb.nearest, color_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{{}, g_bb.vol.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo prev_info{prev, 0, 32};
    const vk::DescriptorBufferInfo cur_info{cur, 0, 32};
    const std::array writes{
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &depth_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &color_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &out_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &prev_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &cur_info,
        },
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.vol_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.vol_layout, 0, writes);
    cmd.pushConstants(g_bb.vol_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    Barrier(cmd, g_bb.vol, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    g_bb.vol_index ^= 1;
    if (++g_bb.vol_passes % 600 == 0) {
        LOG_INFO(Render_Vulkan, "[BB_VOL] {} volumetric passes recorded", g_bb.vol_passes);
    }
    return true;
}

bool InitReflect() {
    if (g_bb.refl_init || g_bb.refl_failed) {
        return g_bb.refl_init;
    }
    g_bb.refl_failed = true;
    const auto device = g_bb.device;
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    if (set_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.refl_set_layout = set_layout;
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 128,
    };
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &g_bb.refl_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    if (layout_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.refl_layout = layout;
    const auto module = CompileSPV(BB_RTREFL_COMP, device);
    if (!module) {
        return false;
    }
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    vk::PipelineShaderStageCreateInfo{
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = g_bb.refl_layout,
            });
    device.destroyShaderModule(module);
    if (pipe_result != vk::Result::eSuccess) {
        return false;
    }
    g_bb.refl_pipeline = pipeline;
    g_bb.refl_failed = false;
    g_bb.refl_init = true;
    BB::Upscale::Checkpoint("rt-reflect: pass mode 2 created (bb_rtrefl.comp: screen-space "
                            "reflections on the scene colour before the upscaler, 48 steps, F0 0.04)");
    return true;
}

// BB_RT_REFLECT=2: the scene colour (linear, render size) with the reflections added, in the
// reflection pass's own image; the depth and camera are the upscaler's.
bool RecordReflect(vk::CommandBuffer cmd, vk::ImageView color_view, vk::ImageView depth_view,
                   vk::Extent2D render) {
    if (!InitReflect() ||
        !EnsureImage(g_bb.refl_out, render, vk::Format::eR16G16B16A16Sfloat,
                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)) {
        return false;
    }
    Barrier(cmd, g_bb.refl_out, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);
    struct {
        float rows[16];
        float inv_rows[16];
    } push{};
    static_assert(sizeof(push) == 128);
    const auto& plan = BB::Upscale::CurrentPlan();
    std::copy(std::begin(plan.world_to_pix), std::end(plan.world_to_pix), push.rows);
    std::copy(std::begin(plan.pix_to_world), std::end(plan.pix_to_world), push.inv_rows);
    const vk::DescriptorImageInfo depth_info{g_bb.nearest, depth_view,
                                             vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const vk::DescriptorImageInfo color_info{g_bb.nearest, color_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{{}, g_bb.refl_out.view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &depth_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &color_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &out_info,
        },
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.refl_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.refl_layout, 0, writes);
    cmd.pushConstants(g_bb.refl_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                      &push);
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    Barrier(cmd, g_bb.refl_out, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    if (++g_bb.refl_frames % 600 == 1) {
        BB::Upscale::Checkpoint(fmt::format("rt-reflect: pass mode 2 ran on {} upscaled frames "
                                            "({}x{})",
                                            g_bb.refl_frames, render.width, render.height));
    }
    return true;
}

float HalfToFloat(u16 h) {
    const u32 sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        v = man != 0 ? std::numeric_limits<float>::quiet_NaN()
                     : std::numeric_limits<float>::infinity();
    } else {
        v = std::ldexp(static_cast<float>(man | 0x400), static_cast<int>(exp) - 25);
    }
    return sign ? -v : v;
}

// The output self-check: once the GPU is past the copy, the five texels (centre and the four
// quarter points) of the upscaled image: NaN / Inf, or all of them one colour, is a warning.
void PollCheck(Scheduler& scheduler) {
    if (!g_bb.check_pending || !scheduler.IsFree(g_bb.check_tick)) {
        return;
    }
    g_bb.check_pending = false;
    vmaInvalidateAllocation(g_bb.allocator, g_bb.check_allocation, 0, VK_WHOLE_SIZE);
    const auto* half = static_cast<const u16*>(g_bb.check_mapped);
    std::array<std::array<float, 4>, 5> px{};
    bool bad = false, same = true;
    std::string points;
    for (u32 i = 0; i < px.size(); ++i) {
        for (u32 c = 0; c < 4; ++c) {
            px[i][c] = HalfToFloat(half[i * 4 + c]);
            bad |= !std::isfinite(px[i][c]);
        }
        for (u32 c = 0; c < 3; ++c) {
            same &= std::fabs(px[i][c] - px[0][c]) < 1e-3f;
        }
        points += fmt::format(" ({:.3f} {:.3f} {:.3f} a{:.2f})", px[i][0], px[i][1], px[i][2],
                              px[i][3]);
    }
    if (bad) {
        BB::Upscale::Checkpoint(fmt::format(
            "WARNING: upscaled image self-check {}: NaN / Inf in the output ({}):{}",
            g_bb.check_done, g_bb.check_what, points));
    } else if (same) {
        BB::Upscale::Checkpoint(fmt::format(
            "WARNING: upscaled image self-check {}: the 5 sampled points are all one colour "
            "({}; a flat / black / broken output, unless the frame really is one colour):{}",
            g_bb.check_done, g_bb.check_what, points));
    } else {
        BB::Upscale::Checkpoint(fmt::format("upscaled image self-check {}: ok ({}):{}",
                                            g_bb.check_done, g_bb.check_what, points));
    }
}

// Records the copy of the five texels of the output (in General layout, last written by a
// compute shader) for PollCheck.
void RecordCheck(Scheduler& scheduler, vk::CommandBuffer cmd, vk::Extent2D output,
                 const std::string& what) {
    if (g_bb.check_pending || g_bb.check_done >= 3) {
        return;
    }
    // The 1st, 2nd and 120th upscale.
    if (g_bb.upscales != 1 && g_bb.upscales != 2 && g_bb.upscales != 120) {
        return;
    }
    if (!g_bb.check_buffer) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = 64,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
        };
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer buffer{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.check_allocation, &info) !=
            VK_SUCCESS) {
            g_bb.check_done = 3;
            return;
        }
        g_bb.check_buffer = buffer;
        g_bb.check_mapped = info.pMappedData;
    }
    std::array<vk::BufferImageCopy, 5> regions{};
    const std::array<std::array<u32, 2>, 5> at{{{output.width / 2, output.height / 2},
                                                {output.width / 4, output.height / 4},
                                                {output.width * 3 / 4, output.height / 4},
                                                {output.width / 4, output.height * 3 / 4},
                                                {output.width * 3 / 4, output.height * 3 / 4}}};
    for (u32 i = 0; i < regions.size(); ++i) {
        regions[i] = vk::BufferImageCopy{
            .bufferOffset = i * 8,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageOffset = {static_cast<s32>(at[i][0]), static_cast<s32>(at[i][1]), 0},
            .imageExtent = {1, 1, 1},
        };
    }
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    cmd.copyImageToBuffer(g_bb.output.image, vk::ImageLayout::eGeneral, g_bb.check_buffer, regions);
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead);
    const vk::MemoryBarrier2 host{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host});
    g_bb.check_tick = scheduler.CurrentTick();
    g_bb.check_pending = true;
    ++g_bb.check_done;
    g_bb.check_what = what;
}

// The luminance self-check (frame 120): the mean Rec.709 luminance of a 64x64 grid of the game's
// presented image (the flip as the presenter would read it natively) and of what we present in
// its place, both in the linear space the presenter's post process encodes once. More than 5 %
// apart is a colour / gamma error (double encode, missing decode), not upscaling detail.
void PollLuminance(Scheduler& scheduler) {
    if (!g_bb.lum_pending || !scheduler.IsFree(g_bb.lum_tick)) {
        return;
    }
    g_bb.lum_pending = false;
    vmaInvalidateAllocation(g_bb.allocator, g_bb.lum_allocation, 0, VK_WHOLE_SIZE);
    const auto* v = static_cast<const u32*>(g_bb.lum_mapped);
    const u32 n = v[2];
    if (n == 0) {
        BB::Upscale::Checkpoint("WARNING: luminance self-check: no grid point was sampled");
        return;
    }
    const double src = v[0] / 65536.0 / n, out = v[1] / 65536.0 / n;
    const double sat_src = v[4] / 65536.0 / n, sat_out = v[5] / 65536.0 / n;
    // Contrast (0.23): the standard deviation of the grid's luminance, game and ours. A washed-out
    // image keeps its mean but loses spread.
    const double sq_src = v[6] / 65536.0 / n, sq_out = v[7] / 65536.0 / n;
    const double sd_src = std::sqrt(std::max(0.0, sq_src - src * src));
    const double sd_out = std::sqrt(std::max(0.0, sq_out - out * out));
    const double sd_diff = sd_src > 1e-4 ? std::fabs(sd_out - sd_src) / sd_src : 0.0;
    // Channel order (0.23): mean red and blue of the game's image (as the presenter reads it) and
    // of ours. Red and blue exchanged between the two is a swapped channel order.
    const double r_src = v[8] / 65536.0 / n, r_out = v[9] / 65536.0 / n;
    const double b_src = v[10] / 65536.0 / n, b_out = v[11] / 65536.0 / n;
    const bool swapped = std::fabs(r_src - b_src) > 0.01 &&
                         std::fabs(r_out - b_src) + std::fabs(b_out - r_src) <
                             0.5 * (std::fabs(r_out - r_src) + std::fabs(b_out - b_src));
    const std::string channels = fmt::format(
        "; red game {:.4f} ours {:.4f}, blue game {:.4f} ours {:.4f}{}", r_src, r_out, b_src,
        b_out, swapped ? " - RED AND BLUE SWAPPED" : "");
    const double diff = src > 1e-4 ? std::fabs(out - src) / src : std::fabs(out - src);
    const double sat_diff =
        sat_src > 1e-3 ? std::fabs(sat_out - sat_src) / sat_src : std::fabs(sat_out - sat_src);
    const std::string text = fmt::format(
        "colour self-check (frame 120, {} grid points of 64x64): luminance game {:.4f}, ours "
        "{:.4f} ({:+.1f} %); saturation game {:.4f}, ours {:.4f} ({:+.1f} %); contrast (luminance "
        "s.d.) game {:.4f}, ours {:.4f} ({:+.1f} %){}",
        n, src, out, src > 1e-4 ? (out - src) / src * 100.0 : 0.0, sat_src, sat_out,
        sat_src > 1e-3 ? (sat_out - sat_src) / sat_src * 100.0 : 0.0, sd_src, sd_out,
        sd_src > 1e-4 ? (sd_out - sd_src) / sd_src * 100.0 : 0.0,
        g_bb.present_encoded ? " (upscaler on sRGB-encoded LDR)" : " (upscaler on linear)") +
        channels;
    if (diff > 0.05 || sat_diff > 0.08 || sd_diff > 0.08 || swapped) {
        LOG_WARNING(Render_Vulkan, "[BB_UPSCALE] WARNING: {} - colour / gamma mismatch", text);
        BB::Upscale::Checkpoint("WARNING: " + text + " - colour / gamma mismatch");
    } else {
        LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}: ok", text);
        BB::Upscale::Checkpoint(text + ": ok");
    }
}

} // namespace

bool Rasterizer::BBInject(VideoCore::ImageId color_id, const VideoCore::ImageViewInfo& view_info,
                          vk::ImageView& out_view, vk::ImageLayout& out_layout) {
    const auto& plan = BB::Upscale::CurrentPlan();
    if (!InitOnce(instance)) {
        return false;
    }
    ReadTimestamps();
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.size.width != plan.render_w || color.info.size.height != plan.render_h) {
        return false;
    }
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    // The upscaler reads the image as the game's shader does: through the game's view format
    // when that only adds / drops the sRGB decode (identity swizzle here; the game's swizzle is
    // applied on the redirected view, BBUpscaledView).
    const vk::Format in_format = instance.GetSupportedFormat(
        SampledFormat(color.info.pixel_format, view_info.format), color.format_features);
    const auto color_view =
        TransientView(scheduler, color.GetImage(), in_format, vk::ImageAspectFlagBits::eColor);
    if (!color_view ||
        !BBRecordUpscale(color.GetImage(), color_view, in_format, color.usage_flags)) {
        return false;
    }
    return BBUpscaledView(view_info, out_view, out_layout);
}

void Rasterizer::BBStampPresentEnd(vk::CommandBuffer cmd) {
    if (!g_bb.stage_active) {
        return;
    }
    Stamp(cmd, 6);
    g_bb.stage_pending[g_bb.stage_slot] = true;
    g_bb.stage_active = false;
    g_bb.stage_slot = (g_bb.stage_slot + 1) % g_bb.stage_pending.size();
}

const char* Rasterizer::BBPresentWhy() {
    return g_bb.present_why;
}

bool Rasterizer::BBPresentUpscale(VideoCore::Image& flip, bool late) {
    const auto& plan = BB::Upscale::CurrentPlan();
    g_bb.present_ready = false;
    if (!InitOnce(instance) || !InitHud()) {
        g_bb.present_why = "the upscaler or the HUD shader could not be initialised";
        return false;
    }
    ReadTimestamps();
    if (flip.info.size.width != plan.render_w || flip.info.size.height != plan.render_h) {
        g_bb.present_why = "the presented image is not the plan's render size";
        return false;
    }
    // The post-processed image before the HUD, blitted into RGBA16F (a format every back end
    // takes, whatever the presented image's format): the upscaler's input, and the HUD's reference.
    const auto features =
        instance.GetPhysicalDevice().getFormatProperties(flip.info.pixel_format).optimalTilingFeatures;
    if (!(features & vk::FormatFeatureFlagBits::eBlitSrc) ||
        !(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        LOG_WARNING(Render_Vulkan, "[BB_UPSCALE] the presented image's format {} cannot be blitted",
                    vk::to_string(flip.info.pixel_format));
        g_bb.present_why = "the presented image's format cannot be blitted";
        return false;
    }
    const vk::Extent2D extent{flip.info.size.width, flip.info.size.height};
    // The upscaler's colour space. An LDR (8-bit sRGB) flip: the back ends (FSR 4, FSR 3, XeSS,
    // DLSS, all run without their HDR flag) are given the perceptual, sRGB-encoded values they
    // are made for in LDR mode: the flip's bytes copied raw into its UNORM twin. Linear values
    // in an LDR-flagged upscale do not come back as the game's image (dark tones lifted, washed
    // out). The HUD composite decodes the result to linear for the presenter. BB_SR_LDR_LINEAR=1
    // keeps the old linear RGBA16F input (A/B). HDR / UNORM / float flips: the blit into RGBA16F
    // keeps what the flip's view reads (linear float, or already-encoded UNORM).
    static const bool ldr_linear = [] {
        const char* v = std::getenv("BB_SR_LDR_LINEAR");
        return v != nullptr && *v == '1';
    }();
    // 0.23 (colour): FSR 4 is given linear light, as deadinside28/bloodborne_pc (GPL-2.0)
    // gpu/upscale/vk_temporal_upscaler.cpp RunScaled does ("FSR 4 treats its colour as linear
    // light ... An sRGB-format frame is decoded by its view already"): the flip through its sRGB
    // view into RGBA16F, the result linear, the composite decodes nothing. Fed the sRGB-encoded
    // bytes, FSR 4 handled perceptual values as light (washed-out image, the colour self-check's
    // -6.7 % luminance of 11-02). BB_SR_LDR_ENCODED=1: the encoded input again (A/B).
    static const bool ldr_encoded = [] {
        const char* v = std::getenv("BB_SR_LDR_ENCODED");
        return v != nullptr && *v == '1';
    }();
    // (Every back end, as the in-place / native AA path, whose colours are right: linear input.)
    const bool fsr4_linear = !ldr_encoded;
    const bool encoded =
        IsSrgb(flip.info.pixel_format) && !plan.hdr && !ldr_linear && !fsr4_linear;
    // 0.23 (channel order): the linear input is the flip's raw bytes in an image of the flip's own
    // sRGB format (decoded to linear on read), not a blit. Build 71's blit into RGBA16F swapped red
    // and blue (blue HUD bar, fire and item icons; purple blood): a blit converts by the image's
    // real format, while every other read of the flip here (final_view, the 0.22 raw copy, the
    // presenter's view) reinterprets its bytes in flip.info.pixel_format order.
    const bool raw_srgb = !encoded && IsSrgb(flip.info.pixel_format) && !plan.hdr;
    // 0.24: the flip as the native presenter reads it (its view format): the game writes BGRA
    // bytes into its R8G8B8A8 flip (deadinside28 gpu/render/presenter.cpp notes the same), and the
    // presenter's B8G8R8A8 view puts red and blue back. Build 73 read the flip as R8G8B8A8 here:
    // red and blue swapped (blue fire, HUD and icons; the self-check's 'game' blue > red on a
    // brown street).
    const vk::Format read_format = PresentReadFormat(flip.info.pixel_format);
    const vk::Format in_format = encoded    ? Unorm(read_format)
                                 : raw_srgb ? read_format
                                            : vk::Format::eR16G16B16A16Sfloat;
    if (!EnsureImage(g_bb.present_in, extent, in_format,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)) {
        g_bb.present_why = "the upscaler input image could not be created";
        return false;
    }
    g_bb.present_encoded = encoded;
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    // The per-stage timestamps of this frame (read back a few frames later).
    g_bb.stage_active = g_bb.stage_queries && !g_bb.stage_pending[g_bb.stage_slot];
    if (g_bb.stage_active) {
        cmd.resetQueryPool(g_bb.stage_queries, g_bb.stage_slot * 8, 8);
        g_bb.stage_written[g_bb.stage_slot] = 0;
    }
    Stamp(cmd, 0);
    Barrier(cmd, g_bb.present_in, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const std::array<vk::Offset3D, 2> corners{
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<s32>(extent.width), static_cast<s32>(extent.height), 1}};
    const vk::ImageBlit blit{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = corners,
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = corners,
    };
    if (encoded || raw_srgb) {
        // Same size class (sRGB and its UNORM twin): the raw bytes, no decode.
        const vk::ImageCopy copy{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {extent.width, extent.height, 1},
        };
        cmd.copyImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.present_in.image,
                      vk::ImageLayout::eTransferDstOptimal, copy);
    } else {
        cmd.blitImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                      g_bb.present_in.image, vk::ImageLayout::eTransferDstOptimal, blit,
                      vk::Filter::eNearest);
    }
    Barrier(cmd, g_bb.present_in, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    Stamp(cmd, 1);
    g_bb.present_src_format = flip.info.pixel_format;
    if (!BBRecordUpscale(g_bb.present_in.image, g_bb.present_in.view, g_bb.present_in.format,
                         g_bb.present_in.usage, late)) {
        g_bb.present_why = "the upscaler dispatch was not recorded (depth / back end)";
        return false;
    }
    g_bb.present_ready = true;
    g_bb.present_why = "";
    g_bb.present_flip = flip.info.guest_address;
    return true;
}

void Rasterizer::BBSetPresentViewFormat(vk::Format view_format) {
    if (g_bb.present_view_format != view_format) {
        g_bb.present_view_format = view_format;
        BB::Upscale::Checkpoint(fmt::format("window-size path: the flip is read as the presenter "
                                            "views it: {}",
                                            vk::to_string(view_format)));
    }
}

bool Rasterizer::BBPresentComposite(vk::CommandBuffer cmd, VideoCore::Image& flip,
                                    vk::Format view_format, vk::ImageView& out_view,
                                    vk::Extent2D& out_size) {
    if (!g_bb.present_ready) {
        if (*g_bb.present_why == '\0') g_bb.present_why = "no window-size upscale was recorded this frame";
        return false;
    }
    if (g_bb.present_flip != flip.info.guest_address) {
        g_bb.present_why = "the upscale was recorded for another presented image";
        return false;
    }
    g_bb.present_ready = false;
    const vk::Extent2D render{flip.info.size.width, flip.info.size.height};
    const vk::Extent2D output = g_bb.output.extent;
    if (!g_bb.hud_init || !g_bb.output.image || !g_bb.present_in.image ||
        g_bb.present_in.extent != render) {
        g_bb.present_why = !g_bb.hud_init        ? "the HUD shader is not ready"
                           : !g_bb.output.image  ? "no window-size output image"
                           : !g_bb.present_in.image ? "no upscaler input image"
                                                    : "the upscaler input is not the presented size";
        return false;
    }
    if (!EnsureImage(g_bb.present, output, vk::Format::eR16G16B16A16Sfloat,
                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)) {
        g_bb.present_why = "the presented image was not created";
        BB::Upscale::OnPresentFailed(g_bb.present_why);
        return false;
    }
    const auto final_view = TransientView(scheduler, flip.GetImage(),
                                          PresentReadFormat(flip.info.pixel_format),
                                          vk::ImageAspectFlagBits::eColor);
    if (!final_view) {
        g_bb.present_why = "no view of the presented image";
        BB::Upscale::OnPresentFailed(g_bb.present_why);
        return false;
    }
    // The luminance self-check buffer (bound every frame, written on frame 120 only).
    if (!g_bb.lum_buffer) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = 48,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        };
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer buffer{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.lum_allocation, &info) !=
            VK_SUCCESS) {
            g_bb.present_why = "the self-check buffer was not created";
            return false;
        }
        g_bb.lum_buffer = buffer;
        g_bb.lum_mapped = info.pMappedData;
    }
    PollLuminance(scheduler);
    ++g_bb.composites;
    const bool measure = g_bb.composites == 120 && !g_bb.lum_pending;
    if (measure) {
        cmd.fillBuffer(g_bb.lum_buffer, 0, 48, 0);
        const vk::MemoryBarrier2 fill{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &fill});
    }
    Stamp(cmd, 4);
    Barrier(cmd, g_bb.present, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite);
    struct {
        u32 size[4];
    } push{{output.width, output.height,
            IsSrgb(view_format) && !IsSrgb(flip.info.pixel_format) ? 1u : 0u,
            (g_bb.present_encoded ? 1u : 0u) | (measure ? 2u : 0u) |
                (BB::Upscale::HudSharpScale(render.width, render.height, output.width,
                                            output.height)
                     ? 8u
                     : 0u)}};
    const vk::DescriptorImageInfo up_info{g_bb.linear, g_bb.output.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo final_info{g_bb.linear, final_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo bare_info{g_bb.linear, g_bb.present_in.view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{{}, g_bb.present.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo lum_info{g_bb.lum_buffer, 0, 48};
    const std::array writes{
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &up_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &final_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &bare_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &out_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &lum_info,
        },
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.hud_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.hud_layout, 0, writes);
    cmd.pushConstants(g_bb.hud_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    Barrier(cmd, g_bb.present, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    // Frame generation's HUD-less image: the same composite with the HUD-less copy in place of
    // the presented image (no HUD added), so it matches the presented frame but for its HUD.
    g_bb.fg_present_ready = false;
    if (BB::FG::WantInputs() &&
        EnsureImage(g_bb.fg_present, output, vk::Format::eR16G16B16A16Sfloat,
                    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)) {
        Barrier(cmd, g_bb.fg_present, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite);
        auto fg_push = push;
        fg_push.size[3] = (g_bb.present_encoded ? 1u : 0u) | 4u | (push.size[3] & 8u);
        const vk::DescriptorImageInfo fg_out{{}, g_bb.fg_present.view, vk::ImageLayout::eGeneral};
        auto fg_writes = writes;
        fg_writes[3].pImageInfo = &fg_out;
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.hud_layout, 0, fg_writes);
        cmd.pushConstants(g_bb.hud_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fg_push),
                          &fg_push);
        cmd.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
        Barrier(cmd, g_bb.fg_present, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
        g_bb.fg_present_ready = true;
    }
    Stamp(cmd, 5);
    if (measure) {
        const vk::MemoryBarrier2 host{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host});
        g_bb.lum_tick = scheduler.CurrentTick();
        g_bb.lum_pending = true;
    }
    static bool said = false;
    if (!said) {
        said = true;
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] presenting the upscaled image {}x{} (game output {}x{}, HUD added "
                 "on top{}{})",
                 output.width, output.height, render.width, render.height,
                 push.size[2] ? ", sRGB decoded for the presenter" : "",
                 g_bb.present_encoded ? ", upscaler ran on sRGB-encoded LDR, decoded to linear "
                                        "once here; the presenter encodes once"
                                      : "");
    }
    out_view = g_bb.present.view;
    out_size = output;
    g_bb.present_why = "";
    return true;
}

bool Rasterizer::BBRecordUpscale(vk::Image color_image, vk::ImageView color_view,
                                 vk::Format color_format, vk::ImageUsageFlags color_usage,
                                 bool late, bool mv_only) {
    const auto& plan = BB::Upscale::CurrentPlan();
    PollCheck(scheduler);
    if (!bb_gb_depth_valid) {
        return false;
    }
    const auto depth_id = texture_cache.FindImage(bb_gb_depth_desc);
    auto& depth = texture_cache.GetImage(depth_id);
    if (depth.info.guest_address != plan.depth_addr || depth.info.size.width != plan.render_w ||
        depth.info.size.height != plan.render_h || !depth.info.props.is_depth) {
        static bool said = false;
        if (!said) {
            said = true;
            LOG_WARNING(Render_Vulkan,
                        "[BB_UPSCALE] depth {:#x} {}x{} does not match the colour {}x{}: no "
                        "upscale",
                        depth.info.guest_address, depth.info.size.width, depth.info.size.height,
                        plan.render_w, plan.render_h);
        }
        return false;
    }
    const vk::Extent2D render{plan.render_w, plan.render_h};
    const vk::Extent2D output{plan.output_w, plan.output_h};
    const auto storage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    const bool mv_zero = BB::Upscale::Cfg().mv_upscale_zero;
    if (!EnsureImage(g_bb.mv, render, vk::Format::eR16G16Sfloat, storage) ||
        (mv_zero && !EnsureImage(g_bb.mv_zero, render, vk::Format::eR16G16Sfloat,
                                 storage | vk::ImageUsageFlagBits::eTransferDst)) ||
        !EnsureImage(g_bb.reactive, render, vk::Format::eR8Unorm, storage) ||
        !EnsureImage(g_bb.lum[0], render, vk::Format::eR16Sfloat, storage) ||
        !EnsureImage(g_bb.lum[1], render, vk::Format::eR16Sfloat, storage) ||
        (!mv_only && !EnsureImage(g_bb.output, output, vk::Format::eR16G16B16A16Sfloat,
                     storage | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eTransferDst))) {
        LOG_ERROR(Render_Vulkan, "[BB_UPSCALE] the upscaler's images were not created");
        return false;
    }

    scheduler.EndRendering();
    runtime.Transit(&depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();

    const auto depth_view =
        TransientView(scheduler, depth.GetImage(), depth.info.pixel_format,
                      vk::ImageAspectFlagBits::eDepth);
    if (!color_view || !depth_view) {
        return false;
    }

    const u32 slot = g_bb.query_slot;
    const bool timed = !mv_only && g_bb.queries && !g_bb.query_pending[slot];
    if (timed) {
        cmd.resetQueryPool(g_bb.queries, slot * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, g_bb.queries, slot * 2);
    }

    // Camera motion vectors and the reactive mask.
    auto& lum_cur = g_bb.lum[g_bb.lum_index];
    auto& lum_prev = g_bb.lum[g_bb.lum_index ^ 1];
    const bool lum_prev_valid = !plan.reset && lum_prev.layout == vk::ImageLayout::eGeneral;
    for (OwnImage* img : {&g_bb.mv, &g_bb.reactive, &lum_cur, &lum_prev}) {
        Barrier(cmd, *img, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);
    }
    {
        struct {
            float rows[16];
            float vp_xy[4];
            float vp_z[4];
            u32 size[4];
        } push{};
        static_assert(sizeof(push) == 112);
        for (int i = 0; i < 16; ++i) push.rows[i] = plan.clip_to_prev_clip[i];
        push.vp_xy[0] = plan.viewport.xscale;
        push.vp_xy[1] = plan.viewport.xoffset;
        push.vp_xy[2] = plan.viewport.yscale;
        push.vp_xy[3] = plan.viewport.yoffset;
        push.vp_z[0] = plan.viewport.zscale != 0.0f ? plan.viewport.zscale : 1.0f;
        push.vp_z[1] = plan.viewport.zoffset;
        push.vp_z[2] = plan.jitter[0];
        push.vp_z[3] = plan.jitter[1];
        push.size[0] = render.width;
        push.size[1] = render.height;
        push.size[2] = plan.camera_valid && plan.viewport.xscale != 0.0f &&
                               plan.viewport.yscale != 0.0f
                           ? 1u
                           : 0u;
        const vk::DescriptorImageInfo depth_info{g_bb.nearest, depth_view,
                                                 vk::ImageLayout::eDepthStencilReadOnlyOptimal};
        const vk::DescriptorImageInfo mv_info{{}, g_bb.mv.view, vk::ImageLayout::eGeneral};
        push.size[3] = lum_prev_valid ? 1u : 0u;
        const vk::DescriptorImageInfo color_info{g_bb.nearest, color_view,
                                                 vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo prev_info{{}, lum_prev.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo cur_info{{}, lum_cur.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo reactive_info{{}, g_bb.reactive.view,
                                                    vk::ImageLayout::eGeneral};
        const std::array writes{
            vk::WriteDescriptorSet{
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &color_info,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &prev_info,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 4,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &cur_info,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 5,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &reactive_info,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &depth_info,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &mv_info,
            },
        };
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.mv_pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.mv_layout, 0, writes);
        cmd.pushConstants(g_bb.mv_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                          &push);
        cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    }
    // Object motion vectors (bb_objmv.h) over the camera ones.
    BB::ObjMotion::Compose(cmd, g_bb.mv.image, g_bb.mv.view, render, plan.reset, late);
    Barrier(cmd, g_bb.mv, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    Barrier(cmd, g_bb.reactive, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    Barrier(cmd, lum_cur, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead);
    g_bb.lum_index ^= 1;
    if (mv_only) {
        // 0.39 (BB_FG_NATIVE): frame generation's inputs only; no upscaler, no output written.
        g_bb.fg_upscaled = true;
        g_bb.fg_plan = plan;
        return true;
    }
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);

    OwnImage& upscale_mv = mv_zero ? g_bb.mv_zero : g_bb.mv;
    if (mv_zero) {
        Barrier(cmd, g_bb.mv_zero, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        cmd.clearColorImage(g_bb.mv_zero.image, vk::ImageLayout::eTransferDstOptimal,
                            vk::ClearColorValue{std::array<float, 4>{}},
                            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        Barrier(cmd, g_bb.mv_zero, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    }

    // BB_RT_REFLECT=2: the reflections go into the colour the upscaler reads (linear HDR only).
    vk::Image upscale_image = color_image;
    vk::ImageView upscale_view = color_view;
    vk::Format upscale_format = color_format;
    vk::ImageUsageFlags upscale_usage = color_usage;
    // Chain: mv_zero (above), then the volumetric pass on the scene colour, then the reflection
    // pass on whatever the volumetric pass produced. Each writes its own image.
    if (VolEnabled() && RecordVol(cmd, depth_view, upscale_view, render, plan)) {
        upscale_image = g_bb.vol.image;
        upscale_view = g_bb.vol.view;
        upscale_format = g_bb.vol.format;
        upscale_usage = g_bb.vol.usage;
    }
    // BB_RT_REFLECT=2: the reflections go into the colour the upscaler reads (linear HDR only).
    if (BB::Upscale::RtReflectLevel() >= 2) {
        const bool linear_hdr = upscale_format == vk::Format::eR16G16B16A16Sfloat ||
                                upscale_format == vk::Format::eB10G11R11UfloatPack32;
        if (plan.reflect_valid && linear_hdr && !g_bb.present_encoded &&
            RecordReflect(cmd, upscale_view, depth_view, render)) {
            upscale_image = g_bb.refl_out.image;
            upscale_view = g_bb.refl_out.view;
            upscale_format = g_bb.refl_out.format;
            upscale_usage = g_bb.refl_out.usage;
        } else if (!g_bb.refl_skip_logged) {
            g_bb.refl_skip_logged = true;
            BB::Upscale::Checkpoint(fmt::format(
                "rt-reflect: pass mode 2 not run on this upscale: colour {}, camera {} (needs "
                "linear RGBA16F or B10G11R11, a valid camera and the pass created)",
                vk::to_string(color_format), plan.reflect_valid ? "valid" : "invalid"));
        }
    }

    BB::SR::Desc desc{};
    desc.cmd = cmd;
    desc.color = {upscale_image, upscale_view, upscale_format, render,
                  vk::ImageLayout::eShaderReadOnlyOptimal, upscale_usage};
    desc.depth = {depth.GetImage(), depth_view, depth.info.pixel_format, render,
                  vk::ImageLayout::eDepthStencilReadOnlyOptimal, depth.usage_flags};
    desc.motion = {upscale_mv.image,
                   upscale_mv.view,
                   upscale_mv.format,
                   render,
                   vk::ImageLayout::eShaderReadOnlyOptimal,
                   upscale_mv.usage};
    desc.reactive = {g_bb.reactive.image, g_bb.reactive.view, g_bb.reactive.format, render,
                     vk::ImageLayout::eShaderReadOnlyOptimal, g_bb.reactive.usage};
    {
        const char* r = std::getenv("BB_REACTIVE");
        desc.has_reactive = r == nullptr || r[0] != '0';
    }
    desc.output = {g_bb.output.image, g_bb.output.view, g_bb.output.format, output,
                   vk::ImageLayout::eGeneral, g_bb.output.usage};
    desc.render = render;
    desc.upscale = output;
    {
        // 0.79 [taa]: exactly the offset the scene passes were drawn with (plan.jitter), per-SDK sign.
        const float applied[2]{plan.jitter[0], plan.jitter[1]};
        float passed[2]{};
        BB::Upscale::TaaPass(applied, plan.camera_valid, passed);
        desc.jitter[0] = passed[0];
        desc.jitter[1] = passed[1];
    }
    desc.reset = plan.reset;
    desc.hdr = plan.hdr;
    // 0.23 (colour): the window-size path with linear input gets the HDR flag, as the in-place
    // (native AA) path's linear RGBA16F input does ("linear HDR, HDR flag" in the log: the runs
    // whose colours are right). LDR-flagged, the back ends apply their LDR handling to it.
    if (color_image == g_bb.present_in.image && !g_bb.present_encoded) {
        desc.hdr = true;
    }
    desc.depth_inverted = plan.depth_inverted;
    desc.delta_ms = plan.delta_ms;
    desc.fov_y = plan.fov_y;
    desc.z_near = plan.z_near;
    desc.z_far = plan.z_far;
    desc.sharpness = BB::Upscale::LiveSharpness();
    desc.quality = BB::Upscale::Cfg().quality;
    desc.camera_valid = plan.camera_valid;
    std::copy(std::begin(plan.view_to_clip), std::end(plan.view_to_clip), desc.view_to_clip);
    std::copy(std::begin(plan.clip_to_prev_clip), std::end(plan.clip_to_prev_clip),
              desc.clip_to_prev_clip);

#ifdef _WIN32
    if (BB::Upscale::Cfg().kind == BB::SR::Kind::Fsr4 &&
        std::getenv("BB_FSR4_DEPTH_INVERT") == nullptr) {
        _putenv_s("BB_FSR4_DEPTH_INVERT", plan.depth_inverted ? "1" : "0");
    }
#endif
    const bool ok = BB::Upscale::RunBackend(desc);
    if (ok && BB::Dlss5::Wanted()) {
        // 0.42 DLSS 5 (bb_dlss5.h): NVIDIA's Neural Rendering on the upscaled output, in place,
        // before frame generation / the HUD read it. BB_DLSS5=0 (default): not called.
        (void)BB::Dlss5::Apply(desc);
    }
    const bool present_path = color_image == g_bb.present_in.image;
    if (present_path) {
        Stamp(cmd, 2);
    }
    // The back ends bind their own pipelines and descriptors; the game's next draw binds its own.
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    if (ok) {
        ++g_bb.upscales;
        g_bb.fg_upscaled = true;
        g_bb.fg_plan = plan;
        // The source's alpha back into the output, non-finite colour cleared (bb_alpha.comp).
        // Not on the window-size path: the presenter reads no alpha (its view maps alpha to
        // one) and the HUD composite clears non-finite colour itself: a 4K read + write saved.
        if (!present_path && InitAlpha()) {
            Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            struct {
                u32 size[4];
            } push{{output.width, output.height, 0u, 0u}};
            const vk::DescriptorImageInfo src_info{g_bb.linear ? g_bb.linear : g_bb.nearest,
                                                   color_view,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo out_info{{}, g_bb.output.view, vk::ImageLayout::eGeneral};
            const std::array writes{
                vk::WriteDescriptorSet{
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &src_info,
                },
                vk::WriteDescriptorSet{
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &out_info,
                },
            };
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.alpha_pipeline);
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.alpha_layout, 0, writes);
            cmd.pushConstants(g_bb.alpha_layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(push), &push);
            cmd.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
            Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
        }
        const bool copy = color_image == g_bb.present_in.image;
        const vk::Format src = copy ? g_bb.present_src_format : color_format;
        const std::string formats = fmt::format(
            "{}{} ({}, {} flag) -> {}{}",
            !copy                  ? ""
            : g_bb.present_encoded       ? "raw UNORM copy of "
            : IsSrgb(g_bb.present_in.format) ? "raw sRGB copy (bytes as they are) of "
                                             : "RGBA16F copy of ",
            vk::to_string(src),
            copy && g_bb.present_encoded ? "sRGB-encoded perceptual values, as LDR upscalers take"
                                         : ColourSpace(src),
            desc.hdr ? "HDR" : "LDR", vk::to_string(g_bb.output.format),
            copy && g_bb.present_encoded ? " (sRGB-encoded, decoded once in the HUD composite)"
                                         : "");
        // The window path gets the same inputs as the in-place path: the game's depth, the
        // camera-reconstructed motion vectors at the render size (bb_mv.comp), its jitter, reset.
        if (copy && (g_bb.upscales == 1 || g_bb.upscales % 600 == 0)) {
            BB::Upscale::Checkpoint(fmt::format(
                "window path upscale {}: MV: {}, jitter {:+.4f},{:+.4f}, depth {}x{} ({:#x}), "
                "reset {}",
                g_bb.upscales, plan.camera_valid ? "camera-reconstructed" : "zero (no camera yet)",
                plan.jitter[0], plan.jitter[1], depth.info.size.width, depth.info.size.height,
                depth.info.guest_address, plan.reset ? "yes" : "no"));
        }
        if (g_bb.upscales == 1) {
            BB::Upscale::SetUpscaleFormats(formats);
        }
        RecordCheck(scheduler, cmd, output,
                    fmt::format("upscale {} {}x{} -> {}x{}, {}", g_bb.upscales, render.width,
                                render.height, output.width, output.height, formats));
    }
    if (timed) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, g_bb.queries,
                            slot * 2 + 1);
        g_bb.query_pending[slot] = true;
        g_bb.query_slot = (slot + 1) % g_bb.query_pending.size();
    }
    return ok;
}

namespace {
bool EnsureDepthCopy(OwnImage& img, vk::Extent2D extent, vk::Format format) {
    if (img.image && img.extent == extent && img.format == format) {
        return true;
    }
    if (img.image) {
        (void)g_bb.device.waitIdle();
        DestroyImage(img);
    }
    const auto usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(g_bb.allocator, &ci, &alloc, &image, &img.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    img.image = image;
    const auto [result, view] = g_bb.device.createImageView(vk::ImageViewCreateInfo{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        DestroyImage(img);
        return false;
    }
    img.view = view;
    img.format = format;
    img.extent = extent;
    img.usage = usage;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

vk::ImageAspectFlags DepthAspects(vk::Format format) {
    switch (format) {
    case vk::Format::eD32SfloatS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD16UnormS8Uint:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        return vk::ImageAspectFlagBits::eDepth;
    }
}

void DepthBarrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout,
                  vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                  vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = img.layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img.image,
        .subresourceRange = {DepthAspects(img.format), 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    img.layout = layout;
}
} // namespace

namespace {
bool HudDiag() {
    static const bool on = [] {
        const char* v = std::getenv("BB_HUD_DIAG");
        // (The launcher clears BB_* variables: bb-analyze.txt in the working folder turns it on.)
        return (v != nullptr && *v == '1') || BB::Analyze::Enabled();
    }();
    return on;
}
float SrgbDecode(u8 v) {
    const float c = v / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
} // namespace

void Rasterizer::BBHudBlendLog(VAddr target, u32 enable, u32 csrc, u32 cfunc, u32 cdst,
                               u32 separate, u32 asrc, u32 afunc, u32 adst, u32 write_mask) {
    if (!HudDiag() || g_bb.hud_flip == 0 || target != g_bb.hud_flip) {
        return;
    }
    // Frames 300, 900, 1500: every draw into the presented image after its HUD-less copy.
    const u64 f = g_bb.hud_frames;
    if (f != 300 && f != 900 && f != 1500) {
        return;
    }
    if (g_bb.hud_logged_frame != f) {
        g_bb.hud_logged_frame = f;
        g_bb.hud_draws_logged = 0;
    }
    if (g_bb.hud_draws_logged++ >= 64) {
        return;
    }
    BB::FG::Printf("[hud-blend] frame %llu draw %u: blend %u colour src %u func %u dst %u | "
                   "separate alpha %u alpha src %u func %u dst %u | write mask 0x%x "
                   "(factors: 0 zero 1 one 4 src_alpha 5 one_minus_src_alpha 6 dst_alpha; "
                   "func 0 add)",
                   static_cast<unsigned long long>(f), g_bb.hud_draws_logged, enable, csrc, cfunc,
                   cdst, separate, asrc, afunc, adst, write_mask);
}

// The HUD's alpha, measured from the HUD-less copy (alpha before the HUD) and the presented image
// (after it): with an "over" blend in alpha, A_after = a + (1 - a) A_before, so
// a = (A_after - A_before) / (1 - A_before). The colours then give the HUD colour
// U = (C_after - (1 - a) C_before) / a in linear light, which must lie in [0, 1] when that alpha
// is the one the colours were blended with. Logged for frames 300, 900, 1500.
void Rasterizer::BBHudAlphaCheck(VideoCore::Image& flip) {
    if (!HudDiag() || !g_bb.hudless.image || flip.info.guest_address != g_bb.hud_flip ||
        g_bb.hudless.extent.width != flip.info.size.width ||
        g_bb.hudless.extent.height != flip.info.size.height) {
        return;
    }
    const u64 f = g_bb.hud_frames;
    if ((f != 300 && f != 900 && f != 1500) || g_bb.diag_pending) {
        return;
    }
    const vk::Format fmt = flip.info.pixel_format;
    if (fmt != vk::Format::eR8G8B8A8Srgb && fmt != vk::Format::eB8G8R8A8Srgb &&
        fmt != vk::Format::eR8G8B8A8Unorm && fmt != vk::Format::eB8G8R8A8Unorm) {
        BB::FG::Printf("[hud-alpha] presented image format %s: not checked",
                       vk::to_string(fmt).c_str());
        return;
    }
    const bool srgb = fmt == vk::Format::eR8G8B8A8Srgb || fmt == vk::Format::eB8G8R8A8Srgb;
    const u32 w = flip.info.size.width, h = flip.info.size.height;
    const vk::DeviceSize bytes = vk::DeviceSize(w) * h * 4;
    if (!g_bb.diag_buffer) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = bytes * 2,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
        };
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer buffer{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.diag_allocation, &info) !=
            VK_SUCCESS) {
            return;
        }
        g_bb.diag_buffer = buffer;
        g_bb.diag_mapped = info.pMappedData;
    }
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.hudless, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    const vk::BufferImageCopy region{
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageExtent = {w, h, 1},
    };
    auto r2 = region;
    r2.bufferOffset = bytes;
    cmd.copyImageToBuffer(g_bb.hudless.image, vk::ImageLayout::eTransferSrcOptimal,
                          g_bb.diag_buffer, region);
    cmd.copyImageToBuffer(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.diag_buffer,
                          r2);
    Barrier(cmd, g_bb.hudless, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    const vk::MemoryBarrier2 host{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host});
    g_bb.diag_pending = true;
    const VmaAllocator allocator = g_bb.allocator;
    const VmaAllocation allocation = g_bb.diag_allocation;
    const u8* mapped = static_cast<const u8*>(g_bb.diag_mapped);
    scheduler.DeferOperation([=] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        const u8* before = mapped;
        const u8* after = mapped + bytes;
        u64 hud = 0, alpha_written = 0, alpha_full = 0, in_range = 0, opaque = 0, translucent = 0;
        double max_out = 0.0;
        u64 hist[11]{};
        for (u64 i = 0; i < u64(w) * h; ++i) {
            const u8* b = before + i * 4;
            const u8* a = after + i * 4;
            if (b[0] == a[0] && b[1] == a[1] && b[2] == a[2]) continue;
            ++hud;
            if (b[3] == 255) {
                ++alpha_full;
                continue;
            }
            if (a[3] == b[3]) continue;
            ++alpha_written;
            const float ab = b[3] / 255.0f, aa = a[3] / 255.0f;
            const float al = std::clamp((aa - ab) / (1.0f - ab), 0.0f, 1.0f);
            hist[std::min(10, int(al * 10.0f))]++;
            if (al >= 0.999f) ++opaque;
            else ++translucent;
            double worst = 0.0;
            for (int c = 0; c < 3; ++c) {
                const float cb = srgb ? SrgbDecode(b[c]) : b[c] / 255.0f;
                const float ca = srgb ? SrgbDecode(a[c]) : a[c] / 255.0f;
                const float u = al > 0.0f ? (ca - (1.0f - al) * cb) / al : 0.0f;
                const double out = u < 0.0f ? -u : (u > 1.0f ? u - 1.0f : 0.0f);
                worst = std::max(worst, out * al); // (in the composite: the error times its alpha)
            }
            max_out = std::max(max_out, worst);
            if (worst <= 2.0 / 255.0) ++in_range;
        }
        BB::FG::Printf("[hud-alpha] %ux%u: HUD pixels %llu; alpha written by the HUD %llu "
                       "(opaque %llu, translucent %llu); scene alpha already 1 there %llu; "
                       "HUD colour consistent with that alpha %llu (%.1f %%), worst %.4f; "
                       "alpha histogram %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                       w, h, (unsigned long long)hud, (unsigned long long)alpha_written,
                       (unsigned long long)opaque, (unsigned long long)translucent,
                       (unsigned long long)alpha_full, (unsigned long long)in_range,
                       alpha_written ? 100.0 * in_range / alpha_written : 0.0, max_out,
                       (unsigned long long)hist[0], (unsigned long long)hist[1],
                       (unsigned long long)hist[2], (unsigned long long)hist[3],
                       (unsigned long long)hist[4], (unsigned long long)hist[5],
                       (unsigned long long)hist[6], (unsigned long long)hist[7],
                       (unsigned long long)hist[8], (unsigned long long)hist[9],
                       (unsigned long long)hist[10]);
        g_bb.diag_pending = false;
    });
}

void Rasterizer::BBNoteSample(VAddr address, u32 width, u32 height) {
    if (g_bb.lut_disabled || !BB::FG::WantInputs()) {
        return;
    }
    if (g_bb.draw_samples.size() < 16) {
        g_bb.draw_samples.emplace_back(address, width, height);
    }
}

void Rasterizer::BBHudCapturePoint(VideoCore::ImageId target_id) {
    if (g_bb.lut_disabled || !BB::FG::WantInputs()) {
        g_bb.draw_samples.clear();
        return;
    }
    auto& target = texture_cache.GetImage(target_id);
    const VAddr addr = target.info.guest_address;
    bool same_size = false;
    for (const auto& [a, w, h] : g_bb.draw_samples) {
        if (w == target.info.size.width && h == target.info.size.height) {
            if (!same_size) g_bb.sampled_by[addr] = a;
            same_size = true;
        }
    }
    g_bb.draw_samples.clear();
    if (g_bb.sampled_by.size() > 256) g_bb.sampled_by.clear();
    // The first draw into A that samples nothing of A's size: its HUD begins (the post-processing
    // passes before it all read a full-size image).
    if (addr != g_bb.lut_a || g_bb.lut_captured || same_size || !InitOnce(instance)) {
        return;
    }
    const vk::Extent2D extent{target.info.size.width, target.info.size.height};
    if (!EnsureImage(g_bb.lut_s, extent, target.info.pixel_format,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)) {
        return;
    }
    scheduler.EndRendering();
    runtime.Transit(&target, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.lut_s, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    cmd.copyImage(target.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.lut_s.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {extent.width, extent.height, 1},
                  });
    Barrier(cmd, g_bb.lut_s, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    g_bb.lut_captured = true;
    g_bb.lut_a_id = target_id;
}

namespace {
bool Rgba8(vk::Format f) {
    return f == vk::Format::eR8G8B8A8Unorm || f == vk::Format::eR8G8B8A8Srgb;
}
} // namespace

void Rasterizer::BBHudlessLut(VideoCore::Image& flip) {
    g_bb.lut_ready = false;
    const bool captured = g_bb.lut_captured;
    g_bb.lut_captured = false;
    // A: the same-size image the presented image's last draw read (learned for the next frame).
    if (const auto it = g_bb.sampled_by.find(flip.info.guest_address); it != g_bb.sampled_by.end()) {
        if (g_bb.lut_a != it->second) {
            g_bb.lut_a = it->second;
            BB::FG::Printf("[fg-hudless] the presented image %#llx is copied from %#llx: its "
                           "HUD-less image is taken there, before the HUD",
                           static_cast<unsigned long long>(flip.info.guest_address),
                           static_cast<unsigned long long>(g_bb.lut_a));
        }
    }
    if (!captured || g_bb.lut_disabled || !g_bb.init) {
        return;
    }
    auto& a = texture_cache.GetImage(g_bb.lut_a_id);
    const vk::Extent2D extent{flip.info.size.width, flip.info.size.height};
    if (a.info.guest_address != g_bb.lut_a || a.info.size.width != extent.width ||
        a.info.size.height != extent.height || !Rgba8(a.info.pixel_format) ||
        !Rgba8(flip.info.pixel_format) || g_bb.lut_s.extent != extent) {
        static bool said = false;
        if (!said) {
            said = true;
            BB::FG::Printf("[fg-hudless] A %s %ux%u, presented %s %ux%u: not the expected 8-bit "
                           "RGBA pair, no HUD-less image",
                           vk::to_string(a.info.pixel_format).c_str(), a.info.size.width,
                           a.info.size.height, vk::to_string(flip.info.pixel_format).c_str(),
                           extent.width, extent.height);
        }
        return;
    }
    if (!g_bb.lut_pipeline && !g_bb.lut_failed_init) {
        g_bb.lut_failed_init = true;
        if (InitHud()) {
            const auto module = CompileSPV(BB_HUDLESS_LUT_COMP, g_bb.device);
            if (module) {
                auto [r, pipe] = g_bb.device.createComputePipeline(
                    {}, vk::ComputePipelineCreateInfo{
                            .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                      .module = module,
                                      .pName = "main"},
                            .layout = g_bb.hud_layout,
                        });
                g_bb.device.destroyShaderModule(module);
                if (r == vk::Result::eSuccess) {
                    g_bb.lut_pipeline = pipe;
                    g_bb.lut_failed_init = false;
                }
            }
        }
        for (u32 i = 0; i < g_bb.lut_buf.size() && g_bb.lut_pipeline; ++i) {
            const VkBufferCreateInfo ci = vk::BufferCreateInfo{
                .size = 16 + 768 * 8,
                .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
            };
            VmaAllocationCreateInfo alloc{};
            alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VkBuffer buffer{};
            VmaAllocationInfo info{};
            if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.lut_alloc[i], &info) !=
                VK_SUCCESS) {
                g_bb.lut_pipeline = vk::Pipeline{};
                g_bb.lut_failed_init = true;
                break;
            }
            g_bb.lut_buf[i] = buffer;
            g_bb.lut_mapped[i] = info.pMappedData;
        }
        if (!g_bb.lut_pipeline) {
            BB::FG::Printf("[fg-hudless] the curve pass could not be created: no HUD-less image");
            g_bb.lut_disabled = true;
            return;
        }
    }
    if (!g_bb.lut_pipeline) {
        return;
    }
    if (g_bb.lut_every == 0) {
        // BB_FG_HUDLESS_CHECK_EVERY: presents between two self-checks (default 120, minimum 8,
        // as the check slot is rewritten by the host); 0 = check only the first frame.
        const char* env = std::getenv("BB_FG_HUDLESS_CHECK_EVERY");
        long v = env ? std::strtol(env, nullptr, 10) : 120;
        g_bb.lut_every = v <= 0 ? ~0u : static_cast<u32>(std::max<long>(v, 8));
        BB::FG::Printf("[fg-hudless] self-check on a sampled frame every %u presents (was every "
                       "frame in 0.22)", g_bb.lut_every);
    }
    const bool have_table = g_bb.lut_table_slot < g_bb.lut_buf.size();
    const bool check = !have_table || (g_bb.lut_recorded % g_bb.lut_every) == 0;
    u32 slot = g_bb.lut_table_slot;
    if (check) {
        slot = g_bb.lut_slot;
        for (u32 i = 0; i < g_bb.lut_buf.size() &&
                        (g_bb.lut_pending[slot] || slot == g_bb.lut_table_slot);
             ++i) {
            slot = (slot + 1) % g_bb.lut_buf.size();
        }
        if (g_bb.lut_pending[slot] || slot == g_bb.lut_table_slot) {
            return; // (no free slot: this frame goes without)
        }
        g_bb.lut_slot = (slot + 1) % g_bb.lut_buf.size();
    } else if (!g_bb.lut_verdict_ok) {
        ++g_bb.lut_recorded;
        return; // the last check failed: the presented image is used until the next check
    }
    ++g_bb.lut_recorded;
    if (!EnsureImage(g_bb.lut_f, extent, vk::Format::eR8G8B8A8Unorm,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled) ||
        !EnsureImage(g_bb.lut_out, extent, vk::Format::eR8G8B8A8Unorm,
                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eSampled)) {
        return;
    }
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.Transit(&a, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.lut_f, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    cmd.copyImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.lut_f.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {extent.width, extent.height, 1},
                  });
    Barrier(cmd, g_bb.lut_f, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    Barrier(cmd, g_bb.lut_out, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite);
    const vk::Buffer buf = g_bb.lut_buf[slot];
    if (check) {
        std::memset(g_bb.lut_mapped[slot], 0, 16 + 768 * 8);
        vmaFlushAllocation(g_bb.allocator, g_bb.lut_alloc[slot], 0, VK_WHOLE_SIZE);
    }
    // (A raw UNORM view of A when it is the sRGB format: the bytes, as the curve sees them.)
    const vk::ImageView a_view =
        TransientView(scheduler, a.GetImage(), vk::Format::eR8G8B8A8Unorm,
                      vk::ImageAspectFlagBits::eColor);
    const vk::ImageView s_view = g_bb.lut_s.format == vk::Format::eR8G8B8A8Unorm
                                     ? g_bb.lut_s.view
                                     : vk::ImageView{};
    if (!a_view || !s_view) {
        static bool said = false;
        if (!said) {
            said = true;
            BB::FG::Printf("[fg-hudless] no UNORM view of A (%s): no HUD-less image",
                           vk::to_string(a.info.pixel_format).c_str());
        }
        return;
    }
    const vk::DescriptorImageInfo t_info{g_bb.nearest, a_view, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo f_info{g_bb.nearest, g_bb.lut_f.view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo s_info{g_bb.nearest, s_view, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo o_info{{}, g_bb.lut_out.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo b_info{buf, 0, 16 + 768 * 8};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &t_info},
        vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &f_info},
        vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &s_info},
        vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &o_info},
        vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageBuffer,
                               .pBufferInfo = &b_info},
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.lut_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.hud_layout, 0, writes);
    const vk::MemoryBarrier2 step{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
    };
    // (The table of an earlier submission is read: order it after that compute work.)
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &step});
    const u32 modes_check[4]{0, 3, 1, 2};
    const u32 modes_fast[1]{1};
    const u32* modes = check ? modes_check : modes_fast;
    const u32 mode_count = check ? 4 : 1;
    for (u32 m = 0; m < mode_count; ++m) {
        const u32 mode = modes[m];
        const u32 push[4]{extent.width, extent.height, mode, check ? 1u : 0u};
        cmd.pushConstants(g_bb.hud_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                          push);
        if (mode == 3) {
            cmd.dispatch(12, 1, 1);
        } else {
            cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        }
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &step});
    }
    const vk::MemoryBarrier2 host{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host});
    // The result in the presenter's view format (raw bytes), as the presented image is read.
    Barrier(cmd, g_bb.lut_out, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    if (!check) {
        g_bb.lut_ready = true;
        return;
    }
    g_bb.lut_pending[slot] = true;
    const VmaAllocator allocator = g_bb.allocator;
    const VmaAllocation allocation = g_bb.lut_alloc[slot];
    const u32* counters = static_cast<const u32*>(g_bb.lut_mapped[slot]);
    scheduler.DeferOperation([=] {
        vmaInvalidateAllocation(allocator, allocation, 0, 16);
        const u32 conflicts = counters[0], mismatches = counters[1], gaps = counters[2],
                  hud = counters[3];
        ++g_bb.lut_frames;
        const bool bad = conflicts != 0 || mismatches != 0;
        if (bad) {
            g_bb.lut_conflict += conflicts != 0;
            g_bb.lut_mismatch += mismatches != 0;
            ++g_bb.lut_bad_streak;
        } else {
            ++g_bb.lut_ok;
            g_bb.lut_bad_streak = 0;
        }
        if (g_bb.lut_frames % 10 == 1 || (bad && g_bb.lut_conflict + g_bb.lut_mismatch <= 8)) {
            BB::FG::Printf("[fg-hudless] frame %llu: curve conflicts %u, mismatches outside the "
                           "HUD %u, gap lookups %u, HUD pixels %u -> %s; totals: %llu good, %llu "
                           "with conflicts, %llu with mismatches",
                           static_cast<unsigned long long>(g_bb.lut_frames), conflicts, mismatches,
                           gaps, hud, bad ? "not used (the presented image given instead)" : "used",
                           static_cast<unsigned long long>(g_bb.lut_ok),
                           static_cast<unsigned long long>(g_bb.lut_conflict),
                           static_cast<unsigned long long>(g_bb.lut_mismatch));
        }
        if (g_bb.lut_bad_streak >= 5 && !g_bb.lut_disabled) {
            g_bb.lut_disabled = true;
            BB::FG::Printf("[fg-hudless] 5 sampled checks in a row failed the self-check (conflicts "
                           "%u, mismatches %u): off for this session, frame generation without "
                           "a HUD-less image on this path",
                           conflicts, mismatches);
        }
        g_bb.lut_verdict_ok = !bad;
        g_bb.lut_table_slot = slot;
        g_bb.lut_pending[slot] = false;
    });
    g_bb.lut_ready = true;
}

// 0.39 (BB_FG_NATIVE): frame generation with no upscaler. At the window-size path's point (before
// the first HUD draw, or at present when no HUD was drawn), with render size = output size: the
// presented image copied raw (it is not changed), then the motion-vector pass (camera + object
// vectors) over the G-buffer depth, as BBRecordUpscale does, but no upscaler and no output.
bool Rasterizer::BBFgNativeAnalyze(VideoCore::Image& flip, bool late) {
    const auto& plan = BB::Upscale::CurrentPlan();
    g_bb.fgn_ready = false;
    if (!InitOnce(instance)) {
        return false;
    }
    if (flip.info.size.width != plan.render_w || flip.info.size.height != plan.render_h) {
        return false;
    }
    const vk::Extent2D extent{flip.info.size.width, flip.info.size.height};
    if (!EnsureImage(g_bb.fgn_src, extent, flip.info.pixel_format,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eSampled)) {
        return false;
    }
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.fgn_src, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    cmd.copyImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.fgn_src.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {extent.width, extent.height, 1},
                  });
    Barrier(cmd, g_bb.fgn_src, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    if (!BBRecordUpscale(g_bb.fgn_src.image, g_bb.fgn_src.view, g_bb.fgn_src.format,
                         g_bb.fgn_src.usage, late, true)) {
        return false;
    }
    g_bb.fgn_ready = true;
    // 0.90: DLSS 5 with no upscaler (Reescalado off / native). The same point and inputs as frame
    // generation: the game's image at its own size (before the HUD when a HUD draw was found),
    // the G-buffer depth and the motion vectors just built. NR runs on an RGBA16F copy; the result
    // goes back into the game's image (what is presented) and into the copy frame generation
    // reads, so the generated frames are made from the same picture that is shown.
    if (BB::Dlss5::Wanted() && bb_gb_depth_valid &&
        EnsureImage(g_bb.d5_work, extent, vk::Format::eR16G16B16A16Sfloat,
                    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                        vk::ImageUsageFlagBits::eTransferSrc |
                        vk::ImageUsageFlagBits::eTransferDst)) {
        auto& depth = texture_cache.GetImage(texture_cache.FindImage(bb_gb_depth_desc));
        scheduler.EndRendering();
        runtime.Transit(&depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        const auto dcmd = scheduler.CommandBuffer();
        const auto depth_view = TransientView(scheduler, depth.GetImage(), depth.info.pixel_format,
                                              vk::ImageAspectFlagBits::eDepth);
        if (depth_view && g_bb.mv.image && g_bb.mv.extent == extent &&
            depth.info.size.width == extent.width && depth.info.size.height == extent.height) {
            const std::array<vk::Offset3D, 2> d5_corners{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<s32>(extent.width), static_cast<s32>(extent.height), 1}};
            const vk::ImageBlit whole{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .srcOffsets = d5_corners,
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstOffsets = d5_corners,
            };
            Barrier(dcmd, g_bb.fgn_src, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            Barrier(dcmd, g_bb.d5_work, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                        vk::AccessFlagBits2::eTransferRead,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
            dcmd.blitImage(g_bb.fgn_src.image, vk::ImageLayout::eTransferSrcOptimal,
                           g_bb.d5_work.image, vk::ImageLayout::eTransferDstOptimal, whole,
                           vk::Filter::eNearest);
            Barrier(dcmd, g_bb.d5_work, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            Barrier(dcmd, g_bb.mv, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
            BB::SR::Desc d{};
            d.cmd = dcmd;
            d.output = {g_bb.d5_work.image, g_bb.d5_work.view, g_bb.d5_work.format, extent,
                        vk::ImageLayout::eGeneral, g_bb.d5_work.usage};
            d.color = d.output;
            d.depth = {depth.GetImage(), depth_view, depth.info.pixel_format, extent,
                       vk::ImageLayout::eDepthStencilReadOnlyOptimal, depth.usage_flags};
            d.motion = {g_bb.mv.image, g_bb.mv.view, g_bb.mv.format, extent,
                        vk::ImageLayout::eShaderReadOnlyOptimal, g_bb.mv.usage};
            d.render = extent;
            d.upscale = extent;
            d.jitter[0] = plan.jitter[0];
            d.jitter[1] = plan.jitter[1];
            d.reset = plan.reset;
            d.depth_inverted = plan.depth_inverted;
            d.delta_ms = plan.delta_ms;
            d.fov_y = plan.fov_y;
            d.z_near = plan.z_near;
            d.z_far = plan.z_far;
            d.camera_valid = plan.camera_valid;
            std::copy(std::begin(plan.view_to_clip), std::end(plan.view_to_clip), d.view_to_clip);
            std::copy(std::begin(plan.clip_to_prev_clip), std::end(plan.clip_to_prev_clip),
                      d.clip_to_prev_clip);
            if (BB::Dlss5::Apply(d)) {
                runtime.Transit(&flip, vk::ImageLayout::eTransferDstOptimal,
                                vk::PipelineStageFlagBits2::eTransfer,
                                vk::AccessFlagBits2::eTransferWrite);
                runtime.FlushBarriers();
                const auto bcmd = scheduler.CommandBuffer();
                Barrier(bcmd, g_bb.d5_work, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
                Barrier(bcmd, g_bb.fgn_src, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
                bcmd.blitImage(g_bb.d5_work.image, vk::ImageLayout::eTransferSrcOptimal,
                               flip.GetImage(), vk::ImageLayout::eTransferDstOptimal, whole,
                               vk::Filter::eNearest);
                bcmd.blitImage(g_bb.d5_work.image, vk::ImageLayout::eTransferSrcOptimal,
                               g_bb.fgn_src.image, vk::ImageLayout::eTransferDstOptimal, whole,
                               vk::Filter::eNearest);
                Barrier(bcmd, g_bb.fgn_src, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
                static u64 n_d5 = 0;
                if (n_d5++ % 600 == 0) {
                    BB::Upscale::Checkpoint(fmt::format(
                        "[dlss5] native (no upscaler): frame {} {}x{}, {} (same image and vectors "
                        "as frame generation)",
                        n_d5, extent.width, extent.height,
                        late ? "at present (no HUD draw found)" : "before the HUD"));
                }
            } else {
                Barrier(dcmd, g_bb.fgn_src, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
            }
        }
    }
    return true;
}

bool Rasterizer::BBFrameGenCapture(u32 slot_index, BB::FG::Inputs& out,
                                   vk::ImageView& hudless_view, vk::Format view_format) {
    out = {};
    hudless_view = vk::ImageView{};
    const bool hudless_ready = g_bb.fg_present_ready;
    const bool raw_ready = g_bb.fg_raw_hudless_ready;
    const bool fgn_ready = g_bb.fgn_ready;
    g_bb.fgn_ready = false;
    g_bb.fg_present_ready = false;
    g_bb.fg_raw_hudless_ready = false;
    // 0.36: per-reason counts of flips with no frame-generation inputs, about once a second.
    static u32 n_flips = 0, n_ok = 0, n_not_up = 0, n_no_depth = 0, n_mismatch = 0, n_alloc = 0,
               n_cam = 0;
    ++n_flips;
    if (n_flips >= 60) {
        if (n_ok != n_flips) {
            BB::FG::Printf("[fg-why] last %u flips: %u with 3D inputs; without: not upscaled this "
                           "flip %u, no G-buffer depth %u, depth/MV size or address mismatch %u, "
                           "copy images %u, camera invalid %u",
                           n_flips, n_ok, n_not_up, n_no_depth, n_mismatch, n_alloc, n_cam);
        }
        n_flips = n_ok = n_not_up = n_no_depth = n_mismatch = n_alloc = n_cam = 0;
    }
    if (!g_bb.fg_upscaled || !bb_gb_depth_valid || !g_bb.init || !g_bb.mv.image) {
        ++(!g_bb.fg_upscaled ? n_not_up : n_no_depth);
        return false;
    }
    g_bb.fg_upscaled = false;
    const auto& plan = g_bb.fg_plan;
    const auto depth_id = texture_cache.FindImage(bb_gb_depth_desc);
    auto& depth = texture_cache.GetImage(depth_id);
    const vk::Extent2D render{plan.render_w, plan.render_h};
    if (depth.info.guest_address != plan.depth_addr || depth.info.size.width != render.width ||
        depth.info.size.height != render.height || !depth.info.props.is_depth ||
        g_bb.mv.extent != render) {
        ++n_mismatch;
        return false;
    }
    auto& slot = g_bb.fg_slots[slot_index % g_bb.fg_slots.size()];
    if (!EnsureDepthCopy(slot.depth, render, depth.info.pixel_format) ||
        !EnsureImage(slot.mv, render, g_bb.mv.format,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)) {
        ++n_alloc;
        return false;
    }
    scheduler.EndRendering();
    runtime.Transit(&depth, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    DepthBarrier(cmd, slot.depth, vk::ImageLayout::eTransferDstOptimal,
                 vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                 vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    Barrier(cmd, slot.mv, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    Barrier(cmd, g_bb.mv, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    cmd.copyImage(depth.GetImage(), vk::ImageLayout::eTransferSrcOptimal, slot.depth.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                      .extent = {render.width, render.height, 1},
                  });
    cmd.copyImage(g_bb.mv.image, vk::ImageLayout::eTransferSrcOptimal, slot.mv.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {render.width, render.height, 1},
                  });
    Barrier(cmd, g_bb.mv, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    DepthBarrier(cmd, slot.depth, vk::ImageLayout::eShaderReadOnlyOptimal,
                 vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                 vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    Barrier(cmd, slot.mv, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    out.valid = plan.camera_valid;
    out.depth = {slot.depth.image, slot.depth.view, slot.depth.format, render, slot.depth.usage};
    out.motion = {slot.mv.image, slot.mv.view, slot.mv.format, render, slot.mv.usage};
    out.render = render;
    out.jitter[0] = plan.jitter[0];
    out.jitter[1] = plan.jitter[1];
    out.reset = plan.reset;
    out.depth_inverted = plan.depth_inverted;
    out.fov_y = plan.fov_y;
    out.z_near = plan.z_near;
    out.z_far = plan.z_far;
    out.camera_valid = plan.camera_valid;
    std::copy(std::begin(plan.view_to_clip), std::end(plan.view_to_clip), out.view_to_clip);
    std::copy(std::begin(plan.clip_to_prev_clip), std::end(plan.clip_to_prev_clip),
              out.clip_to_prev_clip);
    out.sl_frame =
        BB::Upscale::Cfg().kind == BB::SR::Kind::Dlss ? BB::SR::Dlss::LastFrameIndex() : 0;
    if (hudless_ready && g_bb.fg_present.image) {
        // The window-size path: the upscaled image with no HUD added (bb_hud.comp flag 4).
        hudless_view = g_bb.fg_present.view;
        out.hudless_source = "upscaled image before the HUD";
    } else if (fgn_ready && g_bb.fgn_src.image &&
               EnsureImage(g_bb.fg_hudless_view_copy, g_bb.fgn_src.extent, view_format,
                           vk::ImageUsageFlagBits::eTransferDst |
                               vk::ImageUsageFlagBits::eSampled)) {
        // 0.39 (BB_FG_NATIVE): no upscaler: the presented image before its HUD, its raw bytes in
        // the presenter's view format (as the presented image itself is read).
        Barrier(cmd, g_bb.fgn_src, vk::ImageLayout::eTransferSrcOptimal,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        Barrier(cmd, g_bb.fg_hudless_view_copy, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        cmd.copyImage(g_bb.fgn_src.image, vk::ImageLayout::eTransferSrcOptimal,
                      g_bb.fg_hudless_view_copy.image, vk::ImageLayout::eTransferDstOptimal,
                      vk::ImageCopy{
                          .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                          .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                          .extent = {g_bb.fgn_src.extent.width, g_bb.fgn_src.extent.height, 1},
                      });
        Barrier(cmd, g_bb.fg_hudless_view_copy, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
        hudless_view = g_bb.fg_hudless_view_copy.view;
        out.hudless_source = "presented image before its HUD (no upscaler)";
    } else if (g_bb.lut_ready &&
               EnsureImage(g_bb.fg_hudless_view_copy, g_bb.lut_out.extent, view_format,
                           vk::ImageUsageFlagBits::eTransferDst |
                               vk::ImageUsageFlagBits::eSampled)) {
        // The in-place path: A before its HUD through the game's own curve (BBHudlessLut),
        // its bytes in the presenter's view format (as the presented image is read).
        (void)raw_ready;
        Barrier(cmd, g_bb.fg_hudless_view_copy, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        cmd.copyImage(g_bb.lut_out.image, vk::ImageLayout::eTransferSrcOptimal,
                      g_bb.fg_hudless_view_copy.image, vk::ImageLayout::eTransferDstOptimal,
                      vk::ImageCopy{
                          .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                          .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                          .extent = {g_bb.lut_out.extent.width, g_bb.lut_out.extent.height, 1},
                      });
        Barrier(cmd, g_bb.fg_hudless_view_copy, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
        hudless_view = g_bb.fg_hudless_view_copy.view;
        out.hudless_source = "A before its HUD through the game's curve (self-checked)";
    }
    ++g_bb.fg_frames;
    if (hudless_view) ++g_bb.fg_hudless_frames;
    if (g_bb.fg_frames % 600 == 1) {
        BB::FG::Printf("[fg] HUD-less self-check: %llu of %llu 3D frames had an image without "
                       "the HUD (%s), camera %s",
                       static_cast<unsigned long long>(g_bb.fg_hudless_frames),
                       static_cast<unsigned long long>(g_bb.fg_frames), out.hudless_source,
                       plan.camera_valid ? "valid" : "INVALID");
    }
    ++(out.valid ? n_ok : n_cam);
    return out.valid;
}

bool Rasterizer::BBUpscaledView(const VideoCore::ImageViewInfo& view_info, vk::ImageView& out_view,
                                vk::ImageLayout& out_layout) {
    if (!g_bb.output.view) {
        return false;
    }
    out_view = g_bb.output.view;
    if (view_info.mapping != AmdGpu::IdentityMapping) {
        // The game's own swizzle (its view of the source), on the upscaled image too.
        if (const auto view = TransientView(scheduler, g_bb.output.image, g_bb.output.format,
                                            vk::ImageAspectFlagBits::eColor,
                                            LiverpoolToVK::ComponentMapping(view_info.mapping))) {
            out_view = view;
        }
    }
    out_layout = vk::ImageLayout::eGeneral;
    return true;
}

void Rasterizer::BBHudlessCopy(VideoCore::Image& flip) {
    if (!InitOnce(instance)) {
        return;
    }
    const vk::Extent2D extent{flip.info.size.width, flip.info.size.height};
    if (!EnsureImage(g_bb.hudless, extent, flip.info.pixel_format,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
                         vk::ImageUsageFlagBits::eTransferSrc)) {
        return;
    }
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.hudless, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const vk::ImageCopy copy{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {extent.width, extent.height, 1},
    };
    cmd.copyImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.hudless.image,
                  vk::ImageLayout::eTransferDstOptimal, copy);
    Barrier(cmd, g_bb.hudless, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    g_bb.fg_raw_hudless_ready = true;
    g_bb.hud_flip = flip.info.guest_address;
    ++g_bb.hud_frames;
    BB::Upscale::OnHudlessCopied();
}

void Rasterizer::BBOnDraw(u32 num_cb, u64 vs_hash, u64 ps_hash) {
    if (!BB::Upscale::Enabled() && !BB::Analyze::Enabled() && !BB::Upscale::RtReflect()) {
        return;
    }
    if (g_bb.check_pending) {
        PollCheck(scheduler);
    }
    const auto& regs = liverpool->regs;
    std::array<BB::Upscale::Target, AmdGpu::NUM_COLOR_BUFFERS> colors{};
    u32 num_colors = 0;
    u32 mrt = 0;
    for (u32 cb = 0; cb < std::min<u32>(num_cb, cb_descs.size()); ++cb) {
        const auto image_id = cb_descs[cb].image_id;
        if (!image_id) {
            continue;
        }
        const auto& image = texture_cache.GetImage(image_id);
        colors[cb] = {image.info.guest_address, image.info.size.width, image.info.size.height,
                      image.info.pixel_format};
        num_colors = cb + 1;
        ++mrt;
    }
    BB::Upscale::Target depth{};
    const bool has_depth = static_cast<bool>(db_desc.image_id);
    if (has_depth) {
        const auto& image = texture_cache.GetImage(db_desc.image_id);
        depth = {image.info.guest_address, image.info.size.width, image.info.size.height,
                 image.info.pixel_format};
        if (mrt >= 3) {
            bb_gb_depth_desc = db_desc.desc;
            bb_gb_depth_valid = true;
        }
    }
    const auto& vp = regs.viewports[0];
    const auto& vp_ctl = regs.viewport_control;
    // 0.71: in host pixels of the bound targets (a scene target rendered at the internal size has
    // the guest viewport scaled by bb_rt_scale), so the camera vectors unproject the depth the
    // scene was really drawn at.
    BB::Upscale::Viewport viewport{
        .xscale = (vp_ctl.xscale_enable ? vp.xscale : 1.0f) * bb_rt_scale_x,
        .xoffset = (vp_ctl.xoffset_enable ? vp.xoffset : 0.0f) * bb_rt_scale_x,
        .yscale = (vp_ctl.yscale_enable ? vp.yscale : 1.0f) * bb_rt_scale_y,
        .yoffset = (vp_ctl.yoffset_enable ? vp.yoffset : 0.0f) * bb_rt_scale_y,
        .zscale = vp_ctl.zscale_enable ? vp.zscale : 1.0f,
        .zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.0f,
    };
    const auto func = regs.depth_control.depth_func;
    const bool greater =
        func == AmdGpu::CompareFunc::Greater || func == AmdGpu::CompareFunc::GreaterEqual;
    BB::Upscale::OnDraw(std::span(colors.data(), num_colors), has_depth ? &depth : nullptr, greater,
                        viewport, vs_hash, ps_hash);
}


// ---------------------------------------------------------------------------------------------
// BB 0.80 native chain: the upscaled image as the source address's output-size image, the
// content check, and the BB_DUMP_FRAME frame dump.
namespace {
struct HostBuf {
    vk::Buffer buffer{};
    VmaAllocation allocation{};
    void* mapped = nullptr;
    u64 size = 0;
};
bool MakeHostBuf(HostBuf& b, u64 size) {
    const VkBufferCreateInfo ci = vk::BufferCreateInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
    };
    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &b.allocation, &info) != VK_SUCCESS) {
        return false;
    }
    b.buffer = buffer;
    b.mapped = info.pMappedData;
    b.size = size;
    return true;
}
void FreeHostBuf(HostBuf& b) {
    if (b.buffer) vmaDestroyBuffer(g_bb.allocator, b.buffer, b.allocation);
    b = {};
}

// Content check: both images downscaled to 64x36 and compared on the CPU a few frames later.
struct ContentCheck {
    OwnImage a, b; // the internal lit colour, the source's output-size image (both 64x36)
    HostBuf host;
    u64 tick = 0;
    bool pending = false;
    u32 count = 0;
} g_cc;
constexpr u32 kCcW = 64, kCcH = 36;

void PollContent(Scheduler& scheduler) {
    if (!g_cc.pending || !scheduler.IsFree(g_cc.tick)) return;
    g_cc.pending = false;
    vmaInvalidateAllocation(g_bb.allocator, g_cc.host.allocation, 0, VK_WHOLE_SIZE);
    const auto* h = static_cast<const u16*>(g_cc.host.mapped);
    const u32 n = kCcW * kCcH;
    double se = 0.0;
    for (u32 i = 0; i < n; ++i) {
        for (u32 c = 0; c < 3; ++c) {
            float x = HalfToFloat(h[i * 4 + c]), y = HalfToFloat(h[(n + i) * 4 + c]);
            if (!std::isfinite(x)) x = 0.0f;
            if (!std::isfinite(y)) y = 0.0f;
            x = std::max(x, 0.0f) / (1.0f + std::max(x, 0.0f)); // tone-mapped so HDR compares
            y = std::max(y, 0.0f) / (1.0f + std::max(y, 0.0f));
            se += double(x - y) * double(x - y);
        }
    }
    const double mse = se / double(n * 3);
    BB::Upscale::OnContentCheck(mse <= 1e-12 ? 99.0 : 10.0 * std::log10(1.0 / mse));
}

// BB_DUMP_FRAME=N: every render target of frame N, after each pass (a pass ends when the bound
// targets change), plus the upscaled output, as PNG in BB_DUMP_DIR (default bb_dump).
struct DumpEntry {
    HostBuf host;
    std::string name;
    u32 w = 0, h = 0;
    vk::Format format{};
    bool depth = false;
};
struct DumpState {
    s64 frame = -2; // -2 not read yet, -1 off
    std::string dir;
    std::vector<DumpEntry> entries;
    u64 bytes = 0;
    u32 lines = 0;
    bool done = false;
    std::vector<VideoCore::ImageId> last_targets;
    u32 pass = 0;
} g_dump;

u32 DumpBpp(vk::Format f) {
    switch (f) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR32Sfloat:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
        return 4;
    case vk::Format::eR16G16B16A16Sfloat:
        return 8;
    case vk::Format::eR8G8Unorm:
        return 2;
    case vk::Format::eR8Unorm:
        return 1;
    default:
        return 0;
    }
}

float Unpack11(u32 v, u32 mbits) {
    const u32 e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return std::ldexp(float(m), -14 - int(mbits));
    if (e == 31) return 65504.0f;
    return std::ldexp(1.0f + float(m) / float(1u << mbits), int(e) - 15);
}

u8 ToByte(float v, bool hdr) {
    if (!std::isfinite(v)) v = 0.0f;
    if (hdr) v = std::pow(std::max(v, 0.0f) / (1.0f + std::max(v, 0.0f)), 1.0f / 2.2f);
    return u8(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

void WriteDump(DumpEntry& e) {
    std::vector<u8> rgba(size_t(e.w) * e.h * 4, 255);
    const u8* src = static_cast<const u8*>(e.host.mapped);
    const size_t n = size_t(e.w) * e.h;
    for (size_t i = 0; i < n; ++i) {
        u8* o = &rgba[i * 4];
        switch (e.format) {
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Srgb:
            std::memcpy(o, src + i * 4, 4);
            break;
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Srgb:
            o[0] = src[i * 4 + 2]; o[1] = src[i * 4 + 1]; o[2] = src[i * 4]; o[3] = src[i * 4 + 3];
            break;
        case vk::Format::eA2B10G10R10UnormPack32: {
            u32 v; std::memcpy(&v, src + i * 4, 4);
            o[0] = u8((v & 1023) >> 2); o[1] = u8(((v >> 10) & 1023) >> 2);
            o[2] = u8(((v >> 20) & 1023) >> 2); o[3] = u8((v >> 30) * 85);
            break;
        }
        case vk::Format::eB10G11R11UfloatPack32: {
            u32 v; std::memcpy(&v, src + i * 4, 4);
            o[0] = ToByte(Unpack11(v & 2047, 6), true);
            o[1] = ToByte(Unpack11((v >> 11) & 2047, 6), true);
            o[2] = ToByte(Unpack11((v >> 22) & 1023, 5), true);
            break;
        }
        case vk::Format::eR16G16B16A16Sfloat: {
            const auto* hh = reinterpret_cast<const u16*>(src + i * 8);
            for (u32 c = 0; c < 3; ++c) o[c] = ToByte(HalfToFloat(hh[c]), true);
            o[3] = ToByte(HalfToFloat(hh[3]), false);
            break;
        }
        case vk::Format::eR16G16Sfloat: {
            const auto* hh = reinterpret_cast<const u16*>(src + i * 4);
            o[0] = ToByte(HalfToFloat(hh[0]) * 0.5f + 0.5f, false);
            o[1] = ToByte(HalfToFloat(hh[1]) * 0.5f + 0.5f, false);
            o[2] = 0;
            break;
        }
        case vk::Format::eR32Sfloat:
        case vk::Format::eD32Sfloat:
        case vk::Format::eD32SfloatS8Uint: {
            float d; std::memcpy(&d, src + i * 4, 4);
            o[0] = o[1] = o[2] = ToByte(d, false);
            break;
        }
        case vk::Format::eR8G8Unorm:
            o[0] = src[i * 2]; o[1] = src[i * 2 + 1]; o[2] = 0;
            break;
        case vk::Format::eR8Unorm:
            o[0] = o[1] = o[2] = src[i];
            break;
        default:
            break;
        }
    }
    const std::string path = g_dump.dir + "/" + e.name + ".png";
    stbi_write_png(path.c_str(), int(e.w), int(e.h), 4, rgba.data(), int(e.w * 4));
}

bool DumpActive() {
    if (g_dump.frame == -2) {
        const char* v = std::getenv("BB_DUMP_FRAME");
        g_dump.frame = (v != nullptr && *v != '\0') ? std::max<s64>(0, std::atoll(v)) : -1;
        const char* d = std::getenv("BB_DUMP_DIR");
        g_dump.dir = (d != nullptr && *d != '\0') ? d : "bb_dump";
        // 0.83: the launcher cannot pass variables: a trigger file bb_dump_frame.txt in the
        // working folder (or C:\EMULADOR PS4) holds the frame number (600 when empty); it is
        // renamed to bb_dump_frame.used.txt so the next run does not dump again.
        if (g_dump.frame < 0) {
            std::error_code ec;
            for (const std::filesystem::path dir :
                 {std::filesystem::current_path(ec), std::filesystem::path("C:/EMULADOR PS4")}) {
                const auto trigger = dir / "bb_dump_frame.txt";
                if (dir.empty() || !std::filesystem::exists(trigger, ec)) continue;
                s64 n = 600;
                if (FILE* fp = std::fopen(trigger.string().c_str(), "rb")) {
                    char buf[32] = {};
                    const size_t got = std::fread(buf, 1, sizeof(buf) - 1, fp);
                    std::fclose(fp);
                    buf[got] = 0;
                    const long long parsed = std::atoll(buf);
                    if (parsed > 0) n = parsed;
                }
                g_dump.frame = n;
                if (d == nullptr || *d == '\0') g_dump.dir = (dir / "bb_dump").string();
                std::filesystem::rename(trigger, dir / "bb_dump_frame.used.txt", ec);
                BB::Upscale::Checkpoint(fmt::format("BB_DUMP_FRAME: trigger file {} -> frame {}{}",
                                                    trigger.string(), n,
                                                    ec ? " (could not rename it)" : ""));
                break;
            }
        }
        if (g_dump.frame >= 0) {
            std::error_code ec;
            std::filesystem::create_directories(g_dump.dir, ec);
            BB::Upscale::Checkpoint(fmt::format(
                "BB_DUMP_FRAME={}: every render target of that frame after each pass, and the "
                "upscaled output, go to {} as PNG ({})",
                g_dump.frame, std::filesystem::absolute(g_dump.dir, ec).string(),
                ec ? "the folder could not be created" : "folder ready"));
        }
    }
    // 0.84: a frame the native chain check asks for (its first failure, chain still active)
    // replaces the requested one when that has not been dumped yet.
    if (const u64 req = BB::Upscale::DumpRequest(); req != 0 && g_dump.entries.empty() &&
        static_cast<s64>(req) != g_dump.frame) {
        g_dump.frame = static_cast<s64>(req);
        g_dump.done = false;
        g_dump.pass = 0;
        g_dump.lines = 0;
        g_dump.bytes = 0;
        g_dump.last_targets.clear();
        BB::Upscale::Checkpoint(fmt::format("BB_DUMP_FRAME: the native chain check asked for "
                                            "frame {} (dumped to {})",
                                            req, g_dump.dir));
        std::error_code ec;
        std::filesystem::create_directories(g_dump.dir, ec);
    }
    return g_dump.frame >= 0 && !g_dump.done &&
           BB::Upscale::FrameNumber() == static_cast<u64>(g_dump.frame);
}
} // namespace

bool Rasterizer::BBDumpOn() {
    return DumpActive();
}

void Rasterizer::BBDumpLog(const std::string& line) {
    if (!DumpActive() || g_dump.lines >= 4000) return;
    ++g_dump.lines;
    BB::Upscale::Checkpoint("[dump] " + line);
}

void Rasterizer::BBDumpImage(VideoCore::Image& image, const std::string& what) {
    if (!InitOnce(instance) || g_dump.entries.size() >= 160 || g_dump.bytes > (1500ull << 20)) {
        return;
    }
    const vk::Format f = image.info.pixel_format;
    const u32 bpp = DumpBpp(f);
    if (bpp == 0 || image.info.num_samples != 1) {
        BBDumpLog(fmt::format("{} {:#x} {}x{} {}: format not dumped", what, image.info.guest_address,
                              image.info.size.width, image.info.size.height, vk::to_string(f)));
        return;
    }
    DumpEntry e{};
    e.w = image.info.size.width;
    e.h = image.info.size.height;
    e.format = f;
    e.depth = image.info.props.is_depth;
    if (!MakeHostBuf(e.host, u64(e.w) * e.h * bpp)) return;
    const VAddr a = image.info.guest_address;
    u32 gw = 0, gh = 0, hw = 0, hh = 0;
    const bool chain = BB::Upscale::SceneScale(gw, gh, hw, hh);
    const char* version = image.info.bb_scaled ? "internal"
                          : (chain && e.w == gw && e.h == gh && BB::Upscale::PreferVersion(a) >= 0)
                              ? "twin"
                              : "full";
    e.name = fmt::format("{:04d}_{}_{:x}_{}x{}_{}_{}", g_dump.pass, what, a, e.w, e.h,
                         vk::to_string(f), version);
    scheduler.EndRendering();
    if (runtime.Transit(&image, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead)) {
        runtime.FlushBarriers();
    }
    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .imageSubresource = {e.depth ? vk::ImageAspectFlagBits::eDepth
                                     : vk::ImageAspectFlagBits::eColor,
                             0, 0, 1},
        .imageExtent = {e.w, e.h, 1},
    };
    scheduler.CommandBuffer().copyImageToBuffer(image.GetImage(),
                                                vk::ImageLayout::eTransferSrcOptimal,
                                                e.host.buffer, region);
    g_dump.bytes += e.host.size;
    BBDumpLog(fmt::format("pass {} {} {:#x} {}x{} {} ({}) -> {}.png", g_dump.pass, what, a, e.w,
                          e.h, vk::to_string(f), version, e.name));
    g_dump.entries.push_back(std::move(e));
}

void Rasterizer::BBDumpPass(std::span<const VideoCore::ImageId> targets, const std::string& desc) {
    if (g_dump.frame == -1 && BB::Upscale::DumpRequest() == 0) return;
    if (!DumpActive()) {
        if (g_dump.frame >= 0 && !g_dump.done && !g_dump.entries.empty() &&
            BB::Upscale::FrameNumber() > static_cast<u64>(g_dump.frame)) {
            // The frame is over: the targets of its last pass, then wait and write the files.
            g_dump.done = true;
            scheduler.Finish();
            for (auto& e : g_dump.entries) {
                vmaInvalidateAllocation(g_bb.allocator, e.host.allocation, 0, VK_WHOLE_SIZE);
                WriteDump(e);
                FreeHostBuf(e.host);
            }
            BB::Upscale::Checkpoint(fmt::format("BB_DUMP_FRAME: {} images of frame {} written to "
                                                "{} ({} passes)",
                                                g_dump.entries.size(), g_dump.frame, g_dump.dir,
                                                g_dump.pass));
            g_dump.entries.clear();
        }
        return;
    }
    std::vector<VideoCore::ImageId> now(targets.begin(), targets.end());
    if (now != g_dump.last_targets) {
        for (const auto id : g_dump.last_targets) {
            if (id) BBDumpImage(texture_cache.GetImage(id), "rt");
        }
        g_dump.last_targets = now;
        ++g_dump.pass;
    }
    BBDumpLog(desc);
}

void Rasterizer::BBNativeFillSource(VideoCore::ImageId scaled_id,
                                    VideoCore::TextureCache::ImageDesc desc) {
    PollContent(scheduler);
    auto& scaled = texture_cache.GetImage(scaled_id);
    if (!scaled.info.bb_scaled || !g_bb.output.image || scaled.info.bb_guest_w == 0) {
        return;
    }
    const VAddr addr = scaled.info.guest_address;
    const u32 gw = scaled.info.bb_guest_w, gh = scaled.info.bb_guest_h;
    desc.info.size.width = gw;
    desc.info.size.height = gh;
    desc.info.bb_scaled = false;
    desc.info.bb_guest_w = desc.info.bb_guest_h = 0;
    const auto twin_id = texture_cache.FindImage(desc);
    auto& twin = texture_cache.GetImage(twin_id);
    auto& src = texture_cache.GetImage(scaled_id); // (FindImage may have moved the slot storage)
    if (twin.info.bb_scaled || twin.info.size.width != gw || twin.info.size.height != gh ||
        g_bb.output.extent.width != gw || g_bb.output.extent.height != gh) {
        static bool said = false;
        if (!said) {
            said = true;
            BB::Upscale::Checkpoint(fmt::format(
                "WARNING: native chain: no output-size image for the source {:#x} ({}x{}, "
                "upscaled {}x{}): later reads use the redirect only",
                addr, twin.info.size.width, twin.info.size.height, g_bb.output.extent.width,
                g_bb.output.extent.height));
        }
        return;
    }
    scheduler.EndRendering();
    if (runtime.Transit(&twin, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite)) {
        runtime.FlushBarriers();
    }
    const auto cmd = scheduler.CommandBuffer();
    Barrier(cmd, g_bb.output, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);
    vk::ImageBlit full{};
    full.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    full.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    full.srcOffsets[1] = vk::Offset3D{s32(gw), s32(gh), 1};
    full.dstOffsets[1] = vk::Offset3D{s32(gw), s32(gh), 1};
    cmd.blitImage(g_bb.output.image, vk::ImageLayout::eTransferSrcOptimal, twin.GetImage(),
                  vk::ImageLayout::eTransferDstOptimal, full, vk::Filter::eNearest);
    twin.flags |= VideoCore::ImageFlagBits::GpuModified;
    twin.flags &= ~VideoCore::ImageFlagBits::Dirty;
    BB::Upscale::TwinDone(addr, false, true, true);
    static bool said = false;
    if (!said) {
        said = true;
        BB::Upscale::Checkpoint(fmt::format(
            "native chain: the upscaled image ({}x{}) is the output-size image of the lit scene "
            "colour {:#x} ({}): every pass after the upscaler reads it there",
            gw, gh, addr, vk::to_string(twin.info.pixel_format)));
    }
    if (DumpActive()) {
        BBDumpImage(src, "upscaler-input");
        BBDumpImage(twin, "upscaled");
    }
    // Content check every 120 upscales.
    if (!g_cc.pending && (g_cc.count++ % 120) == 60) {
        const auto usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
        if (EnsureImage(g_cc.a, {kCcW, kCcH}, vk::Format::eR16G16B16A16Sfloat, usage) &&
            EnsureImage(g_cc.b, {kCcW, kCcH}, vk::Format::eR16G16B16A16Sfloat, usage) &&
            (g_cc.host.buffer || MakeHostBuf(g_cc.host, u64(kCcW) * kCcH * 8 * 2))) {
            if (runtime.Transit(&src, vk::ImageLayout::eTransferSrcOptimal,
                                vk::PipelineStageFlagBits2::eBlit,
                                vk::AccessFlagBits2::eTransferRead)) {
                runtime.FlushBarriers();
            }
            if (runtime.Transit(&twin, vk::ImageLayout::eTransferSrcOptimal,
                                vk::PipelineStageFlagBits2::eBlit,
                                vk::AccessFlagBits2::eTransferRead)) {
                runtime.FlushBarriers();
            }
            const auto c2 = scheduler.CommandBuffer();
            for (OwnImage* o : {&g_cc.a, &g_cc.b}) {
                Barrier(c2, *o, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
            }
            vk::ImageBlit down{};
            down.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            down.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            down.dstOffsets[1] = vk::Offset3D{s32(kCcW), s32(kCcH), 1};
            down.srcOffsets[1] = vk::Offset3D{s32(src.info.size.width), s32(src.info.size.height), 1};
            c2.blitImage(src.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_cc.a.image,
                         vk::ImageLayout::eTransferDstOptimal, down, vk::Filter::eLinear);
            down.srcOffsets[1] = vk::Offset3D{s32(gw), s32(gh), 1};
            c2.blitImage(twin.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_cc.b.image,
                         vk::ImageLayout::eTransferDstOptimal, down, vk::Filter::eLinear);
            for (OwnImage* o : {&g_cc.a, &g_cc.b}) {
                Barrier(c2, *o, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
            }
            const vk::BufferImageCopy ra{
                .bufferOffset = 0,
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageExtent = {kCcW, kCcH, 1},
            };
            vk::BufferImageCopy rb = ra;
            rb.bufferOffset = u64(kCcW) * kCcH * 8;
            c2.copyImageToBuffer(g_cc.a.image, vk::ImageLayout::eTransferSrcOptimal,
                                 g_cc.host.buffer, ra);
            c2.copyImageToBuffer(g_cc.b.image, vk::ImageLayout::eTransferSrcOptimal,
                                 g_cc.host.buffer, rb);
            g_cc.tick = scheduler.CurrentTick();
            g_cc.pending = true;
        }
    }
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eBlit,
            vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead);
}

} // namespace Vulkan
