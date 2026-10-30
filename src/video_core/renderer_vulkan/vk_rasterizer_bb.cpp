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
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/bb/bb_analyze.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_alpha_comp.h"
#include "video_core/host_shaders/bb_hud_comp.h"
#include "video_core/host_shaders/bb_mv_comp.h"
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
    vk::Sampler nearest{};
    OwnImage mv;
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
    std::vector<std::pair<u64, vk::ImageView>> transient_views;
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
    g_bb.timestamp_period_ns =
        instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    g_bb.init = true;
    return true;
}

void ReadTimestamps() {
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
    std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = i,
            .descriptorType = i < 3 ? vk::DescriptorType::eCombinedImageSampler
                                    : vk::DescriptorType::eStorageImage,
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

const char* Rasterizer::BBPresentWhy() {
    return g_bb.present_why;
}

bool Rasterizer::BBPresentUpscale(VideoCore::Image& flip) {
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
    if (!EnsureImage(g_bb.present_in, extent, vk::Format::eR16G16B16A16Sfloat,
                     vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)) {
        g_bb.present_why = "the RGBA16F input image could not be created";
        return false;
    }
    scheduler.EndRendering();
    runtime.Transit(&flip, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
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
    cmd.blitImage(flip.GetImage(), vk::ImageLayout::eTransferSrcOptimal, g_bb.present_in.image,
                  vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
    Barrier(cmd, g_bb.present_in, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    g_bb.present_src_format = flip.info.pixel_format;
    if (!BBRecordUpscale(g_bb.present_in.image, g_bb.present_in.view, g_bb.present_in.format,
                         g_bb.present_in.usage)) {
        g_bb.present_why = "the upscaler dispatch was not recorded (depth / back end)";
        return false;
    }
    g_bb.present_ready = true;
    g_bb.present_why = "";
    g_bb.present_flip = flip.info.guest_address;
    return true;
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
    const auto final_view = TransientView(scheduler, flip.GetImage(), flip.info.pixel_format,
                                          vk::ImageAspectFlagBits::eColor);
    if (!final_view) {
        g_bb.present_why = "no view of the presented image";
        BB::Upscale::OnPresentFailed(g_bb.present_why);
        return false;
    }
    Barrier(cmd, g_bb.present, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite);
    struct {
        u32 size[4];
    } push{{output.width, output.height,
            IsSrgb(view_format) && !IsSrgb(flip.info.pixel_format) ? 1u : 0u, 0u}};
    const vk::DescriptorImageInfo up_info{g_bb.linear, g_bb.output.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo final_info{g_bb.linear, final_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo bare_info{g_bb.linear, g_bb.present_in.view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{{}, g_bb.present.view, vk::ImageLayout::eGeneral};
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
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.hud_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.hud_layout, 0, writes);
    cmd.pushConstants(g_bb.hud_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    Barrier(cmd, g_bb.present, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    static bool said = false;
    if (!said) {
        said = true;
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] presenting the upscaled image {}x{} (game output {}x{}, HUD added "
                 "on top{})",
                 output.width, output.height, render.width, render.height,
                 push.size[2] ? ", sRGB decoded for the presenter" : "");
    }
    out_view = g_bb.present.view;
    out_size = output;
    g_bb.present_why = "";
    return true;
}

bool Rasterizer::BBRecordUpscale(vk::Image color_image, vk::ImageView color_view,
                                 vk::Format color_format, vk::ImageUsageFlags color_usage) {
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
    if (!EnsureImage(g_bb.mv, render, vk::Format::eR16G16Sfloat, storage) ||
        !EnsureImage(g_bb.output, output, vk::Format::eR16G16B16A16Sfloat,
                     storage | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eTransferDst)) {
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
    const bool timed = g_bb.queries && !g_bb.query_pending[slot];
    if (timed) {
        cmd.resetQueryPool(g_bb.queries, slot * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, g_bb.queries, slot * 2);
    }

    // Camera motion vectors.
    Barrier(cmd, g_bb.mv, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite);
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
    Barrier(cmd, g_bb.mv, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);

    BB::SR::Desc desc{};
    desc.cmd = cmd;
    desc.color = {color_image, color_view, color_format, render,
                  vk::ImageLayout::eShaderReadOnlyOptimal, color_usage};
    desc.depth = {depth.GetImage(), depth_view, depth.info.pixel_format, render,
                  vk::ImageLayout::eDepthStencilReadOnlyOptimal, depth.usage_flags};
    desc.motion = {g_bb.mv.image, g_bb.mv.view, g_bb.mv.format, render,
                   vk::ImageLayout::eShaderReadOnlyOptimal, g_bb.mv.usage};
    desc.output = {g_bb.output.image, g_bb.output.view, g_bb.output.format, output,
                   vk::ImageLayout::eGeneral, g_bb.output.usage};
    desc.render = render;
    desc.upscale = output;
    desc.jitter[0] = plan.jitter[0];
    desc.jitter[1] = plan.jitter[1];
    desc.reset = plan.reset;
    desc.hdr = plan.hdr;
    desc.depth_inverted = plan.depth_inverted;
    desc.delta_ms = plan.delta_ms;
    desc.fov_y = plan.fov_y;
    desc.z_near = plan.z_near;
    desc.z_far = plan.z_far;
    desc.sharpness = BB::Upscale::Cfg().sharpness;
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
    // The back ends bind their own pipelines and descriptors; the game's next draw binds its own.
    Barrier(cmd, g_bb.output, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
    if (ok) {
        ++g_bb.upscales;
        // The source's alpha back into the output, non-finite colour cleared (bb_alpha.comp).
        if (InitAlpha()) {
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
            "{}{} ({}, {} flag) -> {}", copy ? "RGBA16F copy of " : "", vk::to_string(src),
            ColourSpace(src), plan.hdr ? "HDR" : "LDR", vk::to_string(g_bb.output.format));
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
    BB::Upscale::OnHudlessCopied();
}

void Rasterizer::BBOnDraw(u32 num_cb) {
    if (!BB::Upscale::Enabled() && !BB::Analyze::Enabled()) {
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
    BB::Upscale::Viewport viewport{
        .xscale = vp_ctl.xscale_enable ? vp.xscale : 1.0f,
        .xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.0f,
        .yscale = vp_ctl.yscale_enable ? vp.yscale : 1.0f,
        .yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.0f,
        .zscale = vp_ctl.zscale_enable ? vp.zscale : 1.0f,
        .zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.0f,
    };
    const auto func = regs.depth_control.depth_func;
    const bool greater =
        func == AmdGpu::CompareFunc::Greater || func == AmdGpu::CompareFunc::GreaterEqual;
    BB::Upscale::OnDraw(std::span(colors.data(), num_colors), has_depth ? &depth : nullptr,
                        greater, viewport);
}

} // namespace Vulkan
