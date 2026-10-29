// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The rasterizer side of the Bloodborne upscaling (video_core/bb/bb_upscale.h): the upscaler's own
// images, the camera motion vector pass, the GPU timing, and the call into the selected back end,
// recorded in the rasterizer's command buffer at the injection point.
#include <algorithm>
#include <array>
#include <cstdlib>
#include "common/logging/log.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/bb/bb_analyze.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_mv_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
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
    vk::QueryPool queries{};
    std::array<bool, 8> query_pending{};
    u32 query_slot = 0;
    double timestamp_period_ns = 1.0;
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
    const auto module = Compile(HostShaders::BB_MV_COMP, vk::ShaderStageFlagBits::eCompute, device);
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
                            vk::ImageAspectFlags aspect) {
    const auto [result, view] = g_bb.device.createImageView(vk::ImageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        return {};
    }
    const vk::Device device = g_bb.device;
    scheduler.DeferOperation([device, view] { device.destroyImageView(view); });
    return view;
}

} // namespace

bool Rasterizer::BBInject(VideoCore::ImageId color_id, vk::ImageView& out_view,
                          vk::ImageLayout& out_layout) {
    const auto& plan = BB::Upscale::CurrentPlan();
    if (!InitOnce(instance)) {
        return false;
    }
    ReadTimestamps();
    if (!bb_gb_depth_valid) {
        return false;
    }
    const auto depth_id = texture_cache.FindImage(bb_gb_depth_desc);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& color = texture_cache.GetImage(color_id);
    if (depth.info.guest_address != plan.depth_addr || depth.info.size.width != plan.render_w ||
        depth.info.size.height != plan.render_h || !depth.info.props.is_depth ||
        color.info.size.width != plan.render_w || color.info.size.height != plan.render_h) {
        static bool said = false;
        if (!said) {
            said = true;
            LOG_WARNING(Render_Vulkan,
                        "[BB_UPSCALE] depth {:#x} {}x{} does not match the colour {}x{}: no "
                        "upscale",
                        depth.info.guest_address, depth.info.size.width, depth.info.size.height,
                        color.info.size.width, color.info.size.height);
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
    color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});
    depth.Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                  {});
    const auto cmd = scheduler.CommandBuffer();

    const auto color_view =
        TransientView(scheduler, color.GetImage(), color.info.pixel_format,
                      vk::ImageAspectFlagBits::eColor);
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
    desc.color = {color.GetImage(), color_view, color.info.pixel_format, render,
                  vk::ImageLayout::eShaderReadOnlyOptimal, color.usage_flags};
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
    if (timed) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, g_bb.queries,
                            slot * 2 + 1);
        g_bb.query_pending[slot] = true;
        g_bb.query_slot = (slot + 1) % g_bb.query_pending.size();
    }
    if (!ok) {
        return false;
    }
    out_view = g_bb.output.view;
    out_layout = vk::ImageLayout::eGeneral;
    return true;
}

bool Rasterizer::BBUpscaledView(vk::ImageView& out_view, vk::ImageLayout& out_layout) const {
    if (!g_bb.output.view) {
        return false;
    }
    out_view = g_bb.output.view;
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
    flip.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
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
    const auto& regs = liverpool->regs;
    std::array<BB::Upscale::Target, AmdGpu::NUM_COLOR_BUFFERS> colors{};
    u32 num_colors = 0;
    u32 mrt = 0;
    for (u32 cb = 0; cb < std::min<u32>(num_cb, cb_descs.size()); ++cb) {
        const auto image_id = cb_descs[cb].first;
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
    const bool has_depth = static_cast<bool>(db_desc.first);
    if (has_depth) {
        const auto& image = texture_cache.GetImage(db_desc.first);
        depth = {image.info.guest_address, image.info.size.width, image.info.size.height,
                 image.info.pixel_format};
        if (mrt >= 3) {
            bb_gb_depth_desc = db_desc.second;
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
