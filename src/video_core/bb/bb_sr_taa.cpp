// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-taa: the emulator's own TAA (BB_UPSCALER=taa), native resolution. Two compute
// dispatches recorded into the presenter's command buffer (no submit, fence or readback):
// bb_taa.comp (resolve + history) and bb_taa_sharpen.comp (RCAS-like sharpening to the output).
// History: two RGBA16F images and two R32F depth images, ping-pong, at the render size.

#include <algorithm>
#include <array>
#include <span>
#include <vector>
#include <cstdlib>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_taa_comp.h"
#include "video_core/host_shaders/bb_taa_sharpen_comp.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB::SR::Taa {
namespace {

struct Img {
    vk::Image image{};
    vk::ImageView view{};
    VmaAllocation allocation{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

vk::Device g_device{};
VmaAllocator g_alloc{};
bool g_ready = false;
vk::Sampler g_nearest{}, g_linear{};
vk::DescriptorSetLayout g_set[2]{};
vk::PipelineLayout g_layout[2]{};
vk::Pipeline g_pipe[2]{};
Img g_hist[2], g_depth[2];
vk::Extent2D g_extent{};
u32 g_cur = 0;
bool g_valid = false;

float Feedback() {
    static const float f = [] {
        const char* e = std::getenv("BB_TAA_FEEDBACK");
        return e != nullptr && *e ? std::clamp(static_cast<float>(std::atof(e)), 0.02f, 1.0f)
                                  : 0.1f;
    }();
    return f;
}

void Destroy(Img& img) {
    if (img.view) g_device.destroyImageView(img.view);
    if (img.image) vmaDestroyImage(g_alloc, img.image, img.allocation);
    img = {};
}

bool Create(Img& img, vk::Extent2D extent, vk::Format format) {
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(g_alloc, &ici, &aci, &image, &img.allocation, nullptr) != VK_SUCCESS) {
        return false;
    }
    img.image = image;
    auto [vr, v] = g_device.createImageView({
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (vr != vk::Result::eSuccess) {
        Destroy(img);
        return false;
    }
    img.view = v;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

void Barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout& layout,
             vk::ImageLayout new_layout) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
    layout = new_layout;
}

bool MakePipeline(u32 i, std::span<const u32> code, u32 samplers, u32 storages) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (u32 b = 0; b < samplers; ++b) {
        bindings.push_back({b, vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eCompute});
    }
    for (u32 b = 0; b < storages; ++b) {
        bindings.push_back({samplers + b, vk::DescriptorType::eStorageImage, 1,
                            vk::ShaderStageFlagBits::eCompute});
    }
    auto [lr, l] = g_device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (lr != vk::Result::eSuccess) return false;
    g_set[i] = l;
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, 32};
    auto [pr, pl] = g_device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &g_set[i],
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) return false;
    g_layout[i] = pl;
    const auto module = Vulkan::CompileSPV(code, g_device);
    if (!module) return false;
    const vk::ComputePipelineCreateInfo ci{
        .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
        .layout = g_layout[i],
    };
    auto [cr, p] = g_device.createComputePipeline({}, ci);
    g_device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) return false;
    g_pipe[i] = p;
    return true;
}

vk::WriteDescriptorSet Write(u32 binding, const vk::DescriptorImageInfo& info, bool storage) {
    return vk::WriteDescriptorSet{
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = storage ? vk::DescriptorType::eStorageImage
                                  : vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &info,
    };
}

} // namespace

void OnDevice(vk::Device device, VmaAllocator allocator) {
    g_device = device;
    g_alloc = allocator;
    vk::SamplerCreateInfo sci{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    };
    auto [s0, n] = device.createSampler(sci);
    sci.magFilter = sci.minFilter = vk::Filter::eLinear;
    auto [s1, l] = device.createSampler(sci);
    if (s0 != vk::Result::eSuccess || s1 != vk::Result::eSuccess) return;
    g_nearest = n;
    g_linear = l;
    g_ready = MakePipeline(0, BB_TAA_COMP, 5, 2) && MakePipeline(1, BB_TAA_SHARPEN_COMP, 1, 1);
    LOG_INFO(Render_Vulkan, "[taa] TAA nativo {}: feedback {:.2f}, sharpness {:.2f}",
             g_ready ? "ready" : "NOT available", Feedback(), BB::Upscale::LiveSharpness());
}

bool Ready() {
    return g_ready;
}

bool Evaluate(const Desc& desc) {
    if (!g_ready || desc.render != desc.upscale) return false;
    const auto cmd = desc.cmd;
    bool reset = desc.reset || !g_valid;
    if (g_extent != desc.render || !g_hist[0].image) {
        if (g_hist[0].image) {
            (void)g_device.waitIdle(); // resolution change only
            NoteBackendReset("TAA: resolution change");
        }
        for (u32 i = 0; i < 2; ++i) {
            Destroy(g_hist[i]);
            Destroy(g_depth[i]);
            if (!Create(g_hist[i], desc.render, vk::Format::eR16G16B16A16Sfloat) ||
                !Create(g_depth[i], desc.render, vk::Format::eR32Sfloat)) {
                g_extent = {};
                return false;
            }
        }
        g_extent = desc.render;
        reset = true;
    }
    const u32 prev = g_cur;
    const u32 cur = g_cur ^ 1u;
    Img& hp = g_hist[prev];
    Img& hc = g_hist[cur];
    Img& dp = g_depth[prev];
    Img& dc = g_depth[cur];
    // Undefined history on the first frame: give it a layout (reset ignores its contents).
    Barrier(cmd, hp.image, hp.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, dp.image, dp.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, hc.image, hc.layout, vk::ImageLayout::eGeneral);
    Barrier(cmd, dc.image, dc.layout, vk::ImageLayout::eGeneral);

    struct {
        u32 size[4];
        float params[4];
    } push{};
    static_assert(sizeof(push) == 32);
    push.size[0] = desc.render.width;
    push.size[1] = desc.render.height;
    push.size[2] = (reset ? 1u : 0u) | (desc.camera_valid ? 2u : 0u) |
                   (desc.depth_inverted ? 4u : 0u);
    push.params[0] = desc.jitter[0];
    push.params[1] = desc.jitter[1];
    push.params[2] = Feedback();
    // Strong clamping without the camera (no reprojection: plain history).
    push.params[3] = desc.camera_valid ? 1.25f : 0.75f;
    {
        const vk::DescriptorImageInfo i0{g_nearest, desc.color.view, desc.color.layout};
        const vk::DescriptorImageInfo i1{g_nearest, desc.depth.view, desc.depth.layout};
        const vk::DescriptorImageInfo i2{g_nearest, desc.motion.view, desc.motion.layout};
        const vk::DescriptorImageInfo i3{g_linear, hp.view, vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo i4{g_nearest, dp.view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo i5{{}, hc.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo i6{{}, dc.view, vk::ImageLayout::eGeneral};
        const std::array writes{Write(0, i0, false), Write(1, i1, false), Write(2, i2, false),
                                Write(3, i3, false), Write(4, i4, false), Write(5, i5, true),
                                Write(6, i6, true)};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_pipe[0]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_layout[0], 0, writes);
        cmd.pushConstants(g_layout[0], vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmd.dispatch((desc.render.width + 7) / 8, (desc.render.height + 7) / 8, 1);
    }
    Barrier(cmd, hc.image, hc.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, dc.image, dc.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    {
        push.params[0] = desc.sharpness;
        const vk::DescriptorImageInfo i0{g_nearest, hc.view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo i1{{}, desc.output.view, vk::ImageLayout::eGeneral};
        const std::array writes{Write(0, i0, false), Write(1, i1, true)};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_pipe[1]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_layout[1], 0, writes);
        cmd.pushConstants(g_layout[1], vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmd.dispatch((desc.render.width + 7) / 8, (desc.render.height + 7) / 8, 1);
    }
    g_cur = cur;
    g_valid = true;
    if (reset && desc.reset) {
        static u32 cuts = 0;
        if (++cuts <= 20) LOG_INFO(Render_Vulkan, "[taa] history reset (camera cut)");
    }
    static bool said = false;
    if (!said) {
        said = true;
        BB::Upscale::Checkpoint(fmt::format(
            "[taa] first TAA frame {}x{}: camera {}, jitter {:.3f},{:.3f}", desc.render.width,
            desc.render.height, desc.camera_valid ? "yes (reprojected)" : "no (plain history)",
            desc.jitter[0], desc.jitter[1]));
    }
    return true;
}

} // namespace BB::SR::Taa
