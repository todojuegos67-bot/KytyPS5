// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-bbport: Supermedo's native TAA (BB_UPSCALER=taa), from his Bloodborne port
// (github.com/Supermedo/bloodborne_pc @115a0299a, GPL-2.0-or-later): TemporalUpscaler::RecordTaa
// with taa.comp / taa_sharpen.comp unchanged (bb_taa.comp / bb_taa_sharpen.comp). Two RGBA32F
// ping-pong histories (scene RGB, depth in alpha), depth rejection against the previous camera,
// colour clip, weight mix(0.98, 0.85, |mv| / 8); RCAS on the resolved scene. Recorded into the
// caller's command buffer at the upscaler's position (the HUD cut).
// BB_TAA_VARIANT: Supermedo's optional techniques (his BbToggle bits 51-55, all off by default):
// 1 tonemapped blend, 2 YCoCg clip, 4 variance clip (with 2), 8 3x3 reconstruction, 16 old rule.

#include <algorithm>
#include <array>
#include <span>
#include <vector>
#include <cstdlib>
#include <cstring>
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
Img g_hist[2], g_diag; // RGBA32F histories; the 1x1 unused diagnostic target
vk::Extent2D g_extent{};
u32 g_cur = 0;
bool g_valid = false;

u32 Variant() {
    static const u32 v = [] {
        const char* e = std::getenv("BB_TAA_VARIANT");
        return e != nullptr && *e ? static_cast<u32>(std::strtoul(e, nullptr, 0)) & 31u : 0u;
    }();
    return v;
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

bool MakePipeline(u32 i, std::span<const u32> code, u32 samplers, u32 storages, u32 push) {
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
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, push};
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
    // taa.comp: 4 samplers, 3 storage images, 64 bytes; taa_sharpen.comp: 2 storage, 4 bytes.
    g_ready = MakePipeline(0, BB_TAA_COMP, 4, 3, 64) &&
              MakePipeline(1, BB_TAA_SHARPEN_COMP, 0, 2, 4) &&
              Create(g_diag, {1, 1}, vk::Format::eR16G16B16A16Sfloat);
    LOG_INFO(Render_Vulkan, "[taa] Supermedo TAA {}: variant {}, sharpness {:.2f}",
             g_ready ? "ready" : "NOT available", Variant(), BB::Upscale::LiveSharpness());
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
            if (!Create(g_hist[i], desc.render, vk::Format::eR32G32B32A32Sfloat)) {
                g_extent = {};
                return false;
            }
        }
        g_extent = desc.render;
        reset = true;
    }
    // RecordTaa: history[next] is written, history[1 - next] read (Supermedo's taa_next).
    const u32 next = g_cur;
    Img& hr = g_hist[1 - next];
    Img& hw = g_hist[next];
    if (reset) {
        hr.layout = vk::ImageLayout::eUndefined; // undefined history is never sampled on reset
        hw.layout = vk::ImageLayout::eUndefined;
    }
    Barrier(cmd, hr.image, hr.layout, vk::ImageLayout::eGeneral);
    Barrier(cmd, hw.image, hw.layout, vk::ImageLayout::eGeneral);
    Barrier(cmd, g_diag.image, g_diag.layout, vk::ImageLayout::eGeneral);
    {
        const vk::MemoryBarrier2 inputs{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead |
                             vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                .pMemoryBarriers = &inputs});
    }
    struct Params {
        float jitter[2];
        u32 reset, pad;
        float depth[3][4]; // proj, prev_proj, previous_z
    } params{};
    static_assert(sizeof(Params) == 64);
    params.jitter[0] = desc.jitter[0];
    params.jitter[1] = desc.jitter[1];
    params.reset = reset ? 1u : 0u;
    params.pad = Variant() << 8;
    std::memcpy(params.depth[0], desc.taa_proj, sizeof(params.depth[0]));
    std::memcpy(params.depth[1], desc.taa_prev_proj, sizeof(params.depth[1]));
    std::memcpy(params.depth[2], desc.taa_prev_z, sizeof(params.depth[2]));
    {
        // Supermedo's taa_sampler: nearest, clamped to the edge.
        const vk::DescriptorImageInfo i0{g_nearest, desc.color.view, desc.color.layout};
        const vk::DescriptorImageInfo i1{g_nearest, desc.depth.view, desc.depth.layout};
        const vk::DescriptorImageInfo i2{g_nearest, desc.motion.view, desc.motion.layout};
        const vk::DescriptorImageInfo i3{g_nearest, hr.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo i4{{}, desc.output.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo i5{{}, hw.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo i6{{}, g_diag.view, vk::ImageLayout::eGeneral};
        const std::array writes{Write(0, i0, false), Write(1, i1, false), Write(2, i2, false),
                                Write(3, i3, false), Write(4, i4, true), Write(5, i5, true),
                                Write(6, i6, true)};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_pipe[0]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_layout[0], 0, writes);
        cmd.pushConstants(g_layout[0], vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
                          &params);
        cmd.dispatch((desc.render.width + 7) / 8, (desc.render.height + 7) / 8, 1);
    }
    const float strength = std::clamp(desc.sharpness, 0.0f, 2.0f);
    if (strength > 0.0f) {
        const vk::MemoryBarrier2 resolved{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                             vk::AccessFlagBits2::eShaderStorageWrite};
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                .pMemoryBarriers = &resolved});
        const vk::DescriptorImageInfo s0{{}, hw.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo s1{{}, desc.output.view, vk::ImageLayout::eGeneral};
        const std::array writes{Write(0, s0, true), Write(1, s1, true)};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_pipe[1]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_layout[1], 0, writes);
        cmd.pushConstants(g_layout[1], vk::ShaderStageFlagBits::eCompute, 0, sizeof(strength),
                          &strength);
        cmd.dispatch((desc.render.width + 7) / 8, (desc.render.height + 7) / 8, 1);
    }
    {
        const vk::MemoryBarrier2 done{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead};
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &done});
    }
    g_cur = 1 - next;
    g_valid = true;
    static bool said = false;
    if (!said) {
        said = true;
        BB::Upscale::Checkpoint(fmt::format("[taa] first Supermedo TAA frame {}x{}, jitter "
                                            "{:.3f},{:.3f}, variant {}",
                                            desc.render.width, desc.render.height,
                                            desc.jitter[0], desc.jitter[1], Variant()));
    }
    return true;
}

} // namespace BB::SR::Taa
