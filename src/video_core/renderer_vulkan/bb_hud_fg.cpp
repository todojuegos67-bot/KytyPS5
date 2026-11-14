// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_hud_fg.h"

#include <array>
#include <span>
#include <fmt/format.h>
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_hud_mask_comp.h"
#include "video_core/host_shaders/bb_hud_over_comp.h"
#include "video_core/host_shaders/bb_hudless_fix_comp.h"
#include <cstdlib>
#include "video_core/bb/bb_objmv2.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB {

HudFg::HudFg(const Vulkan::Instance& instance_) : instance{instance_} {}

HudFg::~HudFg() {
    const auto device = instance.GetDevice();
    if (!device) return;
    (void)device.waitIdle();
    for (auto& r : ring) Destroy(r);
    Destroy(prev);
    if (mask_pipe) device.destroyPipeline(mask_pipe);
    if (over_pipe) device.destroyPipeline(over_pipe);
    if (mask_layout) device.destroyPipelineLayout(mask_layout);
    if (over_layout) device.destroyPipelineLayout(over_layout);
    if (mask_set) device.destroyDescriptorSetLayout(mask_set);
    if (over_set) device.destroyDescriptorSetLayout(over_set);
    if (sampler) device.destroySampler(sampler);
    if (fix_pipe) device.destroyPipeline(fix_pipe);
    if (fix_layout) device.destroyPipelineLayout(fix_layout);
    if (fix_set) device.destroyDescriptorSetLayout(fix_set);
    if (fix_counts) vmaDestroyBuffer(instance.GetAllocator(), fix_counts, fix_counts_alloc);
}

bool HudFg::Init() {
    if (tried) return ok;
    tried = true;
    const auto device = instance.GetDevice();
    auto [sr, s] = device.createSampler({
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    });
    if (sr != vk::Result::eSuccess) return false;
    sampler = s;
    const auto make = [&](std::span<const vk::DescriptorSetLayoutBinding> b,
                          vk::DescriptorSetLayout& set, vk::PipelineLayout& layout) {
        auto [lr, l] = device.createDescriptorSetLayout({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
            .bindingCount = static_cast<u32>(b.size()),
            .pBindings = b.data(),
        });
        if (lr != vk::Result::eSuccess) return false;
        set = l;
        const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, 16};
        auto [pr, pl] = device.createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
        if (pr != vk::Result::eSuccess) return false;
        layout = pl;
        return true;
    };
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    const std::array<vk::DescriptorSetLayoutBinding, 4> mb{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {3, vk::DescriptorType::eStorageImage, 1, C},
    }};
    const std::array<vk::DescriptorSetLayoutBinding, 2> ob{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eStorageImage, 1, C},
    }};
    if (!make(mb, mask_set, mask_layout) || !make(ob, over_set, over_layout)) {
        return false;
    }
    const auto pipeline = [&](vk::ShaderModule module, vk::PipelineLayout layout,
                              vk::Pipeline& pipe) {
        if (!module) return false;
        auto [cr, p] = device.createComputePipeline(
            {}, vk::ComputePipelineCreateInfo{
                    .stage{.stage = C, .module = module, .pName = "main"},
                    .layout = layout,
                });
        device.destroyShaderModule(module);
        if (cr != vk::Result::eSuccess) return false;
        pipe = p;
        return true;
    };
    if (!pipeline(Vulkan::CompileSPV(BB_HUD_MASK_COMP, device), mask_layout, mask_pipe) ||
        !pipeline(Vulkan::CompileSPV(BB_HUD_OVER_COMP, device), over_layout, over_pipe)) {
        return false;
    }
    ok = true;
    BB::Upscale::Checkpoint("[hud-fg] HUD separated from frame generation: UI layer per frame "
                            "(HUD-less diff + still-pixel fallback) over every generated frame "
                            "(DLSS-G: own overlay pass, FSR 3: FG swapchain UI resource)");
    return true;
}

void HudFg::Destroy(Own& img) {
    const auto device = instance.GetDevice();
    if (img.view) device.destroyImageView(img.view);
    if (img.image) vmaDestroyImage(instance.GetAllocator(), img.image, img.allocation);
    img = {};
}

bool HudFg::Ensure(Own& img, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
    if (img.image && img.extent == extent && img.format == format) return true;
    if (img.image) {
        (void)instance.GetDevice().waitIdle(); // a resize only
        Destroy(img);
    }
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &img.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    img.image = image;
    auto [vr, v] = instance.GetDevice().createImageView({
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
    img.format = format;
    img.extent = extent;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

void HudFg::Barrier(vk::CommandBuffer cmd, Own& img, vk::ImageLayout layout) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = img.layout,
        .newLayout = layout,
        .image = img.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
    img.layout = layout;
}

const BB::FG::Image* HudFg::Build(vk::CommandBuffer cmd, vk::Image frame,
                                  vk::ImageView frame_view, vk::Format format,
                                  vk::Extent2D extent, vk::ImageView hudless) {
    if (!Init()) return nullptr;
    const auto sampled = vk::ImageUsageFlagBits::eSampled;
    ui = &ring[ring_next];
    ring_next = (ring_next + 1) % Ring;
    if (!Ensure(*ui, extent, vk::Format::eR8G8B8A8Unorm,
                sampled | vk::ImageUsageFlagBits::eStorage) ||
        !Ensure(prev, extent, format, sampled | vk::ImageUsageFlagBits::eTransferDst)) {
        return nullptr;
    }
    if (prev.layout == vk::ImageLayout::eUndefined) prev_valid = false;
    // The finished frame: written by the post-process pass (GENERAL).
    const vk::MemoryBarrier2 written{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &written});
    Barrier(cmd, *ui, vk::ImageLayout::eGeneral);
    if (prev.layout != vk::ImageLayout::eShaderReadOnlyOptimal) {
        Barrier(cmd, prev, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    const bool has_h = static_cast<bool>(hudless);
    const vk::DescriptorImageInfo f_info{sampler, frame_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo h_info{sampler, has_h ? hudless : frame_view,
                                         has_h ? vk::ImageLayout::eShaderReadOnlyOptimal
                                               : vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo p_info{sampler, prev.view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo u_info{{}, ui->view, vk::ImageLayout::eGeneral};
    const auto w = [](u32 b, vk::DescriptorType t, const vk::DescriptorImageInfo* i) {
        return vk::WriteDescriptorSet{
            .dstBinding = b, .descriptorCount = 1, .descriptorType = t, .pImageInfo = i};
    };
    const std::array writes{
        w(0, vk::DescriptorType::eCombinedImageSampler, &f_info),
        w(1, vk::DescriptorType::eCombinedImageSampler, &h_info),
        w(2, vk::DescriptorType::eCombinedImageSampler, &p_info),
        w(3, vk::DescriptorType::eStorageImage, &u_info),
    };
    const u32 push[4] = {extent.width, extent.height, (has_h ? 1u : 0u) | (prev_valid ? 2u : 0u),
                         0u};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, mask_pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, mask_layout, 0, writes);
    cmd.pushConstants(mask_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
    cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    // This frame becomes the next one's "previous frame".
    Barrier(cmd, prev, vk::ImageLayout::eTransferDstOptimal);
    cmd.copyImage(frame, vk::ImageLayout::eGeneral, prev.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {extent.width, extent.height, 1},
                  });
    Barrier(cmd, prev, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, *ui, vk::ImageLayout::eShaderReadOnlyOptimal);
    prev_valid = true;
    ++sec_layer;
    if (has_h) {
        ++sec_hudless;
    } else {
        ++sec_static;
    }
    ui_out = {ui->image, ui->view, ui->format, extent,
              vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage};
    return &ui_out;
}

void HudFg::Overlay(vk::CommandBuffer cmd, vk::ImageView rgba_view, vk::Extent2D extent,
                    bool bgra) {
    if (!ok || !rgba_view || ui == nullptr || !ui->image) return;
    const vk::MemoryBarrier2 written{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &written});
    const vk::DescriptorImageInfo u_info{sampler, ui->view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo t_info{{}, rgba_view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &u_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &t_info},
    };
    const u32 push[4] = {extent.width, extent.height, bgra ? 1u : 0u, 0u};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, over_pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, over_layout, 0, writes);
    cmd.pushConstants(over_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
    cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    const vk::MemoryBarrier2 done{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &done});
    ++sec_overlaid;
}

bool HudFg::FixHudless(vk::CommandBuffer cmd, vk::ImageView frame_view, vk::ImageView raw_hud,
                       vk::ImageView raw_hudless, vk::Image target, vk::ImageView target_rgba,
                       vk::Extent2D extent, bool bgra, bool raw_swapped) {
    if (!frame_view || !raw_hud || !raw_hudless || !target || !target_rgba || extent.width == 0 ||
        extent.height == 0 || !Init()) {
        ++sec_fix_skipped;
        return false;
    }
    const auto device = instance.GetDevice();
    if (!fix_tried) {
        fix_tried = true;
        constexpr auto C = vk::ShaderStageFlagBits::eCompute;
        const std::array<vk::DescriptorSetLayoutBinding, 6> fb{{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {2, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {3, vk::DescriptorType::eStorageImage, 1, C},
            {4, vk::DescriptorType::eStorageBuffer, 1, C},
            {5, vk::DescriptorType::eCombinedImageSampler, 1, C},
        }};
        auto [lr, l] = device.createDescriptorSetLayout({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
            .bindingCount = static_cast<u32>(fb.size()),
            .pBindings = fb.data(),
        });
        if (lr == vk::Result::eSuccess) {
            fix_set = l;
            // bb-artifact-fix: the shader's push block is 32 bytes (uvec4 size + vec4 obj) and
            // 32 are pushed below; the range said 16 (invalid: obj was outside the layout).
            const vk::PushConstantRange range{C, 0, 32};
            auto [pr, pl] = device.createPipelineLayout({
                .setLayoutCount = 1,
                .pSetLayouts = &fix_set,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &range,
            });
            if (pr == vk::Result::eSuccess) fix_layout = pl;
        }
        if (fix_layout) {
            if (const auto module = Vulkan::CompileSPV(BB_HUDLESS_FIX_COMP, device)) {
                auto [cr, p] = device.createComputePipeline(
                    {}, vk::ComputePipelineCreateInfo{
                            .stage{.stage = C, .module = module, .pName = "main"},
                            .layout = fix_layout,
                        });
                device.destroyShaderModule(module);
                if (cr == vk::Result::eSuccess) fix_pipe = p;
            }
        }
        if (fix_pipe) {
            const VkBufferCreateInfo ci = vk::BufferCreateInfo{
                .size = FixSlots * 2 * sizeof(u32),
                .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                .sharingMode = vk::SharingMode::eExclusive,
            };
            VmaAllocationCreateInfo aci{};
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            VkBuffer raw{};
            VmaAllocationInfo info{};
            if (vmaCreateBuffer(instance.GetAllocator(), &ci, &aci, &raw, &fix_counts_alloc,
                                &info) == VK_SUCCESS) {
                fix_counts = raw;
                fix_counts_map = static_cast<u32*>(info.pMappedData);
                if (fix_counts_map == nullptr) {
                    vmaDestroyBuffer(instance.GetAllocator(), fix_counts, fix_counts_alloc);
                    fix_counts = vk::Buffer{};
                }
            }
        }
        fix_ok = fix_pipe && fix_counts;
        BB::Upscale::Checkpoint(
            fix_ok ? "[fsr3-hudless] HUD-less image matched to the finished frame outside the HUD "
                     "(bb_hudless_fix.comp); under the HUD: the post-processed copy when its red / "
                     "blue order was corrected (BB_HUDLESS_SWAP), else the finished frame's "
                     "colours with the HUD taken out (BB_HUDLESS_GRADE: unset auto, 1 always, 0 "
                     "never)"
                   : "[fsr3-hudless] fix pass unavailable: HUD-less image as post-processed");
    }
    if (!fix_ok) {
        ++sec_fix_skipped;
        return false;
    }
    // Counts of the slot written FixSlots - 1 frames ago (long done: frames in flight are fewer).
    const u32 slot = fix_slot;
    fix_slot = (fix_slot + 1) % FixSlots;
    if (fix_frames >= FixSlots) {
        sec_hud_px += fix_counts_map[slot * 2];
        sec_fixed_px += fix_counts_map[slot * 2 + 1];
    }
    ++fix_frames;
    sec_px += static_cast<u64>(extent.width) * extent.height;
    cmd.fillBuffer(fix_counts, slot * 2 * sizeof(u32), 2 * sizeof(u32), 0u);
    const vk::MemoryBarrier2 mem{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    const vk::ImageMemoryBarrier2 to_general{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = target,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                            .pMemoryBarriers = &mem,
                                            .imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &to_general});
    const vk::DescriptorImageInfo f_info{sampler, frame_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo a_info{sampler, raw_hud,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo b_info{sampler, raw_hudless,
                                         vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo t_info{{}, target_rgba, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo c_info{fix_counts, 0, VK_WHOLE_SIZE};
    const auto w = [](u32 b, vk::DescriptorType t, const vk::DescriptorImageInfo* i) {
        return vk::WriteDescriptorSet{
            .dstBinding = b, .descriptorCount = 1, .descriptorType = t, .pImageInfo = i};
    };
    // (11-13) BB_FSR3_PLAYER_REAL=1: the player (moving opaque pixels of attachment 7 in the
    // screen's centre) given to FSR 3 as UI: taken from the real frame, no ghost (moves at the
    // real rate). Only FSR 3 reads this image.
    static const bool player_real = [] {
        const char* v = std::getenv("BB_FSR3_PLAYER_REAL");
        return v != nullptr && v[0] == '1';
    }();
    vk::ImageView obj_view{};
    float osx = 1.0f, osy = 1.0f;
    const bool use_obj = player_real && BB::ObjMv2::ObjectTarget(obj_view, osx, osy);
    const vk::DescriptorImageInfo o_info{sampler, use_obj ? obj_view : frame_view,
                                         vk::ImageLayout::eGeneral};
    const std::array writes{
        w(5, vk::DescriptorType::eCombinedImageSampler, &o_info),
        w(0, vk::DescriptorType::eCombinedImageSampler, &f_info),
        w(1, vk::DescriptorType::eCombinedImageSampler, &a_info),
        w(2, vk::DescriptorType::eCombinedImageSampler, &b_info),
        w(3, vk::DescriptorType::eStorageImage, &t_info),
        vk::WriteDescriptorSet{.dstBinding = 4,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageBuffer,
                               .pBufferInfo = &c_info},
    };
    // bb-artifact-fix: BB_HUDLESS_GRADE: under the HUD the HUD-less image gets the finished
    // frame's colours scaled by raw without / with the HUD (bb_hudless_fix.comp) instead of the
    // post-processed copy. Unset (auto): only when the copy's red / blue order was not corrected
    // (`raw_swapped`: the copy itself then has the frame's colours, exactly); 1 always; 0 never.
    static const int grade_mode = [] {
        const char* v = std::getenv("BB_HUDLESS_GRADE");
        if (v != nullptr && v[0] == '0') return 0;
        if (v != nullptr && v[0] == '1') return 1;
        return 2;
    }();
    const bool grade_hud = grade_mode == 1 || (grade_mode == 2 && !raw_swapped);
    struct {
        u32 size[4];
        float obj[4];
    } push{{extent.width, extent.height, bgra ? 1u : 0u, slot},
           {osx, osy, use_obj ? 1.0f : 0.0f, grade_hud ? 1.0f : 0.0f}};
    static_assert(sizeof(push) == 32);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, fix_pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, fix_layout, 0, writes);
    cmd.pushConstants(fix_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = target,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
    ++sec_fix;
    return true;
}

void HudFg::LogHudlessSecond(double seconds) {
    const u32 fixed = sec_fix.exchange(0), skipped = sec_fix_skipped.exchange(0);
    const u64 hud = sec_hud_px.exchange(0), mism = sec_fixed_px.exchange(0),
              px = sec_px.exchange(0);
    if (fixed == 0 && skipped == 0) return;
    const double total = px > 0 ? static_cast<double>(px) : 1.0;
    BB::Upscale::Checkpoint(fmt::format(
        "[fsr3-hudless] 1 s: HUD-less matched in {} frames (skipped {}) | HUD pixels {:.2f}% | "
        "outside the HUD, post-processed copy != finished frame on {:.2f}% (now replaced by the "
        "frame's pixels) | {:.2f} s",
        fixed, skipped, 100.0 * static_cast<double>(hud) / total,
        100.0 * static_cast<double>(mism) / total, seconds));
}

void HudFg::NoteFrame(bool fg_frame, const char* fg_kind, bool fsr3_layer) {
    if (fg_frame) ++sec_fg;
    if (fsr3_layer) ++sec_fsr3;
    last_kind = fg_kind;
}

void HudFg::LogSecond(double seconds) {
    if (sec_fg.load() == 0 && sec_layer.load() == 0) return;
    BB::Upscale::Checkpoint(fmt::format(
        "[hud-fg] 1 s: FG {} | HUD separated in {} of {} FG frames (UI mask from HUD-less {}, "
        "still-pixel fallback {}) | DLSS-G generated frames with the HUD drawn over {} | FSR 3 "
        "UI layer given {} | {:.2f} s",
        last_kind.load(), sec_layer.exchange(0), sec_fg.exchange(0), sec_hudless.exchange(0),
        sec_static.exchange(0), sec_overlaid.exchange(0), sec_fsr3.exchange(0), seconds));
}

} // namespace BB
