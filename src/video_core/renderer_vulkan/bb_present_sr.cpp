// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_present_sr.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_dlss5.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_mv_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace BB {

PresentSr::PresentSr(const Vulkan::Instance& instance_) : instance{instance_} {}

PresentSr::~PresentSr() {
    const auto device = instance.GetDevice();
    if (!device) return;
    (void)device.waitIdle();
    Destroy(color);
    Destroy(motion);
    Destroy(output);
    for (auto& fs : fg_slots) {
        Destroy(fs.depth);
        Destroy(fs.motion);
    }
    if (pipeline) device.destroyPipeline(pipeline);
    if (layout) device.destroyPipelineLayout(layout);
    if (set_layout) device.destroyDescriptorSetLayout(set_layout);
    if (sampler) device.destroySampler(sampler);
    if (queries) device.destroyQueryPool(queries);
}

bool PresentSr::Init() {
    if (init_tried) return init_ok;
    init_tried = true;
    const auto device = instance.GetDevice();
    const vk::SamplerCreateInfo sci{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    };
    auto [sr, s] = device.createSampler(sci);
    if (sr != vk::Result::eSuccess) return false;
    sampler = s;
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    }};
    auto [lr, l] = device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (lr != vk::Result::eSuccess) return false;
    set_layout = l;
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, 112};
    auto [pr, pl] = device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) return false;
    layout = pl;
    const auto module = Vulkan::CompileSPV(BB_MV_COMP, device);
    if (!module) return false;
    const vk::ComputePipelineCreateInfo ci{
        .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
        .layout = layout,
    };
    auto [cr, p] = device.createComputePipeline({}, ci);
    device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) return false;
    pipeline = p;
    auto [qr, q] = device.createQueryPool({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Slots * PerSlot,
    });
    if (qr == vk::Result::eSuccess) {
        queries = q;
        ns_per_tick = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    }
    init_ok = true;
    BB::Upscale::Checkpoint("present upscaler pass ready (motion vectors + colour copy, 1 dispatch)");
    return true;
}

void PresentSr::Destroy(OwnImage& img) {
    const auto device = instance.GetDevice();
    if (img.view) device.destroyImageView(img.view);
    if (img.image) vmaDestroyImage(instance.GetAllocator(), img.image, img.allocation);
    img = {};
}

bool PresentSr::Ensure(OwnImage& img, vk::Extent2D extent, vk::Format format) {
    if (img.image && img.extent == extent && img.format == format) return true;
    if (img.image) {
        (void)instance.GetDevice().waitIdle(); // (size change only: a mode / resolution switch)
        Destroy(img);
    }
    const auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
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
    img.usage = usage;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

void PresentSr::Barrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout new_layout) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = img.layout,
        .newLayout = new_layout,
        .image = img.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
    img.layout = new_layout;
}

void PresentSr::FrameBegin(vk::CommandBuffer cmd) {
    if (!queries) {
        if (!Init() || !queries) return;
    }
    slot = (slot + 1) % Slots;
    cmd.resetQueryPool(queries, slot * PerSlot, PerSlot);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queries, slot * PerSlot);
    slot_written[slot] = false;
    slot_sr[slot] = false;
    slot_open = true;
}

void PresentSr::FrameEnd(vk::CommandBuffer cmd) {
    if (!slot_open) return;
    if (!slot_sr[slot]) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queries, slot * PerSlot + 1);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queries, slot * PerSlot + 2);
    }
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queries, slot * PerSlot + 3);
    slot_written[slot] = true;
    slot_open = false;
    ++frames;
}

bool PresentSr::Poll(double& gpu_ms, double& sr_ms, double& mv_ms) {
    if (!queries || frames < Slots) return false;
    // The slot four frames back: long finished in a normal pipeline; not ready -> skipped.
    const u32 s = (slot + Slots - 4) % Slots;
    if (!slot_written[s]) return false;
    u64 t[PerSlot]{};
    const auto r = instance.GetDevice().getQueryPoolResults(
        queries, s * PerSlot, PerSlot, sizeof(t), t, sizeof(u64), vk::QueryResultFlagBits::e64);
    if (r != vk::Result::eSuccess) return false;
    slot_written[s] = false;
    const double k = ns_per_tick / 1e6;
    gpu_ms = double(t[3] - t[0]) * k;
    mv_ms = double(t[2] - t[1]) * k;
    sr_ms = double(t[3] - t[2]) * k;
    return true;
}

vk::ImageView PresentSr::Run(vk::CommandBuffer cmd, VideoCore::TextureCache& texture_cache,
                             Vulkan::Runtime& runtime, VideoCore::Image& flip,
                             vk::ImageView flip_view, vk::Extent2D in, vk::Extent2D out,
                             SceneDepth& scene, bool srgb_decode, BB::FG::Inputs* fg,
                             VideoCore::ImageId* depth_out) {
    // The jitter this frame was drawn with, then the frame's camera (and the next jitter).
    float jx = 0, jy = 0;
    BB::Upscale::Jitter(jx, jy);
    const BB::Upscale::Camera cam = BB::Upscale::EndFrame(in.width, in.height);
    // bb-clean-021-taa: TAA is native: its output is the render size (the presenter scales it).
    if (BB::Upscale::Cfg().kind == BB::SR::Kind::Taa) {
        out = in;
    }
    const bool depth_ok = scene.valid;
    scene.valid = false;
    if (fg != nullptr) *fg = {};
    // bb-clean-021-fg: with BB_FG on the vectors are made (and copied for FSR 3) even without an
    // upscaler; the frame is then presented as it is.
    // No upscale pass when it would not change the size (except native AA, ratio 1).
    const bool sr = BB::Upscale::BackendReady() && (in != out || BB::Upscale::Ratio() <= 1.01);
    const bool want_fg = fg != nullptr && BB::FG::WantInputs();
    if ((!sr && !want_fg) || !Init()) return {};
    if (in.width > out.width || in.height > out.height || !depth_ok) return {};
    auto depth_id = texture_cache.FindImage(scene.desc);
    if (depth_out != nullptr) *depth_out = depth_id;
    auto& depth = texture_cache.GetImage(depth_id);
    if (!depth.info.props.is_depth || depth.info.size.width != in.width ||
        depth.info.size.height != in.height || depth.info.num_samples > 1) {
        return {};
    }
    if (!Ensure(color, in, vk::Format::eR16G16B16A16Sfloat) ||
        !Ensure(motion, in, vk::Format::eR16G16Sfloat) ||
        !Ensure(output, out, vk::Format::eR16G16B16A16Sfloat)) {
        return {};
    }
    VideoCore::ImageViewInfo dv{};
    dv.format = vk::Format::eR32Sfloat;
    const vk::ImageView depth_view = *depth.FindView(dv, false).image_view;
    runtime.Transit(&depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();

    const bool timed = slot_open;
    if (timed) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queries, slot * PerSlot + 1);
    }
    Barrier(cmd, color, vk::ImageLayout::eGeneral);
    Barrier(cmd, motion, vk::ImageLayout::eGeneral);
    struct {
        float rows[16];
        float vp_xy[4];
        float vp_z[4];
        u32 size[4];
    } push{};
    static_assert(sizeof(push) == 112);
    // bb-clean-021-bbport: Supermedo's camera (bb_mv.comp): reproject rows, projections, jitter.
    std::memcpy(push.rows, cam.reproject, sizeof(cam.reproject));
    std::memcpy(push.rows + 12, cam.proj, sizeof(cam.proj));
    std::memcpy(push.vp_xy, cam.prev_proj, sizeof(cam.prev_proj));
    push.vp_z[2] = jx;
    push.vp_z[3] = jy;
    push.size[0] = in.width;
    push.size[1] = in.height;
    push.size[2] = cam.mv_valid ? 1u : 0u;
    push.size[3] = srgb_decode ? 1u : 0u; // bit 0: sRGB decode
    const vk::DescriptorImageInfo depth_info{sampler, depth_view,
                                             vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const vk::DescriptorImageInfo mv_info{{}, motion.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo src_info{sampler, flip_view,
                                           vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo col_info{{}, color.view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &depth_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &mv_info},
        vk::WriteDescriptorSet{.dstBinding = 2,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &src_info},
        vk::WriteDescriptorSet{.dstBinding = 3,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &col_info},
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
    cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    Barrier(cmd, color, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, motion, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, output, vk::ImageLayout::eGeneral);
    if (timed) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queries,
                            slot * PerSlot + 2);
        slot_sr[slot] = true;
    }

    if (want_fg && CaptureFg(cmd, runtime, depth, in, *fg)) {
        fg->jitter[0] = jx;
        fg->jitter[1] = jy;
        fg->reset = cam.reset;
        fg->depth_inverted = scene.greater;
        fg->fov_y = cam.fov_y;
        fg->z_near = cam.z_near;
        fg->z_far = cam.z_far;
        fg->camera_valid = cam.valid;
        std::memcpy(fg->view_to_clip, cam.view_to_clip, sizeof(fg->view_to_clip));
        std::memcpy(fg->clip_to_prev_clip, cam.clip_to_prev_clip, sizeof(fg->clip_to_prev_clip));
        // A 3D frame is one with the scene depth (movies and 2D flips have none); without the
        // camera the vectors are zero and FSR 3 uses its default near / far / fov.
        fg->valid = true;
    }
    if (!sr) {
        return {};
    }

    BB::SR::Desc desc{};
    desc.cmd = cmd;
    desc.color = {color.image, color.view, color.format, in,
                  vk::ImageLayout::eShaderReadOnlyOptimal, color.usage};
    desc.depth = {depth.GetImage(), depth_view, depth.info.pixel_format, in,
                  vk::ImageLayout::eDepthStencilReadOnlyOptimal, depth.usage_flags};
    desc.motion = {motion.image, motion.view, motion.format, in,
                   vk::ImageLayout::eShaderReadOnlyOptimal, motion.usage};
    desc.output = {output.image, output.view, output.format, out, vk::ImageLayout::eGeneral,
                   output.usage};
    desc.has_reactive = false;
    desc.render = in;
    desc.upscale = out;
    desc.jitter[0] = jx;
    desc.jitter[1] = jy;
    desc.reset = cam.reset || in != last_in || out != last_out;
    // The scene colour is the game's tonemapped LDR image (as in Supermedo's RunScaled, hdr = false): telling
    // DLSS / FSR it is HDR turned on their auto exposure and HDR handling, which shifted the colours (dull, yellow,
    // blue shadows) and softened the picture.
    desc.hdr = false;
    desc.depth_inverted = scene.greater;
    desc.delta_ms = 16.7f;
    desc.fov_y = cam.fov_y;
    desc.z_near = cam.z_near;
    desc.z_far = cam.z_far;
    desc.sharpness = BB::Upscale::LiveSharpness();
    desc.quality = BB::Upscale::Cfg().quality;
    desc.camera_valid = cam.valid;
    std::memcpy(desc.view_to_clip, cam.view_to_clip, sizeof(desc.view_to_clip));
    std::memcpy(desc.clip_to_prev_clip, cam.clip_to_prev_clip, sizeof(desc.clip_to_prev_clip));
    std::memcpy(desc.taa_proj, cam.proj, sizeof(desc.taa_proj));
    std::memcpy(desc.taa_prev_proj, cam.prev_proj, sizeof(desc.taa_prev_proj));
    std::memcpy(desc.taa_prev_z, cam.reproject + 8, sizeof(desc.taa_prev_z));
    // No previous camera: Supermedo's TAA drops its history (the other upscalers as before).
    if (!cam.mv_valid && BB::Upscale::Cfg().kind == BB::SR::Kind::Taa) desc.reset = true;
#ifdef _WIN32
    if (BB::Upscale::Cfg().kind == BB::SR::Kind::Fsr4 &&
        std::getenv("BB_FSR4_DEPTH_INVERT") == nullptr) {
        _putenv_s("BB_FSR4_DEPTH_INVERT", scene.greater ? "1" : "0");
    }
#endif
    last_in = in;
    last_out = out;
    const bool ok = BB::Upscale::RunBackend(desc);
    if (!ok) {
        return {};
    }
    if (BB::Dlss5::Wanted()) {
        // DLSS 5 (BB_DLSS5 1..3, from 1.59): Neural Rendering on the upscaled output, in place.
        (void)BB::Dlss5::Apply(desc);
    }
    Barrier(cmd, output, vk::ImageLayout::eShaderReadOnlyOptimal);
    static bool said = false;
    if (!said) {
        said = true;
        BB::Upscale::Checkpoint(fmt::format("first upscaled frame: {}x{} -> {}x{} ({} {})",
                                            in.width, in.height, out.width, out.height,
                                            BB::SR::KindName(BB::Upscale::Cfg().kind),
                                            BB::SR::QualityName(BB::Upscale::Cfg().quality)));
    }
    return output.view;
}

bool PresentSr::EnsureDepth(OwnImage& img, vk::Extent2D extent, vk::Format format) {
    if (img.image && img.extent == extent && img.format == format) return true;
    if (img.image) {
        (void)instance.GetDevice().waitIdle();
        Destroy(img);
    }
    const auto usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
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
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    });
    if (vr != vk::Result::eSuccess) {
        Destroy(img);
        return false;
    }
    img.view = v;
    img.format = format;
    img.extent = extent;
    img.usage = usage;
    img.layout = vk::ImageLayout::eUndefined;
    return true;
}

static vk::ImageAspectFlags DepthAspects(vk::Format format) {
    switch (format) {
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        return vk::ImageAspectFlagBits::eDepth;
    }
}

void PresentSr::DepthBarrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout new_layout) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = img.layout,
        .newLayout = new_layout,
        .image = img.image,
        .subresourceRange = {DepthAspects(img.format), 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
    img.layout = new_layout;
}

bool PresentSr::CaptureFg(vk::CommandBuffer cmd, Vulkan::Runtime& runtime,
                          VideoCore::Image& depth, vk::Extent2D in, BB::FG::Inputs& fg) {
    auto& fs = fg_slots[fg_next];
    if (!EnsureDepth(fs.depth, in, depth.info.pixel_format) ||
        !Ensure(fs.motion, in, motion.format)) {
        return false;
    }
    fg_next = (fg_next + 1) % FgSlots;
    runtime.Transit(&depth, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    DepthBarrier(cmd, fs.depth, vk::ImageLayout::eTransferDstOptimal);
    Barrier(cmd, fs.motion, vk::ImageLayout::eTransferDstOptimal);
    Barrier(cmd, motion, vk::ImageLayout::eTransferSrcOptimal);
    cmd.copyImage(depth.GetImage(), vk::ImageLayout::eTransferSrcOptimal, fs.depth.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
                      .extent = {in.width, in.height, 1},
                  });
    cmd.copyImage(motion.image, vk::ImageLayout::eTransferSrcOptimal, fs.motion.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {in.width, in.height, 1},
                  });
    Barrier(cmd, motion, vk::ImageLayout::eShaderReadOnlyOptimal);
    DepthBarrier(cmd, fs.depth, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, fs.motion, vk::ImageLayout::eShaderReadOnlyOptimal);
    runtime.Transit(&depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    fg.depth = {fs.depth.image, fs.depth.view, fs.depth.format, in, fs.depth.usage};
    fg.motion = {fs.motion.image, fs.motion.view, fs.motion.format, in, fs.motion.usage};
    fg.render = in;
    static bool said = false;
    if (!said) {
        said = true;
        BB::FG::Printf("[fg] first frame generation inputs: render %ux%u, depth %s, vectors %s (%s)",
                       in.width, in.height, vk::to_string(fs.depth.format).c_str(),
                       vk::to_string(fs.motion.format).c_str(),
                       BB::Upscale::BackendReady() ? "upscaled frame" : "no upscaler");
    }
    return true;
}

} // namespace BB
