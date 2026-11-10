// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_present_sr.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_dlss5.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_mv_comp.h"
#include "video_core/host_shaders/bb_reactive_blend_comp.h"
#include "video_core/host_shaders/bb_reactive_comp.h"
#include "video_core/renderer_vulkan/bb_reactive.h"
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
    Destroy(mask);
    Destroy(prev_color);
    if (r_pipeline) device.destroyPipeline(r_pipeline);
    if (b_pipeline) device.destroyPipeline(b_pipeline);
    if (r_layout) device.destroyPipelineLayout(r_layout);
    if (b_layout) device.destroyPipelineLayout(b_layout);
    if (r_set_layout) device.destroyDescriptorSetLayout(r_set_layout);
    if (b_set_layout) device.destroyDescriptorSetLayout(b_set_layout);
    if (linear) device.destroySampler(linear);
    if (r_stats) vmaDestroyBuffer(instance.GetAllocator(), r_stats, r_stats_alloc);
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
    // bb-reactive: the reactive mask (transparency + self motion) for every upscaler but the
    // emulator's TAA; FSR 4 (no reactive input in its model) gets it as a blend afterwards.
    if (BB::Upscale::Cfg().kind != BB::SR::Kind::Taa && BB::Reactive::AnyOn() &&
        RunReactive(cmd, in, jx, jy,
                    cam.mv_valid && !cam.reset && in == last_in && out == last_out)) {
        desc.has_reactive = true;
        desc.reactive = {mask.image, mask.view, mask.format, in,
                         vk::ImageLayout::eShaderReadOnlyOptimal, mask.usage};
    }
    desc.render = in;
    desc.upscale = out;
    desc.jitter[0] = jx;
    desc.jitter[1] = jy;
    desc.reset = cam.reset || in != last_in || out != last_out;
    {
        // (diag) Once a second: what the upscaler gets. History resets throw the past frames away (a soft
        // picture); frames without a camera have no motion vectors from it; jitter out of +-0.5 px is wrong.
        static u32 n = 0, valid = 0, resets = 0, size_resets = 0, bad_jitter = 0;
        static float jmax = 0.0f;
        static auto t0 = std::chrono::steady_clock::now();
        ++n;
        valid += cam.valid ? 1 : 0;
        resets += cam.reset ? 1 : 0;
        size_resets += (in != last_in || out != last_out) ? 1 : 0;
        const float ja = std::max(std::abs(jx), std::abs(jy));
        jmax = std::max(jmax, ja);
        bad_jitter += ja > 0.5f ? 1 : 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - t0 >= std::chrono::seconds(1)) {
            BB::Upscale::Checkpoint(fmt::format(
                "[sr-in] 1 s: {} frames, camera valid {}, history resets {} (camera {}, size {}), jitter max {:.3f} px "
                "({} out of range), {}x{} -> {}x{}, sharpness {:.2f}",
                n, valid, resets + size_resets, resets, size_resets, jmax, bad_jitter, in.width, in.height,
                out.width, out.height, BB::Upscale::LiveSharpness()));
            n = valid = resets = size_resets = bad_jitter = 0;
            jmax = 0.0f;
            t0 = now;
        }
    }
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
    if (desc.has_reactive && BB::Upscale::Cfg().kind == BB::SR::Kind::Fsr4) {
        BlendFsr4(cmd, in, out, jx, jy);
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

namespace {
float EnvF(const char* name, float fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == 0) return fallback;
    char* end = nullptr;
    const float f = std::strtof(v, &end);
    return end != v && std::isfinite(f) ? f : fallback;
}
} // namespace

bool PresentSr::InitReactive() {
    if (r_tried) return r_ok;
    r_tried = true;
    const auto device = instance.GetDevice();
    const auto physical = instance.GetPhysicalDevice();
    const auto r8 = physical.getFormatProperties(vk::Format::eR8Unorm).optimalTilingFeatures;
    if (!(r8 & vk::FormatFeatureFlagBits::eStorageImage) ||
        !(r8 & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        BB::Upscale::Checkpoint("[reactive] R8 storage images not supported: reactive mask off");
        return false;
    }
    auto [sr, s] = device.createSampler({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    });
    if (sr != vk::Result::eSuccess) return false;
    linear = s;
    const auto make = [&](std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
                          std::span<const u32> code, vk::DescriptorSetLayout& set_layout_out,
                          vk::PipelineLayout& layout_out, vk::Pipeline& pipeline_out) -> bool {
        auto [lr, l] = device.createDescriptorSetLayout({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        if (lr != vk::Result::eSuccess) return false;
        set_layout_out = l;
        const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, push_size};
        auto [pr, pl] = device.createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_out,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
        if (pr != vk::Result::eSuccess) return false;
        layout_out = pl;
        const auto module = Vulkan::CompileSPV(code, device);
        if (!module) return false;
        const vk::ComputePipelineCreateInfo ci{
            .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
            .layout = layout_out,
        };
        auto [cr, p] = device.createComputePipeline({}, ci);
        device.destroyShaderModule(module);
        if (cr != vk::Result::eSuccess) return false;
        pipeline_out = p;
        return true;
    };
    constexpr auto Cis = vk::DescriptorType::eCombinedImageSampler;
    constexpr auto Comp = vk::ShaderStageFlagBits::eCompute;
    const std::array<vk::DescriptorSetLayoutBinding, 7> rb{{
        {0, Cis, 1, Comp},
        {1, Cis, 1, Comp},
        {2, Cis, 1, Comp},
        {3, Cis, 1, Comp},
        {4, Cis, 1, Comp},
        {5, vk::DescriptorType::eStorageImage, 1, Comp},
        {6, vk::DescriptorType::eStorageBuffer, 1, Comp},
    }};
    const std::array<vk::DescriptorSetLayoutBinding, 3> bb{{
        {0, Cis, 1, Comp},
        {1, Cis, 1, Comp},
        {2, vk::DescriptorType::eStorageImage, 1, Comp},
    }};
    if (!make(rb, 64, BB_REACTIVE_COMP, r_set_layout, r_layout, r_pipeline) ||
        !make(bb, 32, BB_REACTIVE_BLEND_COMP, b_set_layout, b_layout, b_pipeline)) {
        BB::Upscale::Checkpoint("[reactive] pipelines not created: reactive mask off");
        return false;
    }
    const VkBufferCreateInfo bci = vk::BufferCreateInfo{
        .size = RSlots * 16 * sizeof(u32),
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &bci, &aci, &buffer, &r_stats_alloc, &info) ==
        VK_SUCCESS) {
        r_stats = buffer;
        r_stats_ptr = static_cast<u32*>(info.pMappedData);
    } else {
        BB::Upscale::Checkpoint("[reactive] stats buffer not created: reactive mask off");
        return false;
    }
    r_ok = true;
    BB::Upscale::Checkpoint(fmt::format(
        "[reactive] on: transparency {} (BB_REACTIVE), self motion {} (BB_SELFMOTION), FSR 4 "
        "blend {:.2f} (BB_FSR4_REACTIVE)",
        BB::Reactive::TransparencyOn() ? "on" : "off", BB::Reactive::SelfMotionOn() ? "on" : "off",
        EnvF("BB_FSR4_REACTIVE", 0.6f)));
    return true;
}

bool PresentSr::RunReactive(vk::CommandBuffer cmd, vk::Extent2D in, float jx, float jy,
                            bool camera_ok) {
    const auto snap = BB::Reactive::Take();
    if (!InitReactive()) return false;
    if (!Ensure(mask, in, vk::Format::eR8Unorm) ||
        !Ensure(prev_color, in, vk::Format::eR16G16B16A16Sfloat)) {
        return false;
    }
    if (prev_color.layout == vk::ImageLayout::eUndefined) prev_valid = false;
    // Tunables (read once).
    static const float t_scale = EnvF("BB_REACTIVE_SCALE", 2.0f);
    static const float t_max = std::clamp(EnvF("BB_REACTIVE_MAX", 0.9f), 0.0f, 1.0f);
    static const float t_maxcov = EnvF("BB_REACTIVE_MAXCOV", 0.5f);
    static const float m_strength = std::clamp(EnvF("BB_SELFMOTION_STRENGTH", 0.8f), 0.0f, 1.0f);
    static const float m_threshold = EnvF("BB_SELFMOTION_THRESHOLD", 0.03f);
    static const float m_gain = EnvF("BB_SELFMOTION_GAIN", 8.0f);

    // The stats slot four frames back (long finished in a normal pipeline): coverage, guard.
    const u32 w_slot = r_slot;
    r_slot = (r_slot + 1) % RSlots;
    ++r_frames;
    if (r_frames > 4 && r_stats_ptr != nullptr) {
        const u32 rd = (w_slot + RSlots - 4) % RSlots;
        vmaInvalidateAllocation(instance.GetAllocator(), r_stats_alloc, 0, VK_WHOLE_SIZE);
        const u32* c = r_stats_ptr + rd * 16;
        const u64 px = u64(in.width) * in.height;
        t_cov_last = px ? float(c[0]) / float(px) : 0.0f;
        // Deferred lighting or a full-screen pass taken for transparency would mark most of
        // the picture: no transparency part then (measured on, applied off) until it drops.
        t_guard = t_cov_last > t_maxcov;
        rs_px += px;
        rs_t += c[0];
        rs_m += c[1];
        rs_r += c[2];
        rs_sum += c[3];
        rs_mvn += c[4];
        for (int i = 0; i < 7; ++i) rs_mve[i] += c[5 + i];
        ++rs_read;
    }

    const bool t_valid = snap.valid && snap.extent == in;
    const bool m_valid = camera_ok && prev_valid; // (the [mv-check] runs without BB_SELFMOTION)
    u32 flags = 0;
    if (t_valid) flags |= 1u;
    if (t_valid && !t_guard) flags |= 2u;
    if (m_valid) flags |= 4u;
    if (m_valid && BB::Reactive::SelfMotionOn()) flags |= 8u;
    rs_frames++;
    rs_tvalid += t_valid ? 1 : 0;
    rs_guarded += (t_valid && t_guard) ? 1 : 0;
    rs_draws += snap.draws;
    rs_copies += snap.copies;

    // Clear this frame's stats slot.
    cmd.fillBuffer(r_stats, w_slot * 64, 64, 0);
    {
        const vk::BufferMemoryBarrier2 bb{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .buffer = r_stats,
            .offset = w_slot * 64,
            .size = 64,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                .pBufferMemoryBarriers = &bb});
    }
    if (!prev_valid || prev_color.layout != vk::ImageLayout::eShaderReadOnlyOptimal) {
        Barrier(cmd, prev_color, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    Barrier(cmd, mask, vk::ImageLayout::eGeneral);

    struct {
        u32 size[4];
        float jitter[4];
        float k[4];
        float k2[4];
    } push{};
    static_assert(sizeof(push) == 64);
    push.size[0] = in.width;
    push.size[1] = in.height;
    push.size[2] = flags;
    push.size[3] = w_slot;
    push.jitter[0] = jx;
    push.jitter[1] = jy;
    push.jitter[2] = prev_jitter[0];
    push.jitter[3] = prev_jitter[1];
    push.k[0] = t_scale;
    push.k[1] = t_max;
    push.k[2] = m_strength;
    push.k[3] = m_threshold;
    push.k2[0] = m_gain;
    const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    const vk::DescriptorImageInfo cur_info{sampler, color.view, ro};
    const vk::DescriptorImageInfo prev_info{linear, prev_color.view, ro};
    const vk::DescriptorImageInfo mv_info{sampler, motion.view, ro};
    const vk::DescriptorImageInfo before_info{sampler, t_valid ? snap.before : color.view, ro};
    const vk::DescriptorImageInfo after_info{sampler, t_valid ? snap.after : color.view, ro};
    const vk::DescriptorImageInfo mask_info{{}, mask.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo stats_info{r_stats, 0, VK_WHOLE_SIZE};
    const auto img = [](u32 binding, const vk::DescriptorImageInfo* info) {
        return vk::WriteDescriptorSet{.dstBinding = binding,
                                      .descriptorCount = 1,
                                      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                      .pImageInfo = info};
    };
    const std::array writes{
        img(0, &cur_info),
        img(1, &prev_info),
        img(2, &mv_info),
        img(3, &before_info),
        img(4, &after_info),
        vk::WriteDescriptorSet{.dstBinding = 5,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &mask_info},
        vk::WriteDescriptorSet{.dstBinding = 6,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageBuffer,
                               .pBufferInfo = &stats_info},
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, r_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, r_layout, 0, writes);
    cmd.pushConstants(r_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    Barrier(cmd, mask, vk::ImageLayout::eShaderReadOnlyOptimal);

    // This frame's colour becomes the next frame's previous one.
    Barrier(cmd, color, vk::ImageLayout::eTransferSrcOptimal);
    Barrier(cmd, prev_color, vk::ImageLayout::eTransferDstOptimal);
    cmd.copyImage(color.image, vk::ImageLayout::eTransferSrcOptimal, prev_color.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {in.width, in.height, 1},
                  });
    Barrier(cmd, color, vk::ImageLayout::eShaderReadOnlyOptimal);
    Barrier(cmd, prev_color, vk::ImageLayout::eShaderReadOnlyOptimal);
    prev_valid = true;
    prev_jitter[0] = jx;
    prev_jitter[1] = jy;
    ReactiveStats(in);
    return true;
}

void PresentSr::ReactiveStats(vk::Extent2D in) {
    static auto t0 = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - t0 < std::chrono::seconds(1)) return;
    t0 = now;
    const auto pct = [&](u64 v) { return rs_px ? 100.0 * double(v) / double(rs_px) : 0.0; };
    BB::Upscale::Checkpoint(fmt::format(
        "[reactive] 1 s: {} frames ({} read), mask >0.1: {:.1f}% (transparency {:.1f}%, self "
        "motion {:.1f}%), mean {:.3f}; transparency snapshots {} frames ({} guarded off, last "
        "coverage {:.1f}%), {:.1f} draws / {:.1f} copies per frame, {}x{}",
        rs_frames, rs_read, pct(rs_r), pct(rs_t), pct(rs_m),
        rs_px ? double(rs_sum) / 255.0 / double(rs_px) : 0.0, rs_tvalid, rs_guarded,
        100.0 * t_cov_last, rs_frames ? double(rs_draws) / rs_frames : 0.0,
        rs_frames ? double(rs_copies) / rs_frames : 0.0, in.width, in.height));
    if (rs_mvn > 0) {
        // [mv-check]: mean reprojection error (sqrt colour, 0..1) of the previous frame with
        // the vectors x1, zero, negated and other scales; the right vectors must be lowest.
        static const float scales[7] = {1.0f, 0.0f, -1.0f, 0.5f, 2.0f, 0.25f, 4.0f};
        double e[7];
        int best = 0;
        for (int i = 0; i < 7; ++i) {
            e[i] = double(rs_mve[i]) / 256.0 / double(rs_mvn);
            if (e[i] < e[best]) best = i;
        }
        BB::Upscale::Checkpoint(fmt::format(
            "[mv-check] 1 s: err vectors {:.4f}, zero {:.4f}, negated {:.4f} (x0.25 {:.4f}, x0.5 "
            "{:.4f}, x2 {:.4f}, x4 {:.4f}), scale best {} over {} moving px{}",
            e[0], e[1], e[2], e[5], e[3], e[4], e[6], scales[best], rs_mvn,
            best == 0 ? "" : "  <- VECTORS NOT BEST"));
    } else {
        BB::Upscale::Checkpoint("[mv-check] 1 s: no pixel moved by the camera (or self motion off)");
    }
    rs_mvn = 0;
    for (auto& v : rs_mve) v = 0;
    rs_px = rs_t = rs_m = rs_r = rs_sum = 0;
    rs_frames = rs_read = rs_tvalid = rs_guarded = rs_draws = rs_copies = 0;
}

void PresentSr::BlendFsr4(vk::CommandBuffer cmd, vk::Extent2D in, vk::Extent2D out, float jx,
                          float jy) {
    static const float strength = std::clamp(EnvF("BB_FSR4_REACTIVE", 0.6f), 0.0f, 1.0f);
    if (strength <= 0.0f || !r_ok) return;
    Barrier(cmd, output, vk::ImageLayout::eGeneral);
    struct {
        u32 size[4];
        float k[4];
    } push{};
    static_assert(sizeof(push) == 32);
    push.size[0] = out.width;
    push.size[1] = out.height;
    push.size[2] = in.width;
    push.size[3] = in.height;
    push.k[0] = jx;
    push.k[1] = jy;
    push.k[2] = strength;
    const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    const vk::DescriptorImageInfo cur_info{linear, color.view, ro};
    const vk::DescriptorImageInfo mask_info{linear, mask.view, ro};
    const vk::DescriptorImageInfo out_info{{}, output.view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &cur_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &mask_info},
        vk::WriteDescriptorSet{.dstBinding = 2,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &out_info},
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, b_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, b_layout, 0, writes);
    cmd.pushConstants(b_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
    Barrier(cmd, output, vk::ImageLayout::eGeneral);
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
