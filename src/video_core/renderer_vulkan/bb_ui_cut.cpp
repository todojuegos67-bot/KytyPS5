// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Supermedo (github.com/Supermedo/bloodborne_pc @115a0299a)
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-bbport: Supermedo's HUD cut (vk_temporal_upscaler.cpp), see bb_ui_cut.h. The
// function bodies follow his; what differs is where the scene comes from (this build's resolution
// patch: the guest targets already have the render size, Supermedo's scaled_session) and that the
// upscale itself is PresentSr::Run (motion vectors, then DLSS / FSR 4.0 / FSR 3.1 / XeSS / TAA
// into an RGBA16F image, blitted into the UI image as his RunScaled does for FSR 4, DLSS and TAA).

#include "video_core/renderer_vulkan/bb_ui_cut.h"

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/bb_ui_composition.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image.h"

namespace BB {

namespace {
// FrameCapture's display list (the presented addresses).
constexpr u32 MaxDisplays = 16;
std::array<std::atomic<VAddr>, MaxDisplays> g_displays{};
std::atomic<u32> g_display_count{0};

bool IsSrgb(vk::Format f) {
    return f == vk::Format::eR8G8B8A8Srgb || f == vk::Format::eB8G8R8A8Srgb ||
           f == vk::Format::eA8B8G8R8SrgbPack32;
}
vk::Format WithSrgb(vk::Format f, bool srgb) {
    switch (f) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        return srgb ? vk::Format::eB8G8R8A8Srgb : vk::Format::eB8G8R8A8Unorm;
    default:
        return f;
    }
}
bool Rgba8(vk::Format f) {
    return f == vk::Format::eR8G8B8A8Unorm || f == vk::Format::eR8G8B8A8Srgb;
}
vk::ImageAspectFlags DepthAspect(vk::Format f) {
    return vk::ImageAspectFlagBits::eDepth |
           (f == vk::Format::eD32SfloatS8Uint || f == vk::Format::eD24UnormS8Uint ||
                    f == vk::Format::eD16UnormS8Uint
                ? vk::ImageAspectFlagBits::eStencil
                : vk::ImageAspectFlags{});
}
} // namespace

void UiCut::NoteDisplayBuffer(VAddr address) {
    const u32 count = g_display_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        if (g_displays[i].load(std::memory_order_relaxed) == address) return;
    }
    if (count < MaxDisplays) {
        g_displays[count].store(address, std::memory_order_relaxed);
        g_display_count.store(count + 1, std::memory_order_release);
    }
}

bool UiCut::IsDisplayBuffer(VAddr address) {
    const u32 count = g_display_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        if (g_displays[i].load(std::memory_order_relaxed) == address) return true;
    }
    return false;
}

UiCut::UiCut(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
             Vulkan::Runtime& runtime_, VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_} {}

UiCut::~UiCut() {
    Destroy(ui_image);
    Destroy(ui_depth_image);
    for (auto& h : hudless_images) Destroy(h);
    for (auto& [_, d] : displays) Destroy(d.image);
}

bool UiCut::Create(OwnImage& img, u32 w, u32 h, vk::Format format, vk::ImageUsageFlags usage,
                   bool mutable_format) {
    Destroy(img);
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .flags = mutable_format ? vk::ImageCreateFlagBits::eMutableFormat
                                : vk::ImageCreateFlags{},
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {w, h, 1},
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
    img.format = format;
    img.width = w;
    img.height = h;
    return true;
}

void UiCut::Destroy(OwnImage& img) {
    const auto device = instance.GetDevice();
    for (auto& v : img.views) device.destroyImageView(v.view);
    img.views.clear();
    if (img.image) vmaDestroyImage(instance.GetAllocator(), img.image, img.allocation);
    img = {};
}

vk::ImageView UiCut::Mirror(OwnImage& img, vk::Format format, vk::ImageAspectFlags aspect,
                            bool alpha_one) {
    for (const auto& v : img.views) {
        if (v.format == format && v.alpha_one == alpha_one) return v.view;
    }
    auto [r, view] = instance.GetDevice().createImageView({
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                       vk::ComponentSwizzle::eIdentity,
                       alpha_one ? vk::ComponentSwizzle::eOne : vk::ComponentSwizzle::eIdentity},
        .subresourceRange = {aspect, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) return {};
    img.views.push_back({format, alpha_one, view});
    return view;
}

bool UiCut::OutputSize(u32& w, u32& h) const {
    if (u32 ow = 0, oh = 0; BB::Upscale::OutputSize(ow, oh) && ow && oh) {
        w = ow;
        h = oh;
        return true;
    }
    return false;
}

bool UiCut::RenderTarget(u32 w, u32 h) const {
    // The game aligns its targets (Supermedo: up to 8 pixels).
    const u32 sw = BB::Upscale::g_scene_w, sh = BB::Upscale::g_scene_h;
    return sw && sh && w >= sw && h >= sh && w - sw < 8 && h - sh < 8;
}

void UiCut::OnFrameStart() {
    // The display pass of an upscaled frame reads the upscaled UI image.
    display_redirect = ui_phase;
    display_fg = display_redirect ? cut_fg : BB::FG::Inputs{};
    display_hudless = display_redirect ? cut_hudless : -1;
    ui_phase = false;
    ui_read_barrier = false;
    done_this_frame = false;
    hudless_taken = false;
    cut_fg = {};
    cut_hudless = -1;
    ldr_target = {};
}

void UiCut::OnColorTarget(VideoCore::ImageId color) {
    if (ui_phase) return;
    const auto& image = texture_cache.GetImage(color);
    if (image.info.pixel_format == vk::Format::eR8G8B8A8Unorm &&
        RenderTarget(image.info.size.width, image.info.size.height)) {
        ldr_target = color;
    }
}

void UiCut::OnDraw(u64 vs_hash, VideoCore::ImageId color, VideoCore::ImageId depth,
                   bool native_viewport, SceneDepth& scene) {
    // A UI draw without a color target (a Scaleform mask: color writes off) writes the stencil
    // the next UI draws test, so it belongs in the UI's output-size depth like them.
    if (ui_phase && !color && depth && depth != ui_depth && ui_color) {
        const auto& depth_image = texture_cache.GetImage(depth);
        const auto& ui_target = texture_cache.GetImage(ui_color);
        if (depth_image.info.size.width == ui_target.info.size.width &&
            depth_image.info.size.height == ui_target.info.size.height) {
            EnsureUiResources(ui_width, ui_height, ui_format, depth_image.info.pixel_format);
            PrepareUiDepth(depth);
            ui_depth = depth;
        }
        return;
    }
    if (!color) return;
    if (ui_phase) {
        // A UI movie can start without stencil and enable it for later text/masks.
        if (color == ui_color && depth && depth != ui_depth) {
            EnsureUiResources(ui_width, ui_height, ui_format,
                              texture_cache.GetImage(depth).info.pixel_format);
            PrepareUiDepth(depth);
            ui_depth = depth;
        }
        return;
    }
    if (done_this_frame || sr == nullptr) return;
    static const u64 ui_trigger_vs = [] {
        const char* hash = std::getenv("BB_UI_TRIGGER_VS");
        return hash ? std::strtoull(hash, nullptr, 16) : 0ull;
    }();
    const auto& image = texture_cache.GetImage(color);
    const bool movie = (ui_trigger_vs && vs_hash == ui_trigger_vs) ||
                       Vulkan::UiComposition::MovieShader(vs_hash);
    (void)native_viewport; // Supermedo: only for his live resolution path (guest targets 1080p)
    const bool ui_draw = movie && Rgba8(image.info.pixel_format) &&
                         RenderTarget(image.info.size.width, image.info.size.height) &&
                         !IsDisplayBuffer(image.info.guest_address);
    const bool scaled = BB::Upscale::Enabled();
    // Native + frame generation (Supermedo's native path): the cut copies the background at the
    // scene's size, the HUD-less image is taken before the UI draws, and the vectors / camera of
    // the frame are made at the cut for the FG (the presenter then shows the display copy).
    const bool native_fg = !scaled && BB::FG::WantInputs();
    const bool active = scaled ? BB::Upscale::BackendReady() : native_fg;
    switch (Vulkan::UiComposition::Choose(scaled || native_fg, ui_draw,
                                          scene.valid && ldr_target == color, active)) {
    case Vulkan::UiComposition::Background::None:
        return;
    case Vulkan::UiComposition::Background::Temporal:
        if (native_fg ? RunNativeFg(scene, color, depth) : RunScaled(scene)) {
            CaptureHudlessUi();
            return;
        }
        if (ui_phase) return; // native: the vectors failed but the UI copy is in place
        // A failed dispatch must not force UI text back to the scene resolution.
        [[fallthrough]];
    case Vulkan::UiComposition::Background::Copy:
        RunUiOnly(color, depth);
        CaptureHudlessUi();
        return;
    }
}

bool UiCut::Active() {
    return BB::Upscale::Enabled() || BB::FG::WantInputs();
}

bool UiCut::RunNativeFg(SceneDepth& scene, VideoCore::ImageId color_id,
                        VideoCore::ImageId depth_id) {
    auto& color = texture_cache.GetImage(ldr_target);
    const vk::Extent2D in{BB::Upscale::g_scene_w, BB::Upscale::g_scene_h};
    VideoCore::ImageViewInfo ci{};
    ci.format = color.info.pixel_format;
    const vk::ImageView color_view = *color.FindView(ci, false).image_view;
    runtime.Transit(&color, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    scheduler.EndRendering();
    BB::FG::Inputs fg{};
    VideoCore::ImageId scene_depth{};
    // No upscaler: Run only makes the vectors / camera and copies them for the FG (want_fg).
    (void)sr->Run(scheduler.CommandBuffer(), texture_cache, runtime, color, color_view, in, in,
                  scene, false, &fg, &scene_depth);
    scheduler.GetDynamicState().Invalidate();
    RunUiOnly(color_id, depth_id);
    if (!ui_phase) return false;
    cut_fg = fg;
    std::scoped_lock lock{counter_mutex};
    ++counters.cuts;
    return fg.valid;
}

void UiCut::EnsureUiResources(u32 w, u32 h, vk::Format color, vk::Format depth) {
    const bool resized = w != ui_width || h != ui_height;
    const bool new_color = !ui_image.image || resized || color != ui_format;
    const bool new_depth = !ui_depth_image.image || resized || depth != ui_depth_format;
    if (!new_color && !new_depth) return;
    scheduler.Finish();
    ui_width = w;
    ui_height = h;
    if (new_color) {
        Create(ui_image, w, h, color,
               vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment |
                   vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
               true);
        ui_format = color;
    }
    if (new_depth) {
        Create(ui_depth_image, w, h, depth,
               vk::ImageUsageFlagBits::eDepthStencilAttachment |
                   vk::ImageUsageFlagBits::eTransferDst,
               false);
        ui_depth_view = Mirror(ui_depth_image, depth, DepthAspect(depth));
        ui_depth_format = depth;
    }
    BB::Upscale::Checkpoint(fmt::format("UI: native composition {}x{} ({}), independent of the "
                                        "upscaler's history",
                                        w, h, vk::to_string(color)));
}

void UiCut::PrepareUiDepth(VideoCore::ImageId depth_id) {
    scheduler.EndRendering();
    const auto aspect = DepthAspect(ui_depth_format);
    // Copy depth if blit is supported; otherwise just clear it (UI depth test still works).
    const auto props = instance.GetPhysicalDevice().getFormatProperties(ui_depth_format);
    const bool depth_blit =
        (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc) &&
        (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
    const bool copy = depth_id && depth_blit &&
                      texture_cache.GetImage(depth_id).info.pixel_format == ui_depth_format;
    if (copy) {
        runtime.Transit(&texture_cache.GetImage(depth_id), vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
    }
    const auto cmd = scheduler.CommandBuffer();
    const vk::ImageSubresourceRange range{aspect, 0, 1, 0, 1};
    vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = ui_depth_image.image,
        .subresourceRange = range,
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    cmd.clearDepthStencilImage(ui_depth_image.image, vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue{1.0f, 0}, range);
    if (copy) {
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        const auto& source = texture_cache.GetImage(depth_id);
        const vk::ImageBlit region{
            .srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
            .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                     vk::Offset3D{s32(source.info.size.width),
                                                  s32(source.info.size.height), 1}},
            .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
            .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                     vk::Offset3D{s32(ui_width), s32(ui_height), 1}},
        };
        cmd.blitImage(source.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                      ui_depth_image.image, vk::ImageLayout::eTransferDstOptimal, region,
                      vk::Filter::eNearest);
    }
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                           vk::PipelineStageFlagBits2::eLateFragmentTests;
    barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eGeneral;
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

void UiCut::RunUiOnly(VideoCore::ImageId color_id, VideoCore::ImageId depth_id) {
    auto& color = texture_cache.GetImage(color_id);
    u32 ow = color.info.size.width, oh = color.info.size.height;
    const bool native = !BB::Upscale::Enabled(); // frame generation without an upscaler
    if (native) {
        ow = BB::Upscale::g_scene_w;
        oh = BB::Upscale::g_scene_h;
        if (!ow || !oh) return;
    } else if (!OutputSize(ow, oh)) {
        return;
    }
    if (ow < color.info.size.width || oh < color.info.size.height) return;
    const auto depth_format = depth_id ? texture_cache.GetImage(depth_id).info.pixel_format
                                       : vk::Format::eD32SfloatS8Uint;
    EnsureUiResources(ow, oh, color.info.pixel_format, depth_format);
    PrepareUiDepth(depth_id);
    // Copy only the background already drawn. The subsequent menu/HUD/text draws execute
    // directly at output resolution, including frames with no scene or camera at all.
    runtime.Transit(&color, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = ui_image.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    const vk::ImageBlit region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(BB::Upscale::g_scene_w),
                                              s32(BB::Upscale::g_scene_h), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{s32(ow), s32(oh), 1}},
    };
    cmd.blitImage(color.GetImage(), vk::ImageLayout::eTransferSrcOptimal, ui_image.image,
                  vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.dstAccessMask =
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eGeneral;
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    done_this_frame = ui_phase = true;
    ui_color = color_id;
    ui_depth = depth_id;
    std::scoped_lock lock{counter_mutex};
    ++counters.ui_only;
}

bool UiCut::RunScaled(SceneDepth& scene) {
    auto& color = texture_cache.GetImage(ldr_target);
    const vk::Extent2D in{BB::Upscale::g_scene_w, BB::Upscale::g_scene_h};
    vk::Extent2D out{color.info.size.width, color.info.size.height};
    if (u32 ow = 0, oh = 0; OutputSize(ow, oh) && ow >= in.width && oh >= in.height) {
        out = {ow, oh};
    }
    if (BB::Upscale::Cfg().kind == BB::SR::Kind::Taa) out = in; // native TAA
    // The scene's colour as the upscaler reads it (sampled, as the presenter's image was).
    VideoCore::ImageViewInfo ci{};
    ci.format = color.info.pixel_format;
    const vk::ImageView color_view = *color.FindView(ci, false).image_view;
    runtime.Transit(&color, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    scheduler.EndRendering();
    auto cmd = scheduler.CommandBuffer();
    BB::FG::Inputs fg{};
    VideoCore::ImageId depth_id{};
    const vk::ImageView produced =
        sr->Run(cmd, texture_cache, runtime, color, color_view, in, out, scene, false, &fg,
                &depth_id);
    // A DLSS interposer may continue the recording in another command buffer.
    scheduler.GetDynamicState().Invalidate();
    if (!produced || !depth_id) return false;
    cmd = scheduler.CommandBuffer();
    const auto& depth = texture_cache.GetImage(depth_id);
    EnsureUiResources(out.width, out.height, color.info.pixel_format, depth.info.pixel_format);
    PrepareUiDepth(depth_id);
    cmd = scheduler.CommandBuffer();
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const auto barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                             vk::ImageLayout new_layout, vk::PipelineStageFlags2 dst_stage,
                             vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = all,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    };
    // Supermedo's RunScaled for FSR 4, DLSS and TAA: the HDR-format output blitted into the
    // output-size UI image.
    barrier(sr->OutputImage(), vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eBlit,
            vk::AccessFlagBits2::eTransferRead);
    barrier(ui_image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
    const vk::ImageBlit region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(out.width), s32(out.height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(out.width), s32(out.height), 1}},
    };
    cmd.blitImage(sr->OutputImage(), vk::ImageLayout::eTransferSrcOptimal, ui_image.image,
                  vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eNearest);
    barrier(sr->OutputImage(), vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal, all, vk::AccessFlagBits2::eShaderRead);
    barrier(ui_image.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead |
                vk::AccessFlagBits2::eColorAttachmentWrite);
    done_this_frame = true; // the UI is not jittered
    ui_phase = true;
    ui_color = ldr_target;
    ui_depth = depth_id;
    cut_fg = fg;
    std::scoped_lock lock{counter_mutex};
    ++counters.cuts;
    return true;
}

void UiCut::CaptureHudlessUi() {
    if (!ui_phase || !ui_image.image || hudless_taken || !BB::FG::WantInputs()) return;
    hudless_taken = true;
    // The background is in the UI image (General): copied before the UI draws into it.
    auto& slot = hudless_images[hudless_next];
    if (slot.width != ui_width || slot.height != ui_height || slot.format != ui_format) {
        scheduler.Finish();
        if (!Create(slot, ui_width, ui_height, ui_format,
                    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                    true)) {
            return;
        }
    }
    scheduler.EndRendering();
    const auto cmd = scheduler.CommandBuffer();
    vk::ImageMemoryBarrier2 b[2]{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .image = ui_image.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = slot.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = b});
    const vk::ImageCopy copy{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {ui_width, ui_height, 1},
    };
    cmd.copyImage(ui_image.image, vk::ImageLayout::eTransferSrcOptimal, slot.image,
                  vk::ImageLayout::eTransferDstOptimal, copy);
    b[0].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    b[0].srcAccessMask = vk::AccessFlagBits2::eTransferRead;
    b[0].dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    b[0].dstAccessMask =
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
    b[0].oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    b[0].newLayout = vk::ImageLayout::eGeneral;
    b[1].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    b[1].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    b[1].dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    b[1].dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    b[1].oldLayout = vk::ImageLayout::eTransferDstOptimal;
    b[1].newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = b});
    cut_hudless = int(hudless_next);
    hudless_next = (hudless_next + 1) % HudlessSlots;
    std::scoped_lock lock{counter_mutex};
    ++counters.hudless;
}

bool UiCut::RedirectColor(VideoCore::ImageId color, const VideoCore::ImageViewInfo& view_info,
                          Target& target) {
    if (ui_phase && color == ui_color) {
        target = {Mirror(ui_image, view_info.format, vk::ImageAspectFlagBits::eColor),
                  vk::ImageLayout::eGeneral, ui_width, ui_height, true};
        return target.view != vk::ImageView{};
    }
    const auto& image = texture_cache.GetImage(color);
    const VAddr address = image.info.guest_address;
    if (!IsDisplayBuffer(address)) {
        display_redirect = false;
        return false;
    }
    std::scoped_lock lock{display_mutex};
    auto& display = displays[address];
    if (!display_redirect) {
        display.valid = false; // not upscaled: the presenter shows the guest buffer
        return false;
    }
    if (!display.image.image || display.image.format != image.info.pixel_format ||
        display.image.width != ui_width || display.image.height != ui_height) {
        scheduler.Finish();
        BB::Upscale::Checkpoint(fmt::format("Display: host buffer {}x{} (live)", ui_width,
                                            ui_height));
        if (!Create(display.image, ui_width, ui_height, image.info.pixel_format,
                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                        vk::ImageUsageFlagBits::eTransferSrc,
                    true)) {
            display.valid = false;
            return false;
        }
    }
    scheduler.EndRendering();
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = display.image.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.CommandBuffer().pipelineBarrier2(
        {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    display.valid = true;
    display.fg = display_fg;
    display.hudless = display_hudless;
    display.rt_format = view_info.format;
    target = {Mirror(display.image, view_info.format, vk::ImageAspectFlagBits::eColor),
              vk::ImageLayout::eGeneral, ui_width, ui_height, false};
    {
        std::scoped_lock clock{counter_mutex};
        ++counters.displays;
    }
    return target.view != vk::ImageView{};
}

bool UiCut::RedirectDepth(VideoCore::ImageId depth, Target& target) {
    if (!ui_phase || depth != ui_depth || !ui_depth_view) return false;
    target = {ui_depth_view, vk::ImageLayout::eGeneral, ui_width, ui_height, true};
    return true;
}

bool UiCut::RedirectSampled(VideoCore::ImageId image, vk::Format format, vk::ImageView& view,
                            vk::ImageLayout& layout) {
    if (!display_redirect || image != ui_color) return false;
    if (!ui_read_barrier) {
        ui_read_barrier = true;
        scheduler.EndRendering();
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = ui_image.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        scheduler.CommandBuffer().pipelineBarrier2(
            {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    }
    view = Mirror(ui_image, format, vk::ImageAspectFlagBits::eColor);
    layout = vk::ImageLayout::eGeneral;
    return view != vk::ImageView{};
}

bool UiCut::DisplayOverride(VAddr address, vk::Format format, Display& out) {
    std::scoped_lock lock{display_mutex};
    const auto it = displays.find(address);
    if (it == displays.end() || !it->second.valid) return false;
    auto& d = it->second;
    out = {};
    out.image = d.image.image;
    // Viewed with the format the display pass WROTE it with (as Supermedo's DisplayOverride returns
    // the display image's own format), alpha one. The presenter's frame format can differ (BGRA vs
    // RGBA order, sRGB vs UNORM): viewing the host copy with it swapped red and blue / shifted the
    // gamma on every frame (menus too). Only the sRGB-ness of the presenter's format is kept.
    const vk::Format rt = d.rt_format != vk::Format::eUndefined ? d.rt_format : format;
    const vk::Format shown = WithSrgb(rt, IsSrgb(format));
    static bool logged = false;
    if (!logged) {
        logged = true;
        BB::Upscale::Checkpoint(fmt::format("Display: written as {}, presenter asks {}, shown as {}",
                                            vk::to_string(rt), vk::to_string(format),
                                            vk::to_string(shown)));
    }
    out.view = Mirror(d.image, shown, vk::ImageAspectFlagBits::eColor, true);
    out.format = shown;
    out.width = d.image.width;
    out.height = d.image.height;
    out.fg = d.fg;
    if (d.hudless >= 0 && hudless_images[d.hudless].image) {
        auto& h = hudless_images[d.hudless];
        out.hudless = h.image;
        out.hudless_view = Mirror(h, WithSrgb(h.format, IsSrgb(format)),
                                  vk::ImageAspectFlagBits::eColor, true);
    }
    // Shown once: the next display pass decides again.
    d.valid = false;
    return out.view != vk::ImageView{};
}

UiCut::Counters UiCut::TakeCounters() {
    std::scoped_lock lock{counter_mutex};
    const Counters c = counters;
    counters = {};
    return c;
}

} // namespace BB
