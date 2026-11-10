// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_reactive.h"

#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image.h"

namespace BB::Reactive {

namespace {

bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v != nullptr && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
                              std::strcmp(v, "false") == 0));
}

struct Snap {
    vk::Image image{};
    vk::ImageView view{};
    VmaAllocation allocation{};
    vk::Format format{};
    vk::Extent2D extent{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

struct State {
    const Vulkan::Instance* instance = nullptr;
    Snap before, after;
    VideoCore::ImageId target{};
    bool open = false;
    bool have_before = false;
    bool closed = false; // `after` holds the current target's state after a run
    u32 draws = 0, copies = 0, runs = 0;
    bool failed = false;
};
State g;

constexpr u32 MaxCopies = 24; // per frame (each is one scene-sized colour copy)

void Destroy(Snap& s) {
    const auto device = g.instance->GetDevice();
    if (s.view) device.destroyImageView(s.view);
    if (s.image) vmaDestroyImage(g.instance->GetAllocator(), s.image, s.allocation);
    s = {};
}

bool Ensure(Snap& s, vk::Extent2D extent, vk::Format format) {
    if (s.image && s.extent == extent && s.format == format) return true;
    if (s.image) {
        (void)g.instance->GetDevice().waitIdle(); // (size / format change only)
        Destroy(s);
    }
    const auto props = g.instance->GetPhysicalDevice().getFormatProperties(format);
    const auto need = vk::FormatFeatureFlagBits::eSampledImage |
                      vk::FormatFeatureFlagBits::eTransferDst;
    if ((props.optimalTilingFeatures & need) != need) return false;
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(g.instance->GetAllocator(), &ici, &aci, &image, &s.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    s.image = image;
    auto [vr, v] = g.instance->GetDevice().createImageView({
        .image = s.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (vr != vk::Result::eSuccess) {
        Destroy(s);
        return false;
    }
    s.view = v;
    s.format = format;
    s.extent = extent;
    s.layout = vk::ImageLayout::eUndefined;
    return true;
}

// The guest colour target (mip 0, layer 0) into `dst`; false when it cannot be copied.
bool Copy(Vulkan::Scheduler& scheduler, Vulkan::Runtime& runtime,
          VideoCore::TextureCache& texture_cache, VideoCore::ImageId id, Snap& dst) {
    if (g.copies >= MaxCopies) return false;
    auto& image = texture_cache.GetImage(id);
    if (!image.backing || !image.GetImage()) return false;
    const vk::Extent2D extent{image.info.size.width, image.info.size.height};
    const vk::Format format = image.backing->image.image_ci.format;
    if (!Ensure(dst, extent, format)) {
        if (!g.failed) {
            g.failed = true;
            BB::Upscale::Checkpoint(fmt::format(
                "[reactive] scene colour {} {}x{} cannot be snapshot: transparency mask off",
                vk::to_string(format), extent.width, extent.height));
        }
        return false;
    }
    scheduler.EndRendering();
    runtime.Transit(&image, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    const vk::ImageMemoryBarrier2 to_dst{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = dst.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst});
    cmd.copyImage(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageCopy{
                      .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                      .extent = {extent.width, extent.height, 1},
                  });
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = dst.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
    dst.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    ++g.copies;
    return true;
}

} // namespace

bool TransparencyOn() {
    static const bool on = EnvOn("BB_REACTIVE");
    return on;
}

bool SelfMotionOn() {
    static const bool on = EnvOn("BB_SELFMOTION");
    return on;
}

bool AnyOn() {
    return TransparencyOn() || SelfMotionOn();
}

void OnDraw(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
            Vulkan::Runtime& runtime, VideoCore::TextureCache& texture_cache,
            VideoCore::ImageId color, bool scene_draw, bool depth_write, bool blend,
            bool opaque_seen) {
    if (!TransparencyOn() || g.failed) return;
    g.instance = &instance;
    bool match = scene_draw && !depth_write && blend && opaque_seen && bool(color);
    if (match) {
        const auto& image = texture_cache.GetImage(color);
        match = image.info.num_samples == 1 && image.info.size.width == BB::Upscale::g_scene_w &&
                image.info.size.height == BB::Upscale::g_scene_h && !image.info.props.is_depth;
    }
    if (match) {
        if (color == g.target && g.have_before) {
            g.open = true; // the run goes on (or a reopened run on the same target)
            ++g.draws;
            return;
        }
        // A new target: the last run of the frame wins.
        g.open = false;
        g.closed = false;
        g.have_before = false;
        g.target = color;
        g.draws = 0;
        if (!Copy(scheduler, runtime, texture_cache, color, g.before)) return;
        g.have_before = true;
        g.open = true;
        g.draws = 1;
        ++g.runs;
        return;
    }
    if (g.open) {
        g.open = false;
        if (Copy(scheduler, runtime, texture_cache, g.target, g.after)) {
            g.closed = true;
        }
    }
}

Snapshot Take() {
    Snapshot s{};
    s.copies = g.copies;
    s.runs = g.runs;
    s.draws = g.draws;
    if (g.have_before && g.closed && g.before.extent == g.after.extent &&
        g.before.format == g.after.format) {
        s.valid = true;
        s.before = g.before.view;
        s.after = g.after.view;
        s.extent = g.before.extent;
    }
    g.open = false;
    g.closed = false;
    g.have_before = false;
    g.target = {};
    g.draws = g.copies = g.runs = 0;
    return s;
}

} // namespace BB::Reactive
