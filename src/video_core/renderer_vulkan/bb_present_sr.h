// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021: DLSS / FSR 4 on the presented image, recorded into the presenter's command
// buffer (the game's own draw scheduler): no submit, no fence, no readback wait of its own.
#pragma once

#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {
class Instance;
class Runtime;
}

namespace BB {

// The scene depth target of the frame, recorded by the rasterizer (one size compare per
// depth target bind, only with an upscaler on).
struct SceneDepth {
    bool valid = false;
    VideoCore::TextureCache::ImageDesc desc;
    float xscale = 0, xoffset = 0, yscale = 0, yoffset = 0, zscale = 1, zoffset = 0;
    bool greater = true;
};

class PresentSr {
public:
    explicit PresentSr(const Vulkan::Instance& instance);
    ~PresentSr();

    // Upscales `flip` (the presented image, its view as the presenter reads it) to `out`.
    // Returns the output view (RGBA16F linear, shader read) or a null view (presenter path as
    // stock). `cmd` must be outside a render pass.
    vk::ImageView Run(vk::CommandBuffer cmd, VideoCore::TextureCache& texture_cache,
                      Vulkan::Runtime& runtime, VideoCore::Image& flip, vk::ImageView flip_view,
                      vk::Extent2D in, vk::Extent2D out, SceneDepth& depth, bool srgb_decode);

    // Timestamps: begin of a frame's GPU work, its end (once per frame, around the flip).
    void FrameBegin(vk::CommandBuffer cmd);
    void FrameEnd(vk::CommandBuffer cmd);
    // The newest finished frame's GPU times (ms), without waiting; false if none is ready.
    bool Poll(double& gpu_ms, double& sr_ms, double& mv_ms);

private:
    struct OwnImage {
        vk::Image image{};
        vk::ImageView view{};
        VmaAllocation allocation{};
        vk::Format format{};
        vk::Extent2D extent{};
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        vk::ImageUsageFlags usage{};
    };
    bool Ensure(OwnImage& img, vk::Extent2D extent, vk::Format format);
    void Destroy(OwnImage& img);
    void Barrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout);
    bool Init();

    const Vulkan::Instance& instance;
    bool init_tried = false, init_ok = false;
    vk::Sampler sampler{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    OwnImage color, motion, output;

    static constexpr u32 Slots = 8, PerSlot = 4; // begin, before mv, after mv / before sr, end
    vk::QueryPool queries{};
    double ns_per_tick = 1.0;
    u32 slot = 0;
    bool slot_open = false;
    bool slot_written[Slots]{};
    bool slot_sr[Slots]{};
    u32 frames = 0;
    vk::Extent2D last_in{}, last_out{};
};

// The HUD split: the first draw into the presented image after the scene that samples no
// scene-sized image starts the HUD. The scene is copied before that draw; the upscaler runs on the
// copy and the HUD (the presented image minus the copy, at the render size) is added back after
// it, so the temporal upscaler never sees the HUD (no trail). A frame without a HUD start is
// upscaled as before.
class HudSplit {
public:
    explicit HudSplit(const Vulkan::Instance& instance);
    ~HudSplit();

    [[nodiscard]] bool IsFlip(VAddr guest) const;
    void NoteScene() {
        scene_seen = true;
    }
    [[nodiscard]] bool SceneSeen() const {
        return scene_seen;
    }
    [[nodiscard]] bool Captured() const {
        return captured;
    }
    [[nodiscard]] VAddr CapturedAddr() const {
        return captured_addr;
    }
    [[nodiscard]] vk::ImageView SceneView() const {
        return scene.view;
    }

    // Copies `flip` (the presented image, its contents up to this draw) into the scene copy.
    // Outside a render pass: `cmd` is the draw's command buffer.
    bool Capture(vk::CommandBuffer cmd, Vulkan::Runtime& runtime, VideoCore::Image& flip);
    // The HUD added to the upscaled scene; a null view when it cannot be done.
    vk::ImageView Composite(vk::CommandBuffer cmd, vk::ImageView flip_view,
                            vk::ImageView upscaled, vk::Extent2D in, vk::Extent2D out,
                            bool srgb_decode);
    // The presenter's flip: records it as a presented image and ends the frame's HUD state.
    void EndFrame(VAddr presented);

private:
    struct OwnImage {
        vk::Image image{};
        vk::ImageView view{};
        VmaAllocation allocation{};
        vk::Extent2D extent{};
        vk::Format format{};
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    bool Ensure(OwnImage& img, vk::Extent2D extent, vk::Format format,
                vk::ImageUsageFlags usage);
    void Destroy(OwnImage& img);
    void Barrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout);
    bool Init();

    const Vulkan::Instance& instance;
    bool init_tried = false, init_ok = false;
    vk::Sampler sampler{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    OwnImage scene, output;
    bool scene_seen = false;
    bool captured = false;
    VAddr captured_addr = 0;
    static constexpr u32 FlipSlots = 4;
    VAddr flips[FlipSlots]{};
    u32 next_flip = 0;
};

} // namespace BB
