// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021: DLSS / FSR 4 on the presented image, recorded into the presenter's command
// buffer (the game's own draw scheduler): no submit, no fence, no readback wait of its own.
#pragma once

#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_upscale.h"
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
                      vk::Extent2D in, vk::Extent2D out, SceneDepth& depth, bool srgb_decode,
                      BB::FG::Inputs* fg = nullptr, VideoCore::ImageId* depth_id = nullptr);
    // The image behind Run's returned view (RGBA16F, SHADER_READ_ONLY_OPTIMAL after Run).
    [[nodiscard]] vk::Extent2D LastOutput() const {
        return output.extent;
    }
    [[nodiscard]] vk::Image OutputImage() const {
        return output.image;
    }

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
    bool EnsureDepth(OwnImage& img, vk::Extent2D extent, vk::Format format);
    void DepthBarrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout);
    // bb-clean-021-fg: this frame's depth and motion vectors copied for FSR 3 frame generation
    // (read at the present, after the next frame may have started drawing into the originals).
    bool CaptureFg(vk::CommandBuffer cmd, Vulkan::Runtime& runtime, VideoCore::Image& depth,
                   vk::Extent2D in, BB::FG::Inputs& fg);
    void Destroy(OwnImage& img);
    void Barrier(vk::CommandBuffer cmd, OwnImage& img, vk::ImageLayout layout);
    bool Init();
    // bb-reactive: the reactive mask (bb_reactive.comp) and FSR 4's blend (bb_reactive_blend.comp).
    bool InitReactive();
    // bb-reactive: DLSS 5 on the native image (no upscaler): returns the output view or null.
    vk::ImageView RunDlss5Native(vk::CommandBuffer cmd, VideoCore::Image& depth,
                                 vk::ImageView depth_view, vk::Extent2D in, float jx, float jy,
                                 const BB::Upscale::Camera& cam, bool greater);
    bool RunReactive(vk::CommandBuffer cmd, vk::Extent2D in, float jx, float jy, bool camera_ok);
    void BlendFsr4(vk::CommandBuffer cmd, vk::Extent2D in, vk::Extent2D out, float jx, float jy);
    void ReactiveStats(vk::Extent2D in);

    const Vulkan::Instance& instance;
    bool init_tried = false, init_ok = false;
    vk::Sampler sampler{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    OwnImage color, motion, output;
    static constexpr u32 FgSlots = 4;
    struct FgSlot {
        OwnImage depth, motion;
    };
    FgSlot fg_slots[FgSlots]{};
    u32 fg_next = 0;

    static constexpr u32 Slots = 8, PerSlot = 4; // begin, before mv, after mv / before sr, end
    vk::QueryPool queries{};
    double ns_per_tick = 1.0;
    u32 slot = 0;
    bool slot_open = false;
    bool slot_written[Slots]{};
    bool slot_sr[Slots]{};
    u32 frames = 0;
    vk::Extent2D last_in{}, last_out{};

    // bb-reactive
    bool r_tried = false, r_ok = false;
    vk::Sampler linear{};
    vk::DescriptorSetLayout r_set_layout{}, b_set_layout{};
    vk::PipelineLayout r_layout{}, b_layout{};
    vk::Pipeline r_pipeline{}, b_pipeline{};
    OwnImage mask, prev_color;
    bool prev_valid = false;
    float prev_jitter[2]{};
    vk::Buffer r_stats{};
    VmaAllocation r_stats_alloc{};
    u32* r_stats_ptr = nullptr;
    static constexpr u32 RSlots = 8;
    u32 r_slot = 0;
    u32 r_frames = 0;
    float t_cov_last = 0.0f; // the transparency coverage last read (the guard)
    bool t_guard = false;
    // The once-a-second [reactive] line.
    u64 rs_mvn = 0, rs_mve[7]{};
    u64 rs_px = 0, rs_t = 0, rs_m = 0, rs_r = 0, rs_sum = 0;
    u32 rs_frames = 0, rs_read = 0, rs_tvalid = 0, rs_guarded = 0, rs_draws = 0, rs_copies = 0;
};

} // namespace BB
