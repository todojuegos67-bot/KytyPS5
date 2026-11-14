// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean: the HUD kept out of frame generation. The finished frame (HUD included) was given to
// DLSS-G / FSR 3 FG with the scene's camera vectors under the HUD too, so the generated frames
// moved the HUD with the camera and it trembled. Each 3D frame now gets a UI layer (premultiplied
// RGBA8, output size: bb_hud_mask.comp) that is drawn over every generated frame: DLSS-G by our
// own pass on its outputs (bb_hud_over.comp), FSR 3 as the FG swapchain's registered UI resource.
// Per frame (never per draw): two compute dispatches and one image copy at the output size.
#pragma once

#include <atomic>
#include "common/types.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}

namespace BB {

class HudFg {
public:
    explicit HudFg(const Vulkan::Instance& instance);
    ~HudFg();

    // `frame` (GENERAL, sampled view `frame_view`), `hudless` (SHADER_READ_ONLY) may be null.
    // Returns the UI layer (SHADER_READ_ONLY_OPTIMAL) or null.
    const BB::FG::Image* Build(vk::CommandBuffer cmd, vk::Image frame, vk::ImageView frame_view,
                               vk::Format format, vk::Extent2D extent, vk::ImageView hudless);
    // The UI layer over one generated frame (GENERAL, `rgba_view` an R8G8B8A8 storage view).
    void Overlay(vk::CommandBuffer cmd, vk::ImageView rgba_view, vk::Extent2D extent, bool bgra);

    // bb-fsr3-hudless: makes `target` (the post-processed HUD-less image, SHADER_READ_ONLY, an
    // R8G8B8A8 storage view `target_rgba` of it) equal to the finished frame (`frame_view`,
    // GENERAL, UNORM 8-bit) wherever the raw UI target with the HUD (`raw_hud`) equals the raw
    // copy without it (`raw_hudless`), both SHADER_READ_ONLY. Leaves `target` SHADER_READ_ONLY.
    // `raw_swapped`: the raw views swap red and blue (bb-artifact-fix, BB_HUDLESS_SWAP); the
    // ratio grade under the HUD (BB_HUDLESS_GRADE) then defaults off.
    bool FixHudless(vk::CommandBuffer cmd, vk::ImageView frame_view, vk::ImageView raw_hud,
                    vk::ImageView raw_hudless, vk::Image target, vk::ImageView target_rgba,
                    vk::Extent2D extent, bool bgra, bool raw_swapped = false);
    // Once a second (presenter): "[fsr3-hudless] ..." (HUD pixels, mismatching pixels fixed).
    void LogHudlessSecond(double seconds);

    // Once a second (presenter): "[hud-fg] ..." with what was done.
    void NoteFrame(bool fg_frame, const char* fg_kind, bool fsr3_layer);
    void LogSecond(double seconds);

private:
    struct Own {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::ImageView view{};
        vk::Format format{};
        vk::Extent2D extent{};
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    bool Init();
    bool Ensure(Own& img, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    void Destroy(Own& img);
    void Barrier(vk::CommandBuffer cmd, Own& img, vk::ImageLayout layout);

    const Vulkan::Instance& instance;
    bool tried = false, ok = false;
    vk::Sampler sampler{};
    vk::DescriptorSetLayout mask_set{}, over_set{};
    vk::PipelineLayout mask_layout{}, over_layout{};
    vk::Pipeline mask_pipe{}, over_pipe{};
    // One UI layer per frame in flight (FSR 3 reads it at that frame's present, later).
    static constexpr u32 Ring = 8;
    Own ring[Ring];
    u32 ring_next = 0;
    Own* ui = nullptr; // this frame's
    Own prev;
    bool prev_valid = false;
    BB::FG::Image ui_out{};

    // bb-fsr3-hudless (FixHudless).
    bool fix_tried = false, fix_ok = false;
    vk::DescriptorSetLayout fix_set{};
    vk::PipelineLayout fix_layout{};
    vk::Pipeline fix_pipe{};
    vk::Buffer fix_counts{};
    VmaAllocation fix_counts_alloc{};
    u32* fix_counts_map = nullptr;
    static constexpr u32 FixSlots = 8; // read back FixSlots - 1 frames later
    u32 fix_slot = 0, fix_frames = 0;
    std::atomic<u32> sec_fix{0}, sec_fix_skipped{0};
    std::atomic<u64> sec_hud_px{0}, sec_fixed_px{0}, sec_px{0};

    // Per-second counters (the GPU thread builds, the present thread logs).
    std::atomic<u32> sec_fg{0}, sec_layer{0}, sec_hudless{0}, sec_static{0}, sec_overlaid{0},
        sec_fsr3{0};
    std::atomic<const char*> last_kind{"off"};
};

} // namespace BB
