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

    // Per-second counters (the GPU thread builds, the present thread logs).
    std::atomic<u32> sec_fg{0}, sec_layer{0}, sec_hudless{0}, sec_static{0}, sec_overlaid{0},
        sec_fsr3{0};
    std::atomic<const char*> last_kind{"off"};
};

} // namespace BB
