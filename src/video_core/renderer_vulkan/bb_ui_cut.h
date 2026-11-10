// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Supermedo (github.com/Supermedo/bloodborne_pc @115a0299a)
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-bbport: the HUD cut of Supermedo's Bloodborne port (TemporalUpscaler OnFrameStart,
// OnColorTarget, OnDraw, RunScaled, RunUiOnly, EnsureUiResources, PrepareUiDepth, RedirectColor,
// RedirectDepth, RedirectSampled, CaptureHudlessUi, DisplayOverride), adapted to this build's
// upscaler (bb_present_sr.cpp: DLSS, FSR 4.0, FSR 3.1, XeSS, TAA).
//
// The game draws its tonemapped 3D picture into an RGBA8 target of the render size (the UI
// target), then the UI (Scaleform) into the same target; the first pass of the next frame copies
// it into a display buffer. The upscaler runs right before the first UI draw, on that target
// alone, into `ui_image` (output size); the later UI draws into that target are redirected to
// `ui_image` with an output-size depth of their own (no jitter), so the HUD never enters the
// upscaler's history and is drawn at the output's pixels. The next frame's display pass reads
// `ui_image` and writes an output-size host copy of the display buffer, which the presenter shows.
// The picture without the HUD at that instant is copied for frame generation.
#pragma once

#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "video_core/bb/bb_fg.h"
#include "video_core/renderer_vulkan/bb_present_sr.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace BB {

class UiCut {
public:
    UiCut(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler, Vulkan::Runtime& runtime,
          VideoCore::TextureCache& texture_cache);
    ~UiCut();

    void SetUpscaler(PresentSr* sr_) {
        sr = sr_;
    }
    // A presented address (the presenter's flip): a display buffer (FrameCapture::IsDisplayBuffer).
    static void NoteDisplayBuffer(VAddr address);
    static bool IsDisplayBuffer(VAddr address);
    // The cut runs with an upscaler, or natively when frame generation wants its inputs (its
    // HUD-less image and the frame's vectors at the cut). Native without FG: no cut.
    static bool Active();

    // The draw into a display buffer (the game's first pass of a frame) starts a frame.
    void OnFrameStart();
    void OnColorTarget(VideoCore::ImageId color);
    void OnDraw(u64 vs_hash, VideoCore::ImageId color, VideoCore::ImageId depth,
                bool native_viewport, SceneDepth& scene);

    struct Target {
        vk::ImageView view{};
        vk::ImageLayout layout{};
        u32 width = 0, height = 0;
        bool native_ui = false;
    };
    bool RedirectColor(VideoCore::ImageId color, const VideoCore::ImageViewInfo& view,
                       Target& target);
    bool RedirectDepth(VideoCore::ImageId depth, Target& target);
    [[nodiscard]] bool RedirectsSampled(VideoCore::ImageId image) const {
        return display_redirect && image == ui_color;
    }
    bool RedirectSampled(VideoCore::ImageId image, vk::Format format, vk::ImageView& view,
                         vk::ImageLayout& layout);
    // The UI is not jittered.
    [[nodiscard]] bool UiPhase() const {
        return ui_phase;
    }

    // The presenter: the host copy of display buffer `address` (output size, General layout) when
    // the frame was upscaled at the HUD cut; its view in `format`.
    struct Display {
        vk::Image image{};
        vk::ImageView view{};
        vk::Format format{};
        u32 width = 0, height = 0;
        // The frame's frame generation inputs and the picture without the HUD (output size,
        // the UI target's format; viewed sRGB like `format`), null when none.
        BB::FG::Inputs fg{};
        vk::Image hudless{};
        vk::ImageView hudless_view{};
    };
    bool DisplayOverride(VAddr address, vk::Format format, Display& display);

    // Per-second log of the presenter: frames cut, HUD-less copies made.
    struct Counters {
        u32 cuts = 0, ui_only = 0, hudless = 0, displays = 0;
    };
    Counters TakeCounters();

private:
    struct OwnImage {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::Format format{};
        u32 width = 0, height = 0;
        struct View {
            vk::Format format;
            bool alpha_one;
            vk::ImageView view;
        };
        std::vector<View> views;
    };
    bool Create(OwnImage& img, u32 w, u32 h, vk::Format format, vk::ImageUsageFlags usage,
                bool mutable_format);
    void Destroy(OwnImage& img);
    vk::ImageView Mirror(OwnImage& img, vk::Format format, vk::ImageAspectFlags aspect,
                         bool alpha_one = false);
    bool RunScaled(SceneDepth& scene);
    bool RunNativeFg(SceneDepth& scene, VideoCore::ImageId color, VideoCore::ImageId depth);
    void RunUiOnly(VideoCore::ImageId color, VideoCore::ImageId depth);
    void EnsureUiResources(u32 w, u32 h, vk::Format color, vk::Format depth);
    void PrepareUiDepth(VideoCore::ImageId depth);
    void CaptureHudlessUi();
    bool OutputSize(u32& w, u32& h) const;
    [[nodiscard]] bool RenderTarget(u32 w, u32 h) const;

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    PresentSr* sr = nullptr;

    bool ui_phase = false, display_redirect = false, done_this_frame = false;
    bool ui_read_barrier = false, hudless_taken = false;
    VideoCore::ImageId ldr_target{}, ui_color{}, ui_depth{};
    OwnImage ui_image, ui_depth_image;
    // Three copies without the HUD: the presenter reads one a frame later than it was cut.
    static constexpr u32 HudlessSlots = 3;
    OwnImage hudless_images[HudlessSlots];
    u32 hudless_next = 0;
    vk::ImageView ui_depth_view{};
    u32 ui_width = 0, ui_height = 0;
    vk::Format ui_format{}, ui_depth_format{};
    // This frame's frame generation inputs (taken at the cut), handed to the display copy.
    BB::FG::Inputs cut_fg{}, display_fg{};
    int cut_hudless = -1, display_hudless = -1; // hudless slot, -1: none

    struct HostDisplay {
        OwnImage image;
        bool valid = false;
        BB::FG::Inputs fg{};
        int hudless = -1;
        vk::Format rt_format = vk::Format::eUndefined; // the format the display pass wrote it with
    };
    std::mutex display_mutex;
    std::unordered_map<VAddr, HostDisplay> displays;
    Counters counters{};
    std::mutex counter_mutex;
};

} // namespace BB
