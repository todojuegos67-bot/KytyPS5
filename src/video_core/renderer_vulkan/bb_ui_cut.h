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
#include <atomic>
#include <mutex>
#include <string>
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
    // bb-gpu6: the draw as the HUD-start rule sees it.
    struct DrawInfo {
        u64 vs_hash = 0, ps_hash = 0;
        bool depth_test = false, blend = false;
        bool samples_scene = false; // reads an image of the scene's size (a post pass)
        bool samples_hud_layer = false; // reads the HUD layer (HudLayer())
        VAddr layer_read = 0;           // the layer candidate it reads (IsLayerCandidate)
        bool clip_disabled = false;     // screen-space vertices (2D)
        VAddr cb0_address = 0;          // colour buffer 0 in the registers (bound or not)
    };
    // bb-hud-layer v2: a full-size 8-bit colour target that received blended 2D draws (no depth
    // test, no scene read) this frame: the HUD layer may be any of them.
    [[nodiscard]] bool IsLayerCandidate(VAddr address) const {
        if (address == 0) return false;
        for (u32 i = 0; i < num_cand; ++i) {
            if (cand[i].address == address) return true;
        }
        return false;
    }
    // bb-hud-layer: the target the UI-movie shaders draw into in game (not the scene target):
    // the HUD layer, composited over the scene later by a post pass. 0: none seen.
    [[nodiscard]] VAddr HudLayer() const {
        return hud_layer;
    }
    // bb-hud-layer: the composite pass of this draw reads `image` (the HUD layer): it gets a
    // transparent image instead (the scene target receives the picture without the HUD).
    bool SubstituteLayer(VideoCore::ImageId image, vk::ImageView& view, vk::ImageLayout& layout);
    // bb-hud-layer: after the draw was recorded: the cut on the HUD-less scene (any upscaler,
    // or native FG), then the HUD layer drawn over the output-size image.
    void AfterDraw(SceneDepth& scene);
    // BB_HUD_TRACE=1: one frame a second, its last draws ([hud-trace]).
    [[nodiscard]] bool Tracing() const {
        return trace_frame;
    }
    void TraceDraw(std::string&& line);
    void OnDraw(const DrawInfo& info, VideoCore::ImageId color, VideoCore::ImageId depth,
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
    // `mapping`: the component mapping of the game's own view (its T# dst_sel), which the
    // redirected view does not apply (bb-artifact-fix: the HUD-less copies are viewed with it).
    bool RedirectSampled(VideoCore::ImageId image, vk::Format format,
                         const AmdGpu::CompMapping& mapping, vk::ImageView& view,
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
        // bb-fsr3-hudless: the UI target with the HUD drawn, copied as the display pass started
        // (same size, format and view as `hudless_view`; SHADER_READ_ONLY), null when none.
        vk::ImageView hud_view{};
        // bb-artifact-fix: `hudless_view` / `hud_view` swap red and blue (the display pass's
        // view of the UI target does: BB_HUDLESS_SWAP).
        bool hudless_swapped = false;
    };
    bool DisplayOverride(VAddr address, vk::Format format, Display& display);

    // Per-second log of the presenter: frames cut, HUD-less copies made.
    struct Counters {
        u32 cuts = 0, ui_only = 0, hudless = 0, displays = 0;
        u32 layer_cuts = 0, layer_composites = 0, layer_failed = 0;
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
            bool swap_rb = false;
        };
        std::vector<View> views;
    };
    bool Create(OwnImage& img, u32 w, u32 h, vk::Format format, vk::ImageUsageFlags usage,
                bool mutable_format);
    void Destroy(OwnImage& img);
    vk::ImageView Mirror(OwnImage& img, vk::Format format, vk::ImageAspectFlags aspect,
                         bool alpha_one = false, bool swap_rb = false);
    bool RunScaled(SceneDepth& scene);
    bool RunNativeFg(SceneDepth& scene, VideoCore::ImageId color, VideoCore::ImageId depth);
    void RunUiOnly(VideoCore::ImageId color, VideoCore::ImageId depth);
    void EnsureUiResources(u32 w, u32 h, vk::Format color, vk::Format depth);
    void PrepareUiDepth(VideoCore::ImageId depth);
    void CaptureHudlessUi();
    bool OutputSize(u32& w, u32& h) const;
    bool InitLayerPipe();
    bool EnsureClearLayer(u32 w, u32 h);
    bool CompositeLayer(VideoCore::ImageId layer);
    [[nodiscard]] bool RenderTarget(u32 w, u32 h) const;

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    PresentSr* sr = nullptr;

    // bb-hud-layer state (see AfterDraw).
    VAddr hud_layer = 0;
    struct Candidate {
        VAddr address = 0;
        u32 width = 0, height = 0, draws = 0;
        u64 first_vs = 0;
    };
    static constexpr u32 MaxCandidates = 8;
    std::array<Candidate, MaxCandidates> cand{};
    u32 num_cand = 0;
    u32 hud_layer_w = 0, hud_layer_h = 0;
    bool layer_pending = false, layer_substituted = false;
    VideoCore::ImageId layer_color{}, layer_depth{}, layer_image{};
    OwnImage clear_layer, layer_tmp;
    bool layer_pipe_tried = false, layer_pipe_ok = false;
    vk::Sampler layer_sampler_nearest{}, layer_sampler_linear{};
    vk::DescriptorSetLayout layer_set{};
    vk::PipelineLayout layer_layout{};
    vk::Pipeline layer_pipe{};
    bool trace_frame = false;
    std::vector<std::string> trace_ring;
    std::vector<std::string> trace_head;
    u32 trace_next = 0, trace_total = 0;
    bool ui_phase = false, display_redirect = false, done_this_frame = false;
    bool ui_read_barrier = false, hudless_taken = false;
    // bb-gpu6: the trigger of this frame's cut (0: a UI-movie shader or none) and the vertex
    // shaders refused as HUD start (a post pass into the UI target came after them).
    u64 cut_vs = 0;
    std::array<u64, 16> bad_vs{};
    u32 num_bad_vs = 0;
    u32 seq_frame = 0;
    std::string seq;
    std::string seq_hdr; // (11-13) [fx-seq]: draws into scene-size non-RGBA8 targets (effects layer study)
    VideoCore::ImageId ldr_target{}, ui_color{}, ui_depth{};
    OwnImage ui_image, ui_depth_image;
    // Three copies without the HUD: the presenter reads one a frame later than it was cut.
    static constexpr u32 HudlessSlots = 3;
    OwnImage hudless_images[HudlessSlots];
    // bb-fsr3-hudless: per slot, the UI target with the HUD (copied at the display pass).
    OwnImage hud_images[HudlessSlots];
    std::array<std::atomic<bool>, HudlessSlots> hud_ok{};
    // bb-artifact-fix: per slot, the display pass's view of the UI target swaps red and blue
    // (its T# dst_sel; the copies are raw memory order, so they are viewed swapped too).
    std::array<std::atomic<bool>, HudlessSlots> raw_swap{};
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
