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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_hud_layer_comp.h"
#include "video_core/renderer_vulkan/bb_ui_composition.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
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
// Why no draw started the HUD cut (logged once a second while no cut happens).
struct CutWhy {
    u64 movie_not_target = 0, target_not_movie = 0, display_buffer = 0, no_scene = 0,
        not_ready = 0, cuts_seen = 0;
    u64 hashes[8]{};
    u32 hash_count = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
} g_why;

void NoteTargetHash(u64 hash) {
    for (u32 i = 0; i < g_why.hash_count; ++i) {
        if (g_why.hashes[i] == hash) return;
    }
    if (g_why.hash_count < 8) g_why.hashes[g_why.hash_count++] = hash;
}

void FlushWhy() {
    const auto now = std::chrono::steady_clock::now();
    if (now - g_why.t0 < std::chrono::seconds(1)) return;
    g_why.t0 = now;
    if (g_why.cuts_seen == 0) {
        std::string hashes;
        for (u32 i = 0; i < g_why.hash_count; ++i) {
            hashes += fmt::format("{}{:#010x}", i ? " " : "", g_why.hashes[i]);
        }
        BB::Upscale::Checkpoint(fmt::format(
            "[hud-cut] no cut this second: UI-movie shader on a non-scene target {}, scene-size "
            "RGBA8 draws by other shaders {} (vs: {}), display buffer {}, scene not ready {}, "
            "upscaler/FG not ready {}",
            g_why.movie_not_target, g_why.target_not_movie, hashes.empty() ? "none" : hashes,
            g_why.display_buffer, g_why.no_scene, g_why.not_ready));
    }
    g_why = CutWhy{};
    g_why.t0 = now;
}

// bb-hud-layer: BB_HUD_CUT=0 turns the HUD-layer cut off (the frame is upscaled with the HUD,
// as before). BB_HUD_TRACE=1 logs one frame's last draws a second. BB_HUD_LAYER_ALPHA=straight
// if the layer is not premultiplied.
bool LayerCutEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_HUD_CUT");
        return !(v != nullptr && v[0] == '0');
    }();
    return on;
}
bool TraceEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_HUD_TRACE"); // on by default for now; 0 silences
        return !(v != nullptr && v[0] == '0');
    }();
    return on;
}
bool LayerStraightAlpha() {
    static const bool on = [] {
        const char* v = std::getenv("BB_HUD_LAYER_ALPHA");
        return v != nullptr && std::strcmp(v, "straight") == 0;
    }();
    return on;
}
// Per second: what the HUD-layer path saw (GPU thread only).
struct LayerWhy {
    u64 movie_layer_draws = 0, candidate_draws = 0, composites = 0, armed = 0, substituted = 0, not_substituted = 0,
        not_ready = 0, pipe_failed = 0, cut_failed = 0, composited = 0;
    VAddr layer = 0;
    u32 layer_w = 0, layer_h = 0;
    vk::Format layer_format{};
    u64 comp_vs = 0, comp_ps = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
} g_layer;

void FlushLayer() {
    const auto now = std::chrono::steady_clock::now();
    if (now - g_layer.t0 < std::chrono::seconds(1)) return;
    BB::Upscale::Checkpoint(fmt::format(
        "[hud-layer] 1 s: 2D layer draws {} (UI-movie {}; last layer {:#x} {}x{} {}), composite draws {} (vs {:#x} "
        "ps {:#x}), armed {}, layer swapped {} / not found in the draw {}, not ready {}, pipeline "
        "failed {}, cut failed {}, HUD drawn over the upscaled scene {} (BB_HUD_CUT=0 disables)",
        g_layer.candidate_draws, g_layer.movie_layer_draws, g_layer.layer, g_layer.layer_w, g_layer.layer_h,
        vk::to_string(g_layer.layer_format), g_layer.composites, g_layer.comp_vs, g_layer.comp_ps,
        g_layer.armed, g_layer.substituted, g_layer.not_substituted, g_layer.not_ready,
        g_layer.pipe_failed, g_layer.cut_failed, g_layer.composited));
    g_layer = LayerWhy{};
    g_layer.t0 = now;
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
    Destroy(clear_layer);
    Destroy(layer_tmp);
    const auto device = instance.GetDevice();
    if (layer_pipe) device.destroyPipeline(layer_pipe);
    if (layer_layout) device.destroyPipelineLayout(layer_layout);
    if (layer_set) device.destroyDescriptorSetLayout(layer_set);
    if (layer_sampler_nearest) device.destroySampler(layer_sampler_nearest);
    if (layer_sampler_linear) device.destroySampler(layer_sampler_linear);
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
    cut_vs = 0;
    layer_pending = false;
    layer_substituted = false;
    layer_color = {};
    layer_depth = {};
    layer_image = {};
    num_cand = 0;
    // [hud-trace] (BB_HUD_TRACE=1): the traced frame's last draws, logged at the next frame start.
    if (trace_frame && trace_total != 0) {
        // The first TraceHead draws of the frame (the display pass and what follows it), then the
        // last draws (from the scene's end to the next display pass), without repeating any.
        const u32 n = u32(trace_ring.size());
        std::string text = fmt::format("[hud-trace] {} draws of one frame; first {}:",
                                       trace_total, u32(trace_head.size()));
        for (const auto& line : trace_head) {
            text += "\n  ";
            text += line;
        }
        const u32 tail = std::min(trace_total, n);
        const u32 start = trace_total > n ? trace_next : 0;
        u32 shown = 0;
        for (u32 i = 0; i < tail; ++i) {
            const u32 index = trace_total - tail + i; // the draw's number in the frame
            if (index < u32(trace_head.size())) continue;
            if (shown++ == 0) text += fmt::format("\n  ... last draws from #{}:", index);
            text += "\n  ";
            text += trace_ring[(start + i) % n];
        }
        BB::Upscale::Checkpoint(text);
    }
    trace_head.clear();
    trace_frame = false;
    trace_total = 0;
    trace_next = 0;
    // [hud-seq]: the frame traced (seq_frame 1) is logged at the next frame start.
    if (seq_frame == 1 && !seq.empty()) {
        BB::Upscale::Checkpoint("[hud-seq] draws into scene-size RGBA8 targets (UI: after the "
                                "cut):" + seq);
    }
    seq.clear();
    static auto t_seq = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - t_seq >= std::chrono::seconds(1)) {
        t_seq = now;
        seq_frame = 1;
        static u32 trace_seconds = 0;
        trace_frame = TraceEnabled() && (trace_seconds++ % 2) == 0; // one frame every 2 s
    } else {
        seq_frame = 0;
    }
}

void UiCut::OnColorTarget(VideoCore::ImageId color) {
    if (ui_phase) return;
    const auto& image = texture_cache.GetImage(color);
    if (image.info.pixel_format == vk::Format::eR8G8B8A8Unorm &&
        RenderTarget(image.info.size.width, image.info.size.height)) {
        ldr_target = color;
    }
}

void UiCut::OnDraw(const DrawInfo& info, VideoCore::ImageId color, VideoCore::ImageId depth,
                   bool native_viewport, SceneDepth& scene) {
    layer_pending = false; // armed by this draw only (a skipped draw never reached AfterDraw)
    layer_substituted = false;
    const u64 vs_hash = info.vs_hash;
    // bb-gpu6 [hud-seq]: once a second, one frame's draws into scene-size RGBA8 targets.
    const bool seq_on = seq_frame == 1;
    if (seq_on && color && seq.size() < 1800) {
        const auto& im = texture_cache.GetImage(color);
        if (Rgba8(im.info.pixel_format) && RenderTarget(im.info.size.width, im.info.size.height)) {
            seq += fmt::format(" {}vs{:#x}/ps{:#x}{}{}{}{}", ui_phase ? "UI:" : "", vs_hash,
                               info.ps_hash, info.samples_scene ? " reads-scene" : "",
                               info.depth_test ? " depth" : "", info.blend ? " blend" : "",
                               color == ldr_target ? "" : " other-target");
        }
    }
    // bb-gpu6: a post pass (reads a scene-size image) into the UI target after a cut started by a
    // non-movie shader: that shader is not the HUD start; it is refused from now on.
    if (ui_phase && cut_vs != 0 && color && color == ui_color && info.samples_scene &&
        num_bad_vs < bad_vs.size() &&
        std::find(bad_vs.begin(), bad_vs.begin() + num_bad_vs, cut_vs) ==
            bad_vs.begin() + num_bad_vs) {
        bad_vs[num_bad_vs++] = cut_vs;
        BB::Upscale::Checkpoint(fmt::format(
            "[hud-cut] post pass vs {:#x} ps {:#x} came after the cut started by vs {:#x}: that "
            "shader no longer starts the HUD cut ({} refused)",
            vs_hash, info.ps_hash, cut_vs, num_bad_vs));
        cut_vs = 0;
    }
    // bb-hud-mask (3rd log, [hud-trace]): in game the HUD starts with a Scaleform stencil mask
    // (vs 0x34e8a281: colour buffer 0 is the scene target, write mask 0, so no colour target is
    // bound) right after the last post pass; the HUD's colour draws follow. That mask draw starts
    // the cut on the scene target, and its stencil goes into the UI's output-size depth.
    bool mask_cut = false;
    if (!ui_phase && !color && depth && ldr_target && !done_this_frame && sr != nullptr &&
        info.cb0_address != 0 && Vulkan::UiComposition::MovieShader(vs_hash) &&
        texture_cache.GetImage(ldr_target).info.guest_address == info.cb0_address) {
        color = ldr_target;
        mask_cut = true;
    }
    const auto mask_depth = [&] {
        if (!mask_cut || !ui_phase || !depth || depth == ui_depth || !ui_color) return;
        const auto& depth_image = texture_cache.GetImage(depth);
        const auto& ui_target = texture_cache.GetImage(ui_color);
        if (depth_image.info.size.width == ui_target.info.size.width &&
            depth_image.info.size.height == ui_target.info.size.height) {
            EnsureUiResources(ui_width, ui_height, ui_format, depth_image.info.pixel_format);
            PrepareUiDepth(depth);
            ui_depth = depth;
        }
    };
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
    // bb-hud-layer: in game the UI-movie shaders draw into a HUD layer of their own (the log: "UI-
    // movie shader on a non-scene target 60" a second, no UI draw into the scene target), and a
    // post pass reading that layer composites it over the scene in the scene target. e022c2aad
    // cut at that pass and let it run into the upscaled image: an opaque full-screen pass, it
    // replaced the upscaled scene with the render-size one (scaling broken). Here the pass runs
    // as the game wrote it, at the render size, with a transparent layer (SubstituteLayer): the
    // scene target gets the picture without the HUD; after the draw (AfterDraw) that picture is
    // upscaled (or copied, native FG) and the real layer is drawn over the output-size image.
    {
        const bool movie_shader = Vulkan::UiComposition::MovieShader(vs_hash);
        const auto& im = texture_cache.GetImage(color);
        const bool scene_target =
            Rgba8(im.info.pixel_format) && RenderTarget(im.info.size.width, im.info.size.height);
        // v2 (the 2nd log: no UI-movie draw at all in game): any full-size 8-bit target that
        // gets blended 2D draws (no depth test, no scene read) is a layer candidate; a draw into
        // the scene target that reads one is the composite.
        const u32 sw = BB::Upscale::g_scene_w, sh = BB::Upscale::g_scene_h;
        const auto f = im.info.pixel_format;
        const bool eight_bit = Rgba8(f) || f == vk::Format::eB8G8R8A8Unorm ||
                               f == vk::Format::eB8G8R8A8Srgb;
        const bool full_size = sw != 0 && sh != 0 && im.info.size.width + 8 >= sw &&
                               im.info.size.height + 8 >= sh;
        const bool layer_like =
            (movie_shader || (info.blend && !info.depth_test && !info.samples_scene)) &&
            eight_bit && full_size;
        if (layer_like && !scene_target && !IsDisplayBuffer(im.info.guest_address) &&
            im.info.guest_address != 0) {
            const VAddr a = im.info.guest_address;
            u32 i = 0;
            while (i < num_cand && cand[i].address != a) ++i;
            if (i == num_cand && num_cand < MaxCandidates) {
                cand[num_cand++] = {a, im.info.size.width, im.info.size.height, 0, vs_hash};
            }
            if (i < num_cand) ++cand[i].draws;
            if (movie_shader) ++g_layer.movie_layer_draws;
            ++g_layer.candidate_draws;
            g_layer.layer = a;
            g_layer.layer_w = im.info.size.width;
            g_layer.layer_h = im.info.size.height;
            g_layer.layer_format = f;
        }
        if (info.samples_hud_layer && info.layer_read != 0 && !movie_shader && scene_target &&
            color == ldr_target && !IsDisplayBuffer(im.info.guest_address)) {
            for (u32 i = 0; i < num_cand; ++i) {
                if (cand[i].address == info.layer_read) {
                    hud_layer = cand[i].address;
                    hud_layer_w = cand[i].width;
                    hud_layer_h = cand[i].height;
                }
            }
            ++g_layer.composites;
            g_layer.comp_vs = vs_hash;
            g_layer.comp_ps = info.ps_hash;
            const bool scaled_now = BB::Upscale::Enabled();
            const bool ready = scaled_now ? BB::Upscale::BackendReady() : BB::FG::WantInputs();
            if (!LayerCutEnabled()) {
                // off: the old behaviour
            } else if (u32 ow = 0, oh = 0;
                       !ready || im.info.pixel_format != vk::Format::eR8G8B8A8Unorm ||
                       (scaled_now && !OutputSize(ow, oh)) ||
                       !EnsureClearLayer(hud_layer_w, hud_layer_h)) {
                ++g_layer.not_ready;
            } else if (!InitLayerPipe()) {
                ++g_layer.pipe_failed;
            } else {
                layer_pending = true;
                layer_substituted = false;
                layer_color = color;
                layer_depth = depth;
                layer_image = {};
                ++g_layer.armed;
                FlushLayer();
                return; // the cut runs after this draw (AfterDraw)
            }
        }
        FlushLayer();
    }
    static const u64 ui_trigger_vs = [] {
        const char* hash = std::getenv("BB_UI_TRIGGER_VS");
        return hash ? std::strtoull(hash, nullptr, 16) : 0ull;
    }();
    const auto& image = texture_cache.GetImage(color);
    const bool listed_movie = (ui_trigger_vs && vs_hash == ui_trigger_vs) ||
                              Vulkan::UiComposition::MovieShader(vs_hash);
    // bb-gpu6 (BB_HUD_TRIGGER): the in-game HUD draws with none of the UI-movie shaders (369 log:
    // "UI-movie shader on a non-scene target 0, scene-size RGBA8 draws by other shaders 210 (vs:
    // 0x1ea5a07e 0x0b0acf50 0x061fbbf6)"). After the scene, a draw into the UI target that reads
    // no scene-size image (not a post pass) and tests no depth starts the HUD: one of those vertex
    // shaders (BB_HUD_VS="0x..,0x.." replaces the list), or with "auto" any blended such draw.
    // A shader followed by a post pass into the UI target is refused afterwards.
    // BB_HUD_TRIGGER=movie: only the UI-movie shaders (gpu5).
    static const int trigger_mode = [] {
        const char* v = std::getenv("BB_HUD_TRIGGER");
        const int m = v == nullptr || std::strcmp(v, "auto") == 0 ? 2
                      : std::strcmp(v, "list") == 0              ? 1
                                                                 : 0;
        BB::Upscale::Checkpoint(fmt::format(
            "[hud-cut] trigger {} (BB_HUD_TRIGGER=auto|list|movie)",
            m == 2 ? "auto: UI-movie shaders, the HUD list, or the first blended 2D draw after "
                     "the scene"
            : m == 1 ? "list: UI-movie shaders and the HUD list"
                     : "movie: UI-movie shaders only"));
        return m;
    }();
    static const std::vector<u64> hud_list = [] {
        std::vector<u64> list{0x1ea5a07e, 0x061fbbf6, 0x0b0acf50};
        if (const char* v = std::getenv("BB_HUD_VS"); v != nullptr && *v != '\0') {
            list.clear();
            std::string text{v};
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = text.find_first_of(", ;", pos);
                const std::string item =
                    text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                if (!item.empty()) list.push_back(std::strtoull(item.c_str(), nullptr, 16));
                if (end == std::string::npos) break;
                pos = end + 1;
            }
        }
        std::string shown;
        for (const u64 h : list) shown += fmt::format(" {:#x}", h);
        BB::Upscale::Checkpoint(fmt::format("[hud-cut] HUD vertex shaders:{}", shown));
        return list;
    }();
    const bool refused = std::find(bad_vs.begin(), bad_vs.begin() + num_bad_vs, vs_hash) !=
                         bad_vs.begin() + num_bad_vs;
    const bool flat = !info.samples_scene && !info.depth_test;
    const bool hud_start =
        !listed_movie && trigger_mode != 0 && flat && !refused &&
        (std::find(hud_list.begin(), hud_list.end(), vs_hash) != hud_list.end() ||
         (trigger_mode == 2 && info.blend));
    const bool movie = listed_movie || hud_start;
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
    {
        const bool target = Rgba8(image.info.pixel_format) &&
                            RenderTarget(image.info.size.width, image.info.size.height);
        if (movie && !target) ++g_why.movie_not_target;
        if (!movie && target && !IsDisplayBuffer(image.info.guest_address)) {
            ++g_why.target_not_movie;
            NoteTargetHash(vs_hash);
        }
        if (movie && target && IsDisplayBuffer(image.info.guest_address)) ++g_why.display_buffer;
        if (ui_draw && !(scene.valid && ldr_target == color)) ++g_why.no_scene;
        if (ui_draw && !active) ++g_why.not_ready;
        if (ui_draw) ++g_why.cuts_seen;
        FlushWhy();
    }
    if (ui_draw && hud_start && scene.valid && ldr_target == color && active) {
        static u32 said = 0;
        if (said < 4) {
            ++said;
            BB::Upscale::Checkpoint(fmt::format(
                "[hud-cut] HUD start by vs {:#x} ps {:#x} (not a UI-movie shader){}", vs_hash,
                info.ps_hash, info.blend ? ", blended" : ""));
        }
    }
    const u64 trigger_vs = hud_start ? vs_hash : 0;
    switch (Vulkan::UiComposition::Choose(scaled || native_fg, ui_draw,
                                          scene.valid && ldr_target == color, active)) {
    case Vulkan::UiComposition::Background::None:
        return;
    case Vulkan::UiComposition::Background::Temporal:
        if (native_fg ? RunNativeFg(scene, color, depth) : RunScaled(scene)) {
            cut_vs = trigger_vs;
            CaptureHudlessUi();
            mask_depth();
            return;
        }
        if (ui_phase) {
            mask_depth();
            return; // native: the vectors failed but the UI copy is in place
        }
        // A failed dispatch must not force UI text back to the scene resolution.
        [[fallthrough]];
    case Vulkan::UiComposition::Background::Copy:
        RunUiOnly(color, depth);
        cut_vs = trigger_vs;
        CaptureHudlessUi();
        mask_depth();
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

bool UiCut::InitLayerPipe() {
    if (layer_pipe_tried) return layer_pipe_ok;
    layer_pipe_tried = true;
    const auto device = instance.GetDevice();
    const auto sampler = [&](vk::Filter filter, vk::Sampler& out) {
        auto [r, smp] = device.createSampler({
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxLod = 0.0f,
        });
        if (r != vk::Result::eSuccess) return false;
        out = smp;
        return true;
    };
    if (!sampler(vk::Filter::eNearest, layer_sampler_nearest) ||
        !sampler(vk::Filter::eLinear, layer_sampler_linear)) {
        return false;
    }
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {2, vk::DescriptorType::eStorageImage, 1, C},
    }};
    auto [lr, set] = device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (lr != vk::Result::eSuccess) return false;
    layer_set = set;
    const vk::PushConstantRange range{C, 0, 16};
    auto [pr, pl] = device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &layer_set,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) return false;
    layer_layout = pl;
    const vk::ShaderModule module = Vulkan::CompileSPV(BB_HUD_LAYER_COMP, device);
    if (!module) return false;
    auto [cr, pipe] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage{.stage = C, .module = module, .pName = "main"},
                .layout = layer_layout,
            });
    device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) return false;
    layer_pipe = pipe;
    layer_pipe_ok = true;
    BB::Upscale::Checkpoint(fmt::format(
        "[hud-layer] HUD-layer cut ready: the composite pass gets a transparent layer, the scene "
        "is upscaled without the HUD, the layer is drawn over the output ({} alpha)",
        LayerStraightAlpha() ? "straight" : "premultiplied"));
    return true;
}

bool UiCut::EnsureClearLayer(u32 w, u32 h) {
    if (w == 0 || h == 0) return false;
    if (clear_layer.image && clear_layer.width == w && clear_layer.height == h) return true;
    if (clear_layer.image) scheduler.Finish();
    if (!Create(clear_layer, w, h, vk::Format::eR8G8B8A8Unorm,
                vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, false)) {
        return false;
    }
    if (!Mirror(clear_layer, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor)) {
        Destroy(clear_layer);
        return false;
    }
    scheduler.EndRendering();
    const auto cmd = scheduler.CommandBuffer();
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = clear_layer.image,
        .subresourceRange = range,
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    cmd.clearColorImage(clear_layer.image, vk::ImageLayout::eTransferDstOptimal,
                        vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}, range);
    b.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    b.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    b.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    b.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    b.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    b.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    BB::Upscale::Checkpoint(fmt::format("[hud-layer] transparent stand-in layer {}x{}", w, h));
    return true;
}

bool UiCut::SubstituteLayer(VideoCore::ImageId image, vk::ImageView& view,
                            vk::ImageLayout& layout) {
    if (!layer_pending || hud_layer == 0 || !image || !clear_layer.image) return false;
    const auto& im = texture_cache.GetImage(image);
    if (im.info.guest_address != hud_layer) return false;
    const vk::ImageView v =
        Mirror(clear_layer, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
    if (!v) return false;
    view = v;
    layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    layer_image = image;
    layer_substituted = true;
    return true;
}

void UiCut::AfterDraw(SceneDepth& scene) {
    if (!layer_pending) return;
    layer_pending = false;
    const VideoCore::ImageId color = layer_color, depth = layer_depth, layer = layer_image;
    const bool substituted = layer_substituted;
    layer_substituted = false;
    layer_color = {};
    layer_depth = {};
    layer_image = {};
    if (!substituted || !color || !layer) {
        // The draw read the HUD as the game drew it: the frame stays as before (no cut).
        ++g_layer.not_substituted;
        return;
    }
    ++g_layer.substituted;
    // From here the scene target has no HUD: the cut must run and the layer must be drawn over.
    const bool scaled = BB::Upscale::Enabled();
    const bool native_fg = !scaled && BB::FG::WantInputs();
    const bool active = scaled ? BB::Upscale::BackendReady() : native_fg;
    // Every upscaler goes through PresentSr::Run (DLSS, FSR 4, FSR 3.1, XeSS, TAA, native AA);
    // without a usable one the HUD-less scene is copied (scaled linearly) as for menus.
    switch (Vulkan::UiComposition::Choose(scaled || native_fg, true,
                                          scene.valid && ldr_target == color, active)) {
    case Vulkan::UiComposition::Background::None:
        break;
    case Vulkan::UiComposition::Background::Temporal:
        if (native_fg ? RunNativeFg(scene, color, depth) : RunScaled(scene)) break;
        if (ui_phase) break;
        [[fallthrough]];
    case Vulkan::UiComposition::Background::Copy:
        RunUiOnly(color, depth);
        break;
    }
    if (!ui_phase) {
        ++g_layer.cut_failed;
        std::scoped_lock lock{counter_mutex};
        ++counters.layer_failed;
        return;
    }
    cut_vs = 0;
    CaptureHudlessUi(); // the picture without the HUD for frame generation
    const bool drawn = CompositeLayer(layer);
    if (drawn) ++g_layer.composited;
    std::scoped_lock lock{counter_mutex};
    ++counters.layer_cuts;
    if (drawn) {
        ++counters.layer_composites;
    } else {
        ++counters.layer_failed;
    }
}

bool UiCut::CompositeLayer(VideoCore::ImageId layer_id) {
    if (!layer_pipe_ok || !layer_id || !ui_image.image || ui_width == 0 || ui_height == 0 ||
        ui_format != vk::Format::eR8G8B8A8Unorm) {
        return false;
    }
    if (!layer_tmp.image || layer_tmp.width != ui_width || layer_tmp.height != ui_height) {
        if (layer_tmp.image) scheduler.Finish();
        if (!Create(layer_tmp, ui_width, ui_height, vk::Format::eR8G8B8A8Unorm,
                    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                    false)) {
            return false;
        }
    }
    const vk::ImageView scene_view =
        Mirror(ui_image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView out_view =
        Mirror(layer_tmp, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
    if (!scene_view || !out_view) return false;
    scheduler.EndRendering();
    auto& layer = texture_cache.GetImage(layer_id);
    VideoCore::ImageViewInfo li{};
    li.format = layer.info.pixel_format;
    const vk::ImageView layer_view = *layer.FindView(li, false).image_view;
    if (!layer_view) return false;
    runtime.Transit(&layer, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const auto cmd = scheduler.CommandBuffer();
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    vk::ImageMemoryBarrier2 b[2]{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = ui_image.image,
            .subresourceRange = range,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = layer_tmp.image,
            .subresourceRange = range,
        }};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = b});
    const vk::DescriptorImageInfo scene_info{layer_sampler_nearest, scene_view,
                                             vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo hud_info{layer_sampler_linear, layer_view,
                                           vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{{}, out_view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &scene_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &hud_info},
        vk::WriteDescriptorSet{.dstBinding = 2,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &out_info},
    };
    const u32 flags = (LayerStraightAlpha() ? 1u : 0u) | (IsSrgb(li.format) ? 2u : 0u);
    const u32 push[4] = {ui_width, ui_height, flags, 0u};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, layer_pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layer_layout, 0, writes);
    cmd.pushConstants(layer_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
    cmd.dispatch((ui_width + 7) / 8, (ui_height + 7) / 8, 1);
    // The result back into the UI image (the display pass of the next frame reads it).
    b[0] = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = ui_image.image,
        .subresourceRange = range,
    };
    b[1] = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = layer_tmp.image,
        .subresourceRange = range,
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = b});
    const vk::ImageCopy copy{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {ui_width, ui_height, 1},
    };
    cmd.copyImage(layer_tmp.image, vk::ImageLayout::eTransferSrcOptimal, ui_image.image,
                  vk::ImageLayout::eTransferDstOptimal, copy);
    b[0] = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = ui_image.image,
        .subresourceRange = range,
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = b});
    scheduler.GetDynamicState().Invalidate();
    return true;
}

void UiCut::TraceDraw(std::string&& line) {
    if (!trace_frame) return;
    constexpr u32 TraceHead = 48;
    if (trace_total < TraceHead) trace_head.push_back(fmt::format("#{} {}", trace_total, line));
    constexpr u32 Keep = 64;
    if (trace_ring.size() != Keep) {
        trace_ring.assign(Keep, std::string{});
        trace_next = 0;
    }
    trace_ring[trace_next] = fmt::format("#{} {}", trace_total, line);
    trace_next = (trace_next + 1) % Keep;
    ++trace_total;
}

UiCut::Counters UiCut::TakeCounters() {
    std::scoped_lock lock{counter_mutex};
    const Counters c = counters;
    counters = {};
    return c;
}

} // namespace BB
