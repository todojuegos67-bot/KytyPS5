// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-rt-reflect (BB_RT_REFLECT=1, default off; BB_RT_REFLECT_QUALITY=1..3): ray-traced
// reflections composited into the game's HDR scene before the post-process / upscaler / HUD.
//  1. Snapshot (right before the composite draw PS 0x3fe9e964): G-buffer depth, RT0 world normal
//     and RT1 shininess at the trace size (bb_rt_gsnap.comp).
//  2. Run (before the first draw that samples the composite's HDR target - RGBA16F or, as in
//     Bloodborne, B10G11R11 - after the forward pass):
//     depth mesh of the final depth (bb_rt_mesh.comp) -> BLAS + TLAS, classification (water =
//     forward surface nearer than the G-buffer depth and facing up; puddles / glossy from RT1),
//     one ray query per reflective pixel (bb_rt_reflect.comp), temporal + a-trous filter and a
//     depth-aware upsample blended into the HDR scene (bb_rt_post.comp).
// Options: BB_RT_REFLECT_STRENGTH (0..2, 1), _WATER_MIN (0..0.9, 0.35), _RIPPLE (0..0.2, 0.02),
// _PUDDLE (shininess, 0.85), _REFINE=0, _HOOK=<n>; BB_RT_DEBUG_VIEW=5/6/7 or F7 (debug).
// No ray query / acceleration structures: nothing is created, the game renders as before.
// BB_RT_VOLUMETRIC=1 / BB_RT_SHADOWS=1 (each independent of BB_RT_REFLECT) run in RunLighting,
// before the composite draw, on their own depth mesh. BB_RT_SHADOWS also needs BB_VOLFOG=1..3:
// its occlusion is cancelled where the game's cascade shadow (the volfog froxel grid) is dark.
#pragma once

#include <array>
#include <chrono>
#include <string>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}

namespace BB {

class RtReflect {
public:
    // G-buffer images at the composite draw (the scene depth bound with the G-buffer MRT).
    struct SnapInput {
        vk::Image depth;
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_has_stencil;
        vk::Image rt0; // GBuffer RT0 (RGB: world normal)
        vk::ImageView rt0_view;
        vk::ImageLayout rt0_layout;
        vk::Image rt1; // GBuffer RT1 (R: shininess), null when not found
        vk::ImageView rt1_view;
        vk::ImageLayout rt1_layout;
        u32 width, height; // depth size
    };
    // The HDR scene after the forward pass and the scene depth (same image as the snapshot's).
    struct PostInput {
        vk::Image hdr;
        vk::ImageView hdr_storage_view; // RGBA16F, or R32UI over a B10G11R11 target (hdr_r11)
        vk::ImageView hdr_view;         // sampled, the target's own format
        bool hdr_r11;
        vk::ImageLayout hdr_layout;
        vk::Image depth;
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_has_stencil;
        u32 hdr_width, hdr_height;
        u32 depth_width, depth_height;
    };

    // bb-rt-light: volumetric light / RT shadows, run before the game's composite draw
    // (PS 0x3fe9e964, which runs with or without the game's SSR) on the lit scene it samples.
    struct LightInput {
        vk::Image depth;
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_has_stencil;
        bool depth_is_copy; // the engine's R32F colour copy: converted to device depth first
        vk::Image normal; // G-buffer RT0, null when not bound
        vk::ImageView normal_view;
        vk::ImageLayout normal_layout;
        vk::Image scene;  // lit scene, RGBA16F
        vk::ImageView scene_storage_view;
        vk::ImageLayout scene_layout;
        u32 depth_width, depth_height;
        u32 scene_width, scene_height;
        // The previous scene render ran this pass too (no skipped frame): temporal history usable.
        bool history_ok;
        // The game's cascade visibility grid of this frame (bb-volfog, TakeGameShadowGate).
        bool gate_ok;
        vk::ImageView gate_view;
        float gate_far;
    };
    // Why RunLighting returned false (per-5 s counters in the rasterizer).
    enum class LightSkip : u8 {
        None,
        NoCamera,      // no 864 B scene block (camera) captured yet
        BadProjection, // projection not invertible
        DepthEncoding, // R32F copy only and its encoding not known yet
        NoLight,       // no validated game light (light-set CB) recently: not run with a guess
        Allocation,    // images / converter / mesh
        Count,
    };
    LightSkip LastSkip() const { return l_skip; }
    bool HistoryRestarted() const { return l_hist_restarted; }
    // A scene render without the volumetric pass (reason), and the 5 s [rt-vol] line (printed
    // whether or not it ran) when due. No-op without BB_RT_VOLUMETRIC=1.
    void VolTick(const char* skip_why) {
        if (skip_why) VolSkip(skip_why);
        LogVolumetric();
    }
    std::string LightSourceText() const;

    explicit RtReflect(const Vulkan::Instance& instance);
    ~RtReflect();

    // Pipelines, buffers and query pools for the features requested by the environment
    // (BB_RT_REFLECT / BB_RT_VOLUMETRIC / BB_RT_SHADOWS). Called once at renderer start.
    bool Init();
    bool LightingOn() const { return ok && ((vol_on && vol_ok) || (sh_on && sh_ok)); }
    bool RunLighting(vk::CommandBuffer cmd, const LightInput& in);
    // One-time readback (5 texels each) of the R32F depth copy and the scene D32 (d32 may be
    // null); logged a few frames later with the camera near / far, and the R32F encoding
    // (raw device depth / view distance / distance over far) chosen from it.
    void ProbeDepth(vk::CommandBuffer cmd, vk::Image r32f, vk::ImageLayout r32f_layout,
                    vk::Image d32, vk::ImageLayout d32_layout, bool d32_stencil, u32 rw, u32 rh,
                    u32 dw, u32 dh);

    bool ReflectOn() const { return ok && refl_on; }
    // Both record into `cmd` outside a render pass and restore the tracked image layouts.
    // Snapshot: false when it did nothing (no pass this frame then).
    bool Snapshot(vk::CommandBuffer cmd, const SnapInput& in);
    // Run: trace, filter and blend into the HDR scene; false when it did nothing (`why` set).
    bool Run(vk::CommandBuffer cmd, const PostInput& in, const char*& why);
    // Frames the rasterizer saw without a pass, by reason (for the 5 s line).
    void NoteSkip(const char* why);
    // (Log) the forward draws into the HDR target between the snapshot and the run.
    void NoteForwardDraws(u32 n) { fwd_draws += n; }
    void NoteWaterDraws(u32 n) { water_draws += n; }
    // The scene depth before the water draws write it (forced depth write, see the rasterizer's
    // BbRtWaterDraw), and its restore after the pass. Layouts restored.
    bool SaveDepth(vk::CommandBuffer cmd, vk::Image depth, vk::ImageLayout layout,
                   vk::Format format, u32 width, u32 height, bool stencil);
    void RestoreDepth(vk::CommandBuffer cmd, vk::Image depth, vk::ImageLayout layout,
                      bool stencil);

private:
    struct Buf {
        vk::Buffer buffer{};
        VmaAllocation alloc{};
        void* mapped = nullptr;
        vk::DeviceAddress address = 0;
        vk::DeviceSize size = 0;
    };
    // A depth mesh and its acceleration structures (one for the reflections, one for the
    // lighting passes: they run at other points of the frame on other depth sizes).
    struct Mesh {
        Buf verts, blas_buf, tlas_buf, scratch, instances;
        vk::AccelerationStructureKHR blas{}, tlas{};
        vk::DeviceSize blas_scratch = 0, tlas_scratch = 0;
        u32 mesh_w = 0, mesh_h = 0, mesh_cell = 0; // grid source size and cell
        u32 gw = 0, gh = 0, tris = 0;
    };
    bool EnsureSize(u32 w, u32 h);
    bool EnsureMesh(Mesh& m, u32 mw, u32 mh, u32 cell);
    void FreeMesh(Mesh& m);
    // Depth mesh -> BLAS -> TLAS for the given (sampleable) depth; params slot already written.
    void BuildMesh(Mesh& m, vk::CommandBuffer cmd, const vk::DescriptorImageInfo& depth_info,
                   const vk::DescriptorBufferInfo& params_info);
    void ReadTimer(vk::QueryPool pool, std::array<bool, 8>& used, u32 slot, double& sum, u32& n);
    bool MakeBuffer(Buf& b, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host,
                    vk::DeviceSize alignment = 0);
    void FreeBuffer(Buf& b);
    struct Img {
        vk::Image image{};
        VmaAllocation alloc{};
        vk::ImageView view{};
    };
    bool MakeImage(Img& i, u32 width, u32 height,
                   vk::Format format = vk::Format::eR16G16B16A16Sfloat);
    bool InitDepthCvt();
    void ReadProbe(const float* proj, float far_dist);
    void FreeImage(Img& i);
    void FreeSized();
    void Log();
    // bb-rt-vol (BB_RT_VOLUMETRIC=1): volumetric light pass, recorded inside Run after the
    // reflections (same TLAS). The lit scene is in GENERAL.
    bool InitVolumetric();
    bool InitShadows();
    void Volumetric(vk::CommandBuffer cmd, vk::ImageView scene_view, u32 scene_w, u32 scene_h,
                    const float* proj, const double* inv, const double* view,
                    const double* reproj, const double* prev_proj_, bool reproj_ok,
                    const vk::DescriptorImageInfo& depth_info);
    void Shadows(vk::CommandBuffer cmd, const LightInput& in, const float* proj,
                 const double* inv, const double* view, const double* reproj,
                 const double* prev_proj_, bool reproj_ok,
                 const vk::DescriptorImageInfo& depth_info,
                 const vk::DescriptorImageInfo& normal_info);
    void LogShadows();
    bool LightDir(const double* view, float out_view[3]);
    bool GameLight();
    void GameFog();
    void LogVolumetric(); // the 5 s [rt-vol] line: ran, or 0 dispatches with the skip reason
    void VolSkip(const char* why);

    const Vulkan::Instance& instance;
    bool tried = false, ok = false;
    bool refl_on = false; // BB_RT_REFLECT=1: reflection pipelines / images
    double ts_period = 1.0;
    vk::Sampler nearest{}, linear{};
    vk::DescriptorSetLayout mesh_set{}, ray_set{}, post_set{}, snap_set{};
    vk::PipelineLayout mesh_layout{}, ray_layout{}, post_layout{}, snap_layout{};
    vk::Pipeline mesh_pipe{}, ray_pipe{}, post_pipe{}, snap_pipe{};
    vk::Pipeline post_r11_pipe{}; // composite into a B10G11R11 HDR target
    // Quality (BB_RT_REFLECT_QUALITY=1..3, default 2): see Init.
    u32 quality = 2, rays = 1, trace_div = 2, spatial_passes = 1;
    float temporal_glossy = 0.15f, temporal_mirror = 0.25f, filter_strength = 1.0f;
    float shine_lo = 0.6f, shine_hi = 0.9f, shine_puddle = 0.85f, rough_lo = 0.3f;
    float max_ray = 150.0f, water_min = 0.35f, ripple = 0.02f;
    // Own images (trace size): snapshot (RGBA32F), raw / temp (RGBA16F), guide x2 (RGBA32F),
    // history x2 (RGBA16F).
    Img snap, raw, tmp;
    std::array<Img, 2> guide, hist;
    bool images_fresh = false;
    // Previous camera (row-major 4x4, column vectors) for the temporal reprojection.
    double prev_view[16]{}, prev_proj[16]{};
    bool have_prev = false;
    u64 frame = 0;
    vk::QueryPool queries{};
    // Snapshot -> run pairing: the run uses only this frame's snapshot; a frame without a run
    // (or a camera cut) restarts the history instead of reprojecting a stale one.
    bool snap_ready = false;
    u64 snap_seq = 0, last_run_seq = 0;
    u32 snap_depth_w = 0, snap_depth_h = 0;
    std::chrono::steady_clock::time_point t0{};
    const char* hist_reset_why = "start";
    u64 hist_resets = 0, skips = 0, fwd_draws = 0, water_draws = 0;
    // Saved scene depth (own image, the scene depth's format; transfer only).
    vk::Image dsave{};
    VmaAllocation dsave_alloc{};
    vk::Format dsave_fmt{};
    u32 dsave_w = 0, dsave_h = 0;
    bool dsave_valid = false;
    const char* last_skip = "none";
    Buf rcounters;

    static constexpr u32 Ring = 8;
    static constexpr vk::DeviceSize ParamSize = 512;
    u32 w = 0, h = 0;   // trace size
    u32 Cell = 4;       // reflection depth-mesh cell (render px)
    Mesh rmesh, lmesh;  // reflections / lighting
    // Lighting pass (composite hook) trace size and camera history.
    u32 lw_ = 0, lh_ = 0;
    double l_prev_view[16]{}, l_prev_proj[16]{};
    bool l_have_prev = false;
    LightSkip l_skip = LightSkip::None;
    bool l_hist_restarted = false;
    float l_prev_light[3]{};
    bool l_have_light = false;
    const char* l_reset_why = "";
    // Restart the temporal accumulation (vol / shadow history) and clear its images.
    void ResetLightHistory(const char* why);
    u32 l_ring = 0;
    Buf params, counters;
    u32 ring_pos = 0;
    std::array<bool, Ring> query_used{};

    // Stats (logged every 5 s).
    u64 dispatches = 0, last_dispatches = 0;
    std::array<u32, 24> last_counts{};
    double gpu_ms_sum = 0.0;
    u32 gpu_ms_n = 0;
    bool logged_camera = false;
    // Strength (BB_RT_REFLECT_STRENGTH, default 1.0) and hit refinement (BB_RT_REFLECT_REFINE=0 off).
    float strength = 1.0f;
    bool refine = true;

    // bb-rt-vol state.
    bool vol_on = false, vol_ok = false, vol_fresh = true;
    float vol_strength = 0.5f, vol_density = 0.02f, vol_dist = 60.0f;
    float vol_color[3]{0.55f, 0.62f, 0.80f};
    u32 vol_steps = 12;
    vk::DescriptorSetLayout vol_set{};
    vk::PipelineLayout vol_layout{};
    vk::Pipeline vol_pipe{};
    Buf vol_params;
    Img vraw;
    std::array<Img, 2> vacc;
    u32 vol_w = 0, vol_h = 0;
    vk::QueryPool vol_queries{};
    std::array<bool, Ring> vol_query_used{};
    u64 vol_dispatches = 0, vol_last_dispatches = 0;
    u64 vol_skips = 0;              // scene renders without a volumetric pass in this 5 s
    const char* vol_skip_why = "none";
    double vol_ms_sum = 0.0;
    u32 vol_ms_n = 0;
    u32 vol_last_counts[2]{};
    int light_ofs = -1;           // scene block float offset of the light direction, -1: none
    int light_forced_ofs = -1;    // BB_RT_VOLUMETRIC_LIGHT_OFS
    bool light_env = false;       // BB_RT_VOLUMETRIC_DIR given
    float light_env_dir[3]{};
    const char* light_src = "none";
    bool light_heuristic = false; // BB_RT_LIGHT_HEURISTIC=1: scene-block search allowed
    bool logged_block = false;
    // Game data (Ghidra, see bb_upscale.h): light-set CB slot 5, fog CB slot 6.
    bool density_env = false, color_env = false;
    bool logged_game_light = false, logged_game_fog = false;
    float game_color[3]{};        // diffuse colour of the chosen directional light (or vol_color)
    float game_dir[3]{};          // its world direction as used (toward the light)
    int game_light_idx = -1;
    float fog_density = 0.0f;     // density at the camera height (1/m)
    float fog_height = 0.0f;      // height falloff 1/H (signed), 0: none
    float fog_tint[3]{1.0f, 1.0f, 1.0f};
    const char* color_src = "fallback (fixed 0.55,0.62,0.80)";
    char game_light_src[96]{};
    float used_dir[3]{};
    float used_color[3]{};
    const char* fog_src = "fallback (BB_RT_VOLUMETRIC_DENSITY default, no height fog)";
    std::chrono::steady_clock::time_point vol_last_log{};
    // bb-fogfx (BB_FOG_FX / BB_VOL, bb_fogfx.h): level 0 = the light-only pass above.
    int fx_level = 0;
    float fx_extra_od = 0.0f, fx_extra_max = 0.0f, fx_noise = 0.0f, fx_halo = 0.0f, fx_g = 0.55f;
    float fx_halo_thr = 1.0f;
    bool fog_game = false;         // the game's fog param has a distance (v0.y = 1/dist > 0)
    float fog_log2_cam = -126.0f;  // log2 of the density at the camera height (no underflow)
    float fog_rgb[3]{};            // game fog colour, v1.rgb (rgb^2)
    bool vol_no_light = false;     // this run has no validated light: no shafts, fog / halo only
    u64 vol_no_light_runs = 0;
    std::chrono::steady_clock::time_point fx_t0{};

    // bb-rt-shadow state (BB_RT_SHADOWS=1).
    bool sh_on = false, sh_ok = false, sh_fresh = true;
    float sh_strength = 0.6f, sh_radius = 0.0047f, sh_dist = 80.0f, sh_amb = 0.5f;
    u32 sh_quality = 2, sh_rays = 2, sh_denoise = 2;
    u64 sh_nogate = 0, sh_nogate_last = 0;
    u32 dbg_view = 0; // BB_RT_DEBUG_VIEW=1..4 (shadow mask, volumetric, depth, N.L); 5-7: reflections
    vk::DescriptorSetLayout sh_set{};
    vk::PipelineLayout sh_layout{};
    vk::Pipeline sh_pipe{};
    Buf sh_params;
    Img sraw;
    std::array<Img, 2> sacc;
    u32 sh_w = 0, sh_h = 0;
    vk::QueryPool sh_queries{};
    std::array<bool, Ring> sh_query_used{};
    u64 sh_dispatches = 0, sh_last_dispatches = 0;
    double sh_ms_sum = 0.0;
    u32 sh_ms_n = 0;
    u32 sh_last_counts[6]{};
    bool sh_normal = false;
    float sh_used_dir[3]{};
    std::chrono::steady_clock::time_point sh_last_log{};
    // Lighting pass totals (mesh build included when the composite hook built it).
    vk::QueryPool l_queries{};
    std::array<bool, Ring> l_query_used{};
    double l_ms_sum = 0.0;
    u32 l_ms_n = 0;
    std::chrono::steady_clock::time_point last_log{};

    // R32F depth copy -> device depth (bb_rt_depthcvt.comp) and its encoding probe.
    int depth_enc = -1;       // 0 raw device depth, 1 view distance, 2 distance / far; -1 unknown
    bool depth_enc_env = false;
    const char* depth_enc_src = "unknown";
    vk::DescriptorSetLayout cvt_set{};
    vk::PipelineLayout cvt_layout{};
    vk::Pipeline cvt_pipe{};
    bool cvt_ok = false, cvt_tried = false;
    Img dcvt;
    u32 dcvt_w = 0, dcvt_h = 0;
    Buf probe;
    int probe_state = 0;      // 0 not taken, 1 recorded (waiting), 2 read
    u32 probe_wait = 0;
    u32 probe_retries = 0;
    bool probe_has_d32 = false;
    u32 probe_rw = 0, probe_rh = 0;
};

// F7 (BB_RT_REFLECT=1): reflection debug view off -> class mask (5) -> reflection only (6) ->
// blend weight (7) -> off. Starts at BB_RT_DEBUG_VIEW when that is 5..7.
void RtReflectCycleDebug();

} // namespace BB
