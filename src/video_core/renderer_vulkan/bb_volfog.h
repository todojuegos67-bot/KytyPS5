// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-volfog (BB_VOLFOG=0..3: 0 off, 1 sutil, 2 normal, 3 denso): froxel volumetric fog in the
// game's HDR scene, before the post-process / upscaler / HUD. No ray tracing needed.
//  - Grid: W x H screen tiles x D radial slices (distance = far * s^2, far = BB_VOLFOG_DIST,
//    100 m), RGBA16F; temporal reprojection with a per-frame slice jitter.
//  - Medium: exponential height haze + low ground mist above a ground reference that follows the
//    camera, broken up by 3D noise carried by the wind (BB_VOLFOG_WIND, see the .cpp for the
//    game's WindWorld). The game's fog CB (slot 6) only tints the ambient when it is fresh.
//  - Moon: direction and shadows from the game's cascade shadow map: the 4096^2 atlas sampled by
//    the deferred shadow-mask pass and FC_ProjSpaceToShadowMatrix[4] / FC_CascadeSelectDist of
//    the 864-byte scene block (floats 112-175 / 176-179), resolved and checked at run time;
//    colour from the light-set CB (slot 5, FUN_026deeb0) light 0.
//  - Point lights: the game's deferred light draws (VS 0x8747a367, GXLightAcc point-light pixel
//    shaders): view-space position, 1/r^2 and diffuse colour of every light of the frame.
// Hooks (vk_rasterizer.cpp): inject before the shadow-mask pass, light collection in
// BindResources, arming at the composite (PS 0x3fe9e964), apply before the first draw that
// reads the composite's HDR target (as bb-rt-reflect, commit c74794af4).
#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}
namespace BB::Upscale {
struct DrawCb;
}

namespace BB {

class VolFog {
public:
    // BB_VOLFOG (0..3); 0 when unset.
    static int Level();
    // BB_RT_SHADOWS=1 with BB_VOLFOG off: the cascade grid only (no fog, no shafts, no Apply).
    static bool GateOnly();
    // A GXLightAcc point-light pixel shader (its buffers carry the frame's lamps).
    static bool IsPointLightPs(u64 hash);

    struct InjectInput {
        vk::Image atlas;
        vk::ImageView atlas_view; // depth aspect, sampled as float
        vk::ImageLayout atlas_layout;
        bool atlas_stencil;
        u32 atlas_w, atlas_h;
        u64 atlas_addr;
        u64 ps_hash;
    };
    struct ApplyInput {
        vk::Image hdr;
        vk::ImageView hdr_storage_view; // RGBA16F, or R32UI over a B10G11R11 target
        bool hdr_r11;
        vk::ImageLayout hdr_layout;
        u32 hdr_w, hdr_h;
        vk::Image depth;
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_stencil;
        u32 depth_w, depth_h;
        bool injected; // the inject ran for this frame (moon shadows valid)
    };

    explicit VolFog(const Vulkan::Instance& instance);
    ~VolFog();
    bool Init();
    bool Ok() const {
        return ok;
    }

    // Every draw's read-only VS / PS buffers (BindResources): the deferred point-light draws.
    void OnDrawCbs(const Upscale::DrawCb* cbs, size_t n);
    // The composite draw: the frame's light draws are over.
    void FrameLightsDone();
    bool Inject(vk::CommandBuffer cmd, const InjectInput& in, const char*& why);
    struct GameShadowGate {
        vk::ImageView view{};
        float far_m = 0.0f;
        const char* source = "none";
    };
    // The game's cascade visibility of this frame's inject (froxel grid, R16F 3D, 1 = lit), once
    // per inject that resolved the cascades; false when there is none (no mask pass, interiors).
    bool TakeGameShadowGate(GameShadowGate& g);
    bool Apply(vk::CommandBuffer cmd, const ApplyInput& in, const char*& why);
    void NoteSkip(const char* why);
    void NoteHdrReader(u64 ps_hash) {
        last_hook_ps = ps_hash;
    }

private:
    struct Buf {
        vk::Buffer buffer{};
        VmaAllocation alloc{};
        void* mapped = nullptr;
        vk::DeviceSize size = 0;
    };
    struct Img {
        vk::Image image{};
        VmaAllocation alloc{};
        vk::ImageView view{};
    };
    bool MakeBuffer(Buf& b, vk::DeviceSize size, vk::BufferUsageFlags usage);
    void FreeBuffer(Buf& b);
    bool MakeImage3D(Img& i, vk::Format format);
    void FreeImage(Img& i);
    // The camera of this frame from the 864-byte scene block (false: none / not invertible).
    bool Camera();
    void ResolveCascades();
    void MoonAndAmbient();
    void ReadTimers();
    void InitImages(vk::CommandBuffer cmd);
    void Log(bool force);
    std::string WindProbe();

    const Vulkan::Instance& instance;
    bool tried = false, ok = false;
    int level = 0;
    const char* level_name = "off";
    u32 gw = 160, gh = 90, gd = 64;
    float far_dist = 100.0f;
    double ts_period = 1.0;
    // Medium / lighting knobs (preset x BB_VOLFOG_* overrides).
    float high_dens = 0.0f, high_h = 40.0f, high_thick = 8.0f, edge_k = 1.0f;
    float haze = 0.008f, haze_h = 30.0f, mist = 0.04f, mist_h = 1.2f, noise_amp = 0.6f,
          noise_freq = 0.12f, scatter = 0.8f, albedo = 0.85f, g_moon = 0.6f, g_lights = 0.3f,
          temporal = 0.12f, ground_below = 1.8f, unshadowed_moon = 0.35f, lights_scale = 3.0f;
    float shaft_scale = 1.0f, cap_rel = 0.25f, light_cap_rel = 3.0f, light_max_r = 12.0f,
          game_fog_mul = 1.0f;
    bool lights_all = false;
    float wind[3]{0.55f, 0.0f, 0.25f};
    const char* wind_src = "default (no game wind decoded: fixed 0.6 m/s drift)";
    int shadow_cmp_env = -1; // BB_VOLFOG_SHADOW_CMP
    bool game_fog_tint = true;
    bool wind_probe = false;

    vk::Sampler nearest{}, linear{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    std::array<vk::Pipeline, 7> pipes{}; // inject, light, integrate, apply, apply R11, measure x2
    Img vis;
    std::array<Img, 2> scat;
    Img integ;
    bool fresh = true;
    bool gate_ready = false;
    bool gate_only = false;

    static constexpr u32 Ring = 8;
    static constexpr vk::DeviceSize ParamSize = 1024;
    static constexpr u32 MaxLights = 64;
    Buf params, lights_buf, counters;
    u32 ring = 0;
    vk::QueryPool queries{};
    std::array<bool, Ring> inj_used{}, app_used{};
    u32 inj_slot_last = 0;

    // Camera (row-major, column vectors) of this frame and of the previous applied frame.
    double V[16]{}, P[16]{}, invP[16]{}, invV[16]{}, PV[16]{};
    double prevV[16]{}, prevP[16]{}, prev_cam[3]{};
    bool have_prev = false;
    u64 last_apply_frame = 0;
    float block[216]{};
    bool cam_ok = false;
    u64 frame = 0;
    float frame_jitter = 0.0f;
    bool frame_has_jitter = false; // set by the inject, consumed by the apply
    float cam_y_smooth = 0.0f;
    bool cam_y_init = false;
    std::chrono::steady_clock::time_point t0{}, last_apply{}, last_log{};

    // Cascades (resolved every inject).
    bool casc_ok = false;
    int casc_combo = -1; // 0 proj-space rows, 1 proj-space transposed, 2 view rows, 3 transposed
    bool casc_baked = false, casc_ndc = false, casc_dist_ok = false;
    float casc_score = 0.0f;
    float shadow_w2s[4][16]{};
    float tile[4][4]{};
    float rmap[4]{1.0f, 1.0f, 0.0f, 0.0f};
    float tbound[4][4]{};
    float cdist[4]{};
    int shadow_cmp = 0;
    bool moon_from_casc = false;
    float casc_dir[3]{};
    std::string casc_why = "no inject yet";
    u64 atlas_addr = 0;
    u32 atlas_w = 0, atlas_h = 0;
    u64 inject_ps = 0;

    // Moon / ambient (per apply).
    bool moon_on = false;
    bool shadow_pass = false; // the frame had a shadow-mask pass (inject attempted)
    float moon_dir[3]{}, moon_col[3]{}, amb[3]{};
    const char* moon_src = "none";
    const char* col_src = "none";
    float last_cb_col[3]{};
    bool have_cb_col = false;
    std::string amb_src = "fixed";

    // Point lights: accumulated over the frame's light draws, closed at the composite.
    struct Light {
        float pos[3];
        float r;
        float col[3];
        float lum;
    };
    std::vector<Light> acc_lights, frame_lights;
    u32 acc_rej_big = 0, acc_rej_cold = 0, frame_rej_big = 0, frame_rej_cold = 0;
    // The game's fog (FUN_02651f10 CB of the forward-lit draws): the most used set with fog of
    // this frame, held while frames show none (sky / effects bind a fog-off set).
    struct GameFogSet {
        float amount = 0.0f, rate = 0.0f, start = 0.0f, inv_h = 0.0f;
        float col[3]{};
        float mix = 0.0f;
        float h_amount = 0.0f, h_base = 0.0f, h_scale = 0.0f;
        u32 count = 0;
    };
    std::vector<GameFogSet> acc_fog;
    u32 acc_fog_off = 0, acc_fog_cbs = 0;
    GameFogSet game_fog{};
    bool game_fog_have = false;
    std::chrono::steady_clock::time_point game_fog_seen{};
    float game_fog_weight = 0.0f;
    u32 frame_fog_sets = 0, frame_fog_off = 0, frame_fog_cbs = 0;
    u64 n_fog_frames = 0, n_nofog_frames = 0;
    // Scene log-average luminance (measure pass), read back a few applies later for the log.
    std::array<u32, 4> app_slot_hist{}, inj_slot_hist{};
    u32 inj_hist_pos = 0;
    u32 app_hist_pos = 0;
    float scene_lum_log = 0.0f;
    u32 acc_draws = 0, frame_draws = 0, acc_parsed = 0, frame_parsed = 0;
    u32 acc_cam_cb = 0, frame_cam_cb = 0;

    // Stats (5 s).
    u64 n_apply = 0, n_inject = 0, n_skip = 0, n_noinject = 0;
    const char* last_skip = "none";
    double inj_ms = 0.0, app_ms = 0.0;
    u32 inj_n = 0, app_n = 0;
    u32 lights_max = 0;
    u64 lights_sum = 0;
    u64 last_hook_ps = 0;
    std::string hist_why = "start";
    u64 hist_resets = 0;
    bool logged_start = false;
};

// Shift+F7 (BB_VOLFOG > 0): debug view off -> fog only (in-scatter) -> fog off -> transmittance.
void VolFogCycleDebug();

} // namespace BB
