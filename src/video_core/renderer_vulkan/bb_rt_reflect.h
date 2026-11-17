// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-rt-reflect (BB_RT_REFLECT=1, default off): ray-traced reflections replacing the output of
// the game's SSR draw (PS 0x6c2323c2) where a ray hits. After that draw, outside the render pass:
// a depth mesh (bb_rt_mesh.comp) -> BLAS + one-instance TLAS (rebuilt each SSR draw) -> one ray
// query per SSR pixel (bb_rt_reflect.comp) writing into the SSR target. BB_RT_REFLECT_DEBUG=1
// tints the pixels a ray hit. Needs Instance::IsRtReflectEnabled() and the game's camera.
// BB_RT_VOLUMETRIC=1 / BB_RT_SHADOWS=1 (each independent of BB_RT_REFLECT) run in RunLighting,
// before the composite draw, on their own depth mesh unless the reflections built one this frame.
#pragma once

#include <array>
#include <chrono>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}

namespace BB {

class RtReflect {
public:
    struct Input {
        vk::Image target;        // SSR output, RGBA16F
        vk::ImageView target_view; // storage view
        vk::ImageLayout target_layout;
        vk::Image depth;          // D32S8 (depth aspect sampled)
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_has_stencil;
        vk::Image normal;         // RGBA8 G-buffer RT0
        vk::ImageView normal_view;
        vk::ImageLayout normal_layout;
        vk::Image color;          // RGBA16F lit scene
        vk::ImageView color_view;
        vk::ImageView color_storage_view; // BB_RT_VOLUMETRIC=1: in-scattering is added here
        vk::ImageLayout color_layout;
        vk::Image shine;          // G-buffer RT1 (R: shininess), null when not bound
        vk::ImageView shine_view;
        vk::ImageLayout shine_layout;
        u32 width, height;        // SSR target / depth size
        u32 color_width, color_height; // lit scene size
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
    };

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

    // Records the pass into `cmd` (outside a render pass). Image layouts are restored.
    // Returns false when it did nothing (not supported, no camera, an allocation failed).
    bool Run(vk::CommandBuffer cmd, const Input& in);

    // BB_RT_REFLECT_FULLRES=1: the full-res (2x SSR size) RGBA16F result of the last Run, in
    // GENERAL layout, same alpha semantics as the game's SSR target. Null when off / failed.
    bool FullRes() const { return scale > 1; }
    vk::ImageView FullView() const { return full.view; }
    u32 FullWidth() const { return w * scale; }
    u32 FullHeight() const { return h * scale; }

private:
    struct Buf {
        vk::Buffer buffer{};
        VmaAllocation alloc{};
        void* mapped = nullptr;
        vk::DeviceAddress address = 0;
        vk::DeviceSize size = 0;
    };
    bool EnsureSize(u32 w, u32 h);
    bool EnsureMesh(u32 mw, u32 mh, u32 cell);
    // Depth mesh -> BLAS -> TLAS for the given (sampleable) depth; params slot already written.
    void BuildMesh(vk::CommandBuffer cmd, const vk::DescriptorImageInfo& depth_info,
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
    void LogVolumetric();

    const Vulkan::Instance& instance;
    bool tried = false, ok = false;
    bool refl_on = false; // BB_RT_REFLECT=1: reflection pipelines / images
    double ts_period = 1.0;
    bool debug = false;
    vk::Sampler nearest{}, linear{};
    vk::DescriptorSetLayout mesh_set{}, ray_set{}, post_set{};
    vk::PipelineLayout mesh_layout{}, ray_layout{}, post_layout{};
    vk::Pipeline mesh_pipe{}, ray_pipe{}, post_pipe{};
    // Quality (BB_RT_REFLECT_QUALITY=1..3, default 2) and BB_RT_REFLECT_MORE.
    u32 quality = 2, rays = 2;
    float temporal_blend = 0.1f, filter_strength = 1.0f;
    bool more = false, use_shine = true;
    // Own images (RGBA16F, SSR size): raw trace, guide (normal, distance) x2, history x2, temp.
    Img raw, tmp, full;
    u32 scale = 1; // trace scale vs. the SSR size (2 with BB_RT_REFLECT_FULLRES=1)
    std::array<Img, 2> guide, hist;
    bool images_fresh = false;
    // Previous camera (row-major 4x4, column vectors) for the temporal reprojection.
    double prev_view[16]{}, prev_proj[16]{};
    bool have_prev = false;
    u64 frame = 0;
    vk::QueryPool queries{};

    u32 Cell = 4;
    static constexpr u32 Ring = 8;
    static constexpr vk::DeviceSize ParamSize = 512;
    u32 w = 0, h = 0;                      // reflection (SSR) size
    u32 mesh_w = 0, mesh_h = 0, mesh_cell = 0; // depth-mesh grid source size and cell
    u32 gw = 0, gh = 0, tris = 0;
    bool mesh_from_reflect = false; // the reflection pass built this frame's TLAS
    // Lighting pass (composite hook) trace size and camera history.
    u32 lw_ = 0, lh_ = 0;
    double l_prev_view[16]{}, l_prev_proj[16]{};
    bool l_have_prev = false;
    u32 l_ring = 0;
    Buf verts, blas_buf, tlas_buf, scratch, instances, params, counters;
    vk::AccelerationStructureKHR blas{}, tlas{};
    vk::DeviceSize blas_scratch = 0, tlas_scratch = 0;
    u32 ring_pos = 0;
    std::array<bool, Ring> query_used{};

    // Stats (logged every 5 s).
    u64 dispatches = 0, last_dispatches = 0;
    std::array<u32, 8> last_counts{};
    double gpu_ms_sum = 0.0;
    u32 gpu_ms_n = 0;
    bool logged_camera = false, logged_shine = false;
    // Strength (BB_RT_REFLECT_STRENGTH, default 1.5) and hit refinement (BB_RT_REFLECT_REFINE=0 off).
    float strength = 1.5f;
    bool refine = true;

    // bb-rt-vol state.
    bool vol_on = false, vol_ok = false, vol_fresh = true;
    float vol_strength = 1.0f, vol_density = 0.02f, vol_dist = 60.0f;
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
    double vol_ms_sum = 0.0;
    u32 vol_ms_n = 0;
    u32 vol_last_counts[2]{};
    int light_ofs = -1;           // scene block float offset of the light direction, -1: none
    int light_forced_ofs = -1;    // BB_RT_VOLUMETRIC_LIGHT_OFS
    bool light_env = false;       // BB_RT_VOLUMETRIC_DIR given
    float light_env_dir[3]{};
    const char* light_src = "none";
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

    // bb-rt-shadow state (BB_RT_SHADOWS=1).
    bool sh_on = false, sh_ok = false, sh_fresh = true;
    float sh_strength = 0.6f, sh_cone = 0.02f, sh_dist = 80.0f;
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
    u32 sh_last_counts[2]{};
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
    bool probe_has_d32 = false;
    u32 probe_rw = 0, probe_rh = 0;
};

} // namespace BB
