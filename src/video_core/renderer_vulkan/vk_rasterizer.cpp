// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <xxhash.h>
#include "common/debug.h"
#include "core/bb_hud_native_hook.h"
#include "core/bb_ssr_probe.h"
#include "video_core/renderer_vulkan/bb_rt_reflect.h"
#include "video_core/renderer_vulkan/bb_volfog.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/bb/bb_gpu_profile.h"
#include "video_core/bb/bb_mv_engine.h"
#include "video_core/bb/bb_objmv2.h"
#include "video_core/bb/bb_fogfx.h"
#include "video_core/bb/bb_texture.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/bb_ui_composition.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {
void DmaForget(VAddr addr, u64 size); // bb-dma-dedup (below Rasterizer::FillBuffer)
void DmaForgetAll();                  // (a shader writing guest memory through flat addresses)
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

namespace {
bool BbRtAnyRequested();
bool BbRtReflectRequested();
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      bb_cut{instance, scheduler, runtime, texture_cache} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    BB::ObjMv2::Init(instance);
    // bb-rt: pipelines / buffers at start, never lazily during play.
    if (BbRtAnyRequested() && instance.IsRtReflectEnabled()) {
        bb_rt_reflect = std::make_unique<BB::RtReflect>(instance);
        if (!bb_rt_reflect->Init()) {
            LOG_ERROR(Render_Vulkan, "[rt] init failed: RT features off");
        }
    } else if (BbRtReflectRequested()) {
        LOG_WARNING(Render_Vulkan, "[rt-reflect] BB_RT_REFLECT=1 requested but this device has no "
                                   "ray query / acceleration structures: reflections off (the "
                                   "game's own reflections are drawn as usual)");
    } else if (BB::FogFx::Level() > 0) {
        LOG_WARNING(Render_Vulkan,
                    "[fog-fx] BB_FOG_FX={} requested but this device has no ray query / "
                    "acceleration structures: volumetric fog off",
                    BB::FogFx::Level());
    }

    // bb-volfog: froxel volumetric fog (no ray tracing: works on any device).
    if (BB::VolFog::Level() > 0 || BB::VolFog::GateOnly()) {
        bb_volfog = std::make_unique<BB::VolFog>(instance);
        if (!bb_volfog->Init()) {
            LOG_ERROR(Render_Vulkan, "[volfog] init failed: volumetric fog off");
            bb_volfog.reset();
        }
    }

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
}

Rasterizer::~Rasterizer() = default;

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

// BB object motion vectors v2 (bb_objmv2.h): last frame's buffers for a motion pipeline's draw.
void Rasterizer::BbObjMvDraw(const GraphicsPipeline* pipeline, u64 identity, u32 count,
                             u32 instances) {
    if (!BB::ObjMv2::Active()) {
        return;
    }
    const auto& key = pipeline->GetGraphicsKey();
    if (!BB::ObjMv2::IsMotionKey(key.mrt_mask, key.color_buffers[BB::ObjMv2::Attachment]) ||
        !key.write_masks[BB::ObjMv2::Attachment]) {
        return; // not a motion draw, or a Mask draw (attachment 7 only keeps the pass going)
    }
    if (BB::MvEngine::Enabled()) {
        // bb-mv-engine: the pass of this draw (depth target, viewport 0, color target 0).
        const auto& regs = liverpool->regs;
        const auto& vp = regs.viewports[0];
        u32 f[4];
        std::memcpy(&f[0], &vp.xscale, 4);
        std::memcpy(&f[1], &vp.xoffset, 4);
        std::memcpy(&f[2], &vp.yscale, 4);
        std::memcpy(&f[3], &vp.yoffset, 4);
        u64 pk = regs.depth_buffer.DepthAddress();
        pk = pk * 0x9e3779b97f4a7c15ull ^ (u64(f[0]) << 32 | f[1]);
        pk = pk * 0x9e3779b97f4a7c15ull ^ (u64(f[2]) << 32 | f[3]);
        pk = pk * 0x9e3779b97f4a7c15ull ^ regs.color_buffers[0].Address();
        BB::MvEngine::SetPass(pk);
    }
    // bb-mv-fill: an Effect draw writes R, G, B of attachment 7 only (Blend: R, G, A or RGBA).
    u32 effect = 0;
    // bb-mv-candles2: with bb-fog-mv the Effect pipeline writes .w too (alpha func Max, which
    // no Blend pipeline uses): before, those draws were taken for Blend draws.
    if (key.blend_controls[BB::ObjMv2::Attachment].enable != 0 &&
        ((!(key.write_masks[BB::ObjMv2::Attachment] & vk::ColorComponentFlagBits::eA) &&
          (key.write_masks[BB::ObjMv2::Attachment] & vk::ColorComponentFlagBits::eB)) ||
         key.blend_controls[BB::ObjMv2::Attachment].alpha_func ==
             AmdGpu::BlendControl::BlendFunc::Max)) {
        effect = BB::ObjMv2::EffectDraw;
        const auto& bc0 = liverpool->regs.blend_control[0];
        if (bc0.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcAlpha) {
            effect |= BB::ObjMv2::EffectSrcAlpha;
        }
    }
    if (key.blend_controls[BB::ObjMv2::Attachment].enable != 0 &&
        liverpool->regs.depth_control.depth_write_enable) {
        effect |= BB::ObjMv2::EffectSolid; // bb-mv-layer
    }
    BB::ObjMv2::OnDraw(push_data, pipeline->GetStage(Shader::SwStage::Vertex), memory,
                       buffer_cache, scheduler,
                       key.stage_hashes[static_cast<u32>(Shader::SwStage::Vertex)], identity,
                       count, instances, key.blend_controls[BB::ObjMv2::Attachment].enable != 0,
                       bb_cut.UiPhase(), effect);
}

namespace {
// BB_SSR_PROBE=1: one line per draw/dispatch (shaders, colour targets, depth, sampled images).
template <typename R>
std::string BbSsrDesc(const Pipeline& pipeline, const R& regs, std::string kind, bool graphics) {
    std::string d = std::move(kind);
    std::string reads;
    for (const auto* stage : pipeline.GetStages()) {
        if (!stage) {
            continue;
        }
        d += fmt::format(" {}:{:#x}", u32(stage->sw_stage), stage->pgm_hash);
        for (const auto& image_desc : stage->images) {
            const auto tsharp = image_desc.GetSharp(*stage);
            if (tsharp.Address() == 0 || reads.size() > 400) {
                continue;
            }
            reads += fmt::format(" {:#x}:{}x{}:fmt{}/{}", tsharp.Address(), u32(tsharp.width) + 1,
                                 u32(tsharp.height) + 1, u32(tsharp.GetDataFmt()),
                                 u32(tsharp.GetNumberFmt()));
        }
    }
    if (graphics) {
        d += " rt:";
        bool any = false;
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            const auto& col = regs.color_buffers[cb];
            if (!col || regs.color_target_mask.GetMask(cb) == 0) {
                continue;
            }
            any = true;
            d += fmt::format(" cb{}={:#x}:{}x{}:fmt{}/{}", cb, VAddr(col.Address()), col.Pitch(),
                             col.Height(), u32(col.GetDataFmt()), u32(col.GetNumberFmt()));
        }
        if (!any) {
            d += " none";
        }
        if (regs.depth_buffer.DepthValid()) {
            d += fmt::format(" depth={:#x}:{}x{}{}{}", regs.depth_buffer.DepthAddress(),
                             regs.depth_buffer.Pitch(), regs.depth_buffer.Height(),
                             regs.depth_control.depth_enable ? " test" : "",
                             regs.depth_control.depth_write_enable ? " write" : "");
        }
    }
    d += " reads:";
    d += reads.empty() ? " none" : reads;
    return d;
}

// bb-rt-reflect (BB_RT_REFLECT=1, default off): ray-traced reflections (bb_rt_reflect.h).
bool BbRtReflectRequested() {
    static const bool on = [] {
        const char* e = std::getenv("BB_RT_REFLECT");
        return e && e[0] == '1';
    }();
    return on;
}
bool BbRtVolRequested() { // BB_RT_VOLUMETRIC=1 or BB_FOG_FX / BB_VOL (bb_fogfx.h)
    return BB::FogFx::VolRequested();
}
bool BbRtShadowRequested() { // BB_RT_SHADOWS=1 (independent of the others)
    static const bool on = [] {
        const char* e = std::getenv("BB_RT_SHADOWS");
        return e && e[0] == '1';
    }();
    return on;
}
bool BbRtLightRequested() {
    return BbRtVolRequested() || BbRtShadowRequested();
}
// bb-rt-vol: the read-only buffers of the draw being bound (light set / fog search).
thread_local std::vector<BB::Upscale::DrawCb> bb_draw_cbs;
bool BbRtAnyRequested() {
    return BbRtReflectRequested() || BbRtLightRequested();
}
// The lighting / composite draw that samples the SSR target directly (VS 0x8747a367).
constexpr u64 BbCompositePsHash = 0x3fe9e964;
} // namespace

namespace {
VAddr BbAddr(const VideoCore::Image& img) {
    return img.info.guest_address;
}
vk::ImageLayout BbLayoutOf(const VideoCore::Image& img) {
    return img.backing->subresource_states.empty() ? img.backing->state.layout
                                                   : img.backing->subresource_states[0].layout;
}
// BB_RT_REFLECT_HOOK=<n>: run before the n-th draw that reads the HDR target after the composite
// (default 1; the "[rt-reflect] frame N after composite" lines list them). In the user's log
// (build 3141ea13) every frame is: composite, ~17 forward draws into the HDR target (B10G11R11,
// MRT with 0x271080000), then a single reader, PS 0x8650418f, which copies it into a same-size
// RGBA16F target (the start of the post-process): that first reader is the hook.
u32 BbRtReflectHookNth() {
    static const u32 n = [] {
        const char* e = std::getenv("BB_RT_REFLECT_HOOK");
        return e && e[0] >= '1' && e[0] <= '9' ? u32(std::clamp(std::atoi(e), 1, 64)) : 0u;
    }();
    return n;
}
} // namespace

namespace {
// bb-rt-water: the forward draws into the HDR target between the composite and the hook, by PS
// (5 s census), and the F6 list (render thread writes, SDL thread reads).
struct BbFwdPs {
    u64 draws = 0;
    bool half_tex = false; // samples an RGBA8 texture of half the scene size
    bool water = false;
};
std::mutex bb_fwd_mutex;
std::map<u64, BbFwdPs> bb_fwd_census;
std::vector<u64> bb_fwd_list; // the last census' PS hashes (F6 cycles through them)
std::atomic<int> bb_fwd_hide{-1};
u64 bb_fwd_hidden_ps = 0;

// BB_RT_REFLECT_WATER_PS=<hex>[,<hex>...]: the water draws' pixel shaders (replaces the automatic
// rule). BB_RT_REFLECT_WATER_AUTO=0: no automatic rule.
const std::vector<u64>& BbWaterPsList() {
    static const std::vector<u64> list = [] {
        std::vector<u64> v;
        const char* e = std::getenv("BB_RT_REFLECT_WATER_PS");
        while (e && *e) {
            char* end = nullptr;
            const u64 h = std::strtoull(e, &end, 16);
            if (end == e) break;
            if (h) v.push_back(h);
            e = *end ? end + 1 : end;
        }
        return v;
    }();
    return list;
}
bool BbWaterAuto() {
    static const bool on = [] {
        const char* e = std::getenv("BB_RT_REFLECT_WATER_AUTO");
        return !(e && e[0] == '0') && BbWaterPsList().empty();
    }();
    return on;
}
} // namespace

// F6 (BB_RT_REFLECT=1): hides the next forward PS of the census (none after the last), so the
// one that draws the water can be found; logged (BB::RtReflectCycleHide, at the end).
void BbRtCycleHide() {
    std::scoped_lock lk{bb_fwd_mutex};
    int i = bb_fwd_hide.load() + 1;
    if (i >= int(bb_fwd_list.size())) i = -1;
    bb_fwd_hide.store(i);
    bb_fwd_hidden_ps = i >= 0 ? bb_fwd_list[i] : 0;
    if (i >= 0) {
        LOG_INFO(Render_Vulkan, "[rt-water] F6: hiding forward PS {:#x} ({} of {})",
                 bb_fwd_hidden_ps, i + 1, bb_fwd_list.size());
    } else {
        LOG_INFO(Render_Vulkan, "[rt-water] F6: hiding none");
    }
}

// bb-rt-water: before BeginRendering of every draw while this frame's reflection pass is pending.
// A forward draw into the HDR target (depth test on, no depth write) is a water draw when its PS
// is listed (BB_RT_REFLECT_WATER_PS) or, automatically, when it samples an RGBA8 texture of half
// the scene's size. HIPÓTESIS (run_refl1 / run_refl2): that texture is the planar reflection
// rendered by the frame's "Reflect" pass (Ghidra: ReflectCamera, GPU label order Shadow,
// DynShadow, Reflect, ..., View) - 854x480 at 1708x960, 640x360 at 1280x720 - that the water
// material (FresnelMin/Max, ReflectionOffset) distorts. Water draws then write the scene depth
// (Ghidra / logs: they test but never write it), so the pass finds the water surface; the depth
// is saved before the first one and restored after the pass. True: the draw is to be skipped (F6).
bool Rasterizer::BbRtWaterDraw(const GraphicsPipeline& pipeline) {
    bb_rt_water_draw = false;
    if (!bb_rt_refl_pending) {
        return false;
    }
    const auto& regs = liverpool->regs;
    if (!regs.depth_control.depth_enable || regs.depth_control.depth_write_enable ||
        !regs.depth_buffer.DepthValid() ||
        regs.depth_buffer.DepthAddress() != bb_rt_depth_last_addr) {
        return false;
    }
    u64 ps = 0;
    for (const auto* stage : pipeline.GetStages()) {
        if (stage && stage->sw_stage == Shader::SwStage::Fragment) {
            ps = stage->pgm_hash;
        }
    }
    bool to_hdr = false, half_tex = false;
    VideoCore::Image* hdr = nullptr;
    for (const auto& image_id : bound_images) {
        auto& img = texture_cache.GetImage(image_id);
        if (img.binding.is_target && BbAddr(img) == bb_rt_hdr_addr) {
            to_hdr = true;
            hdr = &img;
        }
    }
    if (!to_hdr || !ps) {
        return false;
    }
    for (const auto& image_id : bound_images) {
        const auto& img = texture_cache.GetImage(image_id);
        const auto f = img.info.pixel_format;
        if (!img.binding.is_target &&
            (f == vk::Format::eR8G8B8A8Unorm || f == vk::Format::eR8G8B8A8Srgb) &&
            img.info.size.width == hdr->info.size.width / 2 &&
            img.info.size.height == hdr->info.size.height / 2) {
            half_tex = true;
        }
    }
    const auto& list = BbWaterPsList();
    const bool water = std::ranges::find(list, ps) != list.end() || (BbWaterAuto() && half_tex);
    {
        std::scoped_lock lk{bb_fwd_mutex};
        auto& e = bb_fwd_census[ps];
        ++e.draws;
        e.half_tex |= half_tex;
        e.water |= water;
        if (bb_fwd_hide.load() >= 0 && ps == bb_fwd_hidden_ps) {
            return true;
        }
    }
    if (water && bb_rt_reflect->ReflectOn()) {
        bb_rt_water_draw = true;
        ++bb_rt_water_draws;
    }
    return false;
}

void Rasterizer::BbRtWaterLog() {
    static auto last = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (last != std::chrono::steady_clock::time_point{} && now - last < std::chrono::seconds(5)) {
        return;
    }
    last = now;
    std::scoped_lock lk{bb_fwd_mutex};
    std::vector<std::pair<u64, BbFwdPs>> v(bb_fwd_census.begin(), bb_fwd_census.end());
    std::ranges::sort(v, [](const auto& a, const auto& b) { return a.second.draws > b.second.draws; });
    std::string d;
    u32 n_water = 0;
    for (size_t i = 0; i < v.size() && i < 24; ++i) {
        d += fmt::format(" {:#x} x{}{}{}", v[i].first, v[i].second.draws,
                         v[i].second.half_tex ? " half-res-RGBA8" : "",
                         v[i].second.water ? " WATER" : "");
        n_water += v[i].second.water ? 1 : 0;
    }
    if (!v.empty()) {
        LOG_INFO(Render_Vulkan,
                 "[rt-water] 5 s forward PS into the HDR target (draws; WATER = depth written for "
                 "the reflections, rule {}): {} PS, {} water PS:{}",
                 !BbWaterPsList().empty() ? "BB_RT_REFLECT_WATER_PS"
                 : BbWaterAuto()          ? "auto: samples a half-size RGBA8 (planar reflection)"
                                          : "off",
                 v.size(), n_water, d);
    }
    // F6 list: sorted by hash, stable while the scene does not change much.
    bb_fwd_list.clear();
    for (const auto& [h, e] : bb_fwd_census) bb_fwd_list.push_back(h);
    if (bb_fwd_hide.load() >= int(bb_fwd_list.size())) bb_fwd_hide.store(-1);
    bb_fwd_census.clear();
}

void Rasterizer::BbRtRestoreDepth() {
    if (!bb_rt_depth_saved) {
        return;
    }
    bb_rt_depth_saved = false;
    VideoCore::Image* d32 = texture_cache.TryGetImage(bb_rt_depth_last_id, bb_rt_depth_last_uid);
    if (!d32) {
        return;
    }
    scheduler.EndRendering();
    bb_rt_reflect->RestoreDepth(scheduler.CommandBuffer(), d32->GetImage(), BbLayoutOf(*d32),
                                bool(d32->aspect_mask & vk::ImageAspectFlagBits::eStencil));
}

// bb-rt-reflect: at the composite draw (PS 0x3fe9e964, before it is drawn): the G-buffer
// snapshot (scene depth, RT0, RT1 of the G-buffer MRT) and the composite's colour target, the
// HDR target the forward pass (water: material draw phase "NormalDraw_Water") draws into next.
void Rasterizer::BbRtReflectSnapshot(VideoCore::Image* d32, VideoCore::Image* rt0,
                                     VideoCore::Image* rt1) {
    if (!d32) {
        return; // no depth written since the previous composite: the same frame drawn again
    }
    if (bb_rt_refl_pending) {
        // The previous frame's pass never found its hook.
        bb_rt_reflect->NoteSkip("no draw read the HDR target after the forward pass");
    }
    // (a depth saved but never restored belongs to the previous frame: dropped, never copied
    // over this frame's G-buffer depth)
    bb_rt_depth_saved = false;
    bb_rt_refl_pending = false;
    VideoCore::Image* hdr = nullptr;
    for (const auto& image_id : bound_images) {
        auto& img = texture_cache.GetImage(image_id);
        if (img.binding.is_target && !img.info.props.is_depth &&
            (!hdr || u64(img.info.size.width) * img.info.size.height >
                         u64(hdr->info.size.width) * hdr->info.size.height)) {
            hdr = &img;
        }
    }
    const char* why = !hdr                           ? "composite draw has no colour target"
                      : !rt0                         ? "no G-buffer RT0"
                      : d32->info.num_samples > 1    ? "multisampled scene depth"
                      : rt0->info.props.is_depth     ? "G-buffer RT0 is a depth image"
                                                     : nullptr;
    if (why) {
        bb_rt_reflect->NoteSkip(why);
        return;
    }
    if (rt1 && (rt1->info.size.width != rt0->info.size.width ||
                rt1->info.size.height != rt0->info.size.height || rt1->info.props.is_depth)) {
        rt1 = nullptr;
    }
    VideoCore::ImageViewInfo dv{};
    dv.format = vk::Format::eR32Sfloat; // depth aspect (ImageView picks the depth format)
    VideoCore::ImageViewInfo nv{};
    nv.format = vk::Format::eR8G8B8A8Unorm;
    const BB::RtReflect::SnapInput in{
        .depth = d32->GetImage(),
        .depth_view = *d32->FindView(dv, false).image_view,
        .depth_layout = BbLayoutOf(*d32),
        .depth_has_stencil = bool(d32->aspect_mask & vk::ImageAspectFlagBits::eStencil),
        .rt0 = rt0->GetImage(),
        .rt0_view = *rt0->FindView(nv, false).image_view,
        .rt0_layout = BbLayoutOf(*rt0),
        .rt1 = rt1 ? rt1->GetImage() : vk::Image{},
        .rt1_view = rt1 ? *rt1->FindView(nv, false).image_view : vk::ImageView{},
        .rt1_layout = rt1 ? BbLayoutOf(*rt1) : vk::ImageLayout::eUndefined,
        .width = d32->info.size.width,
        .height = d32->info.size.height,
    };
    scheduler.EndRendering();
    if (!bb_rt_reflect->Snapshot(scheduler.CommandBuffer(), in)) {
        bb_rt_reflect->NoteSkip("snapshot not taken (allocation / pipelines)");
        return;
    }
    if (static bool logged = false; !logged) {
        logged = true;
        LOG_INFO(Render_Vulkan,
                 "[rt-reflect] composite PS {:#x}: HDR target {:#x} {}x{} {}, scene depth {:#x} "
                 "{}x{}, RT0 {:#x}, RT1 {}",
                 BbCompositePsHash, BbAddr(*hdr), hdr->info.size.width, hdr->info.size.height,
                 vk::to_string(hdr->info.pixel_format), BbAddr(*d32), d32->info.size.width,
                 d32->info.size.height, BbAddr(*rt0),
                 rt1 ? fmt::format("{:#x}", BbAddr(*rt1)) : std::string("none"));
    }
    bb_rt_hdr_addr = BbAddr(*hdr);
    bb_rt_water_draws = 0;
    BbRtWaterLog();
    bb_rt_refl_pending = true;
    bb_rt_refl_fwd = 0;
    bb_rt_refl_reads = 0;
}

// bb-rt-reflect: before the first draw after the composite that samples the HDR target without
// rendering into it (the post-process chain: the frame's forward pass and its water are done,
// the HUD and the upscaler come later), the reflections are traced and blended into it.
void Rasterizer::BbRtReflectPost(const GraphicsPipeline& pipeline) {
    if (!bb_rt_refl_pending) {
        return;
    }
    const auto& regs = liverpool->regs;
    VideoCore::Image* hdr = nullptr;
    bool writes_hdr = false;
    for (const auto& image_id : bound_images) {
        auto& img = texture_cache.GetImage(image_id);
        if (BbAddr(img) == bb_rt_hdr_addr && !img.info.props.is_depth) {
            if (img.binding.is_target) {
                writes_hdr = true;
            } else {
                hdr = &img;
            }
        }
    }
    if (!writes_hdr && !hdr) {
        return;
    }
    bb_rt_refl_reads += writes_hdr ? 0 : 1;
    const bool trigger = !writes_hdr && bb_rt_refl_reads >= std::max(BbRtReflectHookNth(), 1u);
    if (bb_rt_refl_chain_logged < 2) {
        // The draws touching the HDR target between the composite and the hook (2 frames).
        static u32 lines = 0;
        if (lines++ < 60) {
            LOG_INFO(Render_Vulkan, "[rt-reflect] frame {} after composite, HDR read #{}: {} -> {}",
                     bb_rt_refl_chain_logged, bb_rt_refl_reads,
                     BbSsrDesc(pipeline, regs, writes_hdr ? "writes HDR" : "reads HDR", true),
                     trigger ? "PASS RUNS HERE"
                     : writes_hdr ? (regs.depth_control.depth_enable ? "forward draw" : "draw")
                                  : "skipped (BB_RT_REFLECT_HOOK)");
        }
        if (trigger) {
            ++bb_rt_refl_chain_logged;
        }
    }
    if (writes_hdr) {
        bb_rt_refl_fwd += regs.depth_control.depth_enable ? 1 : 0;
        if (bb_rt_water_draw && !bb_rt_depth_saved) {
            // The first water draw: the scene depth saved before the water writes into it.
            VideoCore::Image* d32 =
                texture_cache.TryGetImage(bb_rt_depth_last_id, bb_rt_depth_last_uid);
            scheduler.EndRendering();
            if (d32 && bb_rt_reflect->SaveDepth(scheduler.CommandBuffer(), d32->GetImage(),
                                                BbLayoutOf(*d32), d32->info.pixel_format,
                                                d32->info.size.width, d32->info.size.height,
                                                bool(d32->aspect_mask &
                                                     vk::ImageAspectFlagBits::eStencil))) {
                bb_rt_depth_saved = true;
            } else {
                bb_rt_water_draw = false; // no copy to restore from: the water writes no depth
            }
        }
        return;
    }
    if (!trigger) {
        return;
    }
    bb_rt_refl_pending = false;
    bb_rt_reflect->NoteForwardDraws(bb_rt_refl_fwd);
    bb_rt_reflect->NoteWaterDraws(bb_rt_water_draws);
    struct Restore {
        Rasterizer& r;
        ~Restore() { r.BbRtRestoreDepth(); }
    } restore{*this};
    const auto hfmt = hdr->info.pixel_format;
    const bool r11 = hfmt == vk::Format::eB10G11R11UfloatPack32;
    if ((!r11 && hfmt != vk::Format::eR16G16B16A16Sfloat) || hdr->info.num_samples > 1) {
        if (static bool logged = false; !logged) {
            logged = true;
            LOG_WARNING(Render_Vulkan, "[rt-reflect] HDR target {:#x} format {} x{} samples: "
                                       "not RGBA16F / B10G11R11, no pass",
                        BbAddr(*hdr), vk::to_string(hfmt), hdr->info.num_samples);
        }
        bb_rt_reflect->NoteSkip("HDR target format not RGBA16F / B10G11R11 (or multisampled)");
        return;
    }
    VideoCore::Image* d32 = texture_cache.TryGetImage(bb_rt_depth_last_id, bb_rt_depth_last_uid);
    if (!d32 || u64(d32->info.size.width) * hdr->info.size.height !=
                    u64(d32->info.size.height) * hdr->info.size.width) {
        bb_rt_reflect->NoteSkip(d32 ? "scene depth aspect differs from the HDR target"
                                    : "scene depth image gone");
        return;
    }
    // B10G11R11: written through an R32UI storage view (the cache's images are mutable-format
    // with extended usage; R32UI storage is always supported, B10G11R11 storage is optional).
    VideoCore::ImageViewInfo sv{};
    sv.format = r11 ? vk::Format::eR32Uint : vk::Format::eR16G16B16A16Sfloat;
    sv.is_storage = true;
    VideoCore::ImageViewInfo cv{};
    cv.format = hfmt;
    VideoCore::ImageViewInfo dv{};
    dv.format = vk::Format::eR32Sfloat;
    const BB::RtReflect::PostInput in{
        .hdr = hdr->GetImage(),
        .hdr_storage_view = *hdr->FindView(sv, false).image_view,
        .hdr_view = *hdr->FindView(cv, false).image_view,
        .hdr_r11 = r11,
        .hdr_layout = BbLayoutOf(*hdr),
        .depth = d32->GetImage(),
        .depth_view = *d32->FindView(dv, false).image_view,
        .depth_layout = BbLayoutOf(*d32),
        .depth_has_stencil = bool(d32->aspect_mask & vk::ImageAspectFlagBits::eStencil),
        .hdr_width = hdr->info.size.width,
        .hdr_height = hdr->info.size.height,
        .depth_width = d32->info.size.width,
        .depth_height = d32->info.size.height,
    };
    scheduler.EndRendering();
    const char* why = nullptr;
    if (!bb_rt_reflect->Run(scheduler.CommandBuffer(), in, why)) {
        bb_rt_reflect->NoteSkip(why);
    }
}

// bb-rt-light (BB_RT_VOLUMETRIC=1 / BB_RT_SHADOWS=1): right before the game's composite draw
// (PS 0x3fe9e964, which runs whether or not the game's SSR is on), outside the render pass: RT
// shadows darken and volumetric light is added into the lit scene (the largest RGBA16F image the
// composite samples, 1708x960 0x272af0000 in the user's log). RT0: the largest RGBA8 it samples.
// Depth: the composite samples no depth-format image, only the engine's R32F copy of the depth
// (Ghidra: the GX window's "DepthR32F" texture, window+0x78, debug view at 0x0259a9ca). The
// lighting uses the frame's scene depth target (D32, tracked in BeginRendering) when it is still
// alive; else the R32F copy, converted to device depth (encoding from a one-time readback that
// compares both, BB_RT_LIGHT_DEPTH_ENC=0/1/2 forces raw / view distance / distance over far).
void Rasterizer::BbRtLightRun(const GraphicsPipeline& pipeline) {
    bool is_composite = false;
    for (const auto* stage : pipeline.GetStages()) {
        if (stage && stage->sw_stage == Shader::SwStage::Fragment &&
            stage->pgm_hash == BbCompositePsHash) {
            is_composite = true;
        }
    }
    const bool light_on = bb_rt_reflect->LightingOn();
    if (!is_composite || !(light_on || bb_rt_reflect->ReflectOn())) {
        return;
    }
    // Per 5 s status (also when nothing ran, with the reason).
    using Skip = BB::RtReflect::LightSkip;
    static u64 n_seen = 0, n_ran = 0, n_dup = 0, n_nodepth = 0, n_noscene = 0, n_r32f = 0,
               n_d32 = 0, n_d32_stale = 0, n_hist_reset = 0;
    static std::array<u64, size_t(Skip::Count)> n_skip{};
    static auto last_status = std::chrono::steady_clock::time_point{};
    ++n_seen;
    const auto status = [&] {
        if (!light_on) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_status != std::chrono::steady_clock::time_point{} &&
            now - last_status < std::chrono::seconds(5)) {
            return;
        }
        last_status = now;
        LOG_INFO(Render_Vulkan,
                 "[rt-light] 5 s: composite draws {}, lighting ran {} (depth: scene D32 {}, R32F "
                 "copy {}; no depth target written since the previous composite {}), history "
                 "restarts {}; skipped: same scene again without new depth {}, no depth {}, no "
                 "scene {}, no camera {}, bad projection {}, R32F encoding unknown {}, no valid "
                 "game light {}, allocation {}; light: {}",
                 n_seen, n_ran, n_d32, n_r32f, n_d32_stale, n_hist_reset, n_dup, n_nodepth,
                 n_noscene, n_skip[size_t(Skip::NoCamera)], n_skip[size_t(Skip::BadProjection)],
                 n_skip[size_t(Skip::DepthEncoding)], n_skip[size_t(Skip::NoLight)],
                 n_skip[size_t(Skip::Allocation)], bb_rt_reflect->LightSourceText());
        n_seen = n_ran = n_dup = n_nodepth = n_noscene = n_r32f = n_d32 = n_d32_stale =
            n_hist_reset = 0;
        n_skip.fill(0);
    };
    VideoCore::Image* depth = nullptr;  // depth-format image sampled by the draw (none so far)
    VideoCore::Image* zcopy = nullptr;  // R32F colour copy of the depth sampled by the draw
    VideoCore::Image* normal = nullptr;
    VideoCore::Image* scene = nullptr;
    const auto area = [](const VideoCore::Image* i) {
        return i ? u64(i->info.size.width) * i->info.size.height : 0ull;
    };
    std::string desc;
    for (const auto& image_id : bound_images) {
        auto& img = texture_cache.GetImage(image_id);
        const auto f = img.info.pixel_format;
        desc += fmt::format(" [{} {:#x} {}x{} {}]", img.binding.is_target ? "target" : "tex",
                            img.info.guest_address, img.info.size.width, img.info.size.height,
                            vk::to_string(f));
        if (img.info.num_samples > 1) {
            continue;
        }
        if (img.info.props.is_depth) {
            if (!depth || (!img.binding.is_target && depth->binding.is_target)) {
                depth = &img;
            }
        } else if (!img.binding.is_target && f == vk::Format::eR32Sfloat &&
                   area(&img) > area(zcopy)) {
            zcopy = &img;
        } else if (!img.binding.is_target && f == vk::Format::eR16G16B16A16Sfloat &&
                   area(&img) > area(scene)) {
            scene = &img;
        } else if (!img.binding.is_target && f == vk::Format::eR8G8B8A8Unorm &&
                   area(&img) > area(normal)) {
            normal = &img;
        }
    }
    // The scene depth (D32): the largest single-sample depth target written since the previous
    // composite draw (BeginRendering, command-stream order) with the scene image's aspect ratio
    // (shadow maps are larger but square; a half-resolution scene depth is sampled with scaled
    // UVs). Consumed here: a second composite draw without a depth write in between is the same
    // frame drawn again. Without a match the last selected D32 is reused while it lives.
    const bool depth_fresh = bb_rt_depth_fresh;
    bb_rt_depth_fresh = false;
    const auto seen = std::move(bb_rt_depth_seen);
    bb_rt_depth_seen.clear();
    const auto same_aspect = [&](u32 w, u32 h) {
        return scene && u64(w) * scene->info.size.height == u64(h) * scene->info.size.width;
    };
    VideoCore::Image* d32 = nullptr;
    VideoCore::Image* gbuf_rt0 = nullptr;
    VideoCore::Image* gbuf_rt1 = nullptr;
    bool d32_reused = false;
    const char* d32_how = "none";
    if (depth_fresh) {
        // 1) The depth bound with the G-buffer MRT (Ghidra FUN_026c39f0: the scene depth
        //    renderer+0x200 is the depth of GBuffer RT0/RT1, Light Acc and the HDR target; the
        //    frame order is Shadow, DynShadow, Reflect, FeedBackBlur, then View: ZPre, Geom,
        //    Lighting, Composite, Forward, Effect - GPU perf labels at 0x049c2664..0x049c27aa).
        //    Most G-buffer passes wins, then the larger one.
        u32 best_g = 0;
        u64 best_ga = 0;
        for (const auto& c : seen) {
            const u64 ar = u64(c.w) * c.h;
            if (c.gbuf_draws == 0 || c.samples != 1 || !same_aspect(c.w, c.h) ||
                c.gbuf_draws < best_g || (c.gbuf_draws == best_g && ar < best_ga)) {
                continue;
            }
            if (auto* img = texture_cache.TryGetImage(c.id, c.uid)) {
                d32 = img;
                best_g = c.gbuf_draws;
                best_ga = ar;
                gbuf_rt0 = c.rt0_uid ? texture_cache.TryGetImage(c.rt0_id, c.rt0_uid) : nullptr;
                gbuf_rt1 = c.rt1_uid ? texture_cache.TryGetImage(c.rt1_id, c.rt1_uid) : nullptr;
                bb_rt_depth_last_id = c.id;
                bb_rt_depth_last_uid = c.uid;
                bb_rt_depth_last_addr = c.addr;
                d32_how = "G-buffer MRT depth";
            }
        }
        u64 best_area = 0;
        for (const auto& c : seen) {
            if (d32) {
                break; // 2) size heuristic only without a G-buffer pass
            }
            const u64 ar = u64(c.w) * c.h;
            if (!c.depth_write || c.samples != 1 || !same_aspect(c.w, c.h) || ar < best_area) {
                continue;
            }
            if (auto* img = texture_cache.TryGetImage(c.id, c.uid)) {
                d32 = img; // later entries (more recently bound) win ties
                best_area = ar;
                d32_how = "size heuristic (no G-buffer MRT seen)";
                bb_rt_depth_last_id = c.id;
                bb_rt_depth_last_uid = c.uid;
                bb_rt_depth_last_addr = c.addr;
            }
        }
        if (!d32 && bb_rt_depth_last_uid != 0) {
            d32 = texture_cache.TryGetImage(bb_rt_depth_last_id, bb_rt_depth_last_uid);
            if (!d32 && bb_rt_depth_last_addr != 0) {
                texture_cache.ForEachImageInRegion(
                    bb_rt_depth_last_addr, 1, [&](VideoCore::ImageId id, VideoCore::Image& img) {
                        if (!d32 && img.info.props.is_depth && img.info.num_samples == 1 &&
                            img.info.guest_address == bb_rt_depth_last_addr &&
                            same_aspect(img.info.size.width, img.info.size.height)) {
                            d32 = &img;
                            bb_rt_depth_last_id = id;
                            bb_rt_depth_last_uid = img.image_uid;
                        }
                    });
            }
            if (d32 && !same_aspect(d32->info.size.width, d32->info.size.height)) {
                d32 = nullptr;
            }
            d32_reused = d32 != nullptr;
            d32_how = d32 ? "reused" : "none";
        }
    } else {
        ++n_d32_stale;
    }
    {
        // The depth targets seen between two composites, logged for the first few composites
        // and whenever none of them qualified as the scene depth (once).
        static int seen_logged = 0;
        static bool miss_logged = false;
        const bool miss = depth_fresh && (!d32 || d32_reused);
        static const char* last_how = nullptr;
        const bool how_changed = depth_fresh && last_how != d32_how && seen_logged < 12;
        if (depth_fresh) {
            last_how = d32_how;
        }
        if (seen_logged < 2 || how_changed || (miss && !miss_logged)) {
            ++seen_logged;
            miss_logged |= miss;
            std::string list;
            for (const auto& c : seen) {
                list += fmt::format(" [{:#x} {}x{} {} x{} test {} write {}{}{}]", c.addr, c.w,
                                    c.h, vk::to_string(c.format), c.samples,
                                    c.depth_test ? "on" : "off", c.depth_write ? "on" : "off",
                                    same_aspect(c.w, c.h) ? "" : ", other aspect",
                                    c.gbuf_draws ? fmt::format(", G-buffer MRT x{}", c.gbuf_draws)
                                                 : std::string{});
            }
            LOG_INFO(Render_Vulkan,
                     "[rt-light] depth targets since the previous composite (scene {}):{} -> "
                     "scene D32 {}{} [{}], G-buffer RT0 {}",
                     scene ? fmt::format("{}x{}", scene->info.size.width, scene->info.size.height)
                           : std::string("none"),
                     list.empty() ? " none" : list,
                     d32 ? fmt::format("{:#x} {}x{}", d32->info.guest_address,
                                       d32->info.size.width, d32->info.size.height)
                         : std::string("none"),
                     d32_reused ? " (reused from an earlier frame)" : "", d32_how,
                     gbuf_rt0 ? fmt::format("{:#x} {}x{}", gbuf_rt0->info.guest_address,
                                            gbuf_rt0->info.size.width, gbuf_rt0->info.size.height)
                              : std::string("none"));
        }
    }
    // Once per scene image between two scene renders: the composite is drawn twice per frame;
    // a second draw on the same scene image would darken / add twice.
    static VAddr last_scene = 0;
    static int dup_logged = 0;
    const VAddr scene_addr = scene ? scene->info.guest_address : 0;
    if (!depth_fresh && scene_addr == last_scene) {
        ++n_dup;
        if (dup_logged < 1) {
            ++dup_logged;
            LOG_INFO(Render_Vulkan,
                     "[rt-light] composite PS {:#x} drawn again on the scene image {:#x} with no "
                     "depth target written in between (images:{}): the RT lighting runs once per "
                     "scene render",
                     BbCompositePsHash, scene_addr, desc);
        }
        status();
        return;
    }
    if (bb_rt_reflect->ReflectOn()) {
        BbRtReflectSnapshot(d32, gbuf_rt0, gbuf_rt1);
    }
    if (!light_on) {
        return;
    }
    // From here on every exit without a run breaks the temporal history.
    const bool history_ok = !bb_rt_light_gap;
    bb_rt_light_gap = true;
    static bool logged = false;
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan,
                 "[rt-light] composite PS {:#x} images:{} -> scene {}, depth: sampled depth {}, "
                 "R32F copy {}, frame scene D32 {}; RT0 {}",
                 BbCompositePsHash, desc.empty() ? " none" : desc,
                 scene ? fmt::format("{:#x} {}x{}", scene->info.guest_address,
                                     scene->info.size.width, scene->info.size.height)
                       : std::string("none"),
                 depth ? fmt::format("{:#x}", depth->info.guest_address) : std::string("none"),
                 zcopy ? fmt::format("{:#x} {}x{}", zcopy->info.guest_address,
                                     zcopy->info.size.width, zcopy->info.size.height)
                       : std::string("none"),
                 d32 ? fmt::format("{:#x} {}x{} {}", d32->info.guest_address,
                                   d32->info.size.width, d32->info.size.height,
                                   vk::to_string(d32->info.pixel_format))
                     : std::string("none"),
                 normal ? fmt::format("{:#x}", normal->info.guest_address)
                        : std::string("none (depth normals)"));
    }
    if (normal && area(normal) < area(scene) / 4) {
        normal = nullptr;
    }
    // G-buffer RT0 (Ghidra FUN_026c67b0 debug view "GBuffer RT0(RGB:WS Normal)" = renderer
    // +0x250): the world-space normals of the pass the depth came from; the RGBA8 the composite
    // samples need not be the normals.
    if (gbuf_rt0 && !gbuf_rt0->info.props.is_depth && gbuf_rt0->info.num_samples == 1) {
        normal = gbuf_rt0;
    }
    if (!scene) {
        ++n_noscene;
        bb_rt_reflect->VolTick("lighting not run: no scene image");
        status();
        return;
    }
    if (!depth) {
        depth = d32;
    }
    if (!depth && !zcopy) {
        ++n_nodepth;
        bb_rt_reflect->VolTick("lighting not run: no depth");
        status();
        return;
    }
    const auto layout_of = [](const VideoCore::Image& img) {
        return img.backing->subresource_states.empty()
                   ? img.backing->state.layout
                   : img.backing->subresource_states[0].layout;
    };
    VideoCore::ImageViewInfo sv{};
    sv.format = vk::Format::eR16G16B16A16Sfloat;
    sv.is_storage = true;
    VideoCore::ImageViewInfo dv{};
    dv.format = vk::Format::eR32Sfloat; // depth aspect (ImageView picks the depth format)
    VideoCore::ImageViewInfo nv{};
    nv.format = vk::Format::eR8G8B8A8Unorm;
    scheduler.EndRendering();
    const auto cmd = scheduler.CommandBuffer();
    // One-time readback of a few texels of the R32F copy and the D32 (encoding evidence).
    if (zcopy && d32) {
        bb_rt_reflect->ProbeDepth(
            cmd, zcopy->GetImage(), layout_of(*zcopy), d32->GetImage(), layout_of(*d32),
            bool(d32->aspect_mask & vk::ImageAspectFlagBits::eStencil), zcopy->info.size.width,
            zcopy->info.size.height, d32->info.size.width, d32->info.size.height);
    } else if (zcopy) {
        bb_rt_reflect->ProbeDepth(cmd, zcopy->GetImage(), layout_of(*zcopy), vk::Image{},
                                  vk::ImageLayout::eUndefined, false, zcopy->info.size.width,
                                  zcopy->info.size.height, 0, 0);
    }
    const bool use_copy = !depth;
    VideoCore::Image* dimg = use_copy ? zcopy : depth;
    VideoCore::ImageViewInfo cvv{};
    cvv.format = vk::Format::eR32Sfloat;
    BB::VolFog::GameShadowGate gate{};
    const bool gate_ok = bb_volfog && bb_volfog->TakeGameShadowGate(gate);
    const BB::RtReflect::LightInput in{
        .depth = dimg->GetImage(),
        .depth_view = *dimg->FindView(use_copy ? cvv : dv, false).image_view,
        .depth_layout = layout_of(*dimg),
        .depth_has_stencil =
            !use_copy && bool(dimg->aspect_mask & vk::ImageAspectFlagBits::eStencil),
        .depth_is_copy = use_copy,
        .normal = normal ? normal->GetImage() : vk::Image{},
        .normal_view = normal ? *normal->FindView(nv, false).image_view : vk::ImageView{},
        .normal_layout = normal ? layout_of(*normal) : vk::ImageLayout::eUndefined,
        .scene = scene->GetImage(),
        .scene_storage_view = *scene->FindView(sv, false).image_view,
        .scene_layout = layout_of(*scene),
        .depth_width = dimg->info.size.width,
        .depth_height = dimg->info.size.height,
        .scene_width = scene->info.size.width,
        .scene_height = scene->info.size.height,
        .history_ok = history_ok,
        .gate_ok = gate_ok,
        .gate_view = gate.view,
        .gate_far = gate.far_m,
        .gate_src = gate_ok ? gate.source : "none",
    };
    if (bb_rt_reflect->RunLighting(cmd, in)) {
        last_scene = scene_addr;
        bb_rt_light_gap = false;
        ++n_ran;
        ++(use_copy ? n_r32f : n_d32);
        n_hist_reset += bb_rt_reflect->HistoryRestarted() ? 1 : 0;
    } else {
        const auto skip = bb_rt_reflect->LastSkip();
        ++n_skip[size_t(skip)];
        static constexpr std::array<const char*, size_t(Skip::Count)> why{
            "lighting not run", "lighting not run: no camera (scene block)",
            "lighting not run: bad projection", "lighting not run: R32F depth encoding unknown",
            "lighting not run: no validated game light", "lighting not run: allocation"};
        bb_rt_reflect->VolTick(why[size_t(skip)]);
    }
    status();
}

// bb-volfog (BB_VOLFOG=1..3, bb_volfog.h), per draw after the resources are bound:
//  - the deferred shadow-mask pass (ps 0xf55643de in the user's [hud-trace] / [jitter-fs] logs, or
//    any full-screen draw sampling a square >= 2048 depth image and the scene depth): the moon
//    visibility of every froxel is injected from that cascade atlas right before it;
//  - the composite (PS 0x3fe9e964) after a G-buffer pass: the frame's point lights are closed and
//    the apply is armed on its HDR target;
//  - the first draw that samples the HDR target without rendering into it (after the forward
//    pass; PS 0x8650418f in the user's logs): the fog is lit, integrated and applied there.
void Rasterizer::BbVolFogDraw(const GraphicsPipeline& pipeline) {
    u64 ps = 0;
    for (const auto* stage : pipeline.GetStages()) {
        if (stage && stage->sw_stage == Shader::SwStage::Fragment) ps = stage->pgm_hash;
    }
    const auto& regs = liverpool->regs;
    if (ps == BbCompositePsHash) {
        if (!bb_vf_gbuf_seen) {
            return; // drawn again with no G-buffer pass in between: the same frame
        }
        VideoCore::Image* hdr = nullptr;
        for (const auto& image_id : bound_images) {
            auto& img = texture_cache.GetImage(image_id);
            if (img.binding.is_target && !img.info.props.is_depth &&
                (!hdr || u64(img.info.size.width) * img.info.size.height >
                             u64(hdr->info.size.width) * hdr->info.size.height)) {
                hdr = &img;
            }
        }
        if (!hdr) {
            return;
        }
        bb_vf_gbuf_seen = false;
        if (bb_vf_pending) {
            bb_volfog->NoteSkip("no draw read the HDR target after the forward pass");
        }
        bb_vf_pending = true;
        bb_vf_hdr_addr = BbAddr(*hdr);
        ++bb_vf_seq;
        bb_volfog->FrameLightsDone();
        return;
    }
    if (bb_vf_pending) {
        VideoCore::Image* hdr = nullptr;
        bool writes = false;
        for (const auto& image_id : bound_images) {
            auto& img = texture_cache.GetImage(image_id);
            if (BbAddr(img) == bb_vf_hdr_addr && !img.info.props.is_depth) {
                if (img.binding.is_target) {
                    writes = true;
                } else {
                    hdr = &img;
                }
            }
        }
        if (writes || !hdr) {
            return; // the forward pass (or an unrelated draw)
        }
        bb_vf_pending = false;
        const auto hfmt = hdr->info.pixel_format;
        const bool r11 = hfmt == vk::Format::eB10G11R11UfloatPack32;
        if ((!r11 && hfmt != vk::Format::eR16G16B16A16Sfloat) || hdr->info.num_samples > 1) {
            bb_volfog->NoteSkip("HDR target not RGBA16F / B10G11R11 (or multisampled)");
            return;
        }
        VideoCore::Image* d32 = texture_cache.TryGetImage(bb_vf_depth_id, bb_vf_depth_uid);
        if (!d32 || u64(d32->info.size.width) * hdr->info.size.height !=
                        u64(d32->info.size.height) * hdr->info.size.width) {
            bb_volfog->NoteSkip(d32 ? "scene depth aspect differs from the HDR target"
                                    : "scene depth image gone");
            return;
        }
        VideoCore::ImageViewInfo sv{};
        sv.format = r11 ? vk::Format::eR32Uint : vk::Format::eR16G16B16A16Sfloat;
        sv.is_storage = true;
        VideoCore::ImageViewInfo dv{};
        dv.format = vk::Format::eR32Sfloat; // depth aspect (ImageView picks the depth format)
        const BB::VolFog::ApplyInput in{
            .hdr = hdr->GetImage(),
            .hdr_storage_view = *hdr->FindView(sv, false).image_view,
            .hdr_r11 = r11,
            .hdr_layout = BbLayoutOf(*hdr),
            .hdr_w = hdr->info.size.width,
            .hdr_h = hdr->info.size.height,
            .depth = d32->GetImage(),
            .depth_view = *d32->FindView(dv, false).image_view,
            .depth_layout = BbLayoutOf(*d32),
            .depth_stencil = bool(d32->aspect_mask & vk::ImageAspectFlagBits::eStencil),
            .depth_w = d32->info.size.width,
            .depth_h = d32->info.size.height,
            .injected = bb_vf_inject_seq == bb_vf_seq,
        };
        if (!(bb_vf_logged & 1)) {
            bb_vf_logged |= 1;
            LOG_INFO(Render_Vulkan,
                     "[volfog] apply hook: PS {:#x} reads the HDR target {:#x} {}x{} {}; scene "
                     "depth {:#x} {}x{} {}",
                     ps, bb_vf_hdr_addr, in.hdr_w, in.hdr_h, vk::to_string(hfmt), BbAddr(*d32),
                     in.depth_w, in.depth_h, vk::to_string(d32->info.pixel_format));
        }
        bb_volfog->NoteHdrReader(ps);
        scheduler.EndRendering();
        const char* why = nullptr;
        if (!bb_volfog->Apply(scheduler.CommandBuffer(), in, why)) {
            bb_volfog->NoteSkip(why);
        }
        return;
    }
    if (bb_vf_inject_seq == bb_vf_seq + 1) {
        return; // this frame's inject is done
    }
    VideoCore::Image* atlas = nullptr;
    bool scene_depth = false;
    const VideoCore::Image* gdepth = texture_cache.TryGetImage(bb_vf_depth_id, bb_vf_depth_uid);
    for (const auto& image_id : bound_images) {
        auto& img = texture_cache.GetImage(image_id);
        if (img.binding.is_target || img.info.num_samples > 1) {
            continue;
        }
        const u32 w = img.info.size.width, h = img.info.size.height;
        if (img.info.props.is_depth && w == h && w >= 2048) {
            if (!atlas || w > atlas->info.size.width) atlas = &img;
        } else if (gdepth && w == gdepth->info.size.width && h == gdepth->info.size.height &&
                   (img.info.props.is_depth || img.info.pixel_format == vk::Format::eR32Sfloat)) {
            scene_depth = true;
        }
    }
    constexpr u64 ShadowMaskPs = 0xf55643de; // [jitter-fs] / [hud-trace] (commit 3141ea133)
    if (!atlas || !(ps == ShadowMaskPs || (scene_depth && regs.num_indices <= 6))) {
        return;
    }
    bb_vf_inject_seq = bb_vf_seq + 1;
    VideoCore::ImageViewInfo dv{};
    dv.format = vk::Format::eR32Sfloat;
    const BB::VolFog::InjectInput in{
        .atlas = atlas->GetImage(),
        .atlas_view = *atlas->FindView(dv, false).image_view,
        .atlas_layout = BbLayoutOf(*atlas),
        .atlas_stencil = bool(atlas->aspect_mask & vk::ImageAspectFlagBits::eStencil),
        .atlas_w = atlas->info.size.width,
        .atlas_h = atlas->info.size.height,
        .atlas_addr = BbAddr(*atlas),
        .ps_hash = ps,
    };
    if (!(bb_vf_logged & 2)) {
        bb_vf_logged |= 2;
        LOG_INFO(Render_Vulkan,
                 "[volfog] inject hook: shadow-mask PS {:#x} ({}), cascade atlas {:#x} {}x{} {}",
                 ps, ps == ShadowMaskPs ? "known hash" : "full-screen + scene depth + atlas rule",
                 in.atlas_addr, in.atlas_w, in.atlas_h, vk::to_string(atlas->info.pixel_format));
    }
    scheduler.EndRendering();
    const char* why = nullptr;
    if (!bb_volfog->Inject(scheduler.CommandBuffer(), in, why)) {
        bb_volfog->NoteSkip(why);
    }
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    BbUiDraw(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    if (bb_rt_reflect && BbRtWaterDraw(*pipeline)) {
        ResetBindings(false);
        return; // F6: this forward PS hidden (water identification)
    }
    const auto state = BeginRendering(pipeline);
    if (BB::SsrProbe::Enabled()) {
        BB::SsrProbe::OnDraw(BbSsrDesc(*pipeline, regs,
                                       fmt::format("draw{} n{}", is_indexed ? "-idx" : "",
                                                   regs.num_indices),
                                       true));
    }

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    if (bb_rt_reflect) {
        BbRtLightRun(*pipeline);
        BbRtReflectPost(*pipeline);
    }
    if (bb_volfog) {
        BbVolFogDraw(*pipeline);
    }

    // bb-clean-021-bbport (Supermedo): screen-space (clip disabled) draws into the upscaler's
    // output-size images.
    push_data.xscale *= bb_target_scale[0];
    push_data.xoffset *= bb_target_scale[0];
    push_data.yscale *= bb_target_scale[1];
    push_data.yoffset *= bb_target_scale[1];
    BbObjMvDraw(pipeline,
                is_indexed ? regs.index_base_address.Address<VAddr>() + index_offset : 0,
                regs.num_indices, regs.num_instances.NumInstances());
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    if (bb_rt_water_draw) {
        // bb-rt-reflect: the water surface writes the scene depth (restored after the pass).
        scheduler.GetDynamicState().SetDepthWriteEnabled(true);
    }
    scheduler.bb_prof_next_label = bb_prof_label;
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();

    if (BB::UiCut::Active()) {
        bb_cut.AfterDraw(bb_scene_depth); // bb-hud-layer: the cut after the HUD composite
    }
    ResetBindings(false);
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
        .is_indirect = true,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    BbUiDraw(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    if (bb_rt_reflect && BbRtWaterDraw(*pipeline)) {
        ResetBindings(false);
        return; // F6: this forward PS hidden (water identification)
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(0, true);
    }

    const auto size = stride * max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(arg_address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4,
                                   vk::AccessFlagBits2::eIndirectCommandRead);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    if (bb_rt_reflect) {
        BbRtLightRun(*pipeline);
        BbRtReflectPost(*pipeline);
    }
    if (bb_volfog) {
        BbVolFogDraw(*pipeline);
    }

    // bb-clean-021-bbport (Supermedo): screen-space (clip disabled) draws into the upscaler's
    // output-size images.
    push_data.xscale *= bb_target_scale[0];
    push_data.xoffset *= bb_target_scale[0];
    push_data.yscale *= bb_target_scale[1];
    push_data.yoffset *= bb_target_scale[1];
    BbObjMvDraw(pipeline, arg_address + offset, max_count, 0);
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    if (bb_rt_water_draw) {
        // bb-rt-reflect: the water surface writes the scene depth (restored after the pass).
        scheduler.GetDynamicState().SetDepthWriteEnabled(true);
    }
    scheduler.bb_prof_next_label = bb_prof_label;
    scheduler.BeginRendering(state);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_offset,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }

    if (BB::UiCut::Active()) {
        bb_cut.AfterDraw(bb_scene_depth); // bb-hud-layer: the cut after the HUD composite
    }
    ResetBindings(false);
}

// bb-gpu-prof: one label per game compute shader.
static u32 BbProfComputeLabel(u64 hash) {
    return BB::GpuProf::KeyLabel(
        hash ^ 0x6373000000000000ull, [&] { return fmt::format("game compute cs {:#x}", hash); },
        0);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return; // (bb-gpu-prof: its copies are timed in the runtime)
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (BB::SsrProbe::Enabled()) {
        BB::SsrProbe::OnDraw(BbSsrDesc(*pipeline, liverpool->regs,
                                       fmt::format("dispatch {}x{}x{}", cs_program.dim_x,
                                                   cs_program.dim_y, cs_program.dim_z),
                                       false));
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    const u32 bb_prof = BB::GpuProf::On() ? BB::GpuProf::Begin(cmdbuf, BbProfComputeLabel(cs.pgm_hash)) : 0u;
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    BB::GpuProf::End(cmdbuf, bb_prof);
    DebugState.IncDispatch();

    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    const u32 bb_prof =
        BB::GpuProf::On()
            ? BB::GpuProf::Begin(cmdbuf, BbProfComputeLabel(
                                             pipeline->GetStage(Shader::SwStage::Compute).pgm_hash))
            : 0u;
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    BB::GpuProf::End(cmdbuf, bb_prof);
    DebugState.IncDispatch();

    ResetBindings(true);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    buffer_cache.TickFrame();
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::OnFence() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    bb_bind_lod_bias = pipeline->IsCompute() ? 0.0f : bb_lod_bias;
    bb_bind_tex_filter = pipeline->IsCompute() ? 0u : bb_tex_filter;

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    bb_draw_cbs.clear();
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }
    if (!bb_draw_cbs.empty()) {
        if (bb_volfog) {
            bb_volfog->OnDrawCbs(bb_draw_cbs.data(), bb_draw_cbs.size());
        }
        BB::Upscale::OnDrawConstants(bb_draw_cbs.data(), bb_draw_cbs.size());
        bb_draw_cbs.clear();
    }

    if (uses_dma) {
        DmaForgetAll(); // bb-dma-dedup: writes anywhere (no V#), nothing can be trusted
        buffer_cache.SynchronizeDmaBuffers();
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const auto& regs = liverpool->regs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) = buffer_cache.ObtainBuffer(range.base_address, size);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        bound_buffers.emplace_back(range.buffer, range.offset, size,
                                   vk::AccessFlagBits2::eVertexAttributeRead);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::BindIndexBuffer(u32 index_offset, bool is_indirect) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr base_address = regs.index_base_address.Address<VAddr>();
    const VAddr index_address = base_address + u64(index_offset) * index_size;

    // Bind index buffer.
    const u32 num_indices =
        is_indirect ? regs.max_index_size : std::min(regs.num_indices, regs.max_index_size);
    const u64 requested_size = u64(num_indices) * index_size;
    const u64 mapped_size =
        base_address != 0 ? memory->ClampRangeSize(index_address, requested_size) : 0;
    u32 index_buffer_size = static_cast<u32>(Common::AlignDown(mapped_size, index_size));
    vk::Buffer index_buffer{};
    u64 buffer_offset{};
    if (index_buffer_size != 0) {
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(index_address, index_buffer_size);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
        bound_buffers.emplace_back(buffer, offset, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = buffer->Handle();
        buffer_offset = offset;
    } else if (!instance.IsMaintenance6Supported()) {
        const auto* null_buffer = buffer_cache.GetNullIndexBuffer();
        index_buffer_size = static_cast<u32>(null_buffer->SizeBytes());
        needs_barrier |= runtime.IsBufferAccessed(null_buffer, 0, index_buffer_size);
        bound_buffers.emplace_back(null_buffer, 0, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = null_buffer->Handle();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsMaintenance5Supported()) {
        cmdbuf.bindIndexBuffer2KHR(index_buffer, buffer_offset, index_buffer_size, index_type);
    } else {
        cmdbuf.bindIndexBuffer(index_buffer, buffer_offset, index_type);
    }
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, src_access] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (const auto& desc : stage.buffers) {
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                if (BB::Upscale::Capture() || BbRtAnyRequested()) {
                    BB::Upscale::OnConstants(stage.flattened_ud_buf.data(),
                                             stage.flattened_ud_buf.size());
                }
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                // bb-clean-021-fg: the camera is in the game's 864-byte scene constant block, a
                // vertex shader's constant buffer (as 1.59 reads it); one float test per buffer.
                if ((stage.sw_stage == Shader::SwStage::Vertex ||
                     stage.sw_stage == Shader::SwStage::Fragment) &&
                    !desc.is_written && size >= 864 &&
                    size <= 16384 && (BB::Upscale::Capture() || BbRtAnyRequested()) &&
                    memory->IsValidMapping(vsharp.base_address, 272)) {
                    BB::Upscale::OnConstants(reinterpret_cast<const u32*>(vsharp.base_address),
                                             static_cast<size_t>(size / 4));
                }
                // bb-rt-vol: the game's light-set (material CB slot 5, 288 B) and fog (slot 6,
                // 84 B of data) constant buffers, layouts from Ghidra (FUN_026deeb0,
                // FUN_02651f10); bound to the forward-lit object draws (any VS / PS). Collected
                // for the whole draw, see OnDrawConstants (after the stage loop).
                // bb-volfog alone: the light-set CB sizes, the point-light draws' PS and the PS
                // buffers that can hold the fog CB (FUN_02651f10, 224 B V# in the logs).
                const bool vf_want =
                    bb_volfog && ((size >= 288 && size <= 320) ||
                                  (stage.sw_stage == Shader::SwStage::Fragment &&
                                   ((size >= 84 && size <= 256) ||
                                    BB::VolFog::IsPointLightPs(stage.pgm_hash))));
                if ((stage.sw_stage == Shader::SwStage::Vertex ||
                     stage.sw_stage == Shader::SwStage::Fragment) &&
                    !desc.is_written && size >= 64 && size <= (bb_volfog ? 2048u : 1024u) &&
                    (BbRtLightRequested() || vf_want) &&
                    bb_draw_cbs.size() < 32 &&
                    memory->IsValidMapping(vsharp.base_address, size)) {
                    bb_draw_cbs.push_back(BB::Upscale::DrawCb{
                        .data = reinterpret_cast<const u32*>(vsharp.base_address),
                        .bytes = static_cast<size_t>(size),
                        .hash = stage.pgm_hash,
                        .stage = stage.sw_stage == Shader::SwStage::Vertex ? 0u : 1u,
                        .index = static_cast<u32>(&desc - stage.buffers.data()),
                    });
                }
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                VideoCore::ObtainBufferFlags flags = {};
                if (desc.is_written) {
                    flags |= VideoCore::ObtainBufferFlags::IsWritten;
                }
                if (desc.is_formatted) {
                    flags |= VideoCore::ObtainBufferFlags::IsTexelBuffer;
                }
                const auto [buffer, offset] =
                    buffer_cache.ObtainBuffer(vsharp.base_address, size, flags);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, src_access);
                if (desc.is_written) {
                    DmaForget(vsharp.base_address, size); // bb-dma-dedup: a shader writes it
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    u32 num_images{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            std::construct_at(&desc, tsharp, image_desc);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            vk::ImageView bb_view = *image_view.image_view;
            vk::ImageLayout bb_layout = image.backing->state.layout;
            // bb-clean-021-bbport (Supermedo's RedirectSampled): the display pass reads the
            // upscaled UI image instead of the render-size UI target.
            if (BB::UiCut::Active() && !is_storage && bb_cut.RedirectsSampled(image_id)) {
                bb_cut.RedirectSampled(image_id, image_view.info.format, image_view.info.mapping,
                                       bb_view, bb_layout);
            } else if (BB::UiCut::Active() && !is_storage) {
                // bb-hud-layer: the HUD composite pass reads a transparent layer (AfterDraw).
                bb_cut.SubstituteLayer(image_id, bb_view, bb_layout);
            }
            image_infos.emplace_back(VK_NULL_HANDLE, bb_view, bb_layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                liverpool->regs.ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        liverpool->regs.ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base, sampler.is_depth,
                                     sampler.is_depth ? 0.0f : bb_bind_lod_bias,
                                     sampler.is_depth ? 0u : bb_bind_tex_filter);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

// bb-gpu6 (BB_LOD_BIAS): with an upscaler the scene is drawn at the render size but shown at
// the output's: its textures need the output's mip level (log2(render / output), the bias DLSS
// and FSR ask for), or they are sampled one mip too blurry and the upscaler cannot bring the
// detail back. Scene draws only (depth test on into the scene-size depth, before the HUD cut).
// BB_LOD_BIAS=off: none; a number: that bias instead; BB_LOD_BIAS_EXTRA adds to the automatic one.
// bb-tex (BB_TEX_QUALITY / BB_ANISO, bb_texture.h): the quality row's LOD offset and the
// anisotropy / trilinear override, on the same scene draws, with or without an upscaler.
// Extra (5) holds alpha-tested (PS discard) and blended draws at a milder offset.
void Rasterizer::BbLodBias(const GraphicsPipeline* pipeline) {
    bb_lod_bias = 0.0f;
    bb_tex_filter = 0;
    static const int mode = [] {
        const char* v = std::getenv("BB_LOD_BIAS");
        if (v != nullptr && (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0)) return 0;
        if (v != nullptr && *v != '\0' && std::strcmp(v, "auto") != 0) return 2;
        return 1;
    }();
    static const float fixed = [] {
        const char* v = std::getenv("BB_LOD_BIAS");
        return v != nullptr ? float(std::atof(v)) : 0.0f;
    }();
    static const float extra = [] {
        const char* v = std::getenv("BB_LOD_BIAS_EXTRA");
        return v != nullptr ? float(std::atof(v)) : 0.0f;
    }();
    const auto& tex = BB::Texture::Get();
    const bool tex_on = tex.level >= 0 || tex.aniso != 0;
    const bool up = BB::Upscale::Enabled() && BB::Upscale::BackendReady();
    if (!tex_on && (mode == 0 || !up)) return;
    const auto& regs = liverpool->regs;
    if (!regs.depth_control.depth_enable || !db_desc.image_id || bb_cut.UiPhase()) return;
    const auto& depth = texture_cache.GetImage(db_desc.image_id);
    if (BB::Upscale::g_scene_w != 0 &&
        (depth.info.size.width != BB::Upscale::g_scene_w ||
         depth.info.size.height != BB::Upscale::g_scene_h)) {
        return;
    }
    float bias = 0.0f;
    u32 ow = 0, oh = 0;
    const bool have_out = up && BB::Upscale::OutputSize(ow, oh) && ow != 0 &&
                          BB::Upscale::g_scene_w != 0;
    if (mode != 0 && have_out) {
        const float ratio = float(BB::Upscale::g_scene_w) / float(ow);
        bias = mode == 2 ? fixed : (ratio < 0.99f ? std::log2(ratio) + extra : 0.0f);
    }
    bool alpha = false;
    if (tex.level >= 0) {
        if (pipeline != nullptr && tex.lod_offset_alpha != tex.lod_offset) {
            for (const auto* stage : pipeline->GetStages()) {
                if (stage && stage->sw_stage == Shader::SwStage::Fragment && stage->has_discard) {
                    alpha = true;
                }
            }
            alpha = alpha || regs.blend_control[0].enable;
        }
        bias += alpha ? tex.lod_offset_alpha : (up ? tex.lod_offset_tu : tex.lod_offset);
    }
    bias = std::round(bias * 8.0f) / 8.0f; // few distinct samplers
    bb_lod_bias = bias;
    bb_tex_filter = BB::Texture::FilterBits();
    static float logged = 1e9f;
    if (!alpha && bias != logged) {
        logged = bias;
        BB::Upscale::Checkpoint(fmt::format(
            "[lod-bias] scene textures {:+.3f} (render {}x{} -> output {}x{}; BB_LOD_BIAS={}, "
            "BB_LOD_BIAS_EXTRA {:+.2f}; BB_TEX_QUALITY {} {:+.2f}, alpha-tested / blended "
            "{:+.2f}; filter bits {:#x})",
            bias, BB::Upscale::g_scene_w, BB::Upscale::g_scene_h, ow, oh,
            mode == 2 ? "fixed" : mode == 0 ? "off" : "auto", extra, tex.level,
            up ? tex.lod_offset_tu : tex.lod_offset, tex.lod_offset_alpha, bb_tex_filter));
    }
}

// bb-hud-mask: the game's colour buffer 0 plus only the motion attachment of BB object motion
// vectors (bb_objmv2.h, attachment 7, the port's own): the in-game HUD's Scaleform draws
// (mrt 0x81 in the 3rd log's [hud-trace]); a single target for the HUD cut.
bool Rasterizer::BbUiSingleTarget(const GraphicsPipeline* pipeline) const {
    const auto& key = pipeline->GetGraphicsKey();
    constexpr u32 motion = 1u << BB::ObjMv2::Attachment;
    return key.mrt_mask == (1u | motion) &&
           BB::ObjMv2::IsMotionKey(key.mrt_mask, key.color_buffers[BB::ObjMv2::Attachment]);
}

void Rasterizer::BbUiDraw(const GraphicsPipeline* pipeline) {
    const auto& regs = liverpool->regs;
    BbLodBias(pipeline);
    if (!BB::UiCut::Active()) {
        return;
    }
    // Supermedo: the pass copying a finished frame to a display buffer starts the frame (here
    // once per run of draws into display buffers).
    const bool display = regs.color_buffers[0] &&
                         BB::UiCut::IsDisplayBuffer(regs.color_buffers[0].Address());
    if (display && !bb_display_pass) {
        bb_cut.OnFrameStart();
    }
    bb_display_pass = display;
    if (cb_descs[0].image_id) {
        bb_cut.OnColorTarget(cb_descs[0].image_id);
    }
    // [hud-trace]: the registers behind the bound targets (a draw with no bound colour target
    // shows why: no colour buffer, target mask 0, colour disabled), for every draw.
    std::string raw;
    if (bb_cut.Tracing()) {
        const auto& key = pipeline->GetGraphicsKey();
        raw = fmt::format(
            " [mrt {:#x} cb0 {:#x} mask {:#x} mode {} display {} db {:#x} stencil {} vp {:.0f}x{:.0f}]",
            key.mrt_mask, regs.color_buffers[0] ? VAddr(regs.color_buffers[0].Address()) : VAddr(0),
            regs.color_target_mask.GetMask(0), u32(regs.color_control.mode), display ? 1 : 0,
            regs.depth_buffer.DepthValid() ? VAddr(regs.depth_buffer.DepthAddress()) : VAddr(0),
            regs.depth_control.stencil_enable ? 1 : 0, regs.viewports[0].xscale * 2,
            regs.viewports[0].yscale * 2);
        if (std::popcount(key.mrt_mask) != 1 && !BbUiSingleTarget(pipeline)) {
            bb_cut.TraceDraw(fmt::format(
                "vs {:#x} (not single-target){}",
                pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash, raw));
        }
    }
    if (std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1 || BbUiSingleTarget(pipeline)) {
        const auto& viewport = regs.viewports[0];
        // bb-gpu6: what the HUD-start rule needs of the draw: does it read a scene-size image
        // (a post pass: fog, tonemap, composites), depth test, blending, its pixel shader.
        BB::UiCut::DrawInfo info{};
        info.vs_hash = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash;
        info.depth_test = regs.depth_control.depth_enable != 0;
        info.blend = pipeline->GetGraphicsKey().blend_controls[0].enable != 0;
        info.clip_disabled = regs.IsClipDisabled();
        info.cb0_address = regs.color_buffers[0] ? VAddr(regs.color_buffers[0].Address()) : 0;
        const u32 sw = BB::Upscale::g_scene_w, sh = BB::Upscale::g_scene_h;
        const bool tracing = bb_cut.Tracing();
        std::string reads;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) continue;
            if (stage->sw_stage == Shader::SwStage::Fragment) info.ps_hash = stage->pgm_hash;
            for (const auto& image_desc : stage->images) {
                const auto tsharp = image_desc.GetSharp(*stage);
                if (tsharp.Address() == 0) continue;
                const u32 w = u32(tsharp.width) + 1, h = u32(tsharp.height) + 1;
                const bool is_layer = bb_cut.IsLayerCandidate(tsharp.Address());
                if (is_layer) {
                    info.samples_hud_layer = true;
                    info.layer_read = tsharp.Address();
                }
                if (sw && sh && w + 8 > sw && h + 8 > sh && w < sw + 8 && h < sh + 8) {
                    info.samples_scene = true;
                }
                if (tracing && reads.size() < 160) {
                    reads += fmt::format(" {:#x}:{}x{}{}", tsharp.Address(), w, h,
                                         is_layer ? "(HUD layer)" : "");
                }
            }
        }
        if (tracing) {
            // BB_HUD_TRACE=1: vs/ps, colour target (address size format), depth, blend, reads.
            std::string target = "none";
            if (cb_descs[0].image_id) {
                const auto& im = texture_cache.GetImage(cb_descs[0].image_id);
                target = fmt::format("{:#x} {}x{} {}", im.info.guest_address,
                                     im.info.size.width, im.info.size.height,
                                     vk::to_string(im.info.pixel_format));
            }
            bb_cut.TraceDraw(fmt::format(
                "{}vs {:#x} ps {:#x} -> {}{}{}{}{} reads:{}{}", bb_cut.UiPhase() ? "[UI] " : "",
                info.vs_hash, info.ps_hash, target, info.depth_test ? " depth" : "",
                info.blend ? " blend" : "", info.clip_disabled ? " 2d" : "",
                UiComposition::MovieShader(info.vs_hash) ? " movie" : "",
                reads.empty() ? " none" : reads, raw));
        }
        bb_cut.OnDraw(info, cb_descs[0].image_id, db_desc.image_id,
                      UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2),
                      bb_scene_depth);
    }
}

void Rasterizer::BbHudNativeMarker(bool begin, u32 seq) {
    // bb-hud-native: the game's Scaleform HAL BeginScene / EndScene, in command stream order.
    bb_cut.OnNativeMarker(begin, seq);
}

vk::Rect2D Rasterizer::BbFullResScissor(u32 index) const {
    // bb-hud-fullres: a scissor covering the whole guest target (a target-size screen / window
    // scissor set for the render size) does not clip; the others (Scaleform's clip rectangles,
    // in the movie viewport's pixels) are scaled like the viewport.
    const auto& regs = liverpool->regs;
    const s32 gw = s32(bb_fullres_guest.first), gh = s32(bb_fullres_guest.second);
    s32 x0 = 0, y0 = 0;
    s32 x1 = s32(bb_fullres_extent.first), y1 = s32(bb_fullres_extent.second);
    bool clipped = false;
    const auto clip = [&](s32 l, s32 t, s32 r, s32 b) {
        if (l <= 0 && t <= 0 && r >= gw && b >= gh) return;
        clipped = true;
        x0 = std::max(x0, s32(std::lround(float(std::max(l, 0)) * bb_target_scale[0])));
        y0 = std::max(y0, s32(std::lround(float(std::max(t, 0)) * bb_target_scale[1])));
        x1 = std::min(x1, s32(std::lround(float(std::max(r, 0)) * bb_target_scale[0])));
        y1 = std::min(y1, s32(std::lround(float(std::max(b, 0)) * bb_target_scale[1])));
    };
    const bool offset = !regs.window_scissor.window_offset_disable;
    const s32 ox = offset ? s32(regs.window_offset.window_x_offset) : 0;
    const s32 oy = offset ? s32(regs.window_offset.window_y_offset) : 0;
    clip(regs.screen_scissor.top_left_x, regs.screen_scissor.top_left_y,
         regs.screen_scissor.bottom_right_x, regs.screen_scissor.bottom_right_y);
    clip(s32(regs.window_scissor.top_left_x) + ox, s32(regs.window_scissor.top_left_y) + oy,
         s32(regs.window_scissor.bottom_right_x) + ox, s32(regs.window_scissor.bottom_right_y) + oy);
    clip(s32(regs.generic_scissor.top_left_x) + ox, s32(regs.generic_scissor.top_left_y) + oy,
         s32(regs.generic_scissor.bottom_right_x) + ox,
         s32(regs.generic_scissor.bottom_right_y) + oy);
    if (regs.mode_control.vport_scissor_enable) {
        const auto& v = regs.viewport_scissors[index];
        clip(v.top_left_x, v.top_left_y, v.bottom_right_x, v.bottom_right_y);
    }
    if (clipped) BB::HudNativeHook::FullResStats::scissors.fetch_add(1, std::memory_order_relaxed);
    return {
        .offset = {x0, y0},
        .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
    };
}

void Rasterizer::BbUpdateUiTransform() {
    // Supermedo's UpdateUiTransform: a native viewport (or native window-space UI vertices) must
    // not be doubled again; the display copy uses the guest render-size coordinates.
    const auto& regs = liverpool->regs;
    const auto& viewport = regs.viewports[0];
    const bool window_space = regs.IsClipDisabled() && !regs.viewport_control.xscale_enable &&
                              !regs.viewport_control.yscale_enable;
    // bb-hud-fullres: a Scaleform draw laid out for the output (its movie viewport) goes 1:1 into
    // the output-size UI image (the viewport / scissors / window-space vertices are not scaled).
    if (u32 uw = 0, uh = 0, rw = 0, rh = 0;
        bb_ui_pass.native_ui && BB::HudNativeHook::FullResUi(uw, uh, rw, rh)) {
        const bool ui_draw = bb_cut.NativeLive() ? bb_cut.NativeUiDraw() : bb_cut.UiPhase();
        const auto scale = UiComposition::FullResScale(
            viewport.xscale * 2, viewport.yscale * 2, window_space, ui_draw,
            bb_ui_pass.guest.first, bb_ui_pass.guest.second, uw, uh, bb_ui_pass.output.first,
            bb_ui_pass.output.second);
        if (scale[0] > 0.0f && scale[1] > 0.0f) {
            bb_target_scale = scale;
            bb_fullres_draw = true;
            bb_fullres_guest = bb_ui_pass.guest;
            bb_fullres_extent = bb_ui_pass.output;
            BB::HudNativeHook::FullResStats::output_draws.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    bb_target_scale = UiComposition::DrawScale(
        viewport.xscale * 2, viewport.yscale * 2, window_space, bb_ui_pass.native_ui,
        bb_ui_pass.guest.first, bb_ui_pass.guest.second, bb_ui_pass.output.first,
        bb_ui_pass.output.second);
}

// bb-jitter-fs (BB_JITTER_FS): the upscaler's sub-pixel jitter moves the scene geometry by the
// viewport offset (UpdateViewportScissorState), not the game's projection. A full-screen pass
// that rebuilds world positions from the jittered scene depth (the deferred shadow mask: scene
// depth + the 4096x4096 cascade atlas, ps 0xf55643de in [hud-trace]) then puts every pixel up to
// half a pixel off its surface along the view ray; on rock facets turned the wrong way that is
// enough to fail the game's shadow bias, and those facets go dark whenever the jitter's y is
// below about -0.2 render pixels: one frame in three, darkest on Halton phase 9 of 18 (the
// flicker of the forest-rock video, every 18 game frames). Shifting such a pass by the same
// jitter makes its interpolated rays / UVs land on the unjittered position of the surface the
// pixel holds (|jitter| < 0.5, so a point-sampled depth fetch still reads the same texel).
// BB_JITTER_FS: 0 off, 1 (default) passes that read the scene depth and a shadow map,
// 2 every full-screen pass reading a depth-like image of the scene / its half / quarter size.
// BB_JITTER_FS_PS=0xhash,... only those pixel shaders; BB_JITTER_FS_SKIP=0xhash,... never.
namespace {
struct JitterFsCfg {
    int mode = 1;
    std::vector<u64> only, skip;
};
const JitterFsCfg& JitterFsConfig() {
    static const JitterFsCfg cfg = [] {
        JitterFsCfg c;
        const auto list = [](const char* s, std::vector<u64>& out) {
            while (s && *s) {
                char* end = nullptr;
                const u64 v = std::strtoull(s, &end, 16);
                if (end == s) break;
                out.push_back(v);
                s = end;
                while (*s == ',' || *s == ' ' || *s == ';') ++s;
            }
        };
        if (const char* s = std::getenv("BB_JITTER_FS"); s && *s) c.mode = std::atoi(s);
        list(std::getenv("BB_JITTER_FS_PS"), c.only);
        list(std::getenv("BB_JITTER_FS_SKIP"), c.skip);
        BB::Upscale::Checkpoint(fmt::format(
            "[jitter-fs] BB_JITTER_FS={} (0 off, 1 shadow-mask passes, 2 every depth-reading "
            "full-screen pass), only {} ps, skip {} ps",
            c.mode, c.only.size(), c.skip.size()));
        return c;
    }();
    return cfg;
}
bool DepthLike(const AmdGpu::Image& t) {
    switch (t.GetDataFmt()) {
    case AmdGpu::DataFormat::Format32:
    case AmdGpu::DataFormat::Format8_24:
    case AmdGpu::DataFormat::Format24_8:
    case AmdGpu::DataFormat::FormatX24_8_32:
        return true;
    case AmdGpu::DataFormat::Format16:
        return t.GetNumberFmt() == AmdGpu::NumberFormat::Unorm;
    default:
        return false;
    }
}
struct JitterFsPass {
    u64 vs = 0, ps = 0;
    u32 w = 0, h = 0;
    bool shadow = false, applied = false;
    u32 draws = 0;
};
struct JitterFsStats {
    std::chrono::steady_clock::time_point start{};
    u32 applied = 0, candidates = 0;
    std::vector<JitterFsPass> passes;
} g_jitter_fs_stats;
bool Near(u32 a, u32 b) {
    return a + 8 > b && a < b + 8;
}
} // namespace

void Rasterizer::BbJitterFsEval(const GraphicsPipeline* pipeline) {
    bb_jitter_fs[0] = bb_jitter_fs[1] = 0.0f;
    const auto& regs = liverpool->regs;
    if (bb_jitter_draw || !BB::Upscale::Capture() || !cb_descs[0].image_id ||
        regs.num_indices > 6 || regs.num_instances.NumInstances() > 1 || regs.IsClipDisabled() ||
        bb_cut.UiPhase() || bb_cut.NativeUiDraw()) {
        return;
    }
    float jx = 0.f, jy = 0.f;
    BB::Upscale::Jitter(jx, jy);
    const auto& cfg = JitterFsConfig();
    if (cfg.mode <= 0 || (jx == 0.0f && jy == 0.0f)) {
        return;
    }
    const u32 sw = BB::Upscale::g_scene_w, sh = BB::Upscale::g_scene_h;
    if (!sw || !sh) {
        return;
    }
    const auto& im = texture_cache.GetImage(cb_descs[0].image_id);
    const u32 tw = im.info.size.width, th = im.info.size.height;
    u32 div = 0;
    for (u32 d : {1u, 2u, 4u}) {
        if (Near(tw, sw / d) && Near(th, sh / d)) {
            div = d;
            break;
        }
    }
    if (div == 0) {
        return;
    }
    bool depth_read = false, shadow_read = false;
    u64 vs = 0, ps = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) continue;
        if (stage->sw_stage == Shader::SwStage::Vertex) vs = stage->pgm_hash;
        if (stage->sw_stage == Shader::SwStage::Fragment) ps = stage->pgm_hash;
        for (const auto& image_desc : stage->images) {
            const auto t = image_desc.GetSharp(*stage);
            if (t.Address() == 0 || !DepthLike(t)) continue;
            const u32 w = u32(t.width) + 1, h = u32(t.height) + 1;
            if ((Near(w, sw) && Near(h, sh)) || (Near(w, tw) && Near(h, th))) {
                depth_read = true;
            } else if (w == h && w >= 1024) {
                shadow_read = true;
            }
        }
    }
    if (!depth_read) {
        return;
    }
    const auto listed = [](const std::vector<u64>& l, u64 h) {
        return std::find(l.begin(), l.end(), h) != l.end();
    };
    bool apply = cfg.mode >= 2 ? true : (div == 1 && shadow_read);
    if (!cfg.only.empty()) apply = listed(cfg.only, ps);
    if (listed(cfg.skip, ps)) apply = false;
    if (apply) {
        bb_jitter_fs[0] = bb_jitter_fs[1] = 1.0f / float(div);
    }
    // [jitter-fs] every 10 s: the full-screen passes reading the scene depth, applied or not.
    auto& st = g_jitter_fs_stats;
    const auto now = std::chrono::steady_clock::now();
    if (st.start.time_since_epoch().count() == 0) st.start = now;
    ++st.candidates;
    st.applied += apply ? 1 : 0;
    bool found = false;
    for (auto& p : st.passes) {
        if (p.vs == vs && p.ps == ps && p.w == tw) {
            ++p.draws;
            found = true;
            break;
        }
    }
    if (!found && st.passes.size() < 24) {
        st.passes.push_back({vs, ps, tw, th, shadow_read, apply, 1});
    }
    const double secs = std::chrono::duration<double>(now - st.start).count();
    if (secs >= 10.0) {
        std::string text = fmt::format(
            "[jitter-fs] {:.1f} s: full-screen passes reading the scene depth {:.1f}/s, shifted "
            "by the jitter {:.1f}/s (BB_JITTER_FS={}; jitter now {:+.3f} {:+.3f}):",
            secs, st.candidates / secs, st.applied / secs, cfg.mode, jx, jy);
        for (const auto& p : st.passes) {
            text += fmt::format(" vs{:#x}/ps{:#x} {}x{}{} {}x{:.1f}/s;", p.vs, p.ps, p.w, p.h,
                                p.shadow ? " shadow" : "", p.applied ? "SHIFT" : "keep",
                                p.draws / secs);
        }
        BB::Upscale::Checkpoint(text);
        st = {};
        st.start = now;
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    // bb-clean-021-bbport (Supermedo's BeginRenderingFull): the UI pass and the display pass
    // drawn into the upscaler's output-size images.
    BB::UiCut::Target bb_redirect{};
    bool bb_color_redirected = false, bb_depth_redirected = false;
    std::pair<u32, u32> bb_guest_extent{1, 1};
    std::pair<vk::ImageView, vk::ImageLayout> bb_original_color{};
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    bool has_attachment = false;
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        has_attachment = true;
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
        if (cb == 0 && BB::UiCut::Active()) {
            bb_color_redirected = bb_cut.RedirectColor(image_id, desc.view_info, bb_redirect);
            if (bb_color_redirected) {
                bb_guest_extent = {image->info.size.width, image->info.size.height};
                bb_original_color = {attachment.image_view, attachment.image_layout};
                attachment.image_view = bb_redirect.view;
                attachment.image_layout = bb_redirect.layout;
            }
        }
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }
    if (bb_color_redirected && BbUiSingleTarget(pipeline)) {
        // bb-hud-mask: a HUD draw into the output-size UI image writes no motion (the HUD is not
        // part of the scene): its motion attachment is left unbound (writes discarded).
        for (u32 cb = 1; cb < state.num_color_attachments; ++cb) {
            state.color_attachments[cb] = {};
        }
    } else if (BB::ObjMv2::IsMotionKey(key.mrt_mask,
                                       key.color_buffers[BB::ObjMv2::Attachment])) {
        BB::ObjMv2::Attach(scheduler, state);
    }

    bb_jitter_draw = false;
    if (auto image_id = db_desc.image_id; image_id) {
        has_attachment = true;
        auto& desc = db_desc.desc;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);
        if (bb_volfog && regs.depth_control.depth_enable &&
            regs.depth_control.depth_write_enable && image.info.num_samples == 1) {
            // bb-volfog: the G-buffer MRT's depth (>= 2 RGBA8 colour targets of its size) is the
            // scene depth (as bb-rt-light, Ghidra FUN_026c39f0).
            u32 n_rgba8 = 0;
            for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
                const auto cid = cb_descs[cb].image_id;
                if (!cid || !state.color_attachments[cb].image_view) continue;
                const auto& ci = texture_cache.GetImage(cid);
                if (ci.info.size.width == image.info.size.width &&
                    ci.info.size.height == image.info.size.height &&
                    (ci.info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
                     ci.info.pixel_format == vk::Format::eR8G8B8A8Srgb)) {
                    ++n_rgba8;
                }
            }
            if (n_rgba8 >= 2) {
                bb_vf_depth_id = image_id;
                bb_vf_depth_uid = image.image_uid;
                bb_vf_gbuf_seen = true;
            }
        }
        if (bb_rt_reflect) {
            const bool writes =
                regs.depth_control.depth_enable && regs.depth_control.depth_write_enable;
            // G-buffer MRT test (see BbRtDepthSeen): colour attachments of the depth's size.
            u32 n_same = 0, n_rgba8 = 0;
            VideoCore::ImageId rt0{}, rt1{};
            u64 rt0_uid = 0, rt1_uid = 0;
            for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
                const auto cid = cb_descs[cb].image_id;
                if (!cid || !state.color_attachments[cb].image_view) {
                    continue;
                }
                const auto& ci = texture_cache.GetImage(cid);
                if (ci.info.size.width != image.info.size.width ||
                    ci.info.size.height != image.info.size.height) {
                    continue;
                }
                ++n_same;
                if (ci.info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
                    ci.info.pixel_format == vk::Format::eR8G8B8A8Srgb) {
                    if (n_rgba8 == 0) {
                        rt0 = cid;
                        rt0_uid = ci.image_uid;
                    } else if (n_rgba8 == 1) {
                        rt1 = cid;
                        rt1_uid = ci.image_uid;
                    }
                    ++n_rgba8;
                }
            }
            const bool gbuf = regs.depth_control.depth_enable && n_same >= 2 && n_rgba8 >= 2 &&
                              image.info.num_samples == 1;
            bool merged = false;
            for (auto& c : bb_rt_depth_seen) {
                if (c.id == image_id && c.uid == image.image_uid) {
                    c.depth_test |= bool(regs.depth_control.depth_enable);
                    c.depth_write |= writes;
                    if (gbuf) {
                        ++c.gbuf_draws;
                        if (!c.rt0_uid) {
                            c.rt0_id = rt0;
                            c.rt0_uid = rt0_uid;
                            c.rt1_id = rt1;
                            c.rt1_uid = rt1_uid;
                        }
                    }
                    merged = true;
                }
            }
            if (!merged && bb_rt_depth_seen.size() < 32) {
                bb_rt_depth_seen.push_back({image_id, image.image_uid, image.info.guest_address,
                                            image.info.size.width, image.info.size.height,
                                            image.info.num_samples, image.info.pixel_format,
                                            bool(regs.depth_control.depth_enable), writes,
                                            gbuf ? 1u : 0u, gbuf ? rt0 : VideoCore::ImageId{},
                                            gbuf ? rt0_uid : 0, gbuf ? rt1 : VideoCore::ImageId{},
                                            gbuf ? rt1_uid : 0});
            }
            bb_rt_depth_fresh |= writes && image.info.num_samples == 1;
        }
        if (BB::Upscale::Capture()) {
            // bb-clean-021: the scene depth (the presented image's size, depth test on): the
            // upscaler's depth and the draws that get the sub-pixel jitter (none without an
            // upscaler: BB_FG alone captures the depth with zero jitter).
            const bool scene_sized = regs.depth_control.depth_enable &&
                                     image.info.size.width == BB::Upscale::g_scene_w &&
                                     image.info.size.height == BB::Upscale::g_scene_h;
            // Supermedo: the UI is not jittered (after the HUD cut).
            bb_jitter_draw = scene_sized && !bb_cut.UiPhase() && !bb_cut.NativeUiDraw();
            BB::Upscale::NoteJitterDraw(bb_jitter_draw, scene_sized && !bb_jitter_draw,
                                        image.info.size.width,
                                        image.info.size.height);
            if (bb_jitter_draw && regs.depth_control.depth_write_enable) {
                const auto& vp = regs.viewports[0];
                const auto& vp_ctl = regs.viewport_control;
                bb_scene_depth.valid = true;
                BB::ObjMv2::NoteSceneDepth(regs.depth_buffer.DepthAddress());
                bb_scene_depth.desc = desc;
                bb_scene_depth.xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
                bb_scene_depth.xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
                bb_scene_depth.yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;
                bb_scene_depth.yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
                bb_scene_depth.zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;
                bb_scene_depth.zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
                const auto f = regs.depth_control.depth_func;
                bb_scene_depth.greater = f == AmdGpu::CompareFunc::Greater ||
                                         f == AmdGpu::CompareFunc::GreaterEqual;
            }
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const bool depth_writable = desc.view_info.is_storage || bb_rt_water_draw;
        const auto new_layout = depth_writable
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
        if (BB::UiCut::Active()) {
            BB::UiCut::Target depth_redirect{};
            bb_depth_redirected = bb_cut.RedirectDepth(image_id, depth_redirect);
            if (bb_depth_redirected) {
                bb_guest_extent = {image.info.size.width, image.info.size.height};
                attachment.image_view = depth_redirect.view;
                attachment.image_layout = depth_redirect.layout;
                bb_redirect = depth_redirect;
            }
        }
    } else {
        state.depth_stencil_attachment = {};
    }
    BbJitterFsEval(pipeline);

    // Supermedo: every attachment of such a pass must be redirected; viewports and scissors
    // are scaled.
    bb_target_scale = {1.0f, 1.0f};
    bb_ui_pass.active = false;
    bb_fullres_draw = false;
    if (bb_color_redirected || bb_depth_redirected) {
        u32 color_targets = 0;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            color_targets += state.color_attachments[cb].image_view ? 1 : 0;
        }
        const bool has_depth = bool(db_desc.image_id);
        if ((color_targets > 0 && !bb_color_redirected) || (has_depth && !bb_depth_redirected) ||
            color_targets > 1) {
            static u32 warned = 0;
            if (warned < 8) {
                ++warned;
                BB::Upscale::Checkpoint(fmt::format(
                    "Upscaler: pass not redirected (color {}, depth {}, {} targets)",
                    bb_color_redirected ? "yes" : "no", bb_depth_redirected ? "yes" : "no",
                    state.num_color_attachments));
            }
            if (bb_color_redirected) {
                state.color_attachments[0].image_view = bb_original_color.first;
                state.color_attachments[0].image_layout = bb_original_color.second;
            }
            if (bb_depth_redirected) {
                auto& depth_image = texture_cache.GetImage(db_desc.image_id);
                state.depth_stencil_attachment.image_view =
                    *texture_cache.FindDepthTarget(db_desc.image_id, db_desc.desc).image_view;
                state.depth_stencil_attachment.image_layout = depth_image.backing->state.layout;
            }
        } else {
            state.width = bb_redirect.width;
            state.height = bb_redirect.height;
            bb_ui_pass = {true, bb_redirect.native_ui, bb_guest_extent,
                          {bb_redirect.width, bb_redirect.height}};
            BbUpdateUiTransform();
        }
    }
    // bb-hud-fullres fallback: no cut (or a pass left unredirected): a Scaleform draw laid out for
    // the output, into the render-size scene target, is scaled down to it (the HUD as before).
    if (u32 uw = 0, uh = 0, rw = 0, rh = 0;
        !bb_ui_pass.active && has_attachment && BB::HudNativeHook::FullResUi(uw, uh, rw, rh) &&
        state.width + 8 > rw && state.width < rw + 8 && state.height + 8 > rh &&
        state.height < rh + 8 && (state.width < uw || state.height < uh)) {
        const auto& viewport = regs.viewports[0];
        const bool window_space = regs.IsClipDisabled() && !regs.viewport_control.xscale_enable &&
                                  !regs.viewport_control.yscale_enable;
        const auto scale = UiComposition::FullResScale(
            viewport.xscale * 2, viewport.yscale * 2, window_space, bb_cut.NativeUiDraw(),
            state.width, state.height, uw, uh, state.width, state.height);
        if (scale[0] > 0.0f && scale[1] > 0.0f) {
            bb_target_scale = scale;
            bb_fullres_draw = true;
            bb_fullres_guest = {state.width, state.height};
            bb_fullres_extent = bb_fullres_guest;
            BB::HudNativeHook::FullResStats::fallback_draws.fetch_add(1,
                                                                      std::memory_order_relaxed);
        }
    }

    // Without attachments the area stays at the max framebuffer size. Draws cannot pass the
    // scissors, so bound it by them.
    if (!has_attachment) {
        state.num_layers = 1;
        const AmdGpu::Scissor scissor = regs.GetCombinedScissor();
        if (scissor.bottom_right_x > 0 && scissor.bottom_right_y > 0) {
            state.width = std::min<u32>(state.width, scissor.bottom_right_x);
            state.height = std::min<u32>(state.height, scissor.bottom_right_y);
        }
    }

    bb_prof_label = BB::GpuProf::On() ? BbProfPassLabel(state, pipeline) : 0u;
    return state;
}

// bb-gpu-prof: "pass WxH <colour formats> <depth format> ps <hash>" (a scope per game pass),
// summed also by "target WxH <formats>" (every pass into the same targets).
u32 Rasterizer::BbProfPassLabel(const RenderState& state, const GraphicsPipeline* pipeline) {
    const auto& key = pipeline->GetGraphicsKey();
    const u64 ps = key.stage_hashes[static_cast<u32>(Shader::SwStage::Fragment)];
    const auto mix = [](u64 h, u64 v) { return (h ^ v) * 0x100000001b3ull + 0x9e3779b97f4a7c15ull; };
    u64 tk = mix(0xcbf29ce484222325ull, u64(state.width) << 32 | state.height);
    std::array<vk::Format, 8> formats{};
    u32 host_targets = 0;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        if (cb_descs[cb].image_id) {
            formats[cb] = texture_cache.GetImage(cb_descs[cb].image_id).info.pixel_format;
        } else if (state.color_attachments[cb].image_view) {
            ++host_targets; // our own attachment (objmv2's motion target)
        }
        tk = mix(tk, u64(formats[cb]) << 8 | cb);
    }
    vk::Format depth{};
    if (db_desc.image_id && state.depth_stencil_attachment.image_view) {
        depth = texture_cache.GetImage(db_desc.image_id).info.pixel_format;
    }
    tk = mix(tk, u64(depth) << 4 | host_targets << 1 | (bb_ui_pass.active ? 1 : 0));
    const auto target_name = [&] {
        std::string n = fmt::format("{}x{}", state.width, state.height);
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (formats[cb] != vk::Format::eUndefined) n += " " + vk::to_string(formats[cb]);
        }
        if (host_targets) n += fmt::format(" +{} own", host_targets);
        if (depth != vk::Format::eUndefined) n += " depth " + vk::to_string(depth);
        if (bb_ui_pass.active) n += " (UI, output size)";
        return n;
    };
    static std::unordered_map<u64, u32> groups; // GPU thread only
    u32& gid = groups[tk];
    if (gid == 0) gid = BB::GpuProf::Label("target " + target_name(), BB::GpuProf::Kind::Group);
    return BB::GpuProf::KeyLabel(
        mix(tk, ps), [&] { return fmt::format("pass {} ps {:#x}", target_name(), ps); }, gid);
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    const auto write_id = texture_cache.FindImage(write_desc);
    auto& write_image = texture_cache.GetImage(write_id);
    // The copy destination is never bound as a target, so its stencil plane is linked here.
    texture_cache.AssociateStencil(write_id, write_desc.info);

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        DmaForget(address, num_bytes); // bb-dma-dedup
    }
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

// bb-dma-dedup: the game's CP DMAs (memory -> memory, Rasterizer::CopyBuffer). Bloodborne DMAs a
// CPU bone palette into one of five shared GX buffers before every skinned draw (0x2568a20, see
// bb_mv_engine.h) and its skin rings through 0x2168a20 -> 0x26b3560 -> 0x26b1c40. Each GPU-side
// copy ends the render pass and waits on a barrier. A copy is redundant when the destination still
// holds exactly what the same copy wrote last time: same destination, same size, the same source
// bytes (hash and byte compare of guest memory, which is what the GPU copy reads when the source is not
// GPU-modified; the source address may differ), and nothing has written the destination since (DMA / fill to an
// overlapping range, a shader's storage write, a CPU write to its pages, an unmap all forget it).
// BB_DMA_DEDUP=1 skips such copies (default off). "[dma]" once a second with BB_GPU_PROFILE=1,
// BB_DMA_DEDUP=1 or BB_DMA_STATS=1.
namespace {
struct DmaEntry {
    VAddr src;
    u32 size;
    u64 hash;
    std::vector<u8> bytes; // what was copied (compared byte for byte, not only by hash)
};
struct DmaState {
    std::recursive_mutex m; // (a fault while hashing must not deadlock the forget path)
    std::map<VAddr, DmaEntry> last; // by destination
    // per second
    u64 calls = 0, host = 0, gpu = 0, gds = 0, gpu_bytes = 0, in_pass = 0, dup = 0, skipped = 0;
    u64 sz[4]{};
    std::unordered_map<VAddr, std::array<u64, 4>> by_dst; // gpu copies, dups, last src, size
    std::chrono::steady_clock::time_point start{};
};
constexpr u32 kDmaMaxBytes = 64u << 10;  // size buckets; overlap search window
constexpr u32 kDmaKeepBytes = 16u << 10; // copies remembered (byte compare): up to 16 KiB each
DmaState& Dma() {
    static DmaState d;
    return d;
}
bool DmaDedupOn() {
    static const bool on = [] {
        const char* v = std::getenv("BB_DMA_DEDUP");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
bool DmaStatsOn() {
    static const bool on = [] {
        const char* v = std::getenv("BB_DMA_STATS");
        return DmaDedupOn() || BB::GpuProf::On() || (v != nullptr && v[0] == '1');
    }();
    return on;
}
// Forget every entry overlapping [addr, addr + size) (caller holds the lock).
void DmaForgetLocked(DmaState& d, VAddr addr, u64 size) {
    if (d.last.empty()) return;
    const VAddr from = addr > kDmaMaxBytes ? addr - kDmaMaxBytes : 0;
    for (auto it = d.last.lower_bound(from); it != d.last.end() && it->first < addr + size;) {
        if (it->first + it->second.size > addr) {
            it = d.last.erase(it);
        } else {
            ++it;
        }
    }
}
void DmaForget(VAddr addr, u64 size) {
    if (!DmaStatsOn()) return;
    auto& d = Dma();
    std::scoped_lock lk{d.m};
    DmaForgetLocked(d, addr, size);
}
void DmaForgetAll() {
    if (!DmaStatsOn()) return;
    auto& d = Dma();
    std::scoped_lock lk{d.m};
    d.last.clear();
}
void DmaLogSecond(DmaState& d) {
    const auto now = std::chrono::steady_clock::now();
    if (d.start.time_since_epoch().count() == 0) d.start = now;
    const double el = std::chrono::duration<double>(now - d.start).count();
    if (el < 1.0) return;
    std::vector<std::pair<VAddr, std::array<u64, 4>>> top(d.by_dst.begin(), d.by_dst.end());
    std::sort(top.begin(), top.end(),
              [](const auto& a, const auto& b) { return a.second[0] > b.second[0]; });
    std::string t;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
        const auto& [dst, v] = top[i];
        t += fmt::format("{}{:#x}<-{:#x} {} B x{:.1f} (dup {:.1f})", t.empty() ? "" : ", ", dst,
                         v[2], v[3], double(v[0]) / el, double(v[1]) / el);
    }
    LOG_INFO(Render_Vulkan,
             "[dma] {:.1f} s: CP DMA copies {:.1f}/s: host memcpy {:.1f}, GPU {:.1f} ({:.0f} KiB/s, "
             "{:.1f} ended a render pass), GDS {:.1f} | sizes <=256 B {:.1f}, <=4 KiB {:.1f}, "
             "<=64 KiB {:.1f}, larger {:.1f} | exact repeats of the destination's last copy "
             "{:.1f}/s, skipped {:.1f}/s (BB_DMA_DEDUP={}) | top destinations: {}",
             el, d.calls / el, d.host / el, d.gpu / el, d.gpu_bytes / 1024.0 / el, d.in_pass / el,
             d.gds / el, d.sz[0] / el, d.sz[1] / el, d.sz[2] / el, d.sz[3] / el, d.dup / el,
             d.skipped / el, DmaDedupOn() ? 1 : 0, t);
    d.calls = d.host = d.gpu = d.gds = d.gpu_bytes = d.in_pass = d.dup = d.skipped = 0;
    d.sz[0] = d.sz[1] = d.sz[2] = d.sz[3] = 0;
    d.by_dst.clear();
    d.start = now;
}
} // namespace

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!DmaStatsOn()) {
        CopyBufferImpl(dst, src, num_bytes, dst_gds, src_gds);
        return;
    }
    // (review) The buffer / texture cache queries and the hash run before the lock: the lock is
    // also taken from the write-fault path (InvalidateMemory), so it never waits on another lock.
    const bool gds = dst_gds || src_gds;
    const bool dst_gpu = !gds && buffer_cache.IsRegionGpuModified(dst, num_bytes);
    const bool src_cpu = !gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
                         !texture_cache.FindImageFromRange(src, num_bytes);
    // A GPU copy that may repeat the destination's last copy with the same source bytes?
    const bool hashable = !gds && src_cpu && num_bytes <= kDmaKeepBytes && dst_gpu &&
                          !texture_cache.FindImageFromRange(dst, num_bytes) &&
                          memory->IsValidMapping(src, num_bytes);
    const u64 h = hashable ? XXH3_64bits(std::bit_cast<const void*>(src), num_bytes) : 0;
    auto& d = Dma();
    std::unique_lock lk{d.m};
    DmaLogSecond(d);
    ++d.calls;
    if (gds) {
        ++d.gds;
        if (!dst_gds) DmaForgetLocked(d, dst, num_bytes);
        lk.unlock();
        CopyBufferImpl(dst, src, num_bytes, dst_gds, src_gds);
        return;
    }
    d.sz[num_bytes <= 256 ? 0 : num_bytes <= 4096 ? 1 : num_bytes <= kDmaMaxBytes ? 2 : 3]++;
    if (!dst_gpu && src_cpu) {
        ++d.host; // the host memcpy path below
        DmaForgetLocked(d, dst, num_bytes);
        lk.unlock();
        CopyBufferImpl(dst, src, num_bytes, dst_gds, src_gds);
        return;
    }
    auto& top = d.by_dst[dst];
    ++top[0];
    top[2] = src;
    top[3] = num_bytes;
    if (hashable) {
        // (Not the source address: the game's GX buffer update 0x26b3560 -> 0x26b1c40 copies the
        // CPU data into fresh transient command-buffer memory every time and DMAs from there.)
        if (const auto it = d.last.find(dst);
            it != d.last.end() && it->second.size == num_bytes && it->second.hash == h &&
            std::memcmp(it->second.bytes.data(), std::bit_cast<const void*>(src), num_bytes) ==
                0) {
            ++d.dup;
            ++top[1];
            if (DmaDedupOn()) {
                ++d.skipped;
                return; // the destination already holds these bytes
            }
        }
    }
    ++d.gpu;
    d.gpu_bytes += num_bytes;
    d.in_pass += scheduler.IsRendering() ? 1 : 0;
    DmaForgetLocked(d, dst, num_bytes);
    if (hashable) {
        if (d.last.size() > 2048) d.last.clear(); // (at most 2048 x 16 KiB kept)
        auto& e = d.last[dst];
        e.src = src;
        e.size = num_bytes;
        e.hash = h;
        e.bytes.assign(std::bit_cast<const u8*>(src), std::bit_cast<const u8*>(src) + num_bytes);
    }
    lk.unlock();
    CopyBufferImpl(dst, src, num_bytes, dst_gds, src_gds);
}

void Rasterizer::CopyBufferImpl(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten |
                                             VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    // bb-dma-dedup: a CPU write (the page around it: the tracker works on whole pages).
    DmaForget(addr & ~VAddr(0xffff), size + 0x20000);
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

// bb-native-hot (fault-around). Ghidra: the game's GX buffer update (0x26b3560 -> 0x26b1c40,
// Ghidra 0x2ab3560 / 0x2ab1c40) takes fresh space from the command buffer with the Gnm
// allocator 0x2ab07f0 (it allocates downward from the buffer's end, 64-byte aligned), memcpys the
// data there and DMAs it; the command packets grow upward from the start. Every frame the same
// command-buffer memory is filled again, page after page, and each 4 KiB page that the buffer
// cache uploaded last frame takes its own write fault (VEH + handler + VirtualProtect under the
// address space lock), ~90k faults/s in run_pt2. After one fault the rest of its aligned block is
// unprotected at once: the next writes to it are plain stores. Only CPU-clean, not GPU-written
// pages are touched, and they are marked CPU-modified, so the GPU still gets the guest's current
// bytes (at worst a page is uploaded that was not written). Pages a texture also watches stay
// protected through the page manager's watcher count.
static u32 FaultAroundPages() {
    static const u32 pages = [] {
        const char* v = std::getenv("BB_NATIVE_FAULTAROUND");
        u32 n = 16;
        if (v && *v) {
            n = static_cast<u32>(std::strtoul(v, nullptr, 10));
        }
        n = std::min<u32>(n, 16); // (the window stays inside the 64 KiB DmaForget covers)
        if (n < 2) {
            LOG_INFO(Render_Vulkan, "[native] fault-around off (BB_NATIVE_FAULTAROUND=0)");
            return 0u;
        }
        n = std::bit_floor(n);
        LOG_INFO(Render_Vulkan,
                 "[native] fault-around on: {} pages ({} KiB) per guest write fault "
                 "(BB_NATIVE_FAULTAROUND=0 off)",
                 n, n * 4);
        return n;
    }();
    return pages;
}

void Rasterizer::FaultAround(VAddr addr) {
    const u32 pages = FaultAroundPages();
    if (pages == 0) {
        return;
    }
    const u64 window = u64{pages} << 12;
    const VAddr begin = addr & ~(window - 1);
    if (!IsMapped(begin, window)) {
        return;
    }
    buffer_cache.FaultAroundMemory(begin, window);
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    DmaForget(addr, size); // bb-dma-dedup
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const AmdGpu::Scissor scsr = regs.GetCombinedScissor();

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            // bb-clean-021-bbport: Supermedo's target_scale (draws into the output-size UI images).
            viewport.x = (xoffset - xscale) * bb_target_scale[0];
            viewport.y = (yoffset - yscale) * bb_target_scale[1];
            // Supermedo (vk_rasterizer.cpp ~1122): jitter geometry, not full-screen passes (a
            // shifted full-screen quad leaves an edge column/row unwritten).
            if (bb_jitter_draw &&
                (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1)) {
                float jx = 0.f, jy = 0.f;
                BB::Upscale::Jitter(jx, jy);
                viewport.x += jx;
                viewport.y += jy;
            } else if (bb_jitter_fs[0] != 0.0f) {
                // bb-jitter-fs: see BbJitterFsEval.
                float jx = 0.f, jy = 0.f;
                BB::Upscale::Jitter(jx, jy);
                viewport.x += jx * bb_jitter_fs[0];
                viewport.y += jy * bb_jitter_fs[1];
            }
            viewport.width = xscale * 2.0f * bb_target_scale[0];
            viewport.height = yscale * 2.0f * bb_target_scale[1];
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x =
                std::min(vp_scsr.bottom_right_x, s16(regs.viewport_scissors[i].bottom_right_x));
            vp_scsr.bottom_right_y =
                std::min(vp_scsr.bottom_right_y, s16(regs.viewport_scissors[i].bottom_right_y));
        }
        if (bb_fullres_draw) {
            scissors.push_back(BbFullResScissor(i));
        } else if (bb_target_scale[0] != 1.0f || bb_target_scale[1] != 1.0f) {
            const auto scale = [](s32 v, float f) { return s32(std::lround(float(v) * f)); };
            const s32 x0 = scale(vp_scsr.top_left_x, bb_target_scale[0]);
            const s32 y0 = scale(vp_scsr.top_left_y, bb_target_scale[1]);
            const s32 x1 = scale(vp_scsr.top_left_x + s32(vp_scsr.GetWidth()), bb_target_scale[0]);
            const s32 y1 = scale(vp_scsr.top_left_y + s32(vp_scsr.GetHeight()), bb_target_scale[1]);
            scissors.push_back({
                .offset = {x0, y0},
                .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
            });
        } else {
            scissors.push_back({
                .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
                .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
            });
        }
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan

namespace BB {
void RtReflectCycleHide() {
    Vulkan::BbRtCycleHide();
}
} // namespace BB
