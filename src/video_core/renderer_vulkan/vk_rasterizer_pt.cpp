// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtracing: the rasterizer side of Joao Vitor Alves' path tracing mod (rt/, GPL-2.0),
// mapped from his vk_rasterizer.cpp hunk for SuperMedo's renderer onto ours. Only built with
// ENABLE_PATHTRACING; every entry point returns at once unless BB_PATHTRACING=1 started the
// engine. What is wired:
//  - frame start: the first draw into a display buffer (as the mod / SuperMedo's upscaler):
//    OnDisplayPass (finishes the tracer's frame), the geometry store's half flips;
//  - geometry: opaque G-buffer draws (5+ colour targets with depth, triangle lists) get the
//    storing vertex shader (pt_capture.h); per draw its index range, store slots, UV select and
//    texture views go to AddDraw / OnDrawBound; the G-buffer targets to OnGBufferPass and their
//    depth is the scene depth;
//  - lighting: the light accumulation draws (VS 0x8747a367: state, targets, constants, the skip
//    when the traced direct light replaced them), the direct-light hook before the hook draw,
//    GI after the light composite (PS 0x32e47788), the AO and directional shadow inputs, the
//    post trigger dispatch (shadows, sky, fog, exposure), the scene colour target, sky dome;
//  - camera: our scene block capture (bb_upscale.cpp, floats 180-191 / 52-63).
// Not wired (defaults off in every preset): water / fog sheets, smoke / flame / glow capture,
// alpha inheritance from depth prepasses, indirect draws into the BVH.

#ifdef ENABLE_PATHTRACING

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <xxhash.h>

#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/bb_ui_cut.h"
#include "video_core/renderer_vulkan/rt/pt_capture.h"
#include "video_core/renderer_vulkan/rt/pt_rr.h"
#include "video_core/renderer_vulkan/rt/vk_ray_tracing.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {
// Vertex shader constant buffers of Bloodborne's G-buffer shaders (SuperMedo's motion_history.h):
// 864-byte scene constants, 416-byte model constants, palettes of 3x4 bone matrices (48 bytes a
// bone; characters 640+ bytes, weapons and props 96-384).
enum class BufferRole { Other, Skeleton, SmallSkeleton };
BufferRole ClassifyBuffer(u32 size) {
    if (size == 864 || size > 16384) {
        return BufferRole::Other;
    }
    if (size >= 640) {
        return BufferRole::Skeleton;
    }
    return size >= 96 && size % 48 == 0 ? BufferRole::SmallSkeleton : BufferRole::Other;
}

std::pair<u32, u32> PtDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
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

RayTracing::SceneInputs Inputs(VideoCore::ImageId depth) {
    RayTracing::SceneInputs in{};
    in.depth = depth;
    float inv_view[12]{}, proj[4]{};
    in.camera_valid = BB::Upscale::PtCamera(inv_view, proj);
    if (in.camera_valid) {
        std::memcpy(in.inv_view.data(), inv_view, sizeof(inv_view));
        std::memcpy(in.projection.data(), proj, sizeof(proj));
    }
    float jx = 0.0f, jy = 0.0f;
    BB::Upscale::Jitter(jx, jy);
    in.jitter = {jx, jy};
    in.scene_targets = nullptr;
    in.reduced = false; // the scene renders at the guest size here
    return in;
}

u64 Low32(u64 hash) {
    return hash & 0xFFFFFFFFull;
}

// The alpha-test reference for shaders whose own one was not found (the mod's 0.5);
// BB_PT_ALPHA_REF overrides it (0.05-0.95).
float DefaultAlphaRef() {
    static const float value = [] {
        const char* v = std::getenv("BB_PT_ALPHA_REF");
        const float f = v && *v ? float(std::atof(v)) : 0.5f;
        return f >= 0.05f && f <= 0.95f ? f : 0.5f;
    }();
    return value;
}

struct AlphaRefStats {
    u64 literal{}, constant{}, unknown{};
} pt_alpha_ref_stats;
} // namespace

void Rasterizer::PtInit() {
    if (!RayTracing::Wanted()) {
        return;
    }
    // The store must exist before the engine reads its addresses (PtCapture, constructor).
    if (instance.IsRayQuerySupported() && instance.IsBufferDeviceAddressSupported()) {
        pt_store = std::make_unique<PtStore>(instance);
        if (!pt_store->Ready()) {
            pt_store.reset();
        }
    }
    ray_tracing = std::make_unique<RayTracing>(instance, scheduler, runtime, texture_cache,
                                               buffer_cache);
    PtCapture::active = ray_tracing->Enabled() && pt_store != nullptr;
    if (!ray_tracing->Enabled()) {
        ray_tracing.reset();
        pt_store.reset();
        PtCapture::active = false;
    }
    if (ray_tracing && PtCapture::active) {
        // DLSS Ray Reconstruction for the traced frames (the upscaler pass asks it first).
        pt_rr = std::make_unique<PtRayReconstruction>(instance, scheduler, runtime, texture_cache,
                                                      *ray_tracing);
    }
    LOG_INFO(Render_Vulkan, "[pt] path tracing engine {}",
             PtCapture::active ? "on: G-buffer draws store their geometry, F9 toggles"
                               : "off (see the 'Ray tracing:' / 'Path tracing:' lines above)");
}

void Rasterizer::PtFrameStart() {
    auto& rt = *ray_tracing;
    rt.OnDisplayPass(Inputs(pt_scene_depth));
    if (pt_store) {
        pt_store->BeginFrame();
    }
    BB::Upscale::PtFrameStart();
    pt_scene_depth = {};
    static u64 frames = 0;
    static auto last = std::chrono::steady_clock::now();
    ++frames;
    const auto now = std::chrono::steady_clock::now();
    if (pt_store && now - last >= std::chrono::seconds(5)) {
        const auto& s = pt_store->stats;
        LOG_INFO(Render_Vulkan,
                 "[pt] geometry store, last 5 s: {} frames, {} draws stored ({:.0f} a frame), "
                 "{:.2f} M vertices a frame, peak {:.2f} M of {} M a frame, {} draws over the "
                 "store, {} small draws left out near the top (BB_PT_STORE_RESERVE), index "
                 "scans {} (cached {}, changed {}){}",
                 frames, s.draws, double(s.draws) / std::max<u64>(frames, 1),
                 double(s.vertices) / std::max<u64>(frames, 1) / 1e6, s.peak / 1e6,
                 (pt_store->Capacity() / 2) >> 20, s.full, s.reserved, s.scans, s.hits, s.stale,
                 RayTracing::Paused() ? " (paused with F9)" : "");
        auto& a = pt_alpha_ref_stats;
        LOG_INFO(Render_Vulkan,
                 "[pt] alpha-tested G-buffer draws, last 5 s: {} with the shader's own reference "
                 "literal, {} with it read from the material constants, {} at {:.2f} "
                 "(BB_PT_ALPHA_REF; reference not found in the shader)",
                 a.literal, a.constant, a.unknown, DefaultAlphaRef());
        a = {};
        pt_store->stats = {};
        frames = 0;
        last = now;
    }
}

bool Rasterizer::PtPrepare(const GraphicsPipeline* pipeline) {
    pt_gbuffer = pt_light_draw = pt_ao_draw = pt_dir_shadow = pt_skip_light = false;
    pt_composite_draw = pt_composite_ps = pt_capture_tex = pt_draw_added = false;
    pt_albedo = pt_normal = pt_alpha = pt_sky = {};
    pt_alpha_channel = 3;
    pt_alpha_ref_cb = -1.0f;
    if (!ray_tracing) {
        return true;
    }
    auto& rt = *ray_tracing;
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;

    // Frame start: the pass copying the finished frame into a display buffer.
    const bool display =
        regs.color_buffers[0] && BB::UiCut::IsDisplayBuffer(regs.color_buffers[0].Address());
    if (display && !pt_display_pass) {
        PtFrameStart();
    }
    pt_display_pass = display;

    const bool has_ps = key.mrt_mask && (pipeline->GetStages()[u32(Shader::SwStage::Fragment)] != nullptr);
    const u64 ps_hash = has_ps ? pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash : 0;
    const u64 vs_hash = (pipeline->GetStages()[u32(Shader::SwStage::Vertex)] != nullptr)
                            ? pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash
                            : 0;
    if (has_ps && rt.SkipPs(ps_hash)) {
        return false;
    }

    // G-buffer pass: 5+ colour targets with depth (the mod's gbuffer_draw); a smaller depth than
    // the scene's is a side view (reflection / cubemap) and stays out.
    if (db_desc.image_id && std::popcount(key.mrt_mask) >= int(PtCapture::GBufferTargets)) {
        const auto& depth = texture_cache.GetImage(db_desc.image_id);
        const bool side = BB::Upscale::g_scene_w && depth.info.size.width < BB::Upscale::g_scene_w;
        if (!side) {
            pt_gbuffer = true;
            pt_scene_depth = db_desc.image_id;
            std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS> targets{};
            for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
                targets[cb] = cb_descs[cb].image_id;
            }
            rt.OnGBufferPass(targets);
            pt_capture_tex = pipeline->PtStore();
        }
    }
    // Scene colour: a scene-size RGBA16F target drawn with the scene depth.
    if (pt_scene_depth && db_desc.image_id == pt_scene_depth && cb_descs[0].image_id &&
        std::popcount(key.mrt_mask) <= 2) {
        const auto& color = texture_cache.GetImage(cb_descs[0].image_id);
        const auto& depth = texture_cache.GetImage(db_desc.image_id);
        if (color.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
            color.info.size.width == depth.info.size.width &&
            color.info.size.height == depth.info.size.height) {
            rt.OnSceneColor(cb_descs[0].image_id);
        }
    }

    pt_composite_draw = rt.WantsDirect() && Rt::DirectHookDraw(ps_hash);
    pt_composite_ps = has_ps && ps_hash == Rt::LightCompositeShader;
    if (rt.WantsLightBuffers() && u32(vs_hash) == Rt::LightAccVertexShader) {
        rt.OnLightDraw(ps_hash);
        rt.OnLightStencil(Rt::PackLightStencil(
            regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid(),
            u32(regs.depth_control.stencil_ref_func), regs.stencil_ref_front.stencil_test_val,
            regs.stencil_ref_front.stencil_mask));
        {
            Rt::LightDrawState st{};
            st.valid = true;
            if ((key.mrt_mask & 1u) && cb_descs[0].image_id) {
                st.diffuse = cb_descs[0].image_id.index;
                st.width = texture_cache.GetImage(cb_descs[0].image_id).info.size.width;
            }
            if ((key.mrt_mask & 2u) && cb_descs[1].image_id) {
                st.specular = cb_descs[1].image_id.index;
            }
            st.mask0 = u8(regs.color_target_mask.GetMask(0));
            st.mask1 = u8(regs.color_target_mask.GetMask(1));
            const auto& b0 = regs.blend_control[0];
            const auto& b1 = regs.blend_control[1];
            st.blend0 = b0.enable && !regs.color_buffers[0].info.blend_bypass;
            st.blend1 = b1.enable && !regs.color_buffers[1].info.blend_bypass;
            st.src0 = u8(b0.color_src_factor);
            st.dst0 = u8(b0.color_dst_factor);
            st.func0 = u8(b0.color_func);
            st.src1 = u8(b1.color_src_factor);
            st.dst1 = u8(b1.color_dst_factor);
            st.func1 = u8(b1.color_func);
            st.depth = regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
            st.zfunc = u8(regs.depth_control.depth_func);
            st.bounds = regs.depth_control.depth_bounds_enable;
            st.bounds_min = regs.depth_bounds_min;
            st.bounds_max = regs.depth_bounds_max;
            st.depth_write = regs.depth_control.depth_write_enable;
            st.cull = u8(regs.polygon_control.CullingMode());
            st.zscale = regs.viewport_control.zscale_enable ? regs.viewports[0].zscale : 1.0f;
            st.zoffset = regs.viewport_control.zoffset_enable ? regs.viewports[0].zoffset : 0.0f;
            rt.OnLightState(st);
        }
        pt_light_draw = ps_hash != 0;
        pt_ao_draw = ps_hash == Rt::DeferredAoShader;
        pt_dir_shadow = rt.WantsDirShadowMask(ps_hash);
        if (pt_light_draw) {
            const VideoCore::ImageId diffuse =
                (key.mrt_mask & 1u) ? cb_descs[0].image_id : VideoCore::ImageId{};
            const VideoCore::ImageId specular =
                (key.mrt_mask & 2u) ? cb_descs[1].image_id : VideoCore::ImageId{};
            const bool blended =
                regs.blend_control[0].enable && !regs.color_buffers[0].info.blend_bypass;
            rt.OnLightTargets(diffuse, specular);
            pt_skip_light = rt.SkipLightDraw(
                diffuse, specular, Rt::DirectHookDraw(ps_hash), blended,
                regs.depth_control.depth_enable && regs.depth_control.depth_write_enable,
                regs.depth_control.stencil_enable &&
                    (regs.stencil_ref_front.stencil_write_mask ||
                     regs.stencil_ref_back.stencil_write_mask));
            pt_skip_light = pt_skip_light || rt.SkipHalfResLight(diffuse, blended, ps_hash);
        }
    }
    if (rt.WantsDirect() && key.mrt_mask) {
        rt.OnAccTargetDraw(cb_descs[0].image_id, vs_hash, ps_hash, key.mrt_mask, pt_light_draw);
    }
    if (pt_composite_draw) {
        rt.OnLightComposite(Inputs(pt_scene_depth), ps_hash);
    }
    if (pt_ao_draw && rt.WantsAo()) {
        rt.OnAoDraw(Inputs(pt_scene_depth));
    }
    return true;
}

void Rasterizer::PtStoreDraw(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset) {
    if (!ray_tracing || !pt_store || !pt_gbuffer || !pipeline->PtStore() ||
        RayTracing::Paused()) {
        return; // push store base 0: the variant stores nothing
    }
    auto& rt = *ray_tracing;
    const auto& regs = liverpool->regs;
    if (regs.primitive_type != AmdGpu::PrimitiveType::TriangleList || regs.num_indices < 3) {
        return;
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch = pipeline->GetFetchShader();
    const auto [base_vertex, first_instance] = PtDrawOffsets(regs, vs, fetch);
    const u32 instances = std::max(regs.num_instances.NumInstances(), 1u);
    const u32 index_size = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
    const VAddr index_address =
        is_indexed ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size : 0;
    const bool restart = regs.enable_primitive_restart != 0;
    PtStore::Range range{base_vertex, regs.num_indices, 0};
    if (is_indexed) {
        const u64 bytes = u64(regs.num_indices) * index_size;
        if (!index_address || !memory->IsValidMapping(index_address, bytes)) {
            return;
        }
        range = pt_store->IndexRange(reinterpret_cast<const void*>(index_address), index_address,
                                     regs.num_indices, index_size, s32(base_vertex), restart);
        if (!range.count) {
            return;
        }
    }
    // Geometry identity (the engine's BLAS cache key): the vertex streams' sharps.
    u64 geometry = vs.pgm_hash | 1ull;
    for (const auto& attr : fetch.attributes) {
        const AmdGpu::Buffer sharp = attr.GetSharp(vs);
        geometry = XXH3_64bits_withSeed(&sharp, sizeof(sharp), geometry);
    }
    // Skinning: a bone palette (character skeleton or small rigid palette) in the VS buffers.
    bool skinned = false, unreadable = false;
    u64 palette = 0;
    for (const auto& resource : vs.buffers) {
        if (resource.IsSpecial()) {
            continue;
        }
        const auto buffer = resource.GetSharp(vs);
        const u32 size = buffer.GetSize();
        if (!buffer.Valid() || buffer.GetStride() != 16) {
            continue;
        }
        const auto role = ClassifyBuffer(size);
        if (role == BufferRole::Other) {
            continue;
        }
        skinned |= role == BufferRole::Skeleton;
        if (memory->IsValidMapping(buffer.base_address, size)) {
            palette = XXH3_64bits_withSeed(reinterpret_cast<const void*>(buffer.base_address),
                                           size, palette);
        } else if (role == BufferRole::Skeleton) {
            unreadable = true;
        }
    }
    const u32 store_base = pt_store->Allocate(u64(range.count) * instances);
    if (!store_base) {
        return;
    }

    // Surface: UV (and normals) from the pixel shader's albedo / alpha-test inputs.
    u32 uv_select = 0;
    bool alpha_test = false;
    float alpha_ref = DefaultAlphaRef();
    Rt::DrawTag tag{};
    tag.vs = vs.pgm_hash;
    tag.mrt_mask = u32(pipeline->GetGraphicsKey().mrt_mask);
    tag.target_mask = regs.color_target_mask.raw;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            tag.bound_mask |= 1u << cb;
        }
    }
    tag.cb0 = regs.color_buffers[0] ? regs.color_buffers[0].Address() : 0;
    tag.zbuf = regs.depth_buffer.DepthValid() ? regs.depth_buffer.DepthAddress() : 0;
    tag.ztest = u32(regs.depth_control.depth_enable);
    tag.zfunc = u32(regs.depth_control.depth_func);
    tag.zwrite = u32(regs.depth_control.depth_write_enable);
    tag.stencil = u32(regs.depth_control.stencil_enable);
    tag.prim = u32(regs.primitive_type);
    tag.cull = u32(regs.polygon_control.CullingMode());
    if ((pipeline->GetStages()[u32(Shader::SwStage::Fragment)] != nullptr)) {
        const auto& ps = pipeline->GetStage(Shader::SwStage::Fragment);
        tag.ps = ps.pgm_hash;
        alpha_test = ps.has_discard;
        tag.caster = u32(alpha_test ? Rt::CasterAlpha::Own : Rt::CasterAlpha::None);
        // bb-pathtracing-2: the game's own alpha-test reference (FindAlphaRef): a literal in the
        // shader, or the material constant read in PtBufferBound for this draw.
        if (alpha_test && ps.rt_info_known) {
            if (ps.rt_alpha_ref > 0.0f && ps.rt_alpha_ref < 1.0f) {
                alpha_ref = ps.rt_alpha_ref;
                ++pt_alpha_ref_stats.literal;
            } else if (ps.rt_alpha_ref_buffer >= 0 && pt_alpha_ref_cb > 0.0f &&
                       pt_alpha_ref_cb < 1.0f) {
                alpha_ref = pt_alpha_ref_cb;
                ++pt_alpha_ref_stats.constant;
            } else {
                ++pt_alpha_ref_stats.unknown;
            }
        }
        if (ps.rt_info_known) {
            const bool alpha_uv = alpha_test && Rt::AlphaSourceWanted() &&
                                  ps.rt_alpha_input >= 0 && ps.rt_alpha_input < 32 &&
                                  (ps.rt_albedo_input < 0 ||
                                   (ps.rt_albedo_input == ps.rt_alpha_input &&
                                    ps.rt_albedo_components == ps.rt_alpha_components));
            const s32 uv_input = alpha_uv ? ps.rt_alpha_input : ps.rt_albedo_input;
            const u8 uv_components = alpha_uv ? ps.rt_alpha_components : ps.rt_albedo_components;
            if (uv_input >= 0 && uv_input < 32) {
                uv_select = Rt::UvSelect(s32(regs.ps_inputs[uv_input].input_offset),
                                         uv_components & 3u, (uv_components >> 4) & 3u);
            }
            if (uv_select) {
                const auto input_param = [&](s32 input) {
                    return input >= 0 && input < 32 ? s32(regs.ps_inputs[input].input_offset)
                                                    : -1;
                };
                uv_select |= Rt::NormalSelect(input_param(ps.rt_normal_inputs[0]),
                                              input_param(ps.rt_normal_inputs[1]));
            }
        }
    }

    const u64 address = pt_store->Address();
    push_data.pt_address = {u32(address), u32(address >> 32)};
    push_data.pt_draw = {store_base, range.first, range.count, first_instance};
    push_data.pt_misc = {uv_select, pt_store->Capacity()};

    rt.AddDraw({
        .key = {.shader = vs.pgm_hash,
                .geometry = geometry,
                .indices = index_address,
                .topology = range.topology,
                .index_count = regs.num_indices,
                .instances = instances,
                .first_instance = first_instance,
                .first_vertex = range.first,
                .vertex_count = range.count},
        .palette = palette,
        .skinned = skinned,
        .unreadable = unreadable,
        .store_base = store_base,
        .first_vertex = range.first,
        .vertex_count = range.count,
        .instances = instances,
        .base_vertex = s32(base_vertex),
        .indices = is_indexed ? reinterpret_cast<const void*>(index_address) : nullptr,
        .index_count = regs.num_indices,
        .index_size = is_indexed ? index_size : 0u,
        .restart = is_indexed && restart,
        .alpha_test = alpha_test,
        .uv_captured = uv_select != 0,
        .normal_captured = Rt::NormalSelectValid(uv_select),
        .depth_only = false,
        .alpha_ref = alpha_ref,
        .blended = false,
        .tag = tag,
    });
    pt_draw_added = true;
}

void Rasterizer::PtAfterBind() {
    if (!ray_tracing) {
        return;
    }
    if (pt_draw_added) {
        const auto valid = [&](const PtView& v) {
            const auto* image = v.image ? texture_cache.TryGetImageRt(v.image, 0) : nullptr;
            return image && image->image_uid == v.uid;
        };
        const PtView albedo = valid(pt_albedo) ? pt_albedo : PtView{};
        const PtView normal = valid(pt_normal) ? pt_normal : PtView{};
        const PtView alpha = valid(pt_alpha) ? pt_alpha : PtView{};
        ray_tracing->OnDrawBound(albedo.view, albedo.image, albedo.uid, normal.view, normal.image,
                                 alpha.view, alpha.image, alpha.uid, pt_alpha_channel);
    }
    if (pt_sky.view) {
        ray_tracing->OnSkyDome(pt_sky.view, pt_sky.image, pt_sky.uid);
        pt_sky = {};
    }
}

void Rasterizer::PtAfterDraw() {
    if (ray_tracing && pt_composite_ps) {
        ray_tracing->OnAfterComposite(Inputs(pt_scene_depth));
    }
}

void Rasterizer::PtDispatch(u64 cs_hash) {
    if (ray_tracing) {
        ray_tracing->OnDispatch(cs_hash, Inputs(pt_scene_depth));
    }
}

void Rasterizer::PtCaptureTextures(const Shader::Info& stage, u32 first_image_idx,
                                   std::span<const u32> sizes) {
    if (!ray_tracing || stage.sw_stage != Shader::SwStage::Fragment) {
        return;
    }
    const auto view_of = [&](s32 resource) -> PtView {
        if (resource < 0 || u32(resource) >= sizes.size()) {
            return {};
        }
        u32 flat = 0;
        for (s32 i = 0; i < resource; ++i) {
            flat += sizes[i];
        }
        if (first_image_idx + flat >= image_infos.size() || flat >= image_bindings.size()) {
            return {};
        }
        const VideoCore::ImageId id = image_bindings[flat].image_id;
        if (!id) {
            return {};
        }
        return {image_infos[first_image_idx + flat].imageView, id,
                texture_cache.GetImage(id).image_uid};
    };
    if (u32(stage.pgm_hash) == Rt::SkyDomePs) {
        pt_sky = view_of(0);
    }
    if (!pt_capture_tex || !stage.rt_info_known) {
        return;
    }
    pt_albedo = view_of(stage.rt_albedo_image);
    if (stage.rt_normal_image != stage.rt_albedo_image) {
        pt_normal = view_of(stage.rt_normal_image);
    }
    if (stage.has_discard && stage.rt_alpha_image >= 0) {
        pt_alpha = view_of(stage.rt_alpha_image);
        pt_alpha_channel = stage.rt_alpha_channel;
    }
}

void Rasterizer::PtBufferBound(const Shader::Info& stage, u32 index, bool formatted, bool written,
                               VAddr address, u32 size) {
    if (!ray_tracing || !size) {
        return;
    }
    auto& rt = *ray_tracing;
    const bool fragment = stage.sw_stage == Shader::SwStage::Fragment;
    const bool vertex = stage.sw_stage == Shader::SwStage::Vertex;
    if (written) {
        rt.OnLightSetStorageWrite(address, size, stage.pgm_hash, u32(stage.sw_stage));
        return;
    }
    if (!memory->IsValidMapping(address, size)) {
        return;
    }
    const auto* data = reinterpret_cast<const void*>(address);
    if (fragment && pt_capture_tex && s32(index) == stage.rt_alpha_ref_buffer &&
        u64(stage.rt_alpha_ref_dword) * 4 + 4 <= size) {
        float ref;
        std::memcpy(&ref, static_cast<const u8*>(data) + u64(stage.rt_alpha_ref_dword) * 4,
                    sizeof(ref));
        pt_alpha_ref_cb = std::isfinite(ref) ? ref : -1.0f;
    }
    if (pt_light_draw && fragment) {
        rt.OnLightBuffer(size, data);
        rt.OnLightSetBuffer(address, size);
    }
    if (pt_composite_ps && fragment) {
        rt.OnCompositeBuffer(index, address, size);
    }
    if (pt_light_draw && vertex && !formatted) {
        rt.OnLightVtxParam(size, data);
    }
}

void Rasterizer::PtImageBound(const Shader::Info& stage, VideoCore::ImageId image_id,
                              bool is_storage) {
    if (!ray_tracing || is_storage || stage.sw_stage != Shader::SwStage::Fragment) {
        return;
    }
    if (pt_ao_draw) {
        ray_tracing->OnAoInput(image_id);
    }
    if (pt_dir_shadow) {
        ray_tracing->OnDirShadowInput(image_id);
    }
    if (pt_composite_draw) {
        ray_tracing->OnCompositeInput(image_id, false);
    }
}

} // namespace Vulkan

#endif // ENABLE_PATHTRACING
