// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtracing (BB_PATHTRACING=1): the geometry store the path tracer builds its BVH from
// (video_core/renderer_vulkan/rt/pt_capture.h). A vertex shader compiled with
// HwVertexRuntimeInfo::bb_pt_store writes, per vertex of an opaque G-buffer draw, its clip-space
// position as the guest computed it (before depth-mode / clip conversions and before the viewport
// jitter) and, when the rasterizer selected one, the UV (half2) and two normals (snorm4x8) from
// the vertex outputs the pixel shader samples its albedo / normal map with. Port of the store in
// Joao Vitor Alves' path tracing mod (GPL-2.0, emit_spirv_special.cpp EmitVertexMotion: uv_select
// layout of rt_surface.h UvSelect / NormalSelect), with the addresses in the push constants
// instead of baked into the SPIR-V (cached shaders stay valid across runs).
//
// Layout at the store address (capacity C vertices, slot 0 unused):
//   positions vec4  at +slot * 16
//   uvs       half2 at +C * 16 + slot * 4
//   normals   2 x snorm4x8 at +C * 20 + slot * 8
// slot = store_base + (VertexIndex - first_vertex) + (InstanceIndex - first_instance) * vertices.

#include <array>
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {
Id PtPush(EmitContext& ctx, u32 member, u32 component, u32 width) {
    const Id vec = width == 2 ? ctx.U32[2] : ctx.U32[4];
    const Id value = ctx.OpLoad(
        vec, ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, vec),
                               ctx.push_data_block, ctx.ConstU32(member)));
    return ctx.OpCompositeExtract(ctx.U32[1], value, component);
}
} // namespace

void EmitBBPtStore(EmitContext& ctx) {
    if (ctx.objmv_prev_mode || !Sirit::ValidId(ctx.output_position) ||
        !Sirit::ValidId(ctx.vertex_index) || !Sirit::ValidId(ctx.instance_id)) {
        return;
    }
    const Id u32_type = ctx.U32[1];
    const Id bool_type = ctx.U1[1];
    const Id f32 = ctx.F32[1];
    const Id base_address = ctx.OpBitcast(
        ctx.U64, ctx.OpLoad(ctx.U32[2], ctx.OpAccessChain(ctx.TypePointer(
                                                              spv::StorageClass::PushConstant,
                                                              ctx.U32[2]),
                                                          ctx.push_data_block,
                                                          ctx.ConstU32(PushData::PtAddressIndex))));
    const Id store_base = PtPush(ctx, PushData::PtDrawIndex, 0, 4);
    const Id first_vertex = PtPush(ctx, PushData::PtDrawIndex, 1, 4);
    const Id vertices = PtPush(ctx, PushData::PtDrawIndex, 2, 4);
    const Id first_instance = PtPush(ctx, PushData::PtDrawIndex, 3, 4);
    const Id uv_select = PtPush(ctx, PushData::PtMiscIndex, 0, 2);
    const Id capacity = PtPush(ctx, PushData::PtMiscIndex, 1, 2);

    const Id vertex = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.vertex_index), first_vertex);
    const Id instance =
        ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.instance_id), first_instance);
    const Id local = ctx.OpIAdd(ctx.U64, ctx.OpUConvert(ctx.U64, vertex),
                                ctx.OpIMul(ctx.U64, ctx.OpUConvert(ctx.U64, instance),
                                           ctx.OpUConvert(ctx.U64, vertices)));
    const Id slot64 = ctx.OpIAdd(ctx.U64, ctx.OpUConvert(ctx.U64, store_base), local);
    // Off (store base 0: not a captured draw, or path tracing paused with F9), out of the draw's
    // scanned range, or past the store: nothing written.
    const Id do_store = ctx.OpLogicalAnd(
        bool_type,
        ctx.OpLogicalAnd(bool_type, ctx.OpINotEqual(bool_type, store_base, ctx.u32_zero_value),
                         ctx.OpULessThan(bool_type, vertex, vertices)),
        ctx.OpULessThan(bool_type, slot64, ctx.OpUConvert(ctx.U64, capacity)));

    const Id store_label = ctx.OpLabel();
    const Id store_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(store_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_store, store_label, store_merge);
    ctx.AddLabel(store_label);

    const Id cap64 = ctx.OpUConvert(ctx.U64, capacity);
    const auto at = [&](Id region, u64 stride) {
        return ctx.OpIAdd(ctx.U64, region, ctx.OpIMul(ctx.U64, slot64, ctx.Constant(ctx.U64, stride)));
    };
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    const Id f32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.F32[4]);
    ctx.OpStore(ctx.OpConvertUToPtr(f32x4_ptr, at(base_address, 16)), position,
                spv::MemoryAccessMask::Aligned, 16u);

    // UV (and normals) from the vertex outputs the rasterizer picked (UvSelect bit 31).
    const Id uv_wanted = ctx.OpINotEqual(
        bool_type, ctx.OpBitwiseAnd(u32_type, uv_select, ctx.ConstU32(0x80000000u)),
        ctx.u32_zero_value);
    const Id uv_label = ctx.OpLabel();
    const Id uv_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(uv_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(uv_wanted, uv_label, uv_merge);
    ctx.AddLabel(uv_label);
    const Id zero_f = ctx.Constant(f32, 0.0f);
    const Id zero4 = ctx.ConstantComposite(ctx.F32[4], zero_f, zero_f, zero_f, zero_f);
    const auto select_param = [&](Id wanted) {
        Id chosen = zero4;
        for (u32 i = 0; i < ctx.output_params.size(); ++i) {
            const auto& param = ctx.output_params[i];
            if (!Sirit::ValidId(param.id) || param.num_components == 0 ||
                param.num_components > 4 || param.is_integer || param.is_array ||
                param.pointer_type.value == 0) {
                continue;
            }
            Id value;
            if (param.num_components == 1) {
                value = ctx.OpCompositeConstruct(ctx.F32[4], ctx.OpLoad(f32, param.id), zero_f,
                                                 zero_f, zero_f);
            } else {
                const Id loaded = ctx.OpLoad(ctx.F32[param.num_components], param.id);
                if (param.num_components == 4) {
                    value = loaded;
                } else {
                    std::array<Id, 4> parts{};
                    for (u32 c = 0; c < 4; ++c) {
                        parts[c] = c < param.num_components
                                       ? ctx.OpCompositeExtract(f32, loaded, c)
                                       : zero_f;
                    }
                    value = ctx.OpCompositeConstruct(ctx.F32[4], parts);
                }
            }
            chosen = ctx.OpSelect(ctx.F32[4], ctx.OpIEqual(bool_type, wanted, ctx.ConstU32(i)),
                                  value, chosen);
        }
        return chosen;
    };
    const Id u32_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, u32_type);
    {
        const Id normal_wanted = ctx.OpINotEqual(
            bool_type, ctx.OpBitwiseAnd(u32_type, uv_select, ctx.ConstU32(0x08200000u)),
            ctx.u32_zero_value);
        const Id n_label = ctx.OpLabel();
        const Id n_merge = ctx.OpLabel();
        ctx.OpSelectionMerge(n_merge, spv::SelectionControlMask::MaskNone);
        ctx.OpBranchConditional(normal_wanted, n_label, n_merge);
        ctx.AddLabel(n_label);
        const auto field = [&](u32 shift) {
            return ctx.OpBitwiseAnd(u32_type,
                                    ctx.OpShiftRightLogical(u32_type, uv_select, ctx.ConstU32(shift)),
                                    ctx.ConstU32(31u));
        };
        const auto bit = [&](u32 mask) {
            return ctx.OpINotEqual(bool_type, ctx.OpBitwiseAnd(u32_type, uv_select, ctx.ConstU32(mask)),
                                   ctx.u32_zero_value);
        };
        const Id va = ctx.OpSelect(ctx.F32[4], bit(0x00200000u), select_param(field(16)), zero4);
        const Id vb = ctx.OpSelect(ctx.F32[4], bit(0x08000000u), select_param(field(22)), zero4);
        const auto pack = [&](Id v) {
            const Id xyz = ctx.OpVectorShuffle(ctx.F32[3], v, v, 0, 1, 2);
            const Id len = ctx.OpDot(f32, xyz, xyz);
            const Id ok = ctx.OpFOrdGreaterThan(bool_type, len, ctx.Constant(f32, 1e-12f));
            const Id unit = ctx.OpNormalize(ctx.F32[3], xyz);
            const Id packed = ctx.OpPackSnorm4x8(
                u32_type, ctx.OpCompositeConstruct(ctx.F32[4], ctx.OpCompositeExtract(f32, unit, 0u),
                                                   ctx.OpCompositeExtract(f32, unit, 1u),
                                                   ctx.OpCompositeExtract(f32, unit, 2u), zero_f));
            return ctx.OpSelect(u32_type, ok, packed, ctx.u32_zero_value);
        };
        const Id normals = ctx.OpIAdd(ctx.U64, base_address,
                                      ctx.OpIMul(ctx.U64, cap64, ctx.Constant(ctx.U64, u64(20))));
        const Id n_address = at(normals, 8);
        ctx.OpStore(ctx.OpConvertUToPtr(u32_ptr, n_address), pack(va),
                    spv::MemoryAccessMask::Aligned, 4u);
        ctx.OpStore(ctx.OpConvertUToPtr(
                        u32_ptr, ctx.OpIAdd(ctx.U64, n_address, ctx.Constant(ctx.U64, u64(4)))),
                    pack(vb), spv::MemoryAccessMask::Aligned, 4u);
        ctx.OpBranch(n_merge);
        ctx.AddLabel(n_merge);
    }
    const Id chosen = select_param(ctx.OpBitwiseAnd(u32_type, uv_select, ctx.ConstU32(0xFFu)));
    const Id comp_u = ctx.OpBitwiseAnd(
        u32_type, ctx.OpShiftRightLogical(u32_type, uv_select, ctx.ConstU32(8u)), ctx.ConstU32(3u));
    const Id comp_v = ctx.OpBitwiseAnd(
        u32_type, ctx.OpShiftRightLogical(u32_type, uv_select, ctx.ConstU32(12u)), ctx.ConstU32(3u));
    const Id uv = ctx.OpCompositeConstruct(ctx.F32[2],
                                           ctx.OpVectorExtractDynamic(f32, chosen, comp_u),
                                           ctx.OpVectorExtractDynamic(f32, chosen, comp_v));
    const Id uvs = ctx.OpIAdd(ctx.U64, base_address,
                              ctx.OpIMul(ctx.U64, cap64, ctx.Constant(ctx.U64, u64(16))));
    ctx.OpStore(ctx.OpConvertUToPtr(u32_ptr, at(uvs, 4)), ctx.OpPackHalf2x16(u32_type, uv),
                spv::MemoryAccessMask::Aligned, 4u);
    ctx.OpBranch(uv_merge);
    ctx.AddLabel(uv_merge);

    ctx.OpBranch(store_merge);
    ctx.AddLabel(store_merge);
}

} // namespace Shader::Backend::SPIRV
