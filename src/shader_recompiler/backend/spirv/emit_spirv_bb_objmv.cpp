// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc:
//                         gpu/shader/backend/spirv/emit_spirv_special.cpp, EmitVertexMotion /
//                         EmitFragmentMotion, GPL-2.0-or-later; adapted)
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB object motion vectors (video_core/bb/bb_objmv.h). In the G-buffer pass every vertex shader
// stores its clip position for this frame (per draw, per vertex: a slot in a device-address array)
// and loads the one the same vertex had in the previous frame; the fragment shader writes the
// screen-space difference (previous minus current, guest viewport pixels) and its depth to colour
// attachment 7. bb-clean-021-objmv: only draws that moved since the previous frame take this
// shader variant (the rest of the G-buffer pass keeps its shaders, attachment 7 write-masked).
// Skinned characters, weapons and other animated objects then get their own motion instead of the
// camera-only motion reconstructed from depth (bb_mv.comp), which smeared them under frame
// generation and the temporal upscalers.
//
// Push constants (resource.h, PushData): motion_param (entry in the parameter array, 0 = none),
// the parameter array's address and the position array's address (lo, hi). Parameter entry,
// 32 bytes: u32x4 (store base, load base, vertices per instance, flags), u32x4 (first vertex,
// first instance, instances, 0). Entry 0 is all zero (nothing stored or loaded).

#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {
constexpr u32 FlagStore = 1;
constexpr u32 FlagLoad = 2;

Id PushU32(EmitContext& ctx, u32 member) {
    return ctx.OpLoad(ctx.U32[1],
                      ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]),
                                        ctx.push_data_block, ctx.ConstU32(member)));
}

Id PushAddress(EmitContext& ctx, u32 member) {
    const Id pair = ctx.OpLoad(
        ctx.U32[2], ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[2]),
                                      ctx.push_data_block, ctx.ConstU32(member)));
    return ctx.OpBitcast(ctx.U64, pair);
}

Id PushF32(EmitContext& ctx, u32 member) {
    return ctx.OpLoad(ctx.F32[1],
                      ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.F32[1]),
                                        ctx.push_data_block, ctx.ConstU32(member)));
}
} // namespace

void EmitBBVertexMotion(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.motion_out_cur)) {
        return;
    }
    const Id u32_type = ctx.U32[1];
    const Id bool_type = ctx.U1[1];
    // The guest clip position, before the host conversions (depth mode, clip space).
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    const Id param_index = PushU32(ctx, PushData::MotionParamIndex);
    const Id params_base = PushAddress(ctx, PushData::MotionParamsAddrIndex);
    const Id positions_base = PushAddress(ctx, PushData::MotionPositionsAddrIndex);
    const auto address = [&](Id base, Id index, u32 stride, u32 extra) {
        return ctx.OpIAdd(ctx.U64,
                          ctx.OpIAdd(ctx.U64, base,
                                     ctx.OpIMul(ctx.U64, ctx.OpUConvert(ctx.U64, index),
                                                ctx.Constant(ctx.U64, u64(stride)))),
                          ctx.Constant(ctx.U64, u64(extra)));
    };
    const Id u32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.U32[4]);
    const Id f32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.F32[4]);
    const Id params =
        ctx.OpLoad(ctx.U32[4], ctx.OpConvertUToPtr(u32x4_ptr, address(params_base, param_index, 32, 0)),
                   spv::MemoryAccessMask::Aligned, 16u);
    const Id offsets = ctx.OpLoad(
        ctx.U32[4], ctx.OpConvertUToPtr(u32x4_ptr, address(params_base, param_index, 32, 16)),
        spv::MemoryAccessMask::Aligned, 16u);
    const Id store_base = ctx.OpCompositeExtract(u32_type, params, 0u);
    const Id load_base = ctx.OpCompositeExtract(u32_type, params, 1u);
    const Id vertices = ctx.OpCompositeExtract(u32_type, params, 2u);
    const Id flags = ctx.OpCompositeExtract(u32_type, params, 3u);
    const Id first_vertex = ctx.OpCompositeExtract(u32_type, offsets, 0u);
    const Id first_instance = ctx.OpCompositeExtract(u32_type, offsets, 1u);
    const Id instances = ctx.OpCompositeExtract(u32_type, offsets, 2u);
    const Id vertex = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.vertex_index), first_vertex);
    const Id instance = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.instance_id), first_instance);
    const Id in_range = ctx.OpLogicalAnd(bool_type, ctx.OpULessThan(bool_type, vertex, vertices),
                                         ctx.OpULessThan(bool_type, instance, instances));
    const Id slot = ctx.OpIAdd(u32_type, vertex, ctx.OpIMul(u32_type, instance, vertices));
    const auto flag = [&](u32 bit) {
        return ctx.OpLogicalAnd(
            bool_type, in_range,
            ctx.OpINotEqual(bool_type, ctx.OpBitwiseAnd(u32_type, flags, ctx.ConstU32(bit)),
                            ctx.u32_zero_value));
    };
    const Id do_store = flag(FlagStore);
    const Id do_load = flag(FlagLoad);

    const Id store_label = ctx.OpLabel();
    const Id store_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(store_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_store, store_label, store_merge);
    ctx.AddLabel(store_label);
    {
        // An indexed draw runs a vertex more than once: atomic component stores (all of them
        // write the same value) avoid a write/write race.
        const Id store_address =
            address(positions_base, ctx.OpIAdd(u32_type, store_base, slot), 16, 0);
        const Id scalar_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, u32_type);
        const Id scope = ctx.ConstU32(static_cast<u32>(spv::Scope::Device));
        for (u32 i = 0; i < 4; ++i) {
            const Id ptr = ctx.OpConvertUToPtr(
                scalar_ptr, ctx.OpIAdd(ctx.U64, store_address, ctx.Constant(ctx.U64, u64(i * 4))));
            const Id bits =
                ctx.OpBitcast(u32_type, ctx.OpCompositeExtract(ctx.F32[1], position, i));
            ctx.OpAtomicExchange(u32_type, ptr, scope, ctx.u32_zero_value, bits);
        }
    }
    ctx.OpBranch(store_merge);
    ctx.AddLabel(store_merge);

    const Id load_label = ctx.OpLabel();
    const Id load_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(load_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_load, load_label, load_merge);
    ctx.AddLabel(load_label);
    const Id loaded = ctx.OpLoad(
        ctx.F32[4],
        ctx.OpConvertUToPtr(f32x4_ptr,
                            address(positions_base, ctx.OpIAdd(u32_type, load_base, slot), 16, 0)),
        spv::MemoryAccessMask::Aligned, 16u);
    ctx.OpBranch(load_merge);
    ctx.AddLabel(load_merge);
    Id previous = ctx.OpPhi(ctx.F32[4], position, store_merge, loaded, load_label);
    // The previous position's z carries "valid" (z itself is not needed for screen motion).
    const Id valid = ctx.OpSelect(ctx.F32[1], do_load, ctx.Constant(ctx.F32[1], 1.0f),
                                  ctx.Constant(ctx.F32[1], 0.0f));
    previous = ctx.OpCompositeInsert(ctx.F32[4], valid, previous, 2u);
    ctx.OpStore(ctx.motion_out_cur, position);
    ctx.OpStore(ctx.motion_out_prev, previous);
}

void EmitBBFragmentMotion(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.motion_frag_out)) {
        return;
    }
    const Id f32_type = ctx.F32[1];
    const Id u32_type = ctx.U32[1];
    const Id bool_type = ctx.U1[1];
    const Id current = ctx.OpLoad(ctx.F32[4], ctx.motion_in_cur);
    const Id previous = ctx.OpLoad(ctx.F32[4], ctx.motion_in_prev);
    const Id xscale = PushF32(ctx, PushData::XScaleIndex);
    const Id yscale = PushF32(ctx, PushData::YScaleIndex);
    const auto ndc = [&](Id clip, u32 component) {
        return ctx.OpFDiv(f32_type, ctx.OpCompositeExtract(f32_type, clip, component),
                          ctx.OpCompositeExtract(f32_type, clip, 3u));
    };
    // Guest viewport pixels, previous minus current (bb_mv.comp's convention).
    const Id dx =
        ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 0u), ndc(current, 0u)), xscale);
    const Id dy =
        ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 1u), ndc(current, 1u)), yscale);
    const Id epsilon = ctx.Constant(f32_type, 1e-5f);
    const Id front = ctx.OpLogicalAnd(
        bool_type,
        ctx.OpFOrdGreaterThan(bool_type, ctx.OpCompositeExtract(f32_type, previous, 3u), epsilon),
        ctx.OpFOrdGreaterThan(bool_type, ctx.OpCompositeExtract(f32_type, current, 3u), epsilon));
    // |v| < 512 px on both axes (false for NaN / Inf too): a draw matched to another one's
    // positions (identity swap) gives a huge vector; it keeps the camera motion instead.
    const Id limit = ctx.Constant(f32_type, 512.0f);
    const Id sane = ctx.OpLogicalAnd(
        bool_type, ctx.OpFOrdLessThan(bool_type, ctx.OpFAbs(f32_type, dx), limit),
        ctx.OpFOrdLessThan(bool_type, ctx.OpFAbs(f32_type, dy), limit));
    const Id loaded = ctx.OpFOrdGreaterThan(bool_type, ctx.OpCompositeExtract(f32_type, previous, 2u),
                                            ctx.Constant(f32_type, 0.5f));
    const Id written =
        ctx.OpLogicalAnd(bool_type, ctx.OpLogicalAnd(bool_type, front, sane), loaded);
    // y: the depth this fragment wrote (bits); bb_objmv.comp takes the vector only where the
    // final scene depth is still this fragment's (a later draw in front of it did not write
    // attachment 7). 0xFFFFFFFF (NaN): no vector.
    const Id depth = Sirit::ValidId(ctx.frag_depth)
                         ? ctx.OpLoad(f32_type, ctx.frag_depth)
                         : ctx.OpCompositeExtract(f32_type,
                                                  ctx.OpLoad(ctx.F32[4], ctx.motion_frag_coord), 2u);
    const Id zero = ctx.Constant(f32_type, 0.0f);
    const Id vector = ctx.OpCompositeConstruct(ctx.F32[2], ctx.OpSelect(f32_type, written, dx, zero),
                                               ctx.OpSelect(f32_type, written, dy, zero));
    ctx.OpStore(ctx.motion_frag_out,
                ctx.OpCompositeConstruct(
                    ctx.U32[2], ctx.OpPackHalf2x16(u32_type, vector),
                    ctx.OpSelect(u32_type, written, ctx.OpBitcast(u32_type, depth),
                                 ctx.ConstU32(0xFFFFFFFFu))));
}

} // namespace Shader::Backend::SPIRV
