// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB object motion vectors v2 (video_core/bb/bb_objmv2.h). A scene vertex shader is emitted twice:
// once as the guest wrote it, and once more as a separate function ("previous pass", called at the
// start of main) whose buffer loads read last frame's snapshot of the same buffers (bone palette,
// object transform, camera) through buffer device addresses, and whose only effect is last
// frame's clip position (a Private variable). Main then outputs both clip positions (locations
// 30 / 31) and the fragment shader writes their screen-space difference to colour attachment 7.
//
// Snapshot table entry (one per draw, address in the push constants, PushData::ObjMvEntryIndex),
// 16 bytes per buffer handle: u32 address lo, address hi, size in dwords (>= 1), use (0: read the
// current buffer, e.g. not snapshotted). Push flags bit 0: the entry holds last frame's buffers
// (else the previous position is flagged invalid and the camera vector stays).

#include <array>
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {
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
// bb-mv-dots: the draw's screen bounding box, from its vertices (vertex shader; push flags bit
// 5): u32 [min x, min y, max x, max y] in 1/65535 of the viewport, 16 bytes before the draw's
// table entry, read back a few frames later. A blended / effect draw covering a large area (fog
// gate, mist wall, moon) is never marked as spark (bit 4), on every frame and tile alike.
// bb-mv-gate counted bright pixels in the fragment shader (1 in 16 by position): the user's
// capture showed a dot grid of exactly 4 render pixels on blended surfaces (puntos.mp4): no
// fragment side effects any more.
void ScreenBounds(EmitContext& ctx, Id position) {
    const Id b = ctx.U1[1];
    const Id f32 = ctx.F32[1];
    const Id ut = ctx.U32[1];
    const Id flags = PushU32(ctx, PushData::ObjMvFlagsIndex);
    const Id has = ctx.OpINotEqual(b, ctx.OpBitwiseAnd(ut, flags, ctx.ConstU32(32u)),
                                   ctx.u32_zero_value);
    const Id bounds_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(has, bounds_label, merge_label);
    ctx.AddLabel(bounds_label);
    const Id w = ctx.OpCompositeExtract(f32, position, 3u);
    const Id front = ctx.OpFOrdGreaterThan(b, w, ctx.Constant(f32, 1e-6f));
    const Id neg = ctx.Constant(f32, -1.0f);
    const Id pos1 = ctx.Constant(f32, 1.0f);
    const auto ndc = [&](std::uint32_t c) {
        const Id v = ctx.OpFDiv(f32, ctx.OpCompositeExtract(f32, position, c), w);
        return ctx.OpFClamp(f32, v, neg, pos1);
    };
    const auto fixed = [&](Id v) {
        return ctx.OpConvertFToU(
            ut, ctx.OpFMul(f32, ctx.OpFAdd(f32, ctx.OpFMul(f32, v, ctx.Constant(f32, 0.5f)),
                                            ctx.Constant(f32, 0.5f)),
                            ctx.Constant(f32, 65535.0f)));
    };
    // A vertex behind the camera: the draw may cover anything (whole viewport).
    const Id x = fixed(ndc(0u));
    const Id y = fixed(ndc(1u));
    const Id zero = ctx.u32_zero_value;
    const Id full = ctx.ConstU32(65535u);
    const Id lo_x = ctx.OpSelect(ut, front, x, zero);
    const Id lo_y = ctx.OpSelect(ut, front, y, zero);
    const Id hi_x = ctx.OpSelect(ut, front, x, full);
    const Id hi_y = ctx.OpSelect(ut, front, y, full);
    const Id base = ctx.OpISub(ctx.U64, PushAddress(ctx, PushData::ObjMvEntryIndex),
                               ctx.Constant(ctx.U64, u64(16)));
    const Id ptr_type = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ut);
    const Id scope = ctx.ConstU32(static_cast<std::uint32_t>(spv::Scope::Device));
    const auto at = [&](u64 off) {
        return ctx.OpConvertUToPtr(ptr_type, ctx.OpIAdd(ctx.U64, base, ctx.Constant(ctx.U64, off)));
    };
    ctx.OpAtomicUMin(ut, at(0), scope, zero, lo_x);
    ctx.OpAtomicUMin(ut, at(4), scope, zero, lo_y);
    ctx.OpAtomicUMax(ut, at(8), scope, zero, hi_x);
    ctx.OpAtomicUMax(ut, at(12), scope, zero, hi_y);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}
} // namespace

bool EmitContext::ObjMvPrevPass() const {
    if (!ObjMvVertex() || info.uses_dma) {
        return false;
    }
    for (const auto& desc : info.buffers) {
        if (desc.is_written) {
            return false;
        }
    }
    return true;
}

Id EmitContext::ObjMvPrevLoad(u32 handle, Id index, Id current, Id type) {
    if (!objmv_prev_mode) {
        return current;
    }
    const Id u32_type = U32[1];
    const Id u32x4_ptr = TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[4]);
    const Id entry_addr = OpIAdd(U64, PushAddress(*this, PushData::ObjMvEntryIndex),
                                 Constant(U64, u64(handle) * 16));
    const Id entry = OpLoad(U32[4], OpConvertUToPtr(u32x4_ptr, entry_addr),
                            spv::MemoryAccessMask::Aligned, 16u);
    const Id base = OpBitcast(U64, OpCompositeConstruct(U32[2], OpCompositeExtract(u32_type, entry, 0u),
                                                        OpCompositeExtract(u32_type, entry, 1u)));
    const Id dwords = OpCompositeExtract(u32_type, entry, 2u);
    const Id use = OpINotEqual(U1[1], OpCompositeExtract(u32_type, entry, 3u), u32_zero_value);
    const Id clamped = OpUMin(u32_type, index, OpISub(u32_type, dwords, u32_one_value));
    const Id addr = OpIAdd(U64, base, OpIMul(U64, OpUConvert(U64, clamped), Constant(U64, u64(4))));
    const Id ptr = OpConvertUToPtr(TypePointer(spv::StorageClass::PhysicalStorageBuffer, type), addr);
    const Id prev = OpLoad(type, ptr, spv::MemoryAccessMask::Aligned, 4u);
    return OpSelect(type, use, prev, current);
}

// Main, first block: the previous-frame function only when push flags bit 0 is set (a uniform
// branch: static draws skip the second vertex run).
void EmitBBObjMvPrevCall(EmitContext& ctx) {
    ctx.OpStore(ctx.objmv_prev_pos, ctx.ConstantNull(ctx.F32[4]));
    const Id flags = PushU32(ctx, PushData::ObjMvFlagsIndex);
    const Id has = ctx.OpINotEqual(ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.u32_one_value),
                                   ctx.u32_zero_value);
    const Id call_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(has, call_label, merge_label);
    ctx.AddLabel(call_label);
    ctx.OpFunctionCall(ctx.void_id, ctx.objmv_prev_func);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

// Main, at the epilogue (before the host clip-space conversions): both clip positions out.
void EmitBBObjMvVertex(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.objmv_out_cur) || ctx.objmv_prev_mode) {
        return;
    }
    const Id f32 = ctx.F32[1];
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    Id previous = position;
    Id valid = ctx.Constant(f32, 0.0f);
    if (Sirit::ValidId(ctx.objmv_prev_func)) {
        previous = ctx.OpLoad(ctx.F32[4], ctx.objmv_prev_pos);
        const Id flags = PushU32(ctx, PushData::ObjMvFlagsIndex);
        const Id has = ctx.OpINotEqual(ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.u32_one_value),
                                       ctx.u32_zero_value);
        valid = ctx.OpSelect(f32, has, ctx.Constant(f32, 1.0f), valid);
    }
    // z of the previous position carries "valid" (screen motion needs x, y, w only).
    previous = ctx.OpCompositeInsert(ctx.F32[4], valid, previous, 2u);
    ScreenBounds(ctx, position);
    ctx.OpStore(ctx.objmv_out_cur, position);
    ctx.OpStore(ctx.objmv_out_prev, previous);
}

// Fragment epilogue: (previous - current) in render pixels, class, weight.
void EmitBBObjMvFragment(EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.objmv_frag_out)) {
        return;
    }
    const Id f32 = ctx.F32[1];
    const Id b = ctx.U1[1];
    const Id current = ctx.OpLoad(ctx.F32[4], ctx.objmv_in_cur);
    const Id previous = ctx.OpLoad(ctx.F32[4], ctx.objmv_in_prev);
    const Id xscale = PushF32(ctx, PushData::XScaleIndex);
    const Id yscale = PushF32(ctx, PushData::YScaleIndex);
    const auto ndc = [&](Id clip, u32 c) {
        return ctx.OpFDiv(f32, ctx.OpCompositeExtract(f32, clip, c),
                          ctx.OpCompositeExtract(f32, clip, 3u));
    };
    const Id dx = ctx.OpFMul(f32, ctx.OpFSub(f32, ndc(previous, 0u), ndc(current, 0u)), xscale);
    const Id dy = ctx.OpFMul(f32, ctx.OpFSub(f32, ndc(previous, 1u), ndc(current, 1u)), yscale);
    const Id eps = ctx.Constant(f32, 1e-5f);
    const Id front =
        ctx.OpLogicalAnd(b, ctx.OpFOrdGreaterThan(b, ctx.OpCompositeExtract(f32, previous, 3u), eps),
                         ctx.OpFOrdGreaterThan(b, ctx.OpCompositeExtract(f32, current, 3u), eps));
    const Id bad = ctx.OpLogicalOr(
        b, ctx.OpLogicalOr(b, ctx.OpIsNan(b, dx), ctx.OpIsInf(b, dx)),
        ctx.OpLogicalOr(b, ctx.OpIsNan(b, dy), ctx.OpIsInf(b, dy)));
    // Larger than 256 render pixels in a frame: a wrong match (camera cut, other instance).
    const Id limit = ctx.Constant(f32, 256.0f);
    const Id sane = ctx.OpLogicalAnd(b, ctx.OpFOrdLessThan(b, ctx.OpFAbs(f32, dx), limit),
                                     ctx.OpFOrdLessThan(b, ctx.OpFAbs(f32, dy), limit));
    const Id written = ctx.OpLogicalAnd(
        b, ctx.OpLogicalAnd(b, ctx.OpLogicalAnd(b, front, ctx.OpLogicalNot(b, bad)), sane),
        ctx.OpFOrdGreaterThan(b, ctx.OpCompositeExtract(f32, previous, 2u), ctx.Constant(f32, 0.5f)));
    // Attachment 7 is RGBA16F: (dx, dy, class, weight). Opaque draws: written over whatever was
    // behind (class 0 where no vector: the camera vector stays), weight 0. Blended draws
    // (particles, effects, fog; attachment 7 blends with src alpha): weight from their MRT 0
    // alpha, so the vector behind is weighted with theirs; no vector -> weight 0 (nothing
    // changes). With bb-fog-mv the class channel is masked for them and .w accumulates.
    const Id zero = ctx.Constant(f32, 0.0f);
    const Id one = ctx.Constant(f32, 1.0f);
    // Class 1: moving opaque draw; 2: static draw run only for the check (flags bit 1, never
    // replaces the camera vector); 3: blended.
    const Id flags = PushU32(ctx, PushData::ObjMvFlagsIndex);
    const Id diag = ctx.OpINotEqual(b, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.ConstU32(2u)),
                                    ctx.u32_zero_value);
    Id cls = ctx.OpSelect(f32, diag, ctx.Constant(f32, 2.0f), one);
    // Class 4 (flags bit 2, BB_NATIVE_MV2): a skinned draw (characters), the player mask's source.
    const Id character = ctx.OpINotEqual(b, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.ConstU32(4u)),
                                         ctx.u32_zero_value);
    cls = ctx.OpSelect(f32, character, ctx.Constant(f32, 4.0f), cls);
    // bb-fog-mv: opaque draws write weight 0: .w of attachment 7 is the coverage of the blended
    // draws in front (fog, smoke, particles), accumulated by blending. The push word is 0 with
    // bb-fog-mv off (BB_FOG_MV=0 / no velocity path): weight 1 as before.
    const Id curve = PushU32(ctx, PushData::ObjMvFogIndex);
    Id weight = ctx.OpSelect(f32, ctx.OpIEqual(b, curve, ctx.u32_zero_value), one, zero);
    if (ctx.runtime_info.hw.fs.bb_objmv_blend) {
        cls = ctx.Constant(f32, 3.0f);
        const auto& mrt0 = ctx.frag_outputs[0];
        if (Sirit::ValidId(mrt0.id) && mrt0.num_components == 4 && !mrt0.is_integer) {
            const Id alpha = ctx.OpFClamp(
                f32, ctx.OpCompositeExtract(f32, ctx.OpLoad(ctx.F32[4], mrt0.id), 3u), zero, one);
            // bb-fog-mv (flags bit 3): density weighting, weight = alpha * smoothstep(lo, hi,
            // alpha) (push word lo | hi << 16, 1/65535): thin wisps leave the vector behind
            // (characters keep theirs), dense layers weigh by their opacity. Attachment 7 then
            // blends (dx, dy) with this weight and accumulates it in .w (the class is kept).
            const Id fog = ctx.OpINotEqual(
                b, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.ConstU32(8u)), ctx.u32_zero_value);
            const Id inv = ctx.Constant(f32, 1.0f / 65535.0f);
            const Id lo_bits = ctx.OpBitwiseAnd(ctx.U32[1], curve, ctx.ConstU32(0xffffu));
            const Id hi_bits = ctx.OpShiftRightLogical(ctx.U32[1], curve, ctx.ConstU32(16u));
            const Id lo = ctx.OpFMul(f32, ctx.OpConvertUToF(f32, lo_bits), inv);
            const Id hi = ctx.OpFMul(f32, ctx.OpConvertUToF(f32, hi_bits), inv);
            const Id span = ctx.OpFMax(f32, ctx.OpFSub(f32, hi, lo), ctx.Constant(f32, 1e-3f));
            const Id t = ctx.OpFClamp(f32, ctx.OpFDiv(f32, ctx.OpFSub(f32, alpha, lo), span),
                                      zero, one);
            const Id two_t = ctx.OpFMul(f32, ctx.Constant(f32, 2.0f), t);
            const Id smooth = ctx.OpFMul(f32, ctx.OpFMul(f32, t, t),
                                         ctx.OpFSub(f32, ctx.Constant(f32, 3.0f), two_t));
            const Id dense = ctx.OpFMul(f32, alpha, smooth);
            weight = ctx.OpSelect(f32, written, ctx.OpSelect(f32, fog, dense, alpha), zero);
            // bb-mv-candles (BB_FG_SPARKS=1: flags bits 16-31 = threshold in 1/4096, 0 = off):
            // the class channel blends with the same weight; the fragment writes 0 there (so B is
            // class * (1 - .w), decoded by compose) or, where the colour it adds (MRT 0 rgb x
            // alpha) is bright (candle flames, glowing effects drawn alpha-blended), 5 with
            // weight 1: its own static-world vector replaces the one behind.
            const Id sbits = ctx.OpShiftRightLogical(ctx.U32[1], flags, ctx.ConstU32(16u));
            const Id spark_on = ctx.OpINotEqual(b, sbits, ctx.u32_zero_value);
            const Id col = ctx.OpLoad(ctx.F32[4], mrt0.id);
            const Id lum = ctx.OpFAdd(
                f32,
                ctx.OpFAdd(f32,
                           ctx.OpFMul(f32, ctx.OpCompositeExtract(f32, col, 0u),
                                      ctx.Constant(f32, 0.2126f)),
                           ctx.OpFMul(f32, ctx.OpCompositeExtract(f32, col, 1u),
                                      ctx.Constant(f32, 0.7152f))),
                ctx.OpFMul(f32, ctx.OpCompositeExtract(f32, col, 2u), ctx.Constant(f32, 0.0722f)));
            const Id added = ctx.OpFMul(f32, lum, alpha);
            const Id sthr = ctx.OpFMul(f32, ctx.OpConvertUToF(f32, sbits),
                                       ctx.Constant(f32, 1.0f / 4096.0f));
            const Id measured = ctx.OpLogicalAnd(
                b, ctx.OpLogicalAnd(b, spark_on, written),
                ctx.OpLogicalAnd(b, ctx.OpFOrdGreaterThan(b, added, sthr),
                                 ctx.OpLogicalNot(b, ctx.OpIsInf(b, added))));
            // bb-mv-gate: flags bit 4: a large draw (measured): never marked, on any pixel.
            const Id large = ctx.OpINotEqual(
                b, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.ConstU32(16u)), ctx.u32_zero_value);
            const Id bright = ctx.OpLogicalAnd(b, measured, ctx.OpLogicalNot(b, large));
            weight = ctx.OpSelect(f32, bright, one, weight);
            cls = ctx.OpSelect(f32, spark_on,
                               ctx.OpSelect(f32, bright, ctx.Constant(f32, 5.0f), zero), cls);
        } else {
            weight = zero;
        }
    }
    if (ctx.runtime_info.hw.fs.bb_objmv_add) {
        // bb-mv-fill: an additive / premultiplied effect (sparks, embers, glows). Attachment 7
        // blends with this weight as src alpha (RGB only), so weight 1 replaces (dx, dy) and the
        // class with the effect's own static-world vector and class 5, weight 0 keeps what was
        // behind. Weight 1 only where the colour it adds is bright: luminance of MRT 0 (x its
        // alpha when the game's blend uses src alpha: push word bit 16) > threshold (bits 0-15,
        // 1/4096). The push word is objmv_fog (unused by these draws).
        cls = ctx.Constant(f32, 5.0f);
        weight = zero;
        const auto& mrt0 = ctx.frag_outputs[0];
        if (Sirit::ValidId(mrt0.id) && mrt0.num_components >= 3 && !mrt0.is_integer) {
            const Id col = ctx.OpLoad(ctx.F32[mrt0.num_components], mrt0.id);
            const Id r = ctx.OpCompositeExtract(f32, col, 0u);
            const Id g = ctx.OpCompositeExtract(f32, col, 1u);
            const Id bl = ctx.OpCompositeExtract(f32, col, 2u);
            Id lum = ctx.OpFAdd(f32,
                                ctx.OpFAdd(f32, ctx.OpFMul(f32, r, ctx.Constant(f32, 0.2126f)),
                                           ctx.OpFMul(f32, g, ctx.Constant(f32, 0.7152f))),
                                ctx.OpFMul(f32, bl, ctx.Constant(f32, 0.0722f)));
            const Id word = PushU32(ctx, PushData::ObjMvFogIndex);
            if (mrt0.num_components == 4) {
                const Id a = ctx.OpFClamp(f32, ctx.OpCompositeExtract(f32, col, 3u), zero, one);
                const Id mul = ctx.OpINotEqual(
                    b, ctx.OpBitwiseAnd(ctx.U32[1], word, ctx.ConstU32(0x10000u)),
                    ctx.u32_zero_value);
                lum = ctx.OpSelect(f32, mul, ctx.OpFMul(f32, lum, a), lum);
            }
            const Id thr = ctx.OpFMul(
                f32,
                ctx.OpConvertUToF(f32, ctx.OpBitwiseAnd(ctx.U32[1], word, ctx.ConstU32(0xffffu))),
                ctx.Constant(f32, 1.0f / 4096.0f));
            // NaN / Inf colours never count (OpFOrdGreaterThan is false for NaN).
            const Id measured = ctx.OpLogicalAnd(
                b, written, ctx.OpLogicalAnd(b, ctx.OpFOrdGreaterThan(b, lum, thr),
                                             ctx.OpLogicalNot(b, ctx.OpIsInf(b, lum))));
            const Id large = ctx.OpINotEqual(
                b, ctx.OpBitwiseAnd(ctx.U32[1], flags, ctx.ConstU32(16u)), ctx.u32_zero_value);
            weight = ctx.OpSelect(f32, ctx.OpLogicalAnd(b, measured, ctx.OpLogicalNot(b, large)),
                                  one, zero);
        }
    }
    ctx.OpStore(ctx.objmv_frag_out,
                ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>{
                                             ctx.OpSelect(f32, written, dx, zero),
                                             ctx.OpSelect(f32, written, dy, zero),
                                             ctx.OpSelect(f32, written, cls, zero), weight}));
}

} // namespace Shader::Backend::SPIRV
