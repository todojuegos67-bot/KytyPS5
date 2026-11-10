// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB object motion vectors v2 (bb-objmv-gpu). Bloodborne has no motion vectors; bb_mv.comp
// rebuilds camera-only vectors from depth, so the player, enemies, weapons and animated props
// smear under DLSS / FSR / XeSS and frame generation.
//
// Per-draw vectors made on the GPU, as an engine with motion vectors does:
//  - Scene draws (scene-sized depth with depth test, before the HUD cut; opaque draws with depth
//    write, and alpha-blended draws: particles / effects) take a pipeline variant whose vertex
//    shader runs the guest program twice (emit_spirv_bb_objmv2.cpp): as written, and once more
//    with every buffer load reading last frame's copy of the same buffer (bone palette, object
//    transform, camera block, user data), through buffer device addresses. Both clip positions
//    go to the fragment shader, which writes (previous - current) in render pixels into colour
//    attachment 7 (RGBA16F: dx, dy, class, weight; blended draws alpha-weighted).
//  - Last frame's buffers: when a motion draw is recorded its read-only vertex-shader buffers
//    (<= 64 KiB) are copied from guest memory into a host-visible snapshot arena (one copy per
//    buffer address and frame) and remembered under the draw's identity (vertex program, index
//    buffer, counts, occurrence in the frame): one O(1) hash lookup a draw, no GPU readback.
//    The next frame's same draw gets a table entry with those addresses.
//  - History = exactly the previous frame with motion draws; reset on a frame without them, a
//    resolution change and a camera cut (the upscaler's reset).
//  - After bb_mv.comp, a compute pass (bb_objmv2.comp) puts the object vectors over the camera
//    vectors in the shared motion image: DLSS, FSR 3.1, FSR 4, XeSS, FSR 3 FG and the DLSS-G
//    bridge all read that image with their existing scaling.
// BB_OBJMV2=0 turns it off (camera-only vectors as build 344). BB_OBJMV2_CHECK=0 turns off the
// reprojection check. Log once a second: "[objmv2] ...".

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace AmdGpu {
union Regs;
}
namespace Shader {
struct PushData;
struct Info;
struct PsColorBuffer;
} // namespace Shader
namespace Core {
class MemoryManager;
}
namespace Vulkan {
class Instance;
class Scheduler;
struct RenderState;
} // namespace Vulkan

namespace BB::ObjMv2 {

constexpr u32 Attachment = 7;

// Mask: another draw of the scene pass; attachment 7 bound with write mask 0, stock shaders, so
// switching between motion and other pipelines never restarts the render pass.
enum class Kind : u8 { None = 0, Opaque = 1, Blend = 2, Mask = 3 };

// Rasterizer: once, with the instance (creates buffers and the compose pass; false: off).
bool Init(const Vulkan::Instance& instance);
bool Active();
// Rasterizer: the scene depth target (scene sized, depth written, before the HUD).
void NoteSceneDepth(u64 depth_address);
// Pipeline cache: which motion variant this draw takes (None: the stock pipeline).
Kind Want(const AmdGpu::Regs& regs);
// Pipeline cache: attachment 7 of a motion pipeline's key.
void FillKey(Kind kind, u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7, u32& cb_shader_mask_raw, void* blend7);
bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7);
// Rasterizer, BeginRendering of a motion pipeline: attachment 7.
void Attach(Vulkan::Scheduler& scheduler, Vulkan::RenderState& state);
// Rasterizer, every motion draw (after BindResources): snapshot + table entry, push constants.
// `identity`: index buffer address or indirect argument address.
void OnDraw(Shader::PushData& push, const Shader::Info& vs, Core::MemoryManager* memory,
            u64 vs_hash, u64 identity, u32 count, u32 instances, bool blend, bool ui_phase);
// Presenter (bb_present_sr.cpp), after bb_mv.comp: the object vectors over the camera ones
// (`mv` RG16F, General); `color` the presented colour (RGBA16F, General) for the check.
void Compose(vk::CommandBuffer cmd, vk::ImageView mv_view, vk::ImageView color_view,
             vk::Extent2D render, bool reset);
// Presenter: the guest flip (end of a frame).
void OnFlip();

} // namespace BB::ObjMv2
