// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-clean-021-objmv: object motion vectors. The motion vectors given to the upscalers (DLSS /
// FSR / XeSS / TAA) and to frame generation are made from the scene depth and the camera
// (bb_mv.comp), so anything that moves by itself (the player, enemies, the cape, branches,
// doors) was taken as static and smeared or doubled.
//
// Ported from bb-combo-159 (bb_objmv.cpp 0.25..1.58; approach and bone palette classification
// after deadinside28/bloodborne_pc, gpu/upscale/vk_object_motion.cpp, GPL-2.0-or-later), reduced
// to the draws that really moved:
//  - Scene draws (G-buffer: scene depth bound, 3+ colour targets) get colour attachment 7
//    (R32G32_UINT) in their pipeline key, write-masked: same shaders, no extra pixel work.
//  - Per draw, a cheap check on the CPU: a bone palette (skinned) or the draw's own small
//    constant buffers (a rigid object's transform) compared with the previous frame's (hash).
//    Only a draw whose constants changed takes the motion variant: its vertex shader stores the
//    clip position per vertex (device-address array, allocated once) and loads the one of the
//    previous frame; its fragment shader writes the vector and its depth to attachment 7.
//  - At the present (bb_present_sr.cpp, after bb_mv.comp) one compute pass puts the object
//    vectors over the camera ones where the scene depth is still that fragment's.
// Anything without a previous position keeps the camera vector.
//
// No readbacks or fence waits (the GPU timestamps are read without waiting, 4 frames later), no
// per-frame allocations. BB_OBJMV=0: off. BB_OBJMV_RIGID=0: skinned / small skeletons only.
// BB_OBJMV_VERTICES_MI (1..16, default 4): positions per frame, in Mi.
// Log, once per second: "[objmv] draws with own vector N of M, cpu X ms, gpu Y ms" (per frame).

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
namespace Gcn {
struct FetchShaderData;
}
} // namespace Shader
namespace Vulkan {
class Instance;
class Scheduler;
struct RenderState;
} // namespace Vulkan

namespace BB::ObjMotion {

constexpr u32 Attachment = 7;

// Pipeline cache, every draw: a scene (G-buffer) draw that gets attachment 7 (register checks
// only; creates the buffers once).
bool WantPipeline(const Vulkan::Instance& instance, const AmdGpu::Regs& regs);
// Pipeline cache, scene draws only (after the stages): this draw moved since the previous frame
// and takes the motion shader variant. Remembered for OnDraw.
bool Decide(const AmdGpu::Regs& regs, const Shader::Info& vs,
            const Shader::Gcn::FetchShaderData* fetch, u64 vs_hash, bool indirect);
// Attachment 7 in the key: write-masked (`tracked` false) or written by the motion shaders.
void FillKey(bool tracked, u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7);
bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7);

// Rasterizer: the motion image as attachment 7 (cleared at the frame's first scene draw).
void Attach(Vulkan::Scheduler& scheduler, Vulkan::RenderState& state);
// Rasterizer, motion-variant draws: slots for this frame's positions and last frame's.
void OnDraw(Shader::PushData& push, const AmdGpu::Regs& regs, bool indexed, u32 index_offset,
            u32 vertex_offset, u32 instance_offset);
// Rasterizer: the scene depth target (its guest address) of this frame.
void NoteSceneDepth(u64 depth_address);

// The present (bb_present_sr.cpp), after the camera pass: the object vectors over `mv`
// (General layout), with the scene depth (sampled, as bb_mv.comp reads it).
void Compose(vk::CommandBuffer cmd, vk::ImageView mv_view, vk::ImageView depth_view,
             vk::Sampler sampler, vk::Extent2D render);
// The flip (end of a frame; BB::Upscale::EndFrame).
void OnFlip();

} // namespace BB::ObjMotion
