// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB object motion vectors (0.25). The motion vectors given to frame generation (FSR 3 FG, DLSS-G)
// and to the temporal upscalers were camera-only: reconstructed from the scene depth and the
// camera matrices (bb_mv.comp), so anything that moves by itself (the player character, enemies,
// weapons: skinned meshes) was taken as static and smeared in generated frames.
//
// Approach (after deadinside28/bloodborne_pc, gpu/upscale/vk_object_motion.cpp and
// docs/motion_vectors.md, GPL-2.0-or-later): every draw of the G-buffer pass gets a pipeline
// variant whose vertex shader stores its clip position per vertex (this frame) and reads the one
// of the previous frame for the same draw and vertex, and whose fragment shader writes the
// difference to an extra colour attachment (7, R32G32_UINT: half2 vector + valid). Only draws
// with a bone palette among their vertex constant buffers (skinned: 640..16384 bytes; small
// skeletons of weapons / props: 96..639, a multiple of 48, only while the palette changes) are
// tracked; the rest write "no vector" and keep the camera motion. After the camera pass, a compute
// pass (bb_objmv.comp) puts the object vectors over the camera ones in the shared MV image, so
// DLSS / FSR / XeSS and FSR 3 FG / DLSS-G all get them.
//
// BB_OBJMV=0 turns it off (camera-only motion, as up to 0.24).

#pragma once

#include <array>
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

// Pipeline cache (RefreshGraphicsKey): this draw is a G-buffer pass draw that should get the
// motion pipeline variant.
bool WantPipeline(const AmdGpu::Regs& regs);
// Fills colour attachment 7 of a motion pipeline's key.
void FillKey(u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7);
bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7);

// Rasterizer, every draw: creates the buffers and the compose pass once (BB upscaling on).
void EnsureInit(const Vulkan::Instance& instance);
// Rasterizer: the motion image as colour attachment 7 of a motion pipeline's render state.
void Attach(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
            Vulkan::RenderState& state);
// Rasterizer, every draw: the motion parameters in the push constants.
void OnDraw(const Vulkan::Instance& instance, Shader::PushData& push, bool motion_pipeline,
            const Shader::Info& vs, const Shader::Gcn::FetchShaderData& fetch,
            const AmdGpu::Regs& regs, bool indexed, u32 index_offset, u32 vertex_offset,
            u32 instance_offset, u64 vs_hash);
// After the camera motion pass (bb_mv.comp; `mv` in General layout, written): the object vectors
// over the camera ones.
void Compose(vk::CommandBuffer cmd, vk::Image mv_image, vk::ImageView mv_view, vk::Extent2D render,
             bool reset);
// The guest flip (end of a frame).
void OnFlip();

} // namespace BB::ObjMotion
