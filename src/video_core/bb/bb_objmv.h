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
// 0.71: rigid objects without a bone palette (doors, lifts, carriages, thrown props) are tracked
// the same way when their own vertex constants (16..1024 bytes, not the scene block: their object
// transform) changed since the previous frame; static objects keep the exact camera motion. When
// most draws change every frame (the camera is inside the per-draw constants) rigid tracking
// pauses, since the camera motion is exact for those. BB_OBJMV_RIGID=0 turns this part off.
//
// Where the vectors come from (nothing is estimated from the final image):
//  - camera: current and previous view-projection from the game's 864-byte scene constant block
//    (bb_upscale.cpp BlockCamera, read from the G-buffer draws' vertex constants), turned into
//    clip_to_prev_clip (FillCamera) and applied per pixel to the G-buffer depth (bb_mv.comp).
//  - objects: the vertex shader's own clip position this frame and last frame for the same draw
//    and vertex (emit_spirv_bb_objmv.cpp), written per pixel by the G-buffer pass itself
//    (attachment 7) and put over the camera vectors (bb_objmv.comp, Compose).
//
// BB_OBJMV=0 turns it off (camera-only motion, as up to 0.24).
//
// 1.20: BB_OBJMV_DEBUG=1 logs per second the character (large skeleton) pixels' object vs camera
// vector magnitudes and how its draws found last frame's positions; BB_MV_VIEW=1 shows the motion
// vectors as colour in place of the upscaled image; BB_OBJMV_LOOSE=0 turns off the identity
// without buffer addresses.

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
// over the camera ones. `late`: the upscale runs after the guest flip (the presenter's late path).
void Compose(vk::CommandBuffer cmd, vk::Image mv_image, vk::ImageView mv_view, vk::Extent2D render,
             bool reset, bool late);
// 1.20 BB_MV_VIEW=1: the motion vectors last composed, as colour (hue: direction, brightness:
// magnitude, 8 px full; blue tint: characters, magenta: other objects), blitted over `output`
// (left in `output_layout`). Nothing without BB_MV_VIEW.
void DebugView(vk::CommandBuffer cmd, vk::Image output, vk::ImageLayout output_layout,
               vk::Extent2D output_size);
// The guest flip (end of a frame).
void OnFlip();
// 1.03: draws with object vectors (last frame's positions) in the frame last composed into the
// motion vectors (0: none composed, camera-only motion).
u32 ComposedDraws();

// 1.40 (bb-fgamd-140): what the compose pass put into the motion image since the last call (the
// [fg-amd] line): pixels with an object vector / of characters (large skeletons) / all, composes,
// frames read back, frames whose upscaler history was reset, frames left camera-only by the
// old reset rule (BB_OBJMV_RESET_SKIP=1), composes without a G-buffer motion pass this frame.
struct CoverageStats {
    u64 object_px = 0, character_px = 0, all_px = 0;
    u32 composes = 0, frames_read = 0, reset_frames = 0, reset_skipped = 0, not_written = 0;
    bool enabled = false, reset_skip_rule = false;
    // 1.41 (bb-fg30-141): character pixels whose object vector is near zero (< 0.25 px) and
    // whose object vector differs from the camera one by more than 1 px (own motion seen).
    u64 character_still_px = 0, character_own_px = 0;
    // 1.46 (bb-fgmv-146): pixels drawn by a motion pipeline that carry no vector, by reason: no
    // previous position for the draw, behind the camera, not finite. All zero with object
    // coverage zero = the motion attachment is never written (pipeline / attachment problem).
    u64 no_prev_px = 0, behind_px = 0, nonfinite_px = 0;
    // 1.50: sums of vector magnitude x16 on character pixels / on pixels without an object vector.
    u64 character_mag16 = 0, background_mag16 = 0, background_px = 0;
};
CoverageStats TakeCoverage();
// 1.41: guest frame numbers for the [fg-amd] pairing check. FrameNo(): flips seen so far (OnFlip
// advances it; the frame being built is FrameNo()). ComposedFrame(): the guest frame whose
// G-buffer object vectors the last compose put into the motion image (~0: none).
u64 FrameNo();
u64 ComposedFrame();

} // namespace BB::ObjMotion
