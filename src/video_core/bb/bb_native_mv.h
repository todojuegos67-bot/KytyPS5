// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB native motion vectors (bb-native-mv): object motion keyed by Bloodborne's own objects.
//
// The game draws every FLVER mesh instance through one engine function (DrawMesh, eboot
// 0x21d16d0 in the base 1.00 image, found by byte signature at load time): rdi = the FLVER model
// resource, rdx = the mesh, r8 = the instance record whose first field points to the instance's
// bone matrices (CPU memory, 3x4 row-major, 48 bytes a bone, also the single transform of a rigid
// mesh). Its bone palette is bones[i] x inverse_bind[i] (eboot 0x21d2110) and its draw goes
// through two GX wrappers (0x216ab70 / 0x216acf0) that write the PM4 draw packet into the Gnmx
// draw command buffer (GfxContext at wrapper rdi + 0x10, write pointer at + 0x20).
//
// core/bb_native_mv_hook.cpp detours DrawMesh (identity: bones pointer + mesh) and both wrappers
// (the command buffer write pointer before the draw is emitted, only when called from DrawMesh).
// The command processor then knows, for each draw packet, which game object drew it.
//
// "Moving" is decided from the game's own CPU transforms, not from GPU buffers: each frame the
// bones of every object are kept, and an object moves when its bones are not where last frame's
// bones go under the frame's common transform (identity in world space; the camera change when
// the game's bones are camera-relative), taken as the median over all objects. Static objects
// keep the 149 camera vectors exactly; only moving objects' draws take the objmv2 motion
// pipeline (vertex shader run again with last frame's buffers of the same game object).
//
// BB_NATIVE_MV=1 (default) on, 0: previous behaviour (objmv2 as build a846). Live: the file named
// by BB_NATIVE_MV_FILE (default bb_native_mv.txt in the working folder) holding 0 / 1 is re-read
// every 0.5 s. BB_NATIVE_MV_HOOK=0: no guest patch at all. Log once a second: "[native-mv] ...".

#pragma once

#include "common/types.h"

namespace BB::NativeMv {

// Live switch (on, and the guest hooks are in place).
bool On();
bool HooksInstalled();
void SetHooksInstalled(bool installed, const char* detail);

// Guest threads (from the detours).
void GuestDrawMesh(u64 model, u64 mesh, u64 instance);
void GuestDrawEmit(u64 gx_context, u64 return_address);
void SetDrawMeshRange(u64 begin, u64 end);
// Guest flip (sceGnmSubmitAndFlipCommandBuffers): the guest frame ends.
void OnGuestFlip();

// Command processor (GPU thread): the guest address of the draw packet about to be drawn.
void SetDrawPacket(u64 packet_address);

struct DrawId {
    u64 key = 0;     // game identity (bones pointer x mesh); 0: none
    bool moving = false;
    bool recent = false; // moved in the last frames (snapshot kept for the next frame)
    bool skinned = false;
};
// The current draw's identity (after SetDrawPacket).
const DrawId& Current();

// objmv2 statistics feed (GPU thread).
void NoteSceneDraw(bool motion_candidate);
void NoteVectorDraw(bool with_previous);

// Presenter, once a frame: live switch file, the per-second log.
void OnPresent();

} // namespace BB::NativeMv
