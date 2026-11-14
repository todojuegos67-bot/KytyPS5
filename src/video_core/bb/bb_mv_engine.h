// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-mv-engine: object identity for the velocity pass taken from Bloodborne's own renderer, not
// from buffer sizes / similarity / draw order. BB_MV_ENGINE=1 (default off).
//
// Eboot (1.09 patch image; base 1.00 in brackets), found by byte signature at load time:
//   0x21d1080 [0x21d2110] BonePalette(rdi model, rsi sink, rdx mesh, rcx bones):
//       n = [model+0x1c] (full skeleton, [model+0x200] != 0) or the mesh's count; allocates
//       n*48 bytes from the thread's frame allocator (TLS -> [+0x10] -> vtable+0x58(size, 16)),
//       writes palette[i] = bones[i] x invbind[i] ([model+0x1f0]), then
//       sink->vtable[+0x20](sink, 0, n, palette)          sites 0x21d1299 / 0x21d1484 / 0x21d15d1
//   0x21d1650 [0x21d26e0] the same for a second consumer (DrawMesh calls it when
//       [sink+0x180] != 0), sink->vtable[+0x28](...)      sites 0x21d1869 / 0x21d1a54 / 0x21d1ba1
// Both are called from DrawMesh 0x21d0640 [0x21d16d0] with rcx = [instance] (the instance's
// bone matrices: CPU memory owned by the object, stable while the object lives).
//
// Hooks (byte-checked 14-byte detours, see core/bb_native_mv_hook.cpp): the entry of both
// functions (thread-local: bones, mesh, sink) and the six SetPalette sites (rax = the palette
// the GPU reads). The palette's guest address is then mapped to (bones x mesh): the object.
// Linear frame allocators reuse addresses every frame, so each address keeps the last few
// records with a hash of its first matrix; the GPU side accepts a record only when guest memory
// still holds that matrix (exact compare, no tolerance), else the draw is a counted fallback.
//
// Camera blocks (864 B): keyed by the pass they are drawn in (depth target + viewport of the
// draw) and their order inside that pass only, instead of their order in the whole frame.
//
// Engine-side previous transforms (second commit, same switch):
//   DrawMesh 0x21d0640 rigid ring branch (0x21d0809 / 0x21d0ff9): when the instance has a ring
//   ([instance+8]) and the material is not GPU-skinned, the sink gets
//       vtable+0x30(sink, cur)   cur  = ring slot [ring+0x98]      (0x21d0895 / 0x21d1042)
//       vtable+0x38(sink, prev)  prev = slot [ring+0x98]-1 (wrap +[ring+0x9c])  (0x21d08fb)
//   slot pointers: model [ring+0xb0], [model+0x200] != 0: [[ring+8] + i*8], else
//   [[ [ring] + i*0x28 + 8 ] + bone*8], bone = u16 [mesh+2]. The engine's own previous world
//   transform of the object: GuestDrawMesh mirrors that (byte-checked) and records cur -> prev.
//   SSAO reprojection 0xe669d0 (debug strings "UseReProj" / "DontUpdatePrevMatrix" 0xe6c4aa,
//   flag 0x54b0800): at 0xe67073 [obj+0x1c0] = last frame's view (4x4, from 0x21aa080),
//   [obj+0x200] = last frame's projection (0x21aa290 of the camera [G+0x3f8]), [obj+0x240]
//   valid; xmm4..7 / [rbp-0x220..-0x1f0] this frame's (stored there right after). GuestCamera
//   records both: the engine's own previous camera.
#pragma once

#include "common/types.h"

namespace Core {
class MemoryManager;
}

namespace BB::MvEngine {

bool Enabled(); // BB_MV_ENGINE=1

// Guest threads (detours).
void GuestPaletteBegin(u64 model, u64 sink, u64 mesh, u64 bones, u32 consumer);
void GuestPaletteSet(u64 palette, u32 consumer);
void SetHooks(bool installed, const char* detail);
void GuestRingUpdate(u64 ring); // SkinRingUpdate 0x21c8c50 entry
void GuestDrawMesh(u64 model, u64 sink, u64 mesh, u64 material, u64 instance);
// regs: the cave's register block; rbp the guest frame; xmm the saved xmm0..7 (16 B each).
void GuestCamera(const u64* regs, u64 rbp, const u8* xmm);
void SetEngineHooks(u32 drawmesh, u32 camera, const char* detail);

// GPU thread: the engine's previous world transform for a buffer the rigid ring branch bound as
// the current one (V# base within the slot). `key`: the ring object (identity).
bool LookupRigid(u64 addr, u64& prev_addr, u64& offset, u64& key);
// The engine camera whose current view equals the block's (floats 8..19, the 864 B block):
// writes the engine's previous view in the block's layout (12 floats). 0: no record matches,
// 1: matched, 2: matched but the engine had no previous camera (first frame / cut).
u32 EngineCameraPrev(const float* block, float prev_view[12]);
bool ViewEqual(const float* a, const float* b); // 12 floats, tolerance 1e-4 relative
// Wind / time diagnostic: floats of the main camera block (outside view / projection) that
// changed from the engine-matched previous block.
void BlockDelta(const float* cur, const float* prev);

// GPU thread.
struct ObjId {
    u64 key = 0;    // bones x mesh x consumer (0: none)
    u64 sink = 0;   // the engine's render sink of the draw (pass identity of FLVER draws)
    u64 offset = 0; // V# base minus the palette's start
};
enum class Miss : u8 { None = 0, NoRecord, GpuWritten, Stale, Unmapped, Count };
// The engine object whose palette lives at `addr` (V# base). False: `why` says why.
bool LookupPalette(Core::MemoryManager* memory, bool gpu_modified, u64 addr, ObjId& out,
                   Miss& why);

// The pass of the draws that follow (rasterizer: depth target, viewport).
void SetPass(u64 pass_key);
// A camera block's identity: (pass, order of the block inside the pass). `ordinal` out.
u64 CameraKey(u64 block_key, u32& ordinal);

// Statistics (GPU thread).
enum class Stat : u8 {
    Draws = 0,          // velocity draws seen
    EngineDraws,        // draws with at least one buffer identified by the engine
    PalLookups,         // palette / small-skeleton buffers looked up
    PalFound,           // ... with an engine identity
    PalPrev,            // ... and last frame's copy of the same object
    PalNew,             // identity, no copy last frame (object appeared): no object vector
    CamLookups,
    CamPass,            // camera block by (pass, order in pass 0)
    CamPassPrev,        // ... and last frame's copy found
    CamFallback,        // order in pass > 0 or no copy: old path
    FallbackOld,        // palette resolved by the old (ring / address / mesh / order) path
    RigidLookups,  // Current-role buffers (16..639 B) looked up in the rigid ring records
    RigidFound,    // ... bound by the rigid ring branch: engine previous transform used
    RigidNoSnap,   // ... but no copy of the previous slot (arena full / no buffer)
    CamEngLookups, // opaque camera blocks (once per block and frame)
    CamEngNoCpu,   // no CPU copy of the block (GPU-written): fallback
    CamEngNoMatch, // view not the engine camera's (shadow / other pass): fallback
    CamEngNoPrev,  // engine camera, no previous (first frame / cut)
    CamEngNoBlock, // engine previous view not among last frame's blocks: fallback
    CamEngUsed,    // last frame's block chosen by the engine's previous view
    Count
};
void Count(Stat s, u64 n = 1);
void CountMiss(Miss m);

// Once a frame (ObjMv2::OnFlip): per-frame maps, the per-second log.
void OnFrame();

} // namespace BB::MvEngine
