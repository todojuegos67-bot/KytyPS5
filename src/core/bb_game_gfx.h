// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-game-gfx: Bloodborne's own post-effect quality settings (YEBIS, Silicon Studio's post-effect
// library linked into the 1.09 eboot), read-only unless a flag asks for a patch.
//
// Decompiled (eboot-patch-v2.elf, vaddr = eboot offset):
//  - CPostEffect::SetAmbientOcclusionParameters at eboot+0xbd2f80 (assert strings
//    "SetAmbientOcclusionParameters", "fOcclusionSampleRadius > 0.0f"): stores the sample count
//    (esi) at this+0xb64, radius / scale / contrast at +0xb68.. ; the enable flag is this+0xb60
//    (setter eboot+0xbd2f10). The C wrappers (eboot+0xbb39b0 / +0xbb39d0) pass the global
//    CPostEffect* at eboot+0x5465ed0. The library default (init, eboot+0xbd0f0e) is 16 samples.
//  - The SSAO pass (eboot+0xbe6fa0, calls DrawRectGPU_AlchemySsao eboot+0xc3d9a0 ->
//    eboot+0xc3d210) clamps that count to [9, 64] at eboot+0xbe7255 and picks the technique
//    tech_AlchemySsao_T1_S9 .. _S64 (an N x N kernel, N = sqrt(count), rounded up to one that
//    exists).
// BB_GPU_PROFILE=1 logs what the game set ("[game-gfx]" every 10 s).
// BB_SSAO_MAX_SAMPLES=9..63 (default unset: the game's code untouched) lowers the clamp's upper
// bound (byte-checked: the 30 bytes at eboot+0xbe7255 must match 1.09). A count the game sets
// at or below it is unchanged; above it, the SSAO kernel is smaller than on PS4 (a lower preset
// the user chooses explicitly).

#pragma once

#include <string>
#include "common/types.h"

namespace BB::GameGfx {

void OnEbootPatched(uintptr_t eboot_base, u64 image_size);
/// One line of the game's current post-effect settings (empty before the eboot is loaded).
std::string Describe();

} // namespace BB::GameGfx
