// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"

// Bloodborne 1.09: the startup Resolution Patch (BB_SCENE_UPSCALE=1, 0.23). The game's own render
// size immediates are rewritten in the loaded eboot to output / upscaler ratio, so the game draws
// its scene at the internal size and the existing scene-below-window upscaler path takes it to the
// window. Replaces the 0.22 host-side shrinking of the scene's render targets (now only with
// BB_SCENE_UPSCALE=shrink).
namespace BB::ResPatch {
// Called once the eboot is loaded and the XML patches applied (core/module.cpp).
void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
// The patch was applied: the scene size (internal) and the output size it was derived from.
bool Active(u32* scene_w = nullptr, u32* scene_h = nullptr, u32* out_w = nullptr,
            u32* out_h = nullptr);
// BB_SCENE_UPSCALE=shrink: the 0.22 target-shrinking path instead. Also true for =native.
bool ShrinkWanted();
// 0.71 BB_SCENE_UPSCALE=native: the upscaler inside the game's render chain. The game renders at
// its output size (no eboot rewrite: the window size, from the launcher's resolution patch), the
// 3D scene's targets are created at the internal size on the host (bb_upscale ScaleImage), and the
// upscaler runs at the first output-size pass that samples the scene colour (before
// post-processing and the HUD), writing an output-size image that pass reads instead.
bool NativeChain();
} // namespace BB::ResPatch
