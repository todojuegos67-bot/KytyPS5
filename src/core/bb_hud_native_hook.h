// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-hud-native (BB_HUD_NATIVE=1): brackets the game's Scaleform UI rendering in the GPU command
// stream. See bb_hud_native_hook.cpp for the eboot evidence.

#pragma once

#include <atomic>
#include <utility>
#include <vector>
#include "common/types.h"

namespace BB::HudNativeHook {

// PM4 NOP payloads written into the UI's draw command buffer (dword 1 of a 3-dword NOP).
constexpr u32 MarkerBegin = 0x6875B0B0u; // after HAL::BeginScene succeeded (before any UI draw)
constexpr u32 MarkerEnd = 0x6875B0B1u;   // after HAL::EndScene (its batches flushed)

// Guest-side counters (the GPU thread logs and resets nothing here; it reads deltas).
struct Stats {
    static std::atomic<u64> begins, begin_failed, ends, written, no_space, no_dcb;
};

// BB_HUD_NATIVE=1 (read once).
bool Enabled();
// The vtable wrappers are in place (after relocation).
bool Installed();

// After the eboot's relocations were applied (its vtables hold absolute addresses).
void OnEbootRelocated(uintptr_t eboot_base, const std::vector<std::pair<u64, u64>>& segments);

// bb-hud-fullres: BB_HUD_FULLRES=1 (default: the value of BB_HUD_NATIVE). The resolution patch
// gives the game's Scaleform movie viewport (GFx::Viewport built at eboot 0x1f58554 and passed to
// Movie::SetViewport, vtable +0x68, in the movie wrapper constructor 0x1f58360) the output size
// instead of the render size: Scaleform projects, tessellates and rasterises its glyphs for the
// output's pixels. Its draws into the render-size scene target are then in output pixels; the
// rasterizer draws them 1:1 into the output-size UI image (HUD cut) or scales them down to the
// render-size target when no cut runs (the HUD as before).
bool FullResRequested();
// Set by the resolution patch once it wrote `ui` into the movie viewport (`render`: the scene).
void SetFullResUi(u32 ui_w, u32 ui_h, u32 render_w, u32 render_h);
// The movie viewport (output size) and the render size; false when the patch did not apply.
bool FullResUi(u32& ui_w, u32& ui_h, u32& render_w, u32& render_h);
struct FullResStats {
    // UI draws drawn 1:1 at output size (HUD cut), drawn scaled down into the render-size
    // target (no cut: fallback), scissors rebuilt, UI passes left unredirected.
    static std::atomic<u64> output_draws, fallback_draws, scissors, cropped_frames;
};

} // namespace BB::HudNativeHook
