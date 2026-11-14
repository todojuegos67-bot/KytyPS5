// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include "common/types.h"

// Bloodborne 1.09, BB_VARIABLE_DT=1 (default off): PC-style variable frame time on top of the
// "PC sin limite" patch set (launcher FPS option 7). The game's frame dt is the measured frame
// time clamped to [game limiter period, BB_VARIABLE_DT_MAX_MS (default 50 ms)] instead of
// [period, 1/60]; the per-frame constants that "PC sin limite 1" froze at a 100 fps / 60 fps
// value are re-evaluated every frame from the live dt; the presenter flips each frame as soon
// as it arrives (BB_VRR_PRESENT, on by default with BB_VARIABLE_DT=1) instead of on the next
// emulated vblank tick.
namespace BB::VarDt {
bool Enabled();          // BB_VARIABLE_DT=1
bool ImmediatePresent(); // Enabled() and BB_VRR_PRESENT != 0
void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
// Guest thread, once per game frame (sceVideoOutSubmitFlip / sceGnmSubmitAndFlip*).
// flip_mode: the guest's flip mode (1 = vsync, 2 = hsync / immediate), -1 if unknown.
void OnGuestFlip(int flip_mode);
// Present thread, after a guest frame was presented.
void OnPresent();
} // namespace BB::VarDt
