// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include "common/types.h"

// Bloodborne 1.09, BB_VARIABLE_DT=1 (default off): PC-style variable frame time, installed by the
// emulator at eboot load with byte checks against the original 1.09 code (no XML patch needed;
// the "PC sin limite" entries 1/2x/3/4/5/6 are recognised and kept, any other FPS patch turns
// this off). Game dt = measured frame time clamped to [frame period, BB_VARIABLE_DT_MAX_MS
// (default 50 ms)]; frame cap = BB_GAME_FPS_CAP (0 / unset = unlimited) in the game's own
// limiter (BB_FPS_CAP keeps its vblank / pacer meaning and is not read here); per-frame
// time-step reads use the live dt, frame-count -> seconds conversions stay 1/30;
// Havok semi-fixed substeps; the presenter flips each frame as soon as it arrives
// (BB_VRR_PRESENT, on by default with BB_VARIABLE_DT=1).
namespace BB::VarDt {
bool Enabled();          // BB_VARIABLE_DT=1
bool ImmediatePresent(); // Enabled() and BB_VRR_PRESENT != 0
void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
// Guest thread, once per game frame (sceVideoOutSubmitFlip / sceGnmSubmitAndFlip*).
// flip_mode: the guest's flip mode (1 = vsync, 2 = hsync / immediate), -1 if unknown.
void OnGuestFlip(int flip_mode);
// Present thread, after a guest frame was presented.
void OnPresent();
// bb-reflex: the period of the frame cap BB_GAME_FPS_CAP gave the game's limiter (>= 10 fps,
// limiter installed), 0 when there is none (unlimited, or BB_VARIABLE_DT off).
float CapPeriod();
// bb-reflex: NVIDIA Reflex paces the frames with that cap (its sleep's minimum interval), so the
// limiter's period becomes 1 ms for this frame; false gives the cap back to the limiter. Only from
// the limiter's own gettimeofday (0x2034977) and every frame while true: it writes the limiter
// object's period (this+0x18), which the limiter rewrites from its code each frame.
void SetLimiterExternal(bool external);
} // namespace BB::VarDt
