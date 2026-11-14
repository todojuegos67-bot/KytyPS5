// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-smooth ("Fluidez adaptativa", BB_SMOOTH=1, default off): the real frame rate is held on a
// rung the game sustains (60 / 48 / 40 / 30, only rungs whose shown rate real x multiplier stays
// inside the display's VRR range), the vblank that paces the game's flips is ticked on absolute
// deadlines (high resolution waitable timer + spin), and FSR 3's generated frames are put on a
// grid with exactly step / multiplier spacing. When the game misses its rung it steps down one
// rung (fast), and steps back up only after a long clean stretch with measured headroom (slow,
// with back-off). Evenness over peak fps.
#pragma once

#include <cstdint>

namespace BB::Smooth {

bool Enabled(); // BB_SMOOTH=1

// The present thread (videoout driver). configured_hz: the vblank rate it would use otherwise;
// flip_rate: the guest's sceVideoOutSetFlipRate (0 = a flip on every vblank).
// Returns the vblank interval to use now in seconds (0 = smooth mode not governing).
double VblankInterval(double configured_hz, int flip_rate);

// Precise wait until the absolute steady-clock time deadline_s (seconds).
void WaitUntil(double deadline_s);
double Now(); // steady clock, seconds

// One vblank tick while governing. took_flip: a flip request was presented at this tick;
// submit_s: when the game submitted it (0 = unknown); present_s: after the present call.
void OnTick(double tick_s, bool took_flip, double submit_s, double present_s);

// Re-phase epoch: changes when the target changes (the FG grid starts again).
uint32_t Epoch();
// The real interval FSR 3's slot grid must use (0 = none: the old held mean).
double GridInterval();
// Planned interval between consecutive shown presents (real + generated), ms.
void NoteShown(double ms);

} // namespace BB::Smooth
