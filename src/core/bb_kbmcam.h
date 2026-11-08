// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"

// (bb-kbmcam-142) Bloodborne 1.09 mouse camera, as the Demon's Souls port's KYTY_KBM_NO_RAMP:
// with keyboard + mouse play on, the player camera's full-tilt turn ramp is skipped and the mouse
// is written straight into the camera's turn (radians per count, no stick curve, no rate cap, no
// smoothing). With kbm play off the game's code path is the original one.
namespace BB::KbmCam {
void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
// The camera hook is installed and has learned the stick signs: the mouse no longer drives the
// emulated stick (the hook turns the camera itself).
bool Direct();
// Every reader of the relative mouse goes through here (the SDL timer and the camera hook), so
// both see every count. Returns this caller's delta since its last call.
void PumpMouse(float* dx, float* dy);
// The stick tilt the mouse timer last sent (-128..127), for the sign check.
void NoteStickOut(int x, int y);
} // namespace BB::KbmCam
