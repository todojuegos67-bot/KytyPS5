// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex-game: NVIDIA Reflex for Streamline's DLSS-G with the simulation markers and the
// Reflex sleep placed from the GAME's own frame loop, the way a PC game integrates Reflex.
// Opt-in: BB_REFLEX_GAME=1 with Streamline DLSS-G (BB_FG=dlss-sl*, or dlss-* with BB_DLSSG_SL=1);
// ignored with BB_REFLEX_NATIVE (that path owns the frame limiter's usleep). Unset (the default):
// no code patch, every function here returns at once, frames are exactly as before.
//
// Eboot 1.09 (vaddr; Ghidra = vaddr + 0x400000), main thread:
//   0x1E4A6C0 main: init, then `do {} while (FrameStep())` (0x1E4A6F0 call 0x2018D20).
//   0x2018D20 FrameStep: [0x5540500]->vtbl+0x18 (system / window events), the frame limiter
//             0x2034770 (this = [0x55404F8], call at 0x2018E22: it waits out the frame's period),
//             then the FD4 task manager run for the frame (0x2018E6F call 0x20512A0 ->
//             0xF88C70 TaskManager::Execute(all groups): input, simulation, draw lists), then
//             post-frame checks ([0x569FA30]->vtbl+0x28, [0x56A54F8]->vtbl+0x98).
// One detour at 0x2018E6B (16 bytes: lea rsi,[rbp-0x40]; call 0x20512A0; mov rdi,[0x569FA30]),
// after BB_VARIABLE_DT's dt hook (0x2018E3D..0x2018E6A), byte-checked after static patching:
//   before the task run (the limiter's wait is over): slReflexSleep(frame) then PCL
//     eSimulationStart(frame), frame = a fresh index of Streamline's one frame index stream;
//   after the task run: PCL eSimulationEnd(frame).
// Render thread (guest GPU submits / flips, the hooks bb_reflex uses): eRenderSubmitStart at the
// frame's first guest submit (not before its eSimulationEnd), and at its flip at the latest; the
// flip hands the frame index to the GPU thread, where DLSS SR's evaluate and DLSS-G's TagFrame use
// it (TagFrame then places no simulation markers of its own). eRenderSubmitEnd / ePresentStart /
// ePresentEnd stay in the presenter (bb_fg_dlssg_sl.cpp), on the same index.
// Every Streamline call runs on two host threads (shadPS4:ReflexGameSleep / ReflexGameMarks),
// none on a guest thread; the game's main thread waits for the sleep at most 50 ms (3 such
// timeouts: no more sleeps, markers go on). Any frame without a game index (not seen by the
// hooks, out of order, hooks not installed) gets the old markers from TagFrame / BeforePresent.
// Log: one [reflex-game] line at install, one per second (markers, sleeps, flips, fallbacks).
#pragma once

#include <cstdint>

namespace BB::ReflexGame {

// BB_REFLEX_GAME=1, DLSS-G through Streamline asked, no BB_REFLEX_NATIVE (decided once).
bool Requested();
// After the eboot's static patching (module.cpp, beside NativeMvHook): the detour, if Requested.
void OnEbootPatched(uintptr_t eboot_base, uint64_t image_size);
// Hooks installed and Streamline's DLSS-G + PCL ready (and not shut down).
bool Active();

// ---- Render thread (guest submits / flips) ----
void OnGuestSubmit();
void OnGuestFlip();
void OnGuestFlipDropped();

// ---- GPU thread ----
// Before the flip's frame is prepared (VideoOutDriver::SubmitFlipInternal): takes the game frame
// index of this flip (one per guest flip, in order).
void OnGpuFlipBegin();
// The game frame index of the frame being prepared (0: none: the caller's old way).
uint32_t GpuFrameId();
// TagFrame used `frame` (a game index or its own).
void OnTagged(uint32_t frame);

// ---- Present thread ----
// Before the present's eRenderSubmitEnd: the game markers posted so far reach Streamline first
// (at most 2 ms), and `frame` is the latest index presented.
void OnPresentFrame(uint32_t frame);

} // namespace BB::ReflexGame
