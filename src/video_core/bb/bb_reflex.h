// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex-native: NVIDIA Reflex through VK_NV_low_latency2, the native PC way: the game's own
// frame limiter wait becomes the Reflex sleep, so there is one wait per frame and the game measures
// its frame time (gettimeofday, this+0x264) after it, as on PC. Only with BB_REFLEX_NATIVE=1 and
// DLSS-G (BB_FG=dlss-*), or BB_REFLEX_NATIVE=force (also without frame generation); never with
// FSR 3 FG. Unset (the default): no device extension, no latency swapchain, sceKernelUsleep linked
// as before, nothing else changes. BB_REFLEX_BOOST=1: low latency boost.
//
// Sleep: the frame limiter (eboot 1.09 0x2034770) sleeps the rest of its period with
// sceKernelUsleep at 0x2034BD2. That call (found by its return address 0x2034BD7) hands the wait
// to the host thread shadPS4:ReflexSleep (vkLatencySleepNV with minimumIntervalUs = the limiter's
// period this+0x18) and the game's main thread blocks until it returns (at most period + 50 ms,
// else the game's usleep for that frame; 3 such timeouts turn Reflex off for good). The first call
// of a frame only: if the limiter's loop calls again in the same frame (Reflex ended before the
// game's own deadline), the game's usleep runs as before. No other sleep or pacing is added.
// Frame ids: one per Reflex sleep, then followed through the render thread (GXRenderThread: guest
// submits, guest flip), the GPU thread (the flip's processing) and the present thread (the game
// frame's vkQueuePresentKHR, VkPresentIdKHR = the id). Markers: SIMULATION_START (after the
// sleep), SIMULATION_END (next frame start or the frame's first guest submit), RENDERSUBMIT_START
// / END (first guest submit / guest flip), PRESENT_START / END (around the game frame's present).
// Every VK_NV_low_latency2 call runs on host threads (ReflexSleep, ReflexMarks, present thread).
// Log: one [reflex-native] line per second (sleeps, timeouts, fallback state, the game's dt vs
// the present interval, and the guest sceKernelUsleep calls: all, at the limiter's return address,
// and elsewhere with the last such return address). The first limiter hit and the first unmatched
// calls are logged once each.

// Device and swapchain set-up: bb_reflex_vk.h.
#pragma once

#include <cstdint>

namespace BB::Reflex {

// BB_REFLEX_NATIVE asked for it and the mode allows it (see above); decided once.
bool Requested();

// ---- Game frame ----
void OnEbootLoaded(uintptr_t eboot_base, uint64_t image_size);
// BB_REFLEX_NATIVE is 1 / force (environment only; safe before logging is up).
bool NativeEnv();
// Every guest sceKernelUsleep while BB_REFLEX_NATIVE is set: true when the wait was done by
// Reflex's sleep (the caller skips its own), false: the caller sleeps usec as usual.
bool OnGuestUsleep(const void* return_address, uint32_t usec);
void OnGuestSubmit();      // render thread, guest command buffer submit
void OnGuestFlip();        // render thread, guest flip about to be queued
void OnGuestFlipDropped(); // the flip was refused (queue full): its id is forgotten
uint64_t OnGpuFlip();      // GPU thread, the flip's frame prepared: its id (0: none)

// ---- Present thread ----
// The game frame about to be presented (its id from OnGpuFlip): PRESENT_START; returns the id
// for its VkPresentIdKHR (0: none).
uint64_t BeginPresent(uint64_t id);
void EndPresent(uint64_t id, bool presented);
// After a game frame's present (presented or not): present interval stats, the per-second log.
void OnGameFramePresented();
// The id for a vkQueueSubmit of the game's GPU work (0: no VkLatencySubmissionPresentIdNV).
uint64_t SubmitPresentId();

bool Active();

} // namespace BB::Reflex
