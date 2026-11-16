// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex: NVIDIA Reflex through VK_NV_low_latency2 when the launcher selects it (BB_FG=dlss-*
// with BB_ANTILAG_DLSSG=reflex) or BB_REFLEX=1 (any mode but FSR 3 FG; it is the only way to get
// it without frame generation: opt-in, see Cfg() in bb_reflex.cpp); BB_REFLEX=0 / off turns it
// off. Not requested: no device extension, no latency swapchain, nothing else changes. Streamline's Reflex is not used: Streamline
// runs with manual hooking and DLSS only (the present never goes through it), it is loaded only
// with the DLSS upscaler, and its frame token is made per DLSS evaluate on the GPU thread, while
// the DLSS-G frames come from NGX (bbport_dlss) in the presenter.
//
// Frame ids: one per game frame, made at the game's frame start (its frame limiter's gettimeofday,
// eboot 1.09 0x2034977, on the main thread), then followed through the render thread
// (GXRenderThread: guest submits, guest flip), the GPU thread (the flip's processing) and the
// present thread (the game frame's vkQueuePresentKHR, VkPresentIdKHR = the id).
// Markers: SIMULATION_START (after the sleep), SIMULATION_END (next frame start or the frame's
// first guest submit), RENDERSUBMIT_START / END (first guest submit / guest flip), PRESENT_START /
// END (around the game frame's present). Sleep: vkLatencySleepNV at the game's frame start.
// Frame cap: with BB_VARIABLE_DT's BB_GAME_FPS_CAP, Reflex's minimumIntervalUs and a 1 ms period
// in the game's limiter (written each frame into the limiter object, bb_vardt.cpp; no code
// patch), so only one of the two waits. BB_REFLEX_CAP=0 leaves the cap to the game's limiter.

// Device and swapchain set-up: bb_reflex_vk.h.
#pragma once

#include <cstdint>
#include <string>

namespace BB::Reflex {

// Asked by the launcher's settings (and not turned off); decided once.
bool Requested();

// ---- Game frame ----
void OnEbootLoaded(uintptr_t eboot_base, uint64_t image_size);
// Every guest gettimeofday (cheap unless it is the limiter's frame start).
void OnGuestClock(const void* return_address);
void OnGuestSubmit(); // render thread, guest command buffer submit
void OnGuestFlip();   // render thread, guest flip about to be queued
void OnGuestFlipDropped(); // the flip was refused (queue full): its id is forgotten
uint64_t OnGpuFlip(); // GPU thread, the flip's frame prepared: its id (0: none)

// ---- Present thread ----
// The game frame about to be presented (its id from OnGpuFlip): PRESENT_START; returns the id
// for its VkPresentIdKHR (0: none).
uint64_t BeginPresent(uint64_t id);
void EndPresent(uint64_t id, bool presented);
// The id for a vkQueueSubmit of the game's GPU work (0: no VkLatencySubmissionPresentIdNV).
uint64_t SubmitPresentId();

bool Active();
// "markers sent N (...)" for the [fg-pacing] line; resets the per-second counters.
std::string TakeSecondSummary(double sec);

} // namespace BB::Reflex
