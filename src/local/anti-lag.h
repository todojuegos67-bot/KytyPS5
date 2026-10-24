#ifndef EMULATOR_SRC_LOCAL_ANTI_LAG_H_
#define EMULATOR_SRC_LOCAL_ANTI_LAG_H_
// KYTY_ANTILAG=1: low latency for any GPU (the AMD counterpart of the Reflex that DLSS-G brings, frame-gen.cpp).
//  - AMD Anti-Lag 2 through VK_AMD_anti_lag when the device has it: vkAntiLagUpdateAMD(INPUT) where the game reads its
//    controller for a frame (the driver may sleep the game's thread there), vkAntiLagUpdateAMD(PRESENT) with the same
//    frame index right before that frame's present. With FSR 3 frame generation the PRESENT mark goes on the real frame
//    handed to the frame generation swapchain (Fsr3::QueuePresent), never on the generated ones it presents itself, as
//    AMD asks for frame generation. maxFPS 0: KYTY_FLIP_FPS / the flip pacing keep the rate (no second limiter).
//  - KYTY_ANTILAG=2 only (experimental: corrupted graphics in a test on NVIDIA), VK_AMD_anti_lag missing: the emulator's limiter, at most one frame queued between the game and the
//    present: the controller read of frame k waits (at most 100 ms) until frame k-2 has been presented.
//  - NVIDIA (VK_NV_low_latency2 + VK_KHR_present_id, no Streamline): Reflex. vkLatencySleepNV waited for at the same
//    controller read, SIMULATION_START / INPUT_SAMPLE there, SIMULATION_END / RENDERSUBMIT_START at the flip,
//    RENDERSUBMIT_END / PRESENT_START / PRESENT_END around the present, whose VkPresentIdKHR is the frame index.
//    With FSR 3 frame generation: on its real swapchain through the patched amd_fidelityfx_vk.dll
//    (kytyFfxGetRealSwapchainVK; its present thread chains the id and marks PRESENT_START / END on real frames), else off.
// Off when Streamline (DLSS / DLSS-G) is loaded: Reflex is then active.
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>

namespace Libs::Graphics::AntiLag {

[[nodiscard]] bool        Requested();           // KYTY_ANTILAG=1 (and no Streamline)
[[nodiscard]] const char* ExtensionName();        // "VK_AMD_anti_lag"
// Device creation: its feature structure chained before `next` when `enabled` (the extension in the list).
void* ChainDeviceFeatures(bool enabled, void* next);
// After vkCreateDevice: which mode runs, logged once.
void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc, bool extension_enabled, bool reflex_enabled);

// NVIDIA Reflex (VK_NV_low_latency2, KYTY_ANTILAG=1 without Streamline): the swapchain's VkSwapchainLatencyCreateInfoNV
// chained before `next`; the swapchain made (sleep mode on) or about to go (VK_NULL_HANDLE); FSR 3's: none (logged).
[[nodiscard]] bool ReflexActive();
void*              ChainSwapchain(void* next, bool fsr3);
void               OnSwapchain(VkSwapchainKHR swapchain, bool fsr3);
// The present's VkPresentIdKHR (0: none) and its PRESENT_END marker.
[[nodiscard]] uint64_t PresentId(uint64_t frame_index);
void                   AfterPresent(uint64_t frame_index);

// The game reads its controller (scePadRead / scePadReadState): once per flip, the frame's INPUT mark (or the limiter).
void OnInput();
// A guest flip reserved: the frame's index (0: none).
[[nodiscard]] uint64_t OnFlipReserved();
// The flip's GPU work done (FlipQueue::Complete) / the flip released to the presenter (FlipQueue::Flip): stage stamps
// for KYTY_LOWLAT_PACE_BEFORE_INPUT (default on with KYTY_LOWLAT=1: the game's thread sleeps at its controller read
// until the latest moment its frame is still ready for its release slot, so frames no longer queue behind the cap)
// and the [lat] stages line.
void OnFlipReady(uint64_t frame_index);
void OnFlipReleased(uint64_t frame_index);
// Before OnFlipReleased: the configured release interval of that frame (the cap: KYTY_FLIP_FPS / the vblank floor, ms)
// when it is a game-view frame under a cap, else 0 (movie, menu, cinematic, loading, no cap: the pacing is off).
void SetPaceCap(double cap_ms, double vblank_ms);
// Right before the frame's present (a new frame, not a repeat): its PRESENT mark.
void OnPresent(uint64_t frame_index);

// KYTY_LOWLAT=1: the emulator's own latency limiter (no Reflex / Anti-Lag / Streamline, nothing on any swapchain or
// present). Each guest flip records the host GPU timeline tick (MasterSemaphore, lane 0) of its frame's last submission;
// at the controller read of the next frame the game's thread waits (vkWaitSemaphores, at most 50 ms, no lock held)
// until no more than KYTY_LOWLAT_QUEUE (1|2, default 2) real frames are still on the GPU.
[[nodiscard]] bool OwnLowLatencyOn();
void               OnFrameSubmitted(VkDevice device, VkSemaphore timeline, uint64_t tick);

// KYTY_LATENCY=1 (the launcher sets it with the metrics overlay): input -> present latency, any Anti-Lag mode or none.
// The game's controller read that starts frame N (the first scePadRead / scePadReadState after its last flip, after any
// Reflex / Anti-Lag 2 / KYTY_LOWLAT wait: the same point as Reflex's INPUT_SAMPLE) is stamped with
// QueryPerformanceCounter; the frame id goes with the guest flip (OnFlipReserved) to the present thread, and
// OnPresented(id) right after vkQueuePresentKHR of that real frame returned (the GPU work done: the present thread
// waited for it; with DLSS-G / FSR 3 frame generation this is the real frame handed to the generating swapchain, never a
// generated one) takes the difference. Once a second, on stdout (parsed by the metrics overlay, metricas.ps1):
//   [lat] input->present mean 38.2 ms, p95 45.1, max 52.0; +scanout 2.1; fg +16.7 (n 60); total 57.0, p95 63.9; mode reflex
// +scanout: an estimate, half the display's refresh interval (present -> the middle of the screen scanned out), not
// measured; fg: the frame generation hold of the real frame, estimated as real interval x g / (g + 1) for g generated
// frames per real one (x2: half a real frame), 0 without frame generation; total = input->present + scanout + fg.
// Off: one cached flag test per controller read / flip / present.
[[nodiscard]] bool LatencyOn();
void               OnPresented(uint64_t frame_index);

} // namespace Libs::Graphics::AntiLag

#endif
