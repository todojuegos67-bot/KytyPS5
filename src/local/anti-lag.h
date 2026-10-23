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
[[nodiscard]] bool        StreamlineReflex();    // Streamline DLSS-G owns Reflex (Anti-Lag then off)
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
// Right before the frame's present (a new frame, not a repeat): its PRESENT mark.
void OnPresent(uint64_t frame_index);

} // namespace Libs::Graphics::AntiLag

#endif
