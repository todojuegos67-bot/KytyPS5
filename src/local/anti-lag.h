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
// Off when Streamline (DLSS / DLSS-G) is loaded: Reflex is then active.
#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>

namespace Libs::Graphics::AntiLag {

[[nodiscard]] bool        Requested();           // KYTY_ANTILAG=1 (and no Streamline)
[[nodiscard]] const char* ExtensionName();        // "VK_AMD_anti_lag"
// Device creation: its feature structure chained before `next` when `enabled` (the extension in the list).
void* ChainDeviceFeatures(bool enabled, void* next);
// After vkCreateDevice: which mode runs, logged once.
void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc, bool extension_enabled);

// The game reads its controller (scePadRead / scePadReadState): once per flip, the frame's INPUT mark (or the limiter).
void OnInput();
// A guest flip reserved: the frame's index (0: none).
[[nodiscard]] uint64_t OnFlipReserved();
// Right before the frame's present (a new frame, not a repeat): its PRESENT mark.
void OnPresent(uint64_t frame_index);

} // namespace Libs::Graphics::AntiLag

#endif
