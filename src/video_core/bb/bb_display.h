// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-fg-074: when the images reach the screen, not when the pacer let them go. Every game present
// carries a VkPresentIdKHR; a thread waits for each id with VK_KHR_present_wait and takes the clock
// when the driver says the image is displayed. With FSR 3 frame generation the patched
// FidelityFX DLL gives every present to the driver (the generated frames too) its own id, so each
// shown image is timed. With DLSS-G Streamline presents the generated frames itself: the counters
// of sl::DLSSGState tell how many were shown, the waits time the real frames.
// One line a second: "[display] shown X fps (real Y, generated Z), interval mean/sd/p99 ms,
// dropped N, source=present_wait|sl_state|none". BB_DISPLAY=0: none of it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::FG::Display {

enum class Path : uint32_t { Plain, Amd, Dlss };

bool Wanted();

void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void QueryFeatures(vk::PhysicalDevice physical);
void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next);
void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc);

// The patched FidelityFX DLL's exports (null: AMD's DLL, no ids for the generated frames).
void SetAmdHooks(VkSwapchainKHR (*real_swapchain)(VkSwapchainKHR),
                 uint32_t (*take_ids)(uint64_t*, uint32_t));
// present ids can be chained to the presents (the extensions and features are on).
bool IdsAvailable();

void NotePresentMode(int mode);
void OnSwapchain(vk::SwapchainKHR swapchain, Path path); // null: going away

// Generated frames the next present is expected to show (0: none).
void SetExpected(int generated);
// One game present: its VkPresentIdKHR (0: none). preferred: the id to use when it is above the
// last one (DLSS-G: Streamline's frame index, the one its Reflex markers carry).
uint64_t Issue(uint64_t preferred);
// slDLSSGGetState after a present: the frames shown for it.
void NoteStreamline(uint32_t presented, uint32_t status, uint32_t generate_max);

// VK_FORMAT label and whether Streamline's format table (sl.chi vulkan.cpp getFormat) has it.
const char* FormatLabel(uint32_t vk_format, bool* known_to_streamline);

} // namespace BB::FG::Display
