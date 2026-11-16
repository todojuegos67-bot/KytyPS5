// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex: the Vulkan side of NVIDIA Reflex (bb_reflex.h): device and swapchain set-up.
#pragma once

#include <string>
#include <vector>
#include "video_core/bb/bb_reflex.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::Reflex {

// ---- Device creation ----
void AddDeviceExtensions(vk::PhysicalDevice physical, const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
// VkPhysicalDevicePresentIdFeaturesKHR when the extensions were added (none already chained).
void* ChainDeviceFeatures(void* next);
void OnDevice(vk::Device device);

// ---- Swapchain (present thread) ----
// The VkSwapchainLatencyCreateInfoNV to chain into the swapchain's create info (nullptr: none).
const void* SwapchainCreateNext(const void* next);
void OnSwapchainCreated(vk::SwapchainKHR swapchain, bool latency_mode);
void OnSwapchainDestroy();

} // namespace BB::Reflex
