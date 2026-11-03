// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Between bb_fg.cpp (settings, DLSS-G) and bb_fg_fsr3.cpp (FSR 3 frame generation).
#pragma once

#include "video_core/bb/bb_fg.h"

namespace BB::FG::Amd {
bool Requested();
bool Active();
QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family);
void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t graphics_family,
              vk::Queue graphics_queue, const QueuePlan& plan, std::mutex* submit_mutex);
bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out);
void DestroySwapchain();
vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images);
vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index);
vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info);
void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format);
int Generated();
double PaceInterval(); // the real-frame limit while generating (0: none)
bool Generating();     // generating frames now
void NoteSlept(double seconds);
} // namespace BB::FG::Amd

namespace BB::FG {
// The interval the game's flips were last held to (bb_fg.cpp PaceGuestFlip; log).
double PaceEffective();
} // namespace BB::FG
