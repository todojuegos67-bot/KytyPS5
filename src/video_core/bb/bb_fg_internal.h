// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Between bb_fg.cpp (settings) and bb_fg_fsr3.cpp (FSR 3 frame generation).
#pragma once

#include <string>
#include <vector>
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
double PaceInterval();
bool Generating();
void NoteSlept(double seconds);
void SetUiLayer(const Image* ui);
bool WantsUiLayer();
void PanelValues(float& real, float& shown, const char*& state);
double DisplayHz();
} // namespace BB::FG::Amd

// bb_fg_dlssg.cpp: NVIDIA DLSS-G (BB_FG=dlss-*).
namespace BB::FG::Dlssg {
void OnDevice(vk::PhysicalDevice physical);
void OnDispatch(vk::Device device);
bool Ready();
int Generated();
void BeforePresent(const Inputs* inputs, vk::Extent2D backbuffer, vk::Format format,
                   vk::Rect2D region, uint32_t image_count);
void AfterPresent();
void SwapchainChanging();
vk::PresentModeKHR PreferredPresentMode(vk::PresentModeKHR current,
                                        const std::vector<vk::PresentModeKHR>& available);
uint32_t PreferredImageCount(uint32_t current, uint32_t min, uint32_t max);
void AcquireTime(double ms);
} // namespace BB::FG::Dlssg
