// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-imgfg-078: image-only frame generation (BB_FG=image-x2 | image-x3). Between each pair of
// real frames one (two) intermediate frame(s) is computed on the GPU from the two finished
// images, the way a screen-capture frame generator does, with no vectors, depth or HUD-less image
// needed: block optical flow on a luma pyramid (locally contrast-normalised), forward and
// backward, a consistency check, warps of both frames to the intermediate time with the
// confidence deciding between the warped mix and the nearest real frame, and a protection of the
// HUD (static high-contrast pixels are taken from the real frame, not warped). When the emulator
// does have the game's own data for the frame (the HUD-less image, the camera motion of the
// upscaler's vectors) they improve the result (BB_IMGFG_GAMEDATA=0: never used): the 3D image is
// interpolated without the HUD and the HUD composited on top unmoved, and the camera motion is
// one more starting guess for the flow (the image match still decides). Where they are missing,
// the pure image path runs.
//
// The module sits at the presentation: BB::FG::BeforePresent notes the frame's inputs, and
// Vulkan::Swapchain::Present calls BeforeRealPresent just before it presents the real frame: the
// intermediate frames are presented from there (their own swapchain images), at the pacer's
// grid, and the real frame follows half (a third) of a real interval later.
#pragma once

#include <cstdint>
#include <mutex>
#include "video_core/bb/bb_fg.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::ImgFg {

// BB_FG=image-xN asked.
bool Requested();
// 2 or 3: the shown frames per real one.
int Multiplier();
// Generating intermediate frames at the moment (the pacer treats it like frame generation).
bool Generating();
// The upscaler's inputs (hudless image, vectors) are wanted.
bool WantInputs();
// Swapchain creation: the usage to add (sampled, when the surface allows it) and the extra images.
vk::ImageUsageFlags SwapchainUsage(vk::PhysicalDevice physical, vk::SurfaceKHR surface);
uint32_t SwapchainExtraImages();

void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t family, vk::Queue queue);
// The swapchain is going away or was created (resources follow the next present's size).
void OnSwapchain(vk::SwapchainKHR swapchain);

// Presenter: before BeforePresent. new_game_frame: a new frame of the game (not a repeat of the
// last one); inputs: its fg inputs (null when the upscaler gave none); region: the game image
// inside the backbuffer.
void NoteFrame(bool new_game_frame, const BB::FG::Inputs* inputs, vk::Extent2D backbuffer,
               vk::Rect2D region);

struct PresentIn {
    vk::SwapchainKHR swapchain{};
    vk::Image image{};   // the real frame's swapchain image, presentSrc
    vk::Extent2D extent{};
    vk::Format format = vk::Format::eUndefined;
};
// The presenter, before Swapchain::Present of the real frame (submit_mutex NOT held: the sleeps
// to the grid must not hold it; the queue work takes it). Presents the
// intermediate frames, waits for the real frame's slot, and returns true with `wait` set to the
// semaphore the real frame's present must also wait for (the capture reads the image).
bool BeforeRealPresent(const PresentIn& in, vk::Semaphore* wait);

} // namespace BB::ImgFg
