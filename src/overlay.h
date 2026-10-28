// The F1 overlay (Dear ImGui, Vulkan backend): drawn into the swapchain image right before present, on the game
// queue. F1 opens / closes the menu, F2 toggles frame generation without the menu. Input is polled (no window
// procedure hook): cursor position and mouse buttons through GetCursorPos / GetAsyncKeyState.
#pragma once
#include "common.h"

namespace inj::overlay {

void OnSwapchain(VkSwapchainKHR swapchain, const VkImage* images, uint32_t count, VkFormat format, VkExtent2D extent);
void OnSwapchainDestroyed(VkSwapchainKHR swapchain);
void PollInput();             // once per present
bool WantsDraw();             // menu open or the OSD on
bool MenuVisible();
void Record(VkCommandBuffer cb, uint32_t image_index);  // the overlay render pass into that swapchain image

void* Texture(VkImageView view);  // an ImGui texture for a preview (SHADER_READ_ONLY_OPTIMAL)
void  ForgetTexture(VkImageView view);

} // namespace inj::overlay
