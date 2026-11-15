// Kyty: the Vulkan functions FidelityFX's frame interpolation swapchain calls directly
// (sdk/src/backends/vk/FrameInterpolationSwapchain). Kyty links no Vulkan import library, so
// ffx_vk_fi_loader.h turns each call into a call through a pointer, which
// src/graphics/host_gpu/renderer/fsr3FrameGen.cpp fills from Kyty's dispatcher.
#pragma once

#define KYTY_FFX_FI_VK_FUNCTIONS(X)                                                                \
	X(vkAcquireNextImageKHR)                                                                       \
	X(vkAllocateCommandBuffers)                                                                    \
	X(vkAllocateDescriptorSets)                                                                    \
	X(vkAllocateMemory)                                                                            \
	X(vkBeginCommandBuffer)                                                                        \
	X(vkBindImageMemory)                                                                           \
	X(vkCmdBeginRenderPass)                                                                        \
	X(vkCmdBindDescriptorSets)                                                                     \
	X(vkCmdBindPipeline)                                                                           \
	X(vkCmdCopyImage)                                                                              \
	X(vkCmdDraw)                                                                                   \
	X(vkCmdEndRenderPass)                                                                          \
	X(vkCmdPipelineBarrier)                                                                        \
	X(vkCmdPushConstants)                                                                          \
	X(vkCmdSetScissor)                                                                             \
	X(vkCmdSetViewport)                                                                            \
	X(vkCreateCommandPool)                                                                         \
	X(vkCreateDescriptorPool)                                                                      \
	X(vkCreateDescriptorSetLayout)                                                                 \
	X(vkCreateFramebuffer)                                                                         \
	X(vkCreateGraphicsPipelines)                                                                   \
	X(vkCreateImage)                                                                               \
	X(vkCreateImageView)                                                                           \
	X(vkCreatePipelineLayout)                                                                      \
	X(vkCreateRenderPass)                                                                          \
	X(vkCreateSemaphore)                                                                           \
	X(vkCreateShaderModule)                                                                        \
	X(vkCreateSwapchainKHR)                                                                        \
	X(vkDestroyCommandPool)                                                                        \
	X(vkDestroyDescriptorPool)                                                                     \
	X(vkDestroyDescriptorSetLayout)                                                                \
	X(vkDestroyFramebuffer)                                                                        \
	X(vkDestroyImage)                                                                              \
	X(vkDestroyImageView)                                                                          \
	X(vkDestroyPipeline)                                                                           \
	X(vkDestroyPipelineLayout)                                                                     \
	X(vkDestroyRenderPass)                                                                         \
	X(vkDestroySemaphore)                                                                          \
	X(vkDestroyShaderModule)                                                                       \
	X(vkDestroySwapchainKHR)                                                                       \
	X(vkDeviceWaitIdle)                                                                            \
	X(vkEndCommandBuffer)                                                                          \
	X(vkFreeCommandBuffers)                                                                        \
	X(vkFreeDescriptorSets)                                                                        \
	X(vkFreeMemory)                                                                                \
	X(vkGetDeviceProcAddr)                                                                         \
	X(vkGetImageMemoryRequirements)                                                                \
	X(vkGetPhysicalDeviceMemoryProperties)                                                         \
	X(vkGetPhysicalDeviceQueueFamilyProperties)                                                    \
	X(vkGetPhysicalDeviceSurfaceSupportKHR)                                                        \
	X(vkGetSemaphoreCounterValue)                                                                  \
	X(vkGetSwapchainImagesKHR)                                                                     \
	X(vkQueuePresentKHR)                                                                           \
	X(vkQueueSubmit)                                                                               \
	X(vkQueueWaitIdle)                                                                             \
	X(vkResetCommandBuffer)                                                                        \
	X(vkResetCommandPool)                                                                          \
	X(vkUpdateDescriptorSets)                                                                      \
	X(vkWaitSemaphores)

// Kyty: how often a swapchain submit waited for another thread to send the signal its timeline wait
// needs, and how often it gave up after 1 s (FrameInterpolationSwapchainVK_Helpers.cpp).
#ifdef __cplusplus
extern "C" void kyty_fi_GetSendGuardStats(unsigned long long* waits, unsigned long long* timeouts);
#endif
