// SPDX-FileCopyrightText: Copyright 2026 bbport contributors, KytyPS5 BB fork
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-dcideas-094: glue for bbport's FSR 4.1.1 replay (fsr411.cpp, from
// https://github.com/deadinside28/bloodborne_pc gpu/upscale/fsr411 at 0fc2921c, GPL-2.0-or-later).
// This renderer has no Vulkan prototypes (VK_NO_PROTOTYPES, vulkan-hpp's dynamic dispatcher), so
// the replay's vk* calls go through the dispatcher; bb_env / bb_env_on as bbport's shim/bb_env.h.
#pragma once
#include <cstdlib>
#include "video_core/renderer_vulkan/vk_common.h"

static inline const char* bb_env(const char* name) { return std::getenv(name); }
static inline int bb_env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}
#define vkAllocateMemory VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory
#define vkBindBufferMemory VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory
#define vkBindImageMemory VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory
#define vkCmdBindPipeline VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBindPipeline
#define vkCmdClearColorImage VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdClearColorImage
#define vkCmdCopyBuffer VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer
#define vkCmdDispatch VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdDispatch
#define vkCmdFillBuffer VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdFillBuffer
#define vkCmdPipelineBarrier VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPipelineBarrier
#define vkCmdResetQueryPool VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdResetQueryPool
#define vkCmdWriteTimestamp VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteTimestamp
#define vkCreateBuffer VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer
#define vkCreateComputePipelines VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines
#define vkCreateDescriptorSetLayout VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout
#define vkCreateImage VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage
#define vkCreateImageView VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImageView
#define vkCreatePipelineLayout VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineLayout
#define vkCreateQueryPool VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateQueryPool
#define vkCreateSampler VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSampler
#define vkCreateShaderModule VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule
#define vkDestroyBuffer VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer
#define vkDestroyDescriptorSetLayout VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyDescriptorSetLayout
#define vkDestroyImage VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage
#define vkDestroyImageView VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImageView
#define vkDestroyPipeline VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipeline
#define vkDestroyPipelineLayout VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineLayout
#define vkDestroyQueryPool VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyQueryPool
#define vkDestroySampler VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroySampler
#define vkDestroyShaderModule VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyShaderModule
#define vkFreeMemory VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory
#define vkGetBufferMemoryRequirements VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements
#define vkGetDeviceProcAddr VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr
#define vkGetImageMemoryRequirements VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements
#define vkGetPhysicalDeviceMemoryProperties VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceProperties VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties
#define vkGetQueryPoolResults VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults
#define vkMapMemory VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory
