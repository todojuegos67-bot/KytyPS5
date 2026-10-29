// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The few Kyty types the Demon's Souls fork's upscaler code (src/local) is written against, mapped
// onto shadPS4's Vulkan objects, so that code is reused as it is instead of rewritten.
#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImageState {
    vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
    vk::Format format = vk::Format::eUndefined;
    vk::Extent3D extent = {1, 1, 1};
    vk::ImageUsageFlags usage{};
    vk::Image image = nullptr;
    VulkanImageState state;
    VmaAllocation allocation = nullptr;
};

struct GraphicContext {
    vk::Instance instance = nullptr;
    vk::PhysicalDevice physical_device = nullptr;
    vk::Device device = nullptr;
    VmaAllocator allocator = nullptr;
    vk::Queue queue = nullptr;
    uint32_t queue_family = 0;
    std::mutex queue_mutex;

    bool CreateImage(const vk::ImageCreateInfo& info, VulkanImage& out) {
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkImage image = VK_NULL_HANDLE;
        const VkImageCreateInfo& ci = info;
        if (vmaCreateImage(allocator, &ci, &alloc, &image, &out.allocation, nullptr) !=
            VK_SUCCESS) {
            out.image = nullptr;
            return false;
        }
        out.image = image;
        out.format = info.format;
        out.extent = info.extent;
        out.usage = info.usage;
        out.state = {};
        return true;
    }
    void DeleteImage(VulkanImage& image) {
        if (image.image) {
            vmaDestroyImage(allocator, static_cast<VkImage>(image.image), image.allocation);
        }
        image.image = nullptr;
        image.allocation = nullptr;
    }
};

namespace FrameGen {
inline void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect,
                       vk::ImageLayout layout, vk::AccessFlags2 access,
                       vk::PipelineStageFlags2 stage) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = image.state.pl_stage;
    barrier.srcAccessMask = image.state.access_mask;
    barrier.dstStageMask = stage;
    barrier.dstAccessMask = access;
    barrier.oldLayout = image.state.layout;
    barrier.newLayout = layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    vk::DependencyInfo dep{};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    command.pipelineBarrier2(dep);
    image.state.layout = layout;
    image.state.access_mask = access;
    image.state.pl_stage = stage;
}
inline bool SharpenProbeDue() {
    return false;
}
inline void SharpenProbeCopy(vk::CommandBuffer, vk::Image, vk::ImageLayout, vk::Format,
                             vk::Extent2D, uint32_t, uint32_t, int) {}
inline void SharpenProbeArm(const char*, float, const char*) {}
} // namespace FrameGen

namespace LocalPlatform {
inline uint32_t DisplayHeight(void*) {
    return 0; // the upscaler writes the game's output size; the presenter scales to the window
}
} // namespace LocalPlatform

} // namespace Libs::Graphics

// The Demon's Souls code prints to stdout; here it goes to shadPS4's log.
inline void BBLogf(const char* format, ...) {
    char text[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    std::size_t n = std::char_traits<char>::length(text);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r')) {
        text[--n] = '\0';
    }
    LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}", text);
}
