//  SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Frame;
}

namespace Vulkan::HostPasses {

class PostProcessingPass {
public:
    struct Settings {
        float gamma = 1.0f;
        u32 hdr = 0;
        u32 srgb_input = 0;
        // Set by Render: input/output size ratio when the input is larger (Catmull-Rom downscale).
        float downscale = 1.0f;
        // bb-overscan: source UV scale about the centre (< 1 crops the edges); the presenter
        // sets OverscanScale() or 1 (frames given to frame generation: its depth and vectors are
        // not cropped, so the colour must not be either).
        float uv_scale = 1.0f;
    };

    static float OverscanScale(); // BB_OVERSCAN_PERCENT (default 0.4, 0 = off) as a UV scale

    void Create(vk::Device device, vk::Format surface_format);

    void Render(vk::CommandBuffer cmdbuf, vk::ImageView input, vk::Extent2D input_size,
                Frame& output, Settings settings);

private:
    vk::UniquePipeline pipeline{};
    vk::UniquePipelineLayout pipeline_layout{};
    vk::UniqueDescriptorSetLayout desc_set_layout{};
    vk::UniqueSampler sampler{};
};

} // namespace Vulkan::HostPasses
