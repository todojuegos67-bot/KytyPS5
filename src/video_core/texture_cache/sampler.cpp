// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/sampler.h"

namespace VideoCore {

Sampler::Sampler(const Vulkan::Instance& instance, const AmdGpu::Sampler& sampler,
                 const AmdGpu::BorderColorBuffer border_color_base, const bool is_depth,
                 float extra_lod_bias, u32 bb_filter) {
    using namespace Vulkan;
    // bb-tex (BB_TEX_QUALITY / BB_ANISO, scene draws only, bb_texture.h): bits 0-4 force the
    // anisotropy ratio of a linear-filtered sampler (also one the game left at 1x), bit 6 turns
    // its point-mip filter into a linear one (trilinear). Point-filtered samplers keep the game's.
    const bool bb_linear = LiverpoolToVK::Filter(sampler.xy_min_filter) == vk::Filter::eLinear;
    const u32 bb_ratio = bb_linear ? (bb_filter & 31u) : 0u;
    const bool game_aniso = AmdGpu::IsAnisoFilter(sampler.xy_mag_filter) ||
                            AmdGpu::IsAnisoFilter(sampler.xy_min_filter);
    const bool anisotropy_enable = instance.IsAnisotropicFilteringSupported() &&
                                   (bb_ratio != 0 ? bb_ratio > 1 : game_aniso);
    const float max_anisotropy =
        anisotropy_enable
            ? std::clamp(bb_ratio != 0 ? float(bb_ratio) : sampler.MaxAniso(), 1.0f,
                         instance.MaxSamplerAnisotropy())
            : 1.0f;
    auto mip_mode = LiverpoolToVK::MipFilter(sampler.mip_filter);
    if ((bb_filter & 64u) && bb_linear && sampler.mip_filter == AmdGpu::MipFilter::Point) {
        mip_mode = vk::SamplerMipmapMode::eLinear;
    }
    auto border_color = LiverpoolToVK::BorderColor(sampler.border_color_type);
    if (border_color == vk::BorderColor::eFloatCustomEXT &&
        !instance.IsCustomBorderColorSupported()) {
        LOG_WARNING(Render_Vulkan, "Custom border color is not supported, falling back to black");
        border_color = vk::BorderColor::eFloatOpaqueBlack;
    }

    const auto custom_color = [&]() -> std::optional<vk::SamplerCustomBorderColorCreateInfoEXT> {
        if (border_color == vk::BorderColor::eFloatCustomEXT) {
            const auto border_color_index = sampler.border_color_ptr.Value();
            const auto border_color_buffer = border_color_base.Address<std::array<float, 4>*>();
            const auto custom_border_color_array = border_color_buffer[border_color_index];

            const vk::SamplerCustomBorderColorCreateInfoEXT ret{
                .customBorderColor =
                    vk::ClearColorValue{
                        .float32 = custom_border_color_array,
                    },
                .format = vk::Format::eR32G32B32A32Sfloat,
            };
            return ret;
        } else {
            return std::nullopt;
        }
    }();

    const vk::SamplerCreateInfo sampler_ci = {
        .pNext = custom_color ? &*custom_color : nullptr,
        .magFilter = LiverpoolToVK::Filter(sampler.xy_mag_filter),
        .minFilter = LiverpoolToVK::Filter(sampler.xy_min_filter),
        .mipmapMode = mip_mode,
        .addressModeU = LiverpoolToVK::ClampMode(sampler.clamp_x),
        .addressModeV = LiverpoolToVK::ClampMode(sampler.clamp_y),
        .addressModeW = LiverpoolToVK::ClampMode(sampler.clamp_z),
        // bb-gpu6 (BB_LOD_BIAS): scene samplers get log2(render/output) more when upscaling.
        .mipLodBias = std::clamp(sampler.LodBias() + extra_lod_bias, -instance.MaxSamplerLodBias(),
                                 instance.MaxSamplerLodBias()),
        .anisotropyEnable = anisotropy_enable,
        .maxAnisotropy = max_anisotropy,
        // GCN compares per instruction; a plain read of a compare sampler is undefined in Vulkan.
        .compareEnable = is_depth,
        .compareOp = LiverpoolToVK::DepthCompare(sampler.depth_compare_func),
        .minLod = sampler.MinLod(),
        .maxLod = sampler.MaxLod(),
        .borderColor = border_color,
        .unnormalizedCoordinates = false, // Handled in shader due to Vulkan limitations.
    };
    auto [sampler_result, smplr] = instance.GetDevice().createSamplerUnique(sampler_ci);
    ASSERT_MSG(sampler_result == vk::Result::eSuccess, "Failed to create sampler: {}",
               vk::to_string(sampler_result));
    handle = std::move(smplr);
}

Sampler::~Sampler() = default;

} // namespace VideoCore
