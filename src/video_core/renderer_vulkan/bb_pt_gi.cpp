// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include "common/logging/log.h"
#include "video_core/host_shaders/bb_pt_gi_accum_comp.h"
#include "video_core/host_shaders/bb_pt_gi_trace_comp.h"
#include "video_core/renderer_vulkan/bb_pt_gi.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB::PathTrace {

namespace {

constexpr float CutCosine = 0.906f;    // cos(25 degrees): a forward turn that resets the history
constexpr float CutTranslation = 3.0f; // view units: a move that resets the history

float EnvNumber(const char* name, float def, float lo, float hi) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) {
        return def;
    }
    char* end = nullptr;
    const float v = std::strtof(e, &end);
    return end != e && std::isfinite(v) ? std::clamp(v, lo, hi) : def;
}

u32 GiSpp() {
    static const u32 v = u32(EnvNumber("BB_PT_GI_SPP", 1.0f, 1.0f, 16.0f));
    return v;
}

u32 GiBounces() {
    static const u32 v = u32(EnvNumber("BB_PT_GI_BOUNCES", 2.0f, 1.0f, 8.0f));
    return v;
}

float GiClamp() {
    static const float v = EnvNumber("BB_PT_GI_CLAMP", 4.0f, 0.01f, 1000.0f);
    return v;
}

float GiBias() {
    static const float v = EnvNumber("BB_PT_GI_BIAS", 0.02f, 0.0f, 1.0f);
    return v;
}

float GiHistory() {
    static const float v = EnvNumber("BB_PT_GI_HISTORY", 16.0f, 1.0f, 256.0f);
    return v;
}

float GiStrength() {
    static const float v = EnvNumber("BB_PT_GI_STRENGTH", 1.0f, 0.0f, 16.0f);
    return v;
}

// Matches bb_pt_gi_trace.comp's push constants (48 bytes).
struct TracePush {
    float p00, p11, p02, p12;
    u32 width, height, spp, bounces, seed;
    float radiance_clamp;
    float bias;
    u32 game_srgb;
};
static_assert(sizeof(TracePush) == 48);

// Matches bb_pt_gi_accum.comp's push constants (100 bytes): the previous-from-current transform
// as three rows of 4, then the previous and current projections.
struct AccumPush {
    float m[12];
    float p00p, p11p, p02p, p12p;
    float p00, p11, p02, p12;
    u32 width, height;
    float max_history;
    u32 cut;
    float strength;
};
static_assert(sizeof(AccumPush) == 100);

} // namespace

bool GiRequested() {
    static const bool on = [] {
        const char* gi = std::getenv("BB_PT_GI");
        const char* debug = std::getenv("BB_PT_DEBUG");
        return !(gi && gi[0] == '0') && !(debug && debug[0] == '1');
    }();
    return on;
}

Gi::Gi(const Vulkan::Instance& instance_) : instance{instance_} {}

Gi::~Gi() {
    const auto device = instance.GetDevice();
    if (sampler) {
        device.destroySampler(sampler);
    }
    FreePass(trace_pass);
    FreePass(accum_pass);
    FreeImage(raw);
    FreeImage(zcur);
    FreeImage(final_img);
    for (auto& img : hist) {
        FreeImage(img);
    }
    for (auto& img : zhist) {
        FreeImage(img);
    }
}

bool Gi::MakeImage(Img& img, vk::Extent2D extent, vk::Format format) {
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &img.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    img.image = image;
    auto [r, v] = instance.GetDevice().createImageView({
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) {
        FreeImage(img);
        return false;
    }
    img.view = v;
    img.general = false;
    return true;
}

void Gi::FreeImage(Img& img) {
    if (img.view) {
        instance.GetDevice().destroyImageView(img.view);
    }
    if (img.image) {
        vmaDestroyImage(instance.GetAllocator(), img.image, img.allocation);
    }
    img = {};
}

bool Gi::MakeImages(vk::Extent2D extent) {
    if (raw.image && size == extent) {
        return true;
    }
    if (raw.image) {
        (void)instance.GetDevice().waitIdle(); // a resize only
    }
    FreeImage(raw);
    FreeImage(zcur);
    FreeImage(final_img);
    for (auto& img : hist) {
        FreeImage(img);
    }
    for (auto& img : zhist) {
        FreeImage(img);
    }
    constexpr auto rgba = vk::Format::eR16G16B16A16Sfloat;
    constexpr auto single = vk::Format::eR32Sfloat;
    const bool ok = MakeImage(raw, extent, rgba) && MakeImage(zcur, extent, single) &&
                    MakeImage(final_img, extent, rgba) &&
                    MakeImage(hist[0], extent, rgba) && MakeImage(hist[1], extent, rgba) &&
                    MakeImage(zhist[0], extent, single) && MakeImage(zhist[1], extent, single);
    if (!ok) {
        LOG_ERROR(Render_Vulkan, "[pt-gi] images {}x{} failed: GI off this frame", extent.width,
                  extent.height);
        return false;
    }
    size = extent;
    history_valid = false;
    return true;
}

bool Gi::MakePass(std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
                  std::span<const u32> spirv, Pass& pass) {
    const auto device = instance.GetDevice();
    auto [lr, set] = device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (lr != vk::Result::eSuccess) {
        return false;
    }
    pass.set = set;
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, push_size};
    auto [pr, layout] = device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &pass.set,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) {
        return false;
    }
    pass.layout = layout;
    const vk::ShaderModule module = Vulkan::CompileSPV(spirv, device);
    if (!module) {
        return false;
    }
    auto [cr, pipe] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage{.stage = vk::ShaderStageFlagBits::eCompute,
                       .module = module,
                       .pName = "main"},
                .layout = pass.layout,
            });
    device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) {
        return false;
    }
    pass.pipe = pipe;
    return true;
}

void Gi::FreePass(Pass& pass) {
    const auto device = instance.GetDevice();
    if (pass.pipe) {
        device.destroyPipeline(pass.pipe);
    }
    if (pass.layout) {
        device.destroyPipelineLayout(pass.layout);
    }
    if (pass.set) {
        device.destroyDescriptorSetLayout(pass.set);
    }
    pass = {};
}

bool Gi::Init() {
    const auto device = instance.GetDevice();
    auto [sr, s] = device.createSampler({
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    });
    if (sr != vk::Result::eSuccess) {
        return false;
    }
    sampler = s;
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    using Type = vk::DescriptorType;
    const std::array<vk::DescriptorSetLayoutBinding, 5> trace_bindings{{
        {0, Type::eAccelerationStructureKHR, 1, C},
        {1, Type::eStorageBuffer, 1, C},
        {2, Type::eCombinedImageSampler, 1, C},
        {3, Type::eStorageImage, 1, C},
        {4, Type::eStorageImage, 1, C},
    }};
    const std::array<vk::DescriptorSetLayoutBinding, 8> accum_bindings{{
        {0, Type::eCombinedImageSampler, 1, C},
        {1, Type::eCombinedImageSampler, 1, C},
        {2, Type::eCombinedImageSampler, 1, C},
        {3, Type::eCombinedImageSampler, 1, C},
        {4, Type::eCombinedImageSampler, 1, C},
        {5, Type::eStorageImage, 1, C},
        {6, Type::eStorageImage, 1, C},
        {7, Type::eStorageImage, 1, C},
    }};
    if (!MakePass(trace_bindings, sizeof(TracePush), BB_PT_GI_TRACE_COMP, trace_pass) ||
        !MakePass(accum_bindings, sizeof(AccumPush), BB_PT_GI_ACCUM_COMP, accum_pass)) {
        LOG_ERROR(Render_Vulkan, "[pt-gi] passes failed: GI off");
        return false;
    }
    ready = true;
    LOG_INFO(Render_Vulkan,
             "[pt-gi] indirect light on (BB_PT_GI=1): {} sample(s) per pixel, {} bounce(s), "
             "clamp {:.2f}, bias {:.3f}, history {:.0f}, strength {:.2f}",
             GiSpp(), GiBounces(), GiClamp(), GiBias(), GiHistory(), GiStrength());
    return true;
}

void Gi::Reset() {
    history_valid = false;
}

vk::ImageView Gi::Run(vk::CommandBuffer cmd, const GiInput& in) {
    if (!ready || !in.tlas || !in.lit || !in.game || !in.proj || !in.view ||
        !MakeImages(in.extent)) {
        return {};
    }
    std::array<Img*, 8> fresh{};
    size_t fresh_count = 0;
    const auto add_fresh = [&](Img& img) {
        if (!img.general) {
            fresh[fresh_count++] = &img;
        }
    };
    add_fresh(raw);
    add_fresh(zcur);
    add_fresh(final_img);
    add_fresh(hist[0]);
    add_fresh(hist[1]);
    add_fresh(zhist[0]);
    add_fresh(zhist[1]);
    std::array<vk::ImageMemoryBarrier2, 7> to_general{};
    for (size_t i = 0; i < fresh_count; ++i) {
        fresh[i]->general = true;
        to_general[i] = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = fresh[i]->image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
    }
    if (fresh_count > 0) {
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = u32(fresh_count),
            .pImageMemoryBarriers = to_general.data(),
        });
    }

    // A camera cut (a move of more than 3 units or a forward turn of more than 25 degrees since
    // the last frame) or a missing previous frame starts the history again.
    const float* v = in.view;
    bool cut = !history_valid;
    if (!cut) {
        const float dx = v[3] - prev_view[3], dy = v[7] - prev_view[7], dz = v[11] - prev_view[11];
        const float fwd = v[8] * prev_view[8] + v[9] * prev_view[9] + v[10] * prev_view[10];
        cut = std::sqrt(dx * dx + dy * dy + dz * dz) > CutTranslation || fwd < CutCosine;
    }

    const u32 cur = slot;
    const u32 prev = slot ^ 1u;

    AccumPush accum{};
    if (!cut) {
        // ti = -R^T t (the inverse of the current view's translation), then M = Vprev * [R^T | ti].
        double ti[3]{};
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                ti[k] -= double(v[j * 4 + k]) * double(v[j * 4 + 3]);
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += double(prev_view[i * 4 + k]) * double(v[j * 4 + k]);
                }
                accum.m[i * 4 + j] = float(sum);
            }
            double t = double(prev_view[i * 4 + 3]);
            for (int k = 0; k < 3; ++k) {
                t += double(prev_view[i * 4 + k]) * ti[k];
            }
            accum.m[i * 4 + 3] = float(t);
        }
    }
    accum.p00p = prev_proj[0];
    accum.p11p = prev_proj[5];
    accum.p02p = prev_proj[2];
    accum.p12p = prev_proj[6];
    accum.p00 = in.proj[0];
    accum.p11 = in.proj[5];
    accum.p02 = in.proj[2];
    accum.p12 = in.proj[6];
    accum.width = in.extent.width;
    accum.height = in.extent.height;
    accum.max_history = GiHistory();
    accum.cut = cut ? 1u : 0u;
    accum.strength = GiStrength();

    const vk::DescriptorImageInfo raw_tex{sampler, raw.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo z_tex{sampler, zcur.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo hist_prev_tex{sampler, hist[prev].view,
                                                vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo zprev_tex{sampler, zhist[prev].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo lit_tex{sampler, in.lit, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo hist_store{{}, hist[cur].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo zhist_store{{}, zhist[cur].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo final_store{{}, final_img.view, vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 8> accum_writes{{
        {.dstBinding = 0, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &raw_tex},
        {.dstBinding = 1, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &z_tex},
        {.dstBinding = 2, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hist_prev_tex},
        {.dstBinding = 3, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &zprev_tex},
        {.dstBinding = 4, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &lit_tex},
        {.dstBinding = 5, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &hist_store},
        {.dstBinding = 6, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &zhist_store},
        {.dstBinding = 7, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &final_store},
    }};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, accum_pass.pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, accum_pass.layout, 0, accum_writes);
    cmd.pushConstants(accum_pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(accum),
                      &accum);
    cmd.dispatch((in.extent.width + 7) / 8, (in.extent.height + 7) / 8, 1);

    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = final_img.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &to_read});

    std::copy(v, v + 12, prev_view);
    std::copy(in.proj, in.proj + 16, prev_proj);
    history_valid = true;
    slot = prev;
    ++frame_index;
    return final_img.view;
}

} // namespace BB::PathTrace
