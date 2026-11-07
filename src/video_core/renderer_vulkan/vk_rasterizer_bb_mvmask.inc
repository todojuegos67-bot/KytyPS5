// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// 1.54 (bb-mvmask-154): included inside vk_rasterizer_bb.cpp's anonymous namespace.
// Whole-image motion field + source / confidence mask (host_shaders/bb_mvmask.comp). Runs once per
// frame after the camera pass, the object compose and the optional BB_FG_MV flow, before the
// upscaler and the frame-generation copy: refines g_bb.mv in place (the one image both the
// upscaler and FG read) and raises g_bb.reactive (the upscalers' reactive / T&C / bias-current-
// colour / responsive mask, and FG's mask copy) where the vectors are not trusted.

// BB_MVMASK=0: off (1.53 vectors: camera + object [+ BB_FG_MV flow]).
bool MvMaskWanted() {
    static const bool on = [] {
        const char* e = std::getenv("BB_MVMASK");
        const bool v = e == nullptr || e[0] != '0';
        BB::FG::Printf("[mvmask] 1.54 BB_MVMASK=%s: %s", e ? e : "1 (default)",
                       v ? "per-pixel motion field = camera reprojection + skinned object vectors "
                           "+ quarter-res photometric flow where those do not explain the image; "
                           "mask (class 0 camera / 1 skinned / 2 flow + confidence) raises the "
                           "upscalers' reactive mask and is given to frame generation"
                         : "off");
        return v;
    }();
    return on;
}

bool InitMvMask() {
    if (g_bb.mvm_init) return !g_bb.mvm_failed;
    g_bb.mvm_init = true;
    g_bb.mvm_failed = true;
    const vk::Device device = g_bb.device;
    std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
    for (u32 i = 0; i < 8; ++i) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            i, i == 7 ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eStorageImage, 1,
            vk::ShaderStageFlagBits::eCompute};
    }
    auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 48};
    auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto module = CompileSPV(BB_MVMASK_COMP, device);
    auto [pipe_result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage = vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eCompute,
                                                           .module = module,
                                                           .pName = "main"},
                .layout = layout,
            });
    device.destroyShaderModule(module);
    if (set_result != vk::Result::eSuccess || layout_result != vk::Result::eSuccess ||
        pipe_result != vk::Result::eSuccess) {
        BB::FG::Printf("[mvmask] the pass could not be created: off");
        return false;
    }
    for (u32 i = 0; i < g_bb.mvm_buf.size(); ++i) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = 64,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        };
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer buffer{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g_bb.allocator, &ci, &alloc, &buffer, &g_bb.mvm_alloc[i], &info) !=
            VK_SUCCESS) {
            BB::FG::Printf("[mvmask] statistics buffer not created: off");
            return false;
        }
        g_bb.mvm_buf[i] = buffer;
        g_bb.mvm_mapped[i] = static_cast<u32*>(info.pMappedData);
        std::memset(info.pMappedData, 0, 64);
    }
    g_bb.mvm_set_layout = set_layout;
    g_bb.mvm_layout = layout;
    g_bb.mvm_pipeline = pipeline;
    g_bb.mvm_failed = false;
    return true;
}

void BBMvMask(vk::CommandBuffer cmd, OwnImage& lum_prev, OwnImage& lum_cur, vk::Extent2D render,
              bool prev_valid) {
    if (!MvMaskWanted() || !InitMvMask()) return;
    const vk::Extent2D quarter{(render.width + 3) / 4, (render.height + 3) / 4};
    const auto storage = vk::ImageUsageFlagBits::eStorage;
    if (!EnsureImage(g_bb.mvm_blocks, quarter, vk::Format::eR16G16B16A16Sfloat, storage) ||
        !EnsureImage(g_bb.mvm_cls, render, vk::Format::eR16Uint,
                     storage | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eSampled) ||
        !EnsureImage(g_bb.mvm_dummy, {1, 1}, vk::Format::eR32G32Uint,
                     storage | vk::ImageUsageFlagBits::eTransferDst)) {
        BB::FG::Printf("[mvmask] images not created: off");
        g_bb.mvm_failed = true;
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (g_bb.mvm_t == std::chrono::steady_clock::time_point{}) g_bb.mvm_t = now;
    // Statistics of the slot written 4 runs ago (finished on the GPU by now), then cleared.
    const u32 slot = static_cast<u32>(g_bb.mvm_no % g_bb.mvm_buf.size());
    if (g_bb.mvm_no >= g_bb.mvm_buf.size()) {
        vmaInvalidateAllocation(g_bb.allocator, g_bb.mvm_alloc[slot], 0, 64);
        for (u32 i = 0; i < 8; ++i) g_bb.mvm_acc[i] += g_bb.mvm_mapped[slot][i];
    }
    ++g_bb.mvm_no;
    ++g_bb.mvm_runs;
    if (!prev_valid) ++g_bb.mvm_noprev;
    const vk::ImageView obj_view = BB::ObjMotion::ComposedObjView();
    if (!obj_view) ++g_bb.mvm_noobj;
    if (std::chrono::duration<double>(now - g_bb.mvm_t).count() >= 1.0) {
        const auto& a = g_bb.mvm_acc;
        const double all = static_cast<double>(std::max<u64>(a[4], 1));
        double real_sd = -1.0, shown_sd = -1.0;
        BB::FG::TakePacing(real_sd, shown_sd);
        char pacing[128];
        if (shown_sd >= 0.0 || real_sd >= 0.0) {
            std::snprintf(pacing, sizeof(pacing),
                          "pacing sd=%.2fms (all presents, real + generated) real sd=%.2fms",
                          shown_sd, real_sd);
        } else {
            std::snprintf(pacing, sizeof(pacing), "pacing sd=n/a (no frame generation)");
        }
        BB::FG::Printf("[mvmask] cam=%.1f%% obj=%.1f%% flow=%.1f%% conf_avg=%.2f %s | %llu runs "
                       "(%llu without a previous image, %llu without object vectors), blocks "
                       "searched %llu, given flow %llu",
                       100.0 * a[0] / all, 100.0 * a[1] / all, 100.0 * a[2] / all,
                       a[4] ? a[3] / 255.0 / all : 0.0, pacing,
                       static_cast<unsigned long long>(g_bb.mvm_runs),
                       static_cast<unsigned long long>(g_bb.mvm_noprev),
                       static_cast<unsigned long long>(g_bb.mvm_noobj),
                       static_cast<unsigned long long>(a[5]),
                       static_cast<unsigned long long>(a[6]));
        for (auto& v : g_bb.mvm_acc) v = 0;
        g_bb.mvm_runs = g_bb.mvm_noprev = g_bb.mvm_noobj = 0;
        g_bb.mvm_t = now;
    }
    cmd.fillBuffer(g_bb.mvm_buf[slot], 0, 64, 0u);
    if (g_bb.mvm_dummy.layout == vk::ImageLayout::eUndefined) {
        Barrier(cmd, g_bb.mvm_dummy, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        cmd.clearColorImage(g_bb.mvm_dummy.image, vk::ImageLayout::eGeneral,
                            vk::ClearColorValue{.uint32 = std::array<u32, 4>{0u, 0u, 0u, 0u}},
                            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    }
    for (OwnImage* img : {&g_bb.mvm_blocks, &g_bb.mvm_cls}) {
        Barrier(cmd, *img, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);
    }
    const vk::MemoryBarrier2 mem{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mem});
    const vk::DescriptorImageInfo prev_info{{}, lum_prev.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo cur_info{{}, lum_cur.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, g_bb.mv.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo obj_info{{}, obj_view ? obj_view : g_bb.mvm_dummy.view,
                                           vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo blk_info{{}, g_bb.mvm_blocks.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo cls_info{{}, g_bb.mvm_cls.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo rea_info{{}, g_bb.reactive.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo buf_info{g_bb.mvm_buf[slot], 0, 64};
    const std::array<const vk::DescriptorImageInfo*, 7> infos{&prev_info, &cur_info, &mv_info,
                                                              &obj_info,  &blk_info, &cls_info,
                                                              &rea_info};
    std::array<vk::WriteDescriptorSet, 8> writes{};
    for (u32 i = 0; i < 7; ++i) {
        writes[i] = vk::WriteDescriptorSet{.dstBinding = i,
                                           .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage,
                                           .pImageInfo = infos[i]};
    }
    writes[7] = vk::WriteDescriptorSet{.dstBinding = 7,
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .pBufferInfo = &buf_info};
    static const u32 radius =
        static_cast<u32>(std::clamp(EnvF("BB_MVMASK_RADIUS", 6.0f), 2.0f, 16.0f));
    static const float rel = EnvF("BB_MVMASK_GAIN", 0.35f);
    static const float abs_gain = EnvF("BB_MVMASK_MIN", 0.08f);
    static const float tex = EnvF("BB_MVMASK_TEXTURE", 0.004f);
    static const float conf_scale = EnvF("BB_MVMASK_CONF", 0.6f);
    static const float reactive_gain = std::clamp(EnvF("BB_MVMASK_REACTIVE", 1.0f), 0.0f, 1.0f);
    struct {
        u32 size[4];
        float tune[4];
        float tune2[4];
    } push{{render.width, render.height, (prev_valid ? 1u : 0u) | (obj_view ? 2u : 0u), radius},
           {rel, abs_gain, tex, conf_scale},
           {reactive_gain, 0.0f, 0.0f, 0.0f}};
    static_assert(sizeof(push) == 48);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.mvm_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.mvm_layout, 0, writes);
    cmd.pushConstants(g_bb.mvm_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((quarter.width + 7) / 8, (quarter.height + 7) / 8, 1);
    const vk::MemoryBarrier2 between{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &between});
    push.size[2] |= 1u << 8;
    cmd.pushConstants(g_bb.mvm_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    Barrier(cmd, g_bb.mvm_cls, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
}
