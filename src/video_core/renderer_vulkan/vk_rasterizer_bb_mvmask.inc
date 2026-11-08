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
                           "+ 1.58 hierarchical flow (quarter-res coarse forward / backward search, "
                           "full-res refinement with spatial predictors, sub-pixel, forward-backward "
                           "and occlusion checks, unjittered) where those do not explain the image; "
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
    std::array<vk::DescriptorSetLayoutBinding, 11> bindings{};
    for (u32 i = 0; i < 11; ++i) {
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
    {
        auto [qr, pool] = device.createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = 8,
        });
        g_bb.mvm_queries = qr == vk::Result::eSuccess ? pool : vk::QueryPool{};
    }
    g_bb.mvm_set_layout = set_layout;
    g_bb.mvm_layout = layout;
    g_bb.mvm_pipeline = pipeline;
    g_bb.mvm_failed = false;
    return true;
}

void BBMvMask(vk::CommandBuffer cmd, OwnImage& lum_prev, OwnImage& lum_cur, vk::Extent2D render,
              bool prev_valid, const float* jitter) {
    if (!MvMaskWanted() || !InitMvMask()) return;
    const vk::Extent2D quarter{(render.width + 3) / 4, (render.height + 3) / 4};
    const auto storage = vk::ImageUsageFlagBits::eStorage;
    if (!EnsureImage(g_bb.mvm_blocks, quarter, vk::Format::eR16G16B16A16Sfloat, storage) ||
        !EnsureImage(g_bb.mvm_cls, render, vk::Format::eR16Uint,
                     storage | vk::ImageUsageFlagBits::eTransferSrc |
                         vk::ImageUsageFlagBits::eSampled) ||
        !EnsureImage(g_bb.mvm_dummy, {1, 1}, vk::Format::eR32G32Uint,
                     storage | vk::ImageUsageFlagBits::eTransferDst) ||
        !EnsureImage(g_bb.mvm_qprev, quarter, vk::Format::eR16Sfloat, storage) ||
        !EnsureImage(g_bb.mvm_qcur, quarter, vk::Format::eR16Sfloat, storage) ||
        !EnsureImage(g_bb.mvm_coarse, quarter, vk::Format::eR16G16B16A16Sfloat, storage)) {
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
        for (u32 i = 0; i < 12; ++i) g_bb.mvm_acc[i] += g_bb.mvm_mapped[slot][i];
    }
    if (g_bb.mvm_queries && g_bb.mvm_q_written[slot]) {
        std::array<u64, 4> data{};
        const auto result = g_bb.device.getQueryPoolResults(
            g_bb.mvm_queries, slot * 2, 2, sizeof(data), data.data(), sizeof(u64) * 2,
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if ((result == vk::Result::eSuccess || result == vk::Result::eNotReady) && data[1] != 0 &&
            data[3] != 0 && data[2] >= data[0]) {
            const double ms =
                static_cast<double>(data[2] - data[0]) * g_bb.timestamp_period_ns / 1e6;
            g_bb.mvm_gpu_sum += ms;
            g_bb.mvm_gpu_max = std::max(g_bb.mvm_gpu_max, ms);
            ++g_bb.mvm_gpu_n;
        }
        g_bb.mvm_q_written[slot] = false;
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
        BB::FG::Printf("[mvmask] 1.58 flow: blocks given flow %llu (%llu from the coarse level, "
                       "mean |flow - camera/object| %.2f px), rejected forward-backward %llu, "
                       "occluded / poor match %llu, pixels from a neighbour block %.2f%%; jitter "
                       "taken out %.3f %.3f px; radius %u px; GPU %.3f ms avg, %.3f max (%u frames, "
                       "render %ux%u)",
                       static_cast<unsigned long long>(a[6]),
                       static_cast<unsigned long long>(a[11]),
                       a[6] ? a[9] / 16.0 / a[6] : 0.0, static_cast<unsigned long long>(a[7]),
                       static_cast<unsigned long long>(a[8]), 100.0 * a[10] / all,
                       g_bb.mvm_last_dj[0], g_bb.mvm_last_dj[1], g_bb.mvm_radius,
                       g_bb.mvm_gpu_n ? g_bb.mvm_gpu_sum / g_bb.mvm_gpu_n : 0.0, g_bb.mvm_gpu_max,
                       g_bb.mvm_gpu_n, render.width, render.height);
        g_bb.mvm_gpu_sum = g_bb.mvm_gpu_max = 0.0;
        g_bb.mvm_gpu_n = 0;
        for (auto& v : g_bb.mvm_acc) v = 0;
        g_bb.mvm_runs = g_bb.mvm_noprev = g_bb.mvm_noobj = 0;
        g_bb.mvm_t = now;
    }
    cmd.fillBuffer(g_bb.mvm_buf[slot], 0, 64, 0u);
    if (g_bb.mvm_queries) {
        cmd.resetQueryPool(g_bb.mvm_queries, slot * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_bb.mvm_queries, slot * 2);
    }
    if (g_bb.mvm_dummy.layout == vk::ImageLayout::eUndefined) {
        Barrier(cmd, g_bb.mvm_dummy, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        cmd.clearColorImage(g_bb.mvm_dummy.image, vk::ImageLayout::eGeneral,
                            vk::ClearColorValue{.uint32 = std::array<u32, 4>{0u, 0u, 0u, 0u}},
                            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    }
    for (OwnImage* img : {&g_bb.mvm_blocks, &g_bb.mvm_cls, &g_bb.mvm_qprev, &g_bb.mvm_qcur,
                          &g_bb.mvm_coarse}) {
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
    const vk::DescriptorImageInfo qp_info{{}, g_bb.mvm_qprev.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo qc_info{{}, g_bb.mvm_qcur.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo co_info{{}, g_bb.mvm_coarse.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo buf_info{g_bb.mvm_buf[slot], 0, 64};
    const std::array<const vk::DescriptorImageInfo*, 10> infos{
        &prev_info, &cur_info, &mv_info, &obj_info, &blk_info, &cls_info, &rea_info,
        &qp_info,   &qc_info,  &co_info};
    std::array<vk::WriteDescriptorSet, 11> writes{};
    for (u32 i = 0; i < 10; ++i) {
        const u32 binding = i < 7 ? i : i + 1;
        writes[i] = vk::WriteDescriptorSet{.dstBinding = binding,
                                           .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage,
                                           .pImageInfo = infos[i]};
    }
    writes[10] = vk::WriteDescriptorSet{.dstBinding = 7,
                                        .descriptorCount = 1,
                                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                                        .pBufferInfo = &buf_info};
    // 1.58: the coarse level searches +-radius / 4 cells (default 32 px: a limb or a particle at
    // 60 fps moving ~2000 px/s at 900p), the full-res refinement +-2 px around its candidates.
    static const u32 radius =
        static_cast<u32>(std::clamp(EnvF("BB_MVMASK_RADIUS", 32.0f), 4.0f, 64.0f));
    g_bb.mvm_radius = radius;
    static const float smooth = EnvF("BB_MVMASK_SMOOTH", 0.25f);
    // Image displacement = unjittered vector + (jitter_prev - jitter_cur) (bb_mv.comp's sample
    // point is pixel + 0.5 - jitter). BB_MVMASK_JITTER=0: not taken out (1.54).
    static const bool unjitter = EnvF("BB_MVMASK_JITTER", 1.0f) != 0.0f;
    float dj[2]{0.0f, 0.0f};
    if (unjitter && jitter != nullptr && g_bb.mvm_have_jitter && prev_valid) {
        dj[0] = g_bb.mvm_jitter_prev[0] - jitter[0];
        dj[1] = g_bb.mvm_jitter_prev[1] - jitter[1];
    }
    if (jitter != nullptr) {
        g_bb.mvm_jitter_prev[0] = jitter[0];
        g_bb.mvm_jitter_prev[1] = jitter[1];
        g_bb.mvm_have_jitter = true;
    }
    g_bb.mvm_last_dj[0] = dj[0];
    g_bb.mvm_last_dj[1] = dj[1];
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
           {reactive_gain, dj[0], dj[1], smooth}};
    static_assert(sizeof(push) == 48);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g_bb.mvm_pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bb.mvm_layout, 0, writes);
    const vk::MemoryBarrier2 between{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    const u32 flags = push.size[2];
    if (prev_valid) {
        // 1.58: mode 2 (pyramid level), mode 3 (coarse forward / backward search).
        for (u32 mode : {2u, 3u}) {
            push.size[2] = flags | (mode << 8);
            cmd.pushConstants(g_bb.mvm_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                              &push);
            cmd.dispatch((quarter.width + 7) / 8, (quarter.height + 7) / 8, 1);
            cmd.pipelineBarrier2(
                vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &between});
        }
    }
    push.size[2] = flags; // mode 0: blocks
    cmd.pushConstants(g_bb.mvm_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((quarter.width + 7) / 8, (quarter.height + 7) / 8, 1);
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &between});
    push.size[2] = flags | (1u << 8);
    cmd.pushConstants(g_bb.mvm_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    if (g_bb.mvm_queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g_bb.mvm_queries,
                            slot * 2 + 1);
        g_bb.mvm_q_written[slot] = true;
    }
    Barrier(cmd, g_bb.mvm_cls, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
}
