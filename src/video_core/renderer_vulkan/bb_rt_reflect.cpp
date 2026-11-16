// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_rt_reflect.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include "common/logging/log.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_rt_mesh_comp.h"
#include "video_core/host_shaders/bb_rt_reflect_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB {

namespace {
// 4x4 inverse (row-major, Gauss-Jordan in double). False when singular.
bool Invert4(const float* m, double out[16]) {
    double a[4][8];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r][c] = m[r * 4 + c];
            a[r][4 + c] = r == c ? 1.0 : 0.0;
        }
    }
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        }
        if (std::fabs(a[piv][c]) < 1e-12) return false;
        if (piv != c) {
            for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[piv][j]);
        }
        const double inv = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] *= inv;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = a[r][4 + c];
    return true;
}

// Row-major (column-vector convention) -> GLSL std140 mat4 (column-major).
void ToGlsl(const double* m, float* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[c * 4 + r] = static_cast<float>(m[r * 4 + c]);
}

struct alignas(16) Params {
    float proj[16];
    float inv_proj[16];
    float view[16];
    u32 size[4];
    float misc[4];
};
static_assert(sizeof(Params) <= 256);

bool EnvOn(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '1';
}
} // namespace

RtReflect::RtReflect(const Vulkan::Instance& instance_) : instance{instance_} {}

RtReflect::~RtReflect() {
    const auto device = instance.GetDevice();
    if (!device) return;
    (void)device.waitIdle();
    FreeSized();
    FreeBuffer(instances);
    FreeBuffer(params);
    FreeBuffer(counters);
    if (queries) device.destroyQueryPool(queries);
    if (mesh_pipe) device.destroyPipeline(mesh_pipe);
    if (ray_pipe) device.destroyPipeline(ray_pipe);
    if (mesh_layout) device.destroyPipelineLayout(mesh_layout);
    if (ray_layout) device.destroyPipelineLayout(ray_layout);
    if (mesh_set) device.destroyDescriptorSetLayout(mesh_set);
    if (ray_set) device.destroyDescriptorSetLayout(ray_set);
    if (nearest) device.destroySampler(nearest);
    if (linear) device.destroySampler(linear);
}

bool RtReflect::MakeBuffer(Buf& b, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host,
                           vk::DeviceSize alignment) {
    const VkBufferCreateInfo bci = vk::BufferCreateInfo{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eShaderDeviceAddress,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (host) {
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    const VkResult r = alignment > 0
                           ? vmaCreateBufferWithAlignment(instance.GetAllocator(), &bci, &aci,
                                                          alignment, &buffer, &b.alloc, &info)
                           : vmaCreateBuffer(instance.GetAllocator(), &bci, &aci, &buffer,
                                             &b.alloc, &info);
    if (r != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "[rt-reflect] buffer of {} bytes: VkResult {}", size, int(r));
        b = {};
        return false;
    }
    b.buffer = buffer;
    b.mapped = info.pMappedData;
    b.size = size;
    b.address = instance.GetDevice().getBufferAddress({.buffer = b.buffer});
    return true;
}

void RtReflect::FreeBuffer(Buf& b) {
    if (b.buffer) vmaDestroyBuffer(instance.GetAllocator(), b.buffer, b.alloc);
    b = {};
}

void RtReflect::FreeSized() {
    const auto device = instance.GetDevice();
    if (blas) device.destroyAccelerationStructureKHR(blas);
    if (tlas) device.destroyAccelerationStructureKHR(tlas);
    blas = tlas = vk::AccelerationStructureKHR{};
    FreeBuffer(verts);
    FreeBuffer(blas_buf);
    FreeBuffer(tlas_buf);
    FreeBuffer(scratch);
    w = h = gw = gh = tris = 0;
}

bool RtReflect::Init() {
    if (tried) return ok;
    tried = true;
    if (!instance.IsRtReflectEnabled()) {
        LOG_WARNING(Render_Vulkan, "[rt-reflect] device has no ray query / acceleration "
                                   "structures: the game's SSR is kept");
        return false;
    }
    debug = EnvOn("BB_RT_REFLECT_DEBUG");
    const auto device = instance.GetDevice();
    const auto sampler = [&](vk::Filter f, vk::Sampler& out) {
        auto [r, s] = device.createSampler({
            .magFilter = f,
            .minFilter = f,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxLod = 0.0f,
        });
        out = s;
        return r == vk::Result::eSuccess;
    };
    if (!sampler(vk::Filter::eNearest, nearest) || !sampler(vk::Filter::eLinear, linear)) {
        return false;
    }
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    const auto make = [&](std::span<const vk::DescriptorSetLayoutBinding> b,
                          vk::DescriptorSetLayout& set, vk::PipelineLayout& layout) {
        auto [lr, l] = device.createDescriptorSetLayout({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
            .bindingCount = static_cast<u32>(b.size()),
            .pBindings = b.data(),
        });
        if (lr != vk::Result::eSuccess) return false;
        set = l;
        auto [pr, pl] = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &set});
        if (pr != vk::Result::eSuccess) return false;
        layout = pl;
        return true;
    };
    const std::array<vk::DescriptorSetLayoutBinding, 3> mb{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eStorageBuffer, 1, C},
        {2, vk::DescriptorType::eUniformBuffer, 1, C},
    }};
    const std::array<vk::DescriptorSetLayoutBinding, 7> rb{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {3, vk::DescriptorType::eStorageImage, 1, C},
        {4, vk::DescriptorType::eAccelerationStructureKHR, 1, C},
        {5, vk::DescriptorType::eStorageBuffer, 1, C},
        {6, vk::DescriptorType::eUniformBuffer, 1, C},
    }};
    if (!make(mb, mesh_set, mesh_layout) || !make(rb, ray_set, ray_layout)) {
        return false;
    }
    const auto pipeline = [&](vk::ShaderModule module, vk::PipelineLayout layout,
                              vk::Pipeline& pipe) {
        if (!module) return false;
        auto [cr, p] = device.createComputePipeline(
            {}, vk::ComputePipelineCreateInfo{
                    .stage{.stage = C, .module = module, .pName = "main"},
                    .layout = layout,
                });
        device.destroyShaderModule(module);
        if (cr != vk::Result::eSuccess) return false;
        pipe = p;
        return true;
    };
    if (!pipeline(Vulkan::CompileSPV(BB_RT_MESH_COMP, device), mesh_layout, mesh_pipe) ||
        !pipeline(Vulkan::CompileSPV(BB_RT_REFLECT_COMP, device), ray_layout, ray_pipe)) {
        LOG_ERROR(Render_Vulkan, "[rt-reflect] compute pipelines failed");
        return false;
    }
    using U = vk::BufferUsageFlagBits;
    if (!MakeBuffer(instances, sizeof(vk::AccelerationStructureInstanceKHR),
                    U::eAccelerationStructureBuildInputReadOnlyKHR, true) ||
        !MakeBuffer(params, ParamSize * Ring, U::eUniformBuffer, true) ||
        !MakeBuffer(counters, 64, U::eStorageBuffer, true)) {
        return false;
    }
    std::memset(counters.mapped, 0, 64);
    auto [qr, qp] = device.createQueryPool({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Ring * 2,
    });
    if (qr == vk::Result::eSuccess) {
        queries = qp;
    }
    ok = true;
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] ray-traced reflections ON (depth mesh {}x{} px cells, debug tint {})",
             Cell, Cell, debug ? "on" : "off");
    return true;
}

bool RtReflect::EnsureSize(u32 width, u32 height) {
    if (width == w && height == h && blas) return true;
    const auto device = instance.GetDevice();
    if (w != 0) {
        (void)device.waitIdle(); // a resize only
    }
    FreeSized();
    w = width;
    h = height;
    gw = (w + Cell - 1) / Cell;
    gh = (h + Cell - 1) / Cell;
    tris = gw * gh * 2;
    using U = vk::BufferUsageFlagBits;
    if (!MakeBuffer(verts, vk::DeviceSize(tris) * 3 * 3 * sizeof(float),
                    U::eStorageBuffer | U::eAccelerationStructureBuildInputReadOnlyKHR, false)) {
        return false;
    }
    // BLAS sizes.
    const vk::AccelerationStructureGeometryKHR geom{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry{.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                      .vertexFormat = vk::Format::eR32G32B32Sfloat,
                      .vertexData{.deviceAddress = verts.address},
                      .vertexStride = 3 * sizeof(float),
                      .maxVertex = tris * 3 - 1,
                      .indexType = vk::IndexType::eNoneKHR,
                  }},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR binfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geom,
    };
    const auto bsz = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, binfo, tris);
    const vk::AccelerationStructureGeometryKHR tgeom{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry{.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                      .data{.deviceAddress = instances.address}}},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR tinfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &tgeom,
    };
    const auto tsz = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tinfo, 1u);
    const auto pc = instance.GetPhysicalDevice()
                        .getProperties2<vk::PhysicalDeviceProperties2,
                                        vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    const vk::DeviceSize align = std::max<vk::DeviceSize>(
        pc.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
            .minAccelerationStructureScratchOffsetAlignment,
        256);
    blas_scratch = 0;
    tlas_scratch = (bsz.buildScratchSize + align - 1) / align * align;
    if (!MakeBuffer(blas_buf, bsz.accelerationStructureSize,
                    U::eAccelerationStructureStorageKHR, false) ||
        !MakeBuffer(tlas_buf, tsz.accelerationStructureSize,
                    U::eAccelerationStructureStorageKHR, false) ||
        !MakeBuffer(scratch, tlas_scratch + tsz.buildScratchSize, U::eStorageBuffer, false,
                    align)) {
        FreeSized();
        return false;
    }
    auto [br, b] = device.createAccelerationStructureKHR({
        .buffer = blas_buf.buffer,
        .size = bsz.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
    });
    auto [tr, t] = device.createAccelerationStructureKHR({
        .buffer = tlas_buf.buffer,
        .size = tsz.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
    });
    if (br != vk::Result::eSuccess || tr != vk::Result::eSuccess) {
        if (br == vk::Result::eSuccess) device.destroyAccelerationStructureKHR(b);
        if (tr == vk::Result::eSuccess) device.destroyAccelerationStructureKHR(t);
        FreeSized();
        return false;
    }
    blas = b;
    tlas = t;
    // The one instance: identity transform (the mesh is in view space).
    vk::AccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1.0f;
    inst.mask = 0xFF;
    inst.flags = static_cast<VkGeometryInstanceFlagsKHR>(
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable |
        vk::GeometryInstanceFlagBitsKHR::eForceOpaque);
    inst.accelerationStructureReference = device.getAccelerationStructureAddressKHR({
        .accelerationStructure = blas,
    });
    std::memcpy(instances.mapped, &inst, sizeof(inst));
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] depth mesh {}x{}: {}x{} cells, {} triangles, BLAS {} KiB, TLAS {} KiB, "
             "scratch {} KiB",
             w, h, gw, gh, tris, bsz.accelerationStructureSize / 1024,
             tsz.accelerationStructureSize / 1024, scratch.size / 1024);
    return true;
}

bool RtReflect::Run(vk::CommandBuffer cmd, const Input& in) {
    if (!Init()) return false;
    float proj[16], view[12], far_dist = 3000.0f;
    if (!BB::Upscale::RtCamera(proj, view, far_dist)) {
        return false;
    }
    double inv[16];
    if (!Invert4(proj, inv)) {
        return false;
    }
    if (!EnsureSize(in.width, in.height)) {
        return false;
    }
    const auto device = instance.GetDevice();

    // Parameters (ring slot).
    const u32 slot = ring_pos++ % Ring;
    Params prm{};
    double pd[16], vd[16]{};
    for (int i = 0; i < 16; ++i) pd[i] = proj[i];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) vd[r * 4 + c] = view[r * 4 + c];
    vd[15] = 1.0;
    ToGlsl(pd, prm.proj);
    ToGlsl(inv, prm.inv_proj);
    ToGlsl(vd, prm.view);
    prm.size[0] = w;
    prm.size[1] = h;
    prm.size[2] = gw;
    prm.size[3] = gh;
    // Assumption (logged once): depth is NDC z in [0, 1] (no remap) and NDC y points up
    // (row 0 of the image is ndc.y = +1, D3D/GNM convention).
    prm.misc[0] = far_dist > 1.0f ? far_dist : 3000.0f;
    prm.misc[1] = 1.0f;
    prm.misc[2] = float(Cell);
    prm.misc[3] = debug ? 1.0f : 0.0f;
    std::memcpy(static_cast<u8*>(params.mapped) + slot * ParamSize, &prm, sizeof(prm));
    if (!logged_camera) {
        logged_camera = true;
        // Near / far from the projection: view z where ndc z = 0 and 1 (column vectors).
        const auto zat = [&](double nz) {
            // Solve (p2 . v) / (p3 . v) = nz along the view axis v = (0, 0, z, 1).
            const double a = proj[10] - nz * proj[14];
            const double b = proj[11] - nz * proj[15];
            return std::fabs(a) > 1e-12 ? -b / a : 0.0;
        };
        LOG_INFO(Render_Vulkan,
                 "[rt-reflect] camera: proj rows [{:.4f} {:.4f} {:.4f} {:.4f}] [{:.4f} {:.4f} "
                 "{:.4f} {:.4f}] [{:.4f} {:.4f} {:.4f} {:.4f}] [{:.4f} {:.4f} {:.4f} {:.4f}], "
                 "fov_y {:.1f} deg, view z at depth 0: {:.3f}, at depth 1: {:.3f}, far {} "
                 "(assumes depth = ndc z in [0,1], ndc y up)",
                 proj[0], proj[1], proj[2], proj[3], proj[4], proj[5], proj[6], proj[7], proj[8],
                 proj[9], proj[10], proj[11], proj[12], proj[13], proj[14], proj[15],
                 proj[5] != 0.0f ? 2.0 * std::atan(1.0 / std::fabs(proj[5])) * 57.29578 : 0.0,
                 zat(0.0), zat(1.0), prm.misc[0]);
    }

    // GPU time of the previous use of this slot.
    if (queries && query_used[slot]) {
        std::array<u64, 2> ts{};
        if (device.getQueryPoolResults(queries, slot * 2, 2, sizeof(ts), ts.data(), sizeof(u64),
                                       vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
            const double period = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
            const double ms = double(ts[1] - ts[0]) * period * 1e-6;
            if (ms > 0.0 && ms < 1000.0) {
                gpu_ms_sum += ms;
                ++gpu_ms_n;
            }
        }
    }
    if (queries) {
        cmd.resetQueryPool(queries, slot * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queries, slot * 2);
        query_used[slot] = true;
    }

    // Everything before (the SSR draw, the previous pass's AS reads / builds) -> this pass.
    const vk::MemoryBarrier2 all_before{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    // Layout changes: the SSR target to GENERAL (storage); inputs only when not sampleable.
    const auto sampleable = [](vk::ImageLayout l) {
        return l == vk::ImageLayout::eGeneral || l == vk::ImageLayout::eShaderReadOnlyOptimal ||
               l == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
               l == vk::ImageLayout::eDepthReadOnlyOptimal ||
               l == vk::ImageLayout::eReadOnlyOptimal;
    };
    struct Tr {
        vk::Image image;
        vk::ImageLayout from;
        vk::ImageAspectFlags aspect;
    };
    std::array<Tr, 4> trs{};
    u32 ntr = 0;
    const auto depth_aspect = in.depth_has_stencil
                                  ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                                  : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth};
    if (in.target_layout != vk::ImageLayout::eGeneral) {
        trs[ntr++] = {in.target, in.target_layout, vk::ImageAspectFlagBits::eColor};
    }
    if (!sampleable(in.depth_layout)) trs[ntr++] = {in.depth, in.depth_layout, depth_aspect};
    if (!sampleable(in.normal_layout)) {
        trs[ntr++] = {in.normal, in.normal_layout, vk::ImageAspectFlagBits::eColor};
    }
    if (!sampleable(in.color_layout)) {
        trs[ntr++] = {in.color, in.color_layout, vk::ImageAspectFlagBits::eColor};
    }
    const auto layouts = [&](bool to_general) {
        std::array<vk::ImageMemoryBarrier2, 4> b{};
        for (u32 i = 0; i < ntr; ++i) {
            b[i] = vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .oldLayout = to_general ? trs[i].from : vk::ImageLayout::eGeneral,
                .newLayout = to_general ? vk::ImageLayout::eGeneral : trs[i].from,
                .image = trs[i].image,
                .subresourceRange{trs[i].aspect, 0, 1, 0, 1},
            };
        }
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = to_general ? 1u : 0u,
            .pMemoryBarriers = &all_before,
            .imageMemoryBarrierCount = ntr,
            .pImageMemoryBarriers = b.data(),
        });
    };
    layouts(true);
    const auto lay = [&](vk::ImageLayout l) {
        return sampleable(l) ? l : vk::ImageLayout::eGeneral;
    };

    // 1. Depth mesh.
    const vk::DescriptorImageInfo depth_info{nearest, in.depth_view, lay(in.depth_layout)};
    const vk::DescriptorBufferInfo verts_info{verts.buffer, 0, verts.size};
    const vk::DescriptorBufferInfo params_info{params.buffer, slot * ParamSize, ParamSize};
    {
        const std::array<vk::WriteDescriptorSet, 3> wr{{
            {.dstBinding = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &depth_info},
            {.dstBinding = 1, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &verts_info},
            {.dstBinding = 2, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &params_info},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, mesh_pipe);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, mesh_layout, 0, wr);
        cmd.dispatch((gw + 7) / 8, (gh + 7) / 8, 1);
    }
    const vk::MemoryBarrier2 to_build{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                         vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &to_build});

    // 2. BLAS (rebuilt), then the TLAS.
    {
        const vk::AccelerationStructureGeometryKHR geom{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry{.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                          .vertexFormat = vk::Format::eR32G32B32Sfloat,
                          .vertexData{.deviceAddress = verts.address},
                          .vertexStride = 3 * sizeof(float),
                          .maxVertex = tris * 3 - 1,
                          .indexType = vk::IndexType::eNoneKHR,
                      }},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR binfo{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = blas,
            .geometryCount = 1,
            .pGeometries = &geom,
            .scratchData{.deviceAddress = scratch.address + blas_scratch},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR brange{.primitiveCount = tris};
        const vk::AccelerationStructureBuildRangeInfoKHR* pbr = &brange;
        cmd.buildAccelerationStructuresKHR(1, &binfo, &pbr);
    }
    const vk::MemoryBarrier2 blas_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &blas_done});
    {
        const vk::AccelerationStructureGeometryKHR tgeom{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry{.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                          .data{.deviceAddress = instances.address}}},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR tinfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = tlas,
            .geometryCount = 1,
            .pGeometries = &tgeom,
            .scratchData{.deviceAddress = scratch.address + tlas_scratch},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR trange{.primitiveCount = 1};
        const vk::AccelerationStructureBuildRangeInfoKHR* ptr = &trange;
        cmd.buildAccelerationStructuresKHR(1, &tinfo, &ptr);
    }
    const vk::MemoryBarrier2 tlas_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &tlas_done});

    // 3. One ray per SSR pixel.
    {
        const vk::DescriptorImageInfo normal_info{nearest, in.normal_view, lay(in.normal_layout)};
        const vk::DescriptorImageInfo color_info{linear, in.color_view, lay(in.color_layout)};
        const vk::DescriptorImageInfo target_info{{}, in.target_view, vk::ImageLayout::eGeneral};
        const vk::DescriptorBufferInfo counters_info{counters.buffer, 0, counters.size};
        const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas,
        };
        const std::array<vk::WriteDescriptorSet, 7> wr{{
            {.dstBinding = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &depth_info},
            {.dstBinding = 1, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &normal_info},
            {.dstBinding = 2, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &color_info},
            {.dstBinding = 3, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &target_info},
            {.pNext = &as_info, .dstBinding = 4, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
            {.dstBinding = 5, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &counters_info},
            {.dstBinding = 6, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &params_info},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ray_pipe);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, ray_layout, 0, wr);
        cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
    if (queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, queries, slot * 2 + 1);
    }
    // Back to the layouts the texture cache tracks; the writes visible to whatever comes next.
    layouts(false);
    if (ntr == 0) {
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                .pMemoryBarriers = &all_before});
    }
    ++dispatches;
    Log();
    return true;
}

void RtReflect::Log() {
    const auto now = std::chrono::steady_clock::now();
    if (last_log != std::chrono::steady_clock::time_point{} &&
        now - last_log < std::chrono::seconds(5)) {
        return;
    }
    last_log = now;
    // Counters written by the GPU (cumulative, wrapping): the last 5 s by difference.
    std::array<u32, 5> c{};
    std::memcpy(c.data(), counters.mapped, sizeof(c));
    std::array<u32, 5> d{};
    for (int i = 0; i < 5; ++i) d[i] = c[i] - last_counts[i];
    last_counts = c;
    const double valid = std::max<double>(d[1], 1.0);
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] 5 s: {} dispatches, {} px, {:.1f}% with depth, hit {:.1f}% of those, "
             "normal: G-buffer RT0 {:.1f}% / depth derivatives {:.1f}%, GPU {:.3f} ms per pass{}",
             dispatches - last_dispatches, d[0], 100.0 * d[1] / std::max<double>(d[0], 1.0),
             100.0 * d[2] / valid, 100.0 * d[3] / valid, 100.0 * d[4] / valid,
             gpu_ms_n ? gpu_ms_sum / gpu_ms_n : 0.0, debug ? " (debug tint)" : "");
    last_dispatches = dispatches;
    gpu_ms_sum = 0.0;
    gpu_ms_n = 0;
}

} // namespace BB
