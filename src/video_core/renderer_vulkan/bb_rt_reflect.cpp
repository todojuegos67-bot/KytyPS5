// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/bb_rt_reflect.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include "common/logging/log.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_rt_mesh_comp.h"
#include "video_core/host_shaders/bb_rt_post_comp.h"
#include "video_core/host_shaders/bb_rt_reflect_comp.h"
#include "video_core/host_shaders/bb_rt_vol_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB {

namespace {
// 4x4 inverse (row-major, Gauss-Jordan in double). False when singular.
template <typename T>
bool Invert4(const T* m, double out[16]) {
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

// Row-major 4x4 product a * b.
void Mul4(const double* a, const double* b, double* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) v += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = v;
        }
}

struct alignas(16) Params {
    float proj[16];
    float inv_proj[16];
    float view[16];
    float reproj[16];    // current view -> previous view
    float prev_proj[16];
    u32 size[4];
    float misc[4];
    float misc2[4];      // history valid, frame, rays, flags (1 shine, 2 more)
    float misc3[4];      // temporal blend, default roughness, filter strength
    float misc4[4];      // strength, hit refinement
};
static_assert(sizeof(Params) <= 512);

struct alignas(16) VolParams {
    float proj[16];
    float inv_proj[16];
    float reproj[16];
    float prev_proj[16];
    u32 size[4];
    float light[4];
    float color[4];
    float misc[4];  // density, max march distance, steps, frame
    float misc2[4]; // history valid, temporal blend, far distance, y sign
};
static_assert(sizeof(VolParams) <= 512);

float EnvFloat(const char* name, float def, float lo, float hi) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return def;
    char* end = nullptr;
    const float v = std::strtof(e, &end);
    return end != e && std::isfinite(v) ? std::clamp(v, lo, hi) : def;
}

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
    FreeBuffer(vol_params);
    if (vol_pipe) device.destroyPipeline(vol_pipe);
    if (vol_layout) device.destroyPipelineLayout(vol_layout);
    if (vol_set) device.destroyDescriptorSetLayout(vol_set);
    if (vol_queries) device.destroyQueryPool(vol_queries);
    if (post_pipe) device.destroyPipeline(post_pipe);
    if (post_layout) device.destroyPipelineLayout(post_layout);
    if (post_set) device.destroyDescriptorSetLayout(post_set);
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

bool RtReflect::MakeImage(Img& i, u32 width, u32 height) {
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {width, height, 1},
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
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &i.alloc, nullptr) !=
        VK_SUCCESS) {
        i = {};
        return false;
    }
    i.image = image;
    auto [r, v] = instance.GetDevice().createImageView({
        .image = i.image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) {
        FreeImage(i);
        return false;
    }
    i.view = v;
    return true;
}

void RtReflect::FreeImage(Img& i) {
    if (i.view) instance.GetDevice().destroyImageView(i.view);
    if (i.image) vmaDestroyImage(instance.GetAllocator(), i.image, i.alloc);
    i = {};
}

void RtReflect::FreeSized() {
    const auto device = instance.GetDevice();
    FreeImage(raw);
    FreeImage(tmp);
    FreeImage(full);
    for (auto& i : guide) FreeImage(i);
    for (auto& i : hist) FreeImage(i);
    FreeImage(vraw);
    for (auto& i : vacc) FreeImage(i);
    vol_w = vol_h = 0;
    vol_fresh = true;
    have_prev = false;
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
    scale = EnvOn("BB_RT_REFLECT_FULLRES") ? 2u : 1u;
    more = EnvOn("BB_RT_REFLECT_MORE");
    strength = EnvFloat("BB_RT_REFLECT_STRENGTH", 1.5f, 0.0f, 4.0f);
    if (const char* e = std::getenv("BB_RT_REFLECT_REFINE"); e && e[0] == '0') {
        refine = false;
    }
    if (const char* q = std::getenv("BB_RT_REFLECT_QUALITY"); q && q[0] >= '1' && q[0] <= '3') {
        quality = u32(q[0] - '0');
    }
    if (const char* e = std::getenv("BB_RT_REFLECT_SHINE"); e && e[0] == '0') {
        use_shine = false;
    }
    // 1: 4x4 px cells, 1 ray; 2: 2x2 px cells, 2 rays; 3: 2x2 px cells, 4 rays.
    Cell = quality == 1 ? 4 : 2;
    rays = quality == 1 ? 1 : quality == 2 ? 2 : 4;
    temporal_blend = quality == 1 ? 0.2f : quality == 2 ? 0.12f : 0.08f;
    filter_strength = quality == 1 ? 0.7f : 1.0f;
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
    const std::array<vk::DescriptorSetLayoutBinding, 10> rb{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {3, vk::DescriptorType::eStorageImage, 1, C},
        {4, vk::DescriptorType::eAccelerationStructureKHR, 1, C},
        {5, vk::DescriptorType::eStorageBuffer, 1, C},
        {6, vk::DescriptorType::eUniformBuffer, 1, C},
        {7, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {8, vk::DescriptorType::eStorageImage, 1, C},
        {9, vk::DescriptorType::eStorageImage, 1, C},
    }};
    if (!make(mb, mesh_set, mesh_layout) || !make(rb, ray_set, ray_layout)) {
        return false;
    }
    {
        const std::array<vk::DescriptorSetLayoutBinding, 11> pb{{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {1, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {2, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {3, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {4, vk::DescriptorType::eStorageImage, 1, C},
            {5, vk::DescriptorType::eStorageImage, 1, C},
            {6, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {7, vk::DescriptorType::eUniformBuffer, 1, C},
            {8, vk::DescriptorType::eCombinedImageSampler, 1, C},
            {9, vk::DescriptorType::eStorageBuffer, 1, C},
            {10, vk::DescriptorType::eStorageImage, 1, C},
        }};
        auto [lr, l] = device.createDescriptorSetLayout({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
            .bindingCount = static_cast<u32>(pb.size()),
            .pBindings = pb.data(),
        });
        if (lr != vk::Result::eSuccess) return false;
        post_set = l;
        const vk::PushConstantRange range{C, 0, 4};
        auto [pr, pl] = device.createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &post_set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
        if (pr != vk::Result::eSuccess) return false;
        post_layout = pl;
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
        !pipeline(Vulkan::CompileSPV(BB_RT_REFLECT_COMP, device), ray_layout, ray_pipe) ||
        !pipeline(Vulkan::CompileSPV(BB_RT_POST_COMP, device), post_layout, post_pipe)) {
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
             "[rt-reflect] strength {:.2f} (BB_RT_REFLECT_STRENGTH: hit weight x strength, hit "
             "colour x(1 + (strength-1)/2)), depth-buffer hit refinement {} (BB_RT_REFLECT_REFINE), "
             "mirrors (roughness < 0.05): 1 exact ray, no spatial blur, 40% temporal weight",
             strength, refine ? "on" : "off");
    vol_on = EnvOn("BB_RT_VOLUMETRIC");
    if (vol_on && !InitVolumetric()) {
        vol_on = false;
        LOG_ERROR(Render_Vulkan, "[rt-vol] init failed: volumetric light off");
    }
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] ray-traced reflections ON: quality {} (depth mesh {}x{} px cells, {} "
             "rays/px, temporal blend {:.2f}, 2-pass bilateral x{:.1f}), more {}, shininess {}, "
             "debug tint {}, trace {}",
             quality, Cell, Cell, rays, temporal_blend, filter_strength, more ? "on" : "off",
             use_shine ? "if bound" : "off", debug ? "on" : "off",
             scale > 1 ? "full-res (BB_RT_REFLECT_FULLRES=1, own image fed to PS 0x3fe9e964)"
                       : "SSR size");
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
                    U::eStorageBuffer | U::eAccelerationStructureBuildInputReadOnlyKHR, false) ||
        !MakeImage(raw, w * scale, h * scale) || !MakeImage(tmp, w * scale, h * scale) ||
        !MakeImage(guide[0], w * scale, h * scale) || !MakeImage(guide[1], w * scale, h * scale) ||
        !MakeImage(hist[0], w * scale, h * scale) || !MakeImage(hist[1], w * scale, h * scale) ||
        (scale > 1 && !MakeImage(full, w * scale, h * scale))) {
        FreeSized();
        return false;
    }
    images_fresh = true;
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
    // Temporal reprojection: current view -> world -> previous view.
    double inv_view[16];
    double rp[16]{};
    const bool cam_prev_ok = have_prev && Invert4(vd, inv_view);
    const bool hist_ok = cam_prev_ok && !images_fresh;
    if (cam_prev_ok) {
        Mul4(prev_view, inv_view, rp);
    }
    double vol_prev_proj[16];
    std::memcpy(vol_prev_proj, prev_proj, sizeof(vol_prev_proj));
    if (hist_ok) {
        ToGlsl(rp, prm.reproj);
        ToGlsl(prev_proj, prm.prev_proj);
    }
    const bool shine_on = use_shine && in.shine && in.shine_view;
    if (!logged_shine) {
        logged_shine = true;
        LOG_INFO(Render_Vulkan, "[rt-reflect] G-buffer RT1 (shininess): {}",
                 shine_on ? "bound, used for roughness / blend"
                          : (use_shine ? "not among the SSR draw's images, default roughness"
                                       : "disabled (BB_RT_REFLECT_SHINE=0)"));
    }
    const u32 cur = u32(frame & 1), prv = cur ^ 1u;
    prm.misc2[0] = hist_ok ? 1.0f : 0.0f;
    prm.misc2[1] = float(frame & 0xffff);
    prm.misc2[2] = float(rays);
    prm.misc2[3] = float((shine_on ? 1u : 0u) | (more ? 2u : 0u));
    prm.misc3[0] = temporal_blend;
    prm.misc3[1] = 0.12f; // default roughness without RT1
    prm.misc3[2] = filter_strength;
    prm.misc3[3] = float(scale);
    prm.misc4[0] = strength;
    prm.misc4[1] = refine ? 1.0f : 0.0f;
    std::memcpy(prev_view, vd, sizeof(prev_view));
    std::memcpy(prev_proj, pd, sizeof(prev_proj));
    have_prev = true;
    ++frame;
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
    std::array<Tr, 5> trs{};
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
    const bool vol_run = vol_on && vol_ok && in.color_storage_view;
    const bool color_general = vol_run || !sampleable(in.color_layout);
    if (color_general && in.color_layout != vk::ImageLayout::eGeneral) {
        trs[ntr++] = {in.color, in.color_layout, vk::ImageAspectFlagBits::eColor};
    }
    if (shine_on && in.shine != in.normal && in.shine != in.color && !sampleable(in.shine_layout)) {
        trs[ntr++] = {in.shine, in.shine_layout, vk::ImageAspectFlagBits::eColor};
    }
    if (images_fresh) {
        // Own images: UNDEFINED -> GENERAL once (they stay GENERAL).
        std::array<vk::ImageMemoryBarrier2, 7> b{};
        const std::array<vk::Image, 7> imgs{raw.image,      tmp.image,      guide[0].image,
                                            guide[1].image, hist[0].image,  hist[1].image,
                                            full.image};
        const u32 nimg = full.image ? 7u : 6u;
        for (u32 i = 0; i < nimg; ++i) {
            b[i] = vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask =
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = imgs[i],
                .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
        }
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = nimg,
                                                .pImageMemoryBarriers = b.data()});
        images_fresh = false;
    }
    const auto layouts = [&](bool to_general) {
        std::array<vk::ImageMemoryBarrier2, 5> b{};
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
        const vk::DescriptorImageInfo color_info{
            linear, in.color_view,
            color_general ? vk::ImageLayout::eGeneral : in.color_layout};
        const vk::DescriptorImageInfo target_info{{}, in.target_view, vk::ImageLayout::eGeneral};
        const vk::DescriptorBufferInfo counters_info{counters.buffer, 0, counters.size};
        const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas,
        };
        const vk::DescriptorImageInfo shine_info{
            nearest, shine_on ? in.shine_view : in.normal_view,
            shine_on ? lay(in.shine_layout) : lay(in.normal_layout)};
        const vk::DescriptorImageInfo raw_info{{}, raw.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo guide_info{{}, guide[cur].view, vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 10> wr{{
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
            {.dstBinding = 7, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &shine_info},
            {.dstBinding = 8, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &raw_info},
            {.dstBinding = 9, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &guide_info},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ray_pipe);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, ray_layout, 0, wr);
        const u32 tw = w * scale, th = h * scale;
        cmd.dispatch((tw + 7) / 8, (th + 7) / 8, 1);

        // 4. Temporal accumulation, then two bilateral passes (the second blends into the target).
        const vk::MemoryBarrier2 cc{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        };
        const vk::DescriptorImageInfo full_info{{}, full.view, vk::ImageLayout::eGeneral};
        const auto post = [&](u32 mode, const Img& src, const Img& dst) {
            // Mode 2 output: the SSR target, or (full-res) our own image; mode 3: the target.
            const vk::DescriptorImageInfo& out_i =
                (mode == 2 && scale > 1) ? full_info : target_info;
            cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                    .pMemoryBarriers = &cc});
            const vk::DescriptorImageInfo src_i{nearest, src.view, vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo hist_i{linear, hist[prv].view, vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo g_i{nearest, guide[cur].view, vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo pg_i{nearest, guide[prv].view, vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo dst_i{{}, dst.view, vk::ImageLayout::eGeneral};
            const std::array<vk::WriteDescriptorSet, 11> pw{{
                {.dstBinding = 0, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &src_i},
                {.dstBinding = 1, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hist_i},
                {.dstBinding = 2, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &g_i},
                {.dstBinding = 3, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &pg_i},
                {.dstBinding = 4, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dst_i},
                {.dstBinding = 5, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &out_i},
                {.dstBinding = 6, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                 .pImageInfo = &shine_info},
                {.dstBinding = 7, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &params_info},
                {.dstBinding = 8, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                 .pImageInfo = &depth_info},
                {.dstBinding = 9, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &counters_info},
                {.dstBinding = 10, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &target_info},
            }};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, post_pipe);
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, post_layout, 0, pw);
            cmd.pushConstants(post_layout, vk::ShaderStageFlagBits::eCompute, 0, 4, &mode);
            const u32 dw = mode == 3 ? w : tw, dh = mode == 3 ? h : th;
            cmd.dispatch((dw + 7) / 8, (dh + 7) / 8, 1);
        };
        post(0, raw, hist[cur]);  // raw + history -> hist[cur] (next frame's history)
        post(1, hist[cur], tmp);  // bilateral step 1
        post(2, tmp, raw);        // bilateral step 2 -> blended into the SSR target (or `full`)
        if (scale > 1) {
            post(3, full, tmp);   // full-res result 2x2-averaged into the SSR target (pyramid)
        }
    }
    if (queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, queries, slot * 2 + 1);
    }
    if (vol_run) {
        Volumetric(cmd, in, proj, inv, vd, rp, vol_prev_proj, cam_prev_ok, depth_info);
    }
    // Back to the layouts the texture cache tracks; the writes visible to whatever comes next.
    layouts(false);
    if (ntr == 0 || scale > 1) {
        // (full-res: our image's writes -> the composite draw's fragment-shader reads)
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
    std::array<u32, 8> c{};
    std::memcpy(c.data(), counters.mapped, sizeof(c));
    std::array<u32, 8> d{};
    for (int i = 0; i < 8; ++i) d[i] = c[i] - last_counts[i];
    last_counts = c;
    const double valid = std::max<double>(d[1], 1.0);
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] 5 s: {} dispatches, {} px, {:.1f}% with depth, hit {:.1f}% of those, "
             "normal: G-buffer RT0 {:.1f}% / depth derivatives {:.1f}%, history reused {:.1f}%, "
             "quality {} ({} rays, {}px cells), trace {}x{}, GPU {:.3f} ms per pass{}",
             dispatches - last_dispatches, d[0], 100.0 * d[1] / std::max<double>(d[0], 1.0),
             100.0 * d[2] / valid, 100.0 * d[3] / valid, 100.0 * d[4] / valid,
             100.0 * d[5] / valid, quality, rays, Cell, w * scale, h * scale,
             gpu_ms_n ? gpu_ms_sum / gpu_ms_n : 0.0, debug ? " (debug tint)" : "");
    LOG_INFO(Render_Vulkan,
             "[rt-reflect] 5 s precision: {} mesh hits refined against the depth buffer, {} "
             "rejected (hidden / fake surface -> game SSR colour), strength {:.2f}",
             d[6], d[7], strength);
    last_dispatches = dispatches;
    gpu_ms_sum = 0.0;
    gpu_ms_n = 0;
}

bool RtReflect::InitVolumetric() {
    const auto device = instance.GetDevice();
    vol_strength = EnvFloat("BB_RT_VOLUMETRIC_STRENGTH", 1.0f, 0.0f, 10.0f);
    vol_density = EnvFloat("BB_RT_VOLUMETRIC_DENSITY", 0.02f, 0.0001f, 1.0f);
    vol_dist = EnvFloat("BB_RT_VOLUMETRIC_DIST", 60.0f, 2.0f, 1000.0f);
    vol_steps = u32(EnvFloat("BB_RT_VOLUMETRIC_STEPS", 12.0f, 2.0f, 64.0f));
    if (const char* e = std::getenv("BB_RT_VOLUMETRIC_DIR"); e && e[0]) {
        float v[3]{};
        if (std::sscanf(e, "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
            const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (l > 1e-4f) {
                for (int i = 0; i < 3; ++i) light_env_dir[i] = v[i] / l;
                light_env = true;
            }
        }
    }
    if (const char* e = std::getenv("BB_RT_VOLUMETRIC_LIGHT_OFS"); e && e[0]) {
        light_forced_ofs = std::clamp(std::atoi(e), 0, 213);
    }
    if (const char* e = std::getenv("BB_RT_VOLUMETRIC_COLOR"); e && e[0]) {
        float v[3]{};
        if (std::sscanf(e, "%f,%f,%f", &v[0], &v[1], &v[2]) == 3) {
            for (int i = 0; i < 3; ++i) vol_color[i] = std::clamp(v[i], 0.0f, 100.0f);
        }
    }
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    const std::array<vk::DescriptorSetLayoutBinding, 8> vb{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {1, vk::DescriptorType::eAccelerationStructureKHR, 1, C},
        {2, vk::DescriptorType::eUniformBuffer, 1, C},
        {3, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {4, vk::DescriptorType::eCombinedImageSampler, 1, C},
        {5, vk::DescriptorType::eStorageImage, 1, C},
        {6, vk::DescriptorType::eStorageImage, 1, C},
        {7, vk::DescriptorType::eStorageBuffer, 1, C},
    }};
    auto [lr, l] = device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(vb.size()),
        .pBindings = vb.data(),
    });
    if (lr != vk::Result::eSuccess) return false;
    vol_set = l;
    const vk::PushConstantRange range{C, 0, 4};
    auto [pr, pl] = device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &vol_set,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) return false;
    vol_layout = pl;
    const auto module = Vulkan::CompileSPV(BB_RT_VOL_COMP, device);
    if (!module) return false;
    auto [cr, pipe] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage{.stage = C, .module = module, .pName = "main"},
                .layout = vol_layout,
            });
    device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) return false;
    vol_pipe = pipe;
    if (!MakeBuffer(vol_params, ParamSize * Ring, vk::BufferUsageFlagBits::eUniformBuffer, true)) {
        return false;
    }
    auto [qr, qp] = device.createQueryPool({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Ring * 2,
    });
    if (qr == vk::Result::eSuccess) vol_queries = qp;
    vol_ok = true;
    LOG_INFO(Render_Vulkan,
             "[rt-vol] ray-traced volumetric light ON: half res (depth size) trace, {} steps, "
             "march {:.0f} units, density {:.4f}, strength {:.2f}, colour ({:.2f}, {:.2f}, {:.2f}), "
             "light dir: {}; temporal + depth-bilateral upsample, added into the lit scene "
             "before the composite",
             vol_steps, vol_dist, vol_density, vol_strength, vol_color[0], vol_color[1],
             vol_color[2],
             light_env ? "BB_RT_VOLUMETRIC_DIR"
             : light_forced_ofs >= 0 ? "BB_RT_VOLUMETRIC_LIGHT_OFS"
                                     : "searched in the scene constant block, else fixed fallback");
    return true;
}

// The main light's direction. The scene constant block (864 bytes, the camera block) is searched
// for a unit vec3 at a vec4 boundary that is not a camera matrix row (rows come in orthogonal
// groups and match the view's axes). This is a heuristic: which float the game really uses for
// the sun / moon is unverified. Fallback: a fixed high moonlight direction.
bool RtReflect::LightDir(const double* view, float out_view[3]) {
    double lw[3] = {-0.3, 0.8, 0.5};
    const char* src = "fallback (fixed world dir -0.3,0.8,0.5; no unit vector in the scene block)";
    float b[216];
    const bool have_block = BB::Upscale::RtSceneBlock(b);
    if (have_block && !logged_block) {
        logged_block = true;
        for (int i = 0; i < 216; i += 8) {
            LOG_INFO(Render_Vulkan,
                     "[rt-vol] scene block [{:3}] {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} "
                     "{:.4f}",
                     i, b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7]);
        }
    }
    const auto unit = [&](int i) {
        if (i < 0 || i + 2 >= 216) return false;
        const float x = b[i], y = b[i + 1], z = b[i + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
        const float l = std::sqrt(x * x + y * y + z * z);
        return std::fabs(l - 1.0f) < 0.01f;
    };
    const auto candidate = [&](int i) {
        if ((i >= 8 && i < 20) || (i >= 52 && i < 68) || (i >= 180 && i < 192)) return false;
        if (!unit(i)) return false;
        const double v[3] = {b[i], b[i + 1], b[i + 2]};
        if (std::max({std::fabs(v[0]), std::fabs(v[1]), std::fabs(v[2])}) > 0.999) return false;
        for (int n : {i - 4, i + 4}) {
            if (unit(n) && std::fabs(b[n] * v[0] + b[n + 1] * v[1] + b[n + 2] * v[2]) < 0.02) {
                return false; // a row of a rotation matrix
            }
        }
        for (int r = 0; r < 3; ++r) {
            const double row = view[r * 4] * v[0] + view[r * 4 + 1] * v[1] + view[r * 4 + 2] * v[2];
            const double col = view[r] * v[0] + view[4 + r] * v[1] + view[8 + r] * v[2];
            if (std::fabs(row) > 0.995 || std::fabs(col) > 0.995) return false;
        }
        return true;
    };
    static char src_buf[96];
    if (light_env) {
        for (int i = 0; i < 3; ++i) lw[i] = light_env_dir[i];
        src = "BB_RT_VOLUMETRIC_DIR";
    } else if (have_block) {
        int ofs = -1;
        if (light_forced_ofs >= 0) {
            ofs = unit(light_forced_ofs) ? light_forced_ofs : -1;
        } else if (light_ofs >= 0 && candidate(light_ofs)) {
            ofs = light_ofs;
        } else {
            for (int i = 20; i + 2 < 216 && ofs < 0; i += 4) {
                if (candidate(i)) ofs = i;
            }
        }
        if (ofs >= 0) {
            if (ofs != light_ofs) {
                LOG_INFO(Render_Vulkan,
                         "[rt-vol] light direction candidate: scene block floats {}-{} = ({:.3f}, "
                         "{:.3f}, {:.3f}){}",
                         ofs, ofs + 2, b[ofs], b[ofs + 1], b[ofs + 2],
                         light_forced_ofs >= 0 ? " (forced)" : " (heuristic, unverified)");
            }
            light_ofs = ofs;
            for (int i = 0; i < 3; ++i) lw[i] = b[ofs + i];
            std::snprintf(src_buf, sizeof(src_buf), "scene block floats %d-%d%s", ofs, ofs + 2,
                          light_forced_ofs >= 0 ? " (forced)" : " (heuristic)");
            src = src_buf;
        }
    }
    // Toward the light, assumed above the horizon (world y up).
    if (lw[1] < 0.0) {
        for (double& v : lw) v = -v;
    }
    double lv[3];
    for (int r = 0; r < 3; ++r) {
        lv[r] = view[r * 4] * lw[0] + view[r * 4 + 1] * lw[1] + view[r * 4 + 2] * lw[2];
    }
    const double l = std::sqrt(lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]);
    if (!(l > 1e-6)) return false;
    for (int i = 0; i < 3; ++i) out_view[i] = static_cast<float>(lv[i] / l);
    light_src = src;
    return true;
}

void RtReflect::Volumetric(vk::CommandBuffer cmd, const Input& in, const float* proj,
                           const double* inv, const double* view, const double* reproj,
                           const double* prev_proj_, bool reproj_ok,
                           const vk::DescriptorImageInfo& depth_info) {
    float lv[3];
    if (!LightDir(view, lv)) return;
    const auto device = instance.GetDevice();
    if (vol_w != w || vol_h != h || !vraw.image) {
        FreeImage(vraw);
        for (auto& i : vacc) FreeImage(i);
        if (!MakeImage(vraw, w, h) || !MakeImage(vacc[0], w, h) || !MakeImage(vacc[1], w, h)) {
            FreeImage(vraw);
            for (auto& i : vacc) FreeImage(i);
            vol_ok = false;
            LOG_ERROR(Render_Vulkan, "[rt-vol] images {}x{} failed: volumetric light off", w, h);
            return;
        }
        vol_w = w;
        vol_h = h;
        vol_fresh = true;
    }
    const u32 slot = u32(vol_dispatches % Ring);
    const u32 cur = u32(vol_dispatches & 1), prv = cur ^ 1u;
    VolParams vp{};
    double pd[16];
    for (int i = 0; i < 16; ++i) pd[i] = proj[i];
    ToGlsl(pd, vp.proj);
    ToGlsl(inv, vp.inv_proj);
    const bool hist = reproj_ok && !vol_fresh;
    if (hist) {
        ToGlsl(reproj, vp.reproj);
        ToGlsl(prev_proj_, vp.prev_proj);
    }
    vp.size[0] = w;
    vp.size[1] = h;
    vp.size[2] = in.color_width;
    vp.size[3] = in.color_height;
    vp.light[0] = lv[0];
    vp.light[1] = lv[1];
    vp.light[2] = lv[2];
    vp.color[0] = vol_color[0];
    vp.color[1] = vol_color[1];
    vp.color[2] = vol_color[2];
    vp.color[3] = vol_strength;
    vp.misc[0] = vol_density;
    vp.misc[1] = vol_dist;
    vp.misc[2] = float(vol_steps);
    vp.misc[3] = float(vol_dispatches & 0xffff);
    vp.misc2[0] = hist ? 1.0f : 0.0f;
    vp.misc2[1] = 0.1f;
    float far_dist = 3000.0f;
    {
        float pj[16], vw[12];
        if (BB::Upscale::RtCamera(pj, vw, far_dist) && !(far_dist > 1.0f)) far_dist = 3000.0f;
    }
    vp.misc2[2] = far_dist;
    vp.misc2[3] = 1.0f;
    std::memcpy(static_cast<u8*>(vol_params.mapped) + slot * ParamSize, &vp, sizeof(vp));

    if (vol_queries && vol_query_used[slot]) {
        std::array<u64, 2> ts{};
        if (device.getQueryPoolResults(vol_queries, slot * 2, 2, sizeof(ts), ts.data(),
                                       sizeof(u64), vk::QueryResultFlagBits::e64) ==
            vk::Result::eSuccess) {
            const double period =
                instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
            const double ms = double(ts[1] - ts[0]) * period * 1e-6;
            if (ms > 0.0 && ms < 1000.0) {
                vol_ms_sum += ms;
                ++vol_ms_n;
            }
        }
    }
    if (vol_queries) {
        cmd.resetQueryPool(vol_queries, slot * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, vol_queries, slot * 2);
        vol_query_used[slot] = true;
    }
    if (vol_fresh) {
        std::array<vk::ImageMemoryBarrier2, 3> b{};
        const std::array<vk::Image, 3> imgs{vraw.image, vacc[0].image, vacc[1].image};
        for (u32 i = 0; i < 3; ++i) {
            b[i] = vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = imgs[i],
                .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
        }
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 3,
                                                .pImageMemoryBarriers = b.data()});
        vol_fresh = false;
    }
    const vk::MemoryBarrier2 cc{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    const vk::DescriptorBufferInfo params_info{vol_params.buffer, slot * ParamSize, ParamSize};
    const vk::DescriptorBufferInfo counters_info{counters.buffer, 0, counters.size};
    const vk::DescriptorImageInfo scene_info{{}, in.color_storage_view, vk::ImageLayout::eGeneral};
    const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas,
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, vol_pipe);
    const auto pass = [&](u32 mode, const Img& src, const Img& hst, const Img& dst, u32 dw,
                          u32 dh) {
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &cc});
        const vk::DescriptorImageInfo src_i{nearest, src.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo hist_i{linear, hst.view, vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo dst_i{{}, dst.view, vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 8> wr{{
            {.dstBinding = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depth_info},
            {.pNext = &as_info, .dstBinding = 1, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
            {.dstBinding = 2, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &params_info},
            {.dstBinding = 3, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &src_i},
            {.dstBinding = 4, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hist_i},
            {.dstBinding = 5, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dst_i},
            {.dstBinding = 6, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &scene_info},
            {.dstBinding = 7, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &counters_info},
        }};
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, vol_layout, 0, wr);
        cmd.pushConstants(vol_layout, vk::ShaderStageFlagBits::eCompute, 0, 4, &mode);
        cmd.dispatch((dw + 7) / 8, (dh + 7) / 8, 1);
    };
    pass(0, vacc[prv], vacc[prv], vraw, w, h);                         // trace
    pass(1, vraw, vacc[prv], vacc[cur], w, h);                         // temporal
    pass(2, vacc[cur], vacc[prv], vraw, in.color_width, in.color_height); // upsample + add
    if (vol_queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, vol_queries, slot * 2 + 1);
    }
    ++vol_dispatches;
    LogVolumetric();
}

void RtReflect::LogVolumetric() {
    const auto now = std::chrono::steady_clock::now();
    if (vol_last_log != std::chrono::steady_clock::time_point{} &&
        now - vol_last_log < std::chrono::seconds(5)) {
        return;
    }
    vol_last_log = now;
    u32 c[2];
    std::memcpy(c, static_cast<const u32*>(counters.mapped) + 8, sizeof(c));
    const u32 samples = c[0] - vol_last_counts[0], lit = c[1] - vol_last_counts[1];
    vol_last_counts[0] = c[0];
    vol_last_counts[1] = c[1];
    LOG_INFO(Render_Vulkan,
             "[rt-vol] 5 s: {} dispatches, {}x{} trace, {} light rays, {:.1f}% unshadowed, GPU "
             "{:.3f} ms per pass, strength {:.2f}, light dir source: {}",
             vol_dispatches - vol_last_dispatches, vol_w, vol_h, samples,
             samples ? 100.0 * lit / samples : 0.0, vol_ms_n ? vol_ms_sum / vol_ms_n : 0.0,
             vol_strength, light_src);
    vol_last_dispatches = vol_dispatches;
    vol_ms_sum = 0.0;
    vol_ms_n = 0;
}

} // namespace BB
