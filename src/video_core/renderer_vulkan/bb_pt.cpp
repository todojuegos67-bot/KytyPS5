// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include "common/logging/log.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "video_core/host_shaders/bb_pt_gather_comp.h"
#include "video_core/host_shaders/bb_pt_view_comp.h"
#include "video_core/renderer_vulkan/bb_pt.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB::PathTrace {

namespace {

constexpr u32 MaxRecords = 16384;
// Refits of the scene acceleration structure between two full builds: the BVH keeps its topology
// while the view-space triangles move, and a full build restores its quality.
constexpr u32 RefitLimit = 16;
constexpr auto BlasFlags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
                           vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;

// One captured draw, read by the gather shader (bb_pt_gather.comp, std430, 48 bytes).
struct Record {
    u32 tri_base;     // first triangle of the draw in the gathered order
    u32 tri_count;    // triangles of all instances
    u32 prims;        // triangles per instance
    u32 index_offset; // byte offset of its indices in the index buffer (indexed draws)
    u32 index_min;    // smallest index value (indexed draws), subtracted to form the slot
    u32 index16;      // 1: 16-bit indices
    u32 indexed;      // 1: indexed
    u32 slot_base;    // first clip position slot
    u32 span;         // slots per instance
    u32 instances;
    u32 pad[2];
};
static_assert(sizeof(Record) == 48);

struct GatherPush {
    u32 tri_total;
    u32 record_count;
    float p00, p11, p02, p12;
};
static_assert(sizeof(GatherPush) == 24);

struct ViewPush {
    float p00, p11, p02, p12;
    float range;
    u32 width;
    u32 height;
    u32 debug;
    float sun_x, sun_y, sun_z;
    float sun_r, sun_g, sun_b;
};
static_assert(sizeof(ViewPush) == 56);

float EnvFloat(const char* name, float def, float lo, float hi) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) {
        return def;
    }
    char* end = nullptr;
    const float v = std::strtof(e, &end);
    return end != e && std::isfinite(v) ? std::clamp(v, lo, hi) : def;
}

} // namespace

bool Requested() {
    static const bool on = [] {
        const char* e = std::getenv("BB_PT");
        return e && e[0] == '1';
    }();
    return on;
}

u32 TriangleCapacity() {
    static const u32 tris = u32(EnvFloat("BB_PT_TRIANGLES", 2097152.0f, 1024.0f, 8388608.0f));
    return tris;
}

float ViewRange() {
    static const float range = EnvFloat("BB_PT_RANGE", 50.0f, 1.0f, 1000.0f);
    return range;
}

bool DebugRequested() {
    static const bool on = [] {
        const char* e = std::getenv("BB_PT_DEBUG");
        return e && e[0] == '1';
    }();
    return on;
}

Tracer::Tracer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
               Core::MemoryManager& memory_)
    : instance{instance_}, scheduler{scheduler_}, memory{memory_} {}

Tracer::~Tracer() {
    const auto device = instance.GetDevice();
    if (tlas) {
        device.destroyAccelerationStructureKHR(tlas);
    }
    if (blas) {
        device.destroyAccelerationStructureKHR(blas);
    }
    FreePass(gather_pass);
    FreePass(view_pass);
    FreeImage(out);
    FreeBuffer(soup);
    FreeBuffer(blas_buf);
    FreeBuffer(tlas_buf);
    FreeBuffer(scratch);
    FreeBuffer(instance_buf);
    FreeBuffer(positions);
    for (auto& index : indices) {
        FreeBuffer(index);
    }
    for (auto& record : records) {
        FreeBuffer(record);
    }
}

bool Tracer::MakeBuffer(Buf& buf, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host) {
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
    VkBuffer handle{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &bci, &aci, &handle, &buf.allocation, &info) !=
        VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "[pt] buffer of {} bytes could not be created", size);
        buf = {};
        return false;
    }
    buf.buffer = handle;
    buf.mapped = info.pMappedData;
    buf.size = size;
    buf.address = instance.GetDevice().getBufferAddress({.buffer = buf.buffer});
    return true;
}

void Tracer::FreeBuffer(Buf& buf) {
    if (buf.buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), buf.buffer, buf.allocation);
    }
    buf = {};
}

void Tracer::FreeImage(Img& img) {
    if (img.view) {
        instance.GetDevice().destroyImageView(img.view);
    }
    if (img.image) {
        vmaDestroyImage(instance.GetAllocator(), img.image, img.allocation);
    }
    img = {};
}

void Tracer::FreePass(Pass& pass) {
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

bool Tracer::MakeScene() {
    const auto device = instance.GetDevice();
    const u32 tris = TriangleCapacity();
    using U = vk::BufferUsageFlagBits;
    if (!MakeBuffer(soup, vk::DeviceSize(tris) * 9 * sizeof(float),
                    U::eStorageBuffer | U::eAccelerationStructureBuildInputReadOnlyKHR, false) ||
        !MakeBuffer(instance_buf, sizeof(vk::AccelerationStructureInstanceKHR),
                    U::eAccelerationStructureBuildInputReadOnlyKHR, true)) {
        return false;
    }
    const vk::AccelerationStructureGeometryKHR geom{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry{.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                      .vertexFormat = vk::Format::eR32G32B32Sfloat,
                      .vertexData{.deviceAddress = soup.address},
                      .vertexStride = 3 * sizeof(float),
                      .maxVertex = tris * 3 - 1,
                      .indexType = vk::IndexType::eNoneKHR,
                  }},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR binfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = BlasFlags,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geom,
    };
    const auto bsz = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, binfo, tris);
    const vk::AccelerationStructureGeometryKHR tgeom{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry{.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                      .data{.deviceAddress = instance_buf.address}}},
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
    tlas_scratch = (std::max(bsz.buildScratchSize, bsz.updateScratchSize) + align - 1) / align * align;
    if (!MakeBuffer(blas_buf, bsz.accelerationStructureSize, U::eAccelerationStructureStorageKHR,
                    false) ||
        !MakeBuffer(tlas_buf, tsz.accelerationStructureSize, U::eAccelerationStructureStorageKHR,
                    false) ||
        !MakeBuffer(scratch, tlas_scratch + tsz.buildScratchSize, U::eStorageBuffer, false)) {
        return false;
    }
    if (scratch.address % align != 0) {
        LOG_ERROR(Render_Vulkan, "[pt] scratch buffer is not aligned to {} bytes", align);
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
        return false;
    }
    blas = b;
    tlas = t;
    vk::AccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1.0f;
    inst.mask = 0xFF;
    inst.flags = static_cast<VkGeometryInstanceFlagsKHR>(
        vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable |
        vk::GeometryInstanceFlagBitsKHR::eForceOpaque);
    inst.accelerationStructureReference = device.getAccelerationStructureAddressKHR({
        .accelerationStructure = blas,
    });
    std::memcpy(instance_buf.mapped, &inst, sizeof(inst));
    return true;
}

bool Tracer::MakePass(std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
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
                .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = pass.layout,
            });
    device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) {
        return false;
    }
    pass.pipe = pipe;
    return true;
}

bool Tracer::MakeOutput(vk::Extent2D extent) {
    if (out.image && out.width == extent.width && out.height == extent.height) {
        return true;
    }
    if (out.image) {
        (void)instance.GetDevice().waitIdle(); // a resize only
    }
    FreeImage(out);
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
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
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &out.allocation, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    out.image = image;
    out.width = extent.width;
    out.height = extent.height;
    auto [r, v] = instance.GetDevice().createImageView({
        .image = out.image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) {
        FreeImage(out);
        return false;
    }
    out.view = v;
    return true;
}

bool Tracer::Init() {
    const u32 triangles = TriangleCapacity();
    position_capacity = 2 * triangles;
    index_capacity = triangles * 3 * 4;
    max_records = MaxRecords;
    const vk::BufferUsageFlags storage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (!MakeBuffer(positions, vk::DeviceSize(position_capacity) * 16,
                    storage | vk::BufferUsageFlagBits::eShaderDeviceAddress, false)) {
        return false;
    }
    for (u32 i = 0; i < FrameSlots; ++i) {
        if (!MakeBuffer(indices[i], index_capacity, storage, true) ||
            !MakeBuffer(records[i], vk::DeviceSize(max_records) * sizeof(Record), storage, true)) {
            return false;
        }
    }
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    const std::array<vk::DescriptorSetLayoutBinding, 4> gather_bindings{{
        {0, vk::DescriptorType::eStorageBuffer, 1, C},
        {1, vk::DescriptorType::eStorageBuffer, 1, C},
        {2, vk::DescriptorType::eStorageBuffer, 1, C},
        {3, vk::DescriptorType::eStorageBuffer, 1, C},
    }};
    const std::array<vk::DescriptorSetLayoutBinding, 3> view_bindings{{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, C},
        {1, vk::DescriptorType::eStorageImage, 1, C},
        {2, vk::DescriptorType::eStorageBuffer, 1, C},
    }};
    if (!MakeScene() ||
        !MakePass(gather_bindings, sizeof(GatherPush), BB_PT_GATHER_COMP, gather_pass) ||
        !MakePass(view_bindings, sizeof(ViewPush), BB_PT_VIEW_COMP, view_pass)) {
        LOG_ERROR(Render_Vulkan, "[pt] acceleration structures or passes failed: path tracer off");
        return false;
    }
    ready = true;
    LOG_INFO(Render_Vulkan,
             "[pt] path tracer on (BB_PT=1): capture {} triangles per frame, {} MiB of clip "
             "positions, {} MiB of indices per frame in flight, {} draws per frame",
             triangles, positions.size >> 20, index_capacity >> 20, max_records);
    return true;
}

bool Tracer::CaptureDraw(const DrawCapture& draw, Shader::PushData& push) {
    if (!ready || draw.instances == 0 || draw.num_indices < 3) {
        ++stat_skipped;
        return false;
    }
    const u32 prims = draw.num_indices / 3;
    const u32 vertices = prims * 3;
    u32 index_min = 0;
    u64 span = vertices; // non-indexed: index i is vertex i
    u8* index_out = nullptr;
    if (draw.index_size != 0) {
        if (draw.index_size != 2 && draw.index_size != 4) {
            ++stat_skipped;
            return false;
        }
        const u64 bytes = u64(vertices) * draw.index_size;
        if (u64(index_bytes_used) + ((bytes + 3) & ~u64(3)) > index_capacity) {
            ++stat_skipped;
            return false;
        }
        if (memory.ClampRangeSize(draw.index_address, bytes) < bytes) {
            ++stat_skipped;
            return false;
        }
        index_out = static_cast<u8*>(indices[frame_slot].mapped) + index_bytes_used;
        memory.CopySparseMemory(draw.index_address, index_out, bytes);
        // The primitive restart marker is not a vertex: it is left out of the range.
        const u32 marker = draw.index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
        u32 lo = ~0u, hi = 0;
        for (u32 i = 0; i < vertices; ++i) {
            u32 v = 0;
            if (draw.index_size == 2) {
                u16 v16{};
                std::memcpy(&v16, index_out + u64(i) * 2, 2);
                v = v16;
            } else {
                std::memcpy(&v, index_out + u64(i) * 4, 4);
            }
            if (v == marker) {
                continue;
            }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (lo > hi) {
            ++stat_skipped;
            return false; // only restart markers
        }
        index_min = lo;
        span = u64(hi) - lo + 1;
    }
    const u64 triangles = u64(prims) * draw.instances;
    const u64 slots = span * draw.instances;
    if (u64(triangles_used) + triangles > TriangleCapacity() ||
        u64(position_slots) + slots > position_capacity || record_count >= max_records ||
        span > 0xFFFFFFFFull / draw.instances || span > position_capacity) {
        ++stat_skipped;
        return false;
    }

    const Record record{
        .tri_base = triangles_used,
        .tri_count = u32(triangles),
        .prims = prims,
        .index_offset = index_bytes_used,
        .index_min = index_min,
        .index16 = draw.index_size == 2 ? 1u : 0u,
        .indexed = draw.index_size != 0 ? 1u : 0u,
        .slot_base = position_slots,
        .span = u32(span),
        .instances = draw.instances,
        .pad = {},
    };
    static_cast<Record*>(records[frame_slot].mapped)[record_count] = record;
    if (index_out) {
        index_bytes_used += u32((u64(vertices) * draw.index_size + 3) & ~u64(3));
    }

    push.pt_slot = position_slots;
    push.pt_span = u32(span);
    push.pt_base = draw.vertex_offset + index_min;
    push.pt_inst_base = draw.instance_offset;
    push.pt_store = {u32(positions.address), u32(positions.address >> 32)};
    push.pt_inst_count = draw.instances;

    position_slots += u32(slots);
    triangles_used += u32(triangles);
    ++record_count;
    ++stat_draws;
    return true;
}

vk::ImageView Tracer::Trace(vk::CommandBuffer cmd, vk::Extent2D extent, const float proj[16],
                            const float sun_view[3], const float sun_color[3], bool have_sun) {
    const bool debug = DebugRequested();
    if (!ready || triangles_used == 0 || record_count == 0 || (!debug && !have_sun) ||
        !MakeOutput(extent)) {
        return {};
    }
    ++stat_traced;
    slot_ticks[frame_slot] = scheduler.CurrentTick();
    const u32 tris = triangles_used;
    const vk::DescriptorBufferInfo records_info{records[frame_slot].buffer, 0,
                                                records[frame_slot].size};
    const vk::DescriptorBufferInfo indices_info{indices[frame_slot].buffer, 0,
                                                indices[frame_slot].size};
    const vk::DescriptorBufferInfo positions_info{positions.buffer, 0, positions.size};
    const vk::DescriptorBufferInfo soup_info{soup.buffer, 0, soup.size};

    const vk::MemoryBarrier2 inputs_ready{
        .srcStageMask = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eMemoryWrite |
                         vk::AccessFlagBits2::eShaderRead |
                         vk::AccessFlagBits2::eAccelerationStructureReadKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &inputs_ready});

    const std::array<vk::WriteDescriptorSet, 4> gather_writes{{
        {.dstBinding = 0, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &records_info},
        {.dstBinding = 1, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indices_info},
        {.dstBinding = 2, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &positions_info},
        {.dstBinding = 3, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &soup_info},
    }};
    const GatherPush gather{tris, record_count, proj[0], proj[5], proj[2], proj[6]};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, gather_pass.pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, gather_pass.layout, 0, gather_writes);
    cmd.pushConstants(gather_pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gather),
                      &gather);
    cmd.dispatch((tris + 63) / 64, 1, 1);

    const vk::MemoryBarrier2 positions_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &positions_read});

    const vk::MemoryBarrier2 soup_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                         vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &soup_done});

    const vk::AccelerationStructureGeometryKHR geom{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry{.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                      .vertexFormat = vk::Format::eR32G32B32Sfloat,
                      .vertexData{.deviceAddress = soup.address},
                      .vertexStride = 3 * sizeof(float),
                      .maxVertex = tris * 3 - 1,
                      .indexType = vk::IndexType::eNoneKHR,
                  }},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    const bool refit = blas_triangles == tris && blas_refits < RefitLimit;
    if (refit) {
        ++blas_refits;
        ++stat_blas_refits;
    } else {
        blas_refits = 0;
        blas_triangles = tris;
        ++stat_blas_full;
    }
    const vk::AccelerationStructureBuildGeometryInfoKHR binfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = BlasFlags,
        .mode = refit ? vk::BuildAccelerationStructureModeKHR::eUpdate
                      : vk::BuildAccelerationStructureModeKHR::eBuild,
        .srcAccelerationStructure = refit ? blas : vk::AccelerationStructureKHR{},
        .dstAccelerationStructure = blas,
        .geometryCount = 1,
        .pGeometries = &geom,
        .scratchData{.deviceAddress = scratch.address},
    };
    const vk::AccelerationStructureBuildRangeInfoKHR brange{.primitiveCount = tris};
    const vk::AccelerationStructureBuildRangeInfoKHR* pbr = &brange;
    cmd.buildAccelerationStructuresKHR(1, &binfo, &pbr);

    const vk::MemoryBarrier2 blas_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &blas_done});

    const vk::AccelerationStructureGeometryKHR tgeom{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry{.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                      .data{.deviceAddress = instance_buf.address}}},
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

    const vk::MemoryBarrier2 tlas_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                         vk::AccessFlagBits2::eAccelerationStructureReadKHR,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &tlas_done});

    const vk::ImageMemoryBarrier2 to_general{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = out.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &to_general});

    const vk::DescriptorImageInfo out_info{{}, out.view, vk::ImageLayout::eGeneral};
    const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas,
    };
    const std::array<vk::WriteDescriptorSet, 3> view_writes{{
        {.pNext = &as_info, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
        {.dstBinding = 1, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &out_info},
        {.dstBinding = 2, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &soup_info},
    }};
    const ViewPush view{proj[0],    proj[5],       proj[2],       proj[6],
                        ViewRange(), extent.width, extent.height, debug ? 1u : 0u,
                        sun_view[0], sun_view[1],  sun_view[2],   sun_color[0],
                        sun_color[1], sun_color[2]};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, view_pass.pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, view_pass.layout, 0, view_writes);
    cmd.pushConstants(view_pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(view), &view);
    cmd.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);

    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = out.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &to_read});
    return out.view;
}

void Tracer::EndFrame() {
    if (!ready) {
        return;
    }
    ++stat_frames;
    stat_triangles += triangles_used;
    if (stat_frames % 600 == 0) {
        LOG_INFO(Render_Vulkan,
                 "[pt] {} frames ({} traced): {} draws captured, {} triangles ({} per frame), {} "
                 "draws skipped (capacity, unmapped or unusable indices), BLAS {} full builds, "
                 "{} refits",
                 stat_frames, stat_traced, stat_draws, stat_triangles, stat_triangles / stat_frames,
                 stat_skipped, stat_blas_full, stat_blas_refits);
    }
    frame_slot = (frame_slot + 1) % FrameSlots;
    if (!scheduler.IsFree(slot_ticks[frame_slot])) {
        scheduler.Wait(slot_ticks[frame_slot]);
    }
    position_slots = 0;
    index_bytes_used = 0;
    triangles_used = 0;
    record_count = 0;
}

} // namespace BB::PathTrace
