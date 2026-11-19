// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "common/logging/log.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/bb_pt.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace BB::PathTrace {

namespace {

constexpr u32 MaxRecords = 16384;

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

Tracer::Tracer(const Vulkan::Instance& instance_, Core::MemoryManager& memory_)
    : instance{instance_}, memory{memory_} {}

Tracer::~Tracer() {
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

void Tracer::EndFrame() {
    if (!ready) {
        return;
    }
    ++stat_frames;
    stat_triangles += triangles_used;
    if (stat_frames % 600 == 0) {
        LOG_INFO(Render_Vulkan,
                 "[pt] {} frames: {} draws captured, {} triangles ({} per frame), {} draws "
                 "skipped (capacity, unmapped or unusable indices)",
                 stat_frames, stat_draws, stat_triangles, stat_triangles / stat_frames,
                 stat_skipped);
    }
    frame_slot = (frame_slot + 1) % FrameSlots;
    position_slots = 0;
    index_bytes_used = 0;
    triangles_used = 0;
    record_count = 0;
}

} // namespace BB::PathTrace
