// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-pathtracing: the geometry store (pt_capture.h).

#ifdef ENABLE_PATHTRACING

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <span>
#include <xxhash.h>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/rt/pt_capture.h"
#include "video_core/renderer_vulkan/vk_instance.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {
// Per half (one frame's G-buffer draws). Bloodborne's busiest areas reference ~2-3 M G-buffer
// vertices a frame; the mod used 8 M. BB_PT_STORE_MVERTS overrides (millions, 1-16).
constexpr u32 DefaultPerHalf = 6u << 20;
// Bytes per slot: position 16 + UV 4 + normals 8.
constexpr u64 SlotBytes = 28;

template <class Index>
PtStore::Range Scan(std::span<const Index> indices, s32 base_vertex, bool restart) {
    u32 lo = std::numeric_limits<u32>::max(), hi = 0;
    for (const Index index : indices) {
        if (restart && index == std::numeric_limits<Index>::max()) {
            continue;
        }
        lo = std::min<u32>(lo, index);
        hi = std::max<u32>(hi, index);
    }
    if (lo > hi || s64(lo) + base_vertex < 0 || s64(hi) + base_vertex > s64(INT32_MAX)) {
        return {};
    }
    return {u32(s64(lo) + base_vertex), hi - lo + 1, 0};
}
} // namespace

PtStore::PtStore(const Instance& instance_) : instance{instance_} {
    if (!instance.IsBufferDeviceAddressSupported()) {
        LOG_WARNING(Render_Vulkan, "[pt] geometry store off: buffer device address unsupported");
        return;
    }
    per_half = DefaultPerHalf;
    if (const char* env = std::getenv("BB_PT_STORE_MVERTS")) {
        const long m = std::strtol(env, nullptr, 10);
        if (m >= 1 && m <= 16) {
            per_half = u32(m) << 20;
        }
    }
    capacity = 1 + 2 * per_half;
    const VkBufferCreateInfo ci = vk::BufferCreateInfo{
        .size = u64(capacity) * SlotBytes,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VkBuffer raw{};
    VmaAllocation alloc{};
    if (vmaCreateBuffer(instance.GetAllocator(), &ci, &aci, &raw, &alloc, nullptr) != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "[pt] geometry store: {} MB not allocated",
                  (u64(capacity) * SlotBytes) >> 20);
        return;
    }
    buffer = raw;
    allocation = alloc;
    address = instance.GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = raw});
    if (!address) {
        vmaDestroyBuffer(instance.GetAllocator(), raw, alloc);
        buffer = allocation = nullptr;
        return;
    }
    PtCapture::positions_address = address;
    PtCapture::uvs_address = address + u64(capacity) * 16;
    PtCapture::normals_address = address + u64(capacity) * 20;
    cursor = 0;
    LOG_INFO(Render_Vulkan,
             "[pt] geometry store: {} M vertices a frame x 2 (BB_PT_STORE_MVERTS), {} MB VRAM",
             per_half >> 20, (u64(capacity) * SlotBytes) >> 20);
}

PtStore::~PtStore() {
    if (buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(buffer),
                         static_cast<VmaAllocation>(allocation));
    }
    PtCapture::positions_address = PtCapture::uvs_address = PtCapture::normals_address = 0;
}

void PtStore::BeginFrame() {
    stats.peak = std::max(stats.peak, cursor);
    half ^= 1u;
    cursor = 0;
    ++frame;
    if (frame % 600 == 0) {
        std::erase_if(ranges, [&](const auto& kv) { return frame - kv.second.used > 600; });
    }
}

u32 PtStore::Allocate(u64 vertices) {
    if (!address || vertices == 0) {
        return 0;
    }
    if (vertices > per_half - cursor) {
        ++stats.full;
        return 0;
    }
    const u32 base = 1 + half * per_half + cursor;
    cursor += u32(vertices);
    ++stats.draws;
    stats.vertices += vertices;
    return base;
}

PtStore::Range PtStore::IndexRange(const void* indices, VAddr address_, u32 count,
                                   u32 index_size, s32 base_vertex, bool restart) {
    auto& e = ranges[Key{address_, count, index_size, base_vertex, restart}];
    e.used = frame;
    // Index lists of world geometry do not change; a reused address is caught within 32 frames.
    if (e.valid && frame - e.verified < 32) {
        ++stats.hits;
        return e.range;
    }
    ++stats.scans;
    Range fresh = index_size == 2
                      ? Scan(std::span(static_cast<const u16*>(indices), count), base_vertex, restart)
                      : Scan(std::span(static_cast<const u32*>(indices), count), base_vertex, restart);
    fresh.topology = XXH3_64bits(indices, size_t(count) * index_size);
    if (e.valid && (fresh.topology != e.range.topology || fresh.first != e.range.first ||
                    fresh.count != e.range.count)) {
        ++stats.stale;
    }
    e.range = fresh;
    e.verified = frame;
    e.valid = true;
    return fresh;
}

} // namespace Vulkan

#endif // ENABLE_PATHTRACING
