// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtracing: the per-vertex geometry store the path tracer (Joao Vitor Alves' mod, rt/)
// builds its BVH from. SuperMedo's renderer wrote it from its object-motion instrumentation; here
// it is a vertex-shader variant of our own (emit_spirv_bb_pt.cpp, HwVertexRuntimeInfo::
// bb_pt_store) used only for opaque G-buffer draws while BB_PATHTRACING=1:
//  - one device buffer, capacity C vertices (slot 0 unused, then two halves used on alternate
//    frames): positions vec4 at slot * 16, UVs half2 at C * 16 + slot * 4, normals 2 x snorm4x8
//    at C * 20 + slot * 8 (the layout the engine's gather pass reads, see pt_compat.h);
//  - per draw the rasterizer scans its index range (cached, re-verified every 32 frames),
//    allocates range x instances slots and pushes (store base, first vertex, vertices, first
//    instance, UV select, C) in the push constants (PushData::Pt*).
// This header is included by the pipeline cache in every build; PtCapture::active stays false
// without ENABLE_PATHTRACING or with the switch off, so nothing changes then.
#pragma once

#include <cstdlib>
#include <unordered_map>
#include "common/types.h"

namespace Vulkan {

struct PtCapture {
    /// Path tracing is on: G-buffer pipelines take the storing vertex shader variant.
    static inline bool active = false;
    /// G-buffer draws: colour targets bound (Bloodborne's G-buffer pass binds 7; the mod's test
    /// is 5+ with depth, vk_rasterizer gbuffer_draw).
    static constexpr u32 GBufferTargets = 5;
    /// The store as the engine reads it (0 = not available).
    static inline u64 positions_address = 0;
    static inline u64 uvs_address = 0;
    static inline u64 normals_address = 0;
    /// BB_PATHTRACING=1 in a build with the engine (read before the pipeline cache preloads:
    /// fragment shaders cached without the tracer's surface info are then compiled again, once,
    /// so G-buffer hits get their textures and alpha test; vk_pipeline_serialization.cpp).
    static bool Requested() {
#ifdef ENABLE_PATHTRACING
        static const bool on = [] {
            const char* e = std::getenv("BB_PATHTRACING");
            return e && e[0] == '1';
        }();
        return on;
#else
        return false;
#endif
    }
};

#ifdef ENABLE_PATHTRACING
class Instance;

/// The store and its per-frame slot allocator (GPU thread only).
class PtStore {
public:
    struct Range {
        u32 first{}, count{};
        u64 topology{}; // hash of the index list (the engine's BLAS identity)
    };

    explicit PtStore(const Instance& instance);
    ~PtStore();
    PtStore(const PtStore&) = delete;
    PtStore& operator=(const PtStore&) = delete;

    [[nodiscard]] bool Ready() const noexcept {
        return address != 0;
    }
    [[nodiscard]] u64 Address() const noexcept {
        return address;
    }
    [[nodiscard]] u32 Capacity() const noexcept {
        return capacity;
    }
    /// At the display pass: the other half takes this frame's stores.
    void BeginFrame();
    /// Slots for one draw (0: the half is full, the draw stays out of the BVH this frame).
    u32 Allocate(u64 vertices);
    /// The vertices an indexed draw references ([min, max] + base vertex) and the hash of its
    /// index list, cached by (address, count, size, base, restart).
    Range IndexRange(const void* indices, VAddr address, u32 count, u32 index_size,
                     s32 base_vertex, bool restart);

    struct Stats {
        u64 draws{}, vertices{}, full{}, scans{}, hits{}, stale{};
        u32 peak{};
    } stats;

private:
    struct Key {
        VAddr address{};
        u32 count{}, index_size{};
        s32 base_vertex{};
        bool restart{};
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            return size_t(k.address * 0x9E3779B97F4A7C15ull ^ (u64(k.count) << 20) ^
                          (u64(k.index_size) << 4) ^ (u64(u32(k.base_vertex)) << 36) ^
                          u64(k.restart));
        }
    };
    struct Entry {
        Range range;
        u64 verified{}, used{};
        bool valid{};
    };

    const Instance& instance;
    void* buffer{};     // VkBuffer
    void* allocation{}; // VmaAllocation
    u64 address{};
    u32 capacity{};
    u32 per_half{};
    u32 half{};
    u32 cursor{};
    u64 frame{};
    std::unordered_map<Key, Entry, KeyHash> ranges;
};
#endif

} // namespace Vulkan
