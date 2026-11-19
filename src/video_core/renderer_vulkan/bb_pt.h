// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtrace: a path tracer over the game's own geometry. Ported from the Bloodborne PC path
// tracing mod (Joaovitor404silva/bloodborne-pc-path-tracing, GPL-2.0): its capture (per-vertex
// clip positions of the scene draws), gather into view-space triangles, acceleration structure
// and ray queries. Independent of bb_rt_reflect / bb_volfog / bb_mv: they are not touched.
//
// BB_PT=1 turns it on (it needs ray query and acceleration structures). Unset or 0: no pipeline
// variant, buffer, pass or hook does anything.
#pragma once

#include <array>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace Core {
class MemoryManager;
}
namespace Shader {
struct PushData;
}
namespace Vulkan {
class Instance;
}

namespace BB::PathTrace {

// BB_PT=1.
bool Requested();
// BB_PT_TRIANGLES (2097152, 1024..8388608): triangle capacity per frame. Draws past it are
// skipped and counted in the periodic log line.
u32 TriangleCapacity();
// BB_PT_RANGE (50, metres): the depth at which the distance view fades to black.
float ViewRange();

// One scene draw of a capture pipeline, as the rasterizer issues it.
struct DrawCapture {
    VAddr index_address = 0;          // guest address of the draw's first index (indexed only)
    u32 index_size = 0;               // 2 or 4; 0 for a non-indexed draw
    u32 num_indices = 0;
    u32 instances = 0;
    u32 vertex_offset = 0;            // gl_VertexIndex of index 0 (Vulkan vertexOffset)
    u32 instance_offset = 0;          // gl_InstanceIndex of instance 0 (Vulkan firstInstance)
};

class Tracer {
public:
    Tracer(const Vulkan::Instance& instance, Core::MemoryManager& memory);
    ~Tracer();

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;

    // Creates the buffers; false leaves the path tracer off (the log says why).
    bool Init();

    // Rasterizer, Draw, before the draw command, for a draw of a capture pipeline (see
    // Pipeline::BbPtCapture). Records the draw's indices into the frame's index buffer and
    // fills push's pt_* fields, so the vertex variant stores this draw's clip positions.
    // False: the draw is not captured and its push fields stay zero (nothing is stored).
    bool CaptureDraw(const DrawCapture& draw, Shader::PushData& push);

    // Once per guest frame, after its draws: logs the counters every 600 frames and moves the
    // tables on to the next frame in flight.
    void EndFrame();

private:
    struct Buf {
        vk::Buffer buffer{};
        VmaAllocation allocation{};
        vk::DeviceSize size = 0;
        vk::DeviceAddress address = 0;
        void* mapped = nullptr;
    };
    bool MakeBuffer(Buf& buf, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host);
    void FreeBuffer(Buf& buf);

    // Frames in flight: the host tables of a frame are not reused before this many frames.
    static constexpr u32 FrameSlots = 4;

    const Vulkan::Instance& instance;
    Core::MemoryManager& memory;
    bool ready = false;

    Buf positions;                          // device: vec4 clip position per slot
    std::array<Buf, FrameSlots> indices;    // host: the frame's copied index data
    std::array<Buf, FrameSlots> records;    // host: the frame's records

    u32 frame_slot = 0;
    u32 position_slots = 0;        // slots used this frame
    u32 index_bytes_used = 0;
    u32 triangles_used = 0;
    u32 record_count = 0;
    u32 max_records = 0;
    u32 position_capacity = 0;     // slots
    u32 index_capacity = 0;        // bytes
    u64 stat_frames = 0;
    u64 stat_draws = 0;
    u64 stat_triangles = 0;
    u64 stat_skipped = 0;
};

} // namespace BB::PathTrace
