// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-rt-reflect (BB_RT_REFLECT=1, default off): ray-traced reflections replacing the output of
// the game's SSR draw (PS 0x6c2323c2) where a ray hits. After that draw, outside the render pass:
// a depth mesh (bb_rt_mesh.comp) -> BLAS + one-instance TLAS (rebuilt each SSR draw) -> one ray
// query per SSR pixel (bb_rt_reflect.comp) writing into the SSR target. BB_RT_REFLECT_DEBUG=1
// tints the pixels a ray hit. Needs Instance::IsRtReflectEnabled() and the game's camera.
#pragma once

#include <array>
#include <chrono>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}

namespace BB {

class RtReflect {
public:
    struct Input {
        vk::Image target;        // SSR output, RGBA16F
        vk::ImageView target_view; // storage view
        vk::ImageLayout target_layout;
        vk::Image depth;          // D32S8 (depth aspect sampled)
        vk::ImageView depth_view;
        vk::ImageLayout depth_layout;
        bool depth_has_stencil;
        vk::Image normal;         // RGBA8 G-buffer RT0
        vk::ImageView normal_view;
        vk::ImageLayout normal_layout;
        vk::Image color;          // RGBA16F lit scene
        vk::ImageView color_view;
        vk::ImageLayout color_layout;
        u32 width, height;        // SSR target / depth size
    };

    explicit RtReflect(const Vulkan::Instance& instance);
    ~RtReflect();

    // Records the pass into `cmd` (outside a render pass). Image layouts are restored.
    // Returns false when it did nothing (not supported, no camera, an allocation failed).
    bool Run(vk::CommandBuffer cmd, const Input& in);

private:
    struct Buf {
        vk::Buffer buffer{};
        VmaAllocation alloc{};
        void* mapped = nullptr;
        vk::DeviceAddress address = 0;
        vk::DeviceSize size = 0;
    };
    bool Init();
    bool EnsureSize(u32 w, u32 h);
    bool MakeBuffer(Buf& b, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host,
                    vk::DeviceSize alignment = 0);
    void FreeBuffer(Buf& b);
    void FreeSized();
    void Log();

    const Vulkan::Instance& instance;
    bool tried = false, ok = false;
    bool debug = false;
    vk::Sampler nearest{}, linear{};
    vk::DescriptorSetLayout mesh_set{}, ray_set{};
    vk::PipelineLayout mesh_layout{}, ray_layout{};
    vk::Pipeline mesh_pipe{}, ray_pipe{};
    vk::QueryPool queries{};

    static constexpr u32 Cell = 4;
    static constexpr u32 Ring = 8;
    static constexpr vk::DeviceSize ParamSize = 256;
    u32 w = 0, h = 0, gw = 0, gh = 0, tris = 0;
    Buf verts, blas_buf, tlas_buf, scratch, instances, params, counters;
    vk::AccelerationStructureKHR blas{}, tlas{};
    vk::DeviceSize blas_scratch = 0, tlas_scratch = 0;
    u32 ring_pos = 0;
    std::array<bool, Ring> query_used{};

    // Stats (logged every 5 s).
    u64 dispatches = 0, last_dispatches = 0;
    std::array<u32, 5> last_counts{};
    double gpu_ms_sum = 0.0;
    u32 gpu_ms_n = 0;
    bool logged_camera = false;
    std::chrono::steady_clock::time_point last_log{};
};

} // namespace BB
