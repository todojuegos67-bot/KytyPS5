// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Compatibility layer for the path tracing mod by Joao Vitor Alves (GPL-2.0,
// https://github.com/Joaovitor404silva/bloodborne-pc-path-tracing, commit 04d398f), which was
// written against SuperMedo's shadPS4 derivative (Supermedo/bloodborne_pc 1.9.8.6). Our renderer
// does not have SuperMedo's ObjectMotion (vertex-position store), SceneTargets (reduced-resolution
// proxies), GpuProfiler or vk_dlss.h (NGX bridge wrapper). This header gives the mod's engine
// (vk_ray_tracing.cpp, vk_dlss_rr.cpp) the types it names, with the behaviour our renderer has
// today:
//  - ObjectMotion: not available (Enabled() false): the moon history follows the camera only.
//  - SceneTargets: our fork renders the scene at the guest size, so RayTracing::SceneInputs::reduced
//    is always false and these are never reached with a live object; they exist to compile.
//  - GpuProfiler: none (Get() returns null). The mod's own timestamp queries (BB_PT_GPU_TIME) do
//    the per-pass GPU timing instead.
//  - PtCapture: the addresses of the per-vertex stores the mod reads geometry from (SuperMedo's
//    Shader::MotionVectors::positions/uvs/normals_address). They stay 0 until our shader
//    recompiler writes clip-space positions per draw (see documents/pathtracing/PATH_TRACING.md,
//    "Remaining work"); with 0 the engine logs "off (object motion capture is off)" and does
//    nothing.
#pragma once

#include <array>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

// The geometry store the mod traces against (one vec4 clip position per vertex, half2 UV, two
// snorm4x8 normals), filled by the vertex shaders of the frame's draws. 0 = not available.
struct PtCapture {
    static inline u64 positions_address = 0;
    static inline u64 uvs_address = 0;
    static inline u64 normals_address = 0;
};

class ObjectMotion {
public:
    [[nodiscard]] bool Enabled() const noexcept {
        return false;
    }
    vk::ImageView PrepareRead(vk::CommandBuffer, u32, u32, bool& valid) {
        valid = false;
        return {};
    }
};

class SceneTargets {
public:
    struct Extent {
        u32 width = 0, height = 0;
    };
    struct Target {
        vk::Image image{};
        vk::ImageView view{};
    };
    [[nodiscard]] bool EligibleScene(const VideoCore::Image&) const noexcept {
        return false;
    }
    [[nodiscard]] bool ProxyCurrent(const VideoCore::Image&) const noexcept {
        return false;
    }
    [[nodiscard]] Extent Size() const noexcept {
        return {};
    }
    Target Read(VideoCore::ImageId, const VideoCore::ImageViewInfo&, vk::PipelineStageFlags2,
                vk::AccessFlags2) {
        return {};
    }
};

class GpuProfiler {
public:
    static GpuProfiler* Get() noexcept {
        return nullptr;
    }
    template <typename Describe>
    void Mark(u64, Describe&&) {}
};

// SuperMedo's vk_dlss.h types used by the Ray Reconstruction guides (vk_dlss_rr.*).
namespace Dlss {
struct Resource {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::ImageAspectFlags aspect;
    u32 width, height;
};
struct RrGuides {
    Resource diffuse_albedo, specular_albedo, normal_roughness, specular_hit;
};
} // namespace Dlss

} // namespace Vulkan
