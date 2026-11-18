// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtracing-2: DLSS Ray Reconstruction (NGX DLSS-D) for the path traced frame.
//
// With BB_PATHTRACING=1, BB_UPSCALER=dlss and NVIDIA's nvngx_dlssd.dll beside shadPS4.exe (or in
// its streamline folder / BB_STREAMLINE_DIR), the upscaler pass (bb_present_sr.cpp) hands the
// frame to Ray Reconstruction instead of DLSS Super Resolution: RR denoises and upscales in one
// network, with the guides the path tracer's mod already makes (vk_dlss_rr.cpp: diffuse /
// specular albedo, normals + roughness from the G-buffer, the reflections' hit distance) and the
// camera matrices. Anything missing (DLL, device support, guides) falls back to DLSS SR for the
// session, logged once. BB_PT_DLSS_RR: auto (the presets' default) / 1 = on when possible, 0 = off.
//
// The runtime is loaded directly (as bb_dlss5.cpp loads nvngx_dlssnr.dll): nvngx_dlssd.dll exports
// the NGX Vulkan entry points (NVSDK_NGX_VULKAN_Init_Ext / CreateFeature / EvaluateFeature /
// ReleaseFeature, checked on 310.x), and its caller check is answered the same way.
#pragma once

#include <memory>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace BB::SR {
struct Desc;
}
namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;
class RayTracing;
class DlssGuides;

class PtRayReconstruction {
public:
    PtRayReconstruction(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        VideoCore::TextureCache& texture_cache, RayTracing& ray_tracing);
    ~PtRayReconstruction();

    /// The live object (null without path tracing).
    static PtRayReconstruction* Current();

    /// True when Ray Reconstruction wrote desc.output this frame (the SR back end is skipped);
    /// false: run DLSS SR as usual.
    bool Evaluate(const BB::SR::Desc& desc);

private:
    struct Ngx;
    bool Wanted() const;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    RayTracing& ray_tracing;
    std::unique_ptr<DlssGuides> guides;
    std::unique_ptr<Ngx> ngx;
    bool failed = false;
    bool active = false; // the last frame went through RR (a switch resets the history)
    u64 frames = 0, fallbacks = 0;
};

} // namespace Vulkan
