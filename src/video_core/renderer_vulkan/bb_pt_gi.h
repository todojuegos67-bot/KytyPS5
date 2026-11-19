// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtrace GI: the indirect light stage of the path tracer (ported from the Bloodborne PC
// path tracing mod's rt_gi.comp, GPL-2.0). Runs after the lit view (bb_pt.h Tracer::Trace) on
// the same frame's acceleration structure. Radiance comes from the presented game frame (the
// screen radiance cache), accumulated over time with camera reprojection.
//
// BB_PT=1 and BB_PT_GI=1 (the default): GI is added to the lit view. BB_PT_GI=0 leaves the lit
// view as it is.
#pragma once

#include <array>
#include <span>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
}

namespace BB::PathTrace {

// BB_PT_GI (1): 0 leaves the lit view without indirect light.
bool GiRequested();

struct GiInput {
    vk::AccelerationStructureKHR tlas{};
    vk::Buffer soup{};
    vk::ImageView game{};         // the presented frame before the path tracer (shader-read)
    vk::ImageView lit{};          // the lit view of Tracer::Trace (shader-read)
    vk::Extent2D extent{};
    const float* proj = nullptr;  // scene projection, 16 floats, row-major
    const float* view = nullptr;  // scene view, 3x4 rows (12 floats)
    bool game_srgb = false;       // the game frame is sRGB-encoded (decoded in the trace pass)
};

class Gi {
public:
    explicit Gi(const Vulkan::Instance& instance);
    ~Gi();

    Gi(const Gi&) = delete;
    Gi& operator=(const Gi&) = delete;

    bool Init();
    // The next frame starts without history (no lit frame was traced in between).
    void Reset();
    // Returns the lit view plus the accumulated indirect light (shader-read), or a null view.
    vk::ImageView Run(vk::CommandBuffer cmd, const GiInput& in);

private:
    struct Img {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::ImageView view{};
        bool general = false; // in VK_IMAGE_LAYOUT_GENERAL since its creation
    };
    struct Pass {
        vk::DescriptorSetLayout set{};
        vk::PipelineLayout layout{};
        vk::Pipeline pipe{};
    };
    bool MakeImage(Img& img, vk::Extent2D extent, vk::Format format);
    void FreeImage(Img& img);
    bool MakeImages(vk::Extent2D extent);
    bool MakePass(std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
                  std::span<const u32> spirv, Pass& pass);
    void FreePass(Pass& pass);

    const Vulkan::Instance& instance;
    bool ready = false;
    vk::Sampler sampler{};
    Pass trace_pass;
    Pass accum_pass;
    Img raw;
    Img zcur;
    Img final_img;
    std::array<Img, 2> hist{};
    std::array<Img, 2> zhist{};
    vk::Extent2D size{};
    bool history_valid = false;
    u32 slot = 0;
    u32 frame_index = 0;
    float prev_view[12]{};
    float prev_proj[16]{};
};

} // namespace BB::PathTrace
