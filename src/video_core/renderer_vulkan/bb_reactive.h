// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reactive: the inputs of the upscalers' reactive / transparency mask.
//
// Transparency: the scene colour target is copied right before the first "transparent" draw of
// a run (jittered scene draw, depth test on, depth write off, blending on: fog, particles,
// smoke, effects) and again when the run ends (the next draw that is not one). The difference
// of the two copies is what the transparent draws added (the FFX "auto reactive" idea, opaque
// only vs final colour). The last run of the frame wins (deferred lighting into another target
// earlier in the frame is replaced by the later effect pass).
//
// Switches: BB_REACTIVE=0 turns the transparency part off (no copies at all).
#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/texture_cache.h"
#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace BB::Reactive {

bool TransparencyOn(); // BB_REACTIVE != 0
bool SelfMotionOn();   // BB_SELFMOTION != 0
bool AnyOn();          // either (the mask pass runs)

// Called by the rasterizer for every draw with an upscaler on, before its resources are bound.
// `scene_draw`: a jittered scene draw (scene-sized depth with the depth test on, not UI).
void OnDraw(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
            Vulkan::Runtime& runtime, VideoCore::TextureCache& texture_cache,
            VideoCore::ImageId color, bool scene_draw, bool depth_write, bool blend,
            bool opaque_seen);

struct Snapshot {
    bool valid = false;
    vk::ImageView before{}, after{};
    vk::Extent2D extent{};
    u32 draws = 0;  // transparent draws in the captured runs
    u32 copies = 0; // snapshot copies this frame
    u32 runs = 0;   // runs (restarts on another target count anew)
};
// The frame's snapshot (images in SHADER_READ_ONLY_OPTIMAL), then the state starts over. A run
// still open at the present uses its last closed copy (none: no transparency mask this frame).
Snapshot Take();

} // namespace BB::Reactive
