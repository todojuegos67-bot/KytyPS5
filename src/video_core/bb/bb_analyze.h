// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB_ANALYZE: opt-in, log-only discovery of the inputs an upscaler / frame generator needs
// (scene colour, depth, motion vectors, jitter). Set env BB_ANALYZE=1. No behaviour change.
#pragma once
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include "common/logging/log.h"
#include <fmt/ranges.h>
#include "common/types.h"
#include "video_core/texture_cache/image.h"

namespace BB::Analyze {

inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_ANALYZE");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

inline std::atomic<u64>& FrameCounter() {
    static std::atomic<u64> n{0};
    return n;
}

inline const char* Guess(vk::Format f) {
    switch (f) {
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR16G16Snorm:
    case vk::Format::eR16G16Unorm:
        return "MV?";
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR16G16B16A16Sfloat:
        return "HDR-COLOR?";
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eA2B10G10R10UnormPack32:
        return "LDR";
    default:
        return "-";
    }
}

// One line per unique render pass signature (formats + size + addresses).
inline void OnPass(const std::array<const VideoCore::Image*, 8>& colors, u32 num_colors,
                   const VideoCore::Image* depth, u32 w, u32 h) {
    static std::mutex m;
    static std::unordered_set<std::string> seen;
    std::string sig = fmt::format("{}x{}", w, h);
    for (u32 i = 0; i < num_colors; ++i) {
        if (const auto* c = colors[i]) {
            sig += fmt::format(" c{}={}@{:#x}[{}]", i, vk::to_string(c->info.pixel_format),
                               c->info.guest_address, Guess(c->info.pixel_format));
        }
    }
    if (depth) {
        sig += fmt::format(" d={}@{:#x}", vk::to_string(depth->info.pixel_format),
                           depth->info.guest_address);
    }
    std::scoped_lock lk{m};
    if (seen.insert(sig).second) {
        LOG_INFO(Render_Vulkan, "[BB_ANALYZE] frame={} pass {}", FrameCounter().load(), sig);
    }
}

template <size_t N>
inline void OnPipeline(u64 hash, u32 mrt_mask, const std::array<size_t, N>& stages) {
    LOG_INFO(Render_Vulkan, "[BB_ANALYZE] pipeline {:#x} mrt_mask={:#x} stages={:#x}", hash,
             mrt_mask, fmt::join(stages, ","));
}

inline void OnPresent(u32 w, u32 h, vk::Format f, VAddr addr) {
    const u64 n = FrameCounter().fetch_add(1);
    if (n < 4 || (n % 600) == 0) {
        LOG_INFO(Render_Vulkan, "[BB_ANALYZE] present frame={} {}x{} {} @{:#x}", n, w, h,
                 vk::to_string(f), addr);
    }
}

} // namespace BB::Analyze
