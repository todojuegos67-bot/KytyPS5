// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "video_core/renderer_vulkan/bb_pt.h"

namespace BB::PathTrace {

namespace {

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

} // namespace BB::PathTrace
