// SPDX-FileCopyrightText: Joao Vitor Alves (path tracing for Bloodborne PC)
// SPDX-License-Identifier: GPL-2.0-only
// From https://github.com/Joaovitor404silva/bloodborne-pc-path-tracing (commit 04d398f), written
// for SuperMedo's Bloodborne PC port; ported to this renderer (BB_PT_* knob names, include paths).
// See documents/pathtracing/PATH_TRACING.md and LICENSES/PathTracing-NOTICE.txt.
#pragma once

namespace Vulkan::Rt {

struct HitchSample {
    double cpu_ms = 0.0;
    double wait_ms = 0.0;
    double gpu_ms = 0.0;
};

inline bool IsHitch(const HitchSample& s, double threshold_ms) {
    return s.cpu_ms > threshold_ms || s.wait_ms > 2.0;
}

}
