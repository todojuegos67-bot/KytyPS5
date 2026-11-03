// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

namespace BB::Texture {

// BB_TEX_QUALITY (launcher "Calidad de texturas"): 0 Bajo, 1 Normal (game's own bias, no change),
// 2 Alto, 3 Ultra. Added to the mip LOD bias of every colour sampler; Bajo samples smaller mips.
inline float LodBiasOffset() {
    static const float offset = [] {
        const char* v = std::getenv("BB_TEX_QUALITY");
        switch (v != nullptr ? v[0] : '1') {
        case '0':
            return 1.0f;
        case '2':
            return -0.5f;
        case '3':
            return -1.0f;
        default:
            return 0.0f;
        }
    }();
    return offset;
}

} // namespace BB::Texture
