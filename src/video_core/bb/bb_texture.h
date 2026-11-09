// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include "common/logging/log.h"

namespace BB::Texture {

// Texture quality settings for colour samplers. Every value comes from the environment and is
// read once per process, so a sampler created in this run always matches the others.
//
// BB_TEX_QUALITY (launcher "Calidad de texturas"): 0 Bajo, 1 Normal, 2 Alto, 3 Ultra, 4 Ultra bajo.
//   Sets the mip LOD bias offset and the anisotropy ratio together:
//     Bajo        LOD +1.0  aniso 1
//     Normal       0.0      aniso 4
//     Alto        -0.5      aniso 8
//     Ultra      -1.0      aniso 16
//     Ultra bajo  +2.0      aniso 1   (one more step below Bajo, same step size as Normal->Bajo;
//                                      anisotropy is already off at Bajo)
//   Unset (or a value other than 0..4): LOD unchanged and aniso left to the game.
//   Values 0..3 keep their meaning, so existing launcher.ini files are unaffected.
// BB_ANISO: 1, 4, 8 or 16. Explicit anisotropy; wins over the value implied by BB_TEX_QUALITY.
//   Any other value is ignored.
//
// When neither variable is set, the game's own anisotropy request is used and the LOD bias is
// unchanged, so the default behaviour is identical to the build without these settings.
struct Settings {
    float lod_bias_offset; // Added to the LOD bias of colour samplers only.
    float aniso;           // 0 = each sampler keeps the game's requested ratio.
};

inline const Settings& Get() {
    static const Settings settings = [] {
        static constexpr std::array<float, 5> lod_bias{1.0f, 0.0f, -0.5f, -1.0f, 2.0f};
        static constexpr std::array<float, 5> aniso{1.0f, 4.0f, 8.0f, 16.0f, 1.0f};
        static constexpr std::array<const char*, 5> names{"Bajo", "Normal", "Alto", "Ultra",
                                                          "Ultra bajo"};

        const char* quality_env = std::getenv("BB_TEX_QUALITY");
        const bool quality_set =
            quality_env != nullptr && quality_env[0] >= '0' && quality_env[0] <= '4';
        // Rows 0..4 map to the table index; the table starts at Bajo (0) and ends at Ultra bajo (4).
        const int row = quality_set ? quality_env[0] - '0' : 1; // Normal

        float ratio = quality_set ? aniso[row] : 0.0f;
        if (const char* aniso_env = std::getenv("BB_ANISO"); aniso_env != nullptr) {
            const long value = std::strtol(aniso_env, nullptr, 10);
            if (value == 1 || value == 4 || value == 8 || value == 16) {
                ratio = static_cast<float>(value);
            }
        }

        // BB_TEX_LOD_EXTRA (0.53): extra colour LOD bias for upscaling, log2(internal / output
        // height) as DLSS / FSR ask, so the mips match the output resolution, not the internal
        // one. The launcher sets it only with an upscaler; clamped to [-3, 0].
        float extra = 0.0f;
        if (const char* extra_env = std::getenv("BB_TEX_LOD_EXTRA"); extra_env != nullptr) {
            extra = std::clamp(static_cast<float>(std::strtod(extra_env, nullptr)), -3.0f, 0.0f);
        }
        const Settings result{
            .lod_bias_offset = (quality_set ? lod_bias[row] : 0.0f) + extra,
            .aniso = ratio,
        };
        if (quality_set) {
            LOG_INFO(Render_Vulkan, "[BB_TEX] Calidad de texturas: {} (BB_TEX_QUALITY={}, LOD bias {:+.1f}, aniso {:.0f}x)",
                     names[row], row, result.lod_bias_offset, ratio);
        } else {
            LOG_INFO(Render_Vulkan, "[BB_TEX] Calidad de texturas: sin BB_TEX_QUALITY (juego por defecto)");
        }
        if (extra != 0.0f) {
            LOG_INFO(Render_Vulkan, "[BB_TEX] LOD extra de reescalado {:+.2f} (BB_TEX_LOD_EXTRA)", extra);
        }
        return result;
    }();
    return settings;
}

// Key for the sampler cache: the settings above, so samplers built under different values never
// share a cache entry.
inline std::uint64_t SettingsKey() {
    const Settings& s = Get();
    return (std::uint64_t{std::bit_cast<std::uint32_t>(s.aniso)} << 32) |
           std::bit_cast<std::uint32_t>(s.lod_bias_offset);
}

} // namespace BB::Texture
