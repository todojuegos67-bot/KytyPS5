// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>

namespace BB::Texture {

// Texture quality settings for colour samplers. Every value comes from the environment and is
// read once per process, so a sampler created in this run always matches the others.
//
// BB_TEX_QUALITY (launcher "Calidad de texturas"): 0 Bajo, 1 Normal, 2 Alto, 3 Ultra.
//   Sets the mip LOD bias offset and the anisotropy ratio together:
//     Bajo   LOD +1.0  aniso 1
//     Normal  0.0      aniso 4
//     Alto   -0.5      aniso 8
//     Ultra  -1.0      aniso 16
//   Unset (or a value other than 0..3): LOD unchanged and aniso left to the game.
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
        static constexpr std::array<float, 4> lod_bias{1.0f, 0.0f, -0.5f, -1.0f};
        static constexpr std::array<float, 4> aniso{1.0f, 4.0f, 8.0f, 16.0f};

        const char* quality_env = std::getenv("BB_TEX_QUALITY");
        const bool quality_set =
            quality_env != nullptr && quality_env[0] >= '0' && quality_env[0] <= '3';
        const int row = quality_set ? quality_env[0] - '0' : 1; // Normal

        float ratio = quality_set ? aniso[row] : 0.0f;
        if (const char* aniso_env = std::getenv("BB_ANISO"); aniso_env != nullptr) {
            const long value = std::strtol(aniso_env, nullptr, 10);
            if (value == 1 || value == 4 || value == 8 || value == 16) {
                ratio = static_cast<float>(value);
            }
        }

        return Settings{
            .lod_bias_offset = quality_set ? lod_bias[row] : 0.0f,
            .aniso = ratio,
        };
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
