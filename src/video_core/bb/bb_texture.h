// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-tex (BB_TEX_QUALITY / BB_ANISO): the launcher's "Calidad de texturas" and "Filtrado
// anisotropico" rows. Both are read once per process. They only touch the samplers of scene
// draws (Rasterizer::BbLodBias: depth test on, the scene's depth target, before the HUD), never
// depth / compare samplers, compute passes, the HUD or post-processing.
//
// BB_TEX_QUALITY: mip LOD bias offset (added to the game's own bias and to the upscaler's
// automatic log2(render / output) bias) and anisotropy:
//   0 Bajo        +1.0   1x
//   1 Normal       0.0   game's own (the PS4 look)
//   2 Alto        -0.5   8x
//   3 Ultra       -1.0   16x
//   4 Ultra bajo  +2.0   1x
//   5 Extra       -1.25 with a temporal upscaler (-1.0 without), 16x, trilinear instead of the
//                 game's point-mip filter, and alpha-tested (discard) / blended draws held at
//                 -0.5 so foliage, hair and fences do not shimmer.
// A forced ratio applies to every linear-filtered scene sampler, also those the game left at
// 1x (as the earlier BB_TEX_QUALITY / BB_ANISO did for all colour samplers).
//   Unset / other: no offset, game's anisotropy.
// BB_ANISO: 1, 4, 8 or 16, explicit ratio; wins over the quality row's.
// BB_TEX_LOD_EXTRA (set by the launcher when upscaling): ignored on this branch, the automatic
// upscaler bias (BB_LOD_BIAS=auto) already applies log2(render / output); adding it would count
// it twice.
#pragma once

#include <array>
#include <cstdlib>
#include <string>
#include "common/logging/log.h"
#include "common/types.h"

namespace BB::Texture {

struct Settings {
    int level = -1;             // BB_TEX_QUALITY 0..5, -1 unset
    float lod_offset = 0.0f;    // opaque scene draws
    float lod_offset_tu = 0.0f; // the same with a temporal upscaler running
    float lod_offset_alpha = 0.0f; // alpha-tested / blended scene draws
    u32 aniso = 0; // forced ratio on linear-filtered scene samplers, 0 = the game's
    bool trilinear = false;     // point-mip -> linear-mip on linear-filtered scene samplers
};

inline const Settings& Get() {
    static const Settings s = [] {
        static constexpr std::array<float, 6> lod{1.0f, 0.0f, -0.5f, -1.0f, 2.0f, -1.0f};
        static constexpr std::array<u32, 6> an{1, 0, 8, 16, 1, 16};
        static constexpr std::array<const char*, 6> names{"Bajo",  "Normal",     "Alto",
                                                          "Ultra", "Ultra bajo", "Extra"};
        Settings r{};
        const char* q = std::getenv("BB_TEX_QUALITY");
        if (q != nullptr && q[0] >= '0' && q[0] <= '5' && q[1] == '\0') {
            r.level = q[0] - '0';
            r.lod_offset = lod[r.level];
            r.lod_offset_tu = r.level == 5 ? -1.25f : lod[r.level];
            r.lod_offset_alpha = r.level == 5 ? -0.5f : lod[r.level];
            r.aniso = an[r.level];
            r.trilinear = r.level == 5;
        }
        if (const char* a = std::getenv("BB_ANISO"); a != nullptr) {
            const long v = std::strtol(a, nullptr, 10);
            if (v == 1 || v == 4 || v == 8 || v == 16) r.aniso = u32(v);
        }
        const bool extra_set = std::getenv("BB_TEX_LOD_EXTRA") != nullptr;
        if (r.level >= 0) {
            LOG_INFO(Render_Vulkan,
                     "[BB_TEX] Calidad de texturas: {} (BB_TEX_QUALITY={}): LOD {:+.2f} (con "
                     "reescalador temporal {:+.2f}, alpha-test / mezcla {:+.2f}), aniso {}{}; "
                     "solo dibujos de escena{}",
                     names[r.level], r.level, r.lod_offset, r.lod_offset_tu, r.lod_offset_alpha,
                     r.aniso ? fmt::format("{}x", r.aniso) : std::string("del juego"),
                     r.trilinear ? ", trilineal forzado" : "",
                     extra_set ? "; BB_TEX_LOD_EXTRA ignorado (el sesgo automatico del "
                                 "reescalado ya lo aplica)"
                               : "");
        } else {
            LOG_INFO(Render_Vulkan,
                     "[BB_TEX] Calidad de texturas: sin BB_TEX_QUALITY (juego por defecto), aniso "
                     "{}",
                     r.aniso ? fmt::format("{}x (BB_ANISO)", r.aniso) : std::string("del juego"));
        }
        return r;
    }();
    return s;
}

// Sampler filter override passed to the texture cache (part of the sampler key).
// Bits 0-4: anisotropy ratio forced on every linear-filtered sampler (0 = the game's own),
// bit 6: point-mip -> linear-mip on linear-filtered samplers.
inline u32 FilterBits() {
    const auto& s = Get();
    return (s.aniso & 31u) | (s.trilinear ? 64u : 0u);
}

} // namespace BB::Texture
