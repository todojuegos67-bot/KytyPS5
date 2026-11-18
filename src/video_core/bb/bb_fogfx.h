// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-fogfx (BB_FOG_FX, launcher "Luces volumetricas y niebla"): drives the volumetric pass of
// bb_rt_reflect (bb_rt_vol.comp, run before the game's composite on the lit scene, before the
// upscaler and the HUD) from the game's own fog and light set:
//   - fog CB slot 6 (FUN_02651f10, FOG MANAGER param): density 1/dist at height 0, height
//     falloff 1/H, colour (rgb^2), camera height -> extra height / distance fog whose optical
//     depth is a multiple of the game's;
//   - light-set CB slot 5 (FUN_026deeb0): the moon / sun direction and colour -> ray-traced
//     shafts with a forward-scattering phase (glow around the moon);
//   - the lit scene itself: lamps and other bright emitters scatter a halo into the fog in
//     front of them, scaled by the fog optical depth of each pixel's own view ray (so a near
//     object in front of a lamp is not haloed);
//   - world-space value-noise density, drifting slowly with a fixed wind.
// BB_FOG_FX: 0 off, 1 sutil, 2 espectacular, 3 maxima. Unset: BB_VOL=1 (the launcher's existing
// "Luces volumetricas y niebla (experimental)" box) means 2, else off. BB_RT_VOLUMETRIC=1 alone
// keeps the older light-only pass (level 0 effects). Needs ray query (VK_KHR_ray_query).
#pragma once

#include <array>
#include <cstdlib>
#include "common/types.h"

namespace BB::FogFx {

struct Preset {
    int level;           // 0..3
    const char* name;
    float extra_od;      // extra fog optical depth, x the game's along the same ray
    float extra_max;     // cap on the extra fog's opacity
    float strength;      // in-scattering (shafts) strength
    u32 steps;           // march steps (half res)
    float march;         // march distance (game units / m)
    float noise;         // density noise amplitude (0..1)
    float halo;          // lamp / emitter halo strength
    float g;             // Henyey-Greenstein forward scattering
};

inline int Level() {
    static const int level = [] {
        if (const char* e = std::getenv("BB_FOG_FX"); e && e[0] >= '0' && e[0] <= '3') {
            return e[0] - '0';
        }
        if (const char* e = std::getenv("BB_VOL"); e && e[0] == '1') return 2;
        return 0;
    }();
    return level;
}

// The volumetric pass is wanted: BB_RT_VOLUMETRIC=1 or a BB_FOG_FX level (or BB_VOL=1).
inline bool VolRequested() {
    static const bool on = [] {
        const char* e = std::getenv("BB_RT_VOLUMETRIC");
        return (e && e[0] == '1') || Level() > 0;
    }();
    return on;
}

inline const Preset& Get() {
    static constexpr std::array<Preset, 4> presets{{
        {0, "off (BB_RT_VOLUMETRIC: light only)", 0.0f, 0.0f, 0.5f, 12, 60.0f, 0.0f, 0.0f, 0.55f},
        {1, "sutil", 0.35f, 0.30f, 0.35f, 10, 60.0f, 0.35f, 0.12f, 0.60f},
        {2, "espectacular", 0.80f, 0.45f, 0.60f, 14, 80.0f, 0.55f, 0.25f, 0.65f},
        {3, "maxima", 1.40f, 0.60f, 0.90f, 20, 100.0f, 0.70f, 0.40f, 0.70f},
    }};
    return presets[Level()];
}

} // namespace BB::FogFx
