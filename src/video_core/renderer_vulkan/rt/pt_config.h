// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Path tracing master switch and quality presets for our port of the path tracing mod by
// Joao Vitor Alves (GPL-2.0, https://github.com/Joaovitor404silva/bloodborne-pc-path-tracing).
//
//   BB_PATHTRACING=1                      master switch (default off: nothing below runs).
//   BB_PATHTRACING_QUALITY=low|medium|high|ultra  cost preset (default low).
//
// A preset only fills BB_PT_* knobs the user has NOT set, so any BB_PT_* in the environment (or
// in the BB_PT_CFG live file, which the engine reads before the environment) still wins. The
// table is PtPresetTable() below and documents/pathtracing/PATH_TRACING.md.
//
// The base set (every preset) is what the mod's own launcher passed (launcher-fork patch,
// RT_ENV): it defines the mod's look (direct + GI on, fill light off, fog volumes off...).
#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Vulkan::Rt {

enum class PtQuality : unsigned { Low = 0, Medium = 1, High = 2, Ultra = 3 };

struct PtKnob {
    const char* name;
    const char* value;
};

inline PtQuality PtQualityFromString(const char* value) {
    if (!value || !*value) {
        return PtQuality::Low;
    }
    const std::string_view v{value};
    if (v == "ultra" || v == "3") {
        return PtQuality::Ultra;
    }
    if (v == "high" || v == "2") {
        return PtQuality::High;
    }
    if (v == "medium" || v == "med" || v == "1") {
        return PtQuality::Medium;
    }
    return PtQuality::Low;
}

inline const char* PtQualityName(PtQuality q) {
    return q == PtQuality::Ultra    ? "ultra"
           : q == PtQuality::High   ? "high"
           : q == PtQuality::Medium ? "medium"
                                    : "low";
}

// Look-defining defaults of the mod's launcher (identical for every preset).
inline constexpr std::array<PtKnob, 13> PtBaseKnobs{{
    {"BB_PT_DIRECT", "1"},
    {"BB_PT_GI", "1"},
    {"BB_PT_COMPACT", "0"},
    {"BB_PT_FILL_LIGHT", "off"},
    {"BB_PT_FILL_ENCLOSED", "0.1"},
    {"BB_PT_SKIN_ALPHA", "1"},
    {"BB_PT_FX_FOG", "0"},
    {"BB_PT_FX_FOG_G", "0"},
    {"BB_PT_GI_LAST_BOUNCE_CACHE", "1"},
    {"BB_PT_GI_SER", "1"},
    {"BB_PT_GI_PIPELINE", "1"},
    {"BB_PT_RESTIR_GI", "1"},
    {"BB_PT_KEEP_GI", "1"},
}};

// Cost knobs per preset. Columns: low, medium, high, ultra. "high" is the mod as its launcher
// shipped it (the engine defaults) plus reflections' wider roughness range; "low" is the cheapest
// setting that keeps traced moon/lamp light, one GI bounce and mirror-like reflections (wet
// floors, puddles, metal); "ultra" spends more rays per pixel for less noise.
// bb-pathtracing-2: low was ~5-6 ms GPU on an RTX 5080 at 1708x960 (run_pt1: build 0.7, GI 2.4,
// direct 2.2); its direct rays, BVH budgets and trace distances are trimmed below.
struct PtPresetRow {
    const char* name;
    std::array<const char*, 4> value; // low, medium, high, ultra
    const char* what;
};

inline constexpr std::array<PtPresetRow, 23> PtPresetTable{{
    // Trace resolution: ReSTIR GI traces 1 of N pixels per frame (4 = 2x2 block, quarter rate;
    // 2 = 2x1, half; 1 = every pixel). The reservoirs and the denoiser fill the rest.
    {"BB_PT_GI_SPARSE", {"4", "2", "1", "1"}, "GI trace rate: 1 pixel of 4 / 2 / 1"},
    {"BB_PT_SPP", {"1", "1", "1", "2"}, "GI paths per traced pixel"},
    {"BB_PT_BOUNCES", {"1", "2", "2", "3"}, "GI bounces"},
    {"BB_PT_BOUNCE_RAYS", {"1", "1", "1", "2"}, "shadow rays per bounce hit"},
    {"BB_PT_GI_TMAX", {"25", "60", "0", "0"}, "GI ray max distance in metres (0 = unlimited)"},
    {"BB_PT_GI_DENOISE", {"2", "2", "2", "3"}, "GI denoiser passes (built-in spatial filter)"},
    {"BB_PT_REFLECT", {"1", "1", "1", "1"}, "traced reflections (puddles, metal, glass, water)"},
    {"BB_PT_REFLECT_MAX_ROUGH", {"0.25", "0.4", "0.5", "0.6"}, "roughest surface that gets a reflection ray"},
    {"BB_PT_RESTIR", {"0", "0", "1", "1"}, "ReSTIR DI for direct light"},
    {"BB_PT_DIRECT_RAYS", {"2", "8", "32", "48"}, "shadow rays per pixel for the local lights (strongest first)"},
    {"BB_PT_SHADOW_LIGHTS", {"2", "3", "4", "6"}, "local lights with traced shadows (shadow-mask pass)"},
    {"BB_PT_LIGHT_SOURCE", {"0.02", "0.02", "0.02", "0.02"}, "lamp / fire source radius as a share of the light's range (soft shadows)"},
    {"BB_PT_AO", {"0", "0", "0", "0"}, "traced AO pass (the game's AO is kept)"},
    {"BB_PT_FAR_START", {"50", "120", "0", "0"}, "beyond this many metres (+10 m fade) no rays: stock"},
    {"BB_PT_KEEP_DIST", {"20", "40", "40", "60"}, "off-screen instances kept in the BVH within (m)"},
    {"BB_PT_MIN_TRIS", {"8", "2", "0", "0"}, "opaque draws under this many triangles stay out of the BVH"},
    {"BB_PT_REFIT_BUDGET", {"131072", "524288", "0", "0"}, "skinned BLAS refit triangles per frame (0 = unlimited)"},
    {"BB_PT_REBUILD_BUDGET", {"393216", "1048576", "1048576", "2097152"}, "BLAS rebuild triangles per frame"},
    {"BB_PT_STORE_MVERTS", {"8", "10", "12", "14"}, "geometry store, M vertices a frame (x2, 28 B each)"},
    {"BB_PT_MIST_STEPS", {"4", "8", "8", "12"}, "water mist ray-march steps"},
    {"BB_PT_MOON_HALF", {"2", "2", "1", "1"}, "moon rays: 2 = half the pixels + shared penumbra, 1 = all"},
    {"BB_PT_ADAPTIVE_MS", {"12", "12", "12", "16"}, "frame-time budget: GI bounces drop above it"},
    {"BB_PT_DLSS_RR", {"auto", "auto", "auto", "auto"}, "DLSS Ray Reconstruction (auto: on with DLSS and nvngx_dlssd.dll)"},
}};

namespace detail {
inline bool PtSetDefault(const char* name, const char* value) {
    if (const char* have = std::getenv(name); have && *have) {
        return false;
    }
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 0) == 0;
#endif
}
} // namespace detail

// Fills the unset BB_PT_* knobs for BB_PATHTRACING_QUALITY, once. Returns the preset used.
inline PtQuality ApplyPtPreset() {
    static const PtQuality quality = [] {
        const PtQuality q = PtQualityFromString(std::getenv("BB_PATHTRACING_QUALITY"));
        for (const PtKnob& k : PtBaseKnobs) {
            detail::PtSetDefault(k.name, k.value);
        }
        for (const PtPresetRow& row : PtPresetTable) {
            detail::PtSetDefault(row.name, row.value[unsigned(q)]);
        }
        return q;
    }();
    return quality;
}

// F9 in game (sdl_window.cpp, only with BB_PATHTRACING=1): paused = stock lighting, and no path
// tracing GPU work at all (vk_ray_tracing.cpp FinishFrame skips the BVH build too).
inline std::atomic<bool> effects_paused{false};

inline bool ToggleEffects() {
    const bool paused = !effects_paused.load(std::memory_order_relaxed);
    effects_paused.store(paused, std::memory_order_relaxed);
    return !paused;
}

inline bool PtMasterSwitch() {
    static const bool on = [] {
        const char* v = std::getenv("BB_PATHTRACING");
        return v && v[0] == '1';
    }();
    return on;
}

} // namespace Vulkan::Rt
