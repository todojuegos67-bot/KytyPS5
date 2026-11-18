// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Path tracing master switch and quality presets for our port of the path tracing mod by
// Joao Vitor Alves (GPL-2.0, https://github.com/Joaovitor404silva/bloodborne-pc-path-tracing).
//
//   BB_PATHTRACING=1                      master switch (default off: nothing below runs).
//   BB_PATHTRACING_QUALITY=low|medium|high  cost preset (default low).
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

enum class PtQuality : unsigned { Low = 0, Medium = 1, High = 2 };

struct PtKnob {
    const char* name;
    const char* value;
};

inline PtQuality PtQualityFromString(const char* value) {
    if (!value || !*value) {
        return PtQuality::Low;
    }
    const std::string_view v{value};
    if (v == "high" || v == "2") {
        return PtQuality::High;
    }
    if (v == "medium" || v == "med" || v == "1") {
        return PtQuality::Medium;
    }
    return PtQuality::Low;
}

inline const char* PtQualityName(PtQuality q) {
    return q == PtQuality::High ? "high" : q == PtQuality::Medium ? "medium" : "low";
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

// Cost knobs per preset. Columns: low, medium, high. "high" is the mod as its launcher shipped
// it (the engine defaults), "low" is the cheapest setting that keeps traced moon/lamp light and
// one bounce.
struct PtPresetRow {
    const char* name;
    std::array<const char*, 3> value; // low, medium, high
    const char* what;
};

inline constexpr std::array<PtPresetRow, 17> PtPresetTable{{
    // Trace resolution: ReSTIR GI traces 1 of N pixels per frame (4 = 2x2 block, quarter rate;
    // 2 = 2x1, half; 1 = every pixel). The reservoirs and the denoiser fill the rest.
    {"BB_PT_GI_SPARSE", {"4", "2", "1"}, "GI trace rate: 1 pixel of 4 / 2 / 1"},
    {"BB_PT_SPP", {"1", "1", "1"}, "GI paths per traced pixel"},
    {"BB_PT_BOUNCES", {"1", "2", "2"}, "GI bounces"},
    {"BB_PT_BOUNCE_RAYS", {"1", "1", "1"}, "shadow rays per bounce hit"},
    {"BB_PT_GI_TMAX", {"30", "60", "0"}, "GI ray max distance in metres (0 = unlimited)"},
    {"BB_PT_REFLECT", {"0", "1", "1"}, "traced reflections (water, puddles, glass)"},
    {"BB_PT_RESTIR", {"0", "0", "1"}, "ReSTIR DI for direct light"},
    {"BB_PT_DIRECT_RAYS", {"4", "12", "32"}, "shadow rays per pixel for the light list"},
    {"BB_PT_SHADOW_LIGHTS", {"2", "3", "4"}, "local lights with traced shadows per pixel"},
    {"BB_PT_AO", {"0", "0", "0"}, "traced AO pass (the game's AO is kept)"},
    {"BB_PT_FAR_START", {"60", "120", "0"}, "beyond this many metres (+10 m fade) no rays: stock"},
    {"BB_PT_KEEP_DIST", {"25", "40", "40"}, "off-screen instances kept in the BVH within (m)"},
    {"BB_PT_MIN_TRIS", {"4", "2", "0"}, "opaque draws under this many triangles stay out of the BVH"},
    {"BB_PT_REFIT_BUDGET", {"262144", "524288", "0"}, "skinned BLAS refit triangles per frame (0 = unlimited)"},
    {"BB_PT_REBUILD_BUDGET", {"524288", "1048576", "1048576"}, "BLAS rebuild triangles per frame"},
    {"BB_PT_MIST_STEPS", {"4", "8", "8"}, "water mist ray-march steps"},
    {"BB_PT_ADAPTIVE_MS", {"12", "12", "12"}, "frame-time budget: GI bounces drop above it"},
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
