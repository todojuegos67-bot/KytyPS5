// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Presets of the path tracing port (pt_config.h): parsing, user values win, one value per row.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "video_core/renderer_vulkan/rt/pt_config.h"

using namespace Vulkan::Rt;

int main() {
    assert(PtQualityFromString(nullptr) == PtQuality::Low);
    assert(PtQualityFromString("") == PtQuality::Low);
    assert(PtQualityFromString("bogus") == PtQuality::Low);
    assert(PtQualityFromString("medium") == PtQuality::Medium);
    assert(PtQualityFromString("high") == PtQuality::High);
    assert(PtQualityFromString("ultra") == PtQuality::Ultra);
    assert(std::strcmp(PtQualityName(PtQuality::Ultra), "ultra") == 0);
    for (size_t i = 0; i < PtPresetTable.size(); ++i) {
        for (size_t j = i + 1; j < PtPresetTable.size(); ++j) {
            assert(std::strcmp(PtPresetTable[i].name, PtPresetTable[j].name) != 0);
        }
    }
    for (const auto& row : PtPresetTable) {
        for (const char* v : row.value) {
            assert(v && *v);
        }
        for (const auto& base : PtBaseKnobs) {
            assert(std::strcmp(row.name, base.name) != 0); // no knob in both tables
        }
    }
    setenv("BB_PATHTRACING_QUALITY", "medium", 1);
    setenv("BB_PT_BOUNCES", "5", 1); // the user's value
    unsetenv("BB_PT_GI_SPARSE");
    assert(ApplyPtPreset() == PtQuality::Medium);
    assert(std::strcmp(std::getenv("BB_PT_BOUNCES"), "5") == 0);
    assert(std::strcmp(std::getenv("BB_PT_GI_SPARSE"), "2") == 0);
    assert(std::strcmp(std::getenv("BB_PT_GI"), "1") == 0);
    assert(ToggleEffects() == false && effects_paused.load());
    assert(ToggleEffects() == true && !effects_paused.load());
    std::printf("pt_config: ok\n");
    return 0;
}
