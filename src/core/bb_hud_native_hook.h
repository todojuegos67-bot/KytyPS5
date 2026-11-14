// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-hud-native (BB_HUD_NATIVE=1): brackets the game's Scaleform UI rendering in the GPU command
// stream. See bb_hud_native_hook.cpp for the eboot evidence.

#pragma once

#include <atomic>
#include <utility>
#include <vector>
#include "common/types.h"

namespace BB::HudNativeHook {

// PM4 NOP payloads written into the UI's draw command buffer (dword 1 of a 3-dword NOP).
constexpr u32 MarkerBegin = 0x6875B0B0u; // after HAL::BeginScene succeeded (before any UI draw)
constexpr u32 MarkerEnd = 0x6875B0B1u;   // after HAL::EndScene (its batches flushed)

// Guest-side counters (the GPU thread logs and resets nothing here; it reads deltas).
struct Stats {
    static std::atomic<u64> begins, begin_failed, ends, written, no_space, no_dcb;
};

// BB_HUD_NATIVE=1 (read once).
bool Enabled();
// The vtable wrappers are in place (after relocation).
bool Installed();

// After the eboot's relocations were applied (its vtables hold absolute addresses).
void OnEbootRelocated(uintptr_t eboot_base, const std::vector<std::pair<u64, u64>>& segments);

} // namespace BB::HudNativeHook
