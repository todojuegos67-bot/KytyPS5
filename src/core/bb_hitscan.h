// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"

// (bb-hitscan-150) Bloodborne 1.09 diagnostic, read-only: finds the guest objects that change
// when the player's attacks connect (for a later BB_RUMBLE_ONHIT). Insert arms:
// guest heap objects with an eboot vtable are collected and their +0x224 float ("damage last
// frame" per GetDamageLastFrame, 0x125cae0) is polled; Home stops and logs
// candidates and the int fields that decreased. Every guest read goes through ReadProcessMemory
// (a failed read is skipped, never a fault); nothing is written to guest memory. Windows only.
namespace BB::HitScan {
void SetEboot(uintptr_t base, u64 size);
void Arm();
void Stop();
} // namespace BB::HitScan
