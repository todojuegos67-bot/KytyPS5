// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace BB::RumbleHits {

// Detours the three "player attack landed" calls of the attack-hit dispatcher (eboot 1.09,
// vaddr 0x185b200) when BB_RUMBLE_HITS is on (default). See bb_rumble_hits.cpp.
void OnEbootPatched(uintptr_t eboot_base, u64 image_size);

} // namespace BB::RumbleHits
