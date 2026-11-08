// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include "common/types.h"

// Bloodborne 1.09: the game's render size immediates rewritten to output / upscaler ratio.
namespace BB::ResPatch {
void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
} // namespace BB::ResPatch
