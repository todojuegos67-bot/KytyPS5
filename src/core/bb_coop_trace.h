// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include "common/types.h"

namespace Core::BBCoop {

/// BB_COOP=1 only: observer hooks on Bloodborne 1.09's HTTP manager (see the .cpp).
void InstallHttpTrace(uintptr_t base, u64 size);

} // namespace Core::BBCoop
