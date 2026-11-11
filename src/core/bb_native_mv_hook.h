// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB native motion vectors: the guest detours (see video_core/bb/bb_native_mv.h). Installed after
// the static CPU patching of the eboot, found by byte signatures (base 1.00 and 1.09 images).

#pragma once

#include "common/types.h"

namespace BB::NativeMvHook {

void OnEbootPatched(uintptr_t eboot_base, u64 image_size);

} // namespace BB::NativeMvHook
