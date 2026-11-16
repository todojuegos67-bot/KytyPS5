// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include "common/types.h"

namespace BB::SsrProbe {

// BB_SSR_PROBE=1 (default off): detours entry and return of the SSR draw function (eboot 1.09
// vaddr 0xea8f00) and logs the GPU draws whose PM4 packets the guest wrote during it.
void OnEbootPatched(uintptr_t eboot_base, u64 image_size);

bool Enabled();
// GPU thread: start of a graphics submission, then the PM4 header of each draw/dispatch.
void OnSubmit();
void OnPacket(const void* header);
// Rasterizer: one line describing the draw just recorded (only called when Enabled()).
void OnDraw(const std::string& desc);

} // namespace BB::SsrProbe
