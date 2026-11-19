// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-pathtrace: a path tracer over the game's own geometry. Ported from the Bloodborne PC path
// tracing mod (Joaovitor404silva/bloodborne-pc-path-tracing, GPL-2.0): its capture (per-vertex
// clip positions of the scene draws), gather into view-space triangles, acceleration structure
// and ray queries. Independent of bb_rt_reflect / bb_volfog / bb_mv: they are not touched.
//
// BB_PT=1 turns it on (it needs ray query and acceleration structures). Unset or 0: no pipeline
// variant, buffer, pass or hook does anything.
#pragma once

#include "common/types.h"

namespace BB::PathTrace {

// BB_PT=1.
bool Requested();
// BB_PT_TRIANGLES (2097152, 1024..8388608): triangle capacity per frame. Draws past it are
// skipped and counted in the periodic log line.
u32 TriangleCapacity();
// BB_PT_RANGE (50, metres): the depth at which the distance view fades to black.
float ViewRange();

} // namespace BB::PathTrace
