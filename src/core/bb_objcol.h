// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// BB_OBJCOL_DIAG=1: object collision diagnostics (log only, the game's data and code paths are
// unchanged apart from two logging detours at the hknpWorld add / remove entries).
//   File side: every obj/oXXXXXX.objbnd.dcx the game opens is read from the host disk on a
//   worker thread (DCX DFLT -> BND4): its .hkx entries and whether they hold Havok physics
//   (static solid) or breakable / destruction data.
//   Game side: hknpWorld::addBodies / removeBodies (1.09 eboot+0xb55b20 / +0xb56940, found by
//   signature): every body the game adds or removes, its motion type, layer, filter, position and
//   the objbnd it came from (the shape lives inside the binder the game keeps in memory).
//   Summary per map: solid-expected objects, objects with collision in the Havok world, and the
//   ones missing (never added / added and removed / created but not in the world).

#pragma once

#include "common/types.h"

namespace BB::ObjCol {

bool Enabled();
void OnGuestFileOpen(const char* guest_path, const char* host_path, u64 size);
void OnEbootPatched(uintptr_t eboot_base, u64 image_size);

} // namespace BB::ObjCol
