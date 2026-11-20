// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include "common/types.h"

// Bloodborne 1.09 (CUSA03173) cheats for the F1 menu "Trucos" section. Everything is off by
// default and lives only in memory; with every cheat off nothing is written to the game.
// Addresses and evidence are in bb_cheats.cpp; each site is byte-checked at eboot load and the
// whole module stays off on any other eboot.
namespace BB::Cheats {

enum class Cheat : u32 {
    InfiniteHp,      // HP = max HP each frame + the game's PlayerNoDead debug flag
    InfiniteStamina, // stamina = max stamina each frame
    InfiniteItems,   // the player's "no goods consume" chr flag (vials, bullets, consumables)
    OneHitKill,      // the game's PlayerExterminate debug flag
    Count,
};

struct Stats {
    bool in_game = false; // local player and PlayerGameData found
    s32 hp = 0, hp_max = 0, stamina = 0, stamina_max = 0, echoes = 0, insight = 0;
};

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);
// Guest thread, once per game frame (next to VarDt::OnGuestFlip).
void OnGuestFlip();

bool Available(); // the 1.09 eboot was recognised
bool Get(Cheat c);
void Set(Cheat c, bool on);
// Queued, applied on the next guest frame (only while in game). Clamped as the game clamps:
// echoes 0..999,999,999, insight 0..99.
void AddEchoes(s32 n);
void AddInsight(s32 n);
Stats Read(); // snapshot for the menu (any thread)
// PlayerGameData of the loaded save (player and its data module validated); false: not in game.
bool PlayerGameData(u64& pgd);

} // namespace BB::Cheats
