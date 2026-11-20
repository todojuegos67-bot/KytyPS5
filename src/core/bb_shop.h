// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "common/types.h"
#include "imgui/imgui_texture.h"

// Bloodborne 1.09: F1 > ARMAS Y OBJETOS. Everything is read at runtime from the game's own files
// (nothing is shipped): gameparam.parambnd.dcx (EquipParamWeapon / EquipParamGoods /
// ShopLineupParam / ItemLotParam), msg/<lang>/item.msgbnd.dcx (names, captions, lore) and
// menu/menu_knowledge_NNNNN.tpf.dcx (icons, BC7, PS4 tiled). Loaded on a worker thread when the
// card opens, freed when it closes; nothing runs while the card has never been opened.
// A purchase is applied on the game thread at the next frame (the game is paused while the
// menu is open): echoes are taken atomically and the item is given by the game's own
// add-item function (the one the Lua AddInventoryItem command uses, with its "item obtained" log).
namespace BB::Shop {

enum class Tab : u8 { Weapons, Items, Tools, Materials, Count };

struct Item {
    s32 id = 0;          // weapon base id (+0) or goods id
    Tab tab = Tab::Items;
    s32 price = 0;       // echoes
    bool game_price = false; // from the game's own shop lineup (else a fixed price, see .cpp)
    u16 icon = 0;
    std::string name, caption, lore; // UTF-8, the game's own text
    ImGui::RefCountedTexture tex;    // empty until uploaded / when the icon could not be read
};

enum class State : u8 { Idle, Loading, Ready, Failed };

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size);

// Menu thread.
void Open();  // starts loading (once per open)
void Close(); // stops loading, frees text and textures
State GetState(std::string* message = nullptr, int* progress = nullptr, int* total = nullptr);
// Snapshot of one tab (items are filled once; textures appear as they are uploaded).
std::vector<const Item*> Items(Tab tab);
// Queue a purchase (applied on the next game frame). false: refused now (with a reason).
bool Buy(const Item& item, std::string& why);
bool BuyAvailable(std::string* why = nullptr); // eboot recognised, not in a co-op room
// Last purchase result text (for the menu), and a counter that changes with each result.
std::string LastResult(u32* serial = nullptr);

// Game thread (from Cheats::OnGuestFlip, with the validated PlayerGameData pointer).
bool HasPending();
void OnGuestFlip(u64 player_game_data);

} // namespace BB::Shop
