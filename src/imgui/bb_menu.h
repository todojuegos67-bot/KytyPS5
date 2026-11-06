// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// (bb-menu-104) The Bloodborne F1 menu, as the Demon's Souls port's: Mando (pad photo with spots,
// click a button then press the physical one), Teclado y ratón (the same with a key / mouse
// button, mouse-look sensitivity), Trofeos (the game's own list from its TRP, earned / not
// earned, dates, filters) and Salir del juego. While it is open the game gets no input.
// Also the Demon's Souls trophy popup (slide-in card) and its synthesized chime.

#include <filesystem>
#include <string>
#include <string_view>

union SDL_Event;
struct SDL_Window;

namespace BbMenu {

// Adds the ImGui layer (once). Main thread.
void Init(SDL_Window* window);

// Called first for every SDL event in WindowSDL::WaitEvent (main thread). true: consumed.
bool OnSdlEvent(SDL_Event& event);

bool IsOpen();

// A trophy unlocked: the popup card + chime. false when the BB popup is off (BB_TROPHY_POPUP=0).
bool PushTrophyToast(const std::filesystem::path& icon, const std::string& name,
                     std::string_view grade);

} // namespace BbMenu
