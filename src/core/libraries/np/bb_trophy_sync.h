// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace Libraries::Np::NpTrophy::BbTrophySync {

struct Proof {
    int trophy_id;
    std::string name;
    std::string reason;
};

// Parses every Bloodborne character slot (userdata0000..0009) in save_dir and returns the
// trophies the save data proves, one entry per trophy (first proof found). Pure, no side effects.
std::vector<Proof> ProveFromSaveDir(const std::filesystem::path& save_dir);

// Picks the Bloodborne save directory (.../savedata/<title>/SPRJ0005) of a user: the one whose
// files were written last. Empty if none.
std::filesystem::path FindSaveDir(const std::filesystem::path& user_home);

} // namespace Libraries::Np::NpTrophy::BbTrophySync
