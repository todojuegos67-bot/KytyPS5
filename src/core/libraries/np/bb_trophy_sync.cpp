// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// (bb-trophysync-148) Retroactive Bloodborne trophies: reads the player's own save
// (userdata0000..0009, plain on shadPS4) and reports only what the save proves.
//
// Layout (Noxde/Bloodborne-save-editor, offsets.rs / constants.rs / bosses.json):
//   FACE marker (appearance) searched from 0xF000; inventory = FACE - 34028;
//   username = inventory - 469; flags AOB = username + 68545; NG = username + 68831.
//   Inventory 1983 x 16 B at inventory; key items 2204 B at username + 32201;
//   storage 1984 x 16 B at inventory + 34268. Slot: [7]==0xB0 && [11]==0x40 -> item,
//   id = 24-bit at +8; [11]==0x10 -> armour; else weapon, id = u32 at +8
//   (id % 10000 / 100 = upgrade level, (id % 100000) - (id % 10000) = imprint).
//   Gems / runes: 40 B entries from offset 84, [8]=1 gem / 2 rune, [12]=shape,
//   effects u32 x6 at +16 (gem effect id % 100 = rating).

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <fmt/format.h>

#include "common/types.h"
#include "core/libraries/np/bb_trophy_sync.h"

namespace Libraries::Np::NpTrophy::BbTrophySync {

namespace {

constexpr size_t kInvToFace = 34028;
constexpr size_t kUserToInv = 469;
constexpr size_t kUserToAob = 68545;
constexpr size_t kUserToNg = 68831;
constexpr size_t kUserToKeyInv = 32201;
constexpr size_t kKeyInvBytes = 2204;
constexpr size_t kInvToStorage = 34268;
constexpr size_t kInvSlots = 1983;
constexpr size_t kStorageSlots = 1984;
constexpr size_t kStartToUpgrade = 84;

struct BossFlag {
    int trophy;
    const char* name;
    size_t rel; // from the flags AOB
    u8 mask;    // defeated value
};
// bosses.json "defeated" flags (not the "mp" map flags)
constexpr BossFlag kBosses[] = {
    {14, "Father Gascoigne", 21727, 128},
    {15, "Vicar Amelia", 21602, 128},
    {16, "Shadow of Yharnam", 22227, 128},
    {17, "Rom, the Vacuous Spider", 22602, 128},
    {18, "The One Reborn", 22352, 128},
    {19, "Micolash", 22108, 63},
    {20, "Mergo's Wet Nurse", 22102, 128},
    {21, "Cleric Beast", 21714, 8},
    {22, "Blood-starved Beast", 21477, 128},
    {23, "Witch of Hemwick", 21352, 128},
    {24, "Darkbeast Paarl", 21464, 8},
    {25, "Amygdala", 22727, 128},
    {26, "Martyr Logarius", 21977, 128},
    {27, "Celestial Emissary", 21839, 8},
    {28, "Ebrietas", 21852, 128},
    {35, "Orphan of Kos", 23102, 128},
    {36, "Ludwig", 22852, 128},
    {37, "Lady Maria", 22977, 128},
    {38, "Living Failures", 22983, 32},
    {39, "Laurence", 22858, 32},
    // Childhood's Beginning: the Moon Presence is only fought in that ending
    {3, "Moon Presence", 21108, 32},
};

struct Named {
    u32 id;
    const char* name;
};
// Hunter's Essence: the 26 hunter weapons (base ids, any imprint / level)
constexpr Named kHunterWeapons[] = {
    {20100000, "Torch"},          {4000000, "Blade of Mercy"},
    {8000000, "Kirkhammer"},      {14200000, "Repeating Pistol"},
    {7000000, "Saw Cleaver"},     {7100000, "Saw Spear"},
    {5000000, "Hunter Axe"},      {22000000, "Threaded Cane"},
    {14000000, "Hunter Pistol"},  {6000000, "Hunter Blunderbuss"},
    {10000000, "Rifle Spear"},    {11000000, "Stake Driver"},
    {20000000, "Hunter's Torch"}, {19000000, "Wooden Shield"},
    {8100000, "Ludwig's Holy Blade"}, {6100000, "Ludwig's Rifle"},
    {18100000, "Flamesprayer"},   {15000000, "Cannon"},
    {13000000, "Tonitrus"},       {18000000, "Rosmarinus"},
    {10100000, "Reiterpallasch"}, {14100000, "Evelyn"},
    {2000000, "Chikage"},         {12000000, "Logarius' Wheel"},
    {9000000, "Beast Claw"},      {5100000, "Burial Blade"},
};
// Old Hunter's Essence: the 16 Old Hunters weapons
constexpr Named kOldHunterWeapons[] = {
    {23000000, "Beasthunter Saif"}, {24000000, "Beast Cutter"},
    {25000000, "Amygdalan Arm"},    {26000000, "Holy Moonlight Sword"},
    {27000000, "Rakuyo"},           {28000000, "Boom Hammer"},
    {29000000, "Bloodletter"},      {30000000, "Church Pick"},
    {31000000, "Whirligig Saw"},    {32000000, "Simon's Bowblade"},
    {38000000, "Kos Parasite"},     {33000000, "Gatling Gun"},
    {34000000, "Fist of Gratia"},   {35000000, "Church Cannon"},
    {36000000, "Piercing Rifle"},   {19100000, "Loch Shield"},
};
// Hunter's Craft: the 9 special hunter tools (item ids)
constexpr Named kHunterTools[] = {
    {2060, "Old Hunter Bone"},   {2020, "Beast Roar"},       {1310, "Empty Phantasm Shell"},
    {2070, "Tiny Tonitrus"},     {2000, "Augur of Ebrietas"}, {2110, "Messenger's Gift"},
    {2080, "Executioner's Gloves"}, {2010, "A Call Beyond"}, {2050, "Choir Bell"},
};

constexpr const char* kTrophyNames[40] = {
    "Bloodborne",
    "Yharnam Sunrise",
    "Honouring Wishes",
    "Childhood's Beginning",
    "Yharnam, Pthumerian Queen",
    "Hunter's Essence",
    "Hunter's Craft",
    "Weapon Master",
    "Blood Gem Master",
    "Rune Master",
    "Cainhurst",
    "The Choir",
    "The Source of the Dream",
    "Nightmare Lecture Building",
    "Father Gascoigne",
    "Vicar Amelia",
    "Shadow of Yharnam",
    "Rom, the Vacuous Spider",
    "The One Reborn",
    "Micolash, Host of the Nightmare",
    "Mergo's Wet Nurse",
    "Cleric Beast",
    "Blood-starved Beast",
    "The Witch of Hemwick",
    "Darkbeast Paarl",
    "Amygdala",
    "Martyr Logarius",
    "Celestial Emissary",
    "Ebrietas, Daughter of the Cosmos",
    "Blood Gem Contact",
    "Rune Contact",
    "Chalice of Pthumeru",
    "Chalice of Ailing Loran",
    "Chalice of Isz",
    "Old Hunter's Essence",
    "Orphan of Kos",
    "Ludwig, the Holy Blade",
    "Lady Maria of the Astral Clocktower",
    "Living Failures",
    "Laurence, the First Vicar",
};

std::set<u32> AllWeaponBases() {
    std::set<u32> s;
    for (const auto& w : kHunterWeapons) {
        s.insert(w.id);
    }
    for (const auto& w : kOldHunterWeapons) {
        s.insert(w.id);
    }
    s.insert(12080000); // Bloody Logarius' Wheel
    s.insert(6180000);  // Bloody Ludwig's Rifle
    s.insert(900000);   // Adella's End-Quest Dagger
    return s;
}

u32 Rd32(const std::vector<u8>& b, size_t o) {
    u32 v;
    std::memcpy(&v, b.data() + o, 4);
    return v;
}

struct SlotData {
    bool ok = false;
    std::u16string name;
    u8 ng = 0;
    std::set<u32> items;               // item ids held (inventory, key items, storage)
    std::map<u32, int> weapons;        // base id -> best upgrade level
    std::string best_weapon_name;
    std::vector<std::pair<u8, std::array<u32, 6>>> upgrades; // type (1 gem / 2 rune), effects
    std::vector<const BossFlag*> bosses;
};

SlotData ParseSlot(const std::vector<u8>& b) {
    SlotData d;
    static const std::set<u32> weapon_bases = AllWeaponBases();
    if (b.size() < 0x20000) {
        return d;
    }
    size_t face = std::string::npos;
    for (size_t i = 0xF000; i + 4 <= b.size(); i++) {
        if (b[i] == 'F' && b[i + 1] == 'A' && b[i + 2] == 'C' && b[i + 3] == 'E') {
            face = i;
            break;
        }
    }
    if (face == std::string::npos || face < kInvToFace + kUserToInv) {
        return d;
    }
    const size_t inv = face - kInvToFace;
    const size_t user = inv - kUserToInv;
    if (user + kUserToNg + 1 > b.size() || inv + kInvToStorage + kStorageSlots * 16 > b.size() ||
        user + kUserToAob + 24000 > b.size()) {
        return d;
    }
    for (size_t i = 0; i < 16; i++) {
        const char16_t c = char16_t(b[user + 1 + i * 2] | (b[user + 2 + i * 2] << 8));
        if (c == 0) {
            break;
        }
        d.name.push_back(c);
    }
    d.ng = b[user + kUserToNg];

    const auto scan = [&](size_t start, size_t end) {
        for (size_t i = start; i + 16 <= end; i += 16) {
            const u8 t7 = b[i + 7], t11 = b[i + 11];
            if (t7 == 0xB0 && t11 == 0x40) {
                d.items.insert(u32(b[i + 8]) | (u32(b[i + 9]) << 8) | (u32(b[i + 10]) << 16));
            } else if (t11 == 0x10) {
                continue; // armour
            } else {
                const u32 sp = Rd32(b, i + 8);
                const u32 imprint = (sp % 100000) - (sp % 10000);
                if (sp % 100 != 0 || (imprint != 0 && imprint != 10000 && imprint != 20000 &&
                                      imprint != 80000)) {
                    continue;
                }
                const u32 base =
                    (sp == 12080000 || sp == 6180000) ? sp : (sp / 100000) * 100000;
                if (!weapon_bases.contains(base)) {
                    continue;
                }
                const int level = int((sp % 10000) / 100);
                if (level > 10) {
                    continue;
                }
                auto [it, ins] = d.weapons.emplace(base, level);
                if (!ins) {
                    it->second = std::max(it->second, level);
                }
            }
        }
    };
    scan(inv, inv + kInvSlots * 16);
    scan(user + kUserToKeyInv, user + kUserToKeyInv + kKeyInvBytes);
    scan(inv + kInvToStorage, inv + kInvToStorage + kStorageSlots * 16);

    for (size_t i = kStartToUpgrade; i + 40 <= b.size(); i += 40) {
        const u8* c = b.data() + i + 8;
        const bool gem = c[0] == 1 && (c[4] == 1 || c[4] == 2 || c[4] == 4 || c[4] == 8 || c[4] == 0x3f);
        const bool rune = c[0] == 2 && (c[4] == 1 || c[4] == 2);
        if (!(gem || rune) || c[1] || c[2] || c[3] || c[5] || c[6] || c[7]) {
            break;
        }
        std::array<u32, 6> e;
        for (int k = 0; k < 6; k++) {
            e[k] = Rd32(b, i + 16 + k * 4);
        }
        d.upgrades.emplace_back(c[0], e);
    }

    const size_t aob = user + kUserToAob;
    for (const auto& f : kBosses) {
        if ((b[aob + f.rel] & f.mask) == f.mask) {
            d.bosses.push_back(&f);
        }
    }
    d.ok = true;
    return d;
}

std::string Utf8(const std::u16string& s) {
    std::string o;
    for (char16_t c : s) {
        if (c < 0x80) {
            o.push_back(char(c));
        } else if (c < 0x800) {
            o.push_back(char(0xC0 | (c >> 6)));
            o.push_back(char(0x80 | (c & 0x3F)));
        } else {
            o.push_back(char(0xE0 | (c >> 12)));
            o.push_back(char(0x80 | ((c >> 6) & 0x3F)));
            o.push_back(char(0x80 | (c & 0x3F)));
        }
    }
    return o;
}

} // namespace

std::vector<Proof> ProveFromSaveDir(const std::filesystem::path& save_dir) {
    std::map<int, std::string> proofs;
    const auto add = [&](int id, std::string why) { proofs.emplace(id, std::move(why)); };

    for (int slot = 0; slot < 10; slot++) {
        const auto path = save_dir / fmt::format("userdata{:04d}", slot);
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            continue;
        }
        std::vector<u8> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const SlotData d = ParseSlot(b);
        if (!d.ok) {
            continue;
        }
        const std::string who = fmt::format("slot {} '{}' NG{}", slot, Utf8(d.name), d.ng);

        for (const BossFlag* bf : d.bosses) {
            add(bf->trophy, fmt::format("{}: {} defeated flag set", who, bf->name));
        }
        const auto has_all_weapons = [&](const auto& list) {
            return std::all_of(std::begin(list), std::end(list),
                               [&](const Named& w) { return d.weapons.contains(w.id); });
        };
        if (has_all_weapons(kHunterWeapons)) {
            add(5, who + ": all 26 hunter weapons held");
        }
        if (has_all_weapons(kOldHunterWeapons)) {
            add(34, who + ": all 16 Old Hunters weapons held");
        }
        if (std::all_of(std::begin(kHunterTools), std::end(kHunterTools),
                        [&](const Named& t) { return d.items.contains(t.id); })) {
            add(6, who + ": all 9 special hunter tools held");
        }
        for (const auto& [base, level] : d.weapons) {
            if (level >= 10) {
                add(7, fmt::format("{}: weapon id {} at +10 held", who, base));
                break;
            }
        }
        int gems = 0, runes = 0, best_rating = 0;
        bool formless = false;
        for (const auto& [type, e] : d.upgrades) {
            if (type == 1) {
                gems++;
                const u32 primary = e[0];
                if (primary >= 10000 && primary < 4000000 && primary % 100 >= 1 &&
                    primary % 100 <= 20) {
                    best_rating = std::max(best_rating, int(primary % 100));
                }
            } else {
                runes++;
                formless |= e[0] >= 1136000 && e[0] <= 1136004;
            }
        }
        if (gems > 0) {
            add(29, fmt::format("{}: {} blood gem(s) held", who, gems));
        }
        if (runes > 0) {
            add(30, fmt::format("{}: {} caryll rune(s) held", who, runes));
        }
        if (best_rating >= 15) {
            add(8, fmt::format("{}: blood gem of rating {} held", who, best_rating));
        }
        if (formless) {
            add(9, who + ": Formless Oedon rune held");
        }
        if (d.items.contains(6109)) {
            add(31, who + ": Pthumeru Chalice held");
        }
        if (d.items.contains(6429)) {
            add(32, who + ": Ailing Loran Chalice held");
        }
        if (d.items.contains(6539)) {
            add(33, who + ": Great Isz Chalice held");
        }
        if (d.items.contains(4012)) {
            add(13, who + ": Lecture Theatre Key held (found only inside the Lecture Building)");
        }
        if (d.items.contains(2080)) {
            add(10, who + ": Executioner's Gloves held (found only in Cainhurst)");
        }
        if (d.items.contains(2010)) {
            add(11, who + ": A Call Beyond held (found only in the Upper Cathedral Ward)");
        }
        for (const BossFlag* bf : d.bosses) {
            if (bf->trophy == 27) {
                add(11, who + ": Celestial Emissary (Upper Cathedral Ward) defeated");
            }
        }
    }
    std::vector<Proof> out;
    for (auto& [id, why] : proofs) {
        out.push_back({id, id >= 0 && id < 40 ? kTrophyNames[id] : "?", std::move(why)});
    }
    return out;
}

std::filesystem::path FindSaveDir(const std::filesystem::path& user_home) {
    std::error_code ec;
    const auto root = user_home / "savedata";
    std::filesystem::path best;
    std::filesystem::file_time_type best_t{};
    for (const auto& title : std::filesystem::directory_iterator(root, ec)) {
        const auto dir = title.path() / "SPRJ0005";
        std::filesystem::file_time_type t{};
        bool any = false;
        for (int slot = 0; slot <= 10; slot++) {
            const auto p = dir / fmt::format("userdata{:04d}", slot);
            std::error_code e2;
            const auto ft = std::filesystem::last_write_time(p, e2);
            if (!e2 && (!any || ft > t)) {
                t = ft;
                any = true;
            }
        }
        if (any && (best.empty() || t > best_t)) {
            best = dir;
            best_t = t;
        }
    }
    return best;
}

} // namespace Libraries::Np::NpTrophy::BbTrophySync
