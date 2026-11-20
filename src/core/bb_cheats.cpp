// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bloodborne 1.09 (CUSA03173) cheats. Addresses are eboot VADDR (Ghidra = vaddr + 0x400000,
// ELF offset = vaddr + 0x4000). Evidence (eboot-patch.elf, 1.09):
//  - GameDataMan: pointer at 0x553B130. Lua "GetLocalPlayerSoulLv" (0x1337250) returns
//    [[GameDataMan]+8]+0x90, so [GameDataMan]+8 = PlayerGameData (PGD).
//  - PGD+0x94 echoes: 0x131D5DB adds to it and clamps to 0x3B9AC9FF (999,999,999); the death
//    penalty paths 0x131D410 / 0x1334E4D subtract a rate of it with the same clamp.
//  - PGD+0x84 insight: 0x131D648 adds and clamps to 99 (cmp 0x64 / mov 0x63); the SpEffect path
//    0x18B64F0 does the same (0..99); Lua "GetHeroPoint" (0x1338670) returns it; the summoning
//    path 0x12F7FEA decrements it by one.
//  - WorldChrMan: pointer at 0x553E878; ChrIns lookup 0x13C97A0 returns [WorldChrMan]+0x60 for
//    entity 10000 (the local player).
//  - ChrIns+0x3B0 module bag, +0x20 data module (back pointer to the ChrIns at +8, as read at
//    0x18C20D0): +0xF8 HP, +0xFC max HP (Lua GetHp 0x132FDD0, GetHpRate 0x132FE10, SetHp
//    0x18BCD40 clamps to +0xFC; getter 0x18BF980); +0x134 stamina, +0x138 max stamina
//    (0x18D5C90 / 0x18CA560: add, clamp to [-50, +0x138], skipped for decreases by the chr flag
//    +0x200 bit 7 or WorldChrManDbg AllNoStaminaConsume 0x553E88A).
//  - Goods use 0x18F9720: decrements the item count by one (0x14D0BE0, inventory PGD+0x1D0)
//    unless data+0x200 bit 9 (byte +0x201 bit 1) is set; the quicksilver bullet consumer
//    0x18F9F00 (goods 900/901) skips on the same bit (testb 2, +0x201).
//  - WorldChrManDbg static flags (debug menu strings "GameData.PlayerNoDead" etc., registered
//    at 0x1920550..): 0x553E888 PlayerNoDead (death check 0x18C1FD6..: a player chr cannot die),
//    0x553E889 PlayerExterminate (damage calc 0x1A2AC5B: player attacks use the one-hit
//    damage path; 0x15A5262 the same for the player's bullets).
#include "core/bb_cheats.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include "common/logging/log.h"
#include "core/bb_shop.h"
#include "core/memory.h"

namespace BB::Cheats {
namespace {

constexpr u64 GameDataManVa = 0x553B130;
constexpr u64 WorldChrManVa = 0x553E878;
constexpr u64 PlayerNoDeadVa = 0x553E888;
constexpr u64 PlayerExterminateVa = 0x553E889;
constexpr u64 AllNoStaminaConsumeVa = 0x553E88A;

constexpr u32 PgdInsight = 0x84, PgdEchoes = 0x94;
constexpr u32 ChrModules = 0x3B0, ModData = 0x20, ModOwner = 0x8;
constexpr u32 DataHp = 0xF8, DataHpMax = 0xFC, DataSt = 0x134, DataStMax = 0x138;
constexpr u32 DataFlagsHi = 0x201;
constexpr u8 NoGoodsConsumeBit = 0x02;
constexpr s32 EchoesMax = 999999999, InsightMax = 99;

struct Sig {
    u32 va;
    u32 len;
    const char* bytes;
    u64 rip_target; // != 0: the instruction is "lea r, [rip+disp32]" (disp at +3) to this vaddr
};
constexpr Sig kSigs[] = {
    // GetLocalPlayerSoulLv: lea rax,[GameDataMan]; mov rax,[rax]; mov rcx,[rax+8]
    {0x1337282, 14, "\x48\x8D\x05\xA7\x3E\x20\x04\x48\x8B\x00\x48\x8B\x48\x08", GameDataManVa},
    // ChrIns lookup: lea r12,[WorldChrMan] ... entity 10000 -> mov rax,[rax+0x60]
    {0x13C97B1, 7, "\x4C\x8D\x25\xC0\x50\x17\x04", WorldChrManVa},
    {0x13C9893, 4, "\x48\x8B\x40\x60", 0},
    // HP getter: [[chr+0x3B0]+0x20]+0xF8
    {0x18BF980, 18, "\x48\x8B\x87\xB0\x03\x00\x00\x48\x8B\x40\x20\x8B\x80\xF8\x00\x00\x00\xC3", 0},
    // stamina add: cmp [r14+0x134], ebx ... mov eax,[r14+0x138]; AllNoStaminaConsume flag
    {0x18D5CA4, 7, "\x41\x39\x9E\x34\x01\x00\x00", 0},
    {0x18D5CFB, 7, "\x41\x8B\x86\x38\x01\x00\x00", 0},
    {0x18D5CE5, 7, "\x48\x8D\x05\x9E\x8B\xC6\x03", AllNoStaminaConsumeVa},
    // goods use: movzx edx, word [data+0x200]; and edx, 0x200
    {0x18F9809, 24,
     "\x48\x8B\x93\xB0\x03\x00\x00\x48\x8B\x52\x20\x0F\xB7\x92\x00\x02\x00\x00\x81\xE2\x00\x02"
     "\x00\x00",
     0},
    // bullets: test byte [data+0x201], 2
    {0x18F9F26, 18, "\x48\x8B\x83\xB0\x03\x00\x00\x48\x8B\x40\x20\xF6\x80\x01\x02\x00\x00\x02",
     0},
    // PlayerNoDead / PlayerExterminate readers
    {0x18C210C, 7, "\x48\x8D\x05\x75\xC7\xC7\x03", PlayerNoDeadVa},
    {0x1A2AC5B, 7, "\x48\x8D\x05\x27\x3C\xB1\x03", PlayerExterminateVa},
    // echoes add (clamped) / insight store (clamped to 99)
    {0x131D5DB, 6, "\x03\xB3\x94\x00\x00\x00", 0},
    {0x131D648, 6, "\x89\x83\x84\x00\x00\x00", 0},
};

bool g_ok = false;
uintptr_t g_base = 0;
std::atomic<bool> g_on[static_cast<u32>(Cheat::Count)] = {};
bool g_applied[static_cast<u32>(Cheat::Count)] = {}; // guest thread only
u64 g_items_data = 0;                                 // data module whose bit we set
std::atomic<s64> g_add_echoes{0}, g_add_insight{0};
std::atomic<bool> g_want_stats{false}; // the menu asked for values (Read) since the last frame
std::mutex g_stats_mtx;
Stats g_stats;

bool Valid(u64 addr, u64 size) {
    auto* mem = Core::Memory::Instance();
    return addr != 0 && mem != nullptr && mem->IsValidMapping(addr, size);
}
template <typename T>
T Get(u64 addr) {
    T v{};
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(T));
    return v;
}
template <typename T>
void Put(u64 addr, T v) {
    std::memcpy(reinterpret_cast<void*>(addr), &v, sizeof(T));
}
u64 Static(u64 va) {
    return g_base + va;
}
// Atomic add with clamp on a guest s32 (the game may update the same field on its own thread).
void AddClamped(u64 addr, s64 n, s32 lo, s32 hi) {
    std::atomic_ref<s32> v{*reinterpret_cast<s32*>(addr)};
    s32 cur = v.load();
    for (;;) {
        const s32 nv = static_cast<s32>(std::clamp<s64>(s64(cur) + n, lo, hi));
        if (v.compare_exchange_weak(cur, nv)) break;
    }
}
void SetFlag(u64 va, bool on) {
    Put<u8>(Static(va), on ? 1 : 0);
}

} // namespace

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size) {
    g_base = eboot_base;
    for (const Sig& s : kSigs) {
        if (u64(s.va) + s.len > image_size ||
            std::memcmp(reinterpret_cast<const void*>(eboot_base + s.va), s.bytes, s.len) != 0) {
            LOG_INFO(Lib_VideoOut, "[cheats] code at 0x{:x} is not Bloodborne 1.09: cheats off",
                     s.va);
            return;
        }
        if (s.rip_target != 0) {
            s32 d = 0;
            std::memcpy(&d, reinterpret_cast<const void*>(eboot_base + s.va + 3), 4);
            if (s64(s.va) + 7 + d != s64(s.rip_target)) {
                LOG_INFO(Lib_VideoOut, "[cheats] 0x{:x} does not point at 0x{:x}: cheats off",
                         s.va, s.rip_target);
                return;
            }
        }
    }
    if (PlayerExterminateVa + 1 > image_size || GameDataManVa + 8 > image_size ||
        WorldChrManVa + 8 > image_size) {
        return;
    }
    g_ok = true;
    LOG_INFO(Lib_VideoOut, "[cheats] Bloodborne 1.09 recognised: F1 > Trucos available");
}

bool Available() {
    return g_ok;
}
bool Get(Cheat c) {
    return g_on[static_cast<u32>(c)].load();
}
void Set(Cheat c, bool on) {
    if (!g_ok) return;
    g_on[static_cast<u32>(c)] = on;
    LOG_INFO(Lib_VideoOut, "[cheats] {} {}", static_cast<u32>(c), on ? "on" : "off");
}
void AddEchoes(s32 n) {
    if (g_ok) g_add_echoes += n;
}
void AddInsight(s32 n) {
    if (g_ok) g_add_insight += n;
}
Stats Read() {
    g_want_stats = true;
    std::scoped_lock lk{g_stats_mtx};
    return g_stats;
}

void OnGuestFlip() {
    if (!g_ok) return;
    const auto on = [](Cheat c) { return g_on[static_cast<u32>(c)].load(); };
    const auto idx = [](Cheat c) { return static_cast<u32>(c); };

    // Global debug flags: written only while a cheat owns them, cleared once when it is turned off.
    const bool hp = on(Cheat::InfiniteHp), kill = on(Cheat::OneHitKill);
    if (hp || g_applied[idx(Cheat::InfiniteHp)]) SetFlag(PlayerNoDeadVa, hp);
    if (kill || g_applied[idx(Cheat::OneHitKill)]) SetFlag(PlayerExterminateVa, kill);
    g_applied[idx(Cheat::InfiniteHp)] = hp;
    g_applied[idx(Cheat::OneHitKill)] = kill;

    // Nothing on, nothing queued and the menu not reading: do not touch game memory at all.
    bool busy = g_items_data != 0 || g_add_echoes.load() != 0 || g_add_insight.load() != 0 ||
                BB::Shop::HasPending();
    for (u32 i = 0; i < idx(Cheat::Count); i++) busy = busy || g_on[i].load();
    if (!g_want_stats.exchange(false) && !busy) return;

    // The local player and its data module.
    u64 player = 0, data = 0, pgd = 0;
    const u64 wcm = Get<u64>(Static(WorldChrManVa));
    if (Valid(wcm + 0x60, 8)) player = Get<u64>(wcm + 0x60);
    if (Valid(player + ChrModules, 8)) {
        const u64 mods = Get<u64>(player + ChrModules);
        if (Valid(mods + ModData, 8)) {
            const u64 d = Get<u64>(mods + ModData);
            if (Valid(d, DataStMax + 4) && Get<u64>(d + ModOwner) == player) data = d;
        }
    }
    const u64 gdm = Get<u64>(Static(GameDataManVa));
    if (Valid(gdm + 8, 8)) {
        const u64 p = Get<u64>(gdm + 8);
        if (Valid(p, PgdEchoes + 4)) pgd = p;
    }

    Stats st{};
    if (data != 0) {
        const s32 hmax = Get<s32>(data + DataHpMax), smax = Get<s32>(data + DataStMax);
        // HP: refill alive players only (a dead player is not revived).
        if (hp && hmax > 0 && hmax <= 100000) {
            const s32 h = Get<s32>(data + DataHp);
            if (h > 0 && h < hmax) Put<s32>(data + DataHp, hmax);
        }
        if (on(Cheat::InfiniteStamina) && smax > 0 && smax <= 100000 &&
            Get<s32>(data + DataSt) < smax) {
            Put<s32>(data + DataSt, smax);
        }
        // Items: the chr flag byte is also changed by the game (other bits, 16-bit RMW), so it
        // is set with an atomic OR every frame and cleared once (atomic AND) when turned off.
        std::atomic_ref<u8> flags{*reinterpret_cast<u8*>(data + DataFlagsHi)};
        if (on(Cheat::InfiniteItems)) {
            flags.fetch_or(NoGoodsConsumeBit);
            g_items_data = data;
        } else if (g_items_data == data) {
            flags.fetch_and(static_cast<u8>(~NoGoodsConsumeBit));
            g_items_data = 0;
        }
        st.hp = Get<s32>(data + DataHp), st.hp_max = hmax;
        st.stamina = Get<s32>(data + DataSt), st.stamina_max = smax;
    }
    if (!on(Cheat::InfiniteItems) && g_items_data != data) {
        g_items_data = 0; // that chr is gone (load screen): nothing to clear
    }
    if (pgd != 0 && data != 0) {
        if (const s64 e = g_add_echoes.exchange(0); e != 0) {
            AddClamped(pgd + PgdEchoes, e, 0, EchoesMax);
            LOG_INFO(Lib_VideoOut, "[cheats] echoes {:+}", e);
        }
        if (const s64 i = g_add_insight.exchange(0); i != 0) {
            AddClamped(pgd + PgdInsight, i, 0, InsightMax);
            LOG_INFO(Lib_VideoOut, "[cheats] insight {:+}", i);
        }
        BB::Shop::OnGuestFlip(pgd); // purchases from F1 > ARMAS Y OBJETOS
        st.in_game = true;
        st.echoes = Get<s32>(pgd + PgdEchoes);
        st.insight = Get<s32>(pgd + PgdInsight);
    }
    std::scoped_lock lk{g_stats_mtx};
    g_stats = st;
}

} // namespace BB::Cheats
