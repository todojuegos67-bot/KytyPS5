// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-rumble-hits: a short rumble pulse when the local player's attack lands on an enemy, on an
// object (crate, barrel...) or on the level geometry (wall / floor hit).
//
// Decompiled (eboot-patch-v2.elf, vaddr = eboot offset; Ghidra = vaddr + 0x400000):
//  - Attack-hit dispatcher 0x185b200(hit = rdi -> r14): rbx = [hit+0x1e0] = attacker entity,
//    r15 = [hit+0x1e8] = hit target; switch on attacker kind ([rbx]->vtbl+0x18, 0..3), then on
//    target kind. In attacker case 3 (0x185b29d):
//      target 1 -> 0x19ed020(hit, chr, attacker)  enemy/character hit; calls the target's
//                  damage module [[chr+0x3b0]+0x98]->vtbl+0x60(.., hit); returns 1 when applied
//      target 2 -> 0x19ed3d0(hit, obj, attacker)  object hit; calls 0x15a50b0(obj, hit), the
//                  object damage: durability u16 at obj+0x1fa, break 0x15a5970 -> 0x15a13a0
//      target 0 -> 0x19ed7d0(hit, coll, attacker) level-collision hit; 0x1dfb3b0 + 0x1c83770,
//                  returns 1 when the hit registers
//    Each site is `mov rdi,r14; mov rsi,r15; mov rdx,rbx; call X; jmp 0x185b3bd` (epilogue that
//    returns eax). The detour runs the same call, then reports (kind, eax, attacker).
//  - Local player: 0x15a50b0 compares [hit+0x1e0] with [[0x553e878]+0x60] (the attacker is the
//    player character); the same pointer is used here.
//  - Game vibration option: [[0x553b130]+0x48] byte +1, divided by 10.0 (0x4926814) by the
//    rumble mixer 0x15715f0 before 0xf91940 (pad setter).
// The game's own rumbles are data driven (TAE event 0x17b97d0 -> 0x131f5e0 and the Lua binding
// "AddRumble" 0x1337ae0 -> 0x131f4d0, rumble ids read from the anim/script data): no rumble id
// for a weak hit exists in the eboot, so a direct pulse is used (large 0.35, small on, 120 ms,
// times the game's option), at most one every 80 ms.

#include "core/bb_rumble_hits.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
#include "common/logging/log.h"
#include "input/controller.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace BB::RumbleHits {
namespace {

constexpr u64 PlayerHolderVaddr = 0x553e878; // [[x]+0x60] = local player ChrIns
constexpr u64 OptionHolderVaddr = 0x553b130; // [[x]+0x48]+1 = vibration option byte
constexpr u64 EpilogueVaddr = 0x185b3bd;

struct Site {
    const char* name;
    u64 at;     // first byte of `mov rdi,r14`
    u32 len;    // bytes replaced (call + jmp included)
    u64 target; // call target
    u8 bytes[19];
};
// mov rdi,r14 / mov rsi,r15 / mov rdx,rbx / call rel32 / jmp epilogue (rel32 or rel8)
constexpr Site Sites[] = {
    {"object", 0x185b2be, 19, 0x19ed3d0,
     {0x4c, 0x89, 0xf7, 0x4c, 0x89, 0xfe, 0x48, 0x89, 0xda, 0xe8, 0x04, 0x21, 0x19, 0x00, 0xe9,
      0xec, 0x00, 0x00, 0x00}},
    {"terrain", 0x185b33c, 16, 0x19ed7d0,
     {0x4c, 0x89, 0xf7, 0x4c, 0x89, 0xfe, 0x48, 0x89, 0xda, 0xe8, 0x86, 0x24, 0x19, 0x00, 0xeb,
      0x71}},
    {"enemy", 0x185b351, 16, 0x19ed020,
     {0x4c, 0x89, 0xf7, 0x4c, 0x89, 0xfe, 0x48, 0x89, 0xda, 0xe8, 0xc1, 0x1c, 0x19, 0x00, 0xeb,
      0x5c}},
};

uintptr_t g_base = 0;
std::atomic<s64> g_last_ms{-1000000};
std::atomic<u32> g_count[3]{};

s64 NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

template <typename T>
T Read(u64 addr) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(T));
    return v;
}

void PS4_SYSV_ABI OnHit(u64 kind, u64 ret, u64 attacker) {
    if (static_cast<u32>(ret) == 0 || attacker == 0 || g_base == 0) {
        return;
    }
    const u64 holder = Read<u64>(g_base + PlayerHolderVaddr);
    if (holder == 0 || Read<u64>(holder + 0x60) != attacker) {
        return; // not the local player's attack
    }
    const s64 now = NowMs();
    s64 last = g_last_ms.load();
    if (now - last < 80 || !g_last_ms.compare_exchange_strong(last, now)) {
        return;
    }
    float option = 1.0f;
    const u64 opt = Read<u64>(g_base + OptionHolderVaddr);
    if (opt != 0) {
        const u64 o2 = Read<u64>(opt + 0x48);
        if (o2 != 0) {
            option = Read<u8>(o2 + 1) / 10.0f;
        }
    }
    if (option <= 0.0f) {
        return;
    }
    if (kind < 3 && g_count[kind].fetch_add(1) == 0) {
        LOG_INFO(Core, "[rumble-hits] first player {} hit: pulse (game option {:.1f})",
                 Sites[kind].name, option);
    }
    Input::RumblePulse(0.35f * option, 1.0f * option, 120);
}

void* AllocExec(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p =
        mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

// mov rdi,r14; mov rsi,r15; mov rdx,rbx; call target; then OnHit(kind, eax, rbx) with rax kept;
// jmp epilogue. At the site rsp is 16-aligned (it was a call site), so the first call is too.
std::vector<u8> BuildCave(u32 kind, u64 target, u64 host, u64 back) {
    std::vector<u8> e;
    const auto put = [&](std::initializer_list<u8> v) { e.insert(e.end(), v); };
    const auto put64 = [&](u64 v) {
        for (int i = 0; i < 8; ++i) {
            e.push_back(u8(v >> (i * 8)));
        }
    };
    put({0x4c, 0x89, 0xf7, 0x4c, 0x89, 0xfe, 0x48, 0x89, 0xda}); // original argument moves
    put({0x48, 0xb8});                                             // mov rax, target
    put64(target);
    put({0xff, 0xd0});                         // call rax
    put({0x55});                               // push rbp
    put({0x48, 0x89, 0xe5});                   // mov rbp, rsp
    put({0x48, 0x83, 0xe4, 0xf0});             // and rsp, -16
    put({0x50, 0x50});                         // push rax x2 (keeps alignment)
    put({0xbf});                               // mov edi, kind
    for (int i = 0; i < 4; ++i) {
        e.push_back(u8(kind >> (i * 8)));
    }
    put({0x89, 0xc6});                         // mov esi, eax
    put({0x48, 0x89, 0xda});                   // mov rdx, rbx (attacker)
    put({0x48, 0xb8});                         // mov rax, host
    put64(host);
    put({0xff, 0xd0});                         // call rax
    put({0x58, 0x58});                         // pop rax x2
    put({0x48, 0x89, 0xec});                   // mov rsp, rbp
    put({0x5d});                               // pop rbp
    put({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    put64(back);
    return e;
}

bool EnabledFlag() {
    const char* v = std::getenv("BB_RUMBLE_HITS");
    return !(v != nullptr && v[0] == '0');
}

} // namespace

void OnEbootPatched(uintptr_t eboot_base, u64 image_size) {
    if (!EnabledFlag()) {
        LOG_INFO(Core, "[rumble-hits] off (BB_RUMBLE_HITS=0)");
        return;
    }
    for (const auto& s : Sites) {
        if (s.at + s.len > image_size ||
            std::memcmp(reinterpret_cast<const void*>(eboot_base + s.at), s.bytes, s.len) != 0) {
            LOG_WARNING(Core,
                        "[rumble-hits] {} site eboot+{:#x}: bytes differ (not eboot 1.09?), "
                        "game code untouched",
                        s.name, s.at);
            return;
        }
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        LOG_WARNING(Core, "[rumble-hits] no executable memory for the detours");
        return;
    }
    g_base = eboot_base;
    u8* w = cave;
    for (u32 k = 0; k < 3; ++k) {
        const auto& s = Sites[k];
        const auto c = BuildCave(k, eboot_base + s.target, reinterpret_cast<u64>(&OnHit),
                                 eboot_base + EpilogueVaddr);
        std::memcpy(w, c.data(), c.size());
        u8* at = reinterpret_cast<u8*>(eboot_base + s.at);
        u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
        const u64 to = reinterpret_cast<u64>(w);
        std::memcpy(j + 6, &to, 8);
        std::memcpy(at, j, sizeof(j));
        std::memset(at + sizeof(j), 0x90, s.len - sizeof(j));
        w += (c.size() + 15) & ~size_t(15);
    }
    LOG_INFO(Core,
             "[rumble-hits] on: attack-hit dispatcher eboot+0x185b200 enemy/object/terrain calls "
             "detoured; pulse 120 ms (large 0.35, small on) x game option, 1 per 80 ms, player "
             "attacks only");
}

} // namespace BB::RumbleHits
