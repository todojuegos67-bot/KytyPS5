// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-rumble-hits: a short rumble pulse when the local player's attack lands on an enemy, on an
// object (crate, barrel...) or on the level geometry (wall / floor hit).
//
// Decompiled (eboot-patch-v2.elf, vaddr = eboot offset; Ghidra = vaddr + 0x400000):
//  - 0x18d6160(attacker ChrIns = rdi, hit position = rsi, ...) is the attacker-side "attack hit"
//    feedback: it reads the attacker's modules [rdi+0x3b0] and its weapon/hit entries
//    [[rdi+0x1c8]+8] (chain +0x58), then calls 0x1a55230([[rdi+0x3b0]+0x40]+8, pos, ...) which
//    runs every hit-effect emitter of the manager [0x553b0a0] (+0x28, 0xd8 each, 0x12e5480).
//    It is the single common call of every per-target hit handler of the hit dispatcher
//    0x185b200 (attacker kind 1 = character):
//      0x18b8730 terrain   -> call at 0x18b890e
//      0x18b79e0 character -> call at 0x18b7b13
//      0x18b8430 object    -> call at 0x18b8703 (after the object damage 0x15a50b0 at 0x18b8507)
//      0x18b8940 kind 3    -> call at 0x18b89e7 (unconditional; returns 0)
//    and of 0x18648c1 / 0x19fe52c. Field log 902fc5a: the dispatcher return value (al) is 0 for
//    every player call, so it does not tell a landed hit; the effect spawn does.
//  - Local player: [[0x553e878]+0x60] (0x15a50b0 compares the attacker with it at 0x15a52ad;
//    log 902fc5a: attacker == player 0x211e0ae30).
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
#include <iterator>
#include <string>
#include <vector>
#include <fmt/format.h>
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
constexpr u64 HitFxVaddr = 0x18d6160;
// push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x58
constexpr u8 HitFxPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x58};

struct Caller {
    const char* name;
    u64 ret; // return address of the call 0x18d6160
};
constexpr Caller Callers[] = {
    {"terrain", 0x18b8913}, {"enemy", 0x18b7b18},  {"object", 0x18b8708},
    {"kind3", 0x18b89ec},   {"other1", 0x18648c6}, {"other2", 0x19fe531},
};
constexpr u32 NumCallers = static_cast<u32>(std::size(Callers));

uintptr_t g_base = 0;
std::atomic<s64> g_last_ms{-1000000};
std::atomic<u32> g_count[NumCallers + 1]{};
// diagnostic: calls per caller (last slot: unknown caller) x (attacker == player)
std::atomic<u32> g_diag[NumCallers + 1][2]{};
std::atomic<u32> g_diag_total{0};
std::atomic<s64> g_diag_last_ms{0};

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

const char* CallerName(u32 k) {
    return k < NumCallers ? Callers[k].name : "unknown";
}

void LogDiag(s64 now) {
    s64 last = g_diag_last_ms.load();
    if (now - last < 5000 || !g_diag_last_ms.compare_exchange_strong(last, now)) {
        return;
    }
    std::string line;
    for (u32 k = 0; k <= NumCallers; ++k) {
        line += fmt::format(" {}[other {} player {}]", CallerName(k), g_diag[k][0].load(),
                            g_diag[k][1].load());
    }
    LOG_INFO(Core, "[rumble-hits] diag hit-fx calls{}", line);
}

// regs: rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax, then the return address
void PS4_SYSV_ABI OnHitFx(const u64* regs) {
    if (g_base == 0) {
        return;
    }
    const u64 attacker = regs[5];
    const u64 ret = regs[10] - g_base;
    u32 k = 0;
    while (k < NumCallers && Callers[k].ret != ret) {
        ++k;
    }
    const u64 holder = Read<u64>(g_base + PlayerHolderVaddr);
    const u64 player = holder != 0 ? Read<u64>(holder + 0x60) : 0;
    const bool is_player = attacker != 0 && attacker == player;
    g_diag[k][is_player ? 1 : 0].fetch_add(1);
    const s64 now = NowMs();
    if (g_diag_total.fetch_add(1) < 12) {
        LOG_INFO(Core, "[rumble-hits] diag hit-fx from {} (eboot+{:#x}) attacker {:#x} player {:#x}",
                 CallerName(k), ret, attacker, player);
    }
    LogDiag(now);
    if (!is_player) {
        return;
    }
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
        static std::atomic<bool> off_logged{false};
        if (!off_logged.exchange(true)) {
            LOG_INFO(Core, "[rumble-hits] player hit but the game's vibration option is 0: no pulse");
        }
        return;
    }
    if (g_count[k].fetch_add(1) == 0) {
        LOG_INFO(Core, "[rumble-hits] first player {} hit: pulse (game option {:.1f})",
                 CallerName(k), option);
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

// Saves the caller-saved registers and xmm0..7, calls host(regs), restores them, runs the
// relocated prologue and jumps back after it (same cave layout as bb_objcol.cpp).
std::vector<u8> BuildCave(u64 host, const u8* prologue, u32 len, u64 back) {
    std::vector<u8> e;
    const auto put = [&](std::initializer_list<u8> v) { e.insert(e.end(), v); };
    const auto put64 = [&](u64 v) {
        for (int i = 0; i < 8; ++i) {
            e.push_back(u8(v >> (i * 8)));
        }
    };
    put({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, 0x53});
    put({0x48, 0x89, 0xe3});
    put({0x48, 0x83, 0xe4, 0xf0});
    put({0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x7f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x7f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdf});
    put({0x48, 0xb8});
    put64(host);
    put({0xff, 0xd0});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x6f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x6f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdc});
    put({0x5b, 0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    e.insert(e.end(), prologue, prologue + len);
    put({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
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
    constexpr u32 len = sizeof(HitFxPrologue);
    u8* at = reinterpret_cast<u8*>(eboot_base + HitFxVaddr);
    if (HitFxVaddr + len > image_size || std::memcmp(at, HitFxPrologue, len) != 0) {
        LOG_WARNING(Core,
                    "[rumble-hits] hit-fx eboot+{:#x}: bytes differ (not eboot 1.09?), game code "
                    "untouched",
                    HitFxVaddr);
        return;
    }
    for (const auto& c : Callers) {
        const u8* call = reinterpret_cast<const u8*>(eboot_base + c.ret - 5);
        s32 rel;
        std::memcpy(&rel, call + 1, 4);
        if (call[0] != 0xe8 || c.ret + rel != HitFxVaddr) {
            LOG_WARNING(Core, "[rumble-hits] caller eboot+{:#x} is not `call hit-fx`: game code "
                              "untouched",
                        c.ret - 5);
            return;
        }
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        LOG_WARNING(Core, "[rumble-hits] no executable memory for the detour");
        return;
    }
    g_base = eboot_base;
    const auto c = BuildCave(reinterpret_cast<u64>(&OnHitFx), at, len,
                             reinterpret_cast<u64>(at + len));
    std::memcpy(cave, c.data(), c.size());
    u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
    const u64 to = reinterpret_cast<u64>(cave);
    std::memcpy(j + 6, &to, 8);
    std::memcpy(at, j, sizeof(j));
    std::memset(at + sizeof(j), 0x90, len - sizeof(j));
    LOG_INFO(Core,
             "[rumble-hits] on: attack hit-fx eboot+{:#x} (attacker = rdi) detoured; pulse 120 ms "
             "(large 0.35, small on) x game option, 1 per 80 ms, player attacks only",
             HitFxVaddr);
}

} // namespace BB::RumbleHits
