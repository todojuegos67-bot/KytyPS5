// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-game-gfx (bb_game_gfx.h).

#include "core/bb_game_gfx.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include "common/logging/log.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace BB::GameGfx {

namespace {

constexpr u64 kPfxGlobal = 0x5465ed0;  // CPostEffect* (YEBIS), data segment
constexpr u64 kAoEnable = 0xb60;       // u8
constexpr u64 kAoSamples = 0xb64;      // s32 (YEBIS default 16; the pass clamps to [9, 64])
constexpr u64 kAoRadius = 0xb68;       // float (+0xb6c scale, +0xb70 contrast)
constexpr u64 kClampSite = 0xbe7255;   // the SSAO pass's clamp (eboot 1.09)
// mov eax,[rbx+0xb64]; mov r13d,9; cmp eax,9; jl +0xd; cmp eax,0x40; mov r13d,0x40;
// cmovle r13d,eax
constexpr std::array<u8, 30> kClampBytes = {0x8b, 0x83, 0x64, 0x0b, 0x00, 0x00, 0x41, 0xbd,
                                            0x09, 0x00, 0x00, 0x00, 0x83, 0xf8, 0x09, 0x7c,
                                            0x0d, 0x83, 0xf8, 0x40, 0x41, 0xbd, 0x40, 0x00,
                                            0x00, 0x00, 0x44, 0x0f, 0x4e, 0xe8};
constexpr u64 kCmpImm = 0x13;  // offset of the "cmp eax,0x40" immediate in kClampBytes
constexpr u64 kMovImm = 0x16;  // offset of the "mov r13d,0x40" immediate

std::atomic<uintptr_t> g_base{0};
u64 g_size = 0;
int g_cap = 0; // patched upper bound (0: untouched)

bool SafeCopy(void* dst, u64 src, u64 len) {
    if (src == 0 || src + len < src) return false;
#ifdef _WIN32
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(src), dst, len, &got) !=
               0 &&
           got == len;
#elif defined(__linux__)
    iovec local{dst, len};
    iovec remote{reinterpret_cast<void*>(src), len};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(len);
#else
    (void)dst;
    (void)len;
    return false;
#endif
}

} // namespace

void OnEbootPatched(uintptr_t eboot_base, u64 image_size) {
    g_base.store(eboot_base, std::memory_order_release);
    g_size = image_size;
    const char* v = std::getenv("BB_SSAO_MAX_SAMPLES");
    if (v == nullptr || v[0] == 0) return;
    const int cap = std::atoi(v);
    if (cap < 9 || cap > 63) {
        LOG_WARNING(Core, "[game-gfx] BB_SSAO_MAX_SAMPLES={} ignored (9..63)", v);
        return;
    }
    if (image_size < kClampSite + kClampBytes.size()) {
        LOG_WARNING(Core, "[game-gfx] BB_SSAO_MAX_SAMPLES: eboot too small, not patched");
        return;
    }
    u8* at = reinterpret_cast<u8*>(eboot_base + kClampSite);
    if (std::memcmp(at, kClampBytes.data(), kClampBytes.size()) != 0) {
        LOG_WARNING(Core,
                    "[game-gfx] BB_SSAO_MAX_SAMPLES: the SSAO clamp at eboot+{:#x} is not the "
                    "1.09 code: not patched",
                    kClampSite);
        return;
    }
    at[kCmpImm] = static_cast<u8>(cap);
    at[kMovImm] = static_cast<u8>(cap);
    g_cap = cap;
    LOG_INFO(Core,
             "[game-gfx] BB_SSAO_MAX_SAMPLES={}: the SSAO pass's sample clamp (eboot+{:#x}) is "
             "now [9, {}] (PS4: [9, 64]); counts the game sets above it get a smaller kernel",
             cap, kClampSite, cap);
}

std::string Describe() {
    const uintptr_t base = g_base.load(std::memory_order_acquire);
    if (base == 0 || g_size < kPfxGlobal + 8) return {};
    u64 pfx = 0;
    if (!SafeCopy(&pfx, base + kPfxGlobal, sizeof(pfx)) || pfx == 0) {
        return "[game-gfx] YEBIS post-effect object not created yet";
    }
    u8 enable = 0;
    s32 samples = 0;
    float ao[3]{};
    if (!SafeCopy(&enable, pfx + kAoEnable, 1) || !SafeCopy(&samples, pfx + kAoSamples, 4) ||
        !SafeCopy(ao, pfx + kAoRadius, sizeof(ao))) {
        return fmt::format("[game-gfx] YEBIS post-effect at {:#x} unreadable", pfx);
    }
    const int clamped = samples < 9 ? 9 : (samples > 64 ? 64 : samples);
    const int used = g_cap != 0 && clamped > g_cap ? g_cap : clamped;
    return fmt::format(
        "[game-gfx] YEBIS SSAO {} | samples set {} (pass uses {}{}) radius {:.3f} scale {:.3f} "
        "contrast {:.3f}",
        enable ? "on" : "off", samples, used,
        g_cap != 0 ? fmt::format(", BB_SSAO_MAX_SAMPLES={}", g_cap) : std::string(), ao[0], ao[1],
        ao[2]);
}

} // namespace BB::GameGfx
