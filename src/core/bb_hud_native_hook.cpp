// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-hud-native: Bloodborne draws its menus and HUD with Scaleform GFx 4 (strings "HAL::BeginScene"
// 0x473eef6, "HAL::beginDisplay", "GFx_DefineFontInfoLoader", "gfxfontlib.swf"). Addresses are
// eboot virtual addresses of the 1.09 image (Ghidra project ghidra-bb2 = vaddr + 0x400000):
//   0x95bf0   Render::HAL::BeginScene (base): profiler event "HAL::BeginScene", sets the 2D state.
//   0x95e10   Render::HAL::EndScene (base): Flush() (vtable +0x70) of the pending batches, then
//             the end event. So the UI draws are all in the command buffer after it returns.
//   0x33d780  FromSoft's HAL subclass BeginScene override: base BeginScene, then
//             GfxContext writes through [this+0x2ca30] (0x1075270 / 0x1073fe0 / 0x1074040:
//             Gnm DrawCommandBuffer helpers: [dcb+8] end, [dcb+0x10] write pointer,
//             [dcb+0x18] reserve callback, the same layout sceGnm pushMarker at 0x1077fe0 uses).
//   0x24b9b20 Orbis HAL EndScene (base EndScene, then +0x11d0 bookkeeping), inherited by the
//             FromSoft subclass.
//   0x51ba090 the FromSoft subclass vtable (vptr; +0x50 BeginScene = 0x33d780, +0x58 EndScene =
//             0x24b9b20; RELATIVE relocations at 0x51ba0e0 / 0x51ba0e8). The HAL's own render
//             events (vtable 0x51ba8f0, +0x10 Begin = 0x23bb640) are empty `ret`s in the retail
//             build, so the game itself writes no marker the emulator could see.
// The two vtable slots are replaced by host wrappers (no game code byte changes): each calls the
// original, then writes a 3-dword PM4 NOP (MarkerBegin / MarkerEnd, sequence) into the HAL's draw
// command buffer when it has room. The GPU thread sees the markers in order with the UI's draws.

#include "core/bb_hud_native_hook.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace BB::HudNativeHook {
namespace {

constexpr int W = -1;

// 0x33d780: push rbp; mov rbp,rsp; push r14; push rbx; mov rbx,rdi; call BeginScene(base);
// test al,al; jz ..; lea rdi,[rbx+0x11d0]; call ..; lea rdi,[rbx+0x2b828]; call ..
constexpr int SigBegin[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xfb, 0xe8,
                            W,    W,    W,    W,    0x84, 0xc0, 0x0f, 0x84, W,    W,    W,
                            W,    0x48, 0x8d, 0xbb, 0xd0, 0x11, 0x00, 0x00, 0xe8, W,    W,
                            W,    W,    0x48, 0x8d, 0xbb, 0x28, 0xb8, 0x02, 0x00, 0xe8};
// Inside it: mov rdi,[rbx+disp32] (the GfxContext / dcb); xor esi,esi; call (0x33d7d3).
constexpr int SigCtx[] = {0x48, 0x8b, 0xbb, W, W, W, W, 0x31, 0xf6, 0xe8};
constexpr u64 BeginFuncSize = 0xa0;
// 0x24b9b20: push rbp; mov rbp,rsp; push rbx; push rax; mov rbx,rdi; call EndScene(base);
// test al,al; je +0x13; add rbx,0x11d0; mov rdi,rbx; call ..
constexpr int SigEnd[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0xe8, W, W, W,
                          W,    0x84, 0xc0, 0x74, 0x13, 0x48, 0x81, 0xc3, 0xd0, 0x11, 0x00, 0x00,
                          0x48, 0x89, 0xdf, 0xe8};

std::atomic<bool> g_installed{false};
u32 g_ctx_off = 0;
std::atomic<u32> g_seq{0};

std::optional<u64> Find(const u8* data, u64 size, const int* sig, u32 n, bool unique = true) {
    if (size < n) return std::nullopt;
    std::optional<u64> found;
    for (u64 i = 0; i + n <= size; ++i) {
        if (data[i] != u8(sig[0])) continue;
        u32 k = 1;
        while (k < n && (sig[k] == W || data[i + k] == u8(sig[k]))) ++k;
        if (k == n) {
            if (!unique) return i;
            if (found) return std::nullopt;
            found = i;
        }
    }
    return found;
}

void* AllocExec(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

bool WriteSlot(u64* slot, u64 value) {
#ifdef _WIN32
    DWORD old = 0;
    if (!VirtualProtect(slot, 8, PAGE_READWRITE, &old)) return false;
    *slot = value;
    DWORD unused = 0;
    VirtualProtect(slot, 8, old, &unused);
    return true;
#else
    const long page = sysconf(_SC_PAGESIZE);
    void* p = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~uintptr_t(page - 1));
    if (mprotect(p, size_t(page), PROT_READ | PROT_WRITE) != 0) return false;
    *slot = value;
    return true; // the data segment is RW (p_flags 6) in the eboot
#endif
}

void Emit(u64 hal, u32 payload) {
    const u64 dcb = *reinterpret_cast<const u64*>(hal + g_ctx_off);
    if (dcb == 0) {
        Stats::no_dcb.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    u32* const end = *reinterpret_cast<u32* const*>(dcb + 0x8);
    u32*& cur = *reinterpret_cast<u32**>(dcb + 0x10);
    if (cur == nullptr || end == nullptr || end < cur || end - cur < 3) {
        Stats::no_space.fetch_add(1, std::memory_order_relaxed); // the next chunk: no marker
        return;
    }
    // Type-3 NOP: count 1 (two body dwords), opcode 0x10.
    cur[0] = 0xC0011000u;
    cur[1] = payload;
    cur[2] = g_seq.fetch_add(1, std::memory_order_relaxed);
    cur += 3;
    Stats::written.fetch_add(1, std::memory_order_relaxed);
}

void PS4_SYSV_ABI AfterBegin(u64 hal, u64 ok) {
    Stats::begins.fetch_add(1, std::memory_order_relaxed);
    if ((ok & 0xff) == 0) {
        Stats::begin_failed.fetch_add(1, std::memory_order_relaxed);
        return; // no 2D scene: EndScene will not flush anything either
    }
    Emit(hal, MarkerBegin);
}

void PS4_SYSV_ABI AfterEnd(u64 hal, u64 ok) {
    Stats::ends.fetch_add(1, std::memory_order_relaxed);
    if ((ok & 0xff) == 0) return;
    Emit(hal, MarkerEnd);
}

// push rbx; mov rbx,rdi; mov rax,orig; call rax; push rax; push rax; mov rdi,rbx; mov rsi,rax;
// mov rax,host; call rax; pop rax; pop rax; pop rbx; ret   (stack 16-aligned at both calls)
std::vector<u8> BuildWrapper(u64 orig, u64 host) {
    std::vector<u8> b{0x53, 0x48, 0x89, 0xfb, 0x48, 0xb8};
    for (int i = 0; i < 8; ++i) b.push_back(u8(orig >> (i * 8)));
    b.insert(b.end(), {0xff, 0xd0, 0x50, 0x50, 0x48, 0x89, 0xdf, 0x48, 0x89, 0xc6, 0x48, 0xb8});
    for (int i = 0; i < 8; ++i) b.push_back(u8(host >> (i * 8)));
    b.insert(b.end(), {0xff, 0xd0, 0x58, 0x58, 0x5b, 0xc3});
    return b;
}

} // namespace

std::atomic<u64> Stats::begins{0}, Stats::begin_failed{0}, Stats::ends{0}, Stats::written{0},
    Stats::no_space{0}, Stats::no_dcb{0};

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_HUD_NATIVE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

bool Installed() {
    return g_installed.load(std::memory_order_acquire);
}

void OnEbootRelocated(uintptr_t eboot_base, const std::vector<std::pair<u64, u64>>& segments) {
    if (!Enabled()) return;
    const u8* img = reinterpret_cast<const u8*>(eboot_base);
    // The code is the segment at 0 (the signatures are searched in its first 64 MiB).
    u64 code = 0;
    for (const auto& [va, size] : segments) {
        if (va == 0) code = std::min<u64>(size, 0x4000000);
    }
    if (code == 0) {
        LOG_WARNING(Core_Linker, "[hud-native] not installed: no code segment at the image base");
        return;
    }
    const auto fail = [](const std::string& why) {
        LOG_WARNING(Core_Linker, "[hud-native] not installed: {} (game code untouched; the "
                                 "heuristic HUD cut stays in use)",
                    why);
    };
    const auto begin = Find(img, code, SigBegin, u32(std::size(SigBegin)));
    const auto end = Find(img, code, SigEnd, u32(std::size(SigEnd)));
    if (!begin || !end) {
        fail(fmt::format("signatures: BeginScene {}, EndScene {}", begin ? "ok" : "missing/ambiguous",
                         end ? "ok" : "missing/ambiguous"));
        return;
    }
    const auto ctx = Find(img + *begin, BeginFuncSize, SigCtx, u32(std::size(SigCtx)), false);
    if (!ctx) {
        fail("GfxContext member load not found in BeginScene");
        return;
    }
    std::memcpy(&g_ctx_off, img + *begin + *ctx + 3, 4);
    if (g_ctx_off == 0 || g_ctx_off > 0x100000) {
        fail(fmt::format("GfxContext offset {:#x} implausible", g_ctx_off));
        return;
    }
    // The vtable slot holding BeginScene (exactly one, 8-aligned, after the code), whose next
    // slot holds EndScene.
    const u64 want_b = eboot_base + *begin, want_e = eboot_base + *end;
    u64* slot = nullptr;
    u32 hits = 0;
    for (const auto& [va, size] : segments) {
        if (va == 0) continue; // vtables live in the data segments
        for (u64 o = (va + 7) & ~u64(7); o + 16 <= va + size; o += 8) {
            u64 v;
            std::memcpy(&v, img + o, 8);
            if (v != want_b) continue;
            ++hits;
            u64 next;
            std::memcpy(&next, img + o + 8, 8);
            if (next == want_e) slot = reinterpret_cast<u64*>(eboot_base + o);
        }
    }
    if (slot == nullptr || hits != 1) {
        fail(fmt::format("vtable slot: {} references to BeginScene, pair {}", hits,
                         slot ? "found" : "not found"));
        return;
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        fail("no executable memory");
        return;
    }
    const auto wb = BuildWrapper(want_b, reinterpret_cast<u64>(&AfterBegin));
    const auto we = BuildWrapper(want_e, reinterpret_cast<u64>(&AfterEnd));
    std::memcpy(cave, wb.data(), wb.size());
    std::memcpy(cave + 64, we.data(), we.size());
    if (!WriteSlot(slot, reinterpret_cast<u64>(cave)) ||
        !WriteSlot(slot + 1, reinterpret_cast<u64>(cave + 64))) {
        fail("vtable not writable");
        return;
    }
    g_installed.store(true, std::memory_order_release);
    LOG_INFO(Core_Linker,
             "[hud-native] Scaleform HAL bracketed: BeginScene eboot+{:#x}, EndScene eboot+{:#x}, "
             "vtable slots eboot+{:#x}/+{:#x}, GfxContext at HAL+{:#x}; PM4 NOP markers {:#x}/{:#x}",
             *begin, *end, reinterpret_cast<uintptr_t>(slot) - eboot_base,
             reinterpret_cast<uintptr_t>(slot) - eboot_base + 8, g_ctx_off, MarkerBegin, MarkerEnd);
}

} // namespace BB::HudNativeHook
