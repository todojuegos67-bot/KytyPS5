// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bloodborne FLVER draw path (addresses of the base 1.00 eboot image; the 1.09 update moves them,
// so every site is found by signature):
//   0x21d0bc0  DrawModel(model, sink, ?, passes, instances[], ...): per mesh, per instance:
//   0x21d16d0  DrawMesh(rdi model, rsi sink, rdx mesh, rcx params, r8 instance, r9d n):
//              [r8] = the instance's bone matrices (3x4, 48 bytes a bone); [r8+8] the GPU skin
//              ring (current / previous buffers, sink vtable +0x40 / +0x48); then per material:
//   0x21d2110  BonePalette(model, sink, mesh, bones): palette[i] = bones[i] x invbind[i]
//              ([model+0x1f0]), count [model+0x1c] (full skeleton, [model+0x200] != 0) or the
//              mesh's [mesh+0x18] / map [mesh+0x20], single bone [mesh+0x16]; sink vtable +0x20.
//   0x216ab70 / 0x216acf0  GX draw (rdi = G; Gnmx GfxContext at G+0x10, its draw command buffer
//              write pointer at G+0x20; DRAW_INDEX_OFFSET_2 0xC0033500 written by 0x10746f0).
// Detours (jmp [rip] into host memory; the relocated prologue runs after the host call):
//   DrawMesh entry (20 bytes)    -> NativeMv::GuestDrawMesh(model, mesh, instance)
//   both GX draw entries (14/17) -> NativeMv::GuestDrawEmit(G, return address)

#include "core/bb_native_mv_hook.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_native_mv.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace BB::NativeMvHook {
namespace {

constexpr int W = -1; // wildcard

// DrawMesh prologue + argument moves (r12 = sink, r14 = params, r15 = model, [rbp-0x78] = r8).
constexpr int SigDrawMesh[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
                               0x54, 0x53, 0x48, 0x81, 0xec, W,    W,    0x00, 0x00, 0x44, 0x89,
                               0x8d, W,    W,    W,    W,    0x4c, 0x89, 0x45, W,    0x49, 0x89,
                               0xce, 0x49, 0x89, 0xf4, 0x4c, 0x89, 0xa5, W,    W,    W,    W,
                               0x49, 0x89, 0xff};
constexpr u32 DrawMeshPrologue = 20;
constexpr u64 DrawMeshSize = 0xb00; // 0xa40 in the base image, slack for 1.09

// Inside DrawMesh: the two GX draw calls (single / instanced).
constexpr int SigDrawCalls[] = {0x48, 0x8b, 0x78, 0x08, 0x48, 0x8d, 0xb7, W,    W,    W,    W,
                                0x44, 0x89, 0xe2, 0x41, 0x83, 0xfe, 0x01, 0x75, 0x10, 0x89, 0xd9,
                                0x44, 0x8b, 0x85, W,    W,    W,    W,    0xe8, W,    W,    W,
                                W,    0xeb, 0x12, 0x44, 0x89, 0xf1, 0x41, 0x89, 0xd8, 0x44, 0x8b,
                                0x8d, W,    W,    W,    W,    0xe8, W,    W,    W,    W};
constexpr u32 CallAOff = 29, CallBOff = 49;

constexpr u8 PrologueA[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                            0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
constexpr u8 PrologueB[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                            0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec};

void PS4_SYSV_ABI HookDrawMesh(const u64* regs) {
    // regs: rbx r11 r10 r9 r8 rdi rsi rdx rcx rax, return address
    NativeMv::GuestDrawMesh(regs[5], regs[7], regs[4]);
}

void PS4_SYSV_ABI HookEmit(const u64* regs) {
    NativeMv::GuestDrawEmit(regs[5], regs[10]);
}

std::optional<u64> Find(const u8* data, u64 size, const int* sig, u32 n) {
    if (size < n) return std::nullopt;
    std::optional<u64> found;
    for (u64 i = 0; i + n <= size; ++i) {
        if (data[i] != u8(sig[0])) continue;
        u32 k = 1;
        while (k < n && (sig[k] == W || data[i + k] == u8(sig[k]))) ++k;
        if (k == n) {
            if (found) return std::nullopt; // not unique
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

struct Emitter {
    std::vector<u8> b;
    void Put(std::initializer_list<u8> v) {
        b.insert(b.end(), v);
    }
    void Put64(u64 v) {
        for (int i = 0; i < 8; ++i) b.push_back(u8(v >> (i * 8)));
    }
};

// Saves the caller-saved registers, calls host(regs) (sysv), restores them, runs the relocated
// prologue and jumps back after it.
std::vector<u8> BuildCave(u64 host, const u8* prologue, u32 len, u64 back) {
    Emitter e;
    e.Put({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, 0x53});
    e.Put({0x48, 0x89, 0xe3});                         // mov rbx, rsp
    e.Put({0x48, 0x83, 0xe4, 0xf0});                   // and rsp, -16
    e.Put({0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00}); // sub rsp, 0x80
    for (u8 i = 0; i < 8; ++i) {                       // movdqu [rsp + i*16], xmm_i
        if (i == 0) {
            e.Put({0xf3, 0x0f, 0x7f, 0x04, 0x24});
        } else {
            e.Put({0xf3, 0x0f, 0x7f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    e.Put({0x48, 0x89, 0xdf}); // mov rdi, rbx
    e.Put({0x48, 0xb8});       // mov rax, host
    e.Put64(host);
    e.Put({0xff, 0xd0}); // call rax
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            e.Put({0xf3, 0x0f, 0x6f, 0x04, 0x24});
        } else {
            e.Put({0xf3, 0x0f, 0x6f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    e.Put({0x48, 0x89, 0xdc}); // mov rsp, rbx
    e.Put({0x5b, 0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    e.b.insert(e.b.end(), prologue, prologue + len);
    e.Put({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.Put64(back);
    return e.b;
}

void Detour(u8* at, const u8* to) {
    u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
    const u64 v = reinterpret_cast<u64>(to);
    std::memcpy(j + 6, &v, 8);
    std::memcpy(at, j, sizeof(j));
}

} // namespace

void OnEbootPatched(uintptr_t eboot_base, u64 image_size) {
    if (const char* v = std::getenv("BB_NATIVE_MV_HOOK"); v != nullptr && v[0] == '0') {
        NativeMv::SetHooksInstalled(false, "BB_NATIVE_MV_HOOK=0: the game's code is untouched");
        return;
    }
    // bb-perf-1111: off by default (the LIMPIO log: 0 scene draws with game identity). The game's
    // code is only patched with BB_NATIVE_MV=1 (or BB_NATIVE_MV_HOOK=1).
    {
        const char* on = std::getenv("BB_NATIVE_MV");
        const char* hook = std::getenv("BB_NATIVE_MV_HOOK");
        if (!(on != nullptr && on[0] == '1') && !(hook != nullptr && hook[0] == '1')) {
            NativeMv::SetHooksInstalled(false, "off by default (BB_NATIVE_MV=1 turns it on): the "
                                               "game's code is untouched");
            return;
        }
    }
    const u8* img = reinterpret_cast<const u8*>(eboot_base);
    // The code is in the first segment; search the whole image once (~1 s worst case is avoided:
    // the first byte filter skips most positions).
    const u64 code = std::min<u64>(image_size, 0x4000000);
    const auto dm = Find(img, code, SigDrawMesh, std::size(SigDrawMesh));
    if (!dm) {
        NativeMv::SetHooksInstalled(false, "DrawMesh signature not found (unknown eboot)");
        return;
    }
    const auto calls = Find(img + *dm, DrawMeshSize, SigDrawCalls, std::size(SigDrawCalls));
    if (!calls) {
        NativeMv::SetHooksInstalled(false, "GX draw calls not found inside DrawMesh");
        return;
    }
    const u64 site = *dm + *calls;
    const auto target = [&](u32 off) {
        s32 rel;
        std::memcpy(&rel, img + site + off + 1, 4);
        return s64(site + off + 5) + rel;
    };
    const s64 wa = target(CallAOff), wb = target(CallBOff);
    if (wa <= 0 || wb <= 0 || u64(wa) + 32 > code || u64(wb) + 32 > code ||
        std::memcmp(img + wa, PrologueA, sizeof(PrologueA)) != 0 ||
        std::memcmp(img + wb, PrologueB, sizeof(PrologueB)) != 0) {
        NativeMv::SetHooksInstalled(false, "GX draw wrapper prologues differ: no patch");
        return;
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        NativeMv::SetHooksInstalled(false, "no executable memory for the caves");
        return;
    }
    u8* at_dm = reinterpret_cast<u8*>(eboot_base + *dm);
    u8* at_wa = reinterpret_cast<u8*>(eboot_base + wa);
    u8* at_wb = reinterpret_cast<u8*>(eboot_base + wb);
    const u32 len_b = sizeof(PrologueB) + 1; // sub rsp, imm8
    const auto c1 = BuildCave(reinterpret_cast<u64>(&HookDrawMesh), at_dm, DrawMeshPrologue,
                              reinterpret_cast<u64>(at_dm + DrawMeshPrologue));
    const auto c2 = BuildCave(reinterpret_cast<u64>(&HookEmit), at_wa, sizeof(PrologueA),
                              reinterpret_cast<u64>(at_wa + sizeof(PrologueA)));
    const auto c3 = BuildCave(reinterpret_cast<u64>(&HookEmit), at_wb, len_b,
                              reinterpret_cast<u64>(at_wb + len_b));
    u8* p1 = cave;
    u8* p2 = p1 + ((c1.size() + 15) & ~size_t(15));
    u8* p3 = p2 + ((c2.size() + 15) & ~size_t(15));
    std::memcpy(p1, c1.data(), c1.size());
    std::memcpy(p2, c2.data(), c2.size());
    std::memcpy(p3, c3.data(), c3.size());
    NativeMv::SetDrawMeshRange(eboot_base + *dm, eboot_base + *dm + DrawMeshSize);
    Detour(at_wa, p2);
    Detour(at_wb, p3);
    Detour(at_dm, p1);
    NativeMv::SetHooksInstalled(
        true, fmt::format("DrawMesh eboot+0x{:x}, GX draw eboot+0x{:x} / +0x{:x}, draw calls at "
                          "eboot+0x{:x}",
                          *dm, wa, wb, site)
                  .c_str());
}

} // namespace BB::NativeMvHook
