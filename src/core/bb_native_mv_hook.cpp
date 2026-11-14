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
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_mv_engine.h"
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

// SkinRingUpdate(rdi ring, rsi bones, rdx ?): ++[ring+0x98] mod [ring+0x9c] first.
constexpr int SigSkinRing[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
                               0x54, 0x53, 0x48, 0x83, 0xec, W,    0x49, 0x89, 0xd4, 0x49, 0x89,
                               0xf7, 0x48, 0x89, 0xfb, 0x8b, 0x83, 0x98, 0x00, 0x00, 0x00, 0xff,
                               0xc0, 0x89, 0x83, 0x98, 0x00, 0x00, 0x00};
constexpr u32 SkinRingPrologue = 17;

constexpr u8 PrologueA[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                            0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
constexpr u8 PrologueB[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                            0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec};

void PS4_SYSV_ABI HookDrawMesh(const u64* regs) {
    // regs: rbx r11 r10 r9 r8 rdi rsi rdx rcx rax, return address
    NativeMv::GuestDrawMesh(regs[5], regs[7], regs[4]);
}

void PS4_SYSV_ABI HookSkinRing(const u64* regs) {
    if (MvEngine::Enabled())
        MvEngine::GuestRingUpdate(regs[5]);
    NativeMv::GuestSkinRing(regs[5], regs[6]);
}

void PS4_SYSV_ABI HookEmit(const u64* regs) {
    NativeMv::GuestDrawEmit(regs[5], regs[10]);
}

// bb-mv-engine (bb_mv_engine.h): BonePalette / its twin entry (rdi model, rsi sink, rdx mesh,
// rcx bones) and their SetPalette call sites (rax = the palette just written).
void PS4_SYSV_ABI HookPalBegin0(const u64* regs) {
    MvEngine::GuestPaletteBegin(regs[5], regs[6], regs[7], regs[8], 0);
}
void PS4_SYSV_ABI HookPalBegin1(const u64* regs) {
    MvEngine::GuestPaletteBegin(regs[5], regs[6], regs[7], regs[8], 1);
}
void PS4_SYSV_ABI HookPalSet0(const u64* regs) {
    MvEngine::GuestPaletteSet(regs[9], 0);
}
void PS4_SYSV_ABI HookPalSet1(const u64* regs) {
    MvEngine::GuestPaletteSet(regs[9], 1);
}

// bb-mv-engine engine transforms: DrawMesh entry (rdi model, rsi sink, rdx mesh, rcx material,
// r8 instance) and the SSAO reprojection site (rbx object, rbp frame, xmm4..7 current view).
bool g_dm_native = false; // BB_NATIVE_MV_DRAWMESH=1 wants the DrawMesh entry too
void PS4_SYSV_ABI HookDrawMeshEngine(const u64* regs) {
    MvEngine::GuestDrawMesh(regs[5], regs[6], regs[7], regs[8], regs[4]);
    if (g_dm_native)
        NativeMv::GuestDrawMesh(regs[5], regs[7], regs[4]);
}
void PS4_SYSV_ABI HookCamera(const u64* regs, u64 rbp, const u8* xmm) {
    MvEngine::GuestCamera(regs, rbp, xmm);
}

// DrawMesh body (0x21d0640, size < DrawMeshSize): the instructions GuestDrawMesh mirrors.
struct BodyCheck {
    const char* what;
    std::vector<u8> bytes;
    u32 count;
};
const std::vector<BodyCheck>& DrawMeshChecks() {
    static const std::vector<BodyCheck> c = {
        // mov rcx,[rcx+0x30]; mov rsi,[rax+0xb0]; cmp byte [rsi+0x200],0; movsxd rsi,[rax+0x98]
        {"rigid current slot",
         {0x48, 0x8b, 0x49, 0x30, 0x48, 0x8b, 0xb0, 0xb0, 0x00, 0x00, 0x00, 0x80, 0xbe,
          0x00, 0x02, 0x00, 0x00, 0x00, 0x48, 0x63, 0xb0, 0x98, 0x00, 0x00, 0x00},
         2},
        // mov rax,[rax+0x38]; mov rcx,[rbp-0x78]; mov rdx,[rcx+8]; movzx ecx,word [rbx+2];
        // mov edi,[rdx+0x98]; lea esi,[rdi-1]
        {"rigid previous slot",
         {0x48, 0x8b, 0x40, 0x38, 0x48, 0x8b, 0x4d, 0x88, 0x48, 0x8b, 0x51, 0x08, 0x0f,
          0xb7, 0x4b, 0x02, 0x8b, 0xba, 0x98, 0x00, 0x00, 0x00, 0x8d, 0x77, 0xff},
         2},
        // mov rax,[rax]; lea rdi,[rsi+rsi*4]; movzx esi,word [rdx+2]; shl rsi,3;
        // add rsi,[rax+rdi*8+8]
        {"slot table",
         {0x48, 0x8b, 0x00, 0x48, 0x8d, 0x3c, 0xb6, 0x0f, 0xb7, 0x72,
          0x02, 0x48, 0xc1, 0xe6, 0x03, 0x48, 0x03, 0x74, 0xf8, 0x08},
         2},
        {"material +0x2c", {0x41, 0x80, 0x7e, 0x2c, 0x00}, 1},
        {"sink +0x171", {0x41, 0xf6, 0x84, 0x24, 0x71, 0x01, 0x00, 0x00, 0x01}, 1},
        {"material +0x1d", {0x41, 0x80, 0x7e, 0x1d, 0x00}, 1},
        {"ring +0xa4", {0x80, 0xb8, 0xa4, 0x00, 0x00, 0x00, 0x00}, 1},
        // bb-mv-skin, skinned branch 0x21d0759: mov rcx,[rcx+0x40]; mov rdx,[rax+0xb0];
        // xor esi,esi; cmp byte [rdx+0x200],0; mov edx,0; je; movsxd rdx,[rax+0x98];
        // mov rsi,[rax+0x10]; mov rax,[rax+0x18]; mov rsi,[rsi+rdx*8]; mov rdx,[rax+rdx*8]
        {"skinned current slots",
         {0x48, 0x8b, 0x49, 0x40, 0x48, 0x8b, 0x90, 0xb0, 0x00, 0x00, 0x00, 0x31, 0xf6,
          0x80, 0xba, 0x00, 0x02, 0x00, 0x00, 0x00, 0xba, 0x00, 0x00, 0x00, 0x00, 0x74,
          0x17, 0x48, 0x63, 0x90, 0x98, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x70, 0x10, 0x48,
          0x8b, 0x40, 0x18, 0x48, 0x8b, 0x34, 0xd6, 0x48, 0x8b, 0x14, 0xd0},
         1},
        // 0x21d079f: mov r8,[rax+0x48]; mov rax,[rbp-0x78]; mov rcx,[rax+8];
        // mov edx,[rcx+0x98]; lea ebx,[rdx-1]; mov edi,ebx; test edx,edx; jg; mov edi,[rcx+0x9c];
        // add edi,ebx; ... movsxd rsi,edi; mov rdi,[rcx+0x10]; mov rsi,[rdi+rsi*8]
        {"skinned previous slots",
         {0x4c, 0x8b, 0x40, 0x48, 0x48, 0x8b, 0x45, 0x88, 0x48, 0x8b, 0x48, 0x08, 0x8b,
          0x91, 0x98, 0x00, 0x00, 0x00, 0x8d, 0x5a, 0xff, 0x89, 0xdf, 0x85, 0xd2, 0x7f,
          0x08, 0x8b, 0xb9, 0x9c, 0x00, 0x00, 0x00, 0x01, 0xdf},
         1},
        {"skinned previous B slot",
         {0x48, 0x63, 0xc3, 0x48, 0x8b, 0x49, 0x18, 0x48, 0x8b, 0x14, 0xc1}, 1},
    };
    return c;
}

// SSAO reprojection (0xe67073): 4 x vmovaps [r12+0x10..0x40] (relocated, 28 bytes); mov byte
// [rbx+0x240],1; cmp byte [rip+flag],0; jne; the stores of the current projection
// ([rbp-0x1f0..-0x220] -> [rbx+0x200..0x230]) and view (xmm4..7 -> [rbx+0x1c0..0x1f0]).
constexpr int SigCamera[] = {
    0xc4, 0xc1, 0x78, 0x29, 0x44, 0x24, 0x10, 0xc4, 0xc1, 0x78, 0x29, 0x4c, 0x24, 0x20, 0xc4, 0xc1,
    0x78, 0x29, 0x54, 0x24, 0x30, 0xc4, 0xc1, 0x78, 0x29, 0x5c, 0x24, 0x40, 0xc6, 0x83, 0x40, 0x02,
    0x00, 0x00, 0x01, 0x80, 0x3d, W,    W,    W,    W,    0x00, 0x75, 0x60, 0xc5, 0xf8, 0x28, 0x85,
    0x10, 0xfe, 0xff, 0xff, 0xc5, 0xf8, 0x29, 0x83, 0x00, 0x02, 0x00, 0x00, 0xc5, 0xf8, 0x28, 0x85,
    0x00, 0xfe, 0xff, 0xff, 0xc5, 0xf8, 0x29, 0x83, 0x10, 0x02, 0x00, 0x00, 0xc5, 0xf8, 0x28, 0x85,
    0xf0, 0xfd, 0xff, 0xff, 0xc5, 0xf8, 0x29, 0x83, 0x20, 0x02, 0x00, 0x00, 0xc5, 0xf8, 0x28, 0x85,
    0xe0, 0xfd, 0xff, 0xff, 0xc5, 0xf8, 0x29, 0x83, 0x30, 0x02, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0xa3,
    0xc0, 0x01, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0xab, 0xd0, 0x01, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0xb3,
    0xe0, 0x01, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0xbb, 0xf0, 0x01, 0x00, 0x00};
constexpr u32 CameraRelocLen = 28;

// BonePalette entry: PrologueA + mov rbx,rcx; mov r12,rdx; mov r14,rsi; mov r15,rdi;
// cmp byte [r15+0x200],0 (two functions: 1.09 0x21d1080 / 0x21d1650).
constexpr u8 SigPalEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
                              0x54, 0x53, 0x50, 0x48, 0x89, 0xcb, 0x49, 0x89, 0xd4, 0x49, 0x89,
                              0xf6, 0x49, 0x89, 0xff, 0x41, 0x80, 0xbf, 0x00, 0x02, 0x00, 0x00,
                              0x00};
constexpr u32 PalEntryPrologue = 14; // push rbp .. push rax (no rip-relative bytes)
// mov rbx,[r14]; xor esi,esi; mov rdi,r14; mov edx,r13d; mov rcx,rax (14 bytes, relocated)
// followed by call [rbx+0x20] (first function) / call [rbx+0x28] (twin).
constexpr u8 SigPalSet[] = {0x49, 0x8b, 0x1e, 0x31, 0xf6, 0x4c, 0x89, 0xf7, 0x44,
                            0x89, 0xea, 0x48, 0x89, 0xc1, 0xff, 0x53};
constexpr u32 PalSetLen = 14;
constexpr u64 PalFuncSize = 0x5d0;

// All occurrences of an exact byte string.
std::vector<u64> FindAll(const u8* data, u64 size, const u8* sig, u32 n) {
    std::vector<u64> out;
    for (u64 i = 0; i + n <= size; ++i) {
        if (data[i] == sig[0] && std::memcmp(data + i, sig, n) == 0) out.push_back(i);
    }
    return out;
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
std::vector<u8> BuildCave(u64 host, const u8* prologue, u32 len, u64 back,
                          bool pass_frame = false) {
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
    if (pass_frame) {
        e.Put({0x48, 0x89, 0xee}); // mov rsi, rbp (the guest frame)
        e.Put({0x48, 0x89, 0xe2}); // mov rdx, rsp (saved xmm0..7)
    }
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
    const u8* img = reinterpret_cast<const u8*>(eboot_base);
    const u64 code = std::min<u64>(image_size, 0x4000000);
    constexpr size_t kCaveBytes = 8192;
    u8* cave = static_cast<u8*>(AllocExec(kCaveBytes));
    if (cave == nullptr) {
        NativeMv::SetHooksInstalled(false, "no executable memory for the caves");
        return;
    }
    u8* next = cave;
    const auto place = [&](const std::vector<u8>& c) {
        u8* p = next;
        std::memcpy(p, c.data(), c.size());
        next += (c.size() + 15) & ~size_t(15);
        return p;
    };
    std::string detail;
    bool any = false;

    // Characters: the per-frame skin ring update (ring, bones).
    if (const auto sr = Find(img, code, SigSkinRing, std::size(SigSkinRing))) {
        u8* at = reinterpret_cast<u8*>(eboot_base + *sr);
        u8* p = place(BuildCave(reinterpret_cast<u64>(&HookSkinRing), at, SkinRingPrologue,
                                reinterpret_cast<u64>(at + SkinRingPrologue)));
        Detour(at, p);
        any = true;
        detail += fmt::format("SkinRingUpdate eboot+0x{:x}", *sr);
    } else {
        detail += "SkinRingUpdate not found";
    }

    // bb-mv-engine (BB_MV_ENGINE=1): palette -> engine object (bones x mesh). Every byte that
    // is relocated is compared first; any mismatch: nothing is patched.
    if (MvEngine::Enabled()) {
        std::string md;
        const auto entries = FindAll(img, code, SigPalEntry, sizeof(SigPalEntry));
        bool ok = entries.size() == 2;
        std::vector<std::pair<u64, u32>> sites; // (offset, consumer)
        if (ok) {
            for (u32 f = 0; f < 2 && ok; ++f) {
                const auto in = FindAll(img + entries[f], PalFuncSize, SigPalSet, sizeof(SigPalSet));
                // consumer from the call's vtable slot: 0x20 first function, 0x28 the twin
                for (u64 o : in) {
                    const u8 slot = img[entries[f] + o + sizeof(SigPalSet)];
                    if (slot != (f == 0 ? 0x20 : 0x28)) ok = false;
                    sites.emplace_back(entries[f] + o, f);
                }
                if (in.size() != 3) ok = false;
            }
        }
        if (ok && (next - cave) + 8 * 0x100 <= ptrdiff_t(kCaveBytes)) {
            for (u32 f = 0; f < 2; ++f) {
                u8* at = reinterpret_cast<u8*>(eboot_base + entries[f]);
                u8* p = place(BuildCave(reinterpret_cast<u64>(f == 0 ? &HookPalBegin0
                                                                     : &HookPalBegin1),
                                        at, PalEntryPrologue,
                                        reinterpret_cast<u64>(at + PalEntryPrologue)));
                Detour(at, p);
            }
            for (const auto& [off, f] : sites) {
                u8* at = reinterpret_cast<u8*>(eboot_base + off);
                u8* p = place(BuildCave(reinterpret_cast<u64>(f == 0 ? &HookPalSet0 : &HookPalSet1),
                                        at, PalSetLen, reinterpret_cast<u64>(at + PalSetLen)));
                Detour(at, p);
            }
            md = fmt::format("BonePalette eboot+0x{:x} / twin eboot+0x{:x}, {} SetPalette sites",
                             entries[0], entries[1], sites.size());
            MvEngine::SetHooks(true, md.c_str());
        } else {
            md = fmt::format("signatures differ (entries {}, sites {}): game code untouched",
                             entries.size(), sites.size());
            MvEngine::SetHooks(false, md.c_str());
        }
    }

    // bb-mv-engine engine transforms: DrawMesh rigid ring branch and the engine camera. The
    // relocated bytes and every instruction mirrored on the host are compared first.
    bool dm_engine = false;
    std::optional<u64> dm_off; // DrawMesh, found before its entry is patched
    if (MvEngine::Enabled()) {
        std::string ed;
        u32 dm_ok = 0, cam_ok = 0;
        const auto dm = Find(img, code, SigDrawMesh, std::size(SigDrawMesh));
        std::string bad;
        if (dm) {
            for (const auto& c : DrawMeshChecks()) {
                const auto hits =
                    FindAll(img + *dm, DrawMeshSize, c.bytes.data(), u32(c.bytes.size()));
                if (hits.size() != c.count) {
                    bad += fmt::format(" {} x{} (want {})", c.what, hits.size(), c.count);
                }
            }
        }
        if (dm && bad.empty() && (next - cave) + 0x200 <= ptrdiff_t(kCaveBytes)) {
            u8* at = reinterpret_cast<u8*>(eboot_base + *dm);
            u8* p =
                place(BuildCave(reinterpret_cast<u64>(&HookDrawMeshEngine), at, DrawMeshPrologue,
                                reinterpret_cast<u64>(at + DrawMeshPrologue)));
            Detour(at, p);
            dm_engine = true;
            dm_off = dm;
            dm_ok = 1;
            ed += fmt::format("DrawMesh eboot+0x{:x} ({} body checks ok)", *dm,
                             DrawMeshChecks().size());
        } else {
            ed += dm ? fmt::format("DrawMesh body differs:{}", bad) : "DrawMesh not found";
        }
        const auto cs = Find(img, code, SigCamera, std::size(SigCamera));
        if (cs && (next - cave) + 0x200 <= ptrdiff_t(kCaveBytes)) {
            u8* at = reinterpret_cast<u8*>(eboot_base + *cs);
            u8* p = place(BuildCave(reinterpret_cast<u64>(&HookCamera), at, CameraRelocLen,
                                    reinterpret_cast<u64>(at + CameraRelocLen), true));
            Detour(at, p);
            cam_ok = 1;
            ed += fmt::format(", SSAO reprojection eboot+0x{:x}", *cs);
        } else {
            ed += ", SSAO reprojection site not found (unique)";
        }
        MvEngine::SetEngineHooks(dm_ok, cam_ok, ed.c_str());
    }

    // FLVER DrawMesh and its two GX draw calls: per-mesh CPU work on the guest thread, not used
    // by the velocity buffer (bb-clean-velocity: only the skin ring). BB_NATIVE_MV_DRAWMESH=1.
    const char* dm_env = std::getenv("BB_NATIVE_MV_DRAWMESH");
    if (dm_env == nullptr || dm_env[0] != '1') {
        detail += ", DrawMesh / GX draw not hooked (BB_NATIVE_MV_DRAWMESH=1 hooks them)";
        NativeMv::SetHooksInstalled(any, detail.c_str());
        return;
    }
    const auto dm = dm_engine ? dm_off : Find(img, code, SigDrawMesh, std::size(SigDrawMesh));
    const auto calls =
        dm ? Find(img + *dm, DrawMeshSize, SigDrawCalls, std::size(SigDrawCalls)) : std::nullopt;
    if (dm && calls) {
        const u64 site = *dm + *calls;
        const auto target = [&](u32 off) {
            s32 rel;
            std::memcpy(&rel, img + site + off + 1, 4);
            return s64(site + off + 5) + rel;
        };
        const s64 wa = target(CallAOff), wb = target(CallBOff);
        if (wa > 0 && wb > 0 && u64(wa) + 32 <= code && u64(wb) + 32 <= code &&
            std::memcmp(img + wa, PrologueA, sizeof(PrologueA)) == 0 &&
            std::memcmp(img + wb, PrologueB, sizeof(PrologueB)) == 0) {
            u8* at_dm = reinterpret_cast<u8*>(eboot_base + *dm);
            u8* at_wa = reinterpret_cast<u8*>(eboot_base + wa);
            u8* at_wb = reinterpret_cast<u8*>(eboot_base + wb);
            const u32 len_b = sizeof(PrologueB) + 1; // sub rsp, imm8
            u8* p1 =
                dm_engine
                    ? nullptr
                    : place(BuildCave(reinterpret_cast<u64>(&HookDrawMesh), at_dm, DrawMeshPrologue,
                                      reinterpret_cast<u64>(at_dm + DrawMeshPrologue)));
            u8* p2 = place(BuildCave(reinterpret_cast<u64>(&HookEmit), at_wa, sizeof(PrologueA),
                                     reinterpret_cast<u64>(at_wa + sizeof(PrologueA))));
            u8* p3 = place(BuildCave(reinterpret_cast<u64>(&HookEmit), at_wb, len_b,
                                     reinterpret_cast<u64>(at_wb + len_b)));
            NativeMv::SetDrawMeshRange(eboot_base + *dm, eboot_base + *dm + DrawMeshSize);
            Detour(at_wa, p2);
            Detour(at_wb, p3);
            if (dm_engine) {
                g_dm_native = true; // the engine detour calls NativeMv too
            } else {
                Detour(at_dm, p1);
            }
            any = true;
            detail += fmt::format(", DrawMesh eboot+0x{:x}, GX draw eboot+0x{:x} / +0x{:x}, "
                                  "draw calls at eboot+0x{:x}",
                                  *dm, wa, wb, site);
        } else {
            detail += ", GX draw wrapper prologues differ (no DrawMesh hook)";
        }
    } else {
        detail += ", DrawMesh not found";
    }
    NativeMv::SetHooksInstalled(any, detail.c_str());
}

} // namespace BB::NativeMvHook
