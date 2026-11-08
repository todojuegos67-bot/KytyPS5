// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The patch lines are the community "Resolution Patch 1280x720 (16:9)" (Kyo, AppVer 01.09) as
// shipped in deadinside28/bloodborne_pc patches/Bloodborne.xml (GPL-2.0); the four scene/UI size
// immediates are the only lines that differ between its 16:9 resolutions and are rewritten here
// (scripts/patches.py resolution_writes in the same project).
#include "core/bb_respatch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "common/types.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::ResPatch {
namespace {
constexpr u64 EbootBase = 0x400000;
struct Line {
    u32 addr;
    u32 size;
    u8 bytes[4];
};
constexpr Line Template[] = {
    {0x055289f8, 4, {0x80, 0x02, 0x00, 0x00}},
    {0x055289fc, 4, {0x68, 0x01, 0x00, 0x00}},
    {0x0183a35d, 4, {0x39, 0x8E, 0xE3, 0x3F}},
    {0x02196a6b, 4, {0xB8, 0x00, 0x05, 0x00}},
    {0x02196a6f, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02196a73, 1, {0x00}},
    {0x02358554, 4, {0xB8, 0x00, 0x05, 0x00}},
    {0x02358558, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x0235855c, 1, {0x00}},
    {0x02196a7a, 4, {0xB8, 0xD0, 0x02, 0x00}},
    {0x02196a7e, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02196a82, 1, {0x00}},
    {0x0235855d, 4, {0xB9, 0xD0, 0x02, 0x00}},
    {0x02358561, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02358565, 1, {0x00}},
    {0x019e83af, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x019e83b3, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x019e83b7, 1, {0x00}},
    {0x01ffd491, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x01ffd495, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01ffd499, 1, {0x00}},
    {0x01ffd4d1, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x01ffd4d5, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01ffd4d9, 1, {0x00}},
    {0x01a44357, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x01a4435b, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a4435f, 1, {0x00}},
    {0x01a44c55, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x01a44c59, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a44c5d, 1, {0x00}},
    {0x01a452c7, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x01a452cb, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a452cf, 1, {0x00}},
    {0x01a44365, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x01a44369, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a4436d, 1, {0x00}},
    {0x01a44c63, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x01a44c67, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a44c6b, 1, {0x00}},
    {0x01a452d5, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x01a452d9, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x01a452dd, 1, {0x00}},
    {0x0212c674, 4, {0xBE, 0x80, 0x07, 0x00}},
    {0x0212c678, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x0212c67c, 1, {0x00}},
    {0x0241848f, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x02418493, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02418497, 1, {0x00}},
    {0x02438bb9, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x02438bbd, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02438bc1, 1, {0x00}},
    {0x02438f59, 4, {0xB8, 0x80, 0x07, 0x00}},
    {0x02438f5d, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02438f61, 1, {0x00}},
    {0x0212c6a4, 4, {0xBE, 0x38, 0x04, 0x00}},
    {0x0212c6a8, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x0212c6ac, 1, {0x00}},
    {0x02418498, 4, {0xB9, 0x38, 0x04, 0x00}},
    {0x0241849c, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x024184a0, 1, {0x00}},
    {0x02438bc7, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x02438bcb, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02438bcf, 1, {0x00}},
    {0x02438f67, 4, {0xB8, 0x38, 0x04, 0x00}},
    {0x02438f6b, 4, {0x00, 0x0F, 0x1F, 0x40}},
    {0x02438f6f, 1, {0x00}},
};
constexpr u32 SceneW = 0x02196A6B, SceneH = 0x02196A7A, UiW = 0x02358554, UiH = 0x0235855D;
// The 1.09 data fingerprint: the two graphics heap sizes (originals, or the community "Increased
// Graphics Heap Sizes" values).
constexpr u32 Heap0 = 0x04b36dd8, Heap1 = 0x04b36de0;

u32 ReadImm(uintptr_t base, u32 addr, u8 opcode) {
    const u8* p = reinterpret_cast<const u8*>(base + addr - EbootBase);
    if (p[0] != opcode) return 0;
    u32 v;
    std::memcpy(&v, p + 1, 4);
    return v;
}
} // namespace

// bb-clean-021: with DLSS / FSR 4 at a ratio > 1, the game's scene and UI size immediates are
// rewritten to output / ratio (aligned to 4), so the game renders (and presents) at the
// upscaler's input size; the presenter upscales to the window. The output is the size the
// user's XML Resolution Patch left in the eboot (the launcher's), else the window size.
// BB_RES_PATCH=0: no rewrite (the upscaler then works on whatever size the game presents).
void OnEbootLoaded(uintptr_t base, u64 image_size) {
    if (!BB::Upscale::Enabled()) {
        return;
    }
    if (const char* e = std::getenv("BB_RES_PATCH"); e != nullptr && e[0] == '0') {
        BB::Upscale::Checkpoint("resolution patch: off (BB_RES_PATCH=0)");
        return;
    }
    const auto inside = [&](u32 addr, u32 size) {
        return addr >= EbootBase && addr - EbootBase + size <= image_size;
    };
    for (const Line& l : Template) {
        if (!inside(l.addr, l.size)) {
            BB::Upscale::Checkpoint("resolution patch: NOT applied (not the 1.09 eboot)");
            return;
        }
    }
    if (!inside(Heap0, 8) || !inside(Heap1, 8)) {
        return;
    }
    u64 h0, h1;
    std::memcpy(&h0, reinterpret_cast<const void*>(base + Heap0 - EbootBase), 8);
    std::memcpy(&h1, reinterpret_cast<const void*>(base + Heap1 - EbootBase), 8);
    const bool v109 = (h0 == 0x8ef00000ull || h0 == 0xf7000000ull) &&
                      (h1 == 0x37600000ull || h1 == 0xa0000000ull);
    if (!v109) {
        BB::Upscale::Checkpoint("resolution patch: NOT applied (not Bloodborne 1.09)");
        return;
    }
    u32 ow = ReadImm(base, SceneW, 0xB8), oh = ReadImm(base, SceneH, 0xB8);
    if (const char* o = std::getenv("BB_RES_PATCH_OUTPUT")) {
        unsigned w = 0, h = 0;
        if (std::sscanf(o, "%ux%u", &w, &h) == 2) {
            ow = w;
            oh = h;
        }
    }
    if (ow < 640 || oh < 360 || ow > 7680 || oh > 4320) {
        ow = EmulatorSettings.GetWindowWidth();
        oh = EmulatorSettings.GetWindowHeight();
    }
    const double ratio = BB::Upscale::Ratio();
    if (ratio <= 1.01) {
        BB::Upscale::Checkpoint(fmt::format(
            "resolution patch: ratio 1 (native AA): the game renders at {}x{} as patched", ow, oh));
        return;
    }
    if (std::abs(static_cast<double>(ow) / oh - 16.0 / 9.0) > 0.02) {
        BB::Upscale::Checkpoint(fmt::format(
            "resolution patch: NOT applied: output {}x{} is not 16:9", ow, oh));
        return;
    }
    constexpr u32 align = 4;
    const u32 rw = std::max<u32>(align, static_cast<u32>(std::lround(ow / ratio / align)) * align);
    const u32 rh = std::max<u32>(align, static_cast<u32>(std::lround(oh / ratio / align)) * align);
    for (const Line& l : Template) {
        u8 b[4];
        std::memcpy(b, l.bytes, 4);
        u32 value = 0;
        if (l.addr == SceneW || l.addr == UiW) value = rw;
        if (l.addr == SceneH || l.addr == UiH) value = rh;
        if (value) {
            b[1] = value & 0xFF;
            b[2] = (value >> 8) & 0xFF;
            b[3] = (value >> 16) & 0xFF;
        }
        std::memcpy(reinterpret_cast<void*>(base + l.addr - EbootBase), b, l.size);
    }
    BB::Upscale::Checkpoint(fmt::format(
        "resolution patch: the game renders at {}x{} (output {}x{} / {:.3f}, {} {})", rw, rh, ow,
        oh, ratio, BB::SR::KindName(BB::Upscale::Cfg().kind),
        BB::SR::QualityName(BB::Upscale::Cfg().quality)));
}
} // namespace BB::ResPatch
