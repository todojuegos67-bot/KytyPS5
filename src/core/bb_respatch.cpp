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

bool g_active = false;
u32 g_sw = 0, g_sh = 0, g_ow = 0, g_oh = 0;

std::string Mode() {
    const char* v = std::getenv("BB_SCENE_UPSCALE");
    return v ? std::string(v) : std::string();
}

// "mov r32, imm32" with the template's encoding (B8/B9 + imm32): the immediate, or 0.
u32 ReadImm(uintptr_t base, u32 addr, u8 opcode) {
    const u8* p = reinterpret_cast<const u8*>(base + addr - EbootBase);
    if (p[0] != opcode) return 0;
    u32 v;
    std::memcpy(&v, p + 1, 4);
    return v;
}
} // namespace

bool ShrinkWanted() {
    const std::string mode = Mode();
    return mode == "shrink" || NativeChain();
}

bool NativeChain() {
    return Mode() == "native";
}

bool Active(u32* scene_w, u32* scene_h, u32* out_w, u32* out_h) {
    if (scene_w) *scene_w = g_sw;
    if (scene_h) *scene_h = g_sh;
    if (out_w) *out_w = g_ow;
    if (out_h) *out_h = g_oh;
    return g_active;
}

void OnEbootLoaded(uintptr_t base, u64 image_size) {
    const std::string mode = Mode();
    {
        // 0.81: which lines of work this build carries (bb-imgfg-078 = bb-merge-075 + image frame generation).
        const std::string features =
            "build 0.89 (bb-dlssgpace-089): DLSS-G even pacing (held pacer step BB_DLSSG_PACE_HOLD, dynamic: lowest multiplier reaching the cap BB_DLSSG_DYN_LOWEST, immediate present for dynamic on 120 Hz+, [dlssg-ft] per-second real/shown frame times) + bb-dlssg-085 DLSS-G fixes (reset only on camera cuts, swapchain-loss recovery, G-SYNC/VRR auto cap 224 -> real 112 for DLSS-G, immediate present under the cap, [hitch] lines) on 0.85 (bb-native-085): 0.85 session path lock ([path] locked=..., BB_PATH_LOCK) + 0.84: 0.84 in-place stability (writes after the upscaler no longer move its point, native chain never in scene mode, dump on the first failed native chain check, kept source and sample index, no present-time switch while in place works, history reset only after 30 frames without it, in-place misses by reason) on 0.83: 0.83 native chain (image copies between one size, history images kept at the output size, stale reads by address, bb_dump_frame.txt trigger) on 0.82: 0.80 native+TAA + 0.77 DLSS-G + 0.81 image FG. 0.80 native: 0.80 native chain (the lit scene colour is the only image upscaled across sizes, lighting / compute of the scene at the internal size, content check, BB_DUMP_FRAME) + 0.79 [taa] jitter log, per-SDK jitter sign, BB_TAA_ALL + 0.76 native chain fixes (twin copies for every stale use, no fallback hold, phase per G-buffer, clean-up when off) + 0.73 native chain + 0.74 display meter/FG present. 0.77 DLSS-G: DLSS-G pacing (grid at the 95th percentile of the free frame time), mailbox present mode, no empty tag set, Reflex-report interval in [display], [dlssg] constants / inputs last second, [fsr3] inputs. 0.81 image FG: 0.78 image-only frame generation (BB_FG=image-x2 / image-x3, [imgfg]), self-test. Carries: frame generation pacing of 0.70 (PaceGuestFlip grid, "
            "FSR 3 midpoint slots kytyFfxSetPaceSlot / kytyFfxTakePaceStats, full-frame and "
            "high-res fixes) + 0.71 native chain (BB_SCENE_UPSCALE=native, viewport / MV pixels "
            "of scaled targets, BB_OBJMV_RIGID, Reflex markers at the first sceGnmSubmit, "
            "BB_REFLEX_GAME_LIMIT) + 0.72 one frame limiter (the Reflex game limit yields to the "
            "pacer grid) + 0.73 per-pass native chain (scene passes at the internal size, the output "
            "chain and compute on output-size twins, native chain check, [mv-source] / [mv-probe]) + "
            "0.74 display meter and FG present ([display] on-screen frame rate, VkPresentIdKHR, "
            "sl::DLSSGState probe, BB_FG_PRESENT, KYTY_FSR3_DISPLAY_IDS / kytyFfxTakeDisplayIds, "
            "[streamline] tagged resources)";
        BB::Upscale::Checkpoint(features);
        LOG_INFO(Loader, "[BB] {}", features);
    }
    if (mode == "native") {
        BB::Upscale::Checkpoint(
            "native render chain (BB_SCENE_UPSCALE=native): the game keeps its own output size "
            "(the launcher's resolution patch at the window size); only its 3D scene targets are "
            "rendered at the upscaler's internal size, and the upscaler runs where post-processing "
            "first reads the scene colour, into an image at the output size");
        return;
    }
    if (mode.empty() || mode[0] != '1') {
        return;
    }
    auto inside = [&](u32 addr, u32 size) {
        return addr >= EbootBase && addr - EbootBase + size <= image_size;
    };
    for (const Line& l : Template) {
        if (!inside(l.addr, l.size)) {
            BB::Upscale::Checkpoint("resolution patch: NOT applied: the eboot is smaller than the "
                                    "1.09 one (another game version)");
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
    // The sites as found (after the user's XML patches): logged for verification.
    const u8* sw = reinterpret_cast<const u8*>(base + SceneW - EbootBase);
    BB::Upscale::Checkpoint(fmt::format(
        "resolution patch: eboot heaps {:#x} {:#x} ({}); scene width site {:02X} {:02X} {:02X} "
        "{:02X} {:02X}, scene {}x{} / UI {}x{} as found",
        h0, h1, v109 ? "the 1.09 executable" : "NOT the 1.09 table", sw[0], sw[1], sw[2], sw[3],
        sw[4], ReadImm(base, SceneW, 0xB8), ReadImm(base, SceneH, 0xB8), ReadImm(base, UiW, 0xB8),
        ReadImm(base, UiH, 0xB9)));
    if (!v109) {
        BB::Upscale::Checkpoint("resolution patch: NOT applied (not Bloodborne 1.09)");
        return;
    }
    // The output: the size the user's own Resolution Patch set (the game's size so far), or the
    // window; BB_RES_PATCH_OUTPUT=WxH overrides.
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
    if (std::abs(static_cast<double>(ow) / oh - 16.0 / 9.0) > 0.02) {
        BB::Upscale::Checkpoint(fmt::format("resolution patch: NOT applied: output {}x{} is not "
                                            "16:9 (the patch's aspect constant is 16:9)",
                                            ow, oh));
        return;
    }
    const auto& cfg = BB::Upscale::Cfg();
    double ratio = BB::SR::QualityRatio(cfg.kind, cfg.quality);
    if (!(ratio >= 1.0) || cfg.quality == BB::SR::Quality::Dynamic) ratio = 1.5;
    if (const char* r = std::getenv("BB_RES_PATCH_RATIO")) {
        ratio = std::clamp(std::atof(r), 1.0, 3.0);
    }
    const u32 align = 8;
    const u32 rw = std::max<u32>(align, static_cast<u32>(std::lround(ow / ratio / align)) * align);
    const u32 rh = std::max<u32>(align, static_cast<u32>(std::lround(oh / ratio / align)) * align);
    // 0.69 native scaling: BB_RES_PATCH_UI=output (default) keeps the game's UI size at the
    // output size, so only the 3D scene chain (the SceneW/H immediates: G-buffer, lighting and the
    // game's own scene-size post-processing) renders at the internal size; the game itself then
    // samples the scene into its output-size UI image, the upscaler is injected at that sample
    // (composite mode, bb_upscale.h) and the HUD, menus and text are drawn by the game at the
    // output size: nothing is laid on top afterwards. BB_RES_PATCH_UI=internal restores 0.64
    // (UI at the internal size, the HUD taken off and put back by bb_hud.comp).
    const char* ui_env = std::getenv("BB_RES_PATCH_UI");
    const bool ui_internal = ui_env != nullptr && std::string(ui_env) == "internal";
    const u32 uw = ui_internal ? rw : ow, uh = ui_internal ? rh : oh;
    for (const Line& l : Template) {
        u8 b[4];
        std::memcpy(b, l.bytes, 4);
        u32 value = 0;
        if (l.addr == SceneW) value = rw;
        if (l.addr == SceneH) value = rh;
        if (l.addr == UiW) value = uw;
        if (l.addr == UiH) value = uh;
        if (value) {
            b[1] = value & 0xFF;
            b[2] = (value >> 8) & 0xFF;
            b[3] = (value >> 16) & 0xFF;
        }
        std::memcpy(reinterpret_cast<void*>(base + l.addr - EbootBase), b, l.size);
    }
    g_active = true;
    g_sw = rw;
    g_sh = rh;
    g_ow = ow;
    g_oh = oh;
    BB::Upscale::Checkpoint(fmt::format(
        "resolution patch (BB_SCENE_UPSCALE=1): the game renders its 3D scene at {}x{} (output "
        "{}x{} / {:.3f}), its UI at {}x{} ({}); the upscaler runs inside the game's frame where "
        "it samples the scene. Target shrinking off.",
        rw, rh, ow, oh, ratio, uw, uh,
        ui_internal ? "BB_RES_PATCH_UI=internal: HUD composited on top at the window size"
                    : "native: the game draws its HUD at the output size"));
}
} // namespace BB::ResPatch
