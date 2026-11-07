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
bool g_hud_sharp = false;
u32 g_sw = 0, g_sh = 0, g_ow = 0, g_oh = 0;

std::string Mode() {
    // 0.93b: unset / "auto" = "1", the DCport placement: the game renders its whole scene
    // chain (scene passes and its own scene-size post effects) at the internal size through its
    // own resolution values, its UI at the output size, and the upscaler runs where the game
    // composites the scene into the output-size image, before the HUD. The generic per-target
    // chain ("native") stays opt-in: in the 0.93 run its later consumers (image copies, depth
    // reads) still saw internal-size images with output-size coordinates.
    const char* v = std::getenv("BB_SCENE_UPSCALE");
    std::string m = v ? std::string(v) : std::string();
    // 1.33 (bb-aapass-133): "1" / unset / "auto" / launcher sceneup = "aapass": the upscaler at
    // Bloodborne's own post-processing entry (the scene-colour sample of its first post pass,
    // matched by shader hash, bb_upscale.cpp): DLAA in place at ratio 1, at a ratio > 1 the
    // native chain (scene at the internal size, post chain and HUD at the output size). The 0.93b
    // resolution patch is BB_SCENE_UPSCALE=scene093 (old behaviour, kept behind the switch).
    if (m.empty() || m == "auto" || m == "1") m = "aapass";
    return m;
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

bool HudSharp() {
    return g_hud_sharp;
}

bool NativeChain() {
    // 0.93: the native render chain is the default (BB_SCENE_UPSCALE unset or "auto"). It is
    // generic: the scene's depth target, its lit HDR colour and the first pass that samples it
    // (post-processing) are found from the GPU command stream, nothing of the game is patched. It
    // only engages when an upscaler runs with a ratio > 1 and the game's 3D renders at its output
    // size (bb_upscale.cpp); otherwise nothing changes. BB_SCENE_UPSCALE=0 turns it off.
    // 1.44: BB_SCENE_UPSCALE=prepost is the native chain with Bloodborne's pass roles fixed by
    // shader hash (bb_upscale.cpp, PP section).
    return Mode() == "native" || Mode() == "prepost"; // 1.35: aapass no longer uses the native chain
}

bool Prepost() {
    static const bool on = Mode() == "prepost";
    return on;
}

bool AaPass() {
    return Mode() == "aapass";
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
            "build 1.49 (bb-combo-149): 1.44 (bb-prepost-144) + bb-trophysync-148 (retroactive Bloodborne trophies from the player's own save, once per boot, normal popup + chime one by one, 3.5 s cards, backup .trophy-backup-before-sync, [trophy-sync] log, BB_TROPHY_SYNC=0 off; platinum rule fix). "
            "build 1.43 (bb-combo-143): content check rejects only flat mid-grey canvases (all samples within 3 of one value, 0x7D +-12; dark scenes never), BB_FLATCHECK=0 turns it off. build 1.41 (bb-combo-141): 1.42 flash guard (bb-flash-142, supersedes 1.40) + FG pairing/coverage diagnostics (bb-fg30-141) + rumble fix/diag (bb-rumble-143) + mouse direct camera (bb-kbmcam-142). "
            "build 1.44 (bb-prepost-144): BB_SCENE_UPSCALE=prepost (opt-in) + DLSS range fix (an out-of-range menu upscale no longer turns DLSS off: r143 Quality never upscaled): DLSS on the lit HDR scene before exposure / bloom / tonemap / HUD, the game at the output size, Bloodborne's passes classified by shader hash (scene passes from the G-buffer draw to cs 0x9a9cf8a9 / 0x200820a1 / ps 0x00000ccc53f28aa5 at the internal size, everything after at the output size, phase-1 copies and samples of scene-size images internal, no history pinning, no learned UI / post targets), [prepost] per-second line with the target table and a mixed-size sample counter; default (aapass resolution patch) unchanged. "
            "build 1.41 (bb-combo-141): 1.42 flash guard (bb-flash-142, supersedes 1.40) + FG pairing/coverage diagnostics (bb-fg30-141) + rumble fix/diag (bb-rumble-143) + mouse direct camera (bb-kbmcam-142). "
            "build 1.41 (bb-fg30-141): 1.40 FSR 3 FG diagnostics + [fg-amd] 1.41 pairing line (colour / depth / camera MV / object MV / HUD-less guest frame ids, what FSR 3 was told, character pixels with near-zero vs own-motion vectors). "
            "build 1.42 (bb-flash-142): GPU content check of every presented HUD canvas (a flat grey scene region shows the last frame / black, max 30 in a row; [flash-guard] reports REJECTED), the canvas stays armed when another flip is presented mid-frame (fewer repeated frames), [flash-guard] other image diagnostics. build 1.40 (bb-flash-140): flash guard: a frame whose flip was not written with its scene (its draws went into a HUD canvas, or a canvas only cleared) repeats the last presented canvas (else black), never the guest image; FG skips pairs across a guard or a present-path switch; [flash-guard] per-second line. build 1.39 (bb-hud-139): HUD canvas carries the UI image alpha (bb_alpha.comp on the window path), + bb-input-137 (rumble / mouse / kbm defaults) + bb-trophy-134 (new game re-arms custom trophies per save slot). build 1.38 (bb-hud-138): the HUD canvas is never turned off for the session (frames without a final pass from the HUD target fall back for that frame only). build 1.37 (bb-hud-137): HUD canvas per flip image (re-presented / out-of-order flips use their own canvas: no white frames), depth-only HUD draws (bar masks) into the depth twin (health / stamina bars back), output = swapchain extent (2560 not 2556), internal 1706x960, [hud] draws by kind. build 1.36 (bb-combo-136): 1.35 scaled HUD canvas + input fixes (mouse, touchpad key, vibration) + mapping wizard + custom trophies. build 1.35 (bb-aapass-135): aapass at a ratio > 1 = the game's resolution patch (scene + post at the internal size) + upscale at the HUD point + output-size HUD canvas, no native chain (BB_SCENE_UPSCALE=native opt-in); native DLAA at the post entry unchanged. build 1.34 (bb-aapass-134): aapass at a ratio > 1 upscales at the scene colour's first read (before the exposure compute) instead of holding to the post entry (1.33: dark scene, doubled characters; BB_AAPASS_CHAIN_HOLD=1 old); native DLAA unchanged. build 1.33 (bb-aapass-133): BB_SCENE_UPSCALE=aapass (default for 1 / sceneup): upscaler / DLAA at the scene-colour sample of the first post pass (ps 0x00000ccc53f28aa5), native DLAA there too (BB_AAPASS=0 old in-place point), 0.93b only as BB_SCENE_UPSCALE=scene093, [aapass] per-second line, + bb-perf-132 (BB_CP_FASTPATCH, [perf] cp2). build 1.32 (bb-aacap-132): 1.31 + 1.29 aacap (HUD point on any post target, scaled frames never locked in place, DLAA write-back) + AA-pass capture v2 (whole frame: 20000 lines, no compressed-texture samples) + build 1.31 (bb-dlss5flk-131): 1.29 + DLSS 5 native / DLSS-G flicker fix (a frame whose DLSS 5 ran at present gives DLSS-G the post-DLSS 5 image as HUD-less colour, not the pre-DLSS 5 copy; DLSS 5 history reset when its point switches before-HUD / at-present; [dlss5-fg] per-second line; BB_DLSS5_FLK_FIX=0 old). build 1.29 (bb-skinmv-129): 1.27 + skinned object MVs keyed by bone palette address first (enemies of one model no longer swap previous positions by draw order, BB_OBJMV_PALETTE_ID=0 old), [fg-skinmv] per-second line. build 1.27 (bb-dlaa-127): 1.26 + DLAA stability (DLAA result written back into the scene colour so storage / compute / copy reads see it, BB_INPLACE_WRITEBACK=0 old; full-screen passes before the in-place point no longer jittered: no double offset of the lit scene, [taa-stab] per-second line, BB_JITTER_FULLSCREEN=1 old). build 1.26 (bb-mix-126): 1.24 + 1.21 F1 menu (full-screen hub, two-pad mapping, exclusive capture, OFL fonts, trophy grid/popup). 1.24: 1.23 + 1.20 (object MVs double-buffered by frame parity, loose identity, barrier, BB_OBJMV_DEBUG, BB_MV_VIEW). 1.23: 1.22 + 1.19 (no tuning/sampling of guest threads, CP back-off yields only). 1.22: 1.17 + 1.15 DLSS 5 stable + 1.16 HUD canvas | 1.16 (bb-hud-116): HUD out of the temporal upscaler in scaled modes: the upscaler / DLSS 5 run on the scene at the HUD point (first HUD draw into the UI or presented image after the scene composite), the HUD / menus are re-rasterised at the output size into an output-size canvas (the game's final UI-to-flip pass re-run on it), FG gets the HUD-less twin of that pass and the real HUD on generated frames ([hud] per-second line, BB_HUD_SHARP=0: 1.14). | build build 1.15 (bb-dlss5fix-115): 1.14 + DLSS 5 temporally stable (A/B off by default, BB_DLSS5_AB=1 runs it once; foveated grid off by default, BB_DLSS5_FOVEATED=1 opt-in). | build 1.17 (bb-mix-117): 1.14 + 1.13 (F1 menu animations, Opciones: inicio directo, custom trophy Despertar en Yharnam, trophy icon animation). build 1.14 (bb-mix-114): 1.11 + 1.12 (camera scan skipped once the block camera is known, [perf] cp ms/s breakdown). build 1.11 (bb-fgfix-111): 1.10 + 1.03 object MVs per index range / HUD-less late fix + FG fix (F1 menu layer no longer forces a re-present every vblank: DLSS-G / FSR 3 FG see only guest flips, inputs every 3D frame, dynamic multiplier and guest-release pacing back; no jitter on draws after the in-place point: HUD bars no longer shimmer with native AA). build 1.10 (bb-mix-110): 1.09 + 1.06 perf (GPU command processor back-off, 4+ present frames, thread priority, BB_AFFINITY, [perf] line). build 1.09: 1.08 + 1.04 F1 menu (mando, teclado y raton, trofeos, salir; trophy popup; npcommid fallback). build 1.08: 1.02 + 1.05. 1.05 (bb-fgpure-105): FG without an upscaler locks its input path for the session (before-HUD or at-present, BB_FGN_LOCK), every 3D frame gets inputs, camera-cut test on far-plane points + 4x-average spike rule (no resets on normal turns), [fg-pure] per-second line + build 1.02 (bb-native-102): native AA (ratio 1) uses the in-place scene-colour point, locked, DLSS 5 after it, no HUD-point/present switching + 1.01 (bb-fix-101): UI canvas follows the scene size by default (BB_RES_PATCH_UI=internal; title, menus and HUD at native layout, no crop/zoom) + 1.00 (bb-merge-100): 0.98 hitch fixes ([hitch] cause tags, DLSS 5 GPU timestamps on 1 frame in 16, chunked buffer residency) + 0.99: merge of 0.96 (bb-dyn-096) + 0.93b (bb-scene-093: DCport scene upscale BB_SCENE_UPSCALE=1, scene+post at the internal size, upscale at the composite before the HUD) + 0.97 (bb-inject-097: shared injection point after post before the HUD for DLAA / DLSS 5 / FG inputs at ratio 1 or BB_SCENE_UPSCALE=0, [inject]) + 0.94 (bb-dcideas-094: [sr-hist] resets on first/cut/size change, [fg-gate] FG off in menus/movies, FSR 4.1.1 with FSR 4.0 fallback, metrics show the real upscaler); 0.96 (bb-dyn-096): dynamic DLSS-G picks the highest multiplier with real * N <= cap (x2 floor, 2 s hysteresis, BB_DLSSG_DYN_LOWEST=1 for the 0.89 rule) on 0.95 (bb-merge-095): merge of 0.91 (bb-fgnative-090: DLSS 5 without an upscaler: NGX address hole reserved whenever BB_DLSS5>=1, DLSS device order kept) + 0.93 (bb-scene-093: native scene chain default on, internal-resolution scene upscale locks in place for ratios > 1, no window fallback) + 0.92 (bb-amdfg-092): 0.92 FSR 3 FG held-step present slots (BB_FSR3_PACE_HOLD), [fsr3-ft] real/shown frame times, FSR 3 FG finite far plane (KYTY_FSR3_FAR_INF), camera motion vectors at present (prev_vp kept until OnFlip ends, from 0.90) + 0.91 (bb-aa-091): 0.91 DLAA jitter cycle >= 32 phases (BB_JITTER_PHASES) + 0.89: DLSS-G even pacing (held pacer step BB_DLSSG_PACE_HOLD, dynamic: lowest multiplier reaching the cap BB_DLSSG_DYN_LOWEST, immediate present for dynamic on 120 Hz+, camera cut threshold 3.0 NDC (BB_SR_CUT_NDC, BB_DLSSG_CUT_NDC), [dlssg-ft] per-second real/shown frame times) + bb-dlssg-085 DLSS-G fixes (reset only on camera cuts, swapchain-loss recovery, G-SYNC/VRR auto cap 224 -> real 112 for DLSS-G, immediate present under the cap, [hitch] lines) on 0.85 (bb-native-085): 0.85 session path lock ([path] locked=..., BB_PATH_LOCK) + 0.84: 0.84 in-place stability (writes after the upscaler no longer move its point, native chain never in scene mode, dump on the first failed native chain check, kept source and sample index, no present-time switch while in place works, history reset only after 30 frames without it, in-place misses by reason) on 0.83: 0.83 native chain (image copies between one size, history images kept at the output size, stale reads by address, bb_dump_frame.txt trigger) on 0.82: 0.80 native+TAA + 0.77 DLSS-G + 0.81 image FG. 0.80 native: 0.80 native chain (the lit scene colour is the only image upscaled across sizes, lighting / compute of the scene at the internal size, content check, BB_DUMP_FRAME) + 0.79 [taa] jitter log, per-SDK jitter sign, BB_TAA_ALL + 0.76 native chain fixes (twin copies for every stale use, no fallback hold, phase per G-buffer, clean-up when off) + 0.73 native chain + 0.74 display meter/FG present. 0.77 DLSS-G: DLSS-G pacing (grid at the 95th percentile of the free frame time), mailbox present mode, no empty tag set, Reflex-report interval in [display], [dlssg] constants / inputs last second, [fsr3] inputs. 0.81 image FG: 0.78 image-only frame generation (BB_FG=image-x2 / image-x3, [imgfg]), self-test. Carries: frame generation pacing of 0.70 (PaceGuestFlip grid, "
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
    if (mode == "aapass") {
        BB::Upscale::Checkpoint(
            "[aapass] BB_SCENE_UPSCALE=aapass (launcher sceneup, 1.35): ratio 1: DLAA at the "
            "game's post entry (ps 0x00000ccc53f28aa5); ratio > 1: the game's resolution patch "
            "renders scene + post at the internal size, the upscaler runs at the HUD point and "
            "the HUD is drawn into an output-size canvas (no native chain, no twin images; "
            "BB_SCENE_UPSCALE=native: the native chain)");
    }
    if (mode == "prepost") {
        BB::Upscale::Checkpoint(
            "[prepost] BB_SCENE_UPSCALE=prepost (1.44, opt-in): the game renders at its output "
            "size (no eboot rewrite); the scene passes (G-buffer draw .. the post entry) run at the "
            "internal size, DLSS runs on the lit HDR scene colour sampled by ps 0x00000ccc53f28aa5 "
            "and the exposure / bloom / tonemap / HUD / flip passes run at the output size. Needs "
            "the launcher's resolution patch at the window size. Default path unchanged.");
        return;
    }
    if (mode == "native") {
        BB::Upscale::Checkpoint(
            "native render chain (BB_SCENE_UPSCALE=native/auto, 0.93 default): the game keeps its own output size "
            "(the launcher's resolution patch at the window size); only its 3D scene targets are "
            "rendered at the upscaler's internal size, and the upscaler runs where post-processing "
            "first reads the scene colour, into an image at the output size");
        return;
    }
    if (mode != "scene093" && mode != "aapass") {
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
    const u32 align = 2; // 1.37: 2560 / 1.5 -> 1706 (8 gave 1704: output 2556)
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
    // 1.01: the default is "internal". With the UI immediates at the output size the game still
    // renders its final (flip) image at the SceneW/H size: the 0.93b log shows "game output
    // 1280x720" with UI 2560x1440, so the HUD / title / menus were laid out in output-size
    // coordinates inside an internal-size target (cropped on the right) and then scaled up by
    // the window path (too big). Layout and target now agree; BB_RES_PATCH_UI=output keeps 0.93b.
    const bool ui_internal = !(ui_env != nullptr && std::string(ui_env) == "output");
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
    {
        // 1.16 (bb-hud-116, from 1.07): the HUD canvas in every scaled mode (ratio > 1).
        const char* hs = std::getenv("BB_HUD_SHARP");
        const bool hs_off = hs != nullptr && *hs == '0';
        g_hud_sharp = ui_internal && !hs_off && (rw < ow || rh < oh);
        BB::Upscale::Checkpoint(fmt::format(
            "[hud] {}: {}", g_hud_sharp ? "before-HUD upscale + output-size HUD canvas" : "off",
            g_hud_sharp
                ? fmt::format("scene {}x{} upscaled at the HUD point, HUD laid out at {}x{} and "
                              "rasterised at the output size {}x{} (x{:.3f})",
                              rw, rh, uw, uh, ow, oh, double(ow) / std::max<u32>(uw, 1))
                : fmt::format("HUD at {}x{} ({})", uw, uh,
                              hs_off         ? "BB_HUD_SHARP=0"
                              : !ui_internal ? "BB_RES_PATCH_UI=output"
                                             : "ratio 1, native AA in place")));
    }
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
