// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bloodborne 1.09 (CUSA03173) player camera, eboot code addresses (patch address = + 0x400000):
//   0x1440290..0x14408c0  the camera's debug-menu registration: [r13+off] <- name, which names
//       +0x1e8 RotHiSpeedRateIncTime [sec], +0x1ec/+0x1f0 RotRange MaxX/MinX [deg] (pitch clamp),
//       +0x204/+0x208 RotSpeed X/Y [deg/sec], +0x20c..+0x218 RotSpeedMin/Max X/Y,
//       +0x21c/+0x220 RotHiSpeed X/Y, +0x224..+0x230 RotHiSpeedMin/Max, +0x234 OptionRotSpeedRate.
//   0x143afa1  RotSpeed = Min + (Max - Min) * option^2 (same for RotHiSpeed), every update.
//   0x143b88c  vmovdqa xmm3,[rax]: the camera's stick vector (lane 0 pitch, lane 1 yaw), saved at
//       [rbp-0x3e0]; dt is xmm2 lane 0 ([rbp-0x250]).
//   0x143cc1b  vucomiss |stick|, 0.998; jbe 0x143cd5e: a full stick takes the ramp branch
//       (ramp [r13+0x138] += dt / RotHiSpeedRateIncTime up to 1, 0x143cd22), otherwise ramp = 0.
//   0x143cd5e..0x143ce5d  pitch [r13+0x140] / yaw [r13+0x144] += stick * lerp(RotSpeed,
//       RotHiSpeed, |ramp|) * dt, yaw wrapped to +-pi, pitch clamped to RotRange.
// The same structure as Demon's Souls' free camera (0.998 full-tilt test, ramp to a hi speed).
//
// Patch (two jmp [rip] detours into host memory, both bytes-checked first):
//   site 1 (0x143b88c, 14 bytes): after the stick load, a host call (sysv) that, with kbm play
//     on, replaces the stick vector by mouse counts * rad_per_count / (RotSpeed * dt), so the
//     game's own linear branch turns the camera by exactly the mouse's angle this frame (then
//     wraps / clamps it itself); then the original cmp / jne.
//   site 2 (0x143cc1b, 14 bytes): with kbm play on, always the linear branch (no ramp, and a
//     vector above 1 is not treated as a full stick); otherwise the original test.
// The stick lanes' signs are learned from the game's own vector while the mouse still drives the
// emulated stick (first mouse motion), or set with BB_KBM_YAW_SIGN / BB_KBM_PITCH_SIGN (+1/-1).
// BB_KBM_CAM=0: no patch. BB_KBM_DEG_PER_COUNT (default 0.12) * sensitivity = degrees a count.
#include "core/bb_kbmcam.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "common/logging/log.h"
#include "imgui/bb_menu.h"
#include "input/input_mouse.h"
#include "SDL3/SDL.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace BB::KbmCam {
namespace {
constexpr u64 EbootBase = 0x400000;
constexpr u64 Site1 = 0x0183b88c, Site1Ret = 0x0183b898, Site1Ne = 0x0183b8e9;
constexpr u64 Site2 = 0x0183cc1b, Site2Hi = 0x0183cc29, Site2Lin = 0x0183cd5e;
constexpr u8 Site1Bytes[] = {0xc5, 0xf9, 0x6f, 0x18, 0x41, 0x80, 0xbd, 0x3c,
                             0x01, 0x00, 0x00, 0x00, 0x75, 0x4f};
constexpr u8 Site2Bytes[] = {0xc5, 0xf8, 0x2e, 0x05, 0x5d, 0x92, 0x4e, 0x03,
                             0x0f, 0x86, 0x35, 0x01, 0x00, 0x00};
// Code read by the hook's maths: the linear branch's rate loads and ramp reset.
struct Check {
    u64 addr;
    u8 bytes[9];
    u32 n;
};
constexpr Check Checks[] = {
    {0x0183cd5e, {0x41, 0xc7, 0x85, 0x38, 0x01, 0x00, 0x00}, 7},       // ramp = 0
    {0x0183cd8f, {0xc4, 0xc1, 0x7a, 0x10, 0x8d, 0x04, 0x02, 0x00, 0x00}, 9}, // RotSpeed X
    {0x0183cdba, {0xc4, 0xc1, 0x7a, 0x10, 0x95, 0x08, 0x02, 0x00, 0x00}, 9}, // RotSpeed Y
    {0x0183b849, {0xc5, 0xf8, 0x28, 0x95, 0xb0, 0xfd, 0xff, 0xff}, 8},       // xmm2 = dt
};
// as(1) output of the two caves (see the header comment); 8-byte holes patched below.
constexpr u8 CaveTemplate[] = {
    0xc5, 0xf9, 0x6f, 0x18, 0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xe4, 0xe0, 0x48, 0x81, 0xec, 0x80,
    0x01, 0x00, 0x00, 0x48, 0x89, 0x04, 0x24, 0x48, 0x89, 0x4c, 0x24, 0x08, 0x48, 0x89, 0x54, 0x24,
    0x10, 0x48, 0x89, 0x74, 0x24, 0x18, 0x48, 0x89, 0x7c, 0x24, 0x20, 0x4c, 0x89, 0x44, 0x24, 0x28,
    0x4c, 0x89, 0x4c, 0x24, 0x30, 0x4c, 0x89, 0x54, 0x24, 0x38, 0x4c, 0x89, 0x5c, 0x24, 0x40, 0xc5,
    0xfa, 0x7f, 0x44, 0x24, 0x50, 0xc5, 0xfa, 0x7f, 0x4c, 0x24, 0x60, 0xc5, 0xfa, 0x7f, 0x54, 0x24,
    0x70, 0xc5, 0xfa, 0x7f, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00, 0xc5, 0xfa, 0x7f, 0xac, 0x24, 0x90,
    0x00, 0x00, 0x00, 0xc5, 0xfa, 0x7f, 0xb4, 0x24, 0xa0, 0x00, 0x00, 0x00, 0xc5, 0xfa, 0x7f, 0xbc,
    0x24, 0xb0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x7f, 0x84, 0x24, 0xc0, 0x00, 0x00, 0x00, 0xc5, 0x7a,
    0x7f, 0x8c, 0x24, 0xd0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x7f, 0x94, 0x24, 0xe0, 0x00, 0x00, 0x00,
    0xc5, 0x7a, 0x7f, 0x9c, 0x24, 0xf0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x7f, 0xa4, 0x24, 0x00, 0x01,
    0x00, 0x00, 0xc5, 0x7a, 0x7f, 0xac, 0x24, 0x10, 0x01, 0x00, 0x00, 0xc5, 0x7a, 0x7f, 0xb4, 0x24,
    0x20, 0x01, 0x00, 0x00, 0xc5, 0x7a, 0x7f, 0xbc, 0x24, 0x30, 0x01, 0x00, 0x00, 0xc5, 0xfa, 0x7f,
    0x9c, 0x24, 0x60, 0x01, 0x00, 0x00, 0x4c, 0x89, 0xef, 0x48, 0x8d, 0xb4, 0x24, 0x60, 0x01, 0x00,
    0x00, 0xc5, 0xf8, 0x28, 0xc2, 0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xff,
    0xd0, 0xc5, 0xfa, 0x6f, 0x9c, 0x24, 0x60, 0x01, 0x00, 0x00, 0xc5, 0xfa, 0x6f, 0x44, 0x24, 0x50,
    0xc5, 0xfa, 0x6f, 0x4c, 0x24, 0x60, 0xc5, 0xfa, 0x6f, 0x54, 0x24, 0x70, 0xc5, 0xfa, 0x6f, 0xa4,
    0x24, 0x80, 0x00, 0x00, 0x00, 0xc5, 0xfa, 0x6f, 0xac, 0x24, 0x90, 0x00, 0x00, 0x00, 0xc5, 0xfa,
    0x6f, 0xb4, 0x24, 0xa0, 0x00, 0x00, 0x00, 0xc5, 0xfa, 0x6f, 0xbc, 0x24, 0xb0, 0x00, 0x00, 0x00,
    0xc5, 0x7a, 0x6f, 0x84, 0x24, 0xc0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x6f, 0x8c, 0x24, 0xd0, 0x00,
    0x00, 0x00, 0xc5, 0x7a, 0x6f, 0x94, 0x24, 0xe0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x6f, 0x9c, 0x24,
    0xf0, 0x00, 0x00, 0x00, 0xc5, 0x7a, 0x6f, 0xa4, 0x24, 0x00, 0x01, 0x00, 0x00, 0xc5, 0x7a, 0x6f,
    0xac, 0x24, 0x10, 0x01, 0x00, 0x00, 0xc5, 0x7a, 0x6f, 0xb4, 0x24, 0x20, 0x01, 0x00, 0x00, 0xc5,
    0x7a, 0x6f, 0xbc, 0x24, 0x30, 0x01, 0x00, 0x00, 0x48, 0x8b, 0x04, 0x24, 0x48, 0x8b, 0x4c, 0x24,
    0x08, 0x48, 0x8b, 0x54, 0x24, 0x10, 0x48, 0x8b, 0x74, 0x24, 0x18, 0x48, 0x8b, 0x7c, 0x24, 0x20,
    0x4c, 0x8b, 0x44, 0x24, 0x28, 0x4c, 0x8b, 0x4c, 0x24, 0x30, 0x4c, 0x8b, 0x54, 0x24, 0x38, 0x4c,
    0x8b, 0x5c, 0x24, 0x40, 0x48, 0x89, 0xec, 0x5d, 0x41, 0x80, 0xbd, 0x3c, 0x01, 0x00, 0x00, 0x00,
    0x75, 0x0e, 0xff, 0x25, 0x00, 0x00, 0x00, 0x00, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,
    0xff, 0x25, 0x00, 0x00, 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x50, 0x48,
    0xb8, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x80, 0x38, 0x00, 0x58, 0x75, 0x18, 0xc5,
    0xf8, 0x2e, 0x05, 0x1e, 0x00, 0x00, 0x00, 0x76, 0x0e, 0xff, 0x25, 0x00, 0x00, 0x00, 0x00, 0x55,
    0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xff, 0x25, 0x00, 0x00, 0x00, 0x00, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0xee, 0x7c, 0x7f, 0x3f
};
constexpr u32 OffHook = 215, OffS1Ret = 424, OffS1Ne = 438, OffCave2 = 446, OffFlag = 449,
              OffS2Hi = 479, OffS2Lin = 493;

std::atomic<u8>* g_flag = nullptr; // read by cave 2: the ramp is skipped
bool g_installed = false;
std::atomic<int> g_yaw_sign{0}, g_pitch_sign{0};
std::atomic<int> g_out_x{0}, g_out_y{0};
std::mutex g_mouse_mutex;
float g_acc_timer_x = 0, g_acc_timer_y = 0, g_acc_hook_x = 0, g_acc_hook_y = 0;
float g_deg_per_count = 0.12f;
u64 g_frames = 0, g_injected = 0;
float g_sum_deg_yaw = 0;
u64 g_last_log = 0;

int EnvSign(const char* n) {
    const char* v = std::getenv(n);
    if (!v || !*v) return 0;
    return std::atoi(v) < 0 ? -1 : 1;
}

// Mouse dx -> the yaw lane, dy (after invert) -> the pitch lane, in the sign the emulated stick
// would have given them.
void PS4_SYSV_ABI Hook(const u8* cam, float* vec, float dt) {
    const bool on = BbMenu::KbmPlay() && Input::GetMouseMode() == Input::MouseMode::Joystick &&
                    !BbMenu::IsOpen();
    g_flag->store(on ? 1 : 0, std::memory_order_relaxed);
    if (!on) {
        std::lock_guard lk{g_mouse_mutex};
        g_acc_hook_x = g_acc_hook_y = 0; // nothing saved up for later
        return;
    }
    float dx = 0, dy = 0;
    {
        std::lock_guard lk{g_mouse_mutex};
        float sx = 0, sy = 0;
        SDL_GetRelativeMouseState(&sx, &sy);
        g_acc_timer_x += sx, g_acc_timer_y += sy;
        dx = g_acc_hook_x + sx, dy = g_acc_hook_y + sy;
        g_acc_hook_x = g_acc_hook_y = 0;
    }
    if (Input::MouseInvertY()) dy = -dy;
    g_frames++;
    // learn the signs from the game's vector while the timer drives the stick
    const int ox = g_out_x.load(std::memory_order_relaxed);
    const int oy = g_out_y.load(std::memory_order_relaxed);
    if (g_yaw_sign.load() == 0 && std::abs(ox) >= 40 && std::abs(vec[1]) > 0.15f) {
        g_yaw_sign = (vec[1] > 0) == (ox > 0) ? 1 : -1;
        LOG_INFO(Input, "[kbm-cam] yaw sign learned: {} (stick x {}, game yaw lane {:.2f})",
                 g_yaw_sign.load(), ox, vec[1]);
    }
    if (g_pitch_sign.load() == 0 && std::abs(oy) >= 40 && std::abs(vec[0]) > 0.15f) {
        g_pitch_sign = (vec[0] > 0) == (oy > 0) ? 1 : -1;
        LOG_INFO(Input, "[kbm-cam] pitch sign learned: {} (stick y {}, game pitch lane {:.2f})",
                 g_pitch_sign.load(), oy, vec[0]);
    }
    if (!Direct()) return;
    float rate_x = 0, rate_y = 0;
    std::memcpy(&rate_x, cam + 0x204, 4);
    std::memcpy(&rate_y, cam + 0x208, 4);
    if (!(dt > 1e-4f) || !(rate_x > 1e-3f) || !(rate_y > 1e-3f)) return;
    if (dx == 0 && dy == 0) return; // no mouse this frame: the pad's own stick, if any
    const float sens = std::max(Input::GetMouseSensitivity(), 0.01f);
    const float rad = g_deg_per_count * sens * 3.14159265f / 180.0f;
    vec[1] = static_cast<float>(g_yaw_sign.load()) * dx * rad / (rate_y * dt);
    vec[0] = static_cast<float>(g_pitch_sign.load()) * dy * rad / (rate_x * dt);
    vec[2] = 0;
    g_injected++;
    g_sum_deg_yaw += std::fabs(dx) * rad * 180.0f / 3.14159265f;
    const u64 now = SDL_GetTicks();
    if (now - g_last_log >= 2000) {
        LOG_INFO(Input,
                 "[kbm-cam] direct: {} of {} frames turned by the mouse, {:.0f} deg yaw, "
                 "{:.3f} deg/count, RotSpeed {:.0f}/{:.0f} deg/s, dt {:.4f}",
                 g_injected, g_frames, g_sum_deg_yaw, g_deg_per_count * sens,
                 rate_x * 57.2958f, rate_y * 57.2958f, dt);
        g_injected = g_frames = 0, g_sum_deg_yaw = 0, g_last_log = now;
    }
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

void Put64(u8* p, u64 v) {
    std::memcpy(p, &v, 8);
}

void Detour(u8* at, const u8* to) {
    // jmp qword [rip+0]; .quad to   (14 bytes)
    u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
    Put64(j + 6, reinterpret_cast<u64>(to));
    std::memcpy(at, j, sizeof(j));
}
} // namespace

bool Direct() {
    return g_installed && g_yaw_sign.load(std::memory_order_relaxed) != 0 &&
           g_pitch_sign.load(std::memory_order_relaxed) != 0 &&
           g_flag->load(std::memory_order_relaxed) != 0;
}

void PumpMouse(float* dx, float* dy) {
    std::lock_guard lk{g_mouse_mutex};
    float sx = 0, sy = 0;
    SDL_GetRelativeMouseState(&sx, &sy);
    if (g_installed) g_acc_hook_x += sx, g_acc_hook_y += sy;
    *dx = g_acc_timer_x + sx, *dy = g_acc_timer_y + sy;
    g_acc_timer_x = g_acc_timer_y = 0;
}

void NoteStickOut(int x, int y) {
    g_out_x.store(x, std::memory_order_relaxed);
    g_out_y.store(y, std::memory_order_relaxed);
}

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size) {
    if (const char* v = std::getenv("BB_KBM_CAM"); v && v[0] == '0') {
        LOG_INFO(Input, "[kbm-cam] off (BB_KBM_CAM=0): the game's camera code kept");
        return;
    }
    if (image_size < Site2Lin - EbootBase + 16) return;
    const auto at = [&](u64 a) { return reinterpret_cast<u8*>(eboot_base + a - EbootBase); };
    bool ok = std::memcmp(at(Site1), Site1Bytes, sizeof(Site1Bytes)) == 0 &&
              std::memcmp(at(Site2), Site2Bytes, sizeof(Site2Bytes)) == 0;
    for (const Check& c : Checks) ok = ok && std::memcmp(at(c.addr), c.bytes, c.n) == 0;
    // the site-2 compare's constant is 0.998 (rip + 8 + 0x34e925d)
    float limit = 0;
    if (ok) std::memcpy(&limit, at(Site2 + 8 + 0x034e925d), 4);
    if (!ok || !(limit > 0.997f && limit < 0.999f)) {
        LOG_INFO(Input, "[kbm-cam] Bloodborne 1.09 camera code not found: no camera patch");
        return;
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (!cave) {
        LOG_WARNING(Input, "[kbm-cam] no executable memory for the camera hook: no patch");
        return;
    }
    std::memcpy(cave, CaveTemplate, sizeof(CaveTemplate));
    g_flag = reinterpret_cast<std::atomic<u8>*>(cave + 2048);
    g_flag->store(0);
    Put64(cave + OffHook, reinterpret_cast<u64>(&Hook));
    Put64(cave + OffS1Ret, reinterpret_cast<u64>(at(Site1Ret)));
    Put64(cave + OffS1Ne, reinterpret_cast<u64>(at(Site1Ne)));
    Put64(cave + OffFlag, reinterpret_cast<u64>(cave + 2048));
    Put64(cave + OffS2Hi, reinterpret_cast<u64>(at(Site2Hi)));
    Put64(cave + OffS2Lin, reinterpret_cast<u64>(at(Site2Lin)));
    g_yaw_sign = EnvSign("BB_KBM_YAW_SIGN");
    g_pitch_sign = EnvSign("BB_KBM_PITCH_SIGN");
    if (const char* v = std::getenv("BB_KBM_DEG_PER_COUNT"); v && *v) {
        g_deg_per_count = std::clamp(static_cast<float>(std::atof(v)), 0.001f, 2.0f);
    }
    Detour(at(Site1), cave);
    Detour(at(Site2), cave + OffCave2);
    g_installed = true;
    LOG_INFO(Input,
             "[kbm-cam] Bloodborne camera patch applied (eboot+0x{:x} stick hook, eboot+0x{:x} "
             "full-tilt ramp skipped while kbm play is on): mouse counts turn the camera "
             "directly, {:.3f} deg/count x sensitivity, no ramp / cap / smoothing; signs yaw {} "
             "pitch {} (0 = learned on the first mouse motion)",
             Site1 - EbootBase, Site2 - EbootBase, g_deg_per_count, g_yaw_sign.load(),
             g_pitch_sign.load());
}

} // namespace BB::KbmCam
