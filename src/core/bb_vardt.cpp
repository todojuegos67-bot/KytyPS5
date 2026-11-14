// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bloodborne 1.09 (CUSA03173), BB_VARIABLE_DT=1. Addresses below are XML / Ghidra addresses
// (eboot vaddr + 0x400000), the same convention as the patch XML.
//
// Evidence (eboot 1.09, vaddr):
//  - 0x2034770 frame limiter (this = *(0x55404F8)): measures the frame with gettimeofday
//    (call 0x2bbe728 -> libc gettimeofday) and sleeps with sceKernelUsleep (0x2bbfe68); stores
//    the measured frame time in seconds at this+0x264 (0x2034cf0); the target period is
//    this+0x18 (1/30 or 1/60 from the mode switch at 0x2034806).
//  - 0x2018E36 original: mov dword [rbp-0x38], 0x3D088889 (1/30): the FD4 task manager is run
//    (call 0x20512a0) with a FIXED 1/30 s per frame, whatever the real frame time is.
//  - "PC sin limite 1" replaces 0x2018E3D with
//        vmovss xmm0,[rbx+0x264]; vmaxss xmm0,xmm0,[rbx+0x18]; vminss xmm0,xmm0,[rbp-0x38]
//        vmovss [rbp-0x38],xmm0;  vmovss [0x56D3FC4],xmm0
//    and the immediate at 0x2018E39 with 1/60, so dt = min(max(measured, period), 1/60): below
//    60 fps every frame advances the game by 1/60 s only (slow motion: 45 fps -> 0.75x).
//  - "PC sin limite 1" also freezes 41 per-frame constants that the game reads as "seconds per
//    frame at 30 fps" (orig 1/30) or per-frame factors (orig 1.0, 0.95) at a 100 fps (0.01) or
//    60 fps (1/60, 0.5, 0.95^0.5) value: they are right only at that frame rate.
// What this does: the upper clamp becomes BB_VARIABLE_DT_MAX_MS (default 50 ms = 1/20 s) and
// the frozen constants are evaluated every frame at the live dt with the same law the patch
// used (linear: orig * dt * 30; decay: 0.95^(dt*30)), so at dt = 0.01 they are byte-identical
// to the patch. The Havok world keeps "PC sin limite 3" (semi-fixed steps h = min(acc, 1/60),
// acc += live dt, acc capped at 1/15 s), so a 1/20 s frame runs 3 physics substeps.
#include "core/bb_vardt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
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

namespace BB::VarDt {
namespace {
constexpr u64 EbootBase = 0x400000;

// "PC sin limite 1" dt hook (bb-physics-unlimited.xml, entry 1, Address 0x02418E3D).
constexpr u64 DtImm = 0x02418E39; // imm32 of mov dword [rbp-0x38], imm  (0x2018E36: C7 45 C8)
constexpr u8 DtMovPrefix[3] = {0xC7, 0x45, 0xC8};
constexpr u64 DtHook = 0x02418E3D;
constexpr u8 DtHookBytes[] = {0xC5, 0xFA, 0x10, 0x83, 0x64, 0x02, 0x00, 0x00, // vmovss [rbx+0x264]
                              0xC5, 0xFA, 0x5F, 0x43, 0x18,                   // vmaxss [rbx+0x18]
                              0xC5, 0xFA, 0x5D, 0x45, 0xC8,                   // vminss [rbp-0x38]
                              0xC5, 0xFA, 0x11, 0x45, 0xC8,                   // vmovss [rbp-0x38]
                              0xC5, 0xFA, 0x11, 0x05, 0x68, 0xB1, 0x6B, 0x03}; // -> live DT
// "PC sin limite 3" Havok semi-fixed step entry jump (0x17F9B71: jmp 0x17F9C77).
constexpr u64 HavokHook = 0x01BF9B71;
constexpr u8 HavokHookBytes[] = {0xE9, 0x01, 0x01, 0x00, 0x00};

constexpr u64 LiveDt = 0x05AD3FC4;   // vaddr 0x56D3FC4 (written by the dt hook every frame)
constexpr u64 FramePtr = 0x059404F8; // vaddr 0x55404F8: the frame limiter object pointer
constexpr u32 OffPeriod = 0x18, OffMeasured = 0x264;

enum class Law : u8 { Linear, Decay };
struct Site {
    u32 addr;
    float orig;     // value in the unpatched eboot (per frame at 30 fps)
    u32 patched;    // the value "PC sin limite 1" writes (fingerprint)
    Law law;
};
constexpr u32 P100 = 0x3C23D70A; // 0.01
constexpr u32 P60 = 0x3C888889;  // 1/60
constexpr u32 PHalf = 0x3F000000; // 0.5
constexpr u32 PDecay = 0x3F798497; // 0.974679 = 0.95^0.5
constexpr float T30 = 1.0f / 30.0f;
constexpr Site Sites[] = {
    // orig 1/30 -> 0.01 (100 fps)
    {0x04BEFDCC, T30, P100, Law::Linear}, {0x04C88688, T30, P100, Law::Linear},
    {0x04CAD65C, T30, P100, Law::Linear}, {0x04CAD660, T30, P100, Law::Linear},
    {0x04CAD664, T30, P100, Law::Linear}, {0x04CADC08, T30, P100, Law::Linear},
    {0x04CADD28, T30, P100, Law::Linear}, {0x04CAF1BC, T30, P100, Law::Linear},
    {0x04CAF1C0, T30, P100, Law::Linear}, {0x04CAF290, T30, P100, Law::Linear},
    {0x04CAF294, T30, P100, Law::Linear}, {0x04CAF340, T30, P100, Law::Linear},
    {0x04CAF344, T30, P100, Law::Linear}, {0x04CB03B8, T30, P100, Law::Linear},
    {0x04CB03BC, T30, P100, Law::Linear}, {0x04CCC3F4, T30, P100, Law::Linear},
    {0x04D2577C, T30, P100, Law::Linear}, {0x04D2585C, T30, P100, Law::Linear},
    {0x04D25864, T30, P100, Law::Linear}, {0x04D25E94, T30, P100, Law::Linear},
    {0x04D265FC, T30, P100, Law::Linear}, {0x04D26618, T30, P100, Law::Linear},
    {0x04D26C9C, T30, P100, Law::Linear}, {0x04D270E4, T30, P100, Law::Linear},
    {0x04D27250, T30, P100, Law::Linear}, {0x04D27258, T30, P100, Law::Linear},
    {0x04D27554, T30, P100, Law::Linear}, {0x04D275F4, T30, P100, Law::Linear},
    {0x04D27720, T30, P100, Law::Linear}, {0x04D27C4C, T30, P100, Law::Linear},
    {0x04D28E98, T30, P100, Law::Linear},
    // orig 1/30 -> 1/60 (60 fps)
    {0x04D29124, T30, P60, Law::Linear}, {0x04D29128, T30, P60, Law::Linear},
    {0x04D29170, T30, P60, Law::Linear}, {0x04D29E74, T30, P60, Law::Linear},
    {0x04D29B80, T30, P60, Law::Linear},
    // orig 1.0 -> 0.5 (60 fps)
    {0x04B693F0, 1.0f, PHalf, Law::Linear}, {0x04BB63E4, 1.0f, PHalf, Law::Linear},
    {0x04BB8474, 1.0f, PHalf, Law::Linear}, {0x04BB91C0, 1.0f, PHalf, Law::Linear},
    {0x04BB98B8, 1.0f, PHalf, Law::Linear},
    // orig 0.95 per-frame damping (vmulps at 0x1A4ABCA) -> 0.95^0.5 (60 fps)
    {0x04D0B5B0, 0.95f, PDecay, Law::Decay}, {0x04D0B5B4, 0.95f, PDecay, Law::Decay},
    {0x04D0B5B8, 0.95f, PDecay, Law::Decay}, {0x04D0B5BC, 0.95f, PDecay, Law::Decay},
    // Not handled: 0x04D25844 (patch writes 1/120; Kyo's 60 FPS++ writes 1/120 at 0x04D25E94
    // instead, so the law is unknown) -> left as the patch sets it.
};
constexpr size_t NumSites = sizeof(Sites) / sizeof(Sites[0]);

bool g_enabled = false, g_immediate = false, g_installed = false;
uintptr_t g_base = 0;
u64 g_size = 0;
bool g_site_ok[NumSites] = {};
float g_max_dt = 0.05f;

using Clock = std::chrono::steady_clock;
double Now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct Acc {
    double sum = 0, mn = 1e9, mx = 0;
    u32 n = 0;
    void Add(double v) {
        sum += v, mn = std::min(mn, v), mx = std::max(mx, v), ++n;
    }
    double Avg() const {
        return n ? sum / n : 0.0;
    }
    void Reset() {
        *this = Acc{};
    }
};
std::mutex g_mutex;
Acc g_measured, g_used, g_flip_iv, g_present_iv;
double g_last_flip = 0, g_last_present = 0, g_last_log = 0, g_period = 0;
u32 g_mode_vsync = 0, g_mode_imm = 0, g_mode_other = 0, g_bad_dt = 0;

u8* At(u64 a) {
    return reinterpret_cast<u8*>(g_base + a - EbootBase);
}
bool Inside(u64 a, u64 n) {
    return a >= EbootBase && a - EbootBase + n <= g_size;
}

// The constants live in the eboot's R+X segment. The loader maps executable guest segments
// read/write on the host (address_space.cpp: PAGE_EXECUTE_READWRITE), the XML patcher writes
// them the same way; this makes sure of it once for the constant pages.
void MakeWritable(u64 lo, u64 hi) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(At(lo)) & ~uintptr_t(0xFFF);
    const uintptr_t b = (reinterpret_cast<uintptr_t>(At(hi)) + 0x1000) & ~uintptr_t(0xFFF);
#ifdef _WIN32
    DWORD old = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(a), b - a, PAGE_EXECUTE_READWRITE, &old)) {
        LOG_WARNING(Lib_VideoOut, "[vardt] VirtualProtect failed ({}), constants may not update",
                    GetLastError());
    }
#else
    mprotect(reinterpret_cast<void*>(a), b - a, PROT_READ | PROT_WRITE | PROT_EXEC);
#endif
}

float Value(const Site& s, float dt) {
    const float f = dt * 30.0f;
    return s.law == Law::Decay ? std::pow(s.orig, f) : s.orig * f;
}

bool ReadFrame(float* measured, float* period) {
    u64 p = 0;
    std::memcpy(&p, At(FramePtr), 8);
    if (p < 0x10000 || (p & 3) != 0) return false;
    std::memcpy(measured, reinterpret_cast<const void*>(p + OffMeasured), 4);
    std::memcpy(period, reinterpret_cast<const void*>(p + OffPeriod), 4);
    return true;
}
} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_VARIABLE_DT");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

bool ImmediatePresent() {
    static const bool on = [] {
        if (!Enabled()) return false;
        const char* v = std::getenv("BB_VRR_PRESENT");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

void OnEbootLoaded(uintptr_t eboot_base, u64 image_size) {
    g_enabled = Enabled();
    g_immediate = ImmediatePresent();
    if (!g_enabled) {
        return;
    }
    g_base = eboot_base;
    g_size = image_size;
    if (!Inside(LiveDt, 4) || !Inside(FramePtr, 8) || !Inside(DtHook, sizeof(DtHookBytes))) {
        LOG_WARNING(Lib_VideoOut, "[vardt] BB_VARIABLE_DT=1 but this is not the Bloodborne 1.09 "
                                  "eboot: off");
        return;
    }
    if (std::memcmp(At(DtImm - 3), DtMovPrefix, 3) != 0 ||
        std::memcmp(At(DtHook), DtHookBytes, sizeof(DtHookBytes)) != 0) {
        LOG_WARNING(Lib_VideoOut,
                    "[vardt] BB_VARIABLE_DT=1 but 'PC sin limite 1. High FPS base' is not "
                    "applied (dt hook at 0x{:x} not found; 60 FPS++ / Uncap FPS++ / no FPS "
                    "patch are not supported): game dt unchanged. Use launcher FPS option 7.",
                    DtHook);
        return;
    }
    const bool havok = std::memcmp(At(HavokHook), HavokHookBytes, sizeof(HavokHookBytes)) == 0;
    if (const char* m = std::getenv("BB_VARIABLE_DT_MAX_MS"); m != nullptr && *m) {
        g_max_dt = std::clamp(static_cast<float>(std::atof(m)), 17.0f, 100.0f) / 1000.0f;
    }
    float old_max = 0;
    std::memcpy(&old_max, At(DtImm), 4);
    std::memcpy(At(DtImm), &g_max_dt, 4);

    u64 lo = ~0ull, hi = 0;
    u32 ok = 0;
    for (size_t i = 0; i < NumSites; ++i) {
        const Site& s = Sites[i];
        u32 cur = 0;
        if (Inside(s.addr, 4)) std::memcpy(&cur, At(s.addr), 4);
        g_site_ok[i] = cur == s.patched;
        if (g_site_ok[i]) {
            ++ok, lo = std::min<u64>(lo, s.addr), hi = std::max<u64>(hi, s.addr);
        } else {
            LOG_WARNING(Lib_VideoOut,
                        "[vardt] constant 0x{:x} = 0x{:08x}, expected 0x{:08x} (the patch "
                        "value): left alone",
                        s.addr, cur, s.patched);
        }
    }
    if (ok) MakeWritable(lo, hi);
    g_installed = true;
    LOG_INFO(Lib_VideoOut,
             "[vardt] on: game dt = clamp(measured frame+0x264, period frame+0x18, {:.1f} ms) "
             "(was max {:.2f} ms, imm 0x{:x}); {}/{} per-frame constants follow the live dt; "
             "Havok semi-fixed step (PC sin limite 3) {}; presenter {}",
             g_max_dt * 1000.0f, old_max * 1000.0f, DtImm, ok, NumSites,
             havok ? "present (substeps of <= 1/60 s)"
                   : "NOT present: physics runs one step per frame",
             g_immediate ? "flips immediately (BB_VRR_PRESENT, no vblank quantization)"
                         : "on the vblank tick (BB_VRR_PRESENT=0)");
}

void OnGuestFlip(int flip_mode) {
    if (!g_installed) {
        return;
    }
    float dt = 0;
    std::memcpy(&dt, At(LiveDt), 4);
    const bool good = std::isfinite(dt) && dt > 0.0001f && dt <= g_max_dt + 0.0001f;
    if (good) {
        for (size_t i = 0; i < NumSites; ++i) {
            if (!g_site_ok[i]) continue;
            const float v = Value(Sites[i], dt);
            std::memcpy(At(Sites[i].addr), &v, 4);
        }
    }
    float measured = 0, period = 0;
    const bool frame = ReadFrame(&measured, &period);
    const double now = Now();
    std::scoped_lock lk{g_mutex};
    if (good) {
        g_used.Add(dt);
    } else {
        ++g_bad_dt;
    }
    if (frame && std::isfinite(measured)) g_measured.Add(measured);
    if (frame) g_period = period;
    if (g_last_flip > 0) g_flip_iv.Add(now - g_last_flip);
    g_last_flip = now;
    (flip_mode == 1 ? g_mode_vsync : flip_mode == 2 ? g_mode_imm : g_mode_other)++;
    if (g_last_log == 0) g_last_log = now;
    if (now - g_last_log < 2.0) {
        return;
    }
    const float c100 = Value(Sites[0], good ? dt : 0.01f);
    LOG_INFO(Lib_VideoOut,
             "[vardt] {:.1f}s {} frames ({:.1f} fps): measured dt avg {:.2f} min {:.2f} max "
             "{:.2f} ms | game dt used avg {:.2f} min {:.2f} max {:.2f} ms (clamp {:.2f}..{:.1f}) "
             "| flip interval avg {:.2f} min {:.2f} max {:.2f} ms | present interval avg {:.2f} "
             "max {:.2f} ms ({} presents) | flipMode vsync {} immediate {} other {} | bad dt {} "
             "| 1/30-site now {:.5f}",
             now - g_last_log, g_flip_iv.n, g_flip_iv.n / (now - g_last_log),
             g_measured.Avg() * 1e3, g_measured.n ? g_measured.mn * 1e3 : 0.0,
             g_measured.mx * 1e3, g_used.Avg() * 1e3, g_used.n ? g_used.mn * 1e3 : 0.0,
             g_used.mx * 1e3, g_period * 1e3, g_max_dt * 1e3, g_flip_iv.Avg() * 1e3,
             g_flip_iv.n ? g_flip_iv.mn * 1e3 : 0.0, g_flip_iv.mx * 1e3,
             g_present_iv.Avg() * 1e3, g_present_iv.mx * 1e3, g_present_iv.n, g_mode_vsync,
             g_mode_imm, g_mode_other, g_bad_dt, c100);
    g_measured.Reset(), g_used.Reset(), g_flip_iv.Reset(), g_present_iv.Reset();
    g_mode_vsync = g_mode_imm = g_mode_other = g_bad_dt = 0;
    g_last_log = now;
}

void OnPresent() {
    if (!g_installed) {
        return;
    }
    const double now = Now();
    std::scoped_lock lk{g_mutex};
    if (g_last_present > 0) g_present_iv.Add(now - g_last_present);
    g_last_present = now;
}

} // namespace BB::VarDt
