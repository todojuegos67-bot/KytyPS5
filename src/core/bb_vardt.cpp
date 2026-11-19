// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Bloodborne 1.09 (CUSA03173), BB_VARIABLE_DT=1: PC-style variable frame time, written by the
// emulator itself at eboot load (no XML patch needed). Addresses in this file are eboot VADDR
// (Ghidra / XML address = vaddr + 0x400000) unless a name says "Xml".
//
// Evidence (eboot 1.09, vaddr):
//  - 0x2034770 frame limiter (this = *(0x55404F8)): measures the frame with gettimeofday and
//    sleeps (sceKernelUsleep) / spins until `period` (this+0x18, written from the immediates at
//    0x2034840/0x203485A/0x203487E/0x20348B3) has elapsed; stores the measured frame time in
//    seconds at this+0x264 (0x2034CF0).
//  - 0x2018E36: mov dword [rbp-0x38], 1/30: the FD4 task manager (call 0x20512A0, rsi = &{vptr,
//    dt}) runs every task with a FIXED 1/30 s; tasks read the dt at [rsi+8] (e.g. the timer
//    object at 0x17B5830: [this+0x28] -= [rsi+8]).
//  - 0x17F9B71: the Havok world step is given a fixed 1/30 s ([rbp-0x48], call 0xB5A2E0).
//  - 0x1572A7D: the FD4PadManager task gets its own fixed 1/30 s.
// What BB_VARIABLE_DT installs (byte-checked against the original 1.09 code; groups that an XML
// patch already wrote identically are left as they are; any other FPS patch => nothing written):
//  1. limiter uncap (Lance McDonald / Kyo structure, as "PC sin limite 1"): one frame per period,
//     no vsync-count waits; period = 1 / BB_GAME_FPS_CAP (0 or unset = unlimited, 1 ms);
//  2. dt hook at 0x2018E3D: dt = clamp(measured this+0x264, period this+0x18,
//     BB_VARIABLE_DT_MAX_MS)
//     also stored to LiveDt (vaddr 0x56D3FC4, bss page slack);
//  3. guest flips requested as immediate (0x21B3271: flip mode argument 0);
//  4. Havok semi-fixed step (cave at 0x17F9C76, as "PC sin limite 3"): acc += LiveDt (cap 1/15 s),
//     steps h = min(acc, 1/60) while acc >= 1/360 s;
//  5. pad manager task dt = LiveDt (cave at 0x2018DFC, as "PC sin limite 6");
//  6. every rip-relative read of a "1/30 s per frame" constant that is used as a per-frame time
//     step (accumulators, update(dt) calls, one-frame minimums) is redirected to LiveDt (exact,
//     same frame); this includes "PC sin limite 4/5" (0x12A441A, 0x180F64B, 0x143E2B6);
//  7. the 1/30 constants that convert FRAME COUNTS (30 fps param data) to SECONDS are left at 1/30
//     (the patches scaled them like time steps, which shortens timers/durations at high fps);
//  8. camera DirectPitchRate (per-frame lerp factor [cam+0x23C] = 0.06 from the ctor at
//     0x143A64D, used at 0x143F97E / 0x143FA9F) is replaced by 1 - 0.94^(dt*30) each frame;
//  9. the per-frame damping 0.95 (vmulps at 0x1A4ABCA) and the cloth stiffness 1.0 defaults keep
//     the reviewed laws (0.95^(dt*30), min(dt*30, 1)), re-evaluated each guest flip;
// 10. BB_RUN_FIX (default on, BB_RUN_FIX=0 off): the ChrCtrl sprint wall check at 0x1514C24
//     measured the frame's move * 30 (m/s only at 30 fps) and halved sprint speed at high fps;
//     it now uses move / frame dt, and its 0.8 / 1.2 per-frame factors follow dt (see RunCmpVa).
#include "core/bb_vardt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include "common/logging/log.h"
#include "imgui/bb_menu.h"
#include "video_core/bb/bb_perf2.h"

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
struct CodeLine {
    u32 va;
    u32 len;
    const char* orig; // original 1.09 bytes
    const char* neu;  // bytes written (identical to the "PC sin limite" XML lines)
};
// Generated from the 1.09 eboot and bb-physics-unlimited.xml (entries 1, 3, 6).
constexpr CodeLine kLimiter[] = {
    {0x20347F3, 2, "\x8B\x11", "\x31\xD2"},
    {0x2034806, 4, "\x48\x63\x14\x91", "\x48\x63\x51\x00"},
    {0x203480F, 15, "\x41\xC7\x44\x24\x10\x02\x00\x00\x00\x41\xC6\x44\x24\x14\x01", "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x2034820, 15, "\x41\xC7\x44\x24\x10\x02\x00\x00\x00\x41\xC6\x44\x24\x14\x01", "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x2034831, 15, "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x01", "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x203484B, 15, "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00", "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x2034863, 10, "\x48\xB9\x01\x00\x00\x00\x1E\x00\x00\x00", "\x48\xB9\x00\x00\x00\x00\x00\x00\x00\x00"},
    {0x203486F, 15, "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00", "\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x2034887, 10, "\x48\xB9\x00\x00\x00\x00\x1E\x00\x00\x00", "\x48\xB9\x00\x00\x00\x00\x00\x00\x00\x00"},
    {0x20348A2, 17, "\x74\x3D\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00", "\xEB\x3D\x41\xC7\x44\x24\x10\x01\x00\x00\x00\x41\xC6\x44\x24\x14\x00"},
    {0x20348F0, 2, "\x78\x08", "\xEB\x08"},
    {0x2034907, 2, "\x78\x08", "\xEB\x08"},
    {0x203491A, 2, "\x74\x21", "\xEB\x21"},
    {0x203494E, 2, "\x74\x0C", "\xEB\x0C"},
    {0x2034D98, 2, "\x74\x05", "\x90\x90"},
    {0x2034FA8, 20, "\x67\xF8\xFF\xFF\x78\xF8\xFF\xFF\x89\xF8\xFF\xFF\xA3\xF8\xFF\xFF\xC7\xF8\xFF\xFF", "\x89\xF8\xFF\xFF\x89\xF8\xFF\xFF\x89\xF8\xFF\xFF\x89\xF8\xFF\xFF\x89\xF8\xFF\xFF"},
};
constexpr CodeLine kDtHook[] = {
    {0x2018E3D, 46, "\x48\x8D\x05\xCC\x76\x52\x03\x48\x83\x38\x00\x75\x21\x48\x8D\x3D\x18\x25\x92\x02\x48\x8D\x15\x65\x25\x92\x02\x48\x8D\x0D\x29\x24\x92\x02\xBE\xB1\x00\x00\x00\x31\xC0\xE8\x45\xC7\x09\x00", "\xC5\xFA\x10\x83\x64\x02\x00\x00\xC5\xFA\x5F\x43\x18\xC5\xFA\x5D\x45\xC8\xC5\xFA\x11\x45\xC8\xC5\xFA\x11\x05\x68\xB1\x6B\x03\x90\x90\x90\x90\x90\x90\x90\x90\x48\x8D\x05\xA5\x76\x52\x03"},
};
constexpr CodeLine kFlip[] = {
    {0x21B3271, 3, "\x44\x89\xE6", "\x48\x31\xF6"},
};
constexpr CodeLine kHavok[] = {
    {0x17F9B71, 7, "\xC7\x45\xB8\x89\x88\x08\x3D", "\xE9\x01\x01\x00\x00\x90\x90"},
    {0x17F9BDE, 6, "\x0F\x85\xC5\x00\x00\x00", "\x0F\x85\x92\x00\x00\x00"},
    {0x17F9C52, 5, "\xE8\x49\x14\xC4\xFE", "\xE9\x47\x00\x00\x00"},
    {0x17F9C76, 105, "\x48\x8D\x3D\x55\x75\x13\x03\x48\x8D\x35\x58\x75\x13\x03\x48\x8D\x0D\xCC\x75\x13\x03\xBA\x26\x00\x00\x00\x31\xC0\xE8\xD1\x4C\x3C\x01\x48\x8D\x3D\xBF\x75\x13\x03\xE8\x05\x4F\x3C\x01\xCC\xE9\x59\xFF\xFF\xFF\x48\x8D\x3D\x22\x75\x13\x03\x48\x8D\x35\xC5\x75\x13\x03\x48\x8D\x0D\x25\x76\x13\x03\xBA\x3B\x00\x00\x00\x31\xC0\xE8\x9E\x4C\x3C\x01\x48\x8D\x3D\x29\x76\x13\x03\xE8\xD2\x4E\x3C\x01\xCC\xE9\x08\xFF\xFF\xFF\x0F\x1F\x40", "\xCC\xC5\xFA\x10\x05\x45\xA3\xED\x03\x0F\x1F\x84\x00\x00\x00\x00\x00\x66\x90\x48\x8D\x05\x30\xA3\xED\x03\xC5\xFA\x58\x00\xC5\xFA\x5D\x05\x98\xB8\xFF\x02\xEB\x14\xE8\xFD\x13\xC4\xFE\x48\x8D\x05\x16\xA3\xED\x03\xC5\xFA\x10\x00\x49\x8B\x7D\x70\xC5\xF8\x2F\x05\x16\xF7\x08\x03\x72\x1A\xC5\xFA\x5D\x0D\x00\x28\x00\x03\xC5\xFA\x5C\xC1\xC5\xFA\x11\x00\xC5\xFA\x11\x4D\xB8\xE9\xA2\xFE\xFF\xFF\xC5\xFA\x11\x00\xE9\x78\xFF\xFF\xFF"},
};
constexpr CodeLine kPad[] = {
    {0x1572A7D, 7, "\xC7\x45\xD0\x89\x88\x08\x3D", "\xE9\x7A\x63\xAA\x00\x90\x90"},
    {0x2018DFA, 16, "\x48\x8D\x3D\x68\x25\x92\x02\x48\x8D\x15\xB5\x25\x92\x02\x48\x8D", "\xEB\x1F\x8B\x05\xC2\xB1\x6B\x03\x89\x45\xD0\xE9\x7A\x9C\x55\xFF"},
};
constexpr CodeLine kXmlStreamCave[] = {
    {0x1DBC181, 33, "\x48\x8D\x3D\xE1\xF1\xB7\x02\x48\x8D\x15\x2E\xF2\xB7\x02\x48\x8D\x0D\xF1\x94\xB7\x02\xBE\xB1\x00\x00\x00\x31\xC0\xE8\x0E\x94\x2F\x00", "\x48\x8D\x3D\x70\x43\x78\x03\x48\x8B\x3F\xC5\xFA\x59\x87\x64\x02\x00\x00\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\xE9\x6E\x01\x00\x00"},
    {0x1DBC308, 8, "\xC5\xFA\x59\x05\x88\xCB\xB6\x02", "\xE9\x74\xFE\xFF\xFF\x90\x90\x90"},
};
constexpr CodeLine kXmlAnimDtCave[] = {
    {0x1F77CEA, 38, "\x75\x24\x48\x8D\x3D\x76\x36\x9C\x02\x48\x8D\x15\xC3\x36\x9C\x02\x48\x8D\x0D\xE0\x34\x9C\x02\xBE\xB1\x00\x00\x00\x31\xC0\xE8\xA3\xD8\x13\x00\x49\x8B\x07", "\xEB\x24\x48\x8D\x1D\x05\x88\x5C\x03\x48\x8B\x1B\xC5\xFA\x10\x83\x64\x02\x00\x00\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\xEB\x42"},
    {0x1F77D48, 2, "\x75\x08", "\xEB\xA2"},
};
constexpr CodeLine kXmlTaskDtCave[] = {
    {0x0FD2E16, 27, "\xF6\x45\xC8\x20\x75\x15\x48\x8D\x15\x9D\x4F\x76\x03\x31\xFF\xBE\x3E\x00\x00\x00\x31\xC0\xE8\x7F\x27\x0E\x01", "\xEB\x19\x41\x56\x41\xC7\x46\x08\x00\x00\x80\x3F\xC4\xC1\x7A\x5E\x46\x08\x41\x5E\xE9\x33\x07\x00\x00\x90\x90"},
    {0x0FD3557, 11, "\x48\x85\xFF\x74\x0B\xC4\xC1\x7A\x10\x46\x08", "\xC4\xC1\x7A\x10\x46\x08\xE9\xB6\xF8\xFF\xFF"},
};
constexpr CodeLine kXmlBadCall[] = {
    {0x2315D71, 8, "\xC5\xFA\x10\x25\xFB\x40\x61\x02", "\xE8\x81\x09\x00\x00\x90\x90\x90"},
};
constexpr CodeLine kXmlBadRet[] = {
    {0x2083EC1, 5, "\xE8\xA2\xCE\xB3\x00", "\xC3\x90\x90\x90\x90"},
};
constexpr CodeLine kXmlCamera[] = {
    {0x1440799, 28, "\x49\x8B\xBD\x38\x03\x00\x00\x49\x8D\x95\x38\x02\x00\x00\x48\x8D\x35\x10\x30\x51\x03\x31\xC9\xC5\xFA\x10\x05\xE4", "\x41\xC7\x85\x38\x02\x00\x00\x92\x0A\x06\x3F\x90\x41\xC7\x85\x3C\x02\x00\x00\x8F\xC2\xF5\x3C\xE9\x54\x00\x00\x00"},
};

constexpr u64 EbootBase = 0x400000; // Ghidra / XML address = vaddr + EbootBase

constexpr u32 LiveDtVa = 0x56D3FC4;   // float, written by the dt hook every frame (bss slack)
constexpr u32 HavokAccVa = 0x56D3FC0; // float, Havok step accumulator (bss slack)
constexpr u32 PitchRateVa = 0x56D3FC8; // float, camera DirectPitchRate for this frame (ours)
// BB_RAGDOLL_FIX (default off): the Havok cave keeps stepping while acc >= 1/360 s (vcomiss at
// 0x17F9CB2), so the last substep of a frame has a variable size h in [1/360, 1/60) that changes
// with the fps (e.g. 40 fps: 1/60 + 1/120 every frame; uncapped: random). The original game
// always stepped a fixed 1/30 s. Variable h makes the Havok solver re-solve resting contacts with
// a different dt each step, which shows up as resting ragdolls (corpses) trembling. The fix only
// re-points that vcomiss at a 0.0165 s threshold (bss slack 0x56D3FCC): every substep is then
// h = min(acc, 1/60) with acc >= 0.0165, i.e. a near-fixed 1/60 step; the remainder stays in acc.
constexpr u32 RagdollThrVa = 0x56D3FCC;  // float threshold read by the patched vcomiss (ours)
constexpr u32 HavokCmpVa = 0x17F9CB2;    // vcomiss xmm0, [rip+disp] in the Havok cave
const u8 HavokCmpOrig[8] = {0xC5, 0xF8, 0x2F, 0x05, 0x16, 0xF7, 0x08, 0x03}; // -> 1/360 @0x48893D0
bool g_ragdoll_fix = false;
u32 g_ragdoll_frames = 0, g_ragdoll_zero_steps = 0;
double g_ragdoll_last_log = 0;
// BB_RUN_FIX (default on with BB_VARIABLE_DT=1; BB_RUN_FIX=0 turns it off): the ChrCtrl physics
// update 0x1514500 (called from 0x1E388F2 for every character, player and NPCs) has a "running
// into a wall" check after the Havok character proxy moved (0x1514B92..0x1514CCD):
//   if (root-motion speed [ctrl+0x1E4] > 5.0 m/s && grounded [proxyOwner+0x212]) {
//       d = |pos [proxyOwner+0x1E0] - previous pos [+0x1F0]|   (the move of THIS frame)
//       if (d * 30.0 < 1.0) { scale [ctrl+0x1E0] = scale <= 0.5 ? 0.5 : scale * 0.8; }   // blocked
//       else ...
//   } else scale = scale >= 1 ? 1 : scale * 1.2;
// and 0x1514500 multiplies the root-motion displacement by the scale when it is < 1. "d * 30" is
// the speed in m/s only at 30 fps (the debug panel 0x1517710 prints the same |pos - prev| * 30
// as "Position : x y z (speed)"). At f fps a character moving at v m/s gives d * 30 = v * 30 / f,
// so while sprinting (v ~ 5-7 m/s) the check fires on its own above ~150-200 fps, and once any
// obstacle, slope or a short frame trips it the scale sinks to 0.5 and stays there whenever
// 0.5 * v * 30 / f < 1, i.e. above ~15 * v (~80-100 fps): the character runs at half speed.
// Fix: 0x1514C24 "vmulss xmm0, xmm0, [30.0f @0x492632C]" becomes "d / frame dt" (LiveDt, the dt
// of this frame: exact at any fps, d * 30 at 30 fps), and the per-frame factors 0.8 / 1.2
// (0x4926334 / 0x4926338, one reader each) follow 0.8^(dt*30) / 1.2^(dt*30). The check is
// reached through a small host cave that also counts the checks for the 5 s log line; if no
// executable memory can be had, an in-place "vdivss xmm0, xmm0, [rip -> LiveDt]" is written.
constexpr u32 RunCmpVa = 0x1514C24;
const u8 RunCmpOrig[8] = {0xC5, 0xFA, 0x59, 0x05, 0x00, 0x17, 0x41, 0x03}; // -> 30.0 @0x492632C
constexpr u32 RunThirtyVa = 0x492632C, RunDecayVa = 0x4926334, RunGrowVa = 0x4926338;
constexpr u32 RunCavePtrVa = 0x56D3FD0; // u64: host cave address for "call [rip+disp]" (bss slack)
constexpr float RunDecay30 = 0.8f, RunGrow30 = 1.2f;
struct alignas(16) RunStats {
    float actual_sum;   // +0: d / dt (m/s)
    u32 n;              // +4: checks (root motion > 5 m/s, grounded)
    float intended_sum; // +8: root-motion speed [ctrl+0x1E4]
    float scale_sum;    // +12: move scale [ctrl+0x1E0] before this frame's update
    float one;          // +16: 1.0 (the threshold, for the cave's compare)
    u32 blocked;        // +20: checks with d / dt < 1 m/s
};
RunStats* g_run = nullptr;   // host memory, written by the guest through the cave
bool g_run_fix = false, g_run_factors = false;
double g_run_last_log = 0;
u32 g_run_frames = 0;
const char* g_run_how = "";

constexpr u32 FramePtrVa = 0x55404F8;  // the frame limiter object pointer
constexpr u32 OffPeriod = 0x18, OffMeasured = 0x264;

constexpr u32 DtMovVa = 0x2018E36; // C7 45 C8 imm32: mov dword [rbp-0x38], 1/30 (max dt after hook)
constexpr u8 DtMovPrefix[3] = {0xC7, 0x45, 0xC8};
constexpr u32 PeriodImmVa[] = {0x2034840, 0x203485A, 0x203487E, 0x20348B3};
constexpr u8 PeriodPrefix[5] = {0x41, 0xC7, 0x44, 0x24, 0x18}; // mov dword [r12+0x18], imm32
// rodata constants the Havok cave reads: 1/360, 1/60, 1/15.
constexpr struct {
    u32 va;
    u32 bits;
} HavokConsts[] = {{0x48893D0, 0x3B360B61}, {0x47FC4C4, 0x3C888889}, {0x47F5534, 0x3D888889}};

enum class Law : u8 {
    Step,  // a per-frame time step: value = dt (orig 1/30 * dt * 30); readers -> LiveDt
    Keep,  // frames (30 fps data) -> seconds, or a time in seconds: must stay 1/30
    Cloth, // Havok cloth "Stiffness" default 1.0: min(dt * 30, 1) (reviewed bb-vardt law)
    Decay, // per-frame damping 0.95: 0.95^(dt * 30)
};
struct Site {
    u32 va;
    float orig;  // unpatched value
    u32 xml;     // the value "PC sin limite 1" writes (0 = not written by it)
    Law law;
    const char* what;
};
constexpr u32 X100 = 0x3C23D70A, X60 = 0x3C888889, XHalf = 0x3F000000, XDecay = 0x3F798497;
constexpr u32 X120 = 0x3C088889, X30 = 0x3D088889;
constexpr float T30 = 1.0f / 30.0f;
constexpr Site Sites[] = {
    // Step: the reader uses the value as this frame's time step (all readers redirected below).
    {0x47EFDCC, T30, X100, Law::Step, "0xAB9F0D default one-frame time when 0"},
    {0x48AD65C, T30, X100, Law::Step, "0xEE596A one-frame minimum (timed list)"},
    {0x48AD660, T30, X100, Law::Step, "0xEE5A8A one-frame minimum (timed list)"},
    {0x48AD664, T30, X100, Law::Step, "0xEE5BAA one-frame minimum (timed list)"},
    {0x48ADC08, T30, X100, Law::Step, "0xEF06F1 one-frame minimum"},
    {0x48ADD28, T30, X100, Law::Step, "0xEF6992 one-frame minimum"},
    {0x48CC3F4, T30, X100, Law::Step, "0x1044943 update(dt) virtual call"},
    {0x492577C, T30, X100, Law::Step, "0x12A4416 timer += dt (PC sin limite 4)"},
    {0x4925E94, T30, X100, Law::Step, "0x143E2B2 x += v * dt (PC sin limite 5)"},
    {0x4926C9C, T30, X100, Law::Step, "0x15DC1F7 update(dt) call"},
    {0x49270E4, T30, X100, Law::Step, "0x173456E update(dt) virtual call"},
    {0x4927554, T30, X100, Law::Step, "0x17F9D26 update(dt) tail call"},
    {0x49275F4, T30, X100, Law::Step, "0x180F647 t = min(t + dt, max) (PC sin limite 4)"},
    {0x4927C4C, T30, X100, Law::Step, "0x18BC4F7 update(dt) calls"},
    {0x4929170, T30, X60, Law::Step, "0x1F77D4A update(dt) indirect call"},
    {0x4929E74, T30, X60, Law::Step, "0x2313DB7 / 0x2315D71 frame dt"},
    {0x4929B80, T30, X60, Law::Step, "0x227784B update(dt) (0xEBBDA0: [+0x64] += dt)"},
    // Keep: int frames * 1/30 = seconds (timers that count down by the task dt), or [sec] values.
    {0x4888688, T30, X100, Law::Keep, "0xDCA3ED frames -> seconds (rate = n / (frames / 30))"},
    {0x48AF1BC, T30, X100, Law::Keep, "0xF1E607 frames -> seconds -> 0x2301FA0 timed list"},
    {0x48AF1C0, T30, X100, Law::Keep, "0xF1E64E frames -> seconds -> 0x2301FA0"},
    {0x48AF290, T30, X100, Law::Keep, "0xF1F7B7 frames -> seconds -> 0x2301FA0"},
    {0x48AF294, T30, X100, Law::Keep, "0xF1F807 frames -> seconds -> 0x2301FA0"},
    {0x48AF340, T30, X100, Law::Keep, "0xF20214 frames -> seconds -> 0x2301FE0"},
    {0x48AF344, T30, X100, Law::Keep, "0xF2024B frames -> seconds -> 0x2301FE0"},
    {0x48B03B8, T30, X100, Law::Keep, "0xF37A64 frames -> seconds -> 0x2302020"},
    {0x48B03BC, T30, X100, Law::Keep, "0xF37A9B frames -> seconds -> 0x2302020"},
    {0x492585C, T30, X100, Law::Keep, "0x12AD841 frames -> seconds"},
    {0x4925864, T30, X100, Law::Keep, "0x12ADF20 frames -> seconds"},
    {0x49265FC, T30, X100, Law::Keep, "0x1535C56 'move direction correction time [sec]' default"},
    {0x4926618, T30, X100, Law::Keep, "0x15369B4 'move direction correction time [sec]' default"},
    {0x4927250, T30, X100, Law::Keep, "0x17B55ED frames -> seconds timer (0x17B5830 -= task dt)"},
    {0x4927258, T30, X100, Law::Keep, "0x17B5754 frames -> seconds"},
    {0x4927720, T30, X100, Law::Keep, "0x181FF25 per-id times in seconds (0xF90C30)"},
    {0x4928E98, T30, X100, Law::Keep, "0x1DBC308 frames * 1/30 / task dt"},
    {0x4929124, T30, X60, Law::Keep, "0x1F23009 param u16 frames -> seconds"},
    {0x4929128, T30, X60, Law::Keep, "0x1F23039 param u16 frames -> seconds"},
    {0x4925844, T30, X120, Law::Keep, "0x12AB70C param u16 frames -> seconds (XML: 1/120)"},
    {0x4926364, T30, X30, Law::Keep, "0x1515F69 frames -> seconds / 1/30 s blend"},
    {0x4927284, T30, X30, Law::Keep, "0x17C8758 1/30 s"},
    {0x492728C, T30, X30, Law::Keep, "0x17C8BB6 1/30 s"},
    // Cloth stiffness defaults (0x7E45BE, 0x7F706D, 0x805898, 0x809D62 -> 0x80D800 "Stiffness").
    {0x47693F0, 1.0f, XHalf, Law::Cloth, "0x47C6DC"},
    {0x47B63E4, 1.0f, XHalf, Law::Cloth, "0x7E45BE"},
    {0x47B8474, 1.0f, XHalf, Law::Cloth, "0x7F706D"},
    {0x47B91C0, 1.0f, XHalf, Law::Cloth, "0x805898"},
    {0x47B98B8, 1.0f, XHalf, Law::Cloth, "0x809D62"},
    // Per-frame damping vector (vmulps at 0x1A4ABCA).
    {0x490B5B0, 0.95f, XDecay, Law::Decay, "0x1A4ABCA"},
    {0x490B5B4, 0.95f, XDecay, Law::Decay, "0x1A4ABCA"},
    {0x490B5B8, 0.95f, XDecay, Law::Decay, "0x1A4ABCA"},
    {0x490B5BC, 0.95f, XDecay, Law::Decay, "0x1A4ABCA"},
};
constexpr size_t NumSites = sizeof(Sites) / sizeof(Sites[0]);

// rip-relative readers of the Step constants: VEX prefix + opcode + modrm (4 bytes), then disp32
// as the last field of the instruction (verified in the 1.09 disassembly).
struct Reader {
    u32 site;
    u32 disp; // vaddr of the disp32
    const char* pre;
};
constexpr Reader Readers[] = {
    {0x47EFDCC, 0x0AB9F11, "\xC5\xFA\x10\x0D"}, {0x48AD65C, 0x0EE596E, "\xC5\xFA\x5F\x05"},
    {0x48AD660, 0x0EE5A8E, "\xC5\xFA\x5F\x05"}, {0x48AD664, 0x0EE5BAE, "\xC5\xFA\x5F\x05"},
    {0x48ADC08, 0x0EF06F5, "\xC5\xF2\x5F\x0D"}, {0x48ADD28, 0x0EF6996, "\xC5\xF2\x5F\x0D"},
    {0x48CC3F4, 0x1044947, "\xC5\xFA\x10\x05"}, {0x492577C, 0x12A441A, "\xC5\xFA\x58\x05"},
    {0x4925E94, 0x143E2B6, "\xC5\xFA\x10\x15"}, {0x4926C9C, 0x15DC1FB, "\xC5\xFA\x10\x05"},
    {0x49270E4, 0x1734572, "\xC5\xFA\x10\x05"}, {0x4927554, 0x17F9D2A, "\xC5\xFA\x10\x05"},
    {0x49275F4, 0x180F64B, "\xC5\xFA\x58\x05"}, {0x4927C4C, 0x18BC4FB, "\xC5\xFA\x10\x05"},
    {0x4927C4C, 0x18BC511, "\xC5\xFA\x10\x05"}, {0x4929170, 0x1F77D4E, "\xC5\xFA\x10\x05"},
    {0x4929E74, 0x2313DBB, "\xC5\xFA\x10\x05"}, {0x4929E74, 0x2315D75, "\xC5\xFA\x10\x25"},
    {0x4929B80, 0x227784F, "\xC5\xFA\x10\x05"}, {0x4929B80, 0x2277896, "\xC5\xFA\x10\x05"},
};
constexpr size_t NumReaders = sizeof(Readers) / sizeof(Readers[0]);

// Camera DirectPitchRate: vmulss xmmN, xmmN, [r13+0x23C] (9 bytes) -> vmulss xmmN, xmmN,
// [rip -> PitchRateVa] (8 bytes) + nop.
struct CamSite {
    u32 va;
    const char* orig;
    const char* pre;
};
constexpr CamSite CamSites[] = {
    {0x143F97E, "\xC4\xC1\x7A\x59\x85\x3C\x02\x00\x00", "\xC5\xFA\x59\x05"},
    {0x143FA9F, "\xC4\xC1\x72\x59\x8D\x3C\x02\x00\x00", "\xC5\xF2\x59\x0D"},
};
constexpr u32 CamCtorVa[] = {0x143A64D, 0x143AA4B}; // mov dword [rbx+0x23C], 0.06
constexpr char CamCtorBytes[] = "\xC7\x83\x3C\x02\x00\x00\x8F\xC2\x75\x3D";
constexpr float PitchRate30 = 0.06f;

bool g_enabled = false, g_immediate = false, g_installed = false;
uintptr_t g_base = 0;
u64 g_size = 0;
bool g_site_live[NumSites] = {}; // re-evaluated every flip
float g_max_dt = 0.05f, g_period = 0.001f;
bool g_cam = false;
const char* g_mode = "off";

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
Acc g_measured, g_used, g_flip_iv, g_present_iv, g_havok_acc;
double g_last_flip = 0, g_last_present = 0, g_last_log = 0, g_frame_period = 0;
u32 g_mode_vsync = 0, g_mode_imm = 0, g_mode_other = 0, g_bad_dt = 0, g_clamp_hi = 0,
    g_clamp_lo = 0;

// bb-spikes: a guest frame at or over kSpikeSec gets one "[spike]" line with what happened since
// the previous flip (the counters of bb_perf2.h, as deltas), capped at kSpikeLogMax lines.
constexpr double kSpikeSec = 0.030;
constexpr u32 kSpikeLogMax = 300;
struct SpikeCounts {
    u64 faults = 0, fault_around = 0, buf_up = 0, tex_up = 0, tex_images = 0;
    u64 waits = 0, wait_us = 0, compile_us = 0, pipelines = 0;
    u64 file_reads = 0, file_read_us = 0, file_read_bytes = 0, commit_calls = 0, commit_us = 0;
    u64 gnm_wait_us = 0, fault_us = 0, cp_busy_us = 0, cp_submit_us = 0, present_us = 0;
};
SpikeCounts g_spike_prev;
u32 g_spike_logged = 0;
bool g_spike_menu = false; // F1 menu open (game paused): those frames are not spikes

SpikeCounts ReadSpikeCounts() {
    const auto& c = BBPerf2::Get();
    const auto ld = [](const std::atomic<u64>& v) { return v.load(std::memory_order_relaxed); };
    return {ld(c.cpu_writes),   ld(c.fault_around_pages), ld(c.buf_up_bytes),
            ld(c.tex_up_bytes), ld(c.tex_uploads),        ld(c.waits),
            ld(c.wait_us),      ld(c.compile_us),         ld(c.gfx_pipelines) + ld(c.cs_pipelines),
            ld(c.file_reads),   ld(c.file_read_us),       ld(c.file_read_bytes),
            ld(c.commit_calls), ld(c.commit_us),          ld(c.gnm_wait_us),
            ld(c.fault_us),     ld(c.cp_busy_us),         ld(c.cp_submit_us),
            ld(c.present_us)};
}

// The 10 s "[perf2]" reader zeroes the counters (exchange), so a counter that went down restarts.
u64 Since(u64 now, u64 before) {
    return now >= before ? now - before : now;
}

SpikeCounts Since(const SpikeCounts& now, const SpikeCounts& before) {
    return {Since(now.faults, before.faults),
            Since(now.fault_around, before.fault_around),
            Since(now.buf_up, before.buf_up),
            Since(now.tex_up, before.tex_up),
            Since(now.tex_images, before.tex_images),
            Since(now.waits, before.waits),
            Since(now.wait_us, before.wait_us),
            Since(now.compile_us, before.compile_us),
            Since(now.pipelines, before.pipelines),
            Since(now.file_reads, before.file_reads),
            Since(now.file_read_us, before.file_read_us),
            Since(now.file_read_bytes, before.file_read_bytes),
            Since(now.commit_calls, before.commit_calls),
            Since(now.commit_us, before.commit_us),
            Since(now.gnm_wait_us, before.gnm_wait_us),
            Since(now.fault_us, before.fault_us),
            Since(now.cp_busy_us, before.cp_busy_us),
            Since(now.cp_submit_us, before.cp_submit_us),
            Since(now.present_us, before.present_us)};
}

u8* AtV(u64 va) {
    return reinterpret_cast<u8*>(g_base + va);
}
bool InsideV(u64 va, u64 n) {
    return va + n <= g_size;
}
u32 Read32(u64 va) {
    u32 v = 0;
    std::memcpy(&v, AtV(va), 4);
    return v;
}
void Write32(u64 va, u32 v) {
    std::memcpy(AtV(va), &v, 4);
}
u32 Bits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}
float Float(u32 b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// The constants live in the eboot's R+X segment. The loader maps executable guest segments
// read/write on the host and the XML patcher writes them the same way; this makes sure of it
// once for the constant pages that are rewritten every frame.
void MakeWritable(u64 lo, u64 hi) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(AtV(lo)) & ~uintptr_t(0xFFF);
    const uintptr_t b = (reinterpret_cast<uintptr_t>(AtV(hi)) + 0x1000) & ~uintptr_t(0xFFF);
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

enum class St : u8 { Orig, Xml, Foreign };
const char* Name(St s) {
    return s == St::Orig ? "original" : s == St::Xml ? "already patched (XML)" : "FOREIGN";
}
template <size_t N>
bool AllInside(const CodeLine (&g)[N]) {
    for (const CodeLine& l : g) {
        if (!InsideV(l.va, l.len)) return false;
    }
    return true;
}
// Orig: every line has the 1.09 bytes; Xml: every line has our bytes; else Foreign.
template <size_t N>
St Check(const CodeLine (&g)[N]) {
    bool orig = true, neu = true;
    for (const CodeLine& l : g) {
        orig = orig && std::memcmp(AtV(l.va), l.orig, l.len) == 0;
        neu = neu && std::memcmp(AtV(l.va), l.neu, l.len) == 0;
    }
    return orig ? St::Orig : neu ? St::Xml : St::Foreign;
}
template <size_t N>
void Apply(const CodeLine (&g)[N], bool to_new) {
    for (const CodeLine& l : g) {
        std::memcpy(AtV(l.va), to_new ? l.neu : l.orig, l.len);
    }
}

float Value(const Site& s, float dt) {
    const float f = dt * 30.0f;
    switch (s.law) {
    case Law::Step:
        return s.orig * f;
    case Law::Keep:
        return s.orig;
    case Law::Cloth:
        return std::min(s.orig * f, 1.0f);
    case Law::Decay:
        return std::pow(s.orig, f);
    }
    return s.orig;
}
float PitchRate(float dt) {
    return 1.0f - std::pow(1.0f - PitchRate30, dt * 30.0f);
}

bool ReadFrame(float* measured, float* period) {
    u64 p = 0;
    std::memcpy(&p, AtV(FramePtrVa), 8);
    if (p < 0x10000 || (p & 3) != 0) return false;
    std::memcpy(measured, reinterpret_cast<const void*>(p + OffMeasured), 4);
    std::memcpy(period, reinterpret_cast<const void*>(p + OffPeriod), 4);
    return true;
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

void InstallRunFix() {
    const char* e = std::getenv("BB_RUN_FIX");
    if (e != nullptr && e[0] == '0') {
        LOG_INFO(Lib_VideoOut, "[runfix] BB_RUN_FIX=0: off (sprint wall check stays 'dist * 30')");
        return;
    }
    if (!InsideV(RunCmpVa, 8) || !InsideV(RunCavePtrVa, 8) || !InsideV(RunGrowVa, 4) ||
        std::memcmp(AtV(RunCmpVa), RunCmpOrig, 8) != 0 || Read32(RunThirtyVa) != Bits(30.0f)) {
        LOG_WARNING(Lib_VideoOut, "[runfix] ChrCtrl wall check at 0x1514C24 is not the expected "
                                  "1.09 code: left alone");
        return;
    }
    // Host cave, entered with "call [rip -> RunCavePtrVa]"; rax, xmm1 and the flags are dead at
    // 0x1514C24 (rax is reloaded at 0x1514C95 on every path, xmm1 at 0x1514C2C, flags at C34).
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave != nullptr) {
        g_run = reinterpret_cast<RunStats*>(cave + 2048);
        *g_run = RunStats{};
        g_run->one = 1.0f;
        const u64 live = reinterpret_cast<u64>(AtV(LiveDtVa));
        const u64 st = reinterpret_cast<u64>(g_run);
        u8 c[128];
        size_t n = 0;
        const auto put = [&](std::initializer_list<u8> b) {
            for (const u8 x : b) c[n++] = x;
        };
        const auto put64 = [&](u64 v) {
            std::memcpy(c + n, &v, 8);
            n += 8;
        };
        put({0x48, 0xB8}), put64(live);                      // mov rax, &LiveDt
        put({0xC5, 0xFA, 0x5E, 0x00});                        // vdivss xmm0, xmm0, [rax]
        put({0x48, 0xB8}), put64(st);                         // mov rax, &stats
        put({0xC5, 0xFA, 0x58, 0x08});                        // vaddss xmm1, xmm0, [rax]
        put({0xC5, 0xFA, 0x11, 0x08});                        // vmovss [rax], xmm1
        put({0xF0, 0xFF, 0x40, 0x04});                        // lock inc dword [rax+4]
        put({0xC4, 0xC1, 0x7A, 0x10, 0x8D, 0xE4, 0x01, 0x00, 0x00}); // vmovss xmm1,[r13+0x1E4]
        put({0xC5, 0xF2, 0x58, 0x48, 0x08});                  // vaddss xmm1, xmm1, [rax+8]
        put({0xC5, 0xFA, 0x11, 0x48, 0x08});                  // vmovss [rax+8], xmm1
        put({0xC4, 0xC1, 0x7A, 0x10, 0x8D, 0xE0, 0x01, 0x00, 0x00}); // vmovss xmm1,[r13+0x1E0]
        put({0xC5, 0xF2, 0x58, 0x48, 0x0C});                  // vaddss xmm1, xmm1, [rax+12]
        put({0xC5, 0xFA, 0x11, 0x48, 0x0C});                  // vmovss [rax+12], xmm1
        put({0xC5, 0xF8, 0x2E, 0x40, 0x10});                  // vucomiss xmm0, [rax+16] (1.0)
        put({0x73, 0x04});                                    // jae +4
        put({0xF0, 0xFF, 0x40, 0x14});                        // lock inc dword [rax+20]
        put({0xC3});                                          // ret
        std::memcpy(cave, c, n);
        const u64 cave_addr = reinterpret_cast<u64>(cave);
        std::memcpy(AtV(RunCavePtrVa), &cave_addr, 8);
        u8 b[8] = {0xFF, 0x15, 0, 0, 0, 0, 0x66, 0x90}; // call [rip+disp32]; 2-byte nop
        const s32 d = static_cast<s32>(static_cast<s64>(RunCavePtrVa) - (RunCmpVa + 6));
        std::memcpy(b + 2, &d, 4);
        std::memcpy(AtV(RunCmpVa), b, 8);
        g_run_how = "host cave (counts the checks)";
    } else {
        u8 b[8] = {0xC5, 0xFA, 0x5E, 0x05, 0, 0, 0, 0}; // vdivss xmm0, xmm0, [rip -> LiveDt]
        const s32 d = static_cast<s32>(static_cast<s64>(LiveDtVa) - (RunCmpVa + 8));
        std::memcpy(b + 4, &d, 4);
        std::memcpy(AtV(RunCmpVa), b, 8);
        g_run_how = "in place (no executable memory for the cave: no speed log)";
    }
    g_run_fix = true;
    if (Read32(RunDecayVa) == Bits(RunDecay30) && Read32(RunGrowVa) == Bits(RunGrow30)) {
        MakeWritable(RunDecayVa, RunGrowVa);
        g_run_factors = true;
    }
    LOG_INFO(Lib_VideoOut,
             "[runfix] on: ChrCtrl sprint wall check 0x1514C24 'move of the frame * 30 < 1 m/s' "
             "-> 'move / frame dt < 1 m/s' ({}); move-scale factors 0.8 / 1.2 per frame "
             "(0x4926334 / 0x4926338) -> 0.8^(dt*30) / 1.2^(dt*30): {}",
             g_run_how, g_run_factors ? "yes" : "NOT FOUND, left per frame");
}

float EnvFloat(const char* name, float def, float lo, float hi, bool* set = nullptr) {
    const char* m = std::getenv(name);
    if (set) *set = m != nullptr && *m;
    if (m == nullptr || !*m) return def;
    const float v = static_cast<float>(std::atof(m));
    return std::isfinite(v) ? std::clamp(v, lo, hi) : def;
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
    bool inside = InsideV(LiveDtVa, 4) && InsideV(PitchRateVa, 4) && InsideV(FramePtrVa, 8) &&
                  InsideV(DtMovVa, 7) && AllInside(kLimiter) && AllInside(kDtHook) &&
                  AllInside(kFlip) && AllInside(kHavok) && AllInside(kPad) &&
                  AllInside(kXmlStreamCave) && AllInside(kXmlAnimDtCave) &&
                  AllInside(kXmlTaskDtCave) && AllInside(kXmlBadCall) && AllInside(kXmlBadRet) &&
                  AllInside(kXmlCamera);
    for (const Site& s : Sites) inside = inside && InsideV(s.va, 4);
    for (const Reader& r : Readers) inside = inside && InsideV(r.disp - 4, 8);
    for (const CamSite& c : CamSites) inside = inside && InsideV(c.va, 9);
    for (const u32 p : PeriodImmVa) inside = inside && InsideV(p, 9);
    if (!inside || std::memcmp(AtV(DtMovVa), DtMovPrefix, 3) != 0) {
        LOG_WARNING(Lib_VideoOut, "[vardt] BB_VARIABLE_DT=1 but this is not the Bloodborne 1.09 "
                                  "eboot: off");
        return;
    }

    // 1) Byte checks. Every group must be the original 1.09 code or exactly what "PC sin limite"
    //    writes; anything else is another FPS patch (60 FPS++, Uncap FPS++, 60FPS no deltatime,
    //    Lance McDonald's 60 FPS ...): then nothing at all is written.
    const St st_dt = Check(kDtHook), st_lim = Check(kLimiter), st_flip = Check(kFlip),
             st_havok = Check(kHavok), st_pad = Check(kPad);
    const u32 dt_imm = Read32(DtMovVa + 3);
    const bool imm_ok = dt_imm == X30 || dt_imm == X60; // 1/30 original, 1/60 PC sin limite 1
    bool havok_consts = true;
    for (const auto& c : HavokConsts) havok_consts = havok_consts && Read32(c.va) == c.bits;
    bool periods_ok = true;
    for (const u32 p : PeriodImmVa) {
        periods_ok = periods_ok && std::memcmp(AtV(p), PeriodPrefix, 5) == 0;
    }
    std::string foreign;
    const auto need = [&](St s, const char* what) {
        if (s == St::Foreign) foreign += std::string(foreign.empty() ? "" : ", ") + what;
    };
    need(st_dt, "dt hook 0x2018E3D");
    need(st_lim, "frame limiter 0x20347F3..0x2034FBB");
    need(st_flip, "flip mode 0x21B3271");
    need(st_havok, "Havok step 0x17F9B71/0x17F9C76");
    need(st_pad, "pad task dt 0x1572A7D/0x2018DFA");
    if (!imm_ok) foreign += std::string(foreign.empty() ? "" : ", ") + "max dt imm 0x2018E39";
    if (!havok_consts) foreign += std::string(foreign.empty() ? "" : ", ") + "Havok constants";
    if (!periods_ok) foreign += std::string(foreign.empty() ? "" : ", ") + "limiter period imm";
    if (!foreign.empty()) {
        LOG_WARNING(Lib_VideoOut,
                    "[vardt] BB_VARIABLE_DT=1 but another FPS patch changed: {}. Nothing is "
                    "written (game timing as that patch makes it). Disable the XML FPS patches "
                    "(60 FPS++, Uncap FPS++, 60FPS no deltatime, ...) to use BB_VARIABLE_DT.",
                    foreign);
        return;
    }
    const bool xml = st_dt == St::Xml || st_lim == St::Xml || st_havok == St::Xml ||
                     st_pad == St::Xml || st_flip == St::Xml;
    g_mode = xml ? "native + XML 'PC sin limite' already present" : "native (no XML)";
    LOG_INFO(Lib_VideoOut,
             "[vardt] byte check: dt hook {}, limiter {}, flip mode {}, Havok step {}, pad dt {}",
             Name(st_dt), Name(st_lim), Name(st_flip), Name(st_havok), Name(st_pad));

    // 2) Code groups (same bytes as the XML lines; written only where still original).
    if (st_dt == St::Orig) Apply(kDtHook, true);
    if (st_lim == St::Orig) Apply(kLimiter, true);
    if (st_flip == St::Orig) Apply(kFlip, true);
    // LiveDt is bss slack (0 at load); the redirected readers must not see dt = 0 before the
    // first frame passes the dt hook.
    Write32(LiveDtVa, Bits(T30));
    if (st_havok == St::Orig) {
        Write32(HavokAccVa, 0);
        Apply(kHavok, true);
    }
    if (st_pad == St::Orig) Apply(kPad, true);
    {
        const char* rf = std::getenv("BB_RAGDOLL_FIX");
        const bool want = rf != nullptr && rf[0] == '1';
        if (want && InsideV(RagdollThrVa, 4) && InsideV(HavokCmpVa, 8) &&
            std::memcmp(AtV(HavokCmpVa), HavokCmpOrig, 8) == 0) {
            Write32(RagdollThrVa, Bits(0.0165f));
            Write32(HavokCmpVa + 4, static_cast<u32>(RagdollThrVa - (HavokCmpVa + 8)));
            g_ragdoll_fix = true;
            LOG_INFO(Lib_VideoOut, "[ragdoll] BB_RAGDOLL_FIX on: Havok substeps fixed ~1/60 s "
                                   "(step while acc >= 16.5 ms, remainder carried)");
        } else if (want) {
            LOG_WARNING(Lib_VideoOut, "[ragdoll] BB_RAGDOLL_FIX=1 but the Havok cave bytes at "
                                      "0x17F9CB2 are not the expected ones: off");
        }
    }

    // 3) Max dt (upper clamp) and the frame cap (the limiter's own period, this+0x18).
    g_max_dt = EnvFloat("BB_VARIABLE_DT_MAX_MS", 50.0f, 17.0f, 100.0f) / 1000.0f;
    bool cap_set = false;
    const float cap = EnvFloat("BB_GAME_FPS_CAP", 0.0f, 0.0f, 1000.0f, &cap_set);
    const bool write_period = cap_set || st_lim == St::Orig;
    if (write_period) {
        g_period = cap >= 10.0f ? 1.0f / cap : 0.001f; // 0 / unset = unlimited (1 ms)
        for (const u32 p : PeriodImmVa) Write32(p + 5, Bits(g_period));
    } else {
        g_period = Float(Read32(PeriodImmVa[1] + 5));
    }
    // dt = min(max(measured, period), max dt): a cap below 1000 / BB_VARIABLE_DT_MAX_MS fps
    // would otherwise slow the game down.
    if (std::isfinite(g_period) && g_period > g_max_dt) g_max_dt = std::min(g_period, 0.1f);
    Write32(DtMovVa + 3, Bits(g_max_dt));

    // 4) XML lines that are wrong with a real dt: put the original code back (exact match only).
    std::string restored;
    const auto restore = [&](auto& g, const char* what) {
        if (Check(g) == St::Xml) {
            Apply(g, false);
            restored += std::string(restored.empty() ? "" : "; ") + what;
        }
    };
    restore(kXmlStreamCave, "0x1DBC308 count*measured/dt cave");
    restore(kXmlAnimDtCave, "0x1F77CEA measured-dt cave (lost the paused=0 case, clobbered rbx)");
    restore(kXmlTaskDtCave, "0xFD3557 cave (dropped the null check, wrote 1.0 into task dt)");
    restore(kXmlBadCall, "0x2315D71 call into 0x23166F7 (mid-function, unbalanced stack)");
    restore(kXmlBadRet, "0x2083EC1 ret inside a framed function");
    restore(kXmlCamera, "0x1440799 camera DirectPitchAng 30 deg / DirectPitchRate 0.03");

    // 5) Readers of the per-frame Step constants -> LiveDt.
    u32 redirected = 0, already = 0, skipped = 0;
    for (const Reader& r : Readers) {
        const s32 want = static_cast<s32>(static_cast<s64>(LiveDtVa) - (r.disp + 4));
        const s32 orig = static_cast<s32>(static_cast<s64>(r.site) - (r.disp + 4));
        s32 cur = 0;
        std::memcpy(&cur, AtV(r.disp), 4);
        if (std::memcmp(AtV(r.disp - 4), r.pre, 4) != 0) {
            ++skipped;
            LOG_WARNING(Lib_VideoOut, "[vardt] reader 0x{:x}: instruction changed, left alone",
                        r.disp - 4);
        } else if (cur == want) {
            ++already;
        } else if (cur == orig) {
            std::memcpy(AtV(r.disp), &want, 4);
            ++redirected;
        } else {
            ++skipped;
            LOG_WARNING(Lib_VideoOut, "[vardt] reader 0x{:x}: disp 0x{:x} unexpected, left alone",
                        r.disp - 4, static_cast<u32>(cur));
        }
    }

    // 6) Camera DirectPitchRate (per-frame lerp) -> PitchRateVa.
    Write32(PitchRateVa, Bits(PitchRate30));
    u32 cam = 0;
    for (const CamSite& c : CamSites) {
        if (std::memcmp(AtV(c.va), c.orig, 9) != 0) continue;
        u8 b[9];
        std::memcpy(b, c.pre, 4);
        const s32 d = static_cast<s32>(static_cast<s64>(PitchRateVa) - (c.va + 8));
        std::memcpy(b + 4, &d, 4);
        b[8] = 0x90;
        std::memcpy(AtV(c.va), b, 9);
        ++cam;
    }
    bool ctor_ok = true;
    for (const u32 v : CamCtorVa) ctor_ok = ctor_ok && std::memcmp(AtV(v), CamCtorBytes, 10) == 0;
    g_cam = cam == 2;

    // 7) Constants.
    u64 lo = ~0ull, hi = 0;
    u32 n_step = 0, n_keep = 0, n_keep_fixed = 0, n_law = 0, n_bad = 0;
    for (size_t i = 0; i < NumSites; ++i) {
        const Site& s = Sites[i];
        const u32 cur = Read32(s.va);
        if (cur != Bits(s.orig) && (s.xml == 0 || cur != s.xml)) {
            ++n_bad;
            LOG_WARNING(Lib_VideoOut,
                        "[vardt] constant 0x{:x} = 0x{:08x}, expected 0x{:08x} or 0x{:08x}: left "
                        "alone",
                        s.va, cur, Bits(s.orig), s.xml);
            continue;
        }
        if (s.law == Law::Keep) {
            if (cur != Bits(s.orig)) {
                Write32(s.va, Bits(s.orig));
                ++n_keep_fixed;
            }
            ++n_keep;
            continue;
        }
        g_site_live[i] = true;
        s.law == Law::Step ? ++n_step : ++n_law;
        lo = std::min<u64>(lo, s.va), hi = std::max<u64>(hi, s.va);
        Write32(s.va, Bits(Value(s, 1.0f / 30.0f)));
    }
    if (hi) MakeWritable(lo, hi);
    g_installed = true;

    LOG_INFO(Lib_VideoOut,
             "[vardt] on ({}): game dt = clamp(measured frame+0x264, period, {:.1f} ms); frame cap "
             "{} (period {:.3f} ms, BB_GAME_FPS_CAP{}); Havok semi-fixed steps h <= 1/60 s; pad dt "
             "= live dt; flips immediate; presenter {}",
             g_mode, g_max_dt * 1000.0f,
             g_period > 0.0015f ? std::to_string(static_cast<int>(std::lround(1.0f / g_period)))
                                : std::string("unlimited"),
             g_period * 1000.0f, cap_set ? " set" : " unset",
             g_immediate ? "flips immediately (BB_VRR_PRESENT)" : "on the vblank tick");
    LOG_INFO(Lib_VideoOut,
             "[vardt] per-frame dt readers -> live dt: {} redirected, {} already, {} left alone "
             "(of {}); constants: {} step, {} frames->seconds kept at 1/30 ({} restored from the "
             "XML value), {} cloth/damping law, {} unexpected; camera DirectPitchRate {}/2 "
             "(ctor 0.06 {})",
             redirected, already, skipped, NumReaders, n_step, n_keep, n_keep_fixed, n_law, n_bad,
             cam, ctor_ok ? "ok" : "NOT FOUND");
    if (!restored.empty()) {
        LOG_INFO(Lib_VideoOut, "[vardt] XML lines put back to the original code: {}", restored);
    }
    InstallRunFix();
}

void OnGuestFlip(int flip_mode) {
    if (!g_installed) {
        return;
    }
    float dt = 0;
    std::memcpy(&dt, AtV(LiveDtVa), 4);
    const bool good = std::isfinite(dt) && dt > 0.0001f && dt <= g_max_dt + 0.0001f;
    if (good) {
        for (size_t i = 0; i < NumSites; ++i) {
            if (g_site_live[i]) Write32(Sites[i].va, Bits(Value(Sites[i], dt)));
        }
        if (g_cam) Write32(PitchRateVa, Bits(PitchRate(dt)));
        if (g_run_factors) {
            Write32(RunDecayVa, Bits(std::pow(RunDecay30, dt * 30.0f)));
            Write32(RunGrowVa, Bits(std::pow(RunGrow30, dt * 30.0f)));
        }
    }
    float measured = 0, period = 0;
    const bool frame = ReadFrame(&measured, &period);
    const float havok_acc = Float(Read32(HavokAccVa));
    const double now = Now();
    std::scoped_lock lk{g_mutex};
    if (good) {
        g_used.Add(dt);
    } else {
        ++g_bad_dt;
    }
    if (frame && std::isfinite(measured)) {
        g_measured.Add(measured);
        if (measured > g_max_dt) ++g_clamp_hi;
        if (measured < period) ++g_clamp_lo;
    }
    if (frame) g_frame_period = period;
    const SpikeCounts spike_now = ReadSpikeCounts();
    const SpikeCounts spike = Since(spike_now, g_spike_prev);
    g_spike_prev = spike_now;
    const u64 fault_max_us = BBPerf2::Get().fault_max_us.exchange(0, std::memory_order_relaxed);
    const bool menu_open = BbMenu::IsOpen();
    if (menu_open) {
        g_spike_menu = true;
    } else if (g_spike_menu) {
        g_spike_menu = false; // the frame that closed the menu spans the pause
    } else if (frame && std::isfinite(measured) && measured >= kSpikeSec &&
               g_spike_logged < kSpikeLogMax) {
        ++g_spike_logged;
        LOG_INFO(Lib_VideoOut,
                 "[spike] {:.1f} ms guest frame at {:.3f} s (monotonic): cpu-write faults {} "
                 "(fault-around {} pages) | buffer upload {:.2f} MiB | texture upload {:.2f} MiB "
                 "({} images) | pipelines {} ({:.1f} ms) | GPU waits {} ({:.1f} ms) | guest file "
                 "reads {} ({:.2f} MiB requested, {:.1f} ms) | host commits {} ({:.1f} ms) | "
                 "guest blocked in gnm submit/flip {:.1f} ms | guest fault handlers {:.1f} ms "
                 "(longest {:.1f} ms) | GPU thread busy {:.1f} ms (Vulkan submit {:.1f} ms) | "
                 "presenter {:.1f} ms",
                 measured * 1e3, now, spike.faults, spike.fault_around,
                 static_cast<double>(spike.buf_up) / (1 << 20),
                 static_cast<double>(spike.tex_up) / (1 << 20), spike.tex_images, spike.pipelines,
                 spike.compile_us / 1e3, spike.waits, spike.wait_us / 1e3, spike.file_reads,
                 static_cast<double>(spike.file_read_bytes) / (1 << 20), spike.file_read_us / 1e3,
                 spike.commit_calls, spike.commit_us / 1e3, spike.gnm_wait_us / 1e3,
                 spike.fault_us / 1e3, fault_max_us / 1e3, spike.cp_busy_us / 1e3,
                 spike.cp_submit_us / 1e3, spike.present_us / 1e3);
    }
    if (std::isfinite(havok_acc)) g_havok_acc.Add(havok_acc);
    if (g_ragdoll_fix) {
        ++g_ragdoll_frames;
        if (good && havok_acc >= dt - 0.0001f) ++g_ragdoll_zero_steps; // no substep this frame
        if (g_ragdoll_last_log == 0) g_ragdoll_last_log = now;
        if (now - g_ragdoll_last_log >= 5.0) {
            LOG_INFO(Lib_VideoOut,
                     "[ragdoll] {:.1f}s: {} frames, {} without a Havok substep; fixed h ~1/60 s, "
                     "acc now {:.2f} ms",
                     now - g_ragdoll_last_log, g_ragdoll_frames, g_ragdoll_zero_steps,
                     havok_acc * 1e3);
            g_ragdoll_frames = g_ragdoll_zero_steps = 0;
            g_ragdoll_last_log = now;
        }
    }
    if (g_run_fix) {
        ++g_run_frames;
        if (g_run_last_log == 0) g_run_last_log = now;
        if (now - g_run_last_log >= 5.0) {
            const double secs = now - g_run_last_log;
            if (g_run != nullptr) {
                // Written by guest threads without a lock: a snapshot, good enough for a log.
                volatile RunStats* v = g_run;
                RunStats s{};
                s.actual_sum = v->actual_sum, s.n = v->n, s.intended_sum = v->intended_sum;
                s.scale_sum = v->scale_sum, s.blocked = v->blocked;
                v->actual_sum = 0.0f;
                v->intended_sum = 0.0f;
                v->scale_sum = 0.0f;
                v->n = 0;
                v->blocked = 0;
                const double k = s.n ? 1.0 / s.n : 0.0;
                LOG_INFO(Lib_VideoOut,
                         "[runfix] {:.1f}s {:.1f} fps: {} fast-move checks (root motion > 5 m/s, "
                         "grounded; mostly the player sprinting): root-motion speed avg {:.2f} "
                         "m/s, actual speed (move / dt) avg {:.2f} m/s, move scale avg {:.3f}, "
                         "blocked (< 1 m/s) {}",
                         secs, g_run_frames / secs, s.n, s.intended_sum * k,
                         s.actual_sum * k, s.scale_sum * k, s.blocked);
            }
            g_run_frames = 0;
            g_run_last_log = now;
        }
    }
    if (g_last_flip > 0) g_flip_iv.Add(now - g_last_flip);
    g_last_flip = now;
    (flip_mode == 1 ? g_mode_vsync : flip_mode == 2 ? g_mode_imm : g_mode_other)++;
    if (g_last_log == 0) g_last_log = now;
    if (now - g_last_log < 2.0) {
        return;
    }
    LOG_INFO(Lib_VideoOut,
             "[vardt] {:.1f}s {} frames ({:.1f} fps): measured dt avg {:.2f} min {:.2f} max "
             "{:.2f} ms | game dt used avg {:.2f} min {:.2f} max {:.2f} ms (clamp {:.2f}..{:.1f}; "
             "frames over max {} under period {}) | flip interval avg {:.2f} min {:.2f} max "
             "{:.2f} ms | present interval avg {:.2f} max {:.2f} ms ({} presents) | flipMode vsync "
             "{} immediate {} other {} | bad dt {} | havok acc after step avg {:.3f} max {:.3f} ms "
             "| pitch rate {:.4f} | step const {:.5f}",
             now - g_last_log, g_flip_iv.n, g_flip_iv.n / (now - g_last_log),
             g_measured.Avg() * 1e3, g_measured.n ? g_measured.mn * 1e3 : 0.0,
             g_measured.mx * 1e3, g_used.Avg() * 1e3, g_used.n ? g_used.mn * 1e3 : 0.0,
             g_used.mx * 1e3, g_frame_period * 1e3, g_max_dt * 1e3, g_clamp_hi, g_clamp_lo,
             g_flip_iv.Avg() * 1e3, g_flip_iv.n ? g_flip_iv.mn * 1e3 : 0.0, g_flip_iv.mx * 1e3,
             g_present_iv.Avg() * 1e3, g_present_iv.mx * 1e3, g_present_iv.n, g_mode_vsync,
             g_mode_imm, g_mode_other, g_bad_dt, g_havok_acc.Avg() * 1e3, g_havok_acc.mx * 1e3,
             Float(Read32(PitchRateVa)), Float(Read32(Sites[0].va)));
    g_measured.Reset(), g_used.Reset(), g_flip_iv.Reset(), g_present_iv.Reset();
    g_havok_acc.Reset();
    g_mode_vsync = g_mode_imm = g_mode_other = g_bad_dt = g_clamp_hi = g_clamp_lo = 0;
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
