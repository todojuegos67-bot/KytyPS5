// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bb-ssr-probe (BB_SSR_PROBE=1, default off; diagnostic only, no image change).
// eboot 1.09 (vaddr = eboot offset; Ghidra = vaddr + 0x400000):
//  - 0xea8d80 SSR filter init (shader "GXSsr"), 0xea8f00 SSR draw (rdi = filter,
//    rsi = render step, gfx ctx = [[rsi+0x430]+8]); only call at 0x22b15a0 in step 0x22b1530,
//    which returns to 0x22b15a5 (rbx = rsi there).
//  - 0xea8f00 only sets cached state (0x21693f0.. are field stores / rets); where the PM4 lands
//    is not known, so the probe snapshots the ctx objects on entry and on return and takes every
//    qword that advanced by 16 B..4 MB from a pointer-like value as a write cursor. The guest
//    ranges [old, new) are matched against the address of each PM4 draw/dispatch header on the
//    GPU thread (needs the guest command buffer itself: not with copy_gpu_buffers).

#include "core/bb_ssr_probe.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace BB::SsrProbe {
namespace {

constexpr u64 SsrDrawVaddr = 0xea8f00;
// push rbp; mov rbp,rsp; push r15..r12; push rbx; sub rsp,0x1a8
constexpr u8 SsrPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                              0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xa8, 0x01, 0x00, 0x00};
constexpr u64 CallVaddr = 0x22b15a0;
constexpr u64 RetVaddr = 0x22b15a5;
// mov rax,[rbx+0x430]; movzx ecx,byte [rax+0x2092]
constexpr u8 RetBytes[] = {0x48, 0x8b, 0x83, 0x30, 0x04, 0x00, 0x00,
                           0x0f, 0xb6, 0x88, 0x92, 0x20, 0x00, 0x00};

constexpr u32 StepSize = 0x500;  // [rsi+0x430]: fields up to +0x480
constexpr u32 CtxSize = 0xbc10;  // [[rsi+0x430]+8]: fields up to +0xbc08
constexpr u32 MaxRanges = 64;
constexpr u32 MaxCursorSlots = (StepSize + CtxSize) / 8;

bool g_enabled = false;
std::atomic<u64> g_calls{0};
std::atomic<u64> g_ranges_total{0};
std::atomic<u64> g_matched{0};
std::atomic<u64> g_packets{0};
std::atomic<u32> g_cursor_hits[MaxCursorSlots]{};

struct Snapshot {
    u64 step = 0;
    u64 ctx = 0;
    std::vector<u8> step_bytes = std::vector<u8>(StepSize);
    std::vector<u8> ctx_bytes = std::vector<u8>(CtxSize);
};
thread_local Snapshot t_snap;

struct Range {
    u64 lo, hi, call;
};
std::mutex g_range_mutex;
std::array<Range, MaxRanges> g_ranges{};
u32 g_range_next = 0;

// GPU thread state
bool g_cur_in = false;
u32 g_draw_index = 0;
u64 g_cur_packet = 0;
std::string g_prev_desc;
u32 g_prev_index = 0;
bool g_group = false;
s64 g_last_group_ms = -100000;
s64 g_last_stats_ms = 0;

s64 NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

template <typename T>
T Read(u64 addr) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(T));
    return v;
}

void Grab(u64 step_holder) {
    t_snap.step = Read<u64>(step_holder + 0x430);
    t_snap.ctx = t_snap.step != 0 ? Read<u64>(t_snap.step + 8) : 0;
    if (t_snap.step != 0) {
        std::memcpy(t_snap.step_bytes.data(), reinterpret_cast<const void*>(t_snap.step), StepSize);
    }
    if (t_snap.ctx != 0) {
        std::memcpy(t_snap.ctx_bytes.data(), reinterpret_cast<const void*>(t_snap.ctx), CtxSize);
    }
}

void AddRange(u64 lo, u64 hi, u64 call) {
    std::scoped_lock lk{g_range_mutex};
    g_ranges[g_range_next] = {lo, hi, call};
    g_range_next = (g_range_next + 1) % MaxRanges;
    g_ranges_total.fetch_add(1);
}

void Diff(const u8* before, u64 obj, u32 size, u32 slot_base, u64 call, std::string& found) {
    for (u32 off = 0; off + 8 <= size; off += 8) {
        u64 a, b;
        std::memcpy(&a, before + off, 8);
        b = Read<u64>(obj + off);
        if (a < 0x100000 || b <= a || b - a < 16 || b - a > (4u << 20)) {
            continue;
        }
        g_cursor_hits[slot_base + off / 8].fetch_add(1);
        AddRange(a, b, call);
        if (found.size() < 300) {
            found += fmt::format(" {}+{:#x}:{:#x}->{:#x}", slot_base == 0 ? "step" : "ctx", off, a,
                                 b);
        }
    }
}

// regs: rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax, then the return address
void PS4_SYSV_ABI OnEnter(const u64* regs) {
    Grab(regs[6]);
}

void PS4_SYSV_ABI OnReturn(const u64* regs) {
    const u64 call = g_calls.fetch_add(1) + 1;
    const u64 step = Read<u64>(regs[0] + 0x430);
    if (step != t_snap.step || step == 0) {
        return;
    }
    std::string found;
    Diff(t_snap.step_bytes.data(), step, StepSize, 0, call, found);
    if (t_snap.ctx != 0 && Read<u64>(step + 8) == t_snap.ctx) {
        Diff(t_snap.ctx_bytes.data(), t_snap.ctx, CtxSize, StepSize / 8, call, found);
    }
    if (call <= 6) {
        LOG_INFO(Render, "[ssr-probe] guest SSR call #{} step {:#x} ctx {:#x}; advanced cursors:{}",
                 call, step, t_snap.ctx, found.empty() ? " none" : found);
    }
}

void* AllocExec(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p =
        mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

// Same cave layout as bb_rumble_hits.cpp: save volatile regs + xmm0..7, host(regs), restore,
// run the relocated bytes, jump back.
std::vector<u8> BuildCave(u64 host, const u8* moved, u32 len, u64 back) {
    std::vector<u8> e;
    const auto put = [&](std::initializer_list<u8> v) { e.insert(e.end(), v); };
    const auto put64 = [&](u64 v) {
        for (int i = 0; i < 8; ++i) {
            e.push_back(u8(v >> (i * 8)));
        }
    };
    put({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, 0x53});
    put({0x48, 0x89, 0xe3});
    put({0x48, 0x83, 0xe4, 0xf0});
    put({0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x7f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x7f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdf});
    put({0x48, 0xb8});
    put64(host);
    put({0xff, 0xd0});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x6f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x6f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdc});
    put({0x5b, 0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    e.insert(e.end(), moved, moved + len);
    put({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
    put64(back);
    return e;
}

void Detour(u8* at, u32 len, u8* cave, u64 host) {
    const auto c = BuildCave(host, at, len, reinterpret_cast<u64>(at + len));
    std::memcpy(cave, c.data(), c.size());
    u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
    const u64 to = reinterpret_cast<u64>(cave);
    std::memcpy(j + 6, &to, 8);
    std::memcpy(at, j, sizeof(j));
    std::memset(at + sizeof(j), 0x90, len - sizeof(j));
}

bool InRange(u64 addr) {
    std::scoped_lock lk{g_range_mutex};
    for (const auto& r : g_ranges) {
        if (addr >= r.lo && addr < r.hi) {
            return true;
        }
    }
    return false;
}

void Stats(s64 now) {
    if (now - g_last_stats_ms < 5000) {
        return;
    }
    g_last_stats_ms = now;
    std::string top;
    for (u32 n = 0; n < 3; ++n) {
        u32 best = 0, best_i = 0;
        for (u32 i = 0; i < MaxCursorSlots; ++i) {
            const u32 h = g_cursor_hits[i].load();
            if (h > best && top.find(fmt::format("#{}#", i)) == std::string::npos) {
                best = h;
                best_i = i;
            }
        }
        if (best == 0) {
            break;
        }
        const bool in_step = best_i < StepSize / 8;
        top += fmt::format(" {}+{:#x}={}#{}#", in_step ? "step" : "ctx",
                           (in_step ? best_i : best_i - StepSize / 8) * 8, best, best_i);
    }
    LOG_INFO(Render,
             "[ssr-probe] stats: guest SSR calls {} ranges {} gpu draws {} matched {}; top "
             "cursors:{}",
             g_calls.load(), g_ranges_total.load(), g_packets.load(), g_matched.load(),
             top.empty() ? " none" : top);
}

} // namespace

bool Enabled() {
    return g_enabled;
}

void OnSubmit() {
    if (!g_enabled) {
        return;
    }
    if (g_group) {
        LOG_INFO(Render, "[ssr-probe] after: (end of submission)");
        g_group = false;
        g_last_group_ms = NowMs();
    }
    g_draw_index = 0;
    g_prev_desc.clear();
}

void OnPacket(const void* header) {
    if (!g_enabled) {
        return;
    }
    g_cur_packet = reinterpret_cast<u64>(header);
    g_cur_in = InRange(g_cur_packet);
    g_packets.fetch_add(1);
    if (g_cur_in) {
        g_matched.fetch_add(1);
    }
}

void OnDraw(const std::string& desc) {
    if (!g_enabled) {
        return;
    }
    const u32 idx = g_draw_index++;
    const s64 now = NowMs();
    Stats(now);
    if (g_cur_in && (g_group || now - g_last_group_ms >= 5000)) {
        if (!g_group) {
            g_group = true;
            LOG_INFO(Render, "[ssr-probe] before #{}: {}", g_prev_index,
                     g_prev_desc.empty() ? "(start of submission)" : g_prev_desc);
        }
        LOG_INFO(Render, "[ssr-probe] SSR #{} pm4 {:#x}: {}", idx, g_cur_packet, desc);
    } else if (g_group) {
        LOG_INFO(Render, "[ssr-probe] after #{}: {}", idx, desc);
        g_group = false;
        g_last_group_ms = now;
    }
    g_cur_in = false;
    g_prev_desc = desc;
    g_prev_index = idx;
}

void OnEbootPatched(uintptr_t eboot_base, u64 image_size) {
    const char* v = std::getenv("BB_SSR_PROBE");
    if (v == nullptr || v[0] != '1') {
        return;
    }
    u8* entry = reinterpret_cast<u8*>(eboot_base + SsrDrawVaddr);
    u8* ret = reinterpret_cast<u8*>(eboot_base + RetVaddr);
    const u8* call = reinterpret_cast<const u8*>(eboot_base + CallVaddr);
    s32 rel = 0;
    if (RetVaddr + sizeof(RetBytes) <= image_size) {
        std::memcpy(&rel, call + 1, 4);
    }
    if (RetVaddr + sizeof(RetBytes) > image_size ||
        std::memcmp(entry, SsrPrologue, sizeof(SsrPrologue)) != 0 || call[0] != 0xe8 ||
        RetVaddr + rel != SsrDrawVaddr || std::memcmp(ret, RetBytes, sizeof(RetBytes)) != 0) {
        LOG_WARNING(Render, "[ssr-probe] SSR draw bytes differ (not eboot 1.09?): probe off");
        return;
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        LOG_WARNING(Render, "[ssr-probe] no executable memory: probe off");
        return;
    }
    Detour(entry, sizeof(SsrPrologue), cave, reinterpret_cast<u64>(&OnEnter));
    Detour(ret, sizeof(RetBytes), cave + 2048, reinterpret_cast<u64>(&OnReturn));
    g_enabled = true;
    LOG_INFO(Render, "[ssr-probe] on: eboot+{:#x} entry and return eboot+{:#x} detoured",
             SsrDrawVaddr, RetVaddr);
}

} // namespace BB::SsrProbe
