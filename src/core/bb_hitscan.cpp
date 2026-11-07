// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "core/bb_hitscan.h"

#include <algorithm>
#include <fmt/format.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "common/elf_info.h"
#include "common/logging/log.h"

#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace BB::HitScan {

static uintptr_t g_base = 0;
static u64 g_size = 0;
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_stop{false};
static std::mutex g_thread_m;
static std::thread g_thread;

void SetEboot(uintptr_t base, u64 size) {
    g_base = base, g_size = size;
}

#ifdef _WIN32
// A failed read (unmapped / protected page) returns false; it never faults.
static bool SafeRead(uintptr_t addr, void* out, size_t n) {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(addr), out, n, &got) &&
           got == n;
}

namespace {
constexpr u32 kWindow = 0x800;      // bytes kept per candidate for the before / after diff
constexpr u32 kDmgOff = 0x224;      // GetDamageLastFrame field
constexpr size_t kMaxObjects = 4'000'000;
constexpr size_t kMaxCandidates = 3000;
constexpr u32 kMaxPerVtable = 64;   // characters: few instances per class

struct Cand {
    uintptr_t addr;
    u32 vt;          // vtable, image offset
    u32 vt_count;
    u32 hits = 0;    // +0x224 went from 0 to non-zero
    float max = 0, last = 0;
    std::vector<u8> before;
};
} // namespace

static void Run() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const uintptr_t vt_lo = g_base + 0x50dc000, vt_hi = g_base + 0x53e0000;
    const uintptr_t img_lo = g_base, img_hi = g_base + g_size;
    std::vector<std::pair<uintptr_t, u32>> objs;
    objs.reserve(1 << 20);
    std::vector<u8> buf(1 << 20);
    u64 scanned = 0;
    uintptr_t a = 0x400000;
    const uintptr_t end = 0x10000000000ull;
    MEMORY_BASIC_INFORMATION mi{};
    while (a < end && !g_stop && objs.size() < kMaxObjects) {
        if (VirtualQuery(reinterpret_cast<LPCVOID>(a), &mi, sizeof(mi)) == 0) break;
        const uintptr_t rb = reinterpret_cast<uintptr_t>(mi.BaseAddress);
        const uintptr_t re = rb + mi.RegionSize;
        const DWORD p = mi.Protect & 0xff;
        const bool rw = mi.State == MEM_COMMIT && !(mi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) &&
                        (p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE);
        if (rw && !(rb < img_hi && re > img_lo)) {
            for (uintptr_t c = rb; c < re && objs.size() < kMaxObjects; c += buf.size()) {
                const size_t n = std::min<uintptr_t>(buf.size(), re - c);
                if (!SafeRead(c, buf.data(), n)) continue;
                scanned += n;
                for (size_t i = 0; i + 8 <= n; i += 8) {
                    u64 v;
                    std::memcpy(&v, buf.data() + i, 8);
                    if (v >= vt_lo && v < vt_hi && (v & 7) == 0) {
                        objs.emplace_back(c + i, static_cast<u32>(v - g_base));
                    }
                }
            }
        }
        a = std::max(re, a + 0x1000);
    }
    std::unordered_map<u32, u32> per_vt;
    for (auto& o : objs) ++per_vt[o.second];
    std::vector<Cand> cands;
    for (auto& o : objs) {
        const u32 cnt = per_vt[o.second];
        if (cnt > kMaxPerVtable) continue;
        Cand c{o.first, o.second, cnt};
        c.before.resize(kWindow);
        if (!SafeRead(c.addr, c.before.data(), kWindow)) continue;
        cands.push_back(std::move(c));
        if (cands.size() >= kMaxCandidates) break;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    LOG_INFO(Loader,
             "[hitscan] armed: {} MB scanned, {} objects with an eboot vtable ({} classes), {} "
             "candidates (<= {} per class), {} ms. Hit enemies now; Home stops.",
             scanned >> 20, objs.size(), per_vt.size(), cands.size(), kMaxPerVtable, ms);
    objs.clear();
    objs.shrink_to_fit();

    u64 polls = 0;
    u32 sec_hits = 0;
    auto sec_t = clock::now();
    while (!g_stop) {
        for (auto& c : cands) {
            float f;
            if (!SafeRead(c.addr + kDmgOff, &f, 4)) continue;
            if (!(f == f)) f = 0; // NaN
            if (f != 0 && c.last == 0) ++c.hits, ++sec_hits;
            c.max = std::max(c.max, f);
            c.last = f;
        }
        ++polls;
        if (clock::now() - sec_t >= std::chrono::seconds(1)) {
            if (sec_hits) LOG_INFO(Loader, "[hitscan] last second: {} +0x224 events", sec_hits);
            sec_hits = 0, sec_t = clock::now();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }

    // Report: +0x224 events first, then int fields that went down (HP-like: both in 0..200000,
    // drop >= 5), objects with fewer instances of their class first.
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
        if (x.hits != y.hits) return x.hits > y.hits;
        return x.vt_count < y.vt_count;
    });
    LOG_INFO(Loader, "[hitscan] stopped after {} polls; {} candidates", polls, cands.size());
    int lines = 0;
    std::vector<u8> after(kWindow);
    for (auto& c : cands) {
        if (lines >= 120) break;
        if (!SafeRead(c.addr, after.data(), kWindow)) continue;
        std::string dec;
        int ndec = 0;
        for (u32 off = 8; off + 4 <= kWindow; off += 4) {
            s32 b, n;
            std::memcpy(&b, c.before.data() + off, 4);
            std::memcpy(&n, after.data() + off, 4);
            if (b > 0 && b <= 200000 && n >= 0 && n < b && b - n >= 5) {
                if (ndec < 8) dec += fmt::format(" +{:#x}:{}->{}", off, b, n);
                ++ndec;
            }
        }
        if (c.hits == 0 && (ndec == 0 || ndec > 6)) continue;
        float fb;
        std::memcpy(&fb, c.before.data() + kDmgOff, 4);
        LOG_INFO(Loader,
                 "[hitscan] obj {:#x} vt {:#x} (x{}) +0x224 events {} max {:.1f} before {:.1f} "
                 "now {:.1f}; int drops {}:{}",
                 c.addr, c.vt, c.vt_count, c.hits, c.max, fb, c.last, ndec, dec);
        ++lines;
    }
    LOG_INFO(Loader, "[hitscan] {} lines", lines);
}
#endif

void Arm() {
#ifdef _WIN32
    if (!g_base || g_size < 0x5600000 ||
        Common::ElfInfo::Instance().GameSerial().find("CUSA03173") == std::string_view::npos) {
        LOG_INFO(Loader, "[hitscan] only for Bloodborne CUSA03173 1.09");
        return;
    }
    std::lock_guard lk{g_thread_m};
    if (g_running) {
        LOG_INFO(Loader, "[hitscan] already armed");
        return;
    }
    if (g_thread.joinable()) g_thread.join();
    g_stop = false, g_running = true;
    LOG_INFO(Loader, "[hitscan] arming (scanning guest memory, read-only)...");
    g_thread = std::thread([] {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        Run();
        g_running = false;
    });
#else
    LOG_INFO(Loader, "[hitscan] Windows only");
#endif
}

void Stop() {
#ifdef _WIN32
    std::lock_guard lk{g_thread_m};
    if (!g_running) {
        LOG_INFO(Loader, "[hitscan] not armed");
        return;
    }
    g_stop = true;
    // the thread logs and ends by itself; it is joined on the next arm
#endif
}

} // namespace BB::HitScan
