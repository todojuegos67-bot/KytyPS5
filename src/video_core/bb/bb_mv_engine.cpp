// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-mv-engine: see bb_mv_engine.h.

#include "video_core/bb/bb_mv_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::MvEngine {
namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 kHist = 4;              // records kept per palette address
constexpr u64 kMaxOffset = 0x4000;    // V# base up to this far into a palette
constexpr u64 kKeepFrames = 16;       // records older than this (GPU frames) are dropped

struct Rec {
    u64 hash = 0; // first matrix (48 bytes) when the game handed the palette to the sink
    u64 key = 0;
    u64 sink = 0;
    u64 stamp = 0;
};
struct Slot {
    std::array<Rec, kHist> r{};
    u32 next = 0;
};

struct Tls {
    u64 model = 0, sink = 0, mesh = 0, bones = 0;
    u32 consumer = 0;
    bool valid = false;
};
thread_local Tls t_cur;

std::mutex g_mx;
std::map<u64, Slot> g_pal; // palette start -> records
std::atomic<u64> g_stamp{1};
std::atomic<u64> g_guest_sets{0}, g_guest_begins{0}, g_guest_nobegin{0}, g_recorded{0};
std::atomic<bool> g_hooks{false};
std::string g_hook_detail;

// GPU thread.
u64 g_pass = 0;
std::unordered_map<u64, u64> g_cam_of;     // block key -> camera key (this frame)
std::unordered_map<u64, u32> g_pass_n;     // pass -> camera blocks seen this frame
std::unordered_set<u64> g_passes_frame, g_sinks_frame;
std::array<u64, size_t(Stat::Count)> g_st{};
std::array<u64, size_t(Miss::Count)> g_miss{};
u64 g_frames = 0, g_passes_sum = 0, g_sinks_sum = 0;
Clock::time_point g_t0 = Clock::now();

u64 Mix(u64 a, u64 b) {
    u64 h = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

u64 Hash48(const void* p) {
    u64 v[6];
    std::memcpy(v, p, 48);
    u64 h = 0x6d76656e67ull;
    for (u64 x : v) h = Mix(h, x);
    return h;
}

void Log(const std::string& text) {
    LOG_INFO(Render_Vulkan, "{}", text);
    BB::Upscale::Checkpoint(text);
}

double Pct(u64 a, u64 b) {
    return b != 0 ? 100.0 * double(a) / double(b) : 0.0;
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_MV_ENGINE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

void SetHooks(bool installed, const char* detail) {
    g_hooks = installed;
    {
        std::scoped_lock lk{g_mx};
        g_hook_detail = detail != nullptr ? detail : "";
    }
    Log(fmt::format("[mv-engine] hooks {}: {}", installed ? "in place" : "NOT installed",
                    detail != nullptr ? detail : ""));
}

void GuestPaletteBegin(u64 model, u64 sink, u64 mesh, u64 bones, u32 consumer) {
    t_cur = {model, sink, mesh, bones, consumer, bones != 0};
    g_guest_begins.fetch_add(1, std::memory_order_relaxed);
}

void GuestPaletteSet(u64 palette, u32 consumer) {
    g_guest_sets.fetch_add(1, std::memory_order_relaxed);
    if (!t_cur.valid || palette == 0 || t_cur.consumer != consumer) {
        g_guest_nobegin.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // The palette has just been written by the game (the loop before the call): first matrix.
    const u64 h = Hash48(reinterpret_cast<const void*>(palette));
    const u64 key = Mix(Mix(Mix(t_cur.bones, t_cur.mesh), consumer), 0x6f626a);
    std::scoped_lock lk{g_mx};
    Slot& s = g_pal[palette];
    s.r[s.next % kHist] = {h, key, t_cur.sink, g_stamp.load(std::memory_order_relaxed)};
    ++s.next;
    g_recorded.fetch_add(1, std::memory_order_relaxed);
}

bool LookupPalette(Core::MemoryManager* memory, bool gpu_modified, u64 addr, ObjId& out,
                   Miss& why) {
    why = Miss::None;
    if (gpu_modified) {
        why = Miss::GpuWritten; // not a CPU frame-allocator palette (skin ring: own identity)
        return false;
    }
    u64 start = 0;
    Slot slot;
    {
        std::scoped_lock lk{g_mx};
        auto it = g_pal.upper_bound(addr);
        if (it == g_pal.begin()) {
            why = Miss::NoRecord;
            return false;
        }
        --it;
        if (addr - it->first >= kMaxOffset) {
            why = Miss::NoRecord;
            return false;
        }
        start = it->first;
        slot = it->second;
    }
    if (memory == nullptr || !memory->IsValidMapping(start, 48)) {
        why = Miss::Unmapped;
        return false;
    }
    const u64 h = Hash48(reinterpret_cast<const void*>(start));
    // The newest record whose first matrix is still in guest memory (exact).
    const Rec* best = nullptr;
    for (const Rec& r : slot.r) {
        if (r.key != 0 && r.hash == h && (best == nullptr || r.stamp > best->stamp)) best = &r;
    }
    if (best == nullptr) {
        why = Miss::Stale;
        return false;
    }
    out = {best->key, best->sink, addr - start};
    g_sinks_frame.insert(best->sink);
    return true;
}

void SetPass(u64 pass_key) {
    g_pass = pass_key;
}

u64 CameraKey(u64 block_key, u32& ordinal) {
    const u64 k = Mix(block_key, g_pass);
    if (const auto it = g_cam_of.find(k); it != g_cam_of.end()) {
        ordinal = u32(it->second & 0xffff);
        return it->second;
    }
    g_passes_frame.insert(g_pass);
    ordinal = g_pass_n[g_pass]++;
    const u64 ck = (Mix(Mix(g_pass, 0x63616d70617373ull), ordinal) & ~0xffffull) | ordinal;
    g_cam_of[k] = ck;
    return ck;
}

void Count(Stat s, u64 n) {
    g_st[size_t(s)] += n;
}

void CountMiss(Miss m) {
    ++g_miss[size_t(m)];
}

void OnFrame() {
    if (!Enabled()) return;
    ++g_frames;
    g_passes_sum += g_passes_frame.size();
    g_sinks_sum += g_sinks_frame.size();
    g_cam_of.clear();
    g_pass_n.clear();
    g_passes_frame.clear();
    g_sinks_frame.clear();
    const u64 stamp = g_stamp.fetch_add(1) + 1;
    const auto now = Clock::now();
    if (now - g_t0 < std::chrono::seconds(1)) return;
    const double secs = std::chrono::duration<double>(now - g_t0).count();
    g_t0 = now;
    size_t addrs = 0;
    {
        std::scoped_lock lk{g_mx};
        for (auto it = g_pal.begin(); it != g_pal.end();) {
            u64 newest = 0;
            for (const Rec& r : it->second.r) newest = std::max(newest, r.stamp);
            if (newest + kKeepFrames < stamp) {
                it = g_pal.erase(it);
            } else {
                ++it;
            }
        }
        addrs = g_pal.size();
    }
    const auto S = [](Stat s) { return g_st[size_t(s)]; };
    const auto M = [](Miss m) { return g_miss[size_t(m)]; };
    const double f = double(std::max<u64>(1, g_frames));
    Log(fmt::format(
        "[mv-engine] {:.1f} s {} frames | hooks {} | caves entered: BonePalette entry {:.1f}/frame, "
        "SetPalette site {:.1f}/frame; recorded {:.1f}/frame (site without entry record {}), {} "
        "addresses (0 entries: the game never took BonePalette, every instance drawn has a GPU "
        "skin ring: DrawMesh eboot+0x21d0740 test [instance+8]) | draws {:.0f}/frame, engine identity {:.1f}% | palettes "
        "{:.0f}/frame: identity {:.1f}%, previous transform found {:.1f}% of identified, new "
        "object {:.1f}/frame | fallbacks: no record {:.1f}/frame, GPU-written {:.1f}/frame, "
        "stale {:.1f}/frame, unmapped {:.1f}/frame, old path used {:.1f}/frame | camera blocks "
        "{:.0f}/frame: pass key {:.1f}% (previous found {:.1f}%), fallback {:.1f}% | passes "
        "recognized {:.1f}/frame, engine sinks {:.1f}/frame",
        secs, g_frames, g_hooks.load() ? "on" : "off", double(g_guest_begins.exchange(0)) / f,
        double(g_guest_sets.exchange(0)) / f, double(g_recorded.exchange(0)) / f,
        g_guest_nobegin.exchange(0), addrs, double(S(Stat::Draws)) / f,
        Pct(S(Stat::EngineDraws), S(Stat::Draws)), double(S(Stat::PalLookups)) / f,
        Pct(S(Stat::PalFound), S(Stat::PalLookups)), Pct(S(Stat::PalPrev), S(Stat::PalFound)),
        double(S(Stat::PalNew)) / f, double(M(Miss::NoRecord)) / f,
        double(M(Miss::GpuWritten)) / f, double(M(Miss::Stale)) / f,
        double(M(Miss::Unmapped)) / f, double(S(Stat::FallbackOld)) / f,
        double(S(Stat::CamLookups)) / f, Pct(S(Stat::CamPass), S(Stat::CamLookups)),
        Pct(S(Stat::CamPassPrev), S(Stat::CamPass)), Pct(S(Stat::CamFallback), S(Stat::CamLookups)),
        double(g_passes_sum) / f, double(g_sinks_sum) / f));
    g_st.fill(0);
    g_miss.fill(0);
    g_frames = g_passes_sum = g_sinks_sum = 0;
}

} // namespace BB::MvEngine
