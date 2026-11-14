// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-smooth: see bb_smooth.h.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_smooth.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

namespace BB::Smooth {
namespace {

double EnvNum(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    return end != v ? d : fallback;
}

int Multiplier() {
    const auto& c = BB::FG::Cfg();
    if (c.kind == BB::FG::Kind::Off) return 1;
    return std::clamp(c.multiplier, 2, 4);
}

struct Window {
    std::vector<double> real;   // ms between real presents
    std::vector<double> work;   // ms, estimated game work per frame (interval - slack)
    std::vector<double> shown;  // ms, planned shown intervals (FSR 3 grid)
    uint32_t ticks = 0, flips = 0, misses = 0, shown_count = 0;
    bool idle = false;
    double start = 0.0;
    void Clear(double now) {
        real.clear();
        work.clear();
        shown.clear();
        ticks = flips = misses = shown_count = 0;
        idle = false;
        start = now;
    }
};

struct State {
    bool governing = false;
    std::vector<int> rungs; // descending, fps
    double hz = -1.0;
    int mult = 1;
    int idx = 0;            // current rung
    int max_idx_ok = 0;
    int clean = 0;          // consecutive clean windows (for stepping up)
    int dirty = 0;          // consecutive windows with misses (for stepping down)
    int up_hold = 5;        // clean windows needed before stepping up (back-off)
    double last_up = -1e9, last_change = -1e9;
    double last_present = 0.0, last_flip_tick = 0.0, last_tick = 0.0;
    std::string steps;      // step changes this second, for the log
    Window w;
    const char* why_off = nullptr;
    bool logged_off = false;
} g;

std::atomic<double> g_grid{0.0};
// NoteShown runs on the FSR 3 present path, which may not be the vblank thread: the shown
// samples and the governing flag it reads are kept apart from g under their own lock.
std::mutex g_shown_mu;
std::vector<double> g_shown;
uint32_t g_shown_count = 0;
std::atomic<bool> g_governing{false};
std::atomic<uint32_t> g_epoch{0};

void BuildRungs() {
    const double hz = BB::FG::Amd::DisplayHz();
    const int mult = Multiplier();
    if (!g.rungs.empty() && std::fabs(hz - g.hz) < 0.5 && mult == g.mult) return;
    g.hz = hz;
    g.mult = mult;
    int hi = static_cast<int>(std::clamp(EnvNum("BB_SMOOTH_MAX", 60.0), 30.0, 60.0));
    // A real cap the user chose (BB_FPS_CAP 35..59, e.g. Soulsborne ligero+ 40 x2 = 80) is the top
    // rung: the mode then holds exactly that and only steps down when the game cannot.
    if (const char* cap = std::getenv("BB_FPS_CAP"); cap != nullptr) {
        const int c = std::atoi(cap);
        if (c >= 35 && c < 60) hi = std::min(hi, c);
    }
    const int lo = static_cast<int>(std::clamp(EnvNum("BB_SMOOTH_MIN", 30.0), 20.0, 60.0));
    // G-SYNC / FreeSync: stay a few % under the refresh (above it, mailbox drops frames or VRR
    // leaves its range and tears / waits on vsync). Rungs divide 240 / 120 evenly as well.
    const double top = hz > 30.0 ? hz * 0.97 : 1e9;
    g.rungs.clear();
    for (int r : {60, 48, 45, 40, 30, 24}) {
        if (r == 45 && hi != 45) continue; // 45 only as a chosen cap
        if (r > hi || r < lo) continue;
        if (r * mult > top) continue;
        g.rungs.push_back(r);
    }
    if (g.rungs.empty()) g.rungs.push_back(std::max(lo, 30));
    g.idx = std::min<int>(g.idx, static_cast<int>(g.rungs.size()) - 1);
    std::string list;
    for (int r : g.rungs) {
        if (!list.empty()) list += "/";
        list += std::to_string(r);
    }
    BBLogf("[smooth] Fluidez adaptativa: display %.1f Hz, x%d, real rungs %s fps (shown up to %d, "
           "kept under %.0f Hz for VRR); lowest shown %d fps%s",
           hz, mult, list.c_str(), g.rungs.front() * mult, top > 1e8 ? 0.0 : top,
           g.rungs.back() * mult,
           g.rungs.back() * mult < 48 ? " (under ~48 Hz: the display may use LFC)" : "");
}

double RungInterval(int idx) {
    return 1.0 / g.rungs[std::clamp<int>(idx, 0, static_cast<int>(g.rungs.size()) - 1)];
}

void Stats(std::vector<double>& v, double& mean, double& sd, double& p99) {
    mean = sd = p99 = 0.0;
    if (v.empty()) return;
    double sum = 0.0, sq = 0.0;
    for (double x : v) {
        sum += x;
        sq += x * x;
    }
    mean = sum / v.size();
    sd = std::sqrt(std::max(0.0, sq / v.size() - mean * mean));
    std::sort(v.begin(), v.end());
    p99 = v[std::min(v.size() - 1, static_cast<size_t>(std::ceil(v.size() * 0.99)) - 1)];
}

double P95(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(std::ceil(v.size() * 0.95)) - 1)];
}

void Change(int to, double now, const char* why) {
    const int from = g.rungs[g.idx];
    if (to < g.idx) g.last_up = now;
    g.idx = to;
    g.last_change = now;
    g.clean = g.dirty = 0;
    g_grid.store(RungInterval(g.idx));
    g_epoch.fetch_add(1);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s%d -> %d (%s)", g.steps.empty() ? "" : "; ", from,
                  g.rungs[g.idx], why);
    g.steps += buf;
}

void EndWindow(double now) {
    auto& w = g.w;
    double rm, rsd, rp99, sm, ssd, sp99;
    {
        std::scoped_lock lk{g_shown_mu};
        w.shown.swap(g_shown);
        w.shown_count = g_shown_count;
        g_shown.clear();
        g_shown_count = 0;
    }
    Stats(w.real, rm, rsd, rp99);
    Stats(w.shown, sm, ssd, sp99);
    const double work95 = P95(w.work);
    const double secs = std::max(now - w.start, 1e-3);
    const bool decide = !w.idle && w.ticks >= 10;
    if (decide) {
        const bool missy = w.misses >= std::max(2u, w.ticks / 25); // >= 4 %
        const bool heavy = w.misses >= std::max(3u, w.ticks / 10); // >= 10 %
        g.dirty = missy ? g.dirty + 1 : 0;
        const int last = static_cast<int>(g.rungs.size()) - 1;
        if ((heavy || g.dirty >= 2) && g.idx < last) {
            if (now - g.last_up < 10.0) g.up_hold = std::min(g.up_hold * 2, 60); // failed step up
            Change(g.idx + 1, now, heavy ? "missed >= 10%" : "missed >= 4% twice");
        } else if (g.idx > 0) {
            const double up_ms = RungInterval(g.idx - 1) * 1e3;
            const bool clean = w.misses == 0 && !w.work.empty() && work95 <= 0.80 * up_ms;
            g.clean = clean ? g.clean + 1 : 0;
            if (g.clean >= g.up_hold) Change(g.idx - 1, now, "headroom");
        }
        if (now - g.last_change > 60.0) g.up_hold = 5; // long stable stretch: back-off decays
    }
    const double real_fps = w.flips / secs;
    const double presents = (w.shown_count > 0 ? w.shown_count : w.flips) / secs;
    BBLogf("[smooth] target %d fps (x%d = %d shown) | real %.1f fps, frame time mean %.2f sd %.2f "
           "p99 %.2f ms, missed %u of %u | generated/shown interval sd %.2f p99 %.2f ms (planned "
           "grid; DLL measured: [midpoint]), presents/s %.1f | work p95 %.1f ms, up after %d/%d "
           "clean s%s | steps: %s",
           g.rungs[g.idx], g.mult, g.rungs[g.idx] * g.mult, real_fps, rm, rsd, rp99, w.misses,
           w.ticks, ssd, sp99, presents, work95, g.clean, g.up_hold, w.idle ? " | idle" : "",
           g.steps.empty() ? "none" : g.steps.c_str());
    g.steps.clear();
    w.Clear(now);
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_SMOOTH");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double VblankInterval(double configured_hz, int flip_rate) {
    if (!Enabled()) return 0.0;
    const char* why = nullptr;
    // Without the 60 FPS patch the game times itself in vblanks (30 = a flip every 2 at 60):
    // changing the vblank rate would change its speed. The launcher passes BB_FPS_CAP=30 then.
    if (const char* cap = std::getenv("BB_FPS_CAP"); cap != nullptr && std::atoi(cap) == 30) {
        why = "BB_FPS_CAP=30 (original 30 FPS, no 60 FPS patch): the game's speed follows the "
              "vblank, not changed";
    } else if (flip_rate != 0) {
        why = "the game flips every few vblanks (sceVideoOutSetFlipRate != 0): not changed";
    }
    if (why != nullptr) {
        if (g.governing || !g.logged_off) {
            BBLogf("[smooth] Fluidez adaptativa not governing: %s", why);
            g.logged_off = true;
        }
        g.governing = false;
        g_governing.store(false);
        g_grid.store(0.0);
        return 0.0;
    }
    (void)configured_hz;
    if (!g.governing) {
        g.governing = true;
        g.logged_off = false;
        BuildRungs();
        g.idx = 0;
        g_grid.store(RungInterval(g.idx));
        g_epoch.fetch_add(1);
        g.w.Clear(Now());
        {
            std::scoped_lock lk{g_shown_mu};
            g_shown.clear();
            g_shown_count = 0;
        }
        g_governing.store(true);
    }
    return RungInterval(g.idx);
}

void WaitUntil(double deadline_s) {
    double left = deadline_s - Now();
    if (left <= 0.0) return;
#if defined(_WIN32)
    thread_local HANDLE timer = [] {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
        if (h == nullptr) h = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        return h;
    }();
    // Sleep to ~1 ms before the deadline (high resolution timer: ~0.5 ms wake-up error), spin
    // the rest.
    if (left > 0.0012 && timer != nullptr) {
        LARGE_INTEGER due{};
        due.QuadPart = -static_cast<LONGLONG>((left - 0.001) * 1e7);
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(timer, INFINITE);
        }
    }
    while (Now() < deadline_s) YieldProcessor();
#else
    if (left > 0.0012) {
        std::this_thread::sleep_for(std::chrono::duration<double>(left - 0.001));
    }
    while (Now() < deadline_s) std::this_thread::yield();
#endif
}

void OnTick(double tick_s, bool took_flip, double submit_s, double present_s) {
    if (!g.governing) return;
    auto& w = g.w;
    ++w.ticks;
    const double interval = RungInterval(g.idx);
    if (took_flip) {
        ++w.flips;
        if (g.last_present > 0.0) {
            const double dt = present_s - g.last_present;
            if (dt > 0.0 && dt < 0.25) w.real.push_back(dt * 1e3);
        }
        g.last_present = present_s;
        g.last_flip_tick = tick_s;
        if (submit_s > 0.0 && submit_s <= tick_s && tick_s - submit_s < 0.25) {
            const double slack = std::min(tick_s - submit_s, interval);
            w.work.push_back((interval - slack) * 1e3);
        }
    } else if (g.last_flip_tick > 0.0 && tick_s - g.last_flip_tick > 0.25) {
        w.idle = true; // loading / pause: no decisions this second
    } else {
        ++w.misses;
    }
    g.last_tick = tick_s;
    if (tick_s - w.start >= 1.0) {
        BuildRungs();
        EndWindow(tick_s);
    }
}

uint32_t Epoch() {
    return g_epoch.load(std::memory_order_relaxed);
}

double GridInterval() {
    return Enabled() ? g_grid.load(std::memory_order_relaxed) : 0.0;
}

void NoteShown(double ms) {
    if (!g_governing.load(std::memory_order_relaxed)) return;
    std::scoped_lock lk{g_shown_mu};
    ++g_shown_count;
    g_shown.push_back(ms);
}

} // namespace BB::Smooth
