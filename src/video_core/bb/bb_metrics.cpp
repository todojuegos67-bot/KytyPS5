// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/bb/bb_metrics.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

namespace BB::Metrics {
namespace {
using Clock = std::chrono::steady_clock;

std::FILE* Open() {
    static std::FILE* file = [] () -> std::FILE* {
        const char* path = std::getenv("BB_METRICS_OUT");
        if (path == nullptr || *path == '\0') {
            return nullptr;
        }
        std::error_code ec;
        const std::filesystem::path p{path};
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        if (std::filesystem::exists(p, ec)) {
            std::filesystem::path old = p;
            old += ".1";
            std::filesystem::remove(old, ec);
            std::filesystem::rename(p, old, ec);
        }
        return std::fopen(path, "ab");
    }();
    return file;
}

std::mutex g_mutex;
std::atomic<int64_t> g_input_ns{0}; // first pad read since the last present (0: none)
Clock::time_point g_last_present{};
Clock::time_point g_window_start{};
std::vector<double> g_frames, g_lat;
std::string g_up = "[up] off";
std::string g_res;

// (bb-metrics-043) The panel's state.
std::atomic<bool> g_panel{false};
Snapshot g_snap;
Clock::time_point g_win_at{}, g_fg_at{}, g_lat_at{};

double Age(Clock::time_point t) {
    if (t.time_since_epoch().count() == 0) return 1e9;
    return std::chrono::duration<double>(Clock::now() - t).count();
}

bool Collect() {
    return g_panel.load(std::memory_order_relaxed);
}

// The Demon's Souls panel's log patterns (ds-menu.cpp Sample), on BB::FG::Printf's lines.
void Parse(const char* text) {
    if (text[0] != '[') return;
    const bool dlssg = std::strncmp(text, "[dlssg]", 7) == 0;
    const bool fsr3 = std::strncmp(text, "[fsr3]", 6) == 0;
    const bool mfg = std::strncmp(text, "[fsr3-mfg]", 10) == 0;
    const bool lat = std::strncmp(text, "[lat] input->present", 20) == 0;
    if (!dlssg && !fsr3 && !mfg && !lat) return;
    static const std::regex re_rs(R"(real ([0-9.]+) fps(?: \(free [0-9.]+\))?, shown ([0-9.]+))");
    static const std::regex re_x(R"(^\[(?:dlssg|fsr3-mfg)\] (?:fixed |dynamic )?x(\d))");
    static const std::regex re_tot(R"(total ([0-9.]+), p95 ([0-9.]+))");
    static const std::regex re_mode(R"(mode ([a-z0-9-]+))");
    std::cmatch m;
    std::scoped_lock lk{g_mutex};
    if (lat) {
        if (std::regex_search(text, m, re_tot)) {
            g_snap.lat_mean = std::strtod(m[1].first, nullptr);
            g_snap.lat_p95 = std::strtod(m[2].first, nullptr);
            g_lat_at = Clock::now();
        }
        if (std::regex_search(text, m, re_mode)) g_snap.lat_mode = m[1].str();
        return;
    }
    if (std::regex_search(text, m, re_x)) g_snap.fg_mult = std::atoi(m[1].first);
    if (std::regex_search(text, m, re_rs)) {
        g_snap.fg_real = std::strtod(m[1].first, nullptr);
        g_snap.fg_shown = std::strtod(m[2].first, nullptr);
        g_fg_at = Clock::now();
    }
}

int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}
} // namespace

bool Enabled() {
    return Open() != nullptr;
}

void OnInput() {
    if (!Collect() && !Enabled()) {
        return;
    }
    int64_t expected = 0;
    g_input_ns.compare_exchange_strong(expected, Now());
}

void SetUpscaleLine(const std::string& text) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    g_up = "[up] " + text;
}

void SetResLine(const std::string& text) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    g_res = "[res] " + text;
}

void Line(const char* text) {
    if (text != nullptr && Collect()) {
        Parse(text);
    }
    std::FILE* file = Open();
    if (file == nullptr || text == nullptr) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    std::fprintf(file, "%s\n", text);
    std::fflush(file);
}

void OnPresent() {
    std::FILE* file = Open();
    if (file == nullptr && !Collect()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    const auto now = Clock::now();
    if (g_last_present.time_since_epoch().count() != 0) {
        g_frames.push_back(std::chrono::duration<double, std::milli>(now - g_last_present).count());
    } else {
        g_window_start = now;
    }
    g_last_present = now;
    if (const int64_t input = g_input_ns.exchange(0); input != 0) {
        g_lat.push_back(static_cast<double>(Now() - input) / 1e6);
    }
    if (now - g_window_start < std::chrono::seconds(1) || g_frames.empty()) {
        return;
    }
    double sum = 0, max = 0;
    for (double f : g_frames) {
        sum += f;
        max = std::max(max, f);
    }
    g_snap.mean_ms = sum / g_frames.size();
    g_snap.max_ms = max;
    g_snap.key++;
    g_win_at = now;
    if (file != nullptr) {
        std::fprintf(file, "[fps] %zu frames: mean %.2f ms, max %.2f ms\n", g_frames.size(),
                     sum / g_frames.size(), max);
    }
    if (!g_lat.empty()) {
        std::sort(g_lat.begin(), g_lat.end());
        double lsum = 0;
        for (double l : g_lat) lsum += l;
        const double p95 = g_lat[std::min(g_lat.size() - 1, (g_lat.size() * 95) / 100)];
        g_snap.own_lat_mean = lsum / g_lat.size();
        g_snap.own_lat_p95 = p95;
        if (file != nullptr) {
            std::fprintf(file, "[lat] input->present mean %.1f ms, p95 %.1f ms\n",
                         lsum / g_lat.size(), p95);
        }
    } else {
        g_snap.own_lat_mean = g_snap.own_lat_p95 = 0;
    }
    if (file != nullptr) {
        std::fprintf(file, "%s\n", g_up.c_str());
        if (!g_res.empty()) std::fprintf(file, "%s\n", g_res.c_str());
        std::fflush(file);
    }
    g_frames.clear();
    g_lat.clear();
    g_window_start = now;
}

bool PanelShown() {
    return g_panel.load(std::memory_order_relaxed);
}

void SetPanelShown(bool shown) {
    g_panel.store(shown, std::memory_order_relaxed);
}

void SetUpscale(const Upscale& up) {
    if (!Collect()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    g_snap.up = up;
    g_snap.has_up = true;
}

void SetDlss5State(int state, const char* detail) {
    std::scoped_lock lk{g_mutex};
    g_snap.dlss5_state = state;
    g_snap.dlss5_detail = detail != nullptr ? detail : "";
}

Snapshot Get() {
    std::scoped_lock lk{g_mutex};
    Snapshot s = g_snap;
    s.window_age_s = Age(g_win_at);
    s.fg_age_s = Age(g_fg_at);
    s.lat_age_s = Age(g_lat_at);
    return s;
}

} // namespace BB::Metrics

// ---- bb-dlss5perf-106: "[perf]" once a second ----
namespace BB::Metrics {
namespace {
std::atomic<uint64_t> g_p_d5_n{0}, g_p_d5_cpu_us{0}, g_p_d5_gpu_n{0}, g_p_d5_gpu_us{0};
std::atomic<uint64_t> g_p_wait_n{0}, g_p_wait_us{0};
} // namespace
std::atomic<uint64_t> g_cp_ns[CpCount]{};
std::atomic<uint64_t> g_cp_draws{0}, g_cp_dispatches{0};
std::atomic<uint64_t> g_cp_patch_scans{0}, g_cp_patch_bytes{0}, g_cp_patch_fast{0},
    g_cp_patch_hits{0};
static bool EnvOn(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}
bool CpFastPatch() {
    static const bool on = EnvOn("BB_CP_FASTPATCH", true);
    return on;
}
bool CpDetail() {
    static const bool on = EnvOn("BB_CP_DETAIL", true);
    return on;
}
void PerfDlss5(double cpu_ms) {
    g_p_d5_n.fetch_add(1, std::memory_order_relaxed);
    g_p_d5_cpu_us.fetch_add(static_cast<uint64_t>(cpu_ms * 1000.0), std::memory_order_relaxed);
}
void PerfDlss5Gpu(double gpu_ms) {
    g_p_d5_gpu_n.fetch_add(1, std::memory_order_relaxed);
    g_p_d5_gpu_us.fetch_add(static_cast<uint64_t>(gpu_ms * 1000.0), std::memory_order_relaxed);
}
void PerfPresentWait(double ms) {
    if (ms < 0.05) return;
    g_p_wait_n.fetch_add(1, std::memory_order_relaxed);
    g_p_wait_us.fetch_add(static_cast<uint64_t>(ms * 1000.0), std::memory_order_relaxed);
}
} // namespace BB::Metrics

namespace AmdGpu { // liverpool.cpp
extern std::atomic<uint64_t> g_bb_cp_backoffs;
extern std::atomic<uint64_t> g_bb_cp_wait_yields;
} // namespace AmdGpu

namespace BB::Metrics {
void PerfLine(int frames_in_flight, int frames_total) {
    static const bool off = [] {
        const char* v = std::getenv("BB_PERF");
        return v != nullptr && v[0] == '0';
    }();
    if (off) return;
    using C = std::chrono::steady_clock;
    static C::time_point last{};
    static uint64_t frames = 0, last_backoffs = 0, last_yields = 0;
    static int max_inflight = 0;
    ++frames;
    max_inflight = std::max(max_inflight, frames_in_flight);
    const auto now = C::now();
    if (last.time_since_epoch().count() == 0) {
        last = now;
        (void)Common::BBThreadBusyReport(1.0);
        return;
    }
    const double wall = std::chrono::duration<double>(now - last).count();
    if (wall < 1.0) return;
    last = now;
    const uint64_t d5n = g_p_d5_n.exchange(0), d5cpu = g_p_d5_cpu_us.exchange(0);
    const uint64_t d5gn = g_p_d5_gpu_n.exchange(0), d5gpu = g_p_d5_gpu_us.exchange(0);
    const uint64_t wn = g_p_wait_n.exchange(0), wus = g_p_wait_us.exchange(0);
    const uint64_t bo = AmdGpu::g_bb_cp_backoffs.load(std::memory_order_relaxed);
    const uint64_t yi = AmdGpu::g_bb_cp_wait_yields.load(std::memory_order_relaxed);
    const std::string threads = Common::BBThreadBusyReport(wall, 6);
    char d5[160] = "dlss5 off";
    if (d5n > 0) {
        std::snprintf(d5, sizeof(d5), "dlss5 %llu evals, CPU %.2f ms/eval, GPU %s, CPU waits 0",
                      static_cast<unsigned long long>(d5n), d5cpu / 1000.0 / d5n,
                      d5gn ? fmt::format("{:.2f} ms ({:.0f}% of the second)", d5gpu / 1000.0 / d5gn,
                                         d5gpu / 1000.0 / d5gn * d5n / (wall * 10.0)).c_str()
                           : "n/a (sampled 1 in 16)");
    }
    double cp[CpCount];
    for (int i = 0; i < CpCount; ++i) cp[i] = g_cp_ns[i].exchange(0) / 1e6 / wall; // ms per second
    const uint64_t cdr = g_cp_draws.exchange(0), cdi = g_cp_dispatches.exchange(0);
    const double nd = static_cast<double>(std::max<uint64_t>(1, cdr + cdi));
    const std::string cpline = fmt::format(
        "cp ms/s: draw+dispatch {:.0f} ({:.0f} draws, {:.0f} dispatches, {:.1f} us each) = pipeline {:.0f}, "
        "bind {:.0f} (buffer cache {:.0f}, texture cache {:.0f}, camera scan {:.0f}), obj motion {:.0f}, "
        "record {:.0f}, other {:.0f}",
        cp[CpDraw], cdr / wall, cdi / wall, cp[CpDraw] * 1000.0 * wall / nd, cp[CpPipe], cp[CpBind],
        cp[CpBuf], cp[CpTex], cp[CpScan], cp[CpObjMv], cp[CpRec],
        std::max(0.0, cp[CpDraw] - cp[CpPipe] - cp[CpBind] - cp[CpObjMv] - cp[CpRec]));
    LOG_INFO(Render_Vulkan, "[perf] {}", cpline);
    {
        // bb-perf-132: where "other" and the unattributed part of "bind" go.
        const uint64_t ps = g_cp_patch_scans.exchange(0), pb = g_cp_patch_bytes.exchange(0);
        const uint64_t pf = g_cp_patch_fast.exchange(0), ph = g_cp_patch_hits.exchange(0);
        static bool said = false;
        if (!said) {
            said = true;
            LOG_INFO(Render_Vulkan,
                     "[perf] bb-perf-132 CP options: BB_CP_FASTPATCH={} BB_CP_DETAIL={}",
                     CpFastPatch() ? 1 : 0, CpDetail() ? 1 : 0);
        }
        LOG_INFO(Render_Vulkan,
                 "[perf] cp2 ms/s: render targets {:.0f}, const patch {:.0f} ({:.0f} buffers, "
                 "{:.1f} MB, {:.0f} fast-rejected, {:.0f} patched, fast {}), bb sample hooks {:.0f}, "
                 "samplers {:.0f}, bind rest {:.0f}",
                 cp[CpPrep], cp[CpPatch], ps / wall, pb / wall / 1e6, pf / wall, ph / wall,
                 CpFastPatch() ? "on" : "off", cp[CpHook], cp[CpSamp],
                 std::max(0.0, cp[CpBind] - cp[CpBuf] - cp[CpTex] - cp[CpScan] - cp[CpPatch] -
                                   cp[CpHook] - cp[CpSamp]));
    }
    LOG_INFO(Render_Vulkan,
             "[perf] {:.0f} fps | threads: {} | GPU cmd processor: {} wait polls, {} sleeps | "
             "present waits {} ({:.1f} ms) | frames in flight {} (max {} of {}) | {}",
             frames / wall, threads.empty() ? "n/a" : threads, yi - last_yields, bo - last_backoffs,
             wn, wus / 1000.0, frames_in_flight, max_inflight, frames_total, d5);
    frames = 0;
    max_inflight = 0;
    last_backoffs = bo;
    last_yields = yi;
}
} // namespace BB::Metrics
