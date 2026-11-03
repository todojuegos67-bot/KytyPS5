// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/bb/bb_metrics.h"

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
