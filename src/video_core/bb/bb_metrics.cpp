// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/bb/bb_metrics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
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

int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}
} // namespace

bool Enabled() {
    return Open() != nullptr;
}

void OnInput() {
    if (!Enabled()) {
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

void OnPresent() {
    std::FILE* file = Open();
    if (file == nullptr) {
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
    std::fprintf(file, "[fps] %zu frames: mean %.2f ms, max %.2f ms\n", g_frames.size(),
                 sum / g_frames.size(), max);
    if (!g_lat.empty()) {
        std::sort(g_lat.begin(), g_lat.end());
        double lsum = 0;
        for (double l : g_lat) lsum += l;
        const double p95 = g_lat[std::min(g_lat.size() - 1, (g_lat.size() * 95) / 100)];
        std::fprintf(file, "[lat] input->present mean %.1f ms, p95 %.1f ms\n", lsum / g_lat.size(),
                     p95);
    }
    std::fprintf(file, "%s\n", g_up.c_str());
    std::fflush(file);
    g_frames.clear();
    g_lat.clear();
    g_window_start = now;
}

} // namespace BB::Metrics
