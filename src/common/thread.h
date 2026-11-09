// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include <chrono>
#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

void SetCurrentThreadRealtime(std::chrono::nanoseconds period_ns);

void SetCurrentThreadPriority(ThreadPriority new_priority);

void SetCurrentThreadName(const char* name);

void SetThreadName(void* thread, const char* name);

// bb-dlss5perf-106: named threads are registered (Windows) so a "[perf]" line can report each
// one's CPU busy % per second. Returns e.g. "GpuCommandProcessor 92%, PresentThread 12%, ..."
// (the busiest few since the previous call); empty off Windows.
std::string BBThreadBusyReport(double wall_seconds, int top = 5);

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

class AccurateTimer {
    std::chrono::nanoseconds target_interval{};
    std::chrono::nanoseconds total_wait{};

    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit AccurateTimer(std::chrono::nanoseconds target_interval);

    void Start();

    // Start() with `wait(duration)` doing the sleep (it returns at the end of the duration; it may
    // do other work meanwhile, as the VideoOut present thread presenting a flip as it arrives).
    template <typename Wait>
    void Start(Wait&& wait) {
        const auto begin_sleep = std::chrono::high_resolution_clock::now();
        if (total_wait.count() > 0) {
            wait(total_wait);
        }
        start_time = std::chrono::high_resolution_clock::now();
        total_wait -=
            std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
    }

    void End();

    std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }
};

std::string GetCurrentThreadName();

} // namespace Common
