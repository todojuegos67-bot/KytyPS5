// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// bb-perf-2: cheap global counters for the 10 s "[perf2]" line (bb_upscale.cpp NoteFrame).
// Relaxed atomics only; each counter is read and reset once every 10 s.

#pragma once

#include <atomic>
#include <cstdlib>
#include "common/types.h"

namespace BBPerf2 {

struct Counters {
    std::atomic<u64> gfx_pipelines{};   // graphics pipelines compiled (cache miss)
    std::atomic<u64> cs_pipelines{};    // compute pipelines compiled
    std::atomic<u64> shaders{};         // shader modules compiled (SPIR-V recompiles)
    std::atomic<u64> compile_us{};      // time spent creating pipelines on the GPU thread
    std::atomic<u64> cpu_writes{};      // CPU write faults on GPU-tracked memory (InvalidateMemory)
    std::atomic<u64> buf_up_bytes{};    // buffer cache uploads (guest -> GPU)
    std::atomic<u64> buf_down_bytes{};  // buffer cache readbacks (GPU -> guest)
    std::atomic<u64> tex_up_bytes{};    // texture cache uploads (incl. detile input)
    std::atomic<u64> tex_uploads{};     // images refreshed from guest memory
    std::atomic<u64> sparse_binds{};    // sparse ranges made resident
    std::atomic<u64> waits{};           // Scheduler::Wait calls that blocked
    std::atomic<u64> wait_us{};         // time blocked in Scheduler::Wait
};

inline Counters& Get() {
    static Counters c;
    return c;
}

inline void Add(std::atomic<u64>& c, u64 v) {
    c.fetch_add(v, std::memory_order_relaxed);
}

// BB_PERF2=1 enables the behavior changes of bb-perf-2 (default off). Counters/logging are always on.
inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_PERF2");
        return v && v[0] == '1';
    }();
    return on;
}

} // namespace BBPerf2
