// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// bb-perf-2: cheap global counters for the 10 s "[perf2]" line (bb_upscale.cpp NoteFrame).
// Relaxed atomics only; each counter is read and reset once every 10 s.

#pragma once

#include <atomic>
#include <chrono>
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
    // bb-native-hot
    std::atomic<u64> fault_around_pages{}; // pages unprotected ahead of a write fault (batched)
    std::atomic<u64> wait_stream_us{};     // of wait_us: staging stream buffer wrap waits
    std::atomic<u64> wait_faultbuf_us{};   // of wait_us: GPU page-fault buffer readback waits
    // bb-native-texhash / GPU-thread starvation split
    std::atomic<u64> tex_skips{};      // image refreshes skipped: guest bytes = last upload
    std::atomic<u64> tex_skip_bytes{}; // their guest size
    std::atomic<u64> tex_gpu_src{};    // of tex_uploads: refreshed from GPU-written buffer data
    std::atomic<u64> cp_busy_us{};     // GpuCommandProcessor thread: wake -> idle time
    std::atomic<u64> cp_guest_wait_us{}; // of cp_busy_us: GFX WAIT_REG_MEM on game memory
    std::atomic<u64> cp_vo_wait_us{};    // of cp_busy_us: GFX wait on a flip label (sleeps)
    // bb-spikes: guest file reads (whole ReadFile call) and host commits (HostFile::Flush)
    std::atomic<u64> file_reads{};
    std::atomic<u64> file_read_us{};
    std::atomic<u64> file_read_bytes{}; // requested bytes
    std::atomic<u64> commit_calls{};
    std::atomic<u64> commit_us{};
};

inline Counters& Get() {
    static Counters c;
    return c;
}

inline void Add(std::atomic<u64>& c, u64 v) {
    c.fetch_add(v, std::memory_order_relaxed);
}

inline u64 MicrosSince(std::chrono::steady_clock::time_point started) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count());
}

// BB_PERF2=1 enables the behavior changes of bb-perf-2 (default off). Counters/logging are always on.
inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_PERF2");
        return v && v[0] == '1';
    }();
    return on;
}

// bb-native-hot: BB_NATIVE_<X>=0 turns one native fast path off (default on), for bisecting.
inline bool NativeOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v && v[0] == '0' && v[1] == '\0');
}

} // namespace BBPerf2
