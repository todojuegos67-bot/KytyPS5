// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// bb-perf-2: cheap global counters for the 10 s "[perf2]" line (bb_upscale.cpp NoteFrame).
// Relaxed atomics only; each counter is read and reset once every 10 s.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include "common/thread.h"
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
    // bb-spikes phases: guest thread blocked in sceGnm submit/flip until the GPU thread took the
    // previous submission (WaitGpuIdle), guest write-fault handler time (and the longest one since
    // the last flip), the GPU thread's end-of-submit work (OnSubmit + Flush: Vulkan submit) and
    // the presenter (Present: acquire, compose, queue present).
    std::atomic<u64> gnm_wait_us{};
    std::atomic<u64> fault_us{};
    std::atomic<u64> fault_max_us{};
    std::atomic<u64> cp_submit_us{};
    std::atomic<u64> present_us{};
    // bb-perf-medir: GPU command thread cost per call (Rasterizer draw/dispatch, image refresh).
    std::atomic<u64> draws{};
    std::atomic<u64> draw_us{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> dispatch_us{};
    std::atomic<u64> refreshes{};
    std::atomic<u64> refresh_us{};
};

inline Counters& Get() {
    static Counters c;
    return c;
}

inline void Add(std::atomic<u64>& c, u64 v) {
    c.fetch_add(v, std::memory_order_relaxed);
}

inline void Max(std::atomic<u64>& c, u64 v) {
    u64 cur = c.load(std::memory_order_relaxed);
    while (v > cur && !c.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

inline u64 MicrosSince(std::chrono::steady_clock::time_point started) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count());
}

// bb-perf-medir: counts one call and its time (us) when it goes out of scope.
struct ScopeUs {
    std::atomic<u64>& n;
    std::atomic<u64>& us;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~ScopeUs() {
        n.fetch_add(1, std::memory_order_relaxed);
        us.fetch_add(MicrosSince(start), std::memory_order_relaxed);
    }
};

// Seconds between "[perf2]" lines: BB_PERF_SEC (1..60), default 5.
inline double PeriodSec() {
    static const double sec = [] {
        const char* v = std::getenv("BB_PERF_SEC");
        const int n = v ? std::atoi(v) : 0;
        return n >= 1 && n <= 60 ? double(n) : 5.0;
    }();
    return sec;
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

// bb-spikes: time each guest thread spends blocked in the HLE waits / sleeps / file calls below,
// per thread (the [spike] line names the threads and calls that held a long frame).
enum class Wait : u32 {
    Equeue,
    Sema,
    PosixSem,
    EventFlag,
    CondWait,
    MutexLock,
    Sleep,
    Join,
    AudioOut,
    FileOpen,
    FileStat,
    Count,
};
inline constexpr std::array<const char*, static_cast<u32>(Wait::Count)> kWaitNames{
    "sceKernelWaitEqueue",
    "sceKernelWaitSema",
    "sem_wait",
    "sceKernelWaitEventFlag",
    "pthread_cond_wait",
    "pthread_mutex_lock",
    "sleep",
    "pthread_join",
    "sceAudioOutOutput",
    "open",
    "stat",
};
inline constexpr u32 kWaitThreads = 64;
inline constexpr u32 kWaitKinds = static_cast<u32>(Wait::Count);

struct WaitTable {
    std::array<std::array<std::atomic<u64>, kWaitKinds>, kWaitThreads> us{};
    std::array<std::string, kWaitThreads> names{};
    std::atomic<u32> threads{0};
    std::mutex names_mutex;
};

inline WaitTable& Waits() {
    static WaitTable t;
    return t;
}

// This thread's row (kWaitThreads or more: not recorded).
inline u32 WaitThreadSlot() {
    thread_local const u32 slot = [] {
        auto& t = Waits();
        const u32 i = t.threads.fetch_add(1, std::memory_order_relaxed);
        if (i < kWaitThreads) {
            std::scoped_lock lk{t.names_mutex};
            t.names[i] = Common::GetCurrentThreadName();
        }
        return i;
    }();
    return slot;
}

class WaitScope {
public:
    explicit WaitScope(Wait kind_) : kind{kind_}, started{std::chrono::steady_clock::now()} {}
    ~WaitScope() {
        const u32 slot = WaitThreadSlot();
        if (slot < kWaitThreads) {
            Add(Waits().us[slot][static_cast<u32>(kind)], MicrosSince(started));
        }
    }
    WaitScope(const WaitScope&) = delete;
    WaitScope& operator=(const WaitScope&) = delete;

private:
    Wait kind;
    std::chrono::steady_clock::time_point started;
};

} // namespace BBPerf2
