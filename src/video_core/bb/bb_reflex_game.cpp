// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex-game: Reflex markers / sleep from the game's frame loop (see bb_reflex_game.h).

#include "video_core/bb/bb_reflex_game.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_reflex.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <sys/mman.h>
#endif

namespace BB::ReflexGame {
namespace {

// PCL marker values (sl::PCLMarker, sl_pcl.h v2.x).
constexpr uint32_t kSimulationStart = 0, kSimulationEnd = 1, kRenderSubmitStart = 2;

// Eboot 1.09: FrameStep 0x2018D20, the task manager run at 0x2018E6F (see bb_reflex_game.h).
constexpr uint64_t kSiteVa = 0x2018E6B;
constexpr uint8_t kSiteBytes[16] = {
    0x48, 0x8D, 0x75, 0xC0,                  // lea rsi,[rbp-0x40] (the task: {vptr, dt})
    0xE8, 0x2C, 0x84, 0x03, 0x00,            // call 0x20512A0 (task manager run)
    0x48, 0x8B, 0x3D, 0xB5, 0x6B, 0x68, 0x03 // mov rdi,[rip+0x3686BB5] = [0x569FA30]
};
constexpr uint64_t kTaskRunVa = 0x20512A0;
constexpr uint64_t kPostObjVa = 0x569FA30;
constexpr uint64_t kSiteBackVa = kSiteVa + sizeof(kSiteBytes); // 0x2018E7B: test rdi,rdi
// FrameStep's prologue (0x2018D20) and the limiter call before the site (0x2018E1F..0x2018E26).
constexpr uint64_t kFrameStepVa = 0x2018D20;
constexpr uint8_t kFrameStepBytes[] = {0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                       0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xEC, 0x18};
constexpr uint64_t kLimiterCallVa = 0x2018E1F;
constexpr uint8_t kLimiterCallBytes[] = {0x48, 0x89, 0xDF, 0xE8, 0x49, 0xB9, 0x01, 0x00};

constexpr auto kSleepTimeout = std::chrono::milliseconds(50);
constexpr uint32_t kMaxSleepTimeouts = 3;

struct Settings {
    bool requested = false;
    bool sleep = true; // BB_REFLEX_GAME_SLEEP=0: markers only
};

const Settings& Cfg() {
    static const Settings s = [] {
        Settings r;
        const char* v = std::getenv("BB_REFLEX_GAME");
        if (v == nullptr || std::strcmp(v, "1") != 0) {
            return r; // the default: silent, nothing installed
        }
        if (BB::Reflex::NativeEnv()) {
            BB::Upscale::Checkpoint("[reflex-game] BB_REFLEX_GAME ignored: BB_REFLEX_NATIVE is set "
                                    "(it owns the frame limiter's wait)");
            return r;
        }
        if (!BB::FG::DlssSl::Requested()) {
            BB::Upscale::Checkpoint("[reflex-game] BB_REFLEX_GAME ignored: it needs DLSS-G through "
                                    "Streamline (BB_FG=dlss-sl*, or dlss-* with BB_DLSSG_SL=1)");
            return r;
        }
        const char* s = std::getenv("BB_REFLEX_GAME_SLEEP");
        r.sleep = !(s != nullptr && s[0] == '0');
        r.requested = true;
        return r;
    }();
    return s;
}

std::atomic<bool> g_installed{false};
std::atomic<bool> g_sleep_off{false};

// ---- Frames (g_frames_mutex) ----
struct SimFrame {
    uint32_t id = 0;
    bool sim_ended = false;
    bool rss_sent = false;       // eRenderSubmitStart posted
    bool submit_pending = false; // the render thread submitted before eSimulationEnd
};
std::mutex g_frames_mutex;
std::deque<SimFrame> g_sim; // started by the game, not flipped yet
bool g_render_open = false; // the render thread's current frame has its eRenderSubmitStart
std::deque<uint32_t> g_gpu; // flipped, not prepared by the GPU thread yet (0: no game frame)
std::atomic<bool> g_render_open_fast{false};
std::atomic<uint32_t> g_cur_sim{0}; // main thread: the frame between FrameStart and SimEnd
std::atomic<std::thread::id> g_main_thread{};
// GPU / present threads.
std::atomic<uint32_t> g_gpu_current{0};
std::atomic<uint32_t> g_max_used{0}; // the highest index TagFrame / a present used

// ---- Per-second counters ----
std::atomic<uint32_t> c_sim_start{0}, c_sim_end{0}, c_rss{0}, c_marker_err{0}, c_sleeps{0},
    c_sleep_fail{0}, c_sleep_timeouts{0}, c_sleep_busy{0}, c_flips_game{0}, c_flips_none{0},
    c_stale{0}, c_tag_game{0}, c_tag_own{0}, c_inactive{0}, c_dropped{0};
std::atomic<uint64_t> c_sleep_us{0};
std::atomic<uint32_t> g_timeouts_total{0};

// ---- shadPS4:ReflexGameMarks: every game marker, in the order posted ----
struct Job {
    uint32_t marker = 0;
    uint32_t frame = 0;
};
std::mutex g_jobs_mutex;
std::condition_variable g_jobs_cv, g_jobs_sent_cv;
std::deque<Job> g_jobs;
uint64_t g_jobs_posted = 0, g_jobs_sent = 0;

// ---- shadPS4:ReflexGameSleep ----
std::mutex g_sleep_mutex;
std::condition_variable g_sleep_cv, g_sleep_done_cv;
uint64_t g_sleep_ticket = 0, g_sleep_taken = 0, g_sleep_done = 0;
uint32_t g_sleep_frame = 0;
bool g_sleep_busy = false;
std::atomic<bool> g_workers{false};

void PostMarker(uint32_t marker, uint32_t frame) {
    if (frame == 0)
        return;
    {
        std::scoped_lock lock{g_jobs_mutex};
        if (g_jobs.size() >= 256) {
            c_dropped.fetch_add(1, std::memory_order_relaxed);
            return; // the worker is stuck: a marker is not worth a wait
        }
        g_jobs.push_back(Job{marker, frame});
        ++g_jobs_posted;
    }
    g_jobs_cv.notify_one();
}

void LogSecond() {
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(1))
        return;
    const double secs = std::chrono::duration<double>(now - last).count();
    last = now;
    const uint32_t sleeps = c_sleeps.exchange(0);
    const uint64_t us = c_sleep_us.exchange(0);
    const uint32_t s0 = c_sim_start.exchange(0), s1 = c_sim_end.exchange(0), s2 = c_rss.exchange(0),
                   err = c_marker_err.exchange(0), dropped = c_dropped.exchange(0),
                   fail = c_sleep_fail.exchange(0), timeouts = c_sleep_timeouts.exchange(0),
                   busy = c_sleep_busy.exchange(0), fg = c_flips_game.exchange(0),
                   fn = c_flips_none.exchange(0), tg = c_tag_game.exchange(0),
                   to = c_tag_own.exchange(0), stale = c_stale.exchange(0),
                   inactive = c_inactive.exchange(0);
    if (s0 + s1 + s2 + err + dropped + sleeps + fail + timeouts + busy + fg + fn + tg + to +
            stale ==
        0) {
        return; // nothing to tell (DLSS-G not active yet / any more): no line
    }
    LOG_INFO(Lib_VideoOut,
             "[reflex-game] {:.2f} s: markers simStart {} simEnd {} renderSubmitStart {} (errors "
             "{}, dropped {}) | slReflexSleep {} (avg {:.2f} ms, failed {}, timeouts {} [{} "
             "total{}], skipped busy {}) | flips with game frame {} / without {} | GPU frames "
             "tagged on the game index {} / own index {} (stale game index {}) | frame starts "
             "while inactive {}",
             secs, s0, s1, s2, err, dropped, sleeps, sleeps ? double(us) / 1e3 / sleeps : 0.0, fail,
             timeouts, g_timeouts_total.load(), g_sleep_off.load() ? ", sleep off" : "", busy, fg,
             fn, tg, to, stale, inactive);
}

void MarkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ReflexGameMarks");
    for (;;) {
        Job job;
        {
            std::unique_lock lock{g_jobs_mutex};
            if (!g_jobs_cv.wait_for(lock, std::chrono::seconds(1),
                                    [] { return !g_jobs.empty(); })) {
                lock.unlock();
                LogSecond();
                continue;
            }
            job = g_jobs.front();
            g_jobs.pop_front();
        }
        if (BB::FG::DlssSl::GameMarker(job.marker, job.frame)) {
            static bool said = false;
            if (!said && job.marker == kSimulationStart) {
                said = true;
                BB::Upscale::Checkpoint(fmt::format(
                    "[reflex-game] ACTIVE: first eSimulationStart from the game's frame loop "
                    "(frame {}, Reflex sleep {})",
                    job.frame,
                    !Cfg().sleep                           ? "off (BB_REFLEX_GAME_SLEEP=0)"
                    : BB::FG::DlssSl::GameSleepAvailable() ? "slReflexSleep"
                                                           : "unavailable (no slReflexSleep)"));
            }
            (job.marker == kSimulationStart ? c_sim_start
             : job.marker == kSimulationEnd ? c_sim_end
                                            : c_rss)
                .fetch_add(1, std::memory_order_relaxed);
        } else {
            c_marker_err.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{g_jobs_mutex};
            ++g_jobs_sent;
        }
        g_jobs_sent_cv.notify_all();
        LogSecond();
    }
}

void SleepLoop() {
    Common::SetCurrentThreadName("shadPS4:ReflexGameSleep");
    for (;;) {
        uint64_t ticket = 0;
        uint32_t frame = 0;
        {
            std::unique_lock lock{g_sleep_mutex};
            g_sleep_cv.wait(lock, [] { return g_sleep_ticket > g_sleep_taken; });
            ticket = g_sleep_taken = g_sleep_ticket;
            frame = g_sleep_frame;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = BB::FG::DlssSl::GameSleep(frame);
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        if (ok) {
            c_sleeps.fetch_add(1, std::memory_order_relaxed);
            c_sleep_us.fetch_add(uint64_t(std::max<int64_t>(us, 0)), std::memory_order_relaxed);
        } else {
            c_sleep_fail.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{g_sleep_mutex};
            g_sleep_done = ticket;
            g_sleep_busy = false;
        }
        g_sleep_done_cv.notify_all();
    }
}

void StartWorkers() {
    if (g_workers.exchange(true))
        return;
    // Detached: idle they only wait on their condition variables; the emulator ends with
    // std::quick_exit (no static destructors run under them).
    std::thread{MarkerLoop}.detach();
    std::thread{SleepLoop}.detach();
}

// Game main thread: the Reflex sleep for `frame` on shadPS4:ReflexGameSleep, waited for at most
// kSleepTimeout. A sleep still running from an earlier frame: none this frame.
void GuestSleep(uint32_t frame) {
    if (!Cfg().sleep || g_sleep_off.load(std::memory_order_relaxed) ||
        !BB::FG::DlssSl::GameSleepAvailable())
        return;
    std::unique_lock lock{g_sleep_mutex};
    if (g_sleep_busy) {
        c_sleep_busy.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_sleep_busy = true;
    const uint64_t ticket = ++g_sleep_ticket;
    g_sleep_frame = frame;
    g_sleep_cv.notify_one();
    const bool done =
        g_sleep_done_cv.wait_for(lock, kSleepTimeout, [ticket] { return g_sleep_done >= ticket; });
    if (!done) {
        c_sleep_timeouts.fetch_add(1, std::memory_order_relaxed);
        if (g_timeouts_total.fetch_add(1) + 1 >= kMaxSleepTimeouts) {
            g_sleep_off.store(true, std::memory_order_relaxed);
        }
    }
}

// ---- Guest hooks (game main thread, on its own stack: short work, no Streamline call) ----

// Before the task manager run: the limiter's wait is over, the frame's simulation starts.
void FrameStart() {
    g_cur_sim.store(0, std::memory_order_relaxed);
    if (!Active()) {
        c_inactive.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_main_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    const uint32_t id = BB::SR::Dlss::NextFrameIndex();
    if (id == 0)
        return;
    GuestSleep(id);
    {
        std::scoped_lock lock{g_frames_mutex};
        g_sim.push_back(SimFrame{.id = id});
        while (g_sim.size() > 6) {
            g_sim.pop_front(); // never flipped (loading, a frame without a flip)
        }
        PostMarker(kSimulationStart, id);
    }
    g_cur_sim.store(id, std::memory_order_relaxed);
}

// After the task manager run: the frame's simulation (and its draw lists) is done.
void SimEnd() {
    const uint32_t id = g_cur_sim.exchange(0, std::memory_order_relaxed);
    if (id == 0)
        return;
    std::scoped_lock lock{g_frames_mutex};
    for (SimFrame& f : g_sim) {
        if (f.id != id)
            continue;
        if (!f.sim_ended) {
            f.sim_ended = true;
            PostMarker(kSimulationEnd, id);
        }
        if (f.submit_pending && !f.rss_sent) {
            f.rss_sent = true;
            PostMarker(kRenderSubmitStart, id);
        }
        f.submit_pending = false;
        return;
    }
}

void PS4_SYSV_ABI HookFrameStart() {
    FrameStart();
}
void PS4_SYSV_ABI HookSimEnd() {
    SimEnd();
}

// ---- The detour ----
void* AllocExec(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p =
        mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

struct Emitter {
    std::vector<uint8_t> b;
    void Put(std::initializer_list<uint8_t> v) {
        b.insert(b.end(), v);
    }
    void Put64(uint64_t v) {
        for (int i = 0; i < 8; ++i)
            b.push_back(uint8_t(v >> (i * 8)));
    }
    void MovRax(uint64_t v) {
        Put({0x48, 0xB8}); // mov rax, imm64
        Put64(v);
    }
    // Calls host() (sysv, no arguments) with every caller-saved register and xmm0..7 kept.
    void HostCall(uint64_t host) {
        Put({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, 0x53});
        Put({0x48, 0x89, 0xE3});                         // mov rbx, rsp
        Put({0x48, 0x83, 0xE4, 0xF0});                   // and rsp, -16
        Put({0x48, 0x81, 0xEC, 0x80, 0x00, 0x00, 0x00}); // sub rsp, 0x80
        for (uint8_t i = 0; i < 8; ++i) {                // movdqu [rsp + i*16], xmm_i
            if (i == 0)
                Put({0xF3, 0x0F, 0x7F, 0x04, 0x24});
            else
                Put({0xF3, 0x0F, 0x7F, uint8_t(0x44 | (i << 3)), 0x24, uint8_t(i * 16)});
        }
        MovRax(host);
        Put({0xFF, 0xD0}); // call rax
        for (uint8_t i = 0; i < 8; ++i) {
            if (i == 0)
                Put({0xF3, 0x0F, 0x6F, 0x04, 0x24});
            else
                Put({0xF3, 0x0F, 0x6F, uint8_t(0x44 | (i << 3)), 0x24, uint8_t(i * 16)});
        }
        Put({0x48, 0x89, 0xDC}); // mov rsp, rbx
        Put({0x5B, 0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5F, 0x5E, 0x5A, 0x59, 0x58});
    }
};

bool Matches(uintptr_t base, uint64_t size, uint64_t va, const uint8_t* bytes, size_t n) {
    return va + n <= size && std::memcmp(reinterpret_cast<const void*>(base + va), bytes, n) == 0;
}

} // namespace

bool Requested() {
    return Cfg().requested;
}

void OnEbootPatched(uintptr_t eboot_base, uint64_t image_size) {
    if (!Requested())
        return;
    const bool ok =
        Matches(eboot_base, image_size, kSiteVa, kSiteBytes, sizeof(kSiteBytes)) &&
        Matches(eboot_base, image_size, kFrameStepVa, kFrameStepBytes, sizeof(kFrameStepBytes)) &&
        Matches(eboot_base, image_size, kLimiterCallVa, kLimiterCallBytes,
                sizeof(kLimiterCallBytes)) &&
        kPostObjVa + 8 <= image_size;
    if (!ok) {
        BB::Upscale::Checkpoint(
            "[reflex-game] the game's frame step (1.09: 0x2018D20, task run 0x2018E6B) is not the "
            "expected code (other version / patch): no hooks, Reflex markers as before");
        return;
    }
    Emitter e;
    e.HostCall(reinterpret_cast<uint64_t>(&HookFrameStart));
    e.Put({0x48, 0x8D, 0x75, 0xC0}); // lea rsi,[rbp-0x40]
    e.MovRax(eboot_base + kTaskRunVa);
    e.Put({0xFF, 0xD0}); // call rax (rsp as at the original call: 16-byte aligned)
    e.HostCall(reinterpret_cast<uint64_t>(&HookSimEnd));
    e.MovRax(eboot_base + kPostObjVa);
    e.Put({0x48, 0x8B, 0x38});         // mov rdi,[rax]
    e.Put({0xFF, 0x25, 0, 0, 0, 0});   // jmp [rip]
    e.Put64(eboot_base + kSiteBackVa); // 0x2018E7B: test rdi,rdi
    auto* cave = static_cast<uint8_t*>(AllocExec(4096));
    if (cave == nullptr || e.b.size() > 4096) {
        BB::Upscale::Checkpoint("[reflex-game] no executable memory for the hook: Reflex markers "
                                "as before");
        return;
    }
    std::memcpy(cave, e.b.data(), e.b.size());
    StartWorkers();
    uint8_t jmp[16] = {0xFF, 0x25, 0, 0, 0, 0};
    const uint64_t to = reinterpret_cast<uint64_t>(cave);
    std::memcpy(jmp + 6, &to, 8);
    jmp[14] = jmp[15] = 0x90;
    std::memcpy(reinterpret_cast<void*>(eboot_base + kSiteVa), jmp, sizeof(jmp));
    g_installed.store(true, std::memory_order_release);
    BB::Upscale::Checkpoint(fmt::format(
        "[reflex-game] hooks installed: game frame step 0x2018D20, detour 0x2018E6B around the "
        "task manager run (0x20512A0): before it {}PCL eSimulationStart, after it "
        "eSimulationEnd; eRenderSubmitStart at the frame's first guest GPU submit (render thread); "
        "the frame index goes with the flip to DLSS SR / DLSS-G (TagFrame); eRenderSubmitEnd / "
        "ePresentStart / ePresentEnd stay in the presenter. Active once Streamline DLSS-G + PCL "
        "are ready",
        Cfg().sleep ? "slReflexSleep (host thread, game waits <= 50 ms) then " : ""));
}

bool Active() {
    return g_installed.load(std::memory_order_acquire) && BB::FG::DlssSl::GameReady();
}

void OnGuestSubmit() {
    if (!g_installed.load(std::memory_order_relaxed) ||
        g_render_open_fast.load(std::memory_order_relaxed) || !Active())
        return;
    std::scoped_lock lock{g_frames_mutex};
    if (g_render_open || g_sim.empty())
        return;
    g_render_open = true;
    g_render_open_fast.store(true, std::memory_order_relaxed);
    SimFrame& f = g_sim.front(); // the oldest frame started and not flipped
    if (f.sim_ended) {
        if (!f.rss_sent) {
            f.rss_sent = true;
            PostMarker(kRenderSubmitStart, f.id);
        }
    } else {
        f.submit_pending = true; // sent right after its eSimulationEnd
    }
}

void OnGuestFlip() {
    if (!g_installed.load(std::memory_order_relaxed))
        return;
    const bool active = Active();
    const bool on_main =
        g_main_thread.load(std::memory_order_relaxed) == std::this_thread::get_id();
    std::scoped_lock lock{g_frames_mutex};
    g_render_open = false;
    g_render_open_fast.store(false, std::memory_order_relaxed);
    uint32_t id = 0;
    if (active && !g_sim.empty()) {
        SimFrame& f = g_sim.front();
        if (!f.sim_ended && on_main && f.id == g_cur_sim.load(std::memory_order_relaxed)) {
            // The main thread flips its own frame inside the task run: its simulation is over.
            f.sim_ended = true;
            PostMarker(kSimulationEnd, f.id);
            g_cur_sim.store(0, std::memory_order_relaxed);
        }
        if (f.sim_ended) {
            if (!f.rss_sent)
                PostMarker(kRenderSubmitStart, f.id);
            id = f.id;
            g_sim.pop_front();
        } else {
            f.submit_pending = false; // a flip of an earlier frame the hooks did not see
        }
    }
    (id != 0 ? c_flips_game : c_flips_none).fetch_add(1, std::memory_order_relaxed);
    g_gpu.push_back(id);
    while (g_gpu.size() > 4)
        g_gpu.pop_front();
}

void OnGuestFlipDropped() {
    if (!g_installed.load(std::memory_order_relaxed))
        return;
    std::scoped_lock lock{g_frames_mutex};
    if (!g_gpu.empty())
        g_gpu.pop_back();
}

void OnGpuFlipBegin() {
    if (!g_installed.load(std::memory_order_relaxed))
        return;
    uint32_t id = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (!g_gpu.empty()) {
            id = g_gpu.front();
            g_gpu.pop_front();
        }
    }
    if (id != 0 && (id <= g_max_used.load(std::memory_order_relaxed) || !Active())) {
        // A newer index was already used (an untagged present / an own index in between):
        // Streamline must see increasing indices, so this frame takes the old way.
        c_stale.fetch_add(1, std::memory_order_relaxed);
        id = 0;
    }
    g_gpu_current.store(id, std::memory_order_relaxed);
}

uint32_t GpuFrameId() {
    if (!g_installed.load(std::memory_order_relaxed))
        return 0;
    return g_gpu_current.load(std::memory_order_relaxed);
}

void OnTagged(uint32_t frame) {
    if (!g_installed.load(std::memory_order_relaxed) || frame == 0)
        return;
    (frame == g_gpu_current.load(std::memory_order_relaxed) ? c_tag_game : c_tag_own)
        .fetch_add(1, std::memory_order_relaxed);
    uint32_t cur = g_max_used.load(std::memory_order_relaxed);
    while (frame > cur && !g_max_used.compare_exchange_weak(cur, frame)) {
    }
}

void OnPresentFrame(uint32_t frame) {
    if (!g_installed.load(std::memory_order_relaxed) || frame == 0)
        return;
    uint32_t cur = g_max_used.load(std::memory_order_relaxed);
    while (frame > cur && !g_max_used.compare_exchange_weak(cur, frame)) {
    }
    std::unique_lock lock{g_jobs_mutex};
    const uint64_t target = g_jobs_posted;
    g_jobs_sent_cv.wait_for(lock, std::chrono::milliseconds(2),
                            [target] { return g_jobs_sent >= target; });
}

} // namespace BB::ReflexGame
