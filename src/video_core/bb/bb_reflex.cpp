// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex: NVIDIA Reflex through VK_NV_low_latency2 (see bb_reflex.h).

#include "video_core/bb/bb_reflex_vk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/thread.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::Reflex {
namespace {

// eboot 1.09 vaddr (eboot-patch-v2.elf, checked in Ghidra and byte-checked at load).
// Frame limiter 0x2034770 (this = *(0x55404F8): main loop 0x2018D7D `lea r13,[0x55404F8]`,
// 0x2018E1C `mov rbx,[r13]; mov rdi,rbx; call 0x2034770` at 0x2018E22). Its wait loop: while the
// frame has time left (period this+0x18), sceKernelUsleep(left) at 0x2034BD2 (PLT 0x2BBFE68 ->
// GOT 0x53E5368 sceKernelUsleep), then gettimeofday (PLT 0x2BBE728 -> GOT 0x53E47C8) and the time
// left again; after the loop it stores the measured frame time (s) at this+0x264 (0x2034CF0
// `vmovss [r12+0x264],xmm0`) and bumps its ring index this+0x260. this+0x20 is the previous
// frame's end time (us), copied at the frame's start: one value per frame.
constexpr uint64_t kUsleepCallVa = 0x2034BCA; // vcvttss2si edi,xmm0; test edi,edi; jle; call
constexpr uint8_t kUsleepCallBytes[] = {0xC5, 0xFA, 0x2C, 0xF8, 0x85, 0xFF, 0x7E,
                                        0x0E, 0xE8, 0x91, 0xB2, 0xB8, 0x00};
constexpr uint64_t kUsleepReturnVa = kUsleepCallVa + sizeof(kUsleepCallBytes); // 0x2034BD7
// The limiter has two waits, chosen by r8b = the result of 0xFE3980 (call at 0x20347C5): r8b != 0
// (taken in the 1.09 runs: the 0x2034BD2 usleep never fired, measured dt exactly the period) is a
// pure gettimeofday spin (0x2034AF0..0x2034B6F, no sleep at all); r8b == 0 is the usleep loop
// (0x2034B90): usleep(left - 5 ms) only if left > 1 ms and > 5 ms, then gettimeofday. With Reflex
// armed the limiter is sent to the usleep loop always (je -> jmp at 0x2034ADA) and its usleep is
// called on every pass (the three skips at 0x2034BA6 / 0x2034BB0 / 0x2034BD0 become nops), so the
// first pass of each frame is the Reflex sleep; later passes with nothing left over 5 ms (usec <=
// 0) return at once (the loop then polls gettimeofday until the game's deadline, as the original
// usleep loop does in its last 5 ms), others are the game's own usleep as before.
struct CodePatch {
    uint64_t va;
    uint8_t len;
    uint8_t orig[6];
    uint8_t neu[6];
};
constexpr CodePatch kWaitPatches[] = {
    {0x2034ADA, 6, {0x0F, 0x84, 0x9A, 0x00, 0x00, 0x00}, {0xE9, 0x9B, 0x00, 0x00, 0x00, 0x90}},
    {0x2034BA6, 2, {0x76, 0x38}, {0x90, 0x90}},
    {0x2034BB0, 2, {0x76, 0x2E}, {0x90, 0x90}},
    {0x2034BD0, 2, {0x7E, 0x0E}, {0x90, 0x90}},
};
constexpr uint64_t kLimiterPtrVa = 0x55404F8;
constexpr uint64_t kLimiterPrologueVa = 0x2034770;
constexpr uint8_t kLimiterPrologue[] = {0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56};
constexpr uint32_t kOffPeriod = 0x18, kOffFrameKey = 0x20, kOffMeasured = 0x264;
constexpr uint32_t kMaxTimeouts = 3;

struct Settings {
    bool env = false; // BB_REFLEX_NATIVE set to 1 / force
    bool requested = false;
    bool boost = false;
    std::string why;
};

std::string Env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
}

// BB_REFLEX_NATIVE=1: with DLSS-G (BB_FG=dlss-*); =force: also without frame generation. Never
// with FSR 3 FG. Anything else (unset, the default): no Reflex at all, nothing changes.
const Settings& Cfg() {
    static const Settings s = [] {
        Settings r;
        const std::string native = Env("BB_REFLEX_NATIVE");
        r.env = native == "1" || native == "force";
        if (!r.env) {
            return r; // silent: the default path logs nothing new
        }
        const auto kind = BB::FG::Cfg().kind;
        if (kind == BB::FG::Kind::Amd) {
            r.why = "FSR 3 frame generation paces its own frames";
        } else if (kind == BB::FG::Kind::Dlss) {
            r.requested = true;
            r.why = "BB_REFLEX_NATIVE with DLSS-G";
        } else if (native == "force") {
            r.requested = true;
            r.why = "BB_REFLEX_NATIVE=force without DLSS-G";
        } else {
            r.why = "BB_REFLEX_NATIVE=1 needs DLSS-G (BB_FG=dlss-*) or BB_REFLEX_NATIVE=force";
        }
        r.boost = Env("BB_REFLEX_BOOST") == "1";
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex-native] {} ({}){}", r.requested ? "asked" : "off", r.why,
            r.requested ? fmt::format(": mode {}", r.boost ? "LowLatencyWithBoost" : "LowLatency")
                        : std::string()));
        return r;
    }();
    return s;
}
// Device.
bool g_ext_added = false;
std::atomic<bool> g_ready{false}; // set once on the render thread, read by the game threads
std::atomic<bool> g_disabled{false};
std::string g_off_reason;
vk::Device g_device{};
PFN_vkSetLatencySleepModeNV g_set_sleep_mode = nullptr;
PFN_vkLatencySleepNV g_sleep = nullptr;
PFN_vkSetLatencyMarkerNV g_set_marker = nullptr;
vk::Semaphore g_sleep_semaphore{};
uint64_t g_sleep_value = 0;

// Swapchain: shared by sleep / markers, exclusive while it is replaced.
std::shared_mutex g_sc_mutex;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
std::atomic<uint32_t> g_applied_interval_us{UINT32_MAX};
std::atomic<bool> g_latency_swapchain{false};
bool g_said_active = false;

// Frame ids (see bb_reflex.h).
std::mutex g_frames_mutex;
uint64_t g_next_id = 0;
std::deque<uint64_t> g_sim;  // started by the game, not flipped yet
uint64_t g_last_sim = 0;     // the latest SIMULATION_START
uint64_t g_sim_end_upto = 0; // SIMULATION_END sent up to this id
uint64_t g_render_id = 0;    // RENDERSUBMIT_START sent, not flipped yet
std::atomic<bool> g_render_open{false};
std::deque<uint64_t> g_gpu; // flipped by the guest, not prepared by the GPU thread yet
                            // (then the id travels with the frame: Frame::bb_reflex_id)
uint64_t g_last_present_id = 0;

// The limiter's sceKernelUsleep (OnGuestUsleep).
uintptr_t g_eboot_base = 0;
std::atomic<uintptr_t> g_usleep_return{0};
std::mutex g_guest_mutex;      // one guest wait at a time (the game's main thread)
uint64_t g_last_frame_key = 0; // under g_guest_mutex: this+0x20 of the frame that slept
std::atomic<uint32_t> g_timeouts_total{0};
uint64_t g_eboot_size = 0; // image size (OnEbootLoaded), for the diagnostics' addresses
// Diagnostics (BB_REFLEX_NATIVE only): which guest calls reach the native sceKernelUsleep, and
// whether the limiter's call site (return address) matches. Per second, with the [reflex-native]
// line; the first few of each kind are logged once.
std::atomic<uint32_t> g_usleep_all{0};     // every guest call that reaches the native usleep
std::atomic<uint32_t> g_usleep_limiter{0}; // return address == the limiter's call site (0x2034BD7)
std::atomic<uint32_t> g_usleep_other{0};   // any other return address (the game's own wait)
std::atomic<uint32_t> g_usleep_nothing{0}; // limiter site, usec <= 0 and no Reflex sleep: no wait
std::atomic<uint32_t> g_limiter_silent{0}; // GuestWait left before its sleep (no limiter / period)
std::atomic<uintptr_t> g_last_other_ra{0};
std::atomic<uint32_t> g_other_logged{0}, g_hit_logged{0};
// Per-second counters ([reflex-native] line).
std::atomic<uint32_t> g_markers{0}, g_sleeps{0}, g_sleep_failed{0}, g_timeouts{0}, g_topups{0},
    g_fallbacks{0};
std::atomic<uint64_t> g_sleep_us{0};
std::atomic<uint32_t> g_sleep_failures_in_row{0};
std::mutex g_stats_mutex; // the dt / present interval samples
std::vector<double> g_dt_ms, g_present_ms;
std::chrono::steady_clock::time_point g_last_present{}, g_last_log{};

// bb-reflex-crash: every VK_NV_low_latency2 call runs on this host thread (or the host present
// thread), never on a guest thread. Guest threads run host code on their guest stacks with the
// TEB's stack limits disabled (stack_asm.cpp: no __chkstk probing) and a single 16 KiB guard page
// between neighbouring guest stacks (stack.cpp), so a deep driver call there (the sleep, the
// semaphore wait, the markers) could step over the guard into another thread's stack. The guest
// threads only queue markers here and wait for the frame-start sleep.
// Two threads, so a marker never waits behind a sleep: shadPS4:ReflexMarks sends the simulation
// and render submit markers in FIFO order, shadPS4:ReflexSleep runs the frame start's sleep.
struct Job {
    uint64_t id = 0;
    VkLatencyMarkerNV marker = VK_LATENCY_MARKER_SIMULATION_START_NV;
};
std::mutex g_jobs_mutex;
std::condition_variable g_jobs_cv, g_jobs_sent_cv;
std::deque<Job> g_jobs;
uint64_t g_jobs_posted = 0, g_jobs_sent = 0;
std::atomic<bool> g_workers_started{false};
std::mutex g_start_mutex;
std::condition_variable g_start_cv, g_done_cv;
uint64_t g_ticket = 0, g_taken_ticket = 0, g_done_ticket = 0, g_start_end_id = 0;
uint32_t g_start_interval_us = 0;
std::atomic<uint32_t> g_last_interval_us{0};
bool g_done_slept = false;

// Under g_sc_mutex (exclusive, or shared from the only frame-start thread).
bool ApplySleepMode(uint32_t interval_us) {
    if (g_swapchain == VK_NULL_HANDLE || g_set_sleep_mode == nullptr)
        return false;
    VkLatencySleepModeInfoNV mode{};
    mode.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV;
    mode.lowLatencyMode = VK_TRUE;
    mode.lowLatencyBoost = Cfg().boost ? VK_TRUE : VK_FALSE;
    mode.minimumIntervalUs = interval_us;
    const VkResult r = g_set_sleep_mode(static_cast<VkDevice>(g_device), g_swapchain, &mode);
    if (r != VK_SUCCESS) {
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex] vkSetLatencySleepModeNV failed ({}): Reflex off, frames as before",
            static_cast<int>(r)));
        g_disabled = true;
        return false;
    }
    g_applied_interval_us = interval_us;
    BB::Upscale::Checkpoint(
        fmt::format("[reflex] sleep mode: low latency on, boost {}, minimum interval {} us ({})",
                    Cfg().boost ? "on" : "off", interval_us,
                    interval_us ? fmt::format("frame cap {:.1f} fps by Reflex", 1e6 / interval_us)
                                : std::string("no cap from Reflex")));
    return true;
}

void Marker(uint64_t id, VkLatencyMarkerNV marker) {
    if (id == 0 || !Active())
        return;
    std::shared_lock lock{g_sc_mutex};
    if (g_swapchain == VK_NULL_HANDLE)
        return;
    VkSetLatencyMarkerInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV;
    info.presentID = id;
    info.marker = marker;
    g_set_marker(static_cast<VkDevice>(g_device), g_swapchain, &info);
    g_markers.fetch_add(1, std::memory_order_relaxed);
}

template <typename T>
void PushBounded(std::deque<T>& q, T v, size_t max) {
    q.push_back(v);
    while (q.size() > max)
        q.pop_front();
}

// Returns false when the sleep did not happen (no swapchain, an error).
bool LatencySleep(uint32_t want) {
    std::shared_lock lock{g_sc_mutex};
    if (g_swapchain == VK_NULL_HANDLE)
        return false;
    if (want != g_applied_interval_us) {
        // Only this thread sleeps; the mode is set under the shared lock from here too.
        if (!ApplySleepMode(want))
            return false;
    }
    VkLatencySleepInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV;
    info.signalSemaphore = static_cast<VkSemaphore>(g_sleep_semaphore);
    info.value = ++g_sleep_value;
    const auto t0 = std::chrono::steady_clock::now();
    const VkResult r = g_sleep(static_cast<VkDevice>(g_device), g_swapchain, &info);
    lock.unlock();
    bool ok = r == VK_SUCCESS;
    if (ok) {
        const vk::SemaphoreWaitInfo wait{
            .semaphoreCount = 1,
            .pSemaphores = &g_sleep_semaphore,
            .pValues = &info.value,
        };
        // A sleep never lasts longer than a frame of the slowest cap; 200 ms means it is lost.
        ok = g_device.waitSemaphores(wait, 200'000'000ull) == vk::Result::eSuccess;
    }
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
            .count();
    if (!ok) {
        g_sleep_failed.fetch_add(1, std::memory_order_relaxed);
        if (g_sleep_failures_in_row.fetch_add(1) + 1 == 30) {
            BB::Upscale::Checkpoint(fmt::format(
                "[reflex-native] 30 sleeps in a row failed (last vkLatencySleepNV {}): Reflex off, "
                "the game's own sceKernelUsleep again",
                static_cast<int>(r)));
            g_disabled = true;
        }
        return false;
    }
    g_sleep_failures_in_row = 0;
    g_sleeps.fetch_add(1, std::memory_order_relaxed);
    g_sleep_us.fetch_add(uint64_t(std::max<int64_t>(us, 0)), std::memory_order_relaxed);
    return true;
}

// Every simulation / render submit marker goes through this one FIFO (shadPS4:ReflexMarks), so
// the driver gets them in the order the frame ids were made and handed on.
void PostMarker(uint64_t id, VkLatencyMarkerNV marker) {
    if (id == 0)
        return;
    {
        std::scoped_lock lock{g_jobs_mutex};
        if (g_jobs.size() >= 256)
            return; // the worker is stuck: markers are not worth a wait
        g_jobs.push_back(Job{.id = id, .marker = marker});
        ++g_jobs_posted;
    }
    g_jobs_cv.notify_one();
}

// Host threads only: the markers posted so far reach the driver first (at most 2 ms).
void WaitMarkersSent() {
    std::unique_lock lock{g_jobs_mutex};
    const uint64_t target = g_jobs_posted;
    g_jobs_sent_cv.wait_for(lock, std::chrono::milliseconds(2),
                            [target] { return g_jobs_sent >= target; });
}

// shadPS4:ReflexSleep: SIMULATION_END of the previous frame, the sleep, SIMULATION_START of the
// new one (the game measures its frame start with its own gettimeofday right after).
bool WorkerFrameStart(uint64_t end_id, uint32_t interval_us) {
    PostMarker(end_id, VK_LATENCY_MARKER_SIMULATION_END_NV);
    WaitMarkersSent();
    const bool slept = LatencySleep(interval_us);
    {
        // Posted before the id is visible to the render thread: its markers come after.
        std::scoped_lock lock{g_frames_mutex};
        const uint64_t id = ++g_next_id;
        PushBounded(g_sim, id, 4);
        g_last_sim = id;
        PostMarker(id, VK_LATENCY_MARKER_SIMULATION_START_NV);
    }
    return slept;
}

void MarkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ReflexMarks");
    for (;;) {
        Job job;
        {
            std::unique_lock lock{g_jobs_mutex};
            g_jobs_cv.wait(lock, [] { return !g_jobs.empty(); });
            job = g_jobs.front();
            g_jobs.pop_front();
        }
        Marker(job.id, job.marker);
        {
            std::scoped_lock lock{g_jobs_mutex};
            ++g_jobs_sent;
        }
        g_jobs_sent_cv.notify_all();
    }
}

void SleepLoop() {
    Common::SetCurrentThreadName("shadPS4:ReflexSleep");
    for (;;) {
        uint64_t ticket = 0, end_id = 0;
        uint32_t interval_us = 0;
        {
            std::unique_lock lock{g_start_mutex};
            g_start_cv.wait(lock, [] { return g_ticket > g_taken_ticket; });
            ticket = g_taken_ticket = g_ticket;
            end_id = g_start_end_id;
            interval_us = g_start_interval_us;
        }
        const bool slept = WorkerFrameStart(end_id, interval_us);
        {
            std::scoped_lock lock{g_start_mutex};
            g_done_ticket = ticket;
            g_done_slept = slept;
        }
        g_done_cv.notify_all();
    }
}

// The limiter object (nullptr: not there yet / implausible).
uint8_t* Limiter() {
    if (g_eboot_base == 0)
        return nullptr;
    uint64_t p = 0;
    std::memcpy(&p, reinterpret_cast<const void*>(g_eboot_base + kLimiterPtrVa), 8);
    return p >= 0x10000 && (p & 3) == 0 ? reinterpret_cast<uint8_t*>(p) : nullptr;
}

template <typename T>
T ReadAt(const uint8_t* base, uint32_t off) {
    T v{};
    std::memcpy(&v, base + off, sizeof(T));
    return v;
}

// Game main thread, in place of the limiter's sceKernelUsleep: the first call of a frame hands
// the wait to shadPS4:ReflexSleep (vkLatencySleepNV, minimum interval = the limiter's period) and
// blocks on it. True: the wait happened there (the guest's usleep is skipped); false: the caller
// sleeps the game's own usleep (later calls of the same frame: the game tops up what Reflex left;
// no swapchain, errors, timeouts).
bool GuestWait(uint32_t usec) {
    const uint8_t* lim = Limiter();
    if (lim == nullptr) {
        g_limiter_silent.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const float period = ReadAt<float>(lim, kOffPeriod);
    const uint64_t frame_key = ReadAt<uint64_t>(lim, kOffFrameKey);
    if (!(period > 0.0005f && period < 0.2f)) {
        g_limiter_silent.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::scoped_lock guest_lock{g_guest_mutex};
    if (frame_key == g_last_frame_key) {
        // The limiter's loop again in the same frame: Reflex's sleep ended before the game's own
        // deadline; the game finishes its wait itself (one deadline, the game's).
        g_topups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_last_frame_key = frame_key;
    uint64_t end_id = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (g_last_sim > g_sim_end_upto) {
            end_id = g_last_sim;
            g_sim_end_upto = g_last_sim;
        }
    }
    const uint32_t interval_us = static_cast<uint32_t>(std::lround(double(period) * 1e6));
    g_last_interval_us.store(interval_us, std::memory_order_relaxed);
    bool done = false, slept = false;
    {
        std::unique_lock lock{g_start_mutex};
        const uint64_t ticket = ++g_ticket;
        g_start_end_id = end_id;
        g_start_interval_us = interval_us;
        g_start_cv.notify_one();
        done = g_done_cv.wait_for(lock, std::chrono::microseconds(interval_us + 50'000),
                                  [ticket] { return g_done_ticket >= ticket; });
        slept = done && g_done_ticket == ticket && g_done_slept;
    }
    if (!done) {
        g_timeouts.fetch_add(1, std::memory_order_relaxed);
        if (g_timeouts_total.fetch_add(1) + 1 == kMaxTimeouts) {
            g_disabled = true;
            BB::Upscale::Checkpoint(fmt::format(
                "[reflex-native] shadPS4:ReflexSleep missed the frame's bound (period + 50 ms) {} "
                "times: Reflex off for good, the game's own sceKernelUsleep again",
                kMaxTimeouts));
        }
    }
    if (!slept) {
        g_fallbacks.fetch_add(1, std::memory_order_relaxed);
        (void)usec; // the caller sleeps the game's own usleep(usec)
    }
    return slept;
}

void MeanSd(const std::vector<double>& v, double& mean, double& sd) {
    mean = sd = 0.0;
    if (v.empty())
        return;
    for (const double x : v)
        mean += x;
    mean /= double(v.size());
    for (const double x : v)
        sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
}

// Under g_frames_mutex: the frame the render thread is submitting (made when there is none).
uint64_t OpenRenderFrame(uint64_t& sim_end) {
    if (g_render_id != 0)
        return g_render_id;
    // The oldest frame started and not flipped: the render thread runs one frame behind the
    // main thread or with it, either way that one.
    uint64_t id = g_sim.empty() ? 0 : g_sim.front();
    if (id == 0) {
        // A flip without a frame start of its own (loading screens, menus outside the main
        // loop's limiter).
        id = ++g_next_id;
        PushBounded(g_sim, id, 4);
    } else if (id > g_sim_end_upto) {
        sim_end = id;
        g_sim_end_upto = id;
    }
    g_render_id = id;
    g_render_open.store(true, std::memory_order_relaxed);
    return id;
}

} // namespace

bool Requested() {
    return Cfg().requested;
}

void AddDeviceExtensions(vk::PhysicalDevice physical, const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    if (!Requested())
        return;
    const auto has = [&](const char* name) {
        return std::find(available.begin(), available.end(), name) != available.end();
    };
    if (BB::SR::Dlss::Loaded() && !BB::SR::Dlss::Manual()) {
        g_off_reason = "Streamline is the Vulkan loader (BB_SL_NO_INTERPOSER_LOADER=0) and loads "
                       "its own Reflex plugin";
    } else if (!has(VK_NV_LOW_LATENCY_2_EXTENSION_NAME)) {
        g_off_reason = "the GPU / driver has no VK_NV_low_latency2";
    } else if (!has(VK_KHR_PRESENT_ID_EXTENSION_NAME)) {
        g_off_reason = "the GPU / driver has no VK_KHR_present_id";
    } else {
        vk::PhysicalDevicePresentIdFeaturesKHR present_id{};
        vk::PhysicalDeviceVulkan12Features f12{};
        f12.pNext = &present_id;
        vk::PhysicalDeviceFeatures2 f{};
        f.pNext = &f12;
        physical.getFeatures2(&f);
        if (!present_id.presentId) {
            g_off_reason = "the presentId feature is unsupported";
        } else if (!f12.timelineSemaphore) {
            g_off_reason = "timeline semaphores are unsupported";
        }
    }
    if (!g_off_reason.empty()) {
        BB::Upscale::Checkpoint(
            fmt::format("[reflex] not available: {}; frames as before", g_off_reason));
        return;
    }
    for (const char* name :
         {VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_NV_LOW_LATENCY_2_EXTENSION_NAME}) {
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (!added)
            extensions.push_back(name);
    }
    g_ext_added = true;
    BB::Upscale::Checkpoint("[reflex] device extensions VK_NV_low_latency2 and VK_KHR_present_id "
                            "added");
}

void* ChainDeviceFeatures(void* next) {
    if (!g_ext_added)
        return next;
    for (auto* s = static_cast<const VkBaseInStructure*>(next); s != nullptr; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR) {
            return next; // already asked for (Streamline's present ids)
        }
    }
    static VkPhysicalDevicePresentIdFeaturesKHR present_id{};
    present_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
    present_id.presentId = VK_TRUE;
    present_id.pNext = next;
    return &present_id;
}

void OnDevice(vk::Device device) {
    if (!g_ext_added)
        return;
    const auto proc = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    const VkDevice d = static_cast<VkDevice>(device);
    g_set_sleep_mode =
        reinterpret_cast<PFN_vkSetLatencySleepModeNV>(proc(d, "vkSetLatencySleepModeNV"));
    g_sleep = reinterpret_cast<PFN_vkLatencySleepNV>(proc(d, "vkLatencySleepNV"));
    g_set_marker = reinterpret_cast<PFN_vkSetLatencyMarkerNV>(proc(d, "vkSetLatencyMarkerNV"));
    if (g_set_sleep_mode == nullptr || g_sleep == nullptr || g_set_marker == nullptr) {
        BB::Upscale::Checkpoint("[reflex] VK_NV_low_latency2 entry points missing: Reflex off, "
                                "frames as before");
        return;
    }
    vk::SemaphoreTypeCreateInfo type{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    const vk::SemaphoreCreateInfo info{.pNext = &type};
    auto [result, semaphore] = device.createSemaphore(info);
    if (result != vk::Result::eSuccess) {
        BB::Upscale::Checkpoint(fmt::format("[reflex] sleep semaphore not made ({}): Reflex off",
                                            vk::to_string(result)));
        return;
    }
    g_device = device;
    g_sleep_semaphore = semaphore;
    if (!g_workers_started.exchange(true)) {
        // Detached: they only wait on their condition variables when idle, and the emulator
        // ends with std::quick_exit (no static destructors run under them).
        std::thread{MarkerLoop}.detach();
        std::thread{SleepLoop}.detach();
    }
    g_ready.store(true, std::memory_order_release);
    BB::Upscale::Checkpoint("[reflex] VK_NV_low_latency2 ready on the device (the swapchain "
                            "is made with latency mode on; sleep and markers on the host threads "
                            "shadPS4:ReflexSleep / ReflexMarks, none on the game's threads)");
}

const void* SwapchainCreateNext(const void* next) {
    if (!g_ready || g_disabled)
        return nullptr;
    static VkSwapchainLatencyCreateInfoNV latency{};
    latency.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV;
    latency.pNext = next;
    latency.latencyModeEnable = VK_TRUE;
    return &latency;
}

void OnSwapchainCreated(vk::SwapchainKHR swapchain, bool latency_mode) {
    if (!g_ready)
        return;
    std::unique_lock lock{g_sc_mutex};
    if (!latency_mode) {
        g_swapchain = VK_NULL_HANDLE;
        g_latency_swapchain = false;
        if (!g_disabled.exchange(true)) {
            BB::Upscale::Checkpoint("[reflex] the swapchain could not be made with latency mode: "
                                    "Reflex off, frames as before");
        }
        return;
    }
    g_swapchain = static_cast<VkSwapchainKHR>(swapchain);
    g_applied_interval_us = UINT32_MAX;
    // The limiter's period of the last frame start (0 before the first: set by the next sleep).
    if (!ApplySleepMode(g_last_interval_us.load(std::memory_order_relaxed))) {
        g_swapchain = VK_NULL_HANDLE;
        return;
    }
    g_latency_swapchain = true;
    if (!g_said_active) {
        g_said_active = true;
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex-native] NVIDIA Reflex ACTIVE (VK_NV_low_latency2; {}): mode {}, sleep {}, "
            "markers simulation / render submit / present per game frame, present ids on the "
            "game frames' presents{}",
            Cfg().why, Cfg().boost ? "LowLatencyWithBoost" : "LowLatency",
            g_usleep_return.load() ? "in place of the frame limiter's sceKernelUsleep (0x2034BD2)"
                                   : "NOT armed (hook not found): the game's own wait",
            BB::FG::Dlss::Requested() ? "; DLSS-G frames presented between them" : ""));
    }
}

void OnSwapchainDestroy() {
    if (!g_ready)
        return;
    std::unique_lock lock{g_sc_mutex};
    g_swapchain = VK_NULL_HANDLE;
    g_latency_swapchain = false;
}

void OnEbootLoaded(uintptr_t eboot_base, uint64_t image_size) {
    if (!Requested())
        return;
    const bool inside = kUsleepReturnVa <= image_size && kLimiterPtrVa + 8 <= image_size;
    const bool match = inside &&
                       std::memcmp(reinterpret_cast<const void*>(eboot_base + kUsleepCallVa),
                                   kUsleepCallBytes, sizeof(kUsleepCallBytes)) == 0 &&
                       std::memcmp(reinterpret_cast<const void*>(eboot_base + kLimiterPrologueVa),
                                   kLimiterPrologue, sizeof(kLimiterPrologue)) == 0;
    if (!match) {
        BB::Upscale::Checkpoint("[reflex-native] the frame limiter's sceKernelUsleep call (1.09, "
                                "0x2034BD2) is not there: no Reflex sleep (other version / patch), "
                                "the game's own wait");
        return;
    }
    bool wait_ok = true;
    for (const CodePatch& p : kWaitPatches)
        wait_ok = wait_ok &&
                  std::memcmp(reinterpret_cast<const void*>(eboot_base + p.va), p.orig, p.len) == 0;
    if (!wait_ok) {
        BB::Upscale::Checkpoint("[reflex-native] the frame limiter's wait branches (0x2034ADA, "
                                "0x2034BA6, 0x2034BB0, 0x2034BD0) are not the 1.09 bytes: no "
                                "Reflex sleep, the game's own wait");
        return;
    }
    // After the byte check of kUsleepCallBytes (it covers the jle at 0x2034BD0).
    for (const CodePatch& p : kWaitPatches)
        std::memcpy(reinterpret_cast<void*>(eboot_base + p.va), p.neu, p.len);
    g_eboot_base = eboot_base;
    g_eboot_size = image_size;
    g_usleep_return = eboot_base + kUsleepReturnVa;
    BB::Upscale::Checkpoint(
        "[reflex-native] hook armed: the frame limiter's (0x2034770) sceKernelUsleep at 0x2034BD2 "
        "becomes vkLatencySleepNV (minimum interval = the limiter's period this+0x18) on "
        "shadPS4:ReflexSleep; the limiter's gettimeofday spin path is redirected to its usleep "
        "loop (0x2034ADA) and the usleep is called on every pass (0x2034BA6/BB0/BD0), so it is "
        "the frame's one wait; the game's gettimeofday / dt after it is unchanged");
}

bool NativeEnv() {
    const char* v = std::getenv("BB_REFLEX_NATIVE");
    return v != nullptr && (std::strcmp(v, "1") == 0 || std::strcmp(v, "force") == 0);
}

// Where a guest return address is: the eboot image (as the vaddr of the 1.09 build) or outside it.
// Diagnostics only (the one-time lines of an unmatched call).
static std::string DescribeGuestAddress(uintptr_t ra) {
    if (g_eboot_base != 0 && ra >= g_eboot_base && ra < g_eboot_base + g_eboot_size) {
        std::string s = fmt::format("eboot vaddr 0x{:x}", ra - g_eboot_base);
        // The 5 bytes before the return address: a call's E8 rel32 and where it goes.
        if (ra - 5 >= g_eboot_base) {
            uint8_t b[5]{};
            std::memcpy(b, reinterpret_cast<const void*>(ra - 5), 5);
            int32_t rel = 0;
            std::memcpy(&rel, b + 1, 4);
            s += fmt::format(" (bytes {:02x} {:02x} {:02x} {:02x} {:02x}", b[0], b[1], b[2], b[3],
                             b[4]);
            if (b[0] == 0xE8) {
                s += fmt::format(": call vaddr 0x{:x}", ra - g_eboot_base + rel);
            }
            s += ")";
        }
        return s;
    }
    return fmt::format("outside the eboot image (0x{:x})", ra);
}

bool OnGuestUsleep(const void* return_address, uint32_t usec) {
    const uintptr_t ra = reinterpret_cast<uintptr_t>(return_address);
    g_usleep_all.fetch_add(1, std::memory_order_relaxed);
    const uintptr_t hook = g_usleep_return.load(std::memory_order_relaxed);
    if (hook == 0 || ra != hook) {
        // Not the frame limiter's call: the game's own wait, as before. Counted (per second and
        // the first three with their addresses) so a run shows whether the limiter's call ever
        // arrives here and from where.
        g_usleep_other.fetch_add(1, std::memory_order_relaxed);
        g_last_other_ra.store(ra, std::memory_order_relaxed);
        if (g_other_logged.fetch_add(1, std::memory_order_relaxed) < 3) {
            BB::Upscale::Checkpoint(fmt::format(
                "[reflex-native] guest sceKernelUsleep({} us) from {}, not the frame limiter's "
                "call at 0x2034BD2 (return 0x2034BD7): the game's own wait",
                usec, DescribeGuestAddress(ra)));
        }
        return false;
    }
    g_usleep_limiter.fetch_add(1, std::memory_order_relaxed);
    if (g_hit_logged.exchange(1, std::memory_order_relaxed) == 0) {
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex-native] hook fired: the frame limiter's sceKernelUsleep({} us) reached the "
            "native path from {} (Reflex {})",
            usec, DescribeGuestAddress(ra), Active() ? "active" : "not active: the game's wait"));
    }
    // The patched loop calls with usec <= 0 (vcvttss2si, signed) when less than 5 ms is left:
    // never hand that to the game's usleep (as u32 it would be huge); no wait, as the original jle.
    const bool nothing_left = static_cast<int32_t>(usec) <= 0;
    bool waited = false;
    if (!Active()) {
        g_fallbacks.fetch_add(1, std::memory_order_relaxed);
    } else {
        waited = GuestWait(usec);
    }
    if (!waited && nothing_left) {
        g_usleep_nothing.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return waited;
}

void OnGuestSubmit() {
    if (g_render_open.load(std::memory_order_relaxed) || !Active())
        return;
    uint64_t id = 0, sim_end = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (g_render_id != 0)
            return;
        id = OpenRenderFrame(sim_end);
    }
    PostMarker(sim_end, VK_LATENCY_MARKER_SIMULATION_END_NV);
    PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
}

void OnGuestFlip() {
    if (Requested() && g_usleep_return.load(std::memory_order_relaxed) != 0) {
        // The game's measured frame time of its latest frame (stored after its wait).
        if (const uint8_t* lim = Limiter()) {
            const float dt = ReadAt<float>(lim, kOffMeasured);
            std::scoped_lock lock{g_stats_mutex};
            if (dt > 0.0f && dt < 1.0f && g_dt_ms.size() < 4096)
                g_dt_ms.push_back(dt * 1e3);
        }
    }
    if (!Active())
        return;
    uint64_t id = 0, sim_end = 0;
    bool opened = false;
    {
        std::scoped_lock lock{g_frames_mutex};
        opened = g_render_id == 0;
        id = OpenRenderFrame(sim_end);
        while (!g_sim.empty() && g_sim.front() <= id)
            g_sim.pop_front();
        g_render_id = 0;
        g_render_open.store(false, std::memory_order_relaxed);
        PushBounded(g_gpu, id, 8);
    }
    PostMarker(sim_end, VK_LATENCY_MARKER_SIMULATION_END_NV);
    if (opened)
        PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
    PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
}

void OnGuestFlipDropped() {
    if (!Active())
        return;
    std::scoped_lock lock{g_frames_mutex};
    if (!g_gpu.empty())
        g_gpu.pop_back();
}

uint64_t OnGpuFlip() {
    if (!Active())
        return 0;
    std::scoped_lock lock{g_frames_mutex};
    if (g_gpu.empty())
        return 0;
    const uint64_t id = g_gpu.front();
    g_gpu.pop_front();
    return id;
}

uint64_t SubmitPresentId() {
    if (!Active())
        return 0;
    std::scoped_lock lock{g_frames_mutex};
    return !g_gpu.empty() ? g_gpu.front() : g_render_id;
}

uint64_t BeginPresent(uint64_t id) {
    if (id == 0 || !Active())
        return 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (id <= g_last_present_id)
            return 0;
    }
    WaitMarkersSent(); // RENDERSUBMIT_END of this id before its PRESENT_START
    Marker(id, VK_LATENCY_MARKER_PRESENT_START_NV);
    return id;
}

void EndPresent(uint64_t id, bool presented) {
    if (id == 0)
        return;
    Marker(id, VK_LATENCY_MARKER_PRESENT_END_NV);
    if (presented) {
        std::scoped_lock lock{g_frames_mutex};
        g_last_present_id = std::max(g_last_present_id, id);
    }
}

void OnGameFramePresented() {
    if (!Requested() || g_usleep_return.load(std::memory_order_relaxed) == 0)
        return;
    const auto now = std::chrono::steady_clock::now();
    std::string line;
    {
        std::scoped_lock lock{g_stats_mutex};
        if (g_last_present.time_since_epoch().count() != 0) {
            const double ms =
                std::chrono::duration<double, std::milli>(now - g_last_present).count();
            if (g_present_ms.size() < 4096)
                g_present_ms.push_back(ms);
        }
        g_last_present = now;
        if (g_last_log.time_since_epoch().count() == 0)
            g_last_log = now;
        const double sec = std::chrono::duration<double>(now - g_last_log).count();
        if (sec < 1.0)
            return;
        g_last_log = now;
        double dt_mean, dt_sd, pr_mean, pr_sd;
        MeanSd(g_dt_ms, dt_mean, dt_sd);
        MeanSd(g_present_ms, pr_mean, pr_sd);
        const uint32_t sleeps = g_sleeps.exchange(0);
        const uint64_t us = g_sleep_us.exchange(0);
        const uint8_t* lim = Limiter();
        const float period = lim != nullptr ? ReadAt<float>(lim, kOffPeriod) : 0.0f;
        const uint32_t usleep_all = g_usleep_all.exchange(0);
        const uint32_t usleep_limiter = g_usleep_limiter.exchange(0);
        const uint32_t usleep_nothing = g_usleep_nothing.exchange(0);
        const uint32_t usleep_other = g_usleep_other.exchange(0);
        const uint32_t limiter_silent = g_limiter_silent.exchange(0);
        const uintptr_t last_other = g_last_other_ra.load(std::memory_order_relaxed);
        const bool last_inside = g_eboot_base != 0 && last_other >= g_eboot_base &&
                                 last_other < g_eboot_base + g_eboot_size;
        const std::string last_other_text =
            last_other == 0 ? std::string("none")
            : last_inside   ? fmt::format("vaddr 0x{:x}", last_other - g_eboot_base)
                            : fmt::format("0x{:x}", last_other);
        line = fmt::format(
            "[reflex-native] {:.2f} s: sleeps {} (avg {:.2f} ms, {} failed), timeouts {} ({} "
            "total), game top-ups {}, fallbacks {} | usleep calls {} (limiter site {} [<=0 us, no "
            "wait: {}], other {}, "
            "last other return {}), limiter exits before sleep {} | state {} | period {:.2f} ms | "
            "game dt (this+0x264) {:.2f} ms sd {:.2f} (n {}) vs present interval {:.2f} ms sd "
            "{:.2f} (n {}) | markers {}",
            sec, sleeps, sleeps ? us / 1e3 / sleeps : 0.0, g_sleep_failed.exchange(0),
            g_timeouts.exchange(0), g_timeouts_total.load(), g_topups.exchange(0),
            g_fallbacks.exchange(0), usleep_all, usleep_limiter, usleep_nothing, usleep_other,
            last_other_text, limiter_silent,
            Active()     ? std::string("reflex")
            : g_disabled ? std::string("fallback (disabled)")
            : !g_ready   ? "fallback (" +
                             (g_off_reason.empty() ? std::string("VK_NV_low_latency2 not set up")
                                                   : g_off_reason) +
                             ")"
                       : std::string("fallback (no latency swapchain)"),
            period * 1e3f, dt_mean, dt_sd, g_dt_ms.size(), pr_mean, pr_sd, g_present_ms.size(),
            g_markers.exchange(0));
        g_dt_ms.clear();
        g_present_ms.clear();
    }
    LOG_INFO(Lib_VideoOut, "{}", line);
}

bool Active() {
    return g_ready.load(std::memory_order_acquire) &&
           g_latency_swapchain.load(std::memory_order_relaxed) &&
           !g_disabled.load(std::memory_order_relaxed);
}

} // namespace BB::Reflex
