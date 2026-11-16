// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-reflex: NVIDIA Reflex through VK_NV_low_latency2 (see bb_reflex.h).

#include "video_core/bb/bb_reflex_vk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <fmt/format.h>
#include "common/thread.h"
#include "core/bb_vardt.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::Reflex {
namespace {

// eboot 1.09 vaddr: the frame limiter (0x2034770) reads the frame start time with
// `lea rdi,[rbp-0x40]; xor esi,esi; call gettimeofday` (0x2034971; PLT 0x2BBE728, GOT import
// gettimeofday), the first thing it does after the previous frame; its return address is
// 0x203497C. The limiter is called once per main loop frame (0x2018E22), right before the FD4
// task manager runs the frame's tasks (0x2018E6F), so this call is the game's frame start.
constexpr uint64_t kClockCallVa = 0x2034971;
constexpr uint8_t kClockCallBytes[] = {0x48, 0x8D, 0x7D, 0xC0, 0x31, 0xF6,
                                       0xE8, 0xAC, 0x9D, 0xB8, 0x00};
constexpr uint64_t kClockReturnVa = kClockCallVa + sizeof(kClockCallBytes);

struct Settings {
    bool requested = false;
    bool boost = false;
    bool cap = true; // BB_REFLEX_CAP=0: the game's limiter keeps BB_GAME_FPS_CAP
    std::string why;
};

const Settings& Cfg() {
    static const Settings s = [] {
        Settings r;
        const auto env = [](const char* name) -> std::string {
            const char* v = std::getenv(name);
            return v != nullptr ? std::string(v) : std::string();
        };
        const std::string force = env("BB_REFLEX");
        const std::string fg_raw = env("BB_FG");
        const std::string al = env("BB_ANTILAG");
        const std::string al_dlssg = env("BB_ANTILAG_DLSSG");
        const auto kind = BB::FG::Cfg().kind;
        const bool fg_off = kind == BB::FG::Kind::Off &&
                            (fg_raw.empty() || fg_raw == "off" || fg_raw == "0");
        // BB_REFLEX: "1" asks for it, "0" / "off" turns it off (the launcher writes "0" whenever
        // it does not pick Reflex), unset follows the DLSS-G setting below.
        if (force == "off" || force == "0") {
            r.why = fmt::format("BB_REFLEX={}", force);
        } else if (kind == BB::FG::Kind::Amd) {
            r.why = "FSR 3 frame generation paces its own frames (its interpolation swapchain)";
        } else if (force == "1") {
            r.requested = true;
            r.why = "BB_REFLEX=1";
        } else if (kind == BB::FG::Kind::Dlss) {
            r.requested = al_dlssg == "reflex";
            r.why = fmt::format("DLSS-G with BB_ANTILAG_DLSSG={}",
                                al_dlssg.empty() ? "(unset)" : al_dlssg);
        } else if (fg_off) {
            // Opt-in only (BB_REFLEX=1). BB_ANTILAG=auto, the launcher's default, used to turn it
            // on for every NVIDIA user without frame generation; review (bb-reflex-rv) made it
            // opt-in: it is untested on hardware and changes the default path (latency-mode
            // swapchain, a sleep inside the game's frame limiter, the fps cap moved from the
            // game's limiter to the driver), which the port's rules keep behind a switch.
            r.requested = false;
            r.why = fmt::format("no frame generation, BB_ANTILAG={}: Reflex only with BB_REFLEX=1",
                                al.empty() ? "(unset)" : al);
        } else {
            r.why = fmt::format("BB_FG={}: not DLSS-G", fg_raw);
        }
        // Anti-Lag+ (ultra), the launcher's lowest-delay choice, asks for Reflex's boost too.
        const std::string boost = env("BB_REFLEX_BOOST");
        r.boost = boost.empty() ? al == "ultra" : boost[0] == '1';
        r.cap = env("BB_REFLEX_CAP") != "0";
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex] {} ({}; BB_ANTILAG={}, BB_ANTILAG_DLSSG={}, BB_REFLEX={}){}",
            r.requested ? "asked" : "off", r.why, al.empty() ? "(unset)" : al,
            al_dlssg.empty() ? "(unset)" : al_dlssg, force.empty() ? "(unset)" : force,
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
std::deque<uint64_t> g_sim;      // started by the game, not flipped yet
uint64_t g_last_sim = 0;         // the latest SIMULATION_START
uint64_t g_sim_end_upto = 0;     // SIMULATION_END sent up to this id
uint64_t g_render_id = 0;        // RENDERSUBMIT_START sent, not flipped yet
std::atomic<bool> g_render_open{false};
std::deque<uint64_t> g_gpu;      // flipped by the guest, not prepared by the GPU thread yet
                                 // (then the id travels with the frame: Frame::bb_reflex_id)
uint64_t g_last_present_id = 0;

// Game frame start hook.
std::atomic<uintptr_t> g_clock_return{0};
std::mutex g_sleep_mutex;
uint32_t g_sleep_failures_in_row = 0;
std::atomic<bool> g_cap_owned{false};

// Per-second counters.
std::atomic<uint32_t> g_markers{0}, g_sleeps{0}, g_sleep_failed{0}, g_frames_started{0};
std::atomic<uint64_t> g_sleep_us{0};

// bb-reflex-crash: every VK_NV_low_latency2 call runs on this host thread (or the host present
// thread), never on a guest thread. Guest threads run host code on their guest stacks with the
// TEB's stack limits disabled (stack_asm.cpp: no __chkstk probing) and a single 16 KiB guard page
// between neighbouring guest stacks (stack.cpp), so a deep driver call there (the sleep, the
// semaphore wait, the markers) could step over the guard into another thread's stack. The guest
// threads only queue markers here and wait for the frame-start sleep.
// Two threads, so a marker never waits behind a sleep: shadPS4:ReflexMarks sends the render
// thread's markers, shadPS4:ReflexSleep the frame start (SIMULATION_END, sleep, SIMULATION_START).
struct Job {
    uint64_t id = 0;
    VkLatencyMarkerNV marker = VK_LATENCY_MARKER_SIMULATION_START_NV;
};
std::mutex g_jobs_mutex;
std::condition_variable g_jobs_cv;
std::deque<Job> g_jobs;
std::mutex g_start_mutex;
std::condition_variable g_start_cv, g_done_cv;
uint64_t g_ticket = 0, g_taken_ticket = 0, g_done_ticket = 0, g_start_end_id = 0;
bool g_done_slept = false;

uint32_t WantedIntervalUs() {
    if (!Cfg().cap) return 0;
    const float period = BB::VarDt::CapPeriod();
    return period > 0.0f ? uint32_t(std::lround(double(period) * 1e6)) : 0u;
}

// Under g_sc_mutex (exclusive, or shared from the only frame-start thread).
bool ApplySleepMode(uint32_t interval_us) {
    if (g_swapchain == VK_NULL_HANDLE || g_set_sleep_mode == nullptr) return false;
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
    BB::Upscale::Checkpoint(fmt::format(
        "[reflex] sleep mode: low latency on, boost {}, minimum interval {} us ({})",
        Cfg().boost ? "on" : "off", interval_us,
        interval_us ? fmt::format("frame cap {:.1f} fps by Reflex", 1e6 / interval_us)
                    : std::string("no cap from Reflex")));
    return true;
}

void Marker(uint64_t id, VkLatencyMarkerNV marker) {
    if (id == 0 || !Active()) return;
    std::shared_lock lock{g_sc_mutex};
    if (g_swapchain == VK_NULL_HANDLE) return;
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
    while (q.size() > max) q.pop_front();
}

// Returns false when the sleep did not happen (no swapchain, an error).
bool LatencySleep() {
    std::shared_lock lock{g_sc_mutex};
    if (g_swapchain == VK_NULL_HANDLE) return false;
    const uint32_t want = WantedIntervalUs();
    if (want != g_applied_interval_us) {
        // Only this thread sleeps; the mode is set under the shared lock from here too.
        if (!ApplySleepMode(want)) return false;
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
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    if (!ok) {
        g_sleep_failed.fetch_add(1, std::memory_order_relaxed);
        if (++g_sleep_failures_in_row == 30) {
            BB::Upscale::Checkpoint(fmt::format(
                "[reflex] 30 sleeps in a row failed (last vkLatencySleepNV {}): Reflex off, the "
                "game's limiter paces the frames again",
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

// shadPS4:ReflexSleep: SIMULATION_END of the previous frame, the sleep, SIMULATION_START of the new one.
bool WorkerFrameStart(uint64_t end_id) {
    Marker(end_id, VK_LATENCY_MARKER_SIMULATION_END_NV);
    const bool slept = LatencySleep();
    uint64_t id = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        id = ++g_next_id;
        PushBounded(g_sim, id, 4);
        g_last_sim = id;
    }
    g_frames_started.fetch_add(1, std::memory_order_relaxed);
    Marker(id, VK_LATENCY_MARKER_SIMULATION_START_NV);
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
    }
}

void SleepLoop() {
    Common::SetCurrentThreadName("shadPS4:ReflexSleep");
    for (;;) {
        uint64_t ticket = 0, end_id = 0;
        {
            std::unique_lock lock{g_start_mutex};
            g_start_cv.wait(lock, [] { return g_ticket > g_taken_ticket; });
            ticket = g_taken_ticket = g_ticket;
            end_id = g_start_end_id;
        }
        const bool slept = WorkerFrameStart(end_id);
        {
            std::scoped_lock lock{g_start_mutex};
            g_done_ticket = ticket;
            g_done_slept = slept;
        }
        g_done_cv.notify_all();
    }
}

// Guest threads: a marker for the worker to send.
void PostMarker(uint64_t id, VkLatencyMarkerNV marker) {
    if (id == 0) return;
    {
        std::scoped_lock lock{g_jobs_mutex};
        if (g_jobs.size() >= 256) return; // the worker is stuck: markers are not worth a wait
        g_jobs.push_back(Job{.id = id, .marker = marker});
    }
    g_jobs_cv.notify_one();
}

// Game main thread, inside the limiter's gettimeofday: waits for the worker's sleep.
void FrameStart() {
    std::scoped_lock sleep_lock{g_sleep_mutex};
    if (!Active()) {
        if (g_cap_owned) {
            g_cap_owned = false;
            BB::VarDt::SetLimiterExternal(false);
        }
        return;
    }
    uint64_t end_id = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (g_last_sim > g_sim_end_upto) {
            end_id = g_last_sim;
            g_sim_end_upto = g_last_sim;
        }
    }
    bool slept = false;
    {
        std::unique_lock lock{g_start_mutex};
        const uint64_t ticket = ++g_ticket;
        g_start_end_id = end_id;
        g_start_cv.notify_one();
        // The sleep itself times out at 200 ms; past 250 the frame goes on without it.
        if (g_done_cv.wait_for(lock, std::chrono::milliseconds(250),
                               [ticket] { return g_done_ticket >= ticket; })) {
            slept = g_done_ticket == ticket && g_done_slept;
        } else {
            g_sleep_failed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // The cap goes to Reflex only while its sleep runs with that minimum interval; otherwise
    // (minimized: no swapchain, errors) the game's limiter has it.
    const uint32_t interval = g_applied_interval_us.load();
    const bool own = slept && interval != 0 && interval != UINT32_MAX;
    // Every frame while Reflex owns the cap (the limiter rewrites its period from its own code
    // each frame); once more when it gives it back.
    if (own || g_cap_owned) {
        BB::VarDt::SetLimiterExternal(own);
    }
    g_cap_owned = own;
}

// Under g_frames_mutex: the frame the render thread is submitting (made when there is none).
uint64_t OpenRenderFrame(uint64_t& sim_end) {
    if (g_render_id != 0) return g_render_id;
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
    if (!Requested()) return;
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
        BB::Upscale::Checkpoint(fmt::format("[reflex] not available: {}; frames as before",
                                            g_off_reason));
        return;
    }
    for (const char* name : {VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_NV_LOW_LATENCY_2_EXTENSION_NAME}) {
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (!added) extensions.push_back(name);
    }
    g_ext_added = true;
    BB::Upscale::Checkpoint("[reflex] device extensions VK_NV_low_latency2 and VK_KHR_present_id "
                            "added");
}

void* ChainDeviceFeatures(void* next) {
    if (!g_ext_added) return next;
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
    if (!g_ext_added) return;
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
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex] sleep semaphore not made ({}): Reflex off", vk::to_string(result)));
        return;
    }
    g_device = device;
    g_sleep_semaphore = semaphore;
    std::thread{MarkerLoop}.detach();
    std::thread{SleepLoop}.detach();
    g_ready.store(true, std::memory_order_release);
    BB::Upscale::Checkpoint("[reflex] VK_NV_low_latency2 ready on the device (the swapchain "
                            "is made with latency mode on; sleep and markers on the host threads "
                            "shadPS4:ReflexSleep / ReflexMarks, none on the game's threads)");
}

const void* SwapchainCreateNext(const void* next) {
    if (!g_ready || g_disabled) return nullptr;
    static VkSwapchainLatencyCreateInfoNV latency{};
    latency.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV;
    latency.pNext = next;
    latency.latencyModeEnable = VK_TRUE;
    return &latency;
}

void OnSwapchainCreated(vk::SwapchainKHR swapchain, bool latency_mode) {
    if (!g_ready) return;
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
    if (!ApplySleepMode(WantedIntervalUs())) {
        g_swapchain = VK_NULL_HANDLE;
        return;
    }
    g_latency_swapchain = true;
    if (!g_said_active) {
        g_said_active = true;
        BB::Upscale::Checkpoint(fmt::format(
            "[reflex] NVIDIA Reflex ACTIVE (VK_NV_low_latency2; {}): mode {}, sleep at the game's "
            "frame start ({}), markers simulation / render submit / present per game frame, "
            "present ids on the game frames' presents{}",
            Cfg().why, Cfg().boost ? "LowLatencyWithBoost" : "LowLatency",
            g_clock_return.load() ? "frame limiter 0x2034770" : "hook NOT armed: no sleep yet",
            BB::FG::Dlss::Requested() ? "; DLSS-G frames presented between them" : ""));
    }
}

void OnSwapchainDestroy() {
    if (!g_ready) return;
    std::unique_lock lock{g_sc_mutex};
    g_swapchain = VK_NULL_HANDLE;
    g_latency_swapchain = false;
}

void OnEbootLoaded(uintptr_t eboot_base, uint64_t image_size) {
    if (!Requested()) return;
    const bool inside = kClockReturnVa <= image_size;
    const bool match =
        inside && std::memcmp(reinterpret_cast<const void*>(eboot_base + kClockCallVa),
                              kClockCallBytes, sizeof(kClockCallBytes)) == 0;
    if (!match) {
        BB::Upscale::Checkpoint("[reflex] the frame limiter's gettimeofday call (1.09, 0x2034971) "
                                "is not there: no Reflex sleep (other version / patch)");
        return;
    }
    g_clock_return = eboot_base + kClockReturnVa;
    BB::Upscale::Checkpoint(fmt::format(
        "[reflex] game frame start hook armed (frame limiter 0x2034770, its gettimeofday at "
        "0x2034977){}",
        BB::VarDt::CapPeriod() > 0.0f && Cfg().cap
            ? fmt::format("; BB_GAME_FPS_CAP {:.0f} fps goes to Reflex's minimum interval, the "
                          "game's limiter at 1 ms while Reflex sleeps",
                          1.0f / BB::VarDt::CapPeriod())
            : std::string()));
}

void OnGuestClock(const void* return_address) {
    const uintptr_t hook = g_clock_return.load(std::memory_order_relaxed);
    if (hook == 0 || reinterpret_cast<uintptr_t>(return_address) != hook) return;
    FrameStart();
}

void OnGuestSubmit() {
    if (g_render_open.load(std::memory_order_relaxed) || !Active()) return;
    uint64_t id = 0, sim_end = 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (g_render_id != 0) return;
        id = OpenRenderFrame(sim_end);
    }
    PostMarker(sim_end, VK_LATENCY_MARKER_SIMULATION_END_NV);
    PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
}

void OnGuestFlip() {
    if (!Active()) return;
    uint64_t id = 0, sim_end = 0;
    bool opened = false;
    {
        std::scoped_lock lock{g_frames_mutex};
        opened = g_render_id == 0;
        id = OpenRenderFrame(sim_end);
        while (!g_sim.empty() && g_sim.front() <= id) g_sim.pop_front();
        g_render_id = 0;
        g_render_open.store(false, std::memory_order_relaxed);
        PushBounded(g_gpu, id, 8);
    }
    PostMarker(sim_end, VK_LATENCY_MARKER_SIMULATION_END_NV);
    if (opened) PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
    PostMarker(id, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
}

void OnGuestFlipDropped() {
    if (!Active()) return;
    std::scoped_lock lock{g_frames_mutex};
    if (!g_gpu.empty()) g_gpu.pop_back();
}

uint64_t OnGpuFlip() {
    if (!Active()) return 0;
    std::scoped_lock lock{g_frames_mutex};
    if (g_gpu.empty()) return 0;
    const uint64_t id = g_gpu.front();
    g_gpu.pop_front();
    return id;
}

uint64_t SubmitPresentId() {
    if (!Active()) return 0;
    std::scoped_lock lock{g_frames_mutex};
    return !g_gpu.empty() ? g_gpu.front() : g_render_id;
}

uint64_t BeginPresent(uint64_t id) {
    if (id == 0 || !Active()) return 0;
    {
        std::scoped_lock lock{g_frames_mutex};
        if (id <= g_last_present_id) return 0;
    }
    Marker(id, VK_LATENCY_MARKER_PRESENT_START_NV);
    return id;
}

void EndPresent(uint64_t id, bool presented) {
    if (id == 0) return;
    Marker(id, VK_LATENCY_MARKER_PRESENT_END_NV);
    if (presented) {
        std::scoped_lock lock{g_frames_mutex};
        g_last_present_id = std::max(g_last_present_id, id);
    }
    if (!BB::FG::Dlss::Ready()) {
        // No [fg-pacing] line without DLSS-G: a [reflex] line every 10 s instead.
        static auto last = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(now - last).count();
        if (sec >= 10.0) {
            last = now;
            BB::Upscale::Checkpoint(fmt::format("[reflex] {:.1f} s: {}", sec,
                                                TakeSecondSummary(sec)));
        }
    }
}

bool Active() {
    return g_ready.load(std::memory_order_acquire) &&
           g_latency_swapchain.load(std::memory_order_relaxed) &&
           !g_disabled.load(std::memory_order_relaxed);
}

std::string TakeSecondSummary(double sec) {
    const uint32_t markers = g_markers.exchange(0);
    const uint32_t sleeps = g_sleeps.exchange(0);
    const uint32_t failed = g_sleep_failed.exchange(0);
    const uint32_t frames = g_frames_started.exchange(0);
    const uint64_t us = g_sleep_us.exchange(0);
    if (!Requested()) {
        return fmt::format("reflex markers sent 0 (Reflex off: {})", Cfg().why);
    }
    if (!Active()) {
        return fmt::format(
            "reflex markers sent {} (Reflex asked but not active: {})", markers,
            !g_off_reason.empty() ? g_off_reason
            : g_disabled          ? std::string("turned off after an error, see [reflex] lines")
            : !g_ready            ? std::string("VK_NV_low_latency2 not set up")
                                  : std::string("no latency swapchain (minimized?)"));
    }
    return fmt::format(
        "reflex markers sent {} ({:.0f}/s) | reflex sleeps {} (avg {:.2f} ms, {} failed) at {} "
        "game frame starts, {}, cap {}",
        markers, sec > 0.0 ? markers / sec : 0.0, sleeps, sleeps ? us / 1e3 / sleeps : 0.0,
        failed, frames, Cfg().boost ? "boost" : "no boost",
        g_cap_owned ? fmt::format("{:.0f} fps by Reflex", 1e6 / g_applied_interval_us.load())
                    : std::string("not Reflex's"));
}

} // namespace BB::Reflex
