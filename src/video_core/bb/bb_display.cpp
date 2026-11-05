// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The display meter (bb_display.h).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include "video_core/bb/bb_display.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_kyty_compat.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace BB::FG::Display {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kSettlePresents = 6;
constexpr uint64_t kWaitTimeoutNs = 50'000'000;

struct Group {
    int expected = 0;
    bool skip = false;
    int seen = 0;
};

struct State {
    bool ext_enabled = false;
    bool feat_id = false;
    bool feat_wait = false;
    bool chained = false;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkWaitForPresentKHR wait = nullptr;
    VkSwapchainKHR (*amd_real)(VkSwapchainKHR) = nullptr;
    uint32_t (*amd_take)(uint64_t*, uint32_t) = nullptr;
    VkPhysicalDevicePresentIdFeaturesKHR id_features{};
    VkPhysicalDevicePresentWaitFeaturesKHR wait_features{};

    std::mutex wait_mutex; // held while a wait is in flight: a swapchain is not destroyed under it
    std::mutex mutex;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    Path path = Path::Plain;
    uint32_t epoch = 0;
    int present_mode = -1;
    std::deque<uint64_t> queue;
    uint64_t last_id = 0;
    int next_expected = 0;
    int issued_expected = 0;
    int last_expected = 0;
    int settle = 0;
    uint32_t win_real = 0;
    uint64_t sl_sum_win = 0;
    uint32_t sl_calls_win = 0;
    uint32_t sl_status = 0;
    uint32_t sl_max = 0;
    bool sl_seen = false;
    uint64_t acct_expected = 0;
    uint64_t acct_dropped = 0;
    int64_t sl_balance = 0;
    int64_t sl_reported = 0;
    std::unordered_map<uint64_t, Group> groups;
    // bb-dlssg-077: the Reflex latency report's GPU completion times of the real frames (Streamline
    // keeps them with the driver's clock, not ours): the ms between consecutive ones.
    std::vector<double> gpu_intervals;
    uint64_t gpu_last_id = 0, gpu_last_end = 0;
    bool started = false;
    std::thread thread;
};

State& S() {
    static State* state = new State; // (never destroyed: the thread outlives the statics)
    return *state;
}

bool InChain(const void* head, VkStructureType type) {
    for (auto* node = static_cast<const VkBaseInStructure*>(head); node != nullptr;
         node = node->pNext) {
        if (node->sType == type) return true;
    }
    return false;
}

bool Has(const std::vector<std::string>& available, const char* name) {
    return std::find(available.begin(), available.end(), name) != available.end();
}

const char* PresentModeName(int mode) {
    switch (mode) {
    case 0:
        return "immediate";
    case 1:
        return "mailbox";
    case 2:
        return "fifo";
    case 3:
        return "fifo-relaxed";
    default:
        return "unknown";
    }
}

struct Window {
    Clock::time_point start{};
    Clock::time_point prev{};
    bool have_prev = false;
    std::vector<double> intervals; // ms between consecutive displayed images
    uint32_t done_all = 0, done_real = 0, done_gen = 0;
    uint32_t waits_ok = 0, waits_timeout = 0;
    uint32_t seen_epoch = 0;
};

struct Wait {
    VkResult result = VK_SUCCESS;
    Clock::time_point at{};
};

Wait WaitFor(State& s, Path path, uint64_t id, uint32_t epoch) {
    std::lock_guard wait_lock{s.wait_mutex};
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    {
        std::lock_guard lock{s.mutex};
        if (s.epoch != epoch || s.swapchain == VK_NULL_HANDLE) {
            return {VK_ERROR_SURFACE_LOST_KHR, {}};
        }
        handle = s.swapchain;
    }
    // (FSR 3: the driver swapchain inside the frame interpolation swapchain, which cannot be
    // destroyed while wait_mutex is held.)
    if (path == Path::Amd) {
        handle = s.amd_real != nullptr ? s.amd_real(handle) : VK_NULL_HANDLE;
        if (handle == VK_NULL_HANDLE) return {VK_ERROR_SURFACE_LOST_KHR, {}};
    }
    Wait w;
    w.result = s.wait(s.device, handle, id, kWaitTimeoutNs);
    w.at = Clock::now();
    return w;
}

void Report(State& s, Window& w, Clock::time_point now) {
    const double win = std::chrono::duration<double>(now - w.start).count();
    uint32_t real_issued = 0, sl_calls = 0, sl_status = 0, sl_max = 0;
    uint64_t sl_sum = 0, acct_expected = 0, acct_dropped = 0;
    Path path;
    int mode;
    bool ids;
    {
        std::lock_guard lock{s.mutex};
        real_issued = std::exchange(s.win_real, 0u);
        sl_calls = std::exchange(s.sl_calls_win, 0u);
        sl_sum = std::exchange(s.sl_sum_win, uint64_t{0});
        sl_status = s.sl_status;
        sl_max = s.sl_max;
        acct_expected = s.acct_expected;
        acct_dropped = s.acct_dropped;
        path = s.path;
        mode = s.present_mode;
    }
    ids = IdsAvailable();
    const bool waited = w.done_all > 0;
    std::vector<double> gpu_iv;
    {
        std::lock_guard lock{s.mutex};
        gpu_iv.swap(s.gpu_intervals);
    }
    double shown = -1.0, real = -1.0, generated = -1.0;
    const char* source = "none";
    const char* interval_of = "none";
    if (path == Path::Amd) {
        if (waited) {
            shown = w.done_all / win;
            real = w.done_real / win;
            generated = w.done_gen / win;
            source = "present_wait";
            interval_of = "all shown images";
        }
    } else if (path == Path::Dlss && sl_calls > 0) {
        shown = static_cast<double>(sl_sum) / win;
        real = real_issued / win;
        generated = std::max(shown - real, 0.0);
        source = "sl_state";
        if (waited) {
            const double diff = std::fabs(static_cast<double>(w.done_all) - static_cast<double>(sl_sum));
            if (diff <= 0.03 * static_cast<double>(sl_sum) + 1.0) {
                source = "present_wait";
                interval_of = "all shown images";
            } else {
                interval_of = "real images only";
            }
        }
    } else if (waited) {
        shown = w.done_all / win;
        real = w.done_real / win;
        generated = 0.0;
        source = "present_wait";
        interval_of = "all shown images";
        if (path == Path::Dlss) {
            shown = real = generated = -1.0;
            source = "none";
            interval_of = "real images only";
        }
    }
    // DLSS-G: Streamline presents the images itself and the present ids do not complete through
    // it (waits 0 ok): the interval then comes from the Reflex report, the real frames' GPU
    // completion on the driver's clock, only if it matches the real frame rate (else not used).
    const std::vector<double>* iv = &w.intervals;
    const char* gpu_note = "";
    if (path == Path::Dlss && !waited && gpu_iv.size() >= 5) {
        double sum = 0.0;
        for (double v : gpu_iv) sum += v;
        const double mean = sum / static_cast<double>(gpu_iv.size());
        const double expect = real_issued > 0 ? 1000.0 * win / real_issued : 0.0;
        if (expect > 0.0 && mean > 0.7 * expect && mean < 1.4 * expect) {
            iv = &gpu_iv;
            interval_of = "real frames' GPU completion, Reflex report; not the display";
        } else {
            gpu_note = " [Reflex report clock did not match the real frame rate: not used]";
        }
    }
    if (real_issued == 0 && !waited && sl_calls == 0) {
        w.intervals.clear();
        w.done_all = w.done_real = w.done_gen = 0;
        w.waits_ok = w.waits_timeout = 0;
        w.start = now;
        return;
    }
    char fps[96];
    if (shown >= 0.0) {
        std::snprintf(fps, sizeof(fps), "shown %.1f fps (real %.1f, generated %.1f)", shown, real,
                      generated);
    } else {
        std::snprintf(fps, sizeof(fps), "shown n/a (real %.1f, generated n/a)", real_issued / win);
    }
    char interval[200];
    if ((*iv).size() >= 5) {
        std::vector<double> sorted = (*iv);
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0, sq = 0.0;
        for (double v : sorted) {
            sum += v;
            sq += v * v;
        }
        const double n = static_cast<double>(sorted.size());
        const double mean = sum / n;
        const double sd = std::sqrt(std::max(sq / n - mean * mean, 0.0));
        const size_t idx = std::min(sorted.size() - 1,
                                    static_cast<size_t>(std::ceil(0.99 * n)) - 1);
        std::snprintf(interval, sizeof(interval),
                      "interval mean %.2f ms, sd %.2f ms, p99 %.2f ms, max %.1f ms (%zu, %s)", mean,
                      sd, sorted[idx], sorted.back(), sorted.size(), interval_of);
    } else {
        std::snprintf(interval, sizeof(interval), "interval n/a (%zu measured)", (*iv).size());
    }
    char why[160]{};
    if (std::strcmp(source, "none") == 0) {
        std::snprintf(why, sizeof(why), " [%s]",
                      !ids ? "present ids / VK_KHR_present_wait not available on this device"
                           : path == Path::Dlss
                                 ? "waits did not complete and Streamline gave no state"
                                 : "no present completed");
    }
    char sl[96]{};
    if (sl_calls > 0) {
        std::snprintf(sl, sizeof(sl), ", Streamline status 0x%x, up to %u generated", sl_status,
                      sl_max);
    }
    if (path == Path::Dlss && !waited && w.waits_timeout > 0 && w.waits_ok == 0) {
        static bool said = false;
        if (!said) {
            said = true;
            Printf("[display] DLSS-G: none of the present ids chained to the game's presents completed "
                   "(%u timeouts, 0 ok): Streamline presents through its own swapchain and the id does "
                   "not reach the driver, so there is no display clock here; shown/real/generated come "
                   "from sl::DLSSGState, the interval from the Reflex report when it matches",
                   w.waits_timeout);
        }
    }
    Printf("[display] %s, %s, dropped %llu (%.2f%% of %llu expected), source=%s%s, present mode %s, "
           "waits %u ok / %u timeout%s",
           fps, interval, static_cast<unsigned long long>(acct_dropped),
           acct_expected != 0 ? 100.0 * static_cast<double>(acct_dropped) /
                                    static_cast<double>(acct_expected)
                              : 0.0,
           static_cast<unsigned long long>(acct_expected), source, why, PresentModeName(mode),
           w.waits_ok, w.waits_timeout, sl);
    if (gpu_note[0] != '\0') Printf("[display]%s", gpu_note);
    w.intervals.clear();
    w.done_all = w.done_real = w.done_gen = 0;
    w.waits_ok = w.waits_timeout = 0;
    w.start = now;
}

void Record(State& s, Window& w, uint64_t id, Path path, Clock::time_point at) {
    if (w.have_prev) {
        const double ms = std::chrono::duration<double, std::milli>(at - w.prev).count();
        if (ms >= 0.0 && ms < 250.0) w.intervals.push_back(ms);
    }
    w.prev = at;
    w.have_prev = true;
    ++w.done_all;
    if (path != Path::Amd) {
        ++w.done_real;
        return;
    }
    const uint64_t group = id >> 2;
    std::lock_guard lock{s.mutex};
    if ((id & 3u) != 3u) {
        ++w.done_gen;
        ++s.groups[group].seen;
        return;
    }
    ++w.done_real;
    const auto it = s.groups.find(group);
    if (it != s.groups.end()) {
        if (!it->second.skip && it->second.expected > 0) {
            s.acct_expected += static_cast<uint64_t>(it->second.expected);
            if (it->second.seen < it->second.expected) {
                s.acct_dropped += static_cast<uint64_t>(it->second.expected - it->second.seen);
            }
        }
        s.groups.erase(it);
    }
    // (groups whose real image never came)
    while (s.groups.size() > 64) s.groups.erase(s.groups.begin());
}

void Run() {
    State& s = S();
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    Window w;
    w.start = Clock::now();
    uint32_t consecutive_errors = 0;
    bool broken = false;
    for (;;) {
        VkSwapchainKHR app = VK_NULL_HANDLE;
        Path path = Path::Plain;
        uint32_t epoch = 0;
        uint64_t ids[64];
        uint32_t n = 0;
        {
            std::lock_guard lock{s.mutex};
            app = s.swapchain;
            path = s.path;
            epoch = s.epoch;
            if (path != Path::Amd) {
                while (n < 64 && !s.queue.empty()) {
                    ids[n++] = s.queue.front();
                    s.queue.pop_front();
                }
            }
        }
        if (w.seen_epoch != epoch) {
            w.seen_epoch = epoch;
            w.have_prev = false;
            w.intervals.clear();
            broken = false;
            consecutive_errors = 0;
        }
        if (path == Path::Amd && app != VK_NULL_HANDLE && s.amd_take != nullptr) {
            n = s.amd_take(ids, 64);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (broken || s.wait == nullptr || app == VK_NULL_HANDLE) continue;
            Wait r;
            for (int attempt = 0; attempt < 4; ++attempt) {
                r = WaitFor(s, path, ids[i], epoch);
                if (r.result != VK_TIMEOUT) break;
                ++w.waits_timeout;
            }
            if (r.result == VK_SUCCESS || r.result == VK_SUBOPTIMAL_KHR) {
                consecutive_errors = 0;
                ++w.waits_ok;
                Record(s, w, ids[i], path, r.at);
            } else if (r.result != VK_TIMEOUT && r.result != VK_ERROR_SURFACE_LOST_KHR) {
                if (++consecutive_errors >= 5) {
                    broken = true;
                    Printf("[display] vkWaitForPresentKHR returned %d five times in a row: waiting "
                           "for presents is off for this swapchain (source=none)",
                           static_cast<int>(r.result));
                }
            }
        }
        const auto now = Clock::now();
        if (now - w.start >= std::chrono::seconds(1)) Report(s, w, now);
        if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace

bool Wanted() {
    static const bool on = [] {
        const char* v = std::getenv("BB_DISPLAY");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return on;
}

void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    if (!Wanted()) return;
    State& s = S();
    if (!Has(available, VK_KHR_PRESENT_ID_EXTENSION_NAME) ||
        !Has(available, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
        BBLogf("[display] VK_KHR_present_id / VK_KHR_present_wait not offered by this device: the "
               "display clock is off (source=none)");
        return;
    }
    for (const char* name : {VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_KHR_PRESENT_WAIT_EXTENSION_NAME}) {
        const bool on = std::any_of(extensions.begin(), extensions.end(),
                                    [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (!on) {
            extensions.push_back(name);
            BBLogf("Frame generation / latency: device extension %s added (display clock)", name);
        }
    }
    s.ext_enabled = true;
}

void QueryFeatures(vk::PhysicalDevice physical) {
    State& s = S();
    if (!Wanted() || !s.ext_enabled) return;
    VkPhysicalDevicePresentIdFeaturesKHR id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
    VkPhysicalDevicePresentWaitFeaturesKHR wait{};
    wait.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
    id.pNext = &wait;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &id;
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2 == nullptr) {
        BBLogf("[display] vkGetPhysicalDeviceFeatures2 unavailable: the display clock is off");
        return;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2(static_cast<VkPhysicalDevice>(physical),
                                                              &features);
    s.feat_id = id.presentId == VK_TRUE;
    s.feat_wait = wait.presentWait == VK_TRUE;
    BBLogf("[display] device features: presentId %s, presentWait %s", s.feat_id ? "yes" : "no",
           s.feat_wait ? "yes" : "no");
}

void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next) {
    State& s = S();
    const auto enabled = [&](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [&](const char* e) { return std::strcmp(e, name) == 0; });
    };
    if (!Wanted() || !s.ext_enabled || !s.feat_id || !s.feat_wait ||
        !enabled(VK_KHR_PRESENT_ID_EXTENSION_NAME) || !enabled(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
        return next;
    }
    if (!InChain(next, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR)) {
        s.id_features = {};
        s.id_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
        s.id_features.presentId = VK_TRUE;
        s.id_features.pNext = next;
        next = &s.id_features;
    }
    if (!InChain(next, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR)) {
        s.wait_features = {};
        s.wait_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
        s.wait_features.presentWait = VK_TRUE;
        s.wait_features.pNext = next;
        next = &s.wait_features;
    }
    s.chained = true;
    return next;
}

void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc) {
    State& s = S();
    if (!Wanted() || !s.chained || get_proc == nullptr) return;
    s.device = static_cast<VkDevice>(device);
    s.wait = reinterpret_cast<PFN_vkWaitForPresentKHR>(get_proc(s.device, "vkWaitForPresentKHR"));
    BBLogf("[display] display clock: VK_KHR_present_id + VK_KHR_present_wait %s",
           s.wait != nullptr ? "on" : "enabled but vkWaitForPresentKHR was not found");
}

void SetAmdHooks(VkSwapchainKHR (*real_swapchain)(VkSwapchainKHR),
                 uint32_t (*take_ids)(uint64_t*, uint32_t)) {
    State& s = S();
    s.amd_real = real_swapchain;
    s.amd_take = take_ids;
}

bool IdsAvailable() {
    const State& s = S();
    return Wanted() && s.chained && s.wait != nullptr;
}

void NotePresentMode(int mode) {
    State& s = S();
    std::lock_guard lock{s.mutex};
    s.present_mode = mode;
}

void OnSwapchain(vk::SwapchainKHR swapchain, Path path) {
    if (!Wanted()) return;
    State& s = S();
    {
        std::lock_guard wait_lock{s.wait_mutex};
        std::lock_guard lock{s.mutex};
        s.swapchain = static_cast<VkSwapchainKHR>(swapchain);
        s.path = path;
        ++s.epoch;
        s.queue.clear();
        s.groups.clear();
    }
    if (swapchain && !s.started) {
        s.started = true;
        s.thread = std::thread(Run);
    }
}

void SetExpected(int generated) {
    State& s = S();
    std::lock_guard lock{s.mutex};
    s.next_expected = std::max(generated, 0);
}

uint64_t Issue(uint64_t preferred) {
    if (!Wanted()) return 0;
    State& s = S();
    std::lock_guard lock{s.mutex};
    if (s.swapchain == VK_NULL_HANDLE) return 0;
    ++s.win_real;
    s.issued_expected = s.next_expected;
    if (s.next_expected != s.last_expected) {
        s.last_expected = s.next_expected;
        s.settle = kSettlePresents;
    }
    if (!(s.chained && s.wait != nullptr)) return 0;
    const uint64_t id = std::max(preferred, s.last_id + 1);
    s.last_id = id;
    if (s.path == Path::Amd) {
        Group& g = s.groups[id];
        g.expected = s.next_expected;
        g.skip = s.settle > 0;
        if (s.settle > 0) --s.settle;
    } else {
        s.queue.push_back(id);
        if (s.queue.size() > 256) s.queue.pop_front();
    }
    return id;
}

void NoteGpuEnds(const uint64_t* frame_ids, const uint64_t* end_us, uint32_t n) {
    if (!Wanted() || n == 0) return;
    // (ordered by frame id; the report is a ring of 64 frames, polled every ~40 presents)
    std::vector<std::pair<uint64_t, uint64_t>> frames;
    frames.reserve(n);
    for (uint32_t i = 0; i < n; ++i) frames.emplace_back(frame_ids[i], end_us[i]);
    std::sort(frames.begin(), frames.end());
    State& s = S();
    std::lock_guard lock{s.mutex};
    for (const auto& f : frames) {
        if (f.first <= s.gpu_last_id) continue;
        if (s.gpu_last_end != 0 && f.second > s.gpu_last_end) {
            const double ms = static_cast<double>(f.second - s.gpu_last_end) / 1000.0;
            if (ms < 250.0) s.gpu_intervals.push_back(ms);
        }
        s.gpu_last_id = f.first;
        s.gpu_last_end = f.second;
    }
    if (s.gpu_intervals.size() > 4096) s.gpu_intervals.erase(s.gpu_intervals.begin(), s.gpu_intervals.begin() + 2048);
}

void NoteStreamline(uint32_t presented, uint32_t status, uint32_t generate_max) {
    if (!Wanted()) return;
    State& s = S();
    std::lock_guard lock{s.mutex};
    s.sl_sum_win += presented;
    ++s.sl_calls_win;
    s.sl_status = status;
    s.sl_max = generate_max;
    s.sl_seen = true;
    if (s.path != Path::Dlss || s.issued_expected <= 0) return;
    if (s.settle > 0) {
        --s.settle;
        return;
    }
    const int64_t expected = 1 + s.issued_expected;
    s.acct_expected += static_cast<uint64_t>(expected);
    s.sl_balance += expected - static_cast<int64_t>(presented);
    const int64_t deficit = s.sl_balance - expected; // (one present of Streamline's lag)
    if (deficit > s.sl_reported) {
        s.acct_dropped += static_cast<uint64_t>(deficit - s.sl_reported);
        s.sl_reported = deficit;
    }
}

const char* FormatLabel(uint32_t vk_format, bool* known_to_streamline) {
    struct Entry {
        uint32_t format;
        const char* name;
        bool known;
    };
    static const Entry table[] = {
        {0, "UNDEFINED", false},
        {9, "R8_UNORM", false},
        {13, "R8_UINT", true},
        {37, "R8G8B8A8_UNORM", true},
        {38, "R8G8B8A8_SNORM", true},
        {43, "R8G8B8A8_SRGB", true},
        {44, "B8G8R8A8_UNORM", true},
        {50, "B8G8R8A8_SRGB", true},
        {58, "A2R10G10B10_UNORM_PACK32", false},
        {64, "A2B10G10R10_UNORM_PACK32", true},
        {74, "R16_UINT", true},
        {76, "R16_SFLOAT", true},
        {77, "R16G16_UNORM", true},
        {83, "R16G16_SFLOAT", true},
        {90, "R16G16B16_SFLOAT", true},
        {97, "R16G16B16A16_SFLOAT", true},
        {98, "R32_UINT", true},
        {100, "R32_SFLOAT", true},
        {101, "R32G32_UINT", true},
        {103, "R32G32_SFLOAT", true},
        {106, "R32G32B32_SFLOAT", true},
        {109, "R32G32B32A32_SFLOAT", true},
        {122, "B10G11R11_UFLOAT_PACK32", true},
        {126, "D32_SFLOAT", true},
        {129, "D24_UNORM_S8_UINT", true},
        {130, "D32_SFLOAT_S8_UINT", true},
    };
    for (const Entry& e : table) {
        if (e.format == vk_format) {
            if (known_to_streamline != nullptr) *known_to_streamline = e.known;
            return e.name;
        }
    }
    if (known_to_streamline != nullptr) *known_to_streamline = false;
    return "other";
}

} // namespace BB::FG::Display
