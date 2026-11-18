// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-fg: BB_FG settings and the swapchain / device hooks (bb_fg.h), from the 1.59
// build's bb_fg.cpp reduced to AMD FSR 3 x2.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace BB::FG {

const char* Env(const char* kyty_name) {
    if (kyty_name == nullptr) {
        return nullptr;
    }
    const std::string_view name{kyty_name};
    if (name.starts_with("KYTY_")) {
        const std::string bb = "BB_" + std::string(name.substr(5));
        if (const char* v = std::getenv(bb.c_str()); v != nullptr) {
            return v;
        }
    }
    return std::getenv(kyty_name);
}

void Printf(const char* format, ...) {
    char text[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    std::size_t n = std::strlen(text);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r')) {
        text[--n] = '\0';
    }
    BBLogf("%s", text);
}

void NotePacing(double, double) {}

namespace {

#if defined(_WIN32)
void SetProcessEnv(const char* name, const char* value) {
    _putenv_s(name, value);
    SetEnvironmentVariableA(name, value);
}
#else
void SetProcessEnv(const char* name, const char* value) {
    setenv(name, value, 1);
}
#endif

// Generated frames per real one this present (KYTY_FSR3_FG_ACTIVE): x(N+1).
std::atomic<int> g_amd_active{1};

void WriteAmdEnv(int frames, int active) {
    char f[8], a[8];
    std::snprintf(f, sizeof(f), "%d", frames);
    std::snprintf(a, sizeof(a), "%d", active);
    SetProcessEnv("KYTY_FSR3_FG_FRAMES", f); // read by the DLL when it makes the swapchain
    SetProcessEnv("KYTY_FSR3_FG_ACTIVE", a); // read on every present
}

Config ParseConfig() {
    Config c{};
    const char* v = std::getenv("BB_FG");
    std::string s = v != nullptr ? v : "off";
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
    if (s == "amd-x2" || s == "amd-2" || s == "amd" || s == "fsr3" || s == "fsr3-x2" ||
        s == "fsr3-2" || s == "fsr-x2") {
        c.kind = Kind::Amd;
        c.multiplier = 2;
    } else if (s == "amd-x3" || s == "amd-3") {
        // (11-12) AMD multi frame generation as in Demon's Souls: the patched FidelityFX DLL
        // interpolates N-1 frames at t = k/N and paces them (KYTY_FSR3_FG_FRAMES / _ACTIVE).
        c.kind = Kind::Amd;
        c.multiplier = 3;
    } else if (s == "amd-x4" || s == "amd-4") {
        c.kind = Kind::Amd;
        c.multiplier = 4;
    } else if (s == "amd-dyn" || s == "amd-dinamico" || s == "amd-dynamic") {
        c.kind = Kind::Amd;
        c.multiplier = 4;
        c.dynamic = true;
    } else if (s == "dlss-x2" || s == "dlss-2" || s == "dlss" || s == "dlssg") {
        c.kind = Kind::Dlss;
        c.multiplier = 2;
    } else if (s == "dlss-x3" || s == "dlss-3") {
        c.kind = Kind::Dlss;
        c.multiplier = 3;
    } else if (s == "dlss-x4" || s == "dlss-4") {
        c.kind = Kind::Dlss;
        c.multiplier = 4;
    } else if (s == "dlss-dyn" || s == "dlss-dynamic" || s == "dlssg-dyn") {
        c.kind = Kind::Dlss;
        c.multiplier = 4;
        c.dynamic = true;
    } else if (s == "dlss-sl" || s == "dlss-sl-x2" || s == "dlssg-sl") {
        c.kind = Kind::Dlss;
        c.multiplier = 2;
        c.streamline = true;
    } else if (s == "dlss-sl-x3") {
        c.kind = Kind::Dlss;
        c.multiplier = 3;
        c.streamline = true;
    } else if (s == "dlss-sl-x4") {
        c.kind = Kind::Dlss;
        c.multiplier = 4;
        c.streamline = true;
    } else if (s == "dlss-sl-dyn") {
        c.kind = Kind::Dlss;
        c.multiplier = 4;
        c.dynamic = true;
        c.streamline = true;
    } else if (s != "off" && s != "0" && !s.empty()) {
        BBLogf("Frame generation: BB_FG=%s is not available in this build (amd-x2, dlss-x2..x4, "
               "dlss-dyn): off",
               s.c_str());
    }
    if (c.kind == Kind::Dlss && !c.streamline) {
        const char* sl = std::getenv("BB_DLSSG_SL");
        c.streamline = sl != nullptr && sl[0] == '1';
    }
    if (c.kind == Kind::Amd) {
        SetProcessEnv("KYTY_FSR3_FG", "1");
        g_amd_active.store(c.dynamic ? 1 : c.multiplier - 1, std::memory_order_relaxed);
        WriteAmdEnv(c.multiplier - 1, g_amd_active.load(std::memory_order_relaxed));
    }
    return c;
}

} // namespace

// bb-fsr3-mfg-mult: the generated frames per real one the DLL is told (KYTY_FSR3_FG_FRAMES at
// swapchain creation, KYTY_FSR3_FG_ACTIVE on every present). Kept here so the FG swapchain's
// creation (bb_fg_fsr3.cpp, which clears both first) can give them back with the right values.
int AmdFrames() {
    const auto& c = Cfg();
    return c.kind == Kind::Amd ? std::clamp(c.multiplier, 2, 4) - 1 : 1;
}

int AmdActiveFrames() {
    return std::clamp(g_amd_active.load(std::memory_order_relaxed), 1, AmdFrames());
}

void ApplyAmdEnv() {
    if (Cfg().kind != Kind::Amd) return;
    // A SetAmdActive on another thread between the load and the write would be overwritten with
    // the older value until the next dynamic change: write again until the value read is stable.
    for (int active = AmdActiveFrames();;) {
        WriteAmdEnv(AmdFrames(), active);
        const int now = AmdActiveFrames();
        if (now == active) break;
        active = now;
    }
}

void SetAmdActive(int multiplier) {
    g_amd_active.store(std::clamp(multiplier, 2, 4) - 1, std::memory_order_relaxed);
    char active[8];
    std::snprintf(active, sizeof(active), "%d", std::clamp(multiplier, 2, 4) - 1);
    SetProcessEnv("KYTY_FSR3_FG_ACTIVE", active);
}

const Config& Cfg() {
    static const Config config = [] {
        Config c = ParseConfig();
        if (c.kind == Kind::Amd) {
            if (c.dynamic) BBLogf("Frame generation: BB_FG=%s (AMD FSR 3, dynamic x2..x4)", std::getenv("BB_FG"));
            else BBLogf("Frame generation: BB_FG=%s (AMD FSR 3, fixed x%d)", std::getenv("BB_FG"),
                        c.multiplier);
        } else if (c.kind == Kind::Dlss) {
            BBLogf("Frame generation: BB_FG=%s (DLSS-G via %s, %s)", std::getenv("BB_FG"),
                   c.streamline ? "Streamline sl.dlss_g" : "NGX",
                   c.dynamic ? "dynamic up to x4" : (c.multiplier == 2   ? "x2"
                                                     : c.multiplier == 3 ? "x3"
                                                                         : "x4"));
        }
        return c;
    }();
    return config;
}

const char* Name() {
    const auto& c = Cfg();
    if (c.kind == Kind::Amd) {
        if (c.dynamic) return "amd-dyn";
        return c.multiplier == 2 ? "amd-x2" : c.multiplier == 3 ? "amd-x3" : "amd-x4";
    }
    if (c.kind == Kind::Dlss) {
        if (c.dynamic) return "dlss-dyn";
        return c.multiplier == 2 ? "dlss-x2" : c.multiplier == 3 ? "dlss-x3" : "dlss-x4";
    }
    return "off";
}

bool On() {
    return Cfg().kind != Kind::Off;
}

bool FgBlocked(const char** why) {
    if (why != nullptr) *why = nullptr;
    return false;
}

bool Fsr3Requested() {
    return Cfg().kind == Kind::Amd && Amd::Requested();
}

bool Fsr3Active() {
    return Cfg().kind == Kind::Amd && Amd::Active();
}

// ---- bb-fg-cap (bb_fg.h) ----

double DisplayRefreshHz() {
    return Amd::DisplayHz(); // QueryDisplayConfig (exact rational rate), the game window's display
}

void RefreshDisplayHz() {
    if (On()) Amd::UpdateDisplayHz();
}

double OutputCapFps() {
    const double hz = DisplayRefreshHz();
    if (!(hz > 30.0)) return 0.0;
    const double cap = hz - hz * hz / 3600.0; // 240 -> 224, 144 -> 138.2, 120 -> 116, 60 -> 59
    static std::atomic<double> logged_hz{0.0};
    if (std::fabs(logged_hz.exchange(hz) - hz) > 0.001) {
        Printf("[fg-cap] display refresh %.3f Hz (measured at run time): shown frames kept under "
               "%.1f fps (refresh - refresh^2/3600 = refresh x %.3f, the VRR margin)",
               hz, cap, cap / hz);
    }
    return cap;
}

namespace {
double EnvNumber(const char* name, double fallback, bool* off = nullptr) {
    const char* v = std::getenv(name);
    if (off != nullptr) *off = false;
    if (v == nullptr || *v == '\0') return fallback;
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
    if (s == "auto" || s == "on" || s == "1") return fallback;
    if (s == "0" || s == "off") {
        if (off != nullptr) *off = true;
        return 0.0;
    }
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    return end != v ? d : fallback;
}
} // namespace

int FixedRealCapFps(int multiplier, const char* who) {
    bool off = false;
    const double forced = EnvNumber("BB_FG_FIXED_CAP", 0.0, &off);
    const int n = std::clamp(multiplier, 2, 4);
    const double hz = DisplayRefreshHz();
    const double cap = OutputCapFps();
    int real = 0;
    const char* how = "";
    if (off) {
        how = "BB_FG_FIXED_CAP=0: no cap";
    } else if (forced >= 10.0) {
        real = static_cast<int>(forced);
        how = "BB_FG_FIXED_CAP";
    } else if (cap > 0.0) {
        real = static_cast<int>(std::floor(cap / n + 1e-9));
        how = "floor(output cap / multiplier)";
    } else {
        how = "display refresh unknown: no cap";
    }
    static std::atomic<int> logged{-1};
    const int key = real * 8 + n;
    if (logged.exchange(key) != key) {
        Printf("[%s] x%d: real fps capped to %d (output cap %.0f = refresh %.3f Hz - "
               "refresh^2/3600; %s; BB_FG_FIXED_CAP=0 off, =<fps> forced)",
               who, n, real, cap, hz, how);
    }
    return real;
}

bool DynamicFg::Update(const char* who, double real, int max_mult) {
    max_mult = std::clamp(max_mult, 2, 4);
    cur = std::clamp(cur, 2, max_mult);
    double cap = OutputCapFps();
    if (!(cap > 0.0)) cap = 59.0; // (60 Hz assumed)
    static const double floor_fps = std::max(EnvNumber("BB_FG_DYN_FLOOR", 50.0), 1.0);
    static const bool hold_on = [] {
        const char* v = std::getenv("BB_FG_DYN_HOLD");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    const bool hold = hold_on && allow_hold;
    const auto hold_of = [&](int n) { return static_cast<int>(std::floor(cap / n + 1e-9)); };
    // Preferred: the largest N whose held real rate stays at or above the floor (x4 at 56 real on
    // a 240 Hz display; x2 at 69 on 144 Hz, where x3 / x4 would hold the game under 50).
    int base = 2;
    for (int n = max_mult; n >= 2; --n) {
        if (hold_of(n) >= floor_fps) {
            base = n;
            break;
        }
    }
    // Fits: the largest N with real x N within the cap as the game runs now.
    int fit = 2;
    for (int n = max_mult; n >= 2; --n) {
        if (real * n <= cap) {
            fit = n;
            break;
        }
    }
    const bool pinned = hold && hold_fps > 0 && real >= 0.97 * hold_fps;
    int want = fit;
    const char* why = "fit";
    if (hold) {
        if (pinned && cur > base) {
            // held at this N's rate, so the game can do more: one step towards the preferred N
            want = cur - 1;
            why = "held, game faster: towards the preferred";
        } else if (real >= 0.95 * hold_of(base)) {
            want = std::max(base, fit);
            why = want == base ? "game reaches the preferred hold" : "fit";
        }
    }
    // Down at once (real x N over the cap drops generated frames); up after 2 agreeing seconds.
    agree = want == pending ? agree + 1 : 0;
    pending = want;
    const int before = cur;
    if (want < cur || (want > cur && agree >= 1)) cur = want;
    hold_fps = hold ? hold_of(cur) : 0;
    Printf("[%s] dynamic: real %.1f fps%s, output cap %.1f (refresh %.3f Hz): preferred x%d (hold "
           "%d >= floor %.0f), fits x%d -> want x%d (%s) -> x%d%s, real held at %d fps%s",
           who, real, pinned ? " (at the hold)" : "", cap, DisplayRefreshHz(), base, hold_of(base),
           floor_fps, fit, want, why, cur, before != cur ? " (changed)" : "", hold_fps,
           hold ? "" : " (no hold)");
    return before != cur;
}

void RealLimiter::Wait(double fps) {
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (!(fps > 0.0)) return; // (a later call starts from its own time: next is in the past)
    const double step = 1.0 / fps;
    if (next > now) {
        const double left = next - now;
        if (left > 0.002) {
            std::this_thread::sleep_for(std::chrono::duration<double>(left - 0.0015));
        }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                   .count() < next) {
            std::this_thread::yield();
        }
        next += step;
    } else {
        // late (or the first): from now, with at most one step of credit kept
        next = std::max(next, now - step) + step;
    }
}

static std::atomic<int> g_dlss_mult{0};

void NoteDlssMultiplier(int multiplier) {
    g_dlss_mult.store(multiplier, std::memory_order_relaxed);
}

int PanelMultiplier() {
    const auto& c = Cfg();
    if (c.kind == Kind::Amd) return Fsr3Requested() ? Amd::Multiplier() : 0;
    if (c.kind == Kind::Dlss && DlssSl::Requested()) {
        DlssSl::SecondStats st{};
        return DlssSl::Stats(st) ? static_cast<int>(st.multiplier) : 0;
    }
    if (c.kind == Kind::Dlss) return g_dlss_mult.load(std::memory_order_relaxed);
    return 0;
}

static std::atomic<float> g_dlss_real{0.0f}, g_dlss_shown{0.0f};

void NoteDlssPanel(float real, float shown) {
    g_dlss_real.store(real, std::memory_order_relaxed);
    g_dlss_shown.store(shown, std::memory_order_relaxed);
}

const char* PanelName() {
    // bb-fg-cap: was "FSR 3 x2" for every AMD mode (amd-x3 / x4 / dyn shown as x2), and DLSS-G
    // without its multiplier. Now the effective multiplier (dyn: the current one).
    thread_local char name[48];
    const auto& c = Cfg();
    const int m = PanelMultiplier();
    const char* base = c.kind == Kind::Dlss ? (DlssSl::Requested() ? "DLSS-G (Streamline)" : "DLSS-G")
                                            : "FSR 3";
    if (m >= 2) {
        std::snprintf(name, sizeof(name), "%s %sx%d", base, c.dynamic ? "dyn " : "", m);
    } else {
        std::snprintf(name, sizeof(name), "%s %s", base,
                      c.dynamic ? "dyn" : (c.multiplier == 2 ? "x2" : c.multiplier == 3 ? "x3" : "x4"));
    }
    return name;
}

bool PanelInfo(float& real, float& shown, const char*& state) {
    real = shown = 0;
    state = "off";
    if (!On()) return false;
    if (Cfg().kind == Kind::Dlss && DlssSl::Requested()) {
        // bb-sl-metrics: DLSS-G through Streamline (the NGX bridge is not used then).
        DlssSl::SecondStats st{};
        if (!DlssSl::Ready()) {
            state = "DLSS-G (Streamline) unavailable";
            return true;
        }
        if (!DlssSl::Stats(st)) {
            state = "starting";
            return true;
        }
        real = st.real_fps;
        shown = st.shown_fps;
        state = st.on ? "on" : "idle";
        return true;
    }
    if (Cfg().kind == Kind::Dlss) {
        if (!Dlss::Ready()) {
            state = "DLSS-G unavailable";
            return true;
        }
        real = g_dlss_real.load(std::memory_order_relaxed);
        shown = g_dlss_shown.load(std::memory_order_relaxed);
        state = shown > real * 1.2f ? "on" : "idle";
        return true;
    }
    if (!Fsr3Requested()) {
        state = "FidelityFX DLL not loaded";
        return true;
    }
    Amd::PanelValues(real, shown, state);
    return true;
}

bool WantInputs() {
    return Fsr3Active() || Dlss::Ready() || DlssSl::Ready();
}

bool WantsUiLayer() {
    return Cfg().kind == Kind::Amd && Amd::WantsUiLayer();
}

void SetUiLayer(const Image* ui) {
    if (Cfg().kind == Kind::Amd) Amd::SetUiLayer(ui);
}

namespace {
bool Has(const std::vector<std::string>& available, const char* name) {
    return std::find(available.begin(), available.end(), name) != available.end();
}
bool Enabled(const std::vector<const char*>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](const char* e) { return std::strcmp(e, name) == 0; });
}
} // namespace

void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    if (Cfg().kind != Kind::Amd || !Amd::Requested()) return;
    // The FidelityFX backend calls vkGetBufferMemoryRequirements2KHR by name.
    for (const char* name : {VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
                             VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME}) {
        if (Has(available, name) && !Enabled(extensions, name)) {
            extensions.push_back(name);
            BBLogf("Frame generation: device extension %s added", name);
        }
    }
}

QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family) {
    if (DlssSl::Requested()) {
        // bb-dlssg-sl: Streamline's extra graphics + compute queues, all in the graphics family
        // after ours (index 0); vk_instance.cpp creates up to 4 there.
        uint32_t g = 0, c = 0;
        DlssSl::Queues(&g, &c);
        QueuePlan plan{};
        uint32_t family_max = 1;
        const auto families = physical.getQueueFamilyProperties();
        if (graphics_family < families.size()) family_max = families[graphics_family].queueCount;
        plan.graphics_count = std::min<uint32_t>({1u + g + c, 4u, family_max});
        SR::Dlss::SetQueues(graphics_family, plan.graphics_count);
        BBLogf("DLSS-G (Streamline): %u graphics queue(s) in family %u (Streamline wants %u "
               "graphics + %u compute)",
               plan.graphics_count, graphics_family, g, c);
        return plan;
    }
    if (Cfg().kind != Kind::Amd) return {};
    return Amd::PlanQueues(physical, graphics_family);
}

void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t graphics_family,
              vk::Queue graphics_queue, const QueuePlan& plan, std::mutex* submit_mutex) {
    if (Cfg().kind != Kind::Amd) return;
    Amd::OnDevice(physical, device, graphics_family, graphics_queue, plan, submit_mutex);
}

bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out) {
    if (Cfg().kind != Kind::Amd) return false;
    return Amd::CreateSwapchain(info, out);
}

void DestroySwapchain(vk::SwapchainKHR) {
    Amd::DestroySwapchain();
}

vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
    return Amd::GetSwapchainImages(swapchain, count, images);
}

vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
    return Amd::AcquireNextImage(swapchain, semaphore, index);
}

vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info) {
    return Amd::QueuePresent(queue, info);
}

void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format) {
    if (Fsr3Active()) {
        Amd::BeforePresent(inputs, swapchain, backbuffer, format);
    }
}

} // namespace BB::FG
