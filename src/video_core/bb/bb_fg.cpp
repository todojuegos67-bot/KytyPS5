// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-fg: BB_FG settings and the swapchain / device hooks (bb_fg.h), from the 1.59
// build's bb_fg.cpp reduced to AMD FSR 3 x2.

#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"

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

Config ParseConfig() {
    Config c{};
    const char* v = std::getenv("BB_FG");
    std::string s = v != nullptr ? v : "off";
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
    if (s == "amd-x2" || s == "amd-2" || s == "amd" || s == "fsr3" || s == "fsr3-x2" ||
        s == "fsr3-2" || s == "fsr-x2") {
        c.kind = Kind::Amd;
        c.multiplier = 2;
    } else if (s == "amd-x3" || s == "amd-3" || s == "fsr3-x3") {
        // bb-amd-mfg: AMD multi frame generation (the patched amd_fidelityfx_vk.dll only).
        c.kind = Kind::Amd;
        c.multiplier = 3;
    } else if (s == "amd-x4" || s == "amd-4" || s == "fsr3-x4") {
        c.kind = Kind::Amd;
        c.multiplier = 4;
    } else if (s == "amd-dyn" || s == "amd-dynamic" || s == "amd-dinamico" || s == "fsr3-dyn") {
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
    } else if (s != "off" && s != "0" && !s.empty()) {
        BBLogf("Frame generation: BB_FG=%s is not available in this build (amd-x2..x4, amd-dyn, "
               "dlss-x2..x4, dlss-dyn): off",
               s.c_str());
    }
    if (c.kind == Kind::Amd) {
        SetProcessEnv("KYTY_FSR3_FG", "1");
    }
    return c;
}

} // namespace

const Config& Cfg() {
    static const Config config = [] {
        Config c = ParseConfig();
        if (c.kind == Kind::Amd) {
            BBLogf("Frame generation: BB_FG=%s (AMD FSR 3, %s x%d)", std::getenv("BB_FG"),
                   c.dynamic ? "dynamic x2 up to" : "fixed", c.multiplier);
        } else if (c.kind == Kind::Dlss) {
            BBLogf("Frame generation: BB_FG=%s (DLSS-G via NGX, %s)", std::getenv("BB_FG"),
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

namespace {
std::atomic<float> g_note_real{0.0f}, g_note_shown{0.0f};
std::atomic<int> g_note_mult{0};
} // namespace

void NoteRates(float real, float shown, int multiplier) {
    g_note_real.store(real, std::memory_order_relaxed);
    g_note_shown.store(shown, std::memory_order_relaxed);
    g_note_mult.store(multiplier, std::memory_order_relaxed);
}

std::string Label() {
    const auto& c = Cfg();
    int m = g_note_mult.load(std::memory_order_relaxed);
    if (m < 2) m = c.multiplier;
    const char* who = c.kind == Kind::Amd ? "FSR 3" : c.kind == Kind::Dlss ? "DLSS-G" : "FG";
    return std::string(who) + (c.dynamic ? " dyn x" : " x") + std::to_string(m);
}

bool PanelInfo(float& real, float& shown, const char*& state) {
    real = shown = 0;
    state = "off";
    if (!On()) return false;
    if (Cfg().kind == Kind::Dlss) {
        if (!Dlss::Ready()) {
            state = "DLSS-G unavailable";
            return true;
        }
        real = g_note_real.load(std::memory_order_relaxed);
        shown = g_note_shown.load(std::memory_order_relaxed);
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
    return Fsr3Active() || Dlss::Ready();
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
