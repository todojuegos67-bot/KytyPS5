// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-fg: BB_FG settings and the swapchain / device hooks (bb_fg.h), from the 1.59
// build's bb_fg.cpp reduced to AMD FSR 3 x2.

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
    } else if (s != "off" && s != "0" && !s.empty()) {
        BBLogf("Frame generation: BB_FG=%s is not available in this build (amd-x2 only): off",
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
        if (c.kind != Kind::Off) {
            BBLogf("Frame generation: BB_FG=%s (AMD FSR 3, fixed x%d)", std::getenv("BB_FG"),
                   c.multiplier);
        }
        return c;
    }();
    return config;
}

const char* Name() {
    return Cfg().kind == Kind::Amd ? "amd-x2" : "off";
}

bool On() {
    return Cfg().kind != Kind::Off;
}

bool FgBlocked(const char** why) {
    if (why != nullptr) *why = nullptr;
    return false;
}

bool Fsr3Requested() {
    return On() && Amd::Requested();
}

bool Fsr3Active() {
    return On() && Amd::Active();
}

bool WantInputs() {
    return Fsr3Active();
}

bool WantsUiLayer() {
    return On() && Amd::WantsUiLayer();
}

void SetUiLayer(const Image* ui) {
    if (On()) Amd::SetUiLayer(ui);
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
    if (!On() || !Amd::Requested()) return;
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
    if (!On()) return {};
    return Amd::PlanQueues(physical, graphics_family);
}

void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t graphics_family,
              vk::Queue graphics_queue, const QueuePlan& plan, std::mutex* submit_mutex) {
    if (!On()) return;
    Amd::OnDevice(physical, device, graphics_family, graphics_queue, plan, submit_mutex);
}

bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out) {
    if (!On()) return false;
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
