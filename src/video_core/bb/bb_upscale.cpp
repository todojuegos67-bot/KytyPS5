// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021: see bb_upscale.h. Config, back end set-up (from bb-combo-159), the camera from
// Bloodborne's scene constant block, jitter and the once-per-second performance line.

#include "video_core/bb/bb_perf2.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/bb/bb_dlss5.h"
#include "video_core/bb/bb_fg.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <fmt/format.h>
#include <imgui.h>
#include "common/logging/log.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace BB::SR {

const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Off:
        return "off";
    case Kind::Native:
        return "native";
    case Kind::Dlss:
        return "DLSS";
    case Kind::Xess:
        return "XeSS";
    case Kind::Fsr3:
        return "FSR 3.1";
    case Kind::Fsr4:
        return "FSR 4";
    case Kind::Taa:
        return "TAA";
    }
    return "?";
}

const char* QualityName(Quality quality) {
    switch (quality) {
    case Quality::Native:
        return "native AA";
    case Quality::UltraQualityPlus:
        return "ultra quality plus";
    case Quality::UltraQuality:
        return "ultra quality";
    case Quality::Quality:
        return "quality";
    case Quality::Balanced:
        return "balanced";
    case Quality::Performance:
        return "performance";
    case Quality::UltraPerformance:
        return "ultra performance";
    case Quality::Dynamic:
        return "dynamic";
    }
    return "?";
}

double QualityRatio(Kind kind, Quality quality) {
    (void)kind;
    switch (quality) {
    case Quality::Native:
        return 1.0;
    case Quality::UltraQualityPlus:
        return 1.3;
    case Quality::UltraQuality:
        return 1.5;
    case Quality::Quality:
        return 1.5;
    case Quality::Balanced:
        return 1.724;
    case Quality::Performance:
        return 2.0;
    case Quality::UltraPerformance:
        return 3.0;
    case Quality::Dynamic:
        return 1.5;
    }
    return 1.0;
}

std::atomic<uint32_t> g_backend_resets{0};
void NoteBackendReset(const char* why) {
    g_backend_resets.fetch_add(1, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "[BB_UPSCALE] history reset: {}", why);
}

} // namespace BB::SR

namespace BB::Upscale {

u32 g_scene_w = 0, g_scene_h = 0;

namespace {

std::string Lower(const char* value) {
    std::string s = value != nullptr ? value : "";
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

#if defined(_WIN32)
void SetEnv(const char* name, const char* value) {
    _putenv_s(name, value);
    SetEnvironmentVariableA(name, value);
}
#else
void SetEnv(const char* name, const char* value) {
    setenv(name, value, 1);
}
#endif

Config ParseConfig() {
    Config c;
    const std::string up = Lower(Env("BB_UPSCALER"));
    if (up == "dlss") {
        c.kind = SR::Kind::Dlss;
    } else if (up == "dlaa") {
        c.kind = SR::Kind::Dlss;
        c.quality = SR::Quality::Native;
    } else if (up == "fsr4") {
        c.kind = SR::Kind::Fsr4;
    } else if (up == "fsr4aa" || up == "fsr4native") {
        // Native TAA through FSR 4 (as dlaa for DLSS): jitter + history at ratio 1, no upscale.
        c.kind = SR::Kind::Fsr4;
        c.quality = SR::Quality::Native;
    } else if (up == "xess") {
        c.kind = SR::Kind::Xess;
    } else if (up == "fsr3" || up == "fsr") {
        c.kind = SR::Kind::Fsr3;
    } else if (up == "taa" || (up.empty() && Env("BB_TAA") != nullptr && Env("BB_TAA")[0] != '0')) {
        // bb-clean-021-taa: the emulator's TAA, native resolution (also the AA with no upscaler).
        c.kind = SR::Kind::Taa;
    }
    // (BB_UPSCALER=native or anything else: the stock path, nothing of this module runs.)
    const std::string q = Lower(Env("BB_UPSCALE_MODE"));
    if (up != "dlaa" && up != "fsr4aa" && up != "fsr4native" && !q.empty()) {
        if (q == "native" || q == "dlaa" || q == "aa" || q == "nativeaa") {
            c.quality = SR::Quality::Native;
        } else if (q == "quality") {
            c.quality = SR::Quality::Quality;
        } else if (q == "balanced") {
            c.quality = SR::Quality::Balanced;
        } else if (q == "performance" || q == "perf") {
            c.quality = SR::Quality::Performance;
        } else if (q == "ultraperformance" || q == "ultraperf" || q == "ultra_performance") {
            c.quality = SR::Quality::UltraPerformance;
        }
    }
    if (const char* s = Env("BB_UPSCALE_SHARPNESS") ? Env("BB_UPSCALE_SHARPNESS")
                                                    : Env("BB_SHARPNESS")) {
        c.sharpness = std::clamp(static_cast<float>(std::atof(s)), 0.0f, 1.0f);
    }
    if (c.kind == SR::Kind::Taa) {
        c.quality = SR::Quality::Native;
        c.sharpness = 0.25f;
        if (const char* s = Env("BB_TAA_SHARPNESS")) {
            c.sharpness = std::clamp(static_cast<float>(std::atof(s)), 0.0f, 1.0f);
        }
    }
    if (const char* s = Env("BB_UPSCALE_JITTER")) {
        c.jitter = s[0] != '0';
    }
    if (c.kind == SR::Kind::Fsr4 && Env("BB_FSR4_PRESET") == nullptr) {
        const char* preset = "quality";
        switch (c.quality) {
        case SR::Quality::Native:
            preset = "native";
            break;
        case SR::Quality::Balanced:
            preset = "balanced";
            break;
        case SR::Quality::Performance:
            preset = "performance";
            break;
        case SR::Quality::UltraPerformance:
            preset = "ultraperf";
            break;
        default:
            break;
        }
        SetEnv("BB_FSR4_PRESET", preset);
    }
    return c;
}

bool Streamline() {
    return Cfg().kind == SR::Kind::Dlss;
}

bool g_backend_dead = false;

// ---- Camera ----
struct Mat {
    double m[16]{};
};

Mat Multiply(const Mat& a, const Mat& b) {
    Mat r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[i * 4 + k] * b.m[k * 4 + j];
            r.m[i * 4 + j] = s;
        }
    return r;
}

bool Invert(const Mat& in, Mat& out) {
    double a[4][8]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = in.m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        if (std::fabs(a[pivot][c]) < 1e-15) return false;
        for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double s = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] *= s;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out.m[i * 4 + j] = a[i][4 + j];
    return true;
}

double Dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double Len3(const double* a) {
    return std::sqrt(Dot3(a, a));
}

// A world -> clip perspective view-projection (1.59 LooksLikeViewProj).
bool LooksLikeViewProj(const Mat& v) {
    const double* r0 = &v.m[0];
    const double* r1 = &v.m[4];
    const double* r2 = &v.m[8];
    const double* r3 = &v.m[12];
    for (double x : v.m)
        if (!std::isfinite(x) || std::fabs(x) > 1e7) return false;
    const double n0 = Len3(r0), n1 = Len3(r1), n2 = Len3(r2), n3 = Len3(r3);
    if (n3 < 0.97 || n3 > 1.03) return false;
    if (n0 < 1e-4 * n3 || n1 < 1e-4 * n3 || n2 < 1e-9) return false;
    const double aspect = n0 / n1;
    if (aspect < 0.2 || aspect > 5.0) return false;
    if (std::fabs(Dot3(r0, r1)) / (n0 * n1) > 0.05) return false;
    if (std::fabs(Dot3(r0, r3)) / (n0 * n3) > 0.25 || std::fabs(Dot3(r1, r3)) / (n1 * n3) > 0.25)
        return false;
    const double c[3]{r2[1] * r3[2] - r2[2] * r3[1], r2[2] * r3[0] - r2[0] * r3[2],
                      r2[0] * r3[1] - r2[1] * r3[0]};
    if (Len3(c) / (n2 * n3) > 0.05) return false;
    return true;
}

// bb-clean-021-bbport: the camera as Supermedo's port (gpu/shadps4/video_core/renderer_vulkan/
// vk_camera_motion.cpp, GPL-2.0-or-later): the FIRST valid 864-byte scene constant block of a
// frame is the main camera; a frame without one drops the history (current and previous).
// Written by the GPU thread (OnConstants), read at the flip.
struct BlockCam {
    float view[12]{};     // world to view, 3x4 rows (floats 8-19)
    float inv_view[12]{}; // view to world, 3x4 rows (floats 180-191)
    float proj[4]{};      // x scale, y scale, z scale, z offset (floats 52, 57, 62, 63)
    Mat vp;               // full projection (52-67) * view, for frame generation / DLSS
    bool valid = false;
};
std::mutex g_cam_mutex;
BlockCam g_cam_cur, g_cam_prev;
bool g_frame_has_camera = false;
struct RtCam {
    float proj[16]{};
    float view[12]{};
    float far_dist = 3000.0f;
    float block[216]{};
    bool valid = false;
} g_rt_cam;

u32 g_frame_no = 0;
u32 g_rt_frame = 0; // bb-rt: frames, counted whether or not jitter is on
float g_jitter[2]{};

float Halton(u32 index, u32 base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

// ---- Stats ----
Stats g_stats;
std::atomic<bool> g_overlay{false};
struct Accum {
    double cpu = 0, wait = 0, gpu = 0, sr = 0, mv = 0;
    double frame_max = 0; // the longest flip-to-flip frame of the second (cpu + wait)
    u32 frames = 0, gpu_n = 0;
    std::chrono::steady_clock::time_point start{};
} g_acc;

// F8 frametime graph: one mean and one peak per second, the last 60 seconds (the Demon's Souls
// Metricas graph). Written and drawn on the presenter's thread, so no lock.
std::vector<float> g_hist_mean, g_hist_max;

} // namespace

const Config& Cfg() {
    static const Config c = ParseConfig();
    return c;
}

bool Enabled() {
    return Cfg().kind != SR::Kind::Off && Cfg().kind != SR::Kind::Native;
}

bool Temporal() {
    return Enabled();
}

namespace {
std::atomic<u32> g_out_w{0}, g_out_h{0};
}

void SetOutputSize(u32 w, u32 h) {
    g_out_w = w;
    g_out_h = h;
}

bool OutputSize(u32& w, u32& h) {
    w = g_out_w.load();
    h = g_out_h.load();
    return w != 0 && h != 0;
}

bool Capture() {
    static const bool on = Enabled() || BB::FG::On();
    return on;
}

double Ratio() {
    return SR::QualityRatio(Cfg().kind, Cfg().quality);
}

namespace {
// bb-clean-021-fg: until the first frame every checkpoint is also appended to bb_startup.log
// (opened, written, flushed and closed per line), so a crash during start-up leaves its last
// step; the previous run's file is kept as bb_startup.prev.log.
std::atomic<bool> g_startup_done{false};
std::mutex g_startup_mutex;

void StartupNote(std::string_view text) {
    if (g_startup_done.load(std::memory_order_relaxed)) return;
    std::scoped_lock lk{g_startup_mutex};
    static bool first = true;
    if (first) {
        first = false;
        std::error_code ec;
        std::filesystem::rename("bb_startup.log", "bb_startup.prev.log", ec);
    }
    if (std::FILE* f = std::fopen("bb_startup.log", "ab")) {
        std::fwrite(text.data(), 1, text.size(), f);
        std::fputc('\n', f);
        std::fflush(f);
        std::fclose(f);
    }
}
} // namespace

void Checkpoint(std::string_view text) {
    LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}", text);
    StartupNote(text);
}

void StartupDone() {
    if (!g_startup_done.load(std::memory_order_relaxed)) {
        StartupNote("start-up done: first frame presented");
        g_startup_done = true;
    }
}

namespace {
// Before the first LiveSharpness call: the launcher's file written before the start is the
// value the environment already gave.
const auto g_sharp_start = std::filesystem::file_time_type::clock::now();
} // namespace

// The launcher's Nitidez slider while playing: BB_UPSCALE_SHARPNESS_FILE (one line
// "BB_UPSCALE_SHARPNESS=0.80"), re-read at most twice a second when its time stamp changes.
float LiveSharpness() {
    static std::atomic<float> value{-1.0f};
    static std::mutex mx;
    static std::chrono::steady_clock::time_point next{};
    static std::filesystem::file_time_type seen = g_sharp_start;
    static const std::string path = [] {
        const char* f = Env("BB_UPSCALE_SHARPNESS_FILE");
        return f != nullptr ? std::string(f) : std::string();
    }();
    if (value.load(std::memory_order_relaxed) < 0.0f) {
        value.store(Cfg().sharpness, std::memory_order_relaxed);
    }
    if (path.empty() || !mx.try_lock()) {
        return value.load(std::memory_order_relaxed);
    }
    std::lock_guard lock{mx, std::adopt_lock};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next) {
        next = now + std::chrono::milliseconds(500);
        std::error_code ec;
        const auto stamp = std::filesystem::last_write_time(path, ec);
        if (!ec && stamp > seen) {
            seen = stamp;
            std::ifstream in(path);
            std::string text;
            std::getline(in, text);
            if (const auto eq = text.find('='); eq != std::string::npos)
                text = text.substr(eq + 1);
            char* end = nullptr;
            const float v = std::strtof(text.c_str(), &end);
            if (end != text.c_str() && std::isfinite(v)) {
                const float s = std::clamp(v, 0.0f, 1.0f);
                if (s != value.load(std::memory_order_relaxed)) {
                    value.store(s, std::memory_order_relaxed);
                    Checkpoint(fmt::format("[sharpen] live sharpness {:.2f} ({})", s, path));
                }
            }
        }
    }
    return value.load(std::memory_order_relaxed);
}

void PreInstance() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const auto& c = Cfg();
    if (!Enabled()) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "[BB_UPSCALE] bb-clean-021: upscaler {} mode {} (ratio {:.3f}), sharpness {:.2f}, "
             "jitter {}",
             SR::KindName(c.kind), SR::QualityName(c.quality), Ratio(), c.sharpness,
             c.jitter ? "on" : "off");
    StartupNote(fmt::format("upscaler {} mode {} (ratio {:.3f}), jitter {}", SR::KindName(c.kind),
                            SR::QualityName(c.quality), Ratio(), c.jitter ? "on" : "off"));
    if (Streamline()) {
#if defined(_WIN32)
        for (const char* var :
             {"DISABLE_VULKAN_OBS_CAPTURE", "DISABLE_VK_LAYER_VALVE_steam_overlay_1",
              "DISABLE_VK_LAYER_VALVE_steam_fossilize_1"}) {
            if (Env(var) == nullptr) {
                SetEnv(var, "1");
            }
        }
        if (Env("VK_LOADER_LAYERS_DISABLE") == nullptr) {
            SetEnv("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_OBS_HOOK");
        }
#endif
        Checkpoint("DLSS: loading Streamline (slInit)");
        if (!SR::Dlss::Preload()) {
            Checkpoint("DLSS: Streamline not loaded: DLSS off, the game runs native");
        }
    } else if (c.kind == SR::Kind::Xess) {
        SR::Xess::Load();
    } else if (c.kind == SR::Kind::Fsr3) {
        SR::Fsr3::Load();
    }
}

const wchar_t* VulkanLoaderOverride() {
    return Streamline() ? SR::Dlss::LoaderPath() : nullptr;
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    (void)instance;
    (void)physical;
    const auto add = [&](const char* name) {
        const bool has = std::find(available.begin(), available.end(), name) != available.end();
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (has && !added) {
            extensions.push_back(name);
        }
        Checkpoint(fmt::format("device extension {} for {}: {}", name, SR::KindName(Cfg().kind),
                               has ? "added" : "NOT available"));
    };
    if (Streamline() && SR::Dlss::Manual()) {
        static std::vector<std::string> names;
        names = SR::Dlss::RequiredDeviceExtensions();
        for (const auto& name : names) {
            if (name == VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) continue;
            add(name.c_str());
        }
    }
    if (Streamline() && SR::Dlss::LoaderPath() != nullptr) {
        add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
    }
    if (Cfg().kind == SR::Kind::Fsr4) {
        SR::Fsr4::AddDeviceExtensions(available, extensions);
    }
    if (Cfg().kind == SR::Kind::Fsr3) {
        // The FidelityFX backend calls vkGetBufferMemoryRequirements2KHR by the extension's name.
        add(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        add(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
    if (Cfg().kind == SR::Kind::Xess) {
        SR::Xess::AddDeviceExtensions(instance, physical, available, extensions);
    }
    // DLSS 5 (bb_dlss5.h, from 1.59): NGX's vendor extensions, only with BB_DLSS5 1..3.
    Dlss5::AddDeviceExtensions(physical, available, extensions);
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next) {
    if (Streamline()) {
        vk::PhysicalDeviceVulkan13Features s13{};
        vk::PhysicalDeviceVulkan12Features s12{};
        s12.pNext = &s13;
        vk::PhysicalDeviceFeatures2 s{};
        s.pNext = &s12;
        physical.getFeatures2(&s);
#define BB_F12(name) {#name, &vk::PhysicalDeviceVulkan12Features::name}
#define BB_F13(name) {#name, &vk::PhysicalDeviceVulkan13Features::name}
        static const std::pair<const char*, vk::Bool32 vk::PhysicalDeviceVulkan12Features::*>
            t12[] = {BB_F12(bufferDeviceAddress),
                     BB_F12(timelineSemaphore),
                     BB_F12(descriptorIndexing),
                     BB_F12(runtimeDescriptorArray),
                     BB_F12(descriptorBindingPartiallyBound),
                     BB_F12(descriptorBindingVariableDescriptorCount),
                     BB_F12(descriptorBindingUpdateUnusedWhilePending),
                     BB_F12(descriptorBindingSampledImageUpdateAfterBind),
                     BB_F12(descriptorBindingStorageImageUpdateAfterBind),
                     BB_F12(shaderSampledImageArrayNonUniformIndexing),
                     BB_F12(shaderStorageImageArrayNonUniformIndexing),
                     BB_F12(scalarBlockLayout),
                     BB_F12(hostQueryReset),
                     BB_F12(shaderFloat16),
                     BB_F12(shaderInt8),
                     BB_F12(storageBuffer8BitAccess),
                     BB_F12(uniformAndStorageBuffer8BitAccess),
                     BB_F12(vulkanMemoryModel),
                     BB_F12(imagelessFramebuffer),
                     BB_F12(separateDepthStencilLayouts),
                     BB_F12(uniformBufferStandardLayout),
                     BB_F12(shaderSubgroupExtendedTypes)};
        static const std::pair<const char*, vk::Bool32 vk::PhysicalDeviceVulkan13Features::*>
            t13[] = {BB_F13(synchronization2),
                     BB_F13(dynamicRendering),
                     BB_F13(maintenance4),
                     BB_F13(shaderDemoteToHelperInvocation),
                     BB_F13(subgroupSizeControl),
                     BB_F13(computeFullSubgroups),
                     BB_F13(pipelineCreationCacheControl),
                     BB_F13(shaderIntegerDotProduct)};
#undef BB_F12
#undef BB_F13
        for (const auto& name : SR::Dlss::RequiredFeatures12()) {
            for (const auto& [n, m] : t12) {
                if (name == n && s12.*m == VK_TRUE) f12.*m = VK_TRUE;
            }
        }
        for (const auto& name : SR::Dlss::RequiredFeatures13()) {
            for (const auto& [n, m] : t13) {
                if (name == n && s13.*m == VK_TRUE) f13.*m = VK_TRUE;
            }
        }
    }
    if (Streamline() && SR::Dlss::LoaderPath() != nullptr) {
        const bool has = std::any_of(extensions.begin(), extensions.end(), [](const char* e) {
            return std::strcmp(e, VK_KHR_PRESENT_ID_EXTENSION_NAME) == 0;
        });
        if (has) {
            static VkPhysicalDevicePresentIdFeaturesKHR present_id{};
            present_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
            present_id.presentId = VK_TRUE;
            present_id.pNext = next;
            next = &present_id;
        }
    }
    if (Cfg().kind == SR::Kind::Fsr4) {
        return SR::Fsr4::ChainDeviceFeatures(physical, extensions, f12, f13, f, next);
    }
    if (Cfg().kind == SR::Kind::Xess) {
        return SR::Xess::ChainDeviceFeatures(extensions, next);
    }
    return next;
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family) {
    Dlss5::OnDevice(instance, physical, device, allocator);
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        SR::Dlss::OnDevice(instance, physical, device);
        break;
    case SR::Kind::Fsr4:
        SR::Fsr4::OnDevice(instance, physical, device, allocator, queue, family);
        break;
    case SR::Kind::Xess:
        SR::Xess::OnDevice(instance, physical, device);
        break;
    case SR::Kind::Fsr3:
        SR::Fsr3::OnDevice(physical, device, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
        break;
    case SR::Kind::Taa:
        SR::Taa::OnDevice(device, allocator);
        break;
    default:
        return;
    }
    Checkpoint(fmt::format("{} back end {}", SR::KindName(Cfg().kind),
                           BackendReady() ? "ready"
                                          : "NOT available: the game runs native (no upscale)"));
}

bool BackendReady() {
    if (g_backend_dead) return false;
    static int fsr4 = -1;
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        return SR::Dlss::Ready();
    case SR::Kind::Fsr4:
        if (fsr4 < 0) fsr4 = SR::Fsr4::Ready() ? 1 : 0;
        return fsr4 == 1;
    case SR::Kind::Xess:
        return SR::Xess::Ready();
    case SR::Kind::Fsr3:
        return SR::Fsr3::Ready();
    case SR::Kind::Taa:
        return SR::Taa::Ready();
    default:
        return false;
    }
}

bool RunBackend(const SR::Desc& desc) {
    if (g_backend_dead) {
        return false;
    }
    if (desc.render.width < 64 || desc.render.height < 64 ||
        desc.upscale.width > desc.render.width * 3u + 2u ||
        desc.upscale.height > desc.render.height * 3u + 2u) {
        return false;
    }
    bool ok = false;
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        ok = SR::Dlss::Evaluate(desc);
        break;
    case SR::Kind::Fsr4:
        ok = SR::Fsr4::Evaluate(desc);
        break;
    case SR::Kind::Xess:
        ok = SR::Xess::Evaluate(desc);
        break;
    case SR::Kind::Fsr3:
        ok = SR::Fsr3::Evaluate(desc);
        break;
    case SR::Kind::Taa:
        ok = SR::Taa::Evaluate(desc);
        break;
    default:
        return false;
    }
    static u32 failures = 0;
    if (!ok && ++failures >= 8) {
        g_backend_dead = true;
        Checkpoint(fmt::format("{} failed {} times: turned off, the game runs native",
                               SR::KindName(Cfg().kind), failures));
    }
    if (ok) failures = 0;
    return ok;
}

// Bloodborne's scene constant block (864 bytes): [0] far plane 3000, [1] 1 / far, [4] [5] the
// render size, floats 8-19 the view, 52-67 the projection, 180-191 the inverse view.
bool RtCamera(float proj[16], float view[12], float& far_dist) {
    std::scoped_lock lk{g_cam_mutex};
    if (!g_rt_cam.valid) {
        return false;
    }
    std::memcpy(proj, g_rt_cam.proj, sizeof(g_rt_cam.proj));
    std::memcpy(view, g_rt_cam.view, sizeof(g_rt_cam.view));
    far_dist = g_rt_cam.far_dist;
    return true;
}
bool RtSceneBlock(float out[216]) {
    std::scoped_lock lk{g_cam_mutex};
    if (!g_rt_cam.valid) {
        return false;
    }
    std::memcpy(out, g_rt_cam.block, sizeof(g_rt_cam.block));
    return true;
}
namespace {
struct RtLighting {
    float light[72]{};
    float fog[24]{};
    u32 light_frame = 0, fog_frame = 0;
    bool light_valid = false, fog_valid = false;
    u64 light_hash = 0;
    size_t light_bytes = 0;
} g_rt_lighting;
bool Finite(const float* f, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(f[i])) return false;
    }
    return true;
}
} // namespace

namespace {
// Light-set CB (Ghidra, eboot 1.09): FUN_02651f10 (per-material constant setup, called from
// the object draw paths FUN_02650690 / FUN_02650f70 / FUN_02654a10 for pass types 0 and 4)
// gets the material's CB slot 5 ((*ctx->vtbl[3])(ctx, 5), vtable 0x57b7880 -> FUN_0264dd30:
// *(G->+0x6070)->+0x248 + slot * 8, i.e. ONE global CB object per slot shared by every draw) and
// FUN_026deeb0 / FUN_026dfe80 pack the LIGHT SET into it: a 288-byte block uploaded with
// FUN_02568a20(gfx, gfx + 0xb730, cb, 0, 0, data), stored in the context's CB table
// (+0x990 = slot 5 of the table at +0x968, dirty range 5..6, dirty bit 0x200000 at +0x1978).
// Layout: floats 0-15 = 4 directions (light 0 = set+0x40 raw world; 1-3 world or, when the
// light's flag is set, transformed by the matrix at owner+0x3f8), 16-31 diffuse rgb^2*a,
// 32-47 specular rgb^2*a, 48-59 ambient terms, 60-63 set+0x230 rgb^2*a, u32 65 = flag
// (set+0x24e, 0 or 1), 68-71 = set+0x90/+0xf0/+0x150/+0x1b0.
// It is consumed by the forward-lit object draws, NOT by the composite (PS 0x3fe9e964, its
// 208 B buffer) or the SSR draw (PS 0x6c2323c2, its 320 B buffer): those two are rejected.
bool IsLightSetCb(const u32* data, size_t bytes, u64 hash) {
    const float* f = reinterpret_cast<const float*>(data);
    const bool rejected = hash == 0x3fe9e964 || hash == 0x6c2323c2 || hash == 0x8747a367;
    if (rejected || bytes < 288 || bytes > 320 || !Finite(f, 72)) return false;
    const auto len = [&](int i) {
        return std::sqrt(f[i] * f[i] + f[i + 1] * f[i + 1] + f[i + 2] * f[i + 2]);
    };
    bool ok = data[65] <= 1u;                        // bool flag
    ok = ok && std::abs(len(0) - 1.0f) < 0.02f;      // light 0: unit world direction
    for (int i = 1; i < 4 && ok; ++i) {
        const float l = len(4 * i);
        ok = l < 1e-4f || std::abs(l - 1.0f) < 0.05f; // unused (0) or unit
    }
    for (int i = 16; i < 48 && ok; ++i) {
        ok = f[i] >= 0.0f && f[i] < 1e4f; // squared colours x intensity
    }
    for (int i = 60; i < 63 && ok; ++i) {
        ok = f[i] >= 0.0f && f[i] < 1e4f;
    }
    return ok && f[16] + f[17] + f[18] > 1e-5f; // light 0 lit
}
// Fog CB (Ghidra, eboot 1.09, FUN_02651f10 right after the light set: CB slot 6, the same
// FUN_02568a20 upload, dirty bit 0x400000, context +0x998). Fog param p (FUN_026d9ea0 lookup in
// the FOG MANAGER table G+0x5888, lerped between two params by ctx+0x340 when they differ),
// camera c = owner+0x3f8 (c+0x34 = camera y):
//   s  = p+0x28 (byte) ? 1 : -1,   iH = 1 / (s * max(p+0x24, 0.1))  -> |iH| <= 10, never 0
//   id = p+0x20 > 0 ? 1 / p+0x20 : 0
//   v0 = (p+0x1c * p+0x2c, id, id * exp2(-c.y * iH), iH)       (FUN_02fbf198 = exp2f)
//   v1 = (p+0x10 rgb^2, p+0x30)
//   v2 = (v0.x, p+0x54, id, iH)
//   v3 = (p+0x4c * p+0x2c, G+0x5b94 + p+0x58, iH * (p+0x50 > 0 ? 1 / p+0x50 : 0), c.y)
//   v4 = (p+0x40 rgb^2, p+0x30)
//   v5.x = int 1. 84 bytes of data (stack 0x118..0xc4); the copy length is the CB object's size
//   (cb+0x2c), not decoded statically, so the V# may be larger (the tail is not the fog).
// v2.x / v2.z / v2.w / v4.w are register copies of v0.x / v0.y / v0.w / v1.w: bit-identical.
// v0.z overflows to +inf (or 0 * inf = NaN when id is 0) once |c.y * iH| > 128, so it is never
// required to be finite (the old Finite(20) test rejected the CB for every such camera height).
const char* FogCbReject(const u32* data, size_t bytes) {
    const float* f = reinterpret_cast<const float*>(data);
    if (bytes < 84) return "smaller than 84 B";
    if (data[8] != data[0]) return "v2.x != v0.x";
    if (data[10] != data[1]) return "v2.z != v0.y";
    if (data[11] != data[3]) return "v2.w != v0.w";
    if (data[19] != data[7]) return "v4.w != v1.w";
    if (!(std::abs(f[3]) <= 10.001f) || f[3] == 0.0f) return "v0.w not 1/H (|.| in (0, 10])";
    if (!(f[1] >= 0.0f) || !std::isfinite(f[1])) return "v0.y (1/dist) not finite >= 0";
    for (int i : {4, 5, 6, 16, 17, 18}) {
        if (!(f[i] >= 0.0f) || !std::isfinite(f[i])) return "fog colour (squared) not >= 0";
    }
    for (int i : {0, 7, 9, 12, 13, 14, 15}) {
        if (!std::isfinite(f[i])) return "v0.x / v1.w / v2.y / v3 not finite";
    }
    return nullptr;
}
struct FogSource {
    u64 hash = 0;
    size_t bytes = 0;
    u32 stage = 0, index = 0;
    bool light_draw = false;
    bool operator==(const FogSource&) const = default;
};
FogSource g_fog_src{};
} // namespace

void OnDrawConstants(const DrawCb* cbs, size_t n) {
    // Diagnostic (kept from bb-rt-vol): the read-only buffers the lighting draws bind.
    for (size_t i = 0; i < n; ++i) {
        const u64 hash = cbs[i].hash;
        const size_t bytes = cbs[i].bytes;
        const bool light_hash = hash == 0x3fe9e964 || hash == 0x8747a367 || hash == 0x6c2323c2;
        if (!((light_hash && bytes >= 64) || (bytes >= 288 && bytes <= 320))) continue;
        const u32* data = cbs[i].data;
        const float* f = reinterpret_cast<const float*>(data);
        static std::vector<std::pair<u64, size_t>> seen;
        std::scoped_lock lk{g_cam_mutex};
        if (seen.size() < 32 &&
            std::ranges::find(seen, std::pair{hash, bytes}) == seen.end()) {
            seen.emplace_back(hash, bytes);
            const auto len = [&](int k) {
                return std::sqrt(f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2]);
            };
            LOG_INFO(Render_Vulkan,
                     "[rt-light] CB candidate: shader {:#x} {} B: f0-7 ({:.3f} {:.3f} {:.3f} "
                     "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}) |v0..v3| {:.3f} {:.3f} {:.3f} {:.3f} "
                     "f16-19 ({:.3f} {:.3f} {:.3f} {:.3f}) u20 {:#x}",
                     hash, bytes, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], len(0), len(4),
                     len(8), len(12), f[16], f[17], f[18], f[19], data[20]);
        }
    }
    int light_i = -1;
    for (size_t i = 0; i < n && light_i < 0; ++i) {
        if (IsLightSetCb(cbs[i].data, cbs[i].bytes, cbs[i].hash)) light_i = int(i);
    }
    // Fog: the first buffer of the draw with the FUN_02651f10 layout; in a light-set draw the
    // buffers after the light set (slot 6 follows slot 5) are tried first.
    int fog_i = -1;
    if (light_i >= 0) {
        for (size_t i = light_i + 1; i < n && fog_i < 0; ++i) {
            if (!FogCbReject(cbs[i].data, cbs[i].bytes)) fog_i = int(i);
        }
    }
    for (size_t i = 0; i < n && fog_i < 0; ++i) {
        if (int(i) != light_i && !FogCbReject(cbs[i].data, cbs[i].bytes)) fog_i = int(i);
    }
    if (light_i < 0 && fog_i < 0) return;
    std::scoped_lock lk{g_cam_mutex};
    if (light_i >= 0) {
        const DrawCb& c = cbs[light_i];
        const float* f = reinterpret_cast<const float*>(c.data);
        if (g_rt_lighting.light_hash != c.hash || g_rt_lighting.light_bytes != c.bytes) {
            LOG_INFO(Render_Vulkan,
                     "[rt-light] light-set CB source: shader {:#x}, {} B V# (strict layout "
                     "check passed): dir0 ({:.3f} {:.3f} {:.3f}) diffuse0 ({:.3f} {:.3f} {:.3f}) "
                     "flag {}",
                     c.hash, c.bytes, f[0], f[1], f[2], f[16], f[17], f[18], c.data[65]);
        }
        std::memcpy(g_rt_lighting.light, f, 72 * sizeof(float));
        g_rt_lighting.light_frame = g_rt_frame;
        g_rt_lighting.light_valid = true;
        g_rt_lighting.light_hash = c.hash;
        g_rt_lighting.light_bytes = c.bytes;
        // Every other read-only buffer of the draws that carry the light set, once per
        // (shader, size): the fog CB (slot 6) should be among them.
        static std::vector<std::pair<u64, size_t>> seen;
        for (size_t i = 0; i < n; ++i) {
            if (int(i) == light_i) continue;
            const DrawCb& o = cbs[i];
            if (seen.size() >= 64 ||
                std::ranges::find(seen, std::pair{o.hash, o.bytes}) != seen.end()) {
                continue;
            }
            seen.emplace_back(o.hash, o.bytes);
            const float* g = reinterpret_cast<const float*>(o.data);
            const size_t nf = std::min<size_t>(24, o.bytes / 4);
            std::string vals;
            for (size_t k = 0; k < nf; ++k) {
                vals += fmt::format("{}{:.5g}", k == 0 ? "" : (k % 4 == 0 ? " | " : " "), g[k]);
            }
            const char* why = FogCbReject(o.data, o.bytes);
            LOG_INFO(Render_Vulkan,
                     "[rt-vol] light-set draw CB: {} shader {:#x} buffer {} ({} B), light set = "
                     "{} shader {:#x} buffer {} ({} B) of {} buffers; fog layout: {}; f0-{}: {}",
                     o.stage == 0 ? "VS" : "PS", o.hash, o.index, o.bytes,
                     c.stage == 0 ? "VS" : "PS", c.hash, c.index, c.bytes, n,
                     why ? why : "MATCH", nf ? nf - 1 : 0, vals);
        }
    }
    if (fog_i >= 0) {
        const DrawCb& c = cbs[fog_i];
        const FogSource src{c.hash, c.bytes, c.stage, c.index, light_i >= 0};
        // A light-set draw's fog CB is preferred: another draw's match only fills in when none
        // was seen for a few frames.
        if (src.light_draw || !g_fog_src.light_draw || g_rt_frame - g_rt_lighting.fog_frame > 4) {
            static std::vector<FogSource> logged; // once per source, first 16
            if (logged.size() < 16 && std::ranges::find(logged, src) == logged.end()) {
                logged.push_back(src);
                const float* f = reinterpret_cast<const float*>(c.data);
                LOG_INFO(Render_Vulkan,
                         "[rt-vol] fog CB source (FUN_02651f10 layout): {} shader {:#x} buffer "
                         "{} ({} B V#){}: v0 ({:.5g} {:.5g} {:.5g} {:.5g}) v1 ({:.4g} {:.4g} "
                         "{:.4g} {:.4g}) v3 ({:.4g} {:.4g} {:.5g} {:.4g}) v5.x {:#x}",
                         c.stage == 0 ? "VS" : "PS", c.hash, c.index, c.bytes,
                         src.light_draw ? ", in a light-set draw" : ", NOT in a light-set draw",
                         f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[12], f[13], f[14],
                         f[15], c.data[20]);
            }
            g_fog_src = src;
            std::memcpy(g_rt_lighting.fog, c.data, 21 * sizeof(float));
            g_rt_lighting.fog_frame = g_rt_frame;
            g_rt_lighting.fog_valid = true;
        }
    }
}
void RtLightSource(u64& hash, size_t& bytes) {
    std::scoped_lock lk{g_cam_mutex};
    hash = g_rt_lighting.light_hash;
    bytes = g_rt_lighting.light_bytes;
}
bool RtLightBlock(float out[72], u64& age_frames) {
    std::scoped_lock lk{g_cam_mutex};
    if (!g_rt_lighting.light_valid) return false;
    std::memcpy(out, g_rt_lighting.light, sizeof(g_rt_lighting.light));
    age_frames = g_rt_frame - g_rt_lighting.light_frame;
    return true;
}
bool RtFogBlock(float out[24], u64& age_frames) {
    std::scoped_lock lk{g_cam_mutex};
    if (!g_rt_lighting.fog_valid) return false;
    std::memcpy(out, g_rt_lighting.fog, sizeof(g_rt_lighting.fog));
    age_frames = g_rt_frame - g_rt_lighting.fog_frame;
    return true;
}

void OnConstants(const u32* data, size_t count) {
    if (count < 216) {
        return;
    }
    const float* d = reinterpret_cast<const float*>(data);
    if (d[0] != 3000.0f || d[4] < 64.0f || d[5] < 64.0f ||
        std::abs(d[1] * d[0] - 1.0f) > 1e-3f) {
        return;
    }
    std::scoped_lock lk{g_cam_mutex};
    // bb-rt-reflect: the latest block (the SSR draw binds the scene block itself).
    std::memcpy(g_rt_cam.proj, d + 52, sizeof(g_rt_cam.proj));
    std::memcpy(g_rt_cam.view, d + 8, sizeof(g_rt_cam.view));
    g_rt_cam.far_dist = d[0];
    std::memcpy(g_rt_cam.block, d, sizeof(g_rt_cam.block));
    g_rt_cam.valid = d[52] != 0.0f && d[57] != 0.0f;
    if (g_frame_has_camera) {
        return; // the first one of a frame is the main camera
    }
    g_cam_prev = g_cam_cur;
    BlockCam& c = g_cam_cur;
    std::memcpy(c.view, d + 8, sizeof(c.view));
    std::memcpy(c.inv_view, d + 180, sizeof(c.inv_view));
    c.proj[0] = d[52];
    c.proj[1] = d[57];
    c.proj[2] = d[62];
    c.proj[3] = d[63];
    c.valid = c.proj[0] != 0.0f && c.proj[1] != 0.0f;
    Mat view, proj;
    for (int i = 0; i < 12; ++i) view.m[i] = d[8 + i];
    view.m[15] = 1.0;
    for (int i = 0; i < 16; ++i) proj.m[i] = d[52 + i];
    c.vp = Multiply(proj, view);
    g_frame_has_camera = true;
}

std::atomic<u32> g_jit_draws{0}, g_jit_skipped{0}, g_jit_w{0}, g_jit_h{0};
void NoteJitterDraw(bool jittered, bool skipped_after_hud, u32 w, u32 h) {
    if (jittered) {
        g_jit_draws.fetch_add(1, std::memory_order_relaxed);
        g_jit_w.store(w, std::memory_order_relaxed);
        g_jit_h.store(h, std::memory_order_relaxed);
    }
    if (skipped_after_hud) g_jit_skipped.fetch_add(1, std::memory_order_relaxed);
}

void Jitter(float& x, float& y) {
    x = g_jitter[0];
    y = g_jitter[1];
}

// a * b for 3x4 affine matrices (rows [R | t]), as Supermedo's vk_camera_motion.cpp.
static void Multiply34(const float* a, const float* b, float* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = c == 3 ? a[r * 4 + 3] : 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
}

Camera EndFrame(u32 flip_w, u32 flip_h) {
    StartupDone();
    Camera cam;
    {
        std::scoped_lock lk{g_cam_mutex};
        if (!g_frame_has_camera) {
            g_cam_cur.valid = g_cam_prev.valid = false; // no camera this frame: drop the history
        }
        g_frame_has_camera = false;
        const BlockCam& cur = g_cam_cur;
        const BlockCam& prev = g_cam_prev;
        Mat inv;
        if (cur.valid && Invert(cur.vp, inv)) {
            const Mat to_prev = Multiply(prev.valid ? prev.vp : cur.vp, inv);
            for (int i = 0; i < 16; ++i) {
                cam.view_to_clip[i] = static_cast<float>(cur.vp.m[i]);
                cam.clip_to_prev_clip[i] = static_cast<float>(to_prev.m[i]);
            }
            const double* r1 = &cur.vp.m[4];
            const double* r2 = &cur.vp.m[8];
            const double* r3 = &cur.vp.m[12];
            cam.fov_y = static_cast<float>(2.0 * std::atan(Len3(r3) / std::max(Len3(r1), 1e-9)));
            const double alpha = Dot3(r2, r3) / Dot3(r3, r3);
            const double beta = r2[3] - alpha * r3[3];
            const double w0 = std::fabs(alpha) > 1e-9 ? -beta / alpha : 1e30;
            const double w1 = std::fabs(1.0 - alpha) > 1e-9 ? beta / (1.0 - alpha) : 1e30;
            double zn = std::min(std::fabs(w0), std::fabs(w1));
            const double zf = std::max(std::fabs(w0), std::fabs(w1));
            if (!(zn > 1e-4)) zn = 0.1;
            cam.z_near = static_cast<float>(zn);
            cam.z_far = zf > 1e7 ? 0.0f : static_cast<float>(zf);
            cam.valid = true;
            cam.reset = !prev.valid;
        }
        // The motion-vector camera (bb_mv.comp): previous view * inverse current view.
        if (cur.valid && prev.valid && prev.proj[0] != 0.0f) {
            Multiply34(prev.view, cur.inv_view, cam.reproject);
            std::memcpy(cam.proj, cur.proj, sizeof(cam.proj));
            std::memcpy(cam.prev_proj, prev.proj, sizeof(cam.prev_proj));
            cam.mv_valid = true;
        }
    }
    ++g_rt_frame;
    g_scene_w = flip_w;
    g_scene_h = flip_h;
    // Next frame's jitter: Halton(2,3) - 0.5 over 8 * ratio^2 phases (at least 8).
    if (Cfg().jitter && Enabled()) {
        const double r = std::max(1.0, Ratio());
        const u32 phases = std::max<u32>(8, static_cast<u32>(std::ceil(8.0 * r * r)));
        const u32 i = (++g_frame_no % phases) + 1;
        g_jitter[0] = Halton(i, 2) - 0.5f;
        g_jitter[1] = Halton(i, 3) - 0.5f;
    } else {
        g_jitter[0] = g_jitter[1] = 0.0f;
    }
    return cam;
}

// bb-perf-2: one "[perf2]" line every 10 s: frame pacing (mean / worst / frames over 50 ms),
// CPU and GPU frame ms, pipeline compiles and their cost, CPU write faults on GPU-tracked memory,
// buffer/texture upload and readback MiB/s, sparse binds and blocking GPU waits.
static void NotePerf2(std::chrono::steady_clock::time_point now, double cpu_ms, double wait_ms,
                      double gpu_ms) {
    static struct {
        std::chrono::steady_clock::time_point start{};
        u64 frames{}, gpu_n{}, slow{};
        double cpu{}, gpu{}, wait{}, worst{};
    } a;
    if (a.start.time_since_epoch().count() == 0) {
        a.start = now;
    }
    const double f = cpu_ms + wait_ms;
    a.cpu += cpu_ms;
    a.wait += wait_ms;
    a.worst = std::max(a.worst, f);
    a.slow += f > 50.0 ? 1 : 0;
    if (gpu_ms > 0) {
        a.gpu += gpu_ms;
        ++a.gpu_n;
    }
    ++a.frames;
    const double el = std::chrono::duration<double>(now - a.start).count();
    if (el < 10.0) {
        return;
    }
    auto& c = BBPerf2::Get();
    const auto take = [](std::atomic<u64>& v) { return v.exchange(0, std::memory_order_relaxed); };
    const u64 gp = take(c.gfx_pipelines), cp = take(c.cs_pipelines), sh = take(c.shaders);
    const double comp_ms = take(c.compile_us) / 1000.0;
    const u64 faults = take(c.cpu_writes), binds = take(c.sparse_binds);
    const u64 fa_pages = take(c.fault_around_pages); // bb-native-hot
    const double wait_stream = take(c.wait_stream_us) / 1000.0;
    const double wait_fbuf = take(c.wait_faultbuf_us) / 1000.0;
    const double mib = 1024.0 * 1024.0 * el;
    const double up = take(c.buf_up_bytes) / mib, down = take(c.buf_down_bytes) / mib;
    const u64 tex_n = take(c.tex_uploads);
    const double tex = take(c.tex_up_bytes) / mib;
    const u64 waits = take(c.waits);
    const double wait_total = take(c.wait_us) / 1000.0;
    // bb-native-texhash and the GPU command thread's busy share (GPU starvation when ~100 %).
    const u64 tex_gpu_n = take(c.tex_gpu_src), skip_n = take(c.tex_skips);
    const double skip_mib = take(c.tex_skip_bytes) / mib;
    const double cp_busy = take(c.cp_busy_us) / 1000.0 / el; // ms per second
    const double cp_game = take(c.cp_guest_wait_us) / 1000.0 / el;
    const double cp_vo = take(c.cp_vo_wait_us) / 1000.0 / el;
    LOG_INFO(Render_Vulkan,
             "[perf2] 10s: fps {:.1f} frame {:.2f} ms (worst {:.1f}, >50ms {}) cpu {:.2f} gpu "
             "{:.2f} fence {:.2f} | compiles {} gfx {} cs {} shaders ({:.0f} ms) | cpu-write "
             "faults {:.0f}/s (fault-around {:.0f} pages/s) | uploads buf {:.1f} tex {:.1f} "
             "MiB/s ({:.0f} img/s) readback {:.2f} MiB/s | sparse binds {} | gpu waits {:.0f}/s {:.1f} ms/s (stream buffer {:.1f}, "
             "fault buffer {:.1f}) | BB_PERF2={} | tex from GPU data {:.0f} img/s, unchanged "
             "skipped {:.0f} img/s {:.1f} MiB/s | gpu thread awake {:.0f} ms/s: waits on game {:.0f}, "
             "on flip {:.0f}, work {:.0f} ms/s ({:.2f} ms/frame)",
             a.frames / el, el * 1000.0 / a.frames, a.worst, a.slow, a.cpu / a.frames,
             a.gpu_n ? a.gpu / a.gpu_n : 0.0, a.wait / a.frames, gp, cp, sh, comp_ms,
             faults / el, fa_pages / el, up, tex, tex_n / el, down, binds, waits / el,
             wait_total / el, wait_stream / el, wait_fbuf / el, BBPerf2::Enabled() ? 1 : 0,
             tex_gpu_n / el, skip_n / el, skip_mib, cp_busy, cp_game, cp_vo,
             std::max(0.0, cp_busy - cp_game - cp_vo - a.wait / el),
             std::max(0.0, cp_busy - cp_game - cp_vo - a.wait / el) * el / a.frames);
    a = {};
    a.start = now;
}

namespace {
double g_sharpen_ms = 0.0;
u32 g_sharpen_n = 0;
} // namespace

void NoteSharpen(double ms) {
    g_sharpen_ms += ms;
    ++g_sharpen_n;
}

void NoteFrame(double cpu_ms, double wait_ms, double gpu_ms, double sr_ms, double mv_ms,
               u32 in_w, u32 in_h, u32 sr_w, u32 sr_h, u32 out_w, u32 out_h, bool upscaled) {
    const auto now = std::chrono::steady_clock::now();
    NotePerf2(now, cpu_ms, wait_ms, gpu_ms);
    if (g_acc.frames == 0 && g_acc.start.time_since_epoch().count() == 0) {
        g_acc.start = now;
    }
    g_acc.cpu += cpu_ms;
    g_acc.wait += wait_ms;
    g_acc.frame_max = std::max(g_acc.frame_max, cpu_ms + wait_ms);
    if (gpu_ms > 0) {
        g_acc.gpu += gpu_ms;
        g_acc.sr += sr_ms;
        g_acc.mv += mv_ms;
        ++g_acc.gpu_n;
    }
    ++g_acc.frames;
    const double elapsed = std::chrono::duration<double>(now - g_acc.start).count();
    if (elapsed < 1.0) {
        return;
    }
    Stats s;
    s.fps = static_cast<float>(g_acc.frames / elapsed);
    s.frame_ms = static_cast<float>(elapsed * 1000.0 / g_acc.frames);
    s.cpu_ms = static_cast<float>(g_acc.cpu / g_acc.frames);
    s.wait_ms = static_cast<float>(g_acc.wait / g_acc.frames);
    const double gn = std::max<u32>(1, g_acc.gpu_n);
    s.gpu_ms = static_cast<float>(g_acc.gpu / gn);
    s.sr_ms = static_cast<float>(g_acc.sr / gn);
    s.mv_ms = static_cast<float>(g_acc.mv / gn);
    s.in_w = in_w;
    s.in_h = in_h;
    s.out_w = out_w;
    s.out_h = out_h;
    s.sr_w = upscaled ? sr_w : 0;
    s.sr_h = upscaled ? sr_h : 0;
    s.upscaled = upscaled;
    g_stats = s;
    g_hist_mean.push_back(s.frame_ms);
    g_hist_max.push_back(static_cast<float>(g_acc.frame_max));
    if (g_hist_mean.size() > 60) {
        g_hist_mean.erase(g_hist_mean.begin());
        g_hist_max.erase(g_hist_max.begin());
    }
    LOG_INFO(
        Render_Vulkan,
        "[perf] fps {:.1f} frame {:.2f} ms | cpu {:.2f} ms (fence wait {:.2f}) | gpu {:.2f} "
        "ms | upscaler {} {} {}x{} -> {} -> {}x{} (window) sr {:.3f} ms mv {:.3f} ms{}",
        s.fps, s.frame_ms, s.cpu_ms, s.wait_ms, s.gpu_ms,
        Enabled() ? SR::KindName(Cfg().kind) : "off",
        Enabled() ? SR::QualityName(Cfg().quality) : "-", in_w, in_h,
        s.sr_w ? fmt::format("{}x{}", s.sr_w, s.sr_h) : std::string("not upscaled"), out_w, out_h,
        s.sr_ms, s.mv_ms,
        (Cfg().kind == SR::Kind::Taa ? fmt::format(" | taa {:.3f} ms", s.sr_ms) : std::string()) +
            (g_sharpen_n != 0 ? fmt::format(" | rcas {:.3f} ms", g_sharpen_ms / g_sharpen_n)
                              : std::string()));
    g_sharpen_ms = 0.0;
    g_sharpen_n = 0;
    g_acc = {};
    g_acc.start = now;
}

const Stats& LastStats() {
    return g_stats;
}

bool OverlayShown() {
    return g_overlay.load(std::memory_order_relaxed);
}

// ---- The F8 panel: Demon's Souls' "Metricas" (ds-menu.cpp, MetricsPanel) row for row. ----
namespace Metrics {
struct Row {
    std::string name, value;
    ImU32 color;
};
constexpr ImU32 kGreen = IM_COL32(50, 205, 50, 255), kRed = IM_COL32(255, 0, 0, 255),
                kBlue = IM_COL32(0, 191, 255, 255), kOrange = IM_COL32(255, 165, 0, 255),
                kWhite = IM_COL32(255, 255, 255, 255), kGold = IM_COL32(255, 215, 0, 255),
                kName = IM_COL32(150, 150, 150, 255);
std::mutex g_mutex;
std::vector<Row> g_sys_rows; // GPU .. Commit, from the sampling thread once a second
std::atomic<bool> g_started{false};
ImFont* g_font = nullptr; // Consolas Bold (consolab.ttf), as Demon's

ImU32 UseColor(double used, double total) {
    if (total <= 0)
        return kOrange;
    const double f = used / total;
    return f < 0.70 ? kGreen : f < 0.85 ? kGold : f < 0.95 ? kOrange : kRed;
}

#if defined(_WIN32)
// NVML (nvml.dll, the driver's: what nvidia-smi queries).
struct Nvml {
    bool ok = false;
    void* dev = nullptr;
    int (*util)(void*, unsigned int*) = nullptr; // nvmlDeviceGetUtilizationRates (gpu, memory)
    int (*temp)(void*, int, unsigned int*) = nullptr;
    int (*clock)(void*, int, unsigned int*) = nullptr;
    int (*mem)(void*, unsigned long long*) = nullptr; // nvmlDeviceGetMemoryInfo (total, free, used)
    void Init() {
        HMODULE m = LoadLibraryW(L"nvml.dll");
        if (m == nullptr) {
            wchar_t sys[MAX_PATH] = {};
            GetSystemDirectoryW(sys, MAX_PATH);
            m = LoadLibraryW((std::wstring(sys) + L"\\nvml.dll").c_str());
        }
        if (m == nullptr)
            return;
        auto init = reinterpret_cast<int (*)()>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlInit_v2")));
        auto get = reinterpret_cast<int (*)(unsigned int, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetHandleByIndex_v2")));
        util = reinterpret_cast<int (*)(void*, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetUtilizationRates")));
        temp = reinterpret_cast<int (*)(void*, int, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetTemperature")));
        clock = reinterpret_cast<int (*)(void*, int, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetClockInfo")));
        mem = reinterpret_cast<int (*)(void*, unsigned long long*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetMemoryInfo")));
        ok = init != nullptr && get != nullptr && util != nullptr && temp != nullptr &&
             mem != nullptr && init() == 0 && get(0, &dev) == 0;
    }
};
// "\Processor Information(_Total)\% Processor Performance" (pdh.dll by name) x the maximum clock.
struct CpuPerf {
    void* query = nullptr;
    void* counter = nullptr;
    long (*collect)(void*) = nullptr;
    long (*value)(void*, unsigned long, unsigned long*, void*) = nullptr;
    int max_mhz = 0;
    void Init() {
        HKEY key = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                          0, KEY_READ, &key) == ERROR_SUCCESS) {
            DWORD mhz = 0, size = sizeof(mhz);
            if (RegQueryValueExW(key, L"~MHz", nullptr, nullptr, reinterpret_cast<BYTE*>(&mhz),
                                 &size) == ERROR_SUCCESS)
                max_mhz = static_cast<int>(mhz);
            RegCloseKey(key);
        }
        HMODULE m = LoadLibraryW(L"pdh.dll");
        if (m == nullptr)
            return;
        auto open = reinterpret_cast<long (*)(const wchar_t*, uintptr_t, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhOpenQueryW")));
        auto add = reinterpret_cast<long (*)(void*, const wchar_t*, uintptr_t, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhAddEnglishCounterW")));
        collect = reinterpret_cast<long (*)(void*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhCollectQueryData")));
        value = reinterpret_cast<long (*)(void*, unsigned long, unsigned long*, void*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhGetFormattedCounterValue")));
        if (open == nullptr || add == nullptr || collect == nullptr || value == nullptr)
            return;
        if (open(nullptr, 0, &query) != 0)
            return;
        if (add(query, L"\\Processor Information(_Total)\\% Processor Performance", 0, &counter) !=
            0)
            counter = nullptr;
        if (counter != nullptr)
            collect(query);
    }
    int Mhz() {
        if (counter == nullptr || max_mhz <= 0)
            return 0;
        struct {
            unsigned long status;
            double value;
        } v{};
        collect(query);
        if (value(counter, 0x200 /* PDH_FMT_DOUBLE */, nullptr, &v) != 0)
            return 0;
        return static_cast<int>(std::lround(v.value * max_mhz / 100.0));
    }
};

void Sample(Nvml& nv, CpuPerf& cpu, uint64_t& idle0, uint64_t& total0) {
    std::vector<Row> rows;
    if (nv.ok) {
        unsigned int u[2] = {}, t = 0, clk = 0;
        unsigned long long m[3] = {};
        nv.util(nv.dev, u);
        nv.temp(nv.dev, 0, &t);
        if (nv.clock != nullptr)
            nv.clock(nv.dev, 0 /* graphics */, &clk);
        nv.mem(nv.dev, m);
        const double used = static_cast<double>(m[2] >> 20),
                     total = static_cast<double>(m[0] >> 20);
        rows.push_back({"GPU", fmt::format("{:3} %   {} C", u[0], t), kOrange});
        if (clk > 0)
            rows.push_back({"GPU reloj", fmt::format("{:4} MHz", clk), kOrange});
        rows.push_back({"VRAM", fmt::format("{:.1f} / {:.0f} GB", used / 1024.0, total / 1024.0),
                        UseColor(used, total)});
    } else {
        rows.push_back({"GPU", "(sin nvidia-smi)", kOrange});
    }
    FILETIME fi, fk, fu;
    if (GetSystemTimes(&fi, &fk, &fu)) {
        const auto q = [](FILETIME f) {
            return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
        };
        const uint64_t idle = q(fi), total = q(fk) + q(fu);
        const double busy = total > total0 ? 100.0 * (1.0 - static_cast<double>(idle - idle0) /
                                                                static_cast<double>(total - total0))
                                           : 0.0;
        idle0 = idle, total0 = total;
        rows.push_back(
            {"CPU",
             fmt::format("{:3} %", static_cast<int>(std::lround(std::clamp(busy, 0.0, 100.0)))),
             kOrange});
    }
    if (const int mhz = cpu.Mhz(); mhz > 0)
        rows.push_back({"CPU reloj", fmt::format("{:4} MHz", mhz), kOrange});
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        const double gb = 1024.0 * 1024.0 * 1024.0;
        const double used = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys) / gb,
                     total = static_cast<double>(ms.ullTotalPhys) / gb;
        rows.push_back(
            {"RAM", fmt::format("{:.1f} / {:.0f} GB", used, total), UseColor(used, total)});
        const double climit = static_cast<double>(ms.ullTotalPageFile) / gb,
                     cused = climit - static_cast<double>(ms.ullAvailPageFile) / gb;
        if (climit > 0)
            rows.push_back({"Commit", fmt::format("{:.1f} / {:.0f} GB", cused, climit),
                            cused / climit > 0.90 ? kRed : UseColor(cused, climit)});
    }
    std::scoped_lock lock(g_mutex);
    g_sys_rows = std::move(rows);
}
#endif

// The sampling thread (Demon's MetricsPanel::Start): once a second while the panel is shown.
void Start() {
    if (g_started.exchange(true))
        return;
#if defined(_WIN32)
    std::thread([] {
        Nvml nv;
        CpuPerf cpu;
        nv.Init();
        cpu.Init();
        uint64_t idle0 = 0, total0 = 0;
        for (;;) {
            if (g_overlay.load(std::memory_order_relaxed))
                Sample(nv, cpu, idle0, total0);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
#endif
}

const char* ModeEs(SR::Quality q) {
    switch (q) {
    case SR::Quality::Native:
        return "Nativo AA";
    case SR::Quality::UltraQualityPlus:
        return "Ultra calidad+";
    case SR::Quality::UltraQuality:
        return "Ultra calidad";
    case SR::Quality::Quality:
        return "Calidad";
    case SR::Quality::Balanced:
        return "Equilibrado";
    case SR::Quality::Performance:
        return "Rendimiento";
    case SR::Quality::UltraPerformance:
        return "Ultra rendimiento";
    case SR::Quality::Dynamic:
        return "Dinamico";
    }
    return "";
}
} // namespace Metrics

void LoadOverlayFont() {
#if defined(_WIN32)
    // Demon's systemOverlay.cpp: %WINDIR%\Fonts\consolab.ttf at 32 px.
    wchar_t win[MAX_PATH] = {};
    if (GetWindowsDirectoryW(win, MAX_PATH) == 0)
        return;
    const std::filesystem::path p = std::filesystem::path(win) / "Fonts" / "consolab.ttf";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec))
        return;
    const auto u8 = p.u8string();
    Metrics::g_font = ImGui::GetIO().Fonts->AddFontFromFileTTF(
        reinterpret_cast<const char*>(u8.c_str()), 32.0f);
#endif
}

void ToggleOverlay() {
    g_overlay.store(!g_overlay.load(std::memory_order_relaxed), std::memory_order_relaxed);
    if (g_overlay.load(std::memory_order_relaxed))
        Metrics::Start();
}

// The F8 panel as Demon's Souls' "Metricas" (ds-menu.cpp, MetricsPanel::Draw): top-centre, no
// background, Consolas Bold 15 / 16 / 34 / 10 px x S, the same rows in the same order (GPU, GPU
// reloj, VRAM, CPU, CPU reloj, RAM, Commit from the sampling thread; Frame, Escala, Resol.,
// Modelo from this build's per-second stats), the FPS line and the 60 s frametime graph.
// Nothing new on the render thread: no GPU queries, no readbacks.
void DrawOverlay() {
    if (!OverlayShown()) {
        return;
    }
    using namespace Metrics;
    const Stats& s = g_stats;
    const bool on = Enabled();
    const bool ready = s.frame_ms > 0;
    const ImGuiIO& io = ImGui::GetIO();
    const float width = io.DisplaySize.x;
    const float S = std::clamp(io.DisplaySize.y / 1080.0f, 1.0f, 2.5f);
    ImFont* f = g_font != nullptr ? g_font : ImGui::GetFont();

    std::vector<Row> rows;
    {
        std::scoped_lock lock(g_mutex);
        rows = g_sys_rows;
    }
    if (ready) {
        rows.push_back({"Frame", fmt::format("{:.1f} ms", s.frame_ms), kOrange});
        const char* dot = " \xC2\xB7 ";
        const char* arrow = "\xE2\x86\x92";
        if (on) {
            const SR::Kind k = Cfg().kind;
            const ImU32 col = k == SR::Kind::Dlss                          ? kGreen
                              : k == SR::Kind::Fsr3 || k == SR::Kind::Fsr4 ? kRed
                              : k == SR::Kind::Xess                        ? kBlue
                                                                           : kOrange;
            const std::string rk = k == SR::Kind::Fsr4   ? "FSR 4"
                                   : k == SR::Kind::Fsr3 ? "FSR"
                                                         : SR::KindName(k);
            if (k == SR::Kind::Taa) {
                rows.push_back({"AA", fmt::format("TAA{}{:.2f} ms", dot, s.sr_ms), kOrange});
            } else {
                rows.push_back({"Escala", rk + dot + ModeEs(Cfg().quality), col});
            }
            const double ratio = s.in_h > 0 ? static_cast<double>(s.out_h) / s.in_h : 0.0;
            const u32 sr_h = s.sr_h ? s.sr_h : s.out_h;
            const double ratio2 = s.in_h > 0 ? static_cast<double>(sr_h) / s.in_h : 0.0;
            (void)ratio;
            rows.push_back({"Resol.",
                            s.in_h > 0 ? fmt::format("{}p {} {}p (x{:.3g}) {} {}p", s.in_h, arrow,
                                                     sr_h, ratio2, arrow, s.out_h)
                                       : fmt::format("{} {}p", arrow, s.out_h),
                            col});
            std::string model;
            if (k == SR::Kind::Dlss) {
                const char* pe = std::getenv("BB_DLSS_PRESET");
                const std::string pk = pe != nullptr && *pe != '\0' ? std::string(1, *pe) : "";
                model = pk == "K"   ? "K (Transformer, DLSS 4)"
                        : pk == "L" ? "L (Transformer, DLSS 4.5)"
                        : pk == "M" ? "M (Transformer, DLSS 4.5)"
                        : pk.empty() ? "auto (NVIDIA)"
                                     : pk;
            } else if (k == SR::Kind::Xess) {
                model = "XeSS";
            } else if (k == SR::Kind::Fsr3) {
                model = "FSR 3.1";
            } else if (k == SR::Kind::Fsr4) {
                model = SR::Fsr4::Active411()
                            ? fmt::format("FSR 4.1.1 ({})", SR::Fsr4::ModelName411())
                            : std::string("FSR 4 (IA)");
            }
            if (!model.empty())
                rows.push_back({"Modelo", model, col});
        } else {
            rows.push_back({"Escala", "Nativo", kOrange});
            rows.push_back({"Resol.", fmt::format("{}p {} {}p", s.in_h, arrow, s.out_h), kOrange});
        }
    }
    if (ready) {
        float fr = 0, fs = 0;
        const char* st = "off";
        if (FG::PanelInfo(fr, fs, st)) {
            const bool gen = std::strcmp(st, "on") == 0;
            FG::DlssSl::SecondStats sl{};
            if (FG::DlssSl::Stats(sl)) {
                // bb-sl-metrics: DLSS-G through Streamline: its own per-second numbers.
                rows.push_back(
                    {"FG",
                     gen ? fmt::format("{} \xC2\xB7 on x{} \xC2\xB7 {:.0f} {} {:.0f} fps "
                                       "({:.0f} gen/s)",
                                       FG::PanelName(), sl.presented_per_frame, fr,
                                       "\xE2\x86\x92", fs, sl.generated_fps)
                         : fmt::format("{} \xC2\xB7 {} (x{} pedido)", FG::PanelName(), st,
                                       sl.multiplier),
                     gen ? kGreen : kOrange});
                rows.push_back({"Reflex", sl.reflex ? "on (Streamline)" : "off",
                                sl.reflex ? kGreen : kOrange});
            } else
            rows.push_back({"FG", gen ? fmt::format("{} \xC2\xB7 on \xC2\xB7 {:.0f} {} {:.0f} fps",
                                                    FG::PanelName(), fr, "\xE2\x86\x92", fs)
                                      : fmt::format("{} \xC2\xB7 {}", FG::PanelName(), st),
                            gen ? kGreen : kOrange});
        } else {
            rows.push_back({"FG", "off", kOrange});
        }
    }
    if (ready) {
        bool d5 = false;
        const std::string d5_line = Dlss5::PanelLine(d5);
        rows.push_back({"DLSS 5", d5_line, d5 ? kGreen : kOrange});
    }
    if (rows.empty())
        rows.push_back({"", "Cargando...", kOrange});

    const float row_px = std::round(15 * S), title_px = std::round(16 * S),
                fps_px = std::round(34 * S), small_px = std::round(10 * S);
    const float pad = std::round(8 * S), row_h = std::ceil(row_px * 1.12f),
                title_h = std::ceil(title_px * 1.4f),
                fps_h = ready ? std::ceil(fps_px * 1.1f) : 0.0f, graph_h = std::round(90 * S);
    float name_w = std::round(90 * S), val_w = std::round(230 * S), wide_w = 0;
    for (const auto& r : rows) {
        const float w = f->CalcTextSizeA(row_px, FLT_MAX, 0, r.value.c_str()).x;
        if (!r.name.empty()) {
            name_w = std::max(name_w, f->CalcTextSizeA(row_px, FLT_MAX, 0, r.name.c_str()).x +
                                          std::round(10 * S));
            val_w = std::max(val_w, w);
        } else {
            wide_w = std::max(wide_w, w);
        }
    }
    const float W = std::min(std::max(name_w + val_w, wide_w) + 2 * pad, width * 0.4f);
    const float x0 = std::floor((width - W) * 0.5f), y0 = std::round(10 * S);
    auto* dl = ImGui::GetForegroundDrawList();

    const char* title = "BLOODBORNE";
    const float tw = f->CalcTextSizeA(title_px, FLT_MAX, 0, title).x;
    dl->AddText(f, title_px, {std::floor(x0 + (W - tw) * 0.5f), y0 + pad}, kWhite, title);
    float y = y0 + pad + title_h;
    const ImVec4 clip{x0, y0, x0 + W, y0 + 4000};
    for (const auto& r : rows) {
        if (!r.name.empty()) {
            dl->AddText(f, row_px, {x0 + pad, y}, kName, r.name.c_str(), nullptr, 0, &clip);
            dl->AddText(f, row_px, {x0 + pad + name_w, y}, r.color, r.value.c_str(), nullptr, 0,
                        &clip);
        } else {
            dl->AddText(f, row_px, {x0 + pad, y}, r.color, r.value.c_str(), nullptr, 0, &clip);
        }
        y += row_h;
    }
    if (ready) {
        float fg_real = 0, fg_shown = 0;
        const char* fg_state = "";
        const bool fg_gen = FG::PanelInfo(fg_real, fg_shown, fg_state) &&
                            std::strcmp(fg_state, "on") == 0 && fg_shown > fg_real * 1.2f;
        const std::string fps =
            fg_gen ? fmt::format("FPS real {:.0f} / mostrados {:.0f} (FG x{:.0f})", fg_real, fg_shown,
                                 [&] {
                                     // bb-sl-metrics: Streamline's real presented-per-frame.
                                     FG::DlssSl::SecondStats sl{};
                                     return FG::DlssSl::Stats(sl) && sl.presented_per_frame > 1
                                                ? float(sl.presented_per_frame)
                                                : std::round(fg_shown / std::max(fg_real, 1.0f));
                                 }())
                   : fmt::format("FPS {:.0f}", s.fps);
        dl->AddText(f, fps_px, {x0 + pad, y}, kWhite, fps.c_str());
        y += fps_h;
    }
    y += pad;

    const float gy = y, gw = W, top = 40.0f;
    for (const float ms : {8.3f, 16.7f, 33.3f}) {
        const float ly = gy + graph_h - ms / top * graph_h;
        dl->AddLine({x0, ly}, {x0 + gw, ly}, IM_COL32(60, 60, 60, 255));
    }
    dl->AddText(f, small_px, {x0, gy}, IM_COL32(128, 128, 128, 255),
                "frametime (ms)  40   verde media, naranja picos");
    dl->AddText(f, small_px, {x0, gy + graph_h - 16.7f / top * graph_h - small_px},
                IM_COL32(128, 128, 128, 255), "16.7");
    const auto line = [&](const std::vector<float>& list, ImU32 col) {
        if (list.size() < 2)
            return;
        const float step = gw / 59.0f, xs = x0 + gw - static_cast<float>(list.size() - 1) * step;
        for (size_t i = 1; i < list.size(); i++) {
            const float y1 = gy + graph_h - std::min(list[i - 1], top) / top * graph_h;
            const float y2 = gy + graph_h - std::min(list[i], top) / top * graph_h;
            dl->AddLine({xs + static_cast<float>(i - 1) * step, y1},
                        {xs + static_cast<float>(i) * step, y2}, col, 2.0f * S);
        }
    };
    line(g_hist_max, IM_COL32(255, 150, 40, 255));
    line(g_hist_mean, IM_COL32(0, 255, 0, 255));
}

} // namespace BB::Upscale
