// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021: see bb_upscale.h. Config, back end set-up (from bb-combo-159), the camera from
// Bloodborne's scene constant block, jitter and the once-per-second performance line.

#include "video_core/bb/bb_upscale.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
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
    }
    // (BB_UPSCALER=native or anything else: the stock path, nothing of this module runs.)
    const std::string q = Lower(Env("BB_UPSCALE_MODE"));
    if (up != "dlaa" && !q.empty()) {
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

// Written by the GPU thread (OnConstants), read at the flip (same thread in shadPS4, the
// mutex only guards against a presenter on another thread; it is taken on a hit, a few times
// per frame).
std::mutex g_cam_mutex;
Mat g_cur_vp;
u64 g_cur_area = 0;
bool g_cur_valid = false;
Mat g_prev_vp;
bool g_prev_valid = false;

u32 g_frame_no = 0;
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
    u32 frames = 0, gpu_n = 0;
    std::chrono::steady_clock::time_point start{};
} g_acc;

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

double Ratio() {
    return SR::QualityRatio(Cfg().kind, Cfg().quality);
}

void Checkpoint(std::string_view text) {
    LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}", text);
}

float LiveSharpness() {
    return Cfg().sharpness;
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
    return next;
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family) {
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        SR::Dlss::OnDevice(instance, physical, device);
        break;
    case SR::Kind::Fsr4:
        SR::Fsr4::OnDevice(instance, physical, device, allocator, queue, family);
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

// Bloodborne's scene constant block (864 bytes; 1.59 BlockCamera): [0] far plane 3000,
// [1] 1 / far, [4] [5] the render size, floats 8-19 the view (3x4 row-major), 52-67 the
// projection (4x4 row-major). The largest pass's block of the frame is its camera.
void OnConstants(const u32* data, size_t count) {
    if (count < 216) {
        return;
    }
    const float* d = reinterpret_cast<const float*>(data);
    if (d[0] != 3000.0f) {
        return; // (the one test every other upload pays)
    }
    if (std::fabs(d[1] * d[0] - 1.0f) > 1e-3f || !(d[4] >= 64.0f) || !(d[5] >= 64.0f) ||
        d[4] > 16384.0f || d[5] > 16384.0f) {
        return;
    }
    const u64 area = static_cast<u64>(d[4]) * static_cast<u64>(d[5]);
    std::scoped_lock lk{g_cam_mutex};
    if (area <= g_cur_area) {
        return;
    }
    Mat view, proj;
    for (int i = 0; i < 12; ++i) view.m[i] = d[8 + i];
    view.m[15] = 1.0;
    for (int i = 0; i < 16; ++i) proj.m[i] = d[52 + i];
    const Mat vp = Multiply(proj, view);
    if (!LooksLikeViewProj(vp)) {
        return;
    }
    g_cur_vp = vp;
    g_cur_area = area;
    g_cur_valid = true;
}

void Jitter(float& x, float& y) {
    x = g_jitter[0];
    y = g_jitter[1];
}

Camera EndFrame(u32 flip_w, u32 flip_h) {
    Camera cam;
    {
        std::scoped_lock lk{g_cam_mutex};
        if (g_cur_valid) {
            Mat inv;
            if (Invert(g_cur_vp, inv)) {
                const Mat prev = g_prev_valid ? g_prev_vp : g_cur_vp;
                const Mat to_prev = Multiply(prev, inv);
                for (int i = 0; i < 16; ++i) {
                    cam.view_to_clip[i] = static_cast<float>(g_cur_vp.m[i]);
                    cam.clip_to_prev_clip[i] = static_cast<float>(to_prev.m[i]);
                }
                const double* r1 = &g_cur_vp.m[4];
                const double* r2 = &g_cur_vp.m[8];
                const double* r3 = &g_cur_vp.m[12];
                cam.fov_y =
                    static_cast<float>(2.0 * std::atan(Len3(r3) / std::max(Len3(r1), 1e-9)));
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
                // A camera cut (a jump far beyond normal motion): history reset.
                if (g_prev_valid) {
                    double diff = 0;
                    for (int i = 0; i < 16; ++i) diff += std::fabs(to_prev.m[i] - (i % 5 == 0));
                    cam.reset = diff > 3.0;
                }
            }
            g_prev_vp = g_cur_vp;
            g_prev_valid = true;
        } else {
            g_prev_valid = false;
        }
        g_cur_valid = false;
        g_cur_area = 0;
    }
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

void NoteFrame(double cpu_ms, double wait_ms, double gpu_ms, double sr_ms, double mv_ms,
               u32 in_w, u32 in_h, u32 out_w, u32 out_h, bool upscaled) {
    const auto now = std::chrono::steady_clock::now();
    if (g_acc.frames == 0 && g_acc.start.time_since_epoch().count() == 0) {
        g_acc.start = now;
    }
    g_acc.cpu += cpu_ms;
    g_acc.wait += wait_ms;
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
    s.upscaled = upscaled;
    g_stats = s;
    LOG_INFO(Render_Vulkan,
             "[perf] fps {:.1f} frame {:.2f} ms | cpu {:.2f} ms (fence wait {:.2f}) | gpu {:.2f} "
             "ms | upscaler {} {} {}x{} -> {}x{} sr {:.3f} ms mv {:.3f} ms",
             s.fps, s.frame_ms, s.cpu_ms, s.wait_ms, s.gpu_ms,
             Enabled() ? SR::KindName(Cfg().kind) : "off",
             Enabled() ? SR::QualityName(Cfg().quality) : "-", in_w, in_h, out_w, out_h,
             s.sr_ms, s.mv_ms);
    g_acc = {};
    g_acc.start = now;
}

const Stats& LastStats() {
    return g_stats;
}

bool OverlayShown() {
    return g_overlay.load(std::memory_order_relaxed);
}

void ToggleOverlay() {
    g_overlay.store(!g_overlay.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

void DrawOverlay() {
    if (!OverlayShown()) {
        return;
    }
    const Stats& s = g_stats;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.55f);
    if (ImGui::Begin("##bbperf", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                         ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs)) {
        ImGui::Text("%.1f fps  %.2f ms", s.fps, s.frame_ms);
        ImGui::Text("CPU %.2f ms (espera %.2f)  GPU %.2f ms", s.cpu_ms, s.wait_ms, s.gpu_ms);
        if (Enabled()) {
            ImGui::Text("%s %s %ux%u -> %ux%u%s", SR::KindName(Cfg().kind),
                        SR::QualityName(Cfg().quality), s.in_w, s.in_h, s.out_w, s.out_h,
                        s.upscaled ? "" : " (sin escalar)");
            ImGui::Text("SR %.2f ms  MV %.2f ms", s.sr_ms, s.mv_ms);
        } else {
            ImGui::Text("Nativo %ux%u -> %ux%u", s.in_w, s.in_h, s.out_w, s.out_h);
        }
    }
    ImGui::End();
}

} // namespace BB::Upscale
