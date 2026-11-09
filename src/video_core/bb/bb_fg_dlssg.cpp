// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-dlssg: NVIDIA DLSS Frame Generation (DLSS-G, multi frame generation) through
// Streamline, for BB_FG=dlss-x2 | dlss-x3 | dlss-x4 | dlss-x5 | dlss-x6 | dlss-dyn. Ported from
// the 1.59 build's bb_fg.cpp (namespace Dlssg) without what made 1.59 slow:
//  - no guest flip pacer, no present-id / present wait, no adaptive / locked step, no sleeps of
//    our own, no Reflex markers on the game's threads (controller read, gnmdriver submits): every
//    Streamline call (frame token, constants, tags, PCL markers, options, state) is made on the
//    presenter's thread at the frame boundary, once per present; nothing per draw, nothing on the
//    GpuCommandProcessor thread, no fence or wait added to the render path. slReflexSleep is off
//    unless BB_DLSSG_SLEEP=1.
//  - Streamline's sl.interposer.dll is the Vulkan loader only with BB_FG=dlss-* (DLSS-G needs its
//    vkCreateDevice / swapchain / present hooks). Its per-command entry points (vkCmdBindPipeline,
//    vkCmdBindDescriptorSets, vkCmdPipelineBarrier, vkBeginCommandBuffer, vkCreateImage) are plain
//    forwards in Streamline 2.14; OnDispatch points the emulator's dispatcher for those straight
//    back at the system loader, so the GPU command thread pays nothing per draw for the
//    interposer (BB_SL_PASSTHROUGH=0: left on the interposer, as 1.59).
// Dropped from 1.59: the DLSS-G preset hook (BB_DLSSG_PRESET), the HUD-less image, the present
// ids / [display] module, the Reflex latency report / [hitch] lines, the VRR cap limiter.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "video_core/bb/bb_fg.h"
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
#if defined(_WIN32) && defined(BB_STREAMLINE)
// (sl_pcl.h's C++23 branch aliases std::to_underlying, which is not valid C++: its own helper.)
#pragma push_macro("__cplusplus")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#undef __cplusplus
#define __cplusplus 202002L
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#pragma clang diagnostic pop
#pragma pop_macro("__cplusplus")
#endif

namespace BB::FG::Dlssg {
namespace {
using Clock = std::chrono::steady_clock;

bool g_ready = false;
bool g_dynamic = false;
int g_generate = 1;  // generated frames per real frame (fixed), or the dynamic maximum
int g_dyn_gen = 1;   // the dynamic multiplier's current generated count
int g_dyn_max = 3;   // min(asked, numFramesToGenerateMax) (3 until Streamline says)
double g_target_fps = 0.0;
int g_mode_set = -1; // DLSS-G last set on (1) or off (0)

int CurrentGenerate() {
    return g_dynamic ? g_dyn_gen : g_generate;
}

bool EnvIs(const char* name, char value) {
    const char* v = Env(name);
    return v != nullptr && *v == value;
}

uint32_t g_sc_images = 0;
// Hardware-accelerated GPU scheduling (DLSS-G on Windows needs it): HwSchMode 2 = on, 1 = off.
const char* HagsState() {
#if defined(_WIN32)
    DWORD v = 0, size = sizeof(v);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers",
                     "HwSchMode", RRF_RT_REG_DWORD, nullptr, &v, &size) == ERROR_SUCCESS) {
        return v == 2 ? "on (HwSchMode 2)" : "OFF (HwSchMode != 2: DLSS-G needs HAGS)";
    }
    return "unknown (HwSchMode not set: OFF by default)";
#else
    return "n/a";
#endif
}
const char* g_sc_mode = "?";
uint32_t g_last_actual = 0, g_last_minwh = 0;
bool g_last_vsync = false;

double RefreshHz() {
#if defined(_WIN32)
    DEVMODEA dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &dm) != 0 && dm.dmDisplayFrequency > 1) {
        return static_cast<double>(dm.dmDisplayFrequency);
    }
#endif
    return 0.0;
}

#if defined(_WIN32) && defined(BB_STREAMLINE)
struct Api {
    PFun_slGetNewFrameToken* get_new_frame_token = nullptr;
    PFun_slSetConstants* set_constants = nullptr;
    PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
    PFun_slFreeResources* free_resources = nullptr;
    PFun_slDLSSGSetOptions* dlssg_set_options = nullptr;
    PFun_slDLSSGGetState* dlssg_get_state = nullptr;
    PFun_slReflexSetOptions* reflex_set_options = nullptr;
    PFun_slReflexSleep* reflex_sleep = nullptr;
    PFun_slPCLSetMarker* pcl_set_marker = nullptr;
} g_api;

sl::FrameToken* g_token = nullptr;
uint32_t g_frame_index = 0;
bool g_options_set = false;
vk::Extent2D g_options_backbuffer{}, g_options_render{};
vk::Format g_options_format = vk::Format::eUndefined;
uint32_t g_options_images = 0;
uint64_t g_presents = 0;
uint64_t g_mode_on_at = 0;
uint32_t g_nogen_run = 0, g_retries = 0;
bool g_resume_pending = false;
Clock::time_point g_backoff_until{};
Clock::time_point g_present_begin{};
bool g_prev_inputs_ok = false;

// The per-second [dlssg] line.
struct Stats {
    Clock::time_point start{};
    uint32_t presents = 0, on = 0, inputs = 0, resets = 0, nocamera = 0, shown = 0;
    double before_ms = 0, after_ms = 0, present_ms = 0, present_max = 0, acquire_ms = 0,
           acquire_max = 0, before_max = 0, after_max = 0;
    uint32_t acquires = 0, generated = 0;
    uint64_t actual_sum = 0;
} g_stats;
uint32_t g_last_status = 0, g_last_max = 0, g_last_presented = 1;
int g_dyn_pend = -1, g_dyn_pend_n = 0;

double Ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

sl::float4x4 Matrix(const float* m) {
    sl::float4x4 r;
    for (uint32_t i = 0; i < 4; ++i) {
        r[i] = sl::float4(m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
    }
    return r;
}

bool Inverted(const float* m, float* out) {
    double a[4][8]{};
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t pivot = c;
        for (uint32_t r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        }
        if (std::fabs(a[pivot][c]) < 1e-12) return false;
        for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double scale = 1.0 / a[c][c];
        for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
        for (uint32_t r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][4 + j]);
    }
    return true;
}

// How far the camera moved since the previous frame (1.59): the mean displacement (NDC) of four
// far points through clip_to_prev_clip. A turn is 0.01-0.3; a cut is above 1.
double CameraDelta(const Inputs* in) {
    if (in == nullptr || !in->camera_valid) return 0.0;
    const float* m = in->clip_to_prev_clip;
    double sum = 0;
    const double pts[4][2] = {{-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}};
    for (const auto& pt : pts) {
        const double v[4] = {pt[0], pt[1], in->depth_inverted ? 1e-4 : 0.9999, 1.0};
        double o[4]{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) o[i] += m[i * 4 + j] * v[j];
        if (std::fabs(o[3]) < 1e-9) return 1e9;
        const double dx = o[0] / o[3] - pt[0], dy = o[1] / o[3] - pt[1];
        sum += std::sqrt(dx * dx + dy * dy);
    }
    return sum / 4.0;
}

// DLSS-G's reset (1.59): the first frame after it had no inputs, and a real camera cut only (the
// upscaler's own reset is not forwarded unless BB_DLSSG_RESET=inputs).
bool Reset(const Inputs* in) {
    static const bool legacy = [] {
        const char* e = std::getenv("BB_DLSSG_RESET");
        return e != nullptr && std::string_view(e) == "inputs";
    }();
    static const double cut = [] {
        const char* e = std::getenv("BB_DLSSG_CUT_NDC");
        return e != nullptr && *e != '\0' ? std::atof(e) : 3.0;
    }();
    const bool first = !g_prev_inputs_ok;
    g_prev_inputs_ok = in != nullptr && in->camera_valid;
    if (in == nullptr) return true;
    if (legacy) return in->reset;
    static double avg = 0.0;
    const double delta = CameraDelta(in);
    const double d = std::min(delta, 99.0);
    const bool is_cut = delta > cut && d > 4.0 * std::max(avg, 0.05);
    avg = avg * 0.8 + d * 0.2;
    return first || is_cut;
}

// The constants of the super resolution path (bb_sr_dlss.cpp): our matrices are row major for
// column vectors, Streamline's row-vector, so each is given transposed.
sl::Constants MakeConstants(const Inputs* in) {
    sl::Constants c{};
    const float identity[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const auto transposed = [](const float* m) {
        float t[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) t[j * 4 + i] = m[i * 4 + j];
        return Matrix(t);
    };
    const bool camera = in != nullptr && in->camera_valid;
    float inverse[16]{}, prev_to_clip[16]{};
    c.cameraViewToClip = transposed(camera ? in->view_to_clip : identity);
    c.clipToCameraView = camera && Inverted(in->view_to_clip, inverse) ? transposed(inverse)
                                                                      : Matrix(identity);
    c.clipToLensClip = Matrix(identity);
    c.clipToPrevClip = transposed(camera ? in->clip_to_prev_clip : identity);
    c.prevClipToClip = camera && Inverted(in->clip_to_prev_clip, prev_to_clip)
                           ? transposed(prev_to_clip)
                           : Matrix(identity);
    const vk::Extent2D render = in != nullptr ? in->render : vk::Extent2D{1, 1};
    c.jitterOffset = in != nullptr ? sl::float2(in->jitter[0], in->jitter[1]) : sl::float2(0, 0);
    c.mvecScale = sl::float2(1.0f / static_cast<float>(std::max(render.width, 1u)),
                             1.0f / static_cast<float>(std::max(render.height, 1u)));
    c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    c.cameraPos = sl::float3(0, 0, 0);
    c.cameraUp = sl::float3(0, 1, 0);
    c.cameraRight = sl::float3(1, 0, 0);
    c.cameraFwd = sl::float3(0, 0, 1);
    c.cameraNear = in != nullptr ? in->z_near : 0.1f;
    c.cameraFar = in != nullptr ? in->z_far : 1000.0f;
    c.cameraFOV = in != nullptr ? in->fov_y : 1.0f;
    c.cameraAspectRatio = static_cast<float>(std::max(render.width, 1u)) /
                          static_cast<float>(std::max(render.height, 1u));
    c.motionVectorsInvalidValue = sl::INVALID_FLOAT;
    c.depthInverted = in == nullptr || in->depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    const bool reset = Reset(in);
    if (in != nullptr && reset) ++g_stats.resets;
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    return c;
}

sl::Resource ToResource(const Image& image) {
    sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
                   static_cast<VkImageView>(image.view),
                   static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    r.width = image.extent.width;
    r.height = image.extent.height;
    r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
    return r;
}

template <typename F>
bool Feature(sl::Feature feature, const char* name, F*& function) {
    function = reinterpret_cast<F*>(SR::Dlss::FeatureProc(static_cast<uint32_t>(feature), name));
    if (function == nullptr) {
        Printf("[dlssg] %s is unavailable: DLSS-G off", name);
        return false;
    }
    return true;
}

// DLSS-G off without keeping its resources, and its viewport freed (1.59 bb-dlssg-085): before a
// swapchain goes away and for the retry after a feature that was not created.
void Release() {
    if (g_api.dlssg_set_options == nullptr) return;
    sl::DLSSGOptions options{};
    options.mode = sl::DLSSGMode::eOff;
    options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
    options.colorWidth = options.mvecDepthWidth = std::max(g_options_backbuffer.width, 1u);
    options.colorHeight = options.mvecDepthHeight = std::max(g_options_backbuffer.height, 1u);
    if (g_options_format != vk::Format::eUndefined) {
        options.colorBufferFormat = options.hudLessBufferFormat =
            static_cast<uint32_t>(static_cast<VkFormat>(g_options_format));
    }
    options.mvecBufferFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
    options.depthBufferFormat = static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT);
    if (const auto r = g_api.dlssg_set_options(sl::ViewportHandle{0}, options); r != sl::Result::eOk) {
        Printf("[dlssg] slDLSSGSetOptions (off) failed (%d)", static_cast<int>(r));
    }
    if (g_api.free_resources != nullptr) {
        if (const auto r = g_api.free_resources(sl::kFeatureDLSS_G, sl::ViewportHandle{0});
            r != sl::Result::eOk) {
            Printf("[dlssg] slFreeResources failed (%d)", static_cast<int>(r));
        }
    }
    g_mode_set = 0;
    g_options_set = false;
}

// bb-dyn-096 (1.59), once a second: the largest multiplier N (2 .. max) with real x N <= the
// target (2% margin); kept while real x current stays within 95-102% of it; a change needs two
// seconds in a row.
void DynamicSecond(double real_fps) {
    if (!g_dynamic || g_mode_set != 1 || real_fps <= 1.0 || g_target_fps <= 1.0) return;
    const double T = g_target_fps;
    const int hi = std::max(g_dyn_max, 1);
    const int cur_m = g_dyn_gen + 1;
    int want = cur_m;
    if (real_fps * cur_m < 0.95 * T || real_fps * cur_m > 1.02 * T) {
        want = 2;
        for (int m = hi + 1; m >= 2; --m) {
            if (real_fps * m <= 1.02 * T) {
                want = m;
                break;
            }
        }
    }
    want = std::clamp(want - 1, 1, hi);
    if (want == g_dyn_gen) {
        g_dyn_pend = -1;
        g_dyn_pend_n = 0;
        return;
    }
    g_dyn_pend_n = want == g_dyn_pend ? g_dyn_pend_n + 1 : 1;
    g_dyn_pend = want;
    if (g_dyn_pend_n < 2) return;
    Printf("[dyn] DLSS-G x%d (from x%d): real %.1f fps, target %.0f", want + 1, cur_m, real_fps, T);
    g_dyn_gen = want;
    g_dyn_pend = -1;
    g_dyn_pend_n = 0;
    g_options_set = false; // numFramesToGenerate again
}

void Second(Clock::time_point now) {
    auto& s = g_stats;
    if (s.start == Clock::time_point{}) {
        s.start = now;
        return;
    }
    const double window = std::chrono::duration<double>(now - s.start).count();
    if (window < 1.0) return;
    const double n = std::max<double>(s.presents, 1.0);
    const double real = s.presents / window;
    const uint32_t shown_per = g_mode_set == 1 ? std::max(g_last_presented, 1u) : 1u;
    Printf("[dlssg] %s x%d (asked x%d); real %.1f fps, shown %.1f fps; on %u of %u presents (%u "
           "with inputs, camera invalid %u, resets %u); Streamline CPU %.3f ms/present (before "
           "%.3f, after %.3f), present call %.2f ms (max %.1f); status 0x%x, generate max %u",
           g_dynamic ? "dynamic" : "fixed", CurrentGenerate() + 1,
           (g_dynamic ? g_dyn_max : g_generate) + 1, real, real * shown_per, s.on, s.presents,
           s.inputs, s.nocamera, s.resets, (s.before_ms + s.after_ms) / n, s.before_ms / n,
           s.after_ms / n, s.present_ms / n, s.present_max, g_last_status, g_last_max);
    // Present call breakdown, once a second: Streamline calls before vkQueuePresentKHR (token,
    // constants, tags, options, PCL markers), inside it (sl.interposer: DLSS-G's generated
    // frame(s) and pacing), after it (PCL, slDLSSGGetState), and the swapchain acquire.
    Printf("[dlssg] present breakdown (ms avg/max): before Streamline %.3f/%.2f, vkQueuePresent "
           "%.2f/%.2f, after %.3f/%.2f, acquire %.2f/%.2f; %u presents, swapchain %u images, %s",
           s.before_ms / n, s.before_max, s.present_ms / n, s.present_max, s.after_ms / n,
           s.after_max, s.acquire_ms / std::max<double>(s.acquires, 1.0), s.acquire_max,
           s.presents, g_sc_images, g_sc_mode);
    // Streamline's own numbers (measured, not real x multiplier): slDLSSGGetState.
    Printf("[dlssg] slDLSSGGetState: status 0x%x%s, numFramesActuallyPresented %u (avg %.2f, >1 on %u "
           "of %u presents), minWidthOrHeight %u, VSync support %s, generate max %u; HAGS %s; "
           "measured shown %.1f fps",
           g_last_status, g_last_status == 0 ? " (eOk)" : " (NOT OK: see sl::DLSSGStatus bits)",
           g_last_actual, static_cast<double>(s.actual_sum) / n, s.generated, s.presents,
           g_last_minwh, g_last_vsync ? "yes" : "no", g_last_max, HagsState(),
           static_cast<double>(s.actual_sum) / window);
    // Real frames: the presents with inputs (game frames) in the window.
    DynamicSecond(s.inputs / window);
    s = {};
    s.start = now;
}
#endif
} // namespace

bool Ready() {
    return g_ready;
}

int Generated() {
    return g_ready && g_mode_set == 1 ? CurrentGenerate() : 0;
}

void OnDevice(vk::PhysicalDevice physical) {
    if (Cfg().kind != Kind::Dlss) return;
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!SR::Dlss::Loaded()) {
        Printf("[dlssg] Streamline was not loaded (sl.interposer.dll not found or not signed by "
               "NVIDIA): DLSS-G off");
        return;
    }
    if (SR::Dlss::LoaderPath() == nullptr) {
        Printf("[dlssg] Streamline is not the Vulkan loader (BB_SL_NO_INTERPOSER_LOADER=1): "
               "DLSS-G needs its swapchain hooks: off");
        return;
    }
    if (!SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeatureDLSS_G), physical)) {
        Printf("[dlssg] DLSS-G is not supported on this device / driver (sl.dlss_g): off");
        return;
    }
    g_api.get_new_frame_token =
        reinterpret_cast<PFun_slGetNewFrameToken*>(SR::Dlss::Proc("slGetNewFrameToken"));
    g_api.set_constants = reinterpret_cast<PFun_slSetConstants*>(SR::Dlss::Proc("slSetConstants"));
    g_api.set_tag_for_frame =
        reinterpret_cast<PFun_slSetTagForFrame*>(SR::Dlss::Proc("slSetTagForFrame"));
    g_api.free_resources = reinterpret_cast<PFun_slFreeResources*>(SR::Dlss::Proc("slFreeResources"));
    if (g_api.get_new_frame_token == nullptr || g_api.set_constants == nullptr ||
        g_api.set_tag_for_frame == nullptr ||
        !Feature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
        !Feature(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
        !Feature(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
        !Feature(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
        return;
    }
    if (EnvIs("BB_DLSSG_SLEEP", '1')) {
        Feature(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
    }
    g_generate = std::clamp(Cfg().multiplier - 1, 1, 5);
    g_dynamic = Cfg().dynamic;
    if (g_dynamic) {
        g_generate = 5; // up to x6; trimmed to numFramesToGenerateMax
        if (const char* t = Env("KYTY_DLSSG_TARGET_FPS"); t != nullptr && std::atof(t) > 0) {
            g_target_fps = std::atof(t);
            Printf("[dlssg] dynamic target %.0f fps (BB_DLSSG_TARGET_FPS)", g_target_fps);
        } else if (const double hz = RefreshHz(); hz > 30.0) {
            g_target_fps = std::floor(hz * 14.0 / 15.0 + 0.01);
            Printf("[dlssg] display %.0f Hz: dynamic target %.0f fps", hz, g_target_fps);
        } else {
            g_target_fps = 224.0;
            Printf("[dlssg] display refresh unknown: dynamic target 224 fps");
        }
        g_dyn_max = std::min(g_generate, 3);
        g_dyn_gen = std::clamp(static_cast<int>(std::floor(g_target_fps / 30.0)) - 1, 1, g_dyn_max);
    }
    // DLSS-G needs Reflex on; its own frame limiter is not used (frameLimitUs 0).
    sl::ReflexOptions reflex{};
    reflex.mode = sl::ReflexMode::eLowLatency;
    if (const char* mode = Env("KYTY_DLSSG_REFLEX"); mode != nullptr) {
        if (std::string_view(mode) == "boost") reflex.mode = sl::ReflexMode::eLowLatencyWithBoost;
        if (std::string_view(mode) == "off") reflex.mode = sl::ReflexMode::eOff;
    }
    reflex.frameLimitUs = 0;
    if (const auto r = g_api.reflex_set_options(reflex); r != sl::Result::eOk) {
        Printf("[dlssg] slReflexSetOptions failed (%d)", static_cast<int>(r));
    }
    g_ready = true;
    Printf("[dlssg] DLSS-G ready: %s x%d, Reflex mode %d, slReflexSleep %s; markers and Streamline "
           "calls on the present thread only (frame boundary)",
           g_dynamic ? "dynamic up to" : "fixed", g_generate + 1, static_cast<int>(reflex.mode),
           g_api.reflex_sleep != nullptr ? "on (BB_DLSSG_SLEEP=1)" : "off");
#else
    (void)physical;
    Printf("[dlssg] this build has no Streamline SDK: DLSS-G off");
#endif
}

void BeforePresent(const Inputs* inputs, vk::Extent2D backbuffer, vk::Format format,
                   vk::Rect2D region, uint32_t image_count) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_ready) return;
    const auto t0 = Clock::now();
    Second(t0);
    const sl::ViewportHandle viewport{0};
    const bool blocked = backbuffer.width == 0 || backbuffer.height == 0 ||
                         region.extent.width == 0 || region.extent.height == 0 ||
                         t0 < g_backoff_until;
    const bool usable = inputs != nullptr && inputs->valid && !blocked;
    ++g_stats.presents;
    if (usable) {
        ++g_stats.inputs;
        if (!inputs->camera_valid) ++g_stats.nocamera;
    }
    // On after 10 presents in a row with inputs, off after 30 without (menus, movies, loading),
    // its resources kept (eRetainResourcesWhenOff), so a fade or a cut does not flap it.
    static uint32_t run_with = 0, run_without = 0;
    run_with = usable ? run_with + 1 : 0;
    run_without = usable ? 0 : run_without + 1;
    const bool mode_on = !blocked && (g_mode_set == 1 ? run_without < 30 : run_with >= 10);
    if (mode_on) ++g_stats.on;
    if (!g_options_set || g_mode_set != (mode_on ? 1 : 0) || g_options_format != format ||
        g_options_images != image_count ||
        (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render))) {
        sl::DLSSGOptions options{};
        options.mode = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (g_mode_set != (mode_on ? 1 : 0)) {
            Printf("[dlssg] %s", mode_on ? "on (3D frame: depth, motion vectors, camera)"
                                         : "off (no 3D frame: menu, loading or movie)");
            if (mode_on) {
                g_mode_on_at = g_presents;
                g_nogen_run = 0;
            }
        }
        g_mode_set = mode_on ? 1 : 0;
        options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
        options.numBackBuffers = image_count;
        options.colorWidth = backbuffer.width;
        options.colorHeight = backbuffer.height;
        options.colorBufferFormat = static_cast<uint32_t>(static_cast<VkFormat>(format));
        // Every format defined, inputs or not (Streamline: "Cannot have undefined format").
        options.mvecDepthWidth = g_options_render.width != 0 ? g_options_render.width : backbuffer.width;
        options.mvecDepthHeight =
            g_options_render.height != 0 ? g_options_render.height : backbuffer.height;
        options.mvecBufferFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
        options.depthBufferFormat = static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT);
        options.hudLessBufferFormat = static_cast<uint32_t>(static_cast<VkFormat>(format));
        if (usable) {
            options.mvecDepthWidth = inputs->render.width;
            options.mvecDepthHeight = inputs->render.height;
            if (inputs->motion.format != vk::Format::eUndefined) {
                options.mvecBufferFormat =
                    static_cast<uint32_t>(static_cast<VkFormat>(inputs->motion.format));
            }
            if (inputs->depth.format != vk::Format::eUndefined) {
                options.depthBufferFormat =
                    static_cast<uint32_t>(static_cast<VkFormat>(inputs->depth.format));
            }
            g_options_render = inputs->render;
        }
        if (const auto r = g_api.dlssg_set_options(viewport, options); r != sl::Result::eOk) {
            Printf("[dlssg] slDLSSGSetOptions failed (%d)", static_cast<int>(r));
        }
        static bool once = false;
        if (!once) {
            once = true;
            Printf("[dlssg] options: %s x%u, backbuffer %ux%u fmt %u, %u images, render %ux%u",
                   g_dynamic ? "dynamic" : "fixed", options.numFramesToGenerate + 1,
                   backbuffer.width, backbuffer.height, options.colorBufferFormat, image_count,
                   options.mvecDepthWidth, options.mvecDepthHeight);
        }
        if (g_resume_pending && g_mode_set == 1) {
            g_resume_pending = false;
            Printf("[dlssg] resumed after swapchain recreate %ux%u", backbuffer.width,
                   backbuffer.height);
        }
        g_options_set = true;
        g_options_backbuffer = backbuffer;
        g_options_format = format;
        g_options_images = image_count;
    }
    // One Streamline frame per present: the DLSS evaluate's index when it ran for this frame
    // (same token for its constants), else a new one.
    {
        const uint32_t sr = SR::Dlss::LastFrameIndex();
        const uint32_t own = usable && sr > g_frame_index ? sr : SR::Dlss::NextFrameIndex();
        g_frame_index = std::max(own, g_frame_index + 1);
    }
    if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
        g_token = nullptr;
        return;
    }
    auto& token = *g_token;
    if (g_api.reflex_sleep != nullptr) g_api.reflex_sleep(token);
    g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
    g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
    g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);
    const sl::Constants constants = MakeConstants(usable ? inputs : nullptr);
    if (const auto r = g_api.set_constants(constants, token, viewport); r != sl::Result::eOk) {
        static uint32_t failures = 0;
        if (failures++ < 8 || failures % 600 == 0) {
            Printf("[dlssg] slSetConstants failed (%d), %u time(s)", static_cast<int>(r), failures);
        }
    }
    if (usable) {
        sl::Resource depth = ToResource(inputs->depth);
        sl::Resource motion = ToResource(inputs->motion);
        const sl::Extent full{0, 0, inputs->render.width, inputs->render.height};
        // sl::Extent is {top, left, width, height}.
        const sl::Extent game{static_cast<uint32_t>(region.offset.y),
                              static_cast<uint32_t>(region.offset.x), region.extent.width,
                              region.extent.height};
        const bool partial = region.offset.x != 0 || region.offset.y != 0 ||
                             region.extent.width != backbuffer.width ||
                             region.extent.height != backbuffer.height;
        sl::ResourceTag tags[3]{
            sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent,
                            &full),
            sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eValidUntilPresent, &full),
            sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer,
                            sl::ResourceLifecycle::eValidUntilPresent, &game),
        };
        const uint32_t count = partial ? 3u : 2u;
        const auto r = g_api.set_tag_for_frame(token, viewport, tags, count, nullptr);
        static uint32_t failures = 0;
        if (r != sl::Result::eOk && (failures++ < 8 || failures % 600 == 0)) {
            Printf("[dlssg] slSetTagForFrame failed (%d), %u time(s)", static_cast<int>(r), failures);
        }
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            Printf("[dlssg] first tags: frame %u, depth %ux%u fmt %u, motion %ux%u fmt %u, "
                   "backbuffer %ux%u fmt %u, game region %d,%d %ux%u%s, camera %s, near %.4g far "
                   "%.4g fov %.4g, depth inverted %d, setTag %d",
                   g_frame_index, depth.width, depth.height, depth.nativeFormat, motion.width,
                   motion.height, motion.nativeFormat, backbuffer.width, backbuffer.height,
                   static_cast<uint32_t>(static_cast<VkFormat>(format)), region.offset.x,
                   region.offset.y, region.extent.width, region.extent.height,
                   partial ? " (tagged as a subrect)" : "", inputs->camera_valid ? "valid" : "INVALID",
                   constants.cameraNear, constants.cameraFar, constants.cameraFOV,
                   static_cast<int>(constants.depthInverted), static_cast<int>(r));
        }
    }
    g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
    g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
    g_present_begin = Clock::now();
    g_stats.before_ms += Ms(t0, g_present_begin);
    g_stats.before_max = std::max(g_stats.before_max, Ms(t0, g_present_begin));
#else
    (void)inputs;
    (void)backbuffer;
    (void)format;
    (void)region;
    (void)image_count;
#endif
}

void AfterPresent() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_ready || g_token == nullptr) return;
    const auto t0 = Clock::now();
    if (g_present_begin != Clock::time_point{}) {
        const double ms = Ms(g_present_begin, t0);
        g_stats.present_ms += ms;
        g_stats.present_max = std::max(g_stats.present_max, ms);
        g_present_begin = {};
    }
    g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
    g_token = nullptr;
    const auto presents = ++g_presents;
    sl::DLSSGState state{};
    if (g_api.dlssg_get_state(sl::ViewportHandle{0}, state, nullptr) == sl::Result::eOk) {
        g_last_status = static_cast<uint32_t>(state.status);
        g_last_max = state.numFramesToGenerateMax;
        g_last_presented = std::max(state.numFramesActuallyPresented, 1u);
        g_last_actual = state.numFramesActuallyPresented;
        g_last_minwh = state.minWidthOrHeight;
        g_last_vsync = state.bIsVsyncSupportAvailable == sl::Boolean::eTrue;
        g_stats.actual_sum += state.numFramesActuallyPresented;
        if (state.numFramesActuallyPresented > 1) ++g_stats.generated;
        // More generated frames asked than the GPU / nvngx_dlssg.dll makes (x5 / x6): as many as
        // it makes.
        if (state.numFramesToGenerateMax != 0) {
            const int max = static_cast<int>(state.numFramesToGenerateMax);
            if (!g_dynamic && g_generate > max) {
                Printf("[dlssg] x%d asked, this GPU / nvngx_dlssg.dll makes up to x%d: x%d",
                       g_generate + 1, max + 1, max + 1);
                g_generate = max;
                g_options_set = false;
            }
            if (g_dynamic && std::min(g_generate, max) != g_dyn_max) {
                g_dyn_max = std::min(g_generate, max);
                Printf("[dyn] DLSS-G up to x%d (numFramesToGenerateMax %d)", g_dyn_max + 1, max);
                if (g_dyn_gen > g_dyn_max) {
                    g_dyn_gen = g_dyn_max;
                    g_options_set = false;
                }
            }
        }
        // On and asked to generate, but nothing generated for 90 presents after 60 settling ones
        // (its NGX feature was not created, e.g. after a lost swapchain): off, freed, on again in
        // a second (1.59 bb-dlssg-085).
        if (g_mode_set == 1 && presents > g_mode_on_at + 60) {
            g_nogen_run = state.numFramesActuallyPresented <= 1 ? g_nogen_run + 1 : 0;
            if (g_nogen_run >= 90) {
                g_nogen_run = 0;
                ++g_retries;
                Printf("[dlssg] not generating (status 0x%x): resources freed, retried in 1 s (%u "
                       "time(s))",
                       g_last_status, g_retries);
                Release();
                g_backoff_until = Clock::now() + std::chrono::seconds(1);
            }
        } else if (g_mode_set != 1) {
            g_nogen_run = 0;
        }
        static uint32_t said_status = ~0u;
        if (g_last_status != said_status) {
            said_status = g_last_status;
            Printf("[dlssg] state: status 0x%x, presented %u, generate max %u, VRAM %llu MiB",
                   g_last_status, state.numFramesActuallyPresented, state.numFramesToGenerateMax,
                   static_cast<unsigned long long>(state.estimatedVRAMUsageInBytes >> 20u));
        }
    }
    g_stats.after_ms += Ms(t0, Clock::now());
    g_stats.after_max = std::max(g_stats.after_max, Ms(t0, Clock::now()));
#endif
}

void SwapchainChanging() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_ready || (g_mode_set != 1 && !g_options_set)) return;
    // The device is idle (Swapchain::Destroy waited): the old swapchain's NGX feature must not
    // outlive it; it is made again by the first options set on the new one.
    Release();
    g_resume_pending = true;
#endif
}

vk::PresentModeKHR PreferredPresentMode(vk::PresentModeKHR current,
                                        const std::vector<vk::PresentModeKHR>& available) {
    std::string mode = "auto";
    if (const char* v = std::getenv("BB_FG_PRESENT"); v != nullptr && *v != '\0') {
        mode = v;
        for (auto& ch : mode) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const auto supported = [&](vk::PresentModeKHR m) {
        return std::find(available.begin(), available.end(), m) != available.end();
    };
    vk::PresentModeKHR wanted = current;
    if (mode == "fifo") {
        wanted = vk::PresentModeKHR::eFifo;
    } else if (mode == "mailbox") {
        wanted = vk::PresentModeKHR::eMailbox;
    } else if (mode == "immediate") {
        wanted = vk::PresentModeKHR::eImmediate;
    } else if (mode == "auto" && (current == vk::PresentModeKHR::eFifo ||
                                  current == vk::PresentModeKHR::eFifoRelaxed)) {
        // DLSS-G on Vulkan has no vsync support (Streamline's guide 22.5). 1.59's rule: immediate
        // on >= 120 Hz with the dynamic multiplier (its shown rate stays under the target), else
        // mailbox (never tears), immediate where mailbox is not offered.
        const bool high = RefreshHz() >= 120.0 && Cfg().dynamic;
        wanted = high && supported(vk::PresentModeKHR::eImmediate) ? vk::PresentModeKHR::eImmediate
                 : supported(vk::PresentModeKHR::eMailbox)        ? vk::PresentModeKHR::eMailbox
                 : supported(vk::PresentModeKHR::eImmediate)      ? vk::PresentModeKHR::eImmediate
                                                                  : current;
    }
    if (!supported(wanted)) wanted = current;
    {
        std::string list;
        for (const auto m : available) list += (list.empty() ? "" : ",") + vk::to_string(m);
        Printf("[dlssg] surface present modes: %s; using %s", list.c_str(),
               vk::to_string(wanted).c_str());
    }
    g_sc_mode = wanted == vk::PresentModeKHR::eMailbox     ? "mailbox"
                : wanted == vk::PresentModeKHR::eImmediate ? "immediate"
                : wanted == vk::PresentModeKHR::eFifo      ? "fifo"
                                                           : "other";
    if (wanted != current) {
        Printf("[dlssg] present mode %s -> %s (DLSS-G on Vulkan has no vsync; BB_FG_PRESENT=%s)",
               vk::to_string(current).c_str(), vk::to_string(wanted).c_str(), mode.c_str());
    }
    return wanted;
}

uint32_t PreferredImageCount(uint32_t current, uint32_t min, uint32_t max) {
    // Each real present becomes 1 + generated presents on the game's swapchain: one image on
    // screen, one queued (mailbox) and the rest for the frames in flight, so acquiring the
    // generated frame's image never waits for a vblank. BB_DLSSG_IMAGES=n: exactly n.
    uint32_t want = std::max(current, min + 2u + static_cast<uint32_t>(std::max(g_generate, 1)));
    if (const char* v = std::getenv("BB_DLSSG_IMAGES"); v != nullptr && std::atoi(v) > 0) {
        want = std::max(static_cast<uint32_t>(std::atoi(v)), min);
    }
    if (max > 0) want = std::min(want, max);
    if (want != current) {
        Printf("[dlssg] swapchain images %u -> %u (min %u, max %u; BB_DLSSG_IMAGES)", current, want,
               min, max);
    }
    g_sc_images = want;
    return want;
}

void AcquireTime(double ms) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    ++g_stats.acquires;
    g_stats.acquire_ms += ms;
    g_stats.acquire_max = std::max(g_stats.acquire_max, ms);
#else
    (void)ms;
#endif
}

void OnDispatch(vk::Device device) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (Cfg().kind != Kind::Dlss || SR::Dlss::LoaderPath() == nullptr) return;
    if (EnvIs("BB_SL_PASSTHROUGH", '0')) {
        Printf("[dlssg] BB_SL_PASSTHROUGH=0: per-command Vulkan calls stay on sl.interposer.dll");
        return;
    }
    HMODULE system = GetModuleHandleW(L"vulkan-1.dll");
    const auto gdpa = system != nullptr ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(reinterpret_cast<void*>(
                                              GetProcAddress(system, "vkGetDeviceProcAddr")))
                                        : nullptr;
    if (gdpa == nullptr) {
        Printf("[dlssg] the system vulkan-1.dll is not loaded: per-command calls stay on the "
               "interposer");
        return;
    }
    auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    const VkDevice dev = static_cast<VkDevice>(device);
    int n = 0;
    const auto set = [&](auto& function, const char* name) {
        if (const PFN_vkVoidFunction p = gdpa(dev, name); p != nullptr) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(p);
            ++n;
        }
    };
    // Streamline 2.14's interposer forwards these unchanged (no plugin hooks them with command
    // list state tracking off): straight to the driver.
    set(d.vkCmdBindPipeline, "vkCmdBindPipeline");
    set(d.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    set(d.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
    set(d.vkBeginCommandBuffer, "vkBeginCommandBuffer");
    set(d.vkCreateImage, "vkCreateImage");
    Printf("[dlssg] %d per-command Vulkan entry points taken from the system loader (the "
           "interposer keeps device creation, swapchain and present only)",
           n);
#else
    (void)device;
#endif
}

} // namespace BB::FG::Dlssg
