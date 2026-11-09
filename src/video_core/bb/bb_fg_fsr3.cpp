// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// AMD FSR 3 frame generation (BB_FG=amd-*), ported from the Demon's Souls fork's
// src/local/fsr3-fg.cpp: the FidelityFX frame interpolation swapchain in place of the window's
// (its own present queue in the graphics family, async compute and image acquire queues), the
// frame generation context configured each present with the HUD-less image, and its prepare pass
// (depth, motion vectors, camera) submitted before the present. The multi frame generation (x3 / x4)
// and the dynamic multiplier need the patched amd_fidelityfx_vk.dll
// (tools/bb/fsr3-mfg/ffx-sdk-v1.1.4-mfg.patch, built by the bb-windows workflow with its command
// list pool fix); AMD's signed DLL stays at x2. Same pacing: KYTY_FSR3_PACING_FLOOR (the DLL's
// present floor), KYTY_VRR_CAP_FPS / the dynamic target (our real-frame limiter), all read as
// BB_* first.
#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "video_core/bb/bb_display.h"
#include "video_core/bb/bb_dyn_multiplier.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_objmv.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/vk/ffx_api_vk.h>
#endif

namespace ImGui::Core {
bool MustKeepDrawing(); // imgui/renderer/imgui_core.h (1.51: overlay up)
}

namespace BB::FG::Amd {

#if defined(_WIN32)
namespace {

bool g_requested = false;
HMODULE g_dll = nullptr;
ffxFunctions g_ffx{};

VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
VkQueue g_game_queue = VK_NULL_HANDLE;
uint32_t g_game_family = 0;
std::mutex* g_submit_mutex = nullptr;
VkQueue g_present_queue = VK_NULL_HANDLE; // FI present: graphics family
VkQueue g_async_queue = VK_NULL_HANDLE;   // compute family, index 0
VkQueue g_acquire_queue = VK_NULL_HANDLE; // compute family, index 1
uint32_t g_compute_family = UINT32_MAX;

PFN_vkGetDeviceProcAddr g_get_device_proc = nullptr;
PFN_vkQueueSubmit g_queue_submit = nullptr;

ffxContext g_swapchain_context = nullptr;
ffxQueryDescSwapchainReplacementFunctionsVK g_functions{};
// Patched DLL: the grid slot of the next present; the presents' intervals (read and cleared).
void (*g_set_pace_slot)(int64_t, int64_t) = nullptr;
void (*g_take_pace_stats)(int64_t*) = nullptr;
// bb-fg-074 (the display patch): the ids of the presents the DLL made, so each can be waited for.
uint32_t (*g_take_display_ids)(uint64_t*, uint32_t) = nullptr;
bool g_display_ids = false; // the game's presents carry ids and the DLL numbers every present

ffxContext g_fg_context = nullptr;
vk::Extent2D g_fg_display{};
vk::Extent2D g_fg_max_render{};
vk::Format g_fg_format = vk::Format::eUndefined;
bool g_fg_failed = false;

struct Slot {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false;
};
VkCommandPool g_pool = VK_NULL_HANDLE;
std::array<Slot, 4> g_slots{};
uint32_t g_slot = 0;

uint64_t g_frame_id = 0;
bool g_generating = false;
int g_state_logged = -1;
std::chrono::steady_clock::time_point g_last_real{};

struct Stats {
    std::chrono::steady_clock::time_point start{};
    uint32_t real = 0, generated_on = 0, no_inputs = 0, hudless = 0, slotted = 0;
    uint32_t hudless_mismatch = 0, hudless_none = 0; // 1.03
    uint64_t presents_at_start = 0;
    double pace_n = 0, pace_sum = 0, pace_sq = 0, pace_max = 0; // real present intervals (s)
    double prepare_ms = 0, present_ms = 0, max_present_ms = 0;
} g_stats;

// 1.40 (bb-fgamd-140): the per-second [fg-amd] line: what FSR 3 frame interpolation got and did.
struct AmdSecond {
    uint32_t prepared = 0;         // prepare dispatches (real frames given to FG)
    uint32_t input_resets = 0;     // inputs with our (upscaler history) reset flag
    uint32_t restarts = 0;         // generation turned on again after a present without it
    uint32_t gaps = 0;             // presents without inputs while generating (repeat / no 3D)
    float mv_scale[2]{};
    float near_v = 0, far_v = 0, fov = 0, jitter[2]{};
    bool depth_inverted = false;
    // 1.41 (bb-fg30-141): frame pairing of each prepared frame (guest frame numbers stamped by
    // the rasterizer: colour, depth copy, camera motion pass, object-vector compose, HUD-less).
    uint32_t paired = 0, off_depth = 0, off_mv = 0, off_obj = 0, off_hudless = 0;
    uint64_t last_ids[5]{~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    uint64_t last_ffx_frame = 0;
    double dt_sum = 0, dt_max = 0;
    uint32_t hudless_given = 0;
} g_amd;
// The frame-generation dispatches the swapchain made (frames interpolated) and how many of them
// came with FSR's own history reset (its frameID chain broken, or the first after a gap).
std::atomic<uint32_t> g_amd_dispatches{0}, g_amd_ffx_resets{0};

std::mutex g_mutex; // the contexts are not thread safe (presents, prepare, destruction)

// 0.92 (bb-amdfg-092): held-step present slots for the patched DLL. With no real fps cap
// (BB_FPS_CAP=0, the default) the guest pacer gave the DLL no slot, and the DLL placed each
// generated frame at its own estimate (moving average x 1/(N+1) minus the variance and a safety
// margin), so generated and real frames alternated short / long: 0.85 AMD x2 run, shown 224 fps
// but interval sd 0.75-1.37 ms, p99 7-9 ms, max 10-14 ms against a 4.46 ms mean. Now each real
// present gets a slot on a held grid (the guest pacer's step, or a held mean of the real present
// intervals when it has none), corrected by 1/8 of the phase error a frame and re-phased only when
// a frame is off by more than 40% of the step; the DLL then puts the N generated frames at
// slot + i * step / (N + 1) and the real one at slot + step. BB_FSR3_PACE_HOLD=0: the old way.
bool HoldPace() {
    static const bool on = [] {
        const char* v = std::getenv("BB_FSR3_PACE_HOLD");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return on;
}

struct Hold {
    double step = 0.0;       // s, the held step (0: none yet)
    double last_slot = 0.0;  // s (steady clock), the previous real frame's slot
    double last_arrive = 0.0;
    double ring[32]{};
    int count = 0, next = 0;
    double planned_last = 0.0; // s, the last planned present (the real frame at slot + step)
} g_hold;

// Per-second samples for [fsr3-ft] (under g_mutex).
struct FrameTimes {
    std::vector<double> real;  // ms between real presents reaching the frame generation swapchain
    std::vector<double> shown; // ms between the presents planned on the slots (all frames)
    uint32_t rephase = 0, own_slots = 0;
} g_ft;

double SteadyNow() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void FtLine(const char* what, std::vector<double>& v, std::string& out) {
    char buf[160];
    if (v.size() < 2) {
        std::snprintf(buf, sizeof(buf), "%s n/a", what);
    } else {
        double sum = 0, sq = 0, mx = 0;
        for (double x : v) {
            sum += x;
            sq += x * x;
            mx = std::max(mx, x);
        }
        const double n = static_cast<double>(v.size());
        const double mean = sum / n;
        std::sort(v.begin(), v.end());
        const size_t i99 = std::min(v.size() - 1, static_cast<size_t>(std::ceil(0.99 * n)) - 1);
        std::snprintf(buf, sizeof(buf), "%s mean %.2f sd %.2f p99 %.2f max %.2f ms (%zu)", what,
                      mean, std::sqrt(std::max(sq / n - mean * mean, 0.0)), v[i99], mx, v.size());
    }
    out += buf;
    v.clear();
}

bool EnvOn(const char* name, bool fallback) {
    const char* value = Env(name);
    if (value == nullptr || *value == '\0') return fallback;
    return *value != '0';
}
float EnvFloat(const char* name, float fallback) {
    const char* value = Env(name);
    return value != nullptr && *value != '\0' ? static_cast<float>(std::atof(value)) : fallback;
}

// AMD-side multi frame generation (the patched DLL): KYTY_FSR3_MFG=N (2..4), KYTY_FSR3_DYNAMIC=1
// (the multiplier picked once a second, bb_dyn_multiplier.h; real frames limited to target / N).
// The DLL reads KYTY_FSR3_FG_FRAMES when it makes the swapchain and KYTY_FSR3_FG_ACTIVE on every
// present.
struct Mfg {
    bool on = false, dynamic = false;
    int max = 2;
    int current = 2;
    double target = 0.0;
    double slept = 0.0;
    double smooth = 0.0;
    Kyty::Local::DynMultiplier dm;
} g_mfg;

double RefreshTarget() {
    DEVMODEA mode{};
    mode.dmSize = sizeof(mode);
    if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 &&
        mode.dmDisplayFrequency > 30) {
        const double hz = mode.dmDisplayFrequency;
        return hz - hz * hz / 3600.0;
    }
    return 0.0;
}

void MfgSetActive(int multiplier) {
    char text[8];
    std::snprintf(text, sizeof(text), "%d", multiplier - 1);
    SetEnvironmentVariableA("KYTY_FSR3_FG_ACTIVE", text);
}

void MfgInit() {
    static bool done = false;
    if (done) return;
    done = true;
    const char* n = Env("KYTY_FSR3_MFG");
    const char* dyn = Env("KYTY_FSR3_DYNAMIC");
    g_mfg.dynamic = dyn != nullptr && *dyn == '1';
    if ((n == nullptr || *n == '\0') && !g_mfg.dynamic) return;
    g_mfg.on = true;
    g_mfg.max = std::clamp(n != nullptr && *n != '\0' ? std::atoi(n) : 4, 2, 4);
    g_mfg.current = g_mfg.max;
    char frames[8];
    std::snprintf(frames, sizeof(frames), "%d", g_mfg.max - 1);
    _putenv_s("KYTY_FSR3_FG_FRAMES", frames);
    SetEnvironmentVariableA("KYTY_FSR3_FG_FRAMES", frames);
    if (g_mfg.dynamic) {
        const float target = EnvFloat("KYTY_FSR3_TARGET_FPS", 0.0f);
        g_mfg.target = target > 0.0f ? target : RefreshTarget();
        if (g_mfg.target <= 0.0) g_mfg.target = 60.0 * g_mfg.max;
        g_mfg.current = 2;
    }
    MfgSetActive(g_mfg.current);
    Printf("[fsr3-mfg] %s, up to x%d (patched FidelityFX DLL only; AMD's signed DLL stays x2)",
           g_mfg.dynamic ? "dynamic multiplier" : "fixed multiplier", g_mfg.max);
    if (g_mfg.dynamic) Printf("[fsr3-mfg] target %.0f fps shown", g_mfg.target);
}

void MfgUpdate(uint32_t frames, double window, double shown) {
    if (!g_mfg.on) return;
    const double real = frames / window;
    const double busy = std::max(window - g_mfg.slept, 0.05);
    const double free = frames / busy;
    g_mfg.slept = 0.0;
    if (g_mfg.dynamic && frames > 0) {
        const double T = g_mfg.target;
        g_mfg.smooth = g_mfg.smooth <= 0.0 ? free : 0.5 * g_mfg.smooth + 0.5 * free;
        static int cap_state = 0;
        const int maxm = std::min(g_mfg.max, std::max(DynBaseCap(g_mfg.smooth, cap_state), 2));
        const int next =
            std::clamp(g_mfg.dm.Window(g_mfg.current, real, T, maxm, false, free), 2, std::max(maxm, 2));
        if (next != g_mfg.current) {
            const int from = g_mfg.current;
            g_mfg.current = next;
            MfgSetActive(next);
            Printf("[dyn] x%d (from x%d): real %.1f fps, shown %.1f (target %.0f), free %.1f; pacer "
                   "interval %.0f us for x%d",
                   next, from, real, std::min(real * from, T), T, g_mfg.dm.last_free, 1e6 * next / T,
                   next);
        }
    }
    Printf("[fsr3-mfg] x%d%s; real %.1f fps (free %.1f), shown %.1f fps", g_mfg.current,
           g_mfg.dynamic ? " dynamic" : "", real, free, shown);
    if (g_mfg.dynamic) {
        Printf("[fsr3-mfg] target %.0f, real limit %.1f", g_mfg.target, g_mfg.target / g_mfg.current);
    }
}

void Message(uint32_t type, const wchar_t* message) {
    static std::atomic<uint32_t> count{0};
    if (count.fetch_add(1, std::memory_order_relaxed) > 200) return;
    char text[1024]{};
    std::wcstombs(text, message, sizeof(text) - 1);
    Printf("FSR3 FG: FidelityFX message (%u): %s", type, text);
}

// The emulator's queue submissions are serialized by the scheduler's lock; FidelityFX submits to
// the game queue through this.
VkResult GameQueueSubmit(uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    std::scoped_lock lock{*g_submit_mutex};
    return g_queue_submit(g_game_queue, count, submits, fence);
}

FfxApiResource ImageResource(const Image& image, bool depth) {
    FfxApiResource resource{};
    resource.resource = static_cast<void*>(static_cast<VkImage>(image.image));
    resource.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
    auto& desc = resource.description;
    desc.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    desc.format = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
    desc.width = image.extent.width;
    desc.height = image.extent.height;
    desc.depth = 1;
    desc.mipCount = 1;
    desc.flags = FFX_API_RESOURCE_FLAGS_NONE;
    desc.usage = depth ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : FFX_API_RESOURCE_USAGE_READ_ONLY;
    return resource;
}

// 0.41: what FSR 3 frame generation needs from the inputs, checked before anything reaches the
// FidelityFX API (the native path, BB_FG_NATIVE, hands over the game's own frame layout, whatever
// the quality patches make of it). nullptr: usable; else the reason, logged once per reason.
const char* InputsProblem(const Inputs& in, vk::Extent2D backbuffer) {
    if (!in.depth.image || !in.motion.image) return "depth or motion-vector image missing";
    if (in.render.width == 0 || in.render.height == 0) return "render size 0";
    if (in.render.width > backbuffer.width || in.render.height > backbuffer.height)
        return "render size larger than the backbuffer (FSR 3 FG needs render <= display)";
    if (in.depth.extent.width < in.render.width || in.depth.extent.height < in.render.height ||
        in.motion.extent.width < in.render.width || in.motion.extent.height < in.render.height)
        return "depth / motion-vector image smaller than the render size";
    if (ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(in.depth.format)) ==
            FFX_API_SURFACE_FORMAT_UNKNOWN ||
        ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(in.motion.format)) ==
            FFX_API_SURFACE_FORMAT_UNKNOWN)
        return "depth or motion-vector format unknown to FidelityFX";
    if (!(in.motion.usage & vk::ImageUsageFlagBits::eSampled) ||
        !(in.depth.usage & vk::ImageUsageFlagBits::eSampled))
        return "depth or motion-vector image not sampled-usable";
    return nullptr;
}

void LogInputs(const char* step, const Inputs& in, vk::Extent2D backbuffer, vk::Format format) {
    Printf("FSR3 FG: %s: backbuffer %ux%u %s; render %ux%u; depth %p %ux%u %s usage 0x%x; motion %p "
           "%ux%u %s usage 0x%x; HUD-less %s %p %ux%u %s; jitter %.3f %.3f, camera %s, depth %s, "
           "fov %.4f, near %.4f, reset %d, context %p, swapchain context %p",
           step, backbuffer.width, backbuffer.height, vk::to_string(format).c_str(),
           in.render.width, in.render.height,
           static_cast<void*>(static_cast<VkImage>(in.depth.image)), in.depth.extent.width,
           in.depth.extent.height, vk::to_string(in.depth.format).c_str(),
           static_cast<uint32_t>(in.depth.usage),
           static_cast<void*>(static_cast<VkImage>(in.motion.image)), in.motion.extent.width,
           in.motion.extent.height, vk::to_string(in.motion.format).c_str(),
           static_cast<uint32_t>(in.motion.usage), in.has_hudless ? "yes" : "no",
           static_cast<void*>(static_cast<VkImage>(in.hudless.image)), in.hudless.extent.width,
           in.hudless.extent.height, vk::to_string(in.hudless.format).c_str(), in.jitter[0],
           in.jitter[1], in.camera_valid ? "valid" : "invalid",
           in.depth_inverted ? "inverted" : "forward", in.fov_y, in.z_near, in.reset ? 1 : 0,
           static_cast<void*>(g_fg_context), static_cast<void*>(g_swapchain_context));
    std::fflush(stdout);
}

void DestroyFrameGeneration() {
    if (g_fg_context != nullptr) {
        ffxConfigureDescFrameGeneration off{};
        off.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        off.frameGenerationEnabled = false;
        g_ffx.Configure(&g_fg_context, &off.header);
        g_ffx.DestroyContext(&g_fg_context, nullptr);
        g_fg_context = nullptr;
    }
}

bool CreateFrameGeneration(vk::Extent2D display, vk::Extent2D render, vk::Format format,
                           bool depth_inverted) {
    DestroyFrameGeneration();
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = g_device;
    backend.vkPhysicalDevice = g_physical;
    backend.vkDeviceProcAddr = g_get_device_proc;

    ffxCreateContextDescFrameGeneration create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.header.pNext = &backend.header;
    // Bloodborne's depth is reversed (1 near, 0 far) with the far plane taken as infinite, as the
    // upscaler gives FSR 3.1 (KYTY_FSR3_DEPTH=normal: as is).
    const char* depth = Env("KYTY_FSR3_DEPTH");
    const bool normal = (depth != nullptr && std::string(depth) == "normal") || !depth_inverted;
    create.flags = normal ? 0u
                          : (FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED |
                             FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE);
    if (EnvOn("KYTY_FSR3_DEBUG", false)) create.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    if (format == vk::Format::eA2B10G10R10UnormPack32 ||
        format == vk::Format::eA2R10G10B10UnormPack32) {
        create.flags |= FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    create.displaySize = {display.width, display.height};
    create.maxRenderSize = {render.width, render.height};
    create.backBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(format));
    const auto result = g_ffx.CreateContext(&g_fg_context, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        Printf("FSR3 FG: frame generation context not created (%u): generation stays off", result);
        g_fg_context = nullptr;
        g_fg_failed = true;
        return false;
    }
    g_fg_display = display;
    g_fg_max_render = render;
    g_fg_format = format;
    Printf("FSR3 FG: frame generation context %ux%u (render up to %ux%u, depth %s)", display.width,
           display.height, render.width, render.height, normal ? "as is" : "reversed, infinite");
    return true;
}

bool EnsureCommandBuffers() {
    if (g_pool != VK_NULL_HANDLE) return true;
    auto* create_pool =
        reinterpret_cast<PFN_vkCreateCommandPool>(g_get_device_proc(g_device, "vkCreateCommandPool"));
    auto* allocate = reinterpret_cast<PFN_vkAllocateCommandBuffers>(
        g_get_device_proc(g_device, "vkAllocateCommandBuffers"));
    auto* create_fence =
        reinterpret_cast<PFN_vkCreateFence>(g_get_device_proc(g_device, "vkCreateFence"));
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = g_game_family;
    if (create_pool(g_device, &pool, nullptr, &g_pool) != VK_SUCCESS) return false;
    for (auto& slot : g_slots) {
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = g_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        if (allocate(g_device, &info, &slot.command) != VK_SUCCESS) return false;
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (create_fence(g_device, &fence, nullptr, &slot.fence) != VK_SUCCESS) return false;
    }
    return true;
}

// The camera's vertical field of view and near plane from the world->clip matrix (row-major,
// clip = M * world), and its axes and position (FSR 3.1.4's camera info).
struct CameraValues {
    float fov = 1.0f, z_near = 0.1f;
    float position[3]{}, up[3]{0, 1, 0}, right[3]{1, 0, 0}, forward[3]{0, 0, 1};
    bool valid = false;
};
CameraValues FromCamera(const float* m, bool camera_valid) {
    CameraValues out;
    if (!camera_valid) return out;
    const auto row = [&](uint32_t i) {
        return std::array<double, 4>{m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]};
    };
    const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
    if (flen < 1e-6) return out;
    const double f[3]{r3[0] / flen, r3[1] / flen, r3[2] / flen};
    const double tf = r3[3] / flen;
    const auto split = [&](const std::array<double, 4>& r, double& s, double axis[3], double& t) {
        const double off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
        double v[3]{r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
        s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (s < 1e-9) return false;
        for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
        t = (r[3] / flen - off * tf) / s;
        return true;
    };
    double sx = 0, sy = 0, x[3]{}, y[3]{}, tx = 0, ty = 0;
    if (!split(r0, sx, x, tx) || !split(r1, sy, y, ty)) return out;
    const double up[3]{f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2],
                       f[0] * x[1] - f[1] * x[0]};
    if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) ty = -ty;
    for (int i = 0; i < 3; ++i) {
        out.right[i] = static_cast<float>(x[i]);
        out.up[i] = static_cast<float>(up[i]);
        out.forward[i] = static_cast<float>(f[i]);
        out.position[i] = static_cast<float>(-(tx * x[i] + ty * up[i] + tf * f[i]));
    }
    out.fov = static_cast<float>(2.0 * std::atan(1.0 / std::fabs(sy)));
    const double z_near = std::fabs(r2[3] / flen);
    out.z_near = z_near > 1e-4 && z_near < 100.0 ? static_cast<float>(z_near) : 0.1f;
    out.valid = std::isfinite(out.fov) && out.fov > 0.1f && out.fov < 3.0f;
    return out;
}

void LogState(int state, const char* text) {
    if (state == g_state_logged) return;
    g_state_logged = state;
    Printf("FSR3 FG: %s", text);
}

void LogPresentIntervals(double window) {
    int64_t pace[8]{};
    g_take_pace_stats(pace);
    const double n = static_cast<double>(pace[1]);
    const double mean_ms = n > 0.0 ? static_cast<double>(pace[2]) / n / 1e6 : 0.0;
    const double var_ms2 =
        n > 0.0 ? static_cast<double>(pace[3]) / n / 1e12 - mean_ms * mean_ms : 0.0;
    NotePacing(-1.0, std::sqrt(std::max(var_ms2, 0.0))); // 1.54: all presents
    Printf("[midpoint] real %.1f fps, shown %.1f fps (%lld presents); interval over all presents: "
           "mean %.2f ms, sd %.2f ms (min %.2f, max %.2f, %lld intervals); placed %u of %u real "
           "flips; late %lld (max %.2f ms)",
           g_stats.real / window, static_cast<double>(pace[0]) / window,
           static_cast<long long>(pace[0]), mean_ms, std::sqrt(std::max(var_ms2, 0.0)),
           static_cast<double>(pace[4]) / 1e6, static_cast<double>(pace[5]) / 1e6,
           static_cast<long long>(pace[1]), g_stats.slotted, g_stats.real,
           static_cast<long long>(pace[6]), static_cast<double>(pace[7]) / 1e6);
}

bool LoadApi() {
    static const bool loaded = [] {
        wchar_t exe[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe);
        dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
        for (const auto* name : {L"amd_fidelityfx_vk.dll", L"fsr3\\amd_fidelityfx_vk.dll"}) {
            g_dll = LoadLibraryW((dir + name).c_str());
            if (g_dll != nullptr) break;
        }
        if (g_dll == nullptr) {
            Printf("FSR3: amd_fidelityfx_vk.dll not found next to the emulator");
            return false;
        }
        ffxLoadFunctions(&g_ffx, g_dll);
        if (g_ffx.CreateContext == nullptr || g_ffx.Configure == nullptr ||
            g_ffx.Dispatch == nullptr) {
            Printf("FSR3: amd_fidelityfx_vk.dll has no FidelityFX API");
            return false;
        }
        const bool patched = GetProcAddress(g_dll, "kytyFfxGetRealSwapchainVK") != nullptr;
        g_set_pace_slot = reinterpret_cast<void (*)(int64_t, int64_t)>(
            GetProcAddress(g_dll, "kytyFfxSetPaceSlot"));
        g_take_pace_stats =
            reinterpret_cast<void (*)(int64_t*)>(GetProcAddress(g_dll, "kytyFfxTakePaceStats"));
        g_take_display_ids = reinterpret_cast<uint32_t (*)(uint64_t*, uint32_t)>(
            GetProcAddress(g_dll, "kytyFfxTakeDisplayIds"));
        Display::SetAmdHooks(reinterpret_cast<VkSwapchainKHR (*)(VkSwapchainKHR)>(
                                 GetProcAddress(g_dll, "kytyFfxGetRealSwapchainVK")),
                             g_take_display_ids);
        Printf("FSR3 FG: amd_fidelityfx_vk.dll loaded (%s)",
               patched ? "patched: multi frame generation x2-x4, dynamic"
                       : "AMD's signed DLL: x2 only");
        return true;
    }();
    return loaded;
}

double CapInterval() {
    static const double interval = [] {
        const char* text = Env("KYTY_VRR_CAP_FPS");
        if (text == nullptr || *text == '\0') return 0.0;
        double cap = std::strtod(text, nullptr);
        if (std::string(text) == "auto") {
            cap = RefreshTarget();
        }
        const char* frames = Env("KYTY_FSR3_FG_FRAMES");
        const int value = frames != nullptr ? std::atoi(frames) : 1;
        const int generated = value >= 3 ? 3 : (value >= 2 ? 2 : 1);
        if (cap <= 0.0) return 0.0;
        Printf("FSR3 FG: shown frames held under %.0f fps (real at most %.1f)", cap,
               cap / (1 + generated));
        return (1 + generated) / cap;
    }();
    if (g_mfg.on && g_mfg.dynamic) {
        return g_mfg.current / g_mfg.target;
    }
    return interval;
}

} // namespace

bool Requested() {
    static const bool requested = [] {
        if (Cfg().kind != Kind::Amd || !EnvOn("KYTY_FSR3_FG", false)) return false;
        if (!LoadApi()) {
            Printf("FSR3 FG: frame generation off");
            return false;
        }
        Printf("FSR3 FG: requested (BB_FG=%s), FidelityFX API loaded", Name());
        g_requested = true;
        return true;
    }();
    return requested && g_requested;
}

QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family) {
    QueuePlan plan{};
    if (!Requested()) return plan;
    const auto families = physical.getQueueFamilyProperties();
    if (graphics_family >= families.size() || families[graphics_family].queueCount < 2) {
        Printf("FSR3 FG: no free graphics queue for its presents: frame generation off");
        g_requested = false;
        return plan;
    }
    for (uint32_t family = 0; family < families.size(); ++family) {
        const auto flags = families[family].queueFlags;
        if (family != graphics_family && (flags & vk::QueueFlagBits::eCompute) &&
            !(flags & vk::QueueFlagBits::eGraphics) && families[family].queueCount >= 2) {
            plan.compute_family = family;
            plan.compute_count = 2;
            break;
        }
    }
    if (plan.compute_family == UINT32_MAX) {
        Printf("FSR3 FG: no async compute queue family: frame generation off");
        g_requested = false;
        return plan;
    }
    plan.graphics_count = 2;
    return plan;
}

void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t graphics_family,
              vk::Queue graphics_queue, const QueuePlan& plan, std::mutex* submit_mutex) {
    if (!Requested() || plan.compute_family == UINT32_MAX || plan.graphics_count < 2) return;
    g_get_device_proc = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    g_device = static_cast<VkDevice>(device);
    g_physical = static_cast<VkPhysicalDevice>(physical);
    g_game_queue = static_cast<VkQueue>(graphics_queue);
    g_game_family = graphics_family;
    g_submit_mutex = submit_mutex;
    g_queue_submit = reinterpret_cast<PFN_vkQueueSubmit>(g_get_device_proc(g_device, "vkQueueSubmit"));
    auto* get_queue =
        reinterpret_cast<PFN_vkGetDeviceQueue>(g_get_device_proc(g_device, "vkGetDeviceQueue"));
    get_queue(g_device, graphics_family, 1, &g_present_queue);
    get_queue(g_device, plan.compute_family, 0, &g_async_queue);
    get_queue(g_device, plan.compute_family, 1, &g_acquire_queue);
    g_compute_family = plan.compute_family;
    if (g_present_queue == VK_NULL_HANDLE || g_async_queue == VK_NULL_HANDLE ||
        g_acquire_queue == VK_NULL_HANDLE || g_queue_submit == nullptr || submit_mutex == nullptr) {
        Printf("FSR3 FG: its queues were not created: frame generation off");
        g_requested = false;
        return;
    }
    ffxConfigureDescGlobalDebug1 debug{};
    debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    debug.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
    debug.fpMessage = &Message;
    g_ffx.Configure(nullptr, &debug.header);
    Printf("FSR3 FG: queues: present (graphics family %u, queue 1), async compute and image "
           "acquire (family %u)",
           graphics_family, plan.compute_family);
}

bool Active() {
    return g_swapchain_context != nullptr;
}

// 1.51 (bb-fgmv-151): the emulator's own overlay (ImGui: trophy toasts, F1 menu, metrics) drawn
// into a separate transparent image this present (vk_presenter.cpp), registered as FSR 3's UI
// resource: the FG swapchain composites it over every presented frame, real and generated, after
// interpolation (it was drawn into the back buffer and interpolated: ghosting).
Image g_ui_layer{};
bool g_ui_layer_set = false;
uint64_t g_ui_layer_frames = 0;

void SetUiLayer(const Image* ui) {
    std::lock_guard lock(g_mutex);
    g_ui_layer_set = ui != nullptr && ui->image;
    if (g_ui_layer_set) g_ui_layer = *ui;
}

bool WantsUiLayer() {
    static const bool on = EnvOn("BB_FG_UI_LAYER", true);
    return on && Active();
}

void RegisterUiLayerLocked() {
    if (g_swapchain_context == nullptr) return;
    ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK ui{};
    ui.header.type = FFX_API_CONFIGURE_DESC_TYPE_FGSWAPCHAIN_REGISTERUIRESOURCE_VK;
    if (g_ui_layer_set) {
        ui.uiResource = ImageResource(g_ui_layer, false);
        ui.uiResource.state = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
        ui.flags = FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA |
                   FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING;
        if (g_ui_layer_frames++ == 0) {
            Printf("[fg-ui] 1.51 overlay (ImGui) given to FSR 3 as its UI layer (%ux%u %s, "
                   "premultiplied, double-buffered): composited after frame generation",
                   g_ui_layer.extent.width, g_ui_layer.extent.height,
                   vk::to_string(g_ui_layer.format).c_str());
        }
    }
    g_ffx.Configure(&g_swapchain_context, &ui.header);
    g_ui_layer_set = false;
}

double PaceInterval() {
    return Active() && g_generating ? CapInterval() : 0.0;
}

bool Generating() {
    return Active() && g_generating;
}

void NoteSlept(double seconds) {
    g_mfg.slept += seconds;
}

int Generated() {
    if (!Active() || !g_generating) return 0;
    return g_mfg.on ? g_mfg.current - 1 : 1;
}

// 1.51 (bb-fgmv-151): present watchdog. r149_0459: presents stopped completing ("source=none
// [no present completed]") while the game kept rendering: the present thread blocked in the FG
// swapchain (its acquire waited forever, UINT64_MAX). The acquire now waits at most
// BB_FG_WATCHDOG_MS (default 500, 0 = forever); on a timeout FSR 3 frame generation is turned
// off for the session and the swapchain is made again as a plain one (out of date -> recreate).
uint64_t WatchdogNs() {
    static const uint64_t ns = [] {
        const char* e = std::getenv("BB_FG_WATCHDOG_MS");
        const long ms = e && *e ? std::strtol(e, nullptr, 10) : 500;
        return ms <= 0 ? UINT64_MAX : static_cast<uint64_t>(ms) * 1'000'000ull;
    }();
    return ns;
}
std::atomic<bool> g_watchdog_off{false};
std::atomic<uint32_t> g_watchdog_trips{0};

void WatchdogTrip(const char* where, VkResult result) {
    const uint32_t n = g_watchdog_trips.fetch_add(1) + 1;
    g_watchdog_off.store(true);
    Printf("[fg-watchdog] 1.51 FSR 3 FG swapchain stalled: %s gave %d after %.0f ms (trip %u): "
           "frame generation OFF for this session, the swapchain is made again without it",
           where, static_cast<int>(result), WatchdogNs() / 1e6, n);
}

bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out) {
    if (!Requested() || g_device == VK_NULL_HANDLE || g_watchdog_off.load()) return false;
    std::lock_guard lock(g_mutex);
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    ffxCreateContextDescFrameGenerationSwapChainVK create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    create.physicalDevice = g_physical;
    create.device = g_device;
    create.swapchain = &swapchain;
    create.allocator = nullptr;
    create.createInfo = static_cast<VkSwapchainCreateInfoKHR>(info);
    create.gameQueue = {g_game_queue, g_game_family, &GameQueueSubmit};
    create.asyncComputeQueue = {g_async_queue, g_compute_family, nullptr};
    create.presentQueue = {g_present_queue, g_game_family, nullptr};
    create.imageAcquireQueue = {g_acquire_queue, g_compute_family, nullptr};
    MfgInit();
    g_display_ids = g_take_display_ids != nullptr && Display::IdsAvailable();
    SetEnvironmentVariableA("KYTY_FSR3_DISPLAY_IDS", g_display_ids ? "1" : nullptr);
    Printf("[display] FSR 3 presents: %s", g_display_ids
               ? "every present to the driver (the generated frames too) carries its own id; the "
                 "display clock waits for each (VK_KHR_present_wait)"
               : g_take_display_ids == nullptr
                     ? "AMD's / an older FidelityFX DLL: no ids for the generated frames, the "
                       "display clock cannot time them (source=none)"
                     : "present ids / VK_KHR_present_wait not available on this device (source=none)");
    {
        // Pacing floor (the patched DLL; AMD's ignores it): no present of the frame interpolation
        // swapchain closer than the display's refresh period x KYTY_FSR3_PACING_FLOOR (default 1.0,
        // 0 off).
        static bool logged = false;
        double factor = 1.0;
        if (const char* text = Env("KYTY_FSR3_PACING_FLOOR"); text != nullptr && *text != '\0') {
            factor = std::strtod(text, nullptr);
        }
        DEVMODEA mode{};
        mode.dmSize = sizeof(mode);
        const bool refresh = EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 &&
                             mode.dmDisplayFrequency > 30;
        if (factor > 0.0 && refresh) {
            const double floor_us = 1000000.0 / mode.dmDisplayFrequency * factor;
            SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", std::to_string(floor_us).c_str());
            if (!logged) {
                Printf("FSR3 FG: present pacing floor %.3f ms (display %lu Hz x %.2f; "
                       "BB_FSR3_PACING_FLOOR=0 off)",
                       floor_us / 1000.0, static_cast<unsigned long>(mode.dmDisplayFrequency), factor);
            }
        } else {
            SetEnvironmentVariableA("KYTY_FSR3_PACING_FLOOR_US", nullptr);
            if (!logged) {
                Printf("FSR3 FG: no present pacing floor (%s)",
                       refresh ? "BB_FSR3_PACING_FLOOR=0" : "display refresh unknown");
            }
        }
        logged = true;
    }
    const auto result = g_ffx.CreateContext(&g_swapchain_context, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK || swapchain == VK_NULL_HANDLE) {
        Printf("FSR3 FG: frame generation swapchain not created (%u): frame generation off", result);
        g_swapchain_context = nullptr;
        g_requested = false;
        return false;
    }
    if (const char* frames = Env("KYTY_FSR3_FG_FRAMES"); frames != nullptr && std::atoi(frames) >= 2) {
        const int generated = std::atoi(frames) >= 3 ? 3 : 2;
        Printf("FSR3 FG: KYTY_FSR3_FG_FRAMES=%s: %d generated frames per real frame (x%d, patched "
               "DLL only)",
               frames, generated, generated + 1);
    }
    g_functions.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    g_ffx.Query(&g_swapchain_context, &g_functions.header);
    if (g_functions.pOutAcquireNextImageKHR == nullptr || g_functions.pOutQueuePresentKHR == nullptr ||
        g_functions.pOutGetSwapchainImagesKHR == nullptr) {
        Printf("FSR3 FG: no replacement swapchain functions: frame generation off");
        g_ffx.DestroyContext(&g_swapchain_context, nullptr);
        g_swapchain_context = nullptr;
        g_requested = false;
        return false;
    }
    Printf("FSR3 FG: frame generation swapchain %ux%u (present mode %d)", info.imageExtent.width,
           info.imageExtent.height, static_cast<int>(info.presentMode));
    *out = swapchain;
    return true;
}

void DestroySwapchain() {
    std::lock_guard lock(g_mutex);
    if (g_swapchain_context == nullptr) return;
    DestroyFrameGeneration();
    g_ffx.DestroyContext(&g_swapchain_context, nullptr);
    g_swapchain_context = nullptr;
    g_functions = {};
    g_generating = false;
    g_state_logged = -1;
    g_stats = {};
    Printf("FSR3 FG: frame generation swapchain released (resize, fullscreen or exit)");
}

vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
    if (g_functions.pOutGetSwapchainImagesKHR == nullptr) return vk::Result::eErrorOutOfDateKHR;
    return static_cast<vk::Result>(g_functions.pOutGetSwapchainImagesKHR(
        g_device, static_cast<VkSwapchainKHR>(swapchain), count, reinterpret_cast<VkImage*>(images)));
}

vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
    if (g_functions.pOutAcquireNextImageKHR == nullptr) return vk::Result::eErrorOutOfDateKHR;
    if (g_watchdog_off.load()) return vk::Result::eErrorOutOfDateKHR;
    const VkResult r = g_functions.pOutAcquireNextImageKHR(
        g_device, static_cast<VkSwapchainKHR>(swapchain), WatchdogNs(),
        static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE, index);
    if (r == VK_TIMEOUT || r == VK_NOT_READY) {
        WatchdogTrip("vkAcquireNextImageKHR", r);
        return vk::Result::eErrorOutOfDateKHR;
    }
    return static_cast<vk::Result>(r);
}

vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info) {
    std::lock_guard lock(g_mutex);
    if (g_swapchain_context == nullptr || g_functions.pOutQueuePresentKHR == nullptr) {
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
            Printf("FSR3 FG: present without a frame generation swapchain: skipped, swapchain made "
                   "again");
        }
        return vk::Result::eErrorOutOfDateKHR;
    }
    const auto begin = std::chrono::steady_clock::now();
    VkPresentInfoKHR present = static_cast<VkPresentInfoKHR>(info);
    present.pNext = nullptr;
    VkPresentIdKHR present_id{};
    uint64_t game_id = 0;
    if (g_display_ids) {
        for (auto* node = static_cast<const VkBaseInStructure*>(info.pNext); node != nullptr;
             node = node->pNext) {
            if (node->sType == VK_STRUCTURE_TYPE_PRESENT_ID_KHR) {
                const auto* id = reinterpret_cast<const VkPresentIdKHR*>(node);
                if (id->swapchainCount > 0 && id->pPresentIds != nullptr) game_id = id->pPresentIds[0];
                break;
            }
        }
        if (game_id != 0) {
            present_id.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
            present_id.swapchainCount = 1;
            present_id.pPresentIds = &game_id;
            present.pNext = &present_id;
        }
    }
    auto slot = BB::FG::PresentSlotNow();
    {
        const double now_s = SteadyNow();
        auto& h = g_hold;
        if (h.last_arrive > 0.0) {
            const double dt = now_s - h.last_arrive;
            if (dt > 0.0 && dt < 0.1) {
                g_ft.real.push_back(dt * 1e3);
                h.ring[h.next] = dt;
                h.next = (h.next + 1) % 32;
                ++h.count;
            } else {
                h.last_slot = 0.0; // a pause: start a new grid
            }
        }
        h.last_arrive = now_s;
        if (slot.qpc == 0 && g_set_pace_slot != nullptr && g_generating && HoldPace()) {
            // The step: the guest pacer's (the real flips are held on it), else a held mean of
            // the real present intervals (moves only when the mean leaves it by 0.5 ms).
            double step = GuestPacing() ? PaceEffective() : CapInterval();
            if (!(step > 0.0 && step < 0.1) && h.count >= 16) {
                const int n = std::min(h.count, 32);
                double sum = 0.0;
                for (int i = 0; i < n; ++i) sum += h.ring[i];
                const double mean = sum / n;
                step = (h.step <= 0.0 || std::fabs(mean - h.step) > 0.0005) ? mean : h.step;
            }
            if (step > 0.0 && step < 0.1) {
                if (std::fabs(step - h.step) > 1e-6) h.step = step;
                double s = now_s;
                const double expected = h.last_slot + step;
                if (h.last_slot > 0.0 && std::fabs(now_s - expected) <= 0.4 * step) {
                    s = expected + (now_s - expected) * 0.125;
                } else {
                    ++g_ft.rephase;
                }
                h.last_slot = s;
                LARGE_INTEGER qpc{}, freq{};
                QueryPerformanceCounter(&qpc);
                QueryPerformanceFrequency(&freq);
                const double f = static_cast<double>(freq.QuadPart);
                slot.qpc = qpc.QuadPart + static_cast<int64_t>((s - now_s) * f);
                slot.interval = static_cast<int64_t>(step * f);
                ++g_ft.own_slots;
                // The planned presents: N generated at slot + i * step / (N + 1), the real at
                // slot + step.
                const int gen = std::max(Generated(), 1);
                double prev = h.planned_last;
                for (int i = 1; i <= gen + 1; ++i) {
                    const double t = s + step * i / (gen + 1);
                    if (prev > 0.0 && t - prev > 0.0 && t - prev < 0.1) {
                        g_ft.shown.push_back((t - prev) * 1e3);
                    }
                    prev = t;
                }
                h.planned_last = prev;
            } else {
                h.last_slot = 0.0;
                h.planned_last = 0.0;
            }
        } else if (slot.qpc == 0) {
            h.last_slot = 0.0;
            h.planned_last = 0.0;
        }
    }
    if (g_set_pace_slot != nullptr) {
        g_set_pace_slot(slot.qpc, slot.interval);
    }
    g_stats.slotted += slot.qpc != 0 ? 1 : 0;
    auto result = g_functions.pOutQueuePresentKHR(static_cast<VkQueue>(queue), &present);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    g_stats.present_ms += ms;
    g_stats.max_present_ms = std::max(g_stats.max_present_ms, ms);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
            Printf("FSR3 FG: frame generation present out of %s memory (%d): swapchain and "
                   "generation context made again",
                   result == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "video" : "host",
                   static_cast<int>(result));
        }
        result = VK_ERROR_OUT_OF_DATE_KHR;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
            Printf("FSR3 FG: frame generation present returned %d", static_cast<int>(result));
        }
    }
    return static_cast<vk::Result>(result);
}

void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format) {
    if (!Active()) return;
    Display::SetExpected(0);
    MfgInit();
    // G-SYNC / the dynamic target: real frames wait just enough for real x (1 + generated) to stay
    // under the cap. Only while generating; movies and menus untouched.
    const double interval = CapInterval();
    if (interval > 0.0 && inputs != nullptr && inputs->valid && !GuestPacing()) {
        static std::chrono::steady_clock::time_point next{};
        const auto step = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(interval));
        auto now = std::chrono::steady_clock::now();
        if (next > now) {
            g_mfg.slept += std::chrono::duration<double>(next - now).count();
            std::this_thread::sleep_until(next - std::chrono::milliseconds(1));
            while (std::chrono::steady_clock::now() < next) std::this_thread::yield();
            now = next;
        }
        next = std::max(next, now - step) + step;
    }
    std::lock_guard lock(g_mutex);
    RegisterUiLayerLocked(); // 1.51: every present, generating or not
    const auto now = std::chrono::steady_clock::now();
    // bb-dcideas-094: never for movies or 2D flips (menus, loading screens).
    bool usable = inputs != nullptr && inputs->valid && !g_fg_failed && !FgBlocked(nullptr) &&
                  !g_watchdog_off.load();
    // 1.51: while an emulator overlay is up (F1 menu, trophy list / toasts: ImGui keeps drawing)
    // FSR 3 FG is bypassed on every present (real frames presented directly) and resumes when it
    // closes. r149_0451 / r149_0459: opening the menu with FG on toggled generation off / on every
    // other present until presents stopped completing. BB_FG_OVERLAY_BYPASS=0: old.
    static const bool overlay_bypass = EnvOn("BB_FG_OVERLAY_BYPASS", true);
    {
        static bool was = false;
        const bool overlay = overlay_bypass && ImGui::Core::MustKeepDrawing();
        if (overlay != was) {
            was = overlay;
            Printf("[fg-ui] 1.51 overlay %s: FSR 3 frame generation %s", overlay ? "open" : "closed",
                   overlay ? "bypassed (real frames presented directly)" : "resumes");
        }
        if (overlay) usable = false;
    }
    static int first_steps = 0; // 0.41: the first 3D frames' steps, logged (a crash shows the last)
    if (usable) {
        if (const char* why = InputsProblem(*inputs, backbuffer)) {
            static std::string said;
            if (said != why) {
                said = why;
                Printf("FSR3 FG: 3D inputs not usable (%s): this frame not generated", why);
                LogInputs("refused inputs", *inputs, backbuffer, format);
            }
            usable = false;
        } else if (first_steps < 3) {
            LogInputs("3D inputs received", *inputs, backbuffer, format);
        }
    }

    if (g_stats.start == std::chrono::steady_clock::time_point{}) g_stats.start = now;
    const double window = std::chrono::duration<double>(now - g_stats.start).count();
    if (window >= 1.0) {
        const uint64_t presents =
            g_functions.pOutGetLastPresentCountFFXAPI != nullptr
                ? g_functions.pOutGetLastPresentCountFFXAPI(static_cast<VkSwapchainKHR>(swapchain))
                : 0;
        const uint64_t shown =
            presents >= g_stats.presents_at_start ? presents - g_stats.presents_at_start : 0;
        Printf("[fsr3] real %.1f fps, shown %.1f fps; generation on %u of %u frames (no 3D %u); "
               "prepare %.2f ms, present call %.2f ms (max %.1f); HUD-less %u",
               g_stats.real / window, static_cast<double>(shown) / window, g_stats.generated_on,
               g_stats.real, g_stats.no_inputs,
               g_stats.real != 0 ? g_stats.prepare_ms / g_stats.real : 0.0,
               g_stats.real != 0 ? g_stats.present_ms / g_stats.real : 0.0, g_stats.max_present_ms,
               g_stats.hudless);
        Printf("[fsr3-ui] HUD-less image given %u, refused (size/format != backbuffer) %u, none "
               "captured %u: generated frames %s",
               g_stats.hudless, g_stats.hudless_mismatch, g_stats.hudless_none,
               g_stats.hudless_mismatch + g_stats.hudless_none == 0
                   ? "reuse the real HUD"
                   : "warp the HUD in frames without a HUD-less image");
        if (g_stats.pace_n >= 10.0) {
            const double mean = g_stats.pace_sum / g_stats.pace_n;
            const double sd =
                std::sqrt(std::max(g_stats.pace_sq / g_stats.pace_n - mean * mean, 0.0));
            NotePacing(sd * 1e3, -1.0); // 1.54: real frames
            Printf("[fsr3] present interval sd %.2f ms (mean %.2f, max %.1f, %.0f frames); pacer "
                   "%.2f ms (%s)",
                   sd * 1e3, mean * 1e3, g_stats.pace_max * 1e3, g_stats.pace_n,
                   (GuestPacing() ? PaceEffective() : CapInterval()) * 1e3,
                   GuestPacing() ? "guest flips" : "present");
        }
        if (g_take_pace_stats != nullptr) {
            LogPresentIntervals(window);
        }
        {
            // 0.92: real / planned-shown frame times (ms) of the last second.
            std::string line = "[fsr3-ft] ";
            FtLine("real", g_ft.real, line);
            line += "; shown (planned slots)";
            FtLine("", g_ft.shown, line);
            char tail[200];
            std::snprintf(tail, sizeof(tail),
                          "; step %.2f ms, own slots %u, re-phased %u, hold %s; DLL measured: "
                          "the [midpoint] line",
                          g_hold.step * 1e3, g_ft.own_slots, g_ft.rephase,
                          HoldPace() ? "on" : "off (BB_FSR3_PACE_HOLD=0)");
            line += tail;
            Printf("%s", line.c_str());
            g_ft.rephase = 0;
            g_ft.own_slots = 0;
        }
        {
            // 1.40: what FSR 3 frame interpolation got (bb-fgamd-140), beside [fg-objmv].
            const auto cov = BB::ObjMotion::TakeCoverage();
            const uint32_t dispatches = g_amd_dispatches.exchange(0, std::memory_order_relaxed);
            const uint32_t ffx_resets = g_amd_ffx_resets.exchange(0, std::memory_order_relaxed);
            const double all = static_cast<double>(std::max<uint64_t>(cov.all_px, 1));
            {
                // 1.41: one line proving which frame each input came from, and what FSR 3 was
                // told for it. "paired" = colour == depth == camera MVs == object MVs == HUD-less.
                const uint32_t n = std::max<uint32_t>(g_amd.prepared, 1);
                const auto id = [](uint64_t v) {
                    return v == ~0ull ? -1ll : static_cast<long long>(v);
                };
                const double chr = static_cast<double>(std::max<uint64_t>(cov.character_px, 1));
                Printf("[fg-amd] 1.41 pairing: %u of %u prepared frames paired (colour == depth "
                       "== camera MV == object MV == HUD-less frame); off: depth %u, camera MV %u, "
                       "object MV %u (no G-buffer motion pass that frame = camera-only), HUD-less "
                       "%u | last ids colour %lld depth %lld mv %lld obj %lld hudless %lld, FSR "
                       "frameID %llu | told FSR: frameTimeDelta mean %.2f ms max %.2f, HUD-less "
                       "colour %u/%u, UI texture none, flags 0x%x, viewSpaceToMeters 1, reset 0 "
                       "unless camera cut | character vectors: %.1f %% of character pixels near "
                       "zero (< 0.25 px), %.1f %% with own motion (> 1 px off camera)",
                       g_amd.paired, g_amd.prepared, g_amd.off_depth, g_amd.off_mv, g_amd.off_obj,
                       g_amd.off_hudless, id(g_amd.last_ids[0]), id(g_amd.last_ids[1]),
                       id(g_amd.last_ids[2]), id(g_amd.last_ids[3]), id(g_amd.last_ids[4]),
                       static_cast<unsigned long long>(g_amd.last_ffx_frame), g_amd.dt_sum / n,
                       g_amd.dt_max, g_amd.hudless_given, g_amd.prepared,
                       EnvOn("KYTY_FSR3_DEBUG_VIEW", false)
                           ? static_cast<uint32_t>(FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW)
                           : 0u,
                       cov.character_px ? 100.0 * static_cast<double>(cov.character_still_px) / chr
                                        : 0.0,
                       cov.character_px ? 100.0 * static_cast<double>(cov.character_own_px) / chr
                                        : 0.0);
            }
            Printf("[fg-amd] 1.40 %.1f s: frames interpolated %u (%.1f/s; real %u, prepared %u, "
                   "FSR history resets %u, restarts after a gap %u, presents without inputs "
                   "while generating %u, input resets %u) | object-MV pixel coverage %.2f %% "
                   "(characters %.2f %%; %u of %u composes read; %u composes on a reset frame%s, "
                   "%u without a G-buffer motion pass)%s | MV scale %.3f %.3f (render px towards "
                   "the previous frame, /render size in FSR), jitter %.3f %.3f | camera near %.6g "
                   "far %.6g fov %.4f rad (%.1f deg), depth %s",
                   window, dispatches, dispatches / window, g_stats.real, g_amd.prepared,
                   ffx_resets, g_amd.restarts, g_amd.gaps, g_amd.input_resets,
                   100.0 * static_cast<double>(cov.object_px) / all,
                   100.0 * static_cast<double>(cov.character_px) / all, cov.frames_read,
                   cov.composes, cov.reset_frames,
                   cov.reset_skip_rule ? ", object vectors DROPPED there (BB_OBJMV_RESET_SKIP=1)"
                                       : ", object vectors kept",
                   cov.not_written, cov.enabled ? "" : " [coverage off]", g_amd.mv_scale[0],
                   g_amd.mv_scale[1], g_amd.jitter[0], g_amd.jitter[1], g_amd.near_v,
                   g_amd.far_v, g_amd.fov, g_amd.fov * 57.29578f,
                   g_amd.depth_inverted ? "inverted (near > far given, infinite flag)"
                                        : "forward");
            Printf("[fg-mv] 1.46 object-MV attachment: %.2f %% of pixels with an object vector, "
                   "%.2f %% drawn by a motion pipeline without one (no previous position %.2f "
                   "%%, behind camera %.2f %%, not finite %.2f %%)%s",
                   100.0 * static_cast<double>(cov.object_px) / all,
                   100.0 * static_cast<double>(cov.no_prev_px + cov.behind_px + cov.nonfinite_px) / all,
                   100.0 * static_cast<double>(cov.no_prev_px) / all,
                   100.0 * static_cast<double>(cov.behind_px) / all,
                   100.0 * static_cast<double>(cov.nonfinite_px) / all,
                   cov.frames_read && cov.object_px == 0 &&
                           cov.no_prev_px + cov.behind_px + cov.nonfinite_px == 0
                       ? " -> the motion attachment is never written by the G-buffer pass"
                       : "");
            Printf("[fg-mv] 1.50 vectors: character pixels %.2f %% of the image with an object "
                   "vector, mean |mv| characters %.2f px vs background %.2f px (render px a real "
                   "frame)",
                   100.0 * static_cast<double>(cov.character_px) / all,
                   cov.character_px ? cov.character_mag16 / 16.0 / cov.character_px : 0.0,
                   cov.background_px ? cov.background_mag16 / 16.0 / cov.background_px : 0.0);
            const AmdSecond keep = g_amd;
            g_amd = {};
            std::copy_n(keep.mv_scale, 2, g_amd.mv_scale);
            std::copy_n(keep.jitter, 2, g_amd.jitter);
            g_amd.near_v = keep.near_v;
            g_amd.far_v = keep.far_v;
            g_amd.fov = keep.fov;
            g_amd.depth_inverted = keep.depth_inverted;
        }
        MfgUpdate(g_stats.real, window, static_cast<double>(shown) / window);
        g_stats = {};
        g_stats.start = now;
        g_stats.presents_at_start = presents;
    }
    g_stats.real++;
    if (!usable) g_stats.no_inputs++;
    if (!usable && g_generating) g_amd.gaps++;
    {
        static std::chrono::steady_clock::time_point prev{};
        const double dt = prev == std::chrono::steady_clock::time_point{}
                              ? 0.0
                              : std::chrono::duration<double>(now - prev).count();
        prev = now;
        if (usable && dt > 0.0 && dt < 0.1) {
            g_stats.pace_n += 1.0;
            g_stats.pace_sum += dt;
            g_stats.pace_sq += dt * dt;
            g_stats.pace_max = std::max(g_stats.pace_max, dt);
        }
    }

    const double delta_ms =
        g_last_real == std::chrono::steady_clock::time_point{}
            ? 16.7
            : std::chrono::duration<double, std::milli>(now - g_last_real).count();
    g_last_real = now;

    if (usable) {
        const vk::Extent2D render = inputs->render;
        if (g_fg_context == nullptr || g_fg_display != backbuffer || g_fg_format != format ||
            render.width > g_fg_max_render.width || render.height > g_fg_max_render.height) {
            Printf("FSR3 FG: creating the frame generation context (display %ux%u, render %ux%u)",
                   backbuffer.width, backbuffer.height, render.width, render.height);
            CreateFrameGeneration(backbuffer,
                                  {std::max(render.width, g_fg_max_render.width),
                                   std::max(render.height, g_fg_max_render.height)},
                                  format, inputs->depth_inverted);
        }
    }
    if (g_fg_context == nullptr) {
        LogState(0, "off (no 3D frame yet: menu, loading or movie)");
        return;
    }

    const bool generate = usable && g_fg_context != nullptr;
    ++g_frame_id;
    ffxConfigureDescFrameGeneration config{};
    config.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    config.swapChain = static_cast<void*>(static_cast<VkSwapchainKHR>(swapchain));
    config.presentCallback = nullptr;
    config.presentCallbackUserContext = nullptr;
    config.frameGenerationCallback = [](ffxDispatchDescFrameGeneration* params,
                                        void* context) -> ffxReturnCode_t {
        g_amd_dispatches.fetch_add(1, std::memory_order_relaxed);
        if (params->reset) g_amd_ffx_resets.fetch_add(1, std::memory_order_relaxed);
        return g_ffx.Dispatch(reinterpret_cast<ffxContext*>(context), &params->header);
    };
    config.frameGenerationCallbackUserContext = &g_fg_context;
    config.frameGenerationEnabled = generate;
    config.allowAsyncWorkloads = EnvOn("KYTY_FSR3_ASYNC", false);
    // 1.51: the UI layer (the emulator overlay) is registered at the top of BeforePresent.
    // 1.40: KYTY_FSR3_HUDLESS=0 gives FSR 3 no HUD-less image (A/B: FSR then interpolates the
    // whole back buffer, HUD included; with it, wherever the back buffer and the HUD-less image
    // differ FSR takes the real frame's pixels as UI, which on characters would look 30 fps).
    static const bool give_hudless = EnvOn("KYTY_FSR3_HUDLESS", true);
    const bool hudless = give_hudless && generate && inputs->has_hudless && inputs->hudless.image &&
                         inputs->hudless.extent == backbuffer && inputs->hudless.format == format;
    config.HUDLessColor = FfxApiResource{};
    if (hudless) {
        config.HUDLessColor = ImageResource(inputs->hudless, false);
        ++g_stats.hudless;
    } else if (generate && !inputs->has_hudless) {
        ++g_stats.hudless_none;
    } else if (generate && inputs->has_hudless) {
        ++g_stats.hudless_mismatch;
        static bool said = false;
        if (!said) {
            said = true;
            Printf("FSR3 FG: HUD-less image %ux%u %s does not match the backbuffer %ux%u %s: not given",
                   inputs->hudless.extent.width, inputs->hudless.extent.height,
                   vk::to_string(inputs->hudless.format).c_str(), backbuffer.width,
                   backbuffer.height, vk::to_string(format).c_str());
        }
    }
    config.flags = EnvOn("KYTY_FSR3_DEBUG_VIEW", false)
                       ? static_cast<uint32_t>(FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW)
                       : 0u;
    config.onlyPresentGenerated = false;
    config.generationRect = {0, 0, static_cast<int32_t>(backbuffer.width),
                             static_cast<int32_t>(backbuffer.height)};
    config.frameID = g_frame_id;
    if (generate && first_steps < 3) {
        Printf("FSR3 FG: configure (frame %llu, HUD-less %s, swapchain %p)",
               static_cast<unsigned long long>(g_frame_id), hudless ? "given" : "not given",
               static_cast<void*>(static_cast<VkSwapchainKHR>(swapchain)));
    }
    if (const auto result = g_ffx.Configure(&g_fg_context, &config.header);
        result != FFX_API_RETURN_OK) {
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1) < 8) Printf("FSR3 FG: configure failed (%u)", result);
        return;
    }
    if (!generate) {
        g_generating = false;
        LogState(1, "off (no 3D frame: menu, loading or movie)");
        return;
    }

    if (!EnsureCommandBuffers()) return;
    static auto* wait_fences =
        reinterpret_cast<PFN_vkWaitForFences>(g_get_device_proc(g_device, "vkWaitForFences"));
    static auto* reset_fences =
        reinterpret_cast<PFN_vkResetFences>(g_get_device_proc(g_device, "vkResetFences"));
    static auto* begin_cb =
        reinterpret_cast<PFN_vkBeginCommandBuffer>(g_get_device_proc(g_device, "vkBeginCommandBuffer"));
    static auto* end_cb =
        reinterpret_cast<PFN_vkEndCommandBuffer>(g_get_device_proc(g_device, "vkEndCommandBuffer"));
    static auto* reset_cb =
        reinterpret_cast<PFN_vkResetCommandBuffer>(g_get_device_proc(g_device, "vkResetCommandBuffer"));
    auto& slot = g_slots[g_slot];
    g_slot = (g_slot + 1) % static_cast<uint32_t>(g_slots.size());
    if (slot.pending) {
        if (const VkResult w = wait_fences(g_device, 1, &slot.fence, VK_TRUE, WatchdogNs());
            w != VK_SUCCESS) {
            // 1.51: the prepare of a few frames ago never finished: do not block the present
            // thread; frame generation off for the session (see WatchdogTrip).
            WatchdogTrip("the prepare fence wait", w);
            g_generating = false;
            return;
        }
        reset_fences(g_device, 1, &slot.fence);
        slot.pending = false;
    }
    reset_cb(slot.command, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    begin_cb(slot.command, &begin);

    const auto camera = FromCamera(inputs->view_to_clip, inputs->camera_valid);
    // Our vectors are render pixels towards the previous frame, as FSR 3.1 takes them
    // (KYTY_FSR3_MV_SIGN=-1 flips them).
    static const float mv_sign = EnvFloat("KYTY_FSR3_MV_SIGN", 1.0f) >= 0 ? 1.0f : -1.0f;
    ffxDispatchDescFrameGenerationPrepare prepare{};
    prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    prepare.frameID = g_frame_id;
    prepare.flags = config.flags;
    prepare.commandList = slot.command;
    prepare.renderSize = {inputs->render.width, inputs->render.height};
    prepare.jitterOffset = {inputs->jitter[0], inputs->jitter[1]};
    prepare.motionVectorScale = {mv_sign, mv_sign};
    prepare.frameTimeDelta = static_cast<float>(std::clamp(delta_ms, 1.0, 100.0));
    prepare.unused_reset = inputs->reset;
    const float z_near = EnvFloat("KYTY_FSR3_NEAR", std::max(inputs->z_near, 0.001f));
    // 0.92: the game's own far plane when known (as the FSR 3.1 upscaler gets it); Bloodborne's
    // projection is not infinite, and an infinite far linearised FG's depth wrongly
    // (KYTY_FSR3_FAR_INF=1: the old infinite far).
    static const bool far_inf = EnvOn("KYTY_FSR3_FAR_INF", false);
    const float z_far = !far_inf && std::isfinite(inputs->z_far) && inputs->z_far > z_near * 2.0f
                            ? inputs->z_far
                            : FLT_MAX;
    if (inputs->depth_inverted) {
        prepare.cameraNear = z_far;
        prepare.cameraFar = z_near;
    } else {
        prepare.cameraNear = z_near;
        prepare.cameraFar = z_far;
    }
    prepare.cameraFovAngleVertical =
        EnvFloat("KYTY_FSR3_FOV", inputs->fov_y > 0.0f ? inputs->fov_y : camera.fov);
    prepare.viewSpaceToMetersFactor = 1.0f;
    {
        // bb-dlssg-077: once, what FSR 3 is given, to set beside "[dlssg] constants" (DLSS-G).
        static bool said = false;
        if (!said && inputs->camera_valid) {
            said = true;
            Printf("[fsr3] inputs: frame %llu motionVectorScale %.3f %.3f (vectors in render pixels "
                   "towards the previous frame) jitter %.4f %.4f near %.6g far %.6g fov %.5g "
                   "depthInverted %d reset %d render %ux%u",
                   static_cast<unsigned long long>(g_frame_id), prepare.motionVectorScale.x,
                   prepare.motionVectorScale.y, prepare.jitterOffset.x, prepare.jitterOffset.y,
                   prepare.cameraNear, prepare.cameraFar, prepare.cameraFovAngleVertical,
                   inputs->depth_inverted ? 1 : 0, inputs->reset ? 1 : 0, inputs->render.width,
                   inputs->render.height);
        }
    }
    prepare.depth = ImageResource(inputs->depth, true);
    prepare.motionVectors = ImageResource(inputs->motion, false);
    {
        // 1.54 (bb-mvmask-154): this FidelityFX API's frame-generation prepare takes depth and
        // motion vectors only (no reactive / T&C input): the mask acts through the vectors
        // (flow-refined) and through the upscaler's reactive mask.
        static bool said = false;
        if (!said) {
            said = true;
            Printf("[mvmask] FSR 3 FG inputs: depth + full motion field (camera + skinned + flow, "
                   "render px, scale %.0f) + HUD-less colour; mask %s (no mask input in this "
                   "FidelityFX FG prepare)",
                   static_cast<double>(mv_sign), inputs->has_mask ? "built" : "not built");
        }
    }
    ++g_amd.prepared;
    if (inputs->reset) ++g_amd.input_resets;
    {
        // 1.41: frame pairing of this prepare's inputs (bb_fg.h Inputs, guest frame numbers).
        const uint64_t c = inputs->color_frame;
        const bool d_ok = inputs->depth_frame == c, m_ok = inputs->mv_frame == c,
                   o_ok = inputs->obj_frame == c,
                   h_ok = !hudless || inputs->hudless_frame == c;
        if (!d_ok) ++g_amd.off_depth;
        if (!m_ok) ++g_amd.off_mv;
        if (!o_ok) ++g_amd.off_obj;
        if (!h_ok) ++g_amd.off_hudless;
        if (c != ~0ull && d_ok && m_ok && o_ok && h_ok) ++g_amd.paired;
        g_amd.last_ids[0] = c;
        g_amd.last_ids[1] = inputs->depth_frame;
        g_amd.last_ids[2] = inputs->mv_frame;
        g_amd.last_ids[3] = inputs->obj_frame;
        g_amd.last_ids[4] = hudless ? inputs->hudless_frame : ~0ull;
        g_amd.last_ffx_frame = g_frame_id;
        g_amd.dt_sum += prepare.frameTimeDelta;
        g_amd.dt_max = std::max<double>(g_amd.dt_max, prepare.frameTimeDelta);
        if (hudless) ++g_amd.hudless_given;
    }
    if (!g_generating) ++g_amd.restarts;
    g_amd.mv_scale[0] = prepare.motionVectorScale.x;
    g_amd.mv_scale[1] = prepare.motionVectorScale.y;
    g_amd.jitter[0] = prepare.jitterOffset.x;
    g_amd.jitter[1] = prepare.jitterOffset.y;
    g_amd.near_v = prepare.cameraNear;
    g_amd.far_v = prepare.cameraFar;
    g_amd.fov = prepare.cameraFovAngleVertical;
    g_amd.depth_inverted = inputs->depth_inverted;
    ffxDispatchDescFrameGenerationPrepareCameraInfo camera_info{};
    camera_info.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
    if (camera.valid) {
        std::copy_n(camera.position, 3, camera_info.cameraPosition);
        std::copy_n(camera.up, 3, camera_info.cameraUp);
        std::copy_n(camera.right, 3, camera_info.cameraRight);
        std::copy_n(camera.forward, 3, camera_info.cameraForward);
        prepare.header.pNext = &camera_info.header;
    }
    const auto prepare_begin = std::chrono::steady_clock::now();
    if (first_steps < 3) {
        Printf("FSR3 FG: prepare dispatch (render %ux%u, command buffer %p, camera info %s, near %g "
               "far %g fov %.4f)",
               prepare.renderSize.width, prepare.renderSize.height,
               static_cast<void*>(slot.command), camera.valid ? "chained" : "none",
               prepare.cameraNear, prepare.cameraFar, prepare.cameraFovAngleVertical);
    }
    const auto result = g_ffx.Dispatch(&g_fg_context, &prepare.header);
    if (first_steps < 3) {
        Printf("FSR3 FG: prepare dispatch returned %u", result);
        ++first_steps;
    }
    end_cb(slot.command);
    if (result != FFX_API_RETURN_OK) {
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1) < 8) Printf("FSR3 FG: prepare failed (%u)", result);
        return;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.command;
    {
        std::scoped_lock queue_lock{*g_submit_mutex};
        g_queue_submit(g_game_queue, 1, &submit, slot.fence);
    }
    slot.pending = true;
    g_stats.prepare_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_begin)
            .count();
    g_stats.generated_on++;
    if (!g_generating) {
        Printf("FSR3 FG: on (3D frame: %ux%u depth and motion vectors, camera %s, HUD-less image %s)",
               inputs->render.width, inputs->render.height, camera.valid ? "found" : "default",
               hudless ? "yes" : "no");
        g_state_logged = 2;
    }
    g_generating = true;
    Display::SetExpected(g_mfg.on ? g_mfg.current - 1 : 1);
}

#else // !_WIN32

void SetUiLayer(const Image*) {}
bool WantsUiLayer() {
    return false;
}

bool Requested() {
    return false;
}
double PaceInterval() {
    return 0.0;
}
bool Generating() {
    return false;
}
void NoteSlept(double) {}
bool Active() {
    return false;
}
int Generated() {
    return 0;
}
QueuePlan PlanQueues(vk::PhysicalDevice, uint32_t) {
    return {};
}
void OnDevice(vk::PhysicalDevice, vk::Device, uint32_t, vk::Queue, const QueuePlan&, std::mutex*) {}
bool CreateSwapchain(const vk::SwapchainCreateInfoKHR&, vk::SwapchainKHR*) {
    return false;
}
void DestroySwapchain() {}
vk::Result GetSwapchainImages(vk::SwapchainKHR, uint32_t*, vk::Image*) {
    return vk::Result::eErrorOutOfDateKHR;
}
vk::Result AcquireNextImage(vk::SwapchainKHR, vk::Semaphore, uint32_t*) {
    return vk::Result::eErrorOutOfDateKHR;
}
vk::Result QueuePresent(vk::Queue, const vk::PresentInfoKHR&) {
    return vk::Result::eErrorOutOfDateKHR;
}
void BeforePresent(const Inputs*, vk::SwapchainKHR, vk::Extent2D, vk::Format) {}

#endif

} // namespace BB::FG::Amd
