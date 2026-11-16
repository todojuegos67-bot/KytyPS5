// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-dlssg-sl: DLSS Frame Generation through NVIDIA Streamline (sl.dlss_g, with Reflex and PCL),
// the way PC games use it, instead of the NGX bridge (bb_fg_dlssg.cpp). Opt-in:
// BB_FG=dlss-sl | dlss-sl-x2 | dlss-sl-x3 | dlss-sl-x4 | dlss-sl-dyn, or BB_FG=dlss-* with
// BB_DLSSG_SL=1. Needs BB_UPSCALER=dlss (Streamline is loaded by DLSS super resolution, manual
// hooking, slSetVulkanInfo after the device; bb_sr_dlss.cpp then loads dlss, dlssg, reflex, pcl).
//
// - Swapchain: vkCreateSwapchainKHR / Destroy / GetSwapchainImages / AcquireNextImage /
//   QueuePresent / DeviceWaitIdle and the Win32 surface go through sl.interposer.dll's proxies
//   (its vkGetDeviceProcAddr / vkGetInstanceProcAddr, ProgrammingGuideManualHooking.md 4.2), so
//   sl.dlss_g attaches to our swapchain and generates + presents its frames inside the present.
// - Per game frame (GPU thread, TagFrame): DLSS super resolution's frame token (one frame index
//   stream, NextFrameIndex / LastFrameIndex), PCL markers SimulationStart / End and
//   RenderSubmitStart, tags depth, motion vectors (render size) and the HUD-less colour (output
//   size), copied in the frame's command buffer (eOnlyValidNow), slDLSSGSetOptions when they
//   change. Constants: DLSS SR's for the same token; set here only when SR did not run.
// - Present thread: RenderSubmitEnd, PresentStart / PresentEnd around the game frame's present.
// - Reflex: slReflexSetOptions(eLowLatency) once (DLSS-G needs Reflex); no slReflexSleep and no
//   pacing of ours: Streamline / the driver pace the generated frames.
// - Log: one [dlssg-sl] line per second (slDLSSGGetState: status, frames presented, ...).
// Not used, and nothing here runs, unless selected.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32) && defined(BB_STREAMLINE)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_helpers_vk.h>
// sl_pcl.h / sl_reflex.h are not included: sl_pcl.h has `using to_underlying =
// std::to_underlying;` under `#if __cplusplus == 202302L`, which does not compile in C++23. The
// few declarations used here, verbatim from the v2.14.16 headers (ABI: enum values, the
// ReflexOptions layout and StructType GUID, the function types):
namespace sl {
enum class PCLMarker : uint32_t {
    eSimulationStart = 0,
    eSimulationEnd = 1,
    eRenderSubmitStart = 2,
    eRenderSubmitEnd = 3,
    ePresentStart = 4,
    ePresentEnd = 5,
};
enum ReflexMode { eOff, eLowLatency, eLowLatencyWithBoost, ReflexMode_eCount };
// {F03AF81A-6D0B-4902-A651-C4965E215434}
SL_STRUCT_BEGIN(ReflexOptions, StructType({ 0xf03af81a, 0x6d0b, 0x4902, { 0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34 } }), kStructVersion1)
    ReflexMode mode = ReflexMode::eOff;
    uint32_t frameLimitUs = 0;
    bool useMarkersToOptimize = false;
    uint16_t virtualKey = 0;
    uint32_t idThread = 0;
SL_STRUCT_END()
} // namespace sl
using PFun_slPCLSetMarker = sl::Result(sl::PCLMarker marker, const sl::FrameToken& frame);
using PFun_slReflexSetOptions = sl::Result(const sl::ReflexOptions& options);
#endif

namespace BB::FG::DlssSl {

bool Requested() {
    return Cfg().kind == Kind::Dlss && Cfg().streamline;
}

void Queues(uint32_t* graphics, uint32_t* compute) {
    if (graphics) *graphics = 0;
    if (compute) *compute = 0;
    if (Requested()) SR::Dlss::RequiredQueues(graphics, compute);
}

#if defined(_WIN32) && defined(BB_STREAMLINE)
namespace {

struct Api {
    PFun_slGetNewFrameToken* get_new_frame_token = nullptr;
    PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
    PFun_slSetConstants* set_constants = nullptr;
    PFun_slDLSSGSetOptions* dlssg_set_options = nullptr;
    PFun_slDLSSGGetState* dlssg_get_state = nullptr;
    PFun_slReflexSetOptions* reflex_set_options = nullptr;
    PFun_slPCLSetMarker* pcl_set_marker = nullptr;
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkVoidFunction create_win32_surface = nullptr;
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
};
Api g{};
bool g_hooked = false, g_ready = false;
const sl::ViewportHandle kViewport{1}; // DLSS super resolution's viewport

std::atomic<uint32_t> g_last_used{0};
std::atomic<uint32_t> g_sec_tagged{0}, g_sec_off{0}, g_sec_own_constants{0}, g_sec_errors{0},
    g_sec_presents{0};
std::atomic<uint32_t> g_max_gen{0};

template <typename F>
bool Feature(sl::Feature feature, const char* name, F*& out) {
    out = reinterpret_cast<F*>(SR::Dlss::FeatureProc(static_cast<uint32_t>(feature), name));
    if (out == nullptr) BBLogf("DLSS-G (Streamline): no %s", name);
    return out != nullptr;
}

sl::FrameToken* Token(uint32_t frame) {
    sl::FrameToken* token = nullptr;
    if (frame == 0 || g.get_new_frame_token(token, &frame) != sl::Result::eOk) return nullptr;
    return token;
}

void Marker(sl::PCLMarker marker, uint32_t frame) {
    if (g.pcl_set_marker == nullptr) return;
    sl::FrameToken* token = Token(frame);
    if (token == nullptr || g.pcl_set_marker(marker, *token) != sl::Result::eOk) {
        g_sec_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

sl::Resource ToResource(const Image& image, vk::ImageLayout layout) {
    sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
                   static_cast<VkImageView>(image.view),
                   static_cast<uint32_t>(static_cast<VkImageLayout>(layout)));
    r.width = image.extent.width;
    r.height = image.extent.height;
    r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
    return r;
}

// Row major for column vectors (ours) -> Streamline's row-vector matrices (transposed), as in
// bb_sr_dlss.cpp.
sl::float4x4 Transposed(const float* m) {
    sl::float4x4 r;
    for (uint32_t i = 0; i < 4; ++i) {
        r[i] = sl::float4(m[0 * 4 + i], m[1 * 4 + i], m[2 * 4 + i], m[3 * 4 + i]);
    }
    return r;
}

bool Inverted(const float* m, float* out) {
    double a[4][8]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        }
        if (std::fabs(a[pivot][c]) < 1e-12) return false;
        for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double s = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] *= s;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][4 + j]);
    }
    return true;
}

// Only when DLSS super resolution did not set this frame's constants.
bool SetOwnConstants(const Inputs& in, const sl::FrameToken& token) {
    static const float identity[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float inv[16]{}, prev_inv[16]{};
    sl::Constants c{};
    c.cameraViewToClip = Transposed(in.camera_valid ? in.view_to_clip : identity);
    c.clipToCameraView = in.camera_valid && Inverted(in.view_to_clip, inv) ? Transposed(inv)
                                                                          : Transposed(identity);
    c.clipToLensClip = Transposed(identity);
    c.clipToPrevClip = Transposed(in.camera_valid ? in.clip_to_prev_clip : identity);
    c.prevClipToClip = in.camera_valid && Inverted(in.clip_to_prev_clip, prev_inv)
                           ? Transposed(prev_inv)
                           : Transposed(identity);
    c.jitterOffset = sl::float2(in.jitter[0], in.jitter[1]);
    c.mvecScale = sl::float2(1.0f / static_cast<float>(std::max(1u, in.render.width)),
                             1.0f / static_cast<float>(std::max(1u, in.render.height)));
    c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    c.cameraPos = sl::float3(0, 0, 0);
    c.cameraUp = sl::float3(0, 1, 0);
    c.cameraRight = sl::float3(1, 0, 0);
    c.cameraFwd = sl::float3(0, 0, 1);
    c.cameraNear = in.z_near;
    c.cameraFar = in.z_far;
    c.cameraFOV = in.fov_y;
    c.cameraAspectRatio = static_cast<float>(in.render.width) /
                          static_cast<float>(std::max(1u, in.render.height));
    c.motionVectorsInvalidValue = sl::INVALID_FLOAT;
    c.depthInverted = in.depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = in.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    return g.set_constants(c, token, kViewport) == sl::Result::eOk;
}

struct OptionsKey {
    bool on = false;
    uint32_t frames = 0, mw = 0, mh = 0, cw = 0, ch = 0;
    bool operator==(const OptionsKey&) const = default;
};
OptionsKey g_options{};
bool g_options_set = false;

// One log line when the reason for staying off (or for being on) changes.
void Reason(uint32_t reason) {
    static std::atomic<uint32_t> last{~0u};
    if (last.exchange(reason, std::memory_order_relaxed) == reason) return;
    static constexpr const char* kText[] = {
        "on (frames tagged)",
        "off: no 3D frame captured for this game frame (2D, HDR or inputs not ready)",
        "off: own constants failed",
        "off: slSetTagForFrame failed",
    };
    BBLogf("[dlssg-sl] %s", kText[reason]);
}

void SetOptions(const OptionsKey& key) {
    if (g_options_set && key == g_options) return;
    sl::DLSSGOptions o{};
    o.mode = key.on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    o.numFramesToGenerate = std::max(1u, key.frames);
    o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    o.mvecDepthWidth = key.mw;
    o.mvecDepthHeight = key.mh;
    o.colorWidth = key.cw;
    o.colorHeight = key.ch;
    const auto r = g.dlssg_set_options(kViewport, o);
    BBLogf("DLSS-G (Streamline): slDLSSGSetOptions(mode %s, frames %u, mvec/depth %ux%u, colour "
           "%ux%u) returned %d",
           key.on ? "on" : "off", o.numFramesToGenerate, key.mw, key.mh, key.cw, key.ch,
           static_cast<int>(r));
    if (r == sl::Result::eOk) {
        g_options = key;
        g_options_set = true;
    } else {
        g_sec_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

uint32_t WantedFrames() {
    const auto& c = Cfg();
    uint32_t want = c.dynamic ? 3u : static_cast<uint32_t>(std::clamp(c.multiplier - 1, 1, 3));
    // numFramesToGenerateMax from the once-a-second slDLSSGGetState (0: not known yet).
    const uint32_t max = g_max_gen.load(std::memory_order_relaxed);
    return max > 0 ? std::min(want, max) : want;
}

} // namespace

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!Requested()) return;
    if (!SR::Dlss::Loaded() || !SR::Dlss::Manual()) {
        BBLogf("DLSS-G (Streamline): needs BB_UPSCALER=dlss with Streamline in manual hooking "
               "(loaded %d, manual %d): DLSS-G off",
               SR::Dlss::Loaded() ? 1 : 0, SR::Dlss::Manual() ? 1 : 0);
        return;
    }
    if (!SR::Dlss::Ready()) {
        BBLogf("DLSS-G (Streamline): Streamline did not take the device: DLSS-G off");
        return;
    }
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(SR::Dlss::Proc("vkGetDeviceProcAddr"));
    const auto gipa =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(SR::Dlss::Proc("vkGetInstanceProcAddr"));
    if (gdpa == nullptr || gipa == nullptr) {
        BBLogf("DLSS-G (Streamline): sl.interposer.dll has no vkGetDeviceProcAddr / "
               "vkGetInstanceProcAddr: DLSS-G off");
        return;
    }
    const VkDevice d = static_cast<VkDevice>(device);
    const VkInstance i = static_cast<VkInstance>(instance);
    g.create_swapchain = reinterpret_cast<PFN_vkCreateSwapchainKHR>(gdpa(d, "vkCreateSwapchainKHR"));
    g.destroy_swapchain =
        reinterpret_cast<PFN_vkDestroySwapchainKHR>(gdpa(d, "vkDestroySwapchainKHR"));
    g.get_swapchain_images =
        reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(gdpa(d, "vkGetSwapchainImagesKHR"));
    g.acquire_next_image =
        reinterpret_cast<PFN_vkAcquireNextImageKHR>(gdpa(d, "vkAcquireNextImageKHR"));
    g.queue_present = reinterpret_cast<PFN_vkQueuePresentKHR>(gdpa(d, "vkQueuePresentKHR"));
    g.device_wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(gdpa(d, "vkDeviceWaitIdle"));
    g.create_win32_surface = gipa(i, "vkCreateWin32SurfaceKHR");
    g.destroy_surface = reinterpret_cast<PFN_vkDestroySurfaceKHR>(gipa(i, "vkDestroySurfaceKHR"));
    g_hooked = g.create_swapchain && g.destroy_swapchain && g.get_swapchain_images &&
               g.acquire_next_image && g.queue_present && g.device_wait_idle &&
               g.create_win32_surface && g.destroy_surface;
    if (!g_hooked) {
        BBLogf("DLSS-G (Streamline): a swapchain proxy is missing: DLSS-G off");
        return;
    }
    g.get_new_frame_token =
        reinterpret_cast<PFun_slGetNewFrameToken*>(SR::Dlss::Proc("slGetNewFrameToken"));
    g.set_tag_for_frame = reinterpret_cast<PFun_slSetTagForFrame*>(SR::Dlss::Proc("slSetTagForFrame"));
    g.set_constants = reinterpret_cast<PFun_slSetConstants*>(SR::Dlss::Proc("slSetConstants"));
    const bool dlssg = SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeatureDLSS_G), physical);
    const bool reflex = SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeatureReflex), physical);
    const bool pcl = SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeaturePCL), physical);
    BBLogf("DLSS-G (Streamline): features supported and loaded: DLSS-G %d, Reflex %d, PCL %d", dlssg,
           reflex, pcl);
    const bool functions =
        g.get_new_frame_token && g.set_tag_for_frame && g.set_constants && dlssg &&
        Feature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g.dlssg_set_options) &&
        Feature(sl::kFeatureDLSS_G, "slDLSSGGetState", g.dlssg_get_state);
    if (reflex) Feature(sl::kFeatureReflex, "slReflexSetOptions", g.reflex_set_options);
    if (pcl) Feature(sl::kFeaturePCL, "slPCLSetMarker", g.pcl_set_marker);
    if (!functions) {
        // The swapchain still goes through the proxies (harmless without DLSS-G options).
        BBLogf("DLSS-G (Streamline): DLSS-G not available: off (swapchain through the proxies)");
        return;
    }
    if (g.reflex_set_options != nullptr) {
        sl::ReflexOptions o{};
        o.mode = sl::ReflexMode::eLowLatency;
        o.useMarkersToOptimize = false;
        const auto r = g.reflex_set_options(o);
        BBLogf("DLSS-G (Streamline): slReflexSetOptions(low latency) returned %d",
               static_cast<int>(r));
    } else {
        BBLogf("DLSS-G (Streamline): no Reflex: DLSS-G may report eFailReflexNotDetectedAtRuntime");
    }
    g_ready = true;
    BBLogf("DLSS-G (Streamline): ready (%s, swapchain through sl.interposer.dll's proxies, no "
           "pacing of ours)",
           Name());
}

bool Hooked() {
    return g_hooked;
}

bool Ready() {
    return g_ready;
}

void* SurfaceProc(vk::Instance) {
    return g_hooked ? reinterpret_cast<void*>(g.create_win32_surface) : nullptr;
}

vk::Result CreateSwapchain(vk::Device device, const vk::SwapchainCreateInfoKHR& info,
                           vk::SwapchainKHR* out) {
    VkSwapchainKHR chain = VK_NULL_HANDLE;
    const VkSwapchainCreateInfoKHR& ci = info;
    const auto r = static_cast<vk::Result>(
        g.create_swapchain(static_cast<VkDevice>(device), &ci, nullptr, &chain));
    *out = chain;
    BBLogf("DLSS-G (Streamline): swapchain %ux%u through Streamline: %d", info.imageExtent.width,
           info.imageExtent.height, static_cast<int>(r));
    g_options_set = false; // set again for the new swapchain
    return r;
}

void DestroySwapchain(vk::Device device, vk::SwapchainKHR swapchain) {
    g.destroy_swapchain(static_cast<VkDevice>(device), static_cast<VkSwapchainKHR>(swapchain),
                        nullptr);
}

vk::Result GetSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain, uint32_t* count,
                              vk::Image* images) {
    return static_cast<vk::Result>(g.get_swapchain_images(
        static_cast<VkDevice>(device), static_cast<VkSwapchainKHR>(swapchain), count,
        reinterpret_cast<VkImage*>(images)));
}

vk::Result AcquireNextImage(vk::Device device, vk::SwapchainKHR swapchain, uint64_t timeout,
                            vk::Semaphore semaphore, uint32_t* index) {
    return static_cast<vk::Result>(g.acquire_next_image(
        static_cast<VkDevice>(device), static_cast<VkSwapchainKHR>(swapchain), timeout,
        static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE, index));
}

vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info) {
    const VkPresentInfoKHR& pi = info;
    return static_cast<vk::Result>(g.queue_present(static_cast<VkQueue>(queue), &pi));
}

vk::Result DeviceWaitIdle(vk::Device device) {
    return static_cast<vk::Result>(g.device_wait_idle(static_cast<VkDevice>(device)));
}

void DestroySurface(vk::Instance instance, vk::SurfaceKHR surface) {
    g.destroy_surface(static_cast<VkInstance>(instance), static_cast<VkSurfaceKHR>(surface),
                      nullptr);
}

uint32_t TagFrame(vk::CommandBuffer cmd, const Inputs* inputs, vk::Extent2D backbuffer,
                  vk::Format format) {
    if (!g_ready) return 0;
    // DLSS super resolution's token when it evaluated this frame (its constants are set),
    // else the next index of the same stream and our own constants.
    uint32_t frame = SR::Dlss::LastFrameIndex();
    bool own = false;
    if (frame == 0 || frame == g_last_used.load(std::memory_order_relaxed)) {
        frame = SR::Dlss::NextFrameIndex();
        own = true;
    }
    g_last_used.store(frame, std::memory_order_relaxed);
    sl::FrameToken* token = Token(frame);
    if (token == nullptr) {
        g_sec_errors.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    Marker(sl::PCLMarker::eSimulationStart, frame);
    Marker(sl::PCLMarker::eSimulationEnd, frame);
    Marker(sl::PCLMarker::eRenderSubmitStart, frame);

    const bool valid = inputs != nullptr && inputs->valid && inputs->render.width > 0 &&
                       inputs->render.height > 0;
    if (!valid) {
        SetOptions(OptionsKey{false, g_options.frames, g_options.mw, g_options.mh, g_options.cw,
                              g_options.ch});
        g_sec_off.fetch_add(1, std::memory_order_relaxed);
        Reason(1);
        return frame;
    }
    if (own) {
        if (!SetOwnConstants(*inputs, *token)) {
            g_sec_errors.fetch_add(1, std::memory_order_relaxed);
            Reason(2);
            return frame;
        }
        g_sec_own_constants.fetch_add(1, std::memory_order_relaxed);
    }
    auto depth = ToResource(inputs->depth, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto motion = ToResource(inputs->motion, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto hudless = ToResource(inputs->hudless, vk::ImageLayout::eShaderReadOnlyOptimal);
    const sl::Extent render{0, 0, inputs->render.width, inputs->render.height};
    const sl::Extent full{0, 0, backbuffer.width, backbuffer.height};
    sl::ResourceTag tags[3]{
        sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow,
                        &render),
        sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors,
                        sl::ResourceLifecycle::eOnlyValidNow, &render),
        sl::ResourceTag(&hudless, sl::kBufferTypeHUDLessColor,
                        sl::ResourceLifecycle::eOnlyValidNow, &full),
    };
    const uint32_t count = inputs->has_hudless ? 3u : 2u;
    sl::CommandBuffer* sl_cmd =
        reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(cmd));
    if (const auto r = g.set_tag_for_frame(*token, kViewport, tags, count, sl_cmd);
        r != sl::Result::eOk) {
        static uint32_t said = 0;
        if (said++ < 8) BBLogf("DLSS-G (Streamline): slSetTagForFrame returned %d", static_cast<int>(r));
        g_sec_errors.fetch_add(1, std::memory_order_relaxed);
        Reason(3);
        return frame;
    }
    SetOptions(OptionsKey{true, WantedFrames(), inputs->render.width, inputs->render.height,
                          backbuffer.width, backbuffer.height});
    g_sec_tagged.fetch_add(1, std::memory_order_relaxed);
    Reason(0);
    static bool said = false;
    if (!said) {
        said = true;
        BBLogf("DLSS-G (Streamline): first tagged frame %u: depth fmt %d %ux%u, motion fmt %d, "
               "HUD-less %s, backbuffer %ux%u fmt %d, constants %s",
               frame, static_cast<int>(inputs->depth.format), inputs->render.width,
               inputs->render.height, static_cast<int>(inputs->motion.format),
               inputs->has_hudless ? "yes" : "no", backbuffer.width, backbuffer.height,
               static_cast<int>(format), own ? "ours" : "DLSS SR's");
    }
    return frame;
}

void BeforePresent(uint32_t frame) {
    if (!g_ready || frame == 0) return;
    Marker(sl::PCLMarker::eRenderSubmitEnd, frame);
    Marker(sl::PCLMarker::ePresentStart, frame);
}

void AfterPresent(uint32_t frame, bool presented) {
    if (!g_ready) return;
    if (frame != 0) {
        Marker(sl::PCLMarker::ePresentEnd, frame);
        if (presented) g_sec_presents.fetch_add(1, std::memory_order_relaxed);
    }
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(1)) return;
    last = now;
    sl::DLSSGState s{};
    const auto r = g.dlssg_get_state(kViewport, s, nullptr);
    if (r == sl::Result::eOk) g_max_gen.store(s.numFramesToGenerateMax, std::memory_order_relaxed);
    BBLogf("[dlssg-sl] state %d status 0x%x presented/frame %u max-gen %u min-size %u vram %.0f MB "
           "| last s: game frames %u tagged %u off %u own-constants %u errors %u | options %s x%u",
           static_cast<int>(r), static_cast<uint32_t>(s.status), s.numFramesActuallyPresented,
           s.numFramesToGenerateMax, s.minWidthOrHeight,
           static_cast<double>(s.estimatedVRAMUsageInBytes) / (1024.0 * 1024.0),
           g_sec_presents.exchange(0), g_sec_tagged.exchange(0), g_sec_off.exchange(0),
           g_sec_own_constants.exchange(0), g_sec_errors.exchange(0),
           g_options.on ? "on" : "off", g_options.frames + 1);
}

#else // no Streamline SDK in this build

void OnDevice(vk::Instance, vk::PhysicalDevice, vk::Device) {
    if (Requested()) BBLogf("DLSS-G (Streamline): not in this build (no Streamline SDK): off");
}
bool Hooked() {
    return false;
}
bool Ready() {
    return false;
}
void* SurfaceProc(vk::Instance) {
    return nullptr;
}
vk::Result CreateSwapchain(vk::Device, const vk::SwapchainCreateInfoKHR&, vk::SwapchainKHR*) {
    return vk::Result::eErrorFeatureNotPresent;
}
void DestroySwapchain(vk::Device, vk::SwapchainKHR) {}
vk::Result GetSwapchainImages(vk::Device, vk::SwapchainKHR, uint32_t*, vk::Image*) {
    return vk::Result::eErrorFeatureNotPresent;
}
vk::Result AcquireNextImage(vk::Device, vk::SwapchainKHR, uint64_t, vk::Semaphore, uint32_t*) {
    return vk::Result::eErrorFeatureNotPresent;
}
vk::Result QueuePresent(vk::Queue, const vk::PresentInfoKHR&) {
    return vk::Result::eErrorFeatureNotPresent;
}
vk::Result DeviceWaitIdle(vk::Device) {
    return vk::Result::eErrorFeatureNotPresent;
}
void DestroySurface(vk::Instance, vk::SurfaceKHR) {}
uint32_t TagFrame(vk::CommandBuffer, const Inputs*, vk::Extent2D, vk::Format) {
    return 0;
}
void BeforePresent(uint32_t) {}
void AfterPresent(uint32_t, bool) {}

#endif

} // namespace BB::FG::DlssSl
