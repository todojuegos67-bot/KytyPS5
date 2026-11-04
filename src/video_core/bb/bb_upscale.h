// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne upscaling (phase 1): DLSS / DLAA (Streamline), XeSS, FSR 3.1 and FSR 4 on the game's
// 3D image before the HUD, with the HUD left to the game at the output size.
//
// How the frame is read (all from the guest's own passes, nothing title specific is hard coded;
// BB_UPSCALE_* overrides exist for every guess):
//  - scene depth: the depth target of the frame's G-buffer pass (the pass with the most colour
//    targets, at least 3, and a depth target). Bloodborne: D32S8, 6-7 MRTs.
//  - scene colour: the last HDR colour target (RGBA16F / B10G11R11) drawn with that depth bound,
//    at its size: the lit, forward-shaded 3D image, before post-processing and the HUD.
//  - injection: when the game first samples the final scene colour (after its last write that
//    frame) the upscaler runs and the sample is redirected to the upscaled image ("scene" mode,
//    internal size == output size: DLAA / native AA). When the 3D renders below the output size the
//    injection moves to the first draw at the output size that samples a render-size image of the
//    scene chain ("composite" mode, LDR), so post-processing stays at the size the game made it for.
//    Everything the game draws after that (its HUD, into the output / flip images) never passes
//    through the upscaler.
//  - motion vectors: Bloodborne has no TAA and no motion vectors; camera motion vectors are made
//    from the scene depth and the view-projection found in the G-buffer draws' vertex constant
//    buffers (this frame's and the previous one's), BB_MV=zero turns them off.
//  - jitter: the scene passes' viewport is offset by a Halton(2,3) sub-pixel sequence while an
//    upscaler runs with reconstructed vectors (BB_UPSCALE_JITTER=0 turns it off).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "common/types.h"
#include "video_core/bb/bb_sr.h"

namespace BB::Upscale {

struct Config {
    SR::Kind kind = SR::Kind::Off;
    SR::Quality quality = SR::Quality::Quality;
    float sharpness = 0.3f;
    bool jitter = true;
    bool mv_reconstruct = true;
    bool hudless_copy = false;
    int force_mode = 0; // 0 auto, 1 scene, 2 composite
    bool to_window = true; // BB_UPSCALE_TO_WINDOW=0: never upscale past the game's own output size
    bool menus = true;     // BB_UPSCALE_MENUS=0: 2D frames (menus, loading, movies) not scaled
    VAddr force_source = 0;
    u32 log_every = 600;
};
const Config& Cfg();
bool Enabled(); // an upscaler (or native) was asked for
// Cfg().sharpness, then the value in BB_UPSCALE_SHARPNESS_FILE (re-read every 0.5 s when it changes).
float LiveSharpness();
// BB_SCENE_UPSCALE=1: the 3D scene's render targets (every colour / depth target of the scene
// passes, learned from one gameplay frame) are created at the internal size; post-processing and
// the HUD stay at the output size, and the upscaler runs in place before post-processing.
// True (and the host size in w, h) when this guest image is one of them at its learned size.
bool ScaleImage(VAddr addr, u32& w, u32& h);
// The guest and host sizes of the scaled scene targets (0 until learned).
bool SceneScale(u32& guest_w, u32& guest_h, u32& host_w, u32& host_h);
// A target of the scaled targets' guest size bound in the same draw as scaled ones (mixed sizes
// in one render pass): scaled from its next use.
void AddSceneTarget(VAddr addr);

// Instance / device creation.
void PreInstance();
const wchar_t* VulkanLoaderOverride(); // Streamline's interposer for DLSS
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions);
void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next);
void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family);
bool BackendReady();
// Records the selected back end; any failure (an error or a crash inside it) turns it off for the
// rest of the session, so the game runs native instead of closing.
bool RunBackend(const SR::Desc& desc);
// Runs fn(arg) under a crash guard (Windows SEH): false on a crash, with the exception code,
// address and faulting module in `fault`.
bool GuardedCall(void (*fn)(void*), void* arg, std::string& fault);
// "<pointer> (<module>+<offset>)" for a function pointer, for the init log.
std::string Where(const void* p);
// DLSS: before the real vkCreateDevice, tries the same create info straight on the system Vulkan
// loader (no Streamline), then with the extensions Streamline adds, then without each of them,
// and logs every result: which addition the driver refuses.
void DryRunDevice(VkInstance instance, VkPhysicalDevice physical, const VkDeviceCreateInfo* info);
// A start-up checkpoint: written and flushed at once to bb_upscale_init.log (next to the working
// directory's shadPS4 log is asynchronous and may be cut off by a crash) and to the log.
void Checkpoint(std::string_view text);

// ---- Frame tracking (GPU command processor thread) ----
struct Target {
    VAddr addr = 0;
    u32 width = 0, height = 0;
    vk::Format format = vk::Format::eUndefined;
};
struct Viewport {
    float xscale = 1, xoffset = 0, yscale = 1, yoffset = 0, zscale = 1, zoffset = 0;
};
// Every draw, once its targets are known.
void OnDraw(std::span<const Target> colors, const Target* depth, bool depth_greater,
            const Viewport& viewport, u64 vs_hash = 0, u64 ps_hash = 0);
// BB_RT_REFLECT=1: the G-buffer pass's targets and shaders logged. BB_RT_REFLECT=2 also adds the
// screen-space reflection pass (bb_rtrefl.comp) to the scene colour before the upscaler.
bool RtReflect();
int RtReflectLevel();
// Every dispatch.
void OnDispatch();
// The current draw is a scene draw whose vertex constant buffers should be scanned for the camera.
// vertex: the vertex stage (its forward view-projection marks a 3D geometry draw to jitter).
bool WantCameraScan(bool vertex);
void OnVertexConstants(const void* data, size_t size);
// A sampled colour image in the current draw/dispatch.
enum class Sample { Keep, Inject, Redirect };
Sample OnSample(VAddr addr, u32 width, u32 height, vk::Format format);
// The current draw is a scene draw to jitter: the offset in pixels.
bool Jitter(float& x, float& y);
// The guest flip (end of the frame).
void OnFlip(VAddr addr, u32 width, u32 height);
// The current draw renders into the presented image (HUD stage): true once per frame, before its
// first HUD draw, when the HUD-less copy should be taken.
bool WantHudlessCopy();

// ---- What an injection needs (filled by OnSample(...) == Inject) ----
struct Plan {
    bool scene_mode = true;
    VAddr color_addr = 0;
    VAddr depth_addr = 0;
    u32 render_w = 0, render_h = 0;
    u32 output_w = 0, output_h = 0;
    bool reset = false;
    bool hdr = true;
    bool depth_inverted = true;
    float jitter[2]{};
    float delta_ms = 16.7f;
    bool camera_valid = false;
    float view_to_clip[16]{};
    float clip_to_prev_clip[16]{};
    bool reflect_valid = false;
    float world_to_pix[16]{}; // the view-projection with the scene viewport and depth mapping
    float pix_to_world[16]{}; // its inverse, from (pixel x, pixel y, buffer depth, 1)
    float fov_y = 1.0f, z_near = 0.1f, z_far = 1000.0f;
    Viewport viewport;
};
const Plan& CurrentPlan();
// The formats of the upscale just recorded ("<input> (<colour space>) -> <output>"), for the log.
void SetUpscaleFormats(std::string text);
void OnInjected(bool ok, float mv_ms_unused);
void OnGpuTime(double ms);
// Per-stage GPU ms of the window-size path (copy, MV + upscaler, alpha, HUD composite, present).
constexpr u32 kStages = 5;
void OnStageTimes(const double (&ms)[kStages]);
void OnHudlessCopied();

// ---- Upscaling to the window size ----
// When the game renders its whole frame (3D, post-processing and HUD) at one size below the
// presentation size (the frame the presenter scales the flip image into: the window), the upscaler
// runs on the post-processed image just before the first HUD draw (an LDR copy of the presented
// image), into an image at the presentation size; the presenter then adds the HUD on top (the
// difference between the final presented image and that copy) and presents it. Any failure turns
// it off for the session and the in-place path runs as before.
void SetPresentSize(u32 width, u32 height); // the presenter's frame size, every flip
// The current draw renders into the presented image: true once per frame, at its first HUD draw
// (it samples no render-size image), when the upscale to the window size should run before it;
// CurrentPlan() then describes it and `flip` is the presented image's address.
bool WantPresentUpscale(VAddr& flip);
void OnPresentUpscaled(bool ok);
// 0.39 (BB_FG_NATIVE, default on): frame generation asked with no upscaler. The module runs its
// native mode: the window-size path's plan (render size = output size) only feeds frame
// generation its depth, motion vectors and camera; no upscaler, the presented image untouched.
bool FgNative();
void OnFgNativeAnalyzed(bool ok); // instead of OnPresentUpscaled in that mode
// No HUD point this frame although the scene was drawn: true once (at the flip), and the presenter
// then runs the window-size upscale on the finished image (CurrentPlan()).
bool TakeLateUpscale();
// The frame just flipped went through the upscaler (in place or to the window size).
bool LastFrameUpscaled();
void OnPresentFailed(std::string_view why);
// The flip being presented had no 3D scene (a menu, loading screen or movie) and an upscaler is
// active: the presenter then scales it to the window with the spatial FSR 1 pass (EASU) instead of
// a plain stretch, since the upscaler itself needs a 3D scene (depth, motion). BB_UPSCALE_MENUS=0
// turns it off. Only ever true for a frame the 3D path did not upscale.
bool MenuUpscaleWanted();
// A 3D frame in window-size mode that was not upscaled: FSR 1 to the window, not a stretch.
bool SceneSpatialFallback();
// 0.48: the flip being presented had no 3D scene (menu, loading screen, movie).
bool FlipIs2D();
// 0.48: the RCAS sharpening (attenuation in stops, 0 = sharpest) for a 2D frame scaled to the
// window by FSR 1: BB_2D_RCAS (default 0.15, text-friendly); negative = the emulator's settings.
float Menu2DRcas();
// 0.48: the presenter's outcome for a 2D flip, counted and logged every 60 2D flips as
// "[2d] last 60 flips: N FSR1, M plain stretch (reasons), K window size".
// kind: 0 FSR 1 (EASU + RCAS), 1 FSR 1 EASU only, 2 already the window size, 3 plain stretch.
void Note2DFlip(int kind, const char* reason);
// Why a 2D flip would not get FSR 1 (for the [2d] line): "" when it would.
const char* Menu2DWhyNot();
// The presenter's verdict for this frame: the window-size image was (or was not) presented.
void OnPresentResult(bool presented, std::string_view why);

} // namespace BB::Upscale
