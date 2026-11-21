#ifndef EMULATOR_SRC_LOCAL_PATHTRACE_H_
#define EMULATOR_SRC_LOCAL_PATHTRACE_H_
// (pt1) BB_PATHTRACING=1: optional ray-traced sun shadows and reflections for Demon's Souls (PPSA01341 01.007.000),
// injected into the game's own intermediate images BEFORE its lighting resolves, so its tonemap, bloom, TAA, the
// upscaler, frame generation and the HUD run unchanged on top. Off by default: unset (or 0), nothing here runs, no
// Vulkan extension is asked for, no image is made, no hook does more than one static load; F9 keeps its old use.
//
// Method (after the GPL-2.0 Bloodborne path-tracing mod by João Vitor Alves, reimplemented for this renderer - no code of
// it is copied, see docs in INFORME.md):
//  1. Geometry: the G-buffer draws' own clip-space output captured by a patched vertex shader (hwrt-scene.cpp, already
//     in this tree), now only inside the game's GBuffer / VirtualGeo profiler scopes (the frame map, src/native),
//     world space through the camera's inverse view-projection; a BLAS per mesh kept across frames (static), moving
//     meshes refit (every frame here), one TLAS a frame built at the END of the GBuffer scope (same frame as the
//     lighting that uses it: no one-frame lag).
//  2. Light and camera: the camera from the G-buffer vertex shaders' SRT (words 29 view-projection, 45 previous); the
//     depth's distance from the PS5 Pro packet's view_to_clip (the depth is not the camera's clip z: measured, hwrt-gi);
//     the sun from the orthographic light matrix of the draws inside the game's "RenderShadowCascade N" scopes (the
//     matrix's depth row; above the horizon, world +y up).
//  3. Passes (pt.comp, compute + ray query): sun shadow (1 ray in the sun cone), temporal (camera reprojection +
//     disocclusion by distance; short history for moving occluders), spatial (edge-aware), compose (min with the game's
//     sun-shadow channel); reflections (GGX sample around the GEOMETRIC normal for surfaces under the roughness cutoff,
//     hit radiance = the game's lit colour of this frame where the hit is on screen), temporal (virtual-image
//     reprojection, neighbourhood clamp), spatial, compose into the game's SSR result (its IBL pass then applies the
//     game's own material response).
//  4. Composition targets found at run time by scope (logged once): the image the ResolveSunShadows dispatches write,
//     the image the RenderSSR dispatches write, the lit colour ResolveLighting writes, the GBuffer's depth and array.
//     Nothing is written to a game image until its calibration from this frame's statistics has passed (which channel
//     of the shadow image is the sun's; the SSR colour scale and alpha convention) and the depth <-> TLAS self-test
//     agrees: before that, or if any check fails, the passes only measure and the picture is the game's.
//
// Switches (read once at start; BB_PATHTRACING_QUALITY and F9 also live):
//   BB_PATHTRACING=1                      master switch (needs a device with ray queries; the log says why if not)
//   BB_PATHTRACING_QUALITY=low|medium|high|ultra (also bajo|medio|alto, 1..4; default low)
//   BB_PT_SHADOWS=0 / BB_PT_REFLECTIONS=0  one part off
//   BB_PT_SHADOW_SCALE, BB_PT_REFL_SCALE  trace resolution (fraction of the render size per axis)
//   BB_PT_ROUGH_CUTOFF, BB_PT_SHADOW_DIST, BB_PT_REFL_DIST, BB_PT_REFL_SPP, BB_PT_SHADOW_HISTORY, BB_PT_REFL_HISTORY
//   BB_PT_SUN_ANGLE (degrees, sun cone half-angle), BB_PT_SHADOW_STRENGTH (0..1), BB_PT_SHADOW_MODE=min|replace
//   BB_PT_SHADOW_CHANNEL=0..3 (skip the calibration), BB_PT_REFL_GAIN=<scale> (skip the colour calibration)
//   BB_PT_ROUGH=<layer>.<byte> (G-buffer roughness: default 2.0, RT5's calibration), BB_PT_CAPTURE_SCOPES=GBuffer,...
//   BB_PT_DEBUG=1 (pt1c) tint: red where our traced sun shadow is occluded, blue where a traced reflection hit (in the
//                 lit HDR image the TAA reads, before the post chain); 2 our shadow alone in its channel; 3 our reflection alone
//   BB_PT_CYCLE=1  rotates bajo -> medio -> alto -> ultra -> off every 5 s window (A/B of all presets in one session)
//   BB_PT_FORCE=1  compose before the calibrations pass (diagnostic)
//   BB_PT_CAPTURE_WHEN_OFF=1 (pt1b) keep capturing and building while F9 has it off (default 0: off costs ~0)
// Every 5 s one "[pt]" block: GPU ms per pass (and per preset), render-thread ms of the capture and of our recording,
// VRAM, the scene, the calibrations and the skip reasons.
//
// (pt1e) What changed, and why (logs 20261118-073624 / 074330):
//  - Reflections are no longer traced into the game's SSR image. They are a fill for "Reflejos RT" (rt-reflections.cpp,
//    KYTY_RT_REFLECTIONS > 0): only on the pixels that path already reflects (its own mask and strength, guided by the
//    game's SSR and material), only where its screen-space ray missed, one hardware ray per block of trace pixels (2x2 on
//    bajo / medio). The hit's colour: this frame's screen where the hit is visible, else the previous composited frame
//    where it was visible then, else a per-mesh radiance learned from on-screen hits (one colour per captured mesh), else
//    nothing (the game's own fill stays). Without "Reflejos RT" no reflection ray is ever traced. The old passes into the
//    SSR image only with BB_PT_REFL_LEGACY=1.
//  - Sun shadows: the target (kRTT_SunShadowResolve, a persistent render target) is kept across frames like the SSR
//    image; rays only where the game's own mask is lit (min() can never brighten); checkerboard on bajo / alto / ultra;
//    the calibration measured on always-traced texels; composed before the first pass seen reading the mask.
//  - F9 off: no job, no capture, no dispatch hook, no fill; the 5 s line counts ON and OFF frames apart and proves the
//    OFF cost is 0.
//  - Budgets per preset (GPU ms of the scene + shadows + fill, ON frames): bajo 2, medio 3, alto 4, ultra 5; over it the
//    trace scales go down (and back up under 60 % of it), logged.
//   BB_PT_AB=1 alternates 5 s with and 5 s without (A/B in the same run: "[pt] A/B" line).
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <span>

namespace Libs::Graphics {
class CommandBuffer;
class TextureCache;
struct TextureBinding;
namespace PathTrace {

[[nodiscard]] bool Requested(); // BB_PATHTRACING=1 (read once)
// Requested and switched on live (F9 toggles; BB_PT_CYCLE windows "off").
[[nodiscard]] bool On();
void               ToggleLive(); // F9 (window.cpp; only while Requested)
[[nodiscard]] const char* PresetName();

// hwrt-scene.cpp's sizing while Requested (0: its own default)
[[nodiscard]] uint32_t CaptureVertices();
[[nodiscard]] uint64_t MeshBudgetBytes();
[[nodiscard]] uint32_t KeepFrames();
// (pt1e) moving meshes refit every N frames (the live preset's; 0: not Requested)
[[nodiscard]] uint32_t RefitEvery();

// The frame map's scope events (src/native/frame-map-probe.cpp, the GPU thread, after the queue's stack changed).
void ScopeBegin(uint32_t pass_id, int32_t queue);
void ScopeEnd(uint32_t pass_id, int32_t queue);
// The walk (graphicsRun.cpp): our GPU work due at this packet of the queue being translated.
[[nodiscard]] bool WorkPending();
void               RunWork(CommandBuffer& buffer);
// CommandProcessor::Flip (the GPU thread).
void Flip();

// table-xpr.inc: a G-buffer draw may be captured here (the GPU thread): a capture scope is open. True when not Requested.
[[nodiscard]] bool CaptureHere();
// A dispatch inside a scope we read images from: it takes the normal path (its bindings reach AfterDispatch).
[[nodiscard]] bool WantDispatchHook();
// renderCompute.cpp after a normal-path dispatch (the GPU thread): its images and which it writes.
void AfterDispatch(TextureCache& cache, std::span<const TextureBinding> images, uint64_t written_mask, uint64_t cs_hash);
// The render targets a draw writes (native-render.h NoteTarget), while g_want_targets.
inline std::atomic<bool> g_want_targets {false};
void NoteTarget(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth);
// Every draw's vertex SRT on the three draw paths (the sun's matrix inside a RenderShadowCascade scope).
void OnDraw(uint64_t vs_hash, std::span<const uint32_t> flattened_srt);
// (pt1c) The game's TAA dispatch (renderCompute.cpp): the colour it reads is the frame's lit HDR image (render size,
// before tonemap); its first read float colour image of the render size is kept as the reflections' hit radiance source
// and the debug tint's target.
void NoteTaaInput(TextureCache& cache, std::span<const TextureBinding> images, uint64_t written_mask, uint64_t cs_hash);
// The PS5 Pro super-resolution packet's view_to_clip (renderSuperResolution.cpp).
void NoteViewToClip(const float* m16);

// (pt1e) The reflections' fill, recorded by rt-reflections.cpp between its trace and its temporal pass (the GPU thread,
// the PS5 Pro packet's upscale, same frame as this frame's TLAS). Every image in the GENERAL layout except where a layout
// is given; trace-sized images: rad (out_a), aux (out_c), gb_cur / gb_prev (view normal, view distance), rad_out (out_a
// with the fill); render-sized: depth, color (this frame's game colour, linear filter), prev_screen (the last composite).
struct ReflFillIo {
	vk::ImageView   rad = nullptr, aux = nullptr, gb_cur = nullptr, gb_prev = nullptr, prev_screen = nullptr, rad_out = nullptr;
	vk::ImageView   depth = nullptr, color = nullptr;
	vk::ImageLayout depth_layout = vk::ImageLayout::eGeneral, color_layout = vk::ImageLayout::eGeneral;
	vk::Extent2D    render {}, trace {};
	vk::Buffer      exposure = nullptr; // the game's exposure buffer (null: 1.0)
	vk::DeviceSize  exposure_base = 0, exposure_range = 0;
	uint32_t        exposure_index = 0;
	bool            row0_up = true;      // rt-reflections' screen y convention (its view space)
	bool            history = false;     // prev_screen and gb_prev hold the last frame's
	float           rough_cutoff = 1.0f; // rt-reflections' roughness cutoff (a pixel above it was not traced: no fill)
	std::array<float, 16> view_to_clip {};
};
// True: the fill can run this frame (BB_PATHTRACING on, F9 on, this frame's TLAS and camera, the reflection part on).
[[nodiscard]] bool ReflFillReady();
// Records the fill (rad -> rad_out, every trace pixel written). False: nothing recorded (rad_out not written).
bool RecordReflFill(vk::CommandBuffer command, const ReflFillIo& io);
// rt-reflections' GPU ms of one frame (its whole chain) with / without our fill: the A/B line.
void NoteReflectionsMs(double ms, bool with_fill);

} // namespace PathTrace
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_PATHTRACE_H_
