#ifndef EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
#define EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
// KYTY_RT_REFLECTIONS=0|1|2|3|4|5 (off / low / medium / high / ultra / extreme (refl10); rt-reflections.cpp, rt-reflections.comp): screen-space
// ray-traced reflections with denoising on the PS5 Pro path's render-resolution color, right before the upscaler
// (KYTY_UPSCALER: DLSS / XeSS / FSR; without an upscaler there is no such point and nothing runs).
// KYTY_RT_REFLECTIONS_STRENGTH (1.0, 0..3), KYTY_RT_REFLECTIONS_MIN (0.3, 0..0.8: the wet surfaces' minimum
// reflectance). Off (the default): nothing created, nothing recorded.
// (refl8) Guided by the game's own SSR (KYTY_RT_REFLECTIONS_GUIDE, default on; 0: refl6's material mask): it decides
// which surfaces reflect and how much; our sharper ray only replaces its SSR color where ours hits (rt-reflections.cpp).
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  RtReflectionsLevel();
[[nodiscard]] bool RtReflectionsWanted();
// In Upscale, after the other screen-space passes: inputs.color becomes the composited copy. False: nothing done (off,
// a movie, no projection, the pass could not be created).
bool RtReflectionsApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// (rpost1) KYTY_REFL_POST=1 (FSR 4): the reflection's added light put on after the upscaler. PostAdd: `target` (RGBA16F,
// storage, general) += the light, false when there is none this frame. F10 (window.cpp): RtReflPostToggle, ON / OFF.
bool RtReflectionsPostAdd(vk::CommandBuffer command, vk::ImageView target, vk::Extent2D size);
[[nodiscard]] bool RtReflPostWanted();
[[nodiscard]] bool RtReflPostLive();
bool RtReflPostToggle();

// frame-gen.cpp: what the reflections read from the KYTY_SSR plumbing (its hooks run while RtReflectionsWanted()).
// The packed G-buffer's copy taken before the deferred lighting (SsrCaptureGbuffer) in this flip or the last.
struct RtReflGbuffer {
	std::array<vk::ImageView, 4> views {};
	vk::Extent2D                 extent {};
	// The depth as the deferred lighting read it (taken with the layers; null: none bound): water detection.
	vk::ImageView                depth = nullptr;
	vk::ImageLayout              depth_layout = vk::ImageLayout::eUndefined;
};
[[nodiscard]] bool RtReflGetGbuffer(RtReflGbuffer& out);
// The number of G-buffer copies taken so far (the periodic stats line: copies per frame).
[[nodiscard]] uint64_t RtReflCopies();
// The camera from the G-buffer draws' constants, with the upscaler's view_to_clip: the world -> view rotation rows, world
// up in view space, the current clip -> previous clip matrix (row-major, column vectors; x, y and w rows meaningful).
struct RtReflCamera {
	std::array<float, 12> view_rows {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	std::array<float, 16> clip_to_prev {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	std::array<float, 3>  up {0, 1, 0};
	bool                  rows = false, prev = false, up_valid = false;
};
void RtReflGetCamera(const std::array<float, 16>& view_to_clip, RtReflCamera& out);
// (rt2) KYTY_RT2=1 (default off): the reflections' and the sun's RT fixes of rt2 (implies KYTY_RTREFL1 and
// KYTY_RT_REFLECTIONS_REPROJ): the camera matrices made consistent in their TAA jitter, the history candidate chosen by
// the material (no per-pixel switching), the sun shadows aligned with the game's cascades (frame-gen.cpp Rt2Wanted).
[[nodiscard]] bool Rt2Wanted();
// (rt3) KYTY_RT3=1 (default off, implies KYTY_RT2): the RT reflections' cost cut. The trace, temporal and a-trous passes at
// the render resolution even with "Nitidez maxima" (rt2 traced at the output size: 2.25x the pixels at FSR Calidad, plus a
// second mirror march per pixel in the composite to bring the larger grid back), and the composite's early out read from
// the trace's mask first (exact: the same result, the heavy per-pixel work only where the trace marked a reflector).
// KYTY_RT3_AB=1 too: 5 s on / 5 s off in the same run and the line "[rt3] GPU ms reflejos ON/OFF".
[[nodiscard]] bool Rt3Wanted();
// (rt4) KYTY_RT4=1 (default off, implies KYTY_RT3 and KYTY_RT2): reflections that cover the screen and stay still on
// stone. Where a ray leaves the screen, is occluded, sees the sky or finds nothing (and where its confidence fades: the
// screen's edges, silhouettes), the game's own specular reflection fills in, blended by the ray's confidence: the
// lighting CS's written image kRTT_SpecularIBL_HalfResX (IBLRendering: its SSR + local box cubemaps + the distant cubemap;
// Ghidra), its scale and model calibrated at run time from confident hits. Not water: the ray along the footprint's
// smoothed normal (the normal map's spread as lobe width), the camera's own reprojection on static surfaces, R2 + hash
// samples. Water keeps its own path. KYTY_RT4_AB=1: 5 s on / 5 s off and "[rt4]" lines for both; KYTY_RT4_FILL (1.0),
// KYTY_RT4_TOKSVIG (0.5), KYTY_RT4_IBL_SLOT (the lighting's written slot; default the RGBA float one at half the width).
[[nodiscard]] bool Rt4Wanted();
// (rt5) KYTY_RT5=1 (default off, implies KYTY_RT4, KYTY_RT3 and KYTY_RT2): the game's own material response and the
// reflection's own reprojection. G-buffer surfaces reflect by their material (no longer only where the game's SSR was
// confident); the composite replaces the game's specular reflection (kRTT_SpecularIBL, Ghidra: the IBL pass after its SSR)
// at the game's split-sum weight: delta = EnvBRDF x (ours - the game's reflected radiance), so a miss changes nothing and
// rough stone gets nearly nothing; rough lobes traced on half the pixels a frame. Static surfaces: the reflection's history
// (temporal and the composite's stabilisation) at its virtual image point, kept in motion and let go by disagreement, not
// speed (walking no longer blurs it). "[rt5]" every 10 s. Water unchanged.
[[nodiscard]] bool Rt5Wanted();
// (refl8) The guide: the game's own screen-space reflection result, a half-resolution RGBA16F input of its deferred
// lighting / specular resolve (KYTY_RT_REFLECTIONS_GUIDE_CS, default 9fb5cc274b5e4f82; slot KYTY_RT_REFLECTIONS_GUIDE_SLOT,
// default the first RGBA16F input at half its written image's size), copied right after that dispatch (frame-gen.cpp
// RtReflGuideAfterDispatch) in this flip or the last: rgb its reflected color, a its confidence (where it hit).
struct RtReflGuide {
	vk::ImageView view = nullptr;
	vk::Extent2D  extent {};
	uint64_t      cs = 0, producer = 0; // (the lighting's hash; the dispatch that wrote the image, 0: not seen)
	int           slot = -1;
};
[[nodiscard]] bool RtReflGetGuide(RtReflGuide& out);
// (combo19) Why the guide is (not) there now: "copied", or the last reason it was not (a string literal).
[[nodiscard]] const char* RtReflGuideStatus();
// (rt4) The copy of the game's specular IBL taken after the lighting dispatch (frame-gen.cpp) in this flip or the last.
[[nodiscard]] bool Rt4GetIbl(RtReflGuide& out);
[[nodiscard]] const char* Rt4IblStatus();

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
