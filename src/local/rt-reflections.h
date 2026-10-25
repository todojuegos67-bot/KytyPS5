#ifndef EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
#define EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
// KYTY_RT_REFLECTIONS=0|1|2|3|4 (off / low / medium / high / ultra; rt-reflections.cpp, rt-reflections.comp): screen-space
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

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
