#ifndef EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
#define EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
// KYTY_RT_REFLECTIONS=0|1|2|3 (off / low / high / ultra; rt-reflections.cpp, rt-reflections.comp): screen-space
// ray-traced reflections with denoising on the PS5 Pro path's render-resolution color, right before the upscaler
// (KYTY_UPSCALER: DLSS / XeSS / FSR; without an upscaler there is no such point and nothing runs).
// KYTY_RT_REFLECTIONS_STRENGTH (1.0, 0..2). Off (the default): nothing created, nothing recorded.
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
};
[[nodiscard]] bool RtReflGetGbuffer(RtReflGbuffer& out);
// The camera from the G-buffer draws' constants, with the upscaler's view_to_clip: the world -> view rotation rows, world
// up in view space, the current clip -> previous clip matrix (row-major, column vectors; x, y and w rows meaningful).
struct RtReflCamera {
	std::array<float, 12> view_rows {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	std::array<float, 16> clip_to_prev {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	std::array<float, 3>  up {0, 1, 0};
	bool                  rows = false, prev = false, up_valid = false;
};
void RtReflGetCamera(const std::array<float, 16>& view_to_clip, RtReflCamera& out);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
