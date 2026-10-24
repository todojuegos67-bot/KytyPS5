#ifndef EMULATOR_SRC_LOCAL_REALISM_H_
#define EMULATOR_SRC_LOCAL_REALISM_H_
// KYTY_REALISM=0|1|2|3 ("Modo Souls realista": off / suave / medio / fuerte; realism.cpp, realism.comp, material.cpp).
// A modest, non-generative realism mode on the PS5 Pro path's HDR color before the upscaler (KYTY_UPSCALER: DLSS /
// XeSS / FSR). Nothing is invented and the art is kept: only physically motivated corrections from data the game
// already has (its G-buffer material, its depth, its sun, its exposure), each one bounded and faded with distance.
//   - a preset: turns on and balances RT shadows, GTAO, SSGI and RT reflections (each env var set by the user wins);
//   - the composite (last before the upscaler): the sun's missing specular on smooth / wet surfaces (GGX, only the part
//     the game's own highlight does not already show), GTAO applied only to the ambient share of the light, an
//     optional gentle local contrast and filmic toe (KYTY_REALISM_TONE);
//   - one line every 10 s with every pass's GPU ms and the total against the level's budget.
#include "local/frame-gen.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics::FrameGen {

// The level (0 off). The preset's env vars are put in place (not overwriting any already set) before anything reads
// them: at static initialization of realism.cpp and again, idempotently, at the first call.
[[nodiscard]] int  RealismLevel();
[[nodiscard]] bool RealismWanted();
// In Upscale, after RT reflections: inputs.color becomes the composited copy. False: nothing done.
bool RealismApply(vk::CommandBuffer command, UpscaleInputs& inputs);

// The passes' GPU time (each pass's ReadTiming hands every frame's ms here): the 10 s line.
enum class RealismPass : int { RtShadows = 0, Ssgi, Gtao, Ssr, RtReflections, Realism, Count };
void RealismNoteGpuMs(RealismPass pass, double ms);

// GTAO hands its denoised AO (trace resolution: visibility in .r) and its view normal / view depth image (.w) to the
// composite instead of compositing it itself, so the occlusion darkens only the ambient share (frame-gen.cpp asks
// RealismTakesAo() before its composite; RealismGetAo is filled for this frame when it did).
[[nodiscard]] bool RealismTakesAo();
struct RealismAo {
	vk::ImageView ao = nullptr, geom = nullptr;
	vk::Extent2D  extent {};
	float         strength = 0.0f;
};
void RealismSetAo(const RealismAo& ao); // (frame-gen.cpp, GtaoApply: this frame's; cleared by RealismApply)

// The RT shadow pass's latched sun (frame-gen.cpp, CShadowApply): toward the light in its view space (the camera's,
// with the screen y convention its calibration found), its fade 0..1 (0: none).
struct RealismSun {
	bool                  valid = false;
	std::array<float, 3>  view {0, 1, 0};
	float                 fade = 0.0f;
	bool                  row0_up = true;
};
void RealismSetSun(const RealismSun& sun); // (each frame CShadowApply runs; invalid when it returned early)

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_REALISM_H_
