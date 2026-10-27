#ifndef EMULATOR_SRC_LOCAL_GODRAYS_H_
#define EMULATOR_SRC_LOCAL_GODRAYS_H_
// KYTY_GODRAYS=0|1|2|3 (godrays.cpp, godrays.comp): crepuscular rays (sun shafts) and volumetric halos around local lights
// on the PS5 Pro path's HDR color, right before the upscaler. 1: screen-space sun shafts (sun on screen) + 2 lights,
// 2: + a view-ray march shadowed against the depth (shafts with the sun off screen) + 4 lights, 3: more taps + 8 lights.
// KYTY_GODRAYS_STRENGTH (1.0, 0..3). Off: nothing created.
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  GodraysLevel();
[[nodiscard]] bool GodraysWanted();
// In Upscale, after ShineApply: inputs.color becomes the composited copy. False: nothing done.
bool GodraysApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// From CShadowApply (KYTY_RT_SHADOWS) every frame it knows the sun: the view-space direction to it and its weight.
void GodraysSetSun(const std::array<double, 3>& view_dir, float weight);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_GODRAYS_H_
