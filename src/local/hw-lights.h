#ifndef EMULATOR_SRC_LOCAL_HW_LIGHTS_H_
#define EMULATOR_SRC_LOCAL_HW_LIGHTS_H_
// KYTY_HWRT_LIGHTS=0|1|2|3 (hw-lights.cpp, hw-lights.comp; needs KYTY_HWRT=1): ray-traced shadows of the game's local
// lights (torches, braziers, hanging lamps) through the scene TLAS, on the PS5 Pro path's HDR color before the upscaler.
// The lights are the game's own shadowed spot / point lights, from their shadow projections (rt-local-lights.cpp
// LiveLights). 1: half-resolution trace, one ray a pixel; 2: full resolution, one ray; 3: full resolution, two rays.
// Off / unset: nothing is created and nothing runs (the light discovery included).
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int HwLightsLevel();
// In Upscale, after CShadowApply: inputs.color becomes the composited copy. False: nothing done (the color untouched).
bool HwLightsApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_HW_LIGHTS_H_
