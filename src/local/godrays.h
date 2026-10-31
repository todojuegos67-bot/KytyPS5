#ifndef EMULATOR_SRC_LOCAL_GODRAYS_H_
#define EMULATOR_SRC_LOCAL_GODRAYS_H_
// KYTY_GODRAYS=0|1|2|3 (godrays.cpp, godrays.comp): crepuscular rays (sun shafts) and volumetric halos around local lights
// on the PS5 Pro path's HDR color, right before the upscaler. 1: screen-space sun shafts (sun on screen) + 2 lights,
// 2: + a view-ray march shadowed against the depth (shafts with the sun off screen) + 4 lights, 3: more taps + 8 lights.
// KYTY_GODRAYS_STRENGTH (1.0, 0..3). Off: nothing created.
#include "local/frame-gen.h"

#include <array>
#include <string>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  GodraysLevel();
[[nodiscard]] bool GodraysWanted();
// (motes-1031) KYTY_MOTES=0|1|2|3 ("Particulas volumetricas": No / Sutil / Media / Densa): floating dust motes in the
// air, anchored in the world and drifting, lit only by the volumetric light (sun shafts, torch / lamp halos) of this
// pass. With KYTY_GODRAYS off the pass runs at level 1 for the motes' light alone (its own light not added).
[[nodiscard]] int MotesLevel();
// (motes-1031, frame-gen.cpp) The camera's view -> world matrix (rows 0..2 of the 4x4, column vectors), from the
// G-buffer pass's view-projection and the upscaler's view -> clip. False: not known this frame.
bool GodraysViewToWorld(const std::array<float, 16>& view_to_clip, std::array<float, 12>& rows);
// In Upscale, after ShineApply: inputs.color becomes the composited copy. False: nothing done.
bool GodraysApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// From CShadowApply (KYTY_RT_SHADOWS) every frame it knows the sun: the view-space direction to it and its weight.
void GodraysSetSun(const std::array<double, 3>& view_dir, float weight);
// (vol-1027) Why CShadowApply gave no sun this frame (the flicker self-check's cause counters): 0 the sun is not latched
// yet / dropped, 1 the RT sun cascades were not copied this frame, 2 no scene TLAS yet, 3 the pass's images / pipeline.
void GodraysNoteSunMiss(int cause);
// (vol-1027, KYTY_RT_CAPTURE frame.json) the pass's state as a JSON object: level, the held view-space sun and its weight,
// the on-screen factor, the TLAS-shadowed march in use, the last frame's feed. "null" when off.
[[nodiscard]] std::string GodraysDescribe();

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_GODRAYS_H_
