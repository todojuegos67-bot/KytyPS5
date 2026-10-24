#ifndef EMULATOR_SRC_LOCAL_SHINE_H_
#define EMULATOR_SRC_LOCAL_SHINE_H_
// KYTY_SHINE=0|1|2|3 (off / sun only / + 2 local lights / + 4 local lights; shine.cpp, shine.comp): specular highlights
// and material-aware sharpening for metal and smooth surfaces (armour, shields, props) on the PS5 Pro path's HDR color,
// right before the upscaler. KYTY_SHINE_STRENGTH (1.0, 0..3), KYTY_SHINE_SHARPEN (0.4, 0..1). Off: nothing created.
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  ShineLevel();
[[nodiscard]] bool ShineWanted();
// In Upscale, after RtReflectionsApply: inputs.color becomes the composited copy. False: nothing done.
bool ShineApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// From CShadowApply (KYTY_RT_SHADOWS) every frame it knows the sun: the view-space direction to it and its weight
// (the shadows' fade x overcast factor).
void ShineSetSun(const std::array<double, 3>& view_dir, float weight);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_SHINE_H_
