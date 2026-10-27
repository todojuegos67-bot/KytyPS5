#ifndef EMULATOR_SRC_LOCAL_VOLSHADOW_H_
#define EMULATOR_SRC_LOCAL_VOLSHADOW_H_
// KYTY_HWRT_VOLSHADOW=0|1|2|3 (No / Bajo / Medio / Alto; volshadow.cpp, volshadow.comp; needs KYTY_HWRT=1 and
// KYTY_GODRAYS >= 2): ray-traced shadowing of the god rays' view-ray march. The sun's visibility along each view ray is
// traced through the scene TLAS (6 / 10 / 16 points a ray, quarter resolution, temporal + bilateral), and godrays.comp
// takes min(its own depth-tested lit length, the TLAS-lit length): shafts are cut by roofs, columns, gates and grates
// even off screen, and never brighter than the approved volumetric (only light it already had is kept or removed).
// Off / unset: nothing is created, nothing runs, godrays binds a dummy and its shader takes the old path unchanged.
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int VolShadowLevel();
// From GodraysApply, before its dispatches: traces this frame's visibility on godrays' quarter grid (qextent) and
// returns the filtered view (RGBA16F, GENERAL; .x = lit fraction, 1 = no cut), or null (nothing done: godrays as before).
// sun_view: godrays' view-space direction to the sun; m: inputs.view_to_clip; row0_up: godrays' y convention.
vk::ImageView VolShadowRun(vk::CommandBuffer command, UpscaleInputs& inputs, vk::Extent2D qextent, const std::array<float, 3>& sun_view,
                           float sun_weight, bool row0_up);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_VOLSHADOW_H_
