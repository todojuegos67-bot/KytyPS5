#ifndef EMULATOR_SRC_LOCAL_DOF_H_
#define EMULATOR_SRC_LOCAL_DOF_H_
// KYTY_DOF=0..3 or No / Suave / Cinematico / Fuerte (dof.cpp, dof.comp; ds-dof-1029): "Profundidad de campo", a
// distance depth of field: the background past the focus distance gets a soft, film-like bokeh blur (bright highlights
// open into round discs), near and mid ground stay sharp. On the PS5 Pro path's render-resolution colour in
// FrameGen::Upscale, after every other pass and before the upscaler, so the HUD (drawn after) stays sharp.
// Unset / 0 (the default): nothing created, nothing recorded, no change.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

// The level (0 off).
[[nodiscard]] int DofLevel();
// In Upscale: inputs.color becomes the blurred copy. False: nothing done.
bool DofApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_DOF_H_
