#ifndef EMULATOR_SRC_LOCAL_HWRT_GI_H_
#define EMULATOR_SRC_LOCAL_HWRT_GI_H_
// KYTY_HWRT_AO=0..3 / KYTY_HWRT_GI=0..3 (hwrt-gi.cpp, hwrt-gi.comp; need KYTY_HWRT=1): ray-traced ambient occlusion and
// one-bounce diffuse GI through the scene TLAS (hwrt-scene.cpp) on the PS5 Pro path's render-resolution color, in
// FrameGen::Upscale where GTAO runs (and in its place: KYTY_HWRT_AO replaces KYTY_GTAO unless KYTY_HWRT_AO_GTAO=1).
// Both unset / 0 (the default): nothing created, nothing recorded, no change.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

// Either switch set (and KYTY_HWRT=1 asked for).
[[nodiscard]] bool HwGiWanted();
// KYTY_HWRT_AO is on and takes GTAO's place (GtaoApply then does nothing).
[[nodiscard]] bool HwAoReplacesGtao();
// In Upscale, in GTAO's slot: inputs.color becomes the composited copy. False: nothing done.
bool HwGiApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_HWRT_GI_H_
