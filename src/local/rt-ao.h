#ifndef EMULATOR_SRC_LOCAL_RT_AO_H_
#define EMULATOR_SRC_LOCAL_RT_AO_H_
// KYTY_AO=0..3 or No / Bajo / Medio / Alto (rt-ao.cpp, rt-ao.comp; ao-1028): screen-space ambient occlusion written from
// scratch (horizon-based visibility, metre radius, temporal + edge-aware denoise, joint-bilateral upsample), applied to
// the ambient term only, on the PS5 Pro path's render-resolution colour in FrameGen::Upscale (GTAO's old slot). No
// hardware ray tracing. Unset / 0 (the default): nothing created, nothing recorded, no change.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

// The level (0 off).
[[nodiscard]] int RtAoLevel();
// In Upscale: inputs.color becomes the composited copy. False: nothing done.
bool RtAoApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_AO_H_
