#ifndef EMULATOR_SRC_LOCAL_TORCH_H_
#define EMULATOR_SRC_LOCAL_TORCH_H_
// KYTY_TORCH=0|1|2|3 ("Luz de antorchas RT": No / Bajo / Medio / Alto; torch.cpp, torch.comp): firelight from torches,
// braziers and bonfires cast onto the nearby G-buffer surfaces (warm, flickering, occlusion-marched in the depth buffer,
// a glint on smooth / wet surfaces), screen-space only. KYTY_TORCH_FLARE=0|1 ("Destellos de lente"): a small lens flare
// from the visible fires, after every other pass. Both on the PS5 Pro path's HDR color before the upscaler (and so
// before the HUD). KYTY_TORCH_STRENGTH (1.0, 0..3), KYTY_TORCH_FLARE_STRENGTH (1.0, 0..3), KYTY_TORCH_LIGHT (the fire
// threshold, exposed luminance, 1.5), KYTY_TORCH_RANGE (1.0, 0.3..3), KYTY_TORCH_DEBUG=1 (the added light only).
// Off (both 0, the default): nothing created.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  TorchLevel();
[[nodiscard]] bool TorchFlareWanted();
// In Upscale, after GTAO (before SSR / RT reflections, so they reflect the lit walls): finds and tracks the fires and,
// at level >= 1, lights the surfaces; inputs.color becomes the lit copy. False: nothing replaced.
bool TorchApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// In Upscale, last (after the god rays): the lens flare from the lights TorchApply tracked this frame.
bool TorchFlareApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_TORCH_H_
