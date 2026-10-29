#ifndef EMULATOR_SRC_LOCAL_SMOKE_H_
#define EMULATOR_SRC_LOCAL_SMOKE_H_
// KYTY_SMOKE=0|1|2 ("Humo y particulas"; smoke.cpp, smoke.comp; default 0 = off, the launcher's default 2): the game's
// particle layer (scene colour after its particle-translucent pass minus the copy taken before it) made softer where
// sprites cut into walls and floors (1, soft particles) and lit warm by the nearby torch lights torch.cpp tracks (2).
// Off: nothing created, nothing recorded, no pre-particle copy for it.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  SmokeLevel();
[[nodiscard]] bool SmokeWanted();
// In Upscale after TorchApply: `game` is the game's own colour (Upscale's inputs as they came in, after the particles);
// inputs.color becomes the composited copy. False: nothing done.
bool SmokeApply(vk::CommandBuffer command, UpscaleInputs& inputs, const UpscaleImage& game);

// torch.cpp: the light buffer TorchApply tracked this flip (torch.comp's Frame, lights at word 32) and its local
// radius (m); false when it did not run this flip. Read only.
[[nodiscard]] bool TorchLightsThisFlip(vk::Buffer& buffer, float& radius);
// frame-gen.cpp: the pre-particle copy of this flip's scene color (see hwrt-particles.h), when it matches the color.
[[nodiscard]] vk::ImageView RtParticlesPreCopy(uint64_t color_address, vk::Extent2D extent);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_SMOKE_H_
