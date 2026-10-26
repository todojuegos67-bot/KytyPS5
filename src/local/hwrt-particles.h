#ifndef EMULATOR_SRC_LOCAL_HWRT_PARTICLES_H_
#define EMULATOR_SRC_LOCAL_HWRT_PARTICLES_H_
// KYTY_HWRT_PARTICLES=0|1|2|3 (hwrt-particles.cpp, hwrt-particles.comp; default 0): light from the game's emissive
// particles (fire, embers, sparks, magic) on the world, shadowed by hardware ray queries into the KYTY_HWRT scene TLAS.
// Needs KYTY_HWRT=1. Off: nothing created, nothing recorded.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  HwParticlesLevel();
[[nodiscard]] bool HwParticlesWanted();
// True while the pass ran this flip (the SSGI emissive light then leaves the particles to it: no double light).
[[nodiscard]] bool HwParticlesActive();
// In Upscale, first (on the game's own color, so color - pre-particle copy is the particles' emission alone):
// inputs.color becomes the composited copy. False: nothing done.
bool HwParticlesApply(vk::CommandBuffer command, UpscaleInputs& inputs);

// frame-gen.cpp: the pre-particle copy of this flip's scene color (KYTY_RT_PARTICLES' hook, also taken while
// HwParticlesWanted()), when it matches the color; null otherwise. Left in SHADER_READ_ONLY_OPTIMAL.
[[nodiscard]] vk::ImageView RtParticlesPreCopy(uint64_t color_address, vk::Extent2D extent);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_HWRT_PARTICLES_H_
