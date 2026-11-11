#ifndef EMULATOR_SRC_LOCAL_PARTICLE_MV_H_
#define EMULATOR_SRC_LOCAL_PARTICLE_MV_H_
// KYTY_FG_PARTICLE_MV=0|1 (particle-mv.cpp, particle-mv.comp; default 0 = off; the launcher sets 1 only in Souls ligero
// (30x2) and Souls ligero + (30x3)): motion vectors for the particle layer (fire, sparks, rain, mist, spells), measured
// by block matching the layer (colour after the particle-translucent pass minus the copy before it) against the
// previous real frame's, and composed into frame generation's copy of the motion vectors only (the game's image and the
// upscaler's input stay untouched). Off: nothing created, nothing recorded, no pre-particle copy for it.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] bool ParticleMvWanted();
// In Upscale (after SmokeApply): `game` is the game's own colour after the particles. Measures this frame's particle
// vectors (kept until the next call). Changes nothing in `inputs`. False: nothing measured (the composite then skips).
bool ParticleMvMeasure(vk::CommandBuffer command, const UpscaleInputs& inputs, const UpscaleImage& game);
// At the guest flip, right after frame generation's motion copy (frame-gen.cpp): the vectors measured this frame mixed
// into that copy, in place. `motion` must have been made with storage usage (R16G16_SFLOAT). False: nothing done.
bool ParticleMvCompose(vk::CommandBuffer command, GraphicContext& graphics, VulkanImage& motion);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_PARTICLE_MV_H_
