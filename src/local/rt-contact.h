#ifndef EMULATOR_SRC_LOCAL_RT_CONTACT_H_
#define EMULATOR_SRC_LOCAL_RT_CONTACT_H_
// KYTY_HWRT_CONTACT=0|1|2|3 (rt-contact.cpp, rt-contact.comp; default 0 = No; 1 Bajo, 2 Medio, 3 Alto; needs KYTY_HWRT=1):
// ray-traced contact shadows. Short hardware ray queries into the scene TLAS (static + this frame's dynamic meshes:
// characters, armour, props) toward the dominant light: the sun (the RTSHADOW pass's latched world direction) and, per
// pixel, the strongest of the game's local shadowed lights (rt-local-lights.cpp, the HWRTLIGHTS lights). Darkens only the
// direct term and never where the pixel is already shadowed (the game's screen-space contact shadows, its cascades, the
// RT sun / lamp passes): see rt-contact.cpp. Off: nothing created, nothing recorded.
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int RtContactLevel();
// frame-gen.cpp CShadowApply, once its composite is recorded (only while KYTY_HWRT_CONTACT is on): the world direction
// toward the sun, its weight (fade x overcast), R (the sun's radiance over the ambient), and the pass's final sun
// visibility image (trace size, .r, GENERAL layout; null when none) with exact = true when that visibility is the whole
// sun term's (KYTY_RT_SUN: the composite replaced the game's own visibility with it).
void RtContactSetSun(const std::array<double, 3>& world_dir, float weight, float r, vk::ImageView vis, vk::Extent2D vis_extent, bool exact);
// In Upscale, after HwLightsApply: inputs.color becomes the composited copy. False: nothing done.
bool RtContactApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_CONTACT_H_
