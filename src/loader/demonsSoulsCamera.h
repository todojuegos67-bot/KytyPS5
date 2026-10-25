#pragma once

namespace Loader {
struct Program;
// (10-25) Demon's Souls' free camera (ChrExFollowCam) speeds a full-tilt horizontal right stick up from its base turn
// rate to a higher one over a ramp; with keyboard and mouse (KYTY_KBM=1) that ramp is removed (KYTY_KBM_NO_RAMP,
// demonsSoulsCamera.cpp).
namespace DemonsSoulsCamera {
// KYTY_KBM_NO_RAMP as asked: 0 the game's ramp, 1 no ramp and the yaw at the game's top rate x stick (the default with
// KYTY_KBM=1 or KYTY_KBM_CALIBRATE=1), 2 no ramp and the yaw at the game's base rate x stick. 0 without keyboard and mouse.
int  NoRampMode();
void InstallNoRamp(Program* program);
} // namespace DemonsSoulsCamera
} // namespace Loader
