#pragma once
#include <cstdint>

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
// (11-10) Direct mouse camera (KYTY_KBM_DIRECT, default on with KYTY_KBM=1): the mouse turns the free camera by its
// angle (demonsSoulsCamera.cpp). Wanted: asked for; Installed: the hooks are in; TakesMouse: mouse counts go to
// DirectAddMouse (the free camera running, the stick signs known) instead of the emulated stick.
bool DirectWanted();
bool DirectInstalled();
bool DirectTakesMouse();
void DirectAddMouse(int dx, int dy);
// (11-10c) While the mouse turned the camera in the last KYTY_KBM_HOLD_MS (default 400), the emulated right stick's X
// is held off centre (HoldStickX, 0..255; 128 = none) so the game sees camera input and its auto-follow behind a running
// player stays off; the hook cancels that held stick's own turn (yaw stick = 0 in frames without counts).
int  DirectHoldStickX();
void DirectNoteStickRouted(int64_t counts); // mouse counts that went to the emulated stick path (log)
void DirectSetStick(int x, int y);    // the emulated right stick last sent (0..255), to learn the signs
void DirectSetDegPerCount(float deg); // degrees per mouse count (deg per count * sensitivity)
} // namespace DemonsSoulsCamera
} // namespace Loader
