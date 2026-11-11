#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_

#include <cstdint>

union SDL_Event;

namespace Libs::Graphics {

void               HostInputInit();
void               HostInputKey(int key_code, bool down);
// The keymap (--keymap) gives that key to a control of the pad.
[[nodiscard]] bool HostInputMapsKey(int key_code);
// Whether the keyboard's default keys count (false in the Steam Deck build unless KYTY_DECK_KEYBOARD=1: there Steam's
// desktop layout sends keys for the Deck's own buttons).
[[nodiscard]] bool HostInputDefaultKeysUsed();
void               HostInputMouseButton(uint8_t mouse_button, bool down);
// Wheel notches (up > 0, down < 0) on the controls mapped to Mouse:WheelUp / Mouse:WheelDown: a short press each.
void               HostInputMouseWheelNotches(int notches); // (des-up-1110: upstream's, renamed)
void               HostInputToggleMouseToJoystick();
// KYTY_KBM=1: keyboard and mouse play (hostInput.cpp). The motion/wheel calls return true when they took the event.
[[nodiscard]] bool HostInputKbmEnabled();
// KYTY_KBM_WARP: the mouse turn (degrees, +x right, +y as the mouse's y) the frame presented does not show yet: what
// the game-frame reads after its flip sent (`shown_seq`, HostInputKbmOnFlip at its flip) plus what is still pending,
// in the camera's degrees (learned from the camera; zero until then without kbm_curve.txt). False without KYTY_KBM or
// with the mouse released.
bool               HostInputKbmWarpDegrees(uint64_t shown_seq, double* yaw, double* pitch);
// A guest flip reserved (videoOut): the game-frame read sequence it shows. (10-25) Also how a pad read tells a new game
// frame from a second read in the same one (the stick is constant over a game frame).
uint64_t           HostInputKbmOnFlip();
void               HostInputFocus(bool focused);
void               HostInputKbmSetLive(float sens_factor, float antidz);
bool               HostInputMouseMotion(int dx, int dy);
bool               HostInputMouseWheel(int y);
[[nodiscard]] bool HostInputWaitEvent(SDL_Event* event);
// The camera's forward direction in world space (FrameGen::OnDraw, render thread): the KYTY_KBM mouse mapping and its
// calibration (KYTY_KBM_CALIBRATE=1) measure the game's yaw rate from it.
void               HostInputCameraForward(float fx, float fy, float fz);

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_ */
