#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_

#include <cstdint>

union SDL_Event;

namespace Libs::Graphics {

void               HostInputInit();
void               HostInputKey(int key_code, bool down);
void               HostInputMouseButton(uint8_t mouse_button, bool down);
void               HostInputToggleMouseToJoystick();
// KYTY_KBM=1: keyboard and mouse play (hostInput.cpp). The motion/wheel calls return true when they took the event.
[[nodiscard]] bool HostInputKbmEnabled();
void               HostInputFocus(bool focused);
bool               HostInputMouseMotion(int dx, int dy);
bool               HostInputMouseWheel(int y);
[[nodiscard]] bool HostInputWaitEvent(SDL_Event* event);
// The camera's forward direction in world space (FrameGen::OnDraw, render thread): the KYTY_KBM mouse mapping and its
// calibration (KYTY_KBM_CALIBRATE=1) measure the game's yaw rate from it.
void               HostInputCameraForward(float fx, float fy, float fz);

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_PRESENTATION_WINDOW_HOST_INPUT_H_ */
