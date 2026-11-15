#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();
// --precompile-only: a hidden window, the device and the renderer as for the game, then the precompile; its exit code.
int                      WindowPrecompileOnly();

// Nits of the display showing the game window; the defaults apply when the host reports none.
struct DisplayLuminance {
	float max_full_frame = 80.0f;
	float max            = 1000.0f;
	float min            = 0.0f;
};

[[nodiscard]] DisplayLuminance WindowGetDisplayLuminance();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
