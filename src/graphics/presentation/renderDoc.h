#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_

namespace Libs::Graphics {

class RenderContext;

// KYTY_RENDERDOC=1 / --rd: loads RenderDoc before the Vulkan instance (agc.cpp); "[rdoc]" lines in the log.
void RenderDocInit();
// RenderDoc loaded and its API bound.
bool RenderDocAvailable();
// F12 (window.cpp): one guest frame captured to <exe folder>\capturas.
void RenderDocRequestCapture();
// Called by the presentation thread after releasing video-out locks.
void RenderDocOnGuestFlip(RenderContext& renderer);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_ */
