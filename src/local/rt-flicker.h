#ifndef EMULATOR_SRC_LOCAL_RT_FLICKER_H_
#define EMULATOR_SRC_LOCAL_RT_FLICKER_H_
// KYTY_RT_FLICKER (default 1; 0 off: nothing created or recorded): the flicker self-check of the ray-traced passes
// (rt-flicker.cpp). In FrameGen::Upscale, around each pass: the pass's contribution (its color out - its color in)
// compared with the previous frame's while the camera holds still. Every 10 s one "RTFLICKER:" line: per pass, the share
// of the points it touches whose contribution changed by more than KYTY_RT_FLICKER_PCT (10) % of the pixel's light.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen::RtFlicker {

// Contact: KYTY_HWRT_CONTACT; Water: KYTY_HWRT_WATER.
// Gi: KYTY_HWRT_GI / KYTY_HWRT_AO (hwrt-gi.cpp), the RT GI + AO on the ambient term.
// Ao: KYTY_AO (rt-ao.cpp), the screen-space ambient occlusion (ao-1028).
enum Pass : uint32_t { Particles = 0, Sun = 1, Lights = 2, Reflections = 3, Volumetric = 4, Gi = 5, Contact = 6, Water = 7, Ao = 8, Count = 9 };
// Once a frame, before the passes: whether the camera holds still since the last frame.
void BeginFrame(const UpscaleInputs& inputs);
// (refl-pro) The camera held still since the last frame / for how many frames in a row.
bool     CameraStill();
uint32_t StillFrames();
// After a pass that replaced inputs.color (before: the color it read).
void Probe(vk::CommandBuffer command, GraphicContext* graphics, Pass pass, const UpscaleImage& before, const UpscaleImage& after);
// (ds-rt-shadow) A pass's own events in this 10 s window, named by a string literal (up to 8 a pass): printed after its
// numbers as "causes: <name> <count>, ..." so a flicker figure comes with what changed (render thread only).
void Cause(Pass pass, const char* name, uint64_t count = 1);

} // namespace Libs::Graphics::FrameGen::RtFlicker

#endif // EMULATOR_SRC_LOCAL_RT_FLICKER_H_
