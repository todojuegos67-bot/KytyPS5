#ifndef EMULATOR_SRC_LOCAL_RT_CAPTURE_H_
#define EMULATOR_SRC_LOCAL_RT_CAPTURE_H_
// KYTY_RT_CAPTURE=1 (rt-capture.cpp, rt-capture.comp): the RT data capture. Once in the world, it records what the RT
// passes see of the game's frame into logs/rtcapture/<stamp>/ next to the emulator: a "still" set (camera held still
// for 2 s) and then a "moving" set (camera moving for 0.5 s). Each set has the pre-upscale colour, depth and motion
// vectors and the four packed G-buffer layers (R32_UINT, as copied at the deferred lighting) plus the G-buffer-time depth,
// as raw bit-exact .bin files and PNG views (one PNG per byte of each G-buffer word), per-bit / per-byte statistics, the
// camera (view_to_clip, the scene view-projection, 64 frames of jitter, motion scale, near / far), the sun and the local
// lights the RT passes know, and frame.json + README.txt. "RTCAPTURE: done <path>" when finished. Off: nothing runs.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen::RtCapture {

bool Enabled();
// In Upscale, before the RT passes (the inputs as the game left them).
void OnFrame(vk::CommandBuffer command, const UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen::RtCapture

#endif // EMULATOR_SRC_LOCAL_RT_CAPTURE_H_
