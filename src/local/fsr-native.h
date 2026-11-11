#ifndef EMULATOR_SRC_LOCAL_FSR_NATIVE_H_
#define EMULATOR_SRC_LOCAL_FSR_NATIVE_H_
// AMD FidelityFX FSR 3 built from source (3rdparty/fidelityfx, SDK 1.1.4) on Linux, where AMD ships no runtime for the
// ffx-api the Windows FSR frame generation loads (amd_fidelityfx_vk.dll, frame-gen.h), and on Windows for its upscaling
// (nothing to download). Any Vulkan GPU.
// - KYTY_UPSCALER=fsr: FSR 3 upscaling in place of Sony's PS5 Pro upscaler the game dispatches, from the inputs DLSS
//   takes there (renderSuperResolution.cpp: the game's jittered HDR color before its post-processing, its depth and
//   motion vectors, the jitter and the reset flag), so the HUD and the post-processing are drawn on the upscaled image.
// - KYTY_FRAMEGEN_BACKEND=fsr (KYTY_FRAMEGEN=1 implied; Linux): FSR 3 frame interpolation, one frame between two of the
//   game's, made and shown by the present thread as NGX's DLSS-G frames are (swapchain.cpp PresentGenerated,
//   FrameGen::Generate).
// KYTY_FSR_SHARPNESS=<0..1> (default 0.6): FSR's RCAS sharpening of the upscaled image (0: none). At Boletaria's start
// (1080 -> 2160) 0.8 had a quarter more edge detail than 0.3; the game's jitter as DLSS takes it served FSR as well as
// any sign change (live switches fsr_jitter_mode, fsr_sharpness_pct, fsr_mv_mode, fsr_debug_view: live-switch.h).
// Elsewhere (macOS) the functions answer false; on Windows the interpolation's (AMD's runtime makes those frames there).
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics {
struct GraphicContext;
namespace FsrNative {

// KYTY_UPSCALER=fsr in a build with the FidelityFX sources.
[[nodiscard]] bool UpscalerRequested();
// KYTY_UPSCALER=dlss where DLSS cannot run (no NVIDIA RTX GPU or runtime, or FSR 3 frame generation, which runs without
// Streamline): the upscaling is FSR 3's from then on (UpscalerRequested), not the game's own TAA (issue #85). False
// where the sources are not built (macOS).
[[nodiscard]] bool UpscaleInsteadOfDlss();
// GPU thread or the recording worker: records the upscale of `inputs` (color, motion and depth in shader-read layouts,
// the output general) into `command`; false when it could not (the caller stretches instead). The images are left in
// their layouts.
[[nodiscard]] bool Upscale(GraphicContext& graphics, vk::CommandBuffer command, const FrameGen::UpscaleInputs& inputs);

// KYTY_FRAMEGEN_BACKEND=fsr in a build with the FidelityFX sources.
[[nodiscard]] bool InterpolationRequested();
struct InterpolationInputs {
	FrameGen::UpscaleImage backbuffer;                  // this frame at the swapchain's size, general layout
	FrameGen::UpscaleImage depth, motion;               // the game's, at the render size, shader-read layouts
	FrameGen::UpscaleImage hudless;                     // the backbuffer without the HUD (shader-read; image null: none)
	vk::Rect2D             region {};                   // where the game image is in the backbuffer
	std::array<float, 2>   jitter_px {};                // the projection's offset in render pixels
	float                  near_plane = 0.1f, fov_y = 1.0f;
	float                  frame_ms = 16.7f;            // the time between the game's last two frames
	bool                   reset    = false;            // no frame before this one to interpolate from
	bool                   pq       = false;            // the backbuffer is HDR10 (ST 2084 PQ), not sRGB
};
// Present thread: records into `output` (general layout, storage) the frame between the last backbuffer given and this
// one; false when there is none to show (the first frame after a reset, or FSR failed).
[[nodiscard]] bool Interpolate(GraphicContext& graphics, vk::CommandBuffer command, const InterpolationInputs& inputs,
                               const FrameGen::UpscaleImage& output);

} // namespace FsrNative
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FSR_NATIVE_H_
