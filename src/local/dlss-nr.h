#ifndef EMULATOR_SRC_LOCAL_DLSS_NR_H_
#define EMULATOR_SRC_LOCAL_DLSS_NR_H_
// DLSS 5 Neural Rendering on Vulkan, foveated (ds-foveated-1029; dlss-nr.cpp, dlss-nr.comp): the launcher's "DLSS foveado".
//
// NVIDIA's NR model (NGX feature 18, nvngx_dlssnr.dll: NOT shipped with the emulator, the user's own copy beside
// kyty_emulator.exe or in its streamline folder) driven through its native Vulkan entry points, on the PS5 Pro path's
// upscaled output (after DLSS / XeSS / FSR, before the game's bloom, depth of field and HUD), inside the foveated
// working grid of FoveatedDLSS5 (github.com/maximedgoulet/FoveatedDLSS5, MIT): full resolution in a centre zone, a
// smoothstep fall-off, a lower density toward the edges (separately per axis), one NR evaluate on the smaller warped
// image with one temporal history, depth and motion vectors warped identically, then stretched back as a ratio on the
// untouched full-res frame (the centre is copied 1:1 both ways: never darkened, never blurred).
//
//   KYTY_FOVEATED=0|1          the foveated grid (1 also turns NR on; 0 / unset: KYTY_DLSS_NR decides, uniform grid)
//   KYTY_DLSS_NR=0|1           NR on the uniform grid (no foveation)
//   KYTY_FOV_WIDTH / _HEIGHT   the full-res centre zone: a fraction of the output (<= 1) or output pixels (default 0.52 / 0.51,
//                              2000 x 1100 at 3840 x 2160, FoveatedDLSS5's suggestion for 4K)
//   KYTY_FOV_FALLOFF           the fall-off on every side: a fraction of the output's height (<= 1) or pixels (default 400 px
//                              at 2160, scaled with the height)
//   KYTY_FOV_EDGE              the density beyond the fall-off, 0.25..1.0 per axis (default 0.5; corners get edge x edge)
//   KYTY_FOV_MEASURE=0         skip the start-up A/B (uniform 300 frames, then foveated 300; off when it saves < 5 %)
//   KYTY_FOV_VIEW=1            the density map over the picture (diagnostic)
//   KYTY_DLSS_NR_STRENGTH      0..1, how much of the model's edit (default 1)
//   KYTY_DLSS_NR_PRESET / _STYLE / _TONE / _STRUCTURE / _SKIN / _AUTOMASK   the model's create-time steering (RenoDX's
//                              defaults: 0, 0, 1, 1, -1, 0)
//   KYTY_DLSS_NR_UNSIGNED=1    load a DLL not signed by NVIDIA (a modded one for pre-Blackwell GPUs)
//
// Missing DLL, a refused init / create / evaluate: one clear log line, NR off, the upscaler's picture as before.
// Every 10 s one "DLSS-NR:" line: the zone, edge scale, NR grid, pixels saved, GPU ms (encode, NR, resolve) and, after
// the A/B, the GPU ms the foveation saved.
#include "local/frame-gen.h"

#include <string>
#include <vector>

namespace Libs::Graphics::FrameGen {

// KYTY_DLSS_NR=1 or KYTY_FOVEATED=1 (read once).
[[nodiscard]] bool DlssNrWanted();
// Before the device: the device extensions the model's Vulkan path needs (VK_NVX_binary_import,
// VK_NVX_image_view_handle; NGX loads its kernels and addresses its images through them), where available.
void DlssNrAddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions);
// In Upscale, after the upscaler wrote inputs.output: the model's edit applied to inputs.output in place. False: nothing.
bool DlssNrApply(vk::CommandBuffer command, const UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_DLSS_NR_H_
