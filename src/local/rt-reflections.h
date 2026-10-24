#ifndef EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
#define EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
// KYTY_RT_REFLECTIONS=0|1|2|3|4 (off / low / medium / high / ultra; rt-reflections.cpp, rt-reflections.comp): screen-space
// ray-traced reflections with denoising on the PS5 Pro path's render-resolution color, right before the upscaler
// (KYTY_UPSCALER: DLSS / XeSS / FSR; without an upscaler there is no such point and nothing runs).
// KYTY_RT_REFLECTIONS_STRENGTH (1.0, 0..3), KYTY_RT_REFLECTIONS_MIN (0.3, 0..0.8: the wet surfaces' minimum
// reflectance). Off (the default): nothing created, nothing recorded.
#include "local/frame-gen.h"

#include <array>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int  RtReflectionsLevel();
[[nodiscard]] bool RtReflectionsWanted();
// In Upscale, after the other screen-space passes: inputs.color becomes the composited copy. False: nothing done (off,
// a movie, no projection, the pass could not be created).
bool RtReflectionsApply(vk::CommandBuffer command, UpscaleInputs& inputs);

// frame-gen.cpp: what the reflections read from the KYTY_SSR plumbing (its hooks run while RtReflectionsWanted()).
// The packed G-buffer's copy taken before the deferred lighting (SsrCaptureGbuffer) in this flip or the last.
struct RtReflGbuffer {
	std::array<vk::ImageView, 4> views {};
	vk::Extent2D                 extent {};
	// The depth as the deferred lighting read it (taken with the layers; null: none bound): water detection.
	vk::ImageView                depth = nullptr;
	vk::ImageLayout              depth_layout = vk::ImageLayout::eUndefined;
};
[[nodiscard]] bool RtReflGetGbuffer(RtReflGbuffer& out);
// The number of G-buffer copies taken so far (the periodic stats line: copies per frame).
[[nodiscard]] uint64_t RtReflCopies();
// The camera from the G-buffer draws' constants, with the upscaler's view_to_clip: the world -> view rotation rows, world
// up in view space, the current clip -> previous clip matrix (row-major, column vectors; x, y and w rows meaningful).
struct RtReflCamera {
	std::array<float, 12> view_rows {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	std::array<float, 16> clip_to_prev {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	std::array<float, 3>  up {0, 1, 0};
	bool                  rows = false, prev = false, up_valid = false;
};
void RtReflGetCamera(const std::array<float, 16>& view_to_clip, RtReflCamera& out);

// The calibration's byte table (16 fields, index layer * 4 + byte, 8 bits each) and its decode, shared with the
// material decode (material.cpp, KYTY_REALISM) so the statistics are taken once: mean / sd over all calibration samples,
// the mean on puddle-looking pixels and on water (count < 20: not measured), the 16-bin histograms. Kept from the last
// calibration attempt that had the G-buffer copy (valid false: none yet, or KYTY_RT_REFLECTIONS off).
struct RtReflByteTable {
	bool                                   valid = false;
	uint32_t                               attempt = 0;
	double                                 samples = 0.0, puddles = 0.0, water = 0.0;
	std::array<double, 16>                 mean {}, sd {}, puddle {}, water_mean {};
	std::array<std::array<uint32_t, 16>, 16> hist {};
	int32_t                                dec_normal = -1, dec_rough = -1; // (the shader's codes; -1 none)
	bool                                   rough_trusted = false, mat_auto = false, row0_up = true;
	float                                  mat_lo = 0.15f, mat_hi = 0.30f;
	double                                 normal_fit = 0.0; // (the decoded normal's mean cos against depth normals)
};
[[nodiscard]] bool RtReflGetByteTable(RtReflByteTable& out);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_REFLECTIONS_H_
