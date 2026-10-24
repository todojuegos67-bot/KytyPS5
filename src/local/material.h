#ifndef EMULATOR_SRC_LOCAL_MATERIAL_H_
#define EMULATOR_SRC_LOCAL_MATERIAL_H_
// The material decode shared by the passes that read Demon's Souls' packed G-buffer (material.cpp; GPU side:
// material.glsl). Every field is decided from statistical evidence and logged with its numbers and its override:
//   normal:   rt-reflections' calibration (mean cos against depth normals), else layer 3 oct 8:8 world (the 10-24 log)
//   rough:    the byte the puddle-looking pixels separate on downward (rt-reflections' byte table; log: L2.b0 0.40 / 0.01)
//   wet:      the byte they separate on upward (log: L1.b2 0.60 / 0.96), kept only when it is not just 1 - rough
//             (realism.comp's correlation between the two)
//   albedo:   three consecutive bytes of a layer whose values follow the lit color's chromaticity and brightness
//             (realism.comp: per-byte correlation with the lit chroma r, g, b and luma)
//   metal:    a two-valued byte (most samples at 0 or 1) whose set pixels are darker and smoother than the rest (a
//             metal's diffuse is black: dark armour); none unless all of that holds
// Overrides (each wins over the calibration): KYTY_REALISM_NORMAL=<layer>:<layout>[w][n],
// KYTY_REALISM_ROUGH / _WET / _METAL=<layer>:<shift>:<bits>[g] | none, KYTY_REALISM_ALBEDO=<layer>:<shift>[r][l] | none
// (r: bytes in b, g, r order; l: linear, not sRGB), KYTY_REALISM_SMOOTH=lo:hi (roughness: smooth below lo, rough from hi).
#include <array>
#include <cstdint>
#include <string>

namespace Libs::Graphics::FrameGen {

enum MaterialField : int { MatNormal = 0, MatRough, MatWet, MatAlbedo, MatMetal, MatFields };

struct MaterialDecode {
	std::array<int32_t, MatFields>     code {-1, -1, -1, -1, -1};   // (material.glsl's codes; -1 none)
	std::array<int, MatFields>         confidence {};               // 0 none, 1 low, 2 medium, 3 high
	std::array<std::string, MatFields> source {};                   // where each came from (logged)
	float                              smooth_lo = 0.25f, smooth_hi = 0.50f; // roughness: smooth / wet below lo, rough from hi
	float                              wet_lo = 0.6f, wet_hi = 0.95f;        // wet field: dry at lo, wet at hi
	uint32_t                           version = 0;                 // bumped on every change (the composite fades over)
};

// The current decode (env overrides applied, rt-reflections' table pulled in when it has a newer attempt).
[[nodiscard]] const MaterialDecode& MaterialGet();
// Pulls rt-reflections' byte table when a new attempt is in (and the env overrides once); true: the decode changed.
bool MaterialRefresh();

// realism.comp's calibration sums (layout below, ints, values x 1000): decides albedo, metal and wet vs gloss, logs.
// True: the decode changed.
constexpr int MatCalN      = 0;                        // samples
constexpr int MatCalChroma = 1;                        // sum r, g, b, r^2, g^2, b^2, Y, Y^2 (lit chroma, tonemapped luma)
constexpr int MatCalField  = MatCalChroma + 8;         // 16 fields x MatCalPer
constexpr int MatCalPer    = 11;                       // v, v^2, v*r, v*g, v*b, v*Y, extreme, n_hi, Y_hi, rough_hi, v*rough
constexpr int MatCalRough  = MatCalField + 16 * MatCalPer; // sum rough, rough^2
constexpr int MatCalInts   = MatCalRough + 2;
bool MaterialAnalyze(const int32_t* sums, uint32_t attempt);
[[nodiscard]] bool MaterialAlbedoKnown();

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_MATERIAL_H_
