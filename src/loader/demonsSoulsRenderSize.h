#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>

namespace Loader {
struct Program;
// KYTY_RENDER_HEIGHT=<720..2160> (run-windows.ps1 -RenderHeight, the launcher's "Render resolution"): Demon's Souls'
// performance mode - the PS5's default, its 3D rendered at 2560x1440 and upscaled by the game to a 3840x2160 output -
// renders its 3D at that height (16:9) instead. The output, the UI and what the game draws at output size stay 4K;
// GPU-bound PCs (a laptop's RTX 3080: 1-1 standing 20.7 -> 24.5 fps at 1080) trade sharpness for frame rate, and 2160
// renders the 3D at the output's own 3840x2160 (an RTX 5090 at 1-1 standing: 68 -> 55 fps, GPU-bound).
// Unset or 1440: the game as it is. The game's own 1080p output modes (r_Wants4K 1 and 2) are no option: their UI
// keeps the 4K layout (off screen). With the game's super resolution path on (KYTY_UPSCALER,
// loader/demonsSoulsSuperResolution.h) the performance mode is display mode 5: the height sizes its 3D instead
// (unset: the PS5 Pro's 2688x1512; at least 1080), the upscaler making the 3840x2160 image from it.
// KYTY_RENDER_ASPECT=<w>:<h> (run-windows.ps1 -Aspect, the launcher's "Aspect ratio"; 21:9 is taken as 64:27): the 3D
// rendered wider than 16:9, up to 32:9, at the render height - 3416x1440 for 21:9 - and its projection's aspect (the
// render setup's 16:9, AspectStore) to match: more of the world to the sides, not a stretched 16:9 view. The game's
// output stays 3840x2160: the wider picture is squeezed into it, the HUD drawn over it as at 16:9, and the presenter
// widens the frames showing the 3D back to the aspect (movies, menus and loading screens stay 16:9: WideAspect()). The
// HUD's elements keep their shape at their sides of the screen (src/local/hud-anchor.h); the in-game menus and the
// markers over the world (lock-on, enemy health, on their targets) are as much wider. (The game's output made as wide,
// 5120x2160, laid its HUD and menus out at 16:9 against its left edge.)
// With the super resolution path (DLSS) and a wider aspect the game's output is as wide too, at 2160 - 5120x2160 at 21:9
// - and its upscaler makes that (KYTY_WIDE_OUTPUT=0: the 3840x2160 output, the 3D squeezed into it, as without DLSS).
// The game finishes its frame at that size (bloom, depth of field, the HUD); it lays its interface out at 3840x2160
// against the left edge, which src/local/hud-anchor.h places (the HUD at its sides, the rest widened as before).
namespace DemonsSoulsRenderSize {
void Install(Program* program);
// The aspect the 3D is rendered at when KYTY_RENDER_ASPECT took effect (Install), else 0 (the game's 16:9).
[[nodiscard]] double WideAspect();
// The game's output's width when the wide output took effect (Install), else 0 (the game's 3840).
[[nodiscard]] uint32_t WideOutputWidth();

// The video settings' mode 3 case (the setter's and the constructor's): the output size, then the render size, as
// RIP-relative pairs of 32-bit width and height (vmovsd), then a short jump (01.007.000); 01.005.000 loads the same
// pairs with vmovddup (0x8eea7b, 0x8ef5e5), which found no case there before (issue #69: its render height kept).
inline constexpr std::string_view ModeCase           = "c5 fb 10 0d ?? ?? ?? ?? c5 fb 10 05 ?? ?? ?? ?? eb";
inline constexpr std::string_view ModeCaseDuplicated = "c5 fb 12 0d ?? ?? ?? ?? c5 fb 12 05 ?? ?? ?? ?? eb";
inline constexpr uint32_t         OutputWidth = 3840, OutputHeight = 2160, RenderWidth = 2560, RenderHeight = 1440;
// Its mode 5 case (the PS5 Pro's, with super resolution: loader/demonsSoulsSuperResolution.h): the same pairs, then
// the super resolution flag set, then the jump (01.007.000).
inline constexpr std::string_view TrinityCase = "c5 fb 10 0d ?? ?? ?? ?? c5 fb 10 05 ?? ?? ?? ?? c6 45 ?? 01 eb";
inline constexpr uint32_t         TrinityWidth = 2688, TrinityHeight = 1512;
// The least render height the game's super resolution runs at (SuperResolution::IsEnabled).
inline constexpr uint32_t SuperResolutionLeastHeight = 1080;
// The wide output's places (01.007.000, DemonsSoulsRenderSize.cpp InstallWideOutput):
// - display mode 4's cases (Cinematic), loading its 3D size and output from the modes' shared 3840x2160 output pair:
//   vmovsd xmm0, [rip + output]; jmp; and the next case's vmovsd xmm1 (0x90090a66d, 0x90090b1d2; mode 3's look alike
//   but load another pair), and the 3840x2160 they load instead: in the mode 3 preset (0x90237f260);
// - the upscaler's context, SuperResolution::InitContext(3840, 2160): mov esi, 3840; mov edx, 2160; call (0x900a521eb);
// - the video settings records' fixed 3840x2160 (mov r64, 2160 << 32 | 3840) that sizes the upscaler's image and the
//   processing after it, in the setter (after vmovaps xmm1, xmm0: 0x90090a6ab), the constructor (after mov r14,
//   [rip + x]: 0x90090b217) and the size query (after mov [rsi + 4], edi: 0x90090ac3f);
// - the per-mode output widths after the render size tables (ModeWidths).
inline constexpr std::string_view CinematicCase        = "c5 fb 10 05 ?? ?? ?? ?? eb ?? c5 fb 10 0d";
inline constexpr uint32_t         CinematicSource[8]   {0x80ffffffu, 2560, 1440, 3840, 2160, 3840, 2160, 1024};
inline constexpr std::string_view UpscalerContext      = "be 00 0f 00 00 ba 70 08 00 00 e8";
inline constexpr std::string_view SetterFixedPair      = "c5 f8 28 c8 48 b9 00 0f 00 00 70 08 00 00 48 ba";
inline constexpr std::string_view ConstructorFixedPair = "4c 8b 35 ?? ?? ?? ?? 48 b9 00 0f 00 00 70 08 00 00";
inline constexpr std::string_view QueryFixedPair       = "89 7e 04 48 ba 00 0f 00 00 70 08 00 00 48 bf";
inline constexpr uint32_t         OutputWidths[5] {1920, 1920, 3840, 3840, 3840};
// The render size query's per-mode tables (modes 1-5): widths, heights.
inline constexpr uint32_t ModeWidths[5] {1600, 1920, 2560, 3840, 2688};
inline constexpr uint32_t ModeHeights[5] {900, 1080, 1440, 2160, 1512};
inline constexpr int      PerformanceMode = 2, TrinityMode = 4; // (indexes in the tables)
// The render setup's constructor storing the projection's aspect, 16/9 as a float: mov dword [rbx+4], 0x3fe38e39
// (01.007.000 0x90090ae7f, 01.005.000 0x9008ef287; the only such store either build has).
inline constexpr std::string_view AspectStore = "c7 43 04 39 8e e3 3f";
inline constexpr double           GameAspect  = 16.0 / 9.0;
inline constexpr double           WidestAspect = 32.0 / 9.0;

// The render size for a height and an aspect, its width a multiple of 8; {0, 0} for a height out of range.
struct Size {
	uint32_t width  = 0;
	uint32_t height = 0;
};
inline Size RenderSizeFor(uint32_t height, double aspect = GameAspect) {
	if (height < 720 || height > OutputHeight) return {};
	const auto width = static_cast<uint32_t>(std::lround(height * aspect / 8.0)) * 8u;
	return {width, height};
}

// Whether a 3D of `size` fits the game's render target memory, made for its own largest 3D, the Cinematic mode's
// 3840x2160: its targets laid out in the GPU's 64 KiB tiles (at 2, 4, 8 and 16 bytes a pixel: 256x128, 128x128, 128x64
// and 64x64 pixels) take no more tiles than at 3840x2160. A target taking more writes past its memory into what
// follows: 4416x1864 at 21:9 (525 tiles at 4 bytes, 3840x2160 510) showed the 3D in red only, 4776x2016 and 32:9's
// 5760x1620 in yellow with black models, 4952x2088 and 5120x2160 ended the game at start-up (eboot+0x803e3a, a tree
// of its own whose node was overwritten); 4264x1800 (510 tiles) and 5120x1440 are right (the RTX 5090, 01.007.000, the
// game's TAA). The pixels alone are no measure: 4416x1864 has fewer than 3840x2160.
inline bool FitsRenderTargets(Size size) {
	constexpr uint32_t tiles[4][2] {{256, 128}, {128, 128}, {128, 64}, {64, 64}};
	for (const auto& tile: tiles) {
		const auto count = [&](uint32_t width, uint32_t height) {
			return static_cast<uint64_t>((width + tile[0] - 1) / tile[0]) * ((height + tile[1] - 1) / tile[1]);
		};
		if (count(size.width, size.height) > count(OutputWidth, OutputHeight)) return false;
	}
	return true;
}
// The highest render height (a multiple of 8) whose size at `aspect` fits (FitsRenderTargets): 2160 at 16:9, 1832 at
// 21:9 (4344x1832), 1512 at 32:9 (5376x1512).
inline uint32_t MaxRenderHeight(double aspect) {
	for (uint32_t height = OutputHeight; height >= 720; height -= 8) {
		if (FitsRenderTargets(RenderSizeFor(height, aspect))) return height;
	}
	return 720;
}

// KYTY_RENDER_ASPECT's value: "<w>:<h>" or a number; 21:9 is the displays' 64:27. 0 when unset, at most 16:9 or wider
// than 32:9 (wider is untested: the game's targets were made for 3840 at most).
inline double ParseAspect(std::string_view text) {
	if (text == "21:9") return 64.0 / 27.0;
	double     w = 0.0, h = 1.0;
	const auto colon = text.find(':');
	const auto number = [](std::string_view part) {
		double value = 0.0, scale = 0.0;
		for (const char c: part) {
			if (c == '.' && scale == 0.0) {
				scale = 1.0;
			} else if (c >= '0' && c <= '9') {
				if (scale == 0.0) {
					value = value * 10.0 + (c - '0');
				} else {
					scale /= 10.0;
					value += (c - '0') * scale;
				}
			} else {
				return -1.0;
			}
		}
		return part.empty() ? -1.0 : value;
	};
	if (colon == std::string_view::npos) {
		w = number(text);
	} else {
		w = number(text.substr(0, colon));
		h = number(text.substr(colon + 1));
	}
	if (w <= 0.0 || h <= 0.0) return 0.0;
	const double aspect = w / h;
	return aspect > GameAspect + 0.01 && aspect <= WidestAspect + 0.01 ? aspect : 0.0;
}
} // namespace DemonsSoulsRenderSize
} // namespace Loader
