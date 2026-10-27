#ifndef EMULATOR_SRC_LOCAL_GBUF_VIEW_H_
#define EMULATOR_SRC_LOCAL_GBUF_VIEW_H_
// KYTY_GBUF_VIEW=<view> (gbuf-view.cpp, gbuf-view.comp; refl7): a diagnostic of Demon's Souls' packed G-buffer before the
// upscaler (KYTY_UPSCALER: DLSS / XeSS / FSR). Views: L<layer>.b<byte> (e.g. L2.b0), <layer>:<shift>:<bits> (any bit
// field, e.g. 2:0:8), mosaic (all 16 byte fields, rows = layers 0..3, columns = bytes 0..3), raw (the 4 words' low 3 bytes
// as RGB, one quadrant per layer), normal (layer 3 oct 8:8 as color), forward (pixels drawn after the G-buffer, e.g.
// water: cyan), tap (KYTY_GBUF_TAP's image), stats (the game's image unchanged; statistics only). A red crosshair marks
// the probe pixel (the centre), whose raw words are logged. Every 5 s one "GBUF:" block: per byte field its mean,
// spread, 16-bin histogram, its correlation with the game's color (with the game's own debug view, e.g.
// KYTY_GAME_CONVARS_EXTRA=ShowRenderTarget=Roughness, the color is the engine's decoded field) and its mean on forward
// pixels; the probe pixel's words and depths. The view can be changed while running: the first line of
// kyty-gbuf-view.txt in the working directory, checked every second.
// KYTY_GBUF_TAP=<cs hash>[:<slot>] (frame-gen.cpp GbufTapAfterDispatch): after that dispatch, its written image (the
// slot, or the first written one) copied for the tap view: the game's own SSR / specular IBL results, per pixel.
// With either set, every dispatch that reads the G-buffer logs once which of its images it writes (its outputs).
// Off (the default): nothing created, nothing recorded; the hooks cost one relaxed load.
#include "local/frame-gen.h"

namespace Libs::Graphics::FrameGen {

[[nodiscard]] bool GbufViewWanted();
// In Upscale, last before the upscaler: inputs.color becomes the view (stats only: unchanged). False: nothing done.
bool GbufViewApply(vk::CommandBuffer command, UpscaleInputs& inputs);

// frame-gen.cpp: KYTY_GBUF_TAP's copy (this flip or the last).
struct GbufTapImage {
	vk::ImageView view = nullptr;
	vk::Extent2D  extent {};
	bool          integer = false; // (an integer format: shown as its first channel's value / 255)
};
[[nodiscard]] bool GbufTapGet(GbufTapImage& out);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_GBUF_VIEW_H_
