#ifndef EMULATOR_SRC_LOCAL_FG_DISCOVER_H_
#define EMULATOR_SRC_LOCAL_FG_DISCOVER_H_
// bruno-astro-1025: KYTY_FG_DISCOVER=1, finding another title's passes for frame generation and the upscalers (the
// todojuegos67-bot fork's discovery captures, reduced to their log summary: no _DebugSnapshots files on this tree).
// Two guest frames are summarized KYTY_FG_DISCOVER_AT seconds after the player's first button (default 20,60,180),
// as "[fgdisc]" lines in the log:
// - the flip image (address, format, size);
// - compute dispatches with at least three images: shader hash, each image's slot, guest address, format, size and
//   use (t sampled, s storage), depth-like (D) and motion-like (M: two-channel 16/32-bit) images marked;
// - TAA-like dispatches (a depth-like and a motion-like image of the size of a storage image they write), with the
//   per-title switches to try (KYTY_FG_TAA_SHADERS, KYTY_FG_TAA_SLOTS=color,depth,motion,history,output);
// - render targets by draw count, and every draw into the flip image with the images it samples (the composite: the
//   scene and the HUD layer; the vertex shader hashes for KYTY_FG_GBUFFER_VS).
// Off: one relaxed load per call.
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics::FgDiscover {

[[nodiscard]] bool Enabled();
// Renderer thread: the current guest frame is being summarized.
[[nodiscard]] bool Recording();
// Renderer thread, at the guest flip (Presenter::PrepareFrame): the frame ends (its summary is printed when it was
// recorded) and the next one may begin recording.
void OnFlip(uint64_t address, uint32_t width, uint32_t height, uint32_t format, bool player_button_seen);
void OnDispatch(uint64_t cs_hash, std::span<const TextureBinding> images, uint32_t groups_x, uint32_t groups_y);
struct Target {
	uint64_t address = 0;
	uint32_t format = 0, width = 0, height = 0;
};
void OnDraw(uint64_t vs_hash, uint64_t ps_hash, std::span<const Target> colors, const Target& depth,
            std::span<const TextureBinding> sampled);

} // namespace Libs::Graphics::FgDiscover

#endif // EMULATOR_SRC_LOCAL_FG_DISCOVER_H_
