#ifndef EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
#define EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
// DLSS-G's HUD-less color (KYTY_FG_HUDLESS=1, off by default): the game image without its interface, so that the
// generated frames do not drag the HUD along with the scene behind it. Demon's Souls draws its final image into the
// flip image with one draw (the composite) that samples the HUD layer and the 3D scene from separate textures. On the
// renderer thread that draw is recorded once more just before it, with the HUD texture cleared to transparent; the
// result is copied aside, and the flip image and the HUD texture are put back as they were, so the game's own draw
// follows unchanged. The copy goes with the presented frame (FrameInputs::hudless); the present thread lays it out as
// the backbuffer (ScaleHudless), which BeforePresent tags as kBufferTypeHUDLessColor.
//
// KYTY_FG_HUDLESS_UI=n: the HUD texture is the composite's candidate n (the log lists them), not the one its pixel
// shader or the guess gives.
// KYTY_FG_HUDLESS_DEBUG=1: the HUD texture is left cleared, so the screen shows what DLSS-G gets (the HUD gone).
#include "frame-gen.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {
struct DrawRenderState;
struct RenderColorInfo;
struct RenderState;
struct TextureBinding;
namespace FrameGen {

// KYTY_FG_HUDLESS=1 with DLSS-G ready (InitializeHudless).
extern bool g_hudless;
[[nodiscard]] inline bool HudlessOn() noexcept {
	return g_hudless;
}
// ... and the current guest frame shows the game's 3D view (SceneNoted): its composite is looked for.
[[nodiscard]] inline bool HudlessCapturing() {
	return g_hudless && SceneNoted();
}

// OnDevice, once DLSS-G is ready: the switches, read once.
void InitializeHudless();

// Renderer thread (Kyty.Gpu), normal draw path, while HudlessCapturing(), not in a speculative translation. FlipSlot:
// the slot of `targets` that is a recent flip image while this frame's composite is still to come (-1: none). Such a
// draw is recorded directly and stored for neither the table nor the native XPR path. BeforeFlipDraw, with everything
// of the draw (`state`) bound and before its render pass begins, when the frame has DLSS-G's inputs so far
// (InputsGathered): `redraw` (the draw) once more without the HUD texture (one of `images`), its result kept, then the
// flip image and the HUD texture as they were.
[[nodiscard]] int FlipSlot(std::span<const RenderColorInfo> targets);
// (BeforeFlipDraw's halves: true when the HUD texture is cleared and the draw's pass begun for `redraw`.)
[[nodiscard]] bool BeginHudlessDraw(CommandBuffer& command, const RenderState& rendering, const DrawRenderState& state,
                                    uint32_t slot, std::span<const TextureBinding> images);
void EndHudlessDraw(CommandBuffer& command, const RenderState& rendering);
template <typename Redraw>
void BeforeFlipDraw(CommandBuffer& command, const RenderState& rendering, const DrawRenderState& state, uint32_t slot,
                    std::span<const TextureBinding> images, Redraw&& redraw) {
	if (BeginHudlessDraw(command, rendering, state, slot, images)) {
		redraw();
		EndHudlessDraw(command, rendering);
	}
}
// Any translating thread (the table and native XPR paths), while HudlessCapturing(): `address` is a recent flip
// image's and this frame's composite is still to come (the draw is left to the normal path).
[[nodiscard]] bool IsFlipTarget(uint64_t address);

// At the guest flip of `flip`, after PrepareInputs: the composite's HUD-less image goes with `inputs`.
void PrepareHudless(FrameInputs& inputs, const Image& flip);
// Present thread, in the present's command buffer after RecordPresentCommands: the frame's HUD-less image laid out as
// the backbuffer (`region`: where the game image is, black around it), for BeforePresent's tag.
void ScaleHudless(GraphicContext& graphics, vk::CommandBuffer command, FrameInputs& inputs, vk::Extent2D backbuffer,
                  vk::Rect2D region, vk::Format format);
// A frame's HUD-less images (DestroyInputs); at shutdown, the capture's own.
void DestroyHudlessInputs(GraphicContext& graphics, FrameInputs& inputs);
void DestroyHudless(GraphicContext& graphics);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_FRAME_GEN_HUDLESS_H_
