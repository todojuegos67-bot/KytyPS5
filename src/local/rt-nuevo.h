#ifndef EMULATOR_SRC_LOCAL_RT_NUEVO_H_
#define EMULATOR_SRC_LOCAL_RT_NUEVO_H_
// (rtq2 b, compilaciones 1-2) KYTY_RTN=0|1|2 ("Reflejos RT nuevos": No / Si / Depuracion; default 0: nothing created,
// copied or recorded). Hardware ray-traced reflections restricted to where the game itself reflects, built in two steps:
//  this one (1) only TRACES and measures - the picture is never changed unless KYTY_RTN=2 shows the debug view;
//  the next (2) will composite over the game's specular IBL.
// Where the game reflects: the game's own SSR tile mask (kRTT_SSR tiles, 16x16 render pixels; an R8_UINT input of the IBL
// CS 9fb5cc274b5e4f82 written by its tile classifier bb1fd2419da34939, which already applies the stencil bit 0x40 the IBL
// tests), copied right after the IBL dispatch (frame-gen.cpp RtNuevoTilesAfterDispatch).
// The ray: from the depth buffer's point, along reflect(view, N) with N the G-buffer's GEOMETRIC normal where the material
// has one (layer 3: 8+8 octahedral for class 0 + flag bit 0 and class 2, 12+12 for class 1; as the IBL decodes it, pc
// 0x740 / 0x8bc / 0x95c), else the shading normal (layer 1, 12+12): coherent with "Suelo estable". One ray query per
// trace pixel (half the render size: 960x540 at 1920x1080) into the KYTY_HWRT scene TLAS of THIS frame (Upscale runs after
// the TLAS build: no frame of delay).
// Checks every 10 s ("[rtn]"): mask share, hit share, the decoded normal against the depth's normal (agreement: proves the
// G-buffer decoding and the world space), the GPU ms of the pass (timestamps).
#include "local/frame-gen.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {
class TextureCache;
struct TextureBinding;
namespace FrameGen {

[[nodiscard]] int  RtNuevoLevel();
[[nodiscard]] bool RtNuevoWanted();
// frame-gen.cpp, after the IBL dispatch (RtReflGuideAfterDispatch): the game's SSR tile mask copied (this flip).
void RtNuevoTilesAfterDispatch(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash,
                               std::span<const TextureBinding> images, uint64_t written_mask);
// The copy of this flip or the last (R8_UINT, shader-read-only layout): false when none.
[[nodiscard]] bool RtNuevoGetTiles(vk::ImageView& view, vk::Extent2D& extent);
// In Upscale (after the other passes, before the upscaler): trace + temporal + spatial (the result for the next frame's
// inject); with KYTY_RTN=2, inputs.color becomes the debug overlay.
bool RtNuevoApply(vk::CommandBuffer command, UpscaleInputs& inputs);
// (compilacion 2) Whether a result waits for the next IBL dispatch.
[[nodiscard]] bool RtNuevoInjectReady();
// frame-gen.cpp, right after the IBL dispatch, with the game's IBL output image (RGBA16F, storage) already in GENERAL:
// the result blended into it (the game then composites it with its own BRDF). False: nothing recorded (logged).
bool RtNuevoInject(GraphicContext& graphics, vk::CommandBuffer command, vk::Image image, vk::Format format, vk::ImageUsageFlags usage,
                   vk::Extent2D extent, uint32_t level, uint32_t layer);

} // namespace FrameGen
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_RT_NUEVO_H_
