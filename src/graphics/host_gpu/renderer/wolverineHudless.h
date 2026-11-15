#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WOLVERINEHUDLESS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WOLVERINEHUDLESS_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cmdRecorder.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <functional>
#include <span>

// Marvel's Wolverine: the HUD-less image for frame generation (59, KYTY_FSR3_FG=1). The game draws its HUD into a UI
// layer of its own; one fullscreen draw at the end of each frame merges the scene and that layer into the flip buffer.
// Before that draw, the merge is run once more with the UI layer cleared, and its result kept: the frame as it would
// be without the HUD. The UI layer is then put back and the game's draw runs as before. The copy goes with the flip,
// and frame generation takes it as the HUD-less colour, so the HUD comes from the real frame (59-fsr-framegen.md).
namespace Libs::Graphics {
class CommandBuffer;
struct RenderColorInfo;
struct TextureBinding;
} // namespace Libs::Graphics
namespace Libs::Graphics::ShaderRecompiler::IR {
struct ImageResource;
} // namespace Libs::Graphics::ShaderRecompiler::IR

namespace Libs::Graphics::WolverineHudless {

// Renderer, before each draw while frame generation is on. `draw` records the same draw again (the HUD-less merge).
void BeforeDraw(CommandBuffer& buffer, const ShaderStageRuntime& pixel, const RenderColorInfo& target,
                const std::function<void()>& draw);
// Renderer, before each dispatch while frame generation is on, with the dispatch's bound images (descriptors
// committed, pipeline bound). `dispatch` records the same dispatch again (the HUD-less merge).
void BeforeDispatch(CommandBuffer& buffer, uint64_t hash, std::span<const ShaderRecompiler::IR::ImageResource> resources,
                    std::span<const TextureBinding> images, const std::function<void()>& dispatch);
// Presenter, as a flip is copied from `surface` in `command`: the HUD-less copy taken of it since its last flip goes
// into `into`, made its size and format. False when there is none.
[[nodiscard]] bool TakeFor(Cmd command, GraphicContext& graphics, const Image& surface, VulkanImage& into);

// Makes `image` a 2D colour image of that size, format and usage (again when they differ).
void Ensure(GraphicContext& graphics, VulkanImage& image, vk::Extent2D extent, vk::Format format,
            vk::ImageUsageFlags usage);
// One barrier from the image's tracked state.
void Transit(Cmd command, VulkanImage& image, vk::ImageLayout layout, vk::AccessFlags2 access);

} // namespace Libs::Graphics::WolverineHudless

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WOLVERINEHUDLESS_H_
