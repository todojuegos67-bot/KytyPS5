#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {
struct GraphicContext;
namespace Pssr {
struct Request;
}
} // namespace Libs::Graphics

// AMD FidelityFX Super Resolution 3.1 for PSSR, on any GPU (FidelityFX SDK 1.1.4, MIT,
// 3rdparty/FidelityFX-SDK). KYTY_FSR3=0 turns it off: PSSR then scales linearly.
namespace Libs::Graphics::Fsr3 {

struct Context;

// A whole image FSR reads or writes: one layer, level 0. FSR's backend makes its own views of it
// and returns it to the general layout after each dispatch.
struct Image {
	VkImage  image;
	VkFormat format; // the view's format; a depth image's own
	uint32_t width;  // the output's: the upscaled size
	uint32_t height;
	bool     depth;
	bool     stencil; // a depth image with a stencil aspect
};

struct Inputs {
	Image        color;
	Image        depth;
	Image        motion;
	const Image* exposure; // null when absent
	Image        output;
};

// What a context fixes at creation.
struct Creation {
	uint32_t max_width; // the colour image, which holds every render size
	uint32_t max_height;
	uint32_t output_width;
	uint32_t output_height;
	uint32_t flags; // Pssr::Flags fixed at creation

};

// Whether FSR takes the request: temporal inputs at the image origin.
[[nodiscard]] bool Accepts(const Pssr::Request& request);
[[nodiscard]] bool Supports(VkFormat format);
[[nodiscard]] Creation CreationFor(const Pssr::Request& request, const Inputs& inputs);
// Builds the context's pipelines and history; null, with a log line, when FSR can't start.
[[nodiscard]] Context* Create(const GraphicContext& graphics, const Creation& creation);
// Records the upscale into command; false, with a log line, when FSR failed.
[[nodiscard]] bool Dispatch(Context* context, vk::CommandBuffer command,
                            const Pssr::Request& request, const Inputs& inputs, bool reset);
// Once the GPU is done with the context's dispatches. Takes null.
void Destroy(Context* context);

} // namespace Libs::Graphics::Fsr3

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR3_H_
