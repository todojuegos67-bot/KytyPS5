#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_

// (ram1b) KYTY_RAM_NATIVE_TEX=1|2 (local/ram-native-tex.h): a texture array of the game's VirtualMemoryGPU zone as a
// sparse (partially resident) image, video memory bound to the layers the game has committed only. See sparseImage.cpp.

#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct ImageInfo;

// Whether an image of `info`, created as `create`, is made sparse here: the switch on, the device able, a sampled 2D
// colour array of the zone (no depth/stencil, not multisampled, its guest layers of equal stride), made on the GPU thread
// (the bindings wait on the GPU thread's scheduler), and no ImageForceDense scope open.
[[nodiscard]] bool SparseImageWanted(const GraphicContext& graphics, const ImageInfo& info, const vk::ImageCreateInfo& create);

// The image made sparse, no memory bound yet (false: nothing made; the caller makes it dense). `image` filled as
// GraphicContext::CreateImage fills it, its `sparse` set.
[[nodiscard]] bool CreateSparseImage(GraphicContext& graphics, const vk::ImageCreateInfo& create, VulkanImage& image);

// The image and all its video memory freed (GraphicContext::DeleteImage: the GPU is done with it).
void DestroySparseImage(GraphicContext& graphics, VulkanImage& image);

// The layers of `image` whose guest slices [address + layer * stride, + stride) are committed now - minus
// [gone_begin, gone_end), about to be unmapped - bound, the others unbound, for the layers whose slices meet
// [begin, end). The binding is queued on the graphics queue; the next submission of `scheduler` waits for it.
void SparseUpdateLayers(GraphicContext& graphics, CommandScheduler& scheduler, VulkanImage& image, uint64_t address,
                        uint64_t stride, uint64_t begin, uint64_t end, uint64_t gone_begin, uint64_t gone_end);

// (=2) The layers bound now (ascending).
void SparseBoundLayers(const VulkanImage& image, std::vector<uint32_t>* layers);

// While one lives on this thread, SparseImageWanted says no (the verification's dense reference image).
struct ImageForceDense {
	ImageForceDense();
	~ImageForceDense();
	ImageForceDense(const ImageForceDense&)            = delete;
	ImageForceDense& operator=(const ImageForceDense&) = delete;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SPARSEIMAGE_H_ */
