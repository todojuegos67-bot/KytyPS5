#ifndef EMULATOR_SRC_LOCAL_SCREENSHOT_H_
#define EMULATOR_SRC_LOCAL_SCREENSHOT_H_
// KYTY_LIVE_FILE `screenshot <path>` (live-control.h): the next frame the presenter shows, as a binary PPM file, read
// back from the GPU, so another window over the emulator's (or none at all) does not matter. The game's picture as the
// presenter got it (its own size, before the scaling to the window); with frame generation shown by the presenter
// (Linux) the finished backbuffer, and the generated frame before it as <path>.gen.ppm. A "SCREENSHOT <path> <w>x<h>"
// line on stdout says when a file is written.
#include "graphics/host_gpu/graphicContext.h"

#include <string>

namespace Libs::Graphics::Screenshot {

// Any thread.
void Request(const std::string& path);
// Any thread: the next `count` frames shown, each box-averaged to SeriesWidth pixels wide, appended to one file at
// `path` ("KSER <w> <h> <count>\n", then each frame's RGB bytes). For a fixed camera, a frame that differs from its
// neighbours is a wrong frame (tools: the flicker check). A "SCREENSHOTS <path> <frames>" line when written.
void RequestSeries(const std::string& path, uint32_t count);
inline constexpr uint32_t SeriesWidth = 320;
// Present thread: the path asked for, once (empty: none).
[[nodiscard]] std::string Take();
// Take()'s path is a frame of a series (RequestSeries): one frame shown, not a file of its own (no ".gen.ppm" beside it).
[[nodiscard]] bool IsSeriesFrame(const std::string& path);

struct Pending {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::Format    format = vk::Format::eUndefined;
	vk::Extent2D  extent {};
	std::string   path;
};
// Present thread: records the copy of `image` (color, one level) into a host buffer, the image left in its layout.
[[nodiscard]] Pending Record(GraphicContext& graphics, vk::CommandBuffer command, VulkanImage& image, const std::string& path);
// After the GPU ran the commands: the file written, the buffer freed.
void Finish(GraphicContext& graphics, Pending& pending);

} // namespace Libs::Graphics::Screenshot

#endif // EMULATOR_SRC_LOCAL_SCREENSHOT_H_
