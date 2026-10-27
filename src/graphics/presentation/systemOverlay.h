#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

union SDL_Event;

namespace Libs::Graphics {

struct GraphicContext;

struct SystemOverlayVisualState {
	bool     active;
	uint64_t revision;
};

// KYTY_FPS_HUD's frame-time graph (empty frame_ms: none) and the detailed panel's rows (empty: none).
struct SystemOverlayPerf {
	std::vector<float> frame_ms; // the last frames' times, oldest first
	std::vector<float> busy_ms;  // the render thread's busy part of each (empty: not drawn)
	std::string        graph_note;
	struct Row {
		std::string label, value;
	};
	std::vector<Row>     rows;
	std::vector<float>   core_load;  // per logical processor (0..1), bars under the processor's row (empty: none)
	std::vector<uint8_t> core_fast;  // 1: a performance core of a hybrid processor
	std::string          core_note;  // the clocks, beside the bars
};

// Small panels over the game image, drawn with every frame (dialog or not): the frame rate
// (KYTY_FPS_HUD), the progress of background work (the shader prefetch) and the debug warp.
struct SystemOverlayHud {
	std::string title;  // large first line (empty: no frame rate panel)
	std::string detail; // small second line
	SystemOverlayPerf perf; // under them
	std::string status; // the progress line (empty: none), over a bar
	float       status_fraction = 0.0f;
	std::string notice; // a text panel (empty: none)
	// KYTY_RTLS_DEBUG (rt-local-lights.h): markers over the game image, in swapchain pixels (shape 0 circle, 1 cross,
	// 2 text only; rgba as IM_COL32).
	struct Marker {
		float       x = 0, y = 0, radius = 0;
		uint32_t    rgba  = 0xffffffffu;
		int         shape = 0;
		std::string label;
	};
	std::vector<Marker> markers;
	vk::Rect2D  region; // the game image in the swapchain image
	vk::Rect2D  drawn;  // out: the panels, in swapchain pixels
};

void                     InitializeSystemOverlayInput();
void                     ShutdownSystemOverlayInput();
bool                     ProcessSystemOverlayInput(const SDL_Event& event);
SystemOverlayVisualState GetSystemOverlayVisualState() noexcept;

class SystemOverlay final {
public:
	explicit SystemOverlay(GraphicContext& graphics);
	~SystemOverlay();
	KYTY_CLASS_NO_COPY(SystemOverlay);

	[[nodiscard]] bool PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t image_count,
	                                SystemOverlayHud* hud = nullptr);
	void               Record(vk::CommandBuffer command, vk::ImageView target);
	void               ReleaseVulkan();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEMOVERLAY_H_
