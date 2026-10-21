#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#include "SDL_events.h"
#include "SDL_video.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

// KYTY_EXCLUSIVE_FULLSCREEN=1 (Windows): classic exclusive full screen through VK_EXT_full_screen_exclusive instead of
// the borderless desktop full screen (swapchain.cpp). The window is the same borderless popup over its monitor; the
// swapchain asks the driver for the display itself (application controlled), released while the window has no focus.
namespace ExclusiveFullscreen {
[[nodiscard]] bool Requested();
// Set by the window's events (main thread), read by the presenter thread: focused and not minimized.
inline std::atomic_bool g_wanted {true};
} // namespace ExclusiveFullscreen

struct WindowLoopState {
	SDL_Event        event {};
	bool             need_exit = false;
	std::atomic_bool paused    = false;
};

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void                                                    UpdateTitle();
	void                                                    Resize(uint32_t width, uint32_t height);
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();
	// SDL window operations must complete on the main thread.
	void RunOnMainThread(std::function<void()> task);
	void DrainMainThreadTasks();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	bool                           full_screen_exclusive_ext = false; // VK_EXT_full_screen_exclusive enabled
	bool                           surface_capabilities2_ext = false; // VK_KHR_get_surface_capabilities2 enabled
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;

	Common::Mutex mutex;

	Common::Mutex                      main_task_mutex;
	Common::CondVar                    main_task_done;
	std::vector<std::function<void()>> main_tasks;            // guarded by main_task_mutex
	uint64_t                           main_tasks_queued = 0; // guarded by main_task_mutex
	uint64_t                           main_tasks_run    = 0; // guarded by main_task_mutex
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
