// Frame injector: shared declarations.
//
// The injector is a proxy vulkan-1.dll placed next to a game's exe. A DX11 game goes through DXVK (d3d11.dll,
// dxgi.dll next to the exe), which loads vulkan-1.dll from the exe's folder first: ours. A native Vulkan game that
// loads vulkan-1.dll by name also gets ours. We forward everything to the system loader
// (%SystemRoot%\System32\vulkan-1.dll) and intercept a few functions (device creation, queues, swapchain, present,
// image creation and command buffer recording) for the overlay, the live analyzer and FSR 3 frame generation.
//
// Nothing here is game specific: per-game knowledge lives only in profiles written by the live analyzer
// (injector_profiles\<exe name>.ini, resource signatures: format + size + usage, never handles, offsets or hashes).
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <string>

namespace inj {

// ---- log (injector.log next to the exe) ----
void Log(const char* fmt, ...);
std::wstring ExeDir();   // with a trailing backslash
std::string  ExeName();  // "GRW.exe" -> "GRW"

// ---- settings (injector.ini next to the exe), changed from the menu (F2 by default) ----
struct Settings {
	bool  fg_enabled     = true;   // FSR 3 frame generation x2
	int   fg_cap_fps     = -1;     // shown-frame cap for VRR: -1 auto (refresh - refresh^2/3600), 0 off, N fps
	bool  osd            = true;   // small FPS line in the corner
	bool  analyze        = true;   // live analyzer ("Analizar vectores")
	bool  use_game_inputs = true;  // feed the analyzer's depth / motion vectors to FG once confident
	float min_confidence = 0.60f;
	int   mv_sign        = -1;     // FSR wants pixels toward the previous position: scale = sign * size
	int   mv_units       = 0;      // 0: UV (scale by render size), 1: pixels (scale 1)
	int   depth_mode     = 0;      // 0 auto (from the depth clear value), 1 normal, 2 reversed
	bool  previews       = true;
	int   menu_key       = VK_F2;  // injector.ini menu_key / fg_key: a name (F1..F12, Insert, Home, End, ...) or a
	int   fg_key         = VK_F3;  // virtual-key number (0x71 or 113)
};
Settings&   Cfg();
void        LoadSettings();
void        SaveSettings();
const char* KeyName(int vk);

// ---- dispatch to the real loader / driver ----
struct Real {
	HMODULE                   loader = nullptr;
	PFN_vkGetInstanceProcAddr gipa   = nullptr;
	PFN_vkGetDeviceProcAddr   gdpa   = nullptr;
	VkInstance                instance = VK_NULL_HANDLE;
	VkPhysicalDevice          physical = VK_NULL_HANDLE;
	VkDevice                  device   = VK_NULL_HANDLE;
	uint32_t                  api_version = VK_API_VERSION_1_1;

#define INJ_FN(name) PFN_##name name = nullptr;
#include "device_functions.inl"
#undef INJ_FN
};
Real& R();
void  LoadDeviceFunctions(VkDevice device);
PFN_vkVoidFunction RealDeviceProc(const char* name);  // for FidelityFX and ImGui: never our hooks

// One lock for every submit / present on any queue we share with the game (vkQueue* are externally synchronized;
// FidelityFX submits on the game queue from its own thread through our callback). Recursive: our present path
// submits while holding it.
std::recursive_mutex& QueueLock();

// The queues added at device creation for frame generation (VK_NULL_HANDLE when unavailable).
struct Queues {
	uint32_t graphics_family = UINT32_MAX;
	VkQueue  game            = VK_NULL_HANDLE;  // learned at the first present (the queue the game presents on)
	uint32_t game_family     = UINT32_MAX;
	VkQueue  present         = VK_NULL_HANDLE;  // graphics family, an extra index
	uint32_t compute_family  = UINT32_MAX;
	VkQueue  async_compute   = VK_NULL_HANDLE;  // compute family, extra index 0
	VkQueue  image_acquire   = VK_NULL_HANDLE;  // compute family, extra index 1
	bool     fg_queues_ok    = false;
};
Queues& Q();

HWND& GameWindow();
uint32_t FindMemoryType(uint32_t bits, VkMemoryPropertyFlags flags);
double   DisplayRefreshHz();

} // namespace inj
