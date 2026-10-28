// The proxy vulkan-1.dll: exports, the real loader, the hooks, and the present path.
#include "analyzer.h"
#include "camera.h"
#include "common.h"
#include "framegen.h"
#include "overlay.h"
#include "upscale.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace inj {

// ---------------------------------------------------------------- log, settings, globals
namespace {
std::mutex g_log_mutex;
FILE*      g_log = nullptr;
Settings   g_settings;
Real       g_real;
Queues     g_queues;
HWND       g_hwnd = nullptr;
std::recursive_mutex g_queue_lock;
std::once_flag g_init_once;
}  // namespace

std::wstring ExeDir() {
	wchar_t path[MAX_PATH] {};
	GetModuleFileNameW(nullptr, path, MAX_PATH);
	std::wstring dir(path);
	return dir.substr(0, dir.find_last_of(L"\\/") + 1);
}
std::string ExeName() {
	char path[MAX_PATH] {};
	GetModuleFileNameA(nullptr, path, MAX_PATH);
	std::string name(path);
	name = name.substr(name.find_last_of("\\/") + 1);
	const auto dot = name.find_last_of('.');
	return dot == std::string::npos ? name : name.substr(0, dot);
}

void Log(const char* fmt, ...) {
	std::lock_guard lock(g_log_mutex);
	if (g_log == nullptr) {
		// Append: the module may be loaded more than once per process, and a run must not erase the last one.
		g_log = _wfopen((ExeDir() + L"injector.log").c_str(), L"a");
		if (g_log == nullptr) return;
		SYSTEMTIME d;
		GetLocalTime(&d);
		std::fprintf(g_log, "\n==== injector log opened %04d-%02d-%02d, pid %lu, module %p ====\n", d.wYear, d.wMonth, d.wDay,
		             GetCurrentProcessId(), static_cast<void*>(GetModuleHandleW(L"vulkan-1.dll")));
	}
	SYSTEMTIME t;
	GetLocalTime(&t);
	std::fprintf(g_log, "%02d:%02d:%02d.%03d ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
	va_list args;
	va_start(args, fmt);
	std::vfprintf(g_log, fmt, args);
	va_end(args);
	std::fputc('\n', g_log);
	std::fflush(g_log);
}

namespace {
thread_local int t_internal = 0;
}
InternalScope::InternalScope() { ++t_internal; }
InternalScope::~InternalScope() { --t_internal; }
bool Internal() { return t_internal > 0; }

size_t FeatureStructSize(VkStructureType type) {
	switch (type) {
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: return sizeof(VkPhysicalDeviceFeatures2);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: return sizeof(VkPhysicalDeviceVulkan11Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: return sizeof(VkPhysicalDeviceVulkan12Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: return sizeof(VkPhysicalDeviceVulkan13Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT: return sizeof(VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV: return sizeof(VkPhysicalDeviceComputeShaderDerivativesFeaturesNV);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: return sizeof(VkPhysicalDeviceShaderIntegerDotProductFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: return sizeof(VkPhysicalDeviceShaderFloat16Int8Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: return sizeof(VkPhysicalDeviceBufferDeviceAddressFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: return sizeof(VkPhysicalDevice16BitStorageFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: return sizeof(VkPhysicalDevice8BitStorageFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES: return sizeof(VkPhysicalDeviceSynchronization2Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES: return sizeof(VkPhysicalDeviceDynamicRenderingFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: return sizeof(VkPhysicalDeviceTimelineSemaphoreFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES: return sizeof(VkPhysicalDeviceDescriptorIndexingFeatures);
	default: return 0;
	}
}

Settings& Cfg() { return g_settings; }
Real& R() { return g_real; }
Queues& Q() { return g_queues; }
HWND& GameWindow() { return g_hwnd; }
std::recursive_mutex& QueueLock() { return g_queue_lock; }

static std::wstring IniPath() { return ExeDir() + L"injector.ini"; }

void LoadSettings() {
	const auto path = IniPath();
	auto geti = [&](const wchar_t* key, int def) { return static_cast<int>(GetPrivateProfileIntW(L"injector", key, def, path.c_str())); };
	auto& s = g_settings;
	s.fg_enabled      = geti(L"fg", 1) != 0;
	s.fg_cap_fps      = geti(L"fg_cap_fps", -1);
	s.osd             = geti(L"osd", 1) != 0;
	s.analyze         = geti(L"analyze", 1) != 0;
	s.use_game_inputs = geti(L"use_game_inputs", 1) != 0;
	s.min_confidence  = geti(L"min_confidence_pct", 60) / 100.0f;
	s.mv_sign         = geti(L"mv_sign", -1) < 0 ? -1 : 1;
	s.mv_units        = geti(L"mv_units", 0) != 0 ? 1 : 0;
	s.depth_mode      = geti(L"depth_mode", 0);
	s.previews        = geti(L"previews", 1) != 0;
	s.up_backend      = std::clamp(geti(L"upscaler", 0), 0, 4);
	s.up_quality      = std::clamp(geti(L"up_quality", 0), 0, 5);
	s.up_sharpness    = std::clamp(geti(L"up_sharpness_pct", 30), 0, 100) / 100.0f;
	s.up_original     = false;  // (A/B always starts on our image)
	s.up_split        = geti(L"up_split", 0) != 0;
	s.up_jitter       = std::clamp(geti(L"up_jitter", 0), 0, 2);
	s.up_input        = std::clamp(geti(L"up_input", 0), 0, 1);
	s.up_native       = geti(L"up_native", 0) != 0;
	s.up_streamline   = geti(L"up_streamline", 1) != 0;
}

void SaveSettings() {
	const auto path = IniPath();
	auto put = [&](const wchar_t* key, int v) { WritePrivateProfileStringW(L"injector", key, std::to_wstring(v).c_str(), path.c_str()); };
	const auto& s = g_settings;
	put(L"fg", s.fg_enabled);
	put(L"fg_cap_fps", s.fg_cap_fps);
	put(L"osd", s.osd);
	put(L"analyze", s.analyze);
	put(L"use_game_inputs", s.use_game_inputs);
	put(L"min_confidence_pct", static_cast<int>(s.min_confidence * 100 + 0.5f));
	put(L"mv_sign", s.mv_sign);
	put(L"mv_units", s.mv_units);
	put(L"depth_mode", s.depth_mode);
	put(L"previews", s.previews);
	put(L"upscaler", s.up_backend);
	put(L"up_quality", s.up_quality);
	put(L"up_sharpness_pct", static_cast<int>(s.up_sharpness * 100 + 0.5f));
	put(L"up_split", s.up_split);
	put(L"up_jitter", s.up_jitter);
	put(L"up_input", s.up_input);
	put(L"up_native", s.up_native);
	put(L"up_streamline", s.up_streamline);
}

double DisplayRefreshHz() {
	DEVMODEW mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) && mode.dmDisplayFrequency > 1) return mode.dmDisplayFrequency;
	return 0.0;
}

uint32_t FindMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
	VkPhysicalDeviceMemoryProperties props {};
	g_real.vkGetPhysicalDeviceMemoryProperties(g_real.physical, &props);
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
		if (bits & (1u << i)) return i;
	return 0;
}

static void InitOnce() {
	std::call_once(g_init_once, [] {
		LoadSettings();
		Log("injector phase 2 (generic proxy vulkan-1.dll: frame generation + upscalers) in %s.exe", ExeName().c_str());
		Log("settings: fg %d, cap %d, osd %d, analyze %d, game inputs %d, min confidence %.2f; upscaler %d, quality %d, sharpness %.2f, "
		    "jitter %d, input %d, native %d, streamline %d", g_settings.fg_enabled, g_settings.fg_cap_fps, g_settings.osd,
		    g_settings.analyze, g_settings.use_game_inputs, g_settings.min_confidence, g_settings.up_backend, g_settings.up_quality,
		    g_settings.up_sharpness, g_settings.up_jitter, g_settings.up_input, g_settings.up_native, g_settings.up_streamline);
		wchar_t sys[MAX_PATH] {};
		GetSystemDirectoryW(sys, MAX_PATH);
		const std::wstring system = std::wstring(sys) + L"\\vulkan-1.dll";
		// 10-28, Wildlands: DXVK freed our vulkan-1.dll after its probe device, the system loader we had loaded stayed
		// in the process under the same module name, and DXVK's next LoadLibraryA("vulkan-1.dll") got THAT one: the
		// real device ran without us. So (1) our module is pinned (never unloaded), and (2) the system loader is loaded
		// from a copy under another name (%LOCALAPPDATA%\injector\vulkan-1-system.dll), so no module named
		// vulkan-1.dll but ours is ever in the process. Falls back to System32 directly if the copy fails.
		std::wstring path = system;
		wchar_t local[MAX_PATH] {};
		if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) > 0) {
			const std::wstring dir = std::wstring(local) + L"\\injector";
			CreateDirectoryW(dir.c_str(), nullptr);
			const std::wstring copy = dir + L"\\vulkan-1-system.dll";
			WIN32_FILE_ATTRIBUTE_DATA a {}, b {};
			const bool have = GetFileAttributesExW(copy.c_str(), GetFileExInfoStandard, &b);
			const bool same = have && GetFileAttributesExW(system.c_str(), GetFileExInfoStandard, &a) &&
			                  a.nFileSizeLow == b.nFileSizeLow && CompareFileTime(&a.ftLastWriteTime, &b.ftLastWriteTime) == 0;
			if (same || CopyFileW(system.c_str(), copy.c_str(), FALSE)) path = copy;
			else Log("could not copy the system loader to %ls (%lu): loading it from System32", copy.c_str(), GetLastError());
		}
		g_real.loader = LoadLibraryW(path.c_str());
		if (g_real.loader == nullptr && path != system) { path = system; g_real.loader = LoadLibraryW(path.c_str()); }
		if (g_real.loader == nullptr) { Log("FATAL: the system Vulkan loader (%ls) not found", path.c_str()); return; }
		g_real.gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_real.loader, "vkGetInstanceProcAddr"));
		Log("system Vulkan loader: %ls", path.c_str());
	});
}

PFN_vkVoidFunction RealDeviceProc(const char* name) {
	PFN_vkVoidFunction f = nullptr;
	if (g_real.device != VK_NULL_HANDLE && g_real.gdpa != nullptr) f = g_real.gdpa(g_real.device, name);
	if (f == nullptr && g_real.gipa != nullptr) f = g_real.gipa(g_real.instance, name);
	return f;
}

void LoadDeviceFunctions(VkDevice device) {
#define INJ_FN(name)                                                                                                 \
	{                                                                                                                \
		auto f = g_real.gdpa(device, #name);                                                                         \
		if (f == nullptr) f = g_real.gipa(g_real.instance, #name);                                                   \
		if (f != nullptr) g_real.name = reinterpret_cast<PFN_##name>(f);                                             \
	}
#include "device_functions.inl"
#undef INJ_FN
}

// ---------------------------------------------------------------- hooks
namespace {

struct SwapInfo {
	VkSwapchainKHR handle = VK_NULL_HANDLE;
	VkExtent2D     extent {};
	VkFormat       format = VK_FORMAT_UNDEFINED;
};
SwapInfo g_swap;

// The pre-present submit ring.
struct Slot {
	VkCommandBuffer cb = VK_NULL_HANDLE;
	VkFence         fence = VK_NULL_HANDLE;
	VkSemaphore     done = VK_NULL_HANDLE;
	bool            pending = false;
};
std::array<Slot, 8> g_slots {};
uint32_t            g_slot = 0;
VkCommandPool       g_pool = VK_NULL_HANDLE;
uint64_t            g_presents = 0;

bool EnsureSlots() {
	if (g_pool != VK_NULL_HANDLE) return true;
	VkCommandPoolCreateInfo pi {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pi.queueFamilyIndex = Q().game_family;
	if (R().vkCreateCommandPool(R().device, &pi, nullptr, &g_pool) != VK_SUCCESS) { g_pool = VK_NULL_HANDLE; return false; }
	for (auto& s: g_slots) {
		VkCommandBufferAllocateInfo ai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		ai.commandPool = g_pool;
		ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;
		R().vkAllocateCommandBuffers(R().device, &ai, &s.cb);
		// A dispatchable handle made by the driver: the loader's dispatch pointer goes in first (as the loader does
		// for buffers allocated through it; we call the driver's function directly).
		*reinterpret_cast<void**>(s.cb) = *reinterpret_cast<void**>(R().device);
		VkFenceCreateInfo fi {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		R().vkCreateFence(R().device, &fi, nullptr, &s.fence);
		VkSemaphoreCreateInfo si {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		R().vkCreateSemaphore(R().device, &si, nullptr, &s.done);
	}
	return true;
}

VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc) {
	auto f = reinterpret_cast<PFN_vkDestroyInstance>(R().gipa(instance, "vkDestroyInstance"));
	Log("vkDestroyInstance %p", static_cast<void*>(instance));
	f(instance, alloc);
	if (instance == R().instance) R().instance = VK_NULL_HANDLE;
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* info,
                                                         const VkAllocationCallbacks* alloc, VkSurfaceKHR* surface) {
	if (info != nullptr) g_hwnd = info->hwnd;
	auto f = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(R().gipa(instance, "vkCreateWin32SurfaceKHR"));
	return f(instance, info, alloc, surface);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
                                                 const VkAllocationCallbacks* alloc, VkDevice* device) {
	auto& r = R();
	Log("vkCreateDevice (instance %p)", static_cast<void*>(r.instance));
	auto  get_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(r.gipa(r.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
	auto  create = reinterpret_cast<PFN_vkCreateDevice>(r.gipa(r.instance, "vkCreateDevice"));
	uint32_t n = 0;
	get_families(physical, &n, nullptr);
	std::vector<VkQueueFamilyProperties> fam(n);
	get_families(physical, &n, fam.data());

	// Frame generation needs three queues of its own: a present queue in the graphics family, and an async compute
	// and an image acquire queue in a compute-only family (KytyPS5's PlanQueues). Added after the game's own.
	std::vector<VkDeviceQueueCreateInfo> queues(info->pQueueCreateInfos, info->pQueueCreateInfos + info->queueCreateInfoCount);
	std::vector<std::vector<float>> priorities(queues.size() + 1);
	uint32_t g = UINT32_MAX, c = UINT32_MAX;
	for (uint32_t i = 0; i < n && g == UINT32_MAX; ++i) if (fam[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) g = i;
	for (uint32_t i = 0; i < n && c == UINT32_MAX; ++i)
		if ((fam[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(fam[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) c = i;
	uint32_t present_index = UINT32_MAX, compute_base = UINT32_MAX;
	bool ok = g != UINT32_MAX && c != UINT32_MAX;
	int  gi = -1, ci = -1;
	for (size_t i = 0; i < queues.size(); ++i) {
		if (queues[i].queueFamilyIndex == g) gi = static_cast<int>(i);
		if (queues[i].queueFamilyIndex == c) ci = static_cast<int>(i);
	}
	if (ok && gi >= 0 && fam[g].queueCount >= queues[gi].queueCount + 1) {
		present_index = queues[gi].queueCount;
		priorities[gi].assign(queues[gi].pQueuePriorities, queues[gi].pQueuePriorities + queues[gi].queueCount);
		priorities[gi].push_back(1.0f);
		queues[gi].queueCount++;
		queues[gi].pQueuePriorities = priorities[gi].data();
	} else ok = false;
	if (ok) {
		if (ci >= 0 && fam[c].queueCount >= queues[ci].queueCount + 2) {
			compute_base = queues[ci].queueCount;
			priorities[ci].assign(queues[ci].pQueuePriorities, queues[ci].pQueuePriorities + queues[ci].queueCount);
			priorities[ci].push_back(1.0f);
			priorities[ci].push_back(1.0f);
			queues[ci].queueCount += 2;
			queues[ci].pQueuePriorities = priorities[ci].data();
		} else if (ci < 0 && fam[c].queueCount >= 2) {
			compute_base = 0;
			priorities.back() = {1.0f, 1.0f};
			VkDeviceQueueCreateInfo q {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
			q.queueFamilyIndex = c;
			q.queueCount = 2;
			q.pQueuePriorities = priorities.back().data();
			queues.push_back(q);
		} else ok = false;
	}
	VkDeviceCreateInfo patched = *info;
	if (ok) {
		patched.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
		patched.pQueueCreateInfos = queues.data();
	}
	// The upscalers' extensions and features (DLSS, FSR 4, XeSS, our push-descriptor passes), where the device has them.
	upscale::DevicePatch up;
	const VkDeviceCreateInfo base = ok ? patched : *info;
	VkDeviceCreateInfo with_up = base;
	{
		InternalScope scope;  // (XeSS asks the driver through "vulkan-1.dll": ours)
		upscale::PatchDevice(physical, base, with_up, up);
	}
	const VkDeviceCreateInfo* made = &with_up;
	VkResult result = create(physical, &with_up, alloc, device);
	if (result != VK_SUCCESS) {
		Log("vkCreateDevice with the upscalers' extensions / features failed (%d): again without them", result);
		up.Restore();
		made = ok ? &patched : info;
		result = create(physical, made, alloc, device);
	}
	if (result != VK_SUCCESS && ok) {
		Log("vkCreateDevice with the frame generation queues failed (%d): again without them", result);
		ok = false;
		made = info;
		result = create(physical, info, alloc, device);
	}
	if (result != VK_SUCCESS) return result;
	upscale::OnDeviceCreated(*made);
	r.physical = physical;
	r.device = *device;
	g_presents = 0;
	r.gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(r.gipa(r.instance, "vkGetDeviceProcAddr"));
	LoadDeviceFunctions(*device);
	VkPhysicalDeviceProperties props {};
	r.vkGetPhysicalDeviceProperties(physical, &props);
	Log("device: %s (Vulkan %u.%u.%u, driver %u)", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
	    VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion);
	auto& q = Q();
	q = {};
	q.graphics_family = g;
	q.game_family = g;
	if (ok) {
		q.compute_family = c;
		r.vkGetDeviceQueue(*device, g, present_index, &q.present);
		r.vkGetDeviceQueue(*device, c, compute_base, &q.async_compute);
		r.vkGetDeviceQueue(*device, c, compute_base + 1, &q.image_acquire);
		q.fg_queues_ok = q.present && q.async_compute && q.image_acquire;
	}
	Log("frame generation queues: %s (graphics family %u present index %u, compute family %u from index %u)",
	    q.fg_queues_ok ? "created" : "NOT available", g, present_index, c, compute_base);
	return result;
}

VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
	if (device == R().device) {
		R().vkDeviceWaitIdle(device);
		upscale::OnDeviceDestroyed();
		camera::Shutdown();
		analyzer::Shutdown();
		overlay::OnDeviceDestroyed();
		framegen::OnDeviceDestroyed();
		Log("device destroyed after %llu presents", static_cast<unsigned long long>(g_presents));
		g_pool = VK_NULL_HANDLE;  // (freed with the device)
		g_slots = {};
	}
	R().vkDestroyDevice(device, alloc);
	if (device == R().device) {
		R().device = VK_NULL_HANDLE;
		Q() = {};
		g_swap = {};
	}
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	std::lock_guard lock(QueueLock());
	if (!Internal())
		for (uint32_t i = 0; i < count; ++i) analyzer::OnSubmit(submits[i].pCommandBuffers, submits[i].commandBufferCount);
	return R().vkQueueSubmit(queue, count, submits, fence);
}

static VkResult Submit2(PFN_vkQueueSubmit2 fn, VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
	std::lock_guard lock(QueueLock());
	std::vector<VkCommandBuffer> cbs;
	for (uint32_t i = 0; i < count && !Internal(); ++i) {
		cbs.clear();
		for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; ++j) cbs.push_back(submits[i].pCommandBufferInfos[j].commandBuffer);
		analyzer::OnSubmit(cbs.data(), static_cast<uint32_t>(cbs.size()));
	}
	return fn(queue, count, submits, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
	return Submit2(R().vkQueueSubmit2, q, n, s, f);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
	return Submit2(R().vkQueueSubmit2KHR, q, n, s, f);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueWaitIdle(VkQueue queue) {
	std::lock_guard lock(QueueLock());
	return R().vkQueueWaitIdle(queue);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueBindSparse(VkQueue queue, uint32_t n, const VkBindSparseInfo* info, VkFence fence) {
	std::lock_guard lock(QueueLock());
	return R().vkQueueBindSparse(queue, n, info, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_DeviceWaitIdle(VkDevice device) {
	std::lock_guard lock(QueueLock());
	return R().vkDeviceWaitIdle(device);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images) {
	if (framegen::Owns(swapchain)) return framegen::GetSwapchainImages(device, swapchain, count, images);
	return R().vkGetSwapchainImagesKHR(device, swapchain, count, images);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                                                       const VkAllocationCallbacks* alloc, VkSwapchainKHR* out) {
	VkSwapchainCreateInfoKHR ci = *info;
	ci.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // the overlay's render pass
	VkResult r;
	const bool fg = framegen::Available();
	if (fg) {
		if (g_swap.handle != VK_NULL_HANDLE && framegen::Owns(g_swap.handle) && ci.oldSwapchain == g_swap.handle) {
			overlay::OnSwapchainDestroyed(g_swap.handle);
			g_swap.handle = VK_NULL_HANDLE;
		}
		r = framegen::CreateSwapchain(device, &ci, out);
	} else {
		r = R().vkCreateSwapchainKHR(device, &ci, alloc, out);
	}
	if (r != VK_SUCCESS) {
		Log("vkCreateSwapchainKHR failed (%d)", r);
		return r;
	}
	g_swap = {*out, ci.imageExtent, ci.imageFormat};
	upscale::SetDisplay(ci.imageExtent, ci.imageFormat);
	uint32_t n = 0;
	Hook_GetSwapchainImagesKHR(device, *out, &n, nullptr);
	std::vector<VkImage> images(n);
	Hook_GetSwapchainImagesKHR(device, *out, &n, images.data());
	analyzer::SetSwapchainImages(images.data(), n);
	overlay::OnSwapchain(*out, images.data(), n, ci.imageFormat, ci.imageExtent);
	Log("swapchain %ux%u format %d, %u images, %s", ci.imageExtent.width, ci.imageExtent.height, ci.imageFormat, n,
	    framegen::Owns(*out) ? "FidelityFX frame interpolation swapchain" : "plain");
	return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL Hook_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* alloc) {
	if (swapchain == VK_NULL_HANDLE) return;
	{
		std::lock_guard lock(QueueLock());
		R().vkDeviceWaitIdle(device);
	}
	overlay::OnSwapchainDestroyed(swapchain);
	if (g_swap.handle == swapchain) g_swap = {};
	if (framegen::Owns(swapchain)) framegen::DestroySwapchain(swapchain);
	else R().vkDestroySwapchainKHR(device, swapchain, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                                                        VkSemaphore semaphore, VkFence fence, uint32_t* index) {
	if (framegen::Owns(swapchain)) return framegen::AcquireNextImage(device, swapchain, timeout, semaphore, fence, index);
	return R().vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, index);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_AcquireNextImage2KHR(VkDevice device, const VkAcquireNextImageInfoKHR* info, uint32_t* index) {
	if (framegen::Owns(info->swapchain))
		return framegen::AcquireNextImage(device, info->swapchain, info->timeout, info->semaphore, info->fence, index);
	return R().vkAcquireNextImage2KHR(device, info, index);
}
VKAPI_ATTR void VKAPI_CALL Hook_SetHdrMetadataEXT(VkDevice device, uint32_t n, const VkSwapchainKHR* sc, const VkHdrMetadataEXT* md) {
	if (n == 1 && framegen::Owns(sc[0])) framegen::SetHdrMetadata(device, n, sc, md);
	else R().vkSetHdrMetadataEXT(device, n, sc, md);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
	std::lock_guard lock(QueueLock());
	++g_presents;
	if (g_presents == 1) {
		Log("first present on device %p, swapchain %p (%ux%u)", static_cast<void*>(R().device),
		    reinterpret_cast<void*>(g_swap.handle), g_swap.extent.width, g_swap.extent.height);
		upscale::OnFirstPresent();  // (DXVK's real device: Streamline is told about it)
	}
	upscale::OnPresent();
	auto& q = Q();
	if (q.game != queue) {
		q.game = queue;
		Log("game present queue %p", static_cast<void*>(queue));
	}
	const bool ours = info->swapchainCount == 1 && info->pSwapchains[0] == g_swap.handle && g_swap.handle != VK_NULL_HANDLE;
	if (!ours) return R().vkQueuePresentKHR(queue, info);
	overlay::PollInput();
	analyzer::EndFrame(g_swap.extent);
	const bool fg = framegen::Owns(g_swap.handle);
	VkPresentInfoKHR present = *info;
	if ((fg || overlay::WantsDraw() || Cfg().analyze) && EnsureSlots()) {
		auto& s = g_slots[g_slot];
		g_slot = (g_slot + 1) % g_slots.size();
		if (s.pending) {
			R().vkWaitForFences(R().device, 1, &s.fence, VK_TRUE, UINT64_MAX);
			s.pending = false;
		}
		R().vkResetFences(R().device, 1, &s.fence);
		R().vkResetCommandBuffer(s.cb, 0);
		VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		R().vkBeginCommandBuffer(s.cb, &bi);
		const auto inputs = analyzer::RecordCopies(s.cb);
		overlay::Record(s.cb, info->pImageIndices[0]);
		framegen::Record(s.cb, g_swap.handle, g_swap.extent, g_swap.format, inputs);
		R().vkEndCommandBuffer(s.cb);
		std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.waitSemaphoreCount = info->waitSemaphoreCount;
		si.pWaitSemaphores = info->pWaitSemaphores;
		si.pWaitDstStageMask = stages.data();
		si.commandBufferCount = 1;
		si.pCommandBuffers = &s.cb;
		si.signalSemaphoreCount = 1;
		si.pSignalSemaphores = &s.done;
		if (R().vkQueueSubmit(queue, 1, &si, s.fence) == VK_SUCCESS) {
			s.pending = true;
			present.waitSemaphoreCount = 1;
			present.pWaitSemaphores = &s.done;
		} else {
			Log("pre-present submit failed");
		}
	}
	if (fg) {
		present.pNext = nullptr;  // (present id / regions / fences: not for the frame interpolation swapchain)
		return framegen::Present(queue, &present);
	}
	return R().vkQueuePresentKHR(queue, &present);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* alloc, VkImage* image) {
	if (Internal()) return R().vkCreateImage(device, info, alloc, image);
	VkImageCreateInfo ci = *info;
	analyzer::PatchImageCreate(ci);
	VkResult r = R().vkCreateImage(device, &ci, alloc, image);
	if (r != VK_SUCCESS && ci.usage != info->usage) {
		ci = *info;
		r = R().vkCreateImage(device, &ci, alloc, image);
	}
	if (r == VK_SUCCESS) analyzer::OnImageCreated(*image, ci);
	return r;
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* alloc) {
	if (image != VK_NULL_HANDLE && !Internal()) analyzer::OnImageDestroyed(image);
	R().vkDestroyImage(device, image, alloc);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImageView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* alloc, VkImageView* view) {
	VkResult r = R().vkCreateImageView(device, info, alloc, view);
	if (r == VK_SUCCESS && !Internal()) analyzer::OnViewCreated(*view, *info);
	return r;
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc) {
	if (view != VK_NULL_HANDLE && !Internal()) analyzer::OnViewDestroyed(view);
	R().vkDestroyImageView(device, view, alloc);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
	if (!Internal()) {
		analyzer::OnBeginCommandBuffer(cb);
		upscale::OnBeginCommandBuffer(cb);
	}
	return R().vkBeginCommandBuffer(cb, info);
}
// A pass begins: the analyzer records it, the camera finder knows whether it is a scene pass, and the upscaler may do its
// work right before it (the HUD point) or end the frame (DXVK's present blit).
static void BeforeBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
	if (Internal()) return;
	const analyzer::PassInfo pass = analyzer::OnBeginRendering(cb, info);
	upscale::BeforeRendering(cb, info, pass);
	camera::SetScenePass(pass.main_depth);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
	BeforeBeginRendering(cb, info);
	R().vkCmdBeginRendering(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* info) {
	BeforeBeginRendering(cb, info);
	R().vkCmdBeginRenderingKHR(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdEndRendering(VkCommandBuffer cb) {
	if (!Internal()) camera::SetScenePass(false);
	R().vkCmdEndRendering(cb);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdEndRenderingKHR(VkCommandBuffer cb) {
	if (!Internal()) camera::SetScenePass(false);
	R().vkCmdEndRenderingKHR(cb);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
	if (!Internal()) analyzer::OnBarrier2(cb, info);
	R().vkCmdPipelineBarrier2(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* info) {
	if (!Internal()) analyzer::OnBarrier2(cb, info);
	R().vkCmdPipelineBarrier2KHR(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                                   VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb,
                                                   const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
	if (!Internal()) analyzer::OnBarrier(cb, ni, im);
	R().vkCmdPipelineBarrier(cb, src, dst, dep, nm, m, nb, b, ni, im);
}

// ---- the state DXVK leaves bound (re-bound after the upscaler's work in its command buffer) ----
VKAPI_ATTR void VKAPI_CALL Hook_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
	upscale::OnBindPipeline(cb, point, pipeline);  // (inside our work too: a graphics bind there is noted, then undone)
	R().vkCmdBindPipeline(cb, point, pipeline);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout,
                                                      uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                                                      uint32_t dynamic_count, const uint32_t* dynamic) {
	upscale::OnBindDescriptorSets(cb, point, layout, first, count, sets, dynamic_count, dynamic);
	R().vkCmdBindDescriptorSets(cb, point, layout, first, count, sets, dynamic_count, dynamic);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,
                                                 uint32_t offset, uint32_t size, const void* data) {
	upscale::OnPushConstants(cb, layout, stages, offset, size, data);
	R().vkCmdPushConstants(cb, layout, stages, offset, size, data);
}

// ---- the camera finder: constant buffers (descriptor templates, buffer memory, mapped allocations) ----
static VkResult CreateTemplate(PFN_vkCreateDescriptorUpdateTemplate fn, VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* info,
                               const VkAllocationCallbacks* alloc, VkDescriptorUpdateTemplate* out) {
	const VkResult r = fn(device, info, alloc, out);
	if (r == VK_SUCCESS && !Internal()) camera::OnCreateTemplate(*out, *info);
	return r;
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDescriptorUpdateTemplate(VkDevice d, const VkDescriptorUpdateTemplateCreateInfo* i,
                                                                   const VkAllocationCallbacks* a, VkDescriptorUpdateTemplate* o) {
	return CreateTemplate(R().vkCreateDescriptorUpdateTemplate, d, i, a, o);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDescriptorUpdateTemplateKHR(VkDevice d, const VkDescriptorUpdateTemplateCreateInfo* i,
                                                                      const VkAllocationCallbacks* a, VkDescriptorUpdateTemplate* o) {
	return CreateTemplate(R().vkCreateDescriptorUpdateTemplateKHR, d, i, a, o);
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyDescriptorUpdateTemplate(VkDevice d, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* a) {
	if (t != VK_NULL_HANDLE) camera::OnDestroyTemplate(t);
	R().vkDestroyDescriptorUpdateTemplate(d, t, a);
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyDescriptorUpdateTemplateKHR(VkDevice d, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* a) {
	if (t != VK_NULL_HANDLE) camera::OnDestroyTemplate(t);
	R().vkDestroyDescriptorUpdateTemplateKHR(d, t, a);
}
VKAPI_ATTR void VKAPI_CALL Hook_UpdateDescriptorSetWithTemplate(VkDevice d, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* data) {
	if (!Internal()) camera::OnUpdateWithTemplate(t, data);  // (before the call: a Halton jitter goes in before the GPU reads it)
	R().vkUpdateDescriptorSetWithTemplate(d, set, t, data);
}
VKAPI_ATTR void VKAPI_CALL Hook_UpdateDescriptorSetWithTemplateKHR(VkDevice d, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* data) {
	if (!Internal()) camera::OnUpdateWithTemplate(t, data);
	R().vkUpdateDescriptorSetWithTemplateKHR(d, set, t, data);
}
VKAPI_ATTR void VKAPI_CALL Hook_UpdateDescriptorSets(VkDevice d, uint32_t nw, const VkWriteDescriptorSet* w, uint32_t nc,
                                                     const VkCopyDescriptorSet* c) {
	if (!Internal()) camera::OnUpdateDescriptorSets(nw, w);
	R().vkUpdateDescriptorSets(d, nw, w, nc, c);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
	if (!Internal()) camera::OnCmdUpdateBuffer(data, size);
	R().vkCmdUpdateBuffer(cb, buffer, offset, size, data);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateBuffer(VkDevice d, const VkBufferCreateInfo* info, const VkAllocationCallbacks* a, VkBuffer* out) {
	const VkResult r = R().vkCreateBuffer(d, info, a, out);
	if (r == VK_SUCCESS && !Internal() && (info->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) camera::OnCreateBuffer(*out, info->size);
	return r;
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyBuffer(VkDevice d, VkBuffer buffer, const VkAllocationCallbacks* a) {
	if (buffer != VK_NULL_HANDLE) camera::OnDestroyBuffer(buffer);
	R().vkDestroyBuffer(d, buffer, a);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_BindBufferMemory(VkDevice d, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
	const VkResult r = R().vkBindBufferMemory(d, buffer, memory, offset);
	if (r == VK_SUCCESS && !Internal()) camera::OnBindBufferMemory(buffer, memory, offset);
	return r;
}
static VkResult BindBuffers2(PFN_vkBindBufferMemory2 fn, VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* infos) {
	const VkResult r = fn(d, n, infos);
	if (r == VK_SUCCESS && !Internal())
		for (uint32_t i = 0; i < n; ++i) camera::OnBindBufferMemory(infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
	return r;
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_BindBufferMemory2(VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* i) {
	return BindBuffers2(R().vkBindBufferMemory2, d, n, i);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_BindBufferMemory2KHR(VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* i) {
	return BindBuffers2(R().vkBindBufferMemory2KHR, d, n, i);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_MapMemory(VkDevice d, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                                              VkMemoryMapFlags flags, void** data) {
	const VkResult r = R().vkMapMemory(d, memory, offset, size, flags, data);
	if (r == VK_SUCCESS && !Internal() && data != nullptr) camera::OnMapMemory(memory, offset, *data);
	return r;
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_MapMemory2KHR(VkDevice d, const VkMemoryMapInfoKHR* info, void** data) {
	const VkResult r = R().vkMapMemory2KHR(d, info, data);
	if (r == VK_SUCCESS && !Internal() && data != nullptr) camera::OnMapMemory(info->memory, info->offset, *data);
	return r;
}
VKAPI_ATTR void VKAPI_CALL Hook_UnmapMemory(VkDevice d, VkDeviceMemory memory) {
	camera::OnUnmapMemory(memory);
	R().vkUnmapMemory(d, memory);
}
VKAPI_ATTR void VKAPI_CALL Hook_FreeMemory(VkDevice d, VkDeviceMemory memory, const VkAllocationCallbacks* a) {
	if (memory != VK_NULL_HANDLE) camera::OnUnmapMemory(memory);
	R().vkFreeMemory(d, memory, a);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hook_GetDeviceProcAddr(VkDevice device, const char* name);

// Extensions whose functions take the swapchain handle (present wait / id, Reflex latency, swapchain maintenance,
// exclusive fullscreen, display timing): the FidelityFX frame interpolation swapchain is not a driver swapchain and
// has no replacement for them, so they are always hidden from the game (DXVK then
// simply does not use them).
const char* const kHiddenExtensions[] = {
    "VK_KHR_present_wait", "VK_KHR_present_wait2", "VK_KHR_present_id", "VK_KHR_present_id2", "VK_NV_low_latency2",
    "VK_EXT_swapchain_maintenance1", "VK_EXT_full_screen_exclusive", "VK_AMD_anti_lag", "VK_GOOGLE_display_timing",
    "VK_EXT_display_control"};

VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumerateDeviceExtensionProperties(VkPhysicalDevice physical, const char* layer,
                                                                       uint32_t* count, VkExtensionProperties* props) {
	auto real = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(R().gipa(R().instance, "vkEnumerateDeviceExtensionProperties"));
	if (layer != nullptr) return real(physical, layer, count, props);
	uint32_t n = 0;
	VkResult r = real(physical, nullptr, &n, nullptr);
	if (r != VK_SUCCESS) return r;
	std::vector<VkExtensionProperties> all(n);
	r = real(physical, nullptr, &n, all.data());
	if (r != VK_SUCCESS && r != VK_INCOMPLETE) return r;
	all.resize(n);
	static bool logged = false;
	std::vector<VkExtensionProperties> kept;
	for (const auto& e: all) {
		bool hide = false;
		for (const char* h: kHiddenExtensions) hide |= std::strcmp(e.extensionName, h) == 0;
		if (!hide) kept.push_back(e);
		else if (!logged) Log("hidden from the game (frame interpolation swapchain): %s", e.extensionName);
	}
	logged = true;
	if (props == nullptr) { *count = static_cast<uint32_t>(kept.size()); return VK_SUCCESS; }
	const uint32_t copy = std::min<uint32_t>(*count, static_cast<uint32_t>(kept.size()));
	std::copy_n(kept.begin(), copy, props);
	*count = copy;
	return copy < kept.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

struct HookEntry { const char* name; PFN_vkVoidFunction fn; bool device; };
#define HOOK(n, f, d) {n, reinterpret_cast<PFN_vkVoidFunction>(&f), d}
const HookEntry kHooks[] = {
    HOOK("vkGetDeviceProcAddr", Hook_GetDeviceProcAddr, true),
    HOOK("vkCreateDevice", Hook_CreateDevice, false),
    HOOK("vkDestroyInstance", Hook_DestroyInstance, false),
    HOOK("vkEnumerateDeviceExtensionProperties", Hook_EnumerateDeviceExtensionProperties, false),
    HOOK("vkCreateWin32SurfaceKHR", Hook_CreateWin32SurfaceKHR, false),
    HOOK("vkDestroyDevice", Hook_DestroyDevice, true),
    HOOK("vkQueueSubmit", Hook_QueueSubmit, true),
    HOOK("vkQueueSubmit2", Hook_QueueSubmit2, true),
    HOOK("vkQueueSubmit2KHR", Hook_QueueSubmit2KHR, true),
    HOOK("vkQueueWaitIdle", Hook_QueueWaitIdle, true),
    HOOK("vkQueueBindSparse", Hook_QueueBindSparse, true),
    HOOK("vkDeviceWaitIdle", Hook_DeviceWaitIdle, true),
    HOOK("vkQueuePresentKHR", Hook_QueuePresentKHR, true),
    HOOK("vkCreateSwapchainKHR", Hook_CreateSwapchainKHR, true),
    HOOK("vkDestroySwapchainKHR", Hook_DestroySwapchainKHR, true),
    HOOK("vkGetSwapchainImagesKHR", Hook_GetSwapchainImagesKHR, true),
    HOOK("vkAcquireNextImageKHR", Hook_AcquireNextImageKHR, true),
    HOOK("vkAcquireNextImage2KHR", Hook_AcquireNextImage2KHR, true),
    HOOK("vkSetHdrMetadataEXT", Hook_SetHdrMetadataEXT, true),
    HOOK("vkCreateImage", Hook_CreateImage, true),
    HOOK("vkDestroyImage", Hook_DestroyImage, true),
    HOOK("vkCreateImageView", Hook_CreateImageView, true),
    HOOK("vkDestroyImageView", Hook_DestroyImageView, true),
    HOOK("vkBeginCommandBuffer", Hook_BeginCommandBuffer, true),
    HOOK("vkCmdBeginRendering", Hook_CmdBeginRendering, true),
    HOOK("vkCmdBeginRenderingKHR", Hook_CmdBeginRenderingKHR, true),
    HOOK("vkCmdPipelineBarrier", Hook_CmdPipelineBarrier, true),
    HOOK("vkCmdPipelineBarrier2", Hook_CmdPipelineBarrier2, true),
    HOOK("vkCmdPipelineBarrier2KHR", Hook_CmdPipelineBarrier2KHR, true),
    HOOK("vkCmdEndRendering", Hook_CmdEndRendering, true),
    HOOK("vkCmdEndRenderingKHR", Hook_CmdEndRenderingKHR, true),
    HOOK("vkCmdBindPipeline", Hook_CmdBindPipeline, true),
    HOOK("vkCmdBindDescriptorSets", Hook_CmdBindDescriptorSets, true),
    HOOK("vkCmdPushConstants", Hook_CmdPushConstants, true),
    HOOK("vkCreateDescriptorUpdateTemplate", Hook_CreateDescriptorUpdateTemplate, true),
    HOOK("vkCreateDescriptorUpdateTemplateKHR", Hook_CreateDescriptorUpdateTemplateKHR, true),
    HOOK("vkDestroyDescriptorUpdateTemplate", Hook_DestroyDescriptorUpdateTemplate, true),
    HOOK("vkDestroyDescriptorUpdateTemplateKHR", Hook_DestroyDescriptorUpdateTemplateKHR, true),
    HOOK("vkUpdateDescriptorSetWithTemplate", Hook_UpdateDescriptorSetWithTemplate, true),
    HOOK("vkUpdateDescriptorSetWithTemplateKHR", Hook_UpdateDescriptorSetWithTemplateKHR, true),
    HOOK("vkUpdateDescriptorSets", Hook_UpdateDescriptorSets, true),
    HOOK("vkCmdUpdateBuffer", Hook_CmdUpdateBuffer, true),
    HOOK("vkCreateBuffer", Hook_CreateBuffer, true),
    HOOK("vkDestroyBuffer", Hook_DestroyBuffer, true),
    HOOK("vkBindBufferMemory", Hook_BindBufferMemory, true),
    HOOK("vkBindBufferMemory2", Hook_BindBufferMemory2, true),
    HOOK("vkBindBufferMemory2KHR", Hook_BindBufferMemory2KHR, true),
    HOOK("vkMapMemory", Hook_MapMemory, true),
    HOOK("vkMapMemory2KHR", Hook_MapMemory2KHR, true),
    HOOK("vkUnmapMemory", Hook_UnmapMemory, true),
    HOOK("vkFreeMemory", Hook_FreeMemory, true),
};

const HookEntry* FindHook(const char* name) {
	if (name == nullptr) return nullptr;
	for (const auto& h: kHooks) if (std::strcmp(h.name, name) == 0) return &h;
	return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hook_GetDeviceProcAddr(VkDevice device, const char* name) {
	if (R().gdpa == nullptr) R().gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(R().gipa(R().instance, "vkGetDeviceProcAddr"));
	PFN_vkVoidFunction real = R().gdpa(device, name);
	const HookEntry* h = FindHook(name);
	if (h != nullptr && h->device && real != nullptr && device == R().device) return h->fn;
	return real;
}

}  // namespace
}  // namespace inj

using namespace inj;

// ---------------------------------------------------------------- exports (vulkan-1.def)
extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL inj_vkCreateInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* alloc, VkInstance* instance) {
	InitOnce();
	if (R().gipa == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
	auto create = reinterpret_cast<PFN_vkCreateInstance>(R().gipa(VK_NULL_HANDLE, "vkCreateInstance"));
	if (Internal()) return create(info, alloc, instance);  // (Streamline's / XeSS's own instances: not the game's)
	// Streamline (slInit) and XeSS load first: DLSS / XeSS need their instance extensions on the game's instance.
	upscale::BeforeCreateInstance();
	std::vector<const char*> extensions(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);
	{
		auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(R().gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
		uint32_t n = 0;
		std::vector<VkExtensionProperties> available;
		if (enumerate != nullptr && enumerate(nullptr, &n, nullptr) == VK_SUCCESS) {
			available.resize(n);
			enumerate(nullptr, &n, available.data());
			available.resize(n);
		}
		upscale::InstanceExtensions(available, extensions);
	}
	VkInstanceCreateInfo patched = *info;
	patched.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	patched.ppEnabledExtensionNames = extensions.data();
	VkResult r = create(&patched, alloc, instance);
	if (r != VK_SUCCESS && patched.enabledExtensionCount != info->enabledExtensionCount) {
		Log("vkCreateInstance with the upscalers' extensions failed (%d): again without them", r);
		r = create(info, alloc, instance);
	}
	if (r == VK_SUCCESS) {
		R().instance = *instance;
		Log("vkCreateInstance -> %p", static_cast<void*>(*instance));
		if (info->pApplicationInfo) R().api_version = info->pApplicationInfo->apiVersion;
		Log("vkCreateInstance: %s / %s", info->pApplicationInfo && info->pApplicationInfo->pApplicationName ? info->pApplicationInfo->pApplicationName : "?",
		    info->pApplicationInfo && info->pApplicationInfo->pEngineName ? info->pApplicationInfo->pEngineName : "?");
	}
	return r;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL inj_vkGetInstanceProcAddr(VkInstance instance, const char* name) {
	InitOnce();
	if (R().gipa == nullptr || name == nullptr) return nullptr;
	if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&inj_vkGetInstanceProcAddr);
	if (std::strcmp(name, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&inj_vkCreateInstance);
	if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&inj::Hook_GetDeviceProcAddr);
	PFN_vkVoidFunction real = R().gipa(instance, name);
	if (real == nullptr) return nullptr;
	if (const auto* h = FindHook(name)) return h->fn;
	return real;
}

VKAPI_ATTR VkResult VKAPI_CALL inj_vkEnumerateInstanceExtensionProperties(const char* layer, uint32_t* count, VkExtensionProperties* props) {
	InitOnce();
	auto f = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(R().gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
	return f(layer, count, props);
}
VKAPI_ATTR VkResult VKAPI_CALL inj_vkEnumerateInstanceLayerProperties(uint32_t* count, VkLayerProperties* props) {
	InitOnce();
	auto f = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(R().gipa(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
	return f(count, props);
}
VKAPI_ATTR VkResult VKAPI_CALL inj_vkEnumerateInstanceVersion(uint32_t* version) {
	InitOnce();
	auto f = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(R().gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
	if (f == nullptr) { *version = VK_API_VERSION_1_0; return VK_SUCCESS; }
	return f(version);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL inj_vkGetDeviceProcAddr(VkDevice device, const char* name) {
	InitOnce();
	return inj::Hook_GetDeviceProcAddr(device, name);
}

}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
	if (reason == DLL_PROCESS_ATTACH) {
		// Pinned: a FreeLibrary by DXVK after its probe device must not unload us (see InitOnce).
		HMODULE self = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
		                   reinterpret_cast<LPCWSTR>(module), &self);
		// Nothing heavy under the loader lock: the system loader is loaded at the first Vulkan call.
	}
	return TRUE;
}
