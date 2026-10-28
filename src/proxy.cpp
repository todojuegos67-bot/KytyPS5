// The proxy vulkan-1.dll: exports, the real loader, the hooks, and the present path.
#include "analyzer.h"
#include "common.h"
#include "framegen.h"
#include "overlay.h"

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

Settings& Cfg() { return g_settings; }
Real& R() { return g_real; }
Queues& Q() { return g_queues; }
HWND& GameWindow() { return g_hwnd; }
std::recursive_mutex& QueueLock() { return g_queue_lock; }

static std::wstring IniPath() { return ExeDir() + L"injector.ini"; }

namespace {
struct KeyEntry { const char* name; int vk; };
const KeyEntry kKeys[] = {{"F1", VK_F1}, {"F2", VK_F2}, {"F3", VK_F3}, {"F4", VK_F4}, {"F5", VK_F5}, {"F6", VK_F6},
    {"F7", VK_F7}, {"F8", VK_F8}, {"F9", VK_F9}, {"F10", VK_F10}, {"F11", VK_F11}, {"F12", VK_F12}, {"Insert", VK_INSERT},
    {"Home", VK_HOME}, {"End", VK_END}, {"Delete", VK_DELETE}, {"PageUp", VK_PRIOR}, {"PageDown", VK_NEXT},
    {"Pause", VK_PAUSE}, {"ScrollLock", VK_SCROLL}, {"NumLock", VK_NUMLOCK}};
int ParseKey(const wchar_t* key, int def) {
	wchar_t buf[64] {};
	GetPrivateProfileStringW(L"injector", key, L"", buf, 64, (ExeDir() + L"injector.ini").c_str());
	std::string text;
	for (const wchar_t* c = buf; *c; ++c) if (*c != L' ') text.push_back(static_cast<char>(*c));
	if (text.empty()) return def;
	for (const auto& k: kKeys) if (_stricmp(k.name, text.c_str()) == 0) return k.vk;
	const long v = std::strtol(text.c_str(), nullptr, 0);
	return v > 0 && v < 256 ? static_cast<int>(v) : def;
}
}  // namespace

const char* KeyName(int vk) {
	for (const auto& k: kKeys) if (k.vk == vk) return k.name;
	static char buf[16];
	std::snprintf(buf, sizeof buf, "0x%02X", vk);
	return buf;
}

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
	s.menu_key        = ParseKey(L"menu_key", VK_F2);
	s.fg_key          = ParseKey(L"fg_key", VK_F3);
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
	WritePrivateProfileStringW(L"injector", L"menu_key", std::wstring(KeyName(s.menu_key), KeyName(s.menu_key) + std::strlen(KeyName(s.menu_key))).c_str(), path.c_str());
	WritePrivateProfileStringW(L"injector", L"fg_key", std::wstring(KeyName(s.fg_key), KeyName(s.fg_key) + std::strlen(KeyName(s.fg_key))).c_str(), path.c_str());
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

static HMODULE g_self = nullptr;
static std::wstring SelfPath() {
	wchar_t p[MAX_PATH] {};
	GetModuleFileNameW(g_self, p, MAX_PATH);
	return p;
}
static bool SelfIsVulkan1() {
	std::wstring p = SelfPath();
	p = p.substr(p.find_last_of(L"\\/") + 1);
	return _wcsicmp(p.c_str(), L"vulkan-1.dll") == 0;
}

// The first line of every load, written with plain Win32 calls from DllMain (no CRT stream, no locks of ours): next
// to the exe (injector.log) and in %LOCALAPPDATA%\injector\boot.log (if the game folder is not writable).
static void BootLine(HMODULE module) {
	wchar_t mod[MAX_PATH] {}, exe[MAX_PATH] {};
	GetModuleFileNameW(module, mod, MAX_PATH);
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	SYSTEMTIME t;
	GetLocalTime(&t);
	char line[1600];
	const int n = std::snprintf(line, sizeof line, "%04d-%02d-%02d %02d:%02d:%02d.%03d BOOT pid %lu, module %ls @ %p, exe %ls\r\n",
	                            t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, GetCurrentProcessId(),
	                            mod, static_cast<void*>(module), exe);
	std::wstring dir(exe);
	dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
	std::wstring targets[2] = {dir + L"injector.log", L""};
	wchar_t local[MAX_PATH] {};
	if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) > 0) {
		CreateDirectoryW((std::wstring(local) + L"\\injector").c_str(), nullptr);
		targets[1] = std::wstring(local) + L"\\injector\\boot.log";
	}
	for (const auto& path: targets) {
		if (path.empty()) continue;
		HANDLE h = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
		                       FILE_ATTRIBUTE_NORMAL, nullptr);
		if (h == INVALID_HANDLE_VALUE) continue;
		DWORD written = 0;
		WriteFile(h, line, n > 0 ? static_cast<DWORD>(n) : 0, &written, nullptr);
		CloseHandle(h);
	}
}

static void InitOnce() {
	std::call_once(g_init_once, [] {
		LoadSettings();
		Log("injector phase 1 (generic proxy vulkan-1.dll) in %s.exe", ExeName().c_str());
		Log("keys: menu %s, frame generation %s (injector.ini menu_key / fg_key)", KeyName(g_settings.menu_key), KeyName(g_settings.fg_key));
		Log("settings: fg %d, cap %d, osd %d, analyze %d, game inputs %d, min confidence %.2f", g_settings.fg_enabled,
		    g_settings.fg_cap_fps, g_settings.osd, g_settings.analyze, g_settings.use_game_inputs, g_settings.min_confidence);
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
		// As winevulkan.dll (the DX11 install: DXVK tries that name first, and no Windows module has it) there is no
		// name clash: the system loader is loaded from System32 as it is (it may already be loaded by an overlay).
		if (!SelfIsVulkan1()) {
			Log("loaded as %ls", SelfPath().c_str());
		} else if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) > 0) {
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

// The loader's own exported trampolines: they dispatch through the handle they get (physical device, device), so
// they work whichever instance owns it. 10-28, Wildlands: DXVK made instances A and B, destroyed B and then created
// a device on A's physical device; looking the instance up by "the last one" gave a destroyed one (null) and the game
// died in vkCreateDevice. Instance-level calls on a physical device now always go through these.
static PFN_vkVoidFunction LoaderExport(const char* name) {
	return g_real.loader != nullptr ? reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(g_real.loader, name)) : nullptr;
}

PFN_vkVoidFunction RealDeviceProc(const char* name) {
	PFN_vkVoidFunction f = nullptr;
	if (g_real.device != VK_NULL_HANDLE && g_real.gdpa != nullptr) f = g_real.gdpa(g_real.device, name);
	if (f == nullptr && g_real.gipa != nullptr && g_real.instance != VK_NULL_HANDLE) f = g_real.gipa(g_real.instance, name);
	if (f == nullptr) f = LoaderExport(name);
	return f;
}

void LoadDeviceFunctions(VkDevice device) {
#define INJ_FN(name)                                                                                                 \
	{                                                                                                                \
		auto f = g_real.gdpa(device, #name);                                                                         \
		if (f == nullptr && g_real.instance != VK_NULL_HANDLE) f = g_real.gipa(g_real.instance, #name);              \
		if (f == nullptr) f = LoaderExport(#name);                                                                   \
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

// Live instances (any number, created and destroyed in any order), which live instances enumerated each physical
// device, and every device we created (DXVK may keep several alive; the one that makes the swapchain is "active").
std::mutex g_objects_mutex;
std::vector<VkInstance> g_live_instances;
std::vector<std::pair<VkPhysicalDevice, VkInstance>> g_physical_owners;
struct DeviceState {
	VkInstance       instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	Queues           queues;
};
std::vector<std::pair<VkDevice, DeviceState>> g_devices;

bool Live(VkInstance instance) {
	return std::find(g_live_instances.begin(), g_live_instances.end(), instance) != g_live_instances.end();
}
// The live instance that enumerated `physical` (latest first), else the last live instance, else none.
VkInstance InstanceFor(VkPhysicalDevice physical) {
	std::lock_guard lock(g_objects_mutex);
	for (auto it = g_physical_owners.rbegin(); it != g_physical_owners.rend(); ++it)
		if (it->first == physical && Live(it->second)) return it->second;
	return g_live_instances.empty() ? VK_NULL_HANDLE : g_live_instances.back();
}
DeviceState* FindDevice(VkDevice device) {
	for (auto& [d, st]: g_devices) if (d == device) return &st;
	return nullptr;
}
bool Known(VkDevice device) {
	std::lock_guard lock(g_objects_mutex);
	return FindDevice(device) != nullptr;
}

void ReleaseActiveObjects() {
	// Our objects on the active device (it stays alive: another device takes over, or it is about to be destroyed).
	if (R().device == VK_NULL_HANDLE) return;
	R().vkDeviceWaitIdle(R().device);
	analyzer::Shutdown();
	overlay::OnDeviceDestroyed();
	framegen::OnDeviceDestroyed();
	for (auto& s: g_slots) {
		if (s.fence) R().vkDestroyFence(R().device, s.fence, nullptr);
		if (s.done) R().vkDestroySemaphore(R().device, s.done, nullptr);
	}
	if (g_pool) R().vkDestroyCommandPool(R().device, g_pool, nullptr);
	g_pool = VK_NULL_HANDLE;
	g_slots = {};
	g_swap = {};
}

void Activate(VkDevice device) {
	DeviceState st;
	{
		std::lock_guard lock(g_objects_mutex);
		const DeviceState* found = FindDevice(device);
		if (found == nullptr) return;
		st = *found;
	}
	auto& r = R();
	r.device   = device;
	r.physical = st.physical;
	if (Live(st.instance)) r.instance = st.instance;
	r.gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(LoaderExport("vkGetDeviceProcAddr"));
	LoadDeviceFunctions(device);
	Q() = st.queues;
	g_presents = 0;
	Log("active device %p (instance %p), frame generation queues %s", static_cast<void*>(device), static_cast<void*>(st.instance),
	    st.queues.fg_queues_ok ? "yes" : "no");
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumeratePhysicalDevices(VkInstance instance, uint32_t* count, VkPhysicalDevice* devices) {
	auto f = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(LoaderExport("vkEnumeratePhysicalDevices"));
	VkResult r = f(instance, count, devices);
	if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && devices != nullptr) {
		std::lock_guard lock(g_objects_mutex);
		for (uint32_t i = 0; i < *count; ++i) g_physical_owners.emplace_back(devices[i], instance);
	}
	return r;
}

VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc) {
	auto f = reinterpret_cast<PFN_vkDestroyInstance>(LoaderExport("vkDestroyInstance"));
	Log("vkDestroyInstance %p", static_cast<void*>(instance));
	f(instance, alloc);
	std::lock_guard lock(g_objects_mutex);
	g_live_instances.erase(std::remove(g_live_instances.begin(), g_live_instances.end(), instance), g_live_instances.end());
	// Only this instance's own entries go (another live instance may have enumerated the same physical device handle).
	g_physical_owners.erase(std::remove_if(g_physical_owners.begin(), g_physical_owners.end(),
	                                       [&](const auto& e) { return e.second == instance; }), g_physical_owners.end());
	if (R().instance == instance) R().instance = g_live_instances.empty() ? VK_NULL_HANDLE : g_live_instances.back();
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* info,
                                                         const VkAllocationCallbacks* alloc, VkSurfaceKHR* surface) {
	if (info != nullptr) g_hwnd = info->hwnd;
	auto f = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(R().gipa(instance, "vkCreateWin32SurfaceKHR"));
	return f(instance, info, alloc, surface);
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
                                                 const VkAllocationCallbacks* alloc, VkDevice* device) {
	const VkInstance owner = InstanceFor(physical);
	Log("vkCreateDevice (physical device %p, instance %p)", static_cast<void*>(physical), static_cast<void*>(owner));
	auto get_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(LoaderExport("vkGetPhysicalDeviceQueueFamilyProperties"));
	auto create = reinterpret_cast<PFN_vkCreateDevice>(LoaderExport("vkCreateDevice"));
	auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(LoaderExport("vkGetPhysicalDeviceProperties"));
	auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(LoaderExport("vkGetDeviceQueue"));
	if (get_families == nullptr || create == nullptr || get_props == nullptr || get_queue == nullptr) {
		Log("loader exports missing: device created without changes");
		auto f = reinterpret_cast<PFN_vkCreateDevice>(LoaderExport("vkCreateDevice"));
		return f != nullptr ? f(physical, info, alloc, device) : VK_ERROR_INITIALIZATION_FAILED;
	}
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
	VkResult result = create(physical, ok ? &patched : info, alloc, device);
	if (result != VK_SUCCESS && ok) {
		Log("vkCreateDevice with the frame generation queues failed (%d): again without them", result);
		ok = false;
		result = create(physical, info, alloc, device);
	}
	if (result != VK_SUCCESS) return result;
	VkPhysicalDeviceProperties props {};
	get_props(physical, &props);
	Log("device: %s (Vulkan %u.%u.%u, driver %u)", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
	    VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion);
	DeviceState st;
	st.instance = owner;
	st.physical = physical;
	auto& q = st.queues;
	q.graphics_family = g;
	q.game_family = g;
	if (ok) {
		q.compute_family = c;
		get_queue(*device, g, present_index, &q.present);
		get_queue(*device, c, compute_base, &q.async_compute);
		get_queue(*device, c, compute_base + 1, &q.image_acquire);
		q.fg_queues_ok = q.present && q.async_compute && q.image_acquire;
	}
	Log("device %p: frame generation queues %s (graphics family %u present index %u, compute family %u from index %u)",
	    static_cast<void*>(*device), q.fg_queues_ok ? "created" : "NOT available", g, present_index, c, compute_base);
	{
		std::lock_guard lock(g_objects_mutex);
		g_devices.emplace_back(*device, st);
	}
	if (R().device == VK_NULL_HANDLE) Activate(*device);  // else: becomes active when it makes a swapchain
	return result;
}

VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
	const bool active = device != VK_NULL_HANDLE && device == R().device;
	if (active) {
		std::lock_guard lock(QueueLock());
		ReleaseActiveObjects();
		Log("device %p destroyed after %llu presents", static_cast<void*>(device), static_cast<unsigned long long>(g_presents));
	} else {
		Log("device %p destroyed (not active)", static_cast<void*>(device));
	}
	auto destroy = R().vkDestroyDevice != nullptr ? R().vkDestroyDevice
	                                              : reinterpret_cast<PFN_vkDestroyDevice>(LoaderExport("vkDestroyDevice"));
	destroy(device, alloc);
	{
		std::lock_guard lock(g_objects_mutex);
		g_devices.erase(std::remove_if(g_devices.begin(), g_devices.end(), [&](const auto& e) { return e.first == device; }),
		                g_devices.end());
	}
	if (active) {
		R().device = VK_NULL_HANDLE;
		Q() = {};
		VkDevice next = VK_NULL_HANDLE;
		{
			std::lock_guard lock(g_objects_mutex);
			if (!g_devices.empty()) next = g_devices.back().first;
		}
		if (next != VK_NULL_HANDLE) Activate(next);
	}
}

VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	std::lock_guard lock(QueueLock());
	for (uint32_t i = 0; i < count; ++i) analyzer::OnSubmit(submits[i].pCommandBuffers, submits[i].commandBufferCount);
	return R().vkQueueSubmit(queue, count, submits, fence);
}

static VkResult Submit2(PFN_vkQueueSubmit2 fn, VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
	std::lock_guard lock(QueueLock());
	std::vector<VkCommandBuffer> cbs;
	for (uint32_t i = 0; i < count; ++i) {
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
	if (device != R().device && Known(device)) {
		std::lock_guard lock(QueueLock());
		Log("swapchain on device %p: it becomes the active device", static_cast<void*>(device));
		ReleaseActiveObjects();
		Activate(device);
	}
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
	if (g_presents == 1) Log("first present on device %p, swapchain %p (%ux%u)", static_cast<void*>(R().device),
	                         reinterpret_cast<void*>(g_swap.handle), g_swap.extent.width, g_swap.extent.height);
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
	if (image != VK_NULL_HANDLE) analyzer::OnImageDestroyed(image);
	R().vkDestroyImage(device, image, alloc);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImageView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* alloc, VkImageView* view) {
	VkResult r = R().vkCreateImageView(device, info, alloc, view);
	if (r == VK_SUCCESS) analyzer::OnViewCreated(*view, *info);
	return r;
}
VKAPI_ATTR void VKAPI_CALL Hook_DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc) {
	if (view != VK_NULL_HANDLE) analyzer::OnViewDestroyed(view);
	R().vkDestroyImageView(device, view, alloc);
}
VKAPI_ATTR VkResult VKAPI_CALL Hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
	analyzer::OnBeginCommandBuffer(cb);
	return R().vkBeginCommandBuffer(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
	analyzer::OnBeginRendering(cb, info);
	R().vkCmdBeginRendering(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* info) {
	analyzer::OnBeginRendering(cb, info);
	R().vkCmdBeginRenderingKHR(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
	analyzer::OnBarrier2(cb, info);
	R().vkCmdPipelineBarrier2(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* info) {
	analyzer::OnBarrier2(cb, info);
	R().vkCmdPipelineBarrier2KHR(cb, info);
}
VKAPI_ATTR void VKAPI_CALL Hook_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                                   VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb,
                                                   const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
	analyzer::OnBarrier(cb, ni, im);
	R().vkCmdPipelineBarrier(cb, src, dst, dep, nm, m, nb, b, ni, im);
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
	auto real = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(LoaderExport("vkEnumerateDeviceExtensionProperties"));
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
    HOOK("vkEnumeratePhysicalDevices", Hook_EnumeratePhysicalDevices, false),
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
};

const HookEntry* FindHook(const char* name) {
	if (name == nullptr) return nullptr;
	for (const auto& h: kHooks) if (std::strcmp(h.name, name) == 0) return &h;
	return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hook_GetDeviceProcAddr(VkDevice device, const char* name) {
	static auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(LoaderExport("vkGetDeviceProcAddr"));
	PFN_vkVoidFunction real = gdpa(device, name);
	const HookEntry* h = FindHook(name);
	if (h != nullptr && h->device && real != nullptr && Known(device)) return h->fn;
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
	VkResult r = create(info, alloc, instance);
	if (r == VK_SUCCESS) {
		R().instance = *instance;
		{
			std::lock_guard lock(g_objects_mutex);
			g_live_instances.push_back(*instance);
		}
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
		inj::g_self = module;
		inj::BootLine(module);
		// Pinned: a FreeLibrary by DXVK after its probe device must not unload us (see InitOnce).
		HMODULE self = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
		                   reinterpret_cast<LPCWSTR>(module), &self);
		// Nothing heavy under the loader lock: the system loader is loaded at the first Vulkan call.
	}
	return TRUE;
}
