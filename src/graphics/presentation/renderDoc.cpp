#include "graphics/presentation/renderDoc.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <renderdoc_app.h>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

// (rdoc1) RenderDoc's in-application API (3rdparty/renderdoc/renderdoc_app.h, MIT).
// KYTY_RENDERDOC=1 (or --rd): renderdoc.dll loaded before the Vulkan instance (KYTY_RENDERDOC_DLL, else
// C:\Program Files\RenderDoc\renderdoc.dll, else the installed one from the registry); its overlay and its own keys off;
// F12 (window.cpp) captures ONE guest frame: Start at a guest flip, End at the next one, every queue of the device
// (graphics and compute: StartFrameCapture(nullptr, nullptr)). The capture goes to <exe folder>\capturas\ds_*.rdc and
// the .rdc files there before it are deleted (only the last one stays). Low-overhead options (memory, disk): only the
// resources the frame uses, no validation, no callstacks, a soft memory limit. Without the dll: one [rdoc] line and the
// game runs as always. Without KYTY_RENDERDOC: nothing here runs.

namespace Libs::Graphics {

enum class RenderDocState : uint32_t {
	Idle,
	Requested,
	Capturing,
};

static RENDERDOC_API_1_6_0*        g_api             = nullptr;
static std::atomic<RenderDocState> g_state           = RenderDocState::Idle;
static std::atomic_bool            g_unavailable_log = false;
static std::filesystem::path       g_capture_dir;

static void RdocLog(const char* format, ...) {
	va_list args;
	va_start(args, format);
	std::fputs("[rdoc] ", stdout);
	std::vfprintf(stdout, format, args);
	va_end(args);
	std::fflush(stdout);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// The process's memory (as the "Memory state" lines): private commit and working set, in MiB.
static void LogProcessMemory(const char* label) {
	struct Counters {
		DWORD  cb;
		DWORD  PageFaultCount;
		SIZE_T PeakWorkingSetSize;
		SIZE_T WorkingSetSize;
		SIZE_T QuotaPeakPagedPoolUsage;
		SIZE_T QuotaPagedPoolUsage;
		SIZE_T QuotaPeakNonPagedPoolUsage;
		SIZE_T QuotaNonPagedPoolUsage;
		SIZE_T PagefileUsage;
		SIZE_T PeakPagefileUsage;
		SIZE_T PrivateUsage;
	};
	using MemInfo      = BOOL(WINAPI*)(HANDLE, Counters*, DWORD);
	static auto memory = reinterpret_cast<MemInfo>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32GetProcessMemoryInfo")));
	Counters counters {};
	counters.cb = sizeof(counters);
	if (memory == nullptr || memory(GetCurrentProcess(), &counters, sizeof(counters)) == 0) {
		RdocLog("memory (%s): unknown\n", label);
		return;
	}
	RdocLog("memory (%s): process commit %llu MiB, working set %llu MiB\n", label,
	        static_cast<unsigned long long>(counters.PrivateUsage >> 20u),
	        static_cast<unsigned long long>(counters.WorkingSetSize >> 20u));
}

static void SetEnv(const char* key, const char* value) {
	_putenv_s(key, value);
	SetEnvironmentVariableA(key, value);
}

static std::filesystem::path ExeFolder() {
	std::array<wchar_t, 32768> exe {};
	const DWORD                size = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
	if (size == 0 || size >= exe.size()) return std::filesystem::current_path();
	return std::filesystem::path(exe.data()).parent_path();
}

#else

static void LogProcessMemory(const char* /*label*/) {}

static void SetEnv(const char* key, const char* value) {
	::setenv(key, value, 1);
}

static std::filesystem::path ExeFolder() {
	std::error_code ec;
	auto            exe = std::filesystem::read_symlink("/proc/self/exe", ec);
	return ec ? std::filesystem::current_path() : exe.parent_path();
}

#endif

// What RenderDoc cannot capture, off before the Vulkan instance and device exist (each has its existing path without
// it): VK_NV_copy_memory_indirect (a copy command per buffer), VK_KHR_pipeline_binary (the static pipeline cache),
// NVIDIA Streamline / NGX (DLSS frame generation, DLSS NR, Reflex) and Anti-Lag. RenderDoc also hides from the
// emulator every device extension it does not support, so any other optional one falls back the same way. The
// upscaler is not changed here (the render size depends on it): CAPTURA-RENDERDOC.cmd picks FSR instead of DLSS.
static void DisableUnsupportedFeatures() {
	struct Off {
		const char* key;
		const char* value;
	};
	static constexpr Off kOff[] = {
	    {"KYTY_COPY_INDIRECT", "0"}, {"KYTY_PIPELINE_BINARIES", "0"}, {"KYTY_FRAMEGEN", "0"},
	    {"KYTY_DLSSG_DYNAMIC", "0"}, {"KYTY_DLSS_NR", "0"},           {"KYTY_FOVEATED", "0"},
	    {"KYTY_ANTILAG", "0"},       {"KYTY_LOWLAT_REFLEX", "0"},
	};
	std::string list;
	for (const auto& off: kOff) {
		SetEnv(off.key, off.value);
		list += std::string(list.empty() ? "" : " ") + off.key + "=" + off.value;
	}
	RdocLog("off for the capture: %s (their existing fallbacks)\n", list.c_str());
	for (const char* key: {"KYTY_UPSCALER", "KYTY_UPSCALE", "KYTY_TAA_REPLACE"}) {
		const char* value = std::getenv(key);
		if (value != nullptr && std::string_view(value) == "dlss") {
			RdocLog("WARNING: %s=dlss: DLSS (Streamline / NGX) does not work under RenderDoc; use CAPTURA-RENDERDOC.cmd "
			        "(it switches to FSR)\n",
			        key);
		}
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool BindRenderDocApi(HMODULE module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(reinterpret_cast<void*>(GetProcAddress(module, "RENDERDOC_GetAPI")));
#else
static bool BindRenderDocApi(void* module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(module, "RENDERDOC_GetAPI"));
#endif
	if (get_api == nullptr) {
		return false;
	}

	void* api = nullptr;
	if (get_api(eRENDERDOC_API_Version_1_6_0, &api) != 1 || api == nullptr) {
		return false;
	}

	g_api = static_cast<RENDERDOC_API_1_6_0*>(api);
	int major = 0, minor = 0, patch = 0;
	g_api->GetAPIVersion(&major, &minor, &patch);

	// Our own key (F12 in window.cpp), no RenderDoc keys, no overlay, no crash handler (the emulator's own).
	g_api->SetCaptureKeys(nullptr, 0);
	g_api->MaskOverlayBits(eRENDERDOC_Overlay_None, eRENDERDOC_Overlay_None);
	g_api->UnloadCrashHandler();

	// Low overhead: only what the frame references, no validation / buffer checks / callstacks, initial contents only
	// where read, debug output muted, no child processes; past 1 GiB of capture-time copies RenderDoc goes to disk.
	struct Option {
		RENDERDOC_CaptureOption option;
		uint32_t                value;
		const char*             name;
	};
	static constexpr Option kOptions[] = {
	    {eRENDERDOC_Option_RefAllResources, 0, "RefAllResources"},
	    {eRENDERDOC_Option_CaptureAllCmdLists, 0, "CaptureAllCmdLists"},
	    {eRENDERDOC_Option_APIValidation, 0, "APIValidation"},
	    {eRENDERDOC_Option_VerifyBufferAccess, 0, "VerifyBufferAccess"},
	    {eRENDERDOC_Option_CaptureCallstacks, 0, "CaptureCallstacks"},
	    {eRENDERDOC_Option_CaptureCallstacksOnlyActions, 0, "CaptureCallstacksOnlyActions"},
	    {eRENDERDOC_Option_SaveAllInitials, 0, "SaveAllInitials"},
	    {eRENDERDOC_Option_DebugOutputMute, 1, "DebugOutputMute"},
	    {eRENDERDOC_Option_HookIntoChildren, 0, "HookIntoChildren"},
	    {eRENDERDOC_Option_DelayForDebugger, 0, "DelayForDebugger"},
	    {eRENDERDOC_Option_AllowVSync, 1, "AllowVSync"},
	    {eRENDERDOC_Option_AllowFullscreen, 1, "AllowFullscreen"},
	    {eRENDERDOC_Option_SoftMemoryLimit, 1024, "SoftMemoryLimit(MB)"},
	};
	std::string refused;
	for (const auto& o: kOptions) {
		if (g_api->SetCaptureOptionU32(o.option, o.value) != 1) refused += std::string(" ") + o.name;
	}

	g_capture_dir = ExeFolder() / "capturas";
	std::error_code ec;
	std::filesystem::create_directories(g_capture_dir, ec);
	const auto templ = (g_capture_dir / "ds").u8string();
	g_api->SetCaptureFilePathTemplate(reinterpret_cast<const char*>(templ.c_str()));

	RdocLog("RenderDoc loaded: API %d.%d.%d; overlay off, own keys off; F12 captures one frame (all queues) to %s_*.rdc; "
	        "options RefAllResources=0 CaptureAllCmdLists=0 APIValidation=0 VerifyBufferAccess=0 CaptureCallstacks=0 "
	        "SaveAllInitials=0 DebugOutputMute=1 HookIntoChildren=0 SoftMemoryLimit=1024 MB%s%s\n",
	        major, minor, patch, reinterpret_cast<const char*>(templ.c_str()), refused.empty() ? "" : "; refused:",
	        refused.c_str());
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}
	LogProcessMemory("before RenderDoc");
	DisableUnsupportedFeatures();

	// Already in the process (started from RenderDoc's UI), else KYTY_RENDERDOC_DLL, else the default install, else
	// the installed one from the registry.
	std::wstring tried;
	auto*        module = GetModuleHandleW(L"renderdoc.dll");
	if (module == nullptr) {
		std::vector<std::filesystem::path> candidates;
		if (const wchar_t* env = _wgetenv(L"KYTY_RENDERDOC_DLL"); env != nullptr && *env != L'\0') {
			candidates.emplace_back(env);
		}
		candidates.emplace_back(L"C:\\Program Files\\RenderDoc\\renderdoc.dll");
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0, KEY_READ,
		                  &key) == ERROR_SUCCESS) {
			std::array<wchar_t, MAX_PATH> path_buffer {};
			DWORD      path_size = static_cast<DWORD>(path_buffer.size() * sizeof(wchar_t));
			const auto result    = RegQueryValueExW(key, L"", nullptr, nullptr, reinterpret_cast<LPBYTE>(path_buffer.data()),
			                                        &path_size);
			RegCloseKey(key);
			if (result == ERROR_SUCCESS) {
				std::wstring icon(path_buffer.data());
				if (const auto comma = icon.find(L','); comma != std::wstring::npos) icon.resize(comma);
				if (!icon.empty() && icon.front() == L'"') icon.erase(0, 1);
				if (!icon.empty() && icon.back() == L'"') icon.pop_back();
				candidates.push_back(std::filesystem::path(icon).parent_path() / L"renderdoc.dll");
			}
		}
		for (const auto& path: candidates) {
			tried += (tried.empty() ? L"" : L"; ") + path.wstring();
			module = LoadLibraryW(path.c_str());
			if (module != nullptr) {
				RdocLog("loaded %s\n", reinterpret_cast<const char*>(path.u8string().c_str()));
				break;
			}
		}
	}
	if (module == nullptr) {
		const auto text = std::filesystem::path(tried).u8string();
		RdocLog("renderdoc.dll NOT found (tried: %s): no captures, the game runs normally. Install RenderDoc or set "
		        "KYTY_RENDERDOC_DLL\n",
		        reinterpret_cast<const char*>(text.c_str()));
		return;
	}

	if (!BindRenderDocApi(module)) {
		RdocLog("renderdoc.dll loaded but its API 1.6.0 is unavailable (RenderDoc too old?): no captures\n");
		return;
	}
	LogProcessMemory("after RenderDoc");
}

#else

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}
	DisableUnsupportedFeatures();

	const char* env    = std::getenv("KYTY_RENDERDOC_DLL");
	const char* name   = env != nullptr && *env != '\0' ? env : "librenderdoc.so";
	auto*       module = ::dlopen(name, RTLD_NOW | RTLD_NOLOAD);
	if (module == nullptr) {
		module = ::dlopen(name, RTLD_NOW);
	}
	if (module == nullptr) {
		RdocLog("%s NOT found: no captures, the game runs normally\n", name);
		return;
	}

	if (!BindRenderDocApi(module)) {
		RdocLog("API 1.6.0 is unavailable: no captures\n");
		::dlclose(module);
	}
}

#endif

bool RenderDocAvailable() {
	return g_api != nullptr;
}

void RenderDocRequestCapture() {
	if (g_api == nullptr) {
		if (!g_unavailable_log.exchange(true)) {
			RdocLog("F12: RenderDoc is not loaded (KYTY_RENDERDOC=1 and RenderDoc installed?)\n");
		}
		return;
	}

	RenderDocState expected = RenderDocState::Idle;
	if (g_state.compare_exchange_strong(expected, RenderDocState::Requested)) {
		RdocLog("F12: capture requested (starts at the next guest flip)\n");
	}
}

// Only the last capture stays: the .rdc files of the folder are deleted before a new one.
static void DeleteOldCaptures() {
	std::error_code ec;
	uint32_t        removed = 0;
	for (const auto& entry: std::filesystem::directory_iterator(g_capture_dir, ec)) {
		if (entry.is_regular_file(ec) && entry.path().extension() == ".rdc" && std::filesystem::remove(entry.path(), ec)) {
			removed++;
		}
	}
	if (removed != 0) RdocLog("deleted %u old capture(s) in capturas\\\n", removed);
}

static void StartCapture() {
	if (g_api->IsFrameCapturing() != 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		RdocLog("capture request ignored: a capture is already active\n");
		return;
	}
	DeleteOldCaptures();
	LogProcessMemory("capture start");
	// The device and window nullptr: every device's queues (graphics and compute) until EndFrameCapture.
	g_api->StartFrameCapture(nullptr, nullptr);
	if (g_api->IsFrameCapturing() == 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		RdocLog("capture failed to start\n");
		return;
	}
	g_state.store(RenderDocState::Capturing, std::memory_order_release);
	RdocLog("capture started (one guest frame)\n");
}

static void EndCapture() {
	const auto ok = g_api->EndFrameCapture(nullptr, nullptr);
	g_state.store(RenderDocState::Idle, std::memory_order_release);
	LogProcessMemory("capture end");
	if (ok == 0) {
		RdocLog("capture FAILED (EndFrameCapture returned 0)\n");
		return;
	}
	const uint32_t count = g_api->GetNumCaptures();
	if (count == 0) {
		RdocLog("capture finished but RenderDoc lists no capture file\n");
		return;
	}
	uint32_t length = 0;
	g_api->GetCapture(count - 1, nullptr, &length, nullptr);
	std::string path(length + 1, '\0');
	g_api->GetCapture(count - 1, path.data(), &length, nullptr);
	path.resize(std::char_traits<char>::length(path.c_str()));
	std::error_code ec;
	const auto      size = std::filesystem::file_size(std::filesystem::path(std::u8string(path.begin(), path.end())), ec);
	RdocLog("capture saved: %s (%llu MiB)\n", path.c_str(), ec ? 0ull : static_cast<unsigned long long>(size >> 20u));
}

void RenderDocOnGuestFlip(RenderContext& renderer) {
	const auto state = g_state.load(std::memory_order_acquire);
	if (g_api == nullptr || state == RenderDocState::Idle) {
		return;
	}

	// Capture boundaries follow presentation and exclude concurrent queue access. One frame: started at this flip,
	// ended at the next.
	RenderLockGuard   render_lock(renderer.GetMutex());
	Common::LockGuard queue_lock(renderer.GetGraphics().queue_mutex);
	if (state == RenderDocState::Requested) {
		StartCapture();
	} else {
		EndCapture();
	}
}

} // namespace Libs::Graphics
