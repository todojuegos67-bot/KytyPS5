// "Demon's Souls.exe": the emulator as a PC port (-DKYTY_DS_PORT=ON). A Windows program without a console that
// starts the game itself: the game folder chosen once (a folder dialog, kept in settings.ini), every option of the
// launcher (Launcher-souls.ps1) and of run-windows.ps1 from settings.ini (ds-settings.cpp), applied in this process
// before the emulator starts; a splash screen with fondo-souls.jpg until the first frame; the in-game options menu
// (ds-menu.cpp). The emulator's own main (src/main.cpp) is compiled in here under another name and called with the
// command line the scripts made.

#define main KytyEmulatorMain
#include "main.cpp" // NOLINT(bugprone-suspicious-include)
#undef main

#include "port/ds-port.h"
#include "port/ds-update.h"
#include "port/ds-precompile.h"
#include "port/ds-steam.h"
#include "port/ds-i18n.h"
#include "port/ds-resume.h"

#include <nlohmann/json.hpp>

#include <windows.h>
#include <shobjidl.h>
#include <shlobj.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <dxgi.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iterator>
#include <map>
#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "stb_image.h"

namespace fs = std::filesystem;

namespace {

std::wstring Wide(const std::string& s) {
	if (s.empty()) return {};
	const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring w(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
	return w;
}

std::string Utf8(const std::wstring& w) {
	if (w.empty()) return {};
	const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
	std::string s(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
	return s;
}

fs::path ExePath() {
	std::wstring path(32768, L'\0');
	path.resize(GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size())));
	return fs::path(path);
}

void Message(const std::string& text, UINT flags = MB_ICONWARNING) {
	MessageBoxW(nullptr, Wide(text).c_str(), L"Demon's Souls", MB_OK | MB_TOPMOST | flags);
}

std::string ReadText(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// (10-23) Launcher-souls.ps1 Start-Game: with under 10 GB of free RAM, the five programs using the most (only read:
// nothing is closed or emptied). false: the player chose to go back.
bool RamWarning() {
	MEMORYSTATUSEX m {};
	m.dwLength = sizeof(m);
	if (!GlobalMemoryStatusEx(&m)) return true;
	const double free_gb = static_cast<double>(m.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
	const double total   = static_cast<double>(m.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
	if (free_gb >= 10.0) return true;
	static const char* const kSkip[] = {"system", "idle", "memory compression", "registry", "csrss.exe", "wininit.exe",
	                                    "winlogon.exe", "services.exe", "lsass.exe", "smss.exe", "svchost.exe", "dwm.exe",
	                                    "explorer.exe", "secure system"};
	using MemInfo = BOOL(WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
	const auto mem_info = reinterpret_cast<MemInfo>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32GetProcessMemoryInfo")));
	std::map<std::string, uint64_t> by_name;
	if (HANDLE snap = mem_info != nullptr ? CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) : INVALID_HANDLE_VALUE; snap != INVALID_HANDLE_VALUE) {
		PROCESSENTRY32W e {};
		e.dwSize = sizeof(e);
		for (BOOL ok = Process32FirstW(snap, &e); ok; ok = Process32NextW(snap, &e)) {
			if (e.th32ProcessID == GetCurrentProcessId() || e.th32ProcessID == 0) continue;
			std::string name = Utf8(e.szExeFile);
			std::string lower = name;
			std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			if (std::any_of(std::begin(kSkip), std::end(kSkip), [&](const char* k) { return lower == k; })) continue;
			HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, e.th32ProcessID);
			if (p == nullptr) continue;
			PROCESS_MEMORY_COUNTERS c {};
			if (mem_info(p, &c, sizeof(c))) by_name[name.size() > 4 && lower.substr(lower.size() - 4) == ".exe" ? name.substr(0, name.size() - 4) : name] += c.WorkingSetSize;
			CloseHandle(p);
		}
		CloseHandle(snap);
	}
	std::vector<std::pair<uint64_t, std::string>> top;
	for (const auto& [name, bytes]: by_name) top.emplace_back(bytes, name);
	std::sort(top.rbegin(), top.rend());
	char head[256];
	std::snprintf(head, sizeof(head), "Solo quedan %.1f GB de RAM libres (de %.0f GB). El juego necesita unos 12 GB y puede ir a tirones o cerrarse.\n\n",
	              free_gb, total);
	std::string text = std::string(head) + "Los programas que mas RAM usan ahora:\n";
	for (size_t i = 0; i < top.size() && i < 5; i++) {
		char line[200];
		std::snprintf(line, sizeof(line), "   - %s: %.1f GB\n", top[i].second.c_str(), static_cast<double>(top[i].first) / (1024.0 * 1024.0 * 1024.0));
		text += line;
	}
	text += "\nCierralos si puedes (navegador, Discord, etc.).\n\nAceptar: jugar igualmente    Cancelar: salir";
	std::printf("[port] RAM: %.1f GB free of %.0f GB (warned)\n", free_gb, total);
	return MessageBoxW(nullptr, Wide(text).c_str(), L"Memoria RAM", MB_OKCANCEL | MB_ICONWARNING | MB_TOPMOST) == IDOK;
}

// ---- (10-27) The PC: Launcher-souls.ps1's Select-BestGpu (the dedicated graphics card before the integrated one: NVIDIA,
// AMD Radeon RX / Pro or Intel Arc before Intel UHD / Iris or AMD "Radeon(TM) Graphics"; else the one with the most
// VRAM), the CPU's name and the RAM, for the menu's TU EQUIPO and BuildLaunch. ----

Port::PcInfo DetectPc() {
	Port::PcInfo pc;
	IDXGIFactory1* factory = nullptr;
	if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) && factory != nullptr) {
		int      best_tier = -1;
		uint64_t best_vram = 0;
		IDXGIAdapter1* adapter = nullptr;
		for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
			DXGI_ADAPTER_DESC1 d {};
			if (SUCCEEDED(adapter->GetDesc1(&d)) && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
				std::string name = Utf8(d.Description);
				while (!name.empty() && name.back() == ' ') name.pop_back();
				std::string up = name;
				std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
				const auto has  = [&](const char* k) { return up.find(k) != std::string::npos; };
				int        tier = 1;
				if (d.VendorId == 0x1414 || has("BASIC") || has("REMOTE DISPLAY") || has("VIRTUAL") || has("PARSEC") || has("DISPLAYLINK")) tier = 0;
				else if (has("RADEON(TM)") && has("GRAPHICS") && !has(" RX")) tier = 1;
				else if (has("UHD") || has("IRIS") || has("HD GRAPHICS") || has("VEGA")) tier = 1;
				else if (d.VendorId == 0x10DE || has("RADEON RX") || has("RADEON PRO") || has("RX ") || has(" ARC")) tier = 2;
				const uint64_t vram = d.DedicatedVideoMemory;
				if (tier > best_tier || (tier == best_tier && vram > best_vram)) {
					best_tier = tier;
					best_vram = vram;
					pc.gpu    = name;
					pc.vram   = vram;
				}
			}
			adapter->Release();
		}
		factory->Release();
	}
	std::string up = pc.gpu;
	std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
	pc.nvidia = pc.gpu.empty() || up.find("NVIDIA") != std::string::npos || up.find("GEFORCE") != std::string::npos ||
	            up.find("RTX") != std::string::npos || up.find("GTX") != std::string::npos || up.find("QUADRO") != std::string::npos;
	pc.amd = up.find("AMD") != std::string::npos || up.find("RADEON") != std::string::npos;
	HKEY key = nullptr;
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) == ERROR_SUCCESS) {
		wchar_t name[256] = {};
		DWORD   size      = sizeof(name) - sizeof(wchar_t);
		if (RegQueryValueExW(key, L"ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(name), &size) == ERROR_SUCCESS) {
			std::string cpu = Utf8(name), out;
			for (char c: cpu)
				if (!(c == ' ' && (out.empty() || out.back() == ' '))) out += c;
			while (!out.empty() && out.back() == ' ') out.pop_back();
			pc.cpu = out;
		}
		RegCloseKey(key);
	}
	MEMORYSTATUSEX m {};
	m.dwLength = sizeof(m);
	if (GlobalMemoryStatusEx(&m)) pc.ram_gb = static_cast<int>(std::llround(static_cast<double>(m.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0)));
	DEVMODEW mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode)) {
		pc.screen_h   = static_cast<int>(mode.dmPelsHeight);
		pc.screen_w   = static_cast<int>(mode.dmPelsWidth);
		pc.refresh_hz = static_cast<int>(mode.dmDisplayFrequency);
	}
	// (10-28) Cores and threads (the Configurador).
	{
		DWORD len = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
		std::vector<char> buf(len);
		if (len > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
			for (DWORD at = 0; at < len;) {
				const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
				pc.cpu_cores++;
				for (WORD g = 0; g < info->Processor.GroupCount; g++) {
					KAFFINITY m2 = info->Processor.GroupMask[g].Mask;
					while (m2 != 0) {
						pc.cpu_threads += static_cast<int>(m2 & 1);
						m2 >>= 1;
					}
				}
				at += info->Size;
			}
		}
	}
	Port::RecommendFor(&pc);
	return pc;
}

// ---- The splash screen: fondo-souls.jpg, a status line under it, until the game's first frame. ----

struct Splash {
	std::atomic<HWND>       window {nullptr};
	std::mutex              mutex;
	std::wstring            status = L"Cargando...";
	std::vector<uint8_t>    bgra;
	int                     w = 0, h = 0;
	std::wstring            hint;             // (10-27c) a second line under the status
	float                   progress = -1.0f; // (10-27c) the start-up precompile's bar (0..1; <0: none)
	std::atomic<bool>       skip {false};     // (10-27c) Esc during the precompile: play now
} g_splash;

LRESULT CALLBACK SplashProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
		case WM_PAINT: {
			PAINTSTRUCT ps;
			HDC         dc = BeginPaint(hwnd, &ps);
			RECT        rc;
			GetClientRect(hwnd, &rc);
			if (!g_splash.bgra.empty()) {
				BITMAPINFO bi {};
				bi.bmiHeader.biSize        = sizeof(bi.bmiHeader);
				bi.bmiHeader.biWidth       = g_splash.w;
				bi.bmiHeader.biHeight      = -g_splash.h;
				bi.bmiHeader.biPlanes      = 1;
				bi.bmiHeader.biBitCount    = 32;
				bi.bmiHeader.biCompression = BI_RGB;
				SetStretchBltMode(dc, HALFTONE);
				StretchDIBits(dc, 0, 0, rc.right, rc.bottom, 0, 0, g_splash.w, g_splash.h, g_splash.bgra.data(), &bi,
				              DIB_RGB_COLORS, SRCCOPY);
			} else {
				FillRect(dc, &rc, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
			}
			std::wstring status;
			{
				std::scoped_lock lock(g_splash.mutex);
				status = g_splash.status;
			}
			const int height = std::max<int>(16, rc.bottom / 36);
			HFONT     font   = CreateFontW(height, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
			                               CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
			HGDIOBJ   old    = SelectObject(dc, font);
			SetBkMode(dc, TRANSPARENT);
			RECT text = {rc.left + height, rc.bottom - height * 3, rc.right - height, rc.bottom - height};
			RECT shade = text;
			OffsetRect(&shade, 2, 2);
			SetTextColor(dc, RGB(0, 0, 0));
			DrawTextW(dc, status.c_str(), -1, &shade, DT_RIGHT | DT_WORDBREAK);
			SetTextColor(dc, RGB(222, 196, 140));
			DrawTextW(dc, status.c_str(), -1, &text, DT_RIGHT | DT_WORDBREAK);
			float        progress;
			std::wstring hint;
			{
				std::scoped_lock lock(g_splash.mutex);
				progress = g_splash.progress;
				hint     = g_splash.hint;
			}
			if (!hint.empty()) {
				// (10-27c) The hint at the top right: bigger, light on a dark backing, with a shadow.
				HFONT hint_font = CreateFontW(height, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
				                          CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
				HGDIOBJ prev = SelectObject(dc, hint_font);
				const int pad   = height / 2;
				const int max_w = std::max<int>(height * 8, (rc.right - rc.left) * 11 / 20);
				RECT      size  = {0, 0, max_w, 0};
				DrawTextW(dc, hint.c_str(), -1, &size, DT_RIGHT | DT_WORDBREAK | DT_CALCRECT);
				RECT box = {rc.right - height - (size.right - size.left) - pad * 2, rc.top + height, rc.right - height,
				            rc.top + height + (size.bottom - size.top) + pad * 2};
				HBRUSH backing = CreateSolidBrush(RGB(12, 10, 8));
				FillRect(dc, &box, backing);
				DeleteObject(backing);
				RECT line = {box.left + pad, box.top + pad, box.right - pad, box.bottom - pad};
				RECT line_shade = line;
				OffsetRect(&line_shade, 2, 2);
				SetTextColor(dc, RGB(0, 0, 0));
				DrawTextW(dc, hint.c_str(), -1, &line_shade, DT_RIGHT | DT_WORDBREAK);
				SetTextColor(dc, RGB(245, 235, 210));
				DrawTextW(dc, hint.c_str(), -1, &line, DT_RIGHT | DT_WORDBREAK);
				SelectObject(dc, prev);
				DeleteObject(hint_font);
			}
			if (progress >= 0.0f) {
				// (10-27c) The precompile's progress bar above the status line.
				RECT bar = {rc.left + height, rc.bottom - height * 4, rc.right - height, rc.bottom - height * 3 - height / 3};
				HBRUSH back = CreateSolidBrush(RGB(30, 26, 20)), fill = CreateSolidBrush(RGB(222, 196, 140));
				FillRect(dc, &bar, back);
				RECT done = bar;
				done.right = bar.left + static_cast<LONG>((bar.right - bar.left) * std::min(1.0f, progress));
				FillRect(dc, &done, fill);
				DeleteObject(back);
				DeleteObject(fill);
			}
			SelectObject(dc, old);
			DeleteObject(font);
			EndPaint(hwnd, &ps);
			return 0;
		}
		case WM_ERASEBKGND: return 1;
		case WM_KEYDOWN:
			if (wp == VK_ESCAPE) g_splash.skip = true;
			return 0;
		case WM_CLOSE: DestroyWindow(hwnd); return 0;
		case WM_DESTROY: PostQuitMessage(0); return 0;
		default: return DefWindowProcW(hwnd, msg, wp, lp);
	}
}

void ShowSplash(const std::string& image) {
	int               channels = 0;
	const std::string bytes    = ReadText(fs::path(Wide(image)));
	auto*             pixels   = bytes.empty() ? nullptr
	                                           : stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()),
	                                                                   &g_splash.w, &g_splash.h, &channels, 4);
	if (pixels != nullptr) {
		g_splash.bgra.assign(pixels, pixels + static_cast<size_t>(g_splash.w) * g_splash.h * 4);
		for (size_t i = 0; i < g_splash.bgra.size(); i += 4) std::swap(g_splash.bgra[i], g_splash.bgra[i + 2]);
		stbi_image_free(pixels);
	}
	std::thread([] {
		WNDCLASSW wc {};
		wc.lpfnWndProc   = SplashProc;
		wc.hInstance     = GetModuleHandleW(nullptr);
		wc.lpszClassName = L"DemonsSoulsSplash";
		wc.hCursor       = LoadCursorW(nullptr, MAKEINTRESOURCEW(32650)); // IDC_APPSTARTING
		wc.hIcon         = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
		RegisterClassW(&wc);
		// 60% of the screen's width, the image's aspect.
		const int sw = GetSystemMetrics(SM_CXSCREEN), sh = GetSystemMetrics(SM_CYSCREEN);
		const int w  = sw * 6 / 10;
		const int h  = g_splash.w > 0 ? w * g_splash.h / g_splash.w : w * 9 / 16;
		HWND hwnd = CreateWindowExW(WS_EX_TOOLWINDOW, wc.lpszClassName, L"Demon's Souls", WS_POPUP | WS_VISIBLE, (sw - w) / 2,
		                            (sh - h) / 2, w, h, nullptr, nullptr, wc.hInstance, nullptr);
		g_splash.window = hwnd;
		MSG msg;
		while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
		g_splash.window = nullptr;
	}).detach();
}

void SplashHint(const std::string& text) {
	{
		std::scoped_lock lock(g_splash.mutex);
		g_splash.hint = Wide(text);
	}
	if (HWND w = g_splash.window.load(); w != nullptr) InvalidateRect(w, nullptr, FALSE);
}

void SplashProgress(float f) {
	{
		std::scoped_lock lock(g_splash.mutex);
		g_splash.progress = f;
	}
	if (HWND w = g_splash.window.load(); w != nullptr) InvalidateRect(w, nullptr, FALSE);
}

void SplashStatus(const std::string& text) {
	{
		std::scoped_lock lock(g_splash.mutex);
		g_splash.status = Wide(text);
	}
	if (HWND w = g_splash.window.load(); w != nullptr) InvalidateRect(w, nullptr, FALSE);
}

// ---- The game folder. ----

bool IsGame(const std::string& path) {
	if (path.empty()) return false;
	const fs::path p = fs::path(Wide(path));
	std::error_code ec;
	if (fs::exists(p / L"eboot.bin", ec)) return true;
	return p.extension() == L".zar" && fs::is_regular_file(p, ec);
}

std::string PickFolder() {
	std::string result;
	IFileOpenDialog* dialog = nullptr;
	if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return result;
	DWORD options = 0;
	dialog->GetOptions(&options);
	dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
	dialog->SetTitle(L"Demon's Souls: elige la carpeta del juego volcado (la que tiene eboot.bin y sce_sys)");
	if (SUCCEEDED(dialog->Show(nullptr))) {
		IShellItem* item = nullptr;
		if (SUCCEEDED(dialog->GetResult(&item))) {
			PWSTR path = nullptr;
			if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
				result = Utf8(path);
				CoTaskMemFree(path);
			}
			item->Release();
		}
	}
	dialog->Release();
	return result;
}

struct GameParam {
	std::string title_id, version, name;
};

GameParam ReadParam(const std::string& game) {
	GameParam param;
	std::ifstream in(fs::path(Wide(game)) / L"sce_sys" / L"param.json");
	if (!in) return param;
	try {
		const auto json = nlohmann::json::parse(in);
		param.title_id  = json.value("titleId", "");
		param.version   = json.value("contentVersion", "");
		if (json.contains("localizedParameters")) {
			const auto& lp   = json["localizedParameters"];
			const auto  lang = lp.value("defaultLanguage", "en-US");
			if (lp.contains(lang)) param.name = lp[lang].value("titleName", "");
		}
	} catch (...) {
	}
	return param;
}

// ---- The CPUs (run-windows.ps1): the largest L3 of a CPU with several, the performance cores of a hybrid one. ----

uint64_t LargestL3Mask() {
	DWORD length = 0;
	GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
	if (length == 0) return 0;
	std::vector<uint8_t> buffer(length);
	if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
	                                      &length))
		return 0;
	uint64_t best_mask = 0, best_size = 0, min_size = ~uint64_t(0);
	int      count     = 0;
	for (DWORD offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
		if (info->Relationship == RelationCache && info->Cache.Level == 3 && info->Cache.GroupMask.Group == 0) {
			count++;
			min_size = std::min<uint64_t>(min_size, info->Cache.CacheSize);
			if (info->Cache.CacheSize > best_size) {
				best_size = info->Cache.CacheSize;
				best_mask = info->Cache.GroupMask.Mask;
			}
		}
		offset += info->Size;
	}
	// (ds-pc-port47) Only a CPU whose L3 caches differ (an asymmetric V-Cache part: 7950X3D, 9950X3D): one CCD for the
	// game. One L3, or several of one size (7950X, Threadripper, Intel): 0, no affinity at all.
	return count >= 2 && best_size > min_size ? best_mask : 0;
}

std::string PerformanceCpus(uint64_t allowed) {
	ULONG length = 0;
	GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) return {};
	std::vector<uint8_t> buffer(length);
	if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), length, &length,
	                                GetCurrentProcess(), 0))
		return {};
	struct Cpu {
		int cpu, core, cls;
	};
	std::vector<Cpu> cpus;
	for (ULONG offset = 0; offset < length;) {
		const auto* set = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data() + offset);
		if (set->Type == CpuSetInformation && set->CpuSet.Group == 0)
			cpus.push_back({set->CpuSet.LogicalProcessorIndex, set->CpuSet.CoreIndex, set->CpuSet.EfficiencyClass});
		offset += set->Size;
	}
	int lo = 255, hi = 0, interrupts = -1;
	for (const auto& c: cpus) {
		lo = std::min(lo, c.cls);
		hi = std::max(hi, c.cls);
		if (c.cpu == 0) interrupts = c.core;
	}
	if (lo == hi) return {};
	std::string list;
	for (const auto& c: cpus) {
		if (c.cls != hi || c.core == interrupts || c.cpu >= 64 || (allowed & (1ull << c.cpu)) == 0) continue;
		if (!list.empty()) list += ",";
		list += std::to_string(c.cpu);
	}
	return list;
}

// ---- The precompile program (kyty_shader_precompile.exe), as run-windows.ps1 runs it before the game. ----

int RunHidden(const fs::path& exe, const std::wstring& args, const fs::path& dir, const fs::path& output, uint64_t affinity) {
	std::wstring        command = L"\"" + exe.wstring() + L"\" " + args;
	SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
	HANDLE              out = CreateFileW(output.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
	STARTUPINFOW        si {};
	si.cb         = sizeof(si);
	si.dwFlags    = STARTF_USESTDHANDLES;
	si.hStdOutput = out;
	si.hStdError  = out;
	PROCESS_INFORMATION pi {};
	const BOOL started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
	                                    dir.c_str(), &si, &pi);
	if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
	if (!started) return -1;
	if (affinity != 0) SetProcessAffinityMask(pi.hProcess, static_cast<DWORD_PTR>(affinity));
	ResumeThread(pi.hThread);
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	return static_cast<int>(code);
}


void PrepareShaders(const fs::path& root, const std::string& game, const GameParam& param, uint64_t affinity, int cpus) {
	const fs::path tool   = root / L"kyty_shader_precompile.exe";
	const fs::path states = root / L"tools" / L"local" / L"static-precompile" / L"pass-states.json";
	std::error_code ec;
	if (!fs::exists(tool, ec) || !fs::exists(states, ec)) return;
	const std::wstring wgame = L"\"" + Wide(game) + L"\"";
	const std::wstring name  = !param.title_id.empty() && !param.version.empty()
	                               ? Wide("seeds-" + param.title_id + "_" + param.version + ".seeds")
	                               : L"seeds.seeds";
	const fs::path seeds = root / name;
	const fs::path stamp_file = fs::path(seeds.wstring() + L".program");
	const auto     tool_time  = fs::last_write_time(tool, ec);
	const std::string stamp   = std::to_string(fs::file_size(tool, ec)) + " " +
	                          std::to_string(tool_time.time_since_epoch().count());
	const fs::path log = root / L"logs" / L"precompile.log";
	const bool     have  = fs::exists(seeds, ec);
	const bool     stale = have && ReadText(stamp_file).find(stamp) == std::string::npos;
	if (!have || stale) {
		SplashStatus("Preparando shaders 1/3: la lista de shaders del juego (solo la primera vez)...");
		const fs::path made = stale ? fs::path(seeds.wstring() + L".new") : seeds;
		if (RunHidden(tool, L"--game " + wgame + L" --make-seeds \"" + made.wstring() + L"\" --states \"" + states.wstring() + L"\"",
		              root, log, affinity) == 0 &&
		    fs::exists(made, ec)) {
			// (10-27c) A new exe that lists the same shaders keeps the seed file as it was (its time too): the prefetch
			// inputs stay current and the start-up precompile's shards done stay done.
			if (stale && ReadText(made) == ReadText(seeds)) {
				fs::remove(made, ec);
				std::printf("[port] shaders: new program, same shader list: the seed file is kept\n");
			} else if (stale) {
				fs::rename(made, seeds, ec);
			}
			std::ofstream(stamp_file, std::ios::binary) << stamp;
		} else if (!have) {
			std::printf("[port] shaders: listing failed: the game compiles its shaders when they first appear\n");
			return;
		}
	}
	// (10-27b) The release's specialization hints (tools/local/static-precompile/hints-<title>_<version>.hints, shipped:
	// recorded play without the game's code), completed with the seed file's code into released-<title>_<version>.seeds
	// next to the exe: the shader prefetch starts with them (PipelineCache::StartShaderPrefetch), what a PC without a
	// recording of its own compiled one by one in the first fights (precompile-windows.ps1 does the same).
	if (!param.title_id.empty() && !param.version.empty()) {
		const std::string id       = param.title_id + "_" + param.version;
		const fs::path    hints    = root / L"tools" / L"local" / L"static-precompile" / Wide("hints-" + id + ".hints");
		const fs::path    released = root / Wide("released-" + id + ".seeds");
		const auto        newer    = [&](const fs::path& a, const fs::path& b) {
            std::error_code e;
            return fs::last_write_time(a, e) > fs::last_write_time(b, e);
		};
		if (fs::exists(hints, ec) && (!fs::exists(released, ec) || newer(seeds, released) || newer(hints, released))) {
			SplashStatus("Preparando shaders 2/3: especializaciones de partidas grabadas (unos segundos)...");
			if (RunHidden(tool, L"--game " + wgame + L" --seeds \"" + seeds.wstring() + L"\" --apply-hints \"" + hints.wstring() +
			                        L"\" --out \"" + released.wstring() + L"\"",
			              root, root / L"logs" / L"apply-hints.log", affinity) != 0) {
				fs::remove(released, ec);
				std::printf("[port] shaders: the release's hints could not be applied: the seeds alone\n");
			} else {
				std::printf("[port] shaders: %s made from the release's hints\n", ("released-" + id + ".seeds").c_str());
			}
		}
	}
	const fs::path status_file = root / L"logs" / L"precompile-status.txt";
	const std::wstring base    = L"--game " + wgame + L" --seeds \"" + seeds.wstring() + L"\" ";
	if (RunHidden(tool, base + L"--status", root, status_file, affinity) != 0) return;
	const std::string status = ReadText(status_file);
	std::printf("[port] shaders: %s\n", status.find("inputs current") != std::string::npos ? "prefetch inputs ready" : "making the prefetch inputs");
	if (status.find("inputs current") == std::string::npos) {
		SplashStatus("Preparando shaders 3/3: para esta grafica (primera vez o driver nuevo: un minuto). Despues se compilan todos los pipelines con su barra de progreso.");
		if (RunHidden(tool, base + L"--no-pipelines --static-inputs --threads " + std::to_wstring(cpus), root, log, affinity) != 0)
			std::printf("[port] shaders: making the prefetch inputs failed\n");
	}
	std::fflush(stdout);
}

// (ds-pc-port37) "Metricas" as the emulator launcher's: its metricas.ps1 window (always on top: GPU, VRAM, CPU, RAM, FPS,
// the frametime graph, the upscaler, frame generation, Anti-Lag and the Souls mode), with the launcher.ini values it
// reads (scalelabel, fglabel, antilagon, soulsmodo, dlsspreset) written from settings.ini as the launcher writes them.
void StartMetricsWindow(const fs::path& root, const Port::Settings& s, const Port::PcInfo& pc) {
	std::error_code ec;
	const fs::path script = root / L"metricas.ps1";
	if (!fs::exists(script, ec)) {
		std::printf("[port] metrics: metricas.ps1 not found next to the exe\n");
		return;
	}
	const std::string souls = s.souls_mode;
	const std::string modo  = souls == "ligero" ? "Modo Souls ligero" : souls == "equilibrado" ? "Modo Souls equilibrado"
	                        : souls == "ultra" ? "Modo Souls ultra" : souls == "extremo" ? "Modo Souls extremo"
	                        : souls == "low" ? "Modo Souls PC low profile" : "Personalizado";
	const bool  fixed = souls == "ligero" || souls == "equilibrado" || souls == "low";
	std::string fg    = fixed ? modo : Port::FrameGenLabel(s.framegen);
	std::string al    = "No";
	if (fixed) al = "Ultra";
	else if (s.framegen.rfind("dlss", 0) == 0) al = s.antilag_dlssg == "ultra" ? "Ultra (Reflex apagado)" : s.antilag_dlssg == "normal" ? "Normal (Reflex apagado)" : "Reflex (DLSS-G)";
	else if (s.framegen.rfind("amd", 0) == 0) {
		const std::string m = s.antilag_mode.empty() || s.antilag_mode == "auto" ? (pc.amd ? "amd2" : "normal") : s.antilag_mode;
		al = m == "amd2" ? "Anti-Lag 2 (AMD)" : m == "ultra" ? "Ultra" : m == "no" ? "No" : "Normal";
	}
	const int preset = s.dlss_preset == "K" ? 1 : s.dlss_preset == "L" ? 2 : s.dlss_preset == "M" ? 3 : 0;
	{
		std::ofstream ini(root / L"launcher.ini", std::ios::binary | std::ios::trunc);
		const char* scale = Port::kResModes[0].label;
		for (int i = 0; i < Port::kResModeCount; i++)
			if (s.res == Port::kResModes[i].id) scale = Port::kResModes[i].label;
		ini << "scalelabel=" << scale << "\r\n"
		    << "fglabel=" << fg << "\r\n"
		    << "antilagon=" << al << "\r\n"
		    << "soulsmodo=" << modo << "\r\n"
		    << "dlsspreset=" << preset << "\r\n";
	}
	std::wstring command = L"\"" + std::wstring(_wgetenv(L"SystemRoot") != nullptr ? _wgetenv(L"SystemRoot") : L"C:\\Windows") +
	                       L"\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"" +
	                       script.wstring() + L"\" -LogDir \"" + (root / L"logs").wstring() + L"\" -Proceso \"Demon's Souls\"";
	if (s.simulate_vram8) command += L" -VramMB 8192";
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | DETACHED_PROCESS, nullptr, root.c_str(), &si, &pi)) {
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		std::printf("[port] metrics: metricas.ps1 started (as the emulator launcher's)\n");
	} else {
		std::printf("[port] metrics: metricas.ps1 did not start (%lu)\n", GetLastError());
	}
}

void BackupSaves(const fs::path& root) {
	std::error_code ec;
	const fs::path  src = root / L"_SaveData";
	if (!fs::exists(src, ec)) return;
	SYSTEMTIME t;
	GetLocalTime(&t);
	wchar_t stamp[32];
	swprintf(stamp, 32, L"%04u%02u%02u-%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
	const fs::path dst = root / L"_SaveData-copias" / stamp;
	fs::create_directories(dst, ec);
	fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
}

void SetEnv(const std::string& key, const std::string& value) {
	// (Both: the C runtime's copy, which the emulator reads, and the process', which DLLs and child processes read.)
	_putenv_s(key.c_str(), value.c_str());
	SetEnvironmentVariableA(key.c_str(), value.empty() ? nullptr : value.c_str());
}

std::atomic<bool> g_restart {false};
std::atomic<bool> g_restart_launcher {false}; // (ds-pc-port38) an update / restore: the new start opens the launcher
fs::path          g_exe;

std::atomic<bool> g_back_to_launcher {false}; // (11-09) the game was started from the launcher (JUGAR)

// (11-09) The game ended (F1's "Cerrar el juego", the window's X, a normal exit): the launcher opens again in a new
// process (the emulator cannot go back to the launcher in this one). Not after a restart (that start is its own).
void ReopenLauncher() {
	if (g_restart.load() || !g_back_to_launcher.exchange(false)) return;
	std::wstring        command = L"\"" + g_exe.wstring() + L"\" --restarted --lanzador";
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, g_exe.parent_path().c_str(), &si, &pi)) {
		AllowSetForegroundWindow(pi.dwProcessId);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
	}
}

void RestartIfAsked() {
	if (!g_restart.exchange(false)) return ReopenLauncher();
	// (ds-pc-port38) After an update or a restore the new version starts at the launcher (--lanzador), never the game by
	// itself; F1's "Reiniciar ahora" still goes straight back to the game (--restarted alone).
	std::wstring command = L"\"" + g_exe.wstring() + L"\" --restarted";
	if (g_restart_launcher.exchange(false)) command += L" --lanzador";
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, g_exe.parent_path().c_str(), &si, &pi)) {
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
	}
}

// (ds-pc-port29) A crash leaves logs\crash-<time>.txt: the exception code, the faulting module and its offset.
// (ds-pc-port38) And always some evidence (the AMD multi frame generation run of 11-01 ended with no crash file: its last
// line was a title-screen stat, then nothing). Why a crash could leave nothing: a DLL loaded later (Streamline's
// sl.interposer / sl.common, a driver or overlay) replaced the unhandled-exception filter; a stack overflow, whose filter
// ran out of stack on the faulting thread; a heap corruption or a C runtime fast fail (invalid parameter, abort,
// std::terminate, a pure virtual call) that never reaches a filter. So: the writing is done by a thread of its own
// (its own stack, started at the beginning), fed by (1) the unhandled-exception filter, re-armed every half second
// (and chained to whatever replaced it), (2) a last vectored handler for the codes no one recovers from (stack overflow,
// heap corruption, fast fail; never access violations: the emulator's guest memory relies on those), (3) the C runtime's
// invalid parameter / pure call / abort / terminate handlers. Each writes crash-<time>.txt (code, module + offset,
// 32 frames) and crash-<time>.dmp (MiniDumpWriteDump, dbghelp.dll loaded by name: no Windows Error Reporting needed).
// A fast fail the kernel ends the process for (no user-mode code runs) is reported at the next start from Windows Error
// Reporting's archive (ReportPreviousWerCrash).
std::wstring g_crash_dir;

struct CrashJob {
	EXCEPTION_POINTERS* info      = nullptr;
	DWORD               thread_id = 0;
	char                what[160] = {};
};
CrashJob          g_crash_job;
HANDLE            g_crash_go   = nullptr;
HANDLE            g_crash_done = nullptr;
std::atomic<bool> g_crash_written {false};
DWORD             g_crash_thread_id = 0;
LPTOP_LEVEL_EXCEPTION_FILTER g_chained_filter = nullptr;
thread_local int  t_in_filter = 0;

std::wstring CrashStamp() {
	SYSTEMTIME t;
	GetLocalTime(&t);
	wchar_t text[64];
	swprintf(text, 64, L"%04u%02u%02u-%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
	return text;
}

void ModuleOf(uintptr_t addr, wchar_t* name, unsigned long long* off) {
	HMODULE mod = nullptr;
	wcscpy_s(name, MAX_PATH, L"?");
	*off = addr;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(addr), &mod) &&
	    mod != nullptr) {
		GetModuleFileNameW(mod, name, MAX_PATH);
		*off = addr - reinterpret_cast<uintptr_t>(mod);
	}
}

// On the crash thread (its own stack): the text file and the minidump.
void WriteCrashFiles(const CrashJob& job) {
	const std::wstring stamp = CrashStamp();
	const DWORD code = job.info != nullptr && job.info->ExceptionRecord != nullptr ? job.info->ExceptionRecord->ExceptionCode : 0;
	const uintptr_t addr =
	    job.info != nullptr && job.info->ExceptionRecord != nullptr ? reinterpret_cast<uintptr_t>(job.info->ExceptionRecord->ExceptionAddress) : 0;
	wchar_t            name[MAX_PATH];
	unsigned long long off = 0;
	ModuleOf(addr, name, &off);
	const std::wstring txt = g_crash_dir + L"\\crash-" + stamp + L".txt";
	if (FILE* f = _wfopen(txt.c_str(), L"w"); f != nullptr) {
		std::fprintf(f, "%s (thread %lu)\n", job.what, job.thread_id);
		std::fwprintf(f, L"exception 0x%08lx at %p\nmodule %ls + 0x%llx\n", code, reinterpret_cast<void*>(addr), name, off);
		if (job.info != nullptr && job.info->ExceptionRecord != nullptr && code == EXCEPTION_ACCESS_VIOLATION &&
		    job.info->ExceptionRecord->NumberParameters >= 2)
			std::fprintf(f, "access: %s 0x%llx\n", job.info->ExceptionRecord->ExceptionInformation[0] == 0 ? "read" : job.info->ExceptionRecord->ExceptionInformation[0] == 1 ? "write" : "execute",
			             static_cast<unsigned long long>(job.info->ExceptionRecord->ExceptionInformation[1]));
		std::fprintf(f, "stack:\n");
		// (ds-pc-port32) The faulting thread's stack from its context (x64 unwind data), up to 32 frames.
		if (job.info != nullptr && job.info->ContextRecord != nullptr) {
			CONTEXT ctx = *job.info->ContextRecord;
			for (int frame = 0; frame < 32 && ctx.Rip != 0; frame++) {
				wchar_t            fn[MAX_PATH];
				unsigned long long fo = 0;
				ModuleOf(static_cast<uintptr_t>(ctx.Rip), fn, &fo);
				std::fwprintf(f, L"  #%02d %ls + 0x%llx\n", frame, fn, fo);
				DWORD64           image = 0;
				RUNTIME_FUNCTION* fe    = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
				if (fe == nullptr) {
					// A leaf function: the return address is at Rsp.
					if (ctx.Rsp == 0 || IsBadReadPtr(reinterpret_cast<void*>(ctx.Rsp), sizeof(DWORD64))) break;
					ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
					ctx.Rsp += 8;
				} else {
					void*   handler_data = nullptr;
					DWORD64 establisher  = 0;
					RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, fe, &ctx, &handler_data, &establisher, nullptr);
				}
			}
		}
		std::fclose(f);
	}
	// The minidump: dbghelp.dll by name (System32's), no link dependency.
	bool dumped = false;
	if (HMODULE dbghelp = LoadLibraryW(L"dbghelp.dll"); dbghelp != nullptr) {
		struct ExceptionInfo {
			DWORD               ThreadId;
			EXCEPTION_POINTERS* ExceptionPointers;
			BOOL                ClientPointers;
		};
		using WriteDump = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, ExceptionInfo*, void*, void*);
		if (auto* write = reinterpret_cast<WriteDump>(reinterpret_cast<void*>(GetProcAddress(dbghelp, "MiniDumpWriteDump"))); write != nullptr) {
			const std::wstring dmp = g_crash_dir + L"\\crash-" + stamp + L".dmp";
			HANDLE file = CreateFileW(dmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (file != INVALID_HANDLE_VALUE) {
				ExceptionInfo ei {job.thread_id, job.info, FALSE};
				// MiniDumpWithIndirectlyReferencedMemory | WithDataSegs | WithHandleData | WithUnloadedModules | WithThreadInfo |
				// WithProcessThreadData: the threads' stacks and what they point to, never the guest's gigabytes.
				const int type = 0x40 | 0x1 | 0x4 | 0x20 | 0x1000 | 0x100;
				dumped = write(GetCurrentProcess(), GetCurrentProcessId(), file, type, job.info != nullptr ? &ei : nullptr, nullptr, nullptr) != FALSE;
				CloseHandle(file);
				if (!dumped) DeleteFileW(dmp.c_str());
			}
		}
	}
	std::printf("[port] crash: %s: exception 0x%08lx in %ls + 0x%llx (thread %lu); logs\\crash-%ls.txt%s\n", job.what, code, name, off,
	            job.thread_id, stamp.c_str(), dumped ? " and .dmp" : " (no minidump)");
	std::fflush(stdout);
}

DWORD WINAPI CrashThread(LPVOID) {
	for (;;) {
		if (WaitForSingleObject(g_crash_go, INFINITE) != WAIT_OBJECT_0) return 0;
		WriteCrashFiles(g_crash_job);
		SetEvent(g_crash_done);
	}
}

// From any thread, however little stack it has left: hands the crash to the crash thread and waits for it (once).
void ReportCrash(EXCEPTION_POINTERS* info, const char* what) {
	if (g_crash_written.exchange(true)) return;
	g_crash_job.info      = info;
	g_crash_job.thread_id = GetCurrentThreadId();
	strncpy_s(g_crash_job.what, what, _TRUNCATE);
	if (g_crash_go != nullptr && g_crash_done != nullptr && g_crash_thread_id != 0 && g_crash_thread_id != GetCurrentThreadId()) {
		SetEvent(g_crash_go);
		WaitForSingleObject(g_crash_done, 30000);
	} else {
		WriteCrashFiles(g_crash_job);
	}
}

// No exception to hand on (abort, terminate, an invalid parameter, a pure call): this thread's context now.
[[noreturn]] void ReportFatalHere(const char* what) {
	static CONTEXT          context;
	static EXCEPTION_RECORD record;
	static EXCEPTION_POINTERS pointers;
	RtlCaptureContext(&context);
	record                  = {};
	record.ExceptionCode    = 0xC0000409; // (STATUS_STACK_BUFFER_OVERRUN: what the C runtime's fast fail reports)
	record.ExceptionAddress = reinterpret_cast<void*>(context.Rip);
	pointers                = {&record, &context};
	ReportCrash(&pointers, what);
	TerminateProcess(GetCurrentProcess(), 3);
	for (;;) Sleep(INFINITE);
}

LONG WINAPI PortCrashFilter(EXCEPTION_POINTERS* info) {
	if (t_in_filter > 0) return EXCEPTION_CONTINUE_SEARCH; // (a chained filter came back to this one)
	t_in_filter++;
	ReportCrash(info, "unhandled exception");
	LONG result = EXCEPTION_CONTINUE_SEARCH;
	if (LPTOP_LEVEL_EXCEPTION_FILTER next = g_chained_filter; next != nullptr && next != &PortCrashFilter) result = next(info);
	t_in_filter--;
	return result;
}

// Last in the vectored chain: only what nothing recovers from.
LONG CALLBACK PortFatalVectored(EXCEPTION_POINTERS* info) {
	const DWORD code = info != nullptr && info->ExceptionRecord != nullptr ? info->ExceptionRecord->ExceptionCode : 0;
	if (code == 0xC00000FD /* stack overflow */) ReportCrash(info, "stack overflow");
	else if (code == 0xC0000374 /* heap corruption */) ReportCrash(info, "heap corruption");
	else if (code == 0xC0000409 /* fast fail reaching user mode */) ReportCrash(info, "fast fail / stack buffer overrun");
	else if (code == 0xC0000420 /* assertion failure */) ReportCrash(info, "assertion failure");
	return EXCEPTION_CONTINUE_SEARCH;
}

void PortInvalidParameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned int, uintptr_t) {
	ReportFatalHere("a C runtime function got an invalid parameter");
}
void PortPureCall() {
	ReportFatalHere("pure virtual function call");
}
void PortAbort(int) {
	ReportFatalHere("abort()");
}
void PortTerminate() {
	ReportFatalHere("std::terminate (an exception nothing caught, or noexcept broken)");
}

// The filter re-armed every half second: a DLL that set its own (Streamline, a driver, an overlay) is chained after ours.
// The C runtime's handlers too (the emulator sets its own reporting ones at start-up: they are chained, kept as well).
DWORD WINAPI CrashRearmThread(LPVOID) {
	bool told = false;
	for (;;) {
		Sleep(500);
		LPTOP_LEVEL_EXCEPTION_FILTER previous = SetUnhandledExceptionFilter(&PortCrashFilter);
		if (previous != &PortCrashFilter && previous != nullptr) {
			g_chained_filter = previous;
			if (!told) {
				wchar_t            name[MAX_PATH];
				unsigned long long off = 0;
				ModuleOf(reinterpret_cast<uintptr_t>(previous), name, &off);
				std::printf("[port] crash reports: the unhandled-exception filter was replaced (by %ls): ours re-armed first, that one chained\n",
				            name);
				std::fflush(stdout);
				told = true;
			}
		}
		if (_get_invalid_parameter_handler() != &PortInvalidParameter) _set_invalid_parameter_handler(&PortInvalidParameter);
		if (_get_purecall_handler() != &PortPureCall) _set_purecall_handler(&PortPureCall);
	}
}

void InstallCrashReports() {
	g_crash_go   = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	g_crash_done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	// (Its own 1 MiB stack, committed now: it must run when the process is out of everything else.)
	if (HANDLE t = CreateThread(nullptr, 1 << 20, &CrashThread, nullptr, 0, &g_crash_thread_id); t != nullptr) CloseHandle(t);
	g_chained_filter = SetUnhandledExceptionFilter(&PortCrashFilter);
	AddVectoredExceptionHandler(0, &PortFatalVectored);
	_set_invalid_parameter_handler(&PortInvalidParameter);
	_set_purecall_handler(&PortPureCall);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT); // (abort: our SIGABRT handler, not a silent fast fail)
	std::signal(SIGABRT, &PortAbort);
	std::set_terminate(&PortTerminate);
	ULONG guarantee = 64 * 1024; // (the main thread: room for the filter after a stack overflow)
	SetThreadStackGuarantee(&guarantee);
	if (HANDLE t = CreateThread(nullptr, 256 * 1024, &CrashRearmThread, nullptr, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr); t != nullptr) {
		SetThreadPriority(t, THREAD_PRIORITY_BELOW_NORMAL);
		CloseHandle(t);
	}
}

// A run the kernel ended (a fast fail: no user-mode code ran, nothing above could write) is in Windows Error Reporting's
// archive or queue: its Report.wer copied to logs\ and its exception code, module and offset in this run's log, once.
void ReportPreviousWerCrash(const fs::path& logs) {
	std::error_code ec;
	const fs::path  seen_file = logs / L"wer-seen.txt";
	const std::string seen    = [&] {
		std::ifstream in(seen_file, std::ios::binary);
		return std::string {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
	}();
	std::string now_seen = seen;
	const auto  program_data = [] {
		wchar_t v[MAX_PATH] = {};
		return GetEnvironmentVariableW(L"ProgramData", v, MAX_PATH) > 0 ? fs::path(v) : fs::path(L"C:\\ProgramData");
	}();
	const auto local = [] {
		wchar_t v[MAX_PATH] = {};
		return GetEnvironmentVariableW(L"LOCALAPPDATA", v, MAX_PATH) > 0 ? fs::path(v) : fs::path();
	}();
	std::vector<fs::path> roots = {program_data / L"Microsoft\\Windows\\WER\\ReportArchive", program_data / L"Microsoft\\Windows\\WER\\ReportQueue"};
	if (!local.empty()) {
		roots.push_back(local / L"Microsoft\\Windows\\WER\\ReportArchive");
		roots.push_back(local / L"Microsoft\\Windows\\WER\\ReportQueue");
	}
	const auto cutoff = fs::file_time_type::clock::now() - std::chrono::hours(72);
	int        found  = 0;
	for (const auto& root: roots) {
		if (!fs::exists(root, ec)) continue;
		for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
			const std::wstring dir = it->path().filename().wstring();
			if (dir.find(L"Demon's Souls") == std::wstring::npos && dir.find(L"Demon") == std::wstring::npos) continue;
			if (it->last_write_time(ec) < cutoff) continue;
			const std::string key = Utf8(dir);
			if (seen.find(key) != std::string::npos) continue;
			const fs::path report = it->path() / L"Report.wer";
			if (!fs::exists(report, ec)) continue;
			// Report.wer is UTF-16: the Sig[n].Name / Sig[n].Value pairs (application, module, offset, exception code).
			std::ifstream in(report, std::ios::binary);
			const std::string bytes {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
			std::string text;
			for (size_t i = (bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0xFF) ? 2 : 0; i + 1 < bytes.size(); i += 2)
				if (bytes[i + 1] == 0 && bytes[i] != 0) text += bytes[i];
			std::string sigs;
			std::map<std::string, std::string> names;
			size_t at = 0;
			while (at < text.size()) {
				size_t eol = text.find_first_of("\r\n", at);
				if (eol == std::string::npos) eol = text.size();
				const std::string line = text.substr(at, eol - at);
				at                     = eol + 1;
				if (line.rfind("Sig[", 0) != 0) continue;
				const size_t close = line.find(']'), eq = line.find('=');
				if (close == std::string::npos || eq == std::string::npos) continue;
				const std::string idx = line.substr(4, close - 4), field = line.substr(close + 2, eq - close - 2), value = line.substr(eq + 1);
				if (field == "Name") names[idx] = value;
				else if (field == "Value") sigs += (sigs.empty() ? "" : ", ") + (names.count(idx) != 0 ? names[idx] : idx) + " " + value;
			}
			fs::copy_file(report, logs / (L"wer-" + dir + L".txt"), fs::copy_options::overwrite_existing, ec);
			std::printf("[port] previous crash (Windows Error Reporting %s): %s\n", key.c_str(), sigs.empty() ? "(no signature)" : sigs.c_str());
			now_seen += key + "\n";
			found++;
		}
	}
	if (found > 0) {
		std::ofstream(seen_file, std::ios::binary | std::ios::trunc) << now_seen;
		std::fflush(stdout);
	}
}

} // namespace

namespace Port {

void CloseSplash() {
	if (HWND w = g_splash.window.load(); w != nullptr) PostMessageW(w, WM_CLOSE, 0, 0);
}

void ReturnToLauncher() {
	ReopenLauncher();
}

void RequestRestart(bool to_launcher) {
	if (to_launcher) g_restart_launcher = true;
	g_restart = true;
	RequestQuit();
}

} // namespace Port

int main(int argc, char* argv[]) {
	// (10-27) The update's helper mode (run elevated when the folder cannot be written): only that, then out.
	if (const int helper = Port::Update::HelperMain(0, nullptr); helper >= 0) return helper;
	g_exe               = ExePath();
	const fs::path root = g_exe.parent_path();
	SetCurrentDirectoryW(root.c_str());
	std::error_code ec;
	// (10-27) What an update left: the exe and the DLLs that were running then, renamed to .old.
	const int olds = Port::Update::RemoveOldFiles(root);
	fs::create_directories(root / L"logs", ec);
	g_crash_dir = (root / L"logs").wstring();
	InstallCrashReports(); // (ds-pc-port38) filter + re-arm, last vectored handler, C runtime handlers, minidump
	// (ds-pc-port29) --fg-relaunch: the game in a fresh process after the launcher (DLSS frame generation, see below).
	bool fg_relaunch = false;
	for (int i = 1; i < argc; i++)
		if (std::string(argv[i]) == "--fg-relaunch") fg_relaunch = true;
	const bool restarted = (argc > 1 && std::string(argv[1]) == "--restarted") || fg_relaunch;
	// (10-28b) The launcher: --sin-lanzador (or --no-launcher) skips it this time, --lanzador shows it this time.
	bool skip_launcher = false, force_launcher = false;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--sin-lanzador" || a == "--no-launcher") skip_launcher = true;
		if (a == "--lanzador" || a == "--launcher") force_launcher = true;
	}
	// (10-27c) --add-to-steam: the Steam shortcut (ds-steam.cpp), its result in a message, then out.
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) != "--add-to-steam") continue;
		std::string report;
		const bool  ok = Port::Steam::AddToSteam(g_exe, &report);
		Message(report, ok ? MB_ICONINFORMATION : MB_ICONWARNING);
		return ok ? 0 : 1;
	}

	// One game at a time (a restart waits for the one before to end).
	HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\DemonsSoulsPC");
	// (A game just closed may still be ending: saving its caches, its precompile stopping. Waited for, up to 30 s, instead
	// of quitting at once.)
	if (GetLastError() == ERROR_ALREADY_EXISTS && WaitForSingleObject(single, restarted ? 60000 : 30000) == WAIT_TIMEOUT) {
		Message("Demon's Souls ya se esta ejecutando.", MB_ICONINFORMATION);
		return 0;
	}

	// The run's log (run-windows.ps1's logs\<time>.out.log / .err.log; F9's debug snapshot names it).
	SYSTEMTIME t;
	GetLocalTime(&t);
	char stamp[40];
	std::snprintf(stamp, sizeof(stamp), "%04u%02u%02u-%02u%02u%02u-%03u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond,
	              t.wMilliseconds);
	const std::string log_out = Utf8((root / L"logs" / Wide(std::string(stamp) + ".out.log")).wstring());
	const std::string log_err = Utf8((root / L"logs" / Wide(std::string(stamp) + ".err.log")).wstring());
	if (_wfreopen(Wide(log_out).c_str(), L"w", stdout) != nullptr)
		SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdout))));
	if (_wfreopen(Wide(log_err).c_str(), L"w", stderr) != nullptr)
		SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stderr))));
	SetEnv("KYTY_RUN_LOG", log_out);
	// (ds-pc-port29) Unbuffered: a crash no longer loses the last lines.
	setvbuf(stdout, nullptr, _IONBF, 0);
	setvbuf(stderr, nullptr, _IONBF, 0);
	if (fg_relaunch) g_back_to_launcher = true; // (11-09) started by the launcher's JUGAR: its end opens it again
	if (fg_relaunch) std::printf("[port] DLSS / Streamline: the game in its own process (no launcher window or D3D before Streamline)\n");
	ReportPreviousWerCrash(root / L"logs"); // (ds-pc-port38) a fast fail of an earlier run, from Windows Error Reporting

	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
	const std::string ini   = Utf8((root / L"settings.ini").wstring());
	const std::string image = Utf8((root / L"fondo-souls.jpg").wstring());
	Port::Settings settings;
	if (!Port::LoadSettings(ini, &settings)) {
		// A first run: the launcher's game-path.txt and ram16.txt when this folder has them.
		const std::string old = ReadText(root / L"game-path.txt");
		if (!old.empty()) {
			std::string path = old;
			if (path.size() >= 3 && static_cast<unsigned char>(path[0]) == 0xEF) path = path.substr(3);
			while (!path.empty() && (path.back() == '\r' || path.back() == '\n' || path.back() == ' ')) path.pop_back();
			settings.game = path;
		}
		settings.profile16 = fs::exists(root / L"ram16.txt", ec);
	}
	// (10-27b) The menus' language: settings.ini's idioma_menu, else (a first start) the Windows UI language; the
	// game's follows it until chosen.
	// (ds-pc-port9) An empty idioma_interfaz: Windows' UI language (GetUserDefaultUILanguage) at every start, not stored.
	const std::string ui_language = settings.menu_language.empty() ? Port::I18n::DetectWindowsLanguage() : settings.menu_language;
	if (!settings.game_language_set) settings.language = Port::I18n::GameLanguageFor(ui_language);
	Port::I18n::SetLanguage(ui_language);
	std::printf("[port] language: %s (%s)\n", ui_language.c_str(), settings.menu_language.empty() ? "Windows" : "chosen");
	if (!restarted && !RamWarning()) return 0;
	ShowSplash(image);
	// (10-27b) With the start menu the game folder is chosen there (its "Carpeta del juego" row, or at JUGAR).
	// (10-27c) No start menu: the exe starts the game (the folder asked once, settings.ini keeps it; options in F1).
	// (10-28b) The launcher comes first (settings.ini menu_inicio=1, the default), and always at a first start (it holds
	// the Configurador, configurado=0); not after F1's "Reiniciar ahora", nor with --sin-lanzador.
	// (ds-pc-port8) The launcher always comes first now (the game has no options menu any more: F1 saves and quits);
	// --sin-lanzador skips it once.
	// (ds-pc-port38) --lanzador (the restart after an update or a restore) shows it even with --restarted: the new version
	// opens at the launcher, never the game by itself. --fg-relaunch (DLSS-G's fresh process after JUGAR) never.
	// (11-09) settings.ini menu_inicio=0 ("No mostrar al iniciar", F1 > Sistema) is honoured again: the game starts
	// directly. A first start (configurado=0) still shows the launcher; --lanzador shows it once.
	const bool start_menu = !fg_relaunch && (force_launcher || (!skip_launcher && !restarted && (settings.start_menu || !settings.configured)));
	if (force_launcher && restarted && !fg_relaunch) std::printf("[port] restarted after an update / restore: the launcher first\n");
	while (!start_menu && !IsGame(settings.game)) {
		if (!settings.game.empty())
			Message("No hay eboot.bin en:\n" + settings.game + "\n\nElige la carpeta del juego volcado (la que tiene eboot.bin y sce_sys).");
		SplashStatus("Elige la carpeta del juego...");
		const std::string picked = PickFolder();
		if (picked.empty()) {
			Port::CloseSplash();
			return 0;
		}
		settings.game = picked;
	}
	Port::SaveSettings(ini, settings);

	GameParam param = ReadParam(settings.game);
	std::printf("[port] Demon's Souls PC %s: %s %s %s; settings %s\n", Port::Update::CurrentVersion(), param.name.c_str(),
	            param.title_id.c_str(), param.version.c_str(), ini.c_str());
	if (olds > 0) std::printf("[port] update: %d .old file(s) of the previous version deleted\n", olds);
	const Port::PcInfo pc = DetectPc();
	std::printf("[port] PC: %s (%llu MB), %s, %d GB RAM%s\n", pc.gpu.c_str(), static_cast<unsigned long long>(pc.vram >> 20), pc.cpu.c_str(),
	            pc.ram_gb, Port::LowRam(settings, pc) ? " (16 GB profile)" : "");
	std::string version = param.title_id + " " + param.version;
	if (!param.title_id.empty() && version != "PPSA01341 01.007.000" && version != "PPSA01341 01.005.000")
		std::printf("[port] untested game version %s (tested: PPSA01341 01.007.000 and 01.005.000)\n", version.c_str());

	const auto make_runtime = [&](const Port::LaunchPlan& plan) {
	Port::Runtime runtime;
	runtime.exe_dir    = Utf8(root.wstring()) + "\\";
	runtime.ini_path   = ini;
	runtime.image_path = image;
	runtime.game_info  = (param.name.empty() ? std::string("Demon's Souls") : param.name) +
	                    (param.title_id.empty() ? std::string() : "  " + version);
	runtime.launched   = settings;
	runtime.notes      = plan.notes;
	runtime.pc         = pc;
	runtime.exe_path   = g_exe;
	{
		// The game's folder, when it is inside the port's (the update never touches it).
		const fs::path game = fs::path(Wide(settings.game));
		const fs::path rel  = game.lexically_relative(root);
		if (!rel.empty() && rel.native()[0] != L'.') runtime.game_rel = rel;
	}
	return runtime;
	};
	// CPU affinity (run-windows.ps1): every CPU, or the one L3 cache's of a CPU with several (the 3D V-Cache CCD).
	DWORD_PTR process_mask = 0, system_mask = 0;
	GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
	const auto cpu_mask = [&]() {
		uint64_t m = system_mask;
		if (!settings.all_ccds)
			if (const uint64_t ccd = LargestL3Mask() & system_mask; ccd != 0) m = ccd;
		return m;
	};
	uint64_t mask = cpu_mask();
	const auto count_cpus = [](uint64_t m) {
		int n = 0;
		for (int bit = 0; bit < 64; bit++) n += (m >> bit) & 1;
		return n;
	};
	int cpus = count_cpus(mask);
	// (10-27b) Shaders first (the seed file, the release's hints, the prefetch inputs: splash "Preparando shaders 1/3..3/3"),
	// then the start menu, which runs the static precompile (precompile-windows.ps1) when this GPU's cache is not current.
	// (Start-up time by phase in the log: "[port] tiempo de arranque: ...".)
	const auto phase_start = std::chrono::steady_clock::now();
	auto       phase_at    = phase_start;
	const auto phase_done  = [&](const char* what) {
		const auto now = std::chrono::steady_clock::now();
		std::printf("[port] tiempo de arranque: %s %.1f s (total %.1f s)\n", what,
		            std::chrono::duration<double>(now - phase_at).count(), std::chrono::duration<double>(now - phase_start).count());
		std::fflush(stdout);
		phase_at = now;
	};
	// (ds-pc-port47) Shader preparation on every CPU of the PC, without the game's V-Cache affinity.
	const int all_cpus = std::max(1, static_cast<int>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)));
	if (IsGame(settings.game)) PrepareShaders(root, settings.game, param, 0, all_cpus);
	phase_done("preparar shaders (1/3-3/3)");
	{
		Port::Precompile::Setup ps;
		ps.root     = root;
		ps.game     = settings.game;
		ps.id       = param.title_id.empty() ? std::string() : param.title_id + "_" + param.version;
		ps.seeds    = root / Wide(ps.id.empty() ? std::string("seeds.seeds") : "seeds-" + ps.id + ".seeds");
		ps.affinity = 0; // (ds-pc-port47: the precompile on every CPU)
		ps.cpus     = cpus;
		ps.cache_key     = Port::DriverCacheKey(settings);
		ps.shadow_filter = Port::ShadowFilterLevel(settings);
		{
			const uint64_t pa = ps.affinity;
			const int      pc = cpus;
			ps.refresh        = [root, ini, pa, pc](Port::Precompile::Setup& setup) {
				Port::Settings now;
				if (!Port::LoadSettings(ini, &now) || now.game.empty() || !IsGame(now.game)) return;
				setup.cache_key     = Port::DriverCacheKey(now);
				setup.shadow_filter = Port::ShadowFilterLevel(now);
				const GameParam p   = ReadParam(now.game);
				const std::string id = p.title_id.empty() ? std::string() : p.title_id + "_" + p.version;
				// The seed file, hints and prefetch inputs of that folder's version: only for another folder or version (the
				// start-up one was prepared at the splash; doing it again here doubled the start-up's shader steps).
				if (now.game != setup.game || id != setup.id) PrepareShaders(root, now.game, p, pa, pc);
				if (now.game != setup.game || id != setup.id)
					std::printf("[port] precompile: game folder %s (%s), was %s (%s)\n", now.game.c_str(), id.c_str(), setup.game.c_str(),
					            setup.id.c_str());
				setup.game  = now.game;
				setup.id    = id;
				setup.seeds = root / Wide(id.empty() ? std::string("seeds.seeds") : "seeds-" + id + ".seeds");
				std::fflush(stdout);
			};
		}
		Port::Precompile::Configure(ps);
		std::error_code pe;
		if (!ps.id.empty() && fs::exists(ps.seeds, pe)) {
			SplashStatus("Comprobando la cache de shaders...");
			const bool current = Port::Precompile::CacheCurrent();
			phase_done("comprobar la cache");
			std::printf("[port] driver pipeline cache: %s\n", current ? "every static recipe in it" : "to be made (start-up precompile)");
			std::fflush(stdout);
			if (!current && start_menu) {
				// (10-28b) The launcher shows first: the precompile runs beside it, with its bar (JUGAR stops it, what
				// is done is kept). (ds-pc-port13: all CPU threads or relaxed, asked the first time)
				Port::Precompile::Choose(ini, &settings);
				Port::Precompile::Start();
			} else if (!current && !fg_relaunch) {
				// (10-27c) Every pipeline of the game into this GPU's driver cache, before the game, with its bar; Esc plays
				// now (what is done is kept, the rest the next start or beside the game).
				g_splash.skip = false;
				Port::Precompile::Choose(ini, &settings); // (ds-pc-port13)
				if (HWND w = g_splash.window.load(); w != nullptr) SetForegroundWindow(w);
				Port::Precompile::Start();
				for (;;) {
					const auto v = Port::Precompile::State();
					if (!v.running) {
						if (v.failed) std::printf("[port] precompile: %s\n", v.text.c_str());
						break;
					}
					if (v.total > 0) SplashProgress(static_cast<float>(v.done) / static_cast<float>(v.total));
					SplashStatus(v.text);
					SplashHint("Esc: jugar ya (lo hecho se guarda, pero el juego tendrá tirones hasta que termine la precompilación)\n"
					           "Si cierras el juego, la precompilación seguirá donde la dejaste.");
					if (g_splash.skip) {
						SplashStatus("Guardando lo hecho...");
						Port::Precompile::Stop();
						break;
					}
					Sleep(200);
				}
				SplashProgress(-1.0f);
				SplashHint("");
				std::printf("[port] precompile: %s\n", Port::Precompile::State().text.c_str());
				phase_done("precompilacion antes del juego");
			}
		}
	}
	if (start_menu) {
		const std::string game_before = settings.game;
		Port::Runtime     menu_rt     = make_runtime(Port::BuildLaunch(settings, param.title_id, pc));
		menu_rt.is_game               = [](const std::string& p) { return IsGame(p); };
		// (ds-pc-port8) The launcher's background: fondo-launcher.jpg (the key art with the logo), else the splash's.
		if (fs::exists(root / L"fondo-launcher.jpg", ec)) menu_rt.image_path = Utf8((root / L"fondo-launcher.jpg").wstring());
		menu_rt.pick_folder           = [] { return PickFolder(); };
		std::printf("[port] launcher: shown\n");
		const bool play = Port::RunStartMenu(menu_rt, &settings);
		// (ds-pc-port38) How the launcher ended, in the log: a log that stops at the cache line was the launcher closed.
		std::printf("[port] launcher: %s\n", play ? "JUGAR" : "closed (Salir, Esc, the window's X, an update or a restart)");
		Port::Precompile::Stop(/*save=*/play); // (closing: out at once, the shards merged at the next start)
		if (!play || !IsGame(settings.game)) {
			// (An update or a "Reiniciar ahora" from the launcher: it closed asking for a restart.)
			if (single != nullptr) ReleaseMutex(single);
			RestartIfAsked();
			return 0;
		}
		settings.configured = true; // (10-28b) the Configurador is the launcher's Sistema tab
		g_back_to_launcher = true; // (11-09) the game's end opens the launcher again
		Port::SaveSettings(ini, settings);
		// (ds-pc-port29) DLSS frame generation (Streamline's sl.interposer, the nvapi redirect, Reflex) must be the first
		// graphics in its process, as in the emulator started from PowerShell: the launcher's SDL renderer has loaded
		// D3D and NVIDIA's driver here (the process closed with no error). The game in a fresh process, no launcher.
		{
			const Port::LaunchPlan fg_plan = Port::BuildLaunch(settings, param.title_id, pc);
			bool                   dlssg   = false;
			// (ds-pc-port38) Not only DLSS frame generation: Streamline (sl.interposer, its nvapi redirect and Reflex) also
			// loads for the DLSS upscaler (KYTY_UPSCALER=dlss, frame-gen.cpp Initialize), and NVIDIA's NGX for DLSS 5 (NR):
			// after the launcher's D3D renderer in the same process that is the same silent end port29 found (no crash
			// filter runs, the log just stops), here at the first 3D frame (the DLSS / NR evaluate) with any frame
			// generation, AMD's included (the 11-01 log: DLSS Rendimiento + AMD MFG dinamico, gone at the title screen).
			const char* why = nullptr;
			for (const auto& [key, value]: fg_plan.env) {
				if (key == "KYTY_FRAMEGEN" && !value.empty() && value != "0") { dlssg = true; why = "DLSS frame generation"; }
				if (why == nullptr && key == "KYTY_UPSCALER" && value == "dlss") { dlssg = true; why = "DLSS (Streamline)"; }
				if (why == nullptr && key == "KYTY_FOVEATED" && value == "1") { dlssg = true; why = "DLSS 5 (NGX)"; }
			}
			if (dlssg) {
				std::printf("[port] %s: starting the game in a fresh process\n", why);
				std::fflush(stdout);
				ReleaseMutex(single);
				CloseHandle(single);
				std::wstring        command = L"\"" + g_exe.wstring() + L"\" --fg-relaunch --sin-lanzador";
				STARTUPINFOW        si {};
				PROCESS_INFORMATION pi {};
				si.cb = sizeof(si);
				if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, root.c_str(), &si, &pi)) {
					CloseHandle(pi.hThread);
					CloseHandle(pi.hProcess);
					Port::CloseSplash();
					return 0;
				}
				std::printf("[port] DLSS frame generation: CreateProcess failed (%lu): in this process\n", GetLastError());
				single = CreateMutexW(nullptr, TRUE, L"Local\\DemonsSoulsPC");
			}
		}
		ShowSplash(image);
		SplashStatus("Iniciando...");
		mask = cpu_mask();
		cpus = count_cpus(mask);
		if (settings.game != game_before) {
			// (Another game folder: its version's seeds, hints and prefetch inputs.)
			param   = ReadParam(settings.game);
			version = param.title_id + " " + param.version;
			std::printf("[port] game folder: %s (%s)\n", settings.game.c_str(), version.c_str());
			PrepareShaders(root, settings.game, param, 0, all_cpus);
		}
	}

	// (10-28) The Configurador at the first start (configurado=0 in settings.ini); later from F1 > Sistema.
	// (10-28b) Folded into the launcher's Sistema tab: shown only if the launcher could not be (no window).
	if (!settings.configured && !start_menu && !skip_launcher) {
		Port::Runtime cfg_rt = make_runtime(Port::BuildLaunch(settings, param.title_id, pc));
		Port::CloseSplash();
		Port::RunConfigurator(cfg_rt, &settings);
		settings.configured = true;
		Port::SaveSettings(ini, settings);
		ShowSplash(image);
		SplashStatus("Iniciando...");
	}

	Port::LaunchPlan plan = Port::BuildLaunch(settings, param.title_id, pc);
	// (10-28) "Calibrar cámara" runs once.
	if (settings.kbm_calibrate) {
		Port::Settings next = settings;
		next.kbm_calibrate  = false;
		Port::SaveSettings(ini, next);
	}
	for (const auto& note: plan.notes) std::printf("[port] %s\n", note.c_str());

	for (auto& [key, value]: plan.env) {
		if ((key == "KYTY_RECORDING_CPUS" || key == "KYTY_RENDER_CPUS") && value == "auto") value = PerformanceCpus(mask);
		// (ds-pc-port38) "Carga de shaders al maximo": the shader prefetch and warmup threads as many as the CPUs the game
		// may run on (the V-Cache CCD: 16 on a 9950X3D), less the render and game threads, not every CPU of the PC
		// (hardware_concurrency: 32 threads squeezed onto 16 CPUs next to the render thread, the in-game hitches).
		// (ds-pc-port46) The warmup threads as the emulator launcher's fixed 16 (was the CCD's CPUs - 2: 14).
		// (ds-pc-port47) The prefetch threads left to the emulator's default (no override).
	}
	for (const auto& [key, value]: plan.env) SetEnv(key, value);
	std::printf("[port] switches: %zu, vblank %d Hz\n", plan.env.size(), plan.vblank);
	std::fflush(stdout);

	if (settings.backup_saves) BackupSaves(root);
	// (ds-pc-port41) Metricas is drawn in the game now (ds-menu.cpp MetricsPanel: metricas.ps1's rows, top-centre); the
	// separate metricas.ps1 window only with KYTY_PORT_METRICS_WINDOW=1.
	if ((settings.metrics == "1" || settings.metrics == "2") && std::getenv("KYTY_PORT_METRICS_WINDOW") != nullptr &&
	    std::string(std::getenv("KYTY_PORT_METRICS_WINDOW")) == "1")
		StartMetricsWindow(root, settings, pc);
	// (ds-pc-port47) The game's threads on its CPUs (the V-Cache CCD) by the process's default CPU sets, the process itself
	// on every CPU: the shader workers use them all (LocalPlatform::ApplyGameCpus, MakeShaderWorkerThread).
	if (mask != system_mask) {
		std::string list;
		for (int bit = 0; bit < 64; bit++)
			if ((mask >> bit) & 1u) list += (list.empty() ? "" : ",") + std::to_string(bit);
		SetEnv("KYTY_GAME_CPUS", list);
	}
	SplashStatus("Iniciando...");

	{
		// (ds-pc-port43) No Windows power throttling (EcoQoS) for the game, normal priority (a relaxed precompile left
		// nothing behind: it runs in its own processes), 1 ms timers.
		PROCESS_POWER_THROTTLING_STATE pt {};
		pt.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
		pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | 0x4 /* IGNORE_TIMER_RESOLUTION */;
		pt.StateMask   = 0;
		const BOOL off = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt));
		SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
		timeBeginPeriod(1);
		std::printf("[port] CPU: power throttling %s, priority normal, timer 1 ms\n", off ? "off" : "unchanged (not supported)");
		std::fflush(stdout);
	}
	Port::Runtime runtime = make_runtime(plan);
	Port::InstallMenu(runtime);
	// (ds-pc-port8) After F1 (save and quit): the title passed by itself and the character back where F1 left.
	Port::Resume::Prepare(root, param.title_id, false); // (ds-pc-port37) "continue where F1 left" removed
	std::atexit(RestartIfAsked);
	std::at_quick_exit(RestartIfAsked);

	std::vector<std::string> args = {Utf8(g_exe.wstring())};
	args.insert(args.end(), plan.args.begin(), plan.args.end());
	std::vector<char*> argv2;
	for (auto& a: args) argv2.push_back(a.data());
	argv2.push_back(nullptr);
	const int code = KytyEmulatorMain(static_cast<int>(args.size()), argv2.data());
	RestartIfAsked();
	return code;
}
