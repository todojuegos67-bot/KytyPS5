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

#include <atomic>
#include <cctype>
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
	uint64_t best_mask = 0, best_size = 0;
	int      count     = 0;
	for (DWORD offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
		if (info->Relationship == RelationCache && info->Cache.Level == 3 && info->Cache.GroupMask.Group == 0) {
			count++;
			if (info->Cache.CacheSize > best_size) {
				best_size = info->Cache.CacheSize;
				best_mask = info->Cache.GroupMask.Mask;
			}
		}
		offset += info->Size;
	}
	return count >= 2 ? best_mask : 0;
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
fs::path          g_exe;

void RestartIfAsked() {
	if (!g_restart.exchange(false)) return;
	std::wstring        command = L"\"" + g_exe.wstring() + L"\" --restarted";
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, g_exe.parent_path().c_str(), &si, &pi)) {
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
	}
}

// (ds-pc-port29) A crash leaves logs\crash-<time>.txt: the exception code, the faulting module and its offset.
std::wstring g_crash_dir;
LONG WINAPI PortCrashFilter(EXCEPTION_POINTERS* info) {
	const DWORD code = info != nullptr && info->ExceptionRecord != nullptr ? info->ExceptionRecord->ExceptionCode : 0;
	void*       addr = info != nullptr && info->ExceptionRecord != nullptr ? info->ExceptionRecord->ExceptionAddress : nullptr;
	HMODULE     mod  = nullptr;
	wchar_t     name[MAX_PATH] = L"?";
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCWSTR>(addr), &mod) &&
	    mod != nullptr)
		GetModuleFileNameW(mod, name, MAX_PATH);
	const unsigned long long off = mod != nullptr ? reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(mod) : 0;
	SYSTEMTIME t;
	GetLocalTime(&t);
	wchar_t file[MAX_PATH + 64];
	swprintf(file, MAX_PATH + 64, L"%ls\\crash-%04u%02u%02u-%02u%02u%02u.txt", g_crash_dir.c_str(), t.wYear, t.wMonth, t.wDay, t.wHour,
	         t.wMinute, t.wSecond);
	if (FILE* f = _wfopen(file, L"w"); f != nullptr) {
		std::fwprintf(f, L"exception 0x%08lx at %p\nmodule %ls + 0x%llx\nstack:\n", code, addr, name, off);
		// (ds-pc-port32) The faulting thread's stack from its context (x64 unwind data), up to 32 frames.
		if (info != nullptr && info->ContextRecord != nullptr) {
			CONTEXT ctx = *info->ContextRecord;
			for (int frame = 0; frame < 32 && ctx.Rip != 0; frame++) {
				HMODULE fm = nullptr;
				wchar_t fn[MAX_PATH] = L"?";
				if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                       reinterpret_cast<LPCWSTR>(ctx.Rip), &fm) && fm != nullptr)
					GetModuleFileNameW(fm, fn, MAX_PATH);
				std::fwprintf(f, L"  #%02d %ls + 0x%llx\n", frame, fn,
				              fm != nullptr ? static_cast<unsigned long long>(ctx.Rip - reinterpret_cast<uintptr_t>(fm)) : static_cast<unsigned long long>(ctx.Rip));
				DWORD64 image = 0;
				RUNTIME_FUNCTION* fe = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
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
	std::printf("[port] crash: exception 0x%08lx in %ls + 0x%llx\n", code, name, off);
	std::fflush(stdout);
	return EXCEPTION_CONTINUE_SEARCH;
}

} // namespace

namespace Port {

void CloseSplash() {
	if (HWND w = g_splash.window.load(); w != nullptr) PostMessageW(w, WM_CLOSE, 0, 0);
}

void RequestRestart() {
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
	SetUnhandledExceptionFilter(PortCrashFilter);
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
	if (GetLastError() == ERROR_ALREADY_EXISTS && WaitForSingleObject(single, restarted ? 60000 : 0) == WAIT_TIMEOUT) {
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
	if (fg_relaunch) std::printf("[port] DLSS frame generation: the game in its own process (no launcher window or D3D before Streamline)\n");

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
	(void)force_launcher;
	const bool start_menu = !skip_launcher && !restarted;
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
	if (IsGame(settings.game)) PrepareShaders(root, settings.game, param, mask != system_mask ? mask : 0, cpus);
	{
		Port::Precompile::Setup ps;
		ps.root     = root;
		ps.game     = settings.game;
		ps.id       = param.title_id.empty() ? std::string() : param.title_id + "_" + param.version;
		ps.seeds    = root / Wide(ps.id.empty() ? std::string("seeds.seeds") : "seeds-" + ps.id + ".seeds");
		ps.affinity = mask != system_mask ? mask : 0;
		ps.cpus     = cpus;
		Port::Precompile::Configure(ps);
		std::error_code pe;
		if (!ps.id.empty() && fs::exists(ps.seeds, pe)) {
			SplashStatus("Comprobando la cache de shaders...");
			const bool current = Port::Precompile::CacheCurrent();
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
		const bool play = Port::RunStartMenu(menu_rt, &settings);
		Port::Precompile::Stop();
		if (!play || !IsGame(settings.game)) return 0;
		settings.configured = true; // (10-28b) the Configurador is the launcher's Sistema tab
		Port::SaveSettings(ini, settings);
		// (ds-pc-port29) DLSS frame generation (Streamline's sl.interposer, the nvapi redirect, Reflex) must be the first
		// graphics in its process, as in the emulator started from PowerShell: the launcher's SDL renderer has loaded
		// D3D and NVIDIA's driver here (the process closed with no error). The game in a fresh process, no launcher.
		{
			const Port::LaunchPlan fg_plan = Port::BuildLaunch(settings, param.title_id, pc);
			bool                   dlssg   = false;
			for (const auto& [key, value]: fg_plan.env)
				if (key == "KYTY_FRAMEGEN" && !value.empty() && value != "0") dlssg = true;
			if (dlssg) {
				std::printf("[port] DLSS frame generation: starting the game in a fresh process\n");
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
			PrepareShaders(root, settings.game, param, mask != system_mask ? mask : 0, cpus);
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
	}
	for (const auto& [key, value]: plan.env) SetEnv(key, value);
	std::printf("[port] switches: %zu, vblank %d Hz\n", plan.env.size(), plan.vblank);
	std::fflush(stdout);

	if (settings.backup_saves) BackupSaves(root);
	if (mask != system_mask) SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(mask));
	SplashStatus("Iniciando...");

	Port::Runtime runtime = make_runtime(plan);
	Port::InstallMenu(runtime);
	// (ds-pc-port8) After F1 (save and quit): the title passed by itself and the character back where F1 left.
	Port::Resume::Prepare(root, param.title_id, settings.continue_f1);
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
