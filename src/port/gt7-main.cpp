// "Gran Turismo 7" PC port: "Gran Turismo 7.exe" (gt7-port.h). The launcher, then kyty_emulator.exe with the chosen
// options (its output to logs\), then the launcher again when the game ends (unless cerrar_lanzador).

#include "port/gt7-i18n.h"
#include "port/gt7-port.h"
#include "port/gt7-settings.h"

#include <windows.h>
#include <dxgi.h>
#include <shobjidl.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace Port {

std::wstring Wide(const std::string& s) {
	if (s.empty()) return {};
	const int    n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring w(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
	return w;
}

std::string Utf8(const std::wstring& w) {
	if (w.empty()) return {};
	const int   n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
	std::string s(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
	return s;
}

bool IsGame(const std::string& path) {
	if (path.empty()) return false;
	const fs::path  p = fs::path(Wide(path));
	std::error_code ec;
	if (fs::exists(p / L"eboot.bin", ec)) return true;
	return (p.extension() == L".zar" || p.filename() == L"eboot.bin") && fs::is_regular_file(p, ec);
}

namespace {

std::string ShowDialog(bool folder, const wchar_t* filter_name, const wchar_t* filter) {
	std::string      result;
	IFileOpenDialog* dialog = nullptr;
	if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return result;
	DWORD options = 0;
	dialog->GetOptions(&options);
	dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
	if (folder) {
		dialog->SetTitle(L"Gran Turismo 7: elige la carpeta del juego volcado (PPSA01317-app, la que tiene eboot.bin)");
	} else {
		COMDLG_FILTERSPEC spec {filter_name, filter};
		dialog->SetFileTypes(1, &spec);
	}
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

PcInfo DetectPc() {
	PcInfo         pc;
	IDXGIFactory1* factory = nullptr;
	if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) && factory != nullptr) {
		int            best_tier = -1;
		uint64_t       best_vram = 0;
		IDXGIAdapter1* adapter   = nullptr;
		for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
			DXGI_ADAPTER_DESC1 d {};
			if (SUCCEEDED(adapter->GetDesc1(&d)) && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
				std::string name = Utf8(d.Description);
				while (!name.empty() && name.back() == ' ') name.pop_back();
				const int      tier = d.VendorId == 0x1414 ? 0 : (d.DedicatedVideoMemory >= (2ull << 30) ? 2 : 1);
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
		wchar_t vendor[64] = {};
		size               = sizeof(vendor) - sizeof(wchar_t);
		if (RegQueryValueExW(key, L"VendorIdentifier", nullptr, nullptr, reinterpret_cast<LPBYTE>(vendor), &size) == ERROR_SUCCESS)
			pc.amd_cpu = std::wstring(vendor).find(L"AMD") != std::wstring::npos;
		RegCloseKey(key);
	}
	if (pc.cpu.find("AMD") != std::string::npos) pc.amd_cpu = true;
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
	return pc;
}

// The CCD with the largest L3 of an asymmetric V-Cache CPU (7950X3D, 9950X3D); 0 on any other.
uint64_t LargestL3Mask() {
	DWORD length = 0;
	GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
	if (length == 0) return 0;
	std::vector<uint8_t> buffer(length);
	if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) return 0;
	uint64_t best_mask = 0, best_size = 0, min_size = ~uint64_t(0);
	int      count = 0;
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
	return count >= 2 && best_size > min_size ? best_mask : 0;
}

// One argument quoted for CommandLineToArgvW / the CRT.
std::wstring Quote(const std::wstring& a) {
	if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
	std::wstring out = L"\"";
	size_t       backslashes = 0;
	for (wchar_t c: a) {
		if (c == L'\\') {
			backslashes++;
		} else if (c == L'"') {
			out.append(backslashes * 2 + 1, L'\\');
			backslashes = 0;
		} else {
			backslashes = 0;
		}
		out += c;
	}
	out.append(backslashes, L'\\');
	out += L"\"";
	return out;
}

std::wstring CommandLine(const Runtime& rt, const LaunchPlan& plan) {
	std::wstring cmd = Quote((rt.exe_dir / L"kyty_emulator.exe").wstring());
	for (const auto& a: plan.args) cmd += L" " + Quote(Wide(a));
	return cmd;
}

void Message(const std::string& text, UINT flags = MB_ICONWARNING) {
	MessageBoxW(nullptr, Wide(text).c_str(), L"Gran Turismo 7", MB_OK | flags);
}

} // namespace

std::string PickFolder() {
	return ShowDialog(true, nullptr, nullptr);
}

std::string PickFile(const wchar_t* filter_name, const wchar_t* filter) {
	return ShowDialog(false, filter_name, filter);
}

std::string BuildCommandLine(const Runtime& rt, const Settings& s) {
	const LaunchPlan plan = BuildLaunch(s, rt.pc.amd_cpu, rt.pc.screen_w, rt.pc.screen_h);
	std::string      out;
	for (const auto& [k, v]: plan.env) out += k + "=" + v + " ";
	return out + Utf8(CommandLine(rt, plan));
}

} // namespace Port

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
	using namespace Port;
	SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
	(void)CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	Runtime rt;
	{
		wchar_t path[MAX_PATH * 2] = {};
		GetModuleFileNameW(nullptr, path, static_cast<DWORD>(std::size(path)));
		rt.exe_path = fs::path(path);
		rt.exe_dir  = rt.exe_path.parent_path();
	}
	SetCurrentDirectoryW(rt.exe_dir.c_str());
	rt.ini_path = Utf8((rt.exe_dir / L"settings.ini").wstring());
	rt.pc       = DetectPc();

	Settings s;
	const bool had_ini = LoadSettings(rt.ini_path, &s);
	ReadWheelJson(Utf8((rt.exe_dir / L"wheel.json").wstring()), &s);
	if (s.menu_language.empty()) s.menu_language = I18n::DetectWindowsLanguage();
	if (!had_ini) {
		s.console_language = I18n::GameLanguageFor(s.menu_language);
		(void)SaveSettings(rt.ini_path, s);
	}
	I18n::SetLanguage(s.menu_language);

	std::error_code ec;
	if (!fs::exists(rt.exe_dir / L"kyty_emulator.exe", ec)) {
		Message("No está kyty_emulator.exe junto a \"Gran Turismo 7.exe\". Descomprime el paquete completo en una carpeta.");
		return 1;
	}

	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD | SDL_INIT_JOYSTICK)) {
		Message(std::string("SDL no arranca: ") + SDL_GetError());
		return 1;
	}

	std::string status;
	for (;;) {
		if (RunLauncher(rt, &s, status) != LauncherResult::Play) break;
		(void)SaveSettings(rt.ini_path, s);
		if (s.wheel) {
			std::string err;
			if (!WriteWheelJson(Utf8((rt.exe_dir / L"wheel.json").wstring()), s, &err)) status = err;
		}

		const LaunchPlan plan = BuildLaunch(s, rt.pc.amd_cpu, rt.pc.screen_w, rt.pc.screen_h);
		for (const auto& [k, v]: plan.env) SetEnvironmentVariableW(Wide(k).c_str(), v.empty() ? nullptr : Wide(v).c_str());

		// The emulator's output: logs\gt7-<date>.log (or its own console, Desarrollador).
		fs::create_directories(rt.exe_dir / L"logs", ec);
		wchar_t stamp[32];
		{
			const std::time_t t = std::time(nullptr);
			std::tm           tm {};
			localtime_s(&tm, &t);
			wcsftime(stamp, std::size(stamp), L"%Y%m%d-%H%M%S", &tm);
		}
		const fs::path      log_path = rt.exe_dir / L"logs" / (std::wstring(L"gt7-") + stamp + L".log");
		SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
		HANDLE              log = INVALID_HANDLE_VALUE;
		STARTUPINFOW        si {};
		si.cb = sizeof(si);
		DWORD flags = CREATE_SUSPENDED | (s.priority == "alta" ? HIGH_PRIORITY_CLASS : NORMAL_PRIORITY_CLASS);
		if (s.dev_console) {
			flags |= CREATE_NEW_CONSOLE;
		} else {
			log = CreateFileW(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (log != INVALID_HANDLE_VALUE) {
				si.dwFlags    = STARTF_USESTDHANDLES;
				si.hStdOutput = log;
				si.hStdError  = log;
				si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
			}
			flags |= CREATE_NO_WINDOW;
		}
		std::wstring        cmd = CommandLine(rt, plan);
		PROCESS_INFORMATION pi {};
		if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, log != INVALID_HANDLE_VALUE, flags, nullptr, rt.exe_dir.c_str(), &si, &pi)) {
			status = "No se pudo iniciar kyty_emulator.exe (error " + std::to_string(GetLastError()) + ").";
			if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
			continue;
		}
		if (s.x3d_ccd) {
			if (const uint64_t mask = LargestL3Mask(); mask != 0) SetProcessAffinityMask(pi.hProcess, static_cast<DWORD_PTR>(mask));
		}
		ResumeThread(pi.hThread);
		CloseHandle(pi.hThread);

		WatchGame(rt, s, pi.hProcess, pi.dwProcessId);

		DWORD code = 0;
		GetExitCodeProcess(pi.hProcess, &code);
		CloseHandle(pi.hProcess);
		if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
		status = code == 0 ? std::string()
		                   : "El juego terminó con el código " + std::to_string(code) + ". Log: logs\\" + Utf8(log_path.filename().wstring());
		if (s.close_launcher && code == 0) break;
	}
	SDL_Quit();
	CoUninitialize();
	return 0;
}
