#include "debug-snapshot.h"

#include "loader/demonsSoulsWarp.h"
#include "loader/systemContent.h"

#include "frame-gen.h"

#include <SDL.h>
#include <SDL_syswm.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace DebugSnapshot {

namespace {

#if defined(_WIN32)
// The window's client area as the screen shows it (what the player saw: the overlay and the presenter's letterbox
// included), as a PNG (Windows Imaging Component).
bool SaveWindowPicture(HWND window, const std::filesystem::path& path, int* width, int* height) {
	RECT  client {};
	POINT origin {0, 0};
	if (!GetClientRect(window, &client) || !ClientToScreen(window, &origin)) return false;
	const int w = client.right - client.left, h = client.bottom - client.top;
	if (w <= 0 || h <= 0) return false;
	*width  = w;
	*height = h;
	HDC     screen = GetDC(nullptr);
	HDC     memory = CreateCompatibleDC(screen);
	HBITMAP bitmap = CreateCompatibleBitmap(screen, w, h);
	auto*   old    = SelectObject(memory, bitmap);
	bool    copied = BitBlt(memory, 0, 0, w, h, screen, origin.x, origin.y, SRCCOPY | CAPTUREBLT) != FALSE;
	std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
	BITMAPINFO           info {};
	info.bmiHeader = {sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, 0, 0, 0, 0, 0};
	copied = copied && GetDIBits(memory, bitmap, 0, static_cast<UINT>(h), pixels.data(), &info, DIB_RGB_COLORS) == h;
	SelectObject(memory, old);
	DeleteObject(bitmap);
	DeleteDC(memory);
	ReleaseDC(nullptr, screen);
	if (!copied) return false;
	for (size_t i = 3; i < pixels.size(); i += 4) pixels[i] = 0xff; // (GDI leaves the alpha 0: opaque)

	const HRESULT com      = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
	bool          saved    = false;
	IWICImagingFactory*    factory = nullptr;
	IWICStream*            stream  = nullptr;
	IWICBitmapEncoder*     encoder = nullptr;
	IWICBitmapFrameEncode* frame   = nullptr;
	IPropertyBag2*         options = nullptr;
	WICPixelFormatGUID     format  = GUID_WICPixelFormat32bppBGRA;
	const UINT             stride  = static_cast<UINT>(w) * 4;
	if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
	    SUCCEEDED(factory->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
	    SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
	    SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache)) && SUCCEEDED(encoder->CreateNewFrame(&frame, &options)) &&
	    SUCCEEDED(frame->Initialize(options)) && SUCCEEDED(frame->SetSize(static_cast<UINT>(w), static_cast<UINT>(h))) &&
	    SUCCEEDED(frame->SetPixelFormat(&format)) && IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA) &&
	    SUCCEEDED(frame->WritePixels(static_cast<UINT>(h), stride, stride * static_cast<UINT>(h), pixels.data())) &&
	    SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit())) {
		saved = true;
	}
	for (IUnknown* object: {static_cast<IUnknown*>(options), static_cast<IUnknown*>(frame), static_cast<IUnknown*>(encoder),
	                        static_cast<IUnknown*>(stream), static_cast<IUnknown*>(factory)}) {
		if (object != nullptr) object->Release();
	}
	if (SUCCEEDED(com)) CoUninitialize();
	return saved;
}

// The run's log: the file the emulator's standard output goes to, as run-windows.ps1 names it (KYTY_RUN_LOG), else as
// the handle tells.
std::string LogFile() {
	if (const char* named = std::getenv("KYTY_RUN_LOG"); named != nullptr && *named != '\0') return named;
	wchar_t    name[MAX_PATH] {};
	const auto length = GetFinalPathNameByHandleW(GetStdHandle(STD_OUTPUT_HANDLE), name, MAX_PATH, FILE_NAME_NORMALIZED);
	if (length == 0 || length >= MAX_PATH) return "(not a file)";
	return std::filesystem::path(name).string();
}

std::vector<std::string> KytySwitches() {
	std::vector<std::string> out;
	if (auto* block = GetEnvironmentStringsA(); block != nullptr) {
		for (const char* at = block; *at != '\0'; at += std::strlen(at) + 1) {
			if (std::strncmp(at, "KYTY_", 5) == 0) out.emplace_back(at);
		}
		FreeEnvironmentStringsA(block);
	}
	return out;
}
#endif

} // namespace

std::string Take(SDL_Window* window) {
	const auto time = std::time(nullptr);
	std::tm    local {};
#if defined(_WIN32)
	localtime_s(&local, &time);
#else
	localtime_r(&time, &local);
#endif
	char stamp[32] {};
	std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
	std::error_code error;
	const auto      folder = std::filesystem::absolute(std::filesystem::path("_DebugSnapshots") / stamp, error);
	std::filesystem::create_directories(folder, error);
	if (error) return {};

	std::string picture = "none";
#if defined(_WIN32)
	SDL_SysWMinfo wm {};
	SDL_VERSION(&wm.version);
	int width = 0, height = 0;
	if (window != nullptr && SDL_GetWindowWMInfo(window, &wm) == SDL_TRUE && wm.subsystem == SDL_SYSWM_WINDOWS &&
	    SaveWindowPicture(wm.info.win.window, folder / "screen.png", &width, &height)) {
		picture = "screen.png " + std::to_string(width) + "x" + std::to_string(height);
	}
#else
	(void)window;
#endif

	char title_id[64] {}, app_ver[64] {};
	(void)Loader::SystemContentParamSfoGetString("TITLE_ID", title_id, sizeof(title_id));
	(void)Loader::SystemContentParamSfoGetString("APP_VER", app_ver, sizeof(app_ver));
	float      xyz[3] {}, yaw = 0.0f;
	const bool placed = Loader::DemonsSoulsWarp::PlayerPlace(xyz, &yaw);
	char       place[96] {};
	if (placed) std::snprintf(place, sizeof(place), "%.2f %.2f %.2f yaw %.3f", xyz[0], xyz[1], xyz[2], yaw);

	std::ofstream info(folder / "info.txt");
	info << "time: " << stamp << "\n";
	info << "game: " << title_id << " " << app_ver << "\n";
	info << "guest frame (flips): " << Libs::Graphics::FrameGen::GuestFlips() << "\n";
	info << "player: " << (placed ? place : "not found (another build, or no character loaded)") << "\n";
	info << "picture: " << picture << "\n";
#if defined(_WIN32)
	info << "log: " << LogFile() << "\n";
	info << "switches:\n";
	for (const auto& entry: KytySwitches()) info << "  " << entry << "\n";
#endif
	info.close();

	std::printf("DEBUG SNAPSHOT %s (flip %llu, player %s)\n", folder.string().c_str(),
	            static_cast<unsigned long long>(Libs::Graphics::FrameGen::GuestFlips()), placed ? place : "unknown");
	std::fflush(stdout);
	return folder.string();
}

} // namespace DebugSnapshot
