#ifndef EMULATOR_SRC_PORT_GT7_PORT_H_
#define EMULATOR_SRC_PORT_GT7_PORT_H_

// "Gran Turismo 7" PC port (-DKYTY_GT7_PORT=ON, Windows): "Gran Turismo 7.exe", a launcher (SDL3 + Dear ImGui, ported
// from the Demon's Souls port's) that starts kyty_emulator.exe beside it with the chosen options, and shows the
// metrics window (FPS, RAM, VRAM) over the game while it runs. The emulator itself is not changed.

#include "port/gt7-settings.h"

#include <windows.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace Port {

struct PcInfo {
	std::string gpu;      // the graphics card's name
	uint64_t    vram = 0; // its dedicated memory (bytes)
	std::string cpu;
	bool        amd_cpu     = false;
	int         cpu_cores   = 0;
	int         cpu_threads = 0;
	int         ram_gb      = 0;
	int         screen_w = 0, screen_h = 0, refresh_hz = 0;
};

struct Runtime {
	std::filesystem::path exe_dir;  // the folder of "Gran Turismo 7.exe"
	std::filesystem::path exe_path; // "Gran Turismo 7.exe"
	std::string           ini_path; // settings.ini (UTF-8)
	PcInfo                pc;
};

enum class LauncherResult { Play, Quit };

// The launcher's window; `settings` is saved to settings.ini on every change and on JUGAR. `status`: a line shown
// under JUGAR (the last game's end, an error).
LauncherResult RunLauncher(const Runtime& runtime, Settings* settings, const std::string& status);
// While the game runs: the metrics window (when on; F10 shows or hides it). Returns when the process has ended.
void WatchGame(const Runtime& runtime, const Settings& settings, HANDLE process, DWORD pid);

// Shader precompile (gt7-menu.cpp; the emulator's side: pipelineSeeds.h). The game's programs and pipelines, as the
// game gave them (guest code and state, valid for any GPU), recorded in _ShaderCache\PPSA01317\guest.seeds, and the
// release's seeds-PPSA01317_<version>.seeds (beside the exe or in seeds\) are translated and compiled by
// "kyty_emulator.exe" with KYTY_PRECOMPILE=1 before the game starts, on all processors.
void MergeShippedSeeds(const Runtime& runtime);
// Whether a precompile is due: seeds there and never precompiled with this GPU, driver and emulator. `why`: the reason.
bool PrecompileDue(const Runtime& runtime, const Settings& settings, std::string* why);
// Forget the last precompile (Rendimiento: "Precompilar de nuevo"): the next JUGAR precompiles again.
void ForgetPrecompile(const Runtime& runtime);
// The emulator in precompile mode (the environment already set for the game) with a progress window and "Omitir".
// Returns a line for the launcher's status (empty when all went well).
std::string RunPrecompile(const Runtime& runtime, const Settings& settings, const std::wstring& command_line);

bool        IsGame(const std::string& path);
std::string PickFolder();
std::string PickFile(const wchar_t* filter_name, const wchar_t* filter);
std::string BuildCommandLine(const Runtime& runtime, const Settings& settings);
std::string Utf8(const std::wstring& w);
std::wstring Wide(const std::string& s);

} // namespace Port

#endif // EMULATOR_SRC_PORT_GT7_PORT_H_
