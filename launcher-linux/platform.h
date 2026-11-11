#ifndef KYTY_LAUNCHER_LINUX_PLATFORM_H
#define KYTY_LAUNCHER_LINUX_PLATFORM_H

#include "settings.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct GpuInfo {
	std::string name;
	uint64_t    memory   = 0; // bytes of its largest device-local heap
	uint32_t    vendor   = 0; // PCI vendor id (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel)
	bool        discrete = false;
};

// A terminal program: how it runs a command in a new window.
struct Terminal {
	std::string              name;   // the program
	std::vector<std::string> before; // its arguments before the command
};

// What the launcher knows about this PC and the emulator's folder, found once at start
// (launcher-windows' Platform.cs).
class LauncherContext {
public:
	LauncherContext();

	fs::path root;       // the folder of run_ds.sh (a package), or the repository's root (a build)
	fs::path run_script; // run_ds.sh
	fs::path emulator;   // kyty_emulator (KYTY_BIN when it is set)
	fs::path precompile_tool;   // kyty_shader_precompile, or empty
	fs::path precompile_script; // precompile-linux.sh, or empty
	fs::path dlss_script;       // get-dlss-linux.sh, or empty
	std::unique_ptr<Settings> settings;

	std::vector<GpuInfo> gpus; // the Vulkan devices, dedicated ones first, then by video memory
	bool                 rtx            = false;
	bool                 amd            = false;
	uint64_t             ram            = 0;
	int                  display_height = 0; // the primary display's, or 0 (set by the window)
	std::string
	    efficiency_cpus; // a hybrid CPU's efficiency cores as a taskset list ("16-23"), or empty
	std::vector<Terminal> terminals; // the terminal programs found, preferred first

	// libkyty_dlss.so where the emulator looks for it (KYTY_DLSS_DIR, next to it, its dlss folder),
	// or empty.
	[[nodiscard]] fs::path DlssLibrary() const;
	[[nodiscard]] bool     DlssAvailable() const { return rtx && !DlssLibrary().empty(); }
	// The terminal chosen on the Advanced page, else the first found; nullptr without one.
	[[nodiscard]] const Terminal* ChosenTerminal() const;
	[[nodiscard]] fs::path        LogsFolder() const;
	[[nodiscard]] fs::path        Readme() const;
	[[nodiscard]] bool            HasMangoHud() const { return mangohud_; }

private:
	bool mangohud_ = false;
};

namespace Platform {

fs::path ExecutableFolder();
// Whether `name` is a program on PATH (or an executable path).
bool HasProgram(const std::string& name);
// A shell word: 'text' with its single quotes escaped.
std::string Quote(const std::string& text);
// Runs argv (no shell) with stdout captured into `output` (when not null); the exit code, -1 when
// it could not run or ran past `timeout_ms` (then killed).
int Run(const std::vector<std::string>& argv, std::string* output, int timeout_ms,
        const fs::path& directory = {});
// Starts argv detached (its own session; this process does not wait for it); false when it could
// not start.
bool Spawn(const std::vector<std::string>& argv, const fs::path& directory = {});
// A folder, file or link in the desktop's own program (xdg-open, else gio open).
void Open(const std::string& target);
// The processors this process may use.
int ProcessorCount();
// A menu entry for this program ($XDG_DATA_HOME/applications/kytyps5-launcher.desktop): its path,
// or empty with the reason in `error`.
fs::path WriteDesktopEntry(const fs::path& folder, const fs::path& icon, std::string* error);

} // namespace Platform

#endif
