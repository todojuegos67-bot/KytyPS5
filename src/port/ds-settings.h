#ifndef EMULATOR_SRC_PORT_DS_SETTINGS_H_
#define EMULATOR_SRC_PORT_DS_SETTINGS_H_

// "Demon's Souls" PC port (-DKYTY_DS_PORT=ON): settings.ini next to the exe, and what it turns into before the
// emulator starts: the environment switches and command-line options that Launcher-souls.ps1 and run-windows.ps1
// passed (the same rules, in the same order; resources/LEEME-PC.txt lists each option).

#include <string>
#include <utility>
#include <vector>

namespace Port {

struct Choice {
	const char* id;    // the value in settings.ini
	const char* label; // the menu's text (Spanish)
};

// Super resolution (Launcher-souls.ps1 $resModes): upscaler and the ratio of the monitor's height, or the game's own
// scaling of a lower render height.
struct ResMode {
	const char* id;
	const char* label;
	double      value;    // ratio (upscalers) or render height (others)
	const char* upscaler; // "dlss", "xess", "fsr" or "" (the game's)
};
extern const ResMode kResModes[];
extern const int     kResModeCount;
extern const Choice  kSharpness[];
extern const int     kSharpnessCount;
extern const Choice  kDlssPresets[];
extern const int     kDlssPresetCount;
extern const Choice  kQualities[];
extern const int     kQualityCount;
extern const Choice  kTextureReduce[];
extern const int     kTextureReduceCount;
extern const Choice  kMetrics[];
extern const int     kMetricsCount;
extern const Choice  kLanguages[];
extern const int     kLanguageCount;

struct Settings {
	// [Juego]
	std::string game;             // the game's folder (eboot.bin) or .zar
	int         language = 3;     // console language (PS5 numbering; 3 = espanol (Espana), 1 = English (US))
	bool        backup_saves = true;
	// [Pantalla]
	bool        fullscreen = true;
	int         width = 2560, height = 1440; // the window's (fullscreen: the desktop's size is used)
	bool        hdr = false;
	bool        vsync = false;
	bool        gsync = false;
	bool        unlock_fps = false;
	bool        aspect_fit = false;
	std::string aspect = "16:9";
	float       fps_cap = 0.0f; // KYTY_FPS_CAP with G-Sync and unlocked fps without frame generation (0: auto)
	// [Reescalado]
	std::string res = "nativo";
	std::string sharpness = "auto";
	std::string dlss_preset = "auto";
	// [FrameGeneration]
	std::string framegen = "sin-fg";
	// [Rendimiento]
	bool        profile16 = false;  // run-windows -Ram16 (ram16.txt): memory only
	bool        low_end = false;    // "Modo graficas de 8 GB": AMD x2 + KYTY_FLIP_FPS=30
	bool        safe_gpu = false;   // KYTY_COMPUTE_CHAIN=0
	std::string quality = "standard";
	std::string texture_reduce = "auto";
	bool        simulate_vram8 = false;
	bool        all_ccds = false;
	// [Interfaz]
	std::string metrics = "0"; // KYTY_FPS_HUD 0/1/2
	bool        rumble = true;
	float       rumble_gain = 4.0f;
	// [Avanzado]
	std::string extra; // KEY=VALUE;KEY=VALUE (run-windows -Set; KEY= removes it)
};

bool        LoadSettings(const std::string& path, Settings* settings); // false: no file (the defaults stay)
bool        SaveSettings(const std::string& path, const Settings& settings);
const char* FrameGenLabel(const std::string& id);

using EnvList = std::vector<std::pair<std::string, std::string>>;

struct LaunchPlan {
	EnvList                  env;     // set before the emulator starts (empty value: removed)
	std::vector<std::string> args;    // the emulator's command line after argv[0]
	std::vector<std::string> notes;   // what was adjusted (logged)
	double                   ratio = 0.0;
	int                      vblank = 60;
};

// `title_id` from sce_sys/param.json (empty: unknown, the Demon's Souls engine settings still apply: this is its port).
LaunchPlan BuildLaunch(const Settings& settings, const std::string& title_id);

// Settings whose change needs a restart (everything the menu does not apply live).
bool NeedsRestart(const Settings& launched, const Settings& now);

} // namespace Port

#endif // EMULATOR_SRC_PORT_DS_SETTINGS_H_
