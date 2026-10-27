#ifndef EMULATOR_SRC_PORT_DS_SETTINGS_H_
#define EMULATOR_SRC_PORT_DS_SETTINGS_H_

// "Demon's Souls" PC port (-DKYTY_DS_PORT=ON): settings.ini next to the exe, and what it turns into before the
// emulator starts: the environment switches and command-line options that Launcher-souls.ps1 and run-windows.ps1
// passed (the same rules, in the same order; resources/LEEME-PC.txt lists each option).

#include <cstdint>
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
// (10-23) Calidad grafica (Launcher-souls.ps1's graphics group): the game's own settings (KYTY_GAME_CONVARS).
extern const Choice  kShadows[];
extern const int     kShadowCount;
extern const Choice  kDetails[];
extern const int     kDetailCount;
extern const Choice  kReflections[];  // also the object fade's levels
extern const int     kReflectionCount;
extern const Choice  kAaSharp[];
extern const int     kAaSharpCount;
// (10-27) Launcher-souls.ps1: Bordes sombras (KYTY_SHADOW_FILTER 0-3), Sombra rincones (the hardware RT AO's floor and
// strength), Texturas (normales / nitidas / extra nitidas), Anti-Lag.
extern const Choice  kShadowEdges[];
extern const int     kShadowEdgeCount;
extern const Choice  kCornerShadows[];
extern const int     kCornerShadowCount;
extern const Choice  kTextureSharpness[];
extern const int     kTextureSharpnessCount;
extern const Choice  kAntiLagAmd[];   // the AMD frame generation's, on an AMD graphics card
extern const int     kAntiLagAmdCount;
extern const Choice  kAntiLagOther[]; // the AMD frame generation's, on NVIDIA / Intel
extern const int     kAntiLagOtherCount;
extern const Choice  kAntiLagDlssg[]; // NVIDIA frame generation's
extern const int     kAntiLagDlssgCount;
extern const Choice  kSoulsModes[];
extern const int     kSoulsModeCount;

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
	bool        profile16 = false;  // run-windows -Ram16 (ram16.txt): memory only, and only on a PC of 20 GB of RAM or less
	bool        simulate_ram16 = false; // (10-24) "Simular 16 GB RAM": KYTY_SIMULATE_RAM_MB=16384 (and the 16 GB profile)
	bool        safe_gpu = false;   // KYTY_COMPUTE_CHAIN=0
	std::string quality = "standard";
	std::string texture_reduce = "0";
	bool        simulate_vram8 = false;
	bool        all_ccds = false;
	bool        antilag = false;     // (before 10-27) KYTY_ANTILAG: now antilag_mode
	std::string antilag_mode = "auto"; // AMD frame generation: auto (amd2 on AMD, normal else), amd2, normal, ultra, no
	std::string antilag_dlssg = "reflex"; // NVIDIA frame generation: reflex, normal, ultra (ours, Reflex off)
	// [Graficos] (10-23, the launcher's "Calidad grafica" and "Modos Souls")
	std::string souls_mode = "ninguno"; // ninguno, ligero (30 x2 AMD), equilibrado (45 x2 AMD), ultra, extremo, low (PC low profile)
	bool        gpu8 = false;        // Optimizacion tarjetas 8 GB (the menu sets the light preset)
	std::string shadows = "maximo";  // maximo, alto (190 m), medio (125 m), bajo (60 m)
	std::string detail = "alto";     // ultra, muy-alto, alto (the game's), normal, bajo
	std::string reflections = "alto"; // ultra, alto (the game's), normal, bajo
	std::string object_fade = "alto";
	std::string aa_sharp = "normal"; // ultra, alto, normal (the game's), bajo, no
	std::string rt_light = "0";      // (10-26: removed from the launcher) KYTY_SSGI, never passed now
	std::string ambient_occlusion = "0"; // (10-26: GTAO removed) never passed now
	// (10-27) Ray tracing (hardware): the launcher's six No / Si rows ($hwTbl). Every Souls mode turns them off.
	// (10-27b, the s25 launcher) Only Reflejos, Sombras sol and Luces antorcha/lamparas (rt_particles) are offered and passed;
	// rt_fog / rt_metal / rt_ao stay in settings.ini but are never passed.
	bool        rt_reflections = false, rt_sun = false, rt_fog = false, rt_particles = false, rt_metal = false, rt_ao = false;
	std::string godrays = "0";       // Luz volumetrica: 0 No, 1 Bajo, 2 Medio, 3 Alto (KYTY_GODRAYS + the hidden RT sun pass)
	bool        pure_game = false;   // Juego original: every visual pass the emulator adds off
	std::string shadow_edges = "0";  // Bordes sombras: 0 Normal, 1 Alta, 2 Muy alta, 3 Ultra (KYTY_SHADOW_FILTER)
	std::string corner_shadow = "0"; // Sombra rincones: 0 Normal, 1 Alta, 2 Maxima (with Oclusion RT only)
	bool        motion_blur = true, dof = true, vignette = true, bloom = true, eye_adapt = true;
	bool        af16 = false;        // KYTY_FORCE_AF=16
	// Texturas (10-25, the launcher's lodbias, default "nitidas"): 0 normales, 1 nitidas (KYTY_TEXTURE_LOD_BIAS=auto),
	// 2 extra nitidas (log2(ratio) - 1 with an upscaler, -0.5 without; at least -2)
	std::string texture_sharpness = "1";
	// [Mandos] (10-23: Launcher-souls.ps1 "Controles")
	bool        kbm = false;         // KYTY_KBM: keyboard and mouse
	float       kbm_sens = 1.0f;     // KYTY_KBM_SENS
	float       kbm_antidz = 0.15f;  // KYTY_KBM_ANTIDZ
	std::string kbm_map;             // KYTY_KBM_MAP
	std::string pad_map;             // KYTY_PAD_MAP
	// [Interfaz]
	std::string metrics = "0"; // KYTY_FPS_HUD 0/1/2
	bool        start_menu = true; // (10-27b) menu_inicio: the launcher-style menu before the game
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

// The PC (ds-main.cpp: the dedicated graphics card first, as Launcher-souls.ps1's Select-BestGpu).
struct PcInfo {
	std::string gpu;        // the graphics card's name
	uint64_t    vram = 0;   // its dedicated memory (bytes)
	std::string cpu;
	int         ram_gb = 0;
	int         screen_h = 0; // the monitor's real height (FSR 900p)
	bool        nvidia = true, amd = false;
	std::string recommended, expected; // the launcher's graphics card table ($gpuDb)
	bool        unsupported = false;
};
void RecommendFor(PcInfo* pc); // fills recommended / expected / unsupported from `gpu`
// (10-24) Launcher-souls.ps1 Test-LowRam: a PC of 20 GB of RAM or less, or "Simular 16 GB RAM".
bool LowRam(const Settings& settings, const PcInfo& pc);

// `title_id` from sce_sys/param.json (empty: unknown, the Demon's Souls engine settings still apply: this is its port).
LaunchPlan BuildLaunch(const Settings& settings, const std::string& title_id, const PcInfo& pc = {});

// Settings whose change needs a restart (everything the menu does not apply live).
bool NeedsRestart(const Settings& launched, const Settings& now);

// The launcher's presets (Set-PresetLight / Max / Ultra / Extreme and Souls equilibrado) on the graphics settings.
void ApplyPreset(Settings* settings, const std::string& preset); // ligero, maximo, equilibrado, ultra, extremo, low
// (10-26) Choosing any Souls mode: every ray tracing row to No.
void RtAllOff(Settings* settings);
// Souls ligero / equilibrado / PC low profile: frame generation fixed at AMD x2, the game at 30 / 45 / 30.
bool FixedFg(const Settings& settings);

} // namespace Port

#endif // EMULATOR_SRC_PORT_DS_SETTINGS_H_
