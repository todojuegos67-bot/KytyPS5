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
	const char* upscaler; // "dlss", "xess", "fsr" (FSR 3.1), "fsr4" (AMD FSR 4) or "" (the game's)
	const char* fsr4_preset = nullptr; // (10-28) FSR 4's model (KYTY_FSR4_PRESET): native, quality, balanced, performance, ultraperf, drs
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
extern const Choice  kIbl[];         // (10-29) Iluminacion IBL: Mitad, Completa
extern const int     kIblCount;
extern const Choice  kTess[];        // (10-29) Teselado: Bajo, Medio, Alto, Ultra, Extremo
extern const int     kTessCount;
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
// (10-28, ds-torch-1028's launcher) The screen-space RT set (c3e58bd5) and the new passes: their rows' levels.
extern const Choice  kRtReflLevels[];  // Reflejos RT: No, Bajo, Medio, Alto, Ultra, Extremo (KYTY_RT_REFLECTIONS 0-5), Hibrido (6: Alto + KYTY_RT_REFLECTIONS_HYBRID)
extern const int     kRtReflLevelCount;
extern const Choice  kRtLightLevels[]; // Luz RT (SSGI): No, Media, Alta, Ultra, Extremo
extern const int     kRtLightLevelCount;
extern const Choice  kSmokeLevels[]; // No, Suave, Antorchas (Humo y particulas)
extern const int     kSmokeLevelCount;
extern const Choice  kRtLevels[];      // No, Bajo, Medio, Alto (Luz volumetrica, Oclusion ambiental, Sombras RT)
extern const int     kRtLevelCount;
extern const Choice  kTorchLevels[];   // Luz de antorchas RT: No, Bajo, Medio, Alto, Hibrido (ds-pc-port10: no longer offered)
extern const int     kTorchLevelCount;
// (ds-pc-port10) DLSS 5 (NVIDIA's Neural Rendering, ds-foveated-1029's foveated grid): No, Suave, Medio, Fuerte
// (KYTY_FOVEATED=1, KYTY_FOV_EDGE 0.75 / 0.5 / 0.35), and its file: Oficial (signed by NVIDIA) / No oficial.
extern const Choice  kDlss5Levels[];
extern const int     kDlss5LevelCount;
extern const Choice  kDlss5Files[];
extern const int     kDlss5FileCount;
extern const Choice  kDlss5Strengths[]; // Intensidad DLSS 5: Baja, Media, Alta (KYTY_DLSS_NR_STRENGTH 0.35 / 0.65 / 1)
extern const int     kDlss5StrengthCount;
extern const Choice  kYesNo[];
extern const int     kYesNoCount;
extern const Choice  kSoulsModes[];
extern const int     kSoulsModeCount;

struct Settings {
	// [Juego]
	std::string game;             // the game's folder (eboot.bin) or .zar
	int         language = 3;     // idioma_juego: console language (PS5 numbering; 3 = espanol (Espana), 1 = English (US))
	bool        game_language_set = false; // (idioma_juego / idioma in settings.ini)
	std::string menu_language;    // (10-27b) idioma_menu: es, en, pt, fr, it, de (empty: the Windows UI language at start)
	bool        backup_saves = true;
	bool        continue_f1 = true; // (ds-pc-port8) continuar_f1: JUGAR after F1 (save and quit) continues by itself where F1 left
	// [Pantalla]
	bool        fullscreen = true;
	int         width = 2560, height = 1440; // the window's (fullscreen: the desktop's size is used)
	// (10-27c) resolucion: the full screen's output resolution, "escritorio" (the desktop's) or WxH (the monitor switched
	// to that mode while the game runs: KYTY_DISPLAY_MODE).
	std::string resolution = "escritorio";
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
	// (ds-pc-port10) dlss5: DLSS 5 0 No, 1 Suave, 2 Medio, 3 Fuerte (NVIDIA RTX and the user's own nvngx_dlssnr.dll beside
	// "Demon's Souls.exe"); dlss5_no_oficial: 1 loads an nvngx_dlssnr.dll not signed by NVIDIA (KYTY_DLSS_NR_UNSIGNED=1).
	std::string dlss5 = "0";
	bool        dlss5_unsigned = false;
	std::string dlss5_strength = "alta"; // dlss5_intensidad: baja, media, alta (KYTY_DLSS_NR_STRENGTH 0.35 / 0.65 / 1)
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
	std::string ibl = "";            // iluminacion_ibl: mitad (HalfResX), completa (Full); empty: by the output (4K completa, else mitad)
	std::string tess = "";           // teselado: bajo, medio, alto, ultra, extremo; empty: by the output (4K alto, else medio)
	std::string reflections = "alto"; // ultra, alto (the game's), normal, bajo
	std::string object_fade = "alto";
	std::string aa_sharp = "normal"; // ultra, alto, normal (the game's), bajo, no
	std::string rt_light = "0";      // (10-26: removed from the launcher) KYTY_SSGI, never passed now
	std::string ambient_occlusion = "0"; // (10-26: GTAO removed) never passed now
	// (10-27) Ray tracing (hardware): the launcher's six No / Si rows ($hwTbl). Every Souls mode turns them off.
	// (10-27b, the s25 launcher) Only Reflejos, Sombras sol and Luces antorcha/lamparas (rt_particles) are offered and passed;
	// rt_fog / rt_metal / rt_ao stay in settings.ini but are never passed.
	bool        rt_reflections = false, rt_sun = false, rt_fog = false, rt_particles = false, rt_metal = false, rt_ao = false;
	// (10-28) Antorchas (the launcher's hwlamp4): 0 No, 1 Bajo, 2 Medio, 3 Alto (KYTY_HWRT_LIGHTS level, with the RT particles).
	std::string rt_lamps = "";       // empty: from rt_particles (an older settings.ini: Si = 2)
	std::string godrays = "0";       // Luz volumetrica: 0 No, 1 Bajo, 2 Medio, 3 Alto (KYTY_GODRAYS + the hidden RT sun pass)
	bool        pure_game = false;   // Juego original: every visual pass the emulator adds off
	// (10-28, ds-torch-1028) The screen-space RT set and the new passes (none needs hardware RT but Hibrido):
	std::string rt_ss_refl = "0";    // reflejos_rt: Reflejos RT 0-6 (KYTY_RT_REFLECTIONS; 4-5: more presence; 6 Hibrido)
	std::string rt_ssgi = "0";       // luz_rt_pantalla: Luz RT (SSGI) 0 No, 1 Media, 2 Alta, 3 Ultra, 4 Extremo (KYTY_SSGI)
	std::string rt_shadows = "0";    // sombras_rt: Sombras RT de pantalla 0-3 (KYTY_RT_SHADOWS, strength 0.35 / 0.5 / 0.7)
	std::string ao = "0";            // oclusion_ao: Path tracing 0-3 (KYTY_AO + KYTY_AO_HW, ds-ptao-1029); default No
	std::string smoke = "1";         // humo_particulas: Humo y particulas 0 No, 1 Suave (KYTY_SMOKE, ds-particles-1029; ds-pc-port10: no Antorchas level); default Suave
	std::string torch = "0";         // luz_antorchas: Luz de antorchas RT 0-3, 4 Hibrido (KYTY_TORCH, KYTY_TORCH_HW)
	bool        lens_flare = false;  // destellos_lente: KYTY_TORCH_FLARE
	bool        rt_hw_exp = false;   // rt_experimental: hardware RT (the launcher's rthwexp; KYTY_RT_HW_EXPERIMENTAL=1)
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
	int         kbm_camera = 10;     // (10-28) cp11_Setting_CameraSpeed with keyboard and mouse (5-15; the launcher's kbmcam2)
	bool        kbm_warp = false;    // (10-28) Camara instantanea (frame warp): KYTY_KBM_WARP
	bool        kbm_calibrate = false; // (10-28) the next start measures the camera into kbm_curve.txt (KYTY_KBM_CALIBRATE), once
	// [Configurador] (10-28)
	bool        configured = false;  // the first-start dialog has been shown
	bool        ram16_boost = true;  // Mejoras para equipos de 16 GB de RAM: the 16 GB profile + the RAM flush (PCs of 20 GB or less)
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
	int         screen_w = 0, refresh_hz = 0; // (10-28) the monitor's width and refresh rate
	int         cpu_cores = 0, cpu_threads = 0; // (10-28)
	bool        nvidia = true, amd = false;
	std::string recommended, expected; // the launcher's graphics card table ($gpuDb)
	bool        unsupported = false;
};
void RecommendFor(PcInfo* pc); // fills recommended / expected / unsupported from `gpu`
// (10-24) Launcher-souls.ps1 Test-LowRam: a PC of 20 GB of RAM or less, or "Simular 16 GB RAM".
bool LowRam(const Settings& settings, const PcInfo& pc);

// `title_id` from sce_sys/param.json (empty: unknown, the Demon's Souls engine settings still apply: this is its port).
LaunchPlan BuildLaunch(const Settings& settings, const std::string& title_id, const PcInfo& pc = {});

// (10-28) The Configurador: the best settings for this PC from the hardware-tier table (ds-settings.cpp, calibrated
// on an RTX 5080 + Ryzen 9 9950X3D at 4K). Writes only graphics, ray tracing and the RAM option into `settings`;
// returns the explanation lines ("Hemos elegido: ...").
std::vector<std::string> Optimize(const PcInfo& pc, Settings* settings);
// (10-29) The output height (full screen mode / desktop, or the window), and the IBL / tessellation defaults for it
// (4K: completa + alto, else mitad + medio). ResolveQualityAuto fills an empty ibl / tess with them.
int  OutputHeight(const Settings& settings, const PcInfo& pc);
void SetQualityForOutput(Settings* settings, int out_h);
void ResolveQualityAuto(Settings* settings, const PcInfo& pc);
// Ray tracing rows (Reflejos, Antorchas, Luz volumetrica, and since 10-28 Oclusion ambiental (8), Luz de antorchas RT (16),
// Destellos de lente (32), Sombras RT (64)) turned off in the menu are applied live (PortLive::rt_off);
// turning them on, or anything else, needs a restart.
int  LiveRtOff(const Settings& launched, const Settings& now); // the PortLive::rt_off mask
// (The Antorchas level: rt_lamps, or rt_particles of an older settings.ini.)
std::string LampLevel(const Settings& settings);

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
