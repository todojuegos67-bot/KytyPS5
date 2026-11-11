#ifndef EMULATOR_SRC_PORT_GT7_SETTINGS_H_
#define EMULATOR_SRC_PORT_GT7_SETTINGS_H_

// "Gran Turismo 7" PC port (-DKYTY_GT7_PORT=ON): its settings (settings.ini next to "Gran Turismo 7.exe") and what
// they turn into on kyty_emulator's command line. Ported from the Demon's Souls port's launcher (ds-settings), with
// only what this emulator (usa-ratt/KytyPS5-GT7) really understands.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Port {

struct Choice {
	const char* id;    // the value in settings.ini
	const char* label; // the menu's text (Spanish; I18n::T translates it)
};

struct Settings {
	// Juego
	std::string game = "E:\\gran turismo 7\\PPSA01317-app"; // carpeta_juego: the dumped game (eboot.bin)
	std::string game_patch;                                  // parche_juego: ETAHen cheat file (.json)
	int         console_language = 3;                        // idioma_juego: PS5 console language (3 = espanol)
	std::string menu_language;                               // idioma_menu: es, en (empty: Windows' language)
	std::string user_name = "Jugador";                       // nombre_usuario (1-16 bytes)
	bool        close_launcher = false;                      // cerrar_lanzador: no launcher after the game
	// Graficos
	bool        fullscreen   = true;        // pantalla_completa (--fullscreen, borderless)
	std::string resolution   = "1920x1080"; // resolucion: escritorio or WxH (--screen-width/--screen-height)
	std::string present      = "mailbox";   // sincronizacion: mailbox, fifo (VSync), immediate
	int         vblank       = 60;          // vblank: --vblank-frequency
	bool        tessellation = false;       // teselado (--tessellation)
	int         reflections_every = 0;      // reflejos_cada: --reflections-every (0: the emulator's own)
	bool        hide_cursor  = true;        // ocultar_cursor (--hide-cursor)
	bool        bindless     = false;       // bindless (--bindless)
	// Reescalado / generacion de fotogramas: kept for later (this base has no DLSS/FSR/XeSS yet: never passed).
	std::string upscaler = "no"; // reescalado
	std::string framegen = "no"; // generacion_fotogramas
	// Rendimiento
	int         gpu        = -1;       // gpu: Vulkan device index (-1: auto)
	std::string amd_cpu    = "auto";   // parches_cpu_amd: auto, si, no (--amd-cpu)
	std::string priority   = "alta";   // prioridad: normal, alta
	bool        x3d_ccd    = false;    // nucleos_x3d: only the CCD with the largest L3 (Ryzen X3D)
	bool        redzone    = true;     // red_zone (--redzone / --no-redzone)
	bool        playgo_hack = false;   // playgo_hack
	bool        cuda_early = false;    // smooth_motion: KYTY_CUDA_EARLY_INIT=1 (NVIDIA Smooth Motion)
	// Mando
	std::string pad_color;             // color_mando: #RRGGBB (empty: the game's)
	int         pad_volume    = 50;    // volumen_mando
	int         pad_vibration = 100;   // vibracion_mando
	// Volante (wheel.json is the emulator's own file: these few values are written into it)
	bool        wheel            = false; // volante (--wheel true)
	int         wheel_ffb_gain   = 50;    // volante_ffb (0-100)
	int         wheel_rotation   = 900;   // volante_grados
	bool        wheel_soft_lock  = true;  // volante_tope_suave
	bool        wheel_reverse_ffb = false; // volante_ffb_invertido
	bool        wheel_soft_cond  = false; // volante_condiciones_software
	bool        wheel_background = false; // volante_segundo_plano
	// Teclado y raton
	bool        kbm        = true;  // teclado_raton: --keymap entries below
	float       mouse_sens = 1.0f;  // sensibilidad_raton (MouseSensitivity=)
	std::string keymap;             // mapa_teclado: Control=Input;Control=Input (empty: the emulator's defaults)
	// Metricas
	bool        metrics      = false; // metricas: the always-on-top window while playing (F10)
	bool        metrics_fps  = true;
	bool        metrics_ram  = true;
	bool        metrics_vram = true;
	bool        metrics_graph = true;
	std::string metrics_pos  = "arriba-izquierda";
	// Desarrollador
	bool        dev_vulkan_validation = false;
	bool        dev_shader_log        = false; // --shader-log-direction File
	bool        dev_printf_file       = false; // --printf-direction File
	bool        dev_renderdoc         = false; // --rd
	bool        dev_profiler          = false; // --profile
	bool        dev_console           = false; // the emulator's console window (else its output goes to logs\)
	std::string dev_args;                      // argumentos_extra: more command line (spaces)
	std::string dev_env;                       // entorno_extra: KEY=VALUE;KEY=VALUE
	bool        configured = false;
};

bool LoadSettings(const std::string& path, Settings* settings); // false: no file (the defaults stay)
bool SaveSettings(const std::string& path, const Settings& settings);

using EnvList = std::vector<std::pair<std::string, std::string>>;

struct LaunchPlan {
	std::vector<std::string> args; // after argv[0]
	EnvList                  env;
};

LaunchPlan BuildLaunch(const Settings& s, bool amd_cpu_detected, int desktop_w, int desktop_h);

// The wheel's values into wheel.json (the rest of the file, its bindings made with launcher.exe, is kept).
bool WriteWheelJson(const std::string& path, const Settings& s, std::string* error);
void ReadWheelJson(const std::string& path, Settings* s);

extern const Choice kResolutions[];
extern const int    kResolutionCount;
extern const Choice kPresentModes[];
extern const int    kPresentModeCount;
extern const Choice kUpscalers[];
extern const int    kUpscalerCount;
extern const Choice kFrameGens[];
extern const int    kFrameGenCount;
extern const Choice kAmdCpu[];
extern const int    kAmdCpuCount;
extern const Choice kPriorities[];
extern const int    kPriorityCount;
extern const Choice kMetricsPos[];
extern const int    kMetricsPosCount;
struct Language {
	int         id;
	const char* label;
};
extern const Language kConsoleLanguages[];
extern const int      kConsoleLanguageCount;

} // namespace Port

#endif // EMULATOR_SRC_PORT_GT7_SETTINGS_H_
