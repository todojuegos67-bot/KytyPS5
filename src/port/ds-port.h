#ifndef EMULATOR_SRC_PORT_DS_PORT_H_
#define EMULATOR_SRC_PORT_DS_PORT_H_

// "Demon's Souls" PC port (-DKYTY_DS_PORT=ON): what its main (ds-main.cpp) hands the in-game options menu (ds-menu.cpp).

#include "port/ds-settings.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace Port {

struct Runtime {
	std::string              exe_dir;    // UTF-8, with a trailing separator
	std::string              ini_path;   // settings.ini
	std::string              image_path; // fondo-souls.jpg (the splash screen's and the menu's background)
	std::string              game_info;  // "Demon's Souls, PPSA01341 01.007.000"
	Settings                 launched;   // what this run started with
	std::vector<std::string> notes;      // what BuildLaunch adjusted
	PcInfo                   pc;         // TU EQUIPO
	std::filesystem::path    exe_path;   // "Demon's Souls.exe" (the update's elevated helper)
	std::filesystem::path    game_rel;   // the game's folder relative to the port's, when inside it (else empty)
	// (10-27b) The start menu's "Carpeta del juego": a folder with eboot.bin (or a .zar), and the Windows folder picker.
	std::function<bool(const std::string&)> is_game;
	std::function<std::string()>            pick_folder;
};

// Installs the port's hooks (PortLive::g_hooks): the menu (F1, or OPTIONS held + R3), the window icon and title, the
// splash screen closing at the first present.
void InstallMenu(const Runtime& runtime);
// The splash screen (ds-main.cpp): closes it.
void CloseSplash();
// Ends the game (as its window's close button) and starts this program again when it has ended.
void RequestRestart();
// Ends the game.
void RequestQuit();
// (10-27b) The launcher-style start menu, before the game (its own window, the F1 menu's pages full screen over
// fondo-souls.jpg, with Jugar / Actualizar version / Volver a la version anterior / Salir; the static shader precompile
// with its progress bar). `settings` gets what was chosen (saved to settings.ini). False: Salir.
bool RunStartMenu(const Runtime& runtime, Settings* settings);

} // namespace Port

#endif // EMULATOR_SRC_PORT_DS_PORT_H_
