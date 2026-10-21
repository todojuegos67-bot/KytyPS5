#ifndef EMULATOR_SRC_PORT_DS_PORT_H_
#define EMULATOR_SRC_PORT_DS_PORT_H_

// "Demon's Souls" PC port (-DKYTY_DS_PORT=ON): what its main (ds-main.cpp) hands the in-game options menu (ds-menu.cpp).

#include "port/ds-settings.h"

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

} // namespace Port

#endif // EMULATOR_SRC_PORT_DS_PORT_H_
