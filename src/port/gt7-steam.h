#ifndef EMULATOR_SRC_PORT_GT7_STEAM_H_
#define EMULATOR_SRC_PORT_GT7_STEAM_H_

// "Gran Turismo 7" PC port (10-27c): "Anadir a Steam": a non-Steam shortcut to "Gran Turismo 7.exe" in every Steam user's
// userdata\<id>\config\shortcuts.vdf (appended; skipped where one to the same exe exists), its icon (the exe's, written
// next to it as "Gran Turismo 7.ico") and its artwork in config\grid (made from fondo-gt7.jpg).

#include <filesystem>
#include <string>

namespace Port::Steam {

// Spanish text of what was done (or why not); true when at least one Steam user has the shortcut now.
// (ds-pc-port37) The launcher's button: runs Anadir-a-Steam.cmd beside the exe (the script players use, which works) in
// its own window. False (report says why) when it is not there or does not start: then AddToSteam below.
bool RunAddToSteamScript(const std::filesystem::path& exe, std::string* report);
bool AddToSteam(const std::filesystem::path& exe, std::string* report);

} // namespace Port::Steam

#endif // EMULATOR_SRC_PORT_GT7_STEAM_H_
