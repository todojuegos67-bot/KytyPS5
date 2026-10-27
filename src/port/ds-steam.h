#ifndef EMULATOR_SRC_PORT_DS_STEAM_H_
#define EMULATOR_SRC_PORT_DS_STEAM_H_

// "Demon's Souls" PC port (10-27c): "Anadir a Steam": a non-Steam shortcut to "Demon's Souls.exe" in every Steam user's
// userdata\<id>\config\shortcuts.vdf (appended; skipped where one to the same exe exists), its icon (the exe's, written
// next to it as "Demon's Souls.ico") and its artwork in config\grid (made from fondo-souls.jpg).

#include <filesystem>
#include <string>

namespace Port::Steam {

// Spanish text of what was done (or why not); true when at least one Steam user has the shortcut now.
bool AddToSteam(const std::filesystem::path& exe, std::string* report);

} // namespace Port::Steam

#endif // EMULATOR_SRC_PORT_DS_STEAM_H_
