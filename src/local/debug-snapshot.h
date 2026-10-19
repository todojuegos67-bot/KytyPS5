#ifndef EMULATOR_SRC_LOCAL_DEBUG_SNAPSHOT_H_
#define EMULATOR_SRC_LOCAL_DEBUG_SNAPSHOT_H_

#include <string>

struct SDL_Window;

// F9: a picture that looks wrong, kept for a later look. The window's picture as shown (screen.png) and what was going
// on (info.txt: the time, the game's frame, the player's position, the run's log file and the KYTY_* switches) go to
// _DebugSnapshots/<date-time>/ next to the emulator, and the game pauses (F9 or Space resumes it). A line
// "DEBUG SNAPSHOT <folder>" in the log marks the moment.
namespace DebugSnapshot {

// The snapshot's folder (empty: it could not be made).
std::string Take(SDL_Window* window);

} // namespace DebugSnapshot

#endif // EMULATOR_SRC_LOCAL_DEBUG_SNAPSHOT_H_
