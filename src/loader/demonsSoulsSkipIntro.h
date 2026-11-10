#pragma once

namespace Loader {
struct Program;
// KYTY_SKIP_INTRO=1: the game's own "skip splash screens" debug switch forced on (demonsSoulsSkipIntro.cpp).
namespace DemonsSoulsSkipIntro {
void Install(Program* program);
// The one-shot opening-cinematic skip: still in place?
bool CinematicSkipArmed();
// Puts the hold-to-skip code back as the game has it (later cinematics play). Safe to call more than once.
void Disarm(const char* reason);
} // namespace DemonsSoulsSkipIntro
} // namespace Loader
