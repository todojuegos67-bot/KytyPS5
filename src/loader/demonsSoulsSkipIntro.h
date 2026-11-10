#pragma once

namespace Loader {
struct Program;
// KYTY_SKIP_INTRO=1: the game's own "skip splash screens" debug switch forced on (demonsSoulsSkipIntro.cpp).
namespace DemonsSoulsSkipIntro {
void Install(Program* program);
} // namespace DemonsSoulsSkipIntro
} // namespace Loader
