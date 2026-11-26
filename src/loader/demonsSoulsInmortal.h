#pragma once

namespace Loader {
struct Program;
// (inmortal1) Read-only diagnostic: logs which "no damage" flag stops a hit from lowering an enemy's HP
// (demonsSoulsInmortal.cpp). Always on, no behavior change.
namespace DemonsSoulsInmortal {
void Install(Program* program);
} // namespace DemonsSoulsInmortal
} // namespace Loader
