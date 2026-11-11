#pragma once

namespace Loader {
struct Program;
// KYTY_ENGINE_POOLS=1 (diagnostics): Demon's Souls' memory pools and the bytes each holds, printed every 60 s
// (demonsSoulsEnginePools.cpp).
namespace DemonsSoulsEnginePools {
void Install(Program* program);
} // namespace DemonsSoulsEnginePools
} // namespace Loader
