#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

// Engine console variables for the game's own command-line file: KYTY_GAME_CONVARS="name=value;name=value"
// gives a read-only open of the PackageCmdLineArgs.txt the game ships a copy with a "+name=value" line per
// entry added (Demon's Souls passes every line of that file to its console at boot), so engine settings
// change without touching the game's files. With the game's super resolution path on
// (loader/demonsSoulsSuperResolution.h), a read of RenderConfig_PS5_Perf.txt gets a copy whose r_Wants4K line asks
// for display mode 5 (WantsTrinity) instead of 3 (WantsHigh).
namespace Loader {
struct Program;
}

namespace Loader::GameArgs {

// The host file a read-only open of `real` should open instead (empty: `real` itself).
std::filesystem::path RedirectRead(const std::filesystem::path& real);
// A stat of `real`: the size of the copy a read would get (nothing: its own).
std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real);

// KYTY_GAME_CONVARS_DUMP=1 (or 2): once the eboot is loaded, the console-variable-like names its image holds
// (identifiers that start r_ or name a rendering/memory element: shadow, texture, stream, lod, fog, volumetric, ssr,
// ao, decal, particle, fur, hair, cloth, probe, reflection, resolution, budget, pool...; 2: every r_/g_/s_ name and
// camelCase identifier of 8+ characters) to the log and to kyty-convar-names.txt in the temporary folder, with the
// KYTY_GAME_CONVARS overrides in effect. Names only: the game registers convars at run time through constructors whose
// default values are not in a string table, so defaults are not printed (a registration hook would be per-build).
void DumpConvarNames(const Program* program);

} // namespace Loader::GameArgs
