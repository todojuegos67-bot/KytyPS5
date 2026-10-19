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
namespace Loader::GameArgs {

// The host file a read-only open of `real` should open instead (empty: `real` itself).
std::filesystem::path RedirectRead(const std::filesystem::path& real);
// A stat of `real`: the size of the copy a read would get (nothing: its own).
std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real);

} // namespace Loader::GameArgs
