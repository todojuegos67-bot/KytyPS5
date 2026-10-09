#pragma once
// kyty_shader_precompile --make-seeds (static-seeds.cpp): the static precompile's seed file, made from the
// game's files. It is tools/local/static-precompile/precompile.py seeds without Python: the same file, byte for
// byte.
#include <cstdint>
#include <filesystem>
#include <string>

namespace StaticSeeds {

struct Options {
	std::filesystem::path game;            // the folder with eboot.bin
	std::filesystem::path out;             // the seed file to write
	std::filesystem::path states;          // pass-states.json; empty: FindStates()
	std::string           stages = "cs,gfx"; // precompile.py seeds --stages: cs, gfx or both
	int64_t               limit  = 0;        // precompile.py seeds --limit: the first N programs only (a smoke test)
};

// pass-states.json where the repository and a release package keep it (tools/local/static-precompile below the
// program's folder, two folders above it in a build tree, or below the working directory), or empty.
std::filesystem::path FindStates();

// Writes the seed file: 0, or 1 where precompile.py stops with an error (nothing is written then).
int Make(const Options& options);

} // namespace StaticSeeds
