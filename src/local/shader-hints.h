#pragma once
// kyty_shader_precompile --make-hints and --apply-hints (shader-hints.cpp): the shaders as recorded play specialized
// them (warmup recordings, PipelineCache's), without the game's code, which a release ships for each game version
// (tools/local/static-precompile/hints-<title>_<version>.hints). A recorded seed file holds the game's code, so only
// the PC that played has one; the hints are completed on every other PC with the code of its own seed file (made
// from its game files by --make-seeds), into the recorded seed file precompile-windows.ps1 compiles with the seeds.
//
// A hint file: the identity line "KytyShaderHints1:<title>_<version>\n", the XXH3-64 of the body, the body
// compressed (one zstd frame). The body is a seed file's (each list: its count, then each entry's word count and
// words): records as the warmup schema writes them (shader-warmup-cache.h Visit), but with the code and the back
// code each [word count, XXH3-64 low, high] and no static key (BuildStageStaticKey makes it from the input info),
// then the pipeline recipes, whose shaders are indices of those records.
#include <filesystem>
#include <string>
#include <vector>

namespace ShaderHints {

// <title>_<version> of the game (the names of its caches) from sce_sys/param.json, or empty.
std::string GameId(const std::filesystem::path& game);

// --make-hints: the hint file of the game version game_id from recordings (warmup files or recorded seed files):
// their records whose code the seed file has (what --make-seeds lists on every PC; a recording can hold another
// version's play too) and the pipeline recipes of those. 0, or 1 (nothing is written).
int Make(const std::filesystem::path& out, const std::filesystem::path& seeds,
         const std::vector<std::filesystem::path>& recordings, const std::string& game_id);

// --apply-hints: the recorded seed file of a hint file of the game version game_id, each record with the code of the
// seed file's program of that word count and hash (a record without one is left out, with the pipelines that use
// it). 0, or 1 (nothing is written).
int Apply(const std::filesystem::path& hints, const std::filesystem::path& seeds, const std::filesystem::path& out,
          const std::string& game_id);

} // namespace ShaderHints
