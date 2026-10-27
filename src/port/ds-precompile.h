#ifndef EMULATOR_SRC_PORT_DS_PRECOMPILE_H_
#define EMULATOR_SRC_PORT_DS_PRECOMPILE_H_

// "Demon's Souls" PC port (10-27b): the static shader and pipeline precompile of precompile-windows.ps1, run by the exe
// itself (the start menu starts it on a PC whose static cache is not this GPU's and driver's): kyty_shader_precompile
// shards on a worker thread, the specializations of recorded play first (released-/hinted-<id>.seeds, from the shipped
// hints and this PC's recordings), then every shader the game ships; the shards merged into
// _PipelineCache/static/<title>_<version>.bin(aries). Stopped, it resumes the next time (finished shards are kept).
// (10-27c) Instead: every pipeline recipe of the static inputs (_PipelineCache/static/<id>.shaders, the splash's step 3)
// compiled by 256 shards into the driver cache the game loads (_PipelineCache/local/<key>/<title>.bin), merged as they
// finish and when stopped (JUGAR); "current" once all 256 are in it for these inputs (<bin>.static-done).

#include <cstdint>
#include <filesystem>
#include <string>

namespace Port::Precompile {

struct Setup {
	std::filesystem::path root;  // the port's folder (the working directory of every run)
	std::string           game;  // UTF-8
	std::string           id;    // <title>_<version>
	std::filesystem::path seeds; // seeds-<id>.seeds (made before)
	uint64_t              affinity = 0;
	int                   cpus     = 1;
};

void Configure(const Setup& setup);
// "--driver-cache-status": the game's driver cache holds every static recipe of this GPU's and driver's inputs.
bool CacheCurrent();
void Start();
void Stop(); // waits for the worker (the running shards are killed; what finished is kept)

struct View {
	bool        running = false, finished = false, failed = false, playable = false; // playable: recorded play merged
	int         done = 0, total = 0;
	std::string text;
};
View State();

} // namespace Port::Precompile

#endif // EMULATOR_SRC_PORT_DS_PRECOMPILE_H_
