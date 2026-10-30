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

namespace Port {
struct Settings;
}

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
// (ds-pc-port13) The CPU use of the precompile: all = every logical CPU thread of the PC (processes of 4 threads, as many
// as the CPUs allow, any CPU), else relaxed (the 10-27c bounds: at most 4 processes of 2 threads on the game's CPUs). The
// "waiting for free memory" gate (8 GB of RAM and of commit left for each new process) applies either way.
void SetAllThreads(bool all);
bool AllThreads();
// settings.ini's precompilar_hilos: "todos" / "relajado"; anything else (not chosen yet): the Yes/No question (a window
// of its own, in the menu's language), its answer stored into *settings and into `ini` (only that key). Then SetAllThreads.
void Choose(const std::string& ini, Settings* settings);
// "--driver-cache-status": the game's driver cache holds every static recipe of this GPU's and driver's inputs.
bool CacheCurrent();
void Start();
void Stop(); // waits for the worker (the running shards are killed; what finished is kept)

struct View {
	bool        running = false, finished = false, failed = false, playable = false; // playable: recorded play merged
	int         done = 0, total = 0;
	std::string text;
	// (ds-pc-port11) The counting text's parts, for the launcher's own (translated) progress panel: the time left in
	// seconds (-1: still estimating), the percent it resumed from (0: a fresh start), waiting for free memory.
	long long eta_s       = -1;
	int       resumed_pct = 0;
	bool      waiting_mem = false;
};
View State();

} // namespace Port::Precompile

#endif // EMULATOR_SRC_PORT_DS_PRECOMPILE_H_
