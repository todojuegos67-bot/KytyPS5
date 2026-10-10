// The static shader and pipeline precompile, a program of its own (tools/local/static-precompile):
//
//   kyty_shader_precompile --game <dir> --seeds <file> [--shard <i>/<n>] [--threads <n>]
//                          [--out <file> | --static-inputs] [--timings <file>] [--no-pipelines]
//   kyty_shader_precompile --game <dir> --merge [--prune]
//   kyty_shader_precompile --game <dir> --seeds <file> --status
//   kyty_shader_precompile --game <dir> --make-seeds <file> [--states <pass-states.json>]
//                          [--stages cs,gfx] [--limit <n>]
//   kyty_shader_precompile --game <dir> --param <file>
//   kyty_shader_precompile --game <dir> --seeds <file> --make-hints <file> --recorded <file> [--recorded <file>...]
//   kyty_shader_precompile --game <dir> --seeds <file> --apply-hints <file> --out <file>
//
// The game <dir> can also be the game packed into a ZArchive (a .zar file, read without extracting it); --param
// copies its sce_sys/param.json to <file> (the scripts read the title and version of a .zar so).
// compiles the shaders and pipelines of a seed file on a headless Vulkan device (the one the emulator
// creates, without a window) into the static pipeline cache _PipelineCache/static/<title>_<version>.bin, which
// the emulator looks up before compiling. With --shard, the i-th of n shares goes into a cache file of
// its own next to it: the NVIDIA driver compiles big shaders nearly one at a time per process, so the
// shares are processes (precompile-windows.ps1 runs them); --merge folds their files into the static
// cache. Where the driver has VK_KHR_pipeline_binary the static cache is the pipelines' binaries instead
// (_PipelineCache/static/<title>_<version>.binaries, read by the emulator when it needs a pipeline): every share
// writes its pipelines' binaries to a shard file, and --merge --prune makes the store of the shards' alone
// (what no seed makes any more is dropped; without --prune the store keeps its pipelines too); exit code
// 3: binaries were left out (the driver began compressing with its own dictionary), to be made by two
// shards (2n) instead. --out
// writes what was compiled as a warmup file, for precompile.py coverage; --static-inputs
// writes it where the emulator's shader prefetch reads it (_PipelineCache/static/<title>_<version>.shaders);
// --timings each pipeline's compile time (ms, SPIR-V words, the seeds' hashes). --status prints
// whether those inputs and the static cache are this GPU's and driver's ("inputs current|stale",
// "static cache current|outdated|stale", outdated: older than the seeds; run-windows.ps1 asks before every launch).
// Run it from the directory the emulator runs in.
// --make-seeds writes the seed file itself, from the game's files (static-seeds.cpp: every shader the game
// ships, paired as its materials and the engine draw them, with the render-pass states of pass-states.json,
// which it finds in tools/local/static-precompile by the program or the working directory): what
// tools/local/static-precompile/precompile.py seeds writes, byte for byte, without Python and without a
// Vulkan device.
// --make-hints writes the specialization hints of warmup recordings (or recorded seed files) of the game version, which
// a release ships: their records of the programs in the seed file without the game's code (shader-hints.h);
// --apply-hints completes a hint file with the code of the seed file into a recorded seed file (--out), the
// specializations of recorded play for a PC that has no recording (precompile-windows.ps1 compiles both).
#include "common/archive.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "loader/systemContent.h"
#include "shader-hints.h"
#include "static-seeds.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <thread>

using Libs::Graphics::PipelineCache;

static int Usage() {
	std::fprintf(stderr, "usage: kyty_shader_precompile --game <dir> --seeds <file> [--shard <i>/<n>] "
	                     "[--threads <n>] [--out <file> | --static-inputs] [--timings <file>] [--no-pipelines]\n"
	                     "       kyty_shader_precompile --game <dir> --merge [--prune]\n"
	                     "       kyty_shader_precompile --game <dir> --seeds <file> --status\n"
	                     "       kyty_shader_precompile --game <dir> --make-seeds <file> [--states <pass-states.json>] "
	                     "[--stages cs,gfx] [--limit <n>]\n"
	                     "       kyty_shader_precompile --game <dir> --param <file>\n"
	                     "       kyty_shader_precompile --game <dir> --seeds <file> --make-hints <file> --recorded <file> "
	                     "[--recorded <file>...]\n"
	                     "       kyty_shader_precompile --game <dir> --seeds <file> --apply-hints <file> --out <file>\n"
	                     "(<dir>: the game folder, or the game packed into a ZArchive: a .zar file)\n");
	return 2;
}

int main(int argc, char* argv[]) {
	std::filesystem::path          game;
	bool                           merge  = false;
	bool                           prune  = false;
	bool                           status = false;
	PipelineCache::PrecompileOptions options;
	StaticSeeds::Options             make_seeds;          // --make-seeds and its options
	bool                             seed_option = false; // --states, --stages or --limit
	bool                             amd         = false; // --amd
	bool                             validate    = false; // --validate: SPIR-V validation of every module
	std::filesystem::path            param_copy;          // --param
	std::filesystem::path              make_hints, apply_hints; // --make-hints, --apply-hints
	std::vector<std::filesystem::path> recordings;              // --recorded (--make-hints)
	options.threads = std::max(1u, std::thread::hardware_concurrency());
	for (int i = 1; i < argc; i++) {
		const std::string_view arg   = argv[i];
		const char*            value = i + 1 < argc ? argv[i + 1] : nullptr;
		if (arg == "--merge") {
			merge = true;
		} else if (arg == "--prune") {
			prune = true;
		} else if (arg == "--status") {
			status = true;
		} else if (arg == "--no-pipelines") {
			options.pipelines = false;
		} else if (arg == "--amd") {
			// What a GPU running wave64 compute natively with 4-byte robust storage alignment (AMD RDNA) gets:
			// 64-lane host subgroups and shader-side storage buffer bounds (with --no-pipelines on another GPU).
			options.host_subgroup_size = 64;
			amd = true;
		} else if (arg == "--validate") {
			validate = true;
		} else if (arg == "--static-inputs") {
			options.static_inputs = true;
		} else if (value == nullptr) {
			return Usage();
		} else if (arg == "--game") {
			game = argv[++i];
		} else if (arg == "--seeds") {
			options.seeds = argv[++i];
		} else if (arg == "--out") {
			options.out = argv[++i];
		} else if (arg == "--timings") {
			options.timings = argv[++i];
		} else if (arg == "--threads") {
			options.threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
		} else if (arg == "--shard") {
			if (std::sscanf(argv[++i], "%u/%u", &options.shard, &options.shards) != 2) return Usage();
		} else if (arg == "--make-seeds") {
			make_seeds.out = argv[++i];
		} else if (arg == "--param") {
			param_copy = argv[++i];
		} else if (arg == "--make-hints") {
			make_hints = argv[++i];
		} else if (arg == "--apply-hints") {
			apply_hints = argv[++i];
		} else if (arg == "--recorded") {
			recordings.emplace_back(argv[++i]);
		} else if (arg == "--states") {
			make_seeds.states = argv[++i];
			seed_option       = true;
		} else if (arg == "--stages") {
			make_seeds.stages = argv[++i];
			seed_option       = true;
		} else if (arg == "--limit") {
			char* end        = nullptr;
			make_seeds.limit = std::strtoll(argv[++i], &end, 10);
			if (end == argv[i] || *end != '\0') return Usage();
			seed_option = true;
		} else {
			return Usage();
		}
	}
	// A game packed into a ZArchive: its files are read inside it ("game.zar!/..."), the archive kept open.
	std::shared_ptr<Common::ArchiveReader> archive;
	if (Common::IsSupportedArchive(game) && Common::File::IsFileExisting(game)) {
		archive = Common::OpenArchive(game);
		game    = Common::MakeArchivePath(game);
	}
	// The game's param.json copied (no Vulkan device, none of the emulator's subsystems).
	if (!param_copy.empty()) {
		if (game.empty()) return Usage();
		Common::File in;
		if (!Common::File::IsFileExisting(game / "eboot.bin") ||
		    !in.Open(game / "sce_sys" / "param.json", Common::File::Mode::Read)) {
			std::fprintf(stderr, "no game (eboot.bin and sce_sys/param.json) in %s\n", game.string().c_str());
			return 1;
		}
		const auto text = in.ReadWholeBuffer();
		Common::File out;
		if (!out.Create(param_copy)) {
			std::fprintf(stderr, "cannot write %s\n", param_copy.string().c_str());
			return 1;
		}
		uint32_t written = 0;
		out.Write(text, &written);
		return written == text.Size() ? 0 : 1;
	}
	// Specialization hints (shader-hints.h): no Vulkan device, none of the emulator's subsystems.
	if (!make_hints.empty() || !apply_hints.empty()) {
		const bool make = !make_hints.empty();
		const bool inputs = make ? !recordings.empty() && options.out.empty() : !options.out.empty() && recordings.empty();
		if (game.empty() || options.seeds.empty() || merge || status || !make_seeds.out.empty() || seed_option ||
		    (make && !apply_hints.empty()) || !inputs)
			return Usage();
		const auto id = ShaderHints::GameId(game);
		if (id.empty()) {
			std::fprintf(stderr, "no title and version (sce_sys/param.json) in %s\n", game.string().c_str());
			return 1;
		}
		const int code = make ? ShaderHints::Make(make_hints, options.seeds, recordings, id)
		                      : ShaderHints::Apply(apply_hints, options.seeds, options.out, id);
		std::fflush(nullptr);
		std::_Exit(code);
	}
	if (!recordings.empty()) return Usage();
	// The seed file from the game's files: no Vulkan device, none of the emulator's subsystems.
	if (!make_seeds.out.empty()) {
		if (game.empty() || merge || status || !options.seeds.empty()) return Usage();
		make_seeds.game = game;
		const int code  = StaticSeeds::Make(make_seeds);
		std::fflush(nullptr);
		std::_Exit(code);
	}
	if (seed_option || game.empty() || (!merge && options.seeds.empty()) || options.threads == 0 ||
	    options.shards == 0 || options.shard >= options.shards)
		return Usage();

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::OutputDirection::Silent;
	config.shader_validation_enabled = validate;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	// The title the caches are named after.
	const auto param_json = game / "sce_sys" / "param.json";
	if (!Common::File::IsFileExisting(param_json)) {
		std::fprintf(stderr, "no %s\n", param_json.string().c_str());
		return 1;
	}
	Loader::SystemContentLoadParamSfo(param_json);

	static Libs::Graphics::GraphicContext graphics;
	if (!Libs::Graphics::CreateHeadlessGraphicContext(graphics)) {
		std::fprintf(stderr, "no Vulkan device\n");
		return 1;
	}
	if (amd) Libs::Graphics::ShaderRecompiler::SetDeviceStorageBufferBounds(false);
	// Compute seeds as the game gets this GPU's programs (PipelineCache::GetComputeProgram: 64-lane host subgroups
	// where wave64 programs run natively, else 32), whatever host subgroups they were listed or recorded with.
	if (options.host_subgroup_size == 0) options.host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
	if (status) {
		// (outdated: this GPU's and driver's, older than the seeds.)
		const bool built = PipelineCache::StaticCacheCurrent(graphics, {});
		std::printf("inputs %s\nstatic cache %s\n",
		            PipelineCache::StaticInputsCurrent(graphics, options.seeds) ? "current" : "stale",
		            !built                                                     ? "stale"
		            : PipelineCache::StaticCacheCurrent(graphics, options.seeds) ? "current"
		                                                                         : "outdated");
		std::fflush(nullptr);
		std::_Exit(0);
	}
	size_t     left_out = 0;
	const bool done     = merge ? PipelineCache::MergePrecompileShards(graphics, prune)
	                            : PipelineCache::Precompile(graphics, options, left_out);
	std::printf("Shader precompile: %s\n", !done ? "failed" : left_out != 0 ? "partial" : "complete");
	std::fflush(nullptr);
	// 3: pipelines' binaries were left out (PipelineBinaryWriter): run this shard as two.
	std::_Exit(!done ? 1 : left_out != 0 ? 3 : 0);
}
