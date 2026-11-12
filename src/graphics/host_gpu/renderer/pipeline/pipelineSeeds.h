#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_

// Guest pipeline seeds (GT7 PC port, as Demon's Souls' precompile): what the GAME gave the
// emulator for each shader program and pipeline it made, independent of the host GPU, so that any
// player's PC can translate the shaders with its own recompiler and switches and make the
// pipelines in its own driver before the game starts (precompile mode).
//
//   <shader cache>/guest.seeds   one record per program and per graphics pipeline
//   (header: magic, format; record: u32 size, u64 id, u64 XXH3 of the payload, payload)
//
// A program record ('P'): the stage, the guest shader code (the expanded code for a compute
// shader that inlines called functions), its back code and user data, the stage input
// description (ShaderVertexInputInfo / ShaderPixelInputInfo / ShaderComputeInputInfo, with the
// host-dependent fields cleared), the resource specialization and the push data cursor: what
// ProgramCache translates and compiles. A pipeline record ('G'): the programs (by their record
// hashes), the rendering formats, the static state and the vertex input layout of the key.
//
// Records are written only when a program permutation or a pipeline is made for the first time in
// a run (never per draw), on a writer thread, each once (by its XXH3 hash). The file is
// cumulative; files can be joined with `copy /b a+b c` (a repeated header is skipped).
// KYTY_PIPELINE_SEEDS=0 turns recording off.
//
// Precompile mode: KYTY_PRECOMPILE=1. The emulator starts as usual up to the Vulkan device, then
// (PipelineCache::PrecompileSeeds) translates every seed program into the shader disk cache and
// makes every pipeline into the driver cache on all processors, saves both and exits (code 0).
// KYTY_PRECOMPILE_SEEDS=<file>[;<file>...]: more seed files read with the player's own (the
// release's seeds-PPSA01317_<version>.seeds). KYTY_PRECOMPILE_PROGRESS=<file>: "<done> <total>
// <compiled> <failed> <final>" written there while it runs; <file>.stop asks it to stop early
// (what was done is saved).

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace Libs::Graphics::PipelineSeeds {

constexpr uint8_t KindProgram  = 'P';
constexpr uint8_t KindPipeline = 'G';

// `format`: the layout of the records (sizes of the structures they copy); a file of another
// format is set aside (renamed .old) and not read.
void Open(const std::filesystem::path& directory, uint64_t format);
[[nodiscard]] bool Recording();

// A program permutation was made: its record (payload[0] == KindProgram) under `hash`, the hash of
// its canonical translation inputs (equal for equal programs). `ids`: the program ids it is known
// by in this run (ShaderProgram::id, normalized_id).
void RecordProgram(uint64_t hash, std::vector<uint8_t> payload, std::span<const uint64_t> ids);
// The record hash of the program with this id in this run, or 0.
[[nodiscard]] uint64_t ProgramHash(uint64_t id);
void RecordPipeline(std::vector<uint8_t> payload);
void Flush();

[[nodiscard]] bool PrecompileRequested();

// Round seedcs: whether a seed of the file read at Open holds this program source (stage, shader
// hash and the code translated, the expanded code for a compute shader that inlines calls), for
// the shader disk cache's misses: a seeded source that misses has another key than the
// precompile's (or failed there); an unseeded one is new to the seeds (a new expansion or
// content). Only asked on a miss. KYTY_SEED_MISS_CHECK=0: not asked.
[[nodiscard]] uint64_t SourceId(uint32_t stage, uint64_t shader_hash, std::span<const uint32_t> code);
[[nodiscard]] bool     MissCheck();
[[nodiscard]] bool     HasSource(uint64_t source_id);
void                   NoteSourceMiss(bool seeded);
void                   MissCounts(uint64_t& seeded, uint64_t& unseeded);

// Every intact record of the player's file and of KYTY_PRECOMPILE_SEEDS's, once each.
struct Record {
	uint64_t             hash = 0;
	std::vector<uint8_t> payload;
};
[[nodiscard]] std::vector<Record> LoadAll(uint32_t* other_format = nullptr);

// Precompile progress (KYTY_PRECOMPILE_PROGRESS) and its stop file.
void               WriteProgress(uint32_t done, uint32_t total, uint32_t compiled, uint32_t failed, bool final);
[[nodiscard]] bool StopRequested();
void               RemoveStopFile();

} // namespace Libs::Graphics::PipelineSeeds

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_ */
