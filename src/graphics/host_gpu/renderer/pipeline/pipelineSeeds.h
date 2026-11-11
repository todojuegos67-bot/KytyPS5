#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_

// Pipeline seeds (GT7 PC port): everything vkCreate*Pipelines was given for each pipeline the game
// made, kept on disk so that a later run can make them all again without the game
// (precompile mode) and fill the driver's pipeline cache before the first race.
//
//   <shader cache>/pipelines.seeds      one record per pipeline (states, layout, SPIR-V hashes)
//   <shader cache>/seed-spirv/<h>.spv   the SPIR-V of each module, by its XXH3 hash
//
// Recording costs nothing per draw: a record is written only when a pipeline is made (the GPU
// thread or a pipeline worker) and a module's SPIR-V only when the module is made, both on a
// writer thread. KYTY_PIPELINE_SEEDS=0 turns recording off.
//
// Precompile mode: KYTY_PRECOMPILE=1. The emulator starts the game as usual up to the creation of
// the Vulkan device, makes every seed's pipeline on all processors into the driver cache, saves it
// and ends (exit code 0). KYTY_PRECOMPILE_PROGRESS=<file>: "<done> <total> <compiled> <failed>"
// written there while it runs; the file <file>.stop asks it to stop early (what was compiled is
// saved).

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace Libs::Graphics::PipelineSeeds {

// Pipeline layout as made by shaders.cpp: set 0 is the pipeline's own, set 1 the bindless table.
struct Layout {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	bool                                        push_descriptors = false;
	bool                                        bindless         = false;
	vk::PushConstantRange                       push_constants {};
};

// Called once the shader cache directory and device are known (PipelineCache's constructor).
// `session`: hash of the shader disk cache session key (GPU and recompiler switches); seeds made
// under another one are not replayed.
void Open(const std::filesystem::path& directory, uint64_t session);
[[nodiscard]] bool Recording();

void NoteModule(vk::ShaderModule module, std::span<const uint32_t> code);
void RecordGraphics(const vk::GraphicsPipelineCreateInfo& info, const Layout& layout);
void RecordCompute(const vk::ComputePipelineCreateInfo& info, const Layout& layout);
void Flush();

[[nodiscard]] bool PrecompileRequested();

struct ReplayResult {
	uint32_t total    = 0; // seeds of this session
	uint32_t other    = 0; // seeds of another GPU / switches (skipped)
	uint32_t compiled = 0;
	uint32_t cached   = 0; // already in a driver cache
	uint32_t failed   = 0;
	bool     stopped  = false;
};
// Makes every seed's pipeline into `driver_cache` (`older`: read-only caches looked up first) on
// all processors.
ReplayResult Replay(vk::Device device, vk::PipelineCache driver_cache,
                    std::span<const vk::PipelineCache> older, vk::DescriptorSetLayout bindless,
                    bool cache_control);

} // namespace Libs::Graphics::PipelineSeeds

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESEEDS_H_ */
