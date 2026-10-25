#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPREWARM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPREWARM_H_

// Record-and-replay compute pipeline prewarm (any game).
//
// Record: every compute pipeline the game makes is appended once to _PipelineCache/<title>.prewarm
// (its SPIR-V, descriptor bindings and creation options, keyed by a hash of all of them).
// Replay: the next run creates every recorded pipeline on many background threads as soon as the
// first compute shader is asked for (in recorded order: what the game needed first comes first) and
// gameplay takes the finished VkPipeline instead of compiling (a pipeline still being built is waited
// for, one not started yet is compiled by gameplay as before).
//
// KYTY_PREWARM=0           off (no record, no replay)
// KYTY_PREWARM_THREADS=<n> replay threads (default: CPUs - 2, at least 1)
// KYTY_PREWARM_WAIT=1      the first compute pipeline waits until the replay is finished (a loading
//                          pause with progress instead of compiles during play)
// The log shows "Prewarm: ..." progress and, every 200 compute pipelines and at exit, how many
// gameplay pipelines the prewarm served ("served") and how many it did not have ("missed").

#include "graphics/host_gpu/graphicContext.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::Prewarm {

struct Binding {
	uint32_t binding = 0;
	uint32_t type    = 0;
	uint32_t count   = 0;
	uint32_t stages  = 0;
};

struct Record {
	uint64_t              key                    = 0;
	uint32_t              required_subgroup_size = 0;
	uint32_t              flags                  = 0; // FlagBindless
	std::vector<Binding>  bindings;
	std::vector<uint32_t> spirv;
};

constexpr uint32_t FlagBindless = 1u;

bool Enabled();

// The SPIR-V of a compute shader module, kept until Register() of its pipeline.
void NoteSpirv(vk::ShaderModule module, std::span<const uint32_t> spirv);

// The pipeline's key (0: unknown module, nothing recorded); records it when the file lacks it.
uint64_t Register(vk::ShaderModule module, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                  uint32_t required_subgroup_size, uint32_t flags);

// Starts the replay once (later calls do nothing).
void EnsureStarted(GraphicContext& graphics, const std::string& title_id, vk::PipelineCache driver_cache);

// The prewarmed pipeline of this key (waits while it is being built), or nullptr. A key taken or
// missed once is not built any more.
vk::Pipeline Take(uint64_t key);

void Shutdown();

// Defined next to the engine's own compute pipeline code (shaders.cpp): creates the pipeline of a
// record with a layout of its own (identically defined, so compatible with gameplay's).
vk::Pipeline BuildComputePipeline(GraphicContext& graphics, const Record& record,
                                  vk::PipelineCache driver_cache);

} // namespace Libs::Graphics::Prewarm

#endif
