#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPREWARM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPREWARM_H_

// Record-and-replay pipeline prewarm (any game), compute and graphics.
//
// Record: _PipelineCache/<title>.prewarm2 is a persistent seed that only grows across sessions.
// Every shader module's SPIR-V is stored once (by content hash) and every pipeline the game makes
// is stored once as its full creation state: for compute the descriptor bindings and subgroup size,
// for graphics the whole VkGraphicsPipelineCreateInfo (stages, vertex input, raster, blend, depth,
// dynamic states, attachment formats, layout bindings), keyed by a hash of all of it.
// Replay: at boot (first pipeline asked for), every recorded pipeline is created on many background
// threads into the shared driver cache, in recorded order. Gameplay takes the finished VkPipeline
// instead of compiling it (one being built is waited for, one not started is compiled as before).
// With the driver cache (_PipelineCache/<title>.bin) still valid this costs ~1 ms per pipeline; after
// a driver update (the .bin is invalidated) it rebuilds everything in the background.
//
// KYTY_PREWARM=0             off (no record, no replay)
// KYTY_PREWARM_THREADS=<n>   replay threads (default: CPUs - 4, at least 1)
// KYTY_PREWARM_WAIT=1        the first pipeline waits until the replay is finished
// KYTY_PREWARM_ONLY=1        precompile step: build the whole seed, save the driver cache, exit
// Log: "Prewarm: ..." progress, and "PipelineStats: ..." every 60 s and at exit.

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

bool Enabled();

// Every shader module that may end in a pipeline (stores its SPIR-V once in the seed).
void NoteSpirv(vk::ShaderModule module, std::span<const uint32_t> spirv);

// Compute: the pipeline's key (0: not recordable); records it when the seed lacks it.
uint64_t Register(vk::ShaderModule module, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                  uint32_t required_subgroup_size, uint32_t layout_flags);

// Graphics: the pipeline's key (0: not recordable); records it when the seed lacks it.
// `info.layout` is ignored (the layout is described by `bindings` + `push_stages`).
uint64_t RegisterGraphics(const vk::GraphicsPipelineCreateInfo&            info,
                          std::span<const vk::DescriptorSetLayoutBinding> bindings,
                          uint32_t push_stages, uint32_t layout_flags);

// Starts the replay once (later calls do nothing). True: KYTY_PREWARM_ONLY=1 and the replay is
// finished (the caller saves the driver cache and exits).
bool EnsureStarted(GraphicContext& graphics, const std::string& title_id, vk::PipelineCache driver_cache);

// The prewarmed pipeline of this key (waits while it is being built), or nullptr. Counts the
// gameplay hit or miss.
vk::Pipeline Take(uint64_t key, bool graphics);

// A gameplay pipeline compiled at runtime (not served by the prewarm), for the stats line.
void NoteRuntimeCompile(bool graphics, uint64_t ns);
// The calling thread is a background compile thread (its compiles do not block a frame).
void MarkAsyncThread();
// A shader translation (guest code -> SPIR-V) at runtime, for the stats line.
void NoteTranslation(uint64_t ns);

void Shutdown();

} // namespace Libs::Graphics::Prewarm

#endif
