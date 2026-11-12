#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// KYTY_TABLE (tableMode.h): the bindings table, after chenxiao07's table sets
// (experiment/perf-40fps-20260926: table-xpr.inc / native-xpr.inc, the perf(table) series of
// 10-11 October, 17542a12f's memo of a pair's last set). There a table draw takes the descriptor
// set its pair and state used last while the set's validation holds. Here, per graphics stage
// slot (vertex 0-2, pixel), the slot's PreparedBindings stay as the last draw left them, and this
// remembers what they were prepared from: the program and the image, sampler and buffer
// descriptors. A draw whose programs came from the programs table and whose stage repeats them
// takes the slot's bindings as they are, once every image passes ReuseTextureBinding (registered,
// not CPU-dirty nor buffer-written, all pages tracked) under the same image set generation and
// every buffer passes BufferCache::ReuseReadBinding (alive, same handle, no CPU-dirty page). Only
// stages without storage images, written or texel buffers, bindless heaps, DMA, dynamic mips or
// light grids are kept. GPU thread only (RenderExecutor).
namespace Libs::Graphics::TableBindings {

// m_graphics_bindings.vertex[0..2], then .pixel.
constexpr uint32_t Slots = 4;

struct Entry {
	bool                                            valid     = false;
	const ShaderRecompiler::IR::CompiledShaderInfo* program   = nullptr;
	bool                                            normalize = false;
	uint64_t                                        image_generation  = 0;
	uint64_t                                        buffer_generation = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> samplers;
	std::vector<ShaderRecompiler::IR::DescriptorValue> buffers;
};

struct Table {
	std::array<Entry, Slots> slots {};
	// KYTY_TABLE=2: the slot's bindings before the normal path prepared them again.
	std::array<std::optional<PreparedBindings>, Slots> verify {};
};

} // namespace Libs::Graphics::TableBindings

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_
