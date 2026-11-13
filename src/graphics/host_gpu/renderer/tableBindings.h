#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

// KYTY_TABLE (tableMode.h): the bindings table, after chenxiao07's table sets
// (experiment/perf-40fps-20260926: table-xpr.inc / native-xpr.inc, the perf(table) series of
// 10-11 October, 17542a12f's memo of a pair's last set). There a table draw takes the descriptor
// set its pair and state used last while the set's validation holds. Here, per graphics stage
// slot (vertex 0-2, pixel), the slot's PreparedBindings stay as the last draw left them, and this
// remembers what they were prepared from: the program and the image and sampler descriptors
// (entries by their hash). Buffers are not part of it: FindBuffers and RebindBuffers run for every
// stage (V#s of per-frame memory made the key miss, and reused buffer views went stale). A draw whose programs came from the programs table and whose stage repeats them
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
	std::vector<ShaderRecompiler::IR::DescriptorValue> images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> samplers;
	uint64_t                                           hash = 0;
	// The bindings the normal path made from them (copied into the stage's slot on a match).
	PreparedBindings prepared;
};

// Entries by a hash of the program and descriptors: a slot's draws alternate between programs,
// so the slot's last draw alone matched too few ("key" misses).
constexpr uint32_t Entries = 4096;

[[nodiscard]] inline uint64_t KeyHash(const void* program, bool normalize,
                                      const ShaderRecompiler::IR::ResourceSnapshot& snapshot);

struct Table {
	std::vector<Entry>               entries = std::vector<Entry>(Entries);
	// The entry each slot's bindings came from (null: none, or not kept).
	std::array<Entry*, Slots>        current {};
	// Diagnostics of "key" misses: per program, the image and sampler list hashes of its last
	// stored entry.
	std::unordered_map<const void*, std::pair<uint64_t, uint64_t>> last_by_program;
	// KYTY_TABLE=2: the slot's bindings before the normal path prepared them again.
	std::array<std::optional<PreparedBindings>, Slots> verify {};
};

inline uint64_t KeyHash(const void* program, bool normalize,
                        const ShaderRecompiler::IR::ResourceSnapshot& snapshot) {
	const auto mix = [](uint64_t h, const void* data, size_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		for (size_t i = 0; i < size; i++) {
			h = (h ^ bytes[i]) * 0x100000001b3ull;
		}
		return h;
	};
	uint64_t h = reinterpret_cast<uint64_t>(program) * 0x9e3779b97f4a7c15ull ^ (normalize ? 1u : 0u);
	for (const auto* list: {&snapshot.images, &snapshot.samplers}) {
		for (const auto& value: *list) {
			h = mix(h, value.dwords.data(), value.dword_count * sizeof(uint32_t));
		}
		h = mix(h, "|", 1);
	}
	return h;
}

[[nodiscard]] inline uint64_t
ListHash(const std::vector<ShaderRecompiler::IR::DescriptorValue>& list) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (const auto& value: list) {
		for (uint32_t i = 0; i < value.dword_count; i++) {
			h = (h ^ value.dwords[i]) * 0x100000001b3ull;
		}
	}
	return h;
}

} // namespace Libs::Graphics::TableBindings

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLEBINDINGS_H_
