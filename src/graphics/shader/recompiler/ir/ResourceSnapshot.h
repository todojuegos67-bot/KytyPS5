#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

// Pattern is a plan's only (RecognizeLoadedPatternFill): the renderer reads its value as the dispatch runs.
enum class UniformFillKind { None, Buffer, Image, Pattern };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	// Per image: a group's candidates in rank order (GroupIndirectImages); empty for other images.
	std::vector<std::vector<DescriptorValue>> image_groups;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	// Words of flattened_srt that a shader wrote and only the shader reads: the guest address and the word's index.
	// flattened_srt holds 0 there; the host copies them on the GPU after uploading it (CopyGpuFills).
	std::vector<std::pair<uint64_t, uint32_t>> gpu_fills;
	// The game's descriptor heap this draw reads images from by slot (base 0: none); the host
	// mirrors it into the image heap.
	struct ImageHeapTable {
		uint64_t base = 0;
		uint64_t size = 0;
	};
	ImageHeapTable               image_heap;
	ImageHeapTable               sampler_heap; // the game's sampler heap, for the same mirror
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	// Per buffer of the shader: the bytes from its descriptor's base its stores can reach,
	// [first, second). Empty when the dispatch does not bound them: written buffers count whole.
	std::vector<std::pair<uint64_t, uint64_t>> buffer_writes;
	// Per buffer: when a store's invocations each reach their own bytes (a scatter by an index list),
	// those bytes, sorted and merged; buffer_writes holds their hull. Empty otherwise.
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> buffer_write_ranges;
	// The form of each form slot (passes/ImageForms.h), ImageFormBits a slot: the shader data's form dword.
	uint32_t image_forms = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
