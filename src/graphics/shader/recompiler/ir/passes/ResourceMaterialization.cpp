#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectDescriptorProbes = 65536u;

// EQAA FMASK formats (kFmask<bits>_S<samples>_F<fragments> with fewer fragments than samples):
// the number of color fragments stored per pixel. 0 for formats with one fragment per sample,
// whose native mapping is the identity.
uint8_t FmaskEqaaFragmentCount(Prospero::BufferFormat format) {
	using F = Prospero::BufferFormat;
	switch (format) {
		case F::kFmask8_S2_F1:
		case F::kFmask8_S4_F1:
		case F::kFmask8_S8_F1:
		case F::kFmask16_S16_F1: return 1;
		case F::kFmask8_S4_F2:
		case F::kFmask16_S8_F2:
		case F::kFmask32_S16_F2: return 2;
		case F::kFmask32_S8_F4:
		case F::kFmask64_S16_F4: return 4;
		case F::kFmask64_S16_F8: return 8;
		default: return 0;
	}
}

// One 32-bit word of an expanded FMASK value (four bits per sample) covering samples
// first_sample..first_sample+7. Without EQAA (fragments == 0) sample i maps to host sample i;
// with EQAA it maps to fragment (i mod fragments), so every index names a host sample that exists.
constexpr uint32_t FmaskSampleMapWord(uint32_t fragments, uint32_t first_sample) {
	if (fragments == 0) {
		return first_sample == 0 ? 0x76543210u : 0xfedcba98u;
	}
	uint32_t word = 0;
	for (uint32_t i = 0; i < 8; i++) {
		word |= ((first_sample + i) % fragments) << (i * 4u);
	}
	return word;
}
static_assert(FmaskSampleMapWord(0, 0) == 0x76543210u && FmaskSampleMapWord(0, 8) == 0xfedcba98u);
static_assert(FmaskSampleMapWord(8, 8) == 0x76543210u);
static_assert(FmaskSampleMapWord(4, 0) == 0x32103210u);
static_assert(FmaskSampleMapWord(2, 8) == 0x10101010u);
static_assert(FmaskSampleMapWord(1, 0) == 0u);

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	// Round 124: a multisample depth or stencil T# (depth tiling) read by an instruction of a
	// single-sample dimension (image_sample, image_load without a sample index) reads its first
	// sample on the console. The shader gets a single-sample image, and the host binds a
	// single-sample copy of that sample (TextureCache::SingleSampleDepth, SingleSampleStencil).
	// Before, the shader sampled a multisample image (not valid SPIR-V): on NVIDIA the samples
	// came back interleaved, rows of two pictures, so a replay's light sprites tested the wrong
	// depth and its 16x16 depth reduction for the CPU read two pictures at once. Multisample
	// colour T#s keep their multisample dimension, as before.
	constexpr uint32_t depth_tiling = static_cast<uint32_t>(Prospero::TileMode::kDepth);
	const bool single_sample_read   = ((descriptor.dwords[3] >> 20u) & 0x1fu) == depth_tiling &&
	                                (requested == Decoder::ImageDimension::Dim2D ||
	                                 requested == Decoder::ImageDimension::Dim2DArray);
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (single_sample_read) {
				return is_array ? Decoder::ImageDimension::Dim2DArray
				                : Decoder::ImageDimension::Dim2D;
			}
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa:
			return single_sample_read ? Decoder::ImageDimension::Dim2D
			                          : Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (!IsValidImageSwizzle(words[3] & 0xfffu)) return false;
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	const auto base_level = (words[3] >> 12u) & 0xfu;
	const auto last_level = (words[3] >> 16u) & 0xfu;
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && last_level >= 1 && last_level <= 3 &&
		       (r128 || max_mip == last_level);
	}
	return base_level <= last_level;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

enum class SamplerClass : uint8_t { Float, Integer, PointInteger };

SamplerClass ClassifySampler(const ImageResource& image) {
	if (image.numeric_class == Prospero::TextureNumericClass::Sint ||
	    (image.numeric_class == Prospero::TextureNumericClass::Uint &&
	     image.conversion_format != Prospero::BufferFormat::kInvalid)) {
		return SamplerClass::PointInteger;
	}
	return image.numeric_class == Prospero::TextureNumericClass::Uint ? SamplerClass::Integer
	                                                               : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return last - base + 1u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.read_memory != nullptr) {
		if (!capture.source.read_memory(capture.source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value>       pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = dynamic_offset & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

thread_local int g_indirect_failure_line = 0;

bool FailIndirect(int line) {
	if (g_indirect_failure_line == 0) {
		g_indirect_failure_line = line;
	}
	return false;
}

// The shader looks the key up in the bindless translation table: nothing to enumerate. Record
// the heap for the host and reserve the two words it patches (region base, entry count); zeros
// resolve every key to the placeholder in slot 0.
bool MaterializeBindlessImage(const DescriptorSource::IndirectDescriptor& indirect,
                               const DescriptorValue& table_value, uint32_t image_index,
                               ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	ShaderBufferResource heap;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, heap)) {
		return FailIndirect(__LINE__);
	}
	const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
	snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
	snapshot.images[image_index] = {.dword_count = 8u};
	snapshot.bindless_heaps.push_back({.base           = heap.Base48(),
	                                   .size           = heap.GetSize(),
	                                   .table_offset   = indirect.table_offset,
	                                   .table_stride   = indirect.table_stride != 0u
	                                                         ? indirect.table_stride
	                                                         : 32u,
	                                   .image          = image_index,
	                                   .mapping_offset = mapping_offset});
	auto& root                      = specialization.images[image_index];
	root.indirect_root              = image_index;
	root.indirect_mapping_offset    = mapping_offset;
	root.indirect_search_iterations = 0;
	root.bindless                   = true;
	return true;
}

bool NormalizeIndirectStoreBuffer(DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    descriptor.Base48() == 0u || descriptor.GetSize() == 0u) {
		value.dwords.fill(0u);
	} else if (descriptor.Stride() != 1u || descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	           descriptor.OutOfBounds() != 3u) {
		return SpecializationFail("indirect buffer stores require linear byte bounds");
	}
	return true;
}

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   const DescriptorSource::IndirectDescriptor& indirect,
                                   uint32_t resource_index, uint32_t dword_count,
                                   const SrtRuntime& runtime, SrtWalker& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   uint32_t maximum_resources, Normalize&& normalize) {
	const auto  descriptor_bytes = dword_count * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	auto& keys = program.material_keys;
	const auto read_keys = [&](const ShaderBufferResource& material, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (step == 4u) {
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, keys))
				return false;
		} else {
			for (auto& key: keys) {
				if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime,
				                     {&key, 1}))
					return false;
				first += step;
			}
		}
		// A key packed into a bit-field of the material word.
		for (auto& key: keys) key = (key >> indirect.key_shift) & indirect.key_mask;
		return true;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	if (sources.empty()) {
		keys.clear();
		DescriptorValue material_value;
		DescriptorValue table_value;
		if ((indirect.material_source != UINT32_MAX &&
		     !clean.EvaluateDescriptor(indirect.material_source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return FailIndirect(__LINE__);
		}
		ShaderBufferResource table;
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return FailIndirect(__LINE__);
		}
		if (indirect.material_source == UINT32_MAX) {
			uint32_t key_count = 0;
			if (indirect.workgroup_axis != UINT32_MAX) {
				if (indirect.workgroup_axis >= runtime.workgroup_counts.size()) return FailIndirect(__LINE__);
				key_count = runtime.workgroup_counts[indirect.workgroup_axis];
				if (key_count == 0u) return FailIndirect(__LINE__);
			} else {
				if (table_value.dword_count != 2u || !clean.Evaluate(indirect.key_count, key_count))
					return FailIndirect(__LINE__);
				if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
			}
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (key_count != 0u && uint64_t {indirect.table_offset} +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return FailIndirect(__LINE__);
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), 0u);
		} else if (!indirect.selector_first.IsEmpty()) {
			ShaderBufferResource material;
			uint32_t             first = 0, count = 0;
			if (!DecodeBufferDescriptor(material_value, material) || material.Type() != 0u ||
			    (indirect.selector_shift != 0u && (material.Base48() & 3u) != 0u) ||
			    material.SwizzleEnabled() ||
			    material.AddTid() || material.OutOfBounds() != 0u ||
			    uint64_t {indirect.selector_offset} + 4u > material.Stride() ||
			    !clean.Evaluate(indirect.selector_first, first) ||
			    !clean.Evaluate(indirect.key_count, count) || count > MaxIndirectDescriptorProbes ||
			    uint64_t {first} + count > material.NumRecords())
				return false;
			if (count != 0u && (uint64_t {first} + count - 1u) * material.Stride() +
			                           indirect.selector_offset + 4u >
			                       uint64_t {UINT32_MAX} + 1u)
				return false;
			if (!read_keys(material, uint64_t {first} * material.Stride() + indirect.selector_offset,
			               material.Stride(), count)) return false;
		} else if (!indirect.selector_mask.IsEmpty()) {
			uint32_t mask = 0;
			uint32_t count = 0;
			if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
			    !clean.Evaluate(indirect.selector_mask, mask) ||
			    !clean.Evaluate(indirect.key_count, count) || count == 0u || count > 32u) {
				return FailIndirect(__LINE__);
			}
			if (count < 32u) mask &= (1u << count) - 1u;
			const auto material_base =
			    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
			keys.reserve(std::popcount(mask));
			while (mask != 0u) {
				const auto index = std::countr_zero(mask);
				const auto offset = static_cast<uint64_t>(indirect.selector_offset) +
				                    static_cast<uint64_t>(index) * indirect.selector_stride;
				if (offset > UINT32_MAX) return FailIndirect(__LINE__);
				uint32_t key = 0;
				if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
				                     runtime, {&key, 1})) return FailIndirect(__LINE__);
				keys.push_back((key >> indirect.key_shift) & indirect.key_mask);
				mask &= mask - 1u;
			}
		} else {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material) || table_value.dword_count != 4u) {
				return FailIndirect(__LINE__);
			}
			// The first aligned offset includes the immediate added after shader U32 arithmetic.
			const auto step = std::max<uint64_t>(4u,
			    std::gcd<uint64_t>(indirect.selector_stride, uint64_t {1} << 32u));
			const uint64_t first = indirect.selector_offset;
			const auto size = material.GetSize();
			const auto limit = std::min(first + (uint64_t {1} << 32u) - step, size >= 4u ? size - 4u : 0u);
			const auto probe_count = size >= 4u && first <= limit ? (limit - first) / step + 1u : 0u;
			if (probe_count > MaxIndirectDescriptorProbes) {
				return FailIndirect(__LINE__);
			}
			if (!read_keys(material, first, step, probe_count)) return FailIndirect(__LINE__);
			if (indirect.selector_shift == 0u) keys.push_back(0u);
		}
		if (indirect.material_source != UINT32_MAX) {
			if (indirect.selector_shift != 0u) {
				for (auto& key: keys) key >>= indirect.selector_shift;
				keys.push_back(0u); // An out-of-range material load returns zero.
			}
			std::ranges::sort(keys);
			keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		}
	}

	const auto children_begin = descriptors.size();
	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto root_resource  = specializations[resource_index];
	const auto key_count = sources.empty() ? keys.size() : sources.size();
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_count * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? keys[entry] : entry;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			const auto table_offset =
			    static_cast<uint32_t>(key * indirect.table_stride) + indirect.table_offset;
			if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(dword_count))) {
				return FailIndirect(__LINE__);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return FailIndirect(__LINE__);
		}
		if (!normalize(candidate)) return false;
		uint32_t ordinal = 0;
		if (entry == 0) {
			descriptors[resource_index] = candidate;
		} else if (descriptors[resource_index] != candidate) {
			const auto found =
			    std::find(descriptors.begin() + children_begin, descriptors.end(), candidate);
			ordinal = static_cast<uint32_t>(found - descriptors.begin() - children_begin + 1u);
			if (found == descriptors.end()) {
				if (descriptors.size() >= maximum_resources) {
					return FailIndirect(__LINE__);
				}
				descriptors.push_back(candidate);
				auto child          = root_resource;
				child.indirect_root = resource_index;
				specializations.push_back(child);
			}
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_count);
	}
	return true;
}
} // namespace

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, 3>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	// what names the use, for the fatal message.
	uint32_t operator()(uint32_t index, const char* what) const {
		if (index >= source_count) {
			EXIT("image remap: %s refers to image %u, but the shader has %u\n", what, index,
			     source_count);
		}
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization,
                                        bool                    float_image_atomics) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the set-1 array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D
				                      : base.dimension;
				image.cube      = base.cube;
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		// Image atomics operate on the raw texel bits, and only the integer atomics decode, so an
		// atomic on an R32 sint or float surface is a uint atomic on the same bits.
		// --no-float-image-atomics still turns the float case off.
		const bool float_atomic =
		    float_image_atomics && base.atomic && format == Prospero::BufferFormat::k32Float;
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt && !float_atomic)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		image.fmask_fragments   = image.fmask ? FmaskEqaaFragmentCount(format) : 0;
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const bool raw_sint_storage = storage && format == Prospero::BufferFormat::k32SInt &&
		                              base.written && !base.read && !base.atomic;
		image.numeric_class         = Prospero::SampledTextureNumericClass(format);
		if (storage) {
			if ((!raw_sint_storage && !base.atomic &&
			     image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
			    // Stores re-pack only raw uint bitfields (PackImageTexel).
			    (image.conversion_format != Prospero::BufferFormat::kInvalid &&
			     Format::GetFormatComponentType(format) != Format::ComponentType::Uint)) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			// Integer image atomics operate on the raw 32-bit texel, independently of
			// the descriptor numeric format. R32 float/sint loads and stores preserve
			// the same register bits through this uint view.
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
			image.clamp_half_float =
			    FloatStoreClampEnabled() && base.written && !base.atomic &&
			    image.numeric_class == Prospero::TextureNumericClass::Float &&
			    (format == Prospero::BufferFormat::k16Float ||
			     format == Prospero::BufferFormat::k16_16Float ||
			     format == Prospero::BufferFormat::k16_16_16_16Float);
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * 2u >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			if (image.numeric_class != image_class.numeric_class ||
			    (!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension))) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			// A bindless sampler is one slot of the host's table for all of its uses: its other
			// classes share the first one's binding rather than adding copies.
			if (!first && base.samplers[index].bindless) {
				mapping[type] = index;
				continue;
			}
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.blocks.size() != program.block_info.size() || program.has_address_writes) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	std::vector<std::vector<uint32_t>> predecessors(blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch:
				successors = terminator.indirect_targets;
				break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
			predecessors[found->second].push_back(i);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	// Host branches only matter when they guard a later resource access. Leave
	// data-only branches (for example a mutable flag checked before a trap) on the GPU.
	std::vector<uint8_t> resource_reachable(blocks.size());
	std::vector<uint32_t> pending;
	for (uint32_t i = 0; i < blocks.size(); ++i) {
		if (!blocks[i].sources.empty() || !blocks[i].srt_reads.empty()) {
			resource_reachable[i] = 1;
			pending.push_back(i);
		}
	}
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		for (const auto predecessor: predecessors[index]) {
			if (resource_reachable[predecessor]) continue;
			resource_reachable[predecessor] = 1;
			pending.push_back(predecessor);
		}
	}
	for (auto& block: blocks) {
		block.condition = std::ranges::any_of(block.successors, [&](uint32_t successor) {
			return resource_reachable[successor] != 0;
		}) ? predicate(block.condition) : Value {};
	}
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.blocks.size() != program.block_info.size() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<uint32_t> visited;
	uint32_t                     index = 0;
	const Inst*                  store = nullptr;
	for (;;) {
		if (!visited.insert(index).second) return {};
		for (const auto& inst: *program.blocks[index]) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = program.block_info[index].terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		const auto next = std::ranges::find(program.block_info, term.true_block, &BlockInfo::id);
		if (next == program.block_info.end()) return {};
		index = static_cast<uint32_t>(next - program.block_info.begin());
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    !IsFullDispatchPredicate(store->Arg(3))) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

namespace {

// Integer operations whose result interval follows from their operands' intervals (see
// WriteInterval).
bool IsIntervalOperation(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::IMul32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::SelectU32: return true;
		default: return false;
	}
}

bool IsBoundedInvocationId(const ResourcePlan& program, const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::GetBuiltin || program.stage != ShaderType::Compute ||
	    !inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() || inst.Arg(1).U32() >= 3u) {
		return false;
	}
	switch (static_cast<StageInputKind>(inst.Arg(0).U32())) {
		case StageInputKind::WorkgroupId:
		case StageInputKind::LocalInvocationId:
		case StageInputKind::LocalInvocationIndex:
		case StageInputKind::GlobalInvocationId: return true;
		default: return false;
	}
}

// A store address operand the host can bound per dispatch: values it evaluates, compute invocation
// IDs, and interval operations on them. The select on the store's own predicate only matters for
// the stored lanes.
bool BoundableWriteOperand(const Program& program, Value value, Value predicate, uint32_t depth) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32u) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		return true;
	}
	if (IsIntervalOperation(inst->GetOpcode())) {
		if (inst->GetOpcode() == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
			return BoundableWriteOperand(program, inst->Arg(1), predicate, depth + 1u);
		}
		const auto first = inst->GetOpcode() == ValueOpcode::SelectU32 ? 1u : 0u;
		for (size_t i = first; i < inst->NumArgs(); ++i) {
			if (!BoundableWriteOperand(program, inst->Arg(i), predicate, depth + 1u)) {
				return false;
			}
		}
		return true;
	}
	return value.GetType() == Type::U32 &&
	       ValidateRuntimeValue(program, value, RuntimeValueType::Integer);
}

} // namespace

// Flat slots the host never needs, so their bytes can be left to the GPU when it wrote them
// (SrtGpuFill): the slot's value is a raw scalar read whose address comes from user data and
// immediates alone, and no value the host evaluates refers to the slot (descriptor dwords and
// keys, branch conditions, uniform fills, buffer write bounds, other reads' addresses). The host
// then only uploads the flat buffer; the shader reads the slot.
static std::vector<uint8_t> GpuFillSlots(const ResourcePlan& plan) {
	const auto           count = plan.srt_reads.size();
	std::vector<uint8_t> host(count, 0u);
	std::vector<Value>   pending;
	std::unordered_set<const Inst*> visited;
	const auto mark = [&](Value root) {
		pending.push_back(root);
		while (!pending.empty()) {
			const auto  value = pending.back().Resolve();
			pending.pop_back();
			const auto* inst  = value.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) {
				continue;
			}
			if (inst->GetOpcode() == ValueOpcode::ReadConst) {
				const auto slot = inst->Arg(1).Resolve();
				if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < count) {
					host[slot.U32()] = 1u;
					pending.push_back(plan.srt_reads[slot.U32()].value);
				}
				continue;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				pending.push_back(inst->Arg(arg));
			}
		}
	};
	for (const auto& source: plan.descriptor_sources) {
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			mark(source.dwords[dword]);
		}
		if (source.indirect_descriptor.has_value()) {
			mark(source.indirect_descriptor->key_count);
			mark(source.indirect_descriptor->selector_first);
			mark(source.indirect_descriptor->selector_mask);
		}
	}
	for (const auto& block: plan.control_flow) {
		mark(block.condition);
	}
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		mark(plan.uniform_fill.values[i]);
	}
	for (const auto& write: plan.buffer_writes) {
		mark(write.index);
		mark(write.offset);
		mark(write.soffset);
		mark(write.predicate);
	}
	// Another read's address operands (the read itself is not a host use of its own slot).
	for (const auto& read: plan.srt_reads) {
		if (const auto* inst = read.value.Resolve().TryInstruction(); inst != nullptr) {
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				mark(inst->Arg(arg));
			}
		}
	}
	// The read's own address must not depend on memory: one host read per slot, the slot's own.
	const auto memory_free = [&](const Inst& read) {
		std::vector<Value>              stack;
		std::unordered_set<const Inst*> seen;
		for (size_t arg = 0; arg < read.NumArgs(); arg++) {
			stack.push_back(read.Arg(arg));
		}
		while (!stack.empty()) {
			const auto* inst = stack.back().Resolve().TryInstruction();
			stack.pop_back();
			if (inst == nullptr || !seen.insert(inst).second) {
				continue;
			}
			switch (inst->GetOpcode()) {
				case ValueOpcode::ReadConst:
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer:
				case ValueOpcode::LoadBufferU32:
				case ValueOpcode::Phi: return false;
				default: break;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				stack.push_back(inst->Arg(arg));
			}
		}
		return true;
	};
	std::vector<uint8_t> fill(count, 0u);
	for (size_t slot = 0; slot < count; slot++) {
		const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
		if (inst == nullptr || host[slot] != 0u ||
		    (slot < plan.clean_flat_slots.size() && plan.clean_flat_slots[slot] != 0u)) {
			continue;
		}
		const auto op    = inst->GetOpcode();
		const auto index = inst->Flags<MemoryFlags>().index;
		if ((op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) ||
		    index >= plan.memory_info.size()) {
			continue;
		}
		const auto kind = plan.memory_info[index].kind;
		if (!((op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
		      (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer))) {
			continue;
		}
		fill[slot] = memory_free(*inst) ? 1u : 0u;
	}
	return fill;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;
	plan.bindless_images            = program.bindless_images;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		target.bindless_sampler = source.bindless_sampler;
		if (target.indirect_descriptor.has_value()) {
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// A proven uniform factor can decide a branch even when its other lanes are unknown.
	// Keep only that Boolean structure, never the varying shader dependency graph.
	Value unknown;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) return Clone(value);
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef || op == ValueOpcode::LogicalNot) {
			const auto operand = ClonePredicate(inst->Arg(0));
			if (operand.IsEmpty() || op == ValueOpcode::ConditionRef) return operand;
			auto& node = plan.value_storage.emplace_back(op);
			node.SetArg(0, operand);
			return Value(&node);
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr) return {};
		auto left = ClonePredicate(inst->Arg(0));
		auto right = ClonePredicate(inst->Arg(1));
		if (left.IsEmpty() && right.IsEmpty()) return {};
		if (left.IsEmpty() || right.IsEmpty()) {
			if (unknown.IsEmpty()) unknown = Value(&plan.value_storage.emplace_back(ValueOpcode::UndefU1));
			if (left.IsEmpty()) left = unknown;
			if (right.IsEmpty()) right = unknown;
		}
		auto& node = plan.value_storage.emplace_back(op);
		node.SetArg(0, left);
		node.SetArg(1, right);
		return Value(&node);
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	bool capture_indirect_reads = false;
	for (const auto& source: plan.descriptor_sources) {
		// A runtime image's T# is the shader's own value: the host reads nothing for it.
		if (!source.indirect_descriptor.has_value() || source.indirect_descriptor->runtime)
			continue;
		const auto& indirect                = *source.indirect_descriptor;
		plan.requires_specialization_memory = true;
		capture_indirect_reads |= indirect.material_source != UINT32_MAX || !indirect.sources.empty();
		MarkCleanFlatSlots(plan, Source(plan, indirect.material_source), plan.clean_flat_slots,
		                   indirect.selector_mask);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_first);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.key_count);
		if (indirect.sources.empty()) {
			MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
		} else {
			for (const auto candidate: indirect.sources) {
				MarkCleanFlatSlots(plan, Source(plan, candidate), plan.clean_flat_slots);
			}
		}
	}
	for (const auto& sampler: plan.info.samplers) {
		const auto* source = Source(plan, sampler.source);
		if (source == nullptr || !source->bindless_sampler.has_value()) {
			continue;
		}
		plan.requires_specialization_memory = true;
		MarkCleanFlatSlots(plan, source, plan.clean_flat_slots);
	}
	plan.capture_specialization_reads |= capture_indirect_reads || !plan.control_flow.empty();
	if (plan.capture_specialization_reads) {
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanFlatSlots(plan, Source(plan, buffer.source), plan.clean_flat_slots);
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanFlatSlots(plan, Source(plan, image.source), plan.clean_flat_slots);
		}
	}
	if (capture_indirect_reads) plan.resource_tracking_complete &= !program.has_address_writes;
	// After the decisions above: the clones below must not count as the shader's own reads.
	// Store addresses bound the bytes a dispatch writes: a title's heap-wide V# otherwise marks
	// the whole heap GPU-written, and every CPU access to it drains the GPU.
	const auto buffer_count = program.info.buffers.size();
	plan.buffer_writes_bounded.assign(buffer_count, 1u);
	std::vector<uint32_t> listed_writes(buffer_count, 0u);
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (program.info.buffers[i].image_alias != BufferResource::NoImageAlias) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) {
				continue;
			}
			const auto flags = inst.Flags<MemoryFlags>();
			if (flags.index >= program.memory_info.size()) {
				plan.buffer_writes_bounded.assign(buffer_count, 0u);
				continue;
			}
			const auto& memory = program.memory_info[flags.index];
			if (memory.kind != ResourceKind::Buffer || memory.resource >= buffer_count) {
				continue;
			}
			const auto buffer = memory.resource;
			if (plan.buffer_writes_bounded[buffer] == 0u) {
				continue;
			}
			const auto predicate = inst.Arg(inst.NumArgs() - 1u).Resolve();
			if (!BoundableWriteOperand(program, inst.Arg(1), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(2), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(3), predicate, 0u)) {
				plan.buffer_writes_bounded[buffer] = 0u;
				continue;
			}
			listed_writes[buffer]++;
			plan.buffer_writes.push_back({.buffer    = buffer,
			                              .immediate = memory.offset,
			                              .index     = Clone(inst.Arg(1)),
			                              .offset    = Clone(inst.Arg(2)),
			                              .soffset   = Clone(inst.Arg(3)),
			                              .predicate = Clone(predicate)});
		}
	}
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (listed_writes[i] == 0u) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	plan.gpu_fill_slots = GpuFillSlots(plan);
	return plan;
}

namespace {

struct WriteInterval {
	uint64_t lo = 0;
	uint64_t hi = 0;
};

// The unsigned 32-bit range a store address operand takes over the dispatch: invocation IDs range
// over the workgroup count and size, everything else the host evaluates exactly. Fails on an
// unknown dispatch size or a range that could wrap.
bool EvaluateWriteInterval(const ResourcePlan& program, const SrtRuntime& runtime, SrtWalker& walker,
                           Value value, Value predicate, WriteInterval& out, uint32_t& budget) {
	constexpr uint64_t Max32 = UINT32_MAX;
	if (budget == 0u) {
		return false;
	}
	--budget;
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return false;
		}
		out = {value.U32(), value.U32()};
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		const auto  axis  = inst->Arg(1).U32();
		if (runtime.workgroup_counts.size() < 3u) {
			return false;
		}
		const auto& count = runtime.workgroup_counts;
		const auto& size  = runtime.workgroup_size;
		uint64_t    total = 0;
		switch (static_cast<StageInputKind>(inst->Arg(0).U32())) {
			case StageInputKind::WorkgroupId: total = count[axis]; break;
			case StageInputKind::LocalInvocationId: total = size[axis]; break;
			case StageInputKind::LocalInvocationIndex:
				total = uint64_t {size[0]} * size[1] * size[2];
				break;
			case StageInputKind::GlobalInvocationId:
				total = uint64_t {count[axis]} * size[axis];
				break;
			default: return false;
		}
		if (total == 0u || total - 1u > Max32) {
			return false;
		}
		out = {0u, total - 1u};
		return true;
	}
	const auto op = inst->GetOpcode();
	if (!IsIntervalOperation(op)) {
		uint32_t exact = 0;
		if (!walker.Evaluate(value, exact)) {
			return false;
		}
		out = {exact, exact};
		return true;
	}
	if (op == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
		return EvaluateWriteInterval(program, runtime, walker, inst->Arg(1), predicate, out, budget);
	}
	std::array<WriteInterval, 3> args {};
	const size_t                 first = op == ValueOpcode::SelectU32 ? 1u : 0u;
	if (inst->NumArgs() > args.size()) {
		return false;
	}
	for (size_t i = first; i < inst->NumArgs(); ++i) {
		if (!EvaluateWriteInterval(program, runtime, walker, inst->Arg(i), predicate, args[i],
		                           budget)) {
			return false;
		}
	}
	const auto& a = args[0];
	const auto& b = args[1];
	switch (op) {
		case ValueOpcode::IAdd32: out = {a.lo + b.lo, a.hi + b.hi}; break;
		case ValueOpcode::IMul32: out = {a.lo * b.lo, a.hi * b.hi}; break;
		case ValueOpcode::ShiftLeftLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo << b.lo, a.hi << b.hi};
			break;
		case ValueOpcode::ShiftRightLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo >> b.hi, a.hi >> b.lo};
			break;
		case ValueOpcode::BitwiseAnd32: out = {0u, std::min(a.hi, b.hi)}; break;
		case ValueOpcode::BitwiseOr32: {
			const auto high = std::max(a.hi, b.hi);
			out = {std::max(a.lo, b.lo), (uint64_t {1} << std::bit_width(high)) - 1u};
			break;
		}
		case ValueOpcode::UMin32: out = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)}; break;
		case ValueOpcode::UMax32: out = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
		case ValueOpcode::BitFieldUExtract: {
			// (base >> offset) masked to count bits: at most both.
			if (b.hi > 31u) {
				return false;
			}
			const auto& count = args[2];
			const auto  mask  = count.hi >= 32u ? Max32 : (uint64_t {1} << count.hi) - 1u;
			out               = {0u, std::min(mask, a.hi >> b.lo)};
			break;
		}
		case ValueOpcode::SelectU32:
			out = {std::min(args[1].lo, args[2].lo), std::max(args[1].hi, args[2].hi)};
			break;
		default: return false;
	}
	return out.hi <= Max32;
}

// See ResourceSnapshot::buffer_write_extents.
void ComputeBufferWriteExtents(const ResourcePlan& program, const SrtRuntime& runtime,
                               SrtWalker& walker, ResourceSnapshot& snapshot) {
	snapshot.buffer_write_extents.assign(program.info.buffers.size(), {});
	if (program.buffer_writes.empty()) {
		return;
	}
	std::vector<uint8_t> failed(program.info.buffers.size(), 0u);
	for (const auto& write: program.buffer_writes) {
		if (write.buffer >= failed.size() || failed[write.buffer] != 0u) {
			continue;
		}
		auto&                extent = snapshot.buffer_write_extents[write.buffer];
		ShaderBufferResource descriptor;
		WriteInterval        index;
		WriteInterval        offset;
		WriteInterval        soffset;
		uint32_t             budget    = 1024u;
		const auto           predicate = write.predicate.Resolve();
		if (program.buffer_writes_bounded[write.buffer] == 0u ||
		    !DecodeBufferDescriptor(snapshot.buffers[write.buffer], descriptor) ||
		    descriptor.SwizzleEnabled() ||
		    !EvaluateWriteInterval(program, runtime, walker, write.index, predicate, index, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.offset, predicate, offset, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.soffset, predicate, soffset,
		                           budget)) {
			failed[write.buffer] = 1u;
			extent               = {};
			continue;
		}
		if (descriptor.AddTid()) {
			index.hi += 63u;
		}
		// PS5 ISA, buffer addressing: base + soffset + offset + index * stride, at most 16 bytes a
		// store (dwordx4, format xyzw), and a store outside the descriptor's range is dropped.
		const uint64_t stride = descriptor.Stride();
		const uint64_t size   = descriptor.GetSize();
		const uint64_t begin =
		    std::min(index.lo * stride + offset.lo + soffset.lo + write.immediate, size);
		const uint64_t end =
		    std::min(index.hi * stride + offset.hi + soffset.hi + write.immediate + 16u, size);
		if (!extent.valid) {
			extent = {.begin = begin, .end = end, .valid = true};
		} else {
			extent.begin = std::min(extent.begin, begin);
			extent.end   = std::max(extent.end, end);
		}
	}
}

} // namespace

int LastIndirectImageFailureLine() {
	return g_indirect_failure_line;
}

namespace {

bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                              ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                              bool& served) {
	served                  = false;
	g_indirect_failure_line = 0;
	snapshot.bindless_heaps.clear();
	snapshot.bindless_sampler_heaps.clear();
	if (!program.resource_tracking_complete ||
	    (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr)) {
		return FailIndirect(__LINE__);
	}
	const bool capture_reads = program.capture_specialization_reads;
	auto& reads = snapshot.specialization_reads;
	reads.clear();
	ReadCapture capture {runtime, reads};
	SrtRuntime observed = runtime;
	if (capture_reads) {
		observed.userdata = &capture;
		observed.read_specialization_memory = runtime.read_specialization_memory != nullptr
		                                         ? CaptureStrictRead : nullptr;
		observed.read_memory = CaptureOrdinaryRead;
		// Reads through the fast page path are captured there.
		observed.capture_ranges = &reads;
	}
	SrtWalker clean(program, CleanRuntime(observed));
	SrtWalker walker(program, observed, program.clean_flat_slots,
	                 capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	SrtTraceSession trace(program, clean, walker, observed);
	if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		return FailIndirect(__LINE__);
	}
	const auto active = std::span<const uint8_t>(program.active_sources);
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	const auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					if (!MaterializeIndirectDescriptor(
					        program, *source->indirect_descriptor, i, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers, ShaderInfo::MaxBuffers,
					        NormalizeIndirectStoreBuffer))
						return false;
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		buffer.write_tracked      = WriteBitmapEnabled() && i < program.info.buffers.size() &&
		                            base.written && base.image_alias == BufferResource::NoImageAlias &&
		                            i < program.buffer_writes_bounded.size() &&
		                            program.buffer_writes_bounded[i] == 0u;
	}
	{
		// Write extents only size the GPU-written range; their reads do not specialize the
		// shader, so they stay out of the specialization-read proof.
		const auto captured = reads.size();
		ComputeBufferWriteExtents(program, observed, walker, snapshot);
		if (capture_reads) {
			reads.resize(captured);
		}
	}
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class              = image.numeric_class,
		    .dimension                  = image.dimension,
		    .mip_count                  = image.mip_count,
		    .conversion_format          = image.conversion_format,
		    .shader_swizzle             = image.shader_swizzle,
		    .indirect_root              = image.indirect_root,
		    .indirect_mapping_offset    = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube                       = image.cube,
		    .bindless                   = image.bindless,
		    .runtime                    = image.runtime,
		    .clamp_half_float           = image.clamp_half_float,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return FailIndirect(__LINE__);
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			if (indirect.runtime) {
				// Bound to set 1 like a bindless image; set 0 sees a null descriptor.
				specialization.images[i].bindless      = true;
				specialization.images[i].runtime       = true;
				specialization.images[i].indirect_root = ImageResource::NoIndirectImage;
				continue;
			}
			if (indirect.bindless) {
				DescriptorValue table_value;
				if (!clean.EvaluateDescriptor(indirect.table_source, table_value) ||
				    !MaterializeBindlessImage(indirect, table_value, i, snapshot, specialization)) {
					return FailIndirect(__LINE__);
				}
				continue;
			}
			if (!MaterializeIndirectDescriptor(
			        program, indirect, i, 8u, observed, clean, snapshot, snapshot.images,
			        specialization.images, UINT32_MAX, [&](DescriptorValue& value) {
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return FailIndirect(__LINE__);
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return FailIndirect(__LINE__);
			}
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		const auto  source_index = program.info.samplers[i].source;
		const auto* source       = Source(program, source_index);
		if (source != nullptr && source->bindless_sampler.has_value()) {
			// The shader indexes the host's mirror of the heap with two patched words (region
			// base, entry count); zeros select the default sampler in slot 0. Set 0 still binds
			// the default sampler for this resource.
			const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			snapshot.samplers[i].dword_count = 4u;
			std::ranges::copy(BindlessDefaultSampler, snapshot.samplers[i].dwords.begin());
			specialization.samplers[i] = {.bindless = true, .bindless_mapping_offset = mapping_offset};
			DescriptorValue      table;
			ShaderBufferResource heap;
			if ((active.empty() || active[source_index]) &&
			    clean.EvaluateDescriptor(source_index, table) &&
			    DecodeBufferDescriptor(table, heap) && heap.Base48() != 0) {
				snapshot.bindless_sampler_heaps.push_back(
				    {.base           = heap.Base48(),
				     .size           = heap.GetSize(),
				     .table_offset   = source->bindless_sampler->table_offset,
				     .sampler        = i,
				     .mapping_offset = mapping_offset});
			}
			continue;
		}
		if (!evaluate(source_index, snapshot.samplers[i])) {
			return FailIndirect(__LINE__);
		}
		if (program.info.samplers[i].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			// MipNone always selects the base level. Explicit point gathers currently require
			// encoded-zero primary and secondary bias. A gather returns the texels of one level,
			// so linear mip filtering gathers the nearest level, as point does (GatherMip rounds):
			// the game's HiZ compute shader 0x37ab7d3ab9bff0b9 samples linearly, and rejecting it
			// skipped its dispatches every frame (the jungle floor went missing).
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires mip filtering None or Point with zero LOD biases");
			}
		}
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	if (!BuildResourceSpecialization(program, snapshot, specialization,
	                                 runtime.float_image_atomics)) {
		return FailIndirect(__LINE__);
	}
	served = trace.FullyServed();
	trace.Succeeded();
	return true;
}

// KYTY_SRT_TRACE_VERIFY=1: a refresh served from a replay trace is evaluated again without it,
// and any difference stops the emulator.
void VerifyTracedRefresh(const ResourcePlan& program, const SrtRuntime& runtime,
                         const ResourceSnapshot& snapshot,
                         const ResourceSpecialization& specialization) {
	ResourceSnapshot       reference_snapshot;
	ResourceSpecialization reference_specialization;
	bool                   served = false;
	auto&                  suppressed = SrtTraceSession::Suppressed();
	suppressed                        = true;
	// The reference lists its GPU-filled slots apart.
	SrtRuntime reference_runtime = runtime;
	if (reference_runtime.gpu_fills != nullptr) {
		reference_runtime.gpu_fills = &reference_snapshot.gpu_fills;
	}
	const bool ok = MaterializeResourcesImpl(program, reference_runtime, reference_snapshot,
	                                         reference_specialization, served);
	suppressed    = false;
	// A GPU-filled slot's host value is a placeholder, and whether its bytes were GPU-written can
	// change between the two walks (a download in between): compare the other slots.
	auto flat           = snapshot.flattened_srt;
	auto reference_flat = reference_snapshot.flattened_srt;
	for (const std::vector<SrtGpuFill>* fills: {&snapshot.gpu_fills, static_cast<const std::vector<SrtGpuFill>*>(&reference_snapshot.gpu_fills)}) {
		for (const auto& fill: *fills) {
			if (fill.flat_offset < flat.size()) flat[fill.flat_offset] = 0u;
			if (fill.flat_offset < reference_flat.size()) reference_flat[fill.flat_offset] = 0u;
		}
	}
	auto reads           = snapshot.specialization_reads;
	auto reference_reads = reference_snapshot.specialization_reads;
	std::ranges::sort(reads);
	reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
	std::ranges::sort(reference_reads);
	reference_reads.erase(std::unique(reference_reads.begin(), reference_reads.end()),
	                      reference_reads.end());
	const char* field = nullptr;
	if (!ok) field = "reference refresh failed";
	else if (flat != reference_flat) field = "flattened_srt";
	else if (snapshot.buffers != reference_snapshot.buffers) field = "buffers";
	else if (snapshot.images != reference_snapshot.images) field = "images";
	else if (snapshot.samplers != reference_snapshot.samplers) field = "samplers";
	else if (snapshot.user_data != reference_snapshot.user_data) field = "user_data";
	else if (snapshot.buffer_write_extents != reference_snapshot.buffer_write_extents) field = "buffer_write_extents";
	else if (snapshot.bindless_heaps != reference_snapshot.bindless_heaps) field = "bindless_heaps";
	else if (snapshot.bindless_sampler_heaps != reference_snapshot.bindless_sampler_heaps) field = "bindless_sampler_heaps";
	else if (!(snapshot.uniform_fill == reference_snapshot.uniform_fill)) field = "uniform_fill";
	else if (reads != reference_reads) field = "specialization_reads";
	else if (!(specialization == reference_specialization)) field = "specialization";
	if (field != nullptr) {
		std::string detail;
		if (std::strcmp(field, "flattened_srt") == 0) {
			const auto& a = snapshot.flattened_srt;
			const auto& b = reference_snapshot.flattened_srt;
			for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
				const auto x = i < a.size() ? a[i] : 0xdeadbeefu;
				const auto y = i < b.size() ? b[i] : 0xdeadbeefu;
				if (x != y) {
					detail += fmt::format(" [{}] {:08x}/{:08x}", i, x, y);
				}
			}
			detail += fmt::format(" fills={}", snapshot.gpu_fills.size());
			for (const auto& fill: snapshot.gpu_fills) {
				detail += fmt::format(" {}@{:x}", fill.flat_offset, fill.address);
			}
		}
		EXIT("SRT trace replay differs from the walk: shader %016" PRIx64 " stage %u: %s%s\n",
		     program.shader_hash, static_cast<uint32_t>(program.stage), field, detail.c_str());
	}
	static std::atomic<uint64_t> verified {0};
	if (verified.fetch_add(1, std::memory_order_relaxed) % 100000 == 0) {
		std::fprintf(stderr, "SRT trace replay verified: %llu\n",
		             static_cast<unsigned long long>(verified.load()));
	}
}

} // namespace

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	bool       served = false;
	const bool ok     = MaterializeResourcesImpl(program, runtime, snapshot, specialization, served);
	static const bool verify = std::getenv("KYTY_SRT_TRACE_VERIFY") != nullptr;
	if (ok && served && verify) {
		VerifyTracedRefresh(program, runtime, snapshot, specialization);
	}
	return ok;
}

const char* UnsupportedResourceSpecialization(const Program&                program,
                                              const ResourceSpecialization& specialization) {
	// ApplyResourceSpecialization's check, before it changes anything: enumerated candidates only
	// sample (EmitImageSample), and an image load or query through one stopped the emulator in
	// GT7's cafe. The shader is skipped for this specialization instead.
	// Fewer resources than the program tracks (ApplyResourceSpecialization's EXIT_IF): a seed whose
	// specialization does not fit this translation stopped the precompile with a fatal error
	// (gt7-precompilar-20261112-054529: 31,367 programs, nothing saved); it fails alone now.
	if (program.info.buffers.size() > specialization.buffers.size() ||
	    program.info.images.size() > specialization.images.size()) {
		return "the specialization holds fewer resources than the program tracks";
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (ImageOpcodeInfoOf(op).access == ImageAccess::None ||
			    op == ValueOpcode::ImageSampleRaw) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				continue;
			}
			const auto resource = program.memory_info[index].resource;
			if (resource >= specialization.images.size()) {
				continue;
			}
			const auto& image = specialization.images[resource];
			if (!image.fmask && !image.bindless && image.indirect_root == resource) {
				return "an image of an enumerated material table is read other than by a sample";
			}
		}
	}
	return nullptr;
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.write_tracked              = source.write_tracked;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.bindless                   = source.bindless;
		image.runtime                    = source.runtime;
		image.clamp_half_float           = source.clamp_half_float;
		image.cube                       = source.cube;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto original_sampler_count = samplers.size();
	for (uint32_t index = 0; index < original_sampler_count && index < specialization.samplers.size();
	     index++) {
		samplers[index].bindless                = specialization.samplers[index].bindless;
		samplers[index].bindless_mapping_offset = specialization.samplers[index].bindless_mapping_offset;
	}
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						for (uint32_t component = 0; component < count; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							const auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(one), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// The host image holds one sample per guest fragment. FMASK's four-bit fragment
				// indices map coverage sample i to host sample i, or with EQAA (fewer fragments
				// than samples) to i mod fragments, so every index names a sample that exists.
				const uint32_t fragments_per_pixel = specialization.images[memory.resource].fmask_fragments;
				const uint32_t indices[] = {FmaskSampleMapWord(fragments_per_pixel, 0),
				                            FmaskSampleMapWord(fragments_per_pixel, 8)};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// Enumerated candidates only sample; a bindless image is loaded per instruction and
			// serves every image operation.
			EXIT_IF(image.indirect_root == memory.resource && !image.bindless &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw);
		}
	}
	// Tracking gives an image index only to the accesses that survived its dead-code pass; an
	// access it removed keeps its T# register from translation, which need not be an image index.
	std::vector<bool> live_image_access(memory_info.size());
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (ImageOpcodeInfoOf(inst.GetOpcode()).access != ImageAccess::None) {
				live_image_access[inst.Flags<MemoryFlags>().index] = true;
			}
		}
	}
	for (uint32_t index = 0; index < memory_info.size(); index++) {
		auto& memory = memory_info[index];
		if (live_image_access[index] && memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap(memory.resource, "image access");
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap(buffer.image_alias, "buffer image alias");
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap(pair.image, "sampled pair");
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap(image.indirect_root, "indirect root");
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap(resource, "indirect resource");
		}
	}
	image_remap.Apply(images);
}

// KYTY_WRITE_BITMAP=1: buffers written at addresses the host cannot bound record the pages they
// write in a GPU bitmap, so a guest read of a page the shader did not write needs no download
// (rework.md, Phase 2 first step; compute shader eaa507075716aa3b adds into a 70 MiB range and
// caused half of all readbacks in a new game).
bool WriteBitmapEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool FloatStoreClampEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FLOAT_STORE_CLAMP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
