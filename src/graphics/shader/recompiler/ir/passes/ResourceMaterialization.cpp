#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ImageForms.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

// A storage image picked from a whole image table by key has a candidate per table entry, and a draw read, interned and
// grouped all of them although the table had not changed since the plan's previous draw. In Marvel's Wolverine's first
// fight (~12,000-entry tables) that was most of the GPU thread's time, and 99.4 % of those tables repeated. Each root
// keeps its last table and what it made: the same words make the same candidates, and the same candidates the same
// groups (BuildResourceSpecialization depends on nothing else), so the root's results are appended instead.
struct IndirectImageReuse {
	struct Root {
		bool                         valid  = false;
		uint64_t                     stored = 0;   // the call that stored it
		std::array<uint64_t, 5>      geometry {};  // table base, size, step, first offset, count
		ResourceSpecialization::Image source;      // the root before materialization
		ImageResource                image;        // the plan's root and table, which normalize the candidates
		DescriptorSource::IndirectDescriptor indirect;
		std::vector<uint32_t>        words;        // the table as read
		std::vector<DescriptorValue> children;     // the candidates it appended
		std::vector<uint32_t>        mapping;      // its flattened_srt words, from the mapping offset on
		// Set once its candidates have gone into groups (GroupIndirectImages): mapping is then the remapped words,
		// root the root as BuildResourceSpecialization left it, and the groups what replaced the candidates.
		bool                                       grouped = false;
		ResourceSpecialization::Image              root;
		std::vector<ResourceSpecialization::Image> group_images;
		std::vector<std::vector<DescriptorValue>>  group_values;
	};
	std::vector<Root>     roots; // by image resource
	std::vector<uint32_t> whole; // this call's roots appended grouped: no candidates of their own
	uint64_t              call    = 0;
	bool                  enabled = true;
};

void IndirectImageReuseDelete::operator()(IndirectImageReuse* reuse) const {
	delete reuse;
}

namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
// A sampled image read from a table of more descriptors than this is read from the image heap.
constexpr uint64_t HeapTableMinDescriptors = 256u;

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
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool NullBufferDescriptor(Value value) {
	const auto* handle = value.Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
	    handle->NumArgs() != 4u) return false;
	for (size_t index = 0; index < 4u; ++index)
		if (handle->Arg(index).Resolve() != Value(0u)) return false;
	return true;
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

bool ValidatePackedUnormSampling(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                                 uint32_t image_index) {
	ShaderTextureResource texture;
	std::copy_n(snapshot.images[image_index].dwords.begin(), std::size(texture.fields),
	            texture.fields);
	if (texture.Type() != Prospero::ImageType::kColor2D ||
	    texture.BaseLevel() != texture.LastLevel() || texture.MinLod() != 0u ||
	    texture.CornerSample()) {
		return SpecializationFail("packed UNorm sampling requires a centered, single-mip 2D image");
	}
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.image != image_index) continue;
		if (pair.sampler >= snapshot.samplers.size() ||
		    program.descriptor_sources[program.info.samplers[pair.sampler].source].indirect_descriptor) {
			return SpecializationFail("packed UNorm sampling requires a direct sampler");
		}
		ShaderSamplerResource sampler;
		std::copy_n(snapshot.samplers[pair.sampler].dwords.begin(), std::size(sampler.fields),
		            sampler.fields);
		const auto linear = static_cast<uint32_t>(Prospero::SamplerFilter::kBilinear);
		const auto clamp = static_cast<uint32_t>(Prospero::SamplerClampMode::kClampLastTexel);
		if (sampler.XyMagFilter() != linear || sampler.XyMinFilter() != linear ||
		    sampler.ClampX() != clamp || sampler.ClampY() != clamp ||
		    sampler.ForceUnormCoords() || sampler.MaxAnisoRatio() != 0u ||
		    sampler.FilterMode() != 0u || sampler.MinLod() != 0u || sampler.ForceSrgb() ||
		    sampler.TruncCoord() || (sampler.fields[0] & (1u << 19u)) != 0u) { // MC_COORD_TRUNC
			return SpecializationFail(fmt::format(
			    "packed UNorm image {} requires normalized bilinear clamp sampling (sampler {})",
			    image_index, pair.sampler));
		}
	}
	return true;
}

enum class SamplerClass : uint8_t { Float, Integer };

SamplerClass ClassifySampler(const ImageResource& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Uint ||
	               image.numeric_class == Prospero::TextureNumericClass::Sint
	           ? SamplerClass::Integer : SamplerClass::Float;
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

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanReads(const std::unordered_set<const Inst*>& planned_reads,
                    const DescriptorSource* source, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value> pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (planned_reads.contains(inst)) {
			auto flags = inst->Flags<SrtReadFlags>();
			flags.clean = 1u;
			inst->SetFlags(flags);
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadTable(uint64_t base, uint64_t size, uint64_t offset,
               const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	if (base > AddressMask || offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	return ReadTable(base & AddressMask & ~uint64_t {3}, size, offset & ~uint64_t {3}, runtime, words);
}

} // namespace

bool NormalizeIndirectBuffer(DescriptorValue& value, bool indexed, const BufferResource& resource) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	const bool formatted_read = resource.formatted && resource.read && !resource.written;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    (!formatted_read && (descriptor.Base48() == 0u || descriptor.GetSize() == 0u))) {
		value.dwords.fill(0u);
		return true;
	}
	// An indexed raw (OOB_SELECT 3) record of stride 1 is a byte view: its index is the byte offset, so its raw byte
	// bound over NumRecords is the index bound the indexed path checks (Marvel's Wolverine's Telambang compound).
	const bool byte_view = indexed && descriptor.OutOfBounds() == 3u && descriptor.Stride() == 1u;
	if (descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	    (descriptor.OutOfBounds() != (indexed ? 0u : 3u) && !byte_view)) {
		return SpecializationFail(fmt::format(
		    "indirect buffers require exact linear index or byte bounds: "
		    "descriptor={:08x},{:08x},{:08x},{:08x} swizzle={} add_tid={} oob={} "
		    "indexed={} formatted={} read={} written={} max_extent={}",
		    value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		    descriptor.SwizzleEnabled(), descriptor.AddTid(), descriptor.OutOfBounds(), indexed,
		    resource.formatted, resource.read, resource.written, resource.max_byte_extent));
	}
	if (resource.formatted) {
		const auto format = Format::GetFormatInfo(descriptor.Format());
		if (!indexed || format.type == Format::ComponentType::Unknown)
			return SpecializationFail("indirect formatted buffers require a supported indexed format");
		if (!formatted_read) {
			const bool scalar = resource.max_byte_extent == 4u &&
			    (descriptor.Format() == Prospero::BufferFormat::k32UInt ||
			     descriptor.Format() == Prospero::BufferFormat::k8SInt);
			const bool rgba16 = resource.max_byte_extent == 16u &&
			    descriptor.Format() == Prospero::BufferFormat::k16_16_16_16Float;
			if ((!scalar && !rgba16) || descriptor.Stride() != format.byte_size ||
			    descriptor.DstSelXYZW() != (rgba16 ? DstSel(4, 5, 6, 7) : DstSel(4, 0, 0, 0)))
				return SpecializationFail("indirect formatted stores require tight identity SInt8, UInt32 or RGBA16Float records");
		}
		if (formatted_read &&
		    (descriptor.Base48() == 0u || descriptor.Stride() == 0u || descriptor.NumRecords() == 0u)) {
			// Every index is out of bounds. Preserve the format and swizzle constants.
			value.dwords[1] &= ~(0x3fffu << 16u);
			value.dwords[2] = 0u;
			return true;
		}
		const auto alignment = std::min(4u, format.byte_size);
		if (descriptor.Stride() < format.byte_size ||
		    (descriptor.Stride() & (alignment - 1u)) != 0u ||
		    (descriptor.Base48() & (alignment - 1u)) != 0u)
			return SpecializationFail("indirect formatted buffers require aligned full-format records");
	} else if ((!indexed && descriptor.Stride() != 1u) ||
	           (indexed && resource.max_byte_extent != 1u && descriptor.Stride() != 4u)) {
		return SpecializationFail("indirect buffer stores require exact linear DWORD or byte bounds");
	}
	return true;
}

namespace {

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	return indirect.sources.empty() && (!indirect.selector || indirect.table_scalar) &&
	       (indirect.key_count.IsEmpty() || indirect.table_maximum) && indirect.workgroup_axis == UINT32_MAX &&
	       table != nullptr && table->dword_count == 4u;
}

bool CanReadIndexedSelector(const ShaderBufferResource& material, uint32_t offset) {
	if (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
	    material.OutOfBounds() != 0u) return false;
	if (material.Format() == Prospero::BufferFormat::kInvalid || material.NumRecords() == 0u ||
	    offset >= material.Stride()) return true;
	return material.Base48() != 0u &&
	       ((material.Base48() | material.Stride() | offset) & 3u) == 0u &&
	       material.NumRecords() <= MaxIndirectDescriptorProbes &&
	       uint64_t {material.NumRecords() - 1u} * material.Stride() + offset + 4u <=
	           uint64_t {UINT32_MAX} + 1u;
}

bool EvaluateIndirectCount(const DescriptorSource::IndirectDescriptor& indirect,
                           const SrtRuntime& runtime, SrtWalker& clean, uint32_t& count) {
	if (indirect.workgroup_axis != UINT32_MAX) {
		if (runtime.workgroup_counts.size() != 3u || indirect.workgroup_axis >= 3u) return false;
		count = std::ranges::find(runtime.workgroup_counts, 0u) != runtime.workgroup_counts.end()
		    ? 0u : runtime.workgroup_counts[indirect.workgroup_axis];
		return true;
	}
	if (!clean.Evaluate(indirect.key_count, count)) return false;
	if (indirect.signed_count && std::bit_cast<int32_t>(count) < 0) count = 0;
	return true;
}

bool ResolveSelectorKeys(const DescriptorSource::IndirectDescriptor& indirect,
                         const DescriptorValue& material_value, const SrtRuntime& runtime,
                         SrtWalker& clean, std::vector<uint32_t>& keys) {
	const auto* selector = &*indirect.selector;
	keys.clear();
	const auto read_selector = [&](uint64_t base, uint64_t size, uint64_t offset,
	                               std::span<uint32_t> words) {
		return selector->global ? ReadTable(base, size, offset, runtime, words)
		                        : ReadScalarTable(base, size, offset, runtime, words);
	};
	const auto read_keys = [&](uint64_t base, uint64_t size, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (count == 0u) return true;
		if (step == 4u) {
			return read_selector(base, size, first, keys);
		}
		if (step < 4u) return false;
		for (uint64_t copied = 0; copied < count;) {
			const auto offset = first + copied * step;
			// Scalar reads align each strided offset independently.
			const auto bias = selector->global ? 0u : offset & 3u;
			const auto length = std::min(count - copied,
			    1u + (uint64_t {MaxIndirectDescriptorProbes - 1u} * 4u - bias) / step);
			keys.resize(count + (bias + (length - 1u) * step) / 4u + 1u);
			const auto batch = std::span(keys).subspan(count);
			if (!read_selector(base, size, offset, batch)) {
				keys.resize(count);
				return false;
			}
			for (uint64_t index = 0; index < length; ++index)
				keys[copied + index] = batch[(bias + index * step) / 4u];
			copied += length;
		}
		keys.resize(count);
		return true;
	};
	// Reads the key of each record whose rank is set.
	const auto read_ranks = [&](uint64_t base, uint64_t size, uint64_t first, uint64_t step,
	                            uint32_t ranks) {
		keys.reserve(std::popcount(ranks));
		for (; ranks != 0u; ranks &= ranks - 1u) {
			const auto offset = first + std::countr_zero(ranks) * step;
			uint32_t key = 0;
			if (offset > UINT32_MAX || !read_selector(base, size, offset, {&key, 1})) return false;
			keys.push_back(key);
		}
		return true;
	};
	if (selector->rows) {
		const auto& rows = *selector->rows;
		DescriptorValue count_value;
		ShaderBufferResource counts, material;
		if (!clean.EvaluateDescriptor(rows.source, count_value) ||
		    !DecodeBufferDescriptor(count_value, counts) ||
		    !DecodeBufferDescriptor(material_value, material) || selector->stride != 4u ||
		    selector->offset != 0u || rows.stride == 0u || (rows.stride & 3u) != 0u ||
		    counts.GetSize() / 4u > MaxIndirectDescriptorProbes) return false;
		std::vector<uint32_t> lengths(counts.GetSize() / 4u);
		if (!ReadScalarTable(counts.Base48(), counts.GetSize(), 0u, runtime, lengths)) return false;
		uint32_t total = 0;
		for (size_t row = 0; row < lengths.size(); ++row) {
			const auto length = lengths[row];
			if (length > MaxIndirectDescriptorProbes - total ||
			    (length != 0u && uint64_t {row} * rows.stride + uint64_t {length} * 4u >
			                         UINT32_MAX + 1ull)) return false;
			total += length;
		}
		keys.reserve(total);
		std::vector<uint32_t> batch;
		for (size_t row = 0; row < lengths.size();) {
			if (lengths[row] == 0u) { ++row; continue; }
			const auto first = uint64_t {row} * rows.stride / 4u;
			auto end = first + lengths[row];
			size_t next = row + 1u;
			for (; next < lengths.size(); ++next) {
				if (lengths[next] == 0u) continue;
				const auto candidate_end =
				    std::max(end, uint64_t {next} * rows.stride / 4u + lengths[next]);
				if (candidate_end - first > MaxIndirectDescriptorProbes) break;
				end = candidate_end;
			}
			// Coalesce small gaps, but interpret only selected prefixes. The strict
			// reader captures the entire fetched span for writable-alias rejection.
			batch.resize(end - first);
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first * 4u, runtime, batch))
				return false;
			for (; row < next; ++row) {
				if (lengths[row] == 0u) continue;
				const auto begin = batch.begin() + (uint64_t {row} * rows.stride / 4u - first);
				keys.insert(keys.end(), begin, begin + lengths[row]);
			}
		}
	} else if (selector->bitset) {
		ShaderBufferResource material;
		std::array<uint32_t, 8> storage {};
		const auto record_words = std::min<size_t>(selector->stride / 4u, storage.size());
		const auto masks = std::span(storage).first(record_words);
		uint32_t first = 0, count = 0;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count != masks.size() * 32u ||
		    selector->stride != masks.size_bytes() || selector->offset != 0u ||
		    material.Type() != 0u || material.Stride() != selector->stride ||
		    material.SwizzleEnabled() || material.AddTid() || material.OutOfBounds() != 0u ||
		    (material.Base48() & 3u) != 0u) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid) {
			const auto words = material.GetSize() / sizeof(uint32_t);
			// Bitmap storage can grow with resolution while its candidate domain stays at 32 keys
			// per record word. Bound scratch memory, preserving each coherent read's capture.
			for (uint64_t first_word = 0; first_word < words; first_word += keys.size()) {
				keys.resize(std::min<uint64_t>(words - first_word, MaxIndirectDescriptorProbes));
				if (!ReadScalarTable(material.Base48(), material.GetSize(), first_word * 4u,
				                     runtime, keys)) return false;
				for (size_t word = 0; word < keys.size(); ++word) masks[word % masks.size()] |= keys[word];
				if (std::ranges::all_of(masks, [](uint32_t mask) { return mask == UINT32_MAX; })) break;
			}
			keys.clear();
			for (uint32_t word = 0; word < masks.size(); ++word) {
				auto mask = masks[word];
				while (mask != 0u) {
					const auto rank = word * 32u + std::countr_zero(mask);
					keys.push_back(first + count - 1u - rank);
					mask &= mask - 1u;
				}
			}
		}
	} else if (!selector->byte_ranges.empty()) {
		if (material_value.dword_count != 2u) return false;
		const auto base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		std::array<bool, 256> selected {};
		for (const auto [offset, size]: selector->byte_ranges) {
			if (size > MaxIndirectDescriptorProbes || ((offset | size) & 3u) != 0u) return false;
			keys.resize(size / 4u);
			if (!ReadScalarTable(base, UINT64_MAX, offset, runtime, keys)) return false;
			for (const auto word: keys)
				for (uint32_t shift = 0; shift < 32u; shift += 8u)
					selected[(word >> shift) & 255u] = true;
		}
		keys.clear();
		for (uint32_t key = 0; key < selected.size(); ++key)
			if (selected[key]) keys.push_back(key);
	} else if (!indirect.selector_first.IsEmpty()) {
		ShaderBufferResource material;
		uint64_t             base = 0, size = UINT64_MAX;
		uint32_t             first = 0, count = 0;
		if (material_value.dword_count == 2u) {
			if (selector->stride == 0u) return false;
			base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		} else {
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			base = material.Base48();
			size = material.GetSize();
		}
		if (!clean.Evaluate(indirect.selector_first, first) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count > MaxIndirectDescriptorProbes)
			return false;
		// SMEM uses its instruction's byte stride; V# stride only supplies its bounds.
		const auto step = selector->stride != 0u ? selector->stride : material.Stride();
		if (selector->stride == 0u &&
		    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
		     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
		     uint64_t {first} + count > material.NumRecords())) return false;
		const auto last = count != 0u ? uint64_t {first} + count - 1u : 0u;
		if (last > UINT32_MAX) return false;
		// Out-of-bounds scalar reads return zero.
		const auto end = count != 0u ? last * step + selector->offset + 4u : 0u;
		if (end > uint64_t {UINT32_MAX} + 1u) return false;
		const auto start = uint64_t {first} * step + selector->offset;
		if (selector->ranks) {
			// A scan selects the ranks some word of its mask's address class sets.
			const auto& mask = *selector->ranks;
			DescriptorValue mask_value;
			ShaderBufferResource masks;
			if (count > 32u || !clean.EvaluateDescriptor(mask.source, mask_value) ||
			    !DecodeBufferDescriptor(mask_value, masks)) return false;
			// Each chunk holds whole strides of the power-of-two class, or one of its words.
			const auto words = mask.stride / 4u;
			const auto chunk =
			    words <= MaxIndirectDescriptorProbes ? MaxIndirectDescriptorProbes : 1u;
			uint32_t ranks = 0;
			for (auto at = uint64_t {mask.offset}; at + 4u <= masks.GetSize();
			     at += uint64_t {std::max(chunk, words)} * 4u) {
				keys.resize(std::min<uint64_t>((masks.GetSize() - at) / 4u, chunk));
				if (!ReadScalarTable(masks.Base48(), masks.GetSize(), at, runtime, keys))
					return false;
				for (size_t word = 0; word < keys.size(); word += words) ranks |= keys[word];
			}
			keys.clear();
			if (count < 32u) ranks &= (1u << count) - 1u;
			if (!read_ranks(base, size, start, step, ranks)) return false;
		} else if (!read_keys(base, size, start, step, count)) {
			return false;
		}
	} else if (!indirect.selector_mask.IsEmpty()) {
		uint32_t mask = 0;
		uint32_t count = 0;
		if (material_value.dword_count != 2u ||
		    !clean.Evaluate(indirect.selector_mask, mask) ||
		    !EvaluateIndirectCount(indirect, runtime, clean, count) || count == 0u || count > 32u) {
			return false;
		}
		if (count < 32u) mask &= (1u << count) - 1u;
		const auto material_base =
		    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
		if (!read_ranks(material_base, UINT64_MAX, selector->offset, selector->stride, mask))
			return false;
	} else if (selector->stride == 0u) {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !CanReadIndexedSelector(material, selector->offset)) return false;
		const auto records = material.NumRecords();
		if (material.Format() != Prospero::BufferFormat::kInvalid && records > selector->first_record &&
		    selector->offset < material.Stride() &&
		    !read_keys(material.Base48(), material.GetSize(),
		               uint64_t {selector->first_record} * material.Stride() + selector->offset,
		               uint64_t {selector->record_step} * material.Stride(),
		               (records - selector->first_record - 1u) / selector->record_step + 1u)) return false;
		keys.push_back(0u); // Every out-of-bounds indexed load returns zero.
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) || selector->stride < 4u ||
		    !std::has_single_bit(selector->stride) || (selector->offset & 3u) != 0u)
			return false;
		// Empty selector bounds denote the complete U32 congruence class. SMEM
		// adds its aligned immediate after the dynamic offset, without U32 wrap.
		const auto domain_count = (uint64_t {1} << 32u) / selector->stride;
		const auto available = material.GetSize() > selector->offset
		    ? material.GetSize() - selector->offset : 0u;
		const auto count = std::min(domain_count,
		    available < 4u ? 0u : (available - 4u) / selector->stride + 1u);
		if (count > MaxIndirectDescriptorProbes ||
		    !read_keys(material.Base48(), material.GetSize(), selector->offset, selector->stride, count)) return false;
		if (count < domain_count) keys.push_back(0u); // An out-of-bounds scalar read returns zero.
	}
	if (selector->byte_ranges.empty()) {
		for (auto& key: keys) {
			auto field = (key >> selector->bit_offset) & selector->bit_mask;
			if (selector->non_negative && static_cast<int32_t>(field) < 0) field = 0u;
			key = std::min(field, selector->maximum);
		}
		if (selector->excluded) std::erase(keys, *selector->excluded);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}
	return true;
}

struct DescriptorValueHash {
	size_t operator()(const DescriptorValue& value) const noexcept {
		uint64_t hash = value.dword_count;
		for (const auto word: value.dwords) hash = (hash ^ word) * 0x100000001b3ull;
		return static_cast<size_t>(hash ^ (hash >> 32u));
	}
};

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   uint32_t source_index,
                                   uint32_t resource_index, uint32_t dword_count, uint32_t read_dwords,
                                   const SrtRuntime& runtime, SrtWalker& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   Normalize&& normalize) {
	const auto& indirect = *program.descriptor_sources[source_index].indirect_descriptor;
	const auto  descriptor_bytes = read_dwords * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	const bool byte_keys = IsBoundedDescriptorTable(program, indirect);
	auto* selector_keys = selector != nullptr
	    ? &program.selector_keys[program.selector_groups[source_index]] : nullptr;
	auto& keys = selector_keys != nullptr ? selector_keys->values : program.material_keys;
	auto& byte_offsets = program.material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	// Each distinct candidate's ordinal, by value: an image heap's rows were searched linearly against every earlier
	// candidate, per draw, and as Marvel's Wolverine's first fight filled its heap that made ~4 s frames.
	std::unordered_map<DescriptorValue, uint32_t, DescriptorValueHash> ordinals;
	const auto intern_candidate = [&](const DescriptorValue& candidate) {
		if (candidate == descriptors[resource_index]) return 0u;
		const auto [found, added] =
		    ordinals.try_emplace(candidate, static_cast<uint32_t>(descriptors.size() - children_begin + 1u));
		if (added) {
			descriptors.push_back(candidate);
			auto child = root_resource;
			child.indirect_root = resource_index;
			specializations.push_back(child);
		}
		return found->second;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	DescriptorValue table_value;
	ShaderBufferResource table;
	const auto failed = [&](std::string_view reason, std::string_view details = {}) {
		const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
		return SpecializationFail(fmt::format(
		    "indirect {} resource={} source={} table_source={} words={:08x},{:08x},{:08x},{:08x} "
		    "base={:#x} size={} type={} runtime_stride={} records={} swizzle={} addtid={} oob={} "
		    "maximum={} offset={} immediate={} record_bytes={} child_count={} child_stride={} "
		    "child_offset={} indexed_pointer={} candidates={} {}",
		    reason, resource_index, source_index, indirect.table_source,
		    table_value.dwords[0], table_value.dwords[1], table_value.dwords[2], table_value.dwords[3],
		    table_base, table_size, table.Type(), table.Stride(), table.NumRecords(),
		    table.SwizzleEnabled(), table.AddTid(), table.OutOfBounds(),
		    indirect.table_maximum ? fmt::format("{}", *indirect.table_maximum) : "none",
		    indirect.table_offset, indirect.table_immediate, indirect.table_record_bytes,
		    pointer != nullptr ? pointer->count : 1u, pointer != nullptr ? pointer->stride : 0u,
		    pointer != nullptr ? pointer->offset : 0u, pointer != nullptr && pointer->indexed,
		    descriptors.size(), details));
	};
	const auto read_failed = [&](std::string_view reason, uint32_t row, uint32_t child,
	                             uint64_t base, uint64_t offset, size_t bytes) {
		return failed(reason, fmt::format("row={} child={} read_base={:#x} read_offset={:#x} "
		                                 "read_address={:#x} requested_bytes={}", row, child, base, offset,
		    (base & AddressMask & ~uint64_t {3}) + (offset & ~uint64_t {3}), bytes));
	};
	const auto normalization_failed = [&](uint64_t offset, uint32_t key, uint64_t step,
	                                      std::string_view key_kind) {
		return failed("candidate", fmt::format("candidate_offset={:#x} {}={} stride={} step={}",
		    offset & ~uint64_t {3}, key_kind, key, indirect.table_stride, step));
	};
	if (sources.empty()) {
		if (selector == nullptr) keys.clear();
		DescriptorValue material_value;
		if ((selector != nullptr && (byte_keys || !selector_keys->resolved) &&
		     !clean.EvaluateDescriptor(selector->source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return false;
		}
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return false;
		}
		if (byte_keys && table_size > UINT32_MAX) return failed("table geometry");
		bool use_selector = selector != nullptr;
		if (byte_keys && selector != nullptr) {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			use_selector = CanReadIndexedSelector(material, selector->offset);
			if (use_selector && material.Format() != Prospero::BufferFormat::kInvalid &&
			    material.NumRecords() != 0u && selector->offset < material.Stride()) {
				const auto begin = material.Base48() + selector->offset;
				const auto end = begin + uint64_t {material.NumRecords() - 1u} * material.Stride() + 4u;
				for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
					const auto& buffer = program.info.buffers[i];
					if (!buffer.written) continue;
					ShaderBufferResource written;
					if (program.descriptor_sources[buffer.source].indirect_descriptor ||
					    !DecodeBufferDescriptor(snapshot.buffers[i], written) ||
					    (written.Base48() != 0u && written.GetSize() != 0u &&
					     begin < written.Base48() + written.GetSize() &&
					     written.Base48() < end)) {
						use_selector = false;
						break;
					}
				}
			}
		}
		if (byte_keys && !use_selector) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes))
				return failed("table geometry");
			const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
			const bool indexed_pointer = pointer != nullptr && pointer->indexed;
			const auto child_count = pointer != nullptr ? pointer->count : 1u;
			uint64_t domain = indirect.table_maximum ? uint64_t {*indirect.table_maximum} + 1u
			                                       : uint64_t {1} << 32u;
			if (!indirect.key_count.IsEmpty()) {
				uint32_t key_count = 0u;
				if (!EvaluateIndirectCount(indirect, runtime, clean, key_count)) return false;
				domain = std::min(domain, uint64_t {key_count});
			}
			if (indexed_pointer &&
			    (!indirect.table_maximum || table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			     table.OutOfBounds() != 0u || (table_base & 3u) != 0u || (table.Stride() & 3u) != 0u ||
			     uint64_t {indirect.table_immediate} + 8u > table.Stride() ||
			     domain > table.NumRecords()))
				return failed("indexed pointer geometry", fmt::format("rows={}", domain));
			const bool affine = indirect.table_stride != 0u || indexed_pointer;
			const bool image_heap = dword_count == 8u && pointer == nullptr;
			const uint64_t step = indexed_pointer ? table.Stride()
			    : indirect.table_maximum ? indirect.table_stride
			    : affine ? std::max(uint64_t {1} << std::countr_zero(indirect.table_stride), uint64_t {4})
			             : 4u;
			const uint64_t first = ((indirect.table_maximum ? indirect.table_offset
			    : indirect.table_offset % step) & ~uint64_t {3}) +
			                       (indirect.table_immediate & ~3u);
			if (!indirect.table_maximum) domain /= step;
			const auto count = table_size >= first + 4u
			    ? std::min((table_size - first - 4u) / step + 1u, domain)
			    : 0u;
			descriptors[resource_index] = {.dword_count = dword_count};
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				const auto& image = program.info.images[resource_index];
				// A resident heap maps slot words instead of materializing its entries.
				ImageHeapView view;
				if (image_heap && runtime.image_heap != nullptr &&
				    image.resource_class == ImageResourceClass::Sampled &&
				    image.mip_mode != ImageMipMode::Dynamic &&
				    runtime.image_heap->Acquire({table_base, table_size, step, first, count, &image},
				                                view)) {
					snapshot.specialization_reads.emplace_back(table_base, table_size);
					if (view.kinds.empty()) return true;
					auto root = root_resource;
					root.indirect_root = resource_index;
					root.indirect_mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
					root.heap = true;
					specializations[resource_index] = root;
					snapshot.flattened_srt.insert(snapshot.flattened_srt.end(),
					    {view.slot_count, static_cast<uint32_t>(view.slots), static_cast<uint32_t>(view.slots >> 32u)});
					for (const auto& kind: view.kinds) {
						descriptors.push_back(kind);
						specializations.push_back(root);
					}
					return true;
				}
			}
			if (child_count == 0u || (!image_heap && count * child_count > MaxIndirectDescriptorProbes))
				return failed("candidate count", fmt::format("rows={} limit={}", count, MaxIndirectDescriptorProbes));
			// Affine SMEM offsets wrap in U32 before adding the aligned immediate.
			// Image heaps retain one bulk snapshot and their absolute byte-offset mapping.
			auto& heap_words = program.material_keys;
			if (image_heap) {
				heap_words.resize(table_size / 4u);
				if (!ReadScalarTable(table_base, table_size, 0u, runtime, heap_words))
					return read_failed("image heap read", 0u, 0u, table_base, 0u, heap_words.size() * sizeof(uint32_t));
			}
			IndirectImageReuse::Root* reuse = nullptr;
			const std::array<uint64_t, 5> geometry {table_base, table_size, step, first, count};
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				if (image_heap && program.image_reuse && program.image_reuse->enabled &&
				    resource_index < program.image_reuse->roots.size()) {
					reuse = &program.image_reuse->roots[resource_index];
					if (reuse->valid && reuse->geometry == geometry && reuse->source == root_resource &&
					    reuse->image == program.info.images[resource_index] && reuse->indirect == indirect &&
					    reuse->words.size() == heap_words.size() &&
					    std::memcmp(reuse->words.data(), heap_words.data(), heap_words.size() * sizeof(uint32_t)) == 0) {
						auto&      flat           = snapshot.flattened_srt;
						const auto mapping_offset = static_cast<uint32_t>(flat.size());
						flat.insert(flat.end(), reuse->mapping.begin(), reuse->mapping.end());
						if (reuse->grouped) {
							auto& root                   = specializations[resource_index];
							root                         = reuse->root;
							root.indirect_mapping_offset = mapping_offset;
							program.image_reuse->whole.push_back(resource_index);
							return true;
						}
						if (reuse->children.empty()) return true;
						descriptors.insert(descriptors.end(), reuse->children.begin(), reuse->children.end());
						auto child          = root_resource;
						child.indirect_root = resource_index;
						specializations.resize(specializations.size() + reuse->children.size(), child);
						auto& root                      = specializations[resource_index];
						root.indirect_root              = resource_index;
						root.indirect_mapping_offset    = mapping_offset;
						root.indirect_search_iterations = 0u;
						return true;
					}
					reuse->valid = false;
				}
			}
			const auto mapping_offset = snapshot.flattened_srt.size();
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			for (uint32_t word = 0; word < count; ++word) {
				const auto table_offset = affine ? first + word * step : uint64_t {word} * 4u;
				uint64_t pointer_base = 0u;
				if (pointer != nullptr) {
					std::array<uint32_t, 2> address {};
					if (!ReadScalarTable(table_base, table_size, table_offset, runtime, address))
						return read_failed("parent pointer read", word, 0u, table_base, table_offset, sizeof(address));
					pointer_base = (uint64_t {address[1]} << 32u) | address[0];
				}
				for (uint32_t child = 0; child < child_count; ++child) {
					DescriptorValue candidate {.dword_count = dword_count};
					if (pointer != nullptr) {
						if (!ReadScalarTable(pointer_base, UINT64_MAX,
						        uint64_t {child} * pointer->stride + pointer->offset, runtime,
						        std::span(candidate.dwords).first(read_dwords)))
							return read_failed("child descriptor read", word, child, pointer_base,
							    uint64_t {child} * pointer->stride + pointer->offset, descriptor_bytes);
					} else if (!image_heap) {
						if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
						                     std::span(candidate.dwords).first(read_dwords))) return false;
					} else {
						const auto table_word = table_offset / 4u;
						std::copy_n(heap_words.begin() + table_word,
						            std::min<size_t>(read_dwords, heap_words.size() - table_word),
						            candidate.dwords.begin());
					}
					if (!normalize(candidate))
						return normalization_failed(table_offset, word, step,
						                            indirect.table_maximum ? "key" : "probe");
					const auto ordinal = intern_candidate(candidate);
					if (ordinal == 0u) continue;
					const auto mapping_word = image_heap ? table_offset / 4u : word * child_count + child;
					snapshot.flattened_srt.resize(mapping_offset + 2u + mapping_word, 0u);
					snapshot.flattened_srt[mapping_offset + 1u + mapping_word] = ordinal;
				}
			}

			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				snapshot.flattened_srt[mapping_offset] =
				    static_cast<uint32_t>(snapshot.flattened_srt.size() - mapping_offset - 1u);
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				if (reuse != nullptr) {
					reuse->valid    = true;
					reuse->stored   = program.image_reuse->call;
					reuse->geometry = geometry;
					reuse->source   = root_resource;
					reuse->image    = program.info.images[resource_index];
					reuse->indirect = indirect;
					std::swap(reuse->words, heap_words);
					reuse->children.assign(descriptors.begin() + static_cast<ptrdiff_t>(children_begin),
					                       descriptors.end());
					reuse->mapping.assign(snapshot.flattened_srt.begin() + static_cast<ptrdiff_t>(mapping_offset),
					                      snapshot.flattened_srt.end());
					reuse->grouped = false;
					reuse->group_images.clear();
					reuse->group_values.clear();
				}
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			uint32_t first_key = 0;
			if (!EvaluateIndirectCount(indirect, runtime, clean, key_count)) return false;
			if (!indirect.selector_first.IsEmpty() && !clean.Evaluate(indirect.selector_first, first_key))
				return false;
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (indirect.selector_first.IsEmpty() && key_count != 0u &&
			         uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return false;
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), first_key);
			if (!keys.empty() && keys.back() < keys.front())
				std::rotate(keys.begin(), keys.begin() + (UINT32_MAX - first_key + 1ull), keys.end());
			if (indirect.workgroup_shift != 0u) {
				for (auto& key: keys)
					key = static_cast<uint32_t>(uint64_t {key} * indirect.workgroup_multiplier >>
					                            indirect.workgroup_shift);
				keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
			}
		} else {
			if (!indirect.selector_mask.IsEmpty() && table_value.dword_count != 2u) return false;
			if (!selector_keys->resolved) {
				if (!ResolveSelectorKeys(indirect, material_value, runtime, clean, keys)) return false;
				selector_keys->resolved = true;
			}
		}
		if (byte_keys) {
			// Keep cached selector values independent of each destination table's
			// U32 arithmetic, and use the same byte keys as the full-heap GPU path.
			byte_offsets.clear();
			byte_offsets.reserve(keys.size());
			for (const auto key: keys) {
				const auto offset = uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
				                    (indirect.table_immediate & ~3u);
				byte_offsets.push_back(offset <= UINT32_MAX ? static_cast<uint32_t>(offset) : UINT32_MAX);
			}
			std::ranges::sort(byte_offsets);
			byte_offsets.erase(std::unique(byte_offsets.begin(), byte_offsets.end()), byte_offsets.end());
		}
	}

	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto selected_keys = byte_keys ? std::span<const uint32_t>(byte_offsets)
	                                    : std::span<const uint32_t>(keys);
	const auto key_count = sources.empty() ? selected_keys.size() : sources.size();
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_count * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? selected_keys[entry] : entry;
		uint64_t table_offset = 0;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			table_offset = byte_keys ? key
			    : uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			          (indirect.table_immediate & ~3u);
			if (!(byte_keys && key == UINT32_MAX) &&
			    !ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(read_dwords))) {
				return read_failed("table descriptor read", key, 0u, table_base, table_offset, descriptor_bytes);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return false;
		}
		if (!normalize(candidate))
			return normalization_failed(table_offset, key, indirect.table_stride, "key");
		uint32_t ordinal = 0;
		if (entry == 0 && !byte_keys) {
			descriptors[resource_index] = candidate;
		} else {
			ordinal = intern_candidate(candidate);
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
	std::vector<std::array<uint32_t, 2>> mapping;
	std::vector<Binding>                bindings;
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

std::vector<ImageHeapChange> DiffImageHeap(std::span<const uint32_t> guest, std::vector<uint32_t>& mirror) {
	EXIT_IF(mirror.size() != guest.size());
	std::vector<ImageHeapChange> changes;
	for (size_t word = 8; word + 8 <= guest.size(); word += 8) {
		if (std::memcmp(&guest[word], &mirror[word], 32) == 0) continue;
		ImageHeapChange change {.slot = static_cast<uint32_t>(word / 8)};
		// One read: the game may be writing the entry, and the next diff then sees the rest.
		std::memcpy(change.descriptor.data(), &guest[word], 32);
		std::memcpy(&mirror[word], change.descriptor.data(), 32);
		changes.push_back(change);
	}
	return changes;
}

uint32_t HeapImageInfo(const DescriptorValue& descriptor, bool r128) {
	if (NullImageDescriptor(descriptor) || !ValidImageDescriptor(descriptor, r128)) return 0u;
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	// shortcut: a texture whose format the shader would have to convert is not shown; give the
	// info dword the conversion when a game reads one by slot.
	if (numeric_class == Prospero::TextureNumericClass::Unsupported || Prospero::IsFmaskTextureFormat(format) ||
	    ImageConversionFormat(format) != Prospero::BufferFormat::kInvalid) return 0u;
	uint32_t filled = 0;
	for (const auto dimension: {Decoder::ImageDimension::Dim2D, Decoder::ImageDimension::Dim2DArray,
	                            Decoder::ImageDimension::Dim3D, Decoder::ImageDimension::Dim1D}) {
		if (DescriptorDimension(descriptor, dimension) == dimension) filled |= ImageHeapFilled(dimension);
	}
	// An array instruction on a 2D texture reads its one layer (as upstream's 65e22adc does outside
	// the heap); without this such a read fell to slot 0 and sampled black.
	if ((filled & ImageHeapFilled(Decoder::ImageDimension::Dim2D)) != 0u && !DescriptorIsCube(descriptor))
		filled |= ImageHeapFilled(Decoder::ImageDimension::Dim2DArray);
	if (filled == 0u) return 0u;
	// The levels below a texture's min LOD (in 1/256 of a level) are not in the game's memory:
	// its streamer has not loaded them. A 2D texture of one layer (the game's material textures
	// are arrays of one) is held from the first loaded level on.
	const auto& words       = descriptor.dwords;
	const auto  last_level  = (words[3] >> 16u) & 0xfu;
	const auto  type        = static_cast<Prospero::ImageType>((words[3] >> 28u) & 0xfu);
	const bool  plain       = !r128 &&
	                          (type == Prospero::ImageType::kColor2D || type == Prospero::ImageType::kColor2DArray) &&
	                          ((words[3] >> 12u) & 0xfu) == 0u && (words[4] & 0x1fffu) == 0u &&
	                          ((words[6] >> 21u) & 1u) == 0u && ((words[5] >> 4u) & 0xfu) == last_level;
	const auto  first_level = plain ? std::min(((words[1] >> 8u) & 0xfffu) >> 8u, last_level) : 0u;
	return (static_cast<uint32_t>(numeric_class) - static_cast<uint32_t>(Prospero::TextureNumericClass::Float)) |
	       (DescriptorIsCube(descriptor) ? ImageHeapCube : 0u) | filled | (first_level << ImageHeapLevelShift);
}

// A sampled image read from the game's descriptor heap is read by slot from the host's image
// heap, so its shader does not depend on which textures the heap holds. The host fills the slots
// from the game's own writes to its heap (GuestImageHeap), so a draw only says where the heap is.
// False when the image is not read that way.
static bool MaterializeHeapImage(const ResourcePlan& program, const DescriptorSource::IndirectDescriptor& indirect,
                                 uint32_t resource_index, const SrtRuntime& runtime, SrtWalker& clean,
                                 ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	const auto& image = program.info.images[resource_index];
	if (!runtime.guest_image_heap || image.resource_class != ImageResourceClass::Sampled || image.depth_compare ||
	    image.atomic || image.mip_mode != ImageMipMode::None || ImageHeapFilled(image.dimension) == 0u ||
	    indirect.pointer_table || !IsBoundedDescriptorTable(program, indirect)) return false;
	DescriptorValue      table_value;
	ShaderBufferResource table;
	if (!clean.EvaluateDescriptor(indirect.table_source, table_value) ||
	    !DecodeBufferDescriptor(table_value, table) || table.Type() != 0u || table.Base48() == 0u) return false;
	const auto size = table.GetSize();
	if (size <= HeapTableMinDescriptors * 32u || size > uint64_t {ImageHeapSlots} * 32u) return false;
	snapshot.image_heap = {table.Base48(), size};
	// The root reads float slots; a slot of another numeric class is read through its own image.
	auto& root         = specialization.images[resource_index];
	root.guest_heap    = true;
	root.indirect_root = resource_index;
	root.numeric_class = Prospero::TextureNumericClass::Float;
	for (const auto numeric_class: {Prospero::TextureNumericClass::Uint, Prospero::TextureNumericClass::Sint}) {
		auto child          = specialization.images[resource_index];
		child.numeric_class = numeric_class;
		specialization.images.push_back(child);
		snapshot.images.push_back({.dword_count = 8u});
	}
	return true;
}

// A sampler read out of a table by key, used only with images of the image heap, is read by
// record from the image heap's sampler arrays, which mirror the game's sampler heap. Its uses are
// then plain float or integer samples (heap images are never compared or gathered by level).
static bool MaterializeHeapSampler(const ResourcePlan& program, const DescriptorSource::IndirectDescriptor& indirect,
                                   uint32_t resource_index, const SrtRuntime& runtime, SrtWalker& clean,
                                   ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	// Upstream gives a material's sampler key a selector (its records are enumerated as candidates);
	// the heap reads the record that selector names (HeapSamplerRecordOf) out of one table of
	// 4-dword records, when the table's records lie on whole 16-byte records.
	const auto* table_source = Source(program, indirect.table_source);
	if (!runtime.guest_image_heap || program.info.samplers[resource_index].gather_lod || !indirect.sources.empty() ||
	    indirect.pointer_table || indirect.workgroup_axis != UINT32_MAX || table_source == nullptr ||
	    table_source->dword_count != 4u || !HeapSamplerRecordOf(indirect)) return false;
	bool used = false;
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.sampler != resource_index) continue;
		if (pair.image >= specialization.images.size() || !specialization.images[pair.image].guest_heap) return false;
		used = true;
	}
	DescriptorValue      table_value;
	ShaderBufferResource table;
	if (!used || !clean.EvaluateDescriptor(indirect.table_source, table_value) ||
	    !DecodeBufferDescriptor(table_value, table) || table.Type() != 0u || table.Base48() == 0u) return false;
	const auto size = table.GetSize();
	if (size == 0u || size > uint64_t {ImageHeapSamplerSlots} * 16u) return false;
	snapshot.sampler_heap                        = {table.Base48(), size};
	specialization.samplers[resource_index].guest_heap = true;
	return true;
}

// With reuse, a root its table's reuse appended whole gets its stored groups, and a root whose candidates were
// materialized afresh this call stores the groups they make.
static void GroupIndirectImages(const ShaderInfo& base, ResourceSnapshot& snapshot,
                                ResourceSpecialization& specialization, IndirectImageReuse* reuse) {
	using Image = ResourceSpecialization::Image;
	auto& images = specialization.images;
	snapshot.image_groups.clear();
	const auto base_count = static_cast<uint32_t>(std::min(base.images.size(), images.size()));
	std::vector<uint8_t> grouped(images.size()); // candidates that a group replaces
	std::vector<Image> group_images;
	std::vector<std::vector<DescriptorValue>> group_values;
	for (uint32_t root = 0; root < base_count; ++root) {
		const auto& root_image = images[root];
		if (root_image.indirect_root != root || root_image.heap || root_image.guest_heap ||
		    base.images[root].resource_class != ImageResourceClass::Storage) continue;
		if (reuse != nullptr && std::ranges::find(reuse->whole, root) != reuse->whole.end()) {
			const auto& stored = reuse->roots[root];
			group_images.insert(group_images.end(), stored.group_images.begin(), stored.group_images.end());
			group_values.insert(group_values.end(), stored.group_values.begin(), stored.group_values.end());
			continue;
		}
		// Ordinal 0 is the root, the others its candidates by index (MaterializeIndirectDescriptor).
		std::vector<uint32_t> members;
		for (uint32_t index = base_count; index < images.size(); ++index)
			if (images[index].indirect_root == root) members.push_back(index);
		if (members.size() + 1u < IndirectImageGroupMin) continue;
		const auto type_of = [&](uint32_t index) {
			auto type = images[index];
			type.indirect_mapping_offset = type.indirect_search_iterations = 0u;
			return type;
		};
		// The groups in a fixed order of their types, so that the same types make the same specialization
		// whatever order the table holds them in.
		std::vector<Image> types;
		bool groupable = true;
		for (const auto member: members) {
			groupable = groupable && images[member].mip_count == 1u && !images[member].fmask;
			if (groupable && std::ranges::find(types, type_of(member)) == types.end()) types.push_back(type_of(member));
		}
		if (!groupable) continue;
		const auto key = [](const Image& type) {
			return std::tuple(static_cast<uint32_t>(type.numeric_class), static_cast<uint32_t>(type.dimension),
			                  static_cast<uint32_t>(type.conversion_format), type.shader_swizzle, type.cube);
		};
		std::ranges::sort(types, [&](const Image& a, const Image& b) { return key(a) < key(b); });
		std::vector<std::vector<DescriptorValue>> values(types.size());
		std::vector<uint32_t> mapped(members.size() + 1u, 0u); // by ordinal
		for (uint32_t ordinal = 1; ordinal <= members.size(); ++ordinal) {
			const auto member = members[ordinal - 1u];
			const auto group  = static_cast<uint32_t>(std::ranges::find(types, type_of(member)) - types.begin());
			mapped[ordinal]   = (group + 1u) << IndirectImageGroupShift | static_cast<uint32_t>(values[group].size());
			values[group].push_back(snapshot.images[member]);
		}
		if (std::ranges::any_of(values, [](const auto& group) { return group.size() >= 1u << IndirectImageGroupShift; }))
			continue;
		auto& flat = snapshot.flattened_srt;
		const auto offset = root_image.indirect_mapping_offset;
		const bool pairs = root_image.indirect_search_iterations != 0u;
		const auto ordinal_at = [&](uint32_t entry) -> uint32_t& {
			return flat[offset + 1u + (pairs ? entry * 2u + 1u : entry)];
		};
		if (offset >= flat.size() || offset + 1u + size_t {flat[offset]} * (pairs ? 2u : 1u) > flat.size()) continue;
		bool mapped_all = true;
		for (uint32_t entry = 0; entry < flat[offset]; ++entry) mapped_all = mapped_all && ordinal_at(entry) < mapped.size();
		if (!mapped_all) continue;
		for (uint32_t entry = 0; entry < flat[offset]; ++entry) ordinal_at(entry) = mapped[ordinal_at(entry)];
		for (const auto member: members) grouped[member] = 1u;
		auto* stored = reuse != nullptr && root < reuse->roots.size() ? &reuse->roots[root] : nullptr;
		if (stored != nullptr && (!stored->valid || stored->stored != reuse->call || stored->grouped)) stored = nullptr;
		if (stored != nullptr) {
			stored->grouped = true;
			stored->root    = root_image;
			stored->mapping.assign(flat.begin() + offset, flat.begin() + offset + 1u + size_t {flat[offset]} * (pairs ? 2u : 1u));
			stored->children = {};
			stored->group_images.clear();
			stored->group_values.clear();
		}
		for (uint32_t group = 0; group < types.size(); ++group) {
			auto image              = types[group];
			image.indirect_capacity = std::bit_ceil(std::max<uint32_t>(static_cast<uint32_t>(values[group].size()), 64u));
			if (stored != nullptr) {
				stored->group_images.push_back(image);
				stored->group_values.push_back(values[group]);
			}
			group_images.push_back(image);
			group_values.push_back(std::move(values[group]));
		}
	}
	if (group_images.empty()) return;
	CompactImages(images, [&](size_t index) { return grouped[index] == 0u; });
	CompactImages(snapshot.images, [&](size_t index) { return grouped[index] == 0u; });
	snapshot.image_groups.resize(images.size());
	for (uint32_t group = 0; group < group_images.size(); ++group) {
		images.push_back(group_images[group]);
		// An exemplar for code that reads one descriptor per image; the host binds the group's.
		snapshot.images.push_back(group_values[group].front());
		snapshot.image_groups.push_back(std::move(group_values[group]));
	}
}

void GroupIndirectImages(const ShaderInfo& base, ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	GroupIndirectImages(base, snapshot, specialization, nullptr);
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization) {
	// A root appended whole is already as this function and its grouping leave it.
	const auto* reuse = program.image_reuse.get();
	const auto  whole = [&](uint32_t index) {
		return reuse != nullptr && std::ranges::find(reuse->whole, index) != reuse->whole.end();
	};
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		if (whole(i)) continue;
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
		if (image.guest_heap) {
			// A slot's own class, cube flag and level range are the heap's, read at run time.
			image.dimension = base.dimension;
			image.mip_count = 1u;
			image.cube      = false;
			continue;
		}
		image.mip_count = ImageMipCount(base, descriptor);
		// A null T# reads zero; it keeps the dimension and cube its instructions declare.
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			if (!NullImageInstructionDims()) {
				// An empty slot reads the null image whatever its view, so under an array instruction it
				// is an array too: a slot empty for one material and an array for the next is one build
				// (CS_AVMaterialQueryColorVolReadBack was rebuilt every 1.5 s between the two).
				image.dimension = base.dimension == Decoder::ImageDimension::Dim2DArray ? Decoder::ImageDimension::Dim2DArray
				                                                                        : Decoder::ImageDimension::Dim2D;
				image.cube          = false;
			}
			continue;
		}
		auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		// A plain 2D texture under an array instruction is a one-layer array: sampling clamps
		// the layer to 0 as the hardware ignores it, and so does a fetch (the emitter clamps its
		// layer), so 2D and array textures share one build.
		if (base.one_layer_safe && !base.written && base.dimension == Decoder::ImageDimension::Dim2DArray &&
		    descriptor_dimension == Decoder::ImageDimension::Dim2D)
			descriptor_dimension = Decoder::ImageDimension::Dim2DArray;
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
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		// See ReadsDepthBits; raw loads only, since filtering would blend the codes.
		const bool depth_bits =
		    !storage && base.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
		    static_cast<Prospero::TileMode>((descriptor.dwords[3] >> 20u) & 0x1fu) ==
		        Prospero::TileMode::kDepth;
		image.conversion_format = depth_bits ? format : ImageConversionFormat(format);
		// SKIP_DEGAMMA (S# bit 31) reads sRGB texels as stored.
		image.skip_degamma = Prospero::IsSrgbFormat(format) &&
		    std::ranges::any_of(program.info.sampled_pairs, [&](const auto& pair) {
			    return pair.image == base_index && pair.sampler < snapshot.samplers.size() &&
			           (snapshot.samplers[pair.sampler].dwords[0] >> 31u) != 0u;
		    });
		// The shader decodes direct 8 and 8_8 sRGB textures where the device has no sRGB view.
		if (!storage && !base.depth_compare && !image.skip_degamma &&
		    image.indirect_root == ImageResource::NoIndirectImage &&
		    (format == Prospero::BufferFormat::k8Srgb || format == Prospero::BufferFormat::k8_8Srgb)) {
			image.srgb_decode = format;
		}
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const auto raw_sint_format = RawSignedStorageFormat(base, format);
		const bool raw_sint_storage = raw_sint_format != Prospero::BufferFormat::kInvalid;
		if (raw_sint_format == Prospero::BufferFormat::k16UInt) image.conversion_format = format;
		const auto packed_storage_format = PackedStorageFormat(base, format);
		if (packed_storage_format != Prospero::BufferFormat::kInvalid) image.conversion_format = format;
		image.numeric_class = Prospero::SampledTextureNumericClass(
		    depth_bits ? Prospero::BufferFormat::k16UNorm
		    : packed_storage_format != Prospero::BufferFormat::kInvalid ? packed_storage_format
		                                                                : Prospero::RemapTextureFormat(format));
		if (!storage && format == Prospero::BufferFormat::k11_11_10UNorm) {
			if (image.indirect_root != ImageResource::NoIndirectImage) {
				return SpecializationFail("packed UNorm sampling requires a direct image");
			}
			if (std::ranges::any_of(program.info.sampled_pairs,
			                       [&](const auto& pair) { return pair.image == i; }) &&
			    !ValidatePackedUnormSampling(program, snapshot, i)) return false;
		}
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			auto message = fmt::format(
			    "sampled image descriptor {} uses unsupported format {}: root={} source={} "
			    "first_use_pc={:#x} r128={} words={:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, static_cast<uint32_t>(format), base_index, base.source, base.first_use_pc,
			    base.r128, descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2],
			    descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5],
			    descriptor.dwords[6], descriptor.dwords[7]);
			const auto* source = Source(program, base.source);
			if (i >= program.info.images.size() && source != nullptr &&
			    source->indirect_descriptor &&
			    IsBoundedDescriptorTable(program, *source->indirect_descriptor) &&
			    specialization.images[base_index].indirect_search_iterations == 0u &&
			    !source->indirect_descriptor->pointer_table) {
				const auto& indirect = *source->indirect_descriptor;
				const size_t mapping = specialization.images[base_index].indirect_mapping_offset;
				const auto& flat = snapshot.flattened_srt;
				if (mapping < flat.size() && flat[mapping] <= flat.size() - mapping - 1u) {
					uint32_t ordinal = 0;
					for (size_t child = program.info.images.size(); child <= i; ++child)
						ordinal += specialization.images[child].indirect_root == base_index;
					uint32_t matches = 0;
					uint64_t first_offset = 0;
					for (uint32_t probe = 0; probe < flat[mapping]; ++probe) {
						if (flat[mapping + 1u + probe] != ordinal) continue;
						if (matches++ == 0u) first_offset = uint64_t {probe} * 4u;
					}
					message += fmt::format(" table_source={} ordinal={} dense_matches={} "
					                       "first_candidate_offset={:#x}",
					                       indirect.table_source, ordinal, matches, first_offset);
				}
			}
			return SpecializationFail(message);
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.guest_heap || whole(root_index)) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		// A heap maps its slot count and slot address; tables map each key.
		size_t mapping_words = 3u;
		if (!root.heap)
			mapping_words = 1u + size_t {key_count} * (root.indirect_search_iterations == 0u ? 1u : 2u);
		if (key_count == 0u ||
		    root.indirect_mapping_offset + mapping_words > snapshot.flattened_srt.size()) {
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
		const auto& base = program.info.images[root_index];
		const bool storage_swizzle = base.unfiltered_access_only &&
		    base.resource_class == ImageResourceClass::Storage;
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		const auto is_1d_or_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim1D ||
			       dimension == Decoder::ImageDimension::Dim2D;
		};
		// The descriptor's type selects the address layout. A 2D or 3D candidate reads the leading
		// coordinates of an equal or larger dimension, but an array with more than one slice reads
		// the next as its slice. Derivative, LOD, clamp and mip operands shift with the dimension.
		const auto leading = [&](Decoder::ImageDimension dimension, const DescriptorValue& value) {
			const bool sliced =
			    static_cast<Prospero::ImageType>((value.dwords[3] >> 28u) & 0xfu) ==
			        Prospero::ImageType::kColor2DArray &&
			    (value.dwords[4] & 0x1fffu) != ((value.dwords[4] >> 16u) & 0x1fffu);
			return !sliced &&
			       (dimension == Decoder::ImageDimension::Dim2D || dimension == base.dimension);
		};
		const bool leading_coordinates = (base.dimension == Decoder::ImageDimension::Dim2D ||
		                                  base.dimension == Decoder::ImageDimension::Dim3D) &&
		    leading(image_class.dimension, snapshot.images[exemplar]) &&
		    std::ranges::none_of(program.memory_info, [&](const MemoryInfo& memory) {
			    return memory.kind == ResourceKind::Image && memory.resource == root_index &&
			           (memory.image_has_mip || (memory.image_sample_flags &
			               (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagDerivative |
			                Decoder::ImageSampleFlagLodClamp)) != 0u);
		    });
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
			const bool unfiltered_coordinates = base.unfiltered_access_only &&
			    base.dimension == Decoder::ImageDimension::Dim2D &&
			    !image.cube && !image_class.cube && is_1d_or_2d(image.dimension) &&
			    is_1d_or_2d(image_class.dimension);
			// A storage table's stores are emitted per candidate with its own conversion, and GroupIndirectImages
			// groups candidates by it: a k5_5_5_1UNorm candidate's R16 packing (PackedStorageFormat) beside
			// k8_8_8_8UNorm ones, as in Marvel's Wolverine's fight, is one table. Other conversions still match.
			const bool packed_storage =
			    storage_swizzle && (PackedStorageFormat(base, image.conversion_format) != Prospero::BufferFormat::kInvalid ||
			                        PackedStorageFormat(base, image_class.conversion_format) != Prospero::BufferFormat::kInvalid);
			const bool prefix_coordinates =
			    leading_coordinates && leading(image.dimension, snapshot.images[candidate]);
			if ((!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension)) &&
			     !unfiltered_coordinates && !prefix_coordinates) ||
			    (image.conversion_format != image_class.conversion_format && !packed_storage) ||
			    (!storage_swizzle && image.shader_swizzle != image_class.shader_swizzle)) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	GroupIndirectImages(program.info, snapshot, specialization, program.image_reuse.get());
	if (!snapshot.image_groups.empty())
		CompactImages(snapshot.image_groups, [&](size_t index) { return !specialization.images[index].fmask; });
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	plan.mapping.resize(base.samplers.size());
	plan.bindings.resize(base.samplers.size());
	std::vector<uint8_t> usage(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.images[pair.image].indirect_resources) {
			if (candidate >= base.images.size()) return false;
			usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[candidate]));
		}
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : static_cast<uint32_t>(plan.bindings.size());
			if (!first) plan.bindings.emplace_back();
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
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
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
				block.images.push_back(memory.resource);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		for (auto* list: {&block.sources, &block.images}) {
			std::ranges::sort(*list);
			list->erase(std::unique(list->begin(), list->end()), list->end());
		}
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
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

static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard) {
	return AffineIndex<3>(value, [&](Value at) { return ResolveActiveU32(at, guard); },
	    [&](Value at) -> std::optional<size_t> {
		    const auto* inst = at.TryInstruction();
		    if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetBuiltin ||
		        inst->Arg(1) != Value(axis)) return {};
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) return 1u;
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) return 2u;
		    return {};
	    });
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
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
		    store->Arg(3).Resolve() != Value(true)) return {};
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

// The operations LaneWriteRanges computes for each invocation.
static bool LaneOpcode(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::GetBuiltin:
		case ValueOpcode::ConditionRef:
		case ValueOpcode::IAdd32:
		case ValueOpcode::ISub32:
		case ValueOpcode::IMul32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4: return true;
		default: return false;
	}
}

// The stores and atomics on written buffers of a compute shader whose addresses are sums,
// products and shifts of immediates, workgroup and invocation IDs and values the host evaluates
// (see BufferWriteExtents). Marvel's Wolverine binds its whole 2.7 GB geometry pool to store a
// few elements (CS 5e04a15a8ee16915: index (K << 1) + group * 64 + local).
static std::vector<ResourcePlan::BufferStore> PlanBufferStores(const Program&                     program,
                                                               const std::function<Value(Value)>& clone) {
	std::vector<ResourcePlan::BufferStore> stores;
	if (program.stage != ShaderType::Compute) return stores;
	std::vector<uint8_t> unbounded(program.info.buffers.size());
	Value guard; // the store's lane predicate: a select on it takes the stored lanes' operand
	std::function<bool(Value, ResourcePlan::BufferStore&, uint32_t)> collect = [&](Value value, auto& store,
	                                                                               uint32_t depth) {
		value            = ResolveActiveU32(value, guard);
		const auto* inst = value.TryInstruction();
		if (value.IsImmediate()) return true;
		if (value.IsEmpty() || inst == nullptr || depth > 32u) return false;
		if (inst->GetOpcode() == ValueOpcode::GetBuiltin) return true;
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			store.values.push_back(clone(value));
			return true;
		}
		const auto op = inst->GetOpcode();
		return (op == ValueOpcode::IAdd32 || op == ValueOpcode::IMul32 || op == ValueOpcode::ShiftLeftLogical32) &&
		       collect(inst->Arg(0), store, depth + 1u) && collect(inst->Arg(1), store, depth + 1u);
	};
	// A part that is no such sum may still be computed for each invocation on the host (LaneWriteRanges):
	// integer operations, compares and selects over the IDs and values the host evaluates, and dword
	// loads from buffers this shader does not write. Marvel's Wolverine scatters records by an index
	// list: CS 96e606211e972f4f and f2f81fa4e946c837 store at list[(group * 64 + local) >> 2][local & 3].
	std::function<bool(Value, ResourcePlan::BufferStore&, uint32_t)> lane = [&](Value value, auto& store,
	                                                                            uint32_t depth) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (value.IsImmediate()) return true;
		if (value.IsEmpty() || inst == nullptr || depth > 32u) return false;
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			store.values.push_back(clone(value));
			return true;
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::LoadBufferU32 || op == ValueOpcode::LoadBufferU32x2 ||
		    op == ValueOpcode::LoadBufferU32x3 || op == ValueOpcode::LoadBufferU32x4) {
			const auto& memory = program.memory_info.at(inst->Flags<MemoryFlags>().index);
			if (memory.kind != ResourceKind::Buffer || memory.typed || memory.formatted || memory.data_bits != 32u ||
			    memory.resource >= program.info.buffers.size() || program.info.buffers[memory.resource].written)
				return false;
			for (uint32_t index = 1; index < inst->NumArgs(); index++) { // index, offset, SGPR offset, predicate
				if (!lane(inst->Arg(index), store, depth + 1u)) return false;
			}
			return true;
		}
		if (!LaneOpcode(op)) return false;
		for (uint32_t index = 0; index < inst->NumArgs(); index++) {
			if (!lane(inst->Arg(index), store, depth + 1u)) return false;
		}
		return true;
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) continue;
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only) continue;
			if (memory.kind != ResourceKind::Buffer || memory.resource >= unbounded.size()) return {};
			ResourcePlan::BufferStore store {.buffer = memory.resource, .immediate = memory.offset};
			guard = inst.Arg(inst.NumArgs() - 1u);
			for (uint32_t part = 0; part < 3u && unbounded[memory.resource] == 0u; part++) {
				const auto value = inst.Arg(1u + part);
				if (collect(value, store, 0u)) {
					store.address[part] = clone(value);
				} else if (lane(value, store, 0u)) {
					store.address[part] = clone(value);
					store.per_lane      = true;
				} else {
					unbounded[memory.resource] = 1u;
				}
			}
			// A predicate the host cannot compute per invocation counts every invocation as storing.
			if (store.per_lane && !lane(guard, store, 0u)) guard = Value(true);
			store.guard = clone(guard);
			stores.push_back(std::move(store));
		}
	}
	std::erase_if(stores, [&](const auto& store) { return unbounded[store.buffer] != 0u; });
	return stores;
}

// The bytes from its buffer's base that a per-lane store (BufferStore::per_lane) reaches, one range an invocation:
// each invocation of the dispatch computed on the host, one value at a time for all of them. Its loads read
// like the plan's other reads, through the runtime (a materialization memo sees them). False when it cannot
// tell: too many invocations, an ID or descriptor it does not follow, or a read that fails. The reference for
// LaneWriteRangesFast, which the GPU thread runs.
static bool LaneWriteRangesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                     const ResourceSnapshot& snapshot, const ResourcePlan::BufferStore& store,
                                     const std::unordered_map<const Inst*, uint32_t>& known,
                                     const ShaderBufferResource&                     target,
                                     std::vector<std::pair<uint64_t, uint64_t>>&     ranges) {
	// A larger dispatch keeps the binding whole: computing it costs the GPU thread more than its bound saves.
	// Marvel's Wolverine's scatters of GPU-scene updates run 64 invocations a dispatch, its others ~1,400.
	constexpr uint64_t MaxLanes   = 2048;
	const auto&        counts     = runtime.workgroup_counts;
	const auto&        size       = runtime.workgroup_size;
	const uint64_t     group_size = uint64_t {size[0]} * size[1] * size[2];
	const uint64_t     lanes      = group_size * counts[0] * counts[1] * counts[2];
	if (lanes == 0u || lanes > MaxLanes || runtime.read_specialization_memory == nullptr) return false;
	struct Column {
		uint32_t              width = 1; // a load of several dwords has them side by side
		std::vector<uint32_t> data;
	};
	std::unordered_map<const Inst*, Column>                   columns;
	std::deque<Column>                                        constants;
	std::vector<std::pair<uint64_t, std::array<uint32_t, 4>>> reads; // each address read once
	const auto uniform = [&](uint32_t word) {
		auto& result = constants.emplace_back();
		result.data.assign(lanes, word);
		return &result;
	};
	const Column* zero  = nullptr; // the missing operands of a one- or two-operand operation
	const auto    zeros = [&] { return zero != nullptr ? zero : (zero = uniform(0u)); };
	// An ID of every invocation, in dispatch order (group by group, local x fastest), with no division.
	const auto builtin = [&](StageInputKind kind, uint32_t axis, std::vector<uint32_t>& out) {
		if (kind != StageInputKind::LocalInvocationId && kind != StageInputKind::WorkgroupId &&
		    kind != StageInputKind::GlobalInvocationId && kind != StageInputKind::LocalInvocationIndex)
			return false;
		uint64_t lane = 0;
		for (uint32_t gz = 0; gz < counts[2]; gz++)
			for (uint32_t gy = 0; gy < counts[1]; gy++)
				for (uint32_t gx = 0; gx < counts[0]; gx++)
					for (uint32_t lz = 0, local = 0; lz < size[2]; lz++)
						for (uint32_t ly = 0; ly < size[1]; ly++)
							for (uint32_t lx = 0; lx < size[0]; lx++, local++) {
								const std::array<uint32_t, 3> l {lx, ly, lz}, g {gx, gy, gz};
								out[lane++] = kind == StageInputKind::LocalInvocationId ? l[axis]
								            : kind == StageInputKind::WorkgroupId       ? g[axis]
								            : kind == StageInputKind::GlobalInvocationId ? g[axis] * size[axis] + l[axis]
								                                                         : local;
							}
		return true;
	};
	std::function<const Column*(Value)> column = [&](Value value) -> const Column* {
		value = value.Resolve();
		if (value.IsImmediate()) return uniform(value.GetType() == Type::U1 ? value.U1() : value.U32());
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return nullptr;
		if (const auto found = known.find(inst); found != known.end()) return uniform(found->second);
		if (const auto found = columns.find(inst); found != columns.end()) return &found->second;
		const auto op = inst->GetOpcode();
		Column     result;
		result.data.resize(lanes);
		if (op == ValueOpcode::GetBuiltin) {
			const auto kind = inst->Arg(0).Resolve();
			const auto axis = inst->Arg(1).Resolve();
			if (!kind.IsImmediate() || !axis.IsImmediate() ||
			    !builtin(static_cast<StageInputKind>(kind.U32()), std::min(axis.U32(), 2u), result.data))
				return nullptr;
		} else if (op == ValueOpcode::LoadBufferU32 || op == ValueOpcode::LoadBufferU32x2 ||
		           op == ValueOpcode::LoadBufferU32x3 || op == ValueOpcode::LoadBufferU32x4) {
			const uint32_t width = op == ValueOpcode::LoadBufferU32 ? 1u : op == ValueOpcode::LoadBufferU32x2 ? 2u
			                       : op == ValueOpcode::LoadBufferU32x3 ? 3u : 4u;
			const auto index = inst->Flags<MemoryFlags>().index; // SrtReadFlags keeps it in the same place
			ShaderBufferResource source;
			if (index >= program.memory_info.size() || program.memory_info[index].resource >= snapshot.buffers.size() ||
			    !DecodeBufferDescriptor(snapshot.buffers[program.memory_info[index].resource], source) ||
			    source.SwizzleEnabled() || source.AddTid())
				return nullptr;
			const auto& memory = program.memory_info[index];
			const auto* indices = column(inst->Arg(1));
			const auto* offsets = column(inst->Arg(2));
			const auto* scalars = column(inst->Arg(3));
			const auto* enabled = column(inst->Arg(4));
			if (indices == nullptr || offsets == nullptr || scalars == nullptr || enabled == nullptr) return nullptr;
			result.width = width;
			result.data.assign(lanes * width, 0u);
			std::vector<uint64_t> addresses(lanes, UINT64_MAX);
			uint64_t              low = UINT64_MAX, high = 0;
			bool                  aligned = true;
			for (uint64_t lane = 0; lane < lanes; lane++) {
				// An inactive invocation or one past the records reads zeros, as robust buffer access gives.
				const uint64_t element = memory.idxen ? indices->data[lane] : 0u;
				const uint64_t offset  = (memory.offen ? offsets->data[lane] : 0u) + uint64_t {memory.offset};
				if (enabled->data[lane] == 0u || (source.Stride() != 0u && element >= source.NumRecords()) ||
				    (source.Stride() == 0u && offset + width * 4u > source.NumRecords()))
					continue;
				addresses[lane] = source.Base48() + element * source.Stride() + offset + scalars->data[lane];
				low             = std::min(low, addresses[lane]);
				high            = std::max(high, addresses[lane] + width * 4u);
				aligned         = aligned && addresses[lane] % 4u == 0u;
			}
			// An index list is a few hundred bytes in a row: one read of them all, not one each.
			std::vector<uint32_t> span;
			if (low != UINT64_MAX && aligned && high - low <= 4096u) {
				span.resize((high - low) / 4u);
				if (!runtime.read_specialization_memory(runtime.userdata, low, span)) return nullptr;
			}
			for (uint64_t lane = 0; lane < lanes; lane++) {
				const auto address = addresses[lane];
				if (address == UINT64_MAX) continue;
				auto* target = result.data.data() + lane * width;
				if (!span.empty()) {
					std::copy_n(span.begin() + static_cast<ptrdiff_t>((address - low) / 4u), width, target);
					continue;
				}
				auto found = std::ranges::find(reads, address, &decltype(reads)::value_type::first);
				if (found == reads.end()) {
					std::array<uint32_t, 4> words {};
					if (!runtime.read_specialization_memory(runtime.userdata, address, std::span(words.data(), width)))
						return nullptr;
					found = reads.insert(reads.end(), {address, words});
				}
				std::copy_n(found->second.begin(), width, target);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2 || op == ValueOpcode::CompositeExtractU32x3 ||
		           op == ValueOpcode::CompositeExtractU32x4) {
			const auto* vector    = column(inst->Arg(0));
			const auto  component = inst->Arg(1).Resolve();
			if (vector == nullptr || !component.IsImmediate() || component.U32() >= vector->width) return nullptr;
			for (uint64_t lane = 0; lane < lanes; lane++) result.data[lane] = vector->data[lane * vector->width + component.U32()];
		} else if (LaneOpcode(op)) {
			std::array<const uint32_t*, 3> args {};
			for (uint32_t index = 0; index < 3u; index++) {
				const auto* argument = index < inst->NumArgs() ? column(inst->Arg(index)) : zeros();
				if (argument == nullptr || argument->width != 1u) return nullptr;
				args[index] = argument->data.data();
			}
			// One loop for each operation, so it runs without a branch per invocation.
			auto* out = result.data.data();
			const auto apply = [&](auto f) {
				for (uint64_t lane = 0; lane < lanes; lane++) out[lane] = f(args[0][lane], args[1][lane], args[2][lane]);
			};
			const auto s = [](uint32_t value) { return static_cast<int32_t>(value); };
			switch (op) {
				case ValueOpcode::ConditionRef: apply([](uint32_t a, uint32_t, uint32_t) { return a; }); break;
				case ValueOpcode::IAdd32: apply([](uint32_t a, uint32_t b, uint32_t) { return a + b; }); break;
				case ValueOpcode::ISub32: apply([](uint32_t a, uint32_t b, uint32_t) { return a - b; }); break;
				case ValueOpcode::IMul32: apply([](uint32_t a, uint32_t b, uint32_t) { return a * b; }); break;
				case ValueOpcode::ShiftLeftLogical32: apply([](uint32_t a, uint32_t b, uint32_t) { return a << (b & 31u); }); break;
				case ValueOpcode::ShiftRightLogical32: apply([](uint32_t a, uint32_t b, uint32_t) { return a >> (b & 31u); }); break;
				case ValueOpcode::ShiftRightArithmetic32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(s(a) >> (b & 31u)); });
					break;
				case ValueOpcode::BitwiseAnd32: apply([](uint32_t a, uint32_t b, uint32_t) { return a & b; }); break;
				case ValueOpcode::BitwiseOr32: apply([](uint32_t a, uint32_t b, uint32_t) { return a | b; }); break;
				case ValueOpcode::BitwiseXor32: apply([](uint32_t a, uint32_t b, uint32_t) { return a ^ b; }); break;
				case ValueOpcode::UMin32: apply([](uint32_t a, uint32_t b, uint32_t) { return std::min(a, b); }); break;
				case ValueOpcode::UMax32: apply([](uint32_t a, uint32_t b, uint32_t) { return std::max(a, b); }); break;
				case ValueOpcode::SMin32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::min(s(a), s(b))); });
					break;
				case ValueOpcode::SMax32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::max(s(a), s(b))); });
					break;
				case ValueOpcode::IEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a == b; }); break;
				case ValueOpcode::INotEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != b; }); break;
				case ValueOpcode::ULessThan32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a < b; }); break;
				case ValueOpcode::ULessThanEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a <= b; }); break;
				case ValueOpcode::UGreaterThan32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a > b; }); break;
				case ValueOpcode::UGreaterThanEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a >= b; }); break;
				case ValueOpcode::SLessThan32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) < s(b); }); break;
				case ValueOpcode::SLessThanEqual32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) <= s(b); }); break;
				case ValueOpcode::SGreaterThan32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) > s(b); }); break;
				case ValueOpcode::SGreaterThanEqual32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) >= s(b); }); break;
				case ValueOpcode::LogicalNot: apply([](uint32_t a, uint32_t, uint32_t) -> uint32_t { return a == 0u; }); break;
				case ValueOpcode::LogicalAnd: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != 0u && b != 0u; }); break;
				case ValueOpcode::LogicalOr: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != 0u || b != 0u; }); break;
				case ValueOpcode::SelectU1:
				case ValueOpcode::SelectU32: apply([](uint32_t a, uint32_t b, uint32_t c) { return a != 0u ? b : c; }); break;
				default: return nullptr;
			}
		} else {
			return nullptr;
		}
		return &columns.emplace(inst, std::move(result)).first->second;
	};
	const auto* guard = column(store.guard);
	std::array<const Column*, 3> parts {};
	for (uint32_t part = 0; part < 3u; part++) parts[part] = column(store.address[part]);
	if (guard == nullptr || std::ranges::find(parts, nullptr) != parts.end()) return false;
	ranges.reserve(lanes);
	for (uint64_t lane = 0; lane < lanes; lane++) {
		const uint64_t element = parts[0]->data[lane];
		// The hardware drops a store past the records; an element is at most 16 bytes.
		if (guard->data[lane] == 0u || (target.Stride() != 0u && element >= target.NumRecords())) continue;
		const uint64_t begin = uint64_t {parts[2]->data[lane]} + store.immediate + parts[1]->data[lane] +
		                       element * target.Stride();
		ranges.emplace_back(begin, begin + 16u);
	}
	return true;
}

namespace {

// A column of LaneWriteRangesFast: in LaneArena::words, or one value for every invocation.
struct LaneColumn {
	uint32_t offset  = 0;
	uint32_t width   = 1;
	bool     uniform = false;
};

// LaneWriteRangesFast's storage, kept by each thread from call to call: no allocation once it has grown.
struct LaneArena {
	bool                                                  busy = false;
	std::vector<uint32_t>                                 words;
	std::vector<std::pair<const Inst*, LaneColumn>>       memo;
	std::vector<uint64_t>                                 addresses;
	std::vector<uint32_t>                                 span;
	std::array<std::vector<uint32_t>, 3>                  broadcast;
	std::unordered_map<uint64_t, std::array<uint32_t, 4>> reads;
};

thread_local LaneArena t_lane_arena;
thread_local bool      t_lane_reference = false;

} // namespace

void SetLaneWriteRangesReference(bool on) {
	t_lane_reference = on;
}

// LaneWriteRanges as LaneWriteRangesReference computes it, for the same reads in the same order: a value the same for
// every invocation is kept once (an immediate, a host value, and what only they feed), and the columns live in a
// per-thread arena instead of a map of vectors. The GPU thread runs it for every per-lane store (~4,000 dispatches a
// second at Trask).
static bool LaneWriteRangesFast(const ResourcePlan& program, const SrtRuntime& runtime, const ResourceSnapshot& snapshot,
                                const ResourcePlan::BufferStore&                store,
                                const std::unordered_map<const Inst*, uint32_t>& known,
                                const ShaderBufferResource&                     target,
                                std::vector<std::pair<uint64_t, uint64_t>>&     ranges) {
	constexpr uint64_t MaxLanes   = 2048; // as LaneWriteRangesReference
	const auto&        counts     = runtime.workgroup_counts;
	const auto&        size       = runtime.workgroup_size;
	const uint64_t     group_size = uint64_t {size[0]} * size[1] * size[2];
	const uint64_t     lanes      = group_size * counts[0] * counts[1] * counts[2];
	if (lanes == 0u || lanes > MaxLanes || runtime.read_specialization_memory == nullptr) return false;
	auto& arena = t_lane_arena;
	if (arena.busy) return LaneWriteRangesReference(program, runtime, snapshot, store, known, target, ranges);
	arena.busy = true;
	struct Release {
		LaneArena& arena;
		~Release() { arena.busy = false; }
	} release {arena};
	auto& words = arena.words;
	words.clear();
	arena.memo.clear();
	arena.reads.clear();
	const auto allocate = [&](uint64_t count) {
		const auto offset = static_cast<uint32_t>(words.size());
		words.resize(words.size() + count);
		return offset;
	};
	const auto uniform = [&](uint32_t word) {
		const LaneColumn result {.offset = allocate(1u), .width = 1u, .uniform = true};
		words[result.offset] = word;
		return result;
	};
	// A column's word of an invocation (its component of a vector).
	const auto at = [&](const LaneColumn& column, uint64_t lane, uint32_t component = 0u) {
		return words[column.offset + (column.uniform ? 0u : lane * column.width) + component];
	};
	const auto builtin = [&](StageInputKind kind, uint32_t axis, uint32_t* out) {
		if (kind != StageInputKind::LocalInvocationId && kind != StageInputKind::WorkgroupId &&
		    kind != StageInputKind::GlobalInvocationId && kind != StageInputKind::LocalInvocationIndex)
			return false;
		uint64_t lane = 0;
		for (uint32_t gz = 0; gz < counts[2]; gz++)
			for (uint32_t gy = 0; gy < counts[1]; gy++)
				for (uint32_t gx = 0; gx < counts[0]; gx++)
					for (uint32_t lz = 0, local = 0; lz < size[2]; lz++)
						for (uint32_t ly = 0; ly < size[1]; ly++)
							for (uint32_t lx = 0; lx < size[0]; lx++, local++) {
								const std::array<uint32_t, 3> l {lx, ly, lz}, g {gx, gy, gz};
								out[lane++] = kind == StageInputKind::LocalInvocationId ? l[axis]
								            : kind == StageInputKind::WorkgroupId       ? g[axis]
								            : kind == StageInputKind::GlobalInvocationId ? g[axis] * size[axis] + l[axis]
								                                                         : local;
							}
		return true;
	};
	const auto column = [&](auto& self, Value value) -> std::optional<LaneColumn> {
		value = value.Resolve();
		if (value.IsImmediate()) return uniform(value.GetType() == Type::U1 ? value.U1() : value.U32());
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return std::nullopt;
		if (const auto found = known.find(inst); found != known.end()) return uniform(found->second);
		for (const auto& [key, memo]: arena.memo) {
			if (key == inst) return memo;
		}
		const auto op = inst->GetOpcode();
		LaneColumn result;
		if (op == ValueOpcode::GetBuiltin) {
			const auto kind = inst->Arg(0).Resolve();
			const auto axis = inst->Arg(1).Resolve();
			if (!kind.IsImmediate() || !axis.IsImmediate()) return std::nullopt;
			result.offset = allocate(lanes);
			if (!builtin(static_cast<StageInputKind>(kind.U32()), std::min(axis.U32(), 2u), words.data() + result.offset))
				return std::nullopt;
		} else if (op == ValueOpcode::LoadBufferU32 || op == ValueOpcode::LoadBufferU32x2 ||
		           op == ValueOpcode::LoadBufferU32x3 || op == ValueOpcode::LoadBufferU32x4) {
			const uint32_t width = op == ValueOpcode::LoadBufferU32 ? 1u : op == ValueOpcode::LoadBufferU32x2 ? 2u
			                       : op == ValueOpcode::LoadBufferU32x3 ? 3u : 4u;
			const auto index = inst->Flags<MemoryFlags>().index;
			ShaderBufferResource source;
			if (index >= program.memory_info.size() || program.memory_info[index].resource >= snapshot.buffers.size() ||
			    !DecodeBufferDescriptor(snapshot.buffers[program.memory_info[index].resource], source) ||
			    source.SwizzleEnabled() || source.AddTid())
				return std::nullopt;
			const auto& memory  = program.memory_info[index];
			// All four first, as the reference reads them.
			const auto indices = self(self, inst->Arg(1));
			const auto offsets = self(self, inst->Arg(2));
			const auto scalars = self(self, inst->Arg(3));
			const auto enabled = self(self, inst->Arg(4));
			if (!indices || !offsets || !scalars || !enabled) return std::nullopt;
			// Every invocation reads the same: one address (the reference's span of it is the same one read).
			const bool same  = indices->uniform && offsets->uniform && scalars->uniform && enabled->uniform;
			const auto count = same ? 1u : lanes;
			result.width     = width;
			result.uniform   = same;
			result.offset    = allocate(count * width);
			auto& addresses  = arena.addresses;
			addresses.assign(count, UINT64_MAX);
			uint64_t low = UINT64_MAX, high = 0;
			bool     aligned = true;
			for (uint64_t lane = 0; lane < count; lane++) {
				const uint64_t element = memory.idxen ? at(*indices, lane) : 0u;
				const uint64_t offset  = (memory.offen ? at(*offsets, lane) : 0u) + uint64_t {memory.offset};
				if (at(*enabled, lane) == 0u || (source.Stride() != 0u && element >= source.NumRecords()) ||
				    (source.Stride() == 0u && offset + width * 4u > source.NumRecords()))
					continue;
				addresses[lane] = source.Base48() + element * source.Stride() + offset + at(*scalars, lane);
				low             = std::min(low, addresses[lane]);
				high            = std::max(high, addresses[lane] + width * 4u);
				aligned         = aligned && addresses[lane] % 4u == 0u;
			}
			auto& span = arena.span;
			span.clear();
			if (low != UINT64_MAX && aligned && high - low <= 4096u) {
				span.resize((high - low) / 4u);
				if (!runtime.read_specialization_memory(runtime.userdata, low, span)) return std::nullopt;
			}
			for (uint64_t lane = 0; lane < count; lane++) {
				const auto address = addresses[lane];
				if (address == UINT64_MAX) continue;
				auto* out = words.data() + result.offset + lane * width;
				if (!span.empty()) {
					std::copy_n(span.begin() + static_cast<ptrdiff_t>((address - low) / 4u), width, out);
					continue;
				}
				auto found = arena.reads.find(address);
				if (found == arena.reads.end()) {
					std::array<uint32_t, 4> read {};
					if (!runtime.read_specialization_memory(runtime.userdata, address, std::span(read.data(), width)))
						return std::nullopt;
					found = arena.reads.emplace(address, read).first;
				}
				std::copy_n(found->second.begin(), width, out);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2 || op == ValueOpcode::CompositeExtractU32x3 ||
		           op == ValueOpcode::CompositeExtractU32x4) {
			const auto vector    = self(self, inst->Arg(0));
			const auto component = inst->Arg(1).Resolve();
			if (!vector || !component.IsImmediate() || component.U32() >= vector->width) return std::nullopt;
			result.uniform = vector->uniform;
			const auto count = result.uniform ? 1u : lanes;
			result.offset  = allocate(count);
			for (uint64_t lane = 0; lane < count; lane++) words[result.offset + lane] = at(*vector, lane, component.U32());
		} else if (LaneOpcode(op)) {
			std::array<LaneColumn, 3> args {};
			for (uint32_t index = 0; index < 3u; index++) {
				const auto argument = index < inst->NumArgs() ? self(self, inst->Arg(index)) : uniform(0u);
				if (!argument || argument->width != 1u) return std::nullopt;
				args[index] = *argument;
			}
			result.uniform   = args[0].uniform && args[1].uniform && args[2].uniform;
			const auto count = result.uniform ? 1u : lanes;
			result.offset    = allocate(count);
			// A uniform operand of a varying operation is spread over the invocations, so each operation is one
			// loop without a branch per invocation.
			std::array<const uint32_t*, 3> in {};
			for (uint32_t index = 0; index < 3u; index++) {
				if (args[index].uniform && !result.uniform) {
					auto& spread = arena.broadcast[index];
					spread.assign(lanes, words[args[index].offset]);
					in[index] = spread.data();
				} else {
					in[index] = words.data() + args[index].offset;
				}
			}
			auto*      out   = words.data() + result.offset;
			const auto apply = [&](auto f) {
				for (uint64_t lane = 0; lane < count; lane++) out[lane] = f(in[0][lane], in[1][lane], in[2][lane]);
			};
			const auto s = [](uint32_t value) { return static_cast<int32_t>(value); };
			switch (op) {
				case ValueOpcode::ConditionRef: apply([](uint32_t a, uint32_t, uint32_t) { return a; }); break;
				case ValueOpcode::IAdd32: apply([](uint32_t a, uint32_t b, uint32_t) { return a + b; }); break;
				case ValueOpcode::ISub32: apply([](uint32_t a, uint32_t b, uint32_t) { return a - b; }); break;
				case ValueOpcode::IMul32: apply([](uint32_t a, uint32_t b, uint32_t) { return a * b; }); break;
				case ValueOpcode::ShiftLeftLogical32: apply([](uint32_t a, uint32_t b, uint32_t) { return a << (b & 31u); }); break;
				case ValueOpcode::ShiftRightLogical32: apply([](uint32_t a, uint32_t b, uint32_t) { return a >> (b & 31u); }); break;
				case ValueOpcode::ShiftRightArithmetic32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(s(a) >> (b & 31u)); });
					break;
				case ValueOpcode::BitwiseAnd32: apply([](uint32_t a, uint32_t b, uint32_t) { return a & b; }); break;
				case ValueOpcode::BitwiseOr32: apply([](uint32_t a, uint32_t b, uint32_t) { return a | b; }); break;
				case ValueOpcode::BitwiseXor32: apply([](uint32_t a, uint32_t b, uint32_t) { return a ^ b; }); break;
				case ValueOpcode::UMin32: apply([](uint32_t a, uint32_t b, uint32_t) { return std::min(a, b); }); break;
				case ValueOpcode::UMax32: apply([](uint32_t a, uint32_t b, uint32_t) { return std::max(a, b); }); break;
				case ValueOpcode::SMin32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::min(s(a), s(b))); });
					break;
				case ValueOpcode::SMax32:
					apply([&](uint32_t a, uint32_t b, uint32_t) { return static_cast<uint32_t>(std::max(s(a), s(b))); });
					break;
				case ValueOpcode::IEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a == b; }); break;
				case ValueOpcode::INotEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != b; }); break;
				case ValueOpcode::ULessThan32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a < b; }); break;
				case ValueOpcode::ULessThanEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a <= b; }); break;
				case ValueOpcode::UGreaterThan32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a > b; }); break;
				case ValueOpcode::UGreaterThanEqual32: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a >= b; }); break;
				case ValueOpcode::SLessThan32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) < s(b); }); break;
				case ValueOpcode::SLessThanEqual32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) <= s(b); }); break;
				case ValueOpcode::SGreaterThan32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) > s(b); }); break;
				case ValueOpcode::SGreaterThanEqual32: apply([&](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return s(a) >= s(b); }); break;
				case ValueOpcode::LogicalNot: apply([](uint32_t a, uint32_t, uint32_t) -> uint32_t { return a == 0u; }); break;
				case ValueOpcode::LogicalAnd: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != 0u && b != 0u; }); break;
				case ValueOpcode::LogicalOr: apply([](uint32_t a, uint32_t b, uint32_t) -> uint32_t { return a != 0u || b != 0u; }); break;
				case ValueOpcode::SelectU1:
				case ValueOpcode::SelectU32: apply([](uint32_t a, uint32_t b, uint32_t c) { return a != 0u ? b : c; }); break;
				default: return std::nullopt;
			}
		} else {
			return std::nullopt;
		}
		arena.memo.emplace_back(inst, result);
		return result;
	};
	// All four first, as the reference evaluates them.
	const auto guard = column(column, store.guard);
	std::array<std::optional<LaneColumn>, 3> parts;
	for (uint32_t part = 0; part < 3u; part++) parts[part] = column(column, store.address[part]);
	if (!guard || std::ranges::any_of(parts, [](const auto& part) { return !part.has_value(); })) return false;
	ranges.reserve(lanes);
	for (uint64_t lane = 0; lane < lanes; lane++) {
		const uint64_t element = at(*parts[0], lane);
		if (at(*guard, lane) == 0u || (target.Stride() != 0u && element >= target.NumRecords())) continue;
		const uint64_t begin = uint64_t {at(*parts[2], lane)} + store.immediate + at(*parts[1], lane) +
		                       element * target.Stride();
		ranges.emplace_back(begin, begin + 16u);
	}
	return true;
}

static bool LaneWriteRanges(const ResourcePlan& program, const SrtRuntime& runtime, const ResourceSnapshot& snapshot,
                            const ResourcePlan::BufferStore&                store,
                            const std::unordered_map<const Inst*, uint32_t>& known,
                            const ShaderBufferResource&                     target,
                            std::vector<std::pair<uint64_t, uint64_t>>&     ranges) {
	return t_lane_reference ? LaneWriteRangesReference(program, runtime, snapshot, store, known, target, ranges)
	                        : LaneWriteRangesFast(program, runtime, snapshot, store, known, target, ranges);
}

// The bytes from each buffer's base the dispatch's stores can reach: every address part's range
// over the workgroup and invocation IDs, with the values the host evaluates in place.
static void BufferWriteExtents(const ResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot) {
	auto& writes = snapshot.buffer_writes;
	writes.clear();
	snapshot.buffer_write_ranges.clear();
	const auto& counts = runtime.workgroup_counts;
	const auto& size   = runtime.workgroup_size;
	if (program.buffer_stores.empty() || counts.size() != 3u || size.size() != 3u) return;
	SrtWalker walker(program, runtime); // its own reads: the bounds do not specialize the shader
	std::array<uint64_t, 11> high {}; // each leaf's largest value: local, group, global xyz, flat local
	high[10] = 1u;
	for (uint32_t axis = 0; axis < 3u; axis++) {
		if (counts[axis] == 0u || size[axis] == 0u) return;
		high[1u + axis] = size[axis] - 1u;
		high[4u + axis] = counts[axis] - 1u;
		high[7u + axis] = uint64_t {counts[axis]} * size[axis] - 1u;
		high[10] *= size[axis];
	}
	high[10] -= 1u;
	const auto leaf = [](Value value) -> std::optional<size_t> {
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetBuiltin ||
		    !inst->Arg(0).Resolve().IsImmediate() || !inst->Arg(1).Resolve().IsImmediate()) return {};
		const auto kind = static_cast<StageInputKind>(inst->Arg(0).Resolve().U32());
		const auto axis = std::min(inst->Arg(1).Resolve().U32(), 2u);
		if (kind == StageInputKind::LocalInvocationId) return 1u + axis;
		if (kind == StageInputKind::WorkgroupId) return 4u + axis;
		if (kind == StageInputKind::GlobalInvocationId) return 7u + axis;
		if (kind == StageInputKind::LocalInvocationIndex) return 10u;
		return {};
	};
	writes.assign(program.info.buffers.size(), {0, UINT64_MAX});
	std::vector<uint8_t>                     state(writes.size()); // 0 none yet, 1 bounded, 2 whole
	std::unordered_map<const Inst*, uint32_t> known;
	// A buffer with a per-lane store: the ranges of all its stores, kept apart (buffer_write_ranges).
	std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges(writes.size());
	std::vector<uint8_t>                                    scattered(writes.size());
	for (const auto& store: program.buffer_stores) {
		auto& extent = writes[store.buffer];
		bool  bounded = state[store.buffer] != 2u;
		known.clear();
		for (const auto value: store.values) {
			uint32_t result = 0;
			bounded         = bounded && walker.Evaluate(value, result);
			if (const auto* inst = value.Resolve().TryInstruction(); inst != nullptr) known[inst] = result;
		}
		if (store.per_lane) {
			ShaderBufferResource descriptor;
			std::vector<std::pair<uint64_t, uint64_t>> reached;
			bounded = bounded && DecodeBufferDescriptor(snapshot.buffers[store.buffer], descriptor) &&
			          !descriptor.SwizzleEnabled() && !descriptor.AddTid() &&
			          LaneWriteRanges(program, runtime, snapshot, store, known, descriptor, reached);
			if (!bounded) {
				extent              = {0, UINT64_MAX};
				state[store.buffer] = 2u;
				continue;
			}
			// Kept sorted as it grows: an index list in order gives its ranges in order, and two stores merge.
			if (!std::ranges::is_sorted(reached)) std::ranges::sort(reached);
			auto&      list   = ranges[store.buffer];
			const auto middle = static_cast<ptrdiff_t>(list.size());
			list.insert(list.end(), reached.begin(), reached.end());
			std::inplace_merge(list.begin(), list.begin() + middle, list.end());
			scattered[store.buffer] = 1u;
			state[store.buffer]     = 1u;
			continue;
		}
		const auto resolve = [&](Value value) {
			const auto active = ResolveActiveU32(value, store.guard);
			value            = active.IsEmpty() ? value.Resolve() : active;
			const auto found = value.TryInstruction() != nullptr ? known.find(value.TryInstruction()) : known.end();
			return found != known.end() ? Value(found->second) : value;
		};
		constexpr uint64_t      cap = uint64_t {1} << 48u; // beyond any guest address: the whole buffer
		std::array<uint64_t, 3> first {}, last {};
		for (uint32_t part = 0; part < 3u && bounded; part++) {
			const auto affine = AffineIndex<11>(store.address[part], resolve, leaf);
			bounded           = affine.has_value();
			if (!bounded) break;
			first[part] = last[part] = (*affine)[0];
			for (size_t index = 1; index < affine->size() && bounded; index++) {
				const auto coefficient = (*affine)[index];
				bounded = coefficient == 0u || high[index] <= cap / coefficient;
				if (bounded) last[part] += coefficient * high[index];
			}
			bounded = bounded && last[part] <= cap;
		}
		ShaderBufferResource descriptor;
		bounded = bounded && DecodeBufferDescriptor(snapshot.buffers[store.buffer], descriptor) &&
		          !descriptor.SwizzleEnabled();
		if (!bounded) {
			extent              = {0, UINT64_MAX};
			state[store.buffer] = 2u;
			continue;
		}
		// The address of an element: SGPR offset + immediate + VGPR offset + index * stride; add_tid
		// adds the lane to the index. An element is at most 16 bytes.
		const uint64_t stride = descriptor.Stride();
		const uint64_t begin  = first[2] + store.immediate + first[1] + first[0] * stride;
		const uint64_t end = last[2] + store.immediate + last[1] + (last[0] + (descriptor.AddTid() ? 63u : 0u)) * stride + 16u;
		extent              = state[store.buffer] == 0u ? std::pair {begin, end}
		                                                : std::pair {std::min(extent.first, begin), std::max(extent.second, end)};
		state[store.buffer] = 1u;
		auto& list = ranges[store.buffer];
		list.insert(std::upper_bound(list.begin(), list.end(), std::pair {begin, end}), std::pair {begin, end});
	}
	snapshot.buffer_write_ranges.assign(writes.size(), {});
	for (size_t buffer = 0; buffer < writes.size(); buffer++) {
		if (scattered[buffer] == 0u || state[buffer] != 1u) continue;
		auto&  list   = ranges[buffer]; // sorted as it grew
		size_t merged = 0;
		for (const auto& range: list) {
			if (merged != 0u && range.first <= list[merged - 1u].second) {
				list[merged - 1u].second = std::max(list[merged - 1u].second, range.second);
			} else {
				list[merged++] = range;
			}
		}
		list.resize(merged);
		// Nothing reached is an empty extent: no byte of the binding becomes GPU-written.
		writes[buffer] = list.empty() ? std::pair<uint64_t, uint64_t> {0, 0}
		                              : std::pair {list.front().first, list.back().second};
		snapshot.buffer_write_ranges[buffer] = std::move(list);
	}
}

// A written buffer whose stores PlanBufferStores cannot bound is marked GPU-written whole at each dispatch;
// tracked, the dispatch also records the pages it writes. Not with BDA accesses, which write memory no
// binding names, nor for a buffer selected by an index, whose stores do not name their binding statically.
void MarkTrackedWrites(Program& program) {
	if (program.stage != ShaderType::Compute || program.info.uses_dma) return;
	const auto stores = PlanBufferStores(program, [](Value value) { return value; });
	// A buffer an indirect one selects can be written through a descriptor index the shader computes.
	std::vector<bool> selected(program.info.buffers.size());
	for (const auto& buffer: program.info.buffers) {
		for (const auto child: buffer.indirect_resources) {
			if (child < selected.size()) selected[child] = true;
		}
	}
	for (uint32_t index = 0; index < program.info.buffers.size(); ++index) {
		auto& buffer   = program.info.buffers[index];
		buffer.tracked = buffer.written && buffer.indirect_root == BufferResource::NoIndirectBuffer &&
		                 buffer.indirect_resources.empty() && !selected[index] &&
		                 std::ranges::none_of(stores, [&](const auto& store) { return store.buffer == index; });
	}
}

// The fetches of ResourcePlan::descriptor_fetch: a descriptor dword that is one of the plan's SRT reads, a user-data
// word or a 32-bit immediate. A read shares its flat slot with no other read; with control flow it must be refreshed by
// exactly one block.
static void BuildDescriptorFetch(ResourcePlan& plan) {
	using Fetch = ResourcePlan::DescriptorFetch;
	std::vector<uint32_t> writers(plan.srt_reads.size(), 0u); // by flat slot
	for (const auto& read: plan.srt_reads)
		if (read.flat_offset < writers.size()) ++writers[read.flat_offset];
	std::vector<uint32_t> blocks(plan.srt_reads.size(), UINT32_MAX); // by read: its block, UINT32_MAX - 1 for several
	for (uint32_t block = 0; block < plan.control_flow.size(); ++block) {
		for (const auto slot: plan.control_flow[block].srt_reads) {
			if (slot >= blocks.size()) continue;
			blocks[slot] = blocks[slot] == UINT32_MAX ? block : UINT32_MAX - 1u;
		}
	}
	std::unordered_map<const Inst*, uint32_t> reads; // read -> index in srt_reads
	for (uint32_t index = 0; index < plan.srt_reads.size(); ++index) {
		const auto& read = plan.srt_reads[index];
		const auto* inst = read.value.ResolveInstruction();
		if (inst == nullptr || read.flat_offset >= writers.size() || writers[read.flat_offset] != 1u) continue;
		if (!plan.control_flow.empty() && blocks[index] >= UINT32_MAX - 1u) continue;
		reads.emplace(inst, index);
	}
	plan.descriptor_fetch.assign(plan.descriptor_sources.size(), {});
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		const auto& descriptor = plan.descriptor_sources[source];
		for (uint32_t dword = 0; dword < descriptor.dword_count && dword < descriptor.dwords.size(); ++dword) {
			auto&      fetch = plan.descriptor_fetch[source][dword];
			const auto value = descriptor.dwords[dword].Resolve();
			if (value.IsImmediate()) {
				if (value.GetType() == Type::U32) fetch = {.kind = Fetch::Immediate, .index = value.U32()};
				continue;
			}
			const auto* inst = value.TryInstruction();
			if (inst == nullptr) continue;
			if (const auto found = reads.find(inst); found != reads.end()) {
				const auto& read = plan.srt_reads[found->second];
				fetch = {.kind  = Fetch::Flat,
				         .index = read.flat_offset,
				         .block = plan.control_flow.empty() ? UINT32_MAX : blocks[found->second],
				         .read  = inst};
			} else if (inst->GetOpcode() == ValueOpcode::GetUserData && inst->NumArgs() == 1 &&
			           inst->Arg(0).GetType() == Type::ScalarReg &&
			           RegIndex(inst->Arg(0).ScalarRegister()) >= plan.user_data_base) {
				fetch = {.kind = Fetch::UserData, .index = RegIndex(inst->Arg(0).ScalarRegister()) - plan.user_data_base};
			}
		}
	}
}

// The reads the host's own values depend on (509f7a346); every other read only fills the flat buffer, and a walk may
// leave it for the GPU to copy (SetSrtGpuFills).
static void MarkSrtHostReads(ResourcePlan& plan) {
	std::unordered_set<const Inst*> host_values;
	std::vector<Value>              pending;
	for (const auto& source: plan.descriptor_sources) {
		pending.insert(pending.end(), source.dwords.begin(), source.dwords.begin() + source.dword_count);
		if (!source.indirect_descriptor.has_value()) continue;
		const auto& indirect = *source.indirect_descriptor;
		pending.insert(pending.end(), {indirect.key_count, indirect.selector_first, indirect.selector_mask});
	}
	for (const auto& block: plan.control_flow) pending.push_back(block.condition);
	const auto& fill = plan.uniform_fill;
	pending.insert(pending.end(), fill.values.begin(), fill.values.end());
	pending.insert(pending.end(), {fill.pattern_size, fill.element_count});
	for (const auto& store: plan.buffer_stores) {
		pending.insert(pending.end(), store.address.begin(), store.address.end());
		pending.insert(pending.end(), store.values.begin(), store.values.end());
		pending.push_back(store.guard);
	}
	for (const auto& read: plan.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		for (size_t i = 0; inst != nullptr && i < inst->NumArgs(); ++i) pending.push_back(inst->Arg(i));
	}
	while (!pending.empty()) {
		const auto value = pending.back();
		pending.pop_back();
		const auto* inst = value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
		if (inst == nullptr || !host_values.insert(inst).second) continue;
		for (size_t i = 0; i < inst->NumArgs(); ++i) pending.push_back(inst->Arg(i));
	}
	plan.srt_read_host.resize(plan.srt_reads.size());
	for (size_t slot = 0; slot < plan.srt_reads.size(); ++slot) {
		const auto offset = plan.srt_reads[slot].flat_offset;
		// A global base: the renderer reads it from the flat buffer to back its pages (BackAddressBases).
		const bool base = std::ranges::any_of(plan.info.address_bases,
		                                      [&](const auto& pair) { return pair[0] == offset || pair[1] == offset; });
		plan.srt_read_host[slot] = base || host_values.contains(plan.srt_reads[slot].value.ResolveInstruction()) ? 1u : 0u;
	}
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = source->Arg(1).Resolve();
			EXIT_IF(!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			        slot.U32() >= program.srt_reads.size());
			return Clone(program.srt_reads[slot.U32()].value);
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
		if (source->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		    source->GetOpcode() == ValueOpcode::ReadConstBuffer ||
		    source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			target.SetFlags(SrtReadFlags {.index = source->Flags<MemoryFlags>().index});
		}
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

	bool capture_indirect_reads = false;
	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			plan.requires_specialization_memory = true;
			capture_indirect_reads = true;
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
	// Destination tables share keys when their complete selector domains agree.
	// Resolve those domains lazily at dispatch time, without repeated guest reads.
	plan.selector_groups.resize(plan.descriptor_sources.size(), UINT32_MAX);
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source].indirect_descriptor;
		if (!descriptor || !descriptor->selector) continue;
		const auto& a = *descriptor;
		uint32_t group = 0;
		for (; group < plan.selector_keys.size(); ++group) {
			const auto& b = *program.descriptor_sources[plan.selector_keys[group].source].indirect_descriptor;
			if (EquivalentSelectorDomain(program, a, b)) break;
		}
		if (group == plan.selector_keys.size()) plan.selector_keys.push_back({.source = source});
		plan.selector_groups[source] = group;
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// Retain integer control expressions with opaque varying leaves. Their identities
	// matter when an EXEC factor also guards a selected value; never clone float work.
	bool capture_control_reads = false;
	bool uniform_control = false;
	const auto control_value_count = plan.value_storage.size();
	struct ControlValue {
		Value value;
		bool complete = false;
	};
	std::unordered_map<const Inst*, ControlValue> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			uniform_control |= value.GetType() == Type::U1;
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		}
		const auto [found, inserted] = predicates.try_emplace(inst);
		auto& entry = found->second;
		auto& result = entry.value;
		const auto opaque = [&]() -> Value {
			if (value.GetType() != Type::U1 && value.GetType() != Type::U32) return {};
			return result = Value(&plan.value_storage.emplace_back(
			    value.GetType() == Type::U1 ? ValueOpcode::UndefU1 : ValueOpcode::UndefU32));
		};
		// A recursive projection remains opaque, including unstructured loop Phis.
		if (!inserted) return entry.complete || !result.IsEmpty() ? result : opaque();
		const auto finish = [&](Value projected) {
			entry.complete = true;
			return result = projected;
		};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			uniform_control = true;
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return finish(Clone(value));
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef) return finish(ClonePredicate(inst->Arg(0)));
		if (op == ValueOpcode::Phi) {
			if (inst->Parent() == nullptr || inst->Parent()->terminator.loop_header ||
			    inst->NumPhiBlocks() != inst->NumArgs() || inst->NumArgs() == 0u ||
			    (value.GetType() != Type::U1 && value.GetType() != Type::U32)) return finish(opaque());
			std::vector<uint32_t> predecessors;
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				const auto predecessor = std::ranges::find(program.blocks, inst->PhiBlock(index));
				if (predecessor == program.blocks.end()) return finish(opaque());
				predecessors.push_back(static_cast<uint32_t>(predecessor - program.blocks.begin()));
			}
			auto& node = plan.value_storage.emplace_back(op, inst->Flags<uint64_t>());
			plan.control_phi_predecessors.emplace(&node, std::move(predecessors));
			for (size_t index = 0; index < inst->NumArgs(); ++index)
				node.AddPhiOperand(nullptr, ClonePredicate(inst->Arg(index)));
			return finish(Value(&node));
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr &&
		    op != ValueOpcode::LogicalNot && op != ValueOpcode::IEqual32 &&
		    op != ValueOpcode::INotEqual32 && op != ValueOpcode::ULessThan32 &&
		    op != ValueOpcode::ULessThanEqual32 &&
		    op != ValueOpcode::SelectU1 && op != ValueOpcode::SelectU32) return finish(opaque());
		auto& node = plan.value_storage.emplace_back(op);
		for (size_t index = 0; index < inst->NumArgs(); ++index)
			node.SetArg(index, ClonePredicate(inst->Arg(index)));
		return finish(Value(&node));
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	if (!uniform_control) {
		plan.control_flow.clear();
		plan.control_phi_predecessors.clear();
		auto first = std::next(plan.value_storage.begin(), control_value_count);
		for (auto it = first; it != plan.value_storage.end(); ++it) it->Invalidate();
		plan.value_storage.erase(first, plan.value_storage.end());
	}
	plan.uniform_fill = program.uniform_fill.fill.kind != UniformFillKind::None
	    ? program.uniform_fill : AnalyzeUniformFill(program);
	for (auto& value: plan.uniform_fill.values) value = Clone(value);
	plan.uniform_fill.pattern_size = Clone(plan.uniform_fill.pattern_size);
	plan.uniform_fill.element_count = Clone(plan.uniform_fill.element_count);
	// Bounds on stores must not change what the plan reads or specializes on. A store value that needs
	// a buffer load (Marvel's Wolverine CS 5e04a15a8ee16915) leaves the plan's reads as they were:
	// BufferWriteExtents reads it at dispatch through the runtime, outside the specialization, and a
	// read that fails there keeps the whole binding.
	const auto reads = std::pair {plan.requires_specialization_memory, plan.capture_specialization_reads};
	plan.buffer_stores = PlanBufferStores(program, Clone);
	std::tie(plan.requires_specialization_memory, plan.capture_specialization_reads) = reads;
	plan.capture_specialization_reads |=
	    capture_indirect_reads || capture_control_reads || !plan.source_reads.empty();
	if (plan.capture_specialization_reads) {
		// Every former flat alias now targets its retained raw read. Mark those reads
		// on the normalized graph, so discarded Phi/EXEC dependencies stay discarded.
		std::unordered_set<const Inst*> planned_reads;
		for (const auto& read: plan.srt_reads) {
			auto* inst = read.value.ResolveInstruction();
			EXIT_IF(inst == nullptr || (inst->GetOpcode() != ValueOpcode::LoadAddressU32 &&
			                           inst->GetOpcode() != ValueOpcode::ReadConstBuffer));
			planned_reads.insert(inst);
		}
		for (const auto& source: plan.descriptor_sources) {
			if (!source.indirect_descriptor.has_value()) continue;
			const auto& indirect = *source.indirect_descriptor;
			if (indirect.selector) {
				MarkCleanReads(planned_reads, Source(plan, indirect.selector->source));
				if (indirect.selector->rows)
					MarkCleanReads(planned_reads, Source(plan, indirect.selector->rows->source));
			}
			MarkCleanReads(planned_reads, nullptr, indirect.selector_mask);
			MarkCleanReads(planned_reads, nullptr, indirect.selector_first);
			MarkCleanReads(planned_reads, nullptr, indirect.key_count);
			if (indirect.sources.empty()) {
				MarkCleanReads(planned_reads, Source(plan, indirect.table_source));
			} else {
				for (const auto candidate: indirect.sources) {
					MarkCleanReads(planned_reads, Source(plan, candidate));
				}
			}
		}
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanReads(planned_reads, Source(plan, buffer.source));
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanReads(planned_reads, Source(plan, image.source));
		}
	}
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	BuildDescriptorFetch(plan);
	MarkSrtHostReads(plan);
	CompileSrtWalker(plan);
	return plan;
}

static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                                     ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                     const std::vector<uint32_t>* walked);

bool BufferTableWanted(const ResourcePlan& program) {
	static const bool off = [] {
		const char* v = std::getenv("KYTY_TABLE_OFF");
		return v != nullptr && v[0] == '1';
	}();
	return !off && !program.info.buffers.empty();
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          const std::vector<uint32_t>* walked) {
	const bool materialized = MaterializeResourcesImpl(program, runtime, snapshot, specialization, walked);
	if (materialized) specialization.buffer_table = BufferTableWanted(program);
	return materialized;
}

static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                                     ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                     const std::vector<uint32_t>* walked) {
	for (auto& keys: program.selector_keys) keys.resolved = false;
	if (!program.resource_tracking_complete ||
	    ((program.requires_specialization_memory || !program.source_reads.empty()) &&
	     runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	if (!program.image_reuse) program.image_reuse.reset(new IndirectImageReuse);
	auto& reuse = *program.image_reuse;
	if (reuse.roots.size() < program.info.images.size()) reuse.roots.resize(program.info.images.size());
	++reuse.call;
	reuse.whole.clear();
	const auto uses_workgroup_count = [](const DescriptorSource& source) {
		return source.indirect_descriptor && source.indirect_descriptor->workgroup_axis != UINT32_MAX;
	};
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty() ||
	    (runtime.indirect_args_address != 0u &&
	     std::ranges::any_of(program.descriptor_sources, uses_workgroup_count));
	auto& reads = snapshot.specialization_reads;
	reads = program.source_reads;
	SrtReadCapture capture {runtime, reads};
	auto observed = capture_reads ? capture.ObservedRuntime() : runtime;
	SrtWalker clean(program, CleanRuntime(observed));
	SrtWalker walker(program, observed,
	                 capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	// The x86 walker loads guest memory itself and records every load's address; where it knows no
	// value the walkers above evaluate it. Without direct reads (the ahead workers) or when the walk
	// fails, they refresh as before.
	if (runtime.direct_reads && walked != nullptr) {
		snapshot.flattened_srt = *walked;
		if (capture_reads) WalkedLoads(program, reads);
		program.srt_walked = true;
	} else {
		program.srt_walked = runtime.direct_reads &&
		                     WalkSrt(program, runtime, snapshot.flattened_srt, capture_reads ? &reads : nullptr);
	}
	// The walk that gave the flat buffer (this one, or the repeat check's) left its GPU copies on the plan.
	snapshot.gpu_fills.clear();
	if (program.srt_walked) snapshot.gpu_fills = program.srt_fills;
	if (program.srt_walked) {
		walker.AdoptWalk(snapshot.flattened_srt);
	} else if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		return SpecializationFail(fmt::format("SRT refresh complete={} reads={} user_data={}",
		    program.srt_plan_complete, program.srt_reads.size(), runtime.user_data.size()));
	}
	const auto active        = std::span<const uint8_t>(program.active_sources);
	const auto active_images = std::span<const uint8_t>(program.active_images);
	std::array<uint32_t, 3> workgroup_counts {};
	if (runtime.indirect_args_address != 0u) {
		for (uint32_t source = 0; source < program.descriptor_sources.size(); ++source) {
			if ((!active.empty() && !active[source]) ||
			    !uses_workgroup_count(program.descriptor_sources[source])) continue;
			if ((runtime.indirect_args_address & 3u) != 0u ||
			    !ReadTable(runtime.indirect_args_address, sizeof(workgroup_counts), 0u,
			               observed, workgroup_counts)) return false;
			observed.workgroup_counts = workgroup_counts;
			break;
		}
	}
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	if (uniform_fill && !fill.pattern_size.IsEmpty())
		uniform_fill = clean.Evaluate(fill.pattern_size, words) && words >= 1 && words <= 4;
	uint32_t elements = 0;
	if (uniform_fill && !fill.element_count.IsEmpty())
		uniform_fill = clean.Evaluate(fill.element_count, elements) && elements != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
		snapshot.uniform_fill.bounded_elements = elements;
	}
	if (fill.fill.kind == UniformFillKind::Pattern) snapshot.uniform_fill = fill.fill; // the renderer reads its value
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value) ||
			    SpecializationFail(fmt::format("descriptor evaluation source={}", source));
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	const auto resource_failed = [&](std::string_view kind, uint32_t resource, uint32_t source,
	                                 size_t candidates) {
		return SpecializationFail(fmt::format("indirect {} resource={} source={} candidates={}",
		    kind, resource, source, candidates));
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
					        program, base.source, i, 4u, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers,
					        [&](DescriptorValue& value) {
						        return NormalizeIndirectBuffer(
						            value, source->indirect_descriptor->indexed, base);
					        }))
						return resource_failed("buffer", i, base.source, snapshot.buffers.size());
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
		buffer.scalar_empty       = base.scalar && descriptor.GetSize() == 0u;
	}
	BufferWriteExtents(program, runtime, snapshot);
	snapshot.images.resize(program.info.images.size());
	snapshot.image_heap   = {};
	snapshot.sampler_heap = {};
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube = image.cube,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return false;
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if ((!active.empty() && !active[image.source]) || (!active_images.empty() && !active_images[i])) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			if (MaterializeHeapImage(program, indirect, i, observed, clean, snapshot, specialization)) continue;
			if (!MaterializeIndirectDescriptor(
			        program, image.source, i, 8u, image.r128 ? 4u : 8u, observed, clean, snapshot, snapshot.images,
			        specialization.images, [&](DescriptorValue& value) {
				        // A broad heap also contains resources for other typed image operations.
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128) ||
				            (bounded_table && DescriptorDimension(value, image.dimension) != image.dimension))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return resource_failed("image", i, image.source, snapshot.images.size());
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return false;
			}
			// An image listed only by unreached blocks reads zero, not its source's descriptor.
			if ((!active_images.empty() && !active_images[i]) ||
			    !ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < snapshot.samplers.size(); ++i) {
		const auto base_index =
		    i < program.info.samplers.size() ? i : specialization.samplers[i].indirect_root;
		if (base_index >= program.info.samplers.size()) return false;
		if (i < program.info.samplers.size()) {
			auto source = program.info.samplers[i].source;
			const auto* descriptor = Source(program, source);
			if (descriptor == nullptr) return false;
			const auto* indirect = descriptor->indirect_descriptor ? &*descriptor->indirect_descriptor : nullptr;
			if (indirect != nullptr && indirect->sources.empty()) {
				snapshot.samplers[i] = {.dword_count = 4u};
				if (MaterializeHeapSampler(program, *indirect, i, observed, clean, snapshot, specialization)) continue;
				if ((active.empty() || active[source]) && !MaterializeIndirectDescriptor(
				        program, source, i, 4u, 4u, observed, clean, snapshot, snapshot.samplers,
				        specialization.samplers,
				        [](DescriptorValue&) { return true; }))
					return resource_failed("sampler", i, source, snapshot.samplers.size());
			} else {
				if (indirect != nullptr) source = indirect->sources[0];
				if (!evaluate(source, snapshot.samplers[i])) return false;
			}
		}
		if (program.info.samplers[base_index].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			const auto perf_mip = (snapshot.samplers[i].dwords[1] >> 24u) & 15u;
			// MipNone always selects the base level. Point/linear selection requires zero
			// primary/secondary bias; linear mip snapping is not implemented.
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u) ||
			    (filter == 2u && perf_mip != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires None, unbiased Point, or unbiased Linear without mip snapping");
			}
		}
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	if (!BuildResourceSpecialization(program, snapshot, specialization)) return false;
	ApplyImageForms(program, snapshot, specialization);
	return true;
}

uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value) {
	// Mirrors BuildResourceSpecialization for sampled roots: entries it would reject, entries of
	// another dimension and entries that need shader-side conversion read as null instead of
	// failing the draw.
	const auto format = static_cast<Prospero::BufferFormat>((value.dwords[1] >> 20u) & 0x1ffu);
	const auto tile = static_cast<Prospero::TileMode>((value.dwords[3] >> 20u) & 0x1fu);
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (NullImageDescriptor(value) || !ValidImageDescriptor(value, root.r128) ||
	    DescriptorDimension(value, root.dimension) != root.dimension ||
	    Prospero::IsFmaskTextureFormat(format) || format == Prospero::BufferFormat::k11_11_10UNorm ||
	    ImageConversionFormat(format) != Prospero::BufferFormat::kInvalid ||
	    (root.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
	     tile == Prospero::TileMode::kDepth) ||
	    numeric_class == Prospero::TextureNumericClass::Unsupported ||
	    (root.depth_compare && numeric_class != Prospero::TextureNumericClass::Float)) {
		value.dwords.fill(0u);
		return 0u;
	}
	return 1u + static_cast<uint32_t>(numeric_class) * 2u + (DescriptorIsCube(value) ? 1u : 0u);
}

ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind) {
	ImageResource resource;
	resource.resource_class         = ImageResourceClass::Sampled;
	resource.read                   = true;
	resource.heap                   = true;
	resource.dimension              = root.dimension;
	resource.unfiltered_access_only = root.unfiltered_access_only;
	// A heap slot's view serves every shader that reads the heap, whatever levels they read.
	resource.base_level_only        = false;
	resource.depth_compare          = root.depth_compare;
	resource.r128                   = root.r128;
	if (kind != 0u) {
		resource.numeric_class = static_cast<Prospero::TextureNumericClass>((kind - 1u) / 2u);
		resource.cube          = ((kind - 1u) & 1u) != 0u;
	}
	return resource;
}

void DropBoundDescriptorWords(Program& program) {
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uint32_t first = 0;
			if (op == ValueOpcode::GetBufferResource) {
				if (std::ranges::any_of(inst.Uses(), [&](const Use& use) {
					return program.memory_info[use.user->Flags<MemoryFlags>().index].kind == ResourceKind::IndirectBuffer;
				})) {
					continue;
				}
				const auto resource = inst.Flags<uint32_t>();
				first = resource < program.info.buffers.size() && program.info.buffers[resource].indirect_root == resource
				            ? 1u : 0u;
			} else if (op == ValueOpcode::GetImageResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < program.info.images.size() && program.info.images[resource].indirect_root == resource
				            ? 1u : 0u;
			} else if (op == ValueOpcode::GetSamplerResource) {
				// A heap sampler has no candidates, but the emitter reads its record by key (LoadHeapSampler):
				// without the key every heap sampler read record 0.
				const auto resource = inst.Flags<uint32_t>();
				first = resource < program.info.samplers.size() &&
				                (!program.info.samplers[resource].indirect_resources.empty() ||
				                 program.info.samplers[resource].guest_heap)
				            ? 1u : 0u;
			} else {
				continue;
			}
			for (size_t index = first; index < inst.NumArgs(); index++) {
				inst.SetArg(index, Value(0u));
			}
		}
	}
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size() ||
	        program.info.samplers.size() > specialization.samplers.size());

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
		image.srgb_decode                = source.srgb_decode;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.indirect_capacity          = source.indirect_capacity;
		image.cube                       = source.cube;
		image.skip_degamma               = source.skip_degamma;
		image.heap                       = source.heap;
		image.guest_heap                 = source.guest_heap;
		image.form_slot                  = source.indirect_mapping_offset == ImageFormMapping
		                                       ? source.indirect_search_iterations
		                                       : ImageResource::NoFormSlot;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto base_sampler_count = samplers.size();
	samplers.reserve(specialization.samplers.size());
	for (uint32_t index = 0; index < specialization.samplers.size(); ++index) {
		const auto& source = specialization.samplers[index];
		if (index >= samplers.size()) {
			EXIT_IF(source.indirect_root >= base_sampler_count);
			samplers.push_back(samplers[source.indirect_root]);
			samplers.back().indirect_resources.clear();
		}
		samplers[index].indirect_root = source.indirect_root;
		samplers[index].indirect_mapping_offset = source.indirect_mapping_offset;
		samplers[index].indirect_search_iterations = source.indirect_search_iterations;
		samplers[index].guest_heap                 = source.guest_heap;
	}
	for (uint32_t index = 0; index < samplers.size(); ++index) {
		const auto root = samplers[index].indirect_root;
		if (root != SamplerResource::NoIndirectSampler) {
			EXIT_IF(root >= samplers.size());
			samplers[root].indirect_resources.push_back(index);
		}
	}
	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	const auto original_sampler_count = samplers.size();
	samplers.reserve(sampler_plan.bindings.size());
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::Integer;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	// Resolve empty scalar-table words before folding the indirect descriptors that use them.
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.kind == ResourceKind::ScalarBuffer &&
			    specialization.buffers[memory.resource].scalar_empty &&
			    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer)
				inst.ReplaceUsesWith(Value(0u));
		}
	}
	const ImageRemap image_remap(specialization);
	// The image memory entries a remaining image instruction uses. Tracking gives only those an image index: an entry
	// whose instructions dead-code elimination removed keeps the translator's T# register number (s44 -> 11), which can
	// be past the image list (PPSA03671 pixel shaders 5b68b75638ea51a3, c8eb03340a65ab88, dffe92699dfd73c2).
	std::vector<bool> used_image_memory(memory_info.size());
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				const bool native_oob = memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob &&
				    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer;
				const bool null_indirect = memory.kind == ResourceKind::IndirectBuffer &&
				    NullBufferDescriptor(inst.Arg(0));
				if (native_oob || null_indirect) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds. An all-zero V# is unbound.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (native_oob && memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						const bool d16 = memory.data_bits == 16u;
						const auto components = d16 ? memory.component_count : count;
						std::array<uint32_t, 4> constants {};
						for (uint32_t component = 0; component < components; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							if (d16 && one == 0x3f800000u) one = 0x3c00u;
							constants[d16 ? component / 2u : component] |=
							    one << (d16 ? (component % 2u) * 16u : 0u);
						}
						for (uint32_t component = 0; component < count; component++) {
							if (constants[component] == 0u) continue;
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(constants[component]), Value(0u)}));
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
			used_image_memory[index] = true;
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
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
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageGatherRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions &&
			        inst.GetOpcode() != ValueOpcode::ImageRead &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite);
		}
	}
	for (uint32_t index = 0; index < memory_info.size(); index++) {
		auto& memory = memory_info[index];
		if (memory.kind == ResourceKind::Image && !memory.planning_only && used_image_memory[index]) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
}

namespace {

using ControlPhis = decltype(ResourcePlan::control_phi_predecessors);

// Instructions come first, so that any value can refer to them by index, then their arguments. Plan Phis have no
// blocks; a control Phi lists the plan blocks of its arguments after them.
void IoValues(Writer& writer, const std::list<Inst>& values, const ControlPhis& phis) {
	IoAll(writer, static_cast<uint32_t>(values.size()));
	for (const auto& inst: values) {
		writer.index.emplace(&inst, static_cast<uint32_t>(writer.index.size()));
		IoAll(writer, inst.GetOpcode(), inst.Flags<uint64_t>());
	}
	for (const auto& inst: values) {
		std::vector<Value> args;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			writer.ok = writer.ok && (inst.GetOpcode() != ValueOpcode::Phi || inst.PhiBlock(index) == nullptr);
			args.push_back(inst.Arg(index));
		}
		IoAll(writer, args);
		if (inst.GetOpcode() != ValueOpcode::Phi) continue;
		const auto found = phis.find(&inst);
		IoAll(writer, found != phis.end() ? found->second : std::vector<uint32_t> {});
	}
}

void IoValues(Reader& reader, std::list<Inst>& values, ControlPhis& phis) {
	uint32_t count = 0;
	Io(reader, count);
	count = reader.Count(count);
	for (uint32_t index = 0; index < count && reader.ok; ++index) {
		ValueOpcode opcode {};
		uint64_t    flags = 0;
		IoAll(reader, opcode, flags);
		reader.ok = reader.ok && opcode < ValueOpcode::Count;
		if (reader.ok) reader.insts.push_back(&values.emplace_back(opcode, flags));
	}
	for (auto* inst: reader.insts) {
		std::vector<Value> args;
		IoAll(reader, args);
		if (!reader.ok) return;
		if (inst->GetOpcode() == ValueOpcode::Phi) {
			std::vector<uint32_t> predecessors;
			IoAll(reader, predecessors);
			for (const auto arg: args) inst->AddPhiOperand(nullptr, arg);
			if (!predecessors.empty()) phis.emplace(inst, std::move(predecessors));
			continue;
		}
		// An argument count the opcode does not have: a record of other code.
		reader.ok = args.size() == inst->NumArgs();
		if (!reader.ok) return;
		for (size_t index = 0; index < args.size(); ++index) inst->SetArg(index, args[index]);
	}
}

// Binds every plan field: one added later fails to compile here until it is stored, derived or listed as scratch.
template <class S, class P> void IoPlan(S& io, P& plan) {
	auto& [stage, shader_hash, user_data_base, user_data_count, values, memory_info, descriptor_sources,
	       selector_groups, control_flow, control_phis, srt_reads,
	       srt_read_host, // derived on load (MarkSrtHostReads)
	       descriptor_fetch, // derived on load (BuildDescriptorFetch)
	       requires_specialization_memory, source_reads, capture_specialization_reads, srt_plan_complete,
	       resource_tracking_complete, info, uniform_fill, buffer_stores,
	       // Derived on load (CompileSrtWalker).
	       srt_code, srt_user_data, srt_captures, srt_outputs, srt_reach, srt_scratch, srt_output_index, srt_loads,
	       srt_walked, srt_fill_address, srt_fills,
	       // The GPU thread's scratch, and the selector keys derived on load.
	       evaluation_contexts, evaluation_value_count, active_sources, active_images, visited_blocks,
	       reachable_blocks, pending_blocks, material_keys, selector_keys, srt_trace, trace, trace_misses, trace_skip,
	       trace_replays, trace_values, trace_ok, image_reuse] = plan;
	IoValues(io, values, control_phis);
	IoAll(io, stage, shader_hash, user_data_base, user_data_count, memory_info, descriptor_sources, selector_groups,
	      control_flow, srt_reads, requires_specialization_memory, source_reads, capture_specialization_reads,
	      srt_plan_complete, resource_tracking_complete, info, uniform_fill, buffer_stores);
}

// What ExtractResourcePlan derives from the stored fields: one key set per selector group (extraction numbers the
// groups densely in order of first use, false when they are not), the descriptor fetches and the walker.
bool DerivePlanState(ResourcePlan& plan) {
	for (uint32_t source = 0; source < plan.selector_groups.size(); ++source) {
		const auto group = plan.selector_groups[source];
		if (group == plan.selector_keys.size()) plan.selector_keys.push_back({.source = source});
		if (group != UINT32_MAX && group >= plan.selector_keys.size()) return false;
	}
	BuildDescriptorFetch(plan);
	MarkSrtHostReads(plan);
	CompileSrtWalker(plan);
	return true;
}

} // namespace

std::vector<uint8_t> SaveResourcePlan(const ResourcePlan& plan) {
	Writer writer;
	IoPlan(writer, plan);
	return writer.ok ? std::move(writer.bytes) : std::vector<uint8_t> {};
}

bool LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan) {
	Reader reader {bytes};
	IoPlan(reader, plan);
	return reader.ok && reader.bytes.empty() && DerivePlanState(plan);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
