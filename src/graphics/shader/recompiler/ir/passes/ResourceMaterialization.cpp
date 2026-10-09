#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"
#include "native-preparation-scratch.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

extern "C" {
volatile std::atomic<uint32_t> kyty_local_specialization_guard_mode {0};
}

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectImageProbes = 65536u;
constexpr uint32_t DenseDescriptorShift   = 5u;

struct IndirectImage {
	uint32_t                     resource = 0;
	std::vector<uint32_t>        keys;
	std::vector<uint32_t>        candidates;
	std::vector<DescriptorValue> descriptors;
	uint32_t                     dropped_candidates = 0;
	uint32_t                     dropped_shapes     = 0;
	std::string                  dropped_summary;
};

struct MaterializedSnapshot {
	ResourceSnapshot           resources;
	std::vector<IndirectImage> indirect_images;
};

struct MaterializeWorkspace {
	MaterializedSnapshot materialized;
	ResourceSpecialization specialization;
	std::vector<DescriptorValue> values;
	std::vector<uint32_t> flattened_srt;
	std::vector<uint8_t> active_sources;
	ResourceSpecializationGuard guard_candidate;
	std::vector<uint32_t> padded_candidates; // BuildResourceSpecialization: per indirect table
};

bool SpecializationFail(MaterializeReport* report, std::string_view message) {
	if (report != nullptr) {
		report->reason = fmt::format("specialization failed: {}", message);
	}
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

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
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
	if (((descriptor.dwords[4] >> 16u) & 0x1fffu) > (descriptor.dwords[4] & 0x1fffu)) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

bool RequiresPointSampler(const ImageResource& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Sint ||
	       image.conversion_format != Prospero::BufferFormat::kInvalid;
}

bool RequiresPointSampler(const ResourceSpecialization::Image& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Sint ||
	       image.conversion_format != Prospero::BufferFormat::kInvalid;
}

bool ReservedImageBitsClear(const DescriptorValue& descriptor) {
	constexpr std::array<uint32_t, 8> reserved = {0x00000000u, 0x20000000u, 0xf0003000u,
	                                              0x00000000u, 0xe000e000u, 0xf9000000u,
	                                              0x00007b00u, 0x00000000u};
	for (uint32_t dword = 0; dword < reserved.size(); dword++) {
		if ((descriptor.dwords[dword] & reserved[dword]) != 0u) {
			return false;
		}
	}
	return true;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t StorageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::DynamicStorage || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return base <= last ? last - base + 1u : 0u;
}

struct IndirectImageClass {
	Decoder::ImageDimension dimension    = Decoder::ImageDimension::Unknown;
	Prospero::BufferFormat  conversion   = Prospero::BufferFormat::kInvalid;
	uint32_t                swizzle      = 0;
	uint32_t                mip_count    = 0;
	bool                    cube         = false;
	Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;

	bool operator==(const IndirectImageClass& other) const = default;
};

IndirectImageClass DescriptorClass(const ImageResource& image, const DescriptorValue& descriptor) {
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	IndirectImageClass result;
	result.dimension  = DescriptorDimension(descriptor, image.dimension);
	result.conversion = ImageConversionFormat(format);
	result.swizzle    = result.conversion == Prospero::BufferFormat::kInvalid
	                        ? 0u
	                        : DescriptorImageSwizzle(descriptor);
	result.mip_count  = StorageMipCount(image, descriptor);
	result.cube       = DescriptorIsCube(descriptor);
	result.numeric_class = Prospero::SampledTextureNumericClass(format);
	return result;
}

std::string ClassText(const IndirectImageClass& value) {
	return fmt::format("dim={} cube={} class={} conv={} swizzle={:03x} mips={}",
	                   static_cast<uint32_t>(value.dimension), static_cast<uint32_t>(value.cube),
	                   static_cast<uint32_t>(value.numeric_class),
	                   static_cast<uint32_t>(value.conversion), value.swizzle, value.mip_count);
}

std::string SpecializationImageText(const ResourceSpecialization::Image& image) {
	return fmt::format("class={} dim={} mips={} conv={} swizzle={:03x} cube={}",
	                   static_cast<uint32_t>(image.numeric_class),
	                   static_cast<uint32_t>(image.dimension), image.mip_count,
	                   static_cast<uint32_t>(image.conversion_format), image.shader_swizzle,
	                   static_cast<uint32_t>(image.cube));
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

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots) {
	if (source == nullptr) {
		return;
	}
	std::vector<Value>       pending(source->dwords.begin(),
	                                 source->dwords.begin() + source->dword_count);
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

uint64_t ScalarBufferSize(const ShaderBufferResource& descriptor) {
	return descriptor.Stride() == 0u
	           ? descriptor.NumRecords()
	           : static_cast<uint64_t>(descriptor.Stride()) * descriptor.NumRecords();
}

bool ReadSpecializationWord(const SrtRuntime& runtime, uint64_t address, uint32_t& word) {
	return runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, &word);
}

// The eight words of a table descriptor at `address`, as eight ReadSpecializationWord calls read
// them: one clean span read (refused unless every word is readable), else those calls, which
// also fail where they did. Tables hold hundreds of descriptors, each word checked on its own.
bool ReadSpecializationDescriptor(const SrtRuntime& runtime, uint64_t address,
                                  std::array<uint32_t, 8>& words) {
	constexpr uint64_t last = (8u - 1u) * sizeof(uint32_t);
	if (runtime.try_read_memory_span != nullptr && address <= AddressMask - last &&
	    runtime.try_read_memory_span(runtime.userdata, address, words.data(), 8u, true)) {
		return true;
	}
	for (uint32_t dword = 0; dword < words.size(); dword++) {
		const auto word_address = address + dword * sizeof(uint32_t);
		if (word_address > AddressMask || !ReadSpecializationWord(runtime, word_address, words[dword])) {
			return false;
		}
	}
	return true;
}

void MakeRangeReadable(const SrtRuntime& runtime, uint64_t base, uint64_t size) {
	if (runtime.sync_memory != nullptr && size != 0u && base <= AddressMask - size) {
		runtime.sync_memory(runtime.userdata, base, size);
	}
}

bool ReadScalarBufferWord(const ShaderBufferResource& descriptor, uint32_t dynamic_offset,
                          uint32_t immediate_offset, const SrtRuntime& runtime, uint32_t& word) {
	const auto byte_offset = static_cast<uint64_t>(dynamic_offset) + immediate_offset;
	const auto aligned     = byte_offset & ~uint64_t {3};
	const auto size        = ScalarBufferSize(descriptor);
	if (aligned > size || size - aligned < sizeof(uint32_t)) {
		word = 0;
		return true;
	}
	const auto base = descriptor.Base48() & ~uint64_t {3};
	if (aligned > AddressMask - base) {
		return false;
	}
	const auto address = base + aligned;
	if (!ReadSpecializationWord(runtime, address, word)) {
		return false;
	}
	return true;
}

// ReadScalarBufferWord for ascending offsets into one buffer (a material table's keys). The words
// come from clean span reads of up to 4 KiB inside the buffer: when every word of a span is
// readable they are the words the scalar reads return; the offsets a refused span covers keep
// the scalar reads. One read per key checked GPU ownership thousands of times per table.
class ScalarBufferWords {
public:
	ScalarBufferWords(const ShaderBufferResource& descriptor, uint32_t immediate_offset,
	                  const SrtRuntime& runtime)
	    : m_descriptor(descriptor), m_immediate(immediate_offset), m_runtime(runtime),
	      m_base(descriptor.Base48() & ~uint64_t {3}), m_size(ScalarBufferSize(descriptor)) {}

	bool Read(uint32_t dynamic_offset, uint32_t& word) {
		const auto aligned = (static_cast<uint64_t>(dynamic_offset) + m_immediate) & ~uint64_t {3};
		if ((aligned < m_begin || aligned >= m_end) && (aligned < m_refused || !Fill(aligned))) {
			return ReadScalarBufferWord(m_descriptor, dynamic_offset, m_immediate, m_runtime, word);
		}
		word = m_words[(aligned - m_begin) / sizeof(uint32_t)];
		return true;
	}

private:
	bool Fill(uint64_t aligned) {
		m_begin = m_end = 0;
		// Whole words inside the buffer whose addresses ReadScalarBufferWord accepts.
		const auto bytes =
		    aligned > m_size || aligned > AddressMask - m_base
		        ? 0
		        : std::min({uint64_t {sizeof(m_words)}, (m_size - aligned) & ~uint64_t {3},
		                    (AddressMask - m_base - aligned + sizeof(uint32_t)) & ~uint64_t {3}});
		if (bytes < 2 * sizeof(uint32_t) || m_runtime.try_read_memory_span == nullptr ||
		    !m_runtime.try_read_memory_span(m_runtime.userdata, m_base + aligned, m_words.data(),
		                                    static_cast<uint32_t>(bytes / sizeof(uint32_t)), true)) {
			m_refused = aligned + std::max<uint64_t>(bytes, sizeof(uint32_t));
			return false;
		}
		m_begin = aligned;
		m_end   = aligned + bytes;
		return true;
	}

	const ShaderBufferResource& m_descriptor;
	const uint32_t              m_immediate;
	const SrtRuntime&           m_runtime;
	const uint64_t              m_base, m_size;
	uint64_t                    m_begin = 0, m_end = 0, m_refused = 0;
	std::array<uint32_t, 1024>  m_words;
};

// Table keys in first-seen order, key 0 first: an open-addressing set with room for every probe
// (0 marks a free slot, key 0 is always in). std::unordered_set hashed and allocated per probe.
class KeyOrder {
public:
	explicit KeyOrder(uint64_t probes): m_mask(std::bit_ceil(2 * probes + 2) - 1), m_slots(m_mask + 1) {
		m_keys.reserve(probes + 1);
		m_keys.push_back(0u);
	}

	void Add(uint32_t key) {
		if (key == 0u) {
			return;
		}
		for (auto slot = (key * 0x9e3779b97f4a7c15ull >> 32u) & m_mask;; slot = (slot + 1u) & m_mask) {
			if (m_slots[slot] == key) {
				return;
			}
			if (m_slots[slot] == 0u) {
				m_slots[slot] = key;
				m_keys.push_back(key);
				return;
			}
		}
	}
	std::vector<uint32_t> Take() { return std::move(m_keys); }

private:
	uint64_t                     m_mask;
	std::vector<uint32_t>        m_slots;
	std::vector<uint32_t>        m_keys;
};

// `enumeration`: what made the candidates, a description made only for a summary of dropped classes.
template <typename Describe>
bool FinishIndirectImage(const ImageResource& image, std::vector<DescriptorValue>& probed,
                         bool over_approximates, const Describe& enumeration, IndirectImage& next,
                         std::string* reason) {
	std::vector<IndirectImageClass> classes;
	std::vector<uint32_t>           tally;
	std::vector<uint32_t>           shape(probed.size(), UINT32_MAX);
	for (uint32_t index = 0; index < probed.size(); index++) {
		if (NullImageDescriptor(probed[index])) {
			continue;
		}
		const auto value = DescriptorClass(image, probed[index]);
		const auto found = std::ranges::find(classes, value);
		shape[index]     = static_cast<uint32_t>(found - classes.begin());
		if (found == classes.end()) {
			classes.push_back(value);
			tally.push_back(1u);
		} else {
			tally[shape[index]]++;
		}
	}
	if (classes.size() > 1u) {
		const auto dominant =
		    static_cast<uint32_t>(std::ranges::max_element(tally) - tally.begin());
		if (over_approximates) {
			for (uint32_t index = 0; index < probed.size(); index++) {
				if (shape[index] != UINT32_MAX && shape[index] != dominant) {
					probed[index].dwords.fill(0);
					next.dropped_candidates++;
				}
			}
			next.dropped_shapes = static_cast<uint32_t>(classes.size() - 1u);
		}
		next.dropped_summary =
		    fmt::format("{} {}={}", enumeration(),
		                over_approximates ? "kept" : "exact probe, refused",
		                ClassText(classes[dominant]));
		for (uint32_t index = 0; index < classes.size(); index++) {
			if (index != dominant) {
				next.dropped_summary +=
				    fmt::format(" | {} x{} {}", over_approximates ? "dropped" : "kept",
				                tally[index], ClassText(classes[index]));
			}
		}
	}
	next.candidates.reserve(probed.size());
	for (const auto& candidate: probed) {
		const auto found = std::ranges::find(next.descriptors, candidate);
		if (found == next.descriptors.end()) {
			if (next.descriptors.size() >= ShaderInfo::MaxImages) {
				if (reason != nullptr) {
					*reason = "indirect image candidates exceed the dense image resource limit";
				}
				return false;
			}
			next.descriptors.push_back(candidate);
			next.candidates.push_back(static_cast<uint32_t>(next.descriptors.size() - 1u));
		} else {
			next.candidates.push_back(static_cast<uint32_t>(found - next.descriptors.begin()));
		}
	}
	return true;
}

bool MaterializeDenseIndirectImage(const DescriptorSource::IndirectImage& indirect,
                                   const ImageResource&   image,
                                   const DescriptorValue& heap_value,
                                   const DescriptorValue* bound_value, const SrtRuntime& runtime,
                                   IndirectImage& result, std::string* reason) {
	const auto note = [reason](const char* why) {
		if (reason != nullptr) {
			*reason = why;
		}
		return false;
	};
	if (heap_value.dword_count != 2u || indirect.key_bound == 0u ||
	    indirect.key_bound > MaxIndirectImageProbes) {
		return note("dense table heap is not an address resource");
	}
	const auto base =
	    ((static_cast<uint64_t>(heap_value.dwords[1]) << 32u) | heap_value.dwords[0]) & AddressMask;

	uint32_t entries = indirect.key_bound;
	if (indirect.bound_source != DescriptorSource::IndirectImage::NoBoundSource) {
		if (bound_value == nullptr || bound_value->dword_count != 1u) {
			return note("dense table loop bound did not evaluate");
		}
		const auto raw   = bound_value->dwords[0];
		const auto count = indirect.bound_signed && (raw & 0x80000000u) != 0u ? 0u : raw;
		if (count > indirect.key_bound) {
			return note("dense table loop bound exceeds the enumeration cap");
		}
		entries = count;
	}
	const uint32_t enumerated = std::max(entries, 1u);
	MakeRangeReadable(runtime, base + indirect.table_offset,
	                  static_cast<uint64_t>(enumerated) << DenseDescriptorShift);
	IndirectImage next;
	std::vector<DescriptorValue> probed;
	probed.reserve(enumerated);
	next.keys.reserve(enumerated);
	for (uint32_t key = 0; key < enumerated; key++) {
		DescriptorValue candidate;
		candidate.dword_count = 8u;
		const auto entry      = static_cast<uint64_t>(indirect.table_offset) +
		                   (static_cast<uint64_t>(key) << DenseDescriptorShift);
		if (key < entries && !ReadSpecializationDescriptor(runtime, base + entry, candidate.dwords)) {
			return note("dense table entry is not readable");
		}
		if (key >= entries || NullImageDescriptor(candidate) ||
		    !ValidImageDescriptor(candidate, image.r128) ||
		    !ReservedImageBitsClear(candidate)) {
			candidate.dwords.fill(0);
		}
		next.keys.push_back(key);
		probed.push_back(candidate);
	}
	if (!FinishIndirectImage(image, probed, true,
	                         [&] {
		                         return fmt::format("dense table=0x{:x} entries={}", indirect.table_offset,
		                                            indirect.key_bound);
	                         },
	                         next, reason)) {
		return false;
	}
	result = std::move(next);
	return true;
}
bool MaterializeAddressProbeIndirectImage(const DescriptorSource::IndirectImage& indirect,
                                          const ImageResource&                   image,
                                          const DescriptorValue&                 material_value,
                                          const DescriptorValue&                 heap_value,
                                          const SrtRuntime& runtime, IndirectImage& result,
                                          std::string* reason) {
	const auto note = [reason](const char* why) {
		if (reason != nullptr) {
			*reason = why;
		}
		return false;
	};
	if (material_value.dword_count != 2u || heap_value.dword_count != 2u ||
	    indirect.item_bound == 0u || indirect.item_bound > MaxIndirectImageProbes ||
	    indirect.selector_stride == 0u) {
		return note("address probe tables are not address resources");
	}
	const auto material_base =
	    ((static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0]) &
	    AddressMask;
	const auto heap_base =
	    ((static_cast<uint64_t>(heap_value.dwords[1]) << 32u) | heap_value.dwords[0]) & AddressMask;
	const auto records = static_cast<uint64_t>(indirect.item_bound);
	MakeRangeReadable(runtime, material_base + indirect.selector_offset,
	                  (records - 1u) * indirect.selector_stride + sizeof(uint32_t));
	KeyOrder keys(records);
	// The keys of a span of records in one clean read (one GPU-ownership check for the span instead of one a key:
	// 32 records of a light table lie within 5 KiB, read for each of its dispatches), word by word where the span
	// read is refused (as before: the same keys, in the same order).
	constexpr uint64_t SpanDwords = 1024; // (ReadShaderMemorySpan's limit for clean reads)
	thread_local std::vector<uint32_t> span;
	const uint64_t span_records = runtime.try_read_memory_span != nullptr && indirect.selector_stride % 4u == 0u &&
	                                      indirect.selector_stride <= (SpanDwords - 1u) * 4u
	                                  ? (SpanDwords - 1u) * 4u / indirect.selector_stride + 1u
	                                  : 1u;
	for (uint64_t item = 0; item < records;) {
		const auto first = material_base + indirect.selector_offset + item * indirect.selector_stride;
		const auto count = std::min<uint64_t>(records - item, span_records);
		const auto words = ((count - 1u) * indirect.selector_stride) / 4u + 1u;
		if (count > 1u && first <= AddressMask - (words - 1u) * 4u) {
			span.resize(words);
			if (runtime.try_read_memory_span(runtime.userdata, first, span.data(), static_cast<uint32_t>(words), true)) {
				for (uint64_t i = 0; i < count; i++) keys.Add(span[i * indirect.selector_stride / 4u]);
				item += count;
				continue;
			}
		}
		for (const auto end = item + count; item < end; item++) {
			const auto address = material_base + indirect.selector_offset + item * indirect.selector_stride;
			uint32_t   key     = 0;
			if (address > AddressMask || !ReadSpecializationWord(runtime, address, key)) {
				return note("record key is not readable");
			}
			keys.Add(key);
		}
	}

	IndirectImage next;
	next.keys = keys.Take();
	std::vector<DescriptorValue> probed;
	probed.reserve(next.keys.size());
	// The T#s of the keys below a bound from one clean read of the table's start (a light table's keys are light
	// indices: its ~30 T# reads, each with its own GPU-ownership checks, became one), the others one at a time, as all
	// of them where that read is refused (a T# the GPU wrote is synchronized first, as before).
	constexpr uint32_t SpanKeys = 128; // (4 KiB of T#s: ReadShaderMemorySpan's limit for clean reads)
	uint32_t           span_keys = 0;
	for (const auto key: next.keys)
		if (key < SpanKeys) span_keys = std::max(span_keys, key + 1u);
	thread_local std::vector<uint32_t> table_span;
	bool                               spanned = false;
	if (runtime.try_read_memory_span != nullptr && span_keys > 1u) {
		const auto first = heap_base + indirect.table_offset;
		const auto words = span_keys << (DenseDescriptorShift - 2u);
		if (first <= AddressMask - uint64_t {words} * sizeof(uint32_t)) {
			table_span.resize(words);
			spanned = runtime.try_read_memory_span(runtime.userdata, first, table_span.data(), words, true);
		}
	}
	// A stale record's key (records past the light count hold anything: half of a light table's keys, 0xfffffffe and
	// pointers) names an entry past the mapping the table starts in: not the table's, a null candidate without a read
	// (each was a clean read attempt of memory far away, ~0.4 us).
	const auto table_end = runtime.mapping_end != nullptr
	                           ? runtime.mapping_end(runtime.userdata, heap_base + indirect.table_offset)
	                           : AddressMask + 1u;
	for (const auto key: next.keys) {
		DescriptorValue candidate;
		candidate.dword_count = 8u;
		const auto entry      = heap_base + indirect.table_offset +
		                   (static_cast<uint64_t>(key) << DenseDescriptorShift);
		bool readable = entry + candidate.dword_count * sizeof(uint32_t) <= std::min(AddressMask, table_end);
		if (readable && spanned && key < span_keys) {
			std::copy_n(table_span.begin() + (static_cast<size_t>(key) << (DenseDescriptorShift - 2u)),
			            candidate.dword_count, candidate.dwords.begin());
		} else if (readable) {
			MakeRangeReadable(runtime, entry, candidate.dword_count * sizeof(uint32_t));
			readable = ReadSpecializationDescriptor(runtime, entry, candidate.dwords);
		}
		if (!readable || NullImageDescriptor(candidate) ||
		    !ValidImageDescriptor(candidate, image.r128) || !ReservedImageBitsClear(candidate)) {
			candidate.dwords.fill(0);
		}
		probed.push_back(candidate);
	}
	if (!FinishIndirectImage(image, probed, true,
	                         [&] {
		                         return fmt::format("address probe records={} stride={} table=0x{:x} keys={}",
		                                            indirect.item_bound, indirect.selector_stride,
		                                            indirect.table_offset, next.keys.size());
	                         },
	                         next, reason)) {
		return false;
	}
	result = std::move(next);
	return true;
}


bool MaterializeIndirectImage(const DescriptorSource::IndirectImage& indirect,
                              const ImageResource& image, const DescriptorValue& material_value,
                              const DescriptorValue& heap_value, const SrtRuntime& runtime,
                              IndirectImage& result, std::string* reason) {
	const auto note = [reason](const char* why) {
		if (reason != nullptr) {
			*reason = why;
		}
		return false;
	};
	ShaderBufferResource material;
	ShaderBufferResource heap;
	if (!DecodeBufferDescriptor(material_value, material) ||
	    !DecodeBufferDescriptor(heap_value, heap)) {
		return note("material or heap is not a buffer descriptor");
	}
	if (material.Stride() != 0u && material.Stride() != indirect.selector_stride) {
		return note("material buffer stride no longer matches the tracked selector");
	}

	const auto period      = uint64_t {1} << 32u;
	const auto step        = std::gcd<uint64_t>(indirect.selector_stride, period);
	const auto residue     = static_cast<uint64_t>(indirect.selector_offset) % step;
	const auto size        = ScalarBufferSize(material);
	const auto limit       = std::min<uint64_t>(UINT32_MAX, size + 3u);
	const auto probe_count = residue <= limit ? (limit - residue) / step + 1u : 0u;
	if (probe_count > MaxIndirectImageProbes) {
		return note("material table is too large to enumerate");
	}

	MakeRangeReadable(runtime, material.Base48() & ~uint64_t {3}, ScalarBufferSize(material));
	MakeRangeReadable(runtime, heap.Base48() & ~uint64_t {3}, ScalarBufferSize(heap));
	KeyOrder          keys(probe_count);
	ScalarBufferWords material_words(material, indirect.selector_immediate, runtime);
	for (uint64_t offset = residue; offset <= limit && probe_count != 0u; offset += step) {
		uint32_t key = 0;
		if (!material_words.Read(static_cast<uint32_t>(offset), key)) {
			return note("material table key is not readable");
		}
		keys.Add(key);
		if (limit - offset < step) {
			break;
		}
	}

	IndirectImage next;
	next.keys = keys.Take();
	std::vector<DescriptorValue> probed;
	probed.reserve(next.keys.size());
	const auto heap_size = ScalarBufferSize(heap);
	const auto heap_base = heap.Base48() & ~uint64_t {3};
	for (const auto key: next.keys) {
		DescriptorValue candidate;
		candidate.dword_count  = 8u;
		const auto heap_offset = key << 5u;
		// Inside the buffer, ReadScalarBufferWord reads the words at heap_base + heap_offset.
		if (uint64_t {heap_offset} + sizeof(candidate.dwords) <= heap_size) {
			if (!ReadSpecializationDescriptor(runtime, heap_base + heap_offset, candidate.dwords)) {
				return note("heap entry is not readable");
			}
		} else {
			for (uint32_t dword = 0; dword < candidate.dword_count; dword++) {
				if (!ReadScalarBufferWord(heap, heap_offset, dword * sizeof(uint32_t), runtime,
				                          candidate.dwords[dword])) {
					return note("heap entry is not readable");
				}
			}
		}
		if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, image.r128) ||
		    !ReservedImageBitsClear(candidate)) {
			candidate.dwords.fill(0);
		}
		probed.push_back(candidate);
	}
	if (!FinishIndirectImage(image, probed, step < indirect.selector_stride,
	                         [&] {
		                         return fmt::format("stride={} step={} probes={} keys={}", indirect.selector_stride,
		                                            step, probe_count, next.keys.size());
	                         },
	                         next, reason)) {
		return false;
	}
	result = std::move(next);
	return true;
}

} // namespace

static bool MaterializeSnapshot(const ResourcePlan& program, const SrtRuntime& runtime,
                                MaterializeWorkspace& workspace, MaterializeReport* report) {
	auto& snapshot = workspace.materialized;
	// A previous failed transaction must not supply state to this invocation.
	snapshot.indirect_images.clear();
	snapshot.resources.uniform_fill = {};
	snapshot.resources.images.clear();
	const auto fail = [report](const char* why) {
		if (report != nullptr) {
			report->reason = why;
		}
		return false;
	};
	if (!program.resource_tracking_complete) {
		return fail("resources were not tracked");
	}

	if (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr) {
		return fail("indirect image needs a specialization memory reader");
	}
	auto& values = workspace.values;
	auto& flattened_srt = workspace.flattened_srt;
	auto& active_sources = workspace.active_sources;
	if (!EvaluateRuntimeSources(program, program.materialization_sources, runtime, values,
	                            flattened_srt, program.clean_flat_slots, active_sources)) {
		return fail("a descriptor source did not evaluate");
	}

	auto&                   next  = snapshot.resources;
	const auto&             fill  = program.uniform_fill;
	const auto              words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	if (words != 0 &&
	    EvaluateUniformValues(program, std::span(fill.values).first(words), runtime,
	                          std::span(stored).first(words)) &&
	    std::all_of(stored.begin(), stored.begin() + words,
	                [&](uint32_t value) { return value == stored[0]; })) {
		next.uniform_fill       = fill.fill;
		next.uniform_fill.value = stored[0];
	}
	auto cursor = values.begin();
	next.buffers.assign(cursor, cursor + program.info.buffers.size());
	cursor += program.info.buffers.size();
	next.flattened_srt.swap(flattened_srt);
	next.images.resize(program.info.images.size());
	for (uint32_t image_index = 0; image_index < program.info.images.size(); image_index++) {
		const auto& image = program.info.images[image_index];
		// Without an indirect image (ExtractResourcePlan), every image takes the next descriptor:
		// the lookup would only miss the cache on the program's descriptor sources.
		const auto* source =
		    program.requires_specialization_memory ? Source(program, image.source) : nullptr;
		if (source != nullptr && source->indirect_image.has_value()) {
			if (!active_sources[image.source]) {
				next.images[image_index].dword_count = 8u;
				continue;
			}
			const auto&           indirect  = *source->indirect_image;
			const bool            dense     = indirect.key_bound != 0u;
			const bool            has_bound = dense && indirect.bound_source !=
			                                               DescriptorSource::IndirectImage::NoBoundSource;
			std::vector<uint32_t> requests;
			if (!dense) {
				requests.push_back(indirect.material_source);
			}
			requests.push_back(indirect.heap_source);
			if (has_bound) {
				requests.push_back(indirect.bound_source);
			}
			SrtRuntime clean_runtime  = runtime;
			clean_runtime.read_memory = runtime.read_specialization_memory;
			std::vector<DescriptorValue> tables;
			if (!EvaluateDescriptorSources(program, requests, clean_runtime, tables)) {
				return fail("an indirect image table source did not evaluate");
			}
			IndirectImage table;
			std::string   detail;
			const auto&            heap_value  = tables[dense ? 0u : 1u];
			const DescriptorValue* bound_value = has_bound ? &tables.back() : nullptr;
			const bool             materialized =
			    dense ? MaterializeDenseIndirectImage(indirect, image, heap_value, bound_value,
			                                          runtime, table, &detail)
			    : indirect.item_bound != 0u
			        ? MaterializeAddressProbeIndirectImage(indirect, image, tables.front(),
			                                               heap_value, runtime, table, &detail)
			        : MaterializeIndirectImage(indirect, image, tables.front(), heap_value,
			                                   runtime, table, &detail);
			if (!materialized) {
				return fail(detail.c_str());
			}
			if (report != nullptr) {
				report->dropped_candidates += table.dropped_candidates;
				report->dropped_shapes += table.dropped_shapes;
				if (!table.dropped_summary.empty()) {
					report->dropped_summary +=
					    fmt::format("\n  image {}: {}", image_index, table.dropped_summary);
				}
			}
			next.images[image_index] = table.descriptors[table.candidates[0]];
			if (table.descriptors.size() > 1u) {
				table.resource = image_index;
				snapshot.indirect_images.push_back(std::move(table));
			}
		} else {
			auto descriptor = *cursor++;
			if (!ValidImageDescriptor(descriptor, image.r128)) {
				descriptor.dwords.fill(0);
			}
			next.images[image_index] = descriptor;
		}
	}
	next.samplers.assign(cursor, cursor + program.info.samplers.size());
	next.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	return true;
}

struct SamplerPlan {
	std::array<uint32_t, ShaderInfo::MaxSamplers> point_sampler {};
	uint32_t                                      sampler_count = 0;
};

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization) {
		for (const auto& image: specialization.images) {
			indices.push_back(image.fmask ? UINT32_MAX : count++);
		}
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != indices.size());
		for (uint32_t index = 0; index < indices.size(); index++) {
			if (indices[index] != UINT32_MAX && indices[index] != index) {
				images[indices[index]] = std::move(images[index]);
			}
		}
		images.resize(count);
	}

	std::vector<uint32_t> indices;
	uint32_t              count = 0;
};

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan);

bool EnumerateAddressProbe(const DescriptorSource::IndirectImage& probe, const ImageResource& root,
                           uint64_t records_base, uint64_t heap_base, const SrtRuntime& runtime,
                           AddressProbeCandidates& result) {
	DescriptorValue records {.dword_count = 2u}, heap {.dword_count = 2u};
	records.dwords[0] = static_cast<uint32_t>(records_base);
	records.dwords[1] = static_cast<uint32_t>(records_base >> 32u);
	heap.dwords[0]    = static_cast<uint32_t>(heap_base);
	heap.dwords[1]    = static_cast<uint32_t>(heap_base >> 32u);
	IndirectImage enumerated;
	if (!MaterializeAddressProbeIndirectImage(probe, root, records, heap, runtime, enumerated, nullptr)) return false;
	result.keys        = std::move(enumerated.keys);
	result.candidates  = std::move(enumerated.candidates);
	result.descriptors = std::move(enumerated.descriptors);
	return true;
}

bool TableIndirectForm(ResourceSpecialization& specialization, std::vector<DescriptorValue>* snapshot_images) {
	auto&    images = specialization.images;
	uint32_t root = ImageResource::NoIndirectImage, candidates = 0;
	for (uint32_t index = 0; index < images.size(); index++) {
		if (images[index].fmask) return false;
		if (images[index].indirect_root == index) {
			if (root != ImageResource::NoIndirectImage) return false;
			root = index;
		} else if (images[index].indirect_root != ImageResource::NoIndirectImage) {
			candidates++;
		}
	}
	if (root == ImageResource::NoIndirectImage) return true;
	constexpr uint32_t capacity = TablePlan::IndirectCapacity;
	// (The candidates follow every other image: BuildResourceSpecialization appends them.)
	if (candidates == 0 || candidates + 1u > capacity || images.size() < candidates + 1u ||
	    (snapshot_images != nullptr && snapshot_images->size() != images.size()))
		return false;
	for (size_t index = images.size() - candidates; index < images.size(); index++)
		if (images[index].indirect_root != root) return false;
	const auto candidate = images.back();
	images.resize(images.size() + (capacity - 1u - candidates), candidate);
	images[root].indirect_mapping_offset    = 0; // (EnterTableMode names the block's mapping)
	images[root].indirect_search_iterations = ImageResource::RuntimeIndirectSearch;
	if (snapshot_images != nullptr) snapshot_images->resize(images.size(), DescriptorValue {.dword_count = 8u});
	return true;
}

bool PortableShaders() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PORTABLE_SHADERS");
		return !(value != nullptr && std::string_view(value) == "0");
	}();
	return enabled;
}

// A null texture reads zero whatever its type. Bound as a 1x1 image of the type the shader declares
// (NullTextureDesc), it takes the module of the texture bound other times; without portable
// modules, and when multisampled, a 2D one.
static Decoder::ImageDimension NullImageDimension(Decoder::ImageDimension declared) {
	switch (declared) {
		case Decoder::ImageDimension::Dim1D:
		case Decoder::ImageDimension::Dim1DArray:
		case Decoder::ImageDimension::Dim2DArray:
		case Decoder::ImageDimension::Dim3D: return PortableShaders() ? declared : Decoder::ImageDimension::Dim2D;
		default: return Decoder::ImageDimension::Dim2D;
	}
}

void CanonicalizeSpecialization(const ShaderInfo& info, ResourceSpecialization& specialization) {
	// What the shader cannot tell apart translates to one permutation: the swizzle of an image it
	// only reads in a format the device converts (only stores and converted texels apply it), and a
	// texture without layers it samples addressing layers (a one-layer view: the layer is clamped).
	for (size_t i = 0; i < specialization.images.size(); i++) {
		auto&      image = specialization.images[i];
		const auto root  = i < info.images.size() ? i : image.indirect_root;
		if (root >= info.images.size()) continue;
		const auto& base = info.images[root];
		if (!base.written && image.conversion_format == Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = ShaderImageIdentitySwizzle;
		}
		using Decoder::ImageDimension;
		const auto layers = image.dimension == ImageDimension::Dim1D        ? ImageDimension::Dim1DArray
		                    : image.dimension == ImageDimension::Dim2D      ? ImageDimension::Dim2DArray
		                    : image.dimension == ImageDimension::Dim2DMsaa  ? ImageDimension::Dim2DMsaaArray
		                                                                    : ImageDimension::Unknown;
		if (!image.cube && layers != ImageDimension::Unknown && base.dimension == layers &&
		    base.resource_class != ImageResourceClass::Storage) {
			image.dimension = layers;
		}
	}
	if (!PortableShaders()) {
		return;
	}
	for (size_t i = 0; i < specialization.buffers.size() && i < info.buffers.size(); i++) {
		auto& buffer = specialization.buffers[i];
		if (RuntimeBufferStride(buffer.packed_stride)) {
			buffer.packed_stride &= BufferAddTidBit;
		}
	}
}

bool PortableFormats(const ShaderInfo& info, ResourceSpecialization& specialization) {
	bool changed = false;
	if (!PortableShaders()) {
		return changed;
	}
	for (size_t i = 0; i < specialization.buffers.size() && i < info.buffers.size(); i++) {
		auto& buffer = specialization.buffers[i];
		if (RuntimeBufferFormat(info.buffers[i]) &&
		    (buffer.descriptor_format != Prospero::BufferFormat::kInvalid ||
		     buffer.descriptor_swizzle != DstSel(4, 5, 6, 7))) {
			buffer.descriptor_format  = Prospero::BufferFormat::kInvalid;
			buffer.descriptor_swizzle = DstSel(4, 5, 6, 7);
			changed                   = true;
		}
	}
	return changed;
}

uint32_t BufferFormatClass(Prospero::BufferFormat format) {
	const auto info = Format::GetFormatInfo(format);
	if (info.type == Format::ComponentType::Unknown) {
		return 0;
	}
	uint32_t layout = 0;
	if (!info.packed_bitfield) {
		const auto bytes = info.component_bits[0] / 8u;
		layout = (static_cast<uint32_t>(std::countr_zero(bytes)) << 2u) | (info.component_count - 1u);
	} else if (info.component_count == 3u) {
		layout = BufferWord::PackedLayouts + (info.component_bits[0] == 11u ? 0u : 1u);
	} else {
		layout = BufferWord::PackedLayouts + (info.component_bits[0] == 2u ? 2u : 3u);
	}
	return layout | static_cast<uint32_t>(info.type) << (BufferWord::TypeShift - BufferWord::LayoutShift);
}

std::vector<Prospero::TextureNumericClass> PredictImageNumericClasses(const Program& program) {
	const auto            count = program.info.images.size();
	std::vector<uint32_t> integer(count), floating(count);
	// The same, counted only up to the first merge with other values: a Phi, or a SelectU32 whose
	// other value is not a constant (the EXEC predication merging a register's old and new values,
	// which may be a float from another path).
	std::vector<uint32_t> direct_integer(count), direct_floating(count);
	std::unordered_map<const Inst*, std::vector<std::pair<const Inst*, size_t>>> uses;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
				if (const auto* definition = inst.Arg(arg).TryInstruction()) uses[definition].emplace_back(&inst, arg);
			}
		}
	}
	// Values that only move data along: followed to their own uses (or producers).
	const auto moves = [](ValueOpcode opcode) {
		switch (opcode) {
			case ValueOpcode::Phi:
			case ValueOpcode::Identity:
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::CompositeExtractU32x3:
			case ValueOpcode::CompositeExtractU32x4:
			case ValueOpcode::CompositeConstructU32x2:
			case ValueOpcode::CompositeConstructU32x3:
			case ValueOpcode::CompositeConstructU32x4:
			case ValueOpcode::SelectU32: return true;
			default: return false;
		}
	};
	// The EXEC predicate of an image read or write: a SelectU32 on it keeps, for the lanes that
	// access the image, the value of its first arm (the register written under that predicate).
	const auto predicate_of = [](const Inst& access) -> const Inst* {
		switch (access.GetOpcode()) {
			case ValueOpcode::ImageRead: return access.Arg(2).Resolve().TryInstruction();
			case ValueOpcode::ImageWrite: return access.Arg(3).Resolve().TryInstruction();
			default: return nullptr;
		}
	};
	const auto exec_merge = [](const Inst* select, const Inst* predicate) {
		return predicate != nullptr && select->Arg(0).Resolve().TryInstruction() == predicate;
	};
	// What reads the texels of `read`: F32 operands (through BitCastF32U32) or U32 operations.
	const auto merges = [](const Inst* user, size_t arg) {
		if (user->GetOpcode() == ValueOpcode::Phi) return true;
		return user->GetOpcode() == ValueOpcode::SelectU32 && !user->Arg(arg == 1 ? 2 : 1).Resolve().IsImmediate();
	};
	const auto classify_uses = [&](const Inst* read, size_t image) {
		const auto* predicate = predicate_of(*read);
		std::vector<std::pair<const Inst*, bool>> work {{read, false}}; // (value, merged on the way)
		std::array<std::unordered_set<const Inst*>, 2> seen {{{read}, {}}}; // by merged
		while (!work.empty() && seen[0].size() + seen[1].size() < 512) {
			const auto [value, merged] = work.back();
			work.pop_back();
			const auto found = uses.find(value);
			if (found == uses.end()) continue;
			for (const auto& [user, arg]: found->second) {
				const auto opcode = user->GetOpcode();
				bool       is_float = false, is_integer = false;
				if (moves(opcode) && !(opcode == ValueOpcode::SelectU32 && arg == 0)) {
					const bool next = merged || (merges(user, arg) && !(arg == 1 && exec_merge(user, predicate)));
					if (seen[next].insert(user).second) work.emplace_back(user, next);
				} else if (opcode == ValueOpcode::BitCastF32U32) {
					is_float = true;
				} else if (TypeOf(opcode) != Type::Void) {
					const auto type = ArgTypeOf(opcode, arg);
					is_float        = type == Type::F32 || type == Type::F16;
					is_integer      = type == Type::U32;
				}
				floating[image] += is_float;
				integer[image] += is_integer;
				if (!merged) {
					direct_floating[image] += is_float;
					direct_integer[image] += is_integer;
				}
			}
		}
	};
	// What makes the texels `write` stores: float bits (BitCastU32F32) or U32 operations, the direct
	// ones up to the first merge as for reads.
	const auto classify_producers = [&](const Inst& write, size_t image) {
		const auto*                     predicate = predicate_of(write);
		std::vector<std::pair<const Inst*, bool>> work; // (value, merged on the way)
		std::array<std::unordered_set<const Inst*>, 2> seen;
		if (auto* inst = write.Arg(2).TryInstruction()) work.emplace_back(inst, false);
		while (!work.empty() && seen[0].size() + seen[1].size() < 512) {
			const auto [value, merged] = work.back();
			work.pop_back();
			if (!seen[merged].insert(value).second) continue;
			const auto opcode = value->GetOpcode();
			bool       is_float = false, is_integer = false;
			if (moves(opcode)) {
				const bool select = opcode == ValueOpcode::SelectU32;
				const bool own    = select && exec_merge(value, predicate);
				const bool mixes  = select && !own && !value->Arg(1).Resolve().IsImmediate() &&
				                   !value->Arg(2).Resolve().IsImmediate();
				const bool next   = merged || opcode == ValueOpcode::Phi || mixes;
				for (size_t arg = select ? 1 : 0; arg < (own ? 2u : value->NumArgs()); arg++) {
					if (auto* inst = value->Arg(arg).TryInstruction()) work.emplace_back(inst, next);
				}
			} else if (opcode == ValueOpcode::BitCastU32F32) {
				is_float = true;
			} else if (TypeOf(opcode) == Type::U32 && ImageOpcodeInfoOf(opcode).access == ImageAccess::None &&
			           BufferAccessOf(opcode) == BufferAccess::None && SharedAccessOf(opcode) == SharedAccess::None) {
				is_integer = true;
			}
			floating[image] += is_float;
			integer[image] += is_integer;
			if (!merged) {
				direct_floating[image] += is_float;
				direct_integer[image] += is_integer;
			}
		}
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = ImageOpcodeInfoOf(inst.GetOpcode()).access;
			if (access != ImageAccess::Read && access != ImageAccess::Write) continue;
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size() || program.memory_info[index].resource >= count) continue;
			const auto image = program.memory_info[index].resource;
			if (access == ImageAccess::Read) classify_uses(&inst, image);
			else classify_producers(inst, image);
		}
	}
	// Uint when every use (or every use before a merge) is an integer one.
	std::vector<Prospero::TextureNumericClass> out(count, Prospero::TextureNumericClass::Float);
	for (size_t image = 0; image < count; image++) {
		if (program.info.images[image].atomic || (integer[image] != 0 && floating[image] == 0) ||
		    (direct_integer[image] != 0 && direct_floating[image] == 0)) {
			out[image] = Prospero::TextureNumericClass::Uint;
		}
	}
	return out;
}

uint32_t BufferDescriptorWord(const DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor) || descriptor.Type() != 0) {
		return 0;
	}
	const uint32_t stride = descriptor.Stride() <= BufferWord::MaxStride ? descriptor.Stride() : 0u;
	return stride | BufferFormatClass(descriptor.Format()) << BufferWord::LayoutShift |
	       descriptor.DstSelXYZW() << BufferWord::DstSelShift;
}

static bool BuildResourceSpecialization(const ResourcePlan& program, MaterializeWorkspace& workspace,
                                        ResourceSnapshot&       specialized_snapshot,
                                        ResourceSpecialization& specialization,
                                        MaterializeReport*      report) {
	auto& snapshot            = workspace.materialized;
	auto& next_snapshot       = snapshot.resources;
	auto& next_specialization = workspace.specialization;
	next_specialization.buffers.clear();
	next_specialization.images.clear();
	next_specialization.buffers.reserve(program.info.buffers.size());
	size_t image_count   = program.info.images.size();
	size_t mapping_words = 0;
	// Portable modules: a table's candidates padded with null images to a power of two, its key
	// mapping found through a header word (IndirectMappingHeader), so the module depends on neither
	// its exact size nor its key count.
	const bool portable = PortableShaders();
	auto&      padded   = workspace.padded_candidates;
	padded.clear();
	for (const auto& table: snapshot.indirect_images) {
		if (table.resource >= program.info.images.size() || table.descriptors.size() < 2u ||
		    image_count + table.descriptors.size() - 1u > ShaderInfo::MaxImages) {
			return SpecializationFail(report, 
			    "indirect image candidates exceed the dense image resource limit");
		}
		auto candidates = table.descriptors.size();
		if (portable && image_count + std::bit_ceil(candidates) - 1u <= ShaderInfo::MaxImages) {
			candidates = std::bit_ceil(candidates);
		}
		padded.push_back(static_cast<uint32_t>(candidates));
		image_count += candidates - 1u;
		mapping_words += (portable ? 2u : 1u) + table.keys.size() * 2u;
	}
	next_snapshot.images.reserve(image_count);
	next_snapshot.flattened_srt.reserve(next_snapshot.flattened_srt.size() + mapping_words);
	const auto header = static_cast<uint32_t>(next_snapshot.flattened_srt.size());
	if (portable) {
		next_snapshot.flattened_srt.resize(header + snapshot.indirect_images.size());
	}
	next_specialization.images.reserve(image_count);
	for (const auto& image: program.info.images) {
		next_specialization.images.push_back({
		    .numeric_class              = image.numeric_class,
		    .dimension                  = image.dimension,
		    .mip_count                  = image.mip_count,
		    .conversion_format          = image.conversion_format,
		    .shader_swizzle             = image.shader_swizzle,
		    .indirect_root              = image.indirect_root,
		    .indirect_mapping_offset    = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube                       = image.cube,
		});
	}
	for (size_t table_index = 0; table_index < snapshot.indirect_images.size(); table_index++) {
		const auto& table      = snapshot.indirect_images[table_index];
		const auto  root_image = next_specialization.images[table.resource];
		for (uint32_t candidate = 1; candidate < padded[table_index]; candidate++) {
			auto image          = root_image;
			image.indirect_root = table.resource;
			next_specialization.images.push_back(image);
			next_snapshot.images.push_back(candidate < table.descriptors.size() ? table.descriptors[candidate]
			                                                                    : DescriptorValue {.dword_count = 8u});
		}
		auto&      root    = next_specialization.images[table.resource];
		const auto mapping = static_cast<uint32_t>(next_snapshot.flattened_srt.size());
		root.indirect_root = table.resource;
		if (portable) {
			root.indirect_mapping_offset    = header + static_cast<uint32_t>(table_index);
			root.indirect_search_iterations = ImageResource::RuntimeIndirectSearch;
			next_snapshot.flattened_srt[root.indirect_mapping_offset] = mapping;
		} else {
			root.indirect_mapping_offset    = mapping;
			root.indirect_search_iterations = std::bit_width(table.keys.size());
		}
		next_snapshot.flattened_srt.resize(mapping + 1u + table.keys.size() * 2u);
		std::vector<uint32_t> order(table.keys.size());
		std::iota(order.begin(), order.end(), 0u);
		std::ranges::sort(order, {}, [&](uint32_t index) { return table.keys[index]; });
		next_snapshot.flattened_srt[mapping] = static_cast<uint32_t>(table.keys.size());
		for (uint32_t entry = 0; entry < order.size(); entry++) {
			const auto source                   = order[entry];
			const auto offset                   = mapping + 1u + entry * 2u;
			next_snapshot.flattened_srt[offset] = table.keys[source];
			next_snapshot.flattened_srt[offset + 1] = table.candidates[source];
		}
		next_snapshot.images[table.resource] = table.descriptors[0];
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		auto&                descriptor_value = next_snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(report, fmt::format("buffer descriptor {} has invalid width", i));
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
		next_specialization.buffers.push_back({
		    .packed_stride     = packed_stride,
		    .descriptor_format = program.info.buffers[i].formatted
		                             ? descriptor.Format()
		                             : Prospero::BufferFormat::kInvalid,
		    .descriptor_swizzle =
		        program.info.buffers[i].formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7),
		    .byte_base_offset = (descriptor.Base48() & 3u) != 0,
		});
	}
	for (uint32_t i = 0; i < next_specialization.images.size(); i++) {
		const auto& descriptor = next_snapshot.images[i];
		auto&       image      = next_specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(report, fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(report, fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = StorageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(report, 
			    fmt::format("storage image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = NullImageDimension(base.dimension);
			image.cube          = false;
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(report, fmt::format(
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
		if (base.atomic && format != Prospero::BufferFormat::k32UInt) {
			return SpecializationFail(report, 
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
				return SpecializationFail(report, "FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
			// Loads of a format the device converts apply no swizzle, and a store writes each channel
			// of the format the component the swizzle routes to it first: one that routes every channel
			// its own component (X001, XXXX, XY00...) writes what an identity one does.
			const auto count    = Format::GetFormatInfo(format).component_count;
			bool       identity = image.conversion_format == Prospero::BufferFormat::kInvalid && count >= 1u &&
			                count <= 4u;
			for (uint32_t channel = 0; identity && channel < count; channel++) {
				uint32_t source = 0;
				while (source < 4u && ((image.shader_swizzle >> (source * 3u)) & 7u) != 4u + channel) source++;
				identity = source == channel;
			}
			if (identity) image.shader_swizzle = ShaderImageIdentitySwizzle;
		}
		const bool raw_sint_storage = storage && format == Prospero::BufferFormat::k32SInt &&
		                              base.written && !base.read && !base.atomic;
		image.numeric_class         = Prospero::SampledTextureNumericClass(format);
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(report, 
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(report, 
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
	}
	CanonicalizeSpecialization(program.info, next_specialization);
	for (uint32_t root_index = 0; root_index < next_specialization.images.size(); root_index++) {
		auto& root = next_specialization.images[root_index];
		if (root.indirect_root != root_index) {
			continue;
		}
		const auto& flattened = next_snapshot.flattened_srt;
		const auto  mapping   = root.indirect_search_iterations != ImageResource::RuntimeIndirectSearch
		                            ? root.indirect_mapping_offset
		                        : root.indirect_mapping_offset < flattened.size()
		                            ? flattened[root.indirect_mapping_offset]
		                            : UINT32_MAX;
		const auto key_count = mapping < flattened.size() ? flattened[mapping] : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(mapping) + 1u + static_cast<size_t>(key_count) * 2u > flattened.size()) {
			return SpecializationFail(report, "indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < next_specialization.images.size(); resource++) {
			if (next_specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(next_snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail(report, "indirect image specialization has no typed candidate");
		}
		const auto& image_class = next_specialization.images[exemplar];
		for (uint32_t candidate = 0; candidate < next_specialization.images.size(); candidate++) {
			auto& image = next_specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(next_snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			if (image.numeric_class != image_class.numeric_class ||
			    image.dimension != image_class.dimension ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle ||
			    image.cube != image_class.cube) {
				return SpecializationFail(report, 
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates: "
				                "resource {} is {}, resource {} is {}",
				                program.info.images[root_index].first_use_pc, exemplar,
				                SpecializationImageText(image_class), candidate,
				                SpecializationImageText(image)));
			}
		}
	}
	SamplerPlan sampler_plan;
	if (!BuildSamplerPlan(program.info, next_specialization.images, sampler_plan)) {
		return SpecializationFail(report, "specialized sampler layout exceeds its resource limit");
	}
	for (uint32_t index = 0; index < program.info.samplers.size(); index++) {
		const auto target = sampler_plan.point_sampler[index];
		if (target != UINT32_MAX && target >= program.info.samplers.size()) {
			next_snapshot.samplers.push_back(next_snapshot.samplers[index]);
		}
	}
	if (kyty_local_preparation_trim_mode.load(std::memory_order_relaxed)) {
		// Runtime snapshots only need stable compaction. The compiler still
		// builds the full index remap when rewriting IR resource references.
		EXIT_IF(next_snapshot.images.size() != next_specialization.images.size());
		uint32_t output = 0;
		for (uint32_t index = 0; index < next_specialization.images.size(); ++index) {
			if (!next_specialization.images[index].fmask) {
				if (output != index) {
					next_snapshot.images[output] = std::move(next_snapshot.images[index]);
				}
				++output;
			}
		}
		next_snapshot.images.resize(output);
	} else {
		ImageRemap(next_specialization).Apply(next_snapshot.images);
	}
	// Publish only after all validation succeeds. Returning the old destination
	// storage to the workspace retains capacity without caching mutable contents.
	std::swap(specialization, next_specialization);
	std::swap(specialized_snapshot, next_snapshot);
	return true;
}

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan) {
	if (base.samplers.size() > plan.point_sampler.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.point_sampler.fill(UINT32_MAX);
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= RequiresPointSampler(images[pair.image]) ? 2u : 1u;
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		if ((usage[index] & 2u) == 0u) {
			continue;
		}
		if ((usage[index] & 1u) == 0u) {
			plan.point_sampler[index] = index;
		} else {
			if (plan.sampler_count >= ShaderInfo::MaxSamplers) {
				return false;
			}
			plan.point_sampler[index] = plan.sampler_count++;
		}
	}
	return true;
}

// These masks are conservative supersets of every descriptor bit consumed by
// BuildResourceSpecialization. Image validity has already been checked by
// MaterializeSnapshot, including array bounds and MSAA restrictions.
static constexpr uint32_t GuardBufferStride = 0xbfff0000u; // stride + swizzle enable
static constexpr uint32_t GuardBufferFlags = 0xc0e00000u;  // type, index stride, add-TID
static constexpr uint32_t GuardBufferFormat = 0x0007ffffu; // format + destination swizzle
static constexpr uint32_t GuardImageFormat = 0x1ff00000u;
static constexpr uint32_t GuardImageInterpretation = 0xf00fffffu; // type, mips, swizzle

static bool MatchesSpecializationGuard(const ResourcePlan& program,
                                       const MaterializedSnapshot& materialized,
                                       const ResourceSpecializationGuard& guard) {
	const auto& snapshot = materialized.resources;
	if (guard.owner != &program || !materialized.indirect_images.empty() ||
	    snapshot.buffers.size() != guard.buffers.size() ||
	    snapshot.images.size() != guard.images.size()) return false;
	for (size_t i = 0; i < guard.buffers.size(); ++i) {
		const auto& descriptor = snapshot.buffers[i];
		const auto& probe = guard.buffers[i];
		if (descriptor.dword_count != 4 ||
		    (descriptor.dwords[0] & 3u) != probe.low_address ||
		    (descriptor.dwords[1] & GuardBufferStride) != probe.stride ||
		    (descriptor.dwords[3] & probe.flag_mask) != probe.flags) return false;
	}
	for (size_t i = 0; i < guard.images.size(); ++i) {
		const auto& descriptor = snapshot.images[i];
		const auto& probe = guard.images[i];
		if (descriptor.dword_count != 8 || NullImageDescriptor(descriptor) != probe.null ||
		    (descriptor.dwords[1] & GuardImageFormat) != probe.format ||
		    (descriptor.dwords[3] & GuardImageInterpretation) != probe.interpretation) return false;
	}
	return true;
}

static bool CaptureSpecializationGuard(const ResourcePlan& program,
                                       const MaterializedSnapshot& materialized,
                                       ResourceSpecializationGuard& guard) {
	// Indirect tables can expand resources and append a fresh SRT lookup table.
	// They continue through the reference path, even when inactive this time.
	if (program.requires_specialization_memory || !materialized.indirect_images.empty() ||
	    std::ranges::any_of(program.descriptor_sources,
	        [](const auto& source) { return source.indirect_image.has_value(); }) ||
	    std::ranges::any_of(program.info.images, [](const auto& image) {
		    return image.indirect_root != ImageResource::NoIndirectImage;
	    })) return false;
	const auto& snapshot = materialized.resources;
	if (snapshot.buffers.size() != program.info.buffers.size() ||
	    snapshot.images.size() != program.info.images.size()) return false;
	guard.buffers.reserve(snapshot.buffers.size());
	for (size_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto& descriptor = snapshot.buffers[i];
		// The reference sanitizes invalid types. Do not reuse that transformation.
		if (descriptor.dword_count != 4 || (descriptor.dwords[3] >> 30u) != 0) return false;
		const auto mask = GuardBufferFlags | (program.info.buffers[i].formatted ? GuardBufferFormat : 0u);
		guard.buffers.push_back({descriptor.dwords[0] & 3u,
		    descriptor.dwords[1] & GuardBufferStride, descriptor.dwords[3] & mask, mask});
	}
	guard.images.reserve(snapshot.images.size());
	for (const auto& descriptor : snapshot.images) {
		if (descriptor.dword_count != 8) return false;
		guard.images.push_back({descriptor.dwords[1] & GuardImageFormat,
		    descriptor.dwords[3] & GuardImageInterpretation, NullImageDescriptor(descriptor)});
	}
	guard.owner = &program;
	return true;
}

static void PublishSpecializationGuard(const ResourcePlan& program,
                                       const ResourceSpecialization& specialization,
                                       ResourceSpecializationGuard& next,
                                       ResourceSpecializationGuard& guard) {
	SamplerPlan samplers;
	EXIT_IF(!BuildSamplerPlan(program.info, specialization.images, samplers));
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		if (samplers.point_sampler[i] != UINT32_MAX &&
		    samplers.point_sampler[i] >= program.info.samplers.size()) next.duplicate_samplers.push_back(i);
	}
	next.compact_images = std::ranges::any_of(specialization.images,
	    [](const auto& image) { return image.fmask; });
	next.specialization = specialization;
	next.publication = guard.publication + 1;
	// The previous guard's storage becomes the next candidate.
	std::swap(guard, next);
}

static void ApplySpecializationGuard(const ResourceSpecializationGuard& guard,
                                     ResourceSnapshot& snapshot) {
	// Copy sampler payloads from this invocation, never from the captured shape.
	snapshot.samplers.reserve(snapshot.samplers.size() + guard.duplicate_samplers.size());
	for (const auto source : guard.duplicate_samplers) snapshot.samplers.push_back(snapshot.samplers[source]);
	if (guard.compact_images) {
		uint32_t output = 0;
		for (uint32_t i = 0; i < guard.specialization.images.size(); ++i) {
			if (!guard.specialization.images[i].fmask) {
				if (output != i) snapshot.images[output] = std::move(snapshot.images[i]);
				++output;
			}
		}
		snapshot.images.resize(output);
	}
}

static std::vector<ResourceBlock> ResourceControlFlow(const Program& program) {
	if (program.blocks.size() != program.block_info.size()) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				if (ValidateRuntimeValue(program, info.condition, RuntimeValueType::Integer)) {
					block.condition = info.condition;
				}
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
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			// Any shader write may alias a scalar predicate read, including on a later loop visit.
			if (buffer == BufferAccess::Write || buffer == BufferAccess::Atomic ||
			    image.access == ImageAccess::Write || image.access == ImageAccess::Atomic ||
			    AddressOpcodeInfoOf(op).access == AddressAccess::Write) {
				return {};
			}
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only) {
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
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis,
                                                      uint32_t depth = 0) {
	value = value.Resolve();
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
	auto left  = FillIndex(inst->Arg(0), axis, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, depth + 1);
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
			const auto index = FillIndex(address->Arg(axis), axis);
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
		    store->Arg(3).Resolve() != Value(0u) || store->Arg(5).Resolve() != Value(true))
			return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = data.Resolve();
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = result.fill.words == 1 ? data : vector->Arg(i);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
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

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
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
		target.indirect_image = source.indirect_image;
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	plan.control_flow = ResourceControlFlow(program);
	for (auto& block: plan.control_flow) {
		block.condition = Clone(block.condition);
	}
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.materialization_sources.reserve(plan.info.buffers.size() + plan.info.images.size() +
	                                     plan.info.samplers.size());
	for (const auto& buffer: plan.info.buffers) {
		plan.materialization_sources.push_back(buffer.source);
	}
	for (const auto& image: plan.info.images) {
		const auto* source = Source(plan, image.source);
		if (source != nullptr && source->indirect_image.has_value()) {
			plan.requires_specialization_memory = true;
		} else {
			plan.materialization_sources.push_back(image.source);
		}
	}
	for (const auto& sampler: plan.info.samplers) {
		plan.materialization_sources.push_back(sampler.source);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	for (const auto& image: plan.info.images) {
		const auto* source = Source(plan, image.source);
		if (source == nullptr || !source->indirect_image.has_value()) {
			continue;
		}
		if (source->indirect_image->key_bound == 0u) {
			MarkCleanFlatSlots(plan, Source(plan, source->indirect_image->material_source),
			                   plan.clean_flat_slots);
		}
		MarkCleanFlatSlots(plan, Source(plan, source->indirect_image->heap_source),
		                   plan.clean_flat_slots);
		if (source->indirect_image->bound_source !=
		    DescriptorSource::IndirectImage::NoBoundSource) {
			MarkCleanFlatSlots(plan, Source(plan, source->indirect_image->bound_source),
			                   plan.clean_flat_slots);
		}
	}
	BuildLinearSrtPlan(plan);
	return plan;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          MaterializeReport* report, ResourceSpecializationGuard* shape_guard,
                          const ResourceSpecialization** borrowed_specialization) {
	if (report != nullptr) {
		*report = {};
	}
	NativePreparationScratch<MaterializeWorkspace> storage;
	auto& workspace = storage.Get();
	if (!MaterializeSnapshot(program, runtime, workspace, report)) {
		return false;
	}
	if (shape_guard != nullptr &&
	    kyty_local_specialization_guard_mode.load(std::memory_order_relaxed) != 0) {
		if (MatchesSpecializationGuard(program, workspace.materialized, *shape_guard)) {
			ApplySpecializationGuard(*shape_guard, workspace.materialized.resources);
			if (borrowed_specialization != nullptr) {
				*borrowed_specialization = &shape_guard->specialization;
			} else {
				specialization = shape_guard->specialization;
			}
			std::swap(snapshot, workspace.materialized.resources);
			return true;
		}
		auto& next = workspace.guard_candidate;
		next.owner = nullptr;
		next.buffers.clear();
		next.images.clear();
		next.duplicate_samplers.clear();
		next.compact_images = false;
		const bool eligible = CaptureSpecializationGuard(program, workspace.materialized, next);
		if (!BuildResourceSpecialization(program, workspace, snapshot, specialization, report)) {
			return false;
		}
		if (eligible) {
			PublishSpecializationGuard(program, specialization, next, *shape_guard);
		}
		if (borrowed_specialization != nullptr) {
			*borrowed_specialization = eligible ? &shape_guard->specialization : nullptr;
		}
		return true;
	}
	if (!BuildResourceSpecialization(program, workspace, snapshot, specialization, report)) {
		return false;
	}
	if (borrowed_specialization != nullptr) {
		*borrowed_specialization = nullptr;
	}
	return true;
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() != specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto buffers = program.info.buffers;
	for (size_t index = 0; index < buffers.size(); index++) {
		buffers[index].packed_stride      = specialization.buffers[index].packed_stride;
		buffers[index].descriptor_format  = specialization.buffers[index].descriptor_format;
		buffers[index].descriptor_swizzle = specialization.buffers[index].descriptor_swizzle;
		buffers[index].byte_base_offset   = specialization.buffers[index].byte_base_offset;
	}
	auto images = program.info.images;
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= program.info.images.size());
			images.push_back(program.info.images[source.indirect_root]);
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
	EXIT_IF(!BuildSamplerPlan(program.info, images, sampler_plan));
	auto samplers      = program.info.samplers;
	auto sampled_pairs = program.info.sampled_pairs;
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < program.info.samplers.size(); index++) {
		const auto target = sampler_plan.point_sampler[index];
		if (target == UINT32_MAX) {
			continue;
		}
		if (target == index) {
			samplers[index].force_point_filtering = true;
		} else {
			EXIT_IF(target != samplers.size());
			auto sampler                  = samplers[index];
			sampler.force_point_filtering = true;
			samplers.push_back(std::move(sampler));
		}
	}
	for (auto& pair: sampled_pairs) {
		if (RequiresPointSampler(images[pair.image])) {
			EXIT_IF(sampler_plan.point_sampler[pair.sampler] == UINT32_MAX);
			pair.sampler = sampler_plan.point_sampler[pair.sampler];
		}
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
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
			if (image_opcode.needs_sampler && RequiresPointSampler(image) &&
			    memory.sampler < program.info.samplers.size()) {
				EXIT_IF(sampler_plan.point_sampler[memory.sampler] == UINT32_MAX);
				memory.sampler = sampler_plan.point_sampler[memory.sampler];
			}
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw);
		}
	}
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap.indices.at(inst.Flags<uint32_t>()));
			}
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap.indices.at(memory.resource);
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap.indices.at(buffer.image_alias);
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap.indices.at(pair.image);
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap.indices.at(image.indirect_root);
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap.indices.at(resource);
		}
	}
	image_remap.Apply(images);
	program.info.buffers       = std::move(buffers);
	program.info.images        = std::move(images);
	program.info.samplers      = std::move(samplers);
	program.info.sampled_pairs = std::move(sampled_pairs);
	program.memory_info        = std::move(memory_info);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
