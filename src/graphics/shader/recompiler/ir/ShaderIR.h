#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "graphics/shader/recompiler/ir/BdaReadPlan.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	Flat,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::Global || kind == ResourceKind::Scratch;
}

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	uint32_t                image_nsa_dwords         = 0;
	uint32_t                image_nsa_addr[Decoder::MaxImageNsaAddressComponents] = {};
	uint32_t                memory_segment                                        = 0;
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    image_cube                                            = false;
	bool                    glc                                                   = false;
	bool                    slc                                                   = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    planning_only                                         = false;

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	bool                   byte_base_offset   = false;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, DynamicStorage };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	// indirect_search_iterations of a portable table (PortableShaders): the search runs until its
	// range is empty, and indirect_mapping_offset names the flattened-SRT word that holds the
	// offset of the key mapping.
	static constexpr uint32_t RuntimeIndirectSearch = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;

	bool operator==(const ImageResource& other) const = default;
};

struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class StageInputKind {
	VertexIndex,
	InstanceIndex,
	BaseInstance,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordNoPerspective,
	WorkgroupId,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size     = false;
	bool     layer          = false;
	bool     viewport       = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control,
	                                                           uint32_t pos_index,
	                                                           uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 43u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	LodStats,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 44u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 51u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) +
	       (stage == ShaderType::Pixel ? static_cast<uint32_t>(DescriptorBindingKind::Count) : 0u);
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 5u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 5u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding;
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 4u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

struct DescriptorBinding {
	DescriptorBindingKind kind = DescriptorBindingKind::Buffers;
	std::vector<uint32_t> resources;

	bool operator==(const DescriptorBinding& other) const = default;
};

// A buffer's runtime descriptor word (shader data, after the memory offsets): what its code reads
// from the V# at run time instead of a specialization, so one module serves every value.
//   [0,12)  stride, when below 4096 and not swizzled (a larger one stays in the specialization)
//   [12,16) layout of a formatted buffer's elements: 0-11 byte-aligned components (count - 1 in
//           the low two bits, log2 of the component bytes in the high two), 12 11_11_10,
//           13 10_11_11, 14 2_10_10_10, 15 10_10_10_2
//   [16,19) Format::ComponentType of the elements (0: unknown format, raw dwords)
//   [20,32) dst_sel
struct BufferWord {
	static constexpr uint32_t StrideBits    = 12;
	static constexpr uint32_t MaxStride     = (1u << StrideBits) - 1u;
	static constexpr uint32_t LayoutShift   = 12;
	static constexpr uint32_t TypeShift     = 16;
	static constexpr uint32_t DstSelShift   = 20;
	static constexpr uint32_t PackedLayouts = 12;
};

// Portable shader modules (KYTY_PORTABLE_SHADERS, default on; read once, fixed for the process):
// what a module could only get from the descriptors the game binds at run time (a buffer's stride
// and format, a null texture's type) no longer selects one, so the modules of a program can be
// compiled ahead from the game files (tools/local/static-precompile). Buffers take those values
// from their buffer words (BufferWord) in the shader data.
bool PortableShaders();

inline constexpr uint32_t BufferAddTidBit = 1u << 20u;

// With buffer words, a buffer whose stride (ShaderBufferResource::PackedStride) its code reads at
// run time: not swizzled (the swizzled address takes the stride apart) and at most MaxStride.
[[nodiscard]] constexpr bool RuntimeBufferStride(uint32_t packed_stride) {
	const auto stride = packed_stride & 0x3fffu;
	return (stride == 0u || ((packed_stride >> 14u) & 1u) == 0u) && stride <= BufferWord::MaxStride;
}

// With buffer words, a formatted buffer whose format its code can decode at run time: one it only
// loads from (stores pack with the specialized format). Its specialization chooses: the V#'s format
// and dst_sel give the fast specialized decode, kInvalid and the default dst_sel (PortableFormats)
// the decode of whatever the buffer word says.
[[nodiscard]] constexpr bool RuntimeBufferFormat(const BufferResource& buffer) {
	return buffer.formatted && !buffer.written;
}

struct BindingLayout {
	uint32_t                       push_data_start_dword = PushData::NoStart;
	uint32_t                       memory_offset_dword = 0;
	uint32_t                       memory_offset_count = 0;
	uint32_t                       buffer_word_count = 0; // one per buffer (BufferWord), or none
	uint32_t                       lod_stats_count = 0;
	uint32_t                       table_block_dwords = 0; // table mode: the block's device address (TablePlan)
	std::vector<uint32_t>          user_data_registers;
	std::vector<DescriptorBinding> descriptors;

	[[nodiscard]] uint32_t BufferWordDword() const {
		return memory_offset_dword + (memory_offset_count + 3u) / 4u;
	}
	[[nodiscard]] uint32_t LodStatsDword() const {
		return BufferWordDword() + buffer_word_count;
	}
	[[nodiscard]] uint32_t TableBlockDword() const {
		return LodStatsDword() + lod_stats_count;
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return TableBlockDword() + table_block_dwords;
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const BindingLayout& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBuffers      = 32;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;
	static constexpr uint32_t MaxSampledPairs = 64;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                             has_bitwise_xor    = false;
	bool                             uses_dma           = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct SpirvRequirements {
	bool subgroup_ballot              = false;
	bool subgroup_shuffle             = false;
	bool subgroup_local_invocation_id = false;
	bool compute_derivatives          = false;
	bool image_gather_extended        = false;
	bool function_lds                 = false;
	bool function_scratch             = false;
	bool pixel_valid_mask             = false;
	bool buffer_int64_atomics         = false;
};

struct BlockInfo {
	uint32_t        id       = 0;
	uint32_t        start_pc = 0;
	uint32_t        end_pc   = 0;
	CFG::Terminator terminator;
	Value           condition;
	Value           indirect_target;
};

struct DescriptorSource {
	struct IndirectImage {
		uint32_t material_source = 0;
		uint32_t heap_source     = 0;
		uint32_t selector_stride = 0;
		uint32_t selector_offset = 0;
		uint32_t key_arg         = 0;
		uint32_t table_offset = 0;
		uint32_t key_bound    = 0;
		static constexpr uint32_t NoBoundSource = UINT32_MAX;
		uint32_t                  bound_source  = NoBoundSource;
		bool                      bound_signed  = false;
		uint32_t item_bound = 0;
		uint32_t selector_immediate = 0;

		bool operator==(const IndirectImage& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectImage> indirect_image;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
};

// Table mode (Program::table_mode): the flattened SRT's slots, the V#s of the buffers the shader reads by device
// address, and the T#s and S#s of its images and samplers. Slot s holds the dword at
// ((high:low) & 0xffff'ffff'ffff) + (offset & ~3) + immediate, dword aligned; a descriptor is its words. An
// operand is an immediate, a user data register or a lower slot. For each draw the renderer evaluates the slots
// `cpu` marks (those the shader reads, `gpu`, the image and sampler words, which choose its descriptor set, and
// what their addresses are read through) as the guest memory holds them then, and gives the shader a block: dword s
// is slot s; from BufferDword(i), buffer i's device address (two dwords), size in dwords (0: none), buffer word
// and stride, resolved from its V#. The shader reads the block through its device address (TableBlockDword).
struct TablePlan {
	struct Operand {
		// ShaderBaseLow/High: the low or high dword of the stage's code address plus `value` (a V# of constants the code
		// embeds, addressed from s_getpc_b64; the renderer has the address from the registers).
		enum class Kind : uint8_t { Immediate, UserData, Slot, ShaderBaseLow, ShaderBaseHigh };
		Kind     kind  = Kind::Immediate;
		uint32_t value = 0; // the immediate, the scalar register, the slot or the offset from the code address
		bool     operator==(const Operand&) const = default;
	};
	struct Slot {
		Operand low, high, offset;
		int32_t immediate = 0; // dword aligned
		bool    operator==(const Slot&) const = default;
	};
	struct Descriptor {
		std::array<Operand, 8> words {};
		uint32_t               count = 0;
		bool                   operator==(const Descriptor&) const = default;
	};
	std::vector<Slot>                   slots;
	std::vector<std::array<Operand, 4>> buffers;
	std::vector<Descriptor>             images, samplers;
	std::vector<uint8_t>                gpu, cpu; // per slot
	// Flat or global memory reads: the renderer readies their page table ranges at every use, from the block's slots
	// and the user data (CompiledShaderInfo::bda_read_plan), as the normal path's PrepareBdaBindings does.
	bool global_memory = false;
	// Reads at offsets no slot fixes (a light's record by the index of its bit in a mask): loads by device address,
	// through the page table, whose ranges the read plan readies (global_memory).
	bool dynamic_reads = false;
	// An image an address probe chooses (DescriptorSource::IndirectImage: records whose keys name T#s in a table, the
	// key the shader's of one record): the renderer enumerates its candidates at every use as resource
	// materialization does (EnumerateAddressProbe), binds them at the root's and its candidates' elements in an order
	// of its own and writes the key mapping into the block at `mapping` ([count, (key, candidate)...] by key), which
	// the shader searches (the root's indirect_mapping_offset: that block dword).
	struct IndirectImage {
		uint32_t                        root = 0; // the image whose indirect_resources are its elements
		DescriptorSource::IndirectImage probe;
		Operand                         records_low, records_high, heap_low, heap_high; // the tables' addresses
		uint32_t                        mapping = 0;
		bool                            operator==(const IndirectImage&) const = default;
	};
	// A table program's indirect image has this many elements (the root and its candidates: a draw or dispatch
	// with fewer candidates binds null images past them), and its mapping room for this many keys.
	static constexpr uint32_t IndirectCapacity      = 16;
	static constexpr uint32_t IndirectMaxKeys       = 64;
	static constexpr uint32_t IndirectMappingDwords = 1 + 2 * IndirectMaxKeys;
	std::vector<IndirectImage> indirect;

	// A buffer's dwords in the block: its device address (2), size in dwords, buffer word (IR::BufferWord, as
	// IR::BufferDescriptorWord makes it) and stride, all from the V# the renderer resolved.
	static constexpr uint32_t BufferDwords = 5;
	[[nodiscard]] uint32_t BufferDword(size_t buffer) const {
		return static_cast<uint32_t>(slots.size() + BufferDwords * buffer);
	}
	[[nodiscard]] uint32_t BlockDwords() const {
		return BufferDword(buffers.size()) + static_cast<uint32_t>(indirect.size()) * IndirectMappingDwords;
	}
	bool                   operator==(const TablePlan&) const = default;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	BdaReadPlan bda_read_plan;
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          table_mode          = false; // see Program::table_mode
	TablePlan                     table_plan;
	ShaderInfo                    info;
	BindingLayout                 bindings;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
};

// Immutable runtime resource analysis retained by the shader cache. It owns descriptor/SRT,
// uniform condition and fill values without retaining translated blocks.
struct LinearSrtPlan;
struct ResourcePlan {
	ResourcePlan() = default;
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept         = default;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::shared_ptr<const LinearSrtPlan> linear_srt;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<ResourceBlock>          control_flow;
	std::vector<uint32_t>               materialization_sources;
	std::vector<SrtRead>                srt_reads;
	std::vector<uint8_t>                clean_flat_slots;
	bool                                requires_specialization_memory = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept         = default;
	Program& operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                      wave_size      = 64;
	// A wave64 compute program whose lanes' results depend on no lane outside their 32-lane half
	// (Frontend::IsWaveLaneLocal): a host with 32-wide subgroups runs it one GCN lane per invocation.
	bool                          lane_local     = false;
	uint32_t                      scratch_dwords = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	std::vector<BlockInfo>        block_info;
	// Decoded MIMG/VMEM metadata carries details such as RDNA2 NSA address registers and
	// storage-image swizzles. Typed memory instructions carry a dense index into these shader-local
	// tables until those fields are consumed by emission.
	std::vector<ExportInfo>       export_info;
	std::vector<Value>            dynamic_reads;
	bool                          shader_info_complete = false;
	BindingLayout                 bindings;
	bool                          binding_layout_complete = false;
	// Table mode (EnterTableMode): the SRT reads and the buffers' V#s come from guest memory through the BDA
	// page table at run time, user data from push constants; no flattened SRT, buffer or shader-data binding.
	bool                          table_mode = false;
	TablePlan                     table_plan;

	std::optional<SpirvRequirements> spirv_requirements;
};

std::string ProgramToString(const Program& program);

void  ValidateProgram(const Program& program, bool require_ssa);
void  ResolveControlFlowIdentities(Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
